# Copyright (c) 2025, Zequn Chen.
# Adapted from PhysGM 7722175 for network-only CPU / Ascend execution.
"""Checkpoint-compatible prediction without rendering or simulation imports."""

import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from easydict import EasyDict
from einops import rearrange
from transformers import AutoModel, DINOv3ViTConfig

from .dense_rep_encoder import DenseRepresentationEncoder
from .dpt import DPTHead
from .transformer import TransformerBlock


class DINOv3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        config = DINOv3ViTConfig(
            hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
            num_attention_heads=16, patch_size=16, num_register_tokens=4,
            use_gated_mlp=False, hidden_act="gelu", query_bias=True,
            key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True,
        )
        self.model = AutoModel.from_config(config, attn_implementation="sdpa")
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        # Parent model.train() must never enable DINO's stochastic training path.
        super().train(False)
        return self

    def forward(self, images):
        # Upstream constructs Resize(256) but never includes it in its Compose.
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with torch.no_grad():
            outputs = self.model(pixel_values=(images - mean) / std)
        patches = outputs.last_hidden_state[:, 1 + self.model.config.num_register_tokens:]
        return patches.reshape(images.shape[0], images.shape[2] // 16, images.shape[3] // 16, -1)


class Processor(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model.num_layers != 24:
            raise ValueError("DPT feature taps [4, 11, 17, 23] require the 24-layer checkpoint")
        self.blocks = nn.ModuleList([
            TransformerBlock(config.model.dim, config.model.transformer.head_dim)
            for _ in range(config.model.num_layers)
        ])
        self.gradient_checkpointing = False

    def forward(self, x):
        outputs = []
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if i in (4, 11, 17, 23):
                outputs.append(x)
        return outputs


class PhysGM(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = EasyDict(config)
        config = self.config
        self.num_global_tokens = config.model.num_global_tokens
        if self.num_global_tokens != 3:
            raise ValueError("The physical heads require exactly three global tokens")
        self.dim_start = self.dim_out = config.model.dim
        self.post_act = config.model.get("post_act", False)
        self.dinov3_extractor = DINOv3FeatureExtractor()
        self.ray_encoder = DenseRepresentationEncoder(
            in_chans=6, embed_dim=32, patch_size=16,
            intermediate_dims=[128], apply_pe=True, input_size_for_pe=256,
        )
        self.dino_proj = nn.Linear(config.model.dinov3_dim + 32, self.dim_start, bias=False)
        self.input_layernorm = nn.LayerNorm(self.dim_start, bias=False)
        self.dpt_head = DPTHead(
            dim_in=self.dim_out,
            output_dim=3 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1,
        )
        self.global_token_init = nn.Parameter(torch.zeros(1, 3, self.dim_start))
        self.processor = Processor(config)
        self.phys_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim_out, bias=False), nn.Linear(self.dim_out, 14, bias=False),
        )
        self.E_token_decoder = self._physical_head()
        self.nu_token_decoder = self._physical_head()
        if device is not None:
            self.to(device)

    def _physical_head(self):
        return nn.Sequential(
            nn.LayerNorm(self.dim_out, bias=False), nn.Linear(self.dim_out, 64),
            nn.GELU(), nn.Linear(64, 2),
        )

    def forward(self, input_dict, *, return_unpruned=False):
        return self.predict(input_dict, return_unpruned=return_unpruned)

    def predict(self, input_dict, *, return_unpruned=False):
        """Predict Gaussians and physical distributions; target-view fields are ignored.

        Required: input_images [B,V,3|4,H,W] float in [0,1], input_intr
        [B,V,4] = fx/fy/cx/cy in pixels, input_c2ws [B,V,4,4].
        Gaussian scale/opacity remain log-scales/logits, rotation is unnormalized,
        and feature contains SH coefficients with the upstream RGB residual.
        """
        images = input_dict["input_images"]
        intr = input_dict["input_intr"]
        c2ws = input_dict["input_c2ws"]
        if images.ndim != 5 or images.shape[2] not in (3, 4):
            raise ValueError("input_images must have shape [B,V,3|4,H,W]")
        B, V, C, H, W = images.shape
        if H % 16 or W % 16:
            raise ValueError("Input height and width must be multiples of 16")
        if intr.shape != (B, V, 4) or c2ws.shape != (B, V, 4, 4):
            raise ValueError("input_intr / input_c2ws shapes do not match the input views")
        start = time.perf_counter()
        if C == 4:
            images = images[:, :, :3] * images[:, :, 3:] + (1 - images[:, :, 3:])
        dino = self.dinov3_extractor(images.reshape(B * V, 3, H, W))
        dino = rearrange(dino, "(b v) h w c -> b v (h w) c", b=B, v=V)

        ray_o = c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1).float()
        x, y = torch.meshgrid(
            torch.arange(W, device=images.device, dtype=intr.dtype),
            torch.arange(H, device=images.device, dtype=intr.dtype), indexing="xy",
        )
        x = (x + 0.5).reshape(1, 1, -1).expand(B, V, -1).contiguous()
        y = (y + 0.5).reshape(1, 1, -1).expand(B, V, -1).contiguous()
        x = (x - intr[:, :, 2:3]) / intr[:, :, 0:1]
        y = (y - intr[:, :, 3:4]) / intr[:, :, 1:2]
        ray_d = F.normalize(torch.stack((x, y, torch.ones_like(x)), -1).float(), dim=-1)
        ray_d = ray_d @ c2ws[:, :, :3, :3].transpose(-1, -2).contiguous()
        plucker = torch.cat((torch.cross(ray_o, ray_d, dim=-1), ray_d), dim=-1)
        rays = self.ray_encoder(rearrange(plucker, "b v (h w) c -> (b v) c h w", h=H, w=W))
        rays = rearrange(rays, "(b v) c h w -> b v (h w) c", b=B, v=V)
        image_tokens = self.dino_proj(torch.cat((dino, rays), -1).flatten(1, 2))
        tokens = torch.cat((self.global_token_init.expand(B, -1, -1), image_tokens), dim=1)
        aggregated = self.processor(self.input_layernorm(tokens))
        phys_logits = self.phys_token_decoder(aggregated[-1][:, 0])
        E_logits = self.E_token_decoder(aggregated[-1][:, 1])
        nu_logits = self.nu_token_decoder(aggregated[-1][:, 2])
        image_tokens = [t[:, 3:].reshape(B, V, -1, self.dim_out) for t in aggregated]
        dense = self.dpt_head(image_tokens, images)
        dense = rearrange(dense, "b v h w d -> b (v h w) d")
        D_sh = (self.config.model.gaussians.sh_degree + 1) ** 2
        xyz, feature, scale, rotation, opacity = dense.split((3, D_sh * 3, 3, 4, 1), dim=-1)
        feature = feature.reshape(B, V * H * W, D_sh, 3)
        if self.post_act:
            xyz = xyz.clamp(-1, 1)
            scale = 0.1 * F.softplus(scale)
            opacity = opacity.sigmoid()
            rotation = F.normalize(rotation, dim=-1)
            feature = 0.5 * feature.tanh() + 0.5
        rgb = rearrange(images, "b v c h w -> b (v h w) c")
        # Avoid an in-place write through a split view on the training graph.
        feature = torch.cat((feature[:, :, :1] + rgb.unsqueeze(2), feature[:, :, 1:]), dim=2)
        gs_config = self.config.model.gaussians
        scale = (scale + gs_config.scale_bias).clamp(max=gs_config.scale_max)
        opacity = opacity + gs_config.opacity_bias
        if gs_config.get("align_to_pixel", True):
            distance = xyz.mean(-1, keepdim=True).sigmoid() * gs_config.max_dist
            xyz = distance * ray_d.reshape(B, -1, 3) + ray_o.reshape(B, -1, 3)
        unpruned = {k: t.float() for k, t in dict(
            xyz=xyz, feature=feature, scale=scale, rotation=rotation, opacity=opacity,
        ).items()}
        gaussians = unpruned
        prune_ratio = gs_config.get("prune_ratio", 0.0)
        if not 0 <= prune_ratio < 1:
            raise ValueError("prune_ratio must be in [0,1)")
        if prune_ratio > 0:
            random_ratio = gs_config.get("random_ratio", 0.0)
            if not 0 <= random_ratio <= 1:
                raise ValueError("random_ratio must be in [0,1]")
            count = xyz.shape[1]
            num_random = int(count * (1 - prune_ratio) * random_ratio)
            num_keep = int(count * (1 - prune_ratio) * (1 - random_ratio))
            indices = opacity.squeeze(-1).argsort(dim=1, descending=True)
            keep = indices[:, :num_keep]
            if num_random:
                rest = indices[:, num_keep:]
                perm = torch.randperm(rest.shape[1], device=rest.device)[:num_random]
                keep = torch.cat((keep, rest[:, perm]), dim=1)
            gaussians = {}
            for k, tensor in unpruned.items():
                flat = tensor.reshape(B, count, -1)
                selected = flat.gather(1, keep.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
                gaussians[k] = selected.reshape(B, -1, *tensor.shape[2:])
        result = {
            "gaussians": gaussians,
            "gaussian_usage": (opacity.sigmoid() > gs_config.get("opacity_threshold", 0.001)).float().mean(1).squeeze(-1),
            "phys_logits": phys_logits,
            "E_mu": E_logits[..., 0], "E_var": F.softplus(E_logits[..., 1]) + 1e-2,
            "nu_mu": nu_logits[..., 0], "nu_var": F.softplus(nu_logits[..., 1]) + 1e-2,
            # Host submission time only; the CLI separately measures synchronized device time.
            "inference_time": time.perf_counter() - start,
        }
        if return_unpruned:
            result["gaussians_unpruned"] = unpruned
        for key in ("pos_avg_inv", "scene_scale", "scene_name"):
            if key in input_dict:
                result[key] = input_dict[key]
        return result


def load_checkpoint(config, path: str | Path, device="cpu"):
    """Load only CPU tensors, preserving all checkpoint keys and freezing DINO."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("model", checkpoint)
    expected_channels = 3 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
    head = state.get("dpt_head.scratch.output_conv2.2.weight")
    if head is not None and head.shape[0] != expected_channels:
        raise ValueError(
            f"Gaussian head has {head.shape[0]} checkpoint channels, configuration requests "
            f"{expected_channels}; check model.gaussians.sh_degree (provided checkpoint: 0)"
        )
    # No random allocation of the 2.4 GB network before replacing its parameters.
    with torch.device("meta"):
        model = PhysGM(config)
    model.load_state_dict(state, strict=True, assign=True)
    # DINO's rotary buffer is nonpersistent and therefore absent from state_dict.
    rope = model.dinov3_extractor.model.rope_embeddings
    rope.inv_freq = 1 / rope.base ** torch.arange(0, 1, 4 / rope.head_dim, dtype=torch.float32, device="cpu")
    model.dinov3_extractor.model.requires_grad_(False)
    model.to(device)
    return model, {"strict": True, "model_entries": len(state), "train_steps_done": checkpoint.get("train_steps_done")}


def initialize_from_dino(config, path, device="cpu"):
    """Fresh non-DINO network using only strictly checked offline DINO weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    source = checkpoint.get("model", checkpoint)
    prefix = "dinov3_extractor.model."
    state = {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
    if not state:
        raise ValueError("No dinov3_extractor.model weights in DINO initialization checkpoint")
    model = PhysGM(config)
    model.dinov3_extractor.model.load_state_dict(state, strict=True)
    # Original PhysGM initialization differs from PyTorch defaults in these heads.
    for module in model.phys_token_decoder.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    for head in (model.E_token_decoder, model.nu_token_decoder):
        nn.init.normal_(head[-1].weight, std=.01)
        nn.init.zeros_(head[-1].bias)
    model.to(device)
    return model, {"mode": "fresh_non_dino", "strict_dino": True, "dino_entries": len(state)}
