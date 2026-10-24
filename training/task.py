"""DDP-visible composition of prediction, differentiable rendering and supervision."""
import torch
from torch import nn
from rendering import StaticGaussianRenderer
from runtime import autocast
from .losses import SupervisedLoss


class PretrainTask(nn.Module):
    def __init__(self, network, config, device):
        super().__init__()
        self.network, self.config, self.device = network, config, device
        self.precision = config.get("amp_dtype", "bf16") if config.get("use_amp", True) else "fp32"
        self.loss = SupervisedLoss(config.training).to(device)
        gs = config.model.gaussians
        self.renderer = StaticGaussianRenderer(sh_degree=gs.sh_degree, near_plane=gs.near_plane,
            far_plane=gs.far_plane, backend=config.training.get("render_backend", "cann"))
        self.checkpoint_renderer = config.training.get("checkpoint_renderer", False)
        self.inspect_gradients = False
        self.network.processor.gradient_checkpointing = config.training.get("checkpoint_transformer", False)
        for weight, head in (("physical_loss", "phys_token_decoder"), ("E_loss", "E_token_decoder"), ("nu_loss", "nu_token_decoder")):
            if not self.loss.weights[weight]:
                getattr(network, head).requires_grad_(False)
        if not self.loss.image_enabled:
            network.dpt_head.requires_grad_(False)

    def forward(self, batch, *, visuals=False, inspect=False):
        inspect = inspect or self.inspect_gradients
        with autocast(self.device, self.precision):
            prediction = self.network(batch)
        if inspect:
            for tensor in prediction["gaussians"].values():
                if tensor.requires_grad:
                    tensor.retain_grad()
        rendered = None
        if self.loss.image_enabled:
            height, width = batch["test_images"].shape[-2:]
            if self.checkpoint_renderer and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                from rendering.static import RenderResult
                keys = ("xyz", "feature", "scale", "rotation", "opacity")
                def render(*tensors):
                    result = self.renderer(dict(zip(keys, tensors)), batch["test_c2ws"], batch["test_intr"], width=width, height=height)
                    return result.rgb, result.alpha
                rgb, alpha = checkpoint(render, *[prediction["gaussians"][k] for k in keys], use_reentrant=False)
                rendered = RenderResult(rgb, alpha, {"activation_checkpointing": True})
            else:
                rendered = self.renderer(prediction["gaussians"], batch["test_c2ws"], batch["test_intr"], width=width, height=height)
        losses = self.loss(prediction, rendered, batch)
        result = {"losses": losses, "gaussian_usage": prediction["gaussian_usage"].detach().mean()}
        if visuals:
            result["visuals"] = {"rgb": rendered.rgb.detach(), "alpha": rendered.alpha.detach()} if rendered is not None else {}
        if inspect:
            result["prediction"] = prediction
        return result
