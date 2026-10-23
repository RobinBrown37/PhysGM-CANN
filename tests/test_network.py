import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
import yaml
from easydict import EasyDict
from plyfile import PlyData

from data.dataset_infer import Dataset
from export import physical_values, save_gaussian_ply
from model.dense_rep_encoder import DenseRepresentationEncoder
from model.dpt import DPTHead
from model.physgm import load_checkpoint
from model.transformer import SelfAttention


ROOT = Path(__file__).resolve().parents[1]


def config():
    return EasyDict(yaml.safe_load((ROOT / "configs/network_npu.yaml").read_text()))


def explicit_attention(module, x):
    B, L, D = x.shape
    q, k, v = module.qkv(x).reshape(B, L, 3, module.num_heads, D // module.num_heads).permute(2, 0, 3, 1, 4)
    weights = ((q @ k.transpose(-2, -1)) * module.scale).softmax(-1)
    output = (weights @ v).transpose(1, 2).reshape(B, L, D)
    return module.proj(output)


@pytest.mark.parametrize("scale", [None, 0.37])
def test_sdpa_matches_attention_forward_and_backward(scale):
    model = SelfAttention(32, head_dim=8, qk_scale=scale).double().eval()
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
    actual = model(x)
    expected = explicit_attention(model, x)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    targets = (x, *model.parameters())
    actual_grads = torch.autograd.grad(actual, targets, upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, targets, upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_attention_dropout_only_in_training():
    model = SelfAttention(32, head_dim=8, attn_drop=0.5)
    x = torch.randn(2, 7, 32)
    model.eval()
    torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
    with patch("model.transformer.F.scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
        model(x)
        assert sdpa.call_args.kwargs["dropout_p"] == 0
        model.train()
        model(x)
        assert sdpa.call_args.kwargs["dropout_p"] == 0.5


@pytest.mark.parametrize("height,width", [(256, 256), (512, 512), (128, 256)])
def test_ray_position_cache_matches_original_interpolation(height, width):
    model = DenseRepresentationEncoder(in_chans=6, embed_dim=32)
    features = torch.randn(1, height // 16 * (width // 16), 32)
    actual = model._interpolate_pos_encoding(features, height, width)
    if features.shape[1] == 256:
        expected = model.pos_embed.unsqueeze(0)
    else:
        expected = F.interpolate(model.pos_embed.reshape(1, 16, 16, 32).permute(0, 3, 1, 2), size=(height // 16, width // 16), mode="bicubic", antialias=True)
        expected = expected.permute(0, 2, 3, 1).reshape(1, -1, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.data_ptr() == model._interpolate_pos_encoding(features, height, width).data_ptr()
    model.load_state_dict(model.state_dict())
    assert not model._pe_cache


def test_dpt_cpu_constant_cache_keeps_backward():
    model = DPTHead(dim_in=8, features=8, out_channels=[8, 8, 8, 8], output_dim=14)
    tokens = [torch.randn(1, 1, 4, 8, requires_grad=True) for _ in range(4)]
    result = model(tokens, torch.randn(1, 1, 3, 32, 32))
    assert result.shape == (1, 1, 32, 32, 14)
    result.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in tokens)
    assert model._pos_cache
    assert all(t.device.type == "cpu" and not t.requires_grad for t in model._pos_cache.values())


def test_dataset_input_only_and_resize():
    cfg = config()
    full = Dataset(cfg)[0]
    cfg.data.resize_h = 256
    small = Dataset(cfg)[0]
    assert full["input_images"].shape == (4, 4, 512, 512)
    assert small["input_images"].shape == (4, 4, 256, 256)
    assert not any(k.startswith("test_") for k in small)
    torch.testing.assert_close(small["input_intr"], full["input_intr"] / 2)
    torch.testing.assert_close(small["input_c2ws"], full["input_c2ws"])
    assert torch.isfinite(small["input_c2ws"]).all()


def test_physical_units():
    prediction = {"E_mu": torch.tensor([0.]), "nu_mu": torch.tensor([0.]), "phys_logits": F.one_hot(torch.tensor([8]), 14).float()}
    assert physical_values(prediction) == {"E": round(10 ** 7.387210 * 0.1, 2), "nu": 0.398, "material": "Rubber"}


@pytest.mark.parametrize("threshold,expected_count", [(None, 2), (0.5, 1), (1.0, 0)])
def test_ply_preserves_raw_attributes(tmp_path, threshold, expected_count):
    gaussians = {"xyz": torch.randn(2, 3), "feature": torch.randn(2, 1, 3), "scale": torch.full((2, 3), -4.), "rotation": torch.randn(2, 4), "opacity": torch.tensor([[-2.], [2.]])}
    path = tmp_path / "test.ply"
    assert save_gaussian_ply(gaussians, path, threshold) == expected_count
    vertices = PlyData.read(path)["vertex"]
    assert len(vertices) == expected_count
    assert "f_rest_44" in vertices.data.dtype.names
    if expected_count:
        assert (vertices["scale_0"] == -4).all()


@pytest.fixture(scope="module")
def weights():
    path = ROOT / "checkpoint.pt"
    if not path.exists():
        pytest.skip("Real checkpoint not present")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def test_real_checkpoint_strict_and_frozen(weights):
    with patch("model.physgm.torch.load", return_value=weights):
        model, report = load_checkpoint(config(), "unused")
    assert report["model_entries"] == 645
    assert not any(t.is_meta for t in model.parameters())
    assert not any(t.is_meta for t in model.buffers())
    assert len(model.state_dict()) == 645
    assert model.dinov3_extractor.model.config._attn_implementation == "sdpa"
    model.train()
    assert not model.dinov3_extractor.model.training
    assert all(not p.requires_grad for p in model.dinov3_extractor.parameters())
    assert all(p.requires_grad for p in model.processor.parameters())
    assert not any(name in sys.modules for name in ("gsplat", "xformers", "warp", "kiui"))


def test_checkpoint_wrong_sh_degree_rejected(weights):
    cfg = config()
    cfg.model.gaussians.sh_degree = 1
    with patch("model.physgm.torch.load", return_value=weights), pytest.raises(ValueError, match="sh_degree"):
        load_checkpoint(cfg, "unused")


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_checkpoint_strict_keys_rejected(weights, change):
    state = dict(weights["model"])
    if change == "missing":
        state.pop("global_token_init")
    else:
        state["unexpected.weight"] = torch.ones(1)
    with patch("model.physgm.torch.load", return_value={"model": state}), pytest.raises(RuntimeError, match="state_dict"):
        load_checkpoint(config(), "unused")


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="Set PHYSGM_TEST_NPU=1 on an exposed Ascend device")
def test_npu_attention_forward_backward_against_fp32_oracle():
    from runtime import select_device

    device = select_device("npu:0")
    reference = SelfAttention(128, head_dim=64, qk_scale=0.37).eval()
    actual_model = SelfAttention(128, head_dim=64, qk_scale=0.37).to(device).eval()
    actual_model.load_state_dict(reference.state_dict())
    x = torch.randn(2, 17, 128, requires_grad=True)
    npu_x = x.detach().to(device).requires_grad_(True)
    expected = explicit_attention(reference, x)
    with torch.autocast("npu", dtype=torch.bfloat16):
        actual = actual_model(npu_x)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.03, atol=0.004)
    upstream = torch.randn_like(expected)
    actual.backward(upstream.to(device))
    expected.backward(upstream)
    pairs = [(npu_x.grad, x.grad), *[(a.grad, b.grad) for a, b in zip(actual_model.parameters(), reference.parameters())]]
    for a, b in pairs:
        assert a is not None and torch.isfinite(a).all() and a.abs().sum() > 0
        # BF16 oracle comparison is limited to this small attention unit,
        # not a substitute for GPU network-output calibration.
        torch.testing.assert_close(a.float().cpu(), b, rtol=0.05, atol=0.04)
