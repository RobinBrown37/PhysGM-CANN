import os

import pytest
import torch

from rendering import StaticGaussianRenderer
from rendering.geometry import quaternion_matrix, sh_colors


def scene(*, degree=0, dtype=torch.float64, device="cpu", count=2):
    g = {
        "xyz": torch.tensor([[0.037, -0.021, 1.10], [-0.046, 0.033, 1.38]], dtype=dtype)[:count],
        "scale": torch.tensor([[-2.5, -2.9, -3.2], [-2.7, -2.4, -3.0]], dtype=dtype)[:count],
        "rotation": torch.tensor([[0.91, 0.13, -0.21, 0.08], [0.87, -0.16, 0.23, 0.11]], dtype=dtype)[:count],
        "opacity": torch.tensor([[-0.7], [0.2]], dtype=dtype)[:count],
        "feature": torch.linspace(-0.16, 0.35, max(1, count)*(degree+1)**2*3, dtype=dtype).reshape(max(1, count), (degree+1)**2, 3)[:count],
    }
    g = {k: v[None].to(device).detach().requires_grad_(True) for k, v in g.items()}
    c2w = torch.eye(4, dtype=dtype, device=device)[None, None]
    intr = torch.tensor([[[23., 25., 15.7, 16.3]]], dtype=dtype, device=device)
    return g, c2w, intr


def test_covariance_rotation_orthogonal_and_sh_dc():
    q = torch.tensor([[0.91, 0.13, -0.21, 0.08]], dtype=torch.float64)
    R = quaternion_matrix(q)
    torch.testing.assert_close(R@R.transpose(-1, -2), torch.eye(3, dtype=q.dtype)[None])
    f = torch.tensor([[[0.2, 0.3, -0.1]]], dtype=q.dtype)
    expected = f[:, 0]*0.28209479177387814+0.5
    torch.testing.assert_close(sh_colors(f, torch.tensor([[0., 0., 1.]], dtype=q.dtype), torch.zeros(3, dtype=q.dtype), 0), expected)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_reference_gradcheck_all_five_attributes(degree):
    g, c2w, intr = scene(degree=degree, count=1)
    intr = intr.clone()
    intr[..., 2:] = torch.tensor([3.2, 2.7])
    render = StaticGaussianRenderer(backend="reference", sh_degree=degree)
    keys = tuple(g)
    def function(*args):
        return render(dict(zip(keys, args)), c2w, intr, width=7, height=5).rgba
    assert torch.autograd.gradcheck(function, tuple(g.values()), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)


@pytest.mark.parametrize("count,near,far", [(0, 0.001, 2.5), (2, 1.5, 2.5), (2, 0.001, 1.0)])
def test_empty_or_culled_background_has_zero_gradients(count, near, far):
    g, c2w, intr = scene(count=count)
    render = StaticGaussianRenderer(backend="reference", near_plane=near, far_plane=far)
    result = render(g, c2w, intr, width=7, height=5, background=(0.2, 0.3, 0.4))
    torch.testing.assert_close(result.alpha, torch.zeros_like(result.alpha))
    torch.testing.assert_close(result.rgb, torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64).view(1, 1, 3, 1, 1).expand_as(result.rgb))
    result.rgba.sum().backward()
    assert all(v.grad is not None and not v.grad.any() for v in g.values())


def test_reference_batch_views_and_background():
    g, c2w, intr = scene(degree=1)
    g = {k: v.expand(2, *v.shape[1:]) for k, v in g.items()}
    c2w = c2w.expand(2, 2, 4, 4).clone()
    c2w[:, 1, 0, 3] = 0.1
    intr = intr.expand(2, 2, 4)
    render = StaticGaussianRenderer(backend="reference", sh_degree=1)
    white = render(g, c2w, intr, width=31, height=25)
    black = render(g, c2w, intr, width=31, height=25, background=(0, 0, 0))
    assert white.rgba.shape == (2, 2, 4, 25, 31)
    torch.testing.assert_close(white.rgb-black.rgb, (1-white.alpha).expand_as(white.rgb))
    assert not torch.allclose(white.rgb[:, 0], white.rgb[:, 1])


def test_reference_scale_guard_and_explicit_backend():
    g, c2w, intr = scene()
    with pytest.raises(ValueError, match="scale limit"):
        StaticGaussianRenderer(backend="reference", reference_max_pairs=1)(g, c2w, intr, width=32, height=32)
    with pytest.raises(ValueError, match="requires NPU"):
        StaticGaussianRenderer()(g, c2w, intr, width=32, height=32)


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="NPU rendering requires an exposed device and built operators")
@pytest.mark.parametrize("degree,width,height,upstream", [(0, 32, 32, "rgb"), (0, 31, 25, "alpha"), (1, 33, 35, "both"), (3, 32, 32, "both")])
def test_native_matches_reference_rgb_alpha_and_raw_gradients(degree, width, height, upstream, record_property):
    g, c2w, intr = scene(degree=degree)
    ref = StaticGaussianRenderer(backend="reference", sh_degree=degree)(g, c2w, intr, width=width, height=height)
    ng = {k: v.detach().float().to("npu:0").requires_grad_(True) for k, v in g.items()}
    native = StaticGaussianRenderer(sh_degree=degree)(ng, c2w.float().to("npu:0"), intr.float().to("npu:0"), width=width, height=height)
    torch.testing.assert_close(native.rgb.cpu().double(), ref.rgb, rtol=1e-3, atol=2e-4)
    torch.testing.assert_close(native.alpha.cpu().double(), ref.alpha, rtol=1e-3, atol=2e-4)
    record_property("rgb_max_abs_error", float((native.rgb.cpu().double()-ref.rgb).abs().max()))
    record_property("alpha_max_abs_error", float((native.alpha.cpu().double()-ref.alpha).abs().max()))
    weight_rgb = torch.linspace(-0.3, 0.7, ref.rgb.numel(), dtype=torch.float64).reshape_as(ref.rgb)
    weight_alpha = torch.linspace(0.2, 0.8, ref.alpha.numel(), dtype=torch.float64).reshape_as(ref.alpha)
    if upstream == "rgb":
        weight_alpha.zero_()
    elif upstream == "alpha":
        weight_rgb.zero_()
    torch.autograd.backward((ref.rgb, ref.alpha), (weight_rgb, weight_alpha))
    torch.autograd.backward((native.rgb, native.alpha), (weight_rgb.float().to("npu:0"), weight_alpha.float().to("npu:0")))
    for key in g:
        assert ng[key].grad is not None and torch.isfinite(ng[key].grad).all()
        torch.testing.assert_close(ng[key].grad.cpu().double(), g[key].grad, rtol=1e-2, atol=2e-4, msg=lambda msg: key+": "+msg)
        error = ng[key].grad.cpu().double()-g[key].grad
        record_property(key+"_gradient_max_abs_error", float(error.abs().max()))
        record_property(key+"_gradient_relative_l2", float(error.norm()/g[key].grad.norm().clamp_min(1e-12)))


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="NPU required")
def test_native_empty_and_alpha_saturation():
    g, c2w, intr = scene(dtype=torch.float32, device="npu:0", count=1)
    with torch.no_grad():
        g["xyz"][0, 0] = torch.tensor([0., 0., 1.], device="npu:0")
        g["opacity"].fill_(20)
        intr[..., 2:] = 0.5
    image = StaticGaussianRenderer()(g, c2w, intr, width=1, height=1)
    torch.testing.assert_close(image.alpha.cpu(), torch.tensor([[[[[0.999]]]]]), atol=2e-4, rtol=1e-3)
    image.alpha.sum().backward()
    assert g["opacity"].grad.abs().max() == 0
    empty = {k: v.detach()[:, :0].requires_grad_(True) for k, v in g.items()}
    result = StaticGaussianRenderer()(empty, c2w, intr, width=1, height=1)
    result.rgba.sum().backward()
    assert not result.alpha.any()
    assert all(v.grad is not None for v in empty.values())


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="NPU required")
def test_native_groups_tail_multiview_and_directional_derivative(record_property):
    # Exercise the four-splat kernel loop and its tail, storage padding and a
    # camera count > 1. Distinct depths avoid a sorting-boundary derivative.
    g, camera, intr = scene(degree=1)
    g = {k: v.repeat(1, 3, *([1]*(v.ndim-2)))[:, :5].detach().requires_grad_(True) for k, v in g.items()}
    with torch.no_grad():
        g["xyz"][0, :, 2] += torch.arange(5, dtype=torch.float64)*0.017
        g["opacity"].sub_(1.1)
    camera = camera.expand(1, 2, 4, 4).clone()
    camera[:, 1, 0, 3] = 0.07
    intr = intr.expand(1, 2, 4)
    ref = StaticGaussianRenderer(backend="reference", sh_degree=1)(g, camera, intr, width=33, height=35)
    ng = {k: v.detach().float().to("npu:0").requires_grad_(True) for k, v in g.items()}
    ncam, ni = camera.float().to("npu:0"), intr.float().to("npu:0")
    renderer = StaticGaussianRenderer(sh_degree=1)
    image = renderer(ng, ncam, ni, width=33, height=35)
    torch.testing.assert_close(image.rgba.cpu().double(), ref.rgba, rtol=1e-3, atol=2e-4)
    ref.rgba.mean().backward()
    image.rgba.mean().backward()
    for key in g:
        torch.testing.assert_close(ng[key].grad.cpu().double(), g[key].grad, rtol=1e-2, atol=2e-4)
    # A full-image finite difference can cross the discrete 1/255 alpha cutoff
    # in tail pixels. Use an interior pixel away from those branch boundaries.
    for value in ng.values():
        value.grad = None
    renderer(ng, ncam, ni, width=33, height=35).rgba[0, 0, :, 16, 16].sum().backward()
    direction = torch.zeros_like(ng["xyz"])
    direction[0, 0, 0] = 1
    analytic = float((ng["xyz"].grad*direction).sum().cpu())
    eps = 1e-3
    with torch.no_grad():
        plus = {**ng, "xyz": ng["xyz"]+eps*direction}
        minus = {**ng, "xyz": ng["xyz"]-eps*direction}
        positive = renderer(plus, ncam, ni, width=33, height=35).rgba[0, 0, :, 16, 16].sum()
        negative = renderer(minus, ncam, ni, width=33, height=35).rgba[0, 0, :, 16, 16].sum()
        finite = float(((positive-negative)/(2*eps)).cpu())
    assert abs(analytic-finite) <= 2e-4+0.01*abs(analytic)
    record_property("directional_derivative_analytic", analytic)
    record_property("directional_derivative_finite_difference", finite)
