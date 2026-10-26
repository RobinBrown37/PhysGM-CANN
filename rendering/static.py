"""Static differentiable RGB/alpha rendering from M1 parameter dictionaries."""
from dataclasses import dataclass

import torch
from torch import nn

from .geometry import project


@dataclass
class RenderResult:
    rgb: torch.Tensor
    alpha: torch.Tensor
    stats: dict

    @property
    def rgba(self):
        return torch.cat((self.rgb, self.alpha), dim=2)


def reference_composite(p, width, height):
    """Small-scene dense CPU oracle, without native kernels or tile culling."""
    order = p["depths"].argsort(stable=True)
    y, x = torch.meshgrid(torch.arange(height, dtype=p["means2d"].dtype, device=p["means2d"].device), torch.arange(width, dtype=p["means2d"].dtype, device=p["means2d"].device), indexing="ij")
    pixels = torch.stack((x+0.5, y+0.5), -1)
    delta = pixels[None] - p["means2d"][order, None, None]
    dx, dy = delta.unbind(-1)
    a, b, c = p["conics"][order].unbind(-1)
    power = 0.5*(a[:, None, None]*dx*dx+c[:, None, None]*dy*dy)+b[:, None, None]*dx*dy
    alpha = (p["opacity"][order, None, None] * (-power).exp()).clamp(max=0.999)
    alpha = torch.where(alpha >= 1/255, alpha, 0)
    transmission = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1-alpha[:-1]), 0), 0)
    weights = alpha * transmission
    rgb = torch.einsum("nhw,nc->chw", weights, p["colors"][order])
    return rgb, weights.sum(0, keepdim=True)


class StaticGaussianRenderer(nn.Module):
    def __init__(self, *, sh_degree=0, near_plane=0.001, far_plane=2.5, eps2d=0.3, backend="cann", reference_max_pairs=2_000_000):
        super().__init__()
        if sh_degree not in (0, 1, 2, 3):
            raise ValueError("Supported SH degrees: 0 through 3")
        if not 0 < near_plane < far_plane or eps2d <= 0:
            raise ValueError("Require 0 < near < far and eps2d > 0")
        if backend not in ("cann", "reference"):
            raise ValueError("backend must be cann or reference")
        self.sh_degree = sh_degree
        self.near_plane, self.far_plane, self.eps2d = near_plane, far_plane, eps2d
        self.backend = backend
        self.reference_max_pairs = reference_max_pairs
        self._geometry = {}

    def forward(self, gaussians, c2ws, intr, *, width, height, background=(1., 1., 1.)):
        return self._render(gaussians, c2ws, intr, width=width, height=height, background=background)

    def render_precomputed(self, means, covariance, colors, opacities, c2ws, intr, *, width, height, background=(1., 1., 1.)):
        """World covariance / RGB / probability opacity, without activation or factorization."""
        return self._render(dict(xyz=means, covariance=covariance, colors=colors, opacity=opacities),
                            c2ws, intr, width=width, height=height, background=background, precomputed=True)

    def _render(self, gaussians, c2ws, intr, *, width, height, background, precomputed=False):
        means = gaussians["xyz"]
        if means.ndim != 3 or means.shape[-1] != 3:
            raise ValueError("Gaussian xyz must have shape [B,N,3]")
        B, N, _ = means.shape
        shapes = {"feature": (B, N, (self.sh_degree+1)**2, 3), "scale": (B, N, 3), "rotation": (B, N, 4), "opacity": (B, N, 1)}
        if precomputed:
            shapes = {'covariance': (B, N, 3, 3), 'colors': (B, N, 3), 'opacity': (B, N, 1)}
        for key, shape in shapes.items():
            tensor = gaussians[key]
            if tensor.shape != shape or tensor.device != means.device or not tensor.is_floating_point():
                raise ValueError(f"{key} must have shape {shape}, floating dtype and the Gaussian device")
        if precomputed:
            if not all(bool(torch.isfinite(gaussians[k]).all()) for k in ('xyz',*shapes)):
                raise ValueError('Precomputed Gaussian inputs must be finite')
            if bool(((gaussians['opacity']<0)|(gaussians['opacity']>1)).any()):
                raise ValueError('Precomputed opacity must be a probability in [0,1]')
        if c2ws.ndim != 4 or c2ws.shape[0] != B or c2ws.shape[-2:] != (4, 4) or intr.shape != (*c2ws.shape[:2], 4):
            raise ValueError("Target c2ws/intr must be [B,T,4,4] / [B,T,4]")
        if c2ws.device != means.device or intr.device != means.device:
            raise ValueError("Cameras and Gaussians must share a device")
        if width <= 0 or height <= 0 or c2ws.shape[1] == 0:
            raise ValueError("Positive image dimensions and at least one target camera are required")
        if self.backend == "cann" and means.device.type != "npu":
            raise ValueError("CANN backend requires NPU tensors; select reference explicitly for CPU")
        if self.backend == "reference" and N*width*height > self.reference_max_pairs:
            raise ValueError("Dense reference scale limit exceeded; use CANN for real scenes")
        # Keep double for CPU numerical checking; native rendering is always FP32.
        dtype = torch.float64 if self.backend == "reference" and means.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=means.device.type, enabled=False):
            g = {k: v.to(dtype) for k, v in gaussians.items() if k in ("xyz", *shapes)}
            cameras, intrinsics = c2ws.to(dtype), intr.to(dtype)
            if not torch.isfinite(cameras).all() or not torch.isfinite(intrinsics).all() or (intrinsics[..., :2] <= 0).any():
                raise ValueError("Target cameras must be finite with positive fx/fy")
            rotations = cameras[..., :3, :3]
            identity = torch.eye(3, dtype=dtype, device=means.device)
            if (rotations @ rotations.transpose(-1, -2) - identity).abs().max() > 1e-4:
                raise ValueError("Target c2ws must be rigid transforms with orthonormal rotation")
            last_row = cameras.new_tensor((0., 0., 0., 1.))
            if (cameras[..., 3, :] - last_row).abs().max() > 1e-5:
                raise ValueError("Target c2ws last row must be [0,0,0,1]")
            bg = torch.as_tensor(background, dtype=dtype, device=means.device)
            if bg.shape != (3,):
                raise ValueError("background must contain three RGB values")
            T = cameras.shape[1]
            rgbs, alphas, visible, intersections = [], [], [], []
            for batch in range(B):
                item = {k: v[batch] for k, v in g.items()}
                for view in range(T):
                    p = project(item, cameras[batch, view], intrinsics[batch, view], width, height,
                                self.near_plane, self.far_plane, self.sh_degree, self.eps2d, precomputed=precomputed)
                    count = p["means2d"].shape[0]
                    visible.append(count)
                    hits = None
                    if count == 0:
                        zero = sum(v.sum()*0 for v in item.values())
                        rgb, alpha = zero.expand(3, height, width), zero.expand(1, height, width)
                    elif self.backend == "reference":
                        rgb, alpha = reference_composite(p, width, height)
                    else:
                        from .native import render, tile_geometry
                        key = (width, height, means.device)
                        if key not in self._geometry:
                            self._geometry[key] = tile_geometry(width, height, means.device)
                        rgb, alpha, hits = render(p, self._geometry[key], width, height)
                    intersections.append(hits)
                    rgbs.append(rgb + (1-alpha)*bg[:, None, None])
                    alphas.append(alpha)
            return RenderResult(
                torch.stack(rgbs).reshape(B, T, 3, height, width),
                torch.stack(alphas).reshape(B, T, 1, height, width),
                {"backend": self.backend, "input_gaussians_per_scene": N, "visible_per_view": visible,
                 "tile_intersections_per_view": intersections, "alpha_threshold": 1/255,
                 "alpha_cap": 0.999, "transmittance_threshold": 1e-4,
                 "cpu_work": ["tile-count scheduling metadata"] if self.backend == "cann" else ["explicit reference backend"]},
            )
