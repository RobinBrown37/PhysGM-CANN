"""Time-varying Gaussian shading on the existing M2 rasterizer.

Inputs use world coordinates, full covariance, SH coefficients and probability
opacity. No Gaussian activations, covariance factorization, or MPM imports.
"""

import math

import torch
from torch import nn

from .geometry import sh_colors
from .static import RenderResult, StaticGaussianRenderer


class DynamicGaussianRenderer(nn.Module):
    def __init__(self, *, renderer=None, **static_options):
        super().__init__()
        if renderer is not None and static_options:
            raise ValueError("Pass an existing renderer or static options, not both")
        self.renderer = (
            renderer
            if renderer is not None
            else StaticGaussianRenderer(**static_options)
        )

    def forward(self, frame, c2ws, intr, *, width, height, background=(1.0, 1.0, 1.0)):
        xyz = frame["xyz"]
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("Dynamic xyz must be [B,N,3]")
        batch, count, _ = xyz.shape
        features = frame["feature"]
        if (
            features.ndim != 4
            or features.shape[:2] != (batch, count)
            or features.shape[-1] != 3
        ):
            raise ValueError("Dynamic feature must be [B,N,C,3]")
        coefficients = features.shape[-2]
        degree = math.isqrt(coefficients) - 1
        if degree not in range(4) or coefficients != (degree + 1) ** 2:
            raise ValueError("Dynamic SH coefficients must contain 1,4,9,16 bases")
        if (
            c2ws.ndim != 4
            or c2ws.shape[0] != batch
            or c2ws.shape[-2:] != (4, 4)
            or c2ws.shape[1] == 0
        ):
            raise ValueError("Dynamic cameras must be [B,T,4,4] with T>0")
        if intr.shape != (*c2ws.shape[:2], 4):
            raise ValueError("Dynamic intrinsics must be [B,T,4]")
        rotation = frame.get("rotation")
        if rotation is not None and (
            rotation.shape != (batch, count, 3, 3) or rotation.device != xyz.device
        ):
            raise ValueError(
                "Shading rotation must be [B,N,3,3] on the Gaussian device"
            )
        if features.device != xyz.device or c2ws.device != xyz.device:
            raise ValueError("Dynamic attributes and cameras must share a device")
        if not bool(torch.isfinite(features).all()) or (
            rotation is not None and not bool(torch.isfinite(rotation).all())
        ):
            raise ValueError("SH and shading rotation must be finite")
        views = []
        # Per-camera SH is the only extra stage. Projection/tile scheduling/sort/
        # composition stay in StaticGaussianRenderer.render_precomputed().
        dtype = (
            torch.float64
            if xyz.dtype == torch.float64 and self.renderer.backend == "reference"
            else torch.float32
        )
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            for view in range(c2ws.shape[1]):
                colors = torch.stack(
                    [
                        sh_colors(
                            features[b].to(dtype),
                            xyz[b].to(dtype),
                            c2ws[b, view, :3, 3].to(dtype),
                            degree,
                            direction_rotation=rotation[b].to(dtype)
                            if degree and rotation is not None
                            else None,
                        )
                        for b in range(batch)
                    ]
                )
                views.append(
                    self.renderer.render_precomputed(
                        xyz,
                        frame["covariance"],
                        colors,
                        frame["opacity"],
                        c2ws[:, view : view + 1],
                        intr[:, view : view + 1],
                        width=width,
                        height=height,
                        background=background,
                    )
                )
        stats = dict(views[0].stats)
        for key in ("visible_per_view", "tile_intersections_per_view"):
            stats[key] = [image.stats[key][b] for b in range(batch) for image in views]
        stats.update(dynamic=True, sh_degree=degree, covariance_factorization=False)
        return RenderResult(
            torch.cat([v.rgb for v in views], 1),
            torch.cat([v.alpha for v in views], 1),
            stats,
        )
