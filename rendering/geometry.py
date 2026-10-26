"""Differentiable pinhole Gaussian projection and real SH (wxyz convention).

Equations follow the gsplat pinhole model: https://github.com/nerfstudio-project/gsplat
All tensor arithmetic stays on the input device; sorting/culling is discrete.
"""
import torch
from torch.nn import functional as F


def quaternion_matrix(quaternion):
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack((
        1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y),
    ), -1).reshape(-1, 3, 3)


def sh_colors(features, means, camera_origin, degree, direction_rotation=None):
    direction = means - camera_origin
    if direction_rotation is not None:
        direction = (direction_rotation @ direction.unsqueeze(-1)).squeeze(-1)
    direction = F.normalize(direction, dim=-1)
    x, y, z = direction.unbind(-1)
    bases = [torch.ones_like(x) * 0.28209479177387814]
    if degree >= 1:
        c = 0.4886025119029199
        bases += [-c*y, c*z, -c*x]
    if degree >= 2:
        bases += [1.0925484305920792*x*y, -1.0925484305920792*y*z,
                  0.31539156525252005*(2*z*z-x*x-y*y),
                  -1.0925484305920792*x*z, 0.5462742152960396*(x*x-y*y)]
    if degree >= 3:
        bases += [-0.5900435899266435*y*(3*x*x-y*y), 2.890611442640554*x*y*z,
                  -0.4570457994644658*y*(4*z*z-x*x-y*y),
                  0.3731763325901154*z*(2*z*z-3*x*x-3*y*y),
                  -0.4570457994644658*x*(4*z*z-x*x-y*y),
                  1.445305721320277*z*(x*x-y*y), -0.5900435899266435*x*(x*x-3*y*y)]
    return ((features[:, :len(bases)] * torch.stack(bases, -1).unsqueeze(-1)).sum(1) + 0.5).clamp_min(0)


def project(gaussians, c2w, intr, width, height, near, far, degree, eps2d=0.3, precomputed=False):
    """Return compact visible attributes; indexing preserves active gradients."""
    means = gaussians["xyz"]
    # Camera poses are rigid transforms. An explicit inverse avoids the generic
    # linalg_inv_ex CPU fallback in TorchNPU and keeps the whole projection on NPU.
    R = c2w[:3, :3].T
    translation = -(R @ c2w[:3, 3])
    camera_means = means @ R.T + translation
    opacity = gaussians["opacity"].squeeze(-1)
    if not precomputed:
        opacity = opacity.sigmoid()
    # These are discrete visibility decisions, as in a splatting rasterizer.
    keep = ((camera_means[:, 2] > near) & (camera_means[:, 2] < far)
            & (opacity >= 1/255)).nonzero().squeeze(-1)
    camera_means = camera_means[keep]
    xyz = means[keep]
    if precomputed:
        covariance = gaussians['covariance'][keep]
    else:
        quats = gaussians["rotation"][keep]
        scales = gaussians["scale"][keep].exp()
        rotation = quaternion_matrix(quats)
        matrix = rotation * scales.unsqueeze(-2)
        covariance = matrix @ matrix.transpose(-1, -2)
    covariance = R @ covariance @ R.T
    tx, ty, tz = camera_means.unbind(-1)
    fx, fy, cx, cy = intr.unbind(-1)
    ux, uy = tx / tz, ty / tz
    clip_x = ux.clamp(min=-(cx/fx + 0.3 * width/(2*fx)), max=(width-cx)/fx + 0.3*width/(2*fx))
    clip_y = uy.clamp(min=-(cy/fy + 0.3 * height/(2*fy)), max=(height-cy)/fy + 0.3*height/(2*fy))
    zero = torch.zeros_like(tz)
    J = torch.stack((fx/tz, zero, -fx*clip_x/tz, zero, fy/tz, -fy*clip_y/tz), -1).reshape(-1, 2, 3)
    covariance2d = J @ covariance @ J.transpose(-1, -2)
    a = covariance2d[:, 0, 0] + eps2d
    b = covariance2d[:, 0, 1]
    d = covariance2d[:, 1, 1] + eps2d
    determinant = a*d-b*b
    conics = torch.stack((d, -b, a), -1) / determinant.unsqueeze(-1)
    means2d = torch.stack((fx*ux+cx, fy*uy+cy), -1)
    colors = gaussians['colors'][keep] if precomputed else sh_colors(gaussians["feature"][keep], xyz, c2w[:3, 3], degree)
    return {
        "means2d": means2d, "conics": conics, "opacity": opacity[keep],
        "colors": colors, "depths": tz,
        "covars2d": torch.stack((a, b, d), -1), "ids": keep,
    }
