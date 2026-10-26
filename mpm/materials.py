"""Constitutive equations ported from PhysGM 7722175 mpm_utils.py.

The names describe the upstream branches, not textbook replacements. In
particular metal has no return map and snow uses the upstream DP stress.
"""

import math
import torch
from .linalg import det3, mm3, svd3

MATERIALS = ("jelly", "metal", "sand", "foam", "snow", "plasticine")


def constitutive(f_trial, mu, lam, yield_stress, cfg, dt, svd_backend="native"):
    material = cfg["material"]
    u, s, v = svd3(f_trial, svd_backend)
    f = f_trial
    # Return maps only change principal stretches: retain the spectral basis.
    if material in ("sand", "foam", "plasticine"):
        eps = (
            s.abs().clamp_min(1.0e-14).log()
            if material == "sand"
            else s.clamp_min(0.01).log()
        )
        tr = eps.sum(-1, keepdim=True)
        dev = eps - tr / 3
        norm = torch.linalg.vector_norm(dev, dim=-1)
        m, l = mu, lam
        if material == "sand":
            phi = math.sin(cfg.get("friction_angle", 25.0) / 180.0 * 3.14159265)
            alpha = math.sqrt(2 / 3) * 2 * phi / (3 - phi)
            delta = norm + (3 * l + 2 * m) / (2 * m) * tr[:, 0] * alpha
            projected = (eps - dev * (delta / norm.clamp_min(1.0e-30))[:, None]).exp()
            new_s = torch.where((tr[:, 0] > 0)[:, None], torch.ones_like(s), projected)
            active = delta > 0
        elif material == "foam":
            trial = 2 * m[:, None] * dev
            trial_norm = torch.linalg.vector_norm(trial, dim=-1)
            y = trial_norm - math.sqrt(2 / 3) * yield_stress
            mu_hat = m * s.clamp_min(0.01).square().mean(-1)
            new_norm = trial_norm - y / (
                1 + cfg.get("plastic_viscosity", 0.0) / (2 * mu_hat * dt)
            )
            new_eps = (
                trial
                * (new_norm / trial_norm.clamp_min(1.0e-30))[:, None]
                / (2 * m[:, None])
                + tr / 3
            )
            new_s = new_eps.exp()
            active = y > 0
        else:
            tau = 2 * m[:, None] * eps + l[:, None] * tr
            cond = tau - tau.mean(-1, keepdim=True)
            active = (torch.linalg.vector_norm(cond, dim=-1) > yield_stress) & (
                yield_stress > 0
            )
            norm_eps = norm + 1.0e-6
            # A damaged particle has mu=0 and an inactive map; prevent 0/0 in
            # unselected tensor branches without altering active calculations.
            delta = norm_eps - yield_stress / (2 * m.clamp_min(1.0e-30))
            correction = (delta / norm_eps)[:, None] * dev
            new_s = (eps - correction).exp()
            damaged_yield = yield_stress - cfg.get(
                "softening", 0.1
            ) * torch.linalg.vector_norm(correction, dim=-1)
            failed = active & (damaged_yield <= 0)
            mu = torch.where(failed, 0.0, mu)
            lam = torch.where(failed, 0.0, lam)
            if cfg.get("hardening", 0) == 1:
                damaged_yield = damaged_yield + 2 * mu * cfg.get("xi", 0.0) * delta
            yield_stress = torch.where(active, damaged_yield, yield_stress)
        old_s = s
        s = torch.where(active[:, None], new_s, s)
        # Apply the stretch correction to F_trial. Algebraically identical to
        # U diag(new_s) V^T, while avoiding reinjecting SVD reconstruction error
        # at every almost-elastic projection in a long FP32 trajectory.
        mapped = f_trial + mm3(u * (s - old_s)[:, None, :], v.transpose(-1, -2))
        f = torch.where(active[:, None, None], mapped, f_trial)

    identity = torch.eye(3, dtype=f.dtype, device=f.device)
    if material == "jelly":
        j = det3(f)
        b = s.square()
        principal = mu[:, None] * j[:, None].pow(-2 / 3) * (
            b - b.mean(-1, keepdim=True)
        ) + lam[:, None] / 2 * (j[:, None].square() - 1)
        stress = mm3(u * principal[:, None, :], u.transpose(-1, -2))
    elif material == "metal":
        j = det3(f)
        rotation = mm3(u, v.transpose(-1, -2))
        stress = (
            mm3(2 * mu[:, None, None] * (f - rotation), f.transpose(-1, -2))
            + identity * (lam * j * (j - 1))[:, None, None]
        )
    else:
        eps = s.clamp_min(0.01 if material == "foam" else 1.0e-6).log()
        principal = 2 * mu[:, None] * eps + lam[:, None] * eps.sum(-1, keepdim=True)
        trial = mm3(u * principal[:, None, :], u.transpose(-1, -2))
        if material == "foam":
            stress = trial
        else:
            trace = trial.diagonal(dim1=-2, dim2=-1).sum(-1)
            hydro = identity * (trace / 3)[:, None, None]
            dev = trial - hydro
            norm = torch.linalg.vector_norm(dev.reshape(-1, 9), dim=-1)
            cohesion = {"sand": 0.0, "snow": 1000.0, "plasticine": 5000.0}[material]
            target = cohesion - 0.16 * trace
            active = (norm + 0.16 * trace - cohesion > 0) & (target < norm)
            scale = target.clamp_min(0) / (norm + 1.0e-9)
            stress = torch.where(
                active[:, None, None], dev * scale[:, None, None] + hydro, trial
            )
    return f, (stress + stress.transpose(-1, -2)) / 2, mu, lam, yield_stress
