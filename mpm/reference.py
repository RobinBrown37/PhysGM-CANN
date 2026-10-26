"""Independent scalar NumPy FP64 oracle for small scenes, never a runtime fallback.

Recomputes SVD after the return map as the Warp source does, deliberately
independent of the device implementation's spectral reuse and batched stencil.
"""

import itertools
import math
import numpy as np


def signed_svd(f):
    u, s, vt = np.linalg.svd(f)
    v = vt.T
    du = -1 if np.linalg.det(u) < 0 else 1
    dv = -1 if np.linalg.det(v) < 0 else 1
    u[:, 2] *= du
    v[:, 2] *= dv
    s[2] *= du * dv
    return u, s, v


def material(f, mu, lam, y, cfg, dt):
    kind = cfg["material"]
    u, s, v = signed_svd(f)
    f = f.copy()
    if kind == "sand":
        eps = np.log(np.maximum(np.abs(s), 1.0e-14))
        tr = eps.sum()
        dev = eps - tr / 3
        norm = np.linalg.norm(dev)
        phi = math.sin(cfg.get("friction_angle", 25.0) / 180 * 3.14159265)
        alpha = math.sqrt(2 / 3) * 2 * phi / (3 - phi)
        delta = norm + (3 * lam + 2 * mu) / (2 * mu) * tr * alpha
        if delta > 0:
            f = (
                u @ v.T
                if tr > 0
                else u @ np.diag(np.exp(eps - dev * delta / norm)) @ v.T
            )
    elif kind in ("foam", "plasticine"):
        eps = np.log(np.maximum(s, 0.01))
        tr = eps.sum()
        dev = eps - tr / 3
        tau = 2 * mu * eps + lam * tr
        cond = tau - tau.mean()
        if kind == "foam":
            st = 2 * mu * dev
            sn = np.linalg.norm(st)
            excess = sn - math.sqrt(2 / 3) * y
            if excess > 0:
                mh = mu * np.mean(np.maximum(s, 0.01) ** 2)
                newnorm = sn - excess / (
                    1 + cfg.get("plastic_viscosity", 0.0) / (2 * mh * dt)
                )
                f = u @ np.diag(np.exp(st * (newnorm / sn) / (2 * mu) + tr / 3)) @ v.T
        elif np.linalg.norm(cond) > y and y > 0:
            norm = np.linalg.norm(dev) + 1.0e-6
            delta = norm - y / (2 * mu)
            corr = delta / norm * dev
            y -= cfg.get("softening", 0.1) * np.linalg.norm(corr)
            if y <= 0:
                mu = 0.0
                lam = 0.0
            f = u @ np.diag(np.exp(eps - corr)) @ v.T
            if cfg.get("hardening", 0) == 1:
                y += 2 * mu * cfg.get("xi", 0.0) * delta
    # The original metal and snow branches intentionally have no return map.
    u, s, v = signed_svd(f)
    j = np.linalg.det(f)
    if kind == "jelly":
        b = s * s
        tau = mu * j ** (-2 / 3) * (b - b.mean()) + lam / 2 * (j * j - 1)
        stress = u @ np.diag(tau) @ u.T
    elif kind == "metal":
        stress = 2 * mu * (f - u @ v.T) @ f.T + np.eye(3) * lam * j * (j - 1)
    else:
        eps = np.log(np.maximum(s, 0.01 if kind == "foam" else 1.0e-6))
        stress = u @ np.diag(2 * mu * eps + lam * eps.sum()) @ u.T
        if kind != "foam":
            tr = np.trace(stress)
            dev = stress - np.eye(3) * tr / 3
            norm = np.linalg.norm(dev)
            k = {"sand": 0.0, "snow": 1000.0, "plasticine": 5000.0}[kind]
            target = k - 0.16 * tr
            if norm + 0.16 * tr - k > 0 and target < norm:
                stress = dev * max(0.0, target) / (norm + 1.0e-9) + np.eye(3) * tr / 3
    return f, (stress + stress.T) / 2, mu, lam, y


def step(state, cfg, dt):
    """Small-scene oracle; supports free flight / box BC for core comparisons."""
    s = {k: v.copy() for k, v in state.items()}
    n = cfg["n_grid"]
    dx = cfg.get("grid_lim", 2.0) / n
    grid = np.zeros((n, n, n, 4), np.float64)
    stencils = []
    for p, x in enumerate(s["x"]):
        if s["selection"][p] != 0:
            stencils.append([])
            continue
        f, tau, mu, lam, y = material(
            s["F_trial"][p], s["mu"][p], s["lam"][p], s["yield_stress"][p], cfg, dt
        )
        s["F"][p] = f
        s["stress"][p] = tau
        s["mu"][p] = mu
        s["lam"][p] = lam
        s["yield_stress"][p] = y
        gp = x / dx
        base = np.trunc(gp - 0.5).astype(int)
        fx = gp - base
        w = np.stack(
            (0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2),
            axis=-1,
        )
        dw = np.stack((fx - 1.5, -2 * (fx - 1), fx - 0.5), axis=-1) / dx
        entries = []
        rpic = cfg.get("rpic_damping", 0.0)
        c = s["C"][p]
        c = np.zeros((3, 3)) if rpic < -0.001 else (1 - rpic) * c + rpic / 2 * (c - c.T)
        for i, j, k in itertools.product(range(3), repeat=3):
            idx = tuple(base + [i, j, k])
            dpos = np.array([i, j, k]) - fx
            wt = w[0, i] * w[1, j] * w[2, k]
            grad = np.array(
                [
                    dw[0, i] * w[1, j] * w[2, k],
                    w[0, i] * dw[1, j] * w[2, k],
                    w[0, i] * w[1, j] * dw[2, k],
                ]
            )
            mass = wt * s["mass"][p]
            grid[idx][0] += mass
            grid[idx][1:] += (
                mass * (s["v"][p] + c @ (dpos * dx)) - dt * s["volume"][p] * tau @ grad
            )
            entries.append((idx, wt, grad, dpos))
        stencils.append(entries)
    velocity = np.zeros((n, n, n, 3), np.float64)
    for idx in itertools.product(range(n), repeat=3):
        if grid[idx][0] > 1.0e-15:
            velocity[idx] = grid[idx][1:] / grid[idx][0] + dt * np.array(
                cfg.get("g", [0, 0, 0])
            )
    if cfg.get("grid_v_damping_scale", 1.1) < 1:
        velocity *= cfg["grid_v_damping_scale"]
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] != "bounding_box":
            raise ValueError(
                "Oracle core scenes support bounding_box only; other BCs have explicit analytical tests"
            )
        for idx in itertools.product(range(n), repeat=3):
            for axis in range(3):
                if (idx[axis] < 3 and velocity[idx][axis] < 0) or (
                    idx[axis] >= n - 3 and velocity[idx][axis] > 0
                ):
                    velocity[idx][axis] = 0
    for p, entries in enumerate(stencils):
        if not entries:
            continue
        v = np.zeros(3)
        c = np.zeros((3, 3))
        grad = np.zeros((3, 3))
        for idx, w, dw, dpos in entries:
            gv = velocity[idx]
            v += gv * w
            c += np.outer(gv, dpos) * w * 4 / dx
            grad += np.outer(gv, dw)
        s["x"][p] += dt * v
        s["v"][p] = v
        s["C"][p] = c
        s["F_trial"][p] = (np.eye(3) + dt * grad) @ s["F"][p]
        if cfg.get("update_cov_with_F", False):
            s["cov"][p] += dt * (grad @ s["cov"][p] + s["cov"][p] @ grad.T)
    return s, grid.reshape(-1, 4), velocity.reshape(-1, 3)
