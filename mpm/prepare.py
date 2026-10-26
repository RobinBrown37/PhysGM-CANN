"""Explicit, reproducible CPU initialization; no Taichi/GPU filling dependency."""

import hashlib
import itertools
import json
import time

import numpy as np
import torch
from export import physical_values
from rendering.geometry import quaternion_matrix

MAT_MAP = dict(
    Wood="metal",
    Metal="metal",
    Plastic="metal",
    Glass="metal",
    Fabric="foam",
    Leather="foam",
    Ceramic="metal",
    Stone="metal",
    Rubber="jelly",
    Paper="foam",
    Sand="sand",
    Snow="snow",
    Plasticine="plasticine",
    Foam="foam",
)


def rotation(degree, axis):
    angle = degree / 180 * 3.1415926
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    if axis == 2:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError("Invalid rotation axis")


def particle_volume(x, n, dx, uniform=False):
    cell = np.floor(x / dx).astype(np.int64)
    if np.any((cell < 0) | (cell >= n)):
        raise ValueError("Initial particle outside volume grid")
    ids = (cell[:, 0] * n + cell[:, 1]) * n + cell[:, 2]
    counts = np.bincount(ids, minlength=n**3)
    volume = (dx**3 / counts[ids]).astype(np.float32)
    return np.full_like(volume, volume.mean()) if uniform else volume


def ray_arrays(occupied, axis, positive):
    """Future hit and false->true transition counts, excluding the origin cell."""
    a = np.moveaxis(occupied, axis, 0)
    if not positive:
        a = a[::-1]
    hit = np.zeros_like(a)
    count = np.zeros(a.shape, dtype=np.int32)
    # A candidate origin is empty in the original kernel. The first future
    # occupied cell starts a run even when density at the origin is nonzero.
    for i in range(len(a) - 2, -1, -1):
        hit[i] = hit[i + 1] | a[i + 1]
        count[i] = count[i + 1] + a[i + 1].astype(np.int32)
        if i + 2 < len(a):
            count[i] -= (a[i + 1] & a[i + 2]).astype(np.int32)
    if not positive:
        hit = hit[::-1]
        count = count[::-1]
    return np.moveaxis(hit, 0, axis), np.moveaxis(count, 0, axis)


def fill_particles(x, opacity, cov, cfg, grid_lim, seed):
    """Port of density / dense fill / constrained smooth / ray internal fill."""
    started = time.perf_counter()
    n = int(cfg.get("n_grid", 200))
    dx = grid_lim / n
    original = x
    origin = np.zeros(3, dtype=np.float32)
    boundary = cfg.get("boundary")
    if boundary is not None:
        lo = np.array(boundary[::2], dtype=np.float32)
        hi = np.array(boundary[1::2], dtype=np.float32)
        mask = ((x > lo) & (x < hi)).all(-1)
        x = x[mask] - lo
        opacity = opacity[mask]
        cov = cov[mask]
        dx = float((hi - lo).max()) / n
        origin = lo
    base = np.floor(x / dx).astype(np.int64)
    if np.any((base < 0) | (base >= n)):
        raise ValueError("Filling input outside grid")
    grid = np.bincount(
        (base[:, 0] * n + base[:, 1]) * n + base[:, 2], minlength=n**3
    ).reshape(n, n, n)
    density = np.zeros(n**3, dtype=np.float32)
    eigen, q = np.linalg.eigh(cov)
    eigen = np.maximum(eigen, 1.0e-8)
    inverse = (q * (1 / eigen)[:, None, :]) @ q.transpose(0, 2, 1)
    radius = np.ceil(np.sqrt(eigen).max(-1) / dx).astype(np.int64)
    corners = np.array(list(itertools.product(range(2), repeat=3)), dtype=np.float32)
    # Group by support radius, then bounded batches. Every original cell and
    # all eight corner samples are retained; no density truncation/downsampling.
    for r in np.unique(radius):
        chosen = np.flatnonzero(radius == r)
        extent = min(int(r), n)
        offsets = np.array(
            list(itertools.product(range(-extent, extent + 1), repeat=3)),
            dtype=np.int64,
        )
        chunk = max(1, min(2048, 500000 // max(len(offsets) * 8, 1)))
        for start in range(0, len(chosen), chunk):
            ids = chosen[start : start + chunk]
            cells = base[ids, None, :] + offsets[None]
            valid = ((cells >= 0) & (cells < n)).all(-1)
            pi, oi = np.nonzero(valid)
            cell = cells[pi, oi]
            particle = ids[pi]
            weight = np.zeros(len(cell), dtype=np.float32)
            for corner in corners:
                dist = x[particle] - (cell + corner) * dx
                weight += np.exp(
                    -0.5 * np.einsum("ni,nij,nj->n", dist, inverse[particle], dist)
                ).astype(np.float32)
            values = opacity[particle] * weight / 8
            flat = (cell[:, 0] * n + cell[:, 1]) * n + cell[:, 2]
            np.add.at(density, flat, values)
    density = density.reshape(n, n, n)
    max_cell = int(
        cfg.get("max_partciels_per_cell", cfg.get("max_particles_per_cell", 1))
    )
    dense = (density > cfg.get("density_threshold", 5.0)) & (grid < max_cell)
    dense_cells = np.argwhere(dense)
    dense_counts = max_cell - grid[dense]
    grid[dense] = max_cell
    smooth_seconds = 0.0
    if cfg.get("smooth", False):
        import mcubes

        st = time.perf_counter()
        density = mcubes.smooth(density, method="constrained", max_iters=500).astype(
            np.float32
        )
        smooth_seconds = time.perf_counter() - st
    occupied = density > cfg.get("search_threshold", 3.0)
    rays = [ray_arrays(occupied, i // 2, i % 2 == 0) for i in range(6)]
    hit = np.ones_like(occupied)
    for direction, (h, _) in enumerate(rays):
        if direction != cfg.get("search_exclude_direction", 5):
            hit &= h
    direction = int(cfg.get("ray_cast_direction", 4))
    odd = (
        np.ones_like(hit) if direction not in range(6) else rays[direction][1] % 2 == 1
    )
    internal = (grid == 0) & hit & odd
    cells = np.concatenate(
        (
            np.repeat(dense_cells, dense_counts, axis=0),
            np.repeat(np.argwhere(internal), max_cell, axis=0),
        ),
        axis=0,
    )
    if len(cells) > int(cfg.get("max_particles_num", 2000000)):
        raise RuntimeError(
            f"Filling requires {len(cells)} new particles, exceeding configured capacity"
        )
    rng = np.random.default_rng(seed)
    added = ((cells + rng.random(cells.shape, dtype=np.float32)) * dx + origin).astype(
        np.float32
    )
    return np.concatenate((original, added)), dict(
        cpu_seconds=time.perf_counter() - started,
        smooth_seconds=smooth_seconds,
        gaussian_density_inputs=len(x),
        dense_added=int(dense_counts.sum()),
        internal_added=int(internal.sum() * max_cell),
        total_added=len(added),
        seed=seed,
        support_radius_max=int(radius.max(initial=0)),
        algorithm="upstream density/corners + constrained smooth + directional ray filling; NumPy seeded RNG",
    )


def prepare_prediction(pack, config, seed=2026):
    started = time.perf_counter()
    cfg = json.loads(json.dumps(config))
    pred = pack["prediction"]
    g = pred.get("gaussians_unpruned", pred["gaussians"])
    g = {k: v[0].detach().cpu().float() for k, v in g.items()}
    physical = physical_values(pred)
    cfg.setdefault("E", physical["E"])
    cfg.setdefault("nu", physical["nu"])
    cfg.setdefault("material", MAT_MAP[physical["material"]])
    cfg.setdefault("n_grid", 50)
    cfg.setdefault("grid_lim", 2.0)
    cfg.setdefault("density", 200.0)
    opacity = g["opacity"][:, 0].sigmoid()
    keep = opacity > cfg.get("opacity_threshold", 0.02)
    ids = torch.arange(len(keep))[keep]
    g = {k: v[keep] for k, v in g.items()}
    opacity = opacity[keep].numpy()
    xyz = g["xyz"].numpy()
    rot = quaternion_matrix(g["rotation"]).numpy()
    matrix = rot * g["scale"].exp().numpy()[:, None, :]
    cov = matrix @ matrix.transpose(0, 2, 1)
    transform = rotation(90.0, 0)
    degrees = cfg.get("rotation_degree", [])
    axes = cfg.get("rotation_axis", [])
    if len(degrees) != len(axes):
        raise ValueError("Rotation axes/degrees length mismatch")
    for degree, axis in zip(degrees, axes):
        transform = rotation(degree, axis) @ transform
    pos = xyz @ transform.T
    cov = transform @ cov @ transform.T
    static = None
    if cfg.get("sim_area") is not None:
        area = np.array(cfg["sim_area"])
        selected = ((pos > area[::2]) & (pos < area[1::2])).all(-1)
        # Keep unselected Gaussians static in original camera/world coordinates.
        world_cov = transform.T @ cov[~selected] @ transform
        static = dict(
            xyz=torch.from_numpy(xyz[~selected]),
            covariance=torch.from_numpy(world_cov),
            opacity=torch.from_numpy(opacity[~selected, None]),
            feature=g["feature"][~selected],
        )
        pos = pos[selected]
        cov = cov[selected]
        opacity = opacity[selected]
        ids = ids[selected]
        g = {k: v[selected] for k, v in g.items()}
    if not len(pos):
        raise ValueError("No Gaussians selected for simulation")
    lo = pos.min(0)
    hi = pos.max(0)
    center = (lo + hi) / 2
    scale = float(cfg.get("scale", 1.0)) / (hi - lo).max()
    pos = ((pos - center) * scale + 1).astype(np.float32)
    cov = (cov * scale**2).astype(np.float32)
    gs_count = len(pos)
    filling = {}
    if cfg.get("particle_filling") is not None:
        pos, filling = fill_particles(
            pos, opacity, cov, cfg["particle_filling"], cfg["grid_lim"], seed
        )
    volume = particle_volume(
        pos, cfg["n_grid"], cfg["grid_lim"] / cfg["n_grid"], cfg["material"] == "sand"
    )
    initial_cov = np.zeros((len(pos), 3, 3), dtype=np.float32)
    initial_cov[:gs_count] = cov
    metadata = dict(
        gaussian_count=gs_count,
        simulation_particles=len(pos),
        gaussian_ids=ids,
        physical_export=physical,
        world_to_sim_rotation=torch.from_numpy(transform),
        normalization_center=torch.from_numpy(center),
        normalization_scale=float(scale),
        feature=g["feature"],
        opacity=torch.from_numpy(opacity[:, None]),
        static_gaussians=static,
        camera_c2ws=pack["input"]["input_c2ws"][:, :1].cpu(),
        camera_intr=pack["input"]["input_intr"][:, :1].cpu(),
        width=int(pack["input"]["input_images"].shape[-1]),
        height=int(pack["input"]["input_images"].shape[-2]),
        filling=filling,
        cpu_initialization_seconds=time.perf_counter() - started,
        configuration_sha256=hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest(),
    )
    return dict(
        format="physgm_mpm_v1",
        config=cfg,
        time=0.0,
        steps=0,
        state=dict(
            x=torch.from_numpy(pos),
            volume=torch.from_numpy(volume),
            initial_cov=torch.from_numpy(initial_cov),
        ),
        metadata=metadata,
    )
