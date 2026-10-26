"""Ordered upstream particle/grid boundary conditions and explicit bug fixes."""

import copy
import numpy as np
import torch


def vector(x, ref):
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def box_mask(x, point, size):
    return ((x - vector(point, x)).abs() < vector(size, x)).all(-1)


class Boundaries:
    def __init__(self, configs, initial_x, grid_n, dx, mode, dt):
        self.mode = mode
        self.items = []
        self.grid_positions = torch.stack(
            torch.meshgrid(
                *[
                    torch.arange(grid_n, device=initial_x.device, dtype=initial_x.dtype)
                    * dx
                ]
                * 3,
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 3)
        self.grid_n, self.dx = grid_n, dx
        indices = torch.stack(
            torch.meshgrid(
                *[torch.arange(grid_n, device=initial_x.device)] * 3, indexing="ij"
            ),
            -1,
        ).reshape(-1, 3)
        self._grid_low = indices < 3
        self._grid_high = indices >= grid_n - 3
        for c in copy.deepcopy(configs):
            self.add(c, initial_x, dt)

    def add(self, c, x, dt):
        kind = c["type"]
        c.setdefault("start_time", 0.0)
        c.setdefault("end_time", 999.0)
        if kind == "release_particles_sequentially":
            axes = [i for i, v in enumerate(c["normal"]) if v != 0]
            if len(axes) != 1:
                raise ValueError("Sequential release requires an axis-aligned normal")
            axis = axes[0]
            layers = 50 if self.mode == "upstream" else int(c["num_layers"])
            if layers <= 0 or c["end_time"] <= c["start_time"]:
                raise ValueError("Invalid release layers/time")
            length = abs(c["start_position"] - c["end_position"]) / layers
            for i in range(layers):
                point = [1.0, 1.0, 1.0]
                point[axis] = c["end_position"]
                size = [1.0, 1.0, 1.0]
                size[axis] = length * (layers - i)
                end = (
                    c["end_time"] * (i + 1) / layers
                    if self.mode == "upstream"
                    else c["start_time"]
                    + (c["end_time"] - c["start_time"]) * (i + 1) / layers
                )
                self.add(
                    dict(
                        type="enforce_particle_translation",
                        point=point,
                        size=size,
                        velocity=[0, 0, 0],
                        start_time=c["start_time"],
                        end_time=end,
                    ),
                    x,
                    dt,
                )
            return
        if kind in ("particle_impulse", "enforce_particle_translation"):
            c.setdefault("point", [1.0, 1.0, 1.0])
            c.setdefault("size", [1.0, 1.0, 1.0])
            c["_mask"] = box_mask(x, c["point"], c["size"])
            if kind == "particle_impulse":
                c["end_time"] = c["start_time"] + dt * c.get("num_dt", 1)
        elif kind == "enforce_particle_velocity_rotation":
            n = vector(c["normal"], x)
            n = n / torch.linalg.vector_norm(n)
            # Preserve the original basis convention. The tangent velocity can
            # then be evaluated algebraically, including at the cylinder axis.
            h = vector([1.0, 1.0, 1.0], x)
            if abs(float(n @ h)) < 0.01:
                h = vector([0.72, 0.37, -0.67], x)
            h = h - (h @ n) * n
            if float(torch.linalg.vector_norm(h)) < 1.0e-12:
                raise ValueError("Upstream cylinder basis is singular for this normal")
            h = h / torch.linalg.vector_norm(h)
            h2 = torch.linalg.cross(h, n)
            offset = x - vector(c["point"], x)
            axial = offset @ n
            radial = torch.linalg.vector_norm(offset - axial[:, None] * n, dim=-1)
            c["_mask"] = (axial.abs() < c["half_height_and_radius"][0]) & (
                radial < c["half_height_and_radius"][1]
            )
            c["_normal"], c["_h1"], c["_h2"] = n, h, h2
        elif kind == "surface_collider":
            n = vector(c["normal"], x)
            if float(torch.linalg.vector_norm(n)) == 0:
                raise ValueError("Surface normal must be nonzero")
            c["_normal"] = n / torch.linalg.vector_norm(n)
            if c["surface"] == "sticky" and c.get("friction", 0) != 0:
                raise ValueError("Sticky friction must be zero")
            c["_mask"] = (
                (self.grid_positions - vector(c["point"], x)) @ c["_normal"]
            ) < 0
        elif kind not in ("bounding_box", "cuboid"):
            raise ValueError(f"Unknown boundary type: {kind}")
        c["start_time"] = float(np.float32(c["start_time"]))
        c["end_time"] = float(np.float32(c["end_time"]))
        self.items.append(c)

    def particles(self, x, velocity, mass, time, dt):
        time = float(np.float32(time))
        # Upstream runs all impulses before all velocity modifiers.
        for kind_group in (
            ("particle_impulse",),
            ("enforce_particle_translation", "enforce_particle_velocity_rotation"),
        ):
            for c in self.items:
                kind = c["type"]
                if (
                    kind not in kind_group
                    or not c["start_time"] <= time < c["end_time"]
                ):
                    continue
                if kind == "particle_impulse":
                    velocity = torch.where(
                        c["_mask"][:, None],
                        velocity + dt * vector(c["force"], x) / mass[:, None],
                        velocity,
                    )
                elif kind == "enforce_particle_translation":
                    velocity = torch.where(
                        c["_mask"][:, None], vector(c["velocity"], x), velocity
                    )
                else:
                    off = x - vector(c["point"], x)
                    target = (
                        -((off @ c["_h2"])[:, None]) * c["_h1"]
                        + (off @ c["_h1"])[:, None] * c["_h2"]
                    ) * c["rotation_scale"] + c["_normal"] * c["translation_scale"]
                    velocity = torch.where(c["_mask"][:, None], target, velocity)
        return velocity

    def grid(self, velocity, time, dt):
        time = float(np.float32(time))
        pos = self.grid_positions
        for c in self.items:
            kind = c["type"]
            active = c["start_time"] <= time < c["end_time"]
            if kind == "bounding_box" and active:
                mask = (self._grid_low & (velocity < 0)) | (
                    self._grid_high & (velocity > 0)
                )
                velocity = torch.where(mask, 0.0, velocity)
            elif kind == "cuboid":
                if active:
                    velocity = torch.where(
                        box_mask(pos, c["point"], c["size"])[:, None],
                        vector(c["velocity"], pos),
                        velocity,
                    )
                    c["point"] = [a + dt * b for a, b in zip(c["point"], c["velocity"])]
                elif c.get("reset", 0) == 1 and time < c["end_time"] + 15 * dt:
                    # Original code resets the entire grid, including pre-start.
                    velocity = torch.zeros_like(velocity)
            elif kind == "surface_collider" and active:
                surface = c["surface"]
                mask = c["_mask"]
                if surface == "cut":
                    z = pos[:, 2]
                    target = velocity * vector([0.3, 0.0, 0.3], pos)
                    target = torch.where(((z < 0.4) | (z > 0.53))[:, None], 0.0, target)
                elif surface == "sticky" or self.mode == "upstream":
                    target = torch.zeros_like(velocity)
                else:
                    n = c["_normal"]
                    vn = (velocity * n).sum(-1)
                    target = (
                        velocity
                        - (vn if surface == "slip" else vn.clamp_max(0))[:, None] * n
                    )
                    length = torch.linalg.vector_norm(target, dim=-1)
                    scale = (length + vn * c.get("friction", 0.0)).clamp_min(
                        0
                    ) / length.clamp_min(1.0e-20)
                    target = torch.where(
                        ((vn < 0) & (length > 1.0e-20))[:, None],
                        target * scale[:, None],
                        target,
                    )
                velocity = torch.where(mask[:, None], target, velocity)
        return velocity

    def export(self):
        return [
            {
                k: (v.cpu() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                for k, v in c.items()
            }
            for c in self.items
        ]

    def restore(self, items):
        self.items = [
            {
                k: (
                    v.to(self.grid_positions.device)
                    if isinstance(v, torch.Tensor)
                    else copy.deepcopy(v)
                )
                for k, v in c.items()
            }
            for c in items
        ]
