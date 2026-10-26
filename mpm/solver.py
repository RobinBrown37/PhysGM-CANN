"""Forward APIC/RPIC MPM. Physics runs entirely on the selected tensor device."""

import copy
import math
import time
import torch
from .boundaries import Boundaries, box_mask
from .linalg import mm3, svd3
from .materials import MATERIALS, constitutive


def stencil(x, n_grid, dx):
    gp = x * (1 / dx)
    base = (gp - 0.5).trunc().to(torch.int64)  # Warp int conversion, not floor
    fx = gp - base.to(x.dtype)
    w = torch.stack(
        (
            0.5 * (1.5 - fx).square(),
            0.75 - (fx - 1).square(),
            0.5 * (fx - 0.5).square(),
        ),
        -1,
    )
    dw = torch.stack((fx - 1.5, -2 * (fx - 1), fx - 0.5), -1) / dx
    off = torch.cartesian_prod(*[torch.arange(3, device=x.device)] * 3)
    weights = w[:, 0, off[:, 0]] * w[:, 1, off[:, 1]] * w[:, 2, off[:, 2]]
    grads = torch.stack(
        (
            dw[:, 0, off[:, 0]] * w[:, 1, off[:, 1]] * w[:, 2, off[:, 2]],
            w[:, 0, off[:, 0]] * dw[:, 1, off[:, 1]] * w[:, 2, off[:, 2]],
            w[:, 0, off[:, 0]] * w[:, 1, off[:, 1]] * dw[:, 2, off[:, 2]],
        ),
        -1,
    )
    coords = base[:, None, :] + off
    ids = (coords[..., 0] * n_grid + coords[..., 1]) * n_grid + coords[..., 2]
    valid = ((coords >= 0) & (coords < n_grid)).all(-1).all(-1)
    return ids, weights, grads, off.to(x.dtype)[None] - fx[:, None], valid


class MPMSimulator:
    def __init__(
        self,
        *,
        device="cpu",
        dtype=torch.float32,
        svd_backend="native",
        transfer_backend="torch",
        physics_mode="upstream",
        validate=True,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        if self.device.type not in ("cpu", "npu") or dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("MPM supports CPU FP32/FP64 or NPU FP32")
        if self.device.type == "npu" and dtype != torch.float32:
            raise ValueError("NPU MPM requires FP32")
        if physics_mode not in ("upstream", "corrected"):
            raise ValueError("Invalid physics mode")
        if transfer_backend not in ("torch", "native"):
            raise ValueError("Invalid transfer backend")
        if transfer_backend == "native" and self.device.type != "npu":
            raise ValueError("Native transfers require NPU")
        self.svd_backend = svd_backend
        self.transfer_backend = transfer_backend
        self.physics_mode = physics_mode
        self.validate = validate
        self.time = 0.0
        self.steps = 0
        self.timings = {}
        self.state = {}

    @torch.no_grad()
    def load_state(self, package):
        if package.get("format") != "physgm_mpm_v1":
            raise ValueError("Expected physgm_mpm_v1 state")
        if package.get("physics_mode", self.physics_mode) != self.physics_mode and (
            package.get("steps", 0) != 0 or package.get("time", 0.0) != 0.0
        ):
            raise ValueError(
                "State physics mode mismatch; only an unstepped initial state can be shared"
            )
        self.cfg = copy.deepcopy(package["config"])
        c = self.cfg
        if c["material"] not in MATERIALS:
            raise ValueError("Unknown material")
        self.n_grid = int(c.get("n_grid", 50))
        self.dx = float(c.get("grid_lim", 2.0)) / self.n_grid
        if self.n_grid < 4 or self.dx <= 0:
            raise ValueError("Invalid grid")
        self.state = {
            k: v.to(
                device=self.device,
                dtype=self.dtype if v.is_floating_point() else v.dtype,
            ).clone()
            for k, v in package["state"].items()
        }
        s = self.state
        x = s["x"]
        n = len(x)
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        if x.shape != (n, 3) or n == 0:
            raise ValueError("Require nonempty [N,3] positions")

        def fill(name, value):
            if name not in s:
                s[name] = torch.full((n,), value, dtype=self.dtype, device=self.device)

        for name in ("v",):
            s.setdefault(name, torch.zeros_like(x))
        for name in ("C", "stress", "F"):
            s.setdefault(name, x.new_zeros(n, 3, 3))
        s.setdefault("F_trial", eye.expand(n, 3, 3).clone())
        s.setdefault("selection", torch.zeros(n, dtype=torch.int64, device=self.device))
        s.setdefault("particle_ids", torch.arange(n, device=self.device))
        fill("Jp", 0.0)
        s.setdefault("initial_cov", x.new_zeros(n, 3, 3))
        s.setdefault("cov", s["initial_cov"].clone())
        for name, default in (
            ("E", 1.0e5),
            ("nu", 0.4),
            ("density", 200.0),
            ("yield_stress", 0.0),
        ):
            fill(name, float(c.get(name, default)))
        if "mu" not in s:
            for p in c.get("additional_material_params", []):
                if any(k not in p for k in ("point", "size", "E", "nu")):
                    raise ValueError(
                        "Additional material parameters require point/size/E/nu"
                    )
                mask = box_mask(x, p["point"], p["size"])
                for k in ("E", "nu", "density"):
                    s[k] = torch.where(mask, float(p.get(k, c.get(k, 200.0))), s[k])
            if bool(((s["nu"] < 0) | (s["nu"] >= 0.5)).any()) or bool(
                (s["E"] <= 0).any()
            ):
                raise ValueError("Require E>0 and 0<=nu<0.5")
            s["mu"] = s["E"] / (2 * (1 + s["nu"]))
            s["lam"] = s["E"] * s["nu"] / ((1 + s["nu"]) * (1 - 2 * s["nu"]))
        if "volume" not in s:
            raise ValueError("Particle volume is required")
        s.setdefault("mass", s["density"] * s["volume"])
        for key, value in s.items():
            shape = (
                (n, 3)
                if key in ("x", "v")
                else (n, 3, 3)
                if key in ("C", "stress", "F", "F_trial", "initial_cov", "cov")
                else (n,)
            )
            if value.shape != shape:
                raise ValueError(
                    f"Invalid state shape for {key}: expected {shape}, got {value.shape}"
                )
        if not all(
            bool(torch.isfinite(v).all()) for v in s.values() if v.is_floating_point()
        ):
            raise ValueError("Nonfinite initial state")
        if bool(((s["nu"] < 0) | (s["nu"] >= 0.5)).any()) or bool((s["E"] <= 0).any()):
            raise ValueError("Require E>0 and 0<=nu<0.5")
        if bool((s["mass"] <= 0).any()) or bool((s["volume"] <= 0).any()):
            raise ValueError("Mass/volume must be positive")
        self.time = float(package.get("time", 0.0))
        self.steps = int(package.get("steps", 0))
        self.metadata = copy.deepcopy(package.get("metadata", {}))
        self.boundaries = Boundaries(
            c.get("boundary_conditions", []),
            x,
            self.n_grid,
            self.dx,
            self.physics_mode,
            c.get("substep_dt", 1.0e-4),
        )
        if "boundary_state" in package:
            self.boundaries.restore(package["boundary_state"])
        self.grid = x.new_zeros(self.n_grid**3, 4)
        self.grid_v = x.new_zeros(self.n_grid**3, 3)
        return self

    def sync(self):
        if self.device.type == "npu":
            torch.npu.synchronize(self.device)

    @torch.no_grad()
    def step(self, dt, *, profile=False):
        # A future caller may invoke physics inside a network BF16 context.
        # The solver's FP32 contract is independent of that outer context.
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._step(dt, profile=profile)

    def _step(self, dt, *, profile=False):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        s = self.state
        c = self.cfg
        active = s["selection"] == 0

        def mark(name, started):
            if profile:
                self.sync()
                self.timings.setdefault(name, []).append(time.perf_counter() - started)
            return time.perf_counter()

        if profile:
            self.sync()
        started = time.perf_counter()
        s["v"] = self.boundaries.particles(s["x"], s["v"], s["mass"], self.time, dt)
        f, stress, mu, lam, y = constitutive(
            s["F_trial"], s["mu"], s["lam"], s["yield_stress"], c, dt, self.svd_backend
        )
        for key, val in (
            ("F", f),
            ("stress", stress),
            ("mu", mu),
            ("lam", lam),
            ("yield_stress", y),
        ):
            mask = active.reshape(-1, *([1] * (val.ndim - 1)))
            s[key] = torch.where(mask, val, s[key])
        started = mark("constitutive", started)
        if self.transfer_backend == "native":
            from .native import p2g

            base = (s["x"] * (1 / self.dx) - 0.5).trunc()
            valid = ((base >= 0) & (base + 2 < self.n_grid)).all(-1)
            if self.validate and not bool((valid | ~active).all()):
                raise RuntimeError(
                    "Active particle stencil leaves the grid; state was not clamped"
                )
            indices, stride = p2g(
                s, self.grid, self.n_grid, self.dx, dt, c.get("rpic_damping", 0.0)
            )
        else:
            ids, w, dw, dpos, valid = stencil(s["x"], self.n_grid, self.dx)
            if self.validate and not bool((valid | ~active).all()):
                raise RuntimeError(
                    "Active particle stencil leaves the grid; state was not clamped"
                )
            ids = ids.clamp(0, self.n_grid**3 - 1)
            self.grid.zero_()
            rpic = c.get("rpic_damping", 0.0)
            C = (
                torch.zeros_like(s["C"])
                if rpic < -0.001
                else (1 - rpic) * s["C"]
                + rpic / 2 * (s["C"] - s["C"].transpose(-1, -2))
            )
            mass = s["mass"] * active
            mom = (w * mass[:, None])[:, :, None] * (
                s["v"][:, None, :] + torch.einsum("pij,pkj->pki", C, dpos * self.dx)
            )
            mom = (
                mom
                - dt
                * s["volume"][:, None, None]
                * torch.einsum("pij,pkj->pki", s["stress"], dw)
                * active[:, None, None]
            )
            updates = torch.cat(((w * mass[:, None])[:, :, None], mom), -1)
            self.grid.index_add_(0, ids.reshape(-1), updates.reshape(-1, 4))
        started = mark("p2g", started)
        gm = self.grid[:, 0]
        self.grid_v = torch.where(
            (gm > 1.0e-15)[:, None],
            self.grid[:, 1:] / gm.clamp_min(1.0e-30)[:, None]
            + dt * s["x"].new_tensor(c.get("g", [0.0, 0.0, 0.0])),
            0.0,
        )
        damping = c.get("grid_v_damping_scale", 1.1)
        if damping < 1:
            self.grid_v *= damping
        self.grid_v = self.boundaries.grid(self.grid_v, self.time, dt)
        started = mark("grid_and_boundaries", started)
        if self.transfer_backend == "native":
            from .native import g2p

            grad = g2p(s, self.grid_v, indices, stride, self.dx, dt)
        else:
            gv = self.grid_v[ids]
            velocity = (gv * w[:, :, None]).sum(1)
            new_c = 4 / self.dx * torch.einsum("pki,pkj,pk->pij", gv, dpos, w)
            grad = torch.einsum("pki,pkj->pij", gv, dw)
            s["x"] = torch.where(active[:, None], s["x"] + dt * velocity, s["x"])
            s["v"] = torch.where(active[:, None], velocity, s["v"])
            s["C"] = torch.where(active[:, None, None], new_c, s["C"])
            next_f = mm3(
                torch.eye(3, device=self.device, dtype=self.dtype) + dt * grad, s["F"]
            )
            s["F_trial"] = torch.where(active[:, None, None], next_f, s["F_trial"])
        if c.get("update_cov_with_F", False):
            next_cov = s["cov"] + dt * (
                mm3(grad, s["cov"]) + mm3(s["cov"], grad.transpose(-1, -2))
            )
            s["cov"] = torch.where(active[:, None, None], next_cov, s["cov"])
        mark("g2p", started)
        self.time += dt
        self.steps += 1

    @torch.no_grad()
    def covariance(self):
        s = self.state
        return (
            s["cov"]
            if self.cfg.get("update_cov_with_F", False)
            else mm3(
                mm3(s["F_trial"], s["initial_cov"]), s["F_trial"].transpose(-1, -2)
            )
        )

    @torch.no_grad()
    def rotation(self):
        u, _, v = svd3(self.state["F_trial"], self.svd_backend)
        return mm3(u, v.transpose(-1, -2)).transpose(-1, -2)

    @torch.no_grad()
    def export_state(self, cpu=False):
        return {
            "format": "physgm_mpm_v1",
            "config": copy.deepcopy(self.cfg),
            "physics_mode": self.physics_mode,
            "time": self.time,
            "steps": self.steps,
            "metadata": copy.deepcopy(self.metadata),
            "state": {
                k: (v.cpu().clone() if cpu else v.clone())
                for k, v in self.state.items()
            },
            "boundary_state": self.boundaries.export(),
        }
