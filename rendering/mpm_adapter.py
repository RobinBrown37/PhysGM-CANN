"""Forward MPM -> world-space render frames; never used by offline replay."""

import torch

from .sequence import compose_frame


class MPMFrameAdapter:
    def __init__(self, metadata, device, dtype=torch.float32):
        self.metadata = metadata
        self.device, self.dtype = device, dtype
        self.count = int(metadata["gaussian_count"])

        def move(value):
            return value.to(
                device=device, dtype=dtype if value.is_floating_point() else value.dtype
            )

        self.rotation = move(metadata["world_to_sim_rotation"])
        self.center = move(metadata["normalization_center"])
        self.scale = float(metadata["normalization_scale"])
        feature, opacity = move(metadata["feature"]), move(metadata["opacity"])
        static = metadata.get("static_gaussians")
        static = {k: move(v) for k, v in static.items()} if static is not None else None
        ids = metadata.get("gaussian_ids", torch.arange(self.count)).to(device)
        if static is not None:
            static_ids = static.get(
                "gaussian_ids", -torch.arange(1, len(static["xyz"]) + 1, device=device)
            )
            ids = torch.cat((ids, static_ids))
        self.shared = dict(
            feature=feature,
            opacity=opacity,
            static=static,
            gaussian_ids=ids,
            dynamic_count=self.count,
            metadata={
                k: v
                for k, v in metadata.items()
                if k
                in (
                    "world_to_sim_rotation",
                    "normalization_center",
                    "normalization_scale",
                    "camera_c2ws",
                    "camera_intr",
                    "input_c2ws",
                    "input_intr",
                    "width",
                    "height",
                )
            },
        )

    @torch.no_grad()
    def frame(self, sim, frame_index):
        n, r = self.count, self.rotation
        xyz = ((sim.state["x"][:n] - 1) / self.scale + self.center) @ r
        covariance = r.T @ (sim.covariance()[:n] / self.scale**2) @ r
        frame = dict(
            xyz=xyz,
            covariance=covariance,
            frame_index=int(frame_index),
            time=float(sim.time),
            step=int(sim.steps),
        )
        if self.shared["feature"].shape[1] > 1:
            # Import only during live higher-order shading; replay has R cached.
            from mpm.linalg import mm3, svd3

            u, _, v = svd3(sim.state["F_trial"][:n], sim.svd_backend)
            inverse_polar = mm3(u, v.transpose(-1, -2)).transpose(-1, -2)
            frame["rotation"] = r.T @ inverse_polar @ r
        return frame

    def render_input(self, frame):
        return compose_frame(self.shared, frame)
