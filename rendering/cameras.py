"""CPU camera planning in the original sphere/look-at convention (+z forward).

The geometry is evaluated in FP64 on CPU; only final rigid c2w/intrinsics are
copied to the rendering device. No GSCamera/CUDA Gaussian-splatting imports.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch


def unit(vector, name):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or length < 1.0e-12:
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    return vector / length


def sim_to_world(points, metadata):
    r = torch.as_tensor(metadata["world_to_sim_rotation"]).cpu().double().numpy()
    center = torch.as_tensor(metadata["normalization_center"]).cpu().double().numpy()
    scale = float(metadata["normalization_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization scale must be finite and positive")
    return ((np.asarray(points, dtype=np.float64) - 1) / scale + center) @ r


def local_basis(vertical):
    up = unit(vertical, "vertical direction")
    h = np.array([1.0, 1.0, 1.0])
    if abs(h @ up) < 0.01:
        h = np.array([0.72, 0.37, -0.67])
    h = h - (h @ up) * up
    if np.linalg.norm(h) < 1.0e-12:
        axis = np.eye(3)[np.argmin(np.abs(up))]
        h = axis - (axis @ up) * up
    h = unit(h, "horizontal direction")
    return np.column_stack((h, np.cross(h, up), up))


def orbit_pose(center, basis, azimuth, elevation, radius):
    if (
        not all(math.isfinite(float(v)) for v in (azimuth, elevation, radius))
        or radius <= 0
    ):
        raise ValueError("Orbit angles must be finite and radius must be positive")
    a, e = np.deg2rad([azimuth, elevation])
    position = (
        center
        + basis
        @ np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
        * radius
    )
    z = unit(center - position, "look-at direction")
    down = -basis[:, 2]
    y = unit(down - (down @ z) * z, "camera vertical; avoid orbit poles")
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((np.cross(y, z), y, z))
    pose[:3, 3] = position
    return pose


def validate_pose(pose):
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Camera pose must be a finite 4x4 matrix")
    r = pose[:3, :3]
    if (
        not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1.0e-5)
        or np.linalg.det(r) <= 0
    ):
        raise ValueError("Camera rotation must be a proper orthonormal matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1.0e-7):
        raise ValueError("Invalid camera homogeneous row")


class CameraPath:
    def __init__(
        self,
        config,
        metadata,
        *,
        mode="config",
        cameras_json=None,
        width=512,
        height=512,
    ):
        if mode not in ("config", "input"):
            raise ValueError("Camera mode must be config or input")
        if width <= 0 or height <= 0:
            raise ValueError("Camera image dimensions must be positive")
        self.config, self.metadata = copy.deepcopy(config), metadata
        self.mode, self.width, self.height = mode, int(width), int(height)
        self.external = None
        source = (
            cameras_json
            if cameras_json is not None
            else self.config.get("_external_cameras")
        )
        if source is not None:
            entries = (
                json.loads(Path(source).read_text())
                if isinstance(source, (str, Path))
                else copy.deepcopy(source)
            )
            if not isinstance(entries, list) or not entries:
                raise ValueError("cameras.json must contain a nonempty camera list")
            self.external = []
            for entry in entries:
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = entry["rotation"], entry["position"]
                validate_pose(pose)
                w, h = int(entry["width"]), int(entry["height"])
                intr = np.array(
                    [
                        entry["fx"],
                        entry["fy"],
                        entry.get("cx", w / 2),
                        entry.get("cy", h / 2),
                    ],
                    dtype=np.float64,
                )
                self.external.append((pose, intr, w, h))
        if mode == "config" and int(config.get("default_camera_index", -1)) < 0:
            center = np.asarray(
                config.get("mpm_space_viewpoint_center", [1.0, 1.0, 1.0]),
                dtype=np.float64,
            )
            up = np.asarray(
                config.get("mpm_space_vertical_upward_axis", [0.0, 0.0, 1.0]),
                dtype=np.float64,
            )
            self.center = sim_to_world(center, metadata)
            self.basis = local_basis(sim_to_world(center + up, metadata) - self.center)

    def _input(self, index):
        poses = (
            torch.as_tensor(
                self.metadata.get("input_c2ws", self.metadata["camera_c2ws"])
            )
            .cpu()
            .double()
            .numpy()[0]
        )
        intr = (
            torch.as_tensor(
                self.metadata.get("input_intr", self.metadata["camera_intr"])
            )
            .cpu()
            .double()
            .numpy()[0]
        )
        if index < 0 or index >= len(poses):
            raise ValueError(
                f"Camera index {index} is out of range for {len(poses)} input cameras"
            )
        return (
            poses[index],
            intr[index],
            int(self.metadata["width"]),
            int(self.metadata["height"]),
        )

    def at(self, absolute_frame):
        if int(absolute_frame) != absolute_frame or absolute_frame < 0:
            raise ValueError("Absolute frame index must be a nonnegative integer")
        cfg = self.config
        index = 0 if self.mode == "input" else int(cfg.get("default_camera_index", -1))
        if index >= 0:
            if self.mode == "input" or self.external is None:
                pose, intr, w, h = self._input(index)
            else:
                if index >= len(self.external):
                    raise ValueError(
                        f"Camera index {index} is out of range for {len(self.external)} external cameras"
                    )
                pose, intr, w, h = self.external[index]
        else:
            if self.external is None:
                # The upstream missing-cameras.json template is 800x800/f=400.
                intr, w, h = np.array([400.0, 400.0, 400.0, 400.0]), 800, 800
            else:
                _, intr, w, h = self.external[0]
            frame = absolute_frame if cfg.get("move_camera", False) else 0
            pose = orbit_pose(
                self.center,
                self.basis,
                cfg.get("init_azimuthm", -45.0) + frame * cfg.get("delta_a", 0.0),
                cfg.get("init_elevation", 0.0) + frame * cfg.get("delta_e", 0.0),
                cfg.get("init_radius", 1.8) + frame * cfg.get("delta_r", 0.0),
            )
        validate_pose(pose)
        intr = np.asarray(intr, dtype=np.float64).copy()
        if w <= 0 or h <= 0 or not np.isfinite(intr).all() or np.any(intr[:2] <= 0):
            raise ValueError("Camera dimensions and fx/fy must be finite and positive")
        intr[[0, 2]] *= self.width / w
        intr[[1, 3]] *= self.height / h
        return dict(
            c2ws=torch.from_numpy(pose.astype(np.float32))[None, None],
            intr=torch.from_numpy(intr.astype(np.float32))[None, None],
            width=self.width,
            height=self.height,
            frame_index=int(absolute_frame),
            mode=self.mode,
        )
