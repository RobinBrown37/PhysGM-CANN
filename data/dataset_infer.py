"""Deterministic input-only loader, preserving upstream camera normalization."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, config, root=None):
        self.config = config
        root = Path(root or Path(__file__).resolve().parents[1])
        listing = Path(config.data.data_path)
        if not listing.is_absolute():
            listing = root / listing
        self.paths = [
            listing.parent / line.strip()
            for line in listing.read_text().splitlines() if line.strip()
        ]
        if not self.paths:
            raise ValueError(f"No scenes in {listing}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        scene = json.loads(path.read_text())
        count = self.config.data.num_input_frames
        if count < 1 or len(scene["frames"]) < count:
            raise ValueError(f"{path}: need {count} input frames")
        frames = scene["frames"][:count]
        arrays = []
        for frame in frames:
            with Image.open(path.parent / frame["file_path"]) as image:
                arrays.append(np.asarray(image.convert("RGBA")))
        images = np.stack(arrays)
        resize_h = self.config.data.get("resize_h", -1)
        resize_w = self.config.data.get("resize_w", -1)
        if resize_h == -1 and resize_w == -1:
            resize_h, resize_w = images.shape[1:3]
        elif resize_h == -1:
            resize_h = int(resize_w / images.shape[2] * images.shape[1])
        elif resize_w == -1:
            resize_w = int(resize_h / images.shape[1] * images.shape[2])
        resize_h = int(round(resize_h / 16)) * 16
        resize_w = int(round(resize_w / 16)) * 16
        if min(resize_h, resize_w) < 16:
            raise ValueError("Resized image dimensions must be at least 16")
        images = np.stack([cv2.resize(image, (resize_w, resize_h)) for image in images])
        start_h = start_w = 0
        if self.config.data.square_crop:
            size = min(resize_h, resize_w)
            start_h, start_w = (resize_h - size) // 2, (resize_w - size) // 2
            images = images[:, start_h:start_h + size, start_w:start_w + size]
        images = torch.from_numpy(images / 255.0).permute(0, 3, 1, 2).float()
        intr = np.array([[f[k] for k in ("fx", "fy", "cx", "cy")] for f in frames], dtype=np.float64)
        intr[:, (0, 2)] *= (resize_w / np.array([f["w"] for f in frames]))[:, None]
        intr[:, (1, 3)] *= (resize_h / np.array([f["h"] for f in frames]))[:, None]
        intr[:, 2] -= start_w
        intr[:, 3] -= start_h
        intr = torch.from_numpy(intr).float()
        c2ws = torch.from_numpy(np.linalg.inv(np.array([f["w2c"] for f in frames]))).float()
        position = c2ws[:, :3, 3].mean(0)
        forward = F.normalize(c2ws[:, :3, 2].mean(0), dim=0)
        down = c2ws[:, :3, 1].mean(0)
        down = F.normalize(down - down.dot(forward) * forward, dim=0)
        right = torch.linalg.cross(down, forward)
        transform = torch.cat((torch.stack((right, down, forward, position), 1), torch.tensor([[0., 0., 0., 1.]])))
        transform_inv = transform.inverse()
        c2ws = transform_inv.unsqueeze(0) @ c2ws
        maximum = c2ws[:, :3, 3].abs().max()
        if not torch.isfinite(c2ws).all() or maximum <= 0:
            raise ValueError(f"{path}: invalid or degenerate cameras")
        scene_scale = 1.0 / (self.config.data.get("scene_scale", 1.0) * maximum)
        c2ws[:, :3, 3] *= scene_scale
        return {
            "scene_name": scene.get("scene_name", path.parent.name),
            "input_images": images, "input_intr": intr, "input_c2ws": c2ws,
            "pos_avg_inv": transform_inv, "scene_scale": scene_scale,
            "input_frame_idx": torch.arange(count),
        }
