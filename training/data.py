"""Real supervised scenes with worker-independent sampling and replayable cursors."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .constants import normalized_physical


def process_frames(frames, directory, config, crop_ratio=1.):
    arrays = []
    for frame in frames:
        with Image.open(directory/frame["file_path"]) as image:
            arrays.append(np.asarray(image.convert("RGBA")))
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"Inconsistent image sizes in {directory}")
    source_h, source_w = arrays[0].shape[:2]
    height, width = config.get("resize_h", 512), config.get("resize_w", -1)
    if height == -1 and width == -1:
        height, width = source_h, source_w
    elif height == -1:
        height = int(width/source_w*source_h)
    elif width == -1:
        width = int(height/source_h*source_w)
    height, width = round(height/16)*16, round(width/16)*16
    if min(height, width) < 16:
        raise ValueError("Resized dimensions must be >=16")
    images = np.stack([cv2.resize(a, (width, height)) for a in arrays])
    intr = np.array([[f[k] for k in ("fx", "fy", "cx", "cy")] for f in frames], dtype=np.float64)
    intr[:, (0, 2)] *= (width/np.array([f["w"] for f in frames]))[:, None]
    intr[:, (1, 3)] *= (height/np.array([f["h"] for f in frames]))[:, None]
    if config.get("square_crop", True):
        size = min(height, width)
        top, left = (height-size)//2, (width-size)//2
        images = images[:, top:top+size, left:left+size]
        intr[:, 2:] -= (left, top)
    images = torch.from_numpy(images/255.).permute(0, 3, 1, 2).float()
    if not 0 < crop_ratio <= 1:
        raise ValueError("Crop ratio must be in (0,1]")
    if crop_ratio < 1:
        h, w = images.shape[-2:]
        images = F.interpolate(images, scale_factor=1/crop_ratio, mode="bilinear", align_corners=False)
        mh, mw = images.shape[-2:]
        top, left = (mh-h)//2, (mw-w)//2
        images = images[:, :, top:top+h, left:left+w]
        intr[:, (0, 2)] *= mw/w
        intr[:, (1, 3)] *= mh/h
        intr[:, 2:] -= (left, top)
    c2ws = torch.from_numpy(np.linalg.inv(np.array([f["w2c"] for f in frames]))).float()
    intr = torch.from_numpy(intr).float()
    if not torch.isfinite(c2ws).all() or not torch.isfinite(intr).all() or (intr[:, :2] <= 0).any():
        raise ValueError(f"Invalid camera in {directory}")
    rotation = c2ws[:, :3, :3]
    if (rotation @ rotation.transpose(-1, -2)-torch.eye(3)).abs().max() > 1e-4:
        raise ValueError(f"Non-rigid camera in {directory}")
    return images, intr, c2ws


class SupervisedDataset(Dataset):
    def __init__(self, config, *, evaluation=False, root=None):
        self.config, self.evaluation = config, evaluation
        root = Path(root or Path(__file__).resolve().parents[1])
        listing = Path(config.data.validation_path if evaluation else config.data.data_path)
        self.listing = listing if listing.is_absolute() else root/listing
        self.paths = [(self.listing.parent/line.strip()).resolve()
                      for line in self.listing.read_text().splitlines() if line.strip()]
        if not self.paths:
            raise ValueError(f"No scenes: {self.listing}")
        self.seed = config.get("seed", 1111)
        self.records = []
        digest = hashlib.sha256()
        for path in self.paths:
            try:
                scene = json.loads(path.read_text())
                labels = None
                t = config.training
                if any(t.get(k, 0) > 0 for k in ("physical_loss", "E_loss", "nu_loss")):
                    labels = normalized_physical(json.loads((path.parent/"physical.json").read_text()))
                if len(scene["frames"]) < max(config.data.num_input_frames, config.data.num_target_frames):
                    raise ValueError("Insufficient frames for sampling without replacement")
                for f in scene["frames"]:
                    relative = Path(f["file_path"])
                    image = relative if relative.is_absolute() else path.parent/relative
                    if not image.is_file():
                        raise FileNotFoundError(image)
                    digest.update(str(image).encode())
                    digest.update(hashlib.sha256(image.read_bytes()).digest())
                digest.update(path.read_bytes())
                if labels is not None:
                    digest.update((path.parent/"physical.json").read_bytes())
                self.records.append((path, scene, labels))
            except Exception as exc:
                raise ValueError(f"Invalid scene {path}: {exc}") from exc
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        # Sampler embeds epoch/draw in each key: persistent/spawn workers see the
        # correct epoch without shared mutable dataset state or prefetch RNG.
        epoch, draw, index = key if isinstance(key, tuple) else (0, int(key), int(key))
        path, scene, labels = self.records[index]
        seed = np.random.SeedSequence([self.seed, 0 if self.evaluation else epoch,
                                       index if self.evaluation else draw, index])
        rng = np.random.default_rng(seed)
        ni, nt = self.config.data.num_input_frames, self.config.data.num_target_frames
        frames = scene["frames"]
        if self.evaluation:
            inputs = np.arange(ni)
        else:
            inputs = rng.choice(len(frames), ni, replace=False)
        if self.config.data.get("target_has_input", True):
            if nt < ni:
                raise ValueError("target_has_input requires target count >= input count")
            remaining = np.setdiff1d(np.arange(len(frames)), inputs)
            extra = remaining[:nt-ni] if self.evaluation else rng.choice(remaining, nt-ni, replace=False)
            targets = np.concatenate((inputs, extra))
        else:
            targets = np.arange(nt) if self.evaluation else rng.choice(len(frames), nt, replace=False)
        minimum = self.config.data.get("random_crop", 1.)
        crop = 1. if self.evaluation else float(rng.uniform(minimum, 1.))
        inp, intr, c2ws = process_frames([frames[i] for i in inputs], path.parent, self.config.data, crop)
        target, tintr, tc2ws = process_frames([frames[i] for i in targets], path.parent, self.config.data, crop)
        # Random nearly symmetric views can make the averaged Gram-Schmidt
        # vectors almost parallel. CPU FP64 avoids amplifying FP32 cancellation
        # into a non-rigid normalized camera (same normalization algorithm).
        c2ws, tc2ws = c2ws.double(), tc2ws.double()
        position = c2ws[:, :3, 3].mean(0)
        forward = F.normalize(c2ws[:, :3, 2].mean(0), dim=0)
        down = c2ws[:, :3, 1].mean(0)
        down = F.normalize(down-down.dot(forward)*forward, dim=0)
        right = torch.linalg.cross(down, forward)
        transform = torch.cat((torch.stack((right, down, forward, position), 1), torch.tensor([[0., 0., 0., 1.]], dtype=torch.float64)))
        if torch.linalg.det(transform[:3, :3]).abs() < 1e-6:
            raise ValueError(f"Degenerate average camera frame: {path}")
        inverse = transform.inverse()
        c2ws, tc2ws = inverse[None] @ c2ws, inverse[None] @ tc2ws
        extent = tc2ws[:, :3, 3].abs().max()
        if not torch.isfinite(extent) or extent <= 0:
            raise ValueError(f"Degenerate scene extent: {path}")
        scale = 1./(self.config.data.get("scene_scale", 1.)*extent)
        c2ws[:, :3, 3] *= scale
        tc2ws[:, :3, 3] *= scale
        result = {"scene_name": scene.get("scene_name", path.parent.name), "input_images": inp,
                  "input_intr": intr, "input_c2ws": c2ws.float(), "test_images": target,
                  "test_intr": tintr, "test_c2ws": tc2ws.float(), "pos_avg_inv": inverse.float(),
                  "scene_scale": scale.float(), "input_frame_idx": torch.tensor(inputs),
                  "test_frame_idx": torch.tensor(targets), "sample_key": torch.tensor([epoch, draw, index])}
        if labels is not None:
            result.update(physical_class=torch.tensor(labels[0]), E=torch.tensor(labels[1], dtype=torch.float32),
                          nu=torch.tensor(labels[2], dtype=torch.float32))
        return result


class CursorBatchSampler(Sampler):
    """Finite deterministic epochs, cursor is advanced only by consumed batches."""
    def __init__(self, size, batch_size, *, seed=1111, rank=0, world_size=1, epoch=0, cursor=0, fixed=False):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.rank, self.world_size, self.epoch, self.cursor, self.fixed = rank, world_size, epoch, cursor, fixed
        self.batches = size//(world_size*batch_size)
        if self.batches == 0:
            raise ValueError("Dataset smaller than one distributed batch")
        if not 0 <= cursor <= self.batches:
            raise ValueError("Invalid data cursor")

    def __len__(self):
        return self.batches-self.cursor

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        for batch in range(self.cursor, self.batches):
            if self.fixed:
                yield [(0, j, j % self.size) for j in range(self.batch_size)]
            else:
                start = (batch*self.world_size+self.rank)*self.batch_size
                yield [(self.epoch, start+j, order[start+j]) for j in range(self.batch_size)]
