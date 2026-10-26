"""Additional wide-FOV before/after image; primary video keeps the M1 camera."""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
import torch
from mpm import MPMSimulator
from rendering import StaticGaussianRenderer
from runtime import select_device, synchronize
from simulate_mpm import render_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("results/mpm_cake"))
    p.add_argument("--focal-scale", type=float, default=0.65)
    args = p.parse_args()
    torch.set_num_threads(4)
    if not 0 < args.focal_scale <= 1:
        raise ValueError("Require 0 < focal scale <= 1")
    device = select_device("npu:0")
    renderer = StaticGaussianRenderer(backend="cann")
    montage = Image.new("RGB", (1024, 512), "white")
    for i, name in enumerate(("initial_state.pt", "final_state.pt")):
        package = torch.load(
            args.directory / name, map_location="cpu", weights_only=False
        )
        sim = MPMSimulator(
            device=device,
            transfer_backend="native",
            physics_mode=package["physics_mode"],
        ).load_state(package)
        meta = package["metadata"].copy()
        meta["camera_intr"] = meta["camera_intr"].clone()
        meta["camera_intr"][..., :2] *= args.focal_scale
        with torch.no_grad():
            result = render_state(sim, renderer, meta, {}, 512, 512)
        synchronize(device)
        rgb = (
            (result.rgb[0, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255)
            .round()
            .astype(np.uint8)
        )
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.text(
            (8, 8),
            f"{sim.physics_mode} | t={sim.time:.2f}s | M1 pose, focal x{args.focal_scale}",
            fill="black",
        )
        montage.paste(image, (i * 512, 0))
    montage.save(args.directory / "overview.png")
    (args.directory / "overview_camera.json").write_text(
        json.dumps(
            dict(
                camera_pose="first M1 input",
                focal_scale=args.focal_scale,
                principal_point="unchanged",
                primary_video_camera="original M1 intrinsics",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
