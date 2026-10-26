"""Resume a real MPM state and verify the subsequent trajectory plus rendering."""

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
from runtime import environment_report, select_device, synchronize
from simulate_mpm import atomic_save, render_state, validate_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state", type=Path, default=Path("results/mpm_cake/final_state.pt")
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/mpm_resume"))
    args = p.parse_args()
    torch.set_num_threads(4)
    device = select_device("npu:0")
    package = torch.load(args.state, map_location="cpu", weights_only=False)
    mode = package["physics_mode"]
    sim = MPMSimulator(
        device=device, transfer_backend="native", physics_mode=mode
    ).load_state(package)
    dt = sim.cfg["substep_dt"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        sim.step(dt)
    atomic_save(sim.export_state(cpu=True), out / "mid_state.pt")
    for _ in range(4):
        sim.step(dt)
    resumed = MPMSimulator(
        device=device, transfer_backend="native", physics_mode=mode
    ).load_state(
        torch.load(out / "mid_state.pt", map_location="cpu", weights_only=False)
    )
    for _ in range(4):
        resumed.step(dt)
    errors = {}
    for key, value in sim.state.items():
        actual = resumed.state[key]
        if value.is_floating_point():
            error = float(
                torch.linalg.vector_norm(actual - value)
                / torch.linalg.vector_norm(value).clamp_min(1)
            )
            if not np.isfinite(error) or error > 1.0e-3:
                raise RuntimeError(f"Resume mismatch: {key}: {error}")
            errors[key] = error
        elif not torch.equal(actual, value):
            raise RuntimeError(f"Resume ID/mask mismatch: {key}")
    if resumed.steps != sim.steps or resumed.time != sim.time:
        raise RuntimeError("Resume counters mismatch")
    renderer = StaticGaussianRenderer(backend="cann")
    meta = package["metadata"]
    with torch.no_grad():
        image = render_state(resumed, renderer, meta, {}, 512, 512)
    synchronize(device)
    rgb = (
        (image.rgb[0, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    png = Image.fromarray(rgb)
    ImageDraw.Draw(png).text(
        (5, 5), f"{mode} resumed | t={resumed.time:.4f}s", fill="black"
    )
    png.save(out / "preview.png")
    report = dict(
        status="passed",
        environment=environment_report(device),
        physics_mode=mode,
        input_step=package["steps"],
        final_step=resumed.steps,
        tolerance=1.0e-3,
        normalized_errors=errors,
        validation=validate_state(resumed),
        render_stats=image.stats,
        gaussian_ids_preserved=torch.equal(
            package["metadata"]["gaussian_ids"], resumed.metadata["gaussian_ids"]
        ),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
