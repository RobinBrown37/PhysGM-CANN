"""Forward MPM and dynamic covariance rendering; no optimizer or MPM autograd."""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from export import to_cpu, write_json
from rendering import StaticGaussianRenderer
from rendering.geometry import sh_colors
from mpm import MPMSimulator
from mpm.prepare import prepare_prediction
from runtime import (
    environment_report,
    memory_report,
    reset_peak_memory,
    select_device,
    synchronize,
)

ROOT = Path(__file__).resolve().parent


def atomic_save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(to_cpu(value), temp)
    temp.replace(path)


def render_state(sim, renderer, metadata, cache, width, height):
    n = metadata["gaussian_count"]
    dev = sim.device
    dtype = sim.dtype
    if not cache:
        for k in (
            "world_to_sim_rotation",
            "normalization_center",
            "feature",
            "opacity",
            "camera_c2ws",
            "camera_intr",
        ):
            cache[k] = metadata[k].to(device=dev, dtype=dtype)
        cache["intr"] = cache["camera_intr"].clone()
        cache["intr"][..., [0, 2]] *= width / metadata["width"]
        cache["intr"][..., [1, 3]] *= height / metadata["height"]
    r = cache["world_to_sim_rotation"]
    scale = metadata["normalization_scale"]
    means = ((sim.state["x"][:n] - 1) / scale + cache["normalization_center"]) @ r
    covariance = r.T @ (sim.covariance()[:n] / scale**2) @ r
    degree = int(round(cache["feature"].shape[1] ** 0.5) - 1)
    rotation = r.T @ sim.rotation()[:n] @ r if degree else None
    colors = sh_colors(
        cache["feature"],
        means,
        cache["camera_c2ws"][0, 0, :3, 3],
        degree,
        direction_rotation=rotation,
    )
    opacity = cache["opacity"]
    static = metadata.get("static_gaussians")
    if static is not None:
        static = {k: v.to(device=dev, dtype=dtype) for k, v in static.items()}
        means = torch.cat((means, static["xyz"]))
        covariance = torch.cat((covariance, static["covariance"]))
        colors = torch.cat(
            (
                colors,
                sh_colors(
                    static["feature"],
                    static["xyz"],
                    cache["camera_c2ws"][0, 0, :3, 3],
                    degree,
                ),
            )
        )
        opacity = torch.cat((opacity, static["opacity"]))
    return renderer.render_precomputed(
        means[None],
        covariance[None],
        colors[None],
        opacity[None],
        cache["camera_c2ws"],
        cache["intr"],
        width=width,
        height=height,
    )


def validate_state(sim):
    result = {}
    for key in ("x", "v", "F_trial", "stress", "mu", "lam", "yield_stress"):
        value = sim.state[key]
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Nonfinite MPM state: {key}")
        result[key + "_absmax"] = float(value.abs().max())
    result["particle_count"] = len(sim.state["x"])
    result["particle_mass"] = float(sim.state["mass"].sum())
    result["grid_mass"] = float(sim.grid[:, 0].sum())
    active = sim.state["selection"] == 0
    active_mass = float((sim.state["mass"] * active).sum())
    result["mass_relative_error"] = abs(result["grid_mass"] - active_mass) / max(
        active_mass, 1.0e-30
    )
    if sim.steps and result["mass_relative_error"] > 1.0e-5:
        raise RuntimeError(f"Mass conservation failed: {result}")
    # Covariance validation only at frame boundaries, on-device; zero covariance
    # belongs to simulation-only filling particles, which are never rendered.
    n = sim.metadata.get("gaussian_count", len(sim.state["x"]))
    cov = sim.covariance()[:n]
    from mpm.linalg import det3

    # The real checkpoint includes extremely thin Gaussians: FP32 construction
    # already produces ~1e-8 relative negative eigenvalues before simulation.
    # Check PSD up to FP32 rounding, without modifying/clamping the covariance.
    scale = cov.diagonal(dim1=-2, dim2=-1).amax(-1)
    normalized = cov / scale.clamp_min(1.0e-30)[:, None, None]
    minors = torch.stack(
        [
            normalized[:, i, i] * normalized[:, j, j] - normalized[:, i, j].square()
            for i, j in ((0, 1), (0, 2), (1, 2))
        ],
        -1,
    )
    determinant = det3(normalized)
    valid = (
        (scale > 0)
        & (normalized.diagonal(dim1=-2, dim2=-1) >= -1.0e-6).all(-1)
        & (minors >= -1.0e-6).all(-1)
        & (determinant >= -1.0e-6)
    )
    if not bool(torch.isfinite(cov).all()) or not bool(valid.all()):
        raise RuntimeError("Rendered covariance fails FP32 PSD validation")
    result["covariance_psd_fp32"] = True
    result["covariance_normalized_minor_min"] = float(minors.min())
    result["covariance_normalized_det_min"] = float(determinant.min())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--prediction-pack", type=Path, default=None)
    source.add_argument("--initial-state", type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "configs/mpm/cake.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/mpm_cake")
    p.add_argument("--device", default="npu:0")
    p.add_argument(
        "--physics-mode", choices=("upstream", "corrected"), default="upstream"
    )
    p.add_argument("--svd-backend", choices=("native", "torch"), default="native")
    p.add_argument(
        "--transfer-backend", choices=("auto", "torch", "native"), default="auto"
    )
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--frames", type=int, help="Explicit debug override; recorded in report"
    )
    p.add_argument("--steps", type=int, help="Explicit substep-only check; no video")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--profile-steps", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    st = time.perf_counter()
    if args.initial_state:
        initial = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    else:
        pack = torch.load(
            args.prediction_pack or ROOT / "results/npu_forward/prediction.pt",
            map_location="cpu",
            weights_only=False,
        )
        initial = prepare_prediction(
            pack, json.loads(args.config.read_text()), args.seed
        )
        initial["physics_mode"] = args.physics_mode
        atomic_save(initial, out / "initial_state.pt")
    write_json(out / "config.json", initial["config"])
    meta = initial["metadata"]
    write_json(
        out / "initialization.json",
        {k: v for k, v in meta.items() if not isinstance(v, (torch.Tensor, dict))}
        | {
            "filling": meta.get("filling", {}),
            "physical_export": meta.get("physical_export", {}),
        },
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared_particles": len(initial["state"]["x"]),
                    "gaussians": meta["gaussian_count"],
                    "seconds": time.perf_counter() - st,
                }
            )
        )
        return
    device = select_device(args.device)
    reset_peak_memory(device)
    transfer_backend = (
        ("native" if device.type == "npu" else "torch")
        if args.transfer_backend == "auto"
        else args.transfer_backend
    )
    sim = MPMSimulator(
        device=device,
        svd_backend=args.svd_backend,
        transfer_backend=transfer_backend,
        physics_mode=args.physics_mode,
    ).load_state(initial)
    cfg = sim.cfg
    dt = cfg.get("substep_dt", 1.0e-4)
    frame_dt = cfg.get("frame_dt", 0.01)
    substeps = round(frame_dt / dt)
    if not np.isclose(substeps * dt, frame_dt, rtol=0, atol=1.0e-10):
        raise ValueError("frame_dt must be an integer multiple of substep_dt")
    frames = args.frames if args.frames is not None else cfg.get("frame_num", 100)
    if frames <= 0 or (args.steps is not None and args.steps <= 0):
        raise ValueError("Positive frame/step count required")
    width = args.width or meta["width"]
    height = args.height or meta["height"]
    report = dict(
        environment=environment_report(device),
        physics_mode=args.physics_mode,
        svd_backend=args.svd_backend,
        transfer_backend=transfer_backend,
        gaussians=meta["gaussian_count"],
        simulation_particles=len(sim.state["x"]),
        filled_particles=len(sim.state["x"]) - meta["gaussian_count"],
        substeps_per_frame=substeps,
        dt=dt,
        frames_requested=frames,
        steps_requested=args.steps,
        initialization=meta.get("filling", {}),
        cpu_initialization_seconds=meta.get("cpu_initialization_seconds"),
        cpu_work=[
            "initialization/filling/volume and cached initial pack",
            "scalar validation and host control",
            "PNG/MP4 and state export",
            "renderer tile-count scheduling",
        ],
        solver_host_particle_fallback=False,
        frames=[],
        start_step=sim.steps,
    )
    renderer = StaticGaussianRenderer(
        backend="cann" if device.type == "npu" else "reference", far_plane=2.5
    )
    cache = {}
    writer = None
    previews = []
    elapsed_render = 0.0
    elapsed_export = 0.0

    if not args.no_render and args.steps is None:
        import cv2

        writer = cv2.VideoWriter(
            str(out / "simulation.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV MP4 encoder unavailable")
        (out / "frames").mkdir(exist_ok=True)
    synchronize(device)
    run_start = time.perf_counter()
    try:
        if args.steps is not None:
            for i in range(args.steps):
                sim.step(dt, profile=i < args.profile_steps)
            report["validation"] = validate_state(sim)
        else:
            for frame in range(frames):
                frame_start = time.perf_counter()
                for j in range(substeps):
                    sim.step(
                        dt,
                        profile=sim.steps - report["start_step"] < args.profile_steps,
                    )
                synchronize(device)
                solver_seconds = time.perf_counter() - frame_start
                checked = validate_state(sim)
                record = dict(
                    frame=frame + 1,
                    time=sim.time,
                    steps=sim.steps,
                    solver_seconds=solver_seconds,
                    **checked,
                )
                if writer is not None:
                    rt = time.perf_counter()
                    image = render_state(sim, renderer, meta, cache, width, height)
                    synchronize(device)
                    record["render_seconds"] = time.perf_counter() - rt
                    elapsed_render += record["render_seconds"]
                    et = time.perf_counter()
                    rgb = (
                        (
                            image.rgb[0, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                            * 255
                        )
                        .round()
                        .astype(np.uint8)
                    )
                    png = Image.fromarray(rgb)
                    draw = ImageDraw.Draw(png)
                    draw.rectangle((0, 0, 260, 22), fill="white")
                    draw.text(
                        (5, 5),
                        f"{args.physics_mode} | frame {frame + 1:02d} | t={sim.time:.2f}s",
                        fill="black",
                    )
                    png.save(out / "frames" / f"{frame + 1:04d}.png")
                    writer.write(np.asarray(png)[:, :, ::-1])
                    if frame in sorted(
                        set([0, frames // 4, frames // 2, 3 * frames // 4, frames - 1])
                    ):
                        previews.append(png.copy())
                    record["export_seconds"] = time.perf_counter() - et
                    elapsed_export += record["export_seconds"]
                report["frames"].append(record)
                print(json.dumps(record), flush=True)
                if (frame + 1) % 10 == 0:
                    atomic_save(sim.export_state(cpu=True), out / "latest_state.pt")
                write_json(out / "progress.json", report)
        synchronize(device)
        report["run_seconds"] = time.perf_counter() - run_start
        report["profiled_stage_seconds"] = {
            k: {"median": statistics.median(v), "samples": v}
            for k, v in sim.timings.items()
        }
        report.update(memory_report(device))
        report["final_step"] = sim.steps
        report["final_time"] = sim.time
        report["render_seconds"] = elapsed_render
        report["image_export_seconds"] = elapsed_export
        atomic_save(sim.export_state(cpu=True), out / "final_state.pt")
        if previews:
            montage = Image.new("RGB", (width * len(previews), height), "white")
            for i, img in enumerate(previews):
                montage.paste(img, (i * width, 0))
            montage.save(out / "preview.png")
        write_json(out / "report.json", report)
    finally:
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
