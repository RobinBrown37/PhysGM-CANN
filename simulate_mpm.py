"""Forward MPM and dynamic covariance rendering; no optimizer or MPM autograd."""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from export import to_cpu, write_json
from rendering import DynamicGaussianRenderer
from rendering.cameras import CameraPath
from rendering.mpm_adapter import MPMFrameAdapter
from rendering.sequence import FrameExporter, SequenceWriter
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
    """M4 compatibility wrapper: same first input camera, shared M5 logic."""
    key = (width, height)
    if cache.get("_size") != key:
        cache.clear()
        cache["_size"] = key
        cache["adapter"] = MPMFrameAdapter(metadata, sim.device, sim.dtype)
        cache["dynamic"] = DynamicGaussianRenderer(renderer=renderer)
        cache["camera"] = CameraPath(
            {}, metadata, mode="input", width=width, height=height
        ).at(0)
    frame = cache["adapter"].frame(sim, sim.steps)
    camera = cache["camera"]
    return cache["dynamic"](
        cache["adapter"].render_input(frame),
        camera["c2ws"].to(sim.device),
        camera["intr"].to(sim.device),
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


def main(argv=None):
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
    p.add_argument("--camera-mode", choices=("input", "config"), default="input")
    p.add_argument(
        "--camera-config",
        type=Path,
        help="Override rendering camera configuration only",
    )
    p.add_argument("--cameras-json", type=Path)
    p.add_argument("--move-camera", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--save-render-frames", action="store_true")
    p.add_argument("--sequence-dir", type=Path)
    p.add_argument("--append-sequence", action="store_true")
    p.add_argument(
        "--save-render-tensors",
        action="store_true",
        help="CPU FP32 RGB/alpha for replay verification",
    )
    p.add_argument("--fps", type=float, help="Playback override; default 1/frame_dt")
    p.add_argument("--background", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    args = p.parse_args(argv)
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
    width = (
        args.width
        if args.width is not None
        else (meta["width"] if args.camera_mode == "input" else 512)
    )
    height = (
        args.height
        if args.height is not None
        else (meta["height"] if args.camera_mode == "input" else 512)
    )
    camera_cfg = dict(cfg)
    if args.camera_config is not None:
        camera_cfg.update(json.loads(args.camera_config.read_text()))
    if args.move_camera is not None:
        camera_cfg["move_camera"] = args.move_camera
    if args.cameras_json is not None:
        camera_cfg["_external_cameras"] = json.loads(args.cameras_json.read_text())
    camera_path = CameraPath(
        camera_cfg, meta, mode=args.camera_mode, width=width, height=height
    )
    # Frame 0 is rendered after the first frame_dt, matching gs_simulation.py.
    # Explicit absolute indices preserve camera phase on resumed simulations.
    start_frame = sim.steps // substeps
    camera_path.at(start_frame)
    fps = args.fps if args.fps is not None else 1 / frame_dt
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
        status="running",
        camera_mode=args.camera_mode,
        start_frame=start_frame,
        fps=fps,
        width=width,
        height=height,
        background=list(args.background),
        camera_config=camera_cfg,
        new_cann_operator_families=0,
    )
    renderer = DynamicGaussianRenderer(
        backend="cann" if device.type == "npu" else "reference", far_plane=2.5
    )
    adapter = MPMFrameAdapter(meta, device, sim.dtype)
    sequence = None
    if args.save_render_frames:
        sequence = SequenceWriter(
            args.sequence_dir or out / "sequence",
            adapter.shared,
            {**camera_cfg, "frame_dt": frame_dt},
            args.physics_mode,
            append=args.append_sequence,
        )
        report["sequence_dir"] = str(sequence.root)
    exporter = None
    elapsed_render = 0.0
    elapsed_export = 0.0
    camera_records = []

    if not args.no_render and args.steps is None:
        exporter = FrameExporter(
            out,
            width,
            height,
            fps,
            total_frames=frames,
            physics_mode=args.physics_mode,
            save_tensors=args.save_render_tensors,
            background=args.background,
            legacy_frames=args.camera_mode == "input" and not args.save_render_frames,
        )
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
                    frame_index=start_frame + frame,
                )
                render_frame = None
                camera = None
                if exporter is not None or sequence is not None:
                    et = time.perf_counter()
                    render_frame = adapter.frame(sim, start_frame + frame)
                    camera = camera_path.at(start_frame + frame)
                    camera_records.append(to_cpu(camera))
                    synchronize(device)
                    record["frame_prepare_seconds"] = time.perf_counter() - et
                if sequence is not None:
                    et = time.perf_counter()
                    sequence.append(render_frame, camera)
                    record["sequence_export_seconds"] = time.perf_counter() - et
                if exporter is not None:
                    rt = time.perf_counter()
                    image = renderer(
                        adapter.render_input(render_frame),
                        camera["c2ws"].to(device),
                        camera["intr"].to(device),
                        width=width,
                        height=height,
                        background=args.background,
                    )
                    synchronize(device)
                    record["render_seconds"] = time.perf_counter() - rt
                    elapsed_render += record["render_seconds"]
                    et = time.perf_counter()
                    exporter.write(image, start_frame + frame, sim.time)
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
        if camera_records:
            atomic_save(camera_records, out / "cameras.pt")
        report["status"] = "passed"
        write_json(out / "report.json", report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "report.json", report)
        raise
    finally:
        if exporter is not None:
            exporter.close()


if __name__ == "__main__":
    main()
