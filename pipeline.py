"""Strict offline M1 prediction -> M4 simulation -> M5 rendering pipeline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--network-config",
        "--config",
        type=Path,
        default=ROOT / "configs/network_npu.yaml",
    )
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoint.pt")
    p.add_argument(
        "--simulation-config",
        "--template-config",
        type=Path,
        default=ROOT / "configs/mpm/cake.json",
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/dynamic_cake")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--camera-mode", choices=("config", "input"), default="config")
    p.add_argument("--camera-config", type=Path)
    p.add_argument("--cameras-json", type=Path)
    p.add_argument("--move-camera", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--background", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    p.add_argument("--frames", type=int)
    p.add_argument("--fps", type=float)
    p.add_argument(
        "--physics-mode", choices=("upstream", "corrected"), default="upstream"
    )
    p.add_argument("--save-render-tensors", action="store_true")
    return p


def commands(args):
    out = args.output_dir.resolve()
    network = [
        sys.executable,
        str(ROOT / "infer_network.py"),
        "--config",
        str(args.network_config.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--device",
        args.device,
        "--amp-dtype",
        args.amp_dtype,
        "--scene-index",
        str(args.scene_index),
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--output-dir",
        str(out / "network"),
    ]
    simulation = [
        sys.executable,
        str(ROOT / "simulate_mpm.py"),
        "--prediction-pack",
        str(out / "network/prediction.pt"),
        "--config",
        str(args.simulation_config.resolve()),
        "--device",
        args.device,
        "--output-dir",
        str(out),
        "--camera-mode",
        args.camera_mode,
        "--save-render-frames",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--physics-mode",
        args.physics_mode,
        "--background",
        *[str(x) for x in args.background],
    ]
    for flag in ("camera_config", "cameras_json"):
        value = getattr(args, flag)
        if value is not None:
            simulation.extend(["--" + flag.replace("_", "-"), str(value.resolve())])
    for flag in ("frames", "fps"):
        if getattr(args, flag) is not None:
            simulation.extend(["--" + flag, str(getattr(args, flag))])
    if args.move_camera is not None:
        simulation.append("--move-camera" if args.move_camera else "--no-move-camera")
    if args.save_render_tensors:
        simulation.append("--save-render-tensors")
    return [("network", network), ("simulation_and_render", simulation)]


def main(argv=None):
    args = parser().parse_args(argv)
    out = args.output_dir.resolve()
    if (out / "sequence/manifest.json").exists():
        raise FileExistsError(
            "Output sequence exists; choose a new --output-dir or resume with simulate_mpm.py"
        )
    for path in (args.network_config, args.checkpoint, args.simulation_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        stages=[],
        network_precision=args.amp_dtype,
        physics_mode=args.physics_mode,
    )
    start = time.perf_counter()
    try:
        for name, cmd in commands(args):
            before = time.perf_counter()
            stage = dict(name=name, argv=cmd, status="running")
            report["stages"].append(stage)
            subprocess.run(cmd, check=True, cwd=ROOT)
            stage.update(status="passed", seconds=time.perf_counter() - before)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if report["stages"]:
            report["stages"][-1]["status"] = "failed"
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - start
        tmp = out / "pipeline_report.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(out / "pipeline_report.json")


if __name__ == "__main__":
    main()
