"""Replay a CPU render sequence with M2 CANN kernels, without network or MPM."""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import argparse
import json
from pathlib import Path
import time

import torch

from rendering import DynamicGaussianRenderer
from rendering.cameras import CameraPath
from rendering.sequence import (
    FrameExporter,
    SequenceReader,
    atomic_json,
    atomic_torch,
    compose_frame,
    cpu_pack,
    signature,
)
from runtime import (
    environment_report,
    memory_report,
    reset_peak_memory,
    select_device,
    synchronize,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/dynamic_replay"))
    p.add_argument("--camera-mode", choices=("recorded", "config", "input"))
    p.add_argument("--camera-config", type=Path)
    p.add_argument("--cameras-json", type=Path)
    p.add_argument("--move-camera", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--backend", choices=("cann", "reference"), default="cann")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--background", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    p.add_argument("--fps", type=float)
    p.add_argument(
        "--compare-to",
        type=Path,
        help="Online render_tensors directory; threshold 2e-4",
    )
    p.add_argument("--save-render-tensors", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--profile",
        action="store_true",
        help="One separate rendering trace after timed replay",
    )
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    reset_peak_memory(device)
    reader = SequenceReader(args.sequence, device)
    if not len(reader):
        raise ValueError("Cannot render an empty sequence")
    cfg = dict(reader.shared["config"])
    if args.camera_config is not None:
        cfg.update(json.loads(args.camera_config.read_text()))
    if args.move_camera is not None:
        cfg["move_camera"] = args.move_camera
    changed = (
        args.camera_config is not None
        or args.cameras_json is not None
        or args.move_camera is not None
    )
    mode = args.camera_mode or ("config" if changed else "recorded")
    if mode == "recorded" and changed:
        raise ValueError("Camera overrides require --camera-mode config or input")
    first = torch.load(
        args.sequence / reader.records[0]["file"],
        map_location="cpu",
        weights_only=False,
    )["camera"]
    width = args.width if args.width is not None else first["width"]
    height = args.height if args.height is not None else first["height"]
    path = (
        CameraPath(
            cfg,
            reader.shared["metadata"],
            mode=mode,
            cameras_json=args.cameras_json,
            width=width,
            height=height,
        )
        if mode != "recorded"
        else None
    )
    renderer = DynamicGaussianRenderer(backend=args.backend)
    exporter = FrameExporter(
        args.output_dir,
        width,
        height,
        args.fps if args.fps is not None else 1 / reader.manifest["frame_dt"],
        total_frames=len(reader),
        physics_mode=reader.shared["physics_mode"],
        save_tensors=args.save_render_tensors,
        background=args.background,
    )
    report = dict(
        status="running",
        environment=environment_report(device),
        camera_mode=mode,
        frames=[],
        source_scene_sha256=reader.manifest["scene_sha256"],
        mpm_substeps_executed=0,
        network_loaded=False,
        new_cann_operator_families=0,
        comparison_tolerance=2.0e-4 if args.compare_to else None,
        cpu_work=[
            "sequence loading",
            "camera planning",
            "tile-count scheduling",
            "PNG/MP4 and report export",
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    maximum = 0.0
    last = None
    camera_records = []
    try:
        with torch.no_grad():
            for frame, camera in reader:
                index = frame["frame_index"]
                if path is not None:
                    camera = path.at(index)
                else:
                    camera = dict(camera)
                    camera["intr"] = camera["intr"].clone()
                    camera["intr"][..., [0, 2]] *= width / camera["width"]
                    camera["intr"][..., [1, 3]] *= height / camera["height"]
                    camera["width"], camera["height"] = width, height
                c2ws, intr = camera["c2ws"].to(device), camera["intr"].to(device)
                inputs = compose_frame(reader.shared, frame)
                synchronize(device)
                st = time.perf_counter()
                image = renderer(
                    inputs,
                    c2ws,
                    intr,
                    width=width,
                    height=height,
                    background=args.background,
                )
                synchronize(device)
                elapsed = time.perf_counter() - st
                record = dict(
                    frame_index=index,
                    time=frame["time"],
                    step=frame["step"],
                    render_seconds=elapsed,
                    stats=image.stats,
                )
                if args.compare_to:
                    target = torch.load(
                        args.compare_to / f"{index:06d}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    if (
                        target["frame_index"] != index
                        or abs(target["time"] - frame["time"]) > 1.0e-12
                    ):
                        raise RuntimeError("Replay comparison frame/time mismatch")
                    errors = {
                        k: float((getattr(image, k).cpu() - target[k]).abs().max())
                        for k in ("rgb", "alpha")
                    }
                    if max(errors.values()) > 2.0e-4:
                        raise RuntimeError(f"Online/replay mismatch: {errors}")
                    maximum = max(maximum, max(errors.values()))
                    record["max_absolute_errors"] = errors
                et = time.perf_counter()
                exporter.write(image, index, frame["time"])
                record["export_seconds"] = time.perf_counter() - et
                camera_records.append(
                    cpu_pack(
                        dict(
                            frame_index=index,
                            c2ws=c2ws,
                            intr=intr,
                            width=width,
                            height=height,
                        )
                    )
                )
                report["frames"].append(record)
                last = (inputs, c2ws, intr)
        report.update(
            status="passed",
            run_seconds=time.perf_counter() - started,
            max_online_replay_absolute_error=maximum if args.compare_to else None,
            gaussian_ids_sha256=signature(reader.shared["gaussian_ids"]),
            **memory_report(device),
        )
        if args.profile and last is not None:
            if device.type != "npu":
                raise ValueError("--profile requires NPU")
            import torch_npu

            with (
                torch.no_grad(),
                torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    schedule=torch_npu.profiler.schedule(
                        wait=0, warmup=0, active=1, repeat=1
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.output_dir / "profiler")
                    ),
                ) as prof,
            ):
                renderer(
                    last[0],
                    last[1],
                    last[2],
                    width=width,
                    height=height,
                    background=args.background,
                )
                synchronize(device)
                prof.step()
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        exporter.close()
        atomic_torch(camera_records, args.output_dir / "cameras.pt")
        atomic_json(report, args.output_dir / "report.json")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
