"""Warm synchronized full-step baselines and a trace of real native MPM."""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mpm import MPMSimulator
from runtime import (
    environment_report,
    memory_report,
    reset_peak_memory,
    select_device,
    synchronize,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--initial-state", type=Path, default=Path("results/mpm_cake/initial_state.pt")
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/mpm_benchmark"))
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    device = select_device("npu:0")
    initial = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    actual = len(initial["state"]["x"])
    report = {"environment": environment_report(device), "cases": []}
    for count in sorted(set([1000, 10000, 200000, actual])):
        if count > actual:
            continue
        package = {
            **initial,
            "state": {k: v[:count] for k, v in initial["state"].items()},
        }
        for backend in ("torch", "native"):
            reset_peak_memory(device)
            sim = MPMSimulator(device=device, transfer_backend=backend).load_state(
                package
            )
            dt = sim.cfg["substep_dt"]
            for _ in range(2):
                sim.step(dt)
            synchronize(device)
            times = []
            for _ in range(7):
                st = time.perf_counter()
                sim.step(dt, profile=True)
                synchronize(device)
                times.append(time.perf_counter() - st)
            report["cases"].append(
                dict(
                    particles=count,
                    transfer_backend=backend,
                    median_seconds=statistics.median(times),
                    p95_seconds=sorted(times)[-1],
                    samples=times,
                    stages={k: statistics.median(v) for k, v in sim.timings.items()},
                    **memory_report(device),
                )
            )
            print(json.dumps(report["cases"][-1]), flush=True)
    if args.profile:
        import torch_npu

        sim = MPMSimulator(device=device, transfer_backend="native").load_state(initial)
        sim.step(sim.cfg["substep_dt"])
        synchronize(device)
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(args.output_dir / "profiler")
            ),
        ) as prof:
            for _ in range(3):
                sim.step(sim.cfg["substep_dt"])
                prof.step()
            synchronize(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
