"""Numerical gate and synchronized SVD microbenchmark on the selected device."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mpm.linalg import det3, svd3
from runtime import environment_report, select_device, synchronize


def cases():
    g = torch.Generator().manual_seed(412)
    q, _ = torch.linalg.qr(torch.randn(1024, 3, 3, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(1024, 3, 3, generator=g, dtype=torch.float64))
    diag = torch.rand(1024, 3, generator=g, dtype=torch.float64) * 2 + 0.1
    mats = [
        torch.eye(3)[None].double(),
        q,
        q @ torch.diag_embed(diag) @ v.mT,
        torch.tensor(
            [
                [[1.0, 0.5, -0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0e-7, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        ),
    ]
    return torch.cat(mats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--output", default="results/mpm_svd.json")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    device = select_device(args.device)
    a = cases().float().to(device)
    u, s, v = svd3(a)
    synchronize(device)
    cpu_a = a.cpu().double()
    u = u.cpu().double()
    s = s.cpu().double()
    v = v.cpu().double()
    reconstruction = torch.linalg.matrix_norm(
        u @ torch.diag_embed(s) @ v.mT - cpu_a
    ) / torch.linalg.matrix_norm(cpu_a).clamp_min(1)
    eye = torch.eye(3, dtype=torch.float64)
    orth = max(
        float(torch.linalg.matrix_norm(u.mT @ u - eye).max()),
        float(torch.linalg.matrix_norm(v.mT @ v - eye).max()),
    )
    report = {
        "environment": environment_report(device),
        "cases": len(a),
        "reconstruction_max": float(reconstruction.max()),
        "orthogonality_max": orth,
        "det_u_min": float(det3(u).min()),
        "det_v_min": float(det3(v).min()),
        "timings": {},
    }
    assert float(reconstruction.max()) <= 1.0e-5, report
    assert orth <= 1.0e-5, report
    for n in (128, 8192, 65536, 200000):
        a = (torch.eye(3)[None] + torch.randn(n, 3, 3) * 0.1).to(device)
        for backend in ("native", "torch"):
            svd3(a, backend)
            synchronize(device)
            times = []
            for _ in range(5):
                started = time.perf_counter()
                svd3(a, backend)
                synchronize(device)
                times.append(time.perf_counter() - started)
            report["timings"][f"{backend}_{n}"] = statistics.median(times)
    if args.profile and device.type == "npu":
        import torch_npu

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(Path(args.output).parent / "svd_profiler")
            ),
        ) as prof:
            svd3(a)
            synchronize(device)
            prof.step()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
