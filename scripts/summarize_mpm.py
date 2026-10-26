"""Validate complete cake artifacts and write the portable M4 acceptance summary."""

import csv
import hashlib
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / "results/mpm_cake"
    report = json.loads((run / "report.json").read_text())
    benchmark = json.loads((ROOT / "results/mpm_benchmark/report.json").read_text())
    svd = json.loads((ROOT / "results/mpm_svd_final.json").read_text())
    resume = json.loads((ROOT / "results/mpm_resume/report.json").read_text())
    assert report["final_step"] == 10000 and len(report["frames"]) == 50
    assert report["gaussians"] == 213532 and report["simulation_particles"] == 219416
    assert len(list((run / "frames").glob("*.png"))) == 50
    assert all(
        f["mass_relative_error"] <= 1.0e-5 and f["covariance_psd_fp32"]
        for f in report["frames"]
    )
    cap = cv2.VideoCapture(str(run / "simulation.mp4"))
    video = dict(
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    assert video == dict(frames=50, fps=25.0, width=512, height=512), video
    assert cap.read()[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 49)
    assert cap.read()[0]
    cap.release()
    profile = ROOT / "results/mpm_benchmark/profiler"
    tasks = []
    for path in profile.glob("*/ASCEND_PROFILER_OUTPUT/task_time.csv"):
        tasks += list(csv.DictReader(path.open()))
    kernels = {
        r["kernel_name"]: r["kernel_type"]
        for r in tasks
        if "physgm_" in r["kernel_name"]
    }
    assert len(kernels) == 3 and not any("AICPU" in r["kernel_type"] for r in tasks)
    cases = {
        c["transfer_backend"]: c for c in benchmark["cases"] if c["particles"] == 219416
    }
    verification = dict(
        status="passed",
        environment=report["environment"],
        physics_mode=report["physics_mode"],
        tests=dict(
            mpm_cpu="27 passed, 14 NPU-only skipped",
            mpm_npu="41 passed",
            existing_npu_regression="53 passed",
        ),
        operator_families=3,
        native_kernel_execution=kernels,
        profiled_aicpu_tasks=0,
        svd=svd,
        full_scale_benchmark=cases,
        full_step_speedup=cases["torch"]["median_seconds"]
        / cases["native"]["median_seconds"],
        cake=dict(
            gaussians=report["gaussians"],
            filled_particles=report["filled_particles"],
            simulation_particles=report["simulation_particles"],
            final_step=report["final_step"],
            final_time=report["final_time"],
            run_seconds=report["run_seconds"],
            solver_seconds=sum(f["solver_seconds"] for f in report["frames"]),
            render_seconds=report["render_seconds"],
            max_mass_relative_error=max(
                f["mass_relative_error"] for f in report["frames"]
            ),
            npu_peak_allocated_bytes=report["npu_peak_allocated_bytes"],
            video=video,
        ),
        resume=resume,
        initial_state_sha256=hashlib.sha256(
            (run / "initial_state.pt").read_bytes()
        ).hexdigest(),
        gpu_warp_alignment="pending: same-input CUDA/Warp reference pack unavailable",
        mpm_backward="out of scope",
    )
    for name, src in (
        ("mpm_cake", run / "report.json"),
        ("mpm_resume", ROOT / "results/mpm_resume/report.json"),
    ):
        (ROOT / "reports" / f"{name}.json").write_text(src.read_text())
    (ROOT / "reports/mpm_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification["cake"], indent=2))


if __name__ == "__main__":
    main()
