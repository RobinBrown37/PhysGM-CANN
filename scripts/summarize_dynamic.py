"""Validate full M5 cake artifacts and write a portable acceptance summary."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tests(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    result = {
        key: sum(int(item.attrib.get(key, 0)) for item in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if result["failures"] or result["errors"]:
        raise RuntimeError(f"Failed test report: {path}")
    return result


def video(path, expected_frames, expected_fps, width, height):
    capture = cv2.VideoCapture(str(path))
    info = dict(
        frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=capture.get(cv2.CAP_PROP_FPS),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    first = capture.read()[0]
    capture.set(cv2.CAP_PROP_POS_FRAMES, expected_frames - 1)
    last = capture.read()[0]
    capture.release()
    expected = dict(
        frames=expected_frames,
        fps=expected_fps,
        width=width + width % 2,
        height=height + height % 2,
    )
    if info != expected or not first or not last:
        raise RuntimeError(
            f"Video validation failed for {path}: {info}, expected {expected}"
        )
    return info


def rendering_kernels(profile_root):
    paths = list(profile_root.glob("*/ASCEND_PROFILER_OUTPUT/task_time.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one completed replay profile, found {len(paths)}")
    rows = list(csv.DictReader(paths[0].open()))
    wanted = ("FlashGaussianBuildMask", "GaussianSort", "CalcRenderFwd")
    kernels = {
        prefix: [
            dict(name=row["kernel_name"], type=row["kernel_type"])
            for row in rows
            if prefix in row["kernel_name"]
        ]
        for prefix in wanted
    }
    if not all(kernels.values()):
        raise RuntimeError(f"Missing M2 rendering kernels in replay profile: {kernels}")
    if any("physgm_" in row["kernel_name"].lower() for row in rows):
        raise RuntimeError("Offline replay unexpectedly executed an MPM custom kernel")
    return kernels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/dynamic_cake")
    parser.add_argument("--tests", type=Path, default=ROOT / "results/dynamic_tests")
    args = parser.parse_args(argv)
    run, test_root = args.run.resolve(), args.tests.resolve()

    network = json.loads((run / "network/run_report.json").read_text())
    simulation = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "sequence/manifest.json").read_text())
    fixed = json.loads((run / "replay_fixed/report.json").read_text())
    orbit = json.loads((run / "replay_orbit/report.json").read_text())
    pipeline = json.loads((run / "pipeline_report.json").read_text())
    count, width, height = (
        len(manifest["frames"]),
        simulation["width"],
        simulation["height"],
    )
    fps = simulation["fps"]

    assert pipeline["status"] == network["status"] == simulation["status"] == "passed"
    assert fixed["status"] == orbit["status"] == "passed"
    assert (
        network["checkpoint"]["strict"]
        and network["checkpoint"]["model_entries"] == 645
    )
    assert network["input_shape"] == [1, 4, 4, 512, 512]
    assert simulation["physics_mode"] == manifest["physics_mode"] == "upstream"
    assert (
        simulation["gaussians"] == 213532
        and simulation["simulation_particles"] == 219416
    )
    assert (
        simulation["final_step"] == 10000 and count == len(simulation["frames"]) == 50
    )
    assert [item["frame_index"] for item in manifest["frames"]] == list(range(50))
    assert all(
        item["mass_relative_error"] <= 1e-5 and item["covariance_psd_fp32"]
        for item in simulation["frames"]
    )
    assert fixed["mpm_substeps_executed"] == 0 and not fixed["network_loaded"]
    assert orbit["mpm_substeps_executed"] == 0 and not orbit["network_loaded"]
    assert fixed["max_online_replay_absolute_error"] <= 2e-4
    assert fixed["gaussian_ids_sha256"] == orbit["gaussian_ids_sha256"]
    assert len(list((run / "rgb").glob("*.png"))) == count
    assert len(list((run / "alpha").glob("*.png"))) == count

    scene = torch.load(
        run / "sequence/scene.pt", map_location="cpu", weights_only=False
    )
    assert scene["dynamic_count"] == simulation["gaussians"]
    assert len(scene["gaussian_ids"]) == simulation["gaussians"]
    for record in manifest["frames"]:
        path = run / "sequence" / record["file"]
        assert sha256(path) == record["file_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)["frame"]
        assert payload["xyz"].shape == (simulation["gaussians"], 3)
        assert payload["covariance"].shape == (simulation["gaussians"], 3, 3)
        assert payload["xyz"].dtype == payload["covariance"].dtype == torch.float32

    videos = {
        "online": video(run / "simulation.mp4", count, fps, width, height),
        "fixed_replay": video(
            run / "replay_fixed/simulation.mp4", count, fps, width, height
        ),
        "moving_replay": video(
            run / "replay_orbit/simulation.mp4", count, fps, width, height
        ),
    }
    kernels = rendering_kernels(run / "replay_fixed/profiler")
    test_reports = {
        name: tests(test_root / filename)
        for name, filename in (
            ("dynamic_cpu", "cpu.xml"),
            ("dynamic_npu", "npu.xml"),
            ("existing_npu", "existing53.xml"),
            ("mpm_npu", "mpm41.xml"),
        )
    }
    summary = dict(
        status="passed",
        environment=simulation["environment"],
        tests=test_reports,
        new_cann_operator_families=0,
        network=dict(
            strict_entries=645,
            input_shape=network["input_shape"],
            precision=network["precision"],
        ),
        simulation=dict(
            physics_mode=simulation["physics_mode"],
            gaussians=simulation["gaussians"],
            filled_particles=simulation["filled_particles"],
            particles=simulation["simulation_particles"],
            frames=count,
            substeps=simulation["final_step"],
            physical_time=simulation["final_time"],
            run_seconds=simulation["run_seconds"],
            render_seconds=simulation["render_seconds"],
            sequence_export_seconds=sum(
                item["sequence_export_seconds"] for item in simulation["frames"]
            ),
            peak_npu_allocated_bytes=simulation["npu_peak_allocated_bytes"],
        ),
        replay=dict(
            maximum_online_offline_absolute_error=fixed[
                "max_online_replay_absolute_error"
            ],
            fixed_seconds=fixed["run_seconds"],
            moving_seconds=orbit["run_seconds"],
            network_loaded=False,
            mpm_substeps_executed=0,
        ),
        render_profile=dict(backend="cann", existing_m2_kernels=kernels, mpm_kernels=0),
        videos=videos,
        sequence=dict(
            format=manifest["format"],
            scene_sha256=manifest["scene_sha256"],
            frame_checksums=True,
        ),
        gpu_warp_alignment="pending: same-input CUDA/Warp reference pack unavailable",
        dynamic_render_backward="out of scope",
        mpm_backward="out of scope",
    )
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "dynamic_cake.json").write_text(json.dumps(simulation, indent=2) + "\n")
    (reports / "dynamic_replay_fixed.json").write_text(
        json.dumps(fixed, indent=2) + "\n"
    )
    (reports / "dynamic_replay_orbit.json").write_text(
        json.dumps(orbit, indent=2) + "\n"
    )
    (reports / "dynamic_verification.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
