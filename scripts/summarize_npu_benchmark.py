"""Validate and summarize the reproducible PhysGM M1-M5 NPU benchmark run."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def load(path):
    value = json.loads(path.read_text())
    if value.get("status") != "passed":
        raise RuntimeError(f"Run did not pass: {path}: {value.get('status')}")
    return value


def timing(records, key):
    values = sorted(float(record[key]) for record in records)
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "p95": values[int(.95 * (len(values) - 1))],
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tests = ET.parse(args.tests).getroot()
    suites = [tests] if tests.tag == "testsuite" else list(tests)
    test_summary = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                    for key in ("tests", "failures", "errors", "skipped")}
    if test_summary["failures"] or test_summary["errors"]:
        raise RuntimeError(f"NPU regression failures: {test_summary}")
    dataset = json.loads(args.dataset_manifest.read_text())
    m1 = load(args.root / "m1_forward_512/run_report.json")
    m1_backward = load(args.root / "m1_backward_256/run_report.json")
    m2 = load(args.root / "m2_forward_512/run_report.json")
    m2_backward = load(args.root / "m2_backward_256/run_report.json")
    inference = load(args.root / "m1_m2_dataset_512/run_report.json")
    train = load(args.root / "m3_dataset_train_100_256/run_report.json")
    train512 = load(args.root / "m3_dataset_train_1_512/run_report.json")
    dynamic = load(args.root / "m5_dynamic_50/report.json")
    replay = load(args.root / "m5_dynamic_50/replay_fixed/report.json")
    orbit = load(args.root / "m5_dynamic_50/replay_orbit/report.json")
    updates = train["updates"]
    summary = {
        "status": "passed",
        "environment": inference["environment"],
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "dataset": {
            "path": str(args.dataset_manifest.parent.resolve()),
            "manifest_sha256": sha256(args.dataset_manifest),
            "archive_sha256": dataset["archive"]["archive_sha256"],
            "scenes": len(dataset["scenes"]),
            "train": sum(item["split"] == "train" for item in dataset["scenes"]),
            "validation": sum(item["split"] == "validation" for item in dataset["scenes"]),
            "quotas": dataset["quotas"],
        },
        "regression": test_summary,
        "M1": {
            "cake_512_forward_median_seconds": m1["forward_median_seconds"],
            "cake_256_backward_seconds": m1_backward["backward_seconds"],
            "dataset_512": inference["summary"]["network_seconds"],
            "dataset_scenes": inference["summary"]["scenes"],
        },
        "M2": {
            "cake_512_four_view_median_seconds": m2["forward_median_seconds"],
            "cake_256_backward_seconds": m2_backward["backward_seconds"],
            "dataset_512_eight_view": inference["summary"]["render_seconds"],
            "dataset_rgb_mse": inference["summary"]["rgb_mse"],
            "dataset_psnr": inference["summary"]["psnr"],
            "dataset_alpha_mse": inference["summary"]["alpha_mse"],
        },
        "physical_prediction": {
            "material_accuracy": inference["summary"]["material_accuracy"],
            "E_normalized_mae": inference["summary"]["E_normalized_mae"],
            "nu_normalized_mae": inference["summary"]["nu_normalized_mae"],
        },
        "M3": {
            "steps": train["global_step"], "micro_steps": train["micro_step"],
            "skipped_windows": train["skipped_windows"],
            "forward_seconds": timing(updates, "forward_seconds"),
            "backward_seconds": timing(updates, "backward_seconds"),
            "optimizer_seconds": timing(updates, "step_seconds"),
            "core_update_seconds": timing([
                {"total": record["forward_seconds"] + record["backward_seconds"] + record["step_seconds"]}
                for record in updates], "total"),
            "validation": train["validation"][-1],
            "dino_unchanged": train["dino_unchanged"],
            "peak_npu_allocated_bytes": train["memory"]["npu_peak_allocated_bytes"],
            "scale_512": {
                "optimizer_steps": train512["global_step"], "micro_steps": train512["micro_step"],
                "forward_seconds": train512["updates"][0]["forward_seconds"],
                "backward_seconds": train512["updates"][0]["backward_seconds"],
                "optimizer_seconds": train512["updates"][0]["step_seconds"],
                "peak_npu_allocated_bytes": train512["memory"]["npu_peak_allocated_bytes"],
            },
        },
        "M4_M5": {
            "physics_mode": dynamic["physics_mode"], "gaussians": dynamic["gaussians"],
            "particles": dynamic["simulation_particles"], "filled_particles": dynamic["filled_particles"],
            "frames": dynamic["frames_requested"], "substeps": dynamic["final_step"],
            "physical_time": dynamic["final_time"], "simulation_seconds": dynamic["run_seconds"],
            "render_seconds": dynamic["render_seconds"],
            "peak_npu_allocated_bytes": dynamic["npu_peak_allocated_bytes"],
            "offline_replay_seconds": replay["run_seconds"],
            "online_replay_max_absolute_error": replay["max_online_replay_absolute_error"],
            "orbit_replay_seconds": orbit["run_seconds"],
        },
        "comparison_artifacts": {
            "per_scene": str((args.root / "m1_m2_dataset_512/per_scene.json").resolve()),
            "reference_samples": str((args.root / "m1_m2_dataset_512/reference_samples.pt").resolve()),
            "dynamic_sequence": str((args.root / "m5_dynamic_50/sequence").resolve()),
        },
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
