"""Create the compact NPU reference bundle required by the GPU benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source, destination):
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_manifest(destination):
    records = []
    outer_manifest = destination / "manifest.json"
    for path in sorted(p for p in destination.rglob("*") if p.is_file() and p != outer_manifest):
        records.append({
            "path": str(path.relative_to(destination)), "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "format": "physgm-gpu-handoff-v1",
        "expected_gpu_path": "/home/developer/datasets/physassets_gpu_reference",
        "required_dataset_path": "/home/developer/datasets/physassets_benchmark_512",
        "required_checkpoint_path": "/mnt/workspace/PhysGM-CANN/checkpoint.pt",
        "files": records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(source_root, destination):
    files = {
        source_root / "summary.json": destination / "npu/summary.json",
        source_root / "regression_final.xml": destination / "npu/regression_final.xml",
        source_root / "m1_m2_dataset_512/per_scene.json": destination / "npu/m1_m2/per_scene.json",
        source_root / "m1_m2_dataset_512/reference_samples.pt": destination / "npu/m1_m2/reference_samples.pt",
        source_root / "m1_m2_dataset_512/run_report.json": destination / "npu/m1_m2/run_report.json",
        source_root / "m1_forward_512/prediction.pt": destination / "npu/m1_m2/cake_prediction.pt",
        source_root / "m3_dataset_train_100_256/config.yaml": destination / "npu/m3_100/config.yaml",
        source_root / "m3_dataset_train_100_256/data_manifest.json": destination / "npu/m3_100/data_manifest.json",
        source_root / "m3_dataset_train_100_256/metrics.jsonl": destination / "npu/m3_100/metrics.jsonl",
        source_root / "m3_dataset_train_100_256/run_report.json": destination / "npu/m3_100/run_report.json",
        source_root / "m3_dataset_train_100_256/training_reference.pt": destination / "npu/m3_100/training_reference.pt",
        source_root / "m3_dataset_train_100_256/training_reference.json": destination / "npu/m3_100/training_reference.json",
        source_root / "m3_dataset_train_1_512/config.yaml": destination / "npu/m3_512/config.yaml",
        source_root / "m3_dataset_train_1_512/metrics.jsonl": destination / "npu/m3_512/metrics.jsonl",
        source_root / "m3_dataset_train_1_512/run_report.json": destination / "npu/m3_512/run_report.json",
        source_root / "m5_dynamic_50/initial_state.pt": destination / "npu/m4_m5/initial_state.pt",
        source_root / "m5_dynamic_50/final_state.pt": destination / "npu/m4_m5/final_state.pt",
        source_root / "m5_dynamic_50/cameras.pt": destination / "npu/m4_m5/cameras.pt",
        source_root / "m5_dynamic_50/config.json": destination / "npu/m4_m5/config.json",
        source_root / "m5_dynamic_50/report.json": destination / "npu/m4_m5/report.json",
        source_root / "m5_dynamic_50/pipeline_report.json": destination / "npu/m4_m5/pipeline_report.json",
        source_root / "m5_dynamic_50/preview.png": destination / "npu/m4_m5/preview.png",
        source_root / "m5_dynamic_50/simulation.mp4": destination / "npu/m4_m5/simulation.mp4",
        ROOT / "NPU-BENCHMARK-REPORT.md": destination / "docs/NPU-BENCHMARK-REPORT.md",
        ROOT / "GPU-BENCHMARK-HANDOFF.md": destination / "docs/GPU-BENCHMARK-HANDOFF.md",
        ROOT / "benchmark_inference.py": destination / "schema/benchmark_inference.py",
        ROOT / "scripts/export_training_reference.py": destination / "schema/export_training_reference.py",
        ROOT / "configs/network_npu.yaml": destination / "configs/network_npu.yaml",
        ROOT / "configs/pretrain_smoke.yaml": destination / "configs/pretrain_smoke.yaml",
        ROOT / "configs/pretrain_npu.yaml": destination / "configs/pretrain_npu.yaml",
        ROOT / "configs/mpm/cake.json": destination / "configs/mpm/cake.json",
    }
    for source, target in files.items():
        copy_file(source, target)
    directories = {
        source_root / "m5_dynamic_50/sequence": destination / "npu/m4_m5/sequence",
        source_root / "m5_dynamic_50/render_tensors": destination / "npu/m4_m5/render_tensors",
    }
    for source, target in directories.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, target)
    return write_manifest(destination)


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for record in manifest["files"]:
        path = directory / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Handoff checksum mismatch: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--refresh-docs", action="store_true")
    args = parser.parse_args()
    if args.verify_existing and args.refresh_docs:
        parser.error("--verify-existing and --refresh-docs are mutually exclusive")
    if args.verify_existing:
        manifest = verify(args.output_dir)
    elif args.refresh_docs:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(args.output_dir)
        copy_file(ROOT / "NPU-BENCHMARK-REPORT.md", args.output_dir / "docs/NPU-BENCHMARK-REPORT.md")
        copy_file(ROOT / "GPU-BENCHMARK-HANDOFF.md", args.output_dir / "docs/GPU-BENCHMARK-HANDOFF.md")
        manifest = write_manifest(args.output_dir)
    else:
        if args.source_root is None:
            parser.error("--source-root is required unless --verify-existing is used")
        if args.output_dir.exists():
            raise FileExistsError(f"Refusing to replace existing output: {args.output_dir}")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
        try:
            manifest = build(args.source_root.resolve(), temporary)
            temporary.replace(args.output_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    total = sum(item["bytes"] for item in manifest["files"])
    print(json.dumps({
        "status": "passed", "files": len(manifest["files"]), "bytes": total,
        "output": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
