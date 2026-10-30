"""Export compact deterministic samples from a large migrated training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sample(tensor, count):
    value = tensor.detach().cpu().contiguous()
    flat = value.reshape(-1)
    if flat.numel() == 0:
        indices = torch.empty(0, dtype=torch.long)
        samples = torch.empty(0, dtype=torch.float32)
    else:
        indices = torch.linspace(0, flat.numel() - 1, min(count, flat.numel())).long().unique()
        samples = flat[indices].float()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "indices": indices, "values": samples,
        "finite": bool(torch.isfinite(samples).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-tensor", type=int, default=32)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model = {
        name: tensor_sample(tensor, args.samples_per_tensor)
        for name, tensor in checkpoint["model"].items()
    }
    optimizer = {}
    for parameter, state in checkpoint["optimizer"]["state"].items():
        optimizer[str(parameter)] = {
            name: tensor_sample(value, args.samples_per_tensor) if isinstance(value, torch.Tensor) else value
            for name, value in state.items()
        }
    metrics_bytes = args.metrics.read_bytes()
    report_bytes = args.run_report.read_bytes()
    result = {
        "format": "physgm-training-reference-v1",
        "checkpoint_sha256": sha256(args.checkpoint),
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "run_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "global_step": checkpoint["global_step"],
        "micro_step": checkpoint["micro_step"],
        "skipped_windows": checkpoint["skipped_windows"],
        "signature": checkpoint["signature"],
        "param_names": checkpoint["param_names"],
        "scheduler": checkpoint["scheduler"],
        "model": model,
        "optimizer": optimizer,
        "metrics": [json.loads(line) for line in metrics_bytes.decode().splitlines()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    manifest = {
        "status": "passed", "format": result["format"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "global_step": result["global_step"], "micro_step": result["micro_step"],
        "model_tensors": len(model), "optimizer_states": len(optimizer),
        "samples_per_tensor": args.samples_per_tensor,
        "output": str(args.output.resolve()), "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
