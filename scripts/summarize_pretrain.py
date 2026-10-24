"""Small, reviewable acceptance report separate from large training checkpoints."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "passed", "runs": {}}
    for name in ("short100", "scale512", "overfit20", "scratch2", "lpips1", "fp16", "continuous", "resumed", "hccl"):
        report = json.loads((args.root/name/"run_report.json").read_text())
        assert report["status"] == "passed" and report["dino_unchanged"]
        result["runs"][name] = {k:report[k] for k in ("status","global_step","micro_step","world_size","start_global_step","memory","elapsed_seconds","dataloader")}
    overfit = json.loads((args.root/"overfit20/run_report.json").read_text())["updates"]
    first, last = statistics.mean(v["total_loss"] for v in overfit[:5]), statistics.mean(v["total_loss"] for v in overfit[-5:])
    assert last < first
    result["overfit_loss"] = {"first5_mean":first,"last5_mean":last,"decreased":True}
    result["resume"] = json.loads((args.root/"resume_comparison.json").read_text())
    result["multi_npu"] = "passed" if result["runs"]["hccl"]["world_size"] >= 2 else "pending: only one exposed device"
    result["gpu_reference_alignment"] = "pending: no reference package"
    result["restored_inference"] = json.loads((args.root/"restored_inference/run_report.json").read_text())["status"]
    result["restored_render"] = json.loads((args.root/"restored_render/run_report.json").read_text())["status"]
    (args.root/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
