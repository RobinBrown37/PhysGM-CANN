"""Compare actual same-device supervised loss, gradients and AdamW update vectors."""
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    a = torch.load(args.continuous/"last.pt", weights_only=True, map_location="cpu", mmap=True)
    b = torch.load(args.resumed/"last.pt", weights_only=True, map_location="cpu", mmap=True)
    ap = torch.load(args.continuous/"update_probe.pt", weights_only=True, map_location="cpu", mmap=True)
    bp = torch.load(args.resumed/"update_probe.pt", weights_only=True, map_location="cpu", mmap=True)
    assert a["global_step"] == b["global_step"] == ap["global_step"] == bp["global_step"]
    assert a["micro_step"] == b["micro_step"]
    assert a["scheduler"] == b["scheduler"]
    assert a["scaler"] == b["scaler"]
    assert torch.equal(ap["sample_key"], bp["sample_key"])
    for ar, br in zip(a["ranks"], b["ranks"]):
        assert ar["data"] == br["data"]
        assert ar["rng"]["python"] == br["rng"]["python"]
        assert ar["rng"]["numpy"] == br["rng"]["numpy"]
        for key in ("torch", "npu"):
            if key in ar["rng"]:
                assert torch.equal(ar["rng"][key], br["rng"][key])
    def last_loss(directory):
        values = [json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()]
        return next(v["total_loss"] for v in reversed(values) if "total_loss" in v)
    loss_a, loss_b = last_loss(args.continuous), last_loss(args.resumed)
    assert abs(loss_b-loss_a) <= 1e-5+1e-3*abs(loss_a)
    groups = {}
    exact = True
    for name, before in ap["parameters_before"].items():
        group = name.split(".")[0]
        entry = groups.setdefault(group, {"gradient_error_squared": 0., "gradient_reference_squared": 0.,
                                           "update_error_squared": 0., "update_reference_squared": 0.})
        ga, gb = ap["gradients"][name].double(), bp["gradients"][name].double()
        da, db = a["model"][name].double()-before.double(), b["model"][name].double()-bp["parameters_before"][name].double()
        entry["gradient_error_squared"] += float((gb-ga).square().sum())
        entry["gradient_reference_squared"] += float(ga.square().sum())
        entry["update_error_squared"] += float((db-da).square().sum())
        entry["update_reference_squared"] += float(da.square().sum())
        exact &= torch.equal(a["model"][name], b["model"][name]) and torch.equal(ga, gb)
    for entry in groups.values():
        for kind in ("gradient", "update"):
            error, reference = entry[kind+"_error_squared"]**.5, entry[kind+"_reference_squared"]**.5
            entry[kind+"_relative_l2"] = error/reference if reference else 0.
            assert error <= .01*reference if reference else error <= 1e-7
    report = {"status": "passed", "global_step": a["global_step"], "loss_continuous": loss_a,
              "loss_resumed": loss_b, "all_parameters_and_gradients_exact": exact, "groups": groups,
              "data_rng_lr_scaler_match": True,
              "tolerances": {"loss_rtol": 1e-3, "loss_atol": 1e-5, "gradient_update_relative_l2": .01, "zero_reference_atol": 1e-7}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
