"""Real-weight network inference and an optional, explicitly synthetic gradient probe."""

import os

# Avoid implicit torch_npu initialization during CPU debug / --help. NPU imports
# happen explicitly in select_device(), before tensors are moved to the device.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import time

import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from data.dataset_infer import Dataset
from export import physical_values, save_gaussian_ply, to_cpu, write_json
from model.physgm import load_checkpoint
from runtime import autocast, environment_report, memory_report, reset_peak_memory, select_device, synchronize


ROOT = Path(__file__).resolve().parent


def gradient_probe(prediction):
    """Exercise every output branch; this is not a supervised training loss."""
    terms = [t.float().square().mean() for t in prediction["gaussians_unpruned"].values()]
    terms.extend(prediction[k].float().square().mean() for k in ("phys_logits", "E_mu", "E_var", "nu_mu", "nu_var"))
    return sum(terms)


def gradient_report(model):
    groups = ("ray_encoder", "dino_proj", "input_layernorm", "processor", "dpt_head", "phys_token_decoder", "E_token_decoder", "nu_token_decoder")
    report = {}
    for name in groups:
        parameters = list(getattr(model, name).parameters())
        gradients = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in gradients)
        magnitude = sum(float(g.detach().float().abs().sum().cpu()) for g in gradients)
        report[name] = {"parameters_with_grad": len(gradients), "trainable_parameters": sum(p.requires_grad for p in parameters), "finite": finite, "absolute_sum": magnitude}
        if len(gradients) != sum(p.requires_grad for p in parameters) or not finite or magnitude == 0:
            raise RuntimeError(f"Gradient probe failed for {name}: {report[name]}")
    dino = list(model.dinov3_extractor.parameters())
    report["dino_frozen"] = all(not p.requires_grad and p.grad is None for p in dino)
    report["dino_eval"] = not model.dinov3_extractor.model.training
    global_grad = model.global_token_init.grad
    report["global_tokens"] = bool(global_grad is not None and torch.isfinite(global_grad).all() and global_grad.abs().sum() > 0)
    if not all(report[k] for k in ("dino_frozen", "dino_eval", "global_tokens")):
        raise RuntimeError("DINO freeze/eval or global-token gradient check failed")
    return report


def validate_prediction(prediction, batch):
    B, V, _, H, W = batch["input_images"].shape
    raw = prediction["gaussians_unpruned"]
    if raw["xyz"].shape != (B, V * H * W, 3):
        raise RuntimeError("Unexpected unpruned Gaussian shape")
    for name, tensor in {**raw, **{k: prediction[k] for k in ("phys_logits", "E_mu", "E_var", "nu_mu", "nu_var")}}.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite prediction: {name}")
    if not (prediction["E_var"] > 0).all() or not (prediction["nu_var"] > 0).all():
        raise RuntimeError("Physical variances must be positive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/network_npu.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoint.pt")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--resize", type=int, help="Override input height (gradient probe: 256)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/cake")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--backward-probe", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8, help="CPU intra-op threads")
    parser.add_argument("--profile", action="store_true", help="Record CPU/NPU operators separately from timing")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup nonnegative")
    if args.backward_probe and args.resize is None:
        args.resize = 256
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "running", "mode": "gradient_probe_not_supervised_training" if args.backward_probe else "inference"}
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(1111)
        config = EasyDict(yaml.safe_load(args.config.read_text()))
        if args.resize is not None:
            config.data.resize_h = args.resize
        precision = args.amp_dtype or (config.amp_dtype if config.get("use_amp", True) else "fp32")
        device = select_device(args.device)
        report["environment"] = environment_report(device)
        report["precision"] = precision
        report["cpu_work"] = ["image decode/resize and camera normalization", "checkpoint load and artifact export", "cached ray bicubic/antialias and DPT FP64 positional constants"]
        cpu_batch = next(iter(DataLoader([Dataset(config)[args.scene_index]], batch_size=1)))
        transfer_start = time.perf_counter()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
        synchronize(device)
        report["input_transfer_seconds"] = time.perf_counter() - transfer_start
        report["input_shape"] = list(batch["input_images"].shape)
        load_start = time.perf_counter()
        model, checkpoint_report = load_checkpoint(config, args.checkpoint, device=device)
        model.train(args.backward_probe)
        synchronize(device)
        report["checkpoint"] = checkpoint_report
        report["model_load_and_transfer_seconds"] = time.perf_counter() - load_start
        print(f"Loaded {checkpoint_report['model_entries']} entries strictly on {device}; input {report['input_shape']}", flush=True)
        reset_peak_memory(device)
        times = []
        first_forward = None
        gradient_context = nullcontext if args.backward_probe else torch.no_grad
        with gradient_context():
            for iteration in range(args.warmup + args.repeats):
                synchronize(device)
                before = time.perf_counter()
                with autocast(device, precision):
                    prediction = model(batch, return_unpruned=True)
                synchronize(device)
                elapsed = time.perf_counter() - before
                if first_forward is None:
                    first_forward = elapsed
                if iteration >= args.warmup:
                    times.append(elapsed)
                # Only the final forward graph is retained for the probe.
                if iteration + 1 < args.warmup + args.repeats:
                    del prediction
        report["first_forward_seconds"] = first_forward
        report["timed_forward_seconds"] = times
        report["warmup_runs"] = args.warmup
        report["forward_median_seconds"] = float(torch.tensor(times, dtype=torch.float64).median())
        report["forward_p95_seconds"] = float(torch.quantile(torch.tensor(times, dtype=torch.float64), 0.95))
        validate_prediction(prediction, batch)
        if args.backward_probe:
            probe = gradient_probe(prediction)
            synchronize(device)
            before = time.perf_counter()
            probe.backward()
            synchronize(device)
            report["backward_seconds"] = time.perf_counter() - before
            report["gradient_probe_scalar"] = float(probe.detach().cpu())
            report["gradients"] = gradient_report(model)
        report["memory"] = memory_report(device)
        if args.profile:
            profile_kwargs = {}
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "npu":
                import torch_npu
                profiler = torch_npu.profiler
                activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
                profile_kwargs["schedule"] = profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
            else:
                profiler = torch.profiler
            # Profiling is a separate replay, outside measured forward/backward.
            model.zero_grad(set_to_none=True)
            with profiler.profile(activities=activities, record_shapes=True, **profile_kwargs) as trace:
                with gradient_context(), autocast(device, precision):
                    profiled = model(batch, return_unpruned=True)
                if args.backward_probe:
                    gradient_probe(profiled).backward()
                synchronize(device)
                if device.type == "npu":
                    trace.step()
            trace.export_chrome_trace(str(args.output_dir / "operators.json"))
            # TorchNPU's profiler exports an event list and does not implement
            # PyTorch's key_averages(). Accept either Chrome trace wire format.
            payload = json.loads((args.output_dir / "operators.json").read_text())
            events = payload if isinstance(payload, list) else payload["traceEvents"]
            operators = sorted({event["name"] for event in events if event.get("ph") == "X" and "name" in event})
            report["attention_operators"] = [op for op in operators if any(s in op.lower() for s in ("attention", "flash", "fusion"))]
            write_json(args.output_dir / "operator_names.json", operators)
            del profiled
        export_start = time.perf_counter()
        cpu_prediction = to_cpu(prediction)
        torch.save({"input": cpu_batch, "prediction": cpu_prediction, "checkpoint": checkpoint_report}, args.output_dir / "prediction.pt")
        physics = physical_values(cpu_prediction)
        write_json(args.output_dir / "predicted_phys.json", physics)
        report["physical_values"] = physics
        report["unpruned_gaussians"] = cpu_prediction["gaussians_unpruned"]["xyz"].shape[1]
        report["ply_points"] = save_gaussian_ply(
            {k: v[0] for k, v in cpu_prediction["gaussians"].items()},
            args.output_dir / "point_clouds.ply",
            config.model.gaussians.get("opacity_threshold", 0.001),
        )
        report["export_seconds"] = time.perf_counter() - export_start
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["total_seconds_including_io"] = time.perf_counter() - started
        write_json(args.output_dir / "run_report.json", report)
    print(f"Passed: {args.output_dir / 'run_report.json'}", flush=True)


if __name__ == "__main__":
    main()
