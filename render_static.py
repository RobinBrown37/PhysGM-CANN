"""Render static RGB/alpha, replay CPU packs and verify image-to-network VJPs.

This diagnostic never constructs a supervised training loop or optimizer.
"""
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict

from data.dataset_infer import Dataset
from export import to_cpu, write_json
from model.physgm import load_checkpoint
from rendering import StaticGaussianRenderer
from runtime import autocast, environment_report, memory_report, reset_peak_memory, select_device, synchronize

ROOT = Path(__file__).resolve().parent
FIELDS = ("xyz", "feature", "scale", "rotation", "opacity")


def image_vjp(image):
    """A fixed spatial upstream gradient, independent of labels or predictions."""
    rgb = torch.linspace(-0.3, 0.7, image.rgb.numel(), device=image.rgb.device).reshape_as(image.rgb) / image.rgb.numel()
    alpha = torch.linspace(0.2, 0.8, image.alpha.numel(), device=image.alpha.device).reshape_as(image.alpha) / image.alpha.numel()
    torch.autograd.backward((image.rgb, image.alpha), (rgb, alpha))


def check_gradients(gaussians, model=None):
    result = {}
    for key in FIELDS:
        grad = gaussians[key].grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"Missing/nonfinite rendered-image gradient: {key}")
        absolute_sum = float(grad.abs().sum().cpu())
        if absolute_sum == 0:
            raise RuntimeError(f"All rendered-image gradients are zero: {key}")
        result[key] = {"finite": True, "absolute_sum": absolute_sum, "nonzero_elements": int((grad != 0).sum().cpu()), "elements": grad.numel()}
    if model is not None:
        modules = {}
        for name in ("ray_encoder", "dino_proj", "input_layernorm", "processor", "dpt_head"):
            params = [p for p in getattr(model, name).parameters() if p.requires_grad]
            gradients = [p.grad for p in params]
            if any(g is None or not torch.isfinite(g).all() for g in gradients):
                raise RuntimeError(f"Missing/nonfinite network image gradient: {name}")
            magnitude = sum(float(g.abs().sum().cpu()) for g in gradients)
            if magnitude == 0:
                raise RuntimeError(f"Zero network image gradient: {name}")
            modules[name] = {"parameters_with_grad": len(gradients), "finite": True, "absolute_sum": magnitude}
        dino_frozen = all(not p.requires_grad and p.grad is None for p in model.dinov3_extractor.parameters())
        physical_unused = all(p.grad is None for name in ("phys_token_decoder", "E_token_decoder", "nu_token_decoder") for p in getattr(model, name).parameters())
        global_grad = model.global_token_init.grad
        global_valid = global_grad is not None and bool(torch.isfinite(global_grad).all()) and bool(global_grad.abs().sum() > 0)
        if not dino_frozen or model.dinov3_extractor.model.training or not physical_unused or not global_valid:
            raise RuntimeError("DINO freeze/eval, global-token or unused physical-head check failed")
        result["network"] = modules
        result.update(dino_frozen=True, dino_eval=True, physical_heads_have_no_image_gradient=True, global_tokens_have_image_gradient=True)
    return result


def pngs(image, directory):
    for batch in range(image.rgb.shape[0]):
        for view in range(image.rgb.shape[1]):
            rgb = image.rgb[batch, view].detach().cpu().permute(1, 2, 0).clamp(0, 1)
            alpha = image.alpha[batch, view, 0].detach().cpu().clamp(0, 1)
            Image.fromarray((rgb.numpy()*255).round().astype(np.uint8)).save(directory/f"rgb_b{batch}_v{view}.png")
            Image.fromarray((alpha.numpy()*255).round().astype(np.uint8)).save(directory/f"alpha_b{batch}_v{view}.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-pack", "--prediction-pack", type=Path, help="M1 prediction.pt or renderer_input.pt; CPU tensors only")
    source.add_argument("--online", action="store_true", help="Network prediction followed by rendering (default without a pack)")
    parser.add_argument("--camera-pack", type=Path, help="CPU dict: c2ws [B,T,4,4], intr [B,T,4], width, height")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/network_npu.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoint.pt")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--backend", choices=("cann", "reference"), default="cann")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16", help="Network precision; renderer stays FP32")
    parser.add_argument("--input-resize", type=int, default=None)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--views", type=int, help="Explicitly select the first K target views")
    parser.add_argument("--backward-probe", action="store_true", help="Fixed RGB/alpha upstream gradients; no optimizer")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/static_render")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0 or (args.views is not None and args.views < 1):
        parser.error("positive repeats/views and nonnegative warmup are required")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "mode": "image_vjp_no_training" if args.backward_probe else "static_inference"}
    started = time.perf_counter()
    try:
        torch.set_num_threads(8)
        torch.manual_seed(1111)
        device = select_device(args.device)
        report["environment"] = environment_report(device)
        cfg = EasyDict(yaml.safe_load(args.config.read_text()))
        model = None
        parameters = dict(sh_degree=cfg.model.gaussians.sh_degree,
                          near_plane=cfg.model.gaussians.near_plane,
                          far_plane=cfg.model.gaussians.far_plane, eps2d=0.3,
                          background=[1., 1., 1.])
        if args.input_pack:
            pack = torch.load(args.input_pack, map_location="cpu", weights_only=True)
            parameters.update(pack.get("parameters", {}))
            if "prediction" in pack:
                source_g = pack["prediction"]["gaussians"]
                cameras, intr = pack["input"]["input_c2ws"], pack["input"]["input_intr"]
                base_h, base_w = pack["input"]["input_images"].shape[-2:]
            else:
                source_g = pack["gaussians"]
                cameras, intr = pack["c2ws"], pack["intr"]
                base_w, base_h = pack["width"], pack["height"]
            gaussians = {k: source_g[k].to(device).detach().requires_grad_(args.backward_probe) for k in FIELDS}
            report["source"] = str(args.input_pack)
            report["source_gaussians"] = gaussians["xyz"].shape[1]
        else:
            if args.input_resize is not None:
                cfg.data.resize_h = args.input_resize
            elif args.backward_probe:
                cfg.data.resize_h = 256
            cpu_batch = next(iter(DataLoader([Dataset(cfg)[0]], batch_size=1)))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
            model, report["checkpoint"] = load_checkpoint(cfg, args.checkpoint, device)
            model.train(args.backward_probe)
            cameras, intr = batch["input_c2ws"], batch["input_intr"]
            base_h, base_w = batch["input_images"].shape[-2:]
            report["source"] = "online M1 prediction"
        if args.camera_pack:
            camera_pack = torch.load(args.camera_pack, map_location="cpu", weights_only=True)
            cameras, intr = camera_pack["c2ws"], camera_pack["intr"]
            base_w, base_h = camera_pack["width"], camera_pack["height"]
        width, height = args.width or base_w, args.height or base_h
        cameras, intr = cameras.to(device).float(), intr.to(device).float().clone()
        intr[..., (0, 2)] *= width/base_w
        intr[..., (1, 3)] *= height/base_h
        if args.views:
            if args.views > cameras.shape[1]:
                raise ValueError("--views exceeds the supplied camera count")
            cameras, intr = cameras[:, :args.views], intr[:, :args.views]
        renderer = StaticGaussianRenderer(
            **{k: parameters[k] for k in ("sh_degree", "near_plane", "far_plane", "eps2d")}, backend=args.backend,
        )
        report["renderer_parameters"] = parameters
        def forward():
            if model is not None:
                with autocast(device, args.amp_dtype):
                    prediction = model(batch)
                current = prediction["gaussians"]
            else:
                current = gaussians
            if args.backward_probe:
                for tensor in current.values():
                    tensor.retain_grad()
            image = renderer(current, cameras, intr, width=width, height=height, background=parameters["background"])
            return current, image
        context = nullcontext if args.backward_probe else torch.no_grad
        reset_peak_memory(device)
        times = []
        with context():
            for i in range(args.warmup+args.repeats):
                synchronize(device)
                before = time.perf_counter()
                current, image = forward()
                synchronize(device)
                elapsed = time.perf_counter()-before
                if i == 0:
                    report["first_forward_seconds"] = elapsed
                if i >= args.warmup:
                    times.append(elapsed)
                if i+1 < args.warmup+args.repeats:
                    del current, image
        report.update(timed_forward_seconds=times, forward_median_seconds=statistics.median(times), forward_p95_seconds=float(np.quantile(times, 0.95)), stats=image.stats, width=width, height=height, views=cameras.shape[1], input_gaussians=current["xyz"].shape[1])
        for name, tensor in (("rgb", image.rgb), ("alpha", image.alpha)):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite {name}")
        if image.alpha.min() < -2e-4 or image.alpha.max() > 1.0002:
            raise RuntimeError("Alpha outside [0,1] beyond numerical tolerance")
        report["alpha_range"] = [float(image.alpha.min().cpu()), float(image.alpha.max().cpu())]
        if args.backward_probe:
            synchronize(device)
            before = time.perf_counter()
            image_vjp(image)
            synchronize(device)
            report["backward_seconds"] = time.perf_counter()-before
            report["gradients"] = check_gradients(current, model)
        report["memory"] = memory_report(device)
        cpu_g = to_cpu(current)
        cpu_output = {"rgb": to_cpu(image.rgb), "alpha": to_cpu(image.alpha)}
        if args.backward_probe:
            cpu_output["gradients"] = {k: to_cpu(current[k].grad) for k in FIELDS}
        torch.save({"gaussians": cpu_g, "c2ws": to_cpu(cameras), "intr": to_cpu(intr), "width": width, "height": height, "parameters": report["renderer_parameters"]}, output/"renderer_input.pt")
        torch.save(cpu_output, output/"renderer_output.pt")
        pngs(image, output)
        if args.profile:
            import torch_npu
            if model is not None:
                model.zero_grad(set_to_none=True)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1), record_shapes=True,
            ) as trace:
                with context():
                    profiled_g, profiled = forward()
                if args.backward_probe:
                    image_vjp(profiled)
                synchronize(device)
                trace.step()
            trace.export_chrome_trace(str(output/"operators.json"))
            payload = json.loads((output/"operators.json").read_text())
            events = payload if isinstance(payload, list) else payload["traceEvents"]
            names = sorted({e["name"] for e in events if e.get("ph") == "X" and "name" in e})
            report["render_operators"] = [n for n in names if any(k in n.lower() for k in ("calcrender", "flashgaussian", "gaussiansort"))]
            write_json(output/"operator_names.json", names)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["total_seconds_including_io"] = time.perf_counter()-started
        write_json(output/"run_report.json", report)
    print(f"Passed: {output/'run_report.json'}", flush=True)


if __name__ == "__main__":
    main()
