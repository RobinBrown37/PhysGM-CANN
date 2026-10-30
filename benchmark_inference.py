"""Run the fixed PhysAssets M1/M2 benchmark and export GPU-comparable samples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from export import MATERIALS, to_cpu, write_json
from model.physgm import load_checkpoint
from rendering import StaticGaussianRenderer
from runtime import autocast, environment_report, memory_report, reset_peak_memory, select_device, synchronize
from training.config import load_config, plain_config
from training.data import SupervisedDataset


ROOT = Path(__file__).resolve().parent
FIELDS = ("xyz", "feature", "scale", "rotation", "opacity")


def device_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p95": float(np.quantile(array, .95)), "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def sampled_reference(scene, archive_scene, prediction, image, gaussian_samples, pixel_samples):
    gaussians = prediction["gaussians_unpruned"]
    count = gaussians["xyz"].shape[1]
    indices = torch.linspace(0, count - 1, gaussian_samples, device=gaussians["xyz"].device).long().unique()
    pixels = torch.cat((image.rgb, image.alpha), 2).permute(0, 1, 3, 4, 2).reshape(-1, 4)
    pixel_indices = torch.linspace(0, pixels.shape[0] - 1, pixel_samples, device=pixels.device).long().unique()
    return {
        "scene_name": scene,
        "archive_scene": archive_scene,
        "gaussian_count": count,
        "gaussian_indices": indices.cpu(),
        "gaussians": {name: gaussians[name][0, indices].detach().float().cpu() for name in FIELDS},
        "physical": {name: prediction[name].detach().float().cpu() for name in ("phys_logits", "E_mu", "E_var", "nu_mu", "nu_var")},
        "render_shape": list(torch.cat((image.rgb, image.alpha), 2).shape),
        "pixel_indices": pixel_indices.cpu(),
        "render_rgba": pixels[pixel_indices].detach().float().cpu(),
    }


def metrics(prediction, image, batch):
    target = batch["test_images"].float()
    target_rgb = target[:, :, :3] * target[:, :, 3:] + 1.0 - target[:, :, 3:]
    rgb_mse = (image.rgb.float() - target_rgb).square().mean()
    alpha_mse = (image.alpha.float() - target[:, :, 3:]).square().mean()
    predicted_class = prediction["phys_logits"].float().argmax(-1)
    return {
        "rgb_mse": float(rgb_mse.cpu()),
        "psnr": float((-10.0 * torch.log10(rgb_mse)).cpu()),
        "alpha_mse": float(alpha_mse.cpu()),
        "predicted_class": int(predicted_class.cpu()),
        "target_class": int(batch["physical_class"].cpu()),
        "material_correct": bool((predicted_class == batch["physical_class"]).cpu()),
        "E_normalized_absolute_error": float((prediction["E_mu"].float() - batch["E"]).abs().cpu()),
        "nu_normalized_absolute_error": float((prediction["nu_mu"].float() - batch["nu"]).abs().cpu()),
    }


def run_once(model, renderer, batch, device, precision):
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), autocast(device, precision):
        prediction = model.predict(batch, return_unpruned=True)
    synchronize(device)
    network_seconds = time.perf_counter() - started
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        image = renderer(
            prediction["gaussians"], batch["test_c2ws"].float(), batch["test_intr"].float(),
            width=batch["test_images"].shape[-1], height=batch["test_images"].shape[-2],
            background=(1.0, 1.0, 1.0),
        )
    synchronize(device)
    render_seconds = time.perf_counter() - started
    return prediction, image, network_seconds, render_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-list", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pretrain_npu.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gaussian-samples", type=int, default=2048)
    parser.add_argument("--pixel-samples", type=int, default=8192)
    parser.add_argument("--save-representatives", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running"}
    try:
        config = load_config(args.config)
        config.data.validation_path = str(args.data_list.resolve())
        config.data.resize_h = 512
        config.data.num_input_frames = 4
        config.data.num_target_frames = 8
        config.data.target_has_input = True
        torch.manual_seed(config.seed)
        torch.set_num_threads(8)
        device = select_device(args.device)
        report["environment"] = environment_report(device)
        manifest_bytes = args.manifest.read_bytes()
        report.update(
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            checkpoint=str(args.checkpoint.resolve()), precision=args.amp_dtype,
            config=plain_config(config),
        )
        dataset = SupervisedDataset(config, evaluation=True)
        count = min(len(dataset), args.limit) if args.limit else len(dataset)
        if count < 1:
            raise ValueError("Benchmark requires at least one scene")
        model, report["checkpoint_load"] = load_checkpoint(config, args.checkpoint, device)
        model.eval()
        renderer = StaticGaussianRenderer(
            sh_degree=config.model.gaussians.sh_degree,
            near_plane=config.model.gaussians.near_plane,
            far_plane=config.model.gaussians.far_plane,
            eps2d=.3, backend="cann" if device.type == "npu" else "reference",
        )
        # Compile/warm up both M1 and M2 with the exact benchmark shape.
        warm_cpu = default_collate([dataset[0]])
        warm = device_batch(warm_cpu, device)
        warm_prediction, warm_image, _, _ = run_once(model, renderer, warm, device, args.amp_dtype)
        del warm_prediction, warm_image, warm, warm_cpu
        reset_peak_memory(device)
        records, references, representatives = [], [], set()
        manifest = json.loads(manifest_bytes)
        material_by_scene = {item["scene_name"]: item["material"] for item in manifest["scenes"]}
        for index in range(count):
            before = time.perf_counter()
            cpu_batch = default_collate([dataset[index]])
            decode_seconds = time.perf_counter() - before
            before = time.perf_counter()
            batch = device_batch(cpu_batch, device)
            synchronize(device)
            transfer_seconds = time.perf_counter() - before
            prediction, image, network_seconds, render_seconds = run_once(model, renderer, batch, device, args.amp_dtype)
            scene = cpu_batch["scene_name"][0]
            archive_scene = dataset.records[index][0].parent.name
            if any(not torch.isfinite(value).all() for value in prediction["gaussians_unpruned"].values()):
                raise RuntimeError(f"Nonfinite M1 output: {scene}")
            if not torch.isfinite(image.rgb).all() or not torch.isfinite(image.alpha).all():
                raise RuntimeError(f"Nonfinite M2 output: {scene}")
            record = {
                "index": index, "scene_name": scene, "archive_scene": archive_scene,
                "material": material_by_scene[archive_scene],
                "decode_seconds": decode_seconds, "transfer_seconds": transfer_seconds,
                "network_seconds": network_seconds, "render_seconds": render_seconds,
                "gaussians": prediction["gaussians"]["xyz"].shape[1],
                **metrics(prediction, image, batch),
            }
            records.append(record)
            references.append(sampled_reference(scene, archive_scene, prediction, image, args.gaussian_samples, args.pixel_samples))
            if args.save_representatives and record["material"] not in representatives:
                representatives.add(record["material"])
                minimal_input = {key: to_cpu(batch[key]) for key in ("input_images", "input_intr", "input_c2ws", "pos_avg_inv", "scene_scale")}
                minimal_prediction = {key: to_cpu(prediction[key]) for key in ("gaussians", "phys_logits", "E_mu", "E_var", "nu_mu", "nu_var", "pos_avg_inv", "scene_scale")}
                torch.save({"input": minimal_input, "prediction": minimal_prediction}, args.output_dir / f"representative_{record['material'].lower()}.pt")
            if (index + 1) % 25 == 0 or index + 1 == count:
                print(f"scenes={index + 1}/{count} m1={network_seconds:.6f}s m2={render_seconds:.6f}s", flush=True)
            del batch, cpu_batch, prediction, image
        torch.save({
            "format": "physgm-gpu-npu-reference-v1", "manifest_sha256": report["manifest_sha256"],
            "precision": args.amp_dtype, "references": references,
        }, args.output_dir / "reference_samples.pt")
        material_records = defaultdict(list)
        for record in records:
            material_records[record["material"]].append(record)
        summary = {
            "scenes": count,
            "network_seconds": quantiles([x["network_seconds"] for x in records]),
            "render_seconds": quantiles([x["render_seconds"] for x in records]),
            "decode_seconds": quantiles([x["decode_seconds"] for x in records]),
            "transfer_seconds": quantiles([x["transfer_seconds"] for x in records]),
            "rgb_mse": statistics.fmean(x["rgb_mse"] for x in records),
            "psnr": statistics.fmean(x["psnr"] for x in records),
            "alpha_mse": statistics.fmean(x["alpha_mse"] for x in records),
            "material_accuracy": statistics.fmean(x["material_correct"] for x in records),
            "E_normalized_mae": statistics.fmean(x["E_normalized_absolute_error"] for x in records),
            "nu_normalized_mae": statistics.fmean(x["nu_normalized_absolute_error"] for x in records),
            "by_material": {material: {
                "scenes": len(values), "network_seconds": quantiles([x["network_seconds"] for x in values]),
                "render_seconds": quantiles([x["render_seconds"] for x in values]),
                "psnr": statistics.fmean(x["psnr"] for x in values),
                "material_accuracy": statistics.fmean(x["material_correct"] for x in values),
            } for material, values in material_records.items()},
        }
        report.update(status="passed", summary=summary, memory=memory_report(device))
        write_json(args.output_dir / "per_scene.json", records)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["total_seconds_including_io"] = time.perf_counter() - started
        write_json(args.output_dir / "run_report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
