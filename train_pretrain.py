"""M3 real RGB/alpha/material/E/nu supervised pretraining on Ascend TorchNPU."""
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, default_collate
import yaml

from export import write_json
from model.physgm import initialize_from_dino, load_checkpoint
from runtime import environment_report, memory_report, reset_peak_memory, select_device, synchronize
from training.config import load_config, plain_config
from training.data import CursorBatchSampler, SupervisedDataset
from training.engine import DistributedContext, accumulate_update
from training.state import (FORMAT, atomic_save, make_scheduler, move_optimizer_state, optimizer_groups,
                            restore_rng, rng_state, seed_all, training_signature, validate_resume)
from training.task import PretrainTask

ROOT = Path(__file__).resolve().parent


def device_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class DataStream:
    def __init__(self, dataset, training, distributed, cursor=None):
        cursor = cursor or {"epoch": 0, "cursor": 0}
        self.sampler = CursorBatchSampler(len(dataset), training.batch_size_per_npu, seed=dataset.seed,
            rank=distributed.rank, world_size=distributed.world_size, epoch=cursor["epoch"],
            cursor=cursor["cursor"], fixed=training.get("fixed_batch", False))
        workers = training.num_workers
        requested_workers = workers
        prefetch = training.get("prefetch_factor", 2)
        # Container /dev/shm is often 64 MiB. One RGBA 512 4/8-view
        # float batch is already ~48 MiB; preflight the real transport size.
        sample = dataset[(self.sampler.epoch, 0, 0)]
        bytes_per_batch = sum(v.numel()*v.element_size() for v in sample.values() if isinstance(v, torch.Tensor))*training.batch_size_per_npu
        shm_available = None
        if workers and Path("/dev/shm").exists():
            stat = os.statvfs("/dev/shm")
            shm_available = stat.f_bavail*stat.f_frsize
            slots = max(0, int(.75*shm_available)//max(1, bytes_per_batch)-2)
            workers = min(workers, slots)
            if workers:
                prefetch = min(prefetch, max(1, slots//workers))
        self.info = {"requested_workers": requested_workers, "actual_workers": workers,
                     "prefetch_factor": prefetch if workers else None, "batch_transport_bytes": bytes_per_batch,
                     "shared_memory_available_bytes": shm_available}
        options = {"num_workers": workers, "pin_memory": training.get("pin_memory", False),
                   "generator": torch.Generator().manual_seed(dataset.seed+distributed.rank)}
        if workers:
            options.update(multiprocessing_context="spawn", persistent_workers=True,
                           prefetch_factor=prefetch)
        self.loader = DataLoader(dataset, batch_sampler=self.sampler, **options)
        self.iterator = None
        self.device, self.last = distributed.device, None

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.sampler.cursor = 0
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        self.sampler.cursor += 1
        self.last = batch
        return device_batch(batch, self.device)

    def state_dict(self):
        return {"epoch": self.sampler.epoch, "cursor": self.sampler.cursor}


def dino_digest(network):
    digest = hashlib.sha256()
    for name, tensor in network.dinov3_extractor.model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_audit(network):
    groups = {}
    samples = {}
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"Frozen parameter received a gradient: {name}")
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Enabled parameter has no gradient: {name}")
        group = name.split(".")[0]
        stat = groups.setdefault(group, {"parameters_with_grad": 0, "absolute_sum": 0.})
        stat["parameters_with_grad"] += 1
        magnitude = float(parameter.grad.detach().float().abs().sum().cpu())
        stat["absolute_sum"] += magnitude
        if group not in samples and magnitude > 0:
            index = int(parameter.grad.detach().abs().reshape(-1).argmax())
            samples[group] = (name, parameter, index, float(parameter.detach().reshape(-1)[index]))
    for name, stat in groups.items():
        if stat["absolute_sum"] == 0:
            raise RuntimeError(f"No nonzero gradient in enabled group: {name}")
    return groups, samples


def export_visuals(task, batch, directory):
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        output = task(batch, visuals=True)
    if not output["visuals"]:
        return
    rgba = batch["test_images"][0].float()
    target = rgba[:, :3]*rgba[:, 3:]+1-rgba[:, 3:]
    rgb = output["visuals"]["rgb"][0]
    alpha = output["visuals"]["alpha"][0]
    for name, tensor in (("target_rgb", target), ("prediction_rgb", rgb), ("target_alpha", rgba[:, 3:]), ("prediction_alpha", alpha)):
        image = tensor.permute(1, 2, 0, 3).flatten(2, 3).clamp(0, 1).mul(255).round().byte().cpu()
        array = image.squeeze(0).numpy() if image.shape[0] == 1 else image.permute(1, 2, 0).numpy()
        Image.fromarray(array).save(directory/f"{name}.png")
    combined = torch.cat((target.permute(1, 2, 0, 3).flatten(2, 3), rgb.permute(1, 2, 0, 3).flatten(2, 3)), 1)
    Image.fromarray(combined.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(directory/"comparison.png")


@torch.no_grad()
def evaluate(task, dataset, device):
    task.eval()
    aggregate = {}
    for i in range(len(dataset)):
        batch = device_batch(default_collate([dataset[i]]), device)
        losses = task(batch)["losses"]
        for key, value in losses.items():
            aggregate[key] = aggregate.get(key, 0.)+float(value.cpu())/len(dataset)
    task.train()
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/pretrain_npu.yaml")
    starts = parser.add_mutually_exclusive_group()
    starts.add_argument("--init-checkpoint", type=Path)
    starts.add_argument("--dinov3-checkpoint", type=Path)
    starts.add_argument("--resume", help="Complete migrated checkpoint, or auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/pretrain")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--max-steps", type=int, help="Stop after this many total updates without changing the schedule")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--audit", action="store_true", help="Audit real Gaussian gradients and group parameter deltas")
    parser.add_argument("--capture-update", action="store_true", help="Save final real clipped gradients and pre-step weights for resume equivalence")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    if args.amp_dtype:
        config.amp_dtype, config.use_amp = args.amp_dtype, args.amp_dtype != "fp32"
    if "LOCAL_RANK" in os.environ:
        args.device = f"npu:{os.environ['LOCAL_RANK']}" if args.device.startswith("npu") else "cpu"
    device = select_device(args.device)
    if device.type == "npu":
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(config.get("deterministic", False))
    distributed = DistributedContext(device)
    directory = args.output_dir.resolve()
    writer = None
    report = {"status": "running", "environment": environment_report(device), "world_size": distributed.world_size}
    report["deterministic"] = torch.are_deterministic_algorithms_enabled()
    started = time.perf_counter()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if args.resume is None and (directory/"last.pt").exists() and not args.eval_only:
            raise ValueError("Output already contains training state; use --resume auto or a new output directory")
        seed_all(config.seed+distributed.rank, device)
        train_data = SupervisedDataset(config)
        validation = SupervisedDataset(config, evaluation=True)
        resume_path = directory/"last.pt" if args.resume == "auto" else Path(args.resume) if args.resume else None
        checkpoint = None
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True, mmap=True)
            if checkpoint.get("format") != FORMAT:
                raise ValueError("Original optimizer is incompatible with full pretraining; use --init-checkpoint")
            network, initialization = load_checkpoint(config, resume_path, device)
            initialization["mode"] = "complete_resume"
        elif args.dinov3_checkpoint:
            network, initialization = initialize_from_dino(config, args.dinov3_checkpoint, device)
        else:
            network, initialization = load_checkpoint(config, args.init_checkpoint or ROOT/"checkpoint.pt", device)
            initialization["mode"] = "warmstart_weights_only"
        task = PretrainTask(network, config, device)
        task.train()
        optimizer, names = optimizer_groups(network, config.training)
        scheduler = make_scheduler(optimizer, config.training)
        scaler = None
        if config.use_amp and config.amp_dtype == "fp16":
            if device.type != "npu":
                raise ValueError("FP16 training requires NPU")
            scaler = torch.npu.amp.GradScaler()
        resources = None
        if config.training.lpips_loss > 0:
            resources = json.loads((Path(config.training.lpips_weights)/"manifest.json").read_text())["files"]
        signature = training_signature(config, [train_data.fingerprint, validation.fingerprint], names, distributed.world_size, resources=resources)
        step = micro_step = skipped = 0
        best = float("inf")
        cursor = None
        if checkpoint is not None:
            validate_resume(checkpoint, signature)
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state(optimizer, device)
            scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            step, micro_step = checkpoint["global_step"], checkpoint["micro_step"]
            skipped, best = checkpoint["skipped_windows"], checkpoint["best_validation"]
            cursor = checkpoint["ranks"][distributed.rank]["data"]
        stream = DataStream(train_data, config.training, distributed, cursor)
        model = task
        if distributed.launched:
            model = DistributedDataParallel(task, device_ids=[device.index] if device.type == "npu" else None,
                                           broadcast_buffers=False, find_unused_parameters=False)
        if checkpoint is not None:
            restore_rng(checkpoint["ranks"][distributed.rank]["rng"], device)
        checkpoint = None
        initial_dino = dino_digest(network)
        if distributed.rank == 0:
            (directory/"config.yaml").write_text(yaml.safe_dump(plain_config(config), sort_keys=False))
            write_json(directory/"data_manifest.json", {"train": [str(p) for p in train_data.paths],
                "validation": [str(p) for p in validation.paths], "fingerprints": signature["value"]["datasets"],
                "source_manifest": json.loads((ROOT/"datasets/physassets_subset/manifest.json").read_text())
                if (ROOT/"datasets/physassets_subset/manifest.json").exists() else None})
            if config.training.get("tensorboard", True):
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(directory/"tensorboard")
        report.update(initialization=initialization, config=plain_config(config), signature=signature["sha256"],
                      trainable_parameters=sum(p.numel() for p in network.parameters() if p.requires_grad),
                      start_global_step=step, dino_initial_sha256=initial_dino, dataloader=stream.info)
        reset_peak_memory(device)
        records, validations, audits = [], [], []

        def log(record):
            if distributed.rank == 0:
                with (directory/"metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False)+"\n")
                if writer is not None:
                    for key, value in record.items():
                        if isinstance(value, (float, int)):
                            writer.add_scalar(key, value, record["global_step"])

        def save(kind):
            ranks = distributed.gather({"rng": rng_state(device), "data": stream.state_dict()})
            if distributed.rank == 0:
                value = {"format": FORMAT, "model": network.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                    "global_step": step, "train_steps_done": micro_step, "micro_step": micro_step,
                    "skipped_windows": skipped, "best_validation": best, "ranks": ranks,
                    "signature": signature, "config": plain_config(config), "param_names": names}
                path = directory/("best.pt" if kind == "best" else f"checkpoint_{step:06d}.pt")
                atomic_save(value, path)
                if kind != "best":
                    link = directory/"last.pt"
                    temporary = directory/".last.tmp"
                    temporary.unlink(missing_ok=True)
                    temporary.symlink_to(path.name)
                    temporary.replace(link)
                    for old in sorted(directory.glob("checkpoint_*.pt"))[:-2]:
                        old.unlink()
            distributed.barrier()

        if args.eval_only:
            if distributed.rank == 0:
                report["validation"] = evaluate(task, validation, device)
            distributed.barrier()
        else:
            limit = args.max_steps if args.max_steps is not None else config.training.train_steps
            if not step <= limit <= config.training.train_steps:
                raise ValueError("Stop limit must be between resumed global_step and configured train_steps")
            audit_pending = args.audit
            while step < limit:
                samples = {}
                gaussian_stats = {}
                def inspect_backward(last):
                    if audit_pending and task.loss.image_enabled:
                        for key, tensor in last["prediction"]["gaussians"].items():
                            grad = tensor.grad
                            if grad is None or not torch.isfinite(grad).all():
                                raise RuntimeError(f"Invalid real Gaussian gradient: {key}")
                            magnitude = float(grad.abs().sum().cpu())
                            if magnitude == 0:
                                raise RuntimeError(f"Zero Gaussian gradient group: {key}")
                            gaussian_stats[key] = {"finite": True, "absolute_sum": magnitude}

                def inspect_step():
                    nonlocal samples
                    if args.capture_update and step+1 == limit and distributed.rank == 0:
                        atomic_save({"global_step": step+1, "sample_key": stream.last["sample_key"],
                            "gradients": {n:p.grad for n,p in network.named_parameters() if p.requires_grad},
                            "parameters_before": {n:p.detach() for n,p in network.named_parameters() if p.requires_grad}},
                            directory/"update_probe.pt")
                    if audit_pending:
                        groups, samples = gradient_audit(network)
                        return groups

                task.inspect_gradients = audit_pending
                try:
                    result = accumulate_update(model, optimizer, scheduler, scaler, stream, device=device,
                        distributed=distributed, accum_steps=config.training.grad_accum_steps,
                        clip_norm=config.training.grad_clip_norm, before_step=inspect_step,
                        after_backward=inspect_backward)
                finally:
                    task.inspect_gradients = False
                micro_step += config.training.grad_accum_steps
                if not result["updated"]:
                    skipped += 1
                    if skipped > config.training.max_overflow_windows:
                        raise RuntimeError("Too many FP16 overflow windows")
                    log({"global_step": step, "micro_step": micro_step, "skipped_windows": skipped})
                    continue
                step += 1
                record = {"global_step": step, "micro_step": micro_step, **result["metrics"], **result["timings"],
                          "epoch": stream.sampler.epoch, "cursor": stream.sampler.cursor,
                          "samples": stream.last["sample_key"].tolist()}
                log(record)
                records.append(record)
                if audit_pending:
                    deltas = {group: {"parameter": name, "sample_index": index,
                        "delta": float(parameter.detach().reshape(-1)[index])-before}
                        for group, (name, parameter, index, before) in samples.items()}
                    # The initial warmup update may have lr=0. Only consume audit
                    # once a real update changed every sampled enabled group.
                    if all(v["delta"] != 0 for v in deltas.values()):
                        audits.append({"global_step": step, "groups": result["audit"], "gaussians": gaussian_stats, "parameter_deltas": deltas})
                        audit_pending = False
                if distributed.rank == 0 and (step % config.training.print_every == 0 or step == limit):
                    print(f"step={step}/{limit} loss={record['total_loss']:.6f} lr={record['lr']:.3g} grad={record['grad_norm']:.4g}", flush=True)
                # Release autograd graphs and renderer caches before eval/save.
                del result
                if step % config.training.evaluation_every == 0 or step == limit:
                    distributed.barrier()
                    val = evaluate(task, validation, device) if distributed.rank == 0 else None
                    vals = distributed.gather(val)
                    val = vals[0]
                    validations.append({"global_step": step, **val})
                    log({"global_step": step, **{"validation/"+k:v for k,v in val.items()}})
                    if val["total_loss"] < best:
                        best = val["total_loss"]
                        save("best")
                if distributed.rank == 0 and (step % config.training.visualization_every == 0 or step == limit):
                    task.eval()
                    export_visuals(task, device_batch(default_collate([validation[0]]), device), directory/f"vis_{step:06d}")
                    task.train()
                if step % config.training.checkpoint_every == 0 or step == limit:
                    save("periodic")
            if audit_pending:
                raise RuntimeError("Audit did not observe nonzero updates in all enabled network groups")
            report.update(global_step=step, micro_step=micro_step, skipped_windows=skipped,
                          updates=records, validation=validations, audits=audits, best_validation=best)
        final_dino = dino_digest(network)
        if final_dino != initial_dino or network.dinov3_extractor.model.training:
            raise RuntimeError("Frozen DINO weights or eval invariant changed")
        report.update(status="passed", dino_final_sha256=final_dino, dino_unchanged=True,
                      memory=memory_report(device), elapsed_seconds=time.perf_counter()-started)
        if distributed.rank == 0:
            write_json(directory/"run_report.json", report)
            export_curves(directory)
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        if distributed.rank == 0:
            write_json(directory/"run_report.json", report)
        raise
    finally:
        if writer is not None:
            writer.close()
        distributed.close()
    return report


def export_curves(directory):
    """Standalone SVG chart without an extra plotting/runtime dependency."""
    from html import escape
    lines = [json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()] if (directory/"metrics.jsonl").exists() else []
    values = [(r["global_step"], r["total_loss"]) for r in lines if "total_loss" in r]
    if not values:
        return
    low, high = min(v for _, v in values), max(v for _, v in values)
    points = " ".join(f"{50+700*s/max(1,values[-1][0]):.2f},{270-220*(v-low)/max(1e-12,high-low):.2f}" for s,v in values)
    (directory/"loss_curve.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="320"><rect width="800" height="320" fill="white"/><text x="50" y="25">Real supervised total loss</text><path d="M50 45V270H760" stroke="black" fill="none"/><polyline points="{escape(points)}" stroke="#2563eb" fill="none"/><text x="50" y="305">optimizer steps 0–{values[-1][0]}; loss {low:.5f}–{high:.5f}</text></svg>\n')


if __name__ == "__main__":
    main()
