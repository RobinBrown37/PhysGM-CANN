"""CPU-serializable complete training state and deterministic per-rank replay."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from export import to_cpu

FORMAT = "physgm-pretrain-v1"


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)


def rng_state(device):
    n = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
              "torch": torch.get_rng_state()}
    if device.type == "npu":
        result["npu"] = torch.npu.get_rng_state(device).cpu()
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if device.type == "npu":
        torch.npu.set_rng_state(state["npu"], device)


def optimizer_groups(network, training):
    groups, names = [[], []], [[], []]
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        group = int(parameter.ndim == 1 or name.endswith(".bias") or getattr(parameter, "_no_weight_decay", False))
        groups[group].append(parameter)
        names[group].append(name)
    optimizer = torch.optim.AdamW([{"params": groups[0], "weight_decay": training.weight_decay},
                                   {"params": groups[1], "weight_decay": 0.}],
        lr=training.lr, betas=(training.beta1, training.beta2), foreach=False, fused=False)
    return optimizer, names


def make_scheduler(optimizer, training):
    warmup, total = training.warmup_steps, training.train_steps
    if not 0 <= warmup <= total or total <= 0:
        raise ValueError("Require 0 <= warmup <= train_steps and train_steps > 0")
    kind = training.get("scheduler_type", "cosine")
    if kind not in ("cosine", "linear", "constant"):
        raise ValueError(f"Unknown scheduler: {kind}")
    def factor(step):
        if step < warmup:
            return step/max(1, warmup)
        progress = min(1., max(0., (step-warmup)/max(1, total-warmup)))
        if kind == "linear":
            return 1-progress
        if kind == "cosine":
            import math
            return .5*(1+math.cos(math.pi*progress))
        return 1.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def training_signature(config, fingerprints, names, world_size, *, resources=None):
    # Logging/worker counts and a stop limit may change without changing math.
    t = config.training
    keys = ("batch_size_per_npu", "grad_accum_steps", "train_steps", "lr", "beta1", "beta2", "weight_decay",
            "warmup_steps", "scheduler_type", "grad_clip_norm", "l2_loss", "opacity_loss", "physical_loss",
            "E_loss", "nu_loss", "lpips_loss", "render_backend", "fixed_batch", "checkpoint_renderer", "checkpoint_transformer")
    value = {"model": config.model, "data": config.data, "seed": config.seed, "use_amp": config.use_amp,
             "amp_dtype": config.amp_dtype, "training": {k: t.get(k) for k in keys},
             "datasets": fingerprints, "param_names": names, "world_size": world_size, "resources": resources}
    if "deterministic" in config:
        value["deterministic"] = config.deterministic
    return {"value": value, "sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    try:
        torch.save(to_cpu(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_resume(checkpoint, signature):
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a migrated complete training checkpoint; use --init-checkpoint for original weights")
    if checkpoint["signature"]["sha256"] != signature["sha256"]:
        old, new = checkpoint["signature"]["value"], signature["value"]
        differences = [k for k in new if old.get(k) != new[k]]
        raise ValueError(f"Resume configuration/data/optimizer mismatch: {differences}")


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for name, value in state.items():
            # AdamW's non-capturable scalar counter deliberately stays on CPU.
            if isinstance(value, torch.Tensor) and name != "step":
                state[name] = value.to(device)
