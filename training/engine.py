"""Device-aware accumulation and successful-update-only scheduling, shared by tests."""
from contextlib import nullcontext
import os
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from runtime import synchronize


class DistributedContext:
    def __init__(self, device):
        self.device = device
        self.rank, self.world_size = 0, 1
        self.launched = "RANK" in os.environ
        if self.launched:
            from datetime import timedelta
            dist.init_process_group("hccl" if device.type == "npu" else "gloo", timeout=timedelta(minutes=10))
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()

    def all_true(self, value):
        value = torch.as_tensor(value, device=self.device, dtype=torch.int32).clone()
        if self.launched:
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def average(self, values):
        keys = sorted(values)
        packed = torch.stack([torch.as_tensor(values[k], device=self.device).detach().float() for k in keys])
        if self.launched:
            dist.all_reduce(packed)
            packed /= self.world_size
        return dict(zip(keys, packed.cpu().tolist()))

    def gather(self, value):
        if not self.launched:
            return [value]
        result = [None]*self.world_size
        dist.all_gather_object(result, value)
        return result

    def barrier(self):
        if self.launched:
            dist.barrier()

    def close(self):
        if self.launched and dist.is_initialized():
            dist.destroy_process_group()


def gradients_finite(parameters):
    flags = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    if not flags:
        raise RuntimeError("No gradients in enabled network")
    return torch.stack(flags).all()


def accumulate_update(model, optimizer, scheduler, scaler, batches, *, device, distributed,
                      accum_steps, clip_norm, before_step=None, after_backward=None):
    """One complete window; no_sync wraps forward AND backward. No hidden NaN repair."""
    import time
    optimizer.zero_grad(set_to_none=True)
    aggregate = {}
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    timings = {"forward_seconds": 0., "backward_seconds": 0., "step_seconds": 0.}
    invalid_window = False
    last = None
    for micro in range(accum_steps):
        context = model.no_sync() if hasattr(model, "no_sync") and micro+1 < accum_steps else nullcontext()
        with context:
            batch = next(batches)
            synchronize(device)
            start = time.perf_counter()
            last = model(batch)
            loss = last["losses"]["total_loss"]
            if not distributed.all_true(torch.isfinite(loss.detach())):
                if scaler is None:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite loss; all ranks abort the complete accumulation window")
                # Finish this window's backward hooks, including the final DDP
                # reduction, then discard all gradients. Skipping backward here
                # would leave DDP's reducer unfinished for the next iteration.
                invalid_window = True
            synchronize(device)
            timings["forward_seconds"] += time.perf_counter()-start
            start = time.perf_counter()
            value = loss/accum_steps
            (scaler.scale(value) if scaler is not None else value).backward()
            synchronize(device)
            timings["backward_seconds"] += time.perf_counter()-start
            for key, tensor in last["losses"].items():
                aggregate[key] = aggregate.get(key, 0)+tensor.detach()/accum_steps
    if scaler is not None:
        scaler.unscale_(optimizer)
    finite_flag = gradients_finite(parameters)
    if scaler is not None:
        # Honor TorchNPU's device-overflow decision even if later math masks
        # it in the final gradient tensors (version pinned in pyproject).
        finite_flag = finite_flag & (not scaler._has_overflow) & (not invalid_window)
    finite = distributed.all_true(finite_flag)
    if not finite:
        if scaler is None:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite gradients; all ranks abort without parameter/scheduler updates")
        # Explicit global decision also handles ranks whose local scaler saw no
        # overflow; update(new_scale) clears the per-optimizer unscale state.
        scaler.update(new_scale=scaler.get_scale()*scaler.get_backoff_factor())
        optimizer.zero_grad(set_to_none=True)
        return {"updated": False, "metrics": distributed.average(aggregate), "timings": timings, "last": last}
    if after_backward is not None:
        after_backward(last)
    norm = clip_grad_norm_(parameters, clip_norm, foreach=False)
    if not distributed.all_true(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradient norm")
    audit = before_step() if before_step is not None else None
    synchronize(device)
    start = time.perf_counter()
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    synchronize(device)
    timings["step_seconds"] = time.perf_counter()-start
    metrics = distributed.average({**aggregate, "grad_norm": norm, "lr": optimizer.param_groups[0]["lr"]})
    return {"updated": True, "metrics": metrics, "timings": timings, "last": last, "audit": audit}
