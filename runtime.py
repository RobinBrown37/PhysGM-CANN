"""Explicit CPU/NPU selection and synchronization; never falls back to CPU."""

import importlib.metadata
import platform
import resource
from contextlib import nullcontext

import torch


def select_device(spec):
    if spec == "cpu":
        return torch.device("cpu")
    if spec != "npu" and not spec.startswith("npu:"):
        raise ValueError("Supported devices: cpu, npu, npu:<logical index>")
    try:
        import torch_npu  # noqa: F401; registers NPU operators and AMP
    except ImportError as exc:
        raise RuntimeError("Cannot import torch_npu; run uv sync and source CANN set_env.sh") from exc
    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable: check device exposure, driver and CANN; CPU debug requires --device cpu")
    device = torch.device(spec if ":" in spec else "npu:0")
    torch.npu.set_device(device)
    return device


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU debug supports fp32 or bf16; use --amp-dtype fp32")
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)


def reset_peak_memory(device):
    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)


def memory_report(device):
    report = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if device.type == "npu":
        report.update(
            npu_peak_allocated_bytes=torch.npu.max_memory_allocated(device),
            npu_peak_reserved_bytes=torch.npu.max_memory_reserved(device),
        )
    return report


def environment_report(device):
    versions = {}
    for name in ("torch", "torch-npu", "transformers", "torchvision", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"python": platform.python_version(), "architecture": platform.machine(), "packages": versions, "device": str(device)}
    if device.type == "npu":
        from torch_npu.npu.utils import get_cann_version
        report["device_name"] = torch.npu.get_device_name(device)
        report["cann_version"] = get_cann_version()
    return report
