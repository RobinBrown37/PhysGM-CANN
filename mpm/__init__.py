"""Forward-only APIC MPM on CPU or Ascend, without Warp/Taichi/CUDA imports."""

from .solver import MPMSimulator

__all__ = ["MPMSimulator"]
