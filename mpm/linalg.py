"""Explicit 3x3 algebra and signed SVD; no generic determinant CPU fallback."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


def mm3(a, b):
    """Tiny matrix products through FP32 vector multiply/reduce, not Cube modes."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def det3(a):
    return (
        a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
        - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
        + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
    )


@lru_cache(None)
def library():
    path = Path(__file__).resolve().parents[1] / "build/mpm/lib/libphysgm_mpm.so"
    if not path.exists():
        raise RuntimeError("Build MPM kernels: .venv/bin/python scripts/build_mpm.py")
    lib = ctypes.CDLL(str(path))
    lib.physgm_svd_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.physgm_svd_launch.restype = None
    return lib


def svd3(a, backend="native"):
    """U,S,V with proper rotations and a signed last singular value, as Warp."""
    if a.device.type == "npu" and backend == "native":
        if a.dtype != torch.float32 or a.ndim != 3 or a.shape[-2:] != (3, 3):
            raise ValueError("Native SVD requires [N,3,3] FP32")
        n = a.shape[0]
        if not n:
            return a.clone(), a.new_empty(0, 3), a.clone()
        stride = (n + 127) // 128 * 128
        packed = torch.nn.functional.pad(
            a.reshape(n, 9).T, (0, stride - n)
        ).contiguous()
        out = a.new_empty(21, stride)
        library().physgm_svd_launch(
            packed.data_ptr(),
            out.data_ptr(),
            stride,
            8,
            torch.npu.current_stream(a.device).npu_stream,
        )
        # The allocator must keep inputs alive until the current stream finishes.
        packed.record_stream(torch.npu.current_stream(a.device))
        return (
            out[:9, :n].T.reshape(n, 3, 3).contiguous(),
            out[9:12, :n].T.contiguous(),
            out[12:, :n].T.reshape(n, 3, 3).contiguous(),
        )
    if backend not in ("native", "torch"):
        raise ValueError("SVD backend must be native or torch")
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    v = vh.transpose(-1, -2)
    du = torch.where(det3(u) < 0, -1.0, 1.0)
    dv = torch.where(det3(v) < 0, -1.0, 1.0)
    u = u.clone()
    v = v.clone()
    s = s.clone()
    u[..., :, 2] *= du[..., None]
    v[..., :, 2] *= dv[..., None]
    s[..., 2] *= du * dv
    return u, s, v
