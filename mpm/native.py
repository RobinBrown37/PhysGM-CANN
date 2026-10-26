"""Forward-only Ascend C APIC transfers using built-in IndexAdd and gather."""

import ctypes
from functools import lru_cache
import torch
from .linalg import library


@lru_cache(None)
def extension():
    lib = library()
    ptr = ctypes.c_void_p
    integer = ctypes.c_int
    real = ctypes.c_float
    lib.physgm_p2g_launch.argtypes = [
        ptr,
        ptr,
        ptr,
        integer,
        integer,
        real,
        real,
        real,
        ptr,
    ]
    lib.physgm_g2p_launch.argtypes = [ptr, ptr, ptr, integer, real, real, ptr]
    lib.physgm_p2g_launch.restype = None
    lib.physgm_g2p_launch.restype = None
    return lib


def pack(fields, n, stride):
    return torch.nn.functional.pad(
        torch.cat([f.reshape(n, -1) for f in fields], -1).T, (0, stride - n)
    ).contiguous()


def p2g(state, grid, n_grid, dx, dt, rpic):
    n = len(state["x"])
    stride = (n + 127) // 128 * 128
    packed = pack(
        [state[k] for k in ("x", "v", "C", "stress", "mass", "volume")]
        + [(state["selection"] == 0).float()],
        n,
        stride,
    )
    out = packed.new_empty(4, 27, stride)
    ids = torch.empty(27, stride, device=packed.device, dtype=torch.int32)
    stream = torch.npu.current_stream(packed.device)
    extension().physgm_p2g_launch(
        packed.data_ptr(),
        out.data_ptr(),
        ids.data_ptr(),
        stride,
        n_grid,
        dx,
        dt,
        rpic,
        stream.npu_stream,
    )
    packed.record_stream(stream)
    indices = ids.reshape(-1).long()
    grid.zero_()
    grid.index_add_(0, indices, out.reshape(4, -1).T.contiguous())
    return indices, stride


def g2p(state, grid_v, indices, stride, dx, dt):
    n = len(state["x"])
    packed = pack([state["x"], state["F"]], n, stride)
    gathered = (
        grid_v.index_select(0, indices)
        .reshape(27, stride, 3)
        .permute(2, 0, 1)
        .contiguous()
    )
    out = packed.new_empty(33, stride)
    stream = torch.npu.current_stream(packed.device)
    extension().physgm_g2p_launch(
        packed.data_ptr(),
        gathered.data_ptr(),
        out.data_ptr(),
        stride,
        dx,
        dt,
        stream.npu_stream,
    )
    packed.record_stream(stream)
    gathered.record_stream(stream)
    active = state["selection"] == 0
    for key, lo, hi, shape in (
        ("x", 0, 3, (n, 3)),
        ("v", 3, 6, (n, 3)),
        ("C", 6, 15, (n, 3, 3)),
        ("F_trial", 15, 24, (n, 3, 3)),
    ):
        value = out[lo:hi, :n].T.reshape(shape).contiguous()
        state[key] = torch.where(
            active.reshape(-1, *([1] * (value.ndim - 1))), value, state[key]
        )
    return out[24:33, :n].T.reshape(n, 3, 3).contiguous()
