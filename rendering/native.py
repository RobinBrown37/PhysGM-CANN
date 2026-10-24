"""Project-local CANN compositor and tile scheduler; first-order autograd."""
import importlib.util
import os
from pathlib import Path
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def extension():
    root = Path(__file__).resolve().parents[1] / "build/gauss_splat"
    paths = list(root.glob("gsplat_c*.so"))
    if len(paths) != 1:
        raise RuntimeError("CANN renderer not built; source CANN and run .venv/bin/python scripts/build_renderer.py")
    vendor = root / "packages/vendors/customize"
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if str(vendor) not in existing.split(":"):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + (":" + existing if existing else "")
    spec = importlib.util.spec_from_file_location("gsplat_c", paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._init_op_api_so_path(str(vendor / "op_api/lib/libcust_opapi.so"))
    return module


class Composite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, conics, opacity, colors, coords, schedule, ids):
        module = extension()
        # The native auxiliary attribute integrates constant 1, hence alpha.
        packed = torch.cat((means, conics, opacity[:, None], colors, torch.ones_like(opacity[:, None])), -1).contiguous()
        outputs = module.calc_render_fwd_double_clip_gsids(packed, coords, schedule, ids.to(torch.int64))
        rgb, alpha, log_transmittance, error, stops, row_stops = outputs
        ctx.save_for_backward(packed, coords, schedule, ids.to(torch.int64), log_transmittance, error, stops, row_stops)
        return rgb, alpha

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_rgb, grad_alpha):
        packed, coords, schedule, ids, cumulative, error, stops, row_stops = ctx.saved_tensors
        grad = extension().calc_render_bwd_var_clip_gsids(
            grad_rgb.contiguous(), grad_alpha.contiguous(), cumulative, error,
            packed, coords, schedule, ids, stops, row_stops,
        )
        return grad[:, :2], grad[:, 2:5], grad[:, 5], grad[:, 6:9], None, None, None


def tile_geometry(width, height, device):
    padded_w, padded_h = (width+31)//32*32, (height+31)//32*32
    y, x = torch.meshgrid(torch.arange(padded_h, device=device), torch.arange(padded_w, device=device), indexing="ij")
    pixel = torch.stack((x, y), -1).float() + 0.5
    coords = pixel.reshape(padded_h//32, 32, padded_w//32, 32, 2).permute(0, 2, 1, 3, 4).reshape(-1, 1024, 2).transpose(1, 2).contiguous()
    grid = torch.stack(torch.meshgrid(torch.arange(0, padded_h, 32, device=device), torch.arange(0, padded_w, 32, device=device), indexing="ij"), -1).reshape(-1, 2).float()
    return coords, grid, padded_w, padded_h


def tile_to_image(tensor, padded_w, padded_h, width, height):
    C = tensor.shape[0]
    return tensor.permute(1, 2, 0).reshape(padded_h//32, padded_w//32, 32, 32, C).transpose(1, 2).reshape(padded_h, padded_w, C)[:height, :width].permute(2, 0, 1)


def render(projected, geometry, width, height):
    import acl
    module = extension()
    coords, grid, padded_w, padded_h = geometry
    p = projected
    N = p["means2d"].shape[0]
    with torch.no_grad():
        # FlashGS copies/gathers in aligned vector blocks. Its compact count is
        # independent of the storage stride, which must be padded to 64 floats.
        storage_count = (N+63)//64*64
        def storage(tensor):
            return torch.nn.functional.pad(tensor, (0, storage_count-N)).contiguous()
        sums, offsets, depths, gs_ids = module.flash_gaussian_build_mask(
            storage(p["means2d"].T[None, None]),
            storage(p["opacity"][None, None, None]),
            storage(p["conics"].T[None, None]),
            storage(p["covars2d"].T[None, None]),
            storage(p["depths"][None, None, None]),
            torch.tensor([[[N]]], dtype=torch.int32, device=coords.device),
            grid, float(width), float(height), 32,
        )
        counts = sums.squeeze(-1)
        sorted_offsets = offsets.squeeze(-1)[:, :, -1].flatten().cumsum(0)
        # Only tile counts/scheduling metadata visit the CPU, not Gaussian data.
        vectors, status = acl.get_device_capability(coords.device.index, 1)
        if status:
            raise RuntimeError(f"ACL device capability failed: {status}")
        counts_cpu = counts.cpu().to(torch.int64)
        schedule = module.get_render_schedule(counts_cpu, vectors).to(coords.device)
        # GaussianSort launches min(tiles,AIV) cores, whereas CalcRender launches
        # all AIV cores. Their schedule header lengths must match independently.
        sort_vectors = min(vectors, grid.shape[0])
        sort_schedule = schedule if sort_vectors == vectors else module.get_render_schedule(counts_cpu, sort_vectors).to(coords.device)
        total = int(sorted_offsets[-1])
        if total == 0:
            zero = sum(p[k].sum() * 0 for k in ("means2d", "conics", "opacity", "colors"))
            return zero.expand(3, height, width), zero.expand(1, height, width), 0
        sorted_ids = module.gaussian_sort(sort_schedule, sums, depths, gs_ids, sorted_offsets, int(sums.max()))
    rgb, alpha = Composite.apply(p["means2d"], p["conics"], p["opacity"], p["colors"], coords, schedule[0, 0].contiguous(), sorted_ids)
    return tile_to_image(rgb, padded_w, padded_h, width, height), tile_to_image(alpha, padded_w, padded_h, width, height), total
