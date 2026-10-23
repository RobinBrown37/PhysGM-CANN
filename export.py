"""CPU-only artifact writing with upstream Gaussian and physical conventions."""

import json

import numpy as np
import torch
from plyfile import PlyData, PlyElement


MATERIALS = ("Wood", "Metal", "Plastic", "Glass", "Fabric", "Leather", "Ceramic", "Stone", "Rubber", "Paper", "Sand", "Snow", "Plasticine", "Foam")


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def physical_values(prediction, index=0):
    """Match pipeline.py's normalization, units and rounding exactly."""
    E_mu = float(prediction["E_mu"][index].float())
    nu_mu = float(prediction["nu_mu"][index].float())
    label = int(prediction["phys_logits"][index].float().argmax())
    return {
        "E": round(10 ** (E_mu * 2.456477 + 7.387210) * 0.1, 2),
        "nu": round(nu_mu * 0.111 + 0.398, 4),
        "material": MATERIALS[label],
    }


def save_gaussian_ply(gaussians, path, opacity_threshold=0.001):
    """Write one scene; keep log-scale/logit and SH values unactivated.

    The 45 zero-padded higher-order SH fields and uint8 normals match upstream.
    Thresholding is an export-only selection, separate from model pruning.
    """
    g = {k: v.detach().cpu().float() for k, v in gaussians.items()}
    mask = torch.ones(g["xyz"].shape[0], dtype=torch.bool)
    if opacity_threshold is not None:
        mask = g["opacity"].squeeze(-1).sigmoid() > opacity_threshold
    g = {k: v[mask].numpy() for k, v in g.items()}
    N = len(g["xyz"])
    columns = {
        **{name: g["xyz"][:, i] for i, name in enumerate(("x", "y", "z"))},
        **{name: np.zeros(N, dtype=np.uint8) for name in ("nx", "ny", "nz")},
        **{f"f_dc_{i}": g["feature"][:, 0, i] for i in range(3)},
    }
    rest = g["feature"][:, 1:].transpose(0, 2, 1).reshape(N, (g["feature"].shape[1] - 1) * 3)
    for i in range(45):
        columns[f"f_rest_{i}"] = rest[:, i] if i < rest.shape[1] else np.zeros(N, dtype=np.float32)
    columns["opacity"] = g["opacity"][:, 0]
    columns.update({f"scale_{i}": g["scale"][:, i] for i in range(3)})
    columns.update({f"rot_{i}": g["rotation"][:, i] for i in range(4)})
    vertices = np.empty(N, dtype=[(k, "u1" if k in ("nx", "ny", "nz") else "f4") for k in columns])
    for key, column in columns.items():
        vertices[key] = column
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
    return N


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
