"""Offline VGG LPIPS v0.1; tensor math matches kiui 0.2.18 (MIT)."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torchvision.models import vgg16


class OfflineLPIPS(nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"LPIPS resource checksum mismatch: {name}")
        self.features = vgg16(weights=None).features
        self.features.load_state_dict(torch.load(directory/"vgg16_features.pt", weights_only=True, map_location="cpu"), strict=True)
        self.lins = nn.ModuleList([nn.Conv2d(c, 1, 1, bias=False) for c in (64, 128, 256, 512, 512)])
        weights = torch.load(directory/"vgg_lpips.pt", weights_only=True, map_location="cpu")
        expected = {f"lin{i}.model.1.weight" for i in range(5)}
        if set(weights) != expected:
            raise ValueError(f"Unexpected LPIPS linear weights: {set(weights)}")
        for i, layer in enumerate(self.lins):
            layer.load_state_dict({"weight": weights[f"lin{i}.model.1.weight"]}, strict=True)
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def _apply(self, function, recurse=True):
        result = super()._apply(function, recurse=recurse)
        if self.shift.device.type == "npu":
            # TorchNPU defaults FP32 convolutions to HF32. VGG max-pool argmax
            # can change, materially altering input gradients. Keep genuine
            # FP32 for LPIPS forward AND backward; BF16 network stays BF16.
            torch.npu.conv.allow_hf32 = False
        return result

    def train(self, mode=True):
        return super().train(False)

    def forward(self, input, target):
        if min(input.shape[-2:]) < 32:
            raise ValueError("VGG LPIPS requires image dimensions >=32")
        a, b = (input-self.shift)/self.scale, (target-self.shift)/self.scale
        values = []
        tap = 0
        for i, layer in enumerate(self.features):
            a, b = layer(a), layer(b)
            if i in (3, 8, 15, 22, 29):
                an = a/(a.square().sum(1, keepdim=True).sqrt()+1e-10)
                bn = b/(b.square().sum(1, keepdim=True).sqrt()+1e-10)
                values.append(self.lins[tap]((an-bn).square()).mean((2, 3), keepdim=True))
                tap += 1
            if i == 29:
                break
        return sum(values)
