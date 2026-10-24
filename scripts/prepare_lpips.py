"""Fetch official frozen LPIPS resources once; training itself remains offline."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

SOURCES = {
    "vgg16_full.pth": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg_lpips.pt": "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth",
}
SOURCE_SHA256 = {
    "vgg16_full.pth": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
    "vgg_lpips.pt": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("weights/lpips"))
    args = parser.parse_args()
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        manifest = json.loads((directory/"manifest.json").read_text())
        if all(hashlib.sha256((directory/k).read_bytes()).hexdigest() == v for k, v in manifest["files"].items()):
            print("Verified existing LPIPS resources")
            return
        raise ValueError("Existing LPIPS resource checksum mismatch")
    source_hashes = {}
    for name, url in SOURCES.items():
        target = directory/name
        if not target.exists():
            temporary = target.with_suffix(".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                digest = hashlib.sha256()
                size = 0
                while data := response.read(1024**2):
                    output.write(data)
                    digest.update(data)
                    size += len(data)
                    if size % (64*1024**2) == 0:
                        print(f"{name}: {size//1024**2} MiB", flush=True)
            temporary.replace(target)
        source_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    if source_hashes != SOURCE_SHA256:
        raise ValueError("Official VGG16/LPIPS resource checksum mismatch")
    # Official torchvision VGG16 uses legacy torch serialization, without mmap.
    state = torch.load(directory/"vgg16_full.pth", weights_only=True, map_location="cpu")
    features = {k[len("features."):]: v for k, v in state.items() if k.startswith("features.")}
    torch.save(features, directory/"vgg16_features.pt")
    files = {k: hashlib.sha256((directory/k).read_bytes()).hexdigest() for k in ("vgg16_features.pt", "vgg_lpips.pt")}
    manifest = {"sources": SOURCES, "source_sha256": source_hashes, "files": files,
                "formula": "kiui==0.2.18 VGG LPIPS, v0.1 linear calibration"}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
