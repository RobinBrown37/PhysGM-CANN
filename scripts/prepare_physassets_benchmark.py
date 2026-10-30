"""Prepare the fixed PhysAssets GPU/NPU benchmark subset from split tar.gz parts.

The archive is read as one concatenated stream; no combined 72 GB file is made.
The first pass validates/indexes scenes and the second pass extracts only eight
views for the deterministic, material-stratified 512-scene benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
import tempfile


MATERIALS = (
    "Wood", "Metal", "Plastic", "Glass", "Fabric", "Leather", "Ceramic",
    "Stone", "Rubber", "Paper", "Sand", "Snow", "Plasticine", "Foam",
)
QUOTAS = {
    "Wood": 50, "Metal": 58, "Plastic": 72, "Glass": 30, "Fabric": 39,
    "Leather": 27, "Ceramic": 37, "Stone": 35, "Rubber": 30, "Paper": 32,
    "Sand": 19, "Snow": 24, "Plasticine": 40, "Foam": 19,
}
VALIDATION_QUOTAS = {
    "Wood": 6, "Metal": 7, "Plastic": 10, "Glass": 4, "Fabric": 5,
    "Leather": 3, "Ceramic": 5, "Stone": 4, "Rubber": 4, "Paper": 4,
    "Sand": 2, "Snow": 3, "Plasticine": 5, "Foam": 2,
}
VIEW_POSITIONS = (0, 11, 21, 32, 5, 16, 26, 37)
EXPECTED_ARCHIVE_SHA256 = "bce802e3b7cbfdb1d9824b2d164d0ad911b428251ed2694a6bc2ad4971425328"


class SplitReader(io.RawIOBase):
    def __init__(self, parts: list[Path], digest: hashlib._Hash | None = None):
        self.parts = parts
        self.digest = digest
        self.index = -1
        self.current = None
        self.raw_bytes = 0
        self._next()

    def readable(self):
        return True

    def _next(self):
        if self.current is not None:
            self.current.close()
        self.index += 1
        self.current = self.parts[self.index].open("rb") if self.index < len(self.parts) else None

    def readinto(self, buffer):
        view = memoryview(buffer)
        total = 0
        while total < len(view) and self.current is not None:
            count = self.current.readinto(view[total:])
            if count:
                chunk = view[total:total + count]
                if self.digest is not None:
                    self.digest.update(chunk)
                self.raw_bytes += count
                total += count
            else:
                self._next()
        return total

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def archive(parts, digest=None):
    reader = SplitReader(parts, digest)
    compressed = gzip.GzipFile(fileobj=io.BufferedReader(reader, 8 << 20))
    return tarfile.open(fileobj=compressed, mode="r|"), reader


def member_key(member):
    path = PurePosixPath(member.name)
    if len(path.parts) < 2:
        return None, None
    return path.parts[-2], path.name


def read_json(stream, member):
    handle = stream.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Cannot read {member.name}")
    return json.load(handle)


def index_archive(parts):
    digest = hashlib.sha256()
    stream, reader = archive(parts, digest)
    scenes = {}
    canonical_paths = None
    members = files = 0
    try:
        for member in stream:
            members += 1
            if not member.isfile():
                continue
            files += 1
            scene, name = member_key(member)
            if name not in ("pose.json", "physical.json"):
                continue
            record = scenes.setdefault(scene, {"scene_name": scene})
            value = read_json(stream, member)
            if name == "physical.json":
                record["physical"] = value
            else:
                paths = [frame["file_path"] for frame in value.get("frames", [])]
                record["frames"] = len(paths)
                if canonical_paths is None:
                    canonical_paths = paths
                elif paths != canonical_paths:
                    record["noncanonical_paths"] = paths
    finally:
        stream.close()
        reader.close()
    if digest.hexdigest() != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"Archive SHA256 mismatch: {digest.hexdigest()}")
    if canonical_paths is None or len(canonical_paths) != 38:
        raise RuntimeError("Expected a canonical 38-view pose layout")
    valid = {}
    raw_materials = Counter()
    rejected = Counter()
    for scene, record in scenes.items():
        physical = record.get("physical", {})
        material = physical.get("material")
        raw_materials[material] += 1
        reason = None
        try:
            young, poisson = float(physical["E"]), float(physical["nu"])
            if material not in MATERIALS or not math.isfinite(young) or young <= 0 or not math.isfinite(poisson):
                reason = "unsupported_or_invalid_physical"
            elif record.get("frames") != 38 or "noncanonical_paths" in record:
                reason = "noncanonical_views"
        except (KeyError, TypeError, ValueError):
            reason = "invalid_physical"
        if reason:
            rejected[reason] += 1
        else:
            valid[scene] = record
    summary = {
        "archive_sha256": digest.hexdigest(), "compressed_bytes": reader.raw_bytes,
        "members": members, "files": files, "scenes": len(scenes),
        "valid_scenes": len(valid), "raw_material_counts": dict(sorted(raw_materials.items())),
        "rejected": dict(rejected), "canonical_frame_paths": canonical_paths,
    }
    return valid, summary


def select_scenes(valid, seed):
    grouped = {material: [] for material in MATERIALS}
    for scene, record in valid.items():
        grouped[record["physical"]["material"]].append(scene)
    selected = []
    split = {}
    for material in MATERIALS:
        candidates = sorted(grouped[material])
        rng = random.Random(f"{seed}:{material}")
        rng.shuffle(candidates)
        quota = QUOTAS[material]
        if len(candidates) < quota:
            raise RuntimeError(f"Not enough {material}: {len(candidates)} < {quota}")
        chosen = candidates[:quota]
        validation = set(chosen[:VALIDATION_QUOTAS[material]])
        for scene in chosen:
            selected.append(scene)
            split[scene] = "validation" if scene in validation else "train"
    return selected, split


def extract_subset(parts, destination, selected, split, summary, valid, seed):
    chosen = set(selected)
    frame_paths = summary["canonical_frame_paths"]
    selected_paths = [frame_paths[position] for position in VIEW_POSITIONS]
    selected_names = {PurePosixPath(path).name for path in selected_paths}
    wanted = selected_names | {"pose.json", "physical.json", "E_nu.json", "phys.json"}
    manifest_scenes = {}
    stream, reader = archive(parts)
    try:
        for member in stream:
            if not member.isfile():
                continue
            scene, name = member_key(member)
            if scene not in chosen or name not in wanted:
                continue
            handle = stream.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot extract {member.name}")
            payload = handle.read()
            if name == "pose.json":
                pose = json.loads(payload)
                frames = pose["frames"]
                pose["frames"] = [frames[position] for position in VIEW_POSITIONS]
                payload = (json.dumps(pose, indent=2) + "\n").encode()
            scene_dir = destination / "scenes" / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / name).write_bytes(payload)
            record = manifest_scenes.setdefault(scene, {
                "scene_name": scene, "material": valid[scene]["physical"]["material"],
                "physical": valid[scene]["physical"], "split": split[scene], "files": {},
            })
            record["files"][name] = hashlib.sha256(payload).hexdigest()
    finally:
        stream.close()
        reader.close()
    missing = chosen - manifest_scenes.keys()
    if missing:
        raise RuntimeError(f"Selected scenes missing from extraction: {sorted(missing)[:5]}")
    expected = wanted
    for scene, record in manifest_scenes.items():
        absent = expected - record["files"].keys()
        if absent:
            raise RuntimeError(f"{scene} missing files: {sorted(absent)}")
    ordered = sorted(manifest_scenes.values(), key=lambda item: (MATERIALS.index(item["material"]), item["scene_name"]))
    manifest = {
        "format": "physassets-gpu-npu-benchmark-v1", "seed": seed,
        "archive": summary, "view_positions": list(VIEW_POSITIONS),
        "input_view_positions": list(VIEW_POSITIONS[:4]),
        "heldout_view_positions": list(VIEW_POSITIONS[4:]),
        "quotas": QUOTAS, "validation_quotas": VALIDATION_QUOTAS,
        "scenes": ordered,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("all", "train", "validation"):
        rows = [f"scenes/{item['scene_name']}/pose.json" for item in ordered if name == "all" or item["split"] == name]
        (destination / f"{name}.txt").write_text("\n".join(rows) + "\n")
    return manifest


def verify_existing(destination):
    manifest = json.loads((destination / "manifest.json").read_text())
    for record in manifest["scenes"]:
        for name, expected in record["files"].items():
            path = destination / "scenes" / record["scene_name"] / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, help="Required when creating a subset; unused for --verify-existing")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        manifest = verify_existing(args.output_dir)
        print(json.dumps({"status": "passed", "scenes": len(manifest["scenes"])}, indent=2))
        return
    if args.archive_dir is None:
        parser.error("--archive-dir is required unless --verify-existing is used")
    parts = sorted(args.archive_dir.glob("dataset.tar.gz.part_*"))
    if len(parts) != 7:
        raise RuntimeError(f"Expected 7 archive parts, found {len(parts)}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output_dir}")
    valid, summary = index_archive(parts)
    selected, split = select_scenes(valid, args.seed)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        manifest = extract_subset(parts, temporary, selected, split, summary, valid, args.seed)
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "passed", "output": str(args.output_dir), "scenes": len(manifest["scenes"]),
        "train": sum(x["split"] == "train" for x in manifest["scenes"]),
        "validation": sum(x["split"] == "validation" for x in manifest["scenes"]),
    }, indent=2))


if __name__ == "__main__":
    main()
