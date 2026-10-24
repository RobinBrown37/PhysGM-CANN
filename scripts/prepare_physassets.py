"""Extract a bounded, reproducible real PhysAssets subset without the 72 GB download."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVISION = "a22b14b6b324f51b0472bc9756c530a8dd17c9ef"
BASE = f"https://huggingface.co/datasets/yaya234/PhysAssets/resolve/{REVISION}"


class RangeReader(io.RawIOBase):
    """Sequential bytes across the upstream's split gzip, bounded before each request."""
    def __init__(self, max_bytes, chunk_bytes=4*1024**2):
        self.limit, self.chunk = max_bytes, chunk_bytes
        self.downloaded = self.part = self.offset = 0
        self.buffer = b""
        self.ranges = []

    def readable(self):
        return True

    def readinto(self, target):
        if not self.buffer:
            if self.downloaded >= self.limit:
                raise RuntimeError(f"Download limit reached ({self.limit} bytes); subset incomplete")
            length = min(self.chunk, self.limit-self.downloaded)
            url = f"{BASE}/dataset.tar.gz.part_{self.part:02d}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={self.offset}-{self.offset+length-1}"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        content_range = response.headers.get("Content-Range", "")
                        if response.status != 206 or not content_range.startswith(f"bytes {self.offset}-"):
                            raise RuntimeError(f"Server did not honor Range: {response.status} {content_range}")
                        size = int(content_range.rsplit("/", 1)[1])
                        payload = response.read(length)
                    if len(payload) != min(length, size-self.offset):
                        raise RuntimeError("Truncated HTTP range")
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
            self.ranges.append({"part": self.part, "offset": self.offset, "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()})
            self.buffer = payload
            self.offset += len(payload)
            self.downloaded += len(payload)
            if self.offset == size:
                self.part += 1
                self.offset = 0
            print(f"Read {self.downloaded / 1024**2:.0f} MiB", flush=True)
        length = min(len(target), len(self.buffer))
        target[:length] = self.buffer[:length]
        self.buffer = self.buffer[length:]
        return length


def prepare(output, count=20, validation_count=4, max_bytes=256*1024**2):
    from training.constants import validate_physical
    from PIL import Image
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != REVISION or len(manifest["scenes"]) != count or len(manifest["validation"]) != validation_count:
            raise ValueError("Existing subset differs; use a different output directory")
        for scene in manifest["scenes"]:
            for name, digest in scene["files"].items():
                if hashlib.sha256((output/scene["directory"]/name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Subset file changed: {scene['directory']}/{name}")
        print(f"Verified existing subset: {manifest_path}")
        return manifest
    if not 0 < validation_count < count:
        raise ValueError("Require 0 < validation-count < scenes")
    output.mkdir(parents=True, exist_ok=True)
    stage = output / ".incomplete"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    selected, rejected, seen = [], [], set()
    reader = RangeReader(max_bytes)
    current = None

    def complete(archive_parent):
        if not (stage/"pose.json").exists():
            return
        try:
            pose = json.loads((stage/"pose.json").read_text())
            physical = json.loads((stage/"physical.json").read_text())
            validate_physical(physical)
            if len(pose["frames"]) < 8:
                raise ValueError("Need at least 8 target views")
            for frame in pose["frames"]:
                relative = PurePosixPath(frame["file_path"])
                if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
                    raise ValueError("Expected a scene-local image filename")
                with Image.open(stage/frame["file_path"]) as image:
                    image.verify()
            name = str(pose["scene_name"])
            if not name or Path(name).name != name or name in seen:
                raise ValueError("Duplicate or invalid scene identity")
            directory = Path("scenes")/name
            (output/directory).parent.mkdir(exist_ok=True)
            shutil.copytree(stage, output/directory, dirs_exist_ok=True)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.iterdir()) if p.is_file()}
            selected.append({"scene_name": name, "directory": str(directory), "archive_parent": archive_parent,
                             "frames": len(pose["frames"]), "physical": physical, "files": hashes})
            seen.add(name)
            print(f"Accepted scene {len(selected)}/{count}: {name} ({physical['material']})", flush=True)
        except (ValueError, KeyError, OSError) as exc:
            rejected.append({"archive_parent": archive_parent, "error": str(exc)})

    try:
        with tarfile.open(fileobj=io.BufferedReader(reader), mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                path = PurePosixPath(member.name)
                parent = str(path.parent)
                if parent != current:
                    if current is not None:
                        complete(current)
                    if len(selected) == count:
                        break
                    shutil.rmtree(stage)
                    stage.mkdir()
                    current = parent
                if path.suffix.lower() not in (".json", ".png", ".jpg", ".jpeg"):
                    continue
                if member.size > 32*1024**2 or path.name in (".", ".."):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                with archive.extractfile(member) as source, (stage/path.name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            else:
                if current is not None:
                    complete(current)
        if len(selected) != count:
            raise RuntimeError(f"Only {len(selected)} complete scenes; requested {count}")
    except Exception:
        (output/"partial_report.json").write_text(json.dumps({"accepted": selected, "rejected": rejected,
                                                           "downloaded_bytes": reader.downloaded}, indent=2))
        raise
    finally:
        shutil.rmtree(stage)
    split = count-validation_count
    train = [s["directory"]+"/pose.json" for s in selected[:split]]
    validation = [s["directory"]+"/pose.json" for s in selected[split:]]
    (output/"train.txt").write_text("\n".join(train)+"\n")
    (output/"validation.txt").write_text("\n".join(validation)+"\n")
    manifest = {"source": "https://huggingface.co/datasets/yaya234/PhysAssets", "revision": REVISION,
                "license": "ODC-By; see upstream dataset card", "archive_checksum_verified": False,
                "downloaded_bytes": reader.downloaded, "ranges": reader.ranges,
                "scenes": selected, "train": train, "validation": validation, "rejected": rejected,
                "materials": dict(Counter(s["physical"]["material"] for s in selected))}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"datasets/physassets_subset")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--validation-scenes", type=int, default=4)
    parser.add_argument("--max-mib", type=int, default=256)
    args = parser.parse_args()
    prepare(args.output_dir.resolve(), args.scenes, args.validation_scenes, args.max_mib*1024**2)
