"""CPU-only frame archives and PNG/video export, independent of MPM kernels."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def cpu_pack(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: cpu_pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_pack(v) for v in value)
    return value


def atomic_torch(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_pack(value), tmp)
    tmp.replace(path)


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    digest = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().contiguous()
            digest.update(str((str(obj.dtype), tuple(obj.shape))).encode())
            digest.update(obj.numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj):
                digest.update(key.encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        else:
            digest.update(json.dumps(obj, sort_keys=True, allow_nan=False).encode())

    visit(value)
    return digest.hexdigest()


def move_pack(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_pack(v, device) for k, v in value.items()}
    return value


def compose_frame(shared, frame):
    """Join shared static/appearance data with one unbatched world-space frame."""
    n = int(shared["dynamic_count"])
    if frame["xyz"].shape != (n, 3) or frame["covariance"].shape != (n, 3, 3):
        raise ValueError(
            "Render frame count/shape mismatch; filling particles must be excluded"
        )
    result = {k: frame[k] for k in ("xyz", "covariance")}
    result.update(feature=shared["feature"], opacity=shared["opacity"])
    if "rotation" in frame:
        result["rotation"] = frame["rotation"]
    static = shared.get("static")
    if static is not None:
        for key in ("xyz", "covariance", "feature", "opacity"):
            result[key] = torch.cat((result[key], static[key]))
        if "rotation" in result:
            eye = torch.eye(
                3, device=result["rotation"].device, dtype=result["rotation"].dtype
            )
            result["rotation"] = torch.cat(
                (result["rotation"], eye.expand(len(static["xyz"]), 3, 3))
            )
    return {k: v.unsqueeze(0) for k, v in result.items()}


class SequenceWriter:
    def __init__(self, directory, shared, config, physics_mode, *, append=False):
        self.root = Path(directory)
        self.shared = cpu_pack(shared)
        self.shared.update(config=config, physics_mode=physics_mode)
        self.fingerprint = signature(self.shared)
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists():
            if not append:
                raise FileExistsError(
                    f"Sequence exists: {self.root}; choose a new directory or --append-sequence"
                )
            self.manifest = json.loads(self.manifest_path.read_text())
            if (
                self.manifest.get("format") != "physgm_render_sequence_v1"
                or self.manifest.get("scene_sha256") != self.fingerprint
            ):
                raise ValueError(
                    "Cannot append: sequence scene/config fingerprint mismatch"
                )
            if (
                signature(
                    torch.load(
                        self.root / "scene.pt", map_location="cpu", weights_only=False
                    )
                )
                != self.fingerprint
            ):
                raise ValueError("Cannot append: shared scene checksum mismatch")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "frames").mkdir(exist_ok=True)
            atomic_torch(self.shared, self.root / "scene.pt")
            self.manifest = dict(
                format="physgm_render_sequence_v1",
                scene="scene.pt",
                scene_sha256=self.fingerprint,
                physics_mode=physics_mode,
                frame_dt=float(config["frame_dt"]),
                frames=[],
            )
            atomic_json(self.manifest, self.manifest_path)

    def append(self, frame, camera):
        frame = cpu_pack(frame)
        camera = cpu_pack(camera)
        compose_frame(self.shared, frame)
        if self.shared["feature"].shape[1] > 1 and "rotation" not in frame:
            raise ValueError(
                "Higher-order SH sequence requires cached shading rotation"
            )
        index = int(frame["frame_index"])
        stamp = float(frame["time"])
        step = int(frame["step"])
        if index < 0 or step < 0 or not math.isfinite(stamp):
            raise ValueError("Invalid sequence frame index/time/step")
        previous = self.manifest["frames"][-1] if self.manifest["frames"] else None
        if previous and (
            index <= previous["frame_index"]
            or stamp <= previous["time"]
            or step <= previous["step"]
        ):
            raise ValueError(
                "Sequence append must advance absolute frame, time and step"
            )
        filename = f"frames/{index:06d}.pt"
        atomic_torch(dict(frame=frame, camera=camera), self.root / filename)
        checksum = file_sha256(self.root / filename)
        self.manifest["frames"].append(
            dict(
                file=filename,
                file_sha256=checksum,
                frame_index=index,
                time=stamp,
                step=step,
            )
        )
        atomic_json(self.manifest, self.manifest_path)


class SequenceReader:
    def __init__(self, directory, device="cpu"):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "physgm_render_sequence_v1":
            raise ValueError("Unsupported render sequence format")
        cpu_shared = torch.load(
            self.root / "scene.pt", map_location="cpu", weights_only=False
        )
        if signature(cpu_shared) != self.manifest["scene_sha256"]:
            raise ValueError("Render sequence scene checksum mismatch")
        self.shared = move_pack(cpu_shared, device)
        self.device = device
        self.records = self.manifest["frames"]
        for a, b in zip(self.records, self.records[1:]):
            if (
                b["frame_index"] <= a["frame_index"]
                or b["time"] <= a["time"]
                or b["step"] <= a["step"]
            ):
                raise ValueError("Sequence indices/time/steps must increase")

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        for record in self.records:
            path = (self.root / record["file"]).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("Frame file escapes sequence directory")
            if record.get("file_sha256") and file_sha256(path) != record["file_sha256"]:
                raise ValueError(
                    f"Render sequence frame checksum mismatch: {record['file']}"
                )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            frame = payload["frame"]
            for key in ("frame_index", "time", "step"):
                if frame[key] != record[key]:
                    raise ValueError(f"Frame metadata mismatch: {key}")
            yield (
                move_pack(frame, self.device),
                move_pack(payload["camera"], self.device),
            )


class FrameExporter:
    """One camera per video; multi-view rendering remains available in the API."""

    def __init__(
        self,
        directory,
        width,
        height,
        fps,
        *,
        total_frames,
        physics_mode,
        save_tensors=False,
        background=(1.0, 1.0, 1.0),
        legacy_frames=False,
    ):
        import cv2

        if (
            width <= 0
            or height <= 0
            or not math.isfinite(fps)
            or fps <= 0
            or total_frames < 1
        ):
            raise ValueError("Positive image size, FPS and frame count are required")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        for kind in ("rgb", "alpha"):
            (self.root / kind).mkdir(exist_ok=True)
        if save_tensors:
            (self.root / "render_tensors").mkdir(exist_ok=True)
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.encoded = (width + width % 2, height + height % 2)
        self.writer = cv2.VideoWriter(
            str(self.root / "simulation.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            self.encoded,
        )
        if not self.writer.isOpened():
            raise RuntimeError("OpenCV MP4 encoder unavailable")
        self.save_tensors = save_tensors
        self.mode = physics_mode
        self.records = []
        self.previews = []
        self.closed = False
        self.legacy_frames = legacy_frames
        if legacy_frames:
            (self.root / "frames").mkdir(exist_ok=True)
        self.preview_positions = set(
            [
                0,
                total_frames // 4,
                total_frames // 2,
                3 * total_frames // 4,
                total_frames - 1,
            ]
        )
        self.background = np.round(np.clip(np.asarray(background), 0, 1) * 255).astype(
            np.uint8
        )

    def write(self, result, frame_index, stamp):
        if result.rgb.shape != (
            1,
            1,
            3,
            self.height,
            self.width,
        ) or result.alpha.shape != (1, 1, 1, self.height, self.width):
            raise ValueError("Video exporter requires B=T=1 and configured dimensions")
        rgb = result.rgb.detach().cpu()
        alpha = result.alpha.detach().cpu()
        if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(alpha).all()):
            raise RuntimeError("Nonfinite rendered RGB/alpha")
        array = (
            (rgb[0, 0].permute(1, 2, 0).clamp(0, 1).numpy() * 255)
            .round()
            .astype(np.uint8)
        )
        mask = (alpha[0, 0, 0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array).save(self.root / "rgb" / f"{frame_index:06d}.png")
        Image.fromarray(mask).save(self.root / "alpha" / f"{frame_index:06d}.png")
        if self.save_tensors:
            atomic_torch(
                dict(rgb=rgb, alpha=alpha, frame_index=frame_index, time=stamp),
                self.root / "render_tensors" / f"{frame_index:06d}.pt",
            )
        if self.legacy_frames:
            legacy = Image.fromarray(array)
            draw = ImageDraw.Draw(legacy)
            draw.rectangle((0, 0, 260, 22), fill="white")
            draw.text(
                (5, 5),
                f"{self.mode} | frame {len(self.records) + 1:02d} | t={stamp:.2f}s",
                fill="black",
            )
            legacy.save(self.root / "frames" / f"{len(self.records) + 1:04d}.png")
            array = np.asarray(legacy)
        canvas = np.empty((self.encoded[1], self.encoded[0], 3), dtype=np.uint8)
        canvas[:] = self.background
        canvas[: self.height, : self.width] = array
        self.writer.write(np.ascontiguousarray(canvas[:, :, ::-1]))
        if len(self.records) in self.preview_positions:
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, min(self.width, 330), 22), fill="white")
            draw.text(
                (5, 5),
                f"{self.mode} | frame {frame_index} | t={stamp:.3f}s",
                fill="black",
            )
            self.previews.append(image)
        self.records.append(dict(frame_index=frame_index, time=stamp))

    def close(self):
        if self.closed:
            return
        self.writer.release()
        self.closed = True
        if self.previews:
            montage = Image.new(
                "RGB", (self.width * len(self.previews), self.height), "white"
            )
            for i, image in enumerate(self.previews):
                montage.paste(image, (i * self.width, 0))
            montage.save(self.root / "preview.png")
        atomic_json(
            dict(
                width=self.width,
                height=self.height,
                encoded_width=self.encoded[0],
                encoded_height=self.encoded[1],
                fps=self.fps,
                physics_mode=self.mode,
                frames=self.records,
            ),
            self.root / "video.json",
        )
