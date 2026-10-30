import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from rendering import DynamicGaussianRenderer, StaticGaussianRenderer, RenderResult
from rendering.cameras import CameraPath, orbit_pose, local_basis
from rendering.geometry import quaternion_matrix
from rendering.mpm_adapter import MPMFrameAdapter
from rendering.sequence import FrameExporter, SequenceReader, SequenceWriter

ROOT = Path(__file__).resolve().parents[1]


def metadata(degree=0):
    return dict(
        gaussian_count=2,
        feature=torch.ones(2, (degree + 1) ** 2, 3, dtype=torch.float64) * 0.15,
        opacity=torch.tensor([[0.6], [0.4]], dtype=torch.float64),
        gaussian_ids=torch.tensor([19, 23]),
        world_to_sim_rotation=torch.tensor(
            [[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64
        ),
        normalization_center=torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        normalization_scale=2.0,
        camera_c2ws=torch.eye(4, dtype=torch.float64)[None, None],
        camera_intr=torch.tensor([[[20.0, 22.0, 8.0, 7.0]]], dtype=torch.float64),
        width=16,
        height=16,
    )


def scene(degree=0, device="cpu", dtype=torch.float64, empty=False):
    n = 0 if empty else 2
    raw = dict(
        xyz=torch.tensor([[[0.037, -0.021, 1.1], [-0.046, 0.033, 1.38]]], dtype=dtype)[
            :, :n
        ],
        scale=torch.tensor([[[-2.5, -2.9, -3.2], [-2.7, -2.4, -3.0]]], dtype=dtype)[
            :, :n
        ],
        rotation=torch.tensor(
            [[[0.91, 0.13, -0.21, 0.08], [0.87, -0.16, 0.23, 0.11]]], dtype=dtype
        )[:, :n],
        opacity=torch.tensor([[[-0.7], [0.2]]], dtype=dtype)[:, :n],
        feature=torch.linspace(
            -0.12, 0.3, max(n, 1) * (degree + 1) ** 2 * 3, dtype=dtype
        ).reshape(1, max(n, 1), (degree + 1) ** 2, 3)[:, :n],
    )
    raw = {k: v.to(device) for k, v in raw.items()}
    rotation = quaternion_matrix(raw["rotation"][0])
    mat = rotation * raw["scale"][0].exp()[:, None, :]
    frame = dict(
        xyz=raw["xyz"],
        covariance=(mat @ mat.mT)[None],
        feature=raw["feature"],
        opacity=raw["opacity"].sigmoid(),
    )
    camera = (
        torch.eye(4, dtype=dtype, device=device)[None, None].expand(1, 2, 4, 4).clone()
    )
    camera[:, 1, 0, 3] = 0.12
    intr = (
        torch.tensor([[[23.0, 25.0, 15.7, 16.3]]], dtype=dtype, device=device)
        .expand(1, 2, 4)
        .clone()
    )
    return frame, camera, intr, raw


def independent_sh(features, positions, origin, rotation, degree):
    """Associated-Legendre recurrence, independent of hardcoded SH polynomials."""
    direction = positions - origin
    if rotation is not None:
        direction = np.einsum("nij,nj->ni", rotation, direction)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    z = direction[:, 2]
    phi = np.arctan2(direction[:, 1], direction[:, 0])
    bases = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            k = abs(m)
            p = np.ones(len(z))
            for j in range(1, k + 1):
                p *= -(2 * j - 1) * np.sqrt(np.maximum(0, 1 - z * z))
            if l > k:
                previous = p
                current = (2 * k + 1) * z * p
                for j in range(k + 2, l + 1):
                    previous, current = (
                        current,
                        ((2 * j - 1) * z * current - (j + k - 1) * previous) / (j - k),
                    )
                p = current
            norm = math.sqrt(
                (2 * l + 1)
                / (4 * math.pi)
                * math.factorial(l - k)
                / math.factorial(l + k)
            )
            basis = norm * p
            if m < 0:
                basis *= math.sqrt(2) * np.sin(k * phi)
            elif m > 0:
                basis *= math.sqrt(2) * np.cos(k * phi)
            bases.append(basis)
    return np.maximum(0, np.einsum("nk,nkc->nc", np.stack(bases, -1), features) + 0.5)


@pytest.mark.parametrize("degree", range(4))
def test_zero_motion_matches_static_and_per_camera_sh(degree):
    frame, cameras, intr, raw = scene(degree)
    renderer = DynamicGaussianRenderer(backend="reference")
    a = renderer(frame, cameras, intr, width=32, height=32)
    b = StaticGaussianRenderer(sh_degree=degree, backend="reference")(
        raw, cameras, intr, width=32, height=32
    )
    torch.testing.assert_close(a.rgba, b.rgba, rtol=0, atol=0)
    for view in range(2):
        expected = independent_sh(
            frame["feature"][0].numpy(),
            frame["xyz"][0].numpy(),
            cameras[0, view, :3, 3].numpy(),
            None,
            degree,
        )
        oracle = renderer.renderer.render_precomputed(
            frame["xyz"],
            frame["covariance"],
            torch.from_numpy(expected)[None],
            frame["opacity"],
            cameras[:, view : view + 1],
            intr[:, view : view + 1],
            width=32,
            height=32,
        )
        torch.testing.assert_close(
            a.rgba[:, view : view + 1], oracle.rgba, rtol=1.0e-12, atol=1.0e-12
        )


@pytest.mark.parametrize("degree", range(4))
def test_rotating_sh_matches_independent_reference(degree):
    frame, cameras, intr, _ = scene(degree)
    rot = torch.tensor(
        [[[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=torch.float64
    ).expand(2, 3, 3)
    frame["rotation"] = rot[None]
    renderer = DynamicGaussianRenderer(backend="reference")
    result = renderer(frame, cameras, intr, width=32, height=32)
    for view in range(2):
        colors = independent_sh(
            frame["feature"][0].numpy(),
            frame["xyz"][0].numpy(),
            cameras[0, view, :3, 3].numpy(),
            rot.numpy(),
            degree,
        )
        expected = renderer.renderer.render_precomputed(
            frame["xyz"],
            frame["covariance"],
            torch.from_numpy(colors)[None],
            frame["opacity"],
            cameras[:, view : view + 1],
            intr[:, view : view + 1],
            width=32,
            height=32,
        )
        torch.testing.assert_close(
            result.rgba[:, view : view + 1], expected.rgba, rtol=1.0e-12, atol=1.0e-12
        )
    unrotated = renderer(
        {k: v for k, v in frame.items() if k != "rotation"},
        cameras,
        intr,
        width=32,
        height=32,
    )
    if degree == 0:
        torch.testing.assert_close(result.rgba, unrotated.rgba, rtol=0, atol=0)
    else:
        assert float((result.rgb - unrotated.rgb).abs().max()) > 1.0e-5


@pytest.mark.parametrize("degree", (0, 3))
def test_mpm_world_adapter_static_ids_and_filling_exclusion(degree):
    meta = metadata(degree)
    r = meta["world_to_sim_rotation"]
    center = meta["normalization_center"]
    scale = meta["normalization_scale"]
    xyz = torch.tensor([[0.1, 0.2, 1.0], [0.2, 0.1, 1.2]], dtype=torch.float64)
    cov = torch.diag(torch.tensor([0.02, 0.01, 0.005], dtype=torch.float64)).expand(
        2, 3, 3
    )
    sim_x = torch.zeros(5, 3, dtype=torch.float64)
    sim_x[:2] = (xyz @ r.T - center) * scale + 1
    sim_cov = torch.zeros(5, 3, 3, dtype=torch.float64)
    sim_cov[:2] = r @ cov @ r.T * scale**2
    q = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    sim = SimpleNamespace(
        state=dict(
            x=sim_x,
            F_trial=(
                q @ torch.diag(torch.tensor([1.2, 0.9, 0.7], dtype=torch.float64))
            ).expand(5, 3, 3),
        ),
        covariance=lambda: sim_cov,
        svd_backend="torch",
        time=0.2,
        steps=1000,
    )
    meta["static_gaussians"] = dict(
        xyz=xyz[:1] + 0.1,
        covariance=cov[:1],
        feature=meta["feature"][:1],
        opacity=meta["opacity"][:1],
        gaussian_ids=torch.tensor([31]),
    )
    adapter = MPMFrameAdapter(meta, torch.device("cpu"), torch.float64)
    frame = adapter.frame(sim, 4)
    composed = adapter.render_input(frame)
    torch.testing.assert_close(frame["xyz"], xyz)
    torch.testing.assert_close(frame["covariance"], cov)
    assert composed["xyz"].shape == (1, 3, 3) and adapter.shared[
        "gaussian_ids"
    ].tolist() == [19, 23, 31]
    torch.testing.assert_close(composed["xyz"][0, 2], xyz[0] + 0.1)
    if degree:
        torch.testing.assert_close(frame["rotation"], (r.T @ q.T @ r).expand(2, 3, 3))
        torch.testing.assert_close(
            composed["rotation"][0, 2], torch.eye(3, dtype=torch.float64)
        )
    else:
        assert "rotation" not in frame


def test_camera_orbit_scaling_indices_and_resume(tmp_path):
    meta = metadata()
    meta.update(
        world_to_sim_rotation=torch.eye(3),
        normalization_center=torch.zeros(3),
        normalization_scale=1.0,
    )
    cfg = dict(
        default_camera_index=-1,
        mpm_space_viewpoint_center=[1, 1, 1],
        mpm_space_vertical_upward_axis=[0, 1, 0],
        init_azimuthm=0,
        init_elevation=0,
        init_radius=1.8,
        move_camera=True,
        delta_a=15.0,
        delta_e=0.0,
        delta_r=0.03,
    )
    path = CameraPath(cfg, meta, width=320, height=240)
    a = path.at(0)
    b = path.at(4)
    torch.testing.assert_close(
        a["intr"], torch.tensor([[[160.0, 120.0, 160.0, 120.0]]])
    )
    expected = np.array([1.0, 0.0, 1.0]) / math.sqrt(2) * 1.8
    np.testing.assert_allclose(
        a["c2ws"][0, 0, :3, 3].numpy(), expected, rtol=1.0e-7, atol=1.0e-7
    )
    assert not torch.equal(a["c2ws"], b["c2ws"])
    restored = CameraPath(cfg, meta, width=320, height=240)
    torch.testing.assert_close(restored.at(4)["c2ws"], b["c2ws"], rtol=0, atol=0)
    pose = b["c2ws"][0, 0].double().numpy()
    toward = (-pose[:3, 3]) @ pose[:3, :3]
    np.testing.assert_allclose(toward, [0, 0, 1.92], rtol=0, atol=2.0e-7)
    cfg["move_camera"] = False
    fixed = CameraPath(cfg, meta)
    torch.testing.assert_close(
        fixed.at(1)["c2ws"], fixed.at(20)["c2ws"], rtol=0, atol=0
    )
    file = tmp_path / "cameras.json"
    file.write_text(
        json.dumps(
            [
                dict(
                    rotation=np.eye(3).tolist(),
                    position=[0, 0, 0],
                    width=1000,
                    height=800,
                    fx=800,
                    fy=700,
                    cx=312,
                    cy=245,
                )
            ]
        )
    )
    external = CameraPath(
        dict(default_camera_index=0), meta, cameras_json=file, width=500, height=400
    )
    torch.testing.assert_close(
        external.at(0)["intr"], torch.tensor([[[400.0, 350.0, 156.0, 122.5]]])
    )
    cached = CameraPath(
        dict(
            default_camera_index=0,
            _external_cameras=json.loads(file.read_text()),
        ),
        meta,
        width=500,
        height=400,
    )
    torch.testing.assert_close(cached.at(0)["intr"], external.at(0)["intr"])
    with pytest.raises(ValueError, match="out of range"):
        CameraPath(dict(default_camera_index=2), meta, cameras_json=file).at(0)
    with pytest.raises(ValueError, match="positive"):
        orbit_pose(np.zeros(3), local_basis([0, 1, 0]), 0, 0, -1)
    with pytest.raises(ValueError, match="nonzero"):
        local_basis([0, 0, 0])
    with pytest.raises(ValueError, match="poles"):
        orbit_pose(np.zeros(3), local_basis([0, 1, 0]), 0, 90, 1)


def small_shared():
    frame, camera, intr, _ = scene(dtype=torch.float32)
    return (
        dict(
            feature=frame["feature"][0],
            opacity=frame["opacity"][0],
            static=None,
            gaussian_ids=torch.tensor([4, 8]),
            dynamic_count=2,
            metadata=metadata(),
        ),
        dict(
            xyz=frame["xyz"][0],
            covariance=frame["covariance"][0],
            frame_index=0,
            time=0.04,
            step=200,
        ),
        dict(
            c2ws=camera[:, :1],
            intr=intr[:, :1],
            width=32,
            height=32,
            mode="input",
            frame_index=0,
        ),
    )


def test_sequence_streaming_append_and_scene_integrity(tmp_path):
    shared, frame, camera = small_shared()
    root = tmp_path / "seq"
    writer = SequenceWriter(root, shared, dict(frame_dt=0.04), "upstream")
    writer.append(frame, camera)
    reader = SequenceReader(root)
    loaded = list(reader)
    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0][0]["xyz"], frame["xyz"], rtol=0, atol=0)
    for value in loaded[0][0].values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and value.dtype == torch.float32
    second = {**frame, "frame_index": 1, "time": 0.08, "step": 400}
    SequenceWriter(root, shared, dict(frame_dt=0.04), "upstream", append=True).append(
        second, camera
    )
    assert len(SequenceReader(root)) == 2
    with pytest.raises(FileExistsError):
        SequenceWriter(root, shared, dict(frame_dt=0.04), "upstream")
    with pytest.raises(ValueError, match="advance"):
        writer.append(frame, camera)
    frame_file = root / "frames/000001.pt"
    original_frame = frame_file.read_bytes()
    payload = torch.load(frame_file, weights_only=False)
    payload["frame"]["xyz"] += 0.01
    torch.save(payload, frame_file)
    with pytest.raises(ValueError, match="frame checksum"):
        list(SequenceReader(root))
    frame_file.write_bytes(original_frame)
    corrupted = torch.load(root / "scene.pt", weights_only=False)
    corrupted["opacity"] += 0.01
    torch.save(corrupted, root / "scene.pt")
    with pytest.raises(ValueError, match="checksum"):
        SequenceReader(root)


def test_video_padding_raw_png_and_fps(tmp_path):
    exporter = FrameExporter(
        tmp_path, 33, 31, 20.0, total_frames=2, physics_mode="upstream"
    )
    image = RenderResult(
        torch.ones(1, 1, 3, 31, 33) * 0.4, torch.ones(1, 1, 1, 31, 33) * 0.7, {}
    )
    exporter.write(image, 0, 0.05)
    exporter.write(image, 1, 0.10)
    exporter.close()
    assert Image.open(tmp_path / "rgb/000000.png").size == (33, 31)
    assert Image.open(tmp_path / "alpha/000000.png").size == (33, 31)
    import cv2

    cap = cv2.VideoCapture(str(tmp_path / "simulation.mp4"))
    assert (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 34
        and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 32
    )
    assert (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2 and cap.get(cv2.CAP_PROP_FPS) == 20
    )
    cap.release()


def test_cpu_replay_imports_no_mpm_or_network(tmp_path):
    shared, frame, camera = small_shared()
    root = tmp_path / "sequence"
    SequenceWriter(root, shared, dict(frame_dt=0.04), "upstream").append(frame, camera)
    code = "import sys; import render_dynamic; render_dynamic.main(sys.argv[1:]); assert 'mpm' not in sys.modules; assert 'model' not in sys.modules"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--sequence",
            str(root),
            "--output-dir",
            str(tmp_path / "render"),
            "--device",
            "cpu",
            "--backend",
            "reference",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / "render/report.json").read_text())
    assert (
        report["status"] == "passed"
        and report["mpm_substeps_executed"] == 0
        and not report["network_loaded"]
    )


def test_pipeline_uses_structured_arguments_and_order(tmp_path):
    import pipeline

    args = pipeline.parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "weights with spaces.pt"),
            "--output-dir",
            str(tmp_path / "out with spaces"),
        ]
    )
    commands = pipeline.commands(args)
    assert [item[0] for item in commands] == ["network", "simulation_and_render"]
    assert str(tmp_path / "weights with spaces.pt") in commands[0][1]
    assert (
        "--save-render-frames" in commands[1][1] and "--camera-mode" in commands[1][1]
    )


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="NPU opt-in")
@pytest.mark.parametrize("degree", range(4))
@pytest.mark.parametrize("case", ("occlusion", "empty", "culling"))
def test_npu_dynamic_multiview_against_cpu(degree, case, record_property):
    frame, cameras, intr, _ = scene(degree, empty=case == "empty")
    if case == "occlusion":
        frame["xyz"][0, 0, 2], frame["xyz"][0, 1, 2] = 1.38, 1.1
    if case == "culling":
        frame["xyz"][0, 0, 2] = -0.1
        frame["xyz"][0, 1, 2] = 3.0
    rotation = torch.eye(3, dtype=torch.float64).expand_as(frame["covariance"]).clone()
    if frame["xyz"].shape[1]:
        rotation[0, 0] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    frame["rotation"] = rotation
    background = (0.2, 0.3, 0.4)
    cpu = DynamicGaussianRenderer(backend="reference")(
        frame, cameras, intr, width=33, height=35, background=background
    )
    npu = DynamicGaussianRenderer(backend="cann")(
        {k: v.float().to("npu:0") for k, v in frame.items()},
        cameras.float().to("npu:0"),
        intr.float().to("npu:0"),
        width=33,
        height=35,
        background=background,
    )
    for key in ("rgb", "alpha"):
        result = getattr(npu, key).cpu().double()
        expected = getattr(cpu, key)
        torch.testing.assert_close(result, expected, rtol=1.0e-3, atol=2.0e-4)
        record_property(key + "_max_abs_error", float((result - expected).abs().max()))
