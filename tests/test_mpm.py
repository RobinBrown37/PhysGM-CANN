import os
import numpy as np
import pytest
import torch

from mpm import MPMSimulator
from mpm.boundaries import Boundaries
from mpm.linalg import det3, svd3
from mpm.materials import MATERIALS, constitutive
from mpm.prepare import fill_particles, particle_volume, ray_arrays, rotation
from mpm import reference
from rendering import StaticGaussianRenderer
from rendering.geometry import quaternion_matrix, sh_colors


def initial(material="plasticine", dtype=torch.float64):
    gen = torch.Generator().manual_seed(131)
    x = torch.rand(16, 3, generator=gen, dtype=dtype) * 0.2 + 0.9
    f = (
        torch.eye(3, dtype=dtype)[None]
        + torch.randn(16, 3, 3, generator=gen, dtype=dtype) * 0.01
    )
    return dict(
        format="physgm_mpm_v1",
        config=dict(
            material=material,
            n_grid=12,
            grid_lim=2.0,
            E=1000.0,
            nu=0.3,
            density=100.0,
            yield_stress=5.0,
            softening=0.1,
            plastic_viscosity=0.1,
            g=[0, -0.98, 0],
            substep_dt=2.0e-4,
        ),
        state=dict(
            x=x,
            volume=torch.ones(16, dtype=dtype) * 1.0e-4,
            F_trial=f,
            initial_cov=torch.eye(3, dtype=dtype).expand(16, 3, 3).clone() * 1.0e-4,
        ),
    )


def test_svd_signed_rank_and_determinant():
    a = torch.tensor(
        [
            [[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, -1.0]],
            [[1.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.float64,
    )
    u, s, v = svd3(a)
    torch.testing.assert_close((u * s[:, None, :]) @ v.mT, a)
    torch.testing.assert_close(det3(a), torch.linalg.det(a))
    assert bool((det3(u) > 0).all()) and bool((det3(v) > 0).all()) and s[0, -1] < 0


def test_plasticine_hardening_damage_state_against_scalar_reference():
    for hardening in (0, 1):
        for softening in (0.1, 1.0e6):
            p = initial("plasticine")
            p["config"].update(hardening=hardening, xi=0.3, softening=softening)
            sim = MPMSimulator(dtype=torch.float64).load_state(p)
            s = sim.state
            s["F_trial"][0] = torch.diag(
                torch.tensor([1.2, 0.8, 0.95], dtype=torch.float64)
            )
            expected = reference.material(
                s["F_trial"][0].numpy(),
                float(s["mu"][0]),
                float(s["lam"][0]),
                float(s["yield_stress"][0]),
                sim.cfg,
                0.0002,
            )
            actual = constitutive(
                s["F_trial"], s["mu"], s["lam"], s["yield_stress"], sim.cfg, 0.0002
            )
            for result, target in zip(actual, expected):
                np.testing.assert_allclose(
                    result[0].numpy(), target, rtol=1.0e-9, atol=1.0e-9
                )
            if softening > 1:
                assert actual[2][0] == 0 and actual[3][0] == 0 and actual[4][0] < 0


@pytest.mark.parametrize("material", MATERIALS)
def test_material_independent_numpy_reference(material):
    p = initial(material)
    sim = MPMSimulator(dtype=torch.float64).load_state(p)
    s = sim.state
    # Yielded/non-yielded, tensile/compressive and damaged-yield cases.
    s["F_trial"][0] = torch.diag(torch.tensor([1.2, 0.8, 0.95], dtype=torch.float64))
    s["yield_stress"][1] = 0
    actual = constitutive(
        s["F_trial"], s["mu"], s["lam"], s["yield_stress"], sim.cfg, 2.0e-4
    )
    for i in range(len(s["x"])):
        expect = reference.material(
            s["F_trial"][i].numpy(),
            float(s["mu"][i]),
            float(s["lam"][i]),
            float(s["yield_stress"][i]),
            sim.cfg,
            2.0e-4,
        )
        for result, target in zip(actual, expect):
            np.testing.assert_allclose(
                result[i].numpy(), target, rtol=1.0e-9, atol=1.0e-9
            )


@pytest.mark.parametrize("material", MATERIALS)
def test_single_step_independent_reference(material):
    sim = MPMSimulator(dtype=torch.float64).load_state(initial(material))
    dt = 2.0e-4
    expected, grid, gv = reference.step(
        {k: v.numpy() for k, v in sim.state.items()}, sim.cfg, dt
    )
    sim.step(dt)
    for key in ("x", "v", "F", "F_trial", "C", "stress", "mu", "lam", "yield_stress"):
        np.testing.assert_allclose(
            sim.state[key].numpy(), expected[key], rtol=1.0e-8, atol=1.0e-8, err_msg=key
        )
    np.testing.assert_allclose(sim.grid.numpy(), grid, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(sim.grid_v.numpy(), gv, rtol=1.0e-8, atol=1.0e-8)
    assert abs(float(sim.grid[:, 0].sum() / sim.state["mass"].sum()) - 1) < 1.0e-12


def test_hundred_steps_and_exact_restore():
    p = initial()
    a = MPMSimulator(dtype=torch.float64).load_state(p)
    b = MPMSimulator(dtype=torch.float64).load_state(p)
    for _ in range(50):
        a.step(2.0e-4)
        b.step(2.0e-4)
    b = MPMSimulator(dtype=torch.float64).load_state(b.export_state(cpu=True))
    for _ in range(50):
        a.step(2.0e-4)
        b.step(2.0e-4)
    for k in a.state:
        torch.testing.assert_close(a.state[k], b.state[k], rtol=0, atol=0)
    assert a.time == b.time and a.steps == b.steps == 100


@pytest.mark.parametrize("rpic", [-1.0, 0.0, 0.5])
def test_rpic_and_incremental_covariance_reference(rpic):
    p = initial("metal")
    p["config"].update(rpic_damping=rpic, update_cov_with_F=True)
    p["state"]["C"] = (
        torch.arange(16 * 9, dtype=torch.float64).reshape(16, 3, 3) * 0.0001
    )
    sim = MPMSimulator(dtype=torch.float64).load_state(p)
    expect, _, _ = reference.step(
        {k: v.numpy() for k, v in sim.state.items()}, sim.cfg, 2.0e-4
    )
    sim.step(2.0e-4)
    for key in ("x", "v", "C", "F_trial", "cov"):
        np.testing.assert_allclose(
            sim.state[key].numpy(), expect[key], rtol=1.0e-9, atol=1.0e-9
        )


def test_hundred_steps_scalar_numpy_reference():
    p = initial("plasticine")
    p["config"]["n_grid"] = 8
    p["state"] = {k: v[:4] for k, v in p["state"].items()}
    sim = MPMSimulator(dtype=torch.float64).load_state(p)
    expected = {k: v.numpy().copy() for k, v in sim.state.items()}
    for _ in range(100):
        expected, _, _ = reference.step(expected, sim.cfg, 2.0e-4)
        sim.step(2.0e-4)
    for key in ("x", "v", "C", "F_trial", "stress"):
        np.testing.assert_allclose(
            sim.state[key].numpy(), expected[key], rtol=1.0e-8, atol=1.0e-8
        )


def test_freefall_selection_rpic_and_out_of_bounds():
    p = initial("jelly")
    p["state"]["F_trial"] = torch.eye(3, dtype=torch.float64).expand(16, 3, 3).clone()
    p["state"]["selection"] = torch.tensor([0] * 15 + [1])
    sim = MPMSimulator(dtype=torch.float64).load_state(p)
    sim.step(2.0e-4)
    torch.testing.assert_close(
        sim.state["v"][:15, 1], torch.full((15,), -0.98 * 2.0e-4, dtype=torch.float64)
    )
    torch.testing.assert_close(sim.state["x"][-1], p["state"]["x"][-1])
    p["state"]["x"][0] = torch.tensor([-2.0, 1.0, 1.0])
    sim = MPMSimulator(dtype=torch.float64).load_state(p)
    with pytest.raises(RuntimeError, match="leaves the grid"):
        sim.step(2.0e-4)


def test_solver_ignores_outer_autocast():
    p = initial()
    a = MPMSimulator().load_state(p)
    b = MPMSimulator().load_state(p)
    a.step(2.0e-4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        b.step(2.0e-4)
    for key in a.state:
        torch.testing.assert_close(a.state[key], b.state[key], rtol=0, atol=0)


def test_surface_compat_corrected_and_cut():
    x = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    c = dict(
        type="surface_collider",
        point=[1, 1, 1],
        normal=[0, 1, 0],
        surface="slip",
        friction=0,
    )
    for mode in ("upstream", "corrected"):
        b = Boundaries([c], x, 10, 0.2, mode, 0.001)
        v = torch.ones(1000, 3, dtype=x.dtype)
        v[:, 1] = -2
        out = b.grid(v, 0, 0.001)
        mask = b.grid_positions[:, 1] < 1
        target = torch.tensor(
            [0.0, 0, 0] if mode == "upstream" else [1.0, 0, 1], dtype=x.dtype
        )
        torch.testing.assert_close(out[mask], target.expand(mask.sum(), 3))
        torch.testing.assert_close(out[~mask], v[~mask])
    cut = {**c, "surface": "cut"}
    b = Boundaries([cut], x, 10, 0.1, "upstream", 0.001)
    v = torch.ones(1000, 3, dtype=x.dtype)
    out = b.grid(v, 0, 0.001)
    inside = (b.grid_positions[:, 2] >= 0.4) & (b.grid_positions[:, 2] <= 0.53)
    torch.testing.assert_close(
        out[inside], torch.tensor([0.3, 0, 0.3], dtype=x.dtype).expand(inside.sum(), 3)
    )


def test_impulse_translation_rotation_and_cuboid_resume():
    x = torch.tensor([[1.0, 1.0, 1.0], [1.1, 1.0, 1.0]], dtype=torch.float64)
    configs = [
        dict(type="particle_impulse", force=[2, 0, 0], num_dt=2),
        dict(
            type="enforce_particle_translation",
            point=[1, 1, 1],
            size=[0.01, 0.1, 0.1],
            velocity=[0, 2, 0],
        ),
    ]
    b = Boundaries(configs, x, 10, 0.2, "upstream", 0.01)
    out = b.particles(x, torch.zeros_like(x), torch.ones(2), 0, 0.01)
    torch.testing.assert_close(
        out, torch.tensor([[0, 2, 0], [0.02, 0, 0]], dtype=x.dtype)
    )
    r = dict(
        type="enforce_particle_velocity_rotation",
        point=[1, 1, 1],
        normal=[0, 1, 0],
        half_height_and_radius=[1, 1],
        rotation_scale=2,
        translation_scale=0.3,
    )
    b = Boundaries([r], x, 10, 0.2, "upstream", 0.01)
    out = b.particles(x, torch.zeros_like(x), torch.ones(2), 0, 0.01)
    torch.testing.assert_close(
        out,
        torch.tensor([[0, 0.3, 0], [0, 0.3, 0.2]], dtype=x.dtype),
        atol=1.0e-14,
        rtol=1.0e-12,
    )
    cuboid = dict(
        type="cuboid",
        point=[1, 1, 1],
        size=[0.3, 0.3, 0.3],
        velocity=[1, 0, 0],
        end_time=0.02,
        reset=1,
    )
    b = Boundaries([cuboid], x, 10, 0.2, "upstream", 0.01)
    b.grid(torch.ones(1000, 3, dtype=x.dtype), 0, 0.01)
    assert b.items[0]["point"][0] == 1.01
    snap = b.export()
    c = Boundaries([], x, 10, 0.2, "upstream", 0.01)
    c.restore(snap)
    torch.testing.assert_close(
        b.grid(torch.ones(1000, 3, dtype=x.dtype), 0.01, 0.01),
        c.grid(torch.ones(1000, 3, dtype=x.dtype), 0.01, 0.01),
    )


def test_release_modes_honor_documented_timing():
    x = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    cfg = dict(
        type="release_particles_sequentially",
        normal=[0, 1, 0],
        start_position=0,
        end_position=1.5,
        num_layers=4,
        start_time=2.0,
        end_time=6.0,
    )
    a = Boundaries([cfg], x, 10, 0.2, "upstream", 0.01)
    b = Boundaries([cfg], x, 10, 0.2, "corrected", 0.01)
    assert len(a.items) == 50 and len(b.items) == 4
    assert b.items[0]["end_time"] == 3 and b.items[-1]["end_time"] == 6


def test_invalid_input_and_time_window():
    p = initial()
    p["config"]["nu"] = 0.5
    with pytest.raises(ValueError, match="nu"):
        MPMSimulator().load_state(p)
    sim = MPMSimulator().load_state(initial())
    with pytest.raises(ValueError, match="dt"):
        sim.step(float("nan"))
    x = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    b = Boundaries(
        [
            dict(
                type="enforce_particle_translation",
                point=[1, 1, 1],
                size=[1, 1, 1],
                velocity=[1, 0, 0],
                end_time=0.01,
            )
        ],
        x,
        4,
        0.5,
        "upstream",
        0.001,
    )
    out = b.particles(x, torch.zeros_like(x), torch.ones(1), 0.01, 0.001)
    torch.testing.assert_close(out, torch.zeros_like(x), rtol=0, atol=0)
    p = initial()
    p.update(physics_mode="upstream", steps=0, time=0.0)
    MPMSimulator(physics_mode="corrected").load_state(p)
    p["steps"] = 1
    with pytest.raises(ValueError, match="mode mismatch"):
        MPMSimulator(physics_mode="corrected").load_state(p)


def test_ray_search_matches_scalar_and_filling_seed():
    rng = np.random.default_rng(41)
    occupied = rng.random((7, 7, 7)) > 0.6
    for direction in range(6):
        hit, count = ray_arrays(occupied, direction // 2, direction % 2 == 0)
        for index in np.ndindex(occupied.shape):
            p = list(index)
            p[direction // 2] += 1 if direction % 2 == 0 else -1
            state = False
            h = False
            c = 0
            while min(p) >= 0 and max(p) < 7:
                new = bool(occupied[tuple(p)])
                h |= new
                c += int(new and not state)
                state = new
                p[direction // 2] += 1 if direction % 2 == 0 else -1
            assert bool(hit[index]) == h and count[index] == c
    x = np.array([[0.5, 0.5, 0.5]], np.float32)
    cov = np.eye(3, dtype=np.float32)[None] * 0.04
    cfg = dict(
        n_grid=4,
        density_threshold=0.0,
        search_threshold=0.1,
        max_partciels_per_cell=2,
        max_particles_num=200,
    )
    a, r = fill_particles(x, np.ones(1, np.float32), cov, cfg, 1.0, 12)
    b, _ = fill_particles(x, np.ones(1, np.float32), cov, cfg, 1.0, 12)
    np.testing.assert_array_equal(a, b)
    assert len(a) > 1
    vol = particle_volume(a, 4, 0.25)
    assert np.isfinite(vol).all() and (vol > 0).all()


def test_precomputed_renderer_matches_raw_and_rotation_sh():
    dtype = torch.float64
    g = dict(
        xyz=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=dtype),
        scale=torch.tensor([[[-2.0, -2.5, -3.0]]], dtype=dtype),
        rotation=torch.tensor([[[1.0, 0, 0, 0]]], dtype=dtype),
        opacity=torch.tensor([[[-0.7]]], dtype=dtype),
        feature=torch.ones(1, 1, 1, 3, dtype=dtype) * 0.2,
    )
    c = torch.eye(4, dtype=dtype)[None, None]
    intr = torch.tensor([[[20.0, 20.0, 8.0, 8.0]]], dtype=dtype)
    renderer = StaticGaussianRenderer(backend="reference")
    a = renderer(g, c, intr, width=16, height=16)
    mat = quaternion_matrix(g["rotation"][0]) * g["scale"][0].exp()[:, None, :]
    cov = mat @ mat.mT
    color = sh_colors(g["feature"][0], g["xyz"][0], torch.zeros(3, dtype=dtype), 0)
    b = renderer.render_precomputed(
        g["xyz"],
        cov[None],
        color[None],
        g["opacity"].sigmoid(),
        c,
        intr,
        width=16,
        height=16,
    )
    torch.testing.assert_close(a.rgb, b.rgb, rtol=0, atol=0)
    torch.testing.assert_close(a.alpha, b.alpha, rtol=0, atol=0)
    with pytest.raises(ValueError, match="probability"):
        renderer.render_precomputed(
            g["xyz"],
            cov[None],
            color[None],
            torch.ones_like(g["opacity"]) * 2,
            c,
            intr,
            width=16,
            height=16,
        )
    f = torch.arange(12, dtype=dtype).reshape(1, 4, 3) / 20
    r = torch.from_numpy(rotation(90, 0)).double()[None]
    point = torch.tensor([[0.2, 0.3, 1.0]], dtype=dtype)
    torch.testing.assert_close(
        sh_colors(f, point, torch.zeros(3, dtype=dtype), 1, direction_rotation=r),
        sh_colors(f, (r @ point[..., None])[..., 0], torch.zeros(3, dtype=dtype), 1),
    )


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="NPU opt-in")
@pytest.mark.parametrize("affine", (False, True))
def test_native_g2p_constant_and_affine_velocity(affine):
    from mpm.native import p2g, g2p

    sim = MPMSimulator(device="npu:0", transfer_backend="native").load_state(initial())
    s = sim.state
    eye = torch.eye(3, device=sim.device)
    s["F"] = eye.expand(len(s["x"]), 3, 3).clone()
    indices, stride = p2g(s, sim.grid, sim.n_grid, sim.dx, 0.0002, 0.0)
    matrix = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, 0.1, 0.2], [-0.1, 0.2, 0.1]], device=sim.device
    )
    if not affine:
        matrix.zero_()
    bias = torch.tensor([0.1, 0.2, 0.3], device=sim.device)
    grid_v = sim.boundaries.grid_positions @ matrix.T + bias
    x = s["x"].clone()
    expected_v = x @ matrix.T + bias
    g2p(s, grid_v, indices, stride, sim.dx, 0.0002)
    torch.testing.assert_close(s["v"], expected_v, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(
        s["x"], x + 0.0002 * expected_v, rtol=1.0e-5, atol=1.0e-6
    )
    torch.testing.assert_close(
        s["C"], matrix.expand_as(s["C"]), rtol=1.0e-5, atol=1.0e-5
    )
    torch.testing.assert_close(
        s["F_trial"],
        (eye + 0.0002 * matrix).expand_as(s["F_trial"]),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="NPU opt-in")
@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("transfer_backend", ("torch", "native"))
def test_npu_reference_hundred_steps(material, transfer_backend):
    p = initial(material)
    a = MPMSimulator(dtype=torch.float64).load_state(p)
    b = MPMSimulator(device="npu:0", transfer_backend=transfer_backend).load_state(p)
    for i in range(100):
        a.step(2.0e-4)
        b.step(2.0e-4)
        if i in (0, 99):
            tol = 1.0e-4 if i == 0 else 1.0e-3
            for key in (
                "x",
                "v",
                "F_trial",
                "C",
                "stress",
                "mu",
                "lam",
                "yield_stress",
            ):
                expected = a.state[key]
                actual = b.state[key].cpu().double()
                error = float(
                    torch.linalg.vector_norm(actual - expected)
                    / torch.linalg.vector_norm(expected).clamp_min(1)
                )
                assert error <= tol, (material, i, key, error)
