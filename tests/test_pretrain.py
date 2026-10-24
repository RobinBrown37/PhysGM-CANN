import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict

from rendering.static import RenderResult
from training.config import load_config
from training.constants import E_MEAN, E_STD, NU_MEAN, NU_STD, normalized_physical
from training.data import CursorBatchSampler, SupervisedDataset, process_frames
from training.engine import DistributedContext, accumulate_update
from training.losses import SupervisedLoss, robust_gaussian_nll
from training.state import atomic_save, make_scheduler, optimizer_groups, restore_rng, rng_state, training_signature, validate_resume

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scenes(tmp_path):
    entries = []
    for scene in range(3):
        directory = tmp_path/str(scene)
        directory.mkdir()
        frames = []
        for i in range(8):
            a = 2*math.pi*i/8
            c2w = np.eye(4)
            c2w[:3, 3] = [math.sin(a), math.cos(a), 2.]
            frame = {"file_path": f"{i}.png", "h": 32, "w": 48, "fx": 30., "fy": 31., "cx": 20., "cy": 14., "w2c": np.linalg.inv(c2w).tolist()}
            rgba = np.full((32, 48, 4), 40+scene*10+i, np.uint8)
            rgba[:, :, 3] = 200
            Image.fromarray(rgba).save(directory/frame["file_path"])
            frames.append(frame)
        (directory/"pose.json").write_text(json.dumps({"scene_name": str(scene), "frames": frames}))
        (directory/"physical.json").write_text(json.dumps({"material": "Leather", "E": 1e6, "nu": .49}))
        entries.append(f"{scene}/pose.json")
    (tmp_path/"train.txt").write_text("\n".join(entries))
    (tmp_path/"validation.txt").write_text(entries[-1])
    cfg = load_config(ROOT/"configs/pretrain_smoke.yaml")
    cfg.data.data_path, cfg.data.validation_path = str(tmp_path/"train.txt"), str(tmp_path/"validation.txt")
    cfg.data.resize_h = 32
    return cfg, tmp_path


def test_true_label_normalization_and_missing_rejection():
    assert normalized_physical({"material": "Leather", "E": 1e6, "nu": .49}) == (5, (6-E_MEAN)/E_STD, (.49-NU_MEAN)/NU_STD)
    for bad in ({"material": "unknown", "E": 1, "nu": .3}, {"material": "Wood", "E": 0, "nu": .3}, {"material": "Wood", "E": 1}):
        with pytest.raises((ValueError, KeyError)):
            normalized_physical(bad)


def test_real_format_sampling_and_missing_labels(scenes):
    cfg, root = scenes
    dataset = SupervisedDataset(cfg)
    batch = dataset[(3, 2, 1)]
    replay = dataset[(3, 2, 1)]
    assert batch["input_images"].shape == (4, 4, 32, 32)
    assert batch["test_images"].shape == (8, 4, 32, 32)
    assert set(batch["input_frame_idx"].tolist()) <= set(batch["test_frame_idx"].tolist())
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            assert torch.equal(batch[key], replay[key])
    assert not torch.equal(batch["input_frame_idx"], dataset[(4, 2, 1)]["input_frame_idx"])
    validation = SupervisedDataset(cfg, evaluation=True)
    assert torch.equal(validation[(1, 4, 0)]["input_frame_idx"], validation[(3, 8, 0)]["input_frame_idx"])
    (root/"1/physical.json").unlink()
    with pytest.raises(ValueError, match="physical.json"):
        SupervisedDataset(cfg)


def test_crop_transforms_offcenter_principal_point(scenes):
    cfg, root = scenes
    scene = json.loads((root/"0/pose.json").read_text())
    _, original, _ = process_frames(scene["frames"][:1], root/"0", cfg.data, 1.)
    _, cropped, _ = process_frames(scene["frames"][:1], root/"0", cfg.data, .8)
    expected = original.clone()
    expected[:, :2] *= 1.25
    expected[:, 2:] = expected[:, 2:]*1.25-4
    torch.testing.assert_close(cropped, expected)


def test_worker_independent_prefetch_and_consumed_cursor(scenes):
    cfg, _ = scenes
    dataset = SupervisedDataset(cfg)
    first = CursorBatchSampler(3, 1, seed=cfg.seed)
    direct = list(DataLoader(dataset, batch_sampler=first, num_workers=0))
    resumed = CursorBatchSampler(3, 1, seed=cfg.seed, cursor=1)
    worker = list(DataLoader(dataset, batch_sampler=resumed, num_workers=1, multiprocessing_context="spawn", timeout=30))
    for expected, actual in zip(direct[1:], worker):
        for key in ("sample_key", "input_images", "input_intr", "input_c2ws", "test_frame_idx"):
            assert torch.equal(expected[key], actual[key])
    rank0 = list(CursorBatchSampler(8, 1, rank=0, world_size=2))
    rank1 = list(CursorBatchSampler(8, 1, rank=1, world_size=2))
    assert {x[0][2] for x in rank0}.isdisjoint({x[0][2] for x in rank1})


def loss_inputs(device="cpu"):
    torch.manual_seed(91)
    prediction = {"phys_logits": torch.randn(2, 14, device=device, requires_grad=True)}
    for name in ("E_mu", "nu_mu"):
        prediction[name] = torch.randn(2, device=device, requires_grad=True)
    for name in ("E_var", "nu_var"):
        prediction[name] = torch.tensor([.8, 1.1], device=device, requires_grad=True)
    rendered = RenderResult(torch.rand(2, 2, 3, 4, 4, device=device, requires_grad=True),
                            torch.rand(2, 2, 1, 4, 4, device=device, requires_grad=True), {})
    batch = {"test_images": torch.rand(2, 2, 4, 4, 4, device=device), "physical_class": torch.tensor([2, 7], device=device),
             "E": torch.tensor([1., -.5], device=device), "nu": torch.tensor([.5, 1.], device=device)}
    return prediction, rendered, batch


def test_joint_loss_matches_original_independent_formula_and_gradients():
    p, r, b = loss_inputs()
    loss = SupervisedLoss({})
    actual = loss(p, r, b)
    gt = b["test_images"]
    expected = F.mse_loss(r.rgb, gt[:, :, :3]*gt[:, :, 3:]+1-gt[:, :, 3:])+.1*F.mse_loss(r.alpha, gt[:, :, 3:])
    expected += .1*F.binary_cross_entropy_with_logits(p["phys_logits"], F.one_hot(b["physical_class"], 14).float())
    for name in ("E", "nu"):
        mu, var = p[name+"_mu"], p[name+"_var"].clamp(min=.01)
        nll = (.5*math.log(2*math.pi)+.5*torch.log(var)+.5*(b[name]-mu)**2/var).clamp(max=100).mean()
        expected += .01*(nll+F.mse_loss(mu, b[name]))
    torch.testing.assert_close(actual["total_loss"], expected)
    inputs = [*p.values(), r.rgb, r.alpha]
    a = torch.autograd.grad(actual["total_loss"], inputs, retain_graph=True)
    e = torch.autograd.grad(expected, inputs)
    for x, y in zip(a, e):
        torch.testing.assert_close(x, y)
        assert torch.isfinite(x).all() and x.abs().sum() > 0
    var = torch.tensor([1e-8], requires_grad=True)
    capped = robust_gaussian_nll(torch.tensor([1e6]), var, torch.zeros(1))
    assert capped == 100
    assert torch.autograd.grad(capped, var)[0] == 0
    assert loss.lpips is None


class ToyTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1).double()

    def forward(self, batch):
        return {"losses": {"total_loss": F.mse_loss(self.linear(batch[0]), batch[1])}}


def training_options():
    return EasyDict(lr=.01, weight_decay=.05, beta1=.9, beta2=.95, train_steps=10, warmup_steps=0, scheduler_type="constant")


def test_accumulation_equivalence_and_no_extra_updates():
    torch.manual_seed(31)
    first = ToyTask()
    second = copy.deepcopy(first)
    options = training_options()
    a, _ = optimizer_groups(first, options)
    b, _ = optimizer_groups(second, options)
    sa, sb = make_scheduler(a, options), make_scheduler(b, options)
    data = (torch.randn(4, 2, dtype=torch.double), torch.randn(4, 1, dtype=torch.double))
    context = DistributedContext(torch.device("cpu"))
    accumulate_update(first, a, sa, None, iter([(data[0][:2], data[1][:2]), (data[0][2:], data[1][2:])]),
        device=torch.device("cpu"), distributed=context, accum_steps=2, clip_norm=30)
    accumulate_update(second, b, sb, None, iter([data]), device=torch.device("cpu"), distributed=context, accum_steps=1, clip_norm=30)
    for x, y in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-14)
    assert sa.last_epoch == sb.last_epoch == 1


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_nonfinite_window_never_updates_or_advances_schedule(bad_gradient):
    task = ToyTask()
    before = copy.deepcopy(task.state_dict())
    optimizer, _ = optimizer_groups(task, training_options())
    scheduler = make_scheduler(optimizer, training_options())
    x, y = torch.ones(1, 2, dtype=torch.double), torch.ones(1, 1, dtype=torch.double)
    if bad_gradient:
        handle = task.linear.weight.register_hook(lambda g: g*float("inf"))
    else:
        y *= float("nan")
    with pytest.raises(FloatingPointError):
        accumulate_update(task, optimizer, scheduler, None, iter([(x, y)]), device=torch.device("cpu"),
                          distributed=DistributedContext(torch.device("cpu")), accum_steps=1, clip_norm=30)
    assert scheduler.last_epoch == 0 and not optimizer.state
    for key, value in task.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(p.grad is None for p in task.parameters())
    if bad_gradient:
        handle.remove()


def test_scheduler_warmup_and_final_boundary():
    options = training_options()
    options.warmup_steps, options.scheduler_type = 2, "cosine"
    optimizer, _ = optimizer_groups(ToyTask(), options)
    scheduler = make_scheduler(optimizer, options)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[:3] == [0, .005, .01]
    assert lrs[-1] == 0 and all(x >= 0 for x in lrs)


def test_complete_state_cpu_safe_load_and_exact_replay(tmp_path):
    task = ToyTask()
    optimizer, _ = optimizer_groups(task, training_options())
    scheduler = make_scheduler(optimizer, training_options())
    context = DistributedContext(torch.device("cpu"))
    def update():
        batch = (torch.randn(2, 2, dtype=torch.double), torch.randn(2, 1, dtype=torch.double))
        return accumulate_update(task, optimizer, scheduler, None, iter([batch]), device=torch.device("cpu"),
                                 distributed=context, accum_steps=1, clip_norm=30)
    update()
    state = {"model": task.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(torch.device("cpu"))}
    atomic_save(state, tmp_path/"state.pt")
    numbers = (random.random(), np.random.random())
    reference = update()["metrics"]
    expected = copy.deepcopy(task.state_dict())
    saved = torch.load(tmp_path/"state.pt", weights_only=True)
    task.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_rng(saved["rng"], torch.device("cpu"))
    assert numbers == (random.random(), np.random.random())
    actual = update()["metrics"]
    assert actual == reference
    for key, value in task.state_dict().items():
        assert torch.equal(value, expected[key])


def test_resume_mismatch_and_original_checkpoint_rejected():
    signature = {"sha256": "new", "value": {"world_size": 1}}
    with pytest.raises(ValueError, match="init-checkpoint"):
        validate_resume({"model": {}}, signature)
    with pytest.raises(ValueError, match="world_size"):
        validate_resume({"format": "physgm-pretrain-v1", "signature": {"sha256": "old", "value": {"world_size": 2}}}, signature)


def test_loader_preflight_fits_container_shared_memory(scenes, monkeypatch):
    from types import SimpleNamespace
    from train_pretrain import DataStream
    import train_pretrain
    cfg, _ = scenes
    context = SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"))
    monkeypatch.setattr(train_pretrain.os, "statvfs", lambda _:SimpleNamespace(f_bavail=64*1024**2, f_frsize=1))
    cfg.training.num_workers = 4
    cfg.data.resize_h = 512
    large = DataStream(SupervisedDataset(cfg), cfg.training, context)
    assert large.info["actual_workers"] == 0
    cfg.data.resize_h = 256
    small = DataStream(SupervisedDataset(cfg), cfg.training, context)
    assert small.info["actual_workers"] == small.info["prefetch_factor"] == 1


def test_real_symmetric_random_views_remain_rigid():
    if not (ROOT/"datasets/physassets_subset/train.txt").exists():
        pytest.skip("Prepare real PhysAssets subset first")
    dataset = SupervisedDataset(load_config(ROOT/"configs/pretrain_smoke.yaml"))
    batch = dataset[(5, 9, 6)]
    rotation = batch["test_c2ws"][:, :3, :3]
    assert float((rotation@rotation.transpose(-1, -2)-torch.eye(3)).abs().max()) < 2e-6


def test_nonreentrant_transformer_checkpoint_matches_output_and_gradients():
    from model.physgm import Processor
    cfg = EasyDict(model={"num_layers":24,"dim":16,"transformer":{"head_dim":8}})
    reference = Processor(cfg).double().train()
    checkpointed = copy.deepcopy(reference)
    checkpointed.gradient_checkpointing = True
    x = torch.randn(1, 5, 16, dtype=torch.double, requires_grad=True)
    cx = x.detach().clone().requires_grad_(True)
    out, actual = reference(x), checkpointed(cx)
    for a,b in zip(out, actual):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    sum(v.square().mean() for v in out).backward()
    sum(v.square().mean() for v in actual).backward()
    for a,b in zip([x,*reference.parameters()],[cx,*checkpointed.parameters()]):
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_gloo_two_process_accumulation_and_collective_abort(tmp_path):
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "1", "PHYSGM_TEST_NPU": "0"}
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                             str(ROOT/"tests/distributed_pretrain_worker.py"), str(tmp_path/"distributed.json")],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((tmp_path/"distributed.json").read_text())
    assert report["rank_parameters_equal"] and report["matches_global_batch"] and report["collective_abort"]


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="Requires exposed NPU")
def test_npu_joint_loss_forward_backward_against_cpu():
    p, r, b = loss_inputs()
    npred = {k: v.detach().npu().requires_grad_(True) for k, v in p.items()}
    nr = RenderResult(r.rgb.detach().npu().requires_grad_(True), r.alpha.detach().npu().requires_grad_(True), {})
    nb = {k:v.npu() for k,v in b.items()}
    cpu = SupervisedLoss({})(p, r, b)
    npu = SupervisedLoss({})(npred, nr, nb)
    cpu["total_loss"].backward()
    npu["total_loss"].backward()
    for key in cpu:
        torch.testing.assert_close(npu[key].cpu(), cpu[key], rtol=1e-4, atol=1e-5)
    for a, e in zip([*npred.values(), nr.rgb, nr.alpha], [*p.values(), r.rgb, r.alpha]):
        torch.testing.assert_close(a.grad.cpu(), e.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="Requires exposed NPU")
def test_lpips_offline_frozen_eval_and_npu_input_gradient():
    from training.perceptual import OfflineLPIPS
    if not (ROOT/"weights/lpips/manifest.json").exists():
        pytest.skip("Prepare official LPIPS resources first")
    cpu = OfflineLPIPS(ROOT/"weights/lpips")
    npu = copy.deepcopy(cpu).npu()
    npu.train()
    assert not npu.training and all(not p.requires_grad for p in npu.parameters())
    x, y = torch.rand(1, 3, 32, 32)*2-1, torch.rand(1, 3, 32, 32)*2-1
    x.requires_grad_(True)
    nx = x.detach().npu().requires_grad_(True)
    expected, actual = cpu(x, y).mean(), npu(nx, y.npu()).mean()
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(nx.grad.cpu(), x.grad, rtol=1e-2, atol=1e-4)
    assert torch.isfinite(nx.grad).all() and nx.grad.abs().sum() > 0


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="Requires exposed NPU")
@pytest.mark.parametrize("nonfinite_loss", [False, True])
def test_npu_fp16_overflow_skips_whole_window_then_recovers(nonfinite_loss):
    class NpuTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1).npu()
        def forward(self, batch):
            with torch.autocast("npu", dtype=torch.float16):
                value = self.linear(batch[0])
            return {"losses":{"total_loss":F.mse_loss(value.float(),batch[1])}}
    device = torch.device("npu:0")
    task = NpuTask()
    optimizer = torch.optim.AdamW(task.parameters(), lr=.01, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
    scaler = torch.npu.amp.GradScaler(init_scale=128.)
    before = copy.deepcopy(task.state_dict())
    hook = task.linear.weight.register_hook(lambda gradient:gradient*float("inf")) if not nonfinite_loss else None
    batch = (torch.ones(1,2,device=device),torch.ones(1,1,device=device))
    bad = (batch[0],batch[1]*float("nan")) if nonfinite_loss else batch
    context = DistributedContext(device)
    failed = accumulate_update(task,optimizer,scheduler,scaler,iter([bad,bad]),device=device,
                                distributed=context,accum_steps=2,clip_norm=30)
    assert not failed["updated"] and scheduler.last_epoch == 0 and scaler.get_scale() == 64.
    assert all(torch.equal(v,before[k]) for k,v in task.state_dict().items())
    assert all(p.grad is None for p in task.parameters())
    if hook is not None:
        hook.remove()
    actual = accumulate_update(task,optimizer,scheduler,scaler,iter([batch,batch]),device=device,
                                distributed=context,accum_steps=2,clip_norm=30)
    assert actual["updated"] and scheduler.last_epoch == 1
    assert any(not torch.equal(v,before[k]) for k,v in task.state_dict().items())


@pytest.mark.skipif(os.environ.get("PHYSGM_TEST_NPU") != "1", reason="Requires exposed NPU")
def test_npu_renderer_nonreentrant_checkpoint_matches_gradients():
    from rendering import StaticGaussianRenderer
    from torch.utils.checkpoint import checkpoint
    g = {"xyz":torch.tensor([[[.037,-.021,1.1]]],device="npu:0"),
         "feature":torch.tensor([[[[.2,.1,.3]]]],device="npu:0"),
         "scale":torch.tensor([[[-2.5,-2.9,-3.2]]],device="npu:0"),
         "rotation":torch.tensor([[[.91,.13,-.21,.08]]],device="npu:0"),
         "opacity":torch.tensor([[[-.7]]],device="npu:0")}
    g = {k:v.requires_grad_(True) for k,v in g.items()}
    cg = {k:v.detach().clone().requires_grad_(True) for k,v in g.items()}
    renderer = StaticGaussianRenderer()
    c2ws = torch.eye(4,device="npu:0")[None,None]
    intr = torch.tensor([[[23.,25.,15.7,16.3]]],device="npu:0")
    keys = tuple(g)
    def render(*values):
        return renderer(dict(zip(keys,values)),c2ws,intr,width=33,height=31).rgba
    expected = render(*g.values())
    actual = checkpoint(render,*cg.values(),use_reentrant=False)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    for name in keys:
        torch.testing.assert_close(cg[name].grad,g[name].grad,rtol=1e-5,atol=1e-6)
