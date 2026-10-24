"""Real two-process Gloo regression for the same accumulation engine used on HCCL."""
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from training.engine import DistributedContext, accumulate_update


class Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1).double()

    def forward(self, batch):
        return {"losses": {"total_loss": F.mse_loss(self.linear(batch[0]), batch[1])}}


def main():
    torch.set_num_threads(1)
    context = DistributedContext(torch.device("cpu"))
    torch.manual_seed(103)
    task = Task()
    reference = copy.deepcopy(task)
    model = DDP(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1)
    x = torch.arange(8, dtype=torch.double).reshape(4, 2)/4
    y = x.sum(-1, keepdim=True)*.3
    batch = [(x[i:i+1], y[i:i+1]) for i in range(context.rank*2, context.rank*2+2)]
    accumulate_update(model, optimizer, scheduler, None, iter(batch), device=context.device,
                      distributed=context, accum_steps=2, clip_norm=30)
    roptimizer = torch.optim.AdamW(reference.parameters(), lr=.01, foreach=False, fused=False)
    F.mse_loss(reference.linear(x), y).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 30)
    roptimizer.step()
    vector = torch.cat([p.detach().flatten() for p in task.parameters()])
    gathered = context.gather(vector)
    same = all(torch.equal(v, vector) for v in gathered)
    matches = all(torch.allclose(a, b, rtol=1e-12, atol=1e-14) for a,b in zip(task.parameters(), reference.parameters()))
    bad = [(x[:1], y[:1]*(float("nan") if context.rank == 1 else 1))]
    aborted = False
    try:
        accumulate_update(model, optimizer, scheduler, None, iter(bad), device=context.device,
                          distributed=context, accum_steps=1, clip_norm=30)
    except FloatingPointError:
        aborted = True
    all_aborted = context.all_true(aborted)
    if context.rank == 0:
        Path(sys.argv[1]).write_text(json.dumps({"rank_parameters_equal":same, "matches_global_batch":matches, "collective_abort":all_aborted}))
    context.close()


if __name__ == "__main__":
    main()
