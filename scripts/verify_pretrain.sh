#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
physgm_uv="${PHYSGM_UV:-../.tools/bin/uv}"
if command -v uv >/dev/null 2>&1; then physgm_uv=uv; fi
physgm_cann="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/home/developer/Ascend/cann-9.1.0}}"
source "$physgm_cann/set_env.sh"
set -u
physgm_results="${PHYSGM_PRETRAIN_RESULTS:-results/pretrain_verification_$(date +%Y%m%d_%H%M%S)}"
"$physgm_uv" sync --locked --group train --group render-build
"$physgm_uv" run --locked --group train python scripts/prepare_physassets.py
"$physgm_uv" run --locked --group train python scripts/prepare_lpips.py
"$physgm_uv" run --locked --group render-build python scripts/build_renderer.py --jobs "${PHYSGM_BUILD_JOBS:-8}" --soc "${PHYSGM_RENDER_SOC:-ascend910b}"
PHYSGM_TEST_NPU=1 "$physgm_uv" run --locked --group train python -m pytest -q --junitxml=build/pretrain_regression.xml -o junit_family=xunit1
physgm_common=(--config configs/pretrain_smoke.yaml --set training.num_workers=0 --set training.warmup_steps=0)
physgm_train() { "$physgm_uv" run --locked --group train python train_pretrain.py "$@"; }
physgm_train --config configs/pretrain_smoke.yaml --audit --output-dir "$physgm_results/short100"
physgm_train --config configs/pretrain_npu.yaml --max-steps 1 --set training.warmup_steps=0 --audit --output-dir "$physgm_results/scale512"
physgm_train "${physgm_common[@]}" --set training.train_steps=20 --set training.scheduler_type=constant --set training.fixed_batch=true --audit --output-dir "$physgm_results/overfit20"
physgm_train --config configs/pretrain_scratch_npu.yaml --set data.resize_h=256 --set training.grad_accum_steps=1 --set training.warmup_steps=0 --set training.lr=0.00001 --set training.num_workers=0 --set training.checkpoint_transformer=true --set training.checkpoint_renderer=true --max-steps 2 --audit --output-dir "$physgm_results/scratch2" --dinov3-checkpoint checkpoint.pt
physgm_train "${physgm_common[@]}" --max-steps 1 --set training.lpips_loss=0.1 --audit --output-dir "$physgm_results/lpips1"
physgm_train "${physgm_common[@]}" --amp-dtype fp16 --max-steps 1 --audit --output-dir "$physgm_results/fp16"
physgm_replay=("${physgm_common[@]}" --set training.train_steps=4 --set training.checkpoint_every=2 --capture-update)
physgm_train "${physgm_replay[@]}" --output-dir "$physgm_results/continuous"
physgm_train "${physgm_replay[@]}" --max-steps 2 --output-dir "$physgm_results/resumed"
physgm_train "${physgm_replay[@]}" --resume auto --output-dir "$physgm_results/resumed"
"$physgm_uv" run --locked --group train python scripts/compare_pretrain_resume.py --continuous "$physgm_results/continuous" --resumed "$physgm_results/resumed" --output "$physgm_results/resume_comparison.json"
physgm_count=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$physgm_uv" run --locked --group train python -c 'import torch, torch_npu; print(torch.npu.device_count())')
physgm_nproc=1
if [[ "$physgm_count" -ge 2 ]]; then physgm_nproc=2; fi
"$physgm_uv" run --locked --group train torchrun --standalone --nproc_per_node="$physgm_nproc" train_pretrain.py "${physgm_common[@]}" --max-steps 1 --set training.grad_accum_steps=2 --audit --output-dir "$physgm_results/hccl"
"$physgm_uv" run --locked python infer_network.py --checkpoint "$physgm_results/resumed/last.pt" --resize 256 --warmup 0 --repeats 1 --output-dir "$physgm_results/restored_inference"
"$physgm_uv" run --locked python render_static.py --input-pack "$physgm_results/restored_inference/prediction.pt" --warmup 0 --repeats 1 --output-dir "$physgm_results/restored_render"
"$physgm_uv" run --locked --group train python scripts/summarize_pretrain.py --root "$physgm_results"
echo "Pretraining verification artifacts: $physgm_results"
