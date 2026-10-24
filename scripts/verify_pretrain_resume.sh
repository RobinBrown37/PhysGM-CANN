#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
physgm_cann="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/home/developer/Ascend/cann-9.1.0}}"
source "$physgm_cann/set_env.sh"
set -u
physgm_results="${1:-results/pretrain_resume}"
physgm_python="${PHYSGM_PYTHON:-.venv/bin/python}"
physgm_options=(--config configs/pretrain_smoke.yaml --set training.num_workers=0 --set training.warmup_steps=0 --set training.train_steps=4 --set training.checkpoint_every=2 --capture-update)
"$physgm_python" -u train_pretrain.py "${physgm_options[@]}" --output-dir "$physgm_results/continuous"
"$physgm_python" -u train_pretrain.py "${physgm_options[@]}" --max-steps 2 --output-dir "$physgm_results/resumed"
"$physgm_python" -u train_pretrain.py "${physgm_options[@]}" --resume auto --output-dir "$physgm_results/resumed"
"$physgm_python" scripts/compare_pretrain_resume.py --continuous "$physgm_results/continuous" --resumed "$physgm_results/resumed" --output "$physgm_results/comparison.json"
"$physgm_python" infer_network.py --checkpoint "$physgm_results/resumed/last.pt" --resize 256 --warmup 0 --repeats 1 --output-dir "$physgm_results/restored_inference"
"$physgm_python" render_static.py --input-pack "$physgm_results/restored_inference/prediction.pt" --warmup 0 --repeats 1 --output-dir "$physgm_results/restored_render"
