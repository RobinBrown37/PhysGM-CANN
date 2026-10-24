#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  physgm_uv=uv
elif [[ -x ../.tools/bin/uv ]]; then
  physgm_uv=../.tools/bin/uv
else
  echo "uv not found; install uv and put it on PATH" >&2
  exit 1
fi

"$physgm_uv" sync --locked
PHYSGM_TEST_NPU=1 "$physgm_uv" run --locked python -m pytest -q tests/test_network.py
"$physgm_uv" run --locked python infer_network.py --device npu:0 --amp-dtype bf16 --output-dir results/npu_forward
"$physgm_uv" run --locked python infer_network.py --device npu:0 --amp-dtype bf16 --backward-probe --resize 256 --profile --output-dir results/npu_backward
