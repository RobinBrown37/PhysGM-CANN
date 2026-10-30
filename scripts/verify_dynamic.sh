#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
physgm_python=.venv/bin/python
dynamic_output=${PHYSGM_DYNAMIC_OUTPUT:-results/dynamic_cake}
test_output=${PHYSGM_DYNAMIC_TEST_OUTPUT:-results/dynamic_tests}
mkdir -p "$test_output"

"$physgm_python" -m pytest -q tests/test_dynamic_render.py --junitxml="$test_output/cpu.xml"
PHYSGM_TEST_NPU=1 "$physgm_python" -m pytest -q tests/test_dynamic_render.py --junitxml="$test_output/npu.xml"
PHYSGM_TEST_NPU=1 "$physgm_python" -m pytest -q \
  tests/test_network.py tests/test_static_render.py tests/test_pretrain.py \
  --junitxml="$test_output/existing53.xml"
PHYSGM_TEST_NPU=1 "$physgm_python" -m pytest -q tests/test_mpm.py \
  --junitxml="$test_output/mpm41.xml"

"$physgm_python" pipeline.py --output-dir "$dynamic_output" --save-render-tensors
"$physgm_python" render_dynamic.py \
  --sequence "$dynamic_output/sequence" \
  --output-dir "$dynamic_output/replay_fixed" \
  --compare-to "$dynamic_output/render_tensors" --profile
"$physgm_python" render_dynamic.py \
  --sequence "$dynamic_output/sequence" \
  --output-dir "$dynamic_output/replay_orbit" \
  --camera-mode config --move-camera
"$physgm_python" scripts/summarize_dynamic.py \
  --run "$dynamic_output" --tests "$test_output"
