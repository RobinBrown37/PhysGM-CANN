#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  physgm_uv=uv
elif [[ -x ../.tools/bin/uv ]]; then
  physgm_uv=../.tools/bin/uv
else
  echo "Install uv and put it on PATH" >&2
  exit 1
fi
physgm_cann="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
if [[ ! -f "$physgm_cann/set_env.sh" ]]; then
  echo "Set ASCEND_HOME_PATH to the installed CANN toolkit" >&2
  exit 1
fi
source "$physgm_cann/set_env.sh"
set -u
"$physgm_uv" sync --locked --group render-build
"$physgm_uv" run --locked --group render-build python scripts/build_renderer.py --jobs "${PHYSGM_BUILD_JOBS:-8}" --soc "${PHYSGM_RENDER_SOC:-ascend910b}"
mkdir -p build
PHYSGM_TEST_NPU=1 "$physgm_uv" run --locked python -m pytest -q --junitxml=build/static_regression.xml -o junit_family=xunit1
"$physgm_uv" run --locked python render_static.py --online --device npu:0 --output-dir results/static_512
"$physgm_uv" run --locked python render_static.py --online --device npu:0 --input-resize 256 --backward-probe --profile --output-dir results/static_network_backward
