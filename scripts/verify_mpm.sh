#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
physgm_python=.venv/bin/python
"$physgm_python" scripts/build_mpm.py
"$physgm_python" -m pytest -q tests/test_mpm.py
PHYSGM_TEST_NPU=1 "$physgm_python" -m pytest -q tests/test_mpm.py
"$physgm_python" scripts/verify_mpm_svd.py --profile --output results/mpm_svd_final.json
"$physgm_python" simulate_mpm.py --prepare-only
"$physgm_python" scripts/benchmark_mpm.py --profile
"$physgm_python" simulate_mpm.py --initial-state results/mpm_cake/initial_state.pt --transfer-backend native
"$physgm_python" scripts/verify_mpm_state.py
"$physgm_python" scripts/export_mpm_overview.py
"$physgm_python" scripts/summarize_mpm.py
