"""Build project-local forward MPM Ascend C kernels (no system installation)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--soc", default="Ascend910B3")
    args = p.parse_args()
    toolkit = os.environ.get("ASCEND_HOME_PATH")
    if not toolkit:
        raise RuntimeError("Source CANN set_env.sh before building")
    cmake = ROOT / ".venv/bin/cmake"
    out = ROOT / "build/mpm"
    # CANN's legacy ascendc_library merges objects in place. Reconfiguring the
    # same merge outputs can feed a packed object back into ld; use a clean tree.
    if out.exists():
        shutil.rmtree(out)
    started = time.perf_counter()
    subprocess.run(
        [
            str(cmake),
            "-S",
            str(ROOT / "mpm/csrc"),
            "-B",
            str(out),
            f"-DSOC_VERSION={args.soc}",
            f"-DASCEND_CANN_PACKAGE_PATH={toolkit}",
            f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        [str(cmake), "--build", str(out), "--parallel", str(args.jobs)], check=True
    )
    (out / "build.json").write_text(
        json.dumps(
            {
                "soc": args.soc,
                "toolkit": toolkit,
                "seconds": time.perf_counter() - started,
                "operator_families": ["svd3x3", "p2g_contributions", "g2p_advect"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
