"""Build the pinned CANN operators into build/, without installing into CANN."""
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--soc", default="ascend910b", choices=("ascend910b", "ascend910_93"))
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=None)
    args = parser.parse_args()
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not toolkit or not (Path(toolkit) / "include").exists():
        raise RuntimeError("Source the target CANN set_env.sh before building")
    source = ROOT / "third_party/gauss_splat"
    output = ROOT / "build/gauss_splat"
    output.mkdir(parents=True, exist_ok=True)
    cmake = ROOT / ".venv/bin/cmake"
    if not cmake.exists():
        raise RuntimeError("Run uv sync --locked --group render-build first")
    flags = ["-std=c++17"]
    for name in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
        value = getattr(torch._C, f"_PYBIND11_{name}", None)
        if value:
            flags.append(f"-D_PYBIND11_{name}={value}")
    common = [
        str(cmake), "-S", str(source), "-B", str(ROOT / "build/renderer_native"),
        "-DCMAKE_BUILD_TYPE=Release", f"-DASCEND_CANN_PACKAGE_PATH={toolkit}",
        f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}", f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DASCEND_COMPUTE_UNIT={args.soc}", "-DENABLE_BINARY_PACKAGE=ON",
        "-DKERNEL_NAME=*",
        f"-Dgauss_splat_PATH={output}", "-Dvendor_name=customize",
        f"-DABI={int(torch.compiled_with_cxx11_abi())}", "-DCOMPILE_WITH_XLA=OFF",
        f"-DEXT_CXX_FLAGS={' '.join(flags)}",
    ]
    started = time.perf_counter()
    for stage in ((0, 1, 2) if args.stage is None else (args.stage,)):
        print(f"Building CANN rendering stage {stage}", flush=True)
        subprocess.run(common + [f"-DBUILD_STAGE={stage}"], check=True)
        subprocess.run([str(cmake), "--build", str(ROOT / "build/renderer_native"), "--parallel", str(args.jobs)], check=True)
    if args.stage is None or args.stage == 2:
        info = output / f"packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/{args.soc}/binary_info_config.json"
        if not info.exists():
            raise RuntimeError("Build incomplete: compiled kernel binary_info_config.json is missing")
        result = {"upstream": "https://gitcode.com/cann/gauss-splat.git", "commit": "baf7591a710a86c9a7d810c5997b037723a91811", "soc": args.soc, "toolkit": toolkit, "torch": torch.__version__, "python": sys.version, "abi": torch.compiled_with_cxx11_abi(), "seconds": time.perf_counter() - started}
        (ROOT / "build/renderer_build.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
