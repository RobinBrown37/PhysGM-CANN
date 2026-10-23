import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import torch

# TorchNPU must register its autograd device before the first CPU backward
# initializes PyTorch's engine in a mixed-device test process.
if os.environ.get("PHYSGM_TEST_NPU") == "1":
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():
        pytest.exit("PHYSGM_TEST_NPU=1 but no NPU is visible", returncode=1)
    torch.npu.set_device(0)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(4)
    torch.manual_seed(23)
