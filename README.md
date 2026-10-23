# PhysGM-CANN：神经网络迁移（M1）

以 PhysGM `7722175ba55db54fd430a7d29a4407b8f8ffe6b6` 为基线，使用 PyTorch＋TorchNPU 执行纯网络预测。包含完整 DINOv3、射线编码、24 层 Transformer、DPT Gaussian 头和三个物理属性头。本阶段交付真实权重推理与梯度探针；静态渲染、MPM 和监督训练尚未接入。

原仓 `/mnt/workspace/PhysGM` 保持不变。原计划见 [plan.md](plan.md)，实际验证结果见 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)。

## 环境

本项目使用独立 uv 环境，Python 3.11、PyTorch `2.7.1+cpu`、TorchNPU `2.7.1.post8`、Transformers `4.57.1`，对应 CANN `9.1.0`。PyTorch CPU 构建提供框架，TorchNPU 提供昇腾设备执行。依据：[Ascend 版本对应表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)、[uv PyTorch 配置](https://docs.astral.sh/uv/guides/integration/pytorch/)。

```bash
cd /mnt/workspace/PhysGM-CANN
# 本工作区已安装 uv；其他机器可使用自己的 uv。
export PATH="/mnt/workspace/.tools/bin:$PATH"
uv sync --locked --python /usr/local/bin/python

# 运行前加载目标机器的 CANN 环境；本机路径如下。
source /home/developer/Ascend/cann-9.1.0/set_env.sh
```

uv 管理 Python 包，驱动与 CANN 由机器提供。默认启动开发测试依赖；只运行推理时可用 `uv sync --locked --no-dev`。权重和示例准备好以后，模型构建与权重加载完全离线，无须 Hugging Face 登录或下载 DINO。

## 推理

```bash
uv run --locked python infer_network.py \
  --checkpoint checkpoint.pt --device npu:0 --amp-dtype bf16 \
  --output-dir results/npu_forward
```

默认处理 cake 的四张 512×512 RGBA 图像，batch size 为 1。`--config` 可指定配置，`--scene-index` 选择列表中的场景，`--resize` 覆盖输入高度并同步缩放内参。数据路径相对项目根目录；场景内图像路径相对 pose.json。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `prediction.pt` | 仅含 CPU Tensor 与普通类型的输入、预测及权重加载信息 |
| `point_clouds.ply` | 按 opacity 阈值筛选的 Gaussian，保留原仓 SH 与参数存储约定 |
| `predicted_phys.json` | 材料类别，以及原仓归一化逆变换后的 E、ν |
| `run_report.json` | 实际设备、版本、输入形状、权重条目、耗时、峰值内存和状态 |

纯 CPU 调试示例：

```bash
uv run --locked python infer_network.py --device cpu --amp-dtype fp32 \
  --resize 64 --warmup 0 --repeats 1 --output-dir results/cpu_smoke
```

NPU 不可用时明确报错，不会自动切换设备。受限沙箱可能看不到设备节点；本次 NPU 验收使用了沙箱外可见的 Ascend910B3。

## 梯度与算子验证

```bash
uv run --locked python infer_network.py --device npu:0 \
  --backward-probe --resize 256 --profile --output-dir results/npu_backward

uv run --locked python -m pytest -q
PHYSGM_TEST_NPU=1 uv run --locked python -m pytest -q

# 一次执行 CPU/NPU 测试、512 推理和 256 反向探针
bash scripts/verify_npu.sh
```

梯度探针使用真实图像、相机和现有权重，对各输出构造平方均值标量。检查所有启用参数的梯度存在，各模块梯度有限且总量非零，并确认全局 token 有梯度、DINO 冻结且保持 eval。没有使用监督标签、优化器或参数更新。

`--profile` 在计时结束后单独重放一次，输出 `operators.json`（Chrome trace）、`operator_names.json` 和 Attention 算子名称。TorchNPU 还会生成被 git 忽略的 `export_only_prof_dir/` 原始 profiling 目录。

计时默认先执行一次预热，再记录三次同步前向的中位数和 p95。首次执行包含位置常量缓存与算子初始化；模型加载、传输、输出 I/O 分别记录，不计入预热后前向。`memory` 是 profiling 重放之前的峰值分配与预留内存。

## Python 接口与兼容性

```python
from pathlib import Path
import torch
import yaml
from easydict import EasyDict
from model.physgm import load_checkpoint
from runtime import select_device, autocast

config = EasyDict(yaml.safe_load(Path("configs/network_npu.yaml").read_text()))
device = select_device("npu:0")
model, load_info = load_checkpoint(config, "checkpoint.pt", device)
model.eval()
# batch 的张量需先移动到 device。
with torch.no_grad(), autocast(device, "bf16"):
    prediction = model.predict(batch, return_unpruned=True)
```

嵌入已有程序时，应在第一次 CPU 或 NPU backward 之前调用 `select_device("npu:0")` 或导入 `torch_npu`，让 PyTorch 的 autograd engine 提前注册 NPU。混合设备回归已覆盖这一初始化顺序。

必需输入为 `input_images: [B,V,3|4,H,W]`（浮点、范围 [0,1]）、`input_intr: [B,V,4]`（fx/fy/cx/cy，像素单位）、`input_c2ws: [B,V,4,4]`。H/W 必须是 16 的倍数。`forward` 委托 `predict`；目标视图字段会被忽略，任何模式都不会触发渲染。

返回 `gaussians`（xyz、feature、scale、rotation、opacity）、`gaussian_usage`、`phys_logits`、`E_mu/E_var`、`nu_mu/nu_var`；可选 `gaussians_unpruned` 为过滤前输出。传入的 `pos_avg_inv/scene_scale/scene_name` 作为元数据保留。Gaussian scale 为 log-scale，opacity 为 logits，rotation 未归一化；激活由后续渲染器负责。PLY 导出保持同一约定。`inference_time` 仅为主机提交耗时，准确耗时以 CLI 同步报告为准。

现有权重含完整 DINOv3 参数，严格匹配 645 个模型条目。其 Gaussian 头为 14 通道，必须使用 `sh_degree=0`；原训练配置的 `sh_degree=1` 不能直接加载这份权重。DINO 架构使用 Transformers 4.57.1 的 ViT-L/16 配置，1024 hidden、4096 MLP、24 层、16 heads、4 register tokens，RoPE 与 LayerNorm 使用该版本官方默认值。

保持原仓实际未 resize 的 DINO 预处理、ImageNet 归一化、白背景合成、Gaussian 后处理及 E/ν 单位转换。DINO 在 `model.train()` 后仍冻结并保持 eval。修复了原 forward 提前读取目标相机、Gaussian split view 原地写入和随机 pruning 索引的问题。

CPU 工作包括图像读取/缩放、相机规范化、权重读取、输出文件，以及固定位置编码的生成缓存：射线编码使用原有 bicubic＋antialias；DPT 位置编码保留原 FP64 频率计算后转 FP32。二者都没有可训练依赖，随后缓存到实际执行设备；神经网络算子不因此切换到 CPU。

当前验证覆盖可见的 910B3。910C 实机、同批 GPU 网络数值对齐、静态渲染和完整预训练分别需要后续验收。
