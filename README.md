# PhysGM-CANN

PhysGM 的昇腾迁移版：  
使用四张 RGBA 图像预测三维 Gaussian 与材料参数，经过 MPM 前向仿真和动态 Gaussian 渲染输出视频；  
同时提供真实 PhysAssets 标签的监督预训练。  
代码基于原仓 [PhysGM](https://github.com/Hihixiaolv/PhysGM) 的 `7722175ba55db54fd430a7d29a4407b8f8ffe6b6` 版本。  
PhysGM 原作者为 Chunji Lv、Zequn Chen、Donglin Di、Weinan Zhang、Hao Li、Wei Chen、Yinjie Lei、Changsheng Li；  
论文见 [arXiv:2508.13911](https://arxiv.org/abs/2508.13911)。  
本仓的 CANN 算子与适配代码是迁移实现，详见[迁移说明](docs/migration.md)。  

当前支持 M1 网络预测、M2 静态可微渲染、M3 监督预训练、M4 MPM 前向仿真和 M5 动态渲染及序列回放。M4/M5 只支持前向推理。  
单卡 Ascend 910B3 上已完成完整 cake 演示；GPU/NPU 对照与范围见[测试结果](docs/benchmark.md)。  


## 环境和安装

已测试环境：Python 3.11、CANN 9.1.0、PyTorch 2.7.1 CPU 构建、TorchNPU 2.7.1.post8、Ascend 910B3。`uv.lock` 锁定 Python 依赖；CANN 工具链及 NPU 驱动需由运行机器提供。命令均从仓库根目录执行。

```bash
source /path/to/Ascend/cann-9.1.0/set_env.sh
uv sync --locked --group train --group mpm --group render-build --group dev
.venv/bin/python scripts/build_renderer.py --jobs 8
.venv/bin/python scripts/build_mpm.py
.venv/bin/python -c 'import torch, torch_npu; print(torch.npu.is_available())'
```

最后一行应输出 `True`。  
原生算子构建结果在忽略的 `build/` 中，新机器或清理构建目录后需重新构建。只做 M1 网络推理时可用 `uv sync --locked`，无需构建渲染或 MPM 算子。模型和渲染默认使用 NPU；`--device cpu` 仅适用于明确支持的调试路径，小规模 CPU renderer 应选 `--backend reference`。

从[原项目权重发布页](https://drive.google.com/file/d/1uWrsUTKQO1rHLX7z3lAyYQp3IC4mdgi3/view?usp=sharing)取得官方 checkpoint，放在仓库根目录 `checkpoint.pt`，或用各入口的权重参数指定位置。当前权重的 SHA256 为 `b4cdf3418d95f497ea2d7ef30cf8783849c9791142304749d7bdf16af5b80b6e`；默认网络配置 `sh_degree=0`，严格加载全部 645 个模型条目。

## 输入数据

仓库自带 `example_data/cake/` 四视图示例，`configs/network_npu.yaml` 默认使用 `example_data/data.txt`。自有推理数据沿用原仓格式：文本清单每行一个 `pose.json` 路径，场景中按 `pose.json` 记录提供 RGBA 图像与相机。可通过修改配置中的 `data.data_path` 指向自己的清单。

训练场景还需同目录 `physical.json`，含 `material`、`E`、`nu`；数据清单拆分为训练和验证两份。PhysAssets 可从[官方数据页](https://huggingface.co/datasets/yaya234/PhysAssets)取得。`configs/pretrain_npu.yaml` 的 `data.data_path` 和 `data.validation_path` 默认指向可选的 20 场景小子集；使用完整 PhysAssets 时先按[训练数据说明](training/README.md)准备清单，再在配置中改为实际路径。随机视图采样与训练恢复均由 M3 入口控制。

## 推理与部署

与原仓相似，`run.sh` 调用完整 `pipeline.py`；默认 cake、四张 512×512 输入、50 帧、每帧 200 个 MPM 子步、配置相机、白背景。完整运行会执行 10,000 个物理子步。输出含预测包、物理 JSON、可恢复仿真状态、逐帧缓存、PNG 和 MP4。

```bash
bash run.sh --checkpoint checkpoint.pt --output-dir results/my_cake
# 等价：.venv/bin/python pipeline.py --checkpoint checkpoint.pt --output-dir results/my_cake
```

需要单独运行或复用中间结果时：

```bash
# M1：预测包 prediction.pt、Gaussian PLY、物理参数 JSON
.venv/bin/python infer_network.py --checkpoint checkpoint.pt --output-dir results/my_network

# M2：对预测包进行静态渲染；可用 --online 从网络直接开始
.venv/bin/python render_static.py --input-pack results/my_network/prediction.pt --output-dir results/my_static

# M4/M5：从预测包仿真，默认启用动态渲染与视频输出
.venv/bin/python simulate_mpm.py --prediction-pack results/my_network/prediction.pt \
  --config configs/mpm/cake.json --output-dir results/my_simulation --save-render-frames

# M5：直接读取逐帧缓存换相机回放，不再运行网络和 MPM
.venv/bin/python render_dynamic.py --sequence results/my_simulation/sequence \
  --camera-mode config --move-camera --output-dir results/my_replay
```

`pipeline.py` 支持 `--camera-mode input|config`、`--cameras-json`、`--frames`、`--width/--height`、`--physics-mode upstream|corrected` 等参数。`upstream` 是与原仓实际仿真行为对应的默认模式；`corrected` 仅启用已记录的碰撞和顺序释放修正。缓存允许更换相机后独立重渲染。逐帧缓存、仿真状态和 checkpoint 可能较大，请使用独立输出目录。

## 监督预训练

与原仓 `main.py` / `start.sh` 用法对应，迁移版的实际训练逻辑位于 `train_pretrain.py`，支持完整权重初始化、仅 DINO 权重的从头训练和断点恢复：

```bash
# 先将训练与验证清单写入配置；示例为 100 次有效更新的功能短训
bash start.sh --config configs/pretrain_smoke.yaml --init-checkpoint checkpoint.pt \
  --output-dir results/my_pretrain_smoke

# 正式配置：512²、4 输入/8 目标、累积 8、100000 次有效更新
.venv/bin/python main.py --config configs/pretrain_npu.yaml \
  --init-checkpoint checkpoint.pt --output-dir results/my_pretrain

# 从头训练：只加载 checkpoint 中冻结的 DINO 权重
.venv/bin/python main.py --config configs/pretrain_scratch_npu.yaml \
  --dinov3-checkpoint checkpoint.pt --output-dir results/my_scratch

# 相同配置、相同输出目录恢复优化器、调度器、随机状态和数据位置
.venv/bin/python main.py --config configs/pretrain_npu.yaml \
  --resume auto --output-dir results/my_pretrain
```

`train_steps` 表示成功的参数更新次数。训练由 RGB、alpha、材料、E 与 ν 标签共同监督，DINO 始终冻结；网络默认 BF16，renderer 与损失 FP32。`--eval-only` 只运行验证。多 NPU 启动：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 main.py \
  --config configs/pretrain_npu.yaml --init-checkpoint checkpoint.pt \
  --output-dir results/my_ddp
```

训练数据、LPIPS、断点格式与配置覆盖详见[训练说明](training/README.md)。

## 测试与基准

```bash
.venv/bin/python -m pytest -q
PHYSGM_TEST_NPU=1 .venv/bin/python -m pytest -q

# 固定 PhysAssets 场景清单的 M1/M2 推理与 GT 指标；需预先准备数据
.venv/bin/python benchmark_inference.py \
  --data-list /path/to/physassets_benchmark_512/all.txt \
  --manifest /path/to/physassets_benchmark_512/manifest.json \
  --checkpoint checkpoint.pt --output-dir results/my_benchmark
```

CPU 测试与 NPU 实测须分别记录；无 NPU 环境不会获得 NPU 验收。社区对照使用的 512 场景抽样、训练设置、GPU/NPU 速度和 GT 精度见[测试结果](docs/benchmark.md)。旧版逐元素对齐门槛没有通过，不能将相近的 GT 指标写成网络中间张量等价。

## 文档与许可

- [迁移说明](docs/migration.md)：M1–M5 的代码路径、原仓行为和自编算子。
- [测试结果](docs/benchmark.md)：固定输入、逐模块测试、速度与精度、未验收范围。
- [渲染接口](rendering/README.md)和[训练接口](training/README.md)。
- 自带的 CANN `gauss-splat` 源码许可证、来源和本地改动见 [UPSTREAM.md](third_party/gauss_splat/UPSTREAM.md) 及其 `LICENSE`。原 PhysGM 论文和源代码归原作者所有；使用数据与权重时遵守各自发布条款。

原论文引用：

```bibtex
@misc{lv2025physgmlargephysicalgaussian,
  title={PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis},
  author={Chunji Lv and Zequn Chen and Donglin Di and Weinan Zhang and Hao Li and Wei Chen and Yinjie Lei and Changsheng Li},
  year={2025},
  eprint={2508.13911},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
