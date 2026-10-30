# PhysGM M1–M5 NPU 基准报告（2026-10-30）

## 结论

单卡 Ascend 910B3、CANN 9.1.0、PyTorch 2.7.1、TorchNPU 2.7.1.post8 上的 NPU 部分全部通过。最终回归为 121 passed、0 failed、0 skipped。官方 checkpoint 严格加载 645 个模型条目，SHA256 为 `b4cdf3418d95f497ea2d7ef30cf8783849c9791142304749d7bdf16af5b80b6e`。

完整 PhysAssets 分卷流通过官方 SHA256 校验。从 21,273 个模型支持标签场景中以种子 `20261030` 按材料分层抽取 512 个场景，保留每场景 4 个输入视图与 4 个 held-out 视图；448 个用于训练、64 个用于验证。子集 manifest SHA256 为 `411cf19146c9417a42c86893bc72b4ac831b61037a08acfc08fd94392a037f31`。

## 实测结果

| 阶段 | 配置 | NPU 结果 |
| --- | --- | --- |
| M1 | 512 场景、4×512² 输入、BF16 | 中位 0.1520 s，p95 0.1780 s |
| M1 backward | cake、4×256²、梯度探针 | 0.3529 s，全部可训练组有限且非零，DINO 冻结 |
| M2 | 512 场景、8×512² 目标、FP32 renderer | 中位 0.2879 s，p95 0.6096 s |
| M2 backward | cake、4×256² | 0.1568 s，五组 Gaussian 梯度通过 |
| M2 图像质量 | 512 场景、8 视图 | 平均 PSNR 25.519 dB，RGB MSE 0.003999，alpha MSE 0.007583 |
| 物理预测 | 512 场景 | 材料准确率 60.94%，归一化 E MAE 0.9547，ν MAE 0.4529 |
| M3 | 256²、100 次有效更新、batch1/累积1 | 核心更新中位 0.7114 s；0 跳步；峰值分配 8.526 GB |
| M3 原规模 | 512²、batch1/累积8、1 次更新 | forward 5.3785 s，backward 4.5145 s，step 0.1417 s；峰值分配 13.706 GB |
| M4/M5 | cake、213,532 Gaussian、219,416 粒子、10,000 子步、50 帧 | 仿真 384.02 s，动态渲染合计 4.25 s，峰值分配 1.439 GB |
| M5 replay | 50 帧固定相机 / 运动相机 | 7.65 / 7.38 s；在线/离线最大绝对误差 0 |

M1/M2 核心时间均排除 checkpoint 加载、首轮编译、图像解码、主机传输和导出。M3 的 forward/backward/optimizer 为每次有效更新内同步计时；总运行时间另包含数据、末尾 64 场景验证、可视化和 checkpoint 保存。M4/M5 使用 `upstream` 物理模式，无粒子主机回退。

## 复现与 GPU 对比输入

数据准备入口为 `scripts/prepare_physassets_benchmark.py`，直接串联读取七个 gzip/tar 分片，不生成 72 GB 合并文件；已生成子集位于 `/home/developer/datasets/physassets_benchmark_512`。M1/M2 批量入口为 `benchmark_inference.py`，完整 NPU 汇总由 `scripts/summarize_npu_benchmark.py` 生成。

项目内结果入口为 `results/npu_benchmark_20261030/`，它链接到 `/home/developer/datasets/physassets_npu_benchmark_20261030`，避免占用已满的 workspace 分区。GPU 对比需要复用以下不可变输入：

- `physassets_benchmark_512/manifest.json` 和 `all/train/validation.txt`；
- `m1_m2_dataset_512/per_scene.json`；
- `m1_m2_dataset_512/reference_samples.pt`；
- `m1_m2_dataset_512/representative_*.pt`；
- `m5_dynamic_50/sequence/`。

机器可读总报告见 `results/npu_benchmark_20261030/summary.json`，最终 JUnit 见 `results/npu_benchmark_20261030/regression_final.xml`。GPU 端尚未执行，因此本报告不声明 GPU/NPU 速度比或跨设备数值误差。

M4/M5 没有对 512 个场景逐一执行完整 10,000 子步；那将产生约 55 小时以上的单卡仿真。当前采用一条完整 cake 全规模流程，加上覆盖材料、本构、边界、转移与动态相机的 NPU 回归测试，GPU 端应使用同一 cake 缓存与配置进行模块对齐。
