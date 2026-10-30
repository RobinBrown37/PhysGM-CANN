# M1 迁移验证报告

日期：2026-10-23。基线：PhysGM `7722175ba55db54fd430a7d29a4407b8f8ffe6b6`。

**纯网络推理和反向探针已在 Ascend910B3（64 GB）通过。910C 尚未实测。** 原仓未修改；现有 checkpoint 未更新，未执行优化器或监督训练。

## 环境与权重

- 独立 uv `.venv`；Python 3.11.4 / aarch64。
- PyTorch 2.7.1+cpu、TorchNPU 2.7.1.post8、Transformers 4.57.1、CANN 9.1.0。
- `checkpoint.pt` 严格加载 645 个模型条目，包含完整 DINOv3。checkpoint 中原有步数 456000 仅作为来源信息，本次没有训练步数。
- Gaussian 头 14 通道，对应 SH degree 0。配置不匹配、缺失或额外权重均报错。
- 设备最初在受限沙箱内不可见；沙箱外 `npu-smi`、TorchNPU 张量运算和后续网络测试均成功。

## 实测

| 验证项 | 结果 |
| --- | --- |
| 完整回归 | 17 passed，包含真实 NPU Attention 对照 |
| CPU 真实权重烟测 | 四视图 64×64、FP32，通过 |
| NPU 真实权重推理 | 四视图 512×512、BF16，通过 |
| 过滤前 Gaussian 数 | 1,048,576 |
| PLY 导出点数 | 237,542，opacity 阈值 0.001 |
| 512 首次前向 | 1.7750 秒 |
| 512 预热后前向中位数 / p95 | 0.1465 / 0.1465 秒 |
| 512 峰值 NPU 分配 / 预留 | 5.118 / 6.184 GiB |
| NPU 反向探针 | 四视图 256×256、BF16，通过 |
| 256 可训练前向中位数 | 0.1325 秒 |
| 256 单次反向 | 0.2730 秒 |
| 256 峰值 NPU 分配 / 预留 | 4.595 / 5.006 GiB |

前向计时为 1 次预热后 3 次同步运行，I/O、模型加载和输入传输单独记录；反向仅计一次，没有优化器更新。profiling 另行重放，不计入上述时间或峰值内存。当前指标只描述本机、单个 cake batch 和本配置。

## 计算图与算子证据

- 射线编码 8、投影 1、输入归一化 1、Transformer 144、DPT 62、材料头 2、E 头 5、ν 头 5 个参数张量均有梯度；各组全部有限且梯度绝对值总和非零。全局 token 有有限且非零的梯度。
- DINO 全部参数冻结、没有梯度，调用 `model.train()` 后仍保持 eval。
- trace 包含 `aten::scaled_dot_product_attention`、`npu::npu_fusion_attention_v3`、`npu::npu_fusion_attention_grad_v3`，以及实际 `aclnnFlashAttentionScore` / `aclnnFlashAttentionScoreGrad` 内核。
- 小规模 Attention 的前向、反向和自定义 scale 与显式 FP32 公式作 BF16 容差对照；CPU 还验证 FP64 前向/梯度和 train/eval dropout。
- 网络运行没有目标视图输入，不导入 gsplat、xFormers、Warp 或 kiui/LPIPS。

## 数据与输出一致性

cake 图像、内参、c2w、相机规范化矩阵和场景缩放与原仓逐元素完全一致。DINO 保留实际未执行 resize 的行为，RGBA 按白背景合成。

Gaussian 保留 log-scale、opacity logits、未归一化 rotation 以及原 SH/RGB residual 约定；PLY 激活阈值仅用于导出筛选。512 示例的物理输出为材料 Plasticine、E=475482.59、ν=0.476；这是网络预测值，没有标签精度评估。

预测包仅含 CPU Tensor 与普通类型，已在不指定 map_location 的情况下重新加载验证。

CPU 执行的工作已明确记录：图像/相机预处理、权重读取和导出；射线 bicubic＋antialias 常量与 DPT FP64 位置常量在 CPU 生成，再缓存到执行设备。这些常量没有可训练依赖。Attention 的实际 NPU 前反向路径已由 trace 确认。

## 复现与产物

使用命令见 [迁移说明](M1-migration_readme.md)，一键验收为 `bash scripts/verify_npu.sh`，运行前加载 CANN 环境。

- [CPU 烟测报告](reports/cpu_smoke.json)
- [NPU 512 推理报告](reports/npu_forward.json)
- [NPU 256 反向与算子报告](reports/npu_backward.json)
- [回归与一致性检查记录](reports/verification.json)
- 完整预测包、PLY、物理 JSON：`results/npu_forward/`。
- 反向探针与 Chrome trace：`results/npu_backward/`。

## 后续验收边界

910C 实机适配、同批 GPU 网络数值对齐尚未验证。原始 GPU 参考包到位后，应比较过滤前 Gaussian、物理 logits、均值/方差及单位转换结果，再根据 dtype 与 GPU 重复运行冻结阈值。

当前梯度标量是人工构造的计算图探针，不是监督预训练损失。本阶段没有实现静态可微渲染、训练标签损失、参数更新、训练恢复、MPM 或动态渲染。
