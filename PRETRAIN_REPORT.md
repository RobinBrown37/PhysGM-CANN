# M3 监督预训练迁移验收报告

日期：2026-10-24。环境：Ascend910B3 64 GB、CANN 9.1.0、Python 3.11、PyTorch 2.7.1+cpu、TorchNPU 2.7.1.post8。完整实施约定见 [PRETRAIN_PLAN.md](PRETRAIN_PLAN.md)，使用方法见 [training/README.md](training/README.md)。

## 交付

真实多视图/物理 Dataset、官方小子集准备、RGB/alpha/BCE/NLL+MSE 联合损失、离线 VGG LPIPS、单卡/HCCL DDP、AdamW/AMP/裁剪/梯度累积、确定性采样与完整 checkpoint、独立验证、图像导出、JSONL/TensorBoard 和 uv train 依赖组已实现。

**真实联合监督、连续100步短训、原分辨率更新和完整保存恢复已通过。CPU/NPU共53项测试，0失败、0跳过。** 原始checkpoint未覆盖；正式训练产物位于 `results/pretrain_final/`。

## 实测验收

| 项目 | 结果 |
| --- | --- |
| CPU与NPU回归 | 53 passed，含原33项及新增20项训练测试 |
| 256真实联合监督 | 五组Gaussian、共享网络和三个物理头梯度有限且组总量非零；各模块观察到真实参数变化 |
| 连续短训 | 从原始权重开始，BF16、256×256、4输入/8目标视图，完成100次更新、100个micro-step |
| 短训初20步/末20步平均联合loss | 0.065987 / 0.039150 |
| 100步NPU峰值分配 | 7.906 GiB，完整运行约309.68秒（含加载、验证、保存与导出） |
| 固定真实batch20步 | 首5步0.042143 → 末5步0.020771，下降50.71% |
| 512原分辨率 | 4输入/8目标视图、累积8，完成1次真实AdamW更新；峰值分配14.214 GiB |
| 从头初始化 | 严格加载415个DINO条目，其余随机；SH1，2次真实更新；Transformer/renderer重计算开启，通过 |
| 可选LPIPS | 官方VGG16/LPIPS离线加载；真实联合损失更新和CPU/NPU图像梯度对照通过 |
| FP16 | 最终默认确定性设置下，NPU GradScaler真实更新通过；非有限损失/梯度整窗跳过与恢复通过 |
| 分布式 | Gloo双进程等效batch与异常一致退出通过；HCCL单rank累积2次更新通过 |
| 保存/恢复 | 连续4步 vs 先2步后恢复2步，实际loss、全部梯度和可训练参数逐元素一致，误差为0 |
| RNG/数据/调度 | 每rank RNG、数据位置、scheduler和scaler一致 |
| 恢复权重闭环 | M1严格加载645项、真实推理及M2静态渲染通过 |
| 独立eval | 最终100步checkpoint验证通过，无backward/参数更新 |
| DINO | 所有训练验收均保持冻结/eval，训练前后SHA256一致 |

连续/恢复对照预先固定loss rtol=1e-3/atol=1e-5、组梯度/更新相对L2误差1%；实际为逐元素一致，没有放宽容差。默认非确定性配置曾出现约1–2%的组梯度差异，因此正式配置启用确定性算法。

100步中每20步验证，RGB PSNR从第20步22.148 dB到第100步23.013 dB；第100步总验证loss为0.085562，材料准确率0.5。只有4个验证场景，联合验证loss并非单调下降，不据此声称全量泛化或收敛。

各项运行报告分别保存同步forward/backward/optimizer计时，排除数据读取/传输和checkpoint I/O；总耗时包含这些工作。部分验收进程并行执行，所列时间不作为隔离性能对比基准。

## 数据与训练约定

官方 PhysAssets 固定 revision `a22b14b6b324f51b0472bc9756c530a8dd17c9ef`，读取首分片 52 MiB，提取20个真实完整场景，16训练/4验证，每场景38个相机视图。图像、原始 material/E/nu 标签均来自官方数据，没有生成或用模型预测代替标签。源范围与逐文件 SHA256 见数据清单；不声称部分压缩包通过整包 checksum。

DINO 始终冻结/eval；其余默认训练334,816,530个参数。完整模型仍为645个条目；从头模式只加载415个DINO条目，并使用原训练SH1配置初始化其余网络。

本轮执行真实联合监督与优化器更新，梯度审计仅作观察，不加入优化目标。100步是功能短训，不是全量收敛或论文指标复现。

完整checkpoint实际包含645个模型条目、229个可训练参数的AdamW状态。已在未注册NPU的CPU进程中，不传map_location、使用weights_only加载，并检查所有Tensor均在CPU且不带梯度。

## 已修复的实际问题

- 原训练入口没有完整恢复 optimizer/scheduler/counter；迁移版保存并核验数学配置、参数名分组、world size、数据清单、每rank RNG和已消费数据位置。
- 修复累积同步范围、额外一步、学习率阶段伪切换和非有限梯度被nan_to_num掩盖的问题；仅成功更新推进有效步数和scheduler。
- 容器 `/dev/shm` 只有64MiB，按真实批Tensor大小限制worker/预取；256实测1worker/prefetch1，512为主进程读图。CPU只承担数据/相机/常量/元数据与导出，网络和训练主计算在NPU。
- 某些近对称随机输入视图使原FP32 Gram-Schmidt的相机正交误差达到0.023。CPU FP64共同规范化后转FP32，保留算法并消除不稳定性；专门回归该真实样本。
- TorchNPU默认HF32卷积改变VGG max-pool分支及图像梯度。LPIPS使用真实FP32，保留原测试容差；冻结/eval及输入梯度验收通过。
- 在首次AMP操作前发布项目算子包路径，并为GE host/proto/tiling显式链接libgraph；完整重建通过，解决独立AMP先使用时的包发现与启动符号问题。
- 默认启用确定性算法，恢复指纹记录该配置；连续/恢复数值容差在执行前固定，不通过放宽阈值处理偏差。

## 产物与复现

- [完整验收摘要](reports/pretrain_verification.json)、[CPU/NPU回归XML](build/pretrain_regression.xml)。
- [100步完整运行报告](reports/pretrain_short100.json)、[512累积报告](reports/pretrain_scale512.json)、[固定batch20步](reports/pretrain_overfit20.json)。
- [从头初始化](reports/pretrain_scratch2.json)、[LPIPS](reports/pretrain_lpips1.json)、[FP16](reports/pretrain_fp16.json)、[HCCL](reports/pretrain_hccl.json)、[独立验证](reports/pretrain_evaluation.json)。
- [严格恢复对照](reports/pretrain_resume.json)、[真实数据来源与SHA256](reports/pretrain_data.json)、[LPIPS资源校验](reports/pretrain_lpips_resources.json)、[最终构建信息](reports/pretrain_build.json)。
- 正式训练：`results/pretrain_final/last.pt`、`best.pt`、最近两份周期断点、TensorBoard、[曲线](results/pretrain_final/loss_curve.svg)和[目标/预测图](results/pretrain_final/vis_000100/comparison.png)。
- 连续/恢复原始梯度和更新前参数：`results/pretrain_resume_deterministic/`；恢复后的M1/M2输出在其 `restored_inference/`、`restored_render/`。

完整矩阵复验：`bash scripts/verify_pretrain.sh`。仅恢复闭环：`bash scripts/verify_pretrain_resume.sh results/my_resume_check`。所有命令和三种初始化方式详见 [训练说明](training/README.md)。

![第100步：上排真实目标，下排预测](results/pretrain_final/vis_000100/comparison.png)

## 尚待独立验收

本机只暴露一个逻辑NPU，HCCL单rank及CPU Gloo双进程通过不等于真实双NPU验收。真实双卡、910C实机与同批GPU gsplat/训练单步数值对齐仍需相应设备和参考包。
