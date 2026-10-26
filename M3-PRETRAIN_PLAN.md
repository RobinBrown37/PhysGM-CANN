# M3 完整监督预训练迁移实施计划

日期：2026-10-24。基于已完成的 M1 网络和 M2 静态可微渲染。交付完整单卡/HCCL DDP 训练、真实联合监督更新、100 步短训、从头初始化、LPIPS 和保存恢复验收，不执行十万步全量收敛训练。

## 数据与损失

- 固定 PhysAssets revision `a22b14b6b324f51b0472bc9756c530a8dd17c9ef`，HTTP Range 顺序解压归档，提取前 20 个完整合法场景，16 训练/4 验证，最多读取 256 MiB；保存原始物理标签、清单与逐文件 SHA256，不声称部分归档完成整包校验。
- Dataset 保持原相机共同规范化和场景缩放，4 输入/8 目标视图、目标包含输入；确定性样本采样，随机裁剪同步焦距和主点；缺失数据明确报错，不生成伪标签。
- E/nu 按原训练统计归一化，不应用推理 E×0.1。默认 RGB MSE 1、alpha MSE 0.1、one-hot BCE 材料 0.1、E/nu 各 NLL+MSE 0.01；NLL 方差下限 0.01、逐元素上限 100。FP32 损失。
- 可选 VGG LPIPS 默认 0，公式对齐 kiui 0.2.18，独立冻结/eval，权重通过准备命令获取，训练运行仅本地加载。

## 执行与接口

- 独立 train_pretrain.py；DDP 包装网络+renderer+loss 的训练封装。M1 predict/forward 保持独立，checkpoint model 键名不变。
- 三种互斥启动：init-checkpoint 严格全权重初始化；dinov3-checkpoint 仅加载 415 个 DINO 条目并按原规则初始化其余层；resume 恢复迁移版完整状态。现有权重 SH0/-7.5/-1.5，从头配置 SH1/-5/-0.2。
- 正式配置：512、4/8 视图、每卡 batch1/累积8、AdamW lr2e-4、betas0.9/0.95、wd0.05、warmup5000、cosine、clip30、100000 有效更新。BF16 网络、FP32 渲染/损失，FP16 使用 NPU scaler。默认关闭 foreach/fused 和非重入 activation checkpointing。
- 累积 no_sync 包含 forward/backward；成功更新才推进 scheduler/global_step。BF16/FP32 非有限立即全 rank 失败，FP16 溢出全 rank 一致跳过整个窗口。禁用监督的独立头提前冻结。
- 完整恢复：模型/优化器参数名分组/scheduler/scaler、计数、每 rank RNG 和数据位置、配置与清单指纹。仅累积窗口结束保存，CPU 序列化并原子替换，保留 latest/best/最近两个周期。原 checkpoint 只有六参数优化器，不能作为全网络 resume。
- 日志 JSONL/TensorBoard、固定验证、时间/内存/梯度、对照图和独立 results 子目录；uv train 组、脚本与 M3 文档。
- 实施补充：默认开启确定性算法并写入恢复指纹；FP32卷积禁用HF32；CPU FP64共同相机规范化与共享内存预检用于处理真实数据/容器条件。NPU主计算路径不转CPU。

## 验收顺序与固定标准

1. CPU：真实格式数据、归一化/裁剪、原损失及梯度、累积/调度边界/异常窗口、完整恢复与启动模式；确定性恢复逐 Tensor 严格一致。
2. 256 实际联合监督更新：原权重、真实 4/8 视图，五组 Gaussian、共享层与三个物理头有限非零梯度，各组实际参数更新，DINO 不变。
3. 512：4/8 视图、完整累积8和 AdamW 更新，记录状态建立后的峰值内存和分项时间。
4. 100 步：256/batch1/累积1/lr1e-5/warmup5，每20步验证；固定真实 batch20步，末5步平均 loss < 首5步；从头模式至少2次更新；LPIPS 独立真实前反向。
5. 恢复：连续与中断训练的采样/计数/LR/RNG 一致；NPU loss rtol=1e-3/atol=1e-5，参数更新向量相对 L2 误差 <=1%，组梯度相对 L2 误差 <=1%（零参考使用绝对 L2 <=1e-7）；先固定容差再运行。恢复 checkpoint 经 M1/M2 推理渲染复验。
6. CPU Gloo 双进程控制测试、torchrun 单进程 HCCL；双卡可用才验收真实双卡，当前只有单个910B3则列待验收。保留现有33项回归。GPU参考单步对齐另行验收。
