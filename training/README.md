# M3 监督预训练迁移

训练使用真实 RGB/alpha、材料、E/ν 标签，连接 M1、M2、联合损失和 AdamW；入口为 `train_pretrain.py`。预训练不需要仿真，M1 的 `predict/forward` 仍只执行预测。总览见 [M1-migration_readme.md](../M1-migration_readme.md)，实施约定见 [M3-PRETRAIN_PLAN.md](../M3-PRETRAIN_PLAN.md)。

## 环境与真实数据

```bash
cd /mnt/workspace/PhysGM-CANN
export PATH="/mnt/workspace/.tools/bin:$PATH"
source /home/developer/Ascend/cann-9.1.0/set_env.sh
uv sync --locked --group train --group render-build
uv run --locked --group render-build python scripts/build_renderer.py --jobs 8
uv run --locked --group train python scripts/prepare_physassets.py
```

数据固定在官方 PhysAssets revision `a22b14b6b324f51b0472bc9756c530a8dd17c9ef`。准备脚本只顺序读取有界归档前缀，默认最多 256 MiB，提取 20 个完整合法场景，16 训练/4 验证；本次实际读取 52 MiB。清单保存源字节范围、每文件 SHA256、原始标签和场景划分；没有声称完成整个压缩包的 checksum。再次运行会校验已有子集。

自有数据使用一行一个 `pose.json` 的文本清单，路径相对清单目录；相邻 `physical.json` 包含 `material`、`E`、`nu`。图像路径相对 pose.json，相机沿用原仓 w2c 和像素内参。默认每场景至少 8 帧；训练不重复抽取视图。标签缺失或错误直接报错。

E 标签为原始训练单位，归一化为 `(log10(max(E,1))-7.387210)/2.456477`，ν 为 `(nu-0.398)/0.111`。这里不应用推理导出 E×0.1。

## 三种启动方式

```bash
# 100 次真实联合监督更新，用于功能/短训验收
uv run --locked --group train python train_pretrain.py \
  --config configs/pretrain_smoke.yaml --init-checkpoint checkpoint.pt \
  --audit --output-dir results/my_pretrain_smoke

# 正式配置：512、4/8 视图、batch1 × 累积8、100000 次有效更新
uv run --locked --group train python train_pretrain.py \
  --config configs/pretrain_npu.yaml --init-checkpoint checkpoint.pt \
  --output-dir results/my_pretrain

# 从头预训练：仅加载原 checkpoint 的 DINO，其余按原规则初始化
uv run --locked --group train python train_pretrain.py \
  --config configs/pretrain_scratch_npu.yaml --dinov3-checkpoint checkpoint.pt \
  --output-dir results/my_scratch

# 完整恢复，配置与上一段训练保持一致
uv run --locked --group train python train_pretrain.py \
  --config configs/pretrain_smoke.yaml --resume auto \
  --output-dir results/my_pretrain_smoke

# 独立验证已有模型，不执行 backward/optimizer.step
uv run --locked --group train python train_pretrain.py \
  --config configs/pretrain_smoke.yaml \
  --init-checkpoint results/my_pretrain_smoke/last.pt --eval-only \
  --output-dir results/my_evaluation
```

三个初始化参数互斥；均未提供时默认加载项目根目录 checkpoint.pt 的全部模型权重。`--resume` 只接受迁移版完整状态；原 checkpoint 的优化器只含六个参数，不适用于新的 334,816,530 个可训练参数，使用 `--init-checkpoint` 进行权重初始化。

现有权重配置 SH0、scale bias -7.5、opacity bias -1.5；从头配置 SH1、-5.0、-0.2。从头只严格加载 415 个 DINO 条目，其余头可随配置构建；完整初始化/恢复严格检查全部网络形状和键。DINO 始终冻结/eval。

`--max-steps N` 是停止位置，不改变总调度长度，适合先运行到 N 再恢复。`--set training.lr=0.00001` 等覆盖使用 YAML 值，支持重复。`--amp-dtype bf16|fp16|fp32` 覆盖精度。配置 `_base_` 相对当前配置文件解析；运行命令以项目根目录为工作目录。

## 多卡、计算图与精度

```bash
uv run --locked --group train torchrun --standalone --nproc_per_node=2 \
  train_pretrain.py --config configs/pretrain_npu.yaml \
  --output-dir results/my_distributed_pretrain
```

torchrun 按 LOCAL_RANK 绑定逻辑 NPU，初始化 HCCL；CPU 控制回归使用 Gloo。DDP 包装整个 PretrainTask（网络、renderer、loss）。累积 no_sync 包含前向和反向；损失除以累积数，每次完整窗口仅更新一次。数据采样在 rank 间分片，采样键包含 epoch、draw、scene index，不依赖 worker 的随机调度。

网络 BF16、renderer/loss FP32；FP16 使用 TorchNPU GradScaler。训练显式关闭 FP32 卷积和矩阵乘 HF32，防止 VGG max-pool 分支改变输入梯度；BF16 算子仍按 BF16 执行。默认 `deterministic: true`，在首次运算前启用 PyTorch/TorchNPU 确定性算法并写入恢复指纹。AdamW 不启用 foreach/fused，动量在 NPU，非 capturable 的标量 step 计数留在 CPU。

默认损失为 RGB MSE 1、alpha MSE 0.1、14 类 one-hot BCE 0.1、E/ν 各 NLL+MSE 0.01。NLL 保留常数项、方差下限 0.01、逐元素上限 100。冻结被关闭监督的独立头；没有对未使用的头伪造零损失连接。

可通过 `training.checkpoint_transformer=true`、`training.checkpoint_renderer=true` 启用非重入重计算，默认均关闭。传给 renderer 的 Gaussian 保持原始 Tensor 计算图，没有 detach、CPU/PLY 重载。

```bash
# 可选 LPIPS：显式准备官方资源，然后离线运行
uv run --locked --group train python scripts/prepare_lpips.py
uv run --locked --group train python train_pretrain.py \
  --config configs/pretrain_smoke.yaml --set training.lpips_loss=0.1 \
  --output-dir results/my_lpips_training
```

LPIPS 默认 0，不导入 VGG、不加载其权重。启用后始终冻结/eval，仍对预测图像求梯度；数学形式对齐 kiui 0.2.18 的 VGG LPIPS v0.1，参考来源为 [kiuikit](https://github.com/ashawkey/kiuikit/blob/main/kiui/lpips.py) 和 [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)。准备脚本校验官方 VGG16/LPIPS 完整 SHA256，训练不自动联网。

BF16/FP32 非有限损失/梯度导致所有 rank 明确失败；FP16 overflow 全 rank 跳过整个窗口、降低 scale，不推进 scheduler/global_step。不用 nan_to_num 修改梯度。只有成功更新才推进有效 step，micro-step 另记；没有原版仅打印“解冻”而未执行解冻的阶段逻辑。

## 数据加载、恢复与产物

默认请求 4 workers/prefetch2；根据实际批 Tensor 大小和 `/dev/shm` 可用量，在创建 worker 前限制预取及 worker 数，不足时使用主进程 CPU 读图。实际设置写入报告。本机容器 64 MiB 共享内存，256 实测 1 worker/prefetch1，512 使用 0 worker。神经网络、投影、光栅化、反向和动量仍在 NPU。

CPU 还负责读图、相机共同规范化、固定位置常量、调度元数据与导出。近对称随机视图的平均坐标系在 CPU FP64 计算 Gram-Schmidt/共同变换后转 FP32，避免 FP32 消减误差产生非刚体相机；不修改原始图像或标签。

| 产物 | 内容 |
| --- | --- |
| config.yaml、data_manifest.json | 实际配置、来源、完整样本和数据指纹 |
| metrics.jsonl、tensorboard/、loss_curve.svg | 分项 loss、LR、梯度范数、验证、曲线 |
| checkpoint_*.pt、last.pt、best.pt | 完整状态；latest 指向周期 checkpoint，保留最近两份周期文件 |
| vis_*/comparison.png | 上排真实目标、下排预测，全部目标视图拼接 |
| vis_*/target_alpha.png、prediction_alpha.png | alpha 对照 |
| run_report.json | 设备、版本、计数、数据加载设置、更新/梯度审计、DINO 校验值与内存 |

完整 checkpoint 使用 CPU Tensor 和普通类型，可 weights_only 加载；保存仅发生在累积窗口结束，写临时文件后原子替换。恢复检查架构、精度、优化器参数名分组、数学配置、数据 SHA256 和 world size。每 rank 的 RNG、epoch 与已消费批次位置一同恢复，预取未消费批次不计入 cursor。更换 world size 或训练数学配置时，使用权重初始化开始新实验。

`--audit` 验证真实五组 Gaussian 梯度、全部启用参数的梯度和各模块的非零更新；初始 warmup 的零 LR 不算有效参数变化证据。DINO 在训练前后通过 SHA256 检查。`--capture-update` 保存最后一次真实梯度及更新前参数，供断点一致性对照，不是训练监督项。

## 复验

```bash
# 数据和构建已准备后的 CPU/NPU 全部回归
PHYSGM_TEST_NPU=1 uv run --locked --group train python -m pytest -q

# 独立的连续训练/恢复/推理/渲染闭环对照
bash scripts/verify_pretrain_resume.sh results/my_resume_check

# 完整矩阵：数据/LPIPS/构建/测试/100步/512/从头/恢复/HCCL
bash scripts/verify_pretrain.sh
```

完整矩阵保存多个全网络 AdamW checkpoint，需要为每份约 4.9 GiB 的状态预留空间。默认使用新时间戳目录，避免覆盖已有实验。当前一张 910B3 已验收单 rank HCCL；真实双卡和同批 GPU 参考对齐需另行验收。
