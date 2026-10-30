# PhysGM GPU 基准接力说明

本文档用于下一次 Codex 会话在 NVIDIA GPU 机器上复现与 NPU 完全同源的 M1–M5 测试。不要重新随机抽样数据，不要使用完整 67 GiB PhysAssets 分卷，也不要覆盖 NPU 参考结果。

## 1. 接力目标与固定路径

GPU 机器必须使用以下绝对路径。数据清单、训练配置和部分报告包含绝对路径，保持一致可以避免无意义的路径修补。

```text
/mnt/workspace/PhysGM                         原 GPU 仓，固定 commit 7722175ba55db54fd430a7d29a4407b8f8ffe6b6
/mnt/workspace/PhysGM-GPU-BENCH               从上述 commit 创建的独立 GPU benchmark worktree
/mnt/workspace/PhysGM-CANN                     当前迁移仓源码与说明
/mnt/workspace/PhysGM-CANN/checkpoint.pt       官方权重
/home/developer/datasets/physassets_benchmark_512
/home/developer/datasets/physassets_gpu_reference
/home/developer/datasets/physassets_gpu_results_YYYYMMDD
```

官方 checkpoint 必须严格匹配645个模型条目：

```text
SHA256 b4cdf3418d95f497ea2d7ef30cf8783849c9791142304749d7bdf16af5b80b6e
```

NPU 基准环境和结果见 `physassets_gpu_reference/npu/summary.json`。NPU 主测试为单卡 Ascend 910B3、CANN 9.1.0、PyTorch 2.7.1、TorchNPU 2.7.1.post8。

## 2. 必须迁移什么

### 2.1 完整迁移 512 场景测试集

`/home/developer/datasets/physassets_benchmark_512` 必须作为一个整体迁移到 GPU 机器的同一路径，不能只复制 `manifest.json` 或部分材料。当前目录约393 MiB、6,148个文件，包含：

- `manifest.json`；
- `all.txt`、`train.txt`、`validation.txt`；
- 512个场景目录；
- 每场景8张512×512 RGBA PNG；
- 每场景 `pose.json`、`physical.json`、`E_nu.json`、`phys.json`。

每个 `pose.json` 已按固定顺序缩减为8帧。位置 `[0,11,21,32]` 对应4个输入视图，位置 `[5,16,26,37]` 对应4个 held-out 视图；在缩减后的文件中就是前4帧输入、后4帧 held-out。场景目录是归档内部数值ID，而 `pose.json.scene_name` 是 Objaverse UUID，两者都必须保留。

完整归档和子集身份：

```text
完整归档 SHA256: bce802e3b7cbfdb1d9824b2d164d0ad911b428251ed2694a6bc2ad4971425328
子集 manifest SHA256: 411cf19146c9417a42c86893bc72b4ac831b61037a08acfc08fd94392a037f31
场景: 512 = 448 train + 64 validation
```

GPU 机器上验证：

```bash
cd /mnt/workspace/PhysGM-CANN
python3 scripts/prepare_physassets_benchmark.py \
  --output-dir /home/developer/datasets/physassets_benchmark_512 \
  --verify-existing
```

### 2.2 完整迁移紧凑 NPU 参考包

整体迁移 `/home/developer/datasets/physassets_gpu_reference` 到 GPU 同一路径。它只包含 GPU 对齐需要的 NPU 参考，不包含约47 GiB的全部 NPU checkpoint/中间结果。主要内容：

```text
npu/summary.json                         NPU总报告
npu/m1_m2/per_scene.json                 512场景逐场景速度与GT指标
npu/m1_m2/reference_samples.pt           M1 Gaussian/物理头及M2像素浮点采样
npu/m1_m2/cake_prediction.pt             固定Gaussian的M2隔离测试输入
npu/m3_100/metrics.jsonl                 100步损失、采样顺序、分项计时
npu/m3_100/training_reference.pt         最终模型/优化器紧凑采样
npu/m3_512/*                             512²累积8的一步参考
npu/m4_m5/initial_state.pt               固定MPM初始状态
npu/m4_m5/final_state.pt                 NPU 10000子步最终状态
npu/m4_m5/sequence/                      50帧动态Gaussian缓存
npu/m4_m5/render_tensors/                50帧在线FP32 RGB/alpha
npu/m4_m5/cameras.pt                     固定相机
manifest.json                            参考包逐文件SHA256
```

GPU 机器上验证：

```bash
cd /mnt/workspace/PhysGM-CANN
python3 scripts/prepare_gpu_handoff.py \
  --output-dir /home/developer/datasets/physassets_gpu_reference \
  --verify-existing
```

### 2.3 代码和 checkpoint

- 原 GPU 仓必须处于 commit `7722175ba55db54fd430a7d29a4407b8f8ffe6b6`。
- 当前 `PhysGM-CANN` 工作区包含尚未提交的 M5 与 benchmark 入口，不能只在 GPU 机器重新 clone 当前 HEAD；应复制当前工作树源码，或先在本机建立明确 commit/patch。
- 复制 `/mnt/workspace/PhysGM-CANN/checkpoint.pt` 到 GPU 同一路径并核对 SHA256。
- 建议从原仓建立独立 worktree，保持原仓不变：

```bash
git -C /mnt/workspace/PhysGM worktree add --detach \
  /mnt/workspace/PhysGM-GPU-BENCH \
  7722175ba55db54fd430a7d29a4407b8f8ffe6b6
```

### 2.4 不需要迁移

- `/home/developer/datasets/physassets/dataset.tar.gz.part_*` 七个完整分片；
- 完整解压后的21,330场景；
- `/home/developer/datasets/physassets_npu_benchmark_20261030` 中约47 GiB的全部训练 checkpoint；
- NPU `.venv`、`build/`、`export_only_prof_dir/`、CANN编译产物；
- 旧的20场景 `datasets/physassets_subset`，除非单独回归旧M3结果。

### 2.5 推荐迁移命令

以下命令在当前 NPU 机器执行，并把 `gpu-host` 替换为实际 GPU 主机。尾部目录层级会保持不变：

```bash
# 393 MiB，必须完整迁移到同一路径
rsync -aH --info=progress2 \
  /home/developer/datasets/physassets_benchmark_512/ \
  gpu-host:/home/developer/datasets/physassets_benchmark_512/

# 约1.1 GiB，精简NPU对齐参考，必须完整迁移到同一路径
rsync -aH --info=progress2 \
  /home/developer/datasets/physassets_gpu_reference/ \
  gpu-host:/home/developer/datasets/physassets_gpu_reference/

# 官方2.4 GiB checkpoint
rsync -aH --info=progress2 \
  /mnt/workspace/PhysGM-CANN/checkpoint.pt \
  gpu-host:/mnt/workspace/PhysGM-CANN/checkpoint.pt
```

源码推荐通过Git传输原仓commit；当前 `PhysGM-CANN` 有未提交文件，因此在建立正式commit/patch之前，应使用rsync复制源码但排除机器相关大文件：

```bash
rsync -aH --info=progress2 \
  --exclude='.venv/' --exclude='build/' --exclude='results/' \
  --exclude='export_only_prof_dir/' --exclude='checkpoint.pt' \
  /mnt/workspace/PhysGM-CANN/ \
  gpu-host:/mnt/workspace/PhysGM-CANN/
```

迁移完成后先执行两条 `--verify-existing`，不要以 `du` 大小相近代替逐文件SHA256验证。

## 3. GPU 环境

先记录硬件与软件，写入 GPU 结果目录的 `environment.json`：GPU型号/数量、显存、驱动、CUDA、cuDNN、Python、PyTorch、torchvision、xFormers、gsplat、Warp和Git commit。

原仓推荐环境是 Python 3.10、PyTorch 2.3.0＋CUDA 12.1：

```bash
conda create -n physgm-gpu python=3.10 -y
conda activate physgm-gpu
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r /mnt/workspace/PhysGM/requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install git+https://github.com/camenduru/simple-knn.git
```

不要安装 TorchNPU。主精度对比关闭 TF32：

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

可另做一组开启TF32的速度测试，但必须标为 `gpu_tf32_speed`，不能替代主精度结果。

## 4. 下一次 Codex 首先要实现的 GPU 基准入口

不要直接运行原仓 `pipeline.py/main.py` 就宣布完成比较。应在 `PhysGM-GPU-BENCH` 中新增独立入口，并让输出协议与 `physassets_gpu_reference/schema/benchmark_inference.py` 一致。

关键修正：

1. 原仓模型构造会先通过 Hugging Face 在线初始化 DINO，且部分入口 `strict=False`。GPU benchmark 必须从官方 checkpoint 离线构建并 `strict=True` 加载645项；缺失、额外或形状不匹配立即失败。
2. checkpoint 对应 `sh_degree=0`、scale bias `-7.5`、opacity bias `-1.5`。不要使用原训练配置的 `sh_degree=1` 加载该权重。
3. M1 网络预测和 M2 gsplat 渲染分别同步计时。不要把图像解码、CPU相机处理、H2D传输、checkpoint加载、首轮CUDA编译或结果导出计入核心时间。
4. 所有计时前后调用 `torch.cuda.synchronize()`；预热至少一个同形状场景。
5. 固定 seed `1111`，验证使用前4帧作为输入、全部8帧作为目标。白背景、RGBA合成、相机共同规范化和场景尺度必须与 NPU benchmark 一致。
6. 主网络使用BF16 autocast；gsplat、MPM和损失使用FP32。

建议实现：

```text
benchmark_gpu.py             512场景M1/M2及参考采样
compare_gpu_npu.py           读取reference_samples.pt并输出逐字段误差
train_gpu_benchmark.py       CUDA版M3同语义训练
benchmark_mpm_gpu.py         Warp M4状态对齐
render_dynamic_gpu.py        固定NPU序列的GPU M5回放
summarize_gpu_benchmark.py   GPU速度、GT指标、GPU/NPU差异总报告
```

## 5. 固定测试顺序

### 5.1 环境和最小门槛

1. 校验两份数据目录、checkpoint SHA和Git commit。
2. CUDA上严格加载645项。
3. 单场景512² M1前向；确认全部输出有限。
4. 用 `cake_prediction.pt` 运行GPU gsplat四视图，先隔离M2再连接M1。

任一步严格加载失败、出现在线模型下载、非有限值或CPU回退时先停止并修复，不进入正式计时。

### 5.2 M1/M2：512场景正式测试

- 输入：`physassets_benchmark_512/all.txt`。
- M1：每场景4个输入视图，BF16。
- M2：同一Gaussian对全部8个目标相机渲染，FP32、白背景。
- 输出：`per_scene.json`、`reference_samples.pt`、`run_report.json`。
- 报告核心时间的 mean/median/p95/min/max，同时单独报告解码与H2D。
- GT指标：RGB MSE/PSNR、alpha MSE、材料准确率、归一化 E/ν MAE。
- 与NPU参考逐字段比较：shape和索引必须完全一致；报告 max/mean/p50/p95/p99 absolute/relative error、allclose比例、材料类别一致率。

M2和M5图像主门槛沿用迁移验收值：`rtol=1e-3, atol=2e-4`。M1 BF16不同Attention内核可能产生累积差异，不得只给一个allclose布尔值；先完整报告各Gaussian字段和物理头误差。建议同时给出 `rtol=2e-2, atol=2e-3` 下的通过比例，任何系统性偏差必须定位到预处理、Attention或后处理，不能通过放宽阈值隐藏。

### 5.3 M3：相同步数训练

公平速度比较必须移植当前M3训练语义到CUDA，而不是直接使用原仓存在计数/累积差异的 `main.py`。可复用 `PhysGM-CANN/training/` 中纯PyTorch的数据、损失、scheduler和断点逻辑，将设备同步/AMP改为CUDA，renderer替换为GPU gsplat。

执行两组：

1. `pretrain_smoke.yaml`：256²、batch1、累积1、100次有效更新、LR `1e-5`、warmup5、cosine；448训练/64验证。
2. `pretrain_npu.yaml`：512²、batch1、累积8、warmup0、完成1次有效更新。

要求：

- 使用官方 checkpoint 权重初始化；DINO冻结并始终eval；
- 100步必须恰好100 optimizer steps/100 micro-steps，0额外更新；
- 512组必须1 optimizer step/8 micro-steps；
- 保存逐步样本key、loss、LR、grad norm、forward/backward/optimizer同步时间；
- 与 `npu/m3_100/metrics.jsonl` 比较采样顺序和学习率；
- 用 `npu/m3_100/training_reference.pt` 比较最终645个模型Tensor和229个optimizer state的固定采样；
- 报告参数采样的最大/平均误差以及验证指标差异，不要求跨后端逐元素bitwise一致；非有限、跳步、DINO变化或数据顺序不同均判失败。

原仓自身训练入口可另做“upstream observed”速度，但不能与上述同语义M3结果混在同一速度比中。

### 5.4 M4：GPU Warp MPM

先使用 `npu/m4_m5/initial_state.pt`，写适配器导入原Warp求解器，避免CPU填充和网络差异污染M4比较。物理模式固定 `upstream`：100³网格、`dt=2e-4`、200子步/帧、50帧、共10000子步。

顺序：1子步 → 100子步 → 保存恢复 → 10000子步。逐字段比较位置、速度、F、APIC矩阵、材料内部状态和质量。小场景/早期状态按M4验收使用：单步归一化误差不超过 `1e-4`，100子步不超过 `1e-3`。完整10000步另外报告误差增长，不用早期阈值伪装长期严格一致。

若原Warp状态格式无法直接接收该CPU pack，先实现显式字段映射并记录；不要重新随机填充后声称是同输入对比。

### 5.5 M5：动态渲染隔离与全链路

隔离测试直接读取 `npu/m4_m5/sequence/` 和 `cameras.pt`，不加载网络、不执行MPM；用GPU gsplat逐帧渲染，并与 `render_tensors/` 中50帧FP32 RGB/alpha比较。序列提供逐帧世界坐标位置、完整3×3协方差、时间、绝对帧号和Gaussian ID。若GPU gsplat接口不能直接接收协方差，使用其官方 covars 接口或实现数学等价适配；不要重新分解后改变符号/旋转约定却不记录。

隔离M5通过后，再执行GPU M1→Warp M4→GPU M5完整cake流程，生成固定相机和运动相机视频。核对50帧、512×512、25 FPS、首尾可解码和Gaussian ID。

## 6. GPU 最终报告必须包含

- 环境、源码commit、全部输入SHA256；
- 512场景总体和14材料分组的M1/M2速度与GT指标；
- M1字段、物理头、M2 RGB/alpha的GPU/NPU误差；
- 100步训练和512一步的同步核心时间、峰值显存、loss/验证差异、最终参数采样误差；
- M4 1/100/10000子步状态误差和速度；
- M5固定序列回放误差、在线全链路结果与视频；
- 明确区分首次编译、预热后核心时间、I/O、导出和总耗时；
- 任何未完成项必须标为 pending/blocked，不能用CPU结果或较小粒子数替代。

GPU输出统一写到新的 `/home/developer/datasets/physassets_gpu_results_YYYYMMDD`，不得写入或修改 `physassets_gpu_reference`。完成后生成一份机器可读 `summary.json` 和一份 Markdown 报告，再计算整个结果清单的SHA256。

## 7. 已知易错点

- 两视图不符合官方checkpoint的4输入协议。
- 8目标视图是本基准/M3训练协议，不是静态renderer的固定架构限制。
- 原仓 `dataset_infer.py` 硬编码 `[0,1,2,3]`，只覆盖输入视图；正式基准必须额外渲染后4个held-out视图。
- 原仓未知材料会静默映射为Wood；本子集已排除57个不支持标签，GPU端仍应严格拒绝未知标签。
- 原仓在线DINO、`strict=False`、sh_degree=1训练配置或不同resize都会破坏可比性。
- CUDA异步计时若缺少 synchronize 会显著低估耗时。
- 训练末尾验证和checkpoint I/O不应计入每步核心速度。
- NPU reference Tensor均保存为CPU Tensor，GPU可直接 `torch.load(..., map_location="cpu", weights_only=True)`。

## 8. 下一会话的第一条检查命令

```bash
set -e
test "$(git -C /mnt/workspace/PhysGM rev-parse HEAD)" = \
  7722175ba55db54fd430a7d29a4407b8f8ffe6b6
sha256sum /mnt/workspace/PhysGM-CANN/checkpoint.pt
python3 /mnt/workspace/PhysGM-CANN/scripts/prepare_physassets_benchmark.py \
  --output-dir /home/developer/datasets/physassets_benchmark_512 --verify-existing
python3 /mnt/workspace/PhysGM-CANN/scripts/prepare_gpu_handoff.py \
  --output-dir /home/developer/datasets/physassets_gpu_reference --verify-existing
nvidia-smi
```

四项校验全部通过后，才开始实现和运行GPU benchmark。
