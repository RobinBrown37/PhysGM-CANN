# PhysGM → TorchNPU / CANN 迁移说明

基线为原 PhysGM `7722175ba55db54fd430a7d29a4407b8f8ffe6b6`。用户安装、训练与部署从[README](../README.md)开始；本文记录迁移设计、原仓之外的代码和适用边界。迁移没有改动原仓工作树。Python 环境由 `pyproject.toml` 与 `uv.lock` 固定；NPU 运行需另装 CANN 9.1.0 和驱动。

| 阶段 | 原仓核心 | 本仓入口与实现 | 计算图 |
| --- | --- | --- | --- |
| M1 网络 | DINOv3、射线编码、Transformer、DPT、材料头 | `infer_network.py`，`model/`，`data/` | 网络可反向，DINO 冻结 |
| M2 静态 Gaussian 渲染 | CUDA gsplat | `render_static.py`，`rendering/static.py`，CANN `gauss-splat` | RGB/alpha 一阶反向 |
| M3 监督预训练 | 图像与物理联合损失 | `main.py` → `train_pretrain.py`，`training/` | 完整监督更新与恢复 |
| M4 MPM | Warp 仿真器 | `simulate_mpm.py`，`mpm/` | 前向 |
| M5 动态 Gaussian 渲染 | 仿真后逐帧渲染 | `pipeline.py`，`render_dynamic.py`，`rendering/dynamic.py` | 前向 |

## M1：网络与权重

保持四视图、512×512 的默认输入、RGBA 白底合成、相机共同规范化、Gaussian 原始参数语义和物理单位转换。DINOv3 ViT-L/16 按架构离线构造，从官方 checkpoint 严格加载 645 个模型条目，不在启动时下载权重。该 checkpoint 对应 `sh_degree=0`、scale bias `-7.5`、opacity bias `-1.5`。DINO 始终冻结且保持 eval。

原 xFormers Attention 改为 PyTorch SDPA，保留 QKV 排列、缩放与 dropout 语义；设备由 `runtime.py` 显式选择，NPU 默认 BF16 autocast。`model.predict(input_dict)` 只需要输入图像与输入相机，`forward` 委托预测，目标图像或目标相机不成为网络前向条件。过滤前 Gaussian 可选保留，供训练与对照使用。

射线 bicubic/antialias 与 DPT 位置编码的固定常量在 CPU 上预计算并缓存，再传到 NPU；图像解码、相机准备、checkpoint 读取和导出也在 CPU。其余网络主计算在 NPU。原仓实际未执行的 DINO resize 不额外引入。M1 反向探针只证明图连通，不是监督损失。

## M2：静态可微渲染与自带 CANN 算子

`StaticGaussianRenderer` 接收未激活的 xyz、SH、log-scale、quaternion、opacity logits 与目标相机，输出 FP32 RGB、alpha、RGBA；投影、激活、协方差与 SH 使用 TorchNPU Tensor 运算。可见性筛选、tile 建表、排序、前向合成和合成反向复用随仓库提供的 CANN `gauss-splat` 源码。实际调用的算子包含 `FlashGaussianBuildMask`、`GaussianSort`、`CalcRenderFwdDoubleClipGsids` 和 `CalcRenderBwdVarClipGsids`。`rendering/native.py`、`rendering/static.py` 是本项目与这些算子之间的适配层，支持多个目标相机和 alpha 输出，并将一阶图像梯度传回 Gaussian 与 M1。

第三方代码的固定来源、许可证和**本地修改清单**见 [third_party/gauss_splat/UPSTREAM.md](../third_party/gauss_splat/UPSTREAM.md)：包括 CANN 9.x 构建路径、逐像素 alpha 阈值、反向透射率还原、小 tile 调度和编译链接修正。M2 不把整套第三方算子声称为本项目从零自编；本项目自编的是适配、可微几何与对原生实现的必要修正。原生合成支持一阶梯度，高阶梯度未提供。构建命令见[渲染接口说明](../rendering/README.md)。

## M3：真实监督预训练

`training/data.py` 读取 PhysAssets 图像、相机和 `physical.json`，提供 4 输入、8 目标（包含输入）、确定性视图采样、同步裁剪与内参调整。`training/task.py` 组合网络、M2 renderer 和 RGB MSE、alpha MSE、14 类 BCE、E/ν Gaussian NLL＋均值 MSE；可选冻结的 VGG LPIPS。损失在 FP32 中计算，DINO 不参与更新。预训练使用 M2 的图像反向；MPM 和动态渲染不在这条训练图里。

迁移版重写了有效更新计数、梯度累积 `no_sync`、非有限值处理、FP16 GradScaler、确定性数据位置、完整 optimizer/scheduler/RNG 断点和严格恢复检查。保留 AdamW、warmup/cosine 及原训练权重。`--init-checkpoint` 是完整模型权重初始化；`--dinov3-checkpoint` 只载入冻结 DINO，并使用 SH1 与原训练头初始化；`--resume` 只接收迁移版完整状态。原官方 checkpoint 所含优化器状态不对应全网络，不能作为完整恢复。详细使用见[训练说明](../training/README.md)。

## M4：MPM 前向与本项目自编 Ascend C 算子

`mpm/` 实现原 Warp 求解器的粒子约束、六种材料（jelly、metal、sand、foam、snow、plasticine）、P2G、网格更新/边界、G2P、APIC/RPIC 和材料内部状态。FP32 仿真子步中粒子与网格保持在 NPU；CPU 执行一次性规范化、密度估计、内部填充、体积计算以及输出序列化。`upstream` 模式遵循原仓实际代码行为；`corrected` 仅修复 surface collider 的最终速度赋值与顺序释放配置。

本项目**自编三类 Ascend C 算子**，与 M2 的第三方渲染算子独立构建：

| 算子族 | 源码 | 职责 |
| --- | --- | --- |
| 3×3 Jacobi SVD | `mpm/csrc/svd.cpp` | 批量小矩阵分解，本构返回映射使用同一谱基 |
| P2G 贡献生成 | `mpm/csrc/transfer.cpp` | 融合 27 邻域权重、网格索引、质量和动量贡献 |
| G2P/粒子更新 | `mpm/csrc/transfer.cpp` | 网格速度采样、速度/APIC、形变梯度和位置更新 |

重复网格索引的累加仍使用 TorchNPU `IndexAdd`，G2P 的网格读取使用 gather；没有另写原子累加算子。本构其余公式、网格更新与边界主要用 TorchNPU Tensor 运算。`mpm/native.py` 通过当前 NPU stream 调用本地构建的算子，不向系统 OPP 安装。CPU FP64 参考与小场景测试验证分解、材料、质量守恒及 100 子步状态。M4 不提供仿真反向。

## M5：动态渲染与缓存回放

`DynamicGaussianRenderer` 接受每帧世界坐标位置、完整协方差、SH、已激活 opacity 与可选着色旋转。`rendering/mpm_adapter.py` 完成仿真坐标逆变换、`FΣFᵀ`、原 Gaussian ID 和静态区域拼接，并排除填充粒子。动态 renderer 按目标相机计算颜色后调用 M2 `render_precomputed()`；没有新 CANN 算子，也不把协方差重新拆成尺度和四元数。

`rendering/sequence.py` 把共享外观与逐帧 CPU FP32 几何分开保存；`render_dynamic.py` 按帧读取，可更换固定、运动或外部相机而不重新运行网络/MPM。`pipeline.py` 串联预测、仿真、渲染和视频输出，延续原仓 `pipeline.py` 入口习惯。M5 只验收前向，缺少运动模糊、时间插值和动态渲染反向。

## 对照范围与限制

在 910B3 上，M1/M2/M3 的网络与静态图像一阶反向已通过；M4/M5 全规模 cake 前向通过。最终 GPU/NPU 对比见[测试结果](benchmark.md)：BF16 深层网络的中间 Gaussian 与严格像素 allclose 没有通过原预设跨设备阈值，不能声称逐元素等价。两端对同一 PhysAssets GT 的静态质量接近，但没有 Gaussian 参数、MPM 轨迹或动态视频 GT。单卡测试不等于双卡 HCCL 实机验收；910C 也未单独测试。
