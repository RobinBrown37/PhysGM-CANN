# M2 静态高斯渲染迁移实施计划

日期：2026-10-24。先完成本计划，再实施与验收。当前源码基线为 M1 提交 `ecbd1a0`；当前可见设备为 Ascend910B3，逻辑设备以 TorchNPU 实际枚举为准。910C 单独保留待验收状态。

## 1. 目标与交付边界

完成网络 Gaussian → 目标相机 RGB/alpha 的静态可微渲染，提供后续预训练可直接调用的、保留计算图的接口。必须验证位置、SH、log-scale、原始 quaternion、opacity logits 的反向，以及图像梯度回到 Gaussian/DPT 头和共享网络的路径。

本轮不增加训练循环、物理监督损失、优化器、调度器或参数更新，不实施 MPM/动态仿真。物理头不参与图像渲染，对其不要求图像梯度。M1 纯预测接口和权重键保持兼容。

## 2. 后端与构建

- 使用官方 `https://gitcode.com/cann/gauss-splat.git`，固定 SHA `baf7591a710a86c9a7d810c5997b037723a91811`，保留许可证和来源。禁止依赖浮动 master。
- 采用 TorchNPU/PyTorch 可微几何投影、quaternion→covariance、SH；使用 CANN 原生 tile 构建/排序和 alpha 合成前反向。这样避免高层 Rasterizer 写死 near=0.2 和不提供 alpha 的限制，投影梯度由 autograd 明确管理。
- 以 CANN compositor 的第四个累积属性通道输入常量 1，得到 alpha=Σ(T·a)，与 RGB 共用一次排序和合成；RGB 加 `(1-alpha)*background`。该通道的反向必须参与位置/形状/opacity 的梯度。
- 检查并修补 CANN 行级 alpha 裁剪、透射率终止及饱和导数，记录与 gsplat 的语义差异。默认采用 1/255 alpha 阈值、0.999 alpha 上限和 1e-4 透射率终止；前反向使用同一策略。像素坐标使用 x+0.5/y+0.5。
- 原生扩展和自定义算子安装在项目 build 目录，不能覆盖系统 CANN。uv 锁定额外构建依赖，构建脚本发现实际 toolkit/Python/ABI，限制并行编译数，保存日志和构建元数据。M1 导入不触发原生渲染扩展。

## 3. 接口与行为

- 新增独立 `StaticGaussianRenderer`，接收 M1 Gaussian 字典和目标 `c2ws`、`intr`、width/height，返回 float32 `rgb [B,T,3,H,W]`、`alpha [B,T,1,H,W]`，提供组合 RGBA 和统计信息。
- `xyz [B,N,3]`、`feature [B,N,K,3]`、`scale [B,N,3]`、`rotation [B,N,4]`、`opacity [B,N,1]` 保持 M1 参数约定。exp/sigmoid/quaternion normalize 各执行一次；SH degree 0–3，按每个目标相机计算方向和颜色，保留 clamp(min=0) 与 +0.5 约定。
- 支持固定 pinhole 相机、near/far、白色/指定背景、批次和多相机、非正方形/非整 tile 分辨率、全部被剔除和空 Gaussian。空输出为背景与 alpha=0，同时保留合法零梯度路径。
- 渲染在 autocast 外以 FP32 执行，`.float()` 不能截断网络梯度；仅可见性、tile membership 和深度排序是离散决策。默认不额外随机抽点/裁剪模型输出，不用 PLY 往返替代在线 Tensor 输入。
- CPU 小规模 reference 后端用于独立公式与梯度验证，设置规模保护，绝不自动回退。NPU 后端缺失时直接报告构建/运行错误。
- 新增静态渲染 CLI：既能回放 M1 的 CPU Tensor 预测包，也能在线调用现有网络；目标相机默认示例四视图，接口允许后续传入独立监督视图。导出 float RGB/alpha、PNG、完整 CPU renderer 输入包和运行报告。

## 4. 验证顺序与成功条件

1. 构建/导入及小场景：验证实际 NPU、自定义前反向算子、非零图像与有效 alpha，记录工具链。
2. CPU double reference：覆盖投影/旋转/SH/白背景、遮挡与排序、近远裁剪、空场景、多 batch/相机和边界尺寸；使用 gradcheck/有限差分验证五组原始属性。排序/裁剪边界的不可导点与数值误差分开报告。
3. NPU 对照：固定相同 Gaussian/相机、RGB/alpha 上游梯度，与 CPU reference 比较浮点图像和全部属性梯度；分别测试 RGB-only、alpha-only 与联合梯度。FP32 图像初始容差 rtol=1e-3/atol=2e-4；梯度 rtol=1e-2/atol=2e-4。失败时定位公式或 kernel，不直接放宽阈值。
4. 真实渲染：回放现有 cake 512 预测包的全部 1,048,576 个 Gaussian，四个目标视图，记录 float 图像、alpha、时间、峰值内存和任何显式预处理/调度 CPU 工作。
5. 网络连接：真实四视图 256 输入 → M1 → renderer → 固定图像上游梯度；检查 DPT、射线、投影、Transformer、global tokens 的有限/非零梯度，DINO 冻结，物理头没有图像梯度。此诊断不增加训练代码。
6. 回归与报告：运行 M1 及新增渲染测试，保存构建、精度、梯度、规模与算子证据；补齐使用文档和后续预训练接口说明。

## 5. 结果记录与限制

CPU reference/NPU 功能与梯度验证、GPU gsplat 数值对齐、910C 实测分别报告，不能互相替代。当前无 GPU 渲染参考包，提供可重放输入输出协议；若原生后端或任何所需梯度受阻，保留可复现日志，完成独立接口和 reference，明确标记未通过项，继续修复可解决的问题。

最终报告必须说明真实规模是否使用全部 Gaussian、是否保留全部可见参数的梯度、RGB/alpha 对原始属性的误差、网络端实际可训练路径、CPU 工作、实测设备及未覆盖的验收项。
