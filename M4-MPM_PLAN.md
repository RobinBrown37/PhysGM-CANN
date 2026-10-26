# M4 MPM 前向仿真实施计划

基线：PhysGM `7722175ba55db54fd430a7d29a4407b8f8ffe6b6`，迁移项目从 `ddc3805` 开始。范围为六种原仓材料、七类边界接口、CPU 初始化与粒子填充、NPU 前向 MPM、动态 Gaussian 渲染、状态恢复；本阶段无 MPM 自动微分。

## 实施顺序

1. 保留 TorchNPU 张量基线与独立 NumPy FP64 小场景参考，固定材料、选择掩码、RPIC/PIC 和边界时间语义。
2. Ascend C SIMD 3×3 Jacobi SVD 先通过数值闸门，再接入返回映射与应力；保留可显式选择的设备 AI CPU SVD 对照。
3. 测量完整子步，优化 P2G 贡献生成与 G2P/粒子更新。最多三类算子，累加使用 IndexAdd，网格读取使用现有 gather。所有状态保留在 NPU。
4. CPU 实现原填充算法，缓存同一初始粒子包；接入预计算 covariance/RGB/opacity 渲染入口。
5. 运行完整 cake：213532 Gaussian、实际填充粒子、100³ 网格、dt=2e-4、200 子步/帧、50 帧，512²、25 FPS。原物理参数和填充均保留。
6. 记录 CPU/NPU 正确性、完整规模性能、执行路径、恢复结果与 M1～M3 回归，发布状态、视频、预览和迁移说明。

## 闸门与模式

SVD 重构/正交误差≤1e-5；质量相对误差≤1e-5；小场景状态单步≤1e-4、100 子步≤1e-3，各字段独立比较。归一化误差为 `||actual-reference||/max(||reference||,1)`，不在失败后调整阈值。

默认 upstream 保持实际材料分支和边界行为；corrected 仅修复 surface collider 的最终速度赋值，以及顺序释放的层数/起止时间。metal/snow 不套入教科书替代模型。

CPU 工作仅包括初始化、NumPy/PyMCubes 填充、体积与相机预处理、导出及少量控制/检查；仿真子步无主机粒子/网格求解 fallback。若需要第四类算子、每步主机求解或 SVD 无法通过闸门，停止扩展并记录原因。

原 CUDA/Warp 数值对齐等待同输入参考包，不将本地 NumPy 检查称为原 CUDA 对齐。

## 开源依据

- [CANN 批量 SVD](https://gitcode.com/cann/ops-math/tree/9.0.0/math/svd)：本机现有实现实测执行于设备 AI CPU，故优化小矩阵分解。
- [CANN IndexAdd](https://gitcode.com/cann/ops-nn/blob/master/index/inplace_scatter_add/docs/aclnnIndexAdd.md)：复用冲突安全的网格累加。
- [Ascend C 原子操作](https://asc.gitcode.com/api/SIMD-API/basic_api/atomic_operations/SetAtomicAdd.html)：底层能力参考；本阶段无需自编网格原子累加。
- [Warp 原仓](https://github.com/NVIDIA/warp/tree/v0.10.1)：CPU/CUDA 执行模型，不直接提供本项目的 Ascend 后端。

构建使用本机 CANN 9.1.0 `ascendc_library`、`dav-c220-vec`，不使用仅适配 Ascend 950 的 SIMT 示例，不升级已验收的 PyTorch/TorchNPU 核心依赖。
