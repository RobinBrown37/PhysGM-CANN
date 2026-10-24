# M2 静态高斯渲染迁移验收报告

日期：2026-10-24。M1 基线提交 `ecbd1a0`，当前设备 Ascend910B3（64 GB）。

**静态 RGB/alpha、五组 Gaussian 属性的一阶反向、图像梯度到现有网络的连接均已通过。** 本轮没有训练循环、优化器或参数更新。910C 和同批 GPU gsplat 对齐仍待独立验收。

## 实际交付

- 独立 `StaticGaussianRenderer`，输入原始 Gaussian Tensor 与固定目标相机，输出 FP32 RGB/alpha/RGBA，保留 autograd 计算图。
- TorchNPU 可微投影、quaternion/covariance、SH 0–3；CANN FlashGS 分块、排序及原生合成前反向。
- 使用辅助累积通道输入 1，在同一次合成中获得 alpha，白背景通过 `RGB_fg+(1-alpha)*background` 计算，alpha 梯度完整回传。
- 完整 CPU 重放输入/输出/属性梯度包、PNG 预览、构建脚本、精度测试和原生算子 trace。
- M1 预测接口及权重键未修改，独立 M1 验收脚本仍可单独运行。

## 实测结果

| 验收项 | 结果 |
| --- | --- |
| M1 + 静态渲染回归 | 33 passed，0 failed |
| CPU double gradcheck | 位置、SH、log-scale、quaternion、opacity logits，SH 0–3，通过 |
| NPU 小场景与 CPU double 对照 | RGB-only、alpha-only、联合梯度，通过 |
| 小场景最大 RGB / alpha 绝对误差 | 1.528e-07 / 2.628e-07 |
| 小场景五组属性最大梯度绝对误差 | 2.443e-06 |
| 多视图、四 Gaussian 分组及尾部 | 前向/反向与参考一致；内部像素方向有限差分通过 |
| 真实 512 四视图渲染输入 | 全部 1,048,576 个 Gaussian，无随机抽样 |
| 512 预热后渲染中位数 / p95 | 0.1665 / 0.1670 秒 |
| 512 纯渲染峰值分配 / 预留 | 1.397 / 1.471 GiB |
| 百万点、512 四视图属性反向 | 五组梯度有限、各组总量非零，通过 |
| 512 纯渲染单次反向 | 0.2750 秒 |
| 512 属性反向峰值分配 / 预留 | 1.794 / 1.971 GiB |
| 在线网络 → 渲染 → 图像 VJP | 256 输入、262,144 个 Gaussian、四视图，通过 |
| 256 网络+渲染前向中位数 / 单次反向 | 0.2004 / 0.6296 秒 |
| 256 网络连接峰值分配 / 预留 | 4.606 / 5.047 GiB |

512 纯渲染前向使用 1 次预热和 3 次同步计时；512 属性反向为另一次真实输入重放，计一次 backward。256 在线路径也执行 1 次预热和 3 次同步计时，反向只计一次。I/O/模型加载不计入预热后前向，profiling 独立重放，不计入所列时间和峰值内存。以上是单个 cake batch 的本机指标。

所有百万点都传入 renderer。根据 near/far 与 alpha 上界做离散筛选：opacity 小于 1/255 的点在任何像素都不可能贡献有效 alpha，因此可以剔除；不将这种标准可见性筛选描述为额外抽样。部分属性/点的零梯度来自裁剪、遮挡或 SH clamp，符合分段可微光栅化行为。

## 预训练反向准备

真实图像上游 VJP 已传到 xyz、feature、scale、rotation、opacity，以及射线编码、输入投影/归一化、24 层 Transformer、DPT 与三个全局 token。DINO 冻结、保持 eval；物理头没有图像梯度符合预期，后续需要材料/E/ν 标签监督。

测试不是只检查梯度存在：先用 double gradcheck 验证参考路径，再在固定相同图像上游梯度下比较原生 RGB/alpha 与五组原始属性梯度。FP32 图像容差 rtol=1e-3/atol=2e-4，梯度 rtol=1e-2/atol=2e-4，实施中未放宽。有限差分避开 alpha 阈值和排序边界；原生合成仅支持一阶反向，不提供二阶导数。

后续预训练可直接将 `result.rgb` / `result.alpha` 接入真实监督；调用中不能 detach、no_grad 或经 PLY 重载。预训练循环、真实标签损失、优化器/AMP、短训及保存恢复尚未实施。

## 原生后端修正与构建

固定上游：`https://gitcode.com/cann/gauss-splat.git`，SHA `baf7591a710a86c9a7d810c5997b037723a91811`。许可证保留于第三方目录。

最终实现修正了以下问题：

- 显式使用 PhysGM 的 near/far、像素中心、单次激活和每相机 SH 计算，避免高层接口默认 near=0.2。
- 逐像素 1/255 alpha 阈值，前反向一致；alpha 上限 0.999，透射率终止 1e-4。
- 合成反向从最终 log-transmittance 还原当前 Gaussian 之前的 T，包括第一个反向步骤；此修正由独立图像/梯度对照验证。
- FlashGS 存储按 64 元素对齐、实际有效计数独立保留；小图排序与合成采用各自实际核心数的调度头。
- 刚体相机使用显式逆变换，避免通用 linalg inverse 的 CPU 回退。
- CANN 9.x 编译器发现、PyTorch ABI 匹配、三阶段构建及二进制清单检查，全部输出到项目 build。

环境为 Python 3.11.4 / aarch64、PyTorch 2.7.1+cpu、TorchNPU 2.7.1.post8、CANN 9.1.0。已完整复跑三阶段构建，`uv sync --locked --group render-build --offline` 通过；无系统 CANN 或系统 Python 修改。

核心 trace 已确认 `FlashGaussianBuildMask`、`GaussianSort`、`CalcRenderFwdDoubleClipGsids`、`CalcRenderBwdVarClipGsids` 及对应 ACLNN 调用。CPU 工作为输入/相机预处理、调度元数据和导出；在线 M1 还保留其固定位置编码的 CPU 常量预计算。投影、SH 与合成前反向在 NPU 上执行。

## 产物与复现

- [构建、API 和预训练接入说明](rendering/README.md)
- [完整实施计划](STATIC_RENDER_PLAN.md)
- [NPU 512 四视图报告](reports/static_512.json)
- [百万点 512 属性反向报告](reports/static_512_backward.json)
- [在线网络图像反向与算子报告](reports/static_network_backward.json)
- [图像/梯度误差与有限差分](reports/static_numeric.json)
- [构建信息](reports/static_build.json) / [回归记录](reports/static_verification.json)
- float Tensor、重放包和四视图 PNG：`results/static_512/`。
- 在线反向及原生算子 Chrome trace：`results/static_network_backward/`。

预览：[RGB](results/render_preview.png) / [alpha](results/static_512/alpha_b0_v0.png)。RGB 预览复制自 `results/static_512/rgb_b0_v0.png`；已人工查看蛋糕重建与 alpha 轮廓。

## 单独保留的验收项

没有 GPU 参考包，因此未声称与原仓实际安装的 gsplat 达成数值一致。CANN 按四 Gaussian 分组/整 tile 决策提前终止，参考小场景禁用提前终止；GPU 对齐还需核查裁剪、相交规则及其梯度边界。910C 未实测。

真实监督训练尚未实施；当前模型图像梯度可用不能替代标签损失、有效参数更新、短训和保存恢复验收。
