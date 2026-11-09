# M2 静态高斯渲染迁移：接口与反向传播

项目总览见 [README](../README.md)，模块差异见 [迁移说明](../docs/migration.md)。

接口提供 RGB/alpha 和一阶反向传播。投影、协方差、SH 和原始参数激活均在 TorchNPU 上执行；分块筛选、排序和合成使用项目内构建的 CANN 算子。原有 M1 `predict/forward` 行为保持不变。

## 构建与运行

```bash
cd /path/to/PhysGM-CANN
source /path/to/Ascend/cann-9.1.0/set_env.sh
uv sync --locked --group render-build
uv run --locked --group render-build python scripts/build_renderer.py --jobs 8

# 回放 M1 的全部百万点预测，不经过 PLY 或重新采样
uv run --locked python render_static.py \
  --input-pack results/npu_forward/prediction.pt --output-dir results/static_512

# 在线 M1 → renderer → 固定图像上游梯度，包含算子跟踪
uv run --locked python render_static.py --online --input-resize 256 \
  --backward-probe --profile --output-dir results/static_network_backward

# 完整构建和验收（会重新生成结果）
bash scripts/verify_static.sh
```

默认构建 `ascend910b`，本机 910B3 已实测；A3 的构建参数为 `--soc ascend910_93`，实机仍需单独验证。构建依赖由 uv 的 `render-build` 组锁定，原生库、CANN 包、日志和构建元数据均位于被 git 忽略的 `build/`。不向系统 CANN 或系统 Python 安装自定义算子。

编译来源固定于官方 [gauss-splat](https://gitcode.com/cann/gauss-splat) 的 `baf7591a710a86c9a7d810c5997b037723a91811`；许可证与本地改动见 [UPSTREAM.md](../third_party/gauss_splat/UPSTREAM.md)。`uv run` 使用现有构建产物；在新机器或清理 build 后需重新构建。

## Python 接口

```python
from rendering import StaticGaussianRenderer

renderer = StaticGaussianRenderer(
    sh_degree=0, near_plane=0.001, far_plane=2.5,
    eps2d=0.3, backend="cann",
)
# model、batch 和固定目标相机均已在同一 NPU 上。
prediction = model.predict(batch)
result = renderer(
    prediction["gaussians"], target_c2ws, target_intr,
    width=512, height=512, background=(1., 1., 1.),
)
rgb = result.rgb      # [B,T,3,H,W]
alpha = result.alpha  # [B,T,1,H,W]
rgba = result.rgba    # [B,T,4,H,W]
# rgb/alpha 保持计算图，后续可直接用于真实图像与 alpha 监督损失。
```

模型可在 BF16 autocast 中预测；renderer 内部显式使用 FP32，并保留 `.float()` 前后的 autograd 连接。训练调用不能包在 `no_grad()` 中，也不能将 Gaussian `detach()`、转 CPU 或经 PLY 重新载入。CPU Tensor 输入包仅供独立重放验证。

输入 Gaussian 字段：

| 字段 | 形状 | 约定 |
| --- | --- | --- |
| `xyz` | `[B,N,3]` | 与相机一致的世界坐标 |
| `feature` | `[B,N,(degree+1)^2,3]` | SH 系数，包含 M1 原有 RGB residual |
| `scale` | `[B,N,3]` | log-scale，renderer 内 exp 一次 |
| `rotation` | `[B,N,4]` | 原始 wxyz quaternion，renderer 内 normalize 一次 |
| `opacity` | `[B,N,1]` | logits，renderer 内 sigmoid 一次 |

目标相机为刚体 `c2ws [B,T,4,4]`，内参 `intr [B,T,4]=(fx,fy,cx,cy)`，单位为当前输出图像的像素。每个视图独立计算 SH 方向；支持 SH 0–3、非方形/非整 tile 分辨率、多个 batch/相机、指定背景及空/全部剔除场景。相机作为固定监督条件；本轮没有验收可训练相机、镜头畸变、正交或鱼眼相机。

Renderer 输出保持 float RGB 原值，允许颜色大于 1；仅 PNG 预览 clamp 到 [0,1]。alpha 的 FP32 累积可能产生约 1e-6 的边界舍入误差，未通过额外 clamp 改变计算图。

## 反向准备状态

- 已验证图像 RGB-only、alpha-only、联合上游梯度到全部五组原始 Gaussian 属性的 VJP；覆盖 quaternion normalization 和 exp/sigmoid 的链式梯度。
- 小场景使用独立 CPU double 累积参考、double gradcheck 和 NPU 方向有限差分。有限差分选在排序、alpha 阈值等离散边界之外。
- 已在真实四视图 256 输入上验证 DPT、共享 Transformer、射线编码、输入投影及全局 token 的图像梯度；DINO 无梯度且保持 eval。
- 物理头没有图像梯度符合架构；后续材料/E/ν 标签损失应独立连接物理头，不能据此误判渲染反向失败。
- CANN 原生合成仅支持一阶梯度；当前预训练的一阶反向可用，高阶梯度未提供。

`--backward-probe` 只传入固定 RGB/alpha 上游梯度，没有真实监督或参数更新。M3 已在 `train_pretrain.py` 中连接真实监督、联合损失、优化器与恢复，详见 [训练说明](../training/README.md)。

## 重放包和验收产物

`render_static.py` 默认在线预测，或用 `--input-pack` 加载 M1 `prediction.pt` / 本模块 `renderer_input.pt`。后者保存完整 `gaussians`、`c2ws`、`intr`、width/height 与渲染参数，全部为 CPU Tensor 和普通类型。可用 `--camera-pack` 替换目标相机，该包包含 `c2ws/intr/width/height`。`--width/--height` 覆盖分辨率时自动按比例缩放内参；`--views K` 明确选择前 K 个目标视图。

`renderer_output.pt` 保存 float RGB/alpha；反向模式还保存五组输入属性梯度。每个 batch/view 的 RGB 与 alpha PNG、统计、同步时间与峰值内存保存在同目录。CPU reference 必须显式选择 `--backend reference --device cpu`，具有规模保护，不自动回退。

渲染侧 CPU 工作包括数据/相机预处理、输出文件以及 tile 数量调度元数据；在线 M1 还保留原有固定位置编码的 CPU 常量预计算。Gaussian 投影、排序、颜色和合成前反向均由 NPU Tensor/算子执行。当前透射率终止由 CANN 按四 Gaussian 分组/整 tile 决策；CPU 小规模参考不启用提前终止。图像精度和梯度按固定测试容差验收；GPU gsplat 跨设备严格 allclose 未通过，GT 对照见 [测试结果](../docs/benchmark.md)。910C 实机尚未测试。

M2 构建改动和与原仓的差异见 [迁移说明](../docs/migration.md)，实测见 [测试结果](../docs/benchmark.md)。

## M5 动态前向与序列回放

`DynamicGaussianRenderer` 复用同一静态 renderer，只把输入换成逐帧世界坐标 `xyz [B,N,3]`、完整 `covariance [B,N,3,3]`、SH `feature [B,N,C,3]`、概率 `opacity [B,N,1]` 与可选 `rotation [B,N,3,3]`。它支持多目标视图，并为每个相机单独求 SH 颜色后调用 `render_precomputed()`；返回仍为 `RenderResult`。

`rendering.mpm_adapter.MPMFrameAdapter` 负责从 MPM 状态恢复世界位置和 `FΣFᵀ`，只取原 Gaussian，排除填充粒子，并按 Gaussian ID 拼接配置区域外的静态部分。`rendering.sequence` 将共享外观与逐帧几何分开保存；所有帧为 CPU FP32、原子写入、逐文件校验，`SequenceReader` 只按需载入一帧。

`render_dynamic.py` 是独立回放入口，不加载网络权重、不运行 MPM 子步。默认复现缓存内记录相机；`--camera-mode config` 重新生成球面配置相机，`--camera-mode input` 使用 M1 输入相机，`--cameras-json` 可替换为外部相机。M5 的实现范围见 [迁移说明](../docs/migration.md)，结果见 [测试结果](../docs/benchmark.md)。
