# Vendored gauss-splat

Source: https://gitcode.com/cann/gauss-splat.git
Commit: baf7591a710a86c9a7d810c5997b037723a91811
License: CANN Open Software License Agreement Version 2.0 (see LICENSE).

Local changes:
- CANN 9.x bisheng compiler path discovery.
- Per-pixel alpha threshold 1/255 in forward and backward, including the single-Gaussian tail.
- Row-level alpha clipping uses 1/255; tile/group transmittance stopping uses 1e-4.
- Correct reverse compositing: undo the current alpha on every reverse step,
  including the first step, because forward saves the final log-transmittance.
- Build uses project-local outputs; no system CANN modification.
- Link GE host/proto/tiling targets explicitly to CANN libgraph so the package
  can be discovered before the first NPU/AMP operation (M3 training startup).

Unused high-level Rasterizer wrappers are retained as upstream context; PhysGM
uses its own differentiable geometry and explicit RGB/alpha adapter.
The adapter pads FlashGS storage to 64 elements while preserving the actual
compact count, and uses separate sort/render scheduling headers when the number
of tiles is smaller than the device AIV count.
