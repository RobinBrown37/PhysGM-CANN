import os
from pathlib import Path

# ACLNN may cache package discovery on its first AMP/optimizer operation.
# Publish the local package before device work without loading an extension.
_vendor = Path(__file__).resolve().parents[1]/"build/gauss_splat/packages/vendors/customize"
if _vendor.is_dir():
    _paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if str(_vendor) not in _paths:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([str(_vendor), *[p for p in _paths if p]])

from .static import StaticGaussianRenderer, RenderResult

__all__ = ["StaticGaussianRenderer", "RenderResult"]
