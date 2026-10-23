# Adapted for Long-LRM by Ziwen 2024
# from timm library https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# and DINO https://github.com/facebookresearch/dino/blob/main/vision_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4., mlp_bias=False, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input 
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim=64, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_flashatt_v2=True):
        super().__init__()
        assert dim % head_dim == 0, 'dim must be divisible by head_dim'
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        # Retain the constructor argument for upstream call-site compatibility.
        # SDPA dispatches to the device backend, including TorchNPU.
        self.use_flashatt_v2 = use_flashatt_v2

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input 
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, sequence, head_dim)
        x = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, head_dim, mlp_ratio=4., mlp_bias=False, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flashatt_v2=True):
        super().__init__()
        self.norm1 = norm_layer(dim, bias=False)
        self.attn = SelfAttention(
            dim, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_flashatt_v2=use_flashatt_v2)
        if drop_path != 0:
            raise ValueError("This checkpoint uses drop_path=0; nonzero drop_path is unsupported")
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim, bias=False)
        self.mlp = Mlp(in_features=dim, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
