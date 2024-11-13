import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.append('../../..')


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """MLP layer, usually used in Transformer"""

    def __init__(
            self,
            in_feat: int,
            mlp_ratio: int = 1,
            out_feat: Optional[int] = None,
            dropout_p: float = 0.1,
            act_layer: nn.Module = nn.Mish,
    ):
        super().__init__()

        mid_feat = in_feat * mlp_ratio
        out_feat = out_feat or in_feat

        self.act = act_layer()

        self.linear1 = nn.Linear(in_feat, mid_feat)
        self.drop1 = nn.Dropout(dropout_p)

        self.linear2 = nn.Linear(mid_feat, out_feat)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.drop1(self.act(self.linear1(x)))
        x = self.drop2(self.linear2(x))
        return x


class DeformAttentionLayer(nn.Module):
    def __init__(self, seq_in: int, seq_out: int, num_feat: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0):
        super().__init__()

        assert num_feat % num_heads == 0
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.num_feat = num_feat
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.scale = self.head_dim ** -0.5
        self.linear_q = nn.Linear(self.seq_in, self.seq_out)
        self.linear_k = nn.Linear(self.seq_in, self.seq_out)
        self.linear_v = nn.Linear(self.seq_in, self.seq_out)
        self.attn_drop = nn.Dropout(dropout_p)
        self.proj = nn.Linear(self.num_feat, self.num_feat)
        self.proj_drop = nn.Dropout(dropout_p)

    def forward(self, x):
        B, L, C = x.shape
        x_transpose = x.permute(0, 2, 1)
        assert C == self.num_feat
        q = self.linear_q(x_transpose).permute(0, 1, 2).reshape(B, self.seq_out, self.num_heads, self.head_dim)
        k = self.linear_k(x_transpose).permute(0, 1, 2).reshape(B, self.seq_out, self.num_heads, self.head_dim)
        v = self.linear_v(x_transpose).permute(0, 1, 2).reshape(B, self.seq_out, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, seq_out, head_dim]
        k = k.permute(0, 2, 1, 3)  # [B, num_heads, seq_out, head_dim]
        v = v.permute(0, 2, 1, 3)  # [B, num_heads, seq_out, head_dim]
        # d = torch.einsum('bd,nd->bn', a, b)
        attn = torch.einsum('nhad,nhbd->nhab', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('nhss,nhsd->nhsd', attn, k)  # [B, num_heads, seq_out, head_dim]
        x = x.permute(0, 2, 1, 3).reshape(B, self.seq_out, self.num_feat)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionLayer(nn.Module):
    """Multi-head scaled self-attension layer"""

    def __init__(self, num_feat: int, num_heads: int = 8, qkv_bias: bool = False, dropout_p: float = 0.0):
        super().__init__()

        assert num_feat % num_heads == 0

        self.num_feat = num_feat
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.num_feat, self.num_feat * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_p)

        self.proj = nn.Linear(self.num_feat, self.num_feat)
        self.proj_drop = nn.Dropout(dropout_p)

    def forward(self, x):
        B, L, C = x.shape
        assert C == self.num_feat

        qkv = self.qkv(x)  # [B, L, num_feat * 3]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # [B, L, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv.unbind(0)  # [B, num_heads, L, head_dim] * 3
        attn = q @ k.transpose(-2, -1) # [B, num_heads, L, L]
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn_score = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)  # [B, L, num_heads, head_dim]
        x = x.reshape(B, L, self.num_feat)  # [B, L, num_feat]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_score


class RelativeAttentionLayer(nn.Module):
    """Multi-head scaled self-attension layer"""

    def __init__(self, num_feat: int, num_heads: int = 8, qkv_bias: bool = False, dropout_p: float = 0.0):
        super().__init__()

        assert num_feat % num_heads == 0

        self.num_feat = num_feat
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.num_feat, self.num_feat * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_p)

        self.proj = nn.Linear(self.num_feat, self.num_feat)
        self.proj_drop = nn.Dropout(dropout_p)
        self.relative_project = nn.Sequential(nn.Linear(self.num_feat, num_heads), nn.GELU(), nn.LayerNorm(num_heads))

    def forward(self, x, relative_pos):
        B, L, C = x.shape
        assert C == self.num_feat
        qkv = self.qkv(x)  # [B, L, num_feat * 3]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # [B, L, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv.unbind(0)  # [B, num_heads, L, head_dim] * 3
        attn = q @ k.transpose(-2, -1) # [B, num_heads, L, L]
        relative_pos = self.relative_project(relative_pos)
        relative_pos = relative_pos.permute(0, 3, 1, 2)

        attn = (attn + relative_pos) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)  # [B, L, num_heads, head_dim]
        x = x.reshape(B, L, self.num_feat)  # [B, L, num_feat]

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CoAttentionLayer(nn.Module):
    """Multi-head scaled self-attension layer"""

    def __init__(self, num_feat: int, coatten: int, num_heads: int = 8, qkv_bias: bool = False, dropout_p: float = 0.0):
        super().__init__()

        assert num_feat % num_heads == 0

        self.num_feat = num_feat
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.scale = self.head_dim ** -0.5
        self.coatten = coatten

        self.qkv = nn.Linear(self.num_feat, self.num_feat * 3, bias=qkv_bias)
        self.attn_drop1 = nn.Dropout(dropout_p)
        self.attn_drop2 = nn.Dropout(dropout_p)

        self.proj = nn.Linear(self.num_feat, self.num_feat)
        self.proj_drop = nn.Dropout(dropout_p)

    def forward(self, x):
        cat_index = self.coatten
        B, L, C = x.shape
        assert C == self.num_feat

        qkv = self.qkv(x)  # [B, L, num_feat * 3]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # [B, L, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv.unbind(0)  # [B, num_heads, L, head_dim] * 3
        q1, k1, v1, = q[:, :, :cat_index, :], k[:, :, :cat_index, :], v[:, :, :cat_index, :]
        q2, k2, v2, = q[:, :, cat_index:, :], k[:, :, cat_index:, :], v[:, :, cat_index:, :]
        attn1 = (self.attn_drop1((q1 @ k2.transpose(-2, -1)) * self.scale) @ v2).transpose(1, 2)
        attn2 = (self.attn_drop2((q2 @ k1.transpose(-2, -1)) * self.scale) @ v1).transpose(1, 2)
        x = torch.cat([attn1, attn2], dim=1).reshape(B, L, self.num_feat)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            in_feat: int,
            out_feat: Optional[int] = None,
            num_heads: int = 8,
            qkv_bias: bool = False,
            mlp_ratio: int = 4,
            dropout_p: float = 0.1,
            droppath_p: float = 0.1,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            coatten: int = 0,
    ):
        super().__init__()

        out_feat = out_feat or in_feat

        self.droppath = DropPath(droppath_p)

        self.norm1 = norm_layer(in_feat)
        self.norm2 = norm_layer(in_feat)
        self.coatten = coatten
        if coatten:
            self.attn = CoAttentionLayer(num_feat=in_feat, coatten=coatten, num_heads=num_heads, qkv_bias=qkv_bias,
                                         dropout_p=dropout_p)
        else:
            self.attn = AttentionLayer(num_feat=in_feat, num_heads=num_heads, qkv_bias=qkv_bias, dropout_p=dropout_p)

        self.mlp = MLP(
            in_feat=in_feat, mlp_ratio=mlp_ratio, out_feat=out_feat, dropout_p=dropout_p, act_layer=act_layer
        )
        self.jump = True if in_feat != out_feat else False
        # self.attention_weight = nn.Parameter(0.3, requires_grad=True)
        # self.mlp_weight = nn.Parameter(0.3, requires_grad=True)

    def forward(self, x):
        x1 = x
        x1, attn_socre = self.attn(self.norm1(x1))
        x = x + self.droppath(x1)
        if self.jump:
            x = self.droppath(self.mlp(self.norm2(x)))
        else:
            x = x + self.droppath(self.mlp(self.norm2(x)))
        return x, attn_socre


class RelativeTransformerBlock(nn.Module):
    def __init__(
            self,
            in_feat: int,
            out_feat: Optional[int] = None,
            num_heads: int = 8,
            qkv_bias: bool = False,
            mlp_ratio: int = 4,
            dropout_p: float = 0.1,
            droppath_p: float = 0.1,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        out_feat = out_feat or in_feat

        self.droppath = DropPath(droppath_p)

        self.norm1 = norm_layer(in_feat)
        self.norm2 = norm_layer(in_feat)
        self.attn = RelativeAttentionLayer(num_feat=in_feat, num_heads=num_heads, qkv_bias=qkv_bias, dropout_p=dropout_p)

        self.mlp = MLP(
            in_feat=in_feat, mlp_ratio=mlp_ratio, out_feat=out_feat, dropout_p=dropout_p, act_layer=act_layer
        )
        self.jump = True if in_feat != out_feat else False
        # self.attention_weight = nn.Parameter(0.3, requires_grad=True)
        # self.mlp_weight = nn.Parameter(0.3, requires_grad=True)

    def forward(self, x, relative_pos,):
        x = x + self.droppath(self.attn(self.norm1(x), relative_pos))
        if self.jump:
            x = self.droppath(self.mlp(self.norm2(x)))
        else:
            x = x + self.droppath(self.mlp(self.norm2(x)))
        return x


class DeformTransformerBlock(nn.Module):
    def __init__(
            self,
            seq_in: int,
            seq_out: int,
            in_feat: int,
            out_feat: Optional[int] = None,
            num_heads: int = 8,
            qkv_bias: bool = False,
            mlp_ratio: int = 4,
            dropout_p: float = 0.1,
            droppath_p: float = 0.1,
            act_layer: nn.Module = nn.Mish,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        out_feat = out_feat or in_feat
        self.droppath = DropPath(droppath_p)

        self.norm1 = norm_layer(in_feat)
        self.norm2 = norm_layer(in_feat)
        self.deform_linear = nn.Linear(seq_in, seq_out)
        self.attn = DeformAttentionLayer(seq_in=seq_in, seq_out=seq_out, num_feat=in_feat, num_heads=num_heads,
                                         qkv_bias=qkv_bias, dropout_p=dropout_p)

        self.mlp = MLP(
            in_feat=in_feat, mlp_ratio=mlp_ratio, out_feat=out_feat, dropout_p=dropout_p, act_layer=act_layer
        )

    def forward(self, x):
        x_transpose = x.permute(0, 2, 1)
        x_transpose = self.deform_linear(x_transpose).permute(0, 2, 1)
        x = x_transpose + self.droppath(self.attn(self.norm1(x)))
        x = x + self.droppath(self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    feat_dim = 128
    sl = 256
    ssl = 312
    inputs = torch.rand((1, sl, feat_dim))
    attn_layer = TransformerBlock(feat_dim)
    print(attn_layer(inputs)[0].shape)
    print(attn_layer(inputs)[1].shape)
    
    deattn_layer = DeformTransformerBlock(sl, ssl, feat_dim)
    print(deattn_layer(inputs).shape)
    
    a = torch.rand((9, 10, 12, 14, 16))
    b1 = a.reshape(9 * 10, 12 * 14, 16)
    b2 = a.reshape(9 * 10, 12, 14, 16)
    b2 = b2.reshape(9 * 10, 12 * 14, 16)
    # print(b1 - b2)
