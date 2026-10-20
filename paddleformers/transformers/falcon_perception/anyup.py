# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Adapted from TII Falcon Perception (Apache-2.0).
"""Native Paddle AnyUp layers with chunked local cross attention."""
import einops as E
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .attention import window_attention
from .norm import RMSNorm


class ResBlock(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super().__init__()
        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else lambda c: nn.Identity()
        p = kernel_size // 2
        self.block = nn.Sequential(
            N(in_channels),
            activation_fn(),
            nn.Conv2D(in_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
            N(out_channels),
            activation_fn(),
            nn.Conv2D(out_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
        )
        self.shortcut = (
            nn.Conv2D(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
            if use_conv_shortcut or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class LearnedFeatureUnification(nn.Layer):
    def __init__(self, out_channels: int, kernel_size: int = 3, init_gaussian_derivatives: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis = nn.Parameter(paddle.randn([out_channels, 1, kernel_size, kernel_size]))

    def forward(self, features: paddle.Tensor) -> paddle.Tensor:
        b, c, h, w = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(b, self.out_channels, c, h, w)
        attn = F.softmax(x, axis=1)
        return attn.mean(axis=2)

    @staticmethod
    def _depthwise_conv(feats, basis, k):
        b, c, h, w = feats.shape
        p = k // 2
        x = F.pad(feats, (p, p, p, p), value=0)
        x = F.conv2d(x, basis.repeat(c, 1, 1, 1), groups=c)
        mask = paddle.ones([1, 1, h, w], dtype=x.dtype, device=x.place)
        denom = F.conv2d(F.pad(mask, (p, p, p, p), value=0), paddle.ones([1, 1, k, k], device=x.place, dtype=x.dtype))
        return x / denom


def _rotate_half(x):
    x1, x2 = x.chunk(2, axis=-1)
    return paddle.cat((-x2, x1), axis=-1)


class AnyUpRoPE(nn.Layer):
    def __init__(self, dim: int, theta: int = 100):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(paddle.empty([2, self.dim]))

    def forward(self, x: paddle.Tensor, coords: paddle.Tensor) -> paddle.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + _rotate_half(x) * angle.sin()


class AttentionWrapper(nn.Layer):
    def __init__(self, qk_dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(paddle.empty([qk_dim * 3, qk_dim]))
        self.in_proj_bias = nn.Parameter(paddle.empty([qk_dim * 3]))

    def forward(self, x_q, x_k, x_v):
        w_q, w_k, w_v = self.in_proj_weight.chunk(3, axis=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        x_q = x_q @ w_q.T + b_q
        x_k = x_k @ w_k.T + b_k
        return (x_q, x_k, x_v)


class FlexCrossAttention(nn.Layer):
    def __init__(self, qk_dim: int, num_heads: int, **kwargs):
        super().__init__()
        self.dim = qk_dim
        self.num_head = num_heads
        self.norm_q = RMSNorm(qk_dim)
        self.norm_k = RMSNorm(qk_dim)
        self.attention = AttentionWrapper(qk_dim)

    def forward(self, query, key, value, mask=None, **kwargs):
        x_q = self.norm_q(query)
        x_k = self.norm_k(key)
        x_q, x_k, x_v = self.attention(x_q, x_k, value)
        x_q = E.rearrange(x_q, "b HW (h d) -> b h HW d", h=self.num_head)
        x_k = E.rearrange(x_k, "b hw (h d) -> b h hw d", h=self.num_head)
        x_v = E.rearrange(value, "b hw (h d) -> b h hw d", h=self.num_head)
        output = window_attention(x_q, x_k, x_v, mask)
        output = E.rearrange(output, "b h hw d -> b hw (h d)")
        return output


class CrossAttentionBlock(nn.Layer):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1, **kwargs):
        super().__init__()
        self.cross_attn = FlexCrossAttention(qk_dim, num_heads)
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2D(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, block_mask, **kwargs):
        b, _, h, w = q.shape
        q = self.conv2d(q)
        q = E.rearrange(q, "b c h w -> b (h w) c")
        k = E.rearrange(k, "b c h w -> b (h w) c")
        v = E.rearrange(v, "b c h w -> b (h w) c")
        features = self.cross_attn(q, k, v, mask=block_mask)
        return E.rearrange(features, "b (h w) c -> b c h w", h=h, w=w)


IMAGENET_MEAN = paddle.to_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = paddle.to_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def create_coordinate(h, w, start=0.0, end=1.0, device=None, dtype=None):
    x = paddle.linspace(start, end, h, device=device, dtype=dtype)
    y = paddle.linspace(start, end, w, device=device, dtype=dtype)
    xx, yy = paddle.meshgrid(x, y, indexing="ij")
    return paddle.stack((xx, yy), -1).view(1, h * w, 2)


class AnyUp(nn.Layer):
    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
        init_gaussian_derivatives=False,
        **kwargs
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.window_ratio = window_ratio
        self._rb_args = dict(
            kernel_size=1, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm, activation_fn=nn.SiLU
        )
        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(
            None, 1, first_layer_k=kernel_size_lfu, init_gaussian_derivatives=init_gaussian_derivatives
        )
        self.cross_decode = CrossAttentionBlock(qk_dim=qk_dim, num_heads=num_heads, window_ratio=window_ratio)
        self.aggregation = self._make_encoder(2 * qk_dim, 3)
        self.rope = AnyUpRoPE(qk_dim)

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0, init_gaussian_derivatives=False):
        pre = (
            nn.Conv2D(in_ch, self.qk_dim, k, padding=k // 2, padding_mode="reflect", bias=False)
            if first_layer_k == 0
            else LearnedFeatureUnification(
                self.qk_dim, first_layer_k, init_gaussian_derivatives=init_gaussian_derivatives
            )
        )
        blocks = [ResBlock(self.qk_dim, self.qk_dim, **self._rb_args) for _ in range(layers)]
        return nn.Sequential(pre, *blocks)

    def upsample(self, enc_img, feats, attn_mask, out_size, vis_attn=False, q_chunk_size=None):
        b, c, h, w = feats.shape
        q = F.adaptive_avg_pool2d(self.query_encoder(enc_img), output_size=out_size)
        k = F.adaptive_avg_pool2d(self.key_encoder(enc_img), output_size=(h, w))
        k = paddle.cat([k, self.key_features_encoder(F.normalize(feats, axis=1))], axis=1)
        k = self.aggregation(k)
        v = feats
        result = self.cross_decode(q, k, v, attn_mask, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
        return result

    def forward(self, images, features, attn_mask, output_size=None, vis_attn=False, q_chunk_size=None):
        output_size = output_size if output_size is not None else images.shape[-2:]
        images = images * 0.5 + 0.5
        images = (images - IMAGENET_MEAN.to(images)) / IMAGENET_STD.to(images)
        images = images.to(features)
        enc = self.image_encoder(images)
        h = enc.shape[-2]
        coords = create_coordinate(h, enc.shape[-1], device=enc.place, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)
        enc = enc.view(enc.shape[0], h, -1, enc.shape[-1]).permute(0, 3, 1, 2)
        result = self.upsample(enc, features, attn_mask, output_size, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
        return result
