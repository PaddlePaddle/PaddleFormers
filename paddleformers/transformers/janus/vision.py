# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The timm-style SigLIP tower used by the Janus understanding path.

The official Janus checkpoint contains a fused-qkv ViT rather than the
Hugging Face SigLIP module used by other PaddleFormers models.  These small
layers intentionally preserve the source module names so converted tensors can
be loaded without a second key-renaming layer.
"""

from __future__ import annotations

from typing import Any, Mapping

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

try:
    from paddle.incubate.nn.functional import fused_linear as _fused_linear
except ImportError:  # pragma: no cover - older CPU-only Paddle builds
    _fused_linear = None


_VISION_DEFAULTS = {
    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4.0,
        "global_pool": "map",
        "class_token": False,
    },
    "siglip_so400m_patch14_384": {
        "image_size": 336,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "class_token": False,
    },
    "siglip_so400m_patch14_224": {
        "image_size": 224,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "class_token": False,
    },
}


def _vision_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    params = dict(params or {})
    model_name = params.get("model_name", "siglip_large_patch16_384")
    if model_name not in _VISION_DEFAULTS:
        raise ValueError(f"unsupported Janus vision model: {model_name}")
    values = dict(_VISION_DEFAULTS[model_name])
    values.update(params)
    return values


def _effective_vision_layers(params: Mapping[str, Any] | None) -> int:
    """Return the number of SigLIP blocks retained by Janus ``select_layer``.

    The original Janus ``create_siglip_vit`` builds a truncated tower rather
    than running every block and selecting a hidden state afterwards.  Keep
    that convention here so the module state and its AOA checkpoint mapping
    describe the same set of blocks.
    """

    values = _vision_params(params)
    total_layers = int(values["layers"])
    select_layer = values.get("select_layer", -1)
    try:
        select_layer = int(select_layer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"select_layer must be an integer, got {select_layer!r}") from exc

    if select_layer <= 0:
        effective_layers = min(total_layers, total_layers + select_layer + 1)
    else:
        effective_layers = min(total_layers, select_layer)
    if effective_layers < 1:
        raise ValueError(f"select_layer={select_layer} selects no vision layers (total layers: {total_layers})")
    return effective_layers


def _layer_norm(layer: nn.LayerNorm, x: paddle.Tensor, high_precision: bool) -> paddle.Tensor:
    """Accumulate normalization in a wider type without changing boundaries."""
    if high_precision and x.dtype in (paddle.float32, paddle.bfloat16):
        # FP32 inputs use FP64 to remove H800 reduction-order drift.  Torch's
        # BF16 LayerNorm accumulates statistics in FP32, so mirror that while
        # retaining BF16 outputs at the operator boundary.
        compute_dtype = "float64" if x.dtype == paddle.float32 else "float32"
        y = F.layer_norm(
            x.astype(compute_dtype),
            layer._normalized_shape,
            layer.weight.astype(compute_dtype) if layer.weight is not None else None,
            layer.bias.astype(compute_dtype) if layer.bias is not None else None,
            layer._epsilon,
        )
        return y.astype(x.dtype)
    return layer(x)


def _linear(layer: nn.Linear, x: paddle.Tensor, high_precision: bool) -> paddle.Tensor:
    """Use FP32 accumulation for BF16 visual projections when requested."""
    if high_precision and x.dtype == paddle.bfloat16:
        # Paddle's eager BF16 matmul and Torch's F.linear choose different
        # reduction kernels.  On CUDA, fused_linear uses the cuBLASLt path and
        # matches Torch's BF16 result (including its final rounding) exactly.
        if _fused_linear is not None and paddle.get_device().startswith("gpu"):
            return _fused_linear(x, layer.weight, layer.bias)
        y = paddle.matmul(x.astype("float32"), layer.weight.astype("float32"))
        if layer.bias is not None:
            y = y + layer.bias.astype("float32")
        return y.astype(x.dtype)
    return layer(x)


class JanusVisionMlp(nn.Layer):
    def __init__(self, dim: int, hidden_dim: int, high_precision: bool = False):
        super().__init__()
        self.high_precision = high_precision
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return _linear(self.fc2, self.act(_linear(self.fc1, x, self.high_precision)), self.high_precision)


class JanusVisionAttention(nn.Layer):
    def __init__(self, dim: int, num_heads: int, high_precision: bool = False):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"vision width {dim} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.high_precision = high_precision
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = _linear(self.qkv, x, self.high_precision).reshape(
            [batch_size, seq_len, 3, self.num_heads, self.head_dim]
        )
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = paddle.unstack(qkv, axis=0)
        if self.high_precision and x.dtype in (paddle.float32, paddle.bfloat16, paddle.float64):
            # CUDA BF16 q@k reductions are not ordered identically by Torch and
            # Paddle.  Keep the checkpoint boundary dtype, but perform the
            # complete score/softmax/context calculation in FP64 whenever the
            # explicit parity mode is enabled.  FP64 inputs are already in the
            # desired work type and follow the same branch.
            q_work, k_work, v_work = q.astype("float64"), k.astype("float64"), v.astype("float64")
            scores = paddle.matmul(q_work * self.scale, k_work, transpose_y=True)
            weights = F.softmax(scores, axis=-1)
            output = paddle.matmul(weights, v_work).astype(q.dtype)
        else:
            # Match official eager path: q @ k^T in the input dtype, then
            # softmax with a float32 upcast for BF16.
            q = q * self.scale
            scores = paddle.matmul(q, k, transpose_y=True)
            weights = F.softmax(scores.astype("float32"), axis=-1).astype(q.dtype)
            output = paddle.matmul(weights, v)
        output = output.transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, dim])
        return _linear(self.proj, output, self.high_precision)


class JanusVisionBlock(nn.Layer):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, high_precision: bool = False):
        super().__init__()
        self.high_precision = high_precision
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = JanusVisionAttention(dim, num_heads, high_precision=high_precision)
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = JanusVisionMlp(dim, int(dim * mlp_ratio), high_precision=high_precision)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = x + self.attn(_layer_norm(self.norm1, x, self.high_precision))
        x = x + self.mlp(_layer_norm(self.norm2, x, self.high_precision))
        return x


class JanusAttentionPool(nn.Layer):
    """State-compatible attention pool retained for checkpoint completeness.

    Janus constructs this module but sets ``ignore_head=True`` in the
    understanding tower, so the visual forward path returns patch features
    before this pool.  Keeping the module allows every source tensor to be
    converted and prevents silently dropping trainable parameters.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"pool width {dim} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.latent = self.create_parameter([1, 1, dim])
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = JanusVisionMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        batch_size, seq_len, dim = x.shape
        latent = paddle.tile(self.latent, [batch_size, 1, 1])
        q = self.q(latent).reshape([batch_size, 1, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        kv = self.kv(x).reshape([batch_size, seq_len, 2, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        k, v = paddle.unstack(kv, axis=0)
        scores = paddle.matmul(q, k, transpose_y=True) * self.scale
        weights = F.softmax(scores.astype("float32"), axis=-1).astype(q.dtype)
        pooled = paddle.matmul(weights, v).transpose([0, 2, 1, 3]).reshape([batch_size, 1, dim])
        pooled = self.proj(pooled)
        return pooled + self.mlp(self.norm(pooled))


class JanusVisionTransformer(nn.Layer):
    def __init__(self, params: Mapping[str, Any] | None = None):
        super().__init__()
        values = _vision_params(params)
        self.image_size = int(values["image_size"])
        self.patch_size = int(values["patch_size"])
        self.width = int(values["width"])
        self.num_heads = int(values["heads"])
        self.global_pool = values.get("global_pool", "map")
        self.class_token = bool(values.get("class_token", False))
        # ``vision_parity_precision`` is the auditable setting.  Accept the
        # original boolean flag for checkpoints written before the setting was
        # made explicit.
        parity_precision = values.get("vision_parity_precision")
        if parity_precision is None:
            parity_precision = "fp64_accumulate" if values.get("paddle_high_precision", False) else "native"
        if parity_precision not in ("native", "fp64_accumulate"):
            raise ValueError(f"unsupported vision parity precision: {parity_precision}")
        self.vision_parity_precision = parity_precision
        self.high_precision = parity_precision == "fp64_accumulate"
        self.patch_embed = nn.Layer()
        self.patch_embed.proj = nn.Conv2D(
            3,
            self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        grid = self.image_size // self.patch_size
        num_patches = grid * grid
        prefix = 1 if self.class_token else 0
        self.pos_embed = self.create_parameter([1, num_patches + prefix, self.width])
        if self.class_token:
            self.cls_token = self.create_parameter([1, 1, self.width])
        self.num_layers = _effective_vision_layers(values)
        self.blocks = nn.LayerList(
            [
                JanusVisionBlock(
                    self.width,
                    self.num_heads,
                    float(values["mlp_ratio"]),
                    high_precision=self.high_precision,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.width, epsilon=1e-6)
        if self.global_pool == "map":
            self.attn_pool = JanusAttentionPool(self.width, self.num_heads, float(values["mlp_ratio"]))
        else:
            self.attn_pool = None

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        pixel_values = pixel_values.astype(self.patch_embed.proj.weight.dtype)
        x = self.patch_embed.proj(pixel_values)
        x = x.flatten(start_axis=2).transpose([0, 2, 1])
        if self.class_token:
            cls = paddle.tile(self.cls_token, [x.shape[0], 1, 1])
            x = paddle.concat([cls, x], axis=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return _layer_norm(self.norm, x, self.high_precision)


class JanusVisionModel(nn.Layer):
    def __init__(self, params: Mapping[str, Any] | None = None):
        super().__init__()
        self.vision_tower = JanusVisionTransformer(params)

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        return self.vision_tower(pixel_values)


class JanusMlpProjector(nn.Layer):
    def __init__(self, params: Mapping[str, Any] | None = None, high_precision: bool = False):
        super().__init__()
        values = dict(params or {})
        self.high_precision = high_precision
        input_dim = int(values.get("input_dim", 1024))
        n_embed = int(values.get("n_embed", 4096))
        depth = int(values.get("depth", 1))
        projector_type = values.get("projector_type", "mlp_gelu")
        if projector_type == "identity":
            self.layers = nn.Identity()
        elif projector_type == "linear":
            self.layers = nn.Linear(input_dim, n_embed)
        elif projector_type == "mlp_gelu":
            modules = [nn.Linear(input_dim, n_embed)]
            for _ in range(1, depth):
                modules.append(nn.GELU(approximate="none"))
                modules.append(nn.Linear(n_embed, n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"unsupported Janus projector type: {projector_type}")

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.high_precision and x.dtype == paddle.bfloat16:
            # Preserve the public Sequential/state-dict shape while routing
            # its Linear layers through the same FP32-accumulation helper.
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    x = _linear(layer, x, self.high_precision)
                else:
                    x = layer(x)
            return x
        return self.layers(x)


__all__ = [
    "JanusAttentionPool",
    "JanusMlpProjector",
    "JanusVisionModel",
    "JanusVisionTransformer",
    "_effective_vision_layers",
]
