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
"""Paddle Molmo model."""

import math
from typing import Callable, Optional, cast

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import MolmoConfig


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def rotate_every_two(x: paddle.Tensor) -> paddle.Tensor:
    shape = x.shape
    x = x.reshape(shape[:-1] + [shape[-1] // 2, 2])
    x1 = x[..., 0]
    x2 = x[..., 1]
    x = paddle.stack((-x2, x1), axis=-1)
    return x.reshape(shape)


def apply_rotary_pos_emb(q, k, cos, sin, rope_impl: str = "interleave", unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotate_fn = rotate_every_two if rope_impl == "interleave" else rotate_half

    q_type, k_type = q.dtype, k.dtype
    q = q.astype(paddle.float32)
    k = k.astype(paddle.float32)
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed.astype(q_type), k_embed.astype(k_type)


class MolmoRotaryEmbedding(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.rope_type = "default"
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_type = config.rope_parameters.get("rope_type", "default")

        rope_init_fn = self._compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def _compute_default_rope_parameters(
        config: Optional[MolmoConfig] = None,
        seq_len: Optional[int] = None,
    ):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            if self.rope_type == "default":
                inv_freq = self._compute_default_rope_parameters(self.config)[0]
            else:
                inv_freq = self.inv_freq.astype(paddle.float32)
            inv_freq_expanded = inv_freq[None, :, None].expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            if getattr(self.config, "rope_impl", "interleave") == "interleave":
                emb = freqs.repeat_interleave(2, axis=-1)
            else:
                emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos, sin


class MolmoRMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.astype(paddle.float32)
            variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
            hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
            hidden_states = hidden_states.astype(input_dtype)
        weight = self.weight.astype(input_dtype) if self.weight.dtype != input_dtype else self.weight
        return weight * hidden_states


def _make_molmo_norm(config: MolmoConfig, hidden_size: int) -> nn.Layer:
    if config.layer_norm_type == "rms":
        return MolmoRMSNorm(hidden_size, config.layer_norm_eps)
    return GeneralNorm.create(
        config=config,
        norm_type="layer_norm",
        hidden_size=hidden_size,
        has_bias=False,
        norm_eps=config.layer_norm_eps,
        input_is_parallel=config.sequence_parallel,
    )


class MolmoAttention(nn.Layer):
    def __init__(self, config: MolmoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        if config.tensor_model_parallel_size > 1:
            assert self.num_heads % config.tensor_model_parallel_size == 0, (
                f"num_heads ({self.num_heads}) must be divisible by "
                f"tensor_model_parallel_size ({config.tensor_model_parallel_size})"
            )
            self.num_heads = self.num_heads // config.tensor_model_parallel_size
            assert self.num_key_value_heads % config.tensor_model_parallel_size == 0, (
                f"num_key_value_heads ({self.num_key_value_heads}) must be divisible by "
                f"tensor_model_parallel_size ({config.tensor_model_parallel_size})"
            )
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        q_hidden_size = self.head_dim * config.num_attention_heads
        kv_hidden_size = self.head_dim * config.num_key_value_heads

        self.q_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size,
            has_bias=config.qkv_bias,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.qkv_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.qkv_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

        self.q_norm: Optional[nn.Layer] = None
        self.k_norm: Optional[nn.Layer] = None
        if config.attention_layer_norm:
            q_norm_size = q_hidden_size
            kv_norm_size = kv_hidden_size
            if config.tensor_model_parallel_size > 1:
                q_norm_size = q_norm_size // config.tensor_model_parallel_size
                kv_norm_size = kv_norm_size // config.tensor_model_parallel_size
            self.q_norm = MolmoRMSNorm(q_norm_size, config.layer_norm_eps)
            self.k_norm = MolmoRMSNorm(kv_norm_size, config.layer_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[paddle.Tensor, list[paddle.Tensor] | None]:
        if self.config.sequence_parallel:
            seq_len = self.config.max_sequence_length
            batch_size = hidden_states.shape[0] * self.config.tensor_model_parallel_size // seq_len
        else:
            batch_size, seq_len = hidden_states.shape[:2]

        q_shape = (batch_size, seq_len, -1, self.head_dim)
        kv_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.q_norm is not None and self.k_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if self.config.clip_qkv is not None:
            query_states = paddle.clip(query_states, min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states = paddle.clip(key_states, min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states = paddle.clip(value_states, min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.reshape(q_shape).transpose([0, 2, 1, 3])
        key_states = key_states.reshape(kv_shape).transpose([0, 2, 1, 3])
        value_states = value_states.reshape(kv_shape).transpose([0, 2, 1, 3])

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            rope_impl=getattr(self.config, "rope_impl", "interleave"),
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MolmoMLP(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.ff_proj = GeneralLinear.create(
            config.hidden_size,
            config.intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.ff_out = GeneralLinear.create(
            config.intermediate_size // 2,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x, gate = self.ff_proj(x).chunk(2, axis=-1)
        return self.ff_out(paddle.nn.functional.silu(gate) * x)


class MolmoDecoderLayer(nn.Layer):
    def __init__(self, config: MolmoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.norm_after = config.norm_after
        self.self_attn = MolmoAttention(config=config, layer_idx=layer_idx)
        self.mlp = MolmoMLP(config)

        self.attn_norm = _make_molmo_norm(config, config.hidden_size)
        self.ff_norm = _make_molmo_norm(config, config.hidden_size)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[paddle.Tensor] | paddle.Tensor:
        residual = hidden_states

        if not self.norm_after:
            hidden_states = self.attn_norm(hidden_states)

        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if self.norm_after:
            attn_out = self.attn_norm(attn_out)

        hidden_states = residual + attn_out

        residual = hidden_states

        if not self.norm_after:
            hidden_states = self.ff_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        if self.norm_after:
            hidden_states = self.ff_norm(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


def _vision_cfg(config: MolmoConfig):
    return config.vision_backbone


def _expand_token(token: paddle.Tensor, batch_size: int) -> paddle.Tensor:
    return token.reshape([1, 1, -1]).expand([batch_size, -1, -1])


def _quick_gelu(x: paddle.Tensor) -> paddle.Tensor:
    return x * F.sigmoid(1.702 * x)


class MolmoVisionLayerNormFp32(nn.LayerNorm):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        input_dtype = x.dtype
        with paddle.amp.auto_cast(False):
            out = F.layer_norm(
                x.astype(paddle.float32),
                self._normalized_shape,
                self.weight.astype(paddle.float32),
                self.bias.astype(paddle.float32),
                self._epsilon,
            )
        return out.astype(input_dtype)


class MolmoVisionMLP(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        v_cfg = _vision_cfg(config)
        self.w1 = nn.Linear(v_cfg["image_emb_dim"], v_cfg["image_mlp_dim"], bias_attr=True)
        self.w2 = nn.Linear(v_cfg["image_mlp_dim"], v_cfg["image_emb_dim"], bias_attr=True)
        self.activation = v_cfg.get("image_mlp_activations", "quick_gelu")

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.w1(x)
        if self.activation == "quick_gelu":
            x = _quick_gelu(x)
        elif self.activation == "gelu":
            x = F.gelu(x, approximate=False)
        else:
            raise NotImplementedError(f"Unknown Molmo vision activation: {self.activation}")
        return self.w2(x)


class MolmoVisionAttention(nn.Layer):
    def __init__(self, config: MolmoConfig, is_vit_layer: bool = True):
        super().__init__()
        self.config = config
        v_cfg = _vision_cfg(config)
        self.embed_dim = v_cfg["image_emb_dim"]
        self.num_heads = v_cfg["image_num_heads"]
        self.head_dim = v_cfg["image_head_dim"]
        self.num_key_value_heads = v_cfg["image_num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_vit_layer = is_vit_layer

        nlayers = 1 if (is_vit_layer or config.vit_layers is None) else len(config.vit_layers)
        self.wq = nn.Linear(nlayers * self.embed_dim, self.num_heads * self.head_dim, bias_attr=True)
        self.wk = nn.Linear(nlayers * self.embed_dim, self.num_key_value_heads * self.head_dim, bias_attr=True)
        self.wv = nn.Linear(nlayers * self.embed_dim, self.num_key_value_heads * self.head_dim, bias_attr=True)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias_attr=True)

    def _split_heads(self, hidden_states: paddle.Tensor, num_heads: int) -> paddle.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + [num_heads, self.head_dim])

    def _merge_heads(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + [self.embed_dim])

    def forward(self, inputs_q: paddle.Tensor, inputs_kv: paddle.Tensor | None = None) -> paddle.Tensor:
        inputs_k = inputs_kv if inputs_kv is not None else inputs_q
        inputs_v = inputs_kv if inputs_kv is not None else inputs_q

        xq = self._split_heads(self.wq(inputs_q), self.num_heads)
        xk = self._split_heads(self.wk(inputs_k), self.num_key_value_heads)
        xv = self._split_heads(self.wv(inputs_v), self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, axis=2)
            xv = xv.repeat_interleave(self.num_key_value_groups, axis=2)

        original_dtype = xq.dtype
        if getattr(self.config, "float32_attention", True):
            xq = xq.astype(paddle.float32)
            xk = xk.astype(paddle.float32)

        attention_type = getattr(self.config, "vision_attention_type", "sdpa")
        if getattr(self.config, "_attn_implementation", None) == "eager":
            attention_type = "direct"
        if attention_type == "direct":
            scores = paddle.einsum("...qhd,...khd->...hqk", xq / math.sqrt(self.head_dim), xk)
            weights = F.softmax(scores, axis=-1).astype(xq.dtype)
            attn_output = paddle.einsum("...hqk,...khd->...qhd", weights.astype(xv.dtype), xv)
        elif attention_type == "sdpa":
            if getattr(self.config, "float32_attention", True):
                xv = xv.astype(paddle.float32)
            attn_output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=0.0,
                is_causal=False,
                training=self.training,
            )
        else:
            raise NotImplementedError(f"Unknown Molmo vision attention type: {attention_type}")

        attn_output = self._merge_heads(attn_output.astype(original_dtype))
        return self.wo(attn_output)


class MolmoVisionResidualAttentionBlock(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        v_cfg = _vision_cfg(config)
        self.attention = MolmoVisionAttention(config)
        self.feed_forward = MolmoVisionMLP(config)
        self.attention_norm = nn.LayerNorm(v_cfg["image_emb_dim"], epsilon=v_cfg["image_norm_eps"])
        self.ffn_norm = nn.LayerNorm(v_cfg["image_emb_dim"], epsilon=v_cfg["image_norm_eps"])

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MolmoVisionBlockCollection(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        v_cfg = _vision_cfg(config)
        self.resblocks = nn.LayerList(
            [MolmoVisionResidualAttentionBlock(config) for _ in range(v_cfg["image_num_layers"])]
        )

    def forward(self, x: paddle.Tensor) -> list[paddle.Tensor]:
        hidden_states = []
        for block in self.resblocks:
            x = block(x)
            hidden_states.append(x)
        return hidden_states


class MolmoVisionTransformer(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config
        v_cfg = _vision_cfg(config)
        self.scale = v_cfg["image_emb_dim"] ** -0.5
        self.class_embedding = self.create_parameter(
            shape=[v_cfg["image_emb_dim"]],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.num_prefix_tokens = 1
        self.positional_embedding = self.create_parameter(
            shape=[v_cfg["image_num_pos"], v_cfg["image_emb_dim"]],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )
        patch_size = v_cfg["image_patch_size"]
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, v_cfg["image_emb_dim"], bias_attr=False)
        self.pre_ln = MolmoVisionLayerNormFp32(v_cfg["image_emb_dim"], epsilon=v_cfg["image_norm_eps"])
        self.transformer = MolmoVisionBlockCollection(config)

    def add_pos_emb(self, x: paddle.Tensor, patch_num: tuple[int, int]) -> paddle.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]
        side = int(math.sqrt(pos_emb.shape[0]))
        pos_emb = pos_emb.reshape([side, side, pos_emb.shape[-1]])
        patch_h, patch_w = patch_num
        if pos_emb.shape[0] != patch_h or pos_emb.shape[1] != patch_w:
            pos_emb = pos_emb.unsqueeze(0).transpose([0, 3, 1, 2])
            pos_emb = F.interpolate(pos_emb, size=[patch_h, patch_w], mode="bicubic", align_corners=False)
            pos_emb = pos_emb.transpose([0, 2, 3, 1]).squeeze(0)
        pos_emb = pos_emb.reshape([-1, pos_emb.shape[-1]])
        emb = paddle.concat([cls_emb.unsqueeze(0), pos_emb.unsqueeze(0)], axis=1).astype(x.dtype)
        return x + emb

    def forward(self, x: paddle.Tensor, patch_num: tuple[int, int] | None = None) -> list[paddle.Tensor]:
        if patch_num is None:
            patch_num = self.config.image_num_patch
        x = self.patch_embedding(x)
        x = paddle.concat([_expand_token(self.class_embedding, x.shape[0]).astype(x.dtype), x], axis=1)
        x = self.add_pos_emb(x, patch_num)
        x = self.pre_ln(x)
        return self.transformer(x)


class MolmoVisionProjectorMLP(nn.Layer):
    def __init__(self, config: MolmoConfig, input_dim: int):
        super().__init__()
        hidden = config.intermediate_size // 2
        self.w1 = nn.Linear(input_dim, hidden, bias_attr=False)
        self.w2 = nn.Linear(hidden, config.hidden_size, bias_attr=False)
        self.w3 = nn.Linear(input_dim, hidden, bias_attr=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MolmoPretrainedVisionBackbone(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.config = config
        v_cfg = _vision_cfg(config)
        self.image_vit = MolmoVisionTransformer(config)
        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = v_cfg["image_emb_dim"] * len(config.vit_layers)
            if config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = self.create_parameter(
                    shape=[2, image_dim],
                    dtype=paddle.get_default_dtype(),
                    default_initializer=nn.initializer.Constant(0.0),
                )
            else:
                raise NotImplementedError(f"Unsupported image_padding_embed: {config.image_padding_embed}")

        if config.image_pooling_2d != "attention-meanq":
            raise NotImplementedError(f"Unsupported image_pooling_2d: {config.image_pooling_2d}")
        self.image_pooling_2d = MolmoVisionAttention(config, is_vit_layer=False)
        self.image_projector = MolmoVisionProjectorMLP(config, v_cfg["image_emb_dim"])

    def encode_image(self, images: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        cfg = self.config
        batch_size, num_crops, num_patch, n_pixels = images.shape
        mask = ~paddle.all(
            images.reshape([batch_size * num_crops, num_patch, n_pixels]) == -1, axis=[1, 2], keepdim=True
        )

        images = images.reshape([batch_size * num_crops, num_patch, n_pixels])
        all_features = self.image_vit(images)
        if cfg.vit_layers is not None:
            features = [all_features[layer] for layer in cfg.vit_layers]
            image_features = paddle.concat(features, axis=-1)
        else:
            image_features = all_features[-1]

        cls_embed = None
        if self.num_prefix_tokens > 0:
            cls_embed = image_features[:, 0]
            image_features = image_features[:, 1:]

        image_features = image_features * mask.astype(image_features.dtype)
        image_features = image_features.reshape([batch_size, num_crops, num_patch, -1])
        if cls_embed is not None:
            cls_embed = cls_embed.reshape([batch_size, num_crops, -1])
        return image_features, cls_embed

    def forward(self, images: paddle.Tensor, image_masks: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        cfg = self.config
        batch_size, num_image = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

        if cfg.image_padding_embed:
            if cfg.image_padding_embed != "pad_and_partial_pad":
                raise NotImplementedError(f"Unsupported image_padding_embed: {cfg.image_padding_embed}")
            pad_embed = self.pad_embed[:, None, None, None, :]
            all_pad = image_masks == 0
            partial_pad = paddle.logical_and(image_masks < 1, paddle.logical_not(all_pad)).astype(image_features.dtype)
            all_pad = all_pad.astype(image_features.dtype)
            image_features = image_features + pad_embed[0] * all_pad.unsqueeze(-1)
            image_features = image_features + pad_embed[1] * partial_pad.unsqueeze(-1)

        image_features = image_features.reshape([batch_size, num_image] + list(cfg.image_num_patch) + [-1])
        if cfg.image_num_patch[0] % cfg.image_pooling_h == 1:
            image_features = F.pad(image_features, [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])

        h_patch, w_patch = cfg.image_num_patch
        h_blocks = (h_patch + cfg.image_pooling_h - 1) // cfg.image_pooling_h
        w_blocks = (w_patch + cfg.image_pooling_w - 1) // cfg.image_pooling_w
        c = image_features.shape[-1]
        image_features = image_features.reshape(
            [
                batch_size,
                num_image,
                h_blocks,
                cfg.image_pooling_h,
                w_blocks,
                cfg.image_pooling_w,
                c,
            ]
        )
        image_features = image_features.transpose([0, 1, 2, 4, 3, 5, 6])
        image_features = image_features.reshape(
            [batch_size * num_image * h_blocks * w_blocks, cfg.image_pooling_h * cfg.image_pooling_w, c]
        )

        query = image_features.mean(axis=-2, keepdim=True)
        image_features = self.image_pooling_2d(query, image_features)
        image_features = image_features.reshape([batch_size, num_image, h_blocks * w_blocks, -1])
        image_features = self.image_projector(image_features)
        return image_features, cls_embed


class MolmoPretrainedModel(PretrainedModel):
    config_class = MolmoConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: MolmoConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        n_kv_groups = n_heads // n_kv_heads if n_kv_heads else 1

        aoa_statements = [
            f"model.transformer.wte.embedding -> {model_prefix}embed_tokens.embedding.weight",
            f"model.transformer.wte.new_embedding -> {model_prefix}embed_tokens.new_embedding.weight",
            f"model.transformer.ln_f.weight -> {model_prefix}norm.weight",
        ]

        if getattr(config, "vision_backbone", None) is not None:
            v_prefix = f"{model_prefix}vision_backbone"
            aoa_statements += [
                f"model.vision_backbone.pad_embed -> {v_prefix}.pad_embed",
                f"model.vision_backbone.image_vit.class_embedding -> {v_prefix}.image_vit.class_embedding",
                f"model.vision_backbone.image_vit.positional_embedding -> {v_prefix}.image_vit.positional_embedding",
                f"model.vision_backbone.image_vit.patch_embedding.weight^T -> {v_prefix}.image_vit.patch_embedding.weight",
                f"model.vision_backbone.image_vit.pre_ln.weight -> {v_prefix}.image_vit.pre_ln.weight",
                f"model.vision_backbone.image_vit.pre_ln.bias -> {v_prefix}.image_vit.pre_ln.bias",
                f"model.vision_backbone.image_pooling_2d.wq.weight^T -> {v_prefix}.image_pooling_2d.wq.weight",
                f"model.vision_backbone.image_pooling_2d.wq.bias -> {v_prefix}.image_pooling_2d.wq.bias",
                f"model.vision_backbone.image_pooling_2d.wk.weight^T -> {v_prefix}.image_pooling_2d.wk.weight",
                f"model.vision_backbone.image_pooling_2d.wk.bias -> {v_prefix}.image_pooling_2d.wk.bias",
                f"model.vision_backbone.image_pooling_2d.wv.weight^T -> {v_prefix}.image_pooling_2d.wv.weight",
                f"model.vision_backbone.image_pooling_2d.wv.bias -> {v_prefix}.image_pooling_2d.wv.bias",
                f"model.vision_backbone.image_pooling_2d.wo.weight^T -> {v_prefix}.image_pooling_2d.wo.weight",
                f"model.vision_backbone.image_pooling_2d.wo.bias -> {v_prefix}.image_pooling_2d.wo.bias",
                f"model.vision_backbone.image_projector.w1.weight^T -> {v_prefix}.image_projector.w1.weight",
                f"model.vision_backbone.image_projector.w2.weight^T -> {v_prefix}.image_projector.w2.weight",
                f"model.vision_backbone.image_projector.w3.weight^T -> {v_prefix}.image_projector.w3.weight",
            ]
            image_num_layers = config.vision_backbone["image_num_layers"]
            for layer_id in range(image_num_layers):
                src = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_id}"
                dst = f"{v_prefix}.image_vit.transformer.resblocks.{layer_id}"
                aoa_statements += [
                    f"{src}.attention.wq.weight^T -> {dst}.attention.wq.weight",
                    f"{src}.attention.wq.bias -> {dst}.attention.wq.bias",
                    f"{src}.attention.wk.weight^T -> {dst}.attention.wk.weight",
                    f"{src}.attention.wk.bias -> {dst}.attention.wk.bias",
                    f"{src}.attention.wv.weight^T -> {dst}.attention.wv.weight",
                    f"{src}.attention.wv.bias -> {dst}.attention.wv.bias",
                    f"{src}.attention.wo.weight^T -> {dst}.attention.wo.weight",
                    f"{src}.attention.wo.bias -> {dst}.attention.wo.bias",
                    f"{src}.feed_forward.w1.weight^T -> {dst}.feed_forward.w1.weight",
                    f"{src}.feed_forward.w1.bias -> {dst}.feed_forward.w1.bias",
                    f"{src}.feed_forward.w2.weight^T -> {dst}.feed_forward.w2.weight",
                    f"{src}.feed_forward.w2.bias -> {dst}.feed_forward.w2.bias",
                    f"{src}.attention_norm.weight -> {dst}.attention_norm.weight",
                    f"{src}.attention_norm.bias -> {dst}.attention_norm.bias",
                    f"{src}.ffn_norm.weight -> {dst}.ffn_norm.weight",
                    f"{src}.ffn_norm.bias -> {dst}.ffn_norm.bias",
                ]

        aoa_statements += [
            f"model.transformer.blocks.$LAYER_ID.attn_norm.weight -> {model_prefix}layers.$LAYER_ID.attn_norm.weight",
            f"model.transformer.blocks.$LAYER_ID.ff_norm.weight -> {model_prefix}layers.$LAYER_ID.ff_norm.weight",
        ]

        if config.attention_layer_norm:
            aoa_statements += [
                f"model.transformer.blocks.$LAYER_ID.q_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.transformer.blocks.$LAYER_ID.k_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            ]

        if n_kv_groups == 1:
            aoa_statements.append(
                f"model.transformer.blocks.$LAYER_ID.att_proj.weight^T -> "
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_proj.weight, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_proj.weight, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.v_proj.weight, axis = 1"
            )
        else:
            aoa_statements.append(
                f"model.transformer.blocks.$LAYER_ID.att_proj.weight^T -> "
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_proj.weight, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_proj.weight, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.v_proj.weight, "
                f"fused_qkv_old, num_heads={n_heads}, num_key_value_groups={n_kv_groups}"
            )

        aoa_statements.append(
            f"model.transformer.blocks.$LAYER_ID.attn_out.weight^T -> "
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight"
        )

        aoa_statements.append(
            f"model.transformer.blocks.$LAYER_ID.ff_proj.weight^T -> "
            f"{model_prefix}layers.$LAYER_ID.mlp.ff_proj.weight"
        )

        aoa_statements.append(
            f"model.transformer.blocks.$LAYER_ID.ff_out.weight^T -> "
            f"{model_prefix}layers.$LAYER_ID.mlp.ff_out.weight"
        )

        if cls != cls.base_model_class:
            if config.weight_tying or config.tie_word_embeddings:
                aoa_statements.append("model.transformer.wte.embedding -> lm_head.weight")
            else:
                aoa_statements.append("model.transformer.ff_out.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: MolmoConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        n_kv_groups = n_heads // n_kv_heads if n_kv_heads else 1

        aoa_statements = [
            f"{model_prefix}embed_tokens.embedding.weight -> model.transformer.wte.embedding",
            f"{model_prefix}embed_tokens.new_embedding.weight -> model.transformer.wte.new_embedding",
            f"{model_prefix}norm.weight -> model.transformer.ln_f.weight",
        ]

        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.attn_norm.weight -> model.transformer.blocks.$LAYER_ID.attn_norm.weight",
            f"{model_prefix}layers.$LAYER_ID.ff_norm.weight -> model.transformer.blocks.$LAYER_ID.ff_norm.weight",
        ]

        if config.attention_layer_norm:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.transformer.blocks.$LAYER_ID.q_norm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.transformer.blocks.$LAYER_ID.k_norm.weight",
            ]

        if n_kv_groups == 1:
            aoa_statements.append(
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_proj.weight^T, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_proj.weight^T, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.v_proj.weight^T -> "
                f"model.transformer.blocks.$LAYER_ID.att_proj.weight, axis = 0"
            )
        else:
            aoa_statements.append(
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_proj.weight^T, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_proj.weight^T, "
                f"{model_prefix}layers.$LAYER_ID.self_attn.v_proj.weight^T -> "
                f"model.transformer.blocks.$LAYER_ID.att_proj.weight, axis = 0"
            )

        aoa_statements.append(
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> "
            f"model.transformer.blocks.$LAYER_ID.attn_out.weight"
        )

        aoa_statements.append(
            f"{model_prefix}layers.$LAYER_ID.mlp.ff_proj.weight^T -> "
            f"model.transformer.blocks.$LAYER_ID.ff_proj.weight"
        )

        aoa_statements.append(
            f"{model_prefix}layers.$LAYER_ID.mlp.ff_out.weight^T -> "
            f"model.transformer.blocks.$LAYER_ID.ff_out.weight"
        )

        if not (config.weight_tying or config.tie_word_embeddings) and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> model.transformer.ff_out.weight")

        return {"aoa_statements": aoa_statements}


class MolmoExtendedEmbedding(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.additional_vocab_size = getattr(config, "additional_vocab_size", 0) or 0

        if config.tensor_model_parallel_size > 1:
            self.embedding = nn.Embedding(self.embedding_size, config.hidden_size)
        else:
            self.embedding = GeneralEmbedding.create(
                config=config,
                num_embeddings=self.embedding_size,
                embedding_dim=config.hidden_size,
            )
        if self.additional_vocab_size > 0:
            if config.tensor_model_parallel_size > 1:
                self.new_embedding = nn.Embedding(self.additional_vocab_size, config.hidden_size)
            else:
                self.new_embedding = GeneralEmbedding.create(
                    config=config,
                    num_embeddings=self.additional_vocab_size,
                    embedding_dim=config.hidden_size,
                )
        else:
            self.new_embedding = None

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.new_embedding is not None:
            base_mask = x < self.embedding_size
            base_ids = paddle.where(base_mask, x, paddle.zeros_like(x))
            extra_ids = paddle.where(base_mask, paddle.zeros_like(x), x - self.embedding_size)
            base_embeds = self.embedding(base_ids)
            extra_embeds = self.new_embedding(extra_ids)
            return paddle.where(base_mask.unsqueeze(-1), base_embeds, extra_embeds)
        return self.embedding(x)

    @property
    def weight(self) -> paddle.Tensor:
        if self.new_embedding is not None:
            return paddle.concat([self.embedding.weight, self.new_embedding.weight], axis=0)
        return self.embedding.weight


class MolmoLMHead(nn.Layer):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        vocab_size = config.embedding_size if config.embedding_size != config.vocab_size else config.vocab_size
        self.weight = self.create_parameter(
            shape=[vocab_size, config.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierNormal(1.0),
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.matmul(hidden_states, self.weight, transpose_y=True)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.weight.shape[1]}, "
            f"vocab_size={self.weight.shape[0]}, "
            f"dtype={self.weight.dtype}, vocab_parallel=False"
        )


@register_base_model
class MolmoModel(MolmoPretrainedModel):
    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = MolmoExtendedEmbedding(config)
        self.layers = nn.LayerList(
            [MolmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = _make_molmo_norm(config, config.hidden_size)
        self.rotary_emb = MolmoRotaryEmbedding(config=config)
        self.vision_backbone = None
        if getattr(config, "vision_backbone", None) is not None:
            self.vision_backbone = MolmoPretrainedVisionBackbone(config)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        images: paddle.Tensor | None = None,
        image_masks: paddle.Tensor | None = None,
        image_input_idx: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = False,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        has_image = images is not None
        if has_image and inputs_embeds is not None:
            raise ValueError("Cannot provide both images and inputs_embeds.")
        if has_image and past_key_values is not None:
            raise ValueError("Cached key and values should not be used with images.")

        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.embedding.weight.dtype)
        inputs_embeds = cast(paddle.Tensor, inputs_embeds)
        bsz, seq_length, _ = inputs_embeds.shape

        if images is not None:
            if self.vision_backbone is None:
                raise ValueError("This Molmo config does not define a vision backbone.")
            if image_masks is None or image_input_idx is None:
                raise ValueError("images requires image_masks and image_input_idx.")
            image_features, _ = self.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            if list(image_input_idx.shape) != [bsz, num_image, num_patch]:
                raise ValueError(
                    f"image_input_idx shape {list(image_input_idx.shape)} does not match "
                    f"[batch={bsz}, num_image={num_image}, num_patch={num_patch}]"
                )
            image_features = image_features.reshape([bsz, num_image * num_patch, -1])
            image_input_idx = image_input_idx.reshape([bsz, num_image * num_patch])
            valid = image_input_idx >= 0
            safe_idx = paddle.where(valid, image_input_idx, paddle.zeros_like(image_input_idx))
            safe_idx = safe_idx.unsqueeze(-1).expand([-1, -1, inputs_embeds.shape[-1]])
            image_features = image_features.astype(inputs_embeds.dtype) * valid.unsqueeze(-1).astype(
                inputs_embeds.dtype
            )
            image_delta = paddle.zeros_like(inputs_embeds)
            image_delta = paddle.put_along_axis(image_delta, safe_idx, image_features, axis=1, reduce="add")
            inputs_embeds = inputs_embeds + image_delta

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(kv_seq_len, seq_length + kv_seq_len, dtype=paddle.int64).unsqueeze(0).tile((bsz, 1))
            )

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_length,
            "cache_length": kv_seq_len,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        all_hidden_states = [] if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    position_embeddings,
                    past_key_values,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        all_hidden_states = tuple(all_hidden_states) if all_hidden_states else None

        if not return_dict:
            outputs = [hidden_states]
            if output_hidden_states:
                outputs.append(all_hidden_states)
            return tuple(outputs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None,
        attn_mask_startend_row_indices: paddle.Tensor | None,
        position_ids: paddle.Tensor,
        position_embeddings: tuple,
        past_key_values: Cache | None,
        use_cache: bool,
    ):
        cos, sin = position_embeddings
        cos = cos.clone()
        sin = sin.clone()
        position_embeddings_safe = (cos, sin)
        hidden_states = recompute(
            layer_module,
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings_safe,
            past_key_values,
            use_cache,
            use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states


class MolmoForCausalLM(MolmoPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        self.config = config
        self.model = MolmoModel(config)
        lm_head_config = config
        if config.embedding_size != config.vocab_size:
            from copy import copy as _copy

            lm_head_config = _copy(config)
            lm_head_config.vocab_size = config.embedding_size
        if config.tensor_model_parallel_size > 1:
            self.lm_head = MolmoLMHead(config)
        else:
            self.lm_head = GeneralLMHead(lm_head_config)
        self.criterion = CriterionLayer(lm_head_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens.embedding

    def set_input_embeddings(self, value):
        self.model.embed_tokens.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        images: paddle.Tensor | None = None,
        image_masks: paddle.Tensor | None = None,
        image_input_idx: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Cache | None = None,
        output_hidden_states: bool | None = False,
        return_dict: bool = False,
        **kwargs,
    ):
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "Both attn_mask_startend_row_indices and attention_mask provided; "
                "attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            images=images if images is not None else kwargs.pop("images", None),
            image_masks=image_masks if image_masks is not None else kwargs.pop("image_masks", None),
            image_input_idx=image_input_idx if image_input_idx is not None else kwargs.pop("image_input_idx", None),
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MolmoForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = MolmoConfig
    _decoder_layer_cls = MolmoDecoderLayer
    _get_tensor_parallel_mappings = MolmoModel._get_tensor_parallel_mappings
    _init_weights = MolmoModel._init_weights
    _keep_in_fp32_modules = MolmoModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = MolmoModel.transpose_weight_keys
    _gen_aoa_config = MolmoForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = MolmoForCausalLM._gen_inv_aoa_config


__all__ = [
    "MolmoPretrainedModel",
    "MolmoModel",
    "MolmoForCausalLM",
    "MolmoForCausalLMPipe",
]
