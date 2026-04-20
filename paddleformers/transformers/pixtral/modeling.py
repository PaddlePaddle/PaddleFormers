# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import Optional

import paddle
from paddle import nn

from ...nn.activation import ACT2FN
from ..model_outputs import BaseModelOutput
from ..model_utils import PretrainedModel
from ..modeling_rope_utils import dynamic_rope_update
from .configuration import PixtralVisionConfig


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        h_grid = paddle.arange(height).unsqueeze(1).tile([1, width]).reshape([-1, 1])
        w_grid = paddle.arange(width).unsqueeze(0).tile([height, 1]).reshape([-1, 1])
        positions.append((h_grid * max_width + w_grid).squeeze(-1))
    return paddle.concat(positions, axis=0)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: paddle.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights.astype("float32"), axis=-1).astype(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = paddle.matmul(attn_weights, value)
    attn_output = attn_output.transpose([0, 2, 1, 3])
    return attn_output, attn_weights


class PixtralRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor

    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        if self.rope_type != "default":
            raise ValueError(f"PixtralVisionModel only supports default RoPE, but got {self.rope_type}")
        inv_freq, _ = self.compute_default_rope_parameters(config)
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(config: PixtralVisionConfig):
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        max_patches_per_side = config.image_size // config.patch_size
        h = paddle.arange(max_patches_per_side, dtype=paddle.float32)
        w = paddle.arange(max_patches_per_side, dtype=paddle.float32)
        freqs = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
        freqs_h = paddle.outer(h, freqs[::2])
        freqs_w = paddle.outer(w, freqs[1::2])
        inv_freq = paddle.concat(
            [
                freqs_h[:, None, :].tile([1, max_patches_per_side, 1]),
                freqs_w[None, :, :].tile([max_patches_per_side, 1, 1]),
            ],
            axis=-1,
        ).reshape([-1, dim // 2])
        inv_freq = paddle.concat((inv_freq, inv_freq), axis=-1)
        return inv_freq, 1.0

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            freqs = self.inv_freq[position_ids]
            cos = freqs.cos()
            sin = freqs.sin()
        return cos.unsqueeze(0).astype(x.dtype), sin.unsqueeze(0).astype(x.dtype)


class PixtralAttention(nn.Layer):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        self.num_key_value_groups = 1

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        batch_size, patches, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape([batch_size, patches, self.num_heads, self.head_dim])
        key_states = self.k_proj(hidden_states).reshape([batch_size, patches, self.num_heads, self.head_dim])
        value_states = self.v_proj(hidden_states).reshape([batch_size, patches, self.num_heads, self.head_dim])

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape([batch_size, patches, self.embed_dim])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PixtralMLP(nn.Layer):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PixtralRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class PixtralAttentionLayer(nn.Layer):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP(config)
        self.attention = PixtralAttention(config)
        self.ffn_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PixtralTransformer(nn.Layer):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.layers = nn.LayerList([PixtralAttentionLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class PixtralPreTrainedModel(PretrainedModel):
    config_class = PixtralVisionConfig
    base_model_prefix = "vision_encoder"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    _no_split_modules = ["PixtralAttentionLayer"]

    def _init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Conv2D)):
            weight = paddle.randn(layer.weight.shape, dtype="float32") * self.config.initializer_range
            layer.weight.set_value(weight.astype(layer.weight.dtype))
            if getattr(layer, "bias", None) is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            weight = paddle.randn(layer.weight.shape, dtype="float32") * self.config.initializer_range
            layer.weight.set_value(weight.astype(layer.weight.dtype))


def generate_block_attention_mask(patch_embeds_list, tensor):
    seq_len = tensor.shape[1]
    d_min = paddle.finfo(tensor.dtype).min
    causal_mask = paddle.full((seq_len, seq_len), fill_value=d_min, dtype=tensor.dtype)
    block_end_idx = paddle.to_tensor(patch_embeds_list, dtype="int64").cumsum(axis=-1)
    block_start_idx = paddle.concat([paddle.zeros([1], dtype="int64"), block_end_idx[:-1]])
    for start, end in zip(block_start_idx.numpy().tolist(), block_end_idx.numpy().tolist()):
        causal_mask[start:end, start:end] = 0
    return causal_mask.unsqueeze(0).unsqueeze(0).tile([tensor.shape[0], 1, 1, 1])


class PixtralVisionModel(PixtralPreTrainedModel):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__(config)
        self.config = config
        self.patch_conv = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias_attr=False,
        )
        self.patch_size = config.patch_size
        self.ln_pre = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformer(config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.patch_conv

    def forward(
        self,
        pixel_values: paddle.Tensor,
        image_sizes: Optional[paddle.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple | BaseModelOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if image_sizes is None:
            batch_size, _, height, width = pixel_values.shape
            image_sizes = paddle.to_tensor([[height, width]] * batch_size, dtype="int64")

        patch_embeds = self.patch_conv(pixel_values.astype(self.patch_conv.weight.dtype))
        patch_embeds_list = [
            embed[:, : size[0] // self.patch_size, : size[1] // self.patch_size]
            for embed, size in zip(patch_embeds, image_sizes.numpy().tolist())
        ]

        patch_embeds = paddle.concat([p.flatten(start_axis=1).transpose([1, 0]) for p in patch_embeds_list], axis=0)
        patch_embeds = patch_embeds.unsqueeze(0)
        patch_embeds = self.ln_pre(patch_embeds)

        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
        )
        position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

        attention_mask = (
            None
            if self.config._attn_implementation != "eager"
            else generate_block_attention_mask([p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds)
        )
        outputs = self.transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            result = (outputs.last_hidden_state,)
            if output_hidden_states:
                result += (outputs.hidden_states,)
            return result

        return outputs


PixtralVision = PixtralVisionModel

__all__ = ["PixtralVision", "PixtralVisionModel", "PixtralPreTrainedModel"]
