# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Phi-4-Multimodal model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...utils.log import logger
from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS
from .configuration import (
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalVisionConfig,
)

# ======================= Utility functions =======================


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = paddle.concat([q_embed, q_pass], axis=-1)
    k_embed = paddle.concat([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


def _create_lora_parameter(layer: nn.Layer, shape):
    return layer.create_parameter(
        shape=shape,
        default_initializer=nn.initializer.Constant(0.0),
    )


def _lora_delta(hidden_states: paddle.Tensor, lora_a: paddle.Tensor, lora_b: paddle.Tensor, scaling: float):
    input_dtype = hidden_states.dtype
    delta = F.linear(hidden_states, lora_a.astype(input_dtype))
    delta = F.linear(delta, lora_b.astype(input_dtype))
    return delta * scaling


def _active_lora_adapter(config):
    adapter = getattr(config, "_active_lora_adapter", None)
    if adapter in ("vision", "speech"):
        return adapter
    return None


def _lora_adapter_from_input_mode(input_mode, image_pixel_values=None, audio_input_features=None):
    if input_mode is not None:
        if isinstance(input_mode, paddle.Tensor):
            input_modes = paddle.unique(input_mode.flatten()).tolist()
            if len(input_modes) != 1:
                raise ValueError("Phi-4 multimodal does not support mixing different input modes in the same batch.")
            input_mode = int(input_modes[0])
        if input_mode in (1, 3):
            return "vision"
        if input_mode == 2:
            return "speech"
        return None
    if image_pixel_values is not None:
        return "vision"
    if audio_input_features is not None:
        return "speech"
    return None


def _merge_multimodal_embeddings(
    input_ids: paddle.Tensor,
    inputs_embeds: paddle.Tensor,
    multimodal_embeds: paddle.Tensor,
    token_id: int,
    modality: str,
) -> paddle.Tensor:
    """Replace multimodal placeholder embeddings without a Python loop over tokens."""
    placeholder_mask = input_ids == token_id
    expanded_mask = placeholder_mask.unsqueeze(-1).expand_as(inputs_embeds)
    if inputs_embeds[expanded_mask].numel() != multimodal_embeds.numel():
        raise ValueError(
            f"{modality} features and {modality} tokens do not match: "
            f"tokens={placeholder_mask.sum().item()}, features={multimodal_embeds.shape[0]}."
        )
    return inputs_embeds.masked_scatter(expanded_mask, multimodal_embeds)


# ======================= Vision Encoder =======================


class Phi4MultimodalVisionMLP(nn.Layer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Phi4MultimodalVisionAttention(nn.Layer):
    # Keep the explicit eager operator order aligned with upstream
    # SiglipAttention. The upstream eager path does not use PyTorch SDPA.
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = list(input_shape) + [self.num_heads, self.head_dim]

        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1, dtype=paddle.float32).astype(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        attn_output = attn_output.reshape(list(input_shape) + [-1])
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Phi4MultimodalVisionEncoderLayer(nn.Layer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.self_attn = Phi4MultimodalVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = Phi4MultimodalVisionMLP(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Phi4MultimodalVisionEncoder(nn.Layer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.LayerList([Phi4MultimodalVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    @paddle.jit.not_to_static
    def recompute_training_full(self, layer_module, hidden_states, attention_mask):
        return recompute(layer_module, hidden_states, attention_mask)

    def forward(
        self,
        inputs_embeds: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(encoder_layer, hidden_states, attention_mask)
            else:
                hidden_states = encoder_layer(hidden_states, attention_mask)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states


class Phi4MultimodalVisionEmbeddings(nn.Layer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches_per_side = config.image_size // self.patch_size

        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.position_embedding = nn.Embedding(self.num_patches_per_side**2, config.hidden_size)

    def forward(self, pixel_values: paddle.Tensor, patch_attention_mask: paddle.Tensor) -> paddle.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose([0, 2, 1])

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = paddle.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side).astype(
            pixel_values.dtype
        )
        position_ids = paddle.full(
            shape=[batch_size, max_nb_patches_h * max_nb_patches_w], fill_value=0, dtype="int64"
        )

        nb_patches_h = patch_attention_mask[:, :, 0].astype("int64").sum(axis=1)
        nb_patches_w = patch_attention_mask[:, 0, :].astype("int64").sum(axis=1)

        step_h = 1.0 / nb_patches_h.astype("float32")
        step_w = 1.0 / nb_patches_w.astype("float32")

        max_patches_h = patch_attention_mask.shape[1]
        max_patches_w = patch_attention_mask.shape[2]
        h_indices = paddle.arange(max_patches_h, dtype="float32")
        w_indices = paddle.arange(max_patches_w, dtype="float32")

        fractional_coords_h = h_indices[None, :] * step_h[:, None]
        fractional_coords_w = w_indices[None, :] * step_w[:, None]

        fractional_coords_h = paddle.clip(fractional_coords_h, max=(1.0 - 1e-6))
        fractional_coords_w = paddle.clip(fractional_coords_w, max=(1.0 - 1e-6))

        fractional_coords_h = fractional_coords_h.astype(pixel_values.dtype)
        fractional_coords_w = fractional_coords_w.astype(pixel_values.dtype)

        bucket_coords_h = paddle.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = paddle.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = bucket_coords_h[:, :, None] * self.num_patches_per_side + bucket_coords_w[:, None, :]
        pos_ids = pos_ids.reshape([batch_size, -1])

        flat_mask = patch_attention_mask.reshape([batch_size, -1]).astype("bool")
        for i in range(batch_size):
            mask_i = flat_mask[i]
            position_ids[i][mask_i] = pos_ids[i][mask_i]

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Phi4MultimodalVisionMultiheadAttentionPoolingHead(nn.Layer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.probe = self.create_parameter(
            shape=[1, 1, config.hidden_size],
            default_initializer=nn.initializer.Normal(std=1.0),
        )
        self.attention = nn.MultiHeadAttention(config.hidden_size, config.num_attention_heads)
        self.layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = Phi4MultimodalVisionMLP(config)

    def forward(self, hidden_state: paddle.Tensor, attention_mask: paddle.Tensor) -> paddle.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.expand([batch_size, -1, -1])

        # attention_mask: [B, S] bool -> key_padding_mask for MHA
        # Paddle MHA uses attn_mask, we need to convert
        # ~attention_mask gives True for padded positions
        key_padding_mask = ~attention_mask.astype("bool")
        # Convert to float mask: 0 for valid, -inf for padded
        attn_mask = key_padding_mask.astype(hidden_state.dtype) * paddle.finfo(hidden_state.dtype).min
        attn_mask = attn_mask.unsqueeze([1, 2])  # [B, 1, 1, S]

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attn_mask)

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Phi4MultimodalVisionModel(nn.Layer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Phi4MultimodalVisionEmbeddings(config)
        self.encoder = Phi4MultimodalVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.head = Phi4MultimodalVisionMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values: paddle.Tensor,
        patch_attention_mask: Optional[paddle.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        batch_size = pixel_values.shape[0]
        if patch_attention_mask is None:
            patch_attention_mask = paddle.ones(
                shape=[
                    batch_size,
                    pixel_values.shape[2] // self.config.patch_size,
                    pixel_values.shape[3] // self.config.patch_size,
                ],
                dtype="bool",
            )

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask_flat = patch_attention_mask.reshape([batch_size, -1])
        # Create bidirectional attention mask
        mask_expanded = patch_attention_mask_flat.unsqueeze([1, 2]).astype(hidden_states.dtype)
        attention_mask = (1.0 - mask_expanded) * paddle.finfo(hidden_states.dtype).min

        last_hidden_state, all_hidden_states = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(
            hidden_state=last_hidden_state,
            attention_mask=patch_attention_mask_flat,
        )

        return last_hidden_state, pooled_output, all_hidden_states


class Phi4MultimodalImageEmbedding(nn.Layer):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.vision_config.feature_layer
        self.crop_size = config.vision_config.crop_size
        self.image_dim_out = config.vision_config.hidden_size

        n_patches = config.vision_config.image_size // config.vision_config.patch_size
        if n_patches % 2 != 0:
            self.img_processor_padding = nn.Pad2D([0, 1, 0, 1], mode="reflect")
            n_patches += 1
        self.num_img_tokens = (n_patches // 2) ** 2

        self.drop = nn.Dropout(config.embd_pdrop)
        self.img_processor = Phi4MultimodalVisionModel(config.vision_config)
        self.image_token_compression = nn.AvgPool2D(kernel_size=2, stride=2)
        self.img_projection_up = nn.Linear(self.image_dim_out, config.hidden_size)
        self.img_projection_down = nn.Linear(config.hidden_size, config.hidden_size)
        self.global_img_feature_extensor = self.create_parameter(
            shape=[1, 1, self.image_dim_out],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.sub_img_feature_extensor = self.create_parameter(
            shape=[1, 1, 1, self.image_dim_out],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def _repeat_sub_img_feature_extensor(self, repeat_height: int) -> paddle.Tensor:
        return paddle.tile(self.sub_img_feature_extensor, repeat_times=[1, repeat_height, 1, 1])

    def get_img_features(self, img_embeds: paddle.Tensor, attention_mask=None) -> paddle.Tensor:
        _, _, all_hidden_states = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask, output_hidden_states=True
        )
        img_feature = all_hidden_states[self.layer_idx]

        patch_feature = img_feature
        width = int(math.sqrt(patch_feature.shape[1]))
        patch_feature = patch_feature.reshape([-1, width, width, patch_feature.shape[-1]])
        # convert to NCHW
        patch_feature = patch_feature.transpose([0, 3, 1, 2])
        if hasattr(self, "img_processor_padding"):
            patch_feature = self.img_processor_padding(patch_feature)
        patch_feature = self.image_token_compression(patch_feature)
        # convert to NHWC
        patch_feature = patch_feature.transpose([0, 2, 3, 1])
        patch_feature = patch_feature.reshape(
            [-1, patch_feature.shape[1] * patch_feature.shape[2], patch_feature.shape[-1]]
        )
        return patch_feature

    def forward(
        self,
        input_ids: paddle.Tensor,
        inputs_embeds: paddle.Tensor,
        image_pixel_values: paddle.Tensor,
        image_sizes: Optional[paddle.Tensor] = None,
        image_attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        image_pixel_values = image_pixel_values.astype(self.img_processor.embeddings.patch_embedding.weight.dtype)

        target_dtype = self.img_projection_up.bias.dtype

        batch_size = image_pixel_values.shape[0]

        img_features = self.get_img_features(
            image_pixel_values.flatten(0, 1),
            attention_mask=image_attention_mask.flatten(0, 1).astype("bool")
            if image_attention_mask is not None
            else None,
        )
        base_feat_size = int(np.sqrt(img_features.shape[1]))
        img_features = img_features.reshape([batch_size, -1, base_feat_size**2, self.image_dim_out])
        image_sizes_flat = image_sizes.reshape([-1, 2])

        output_imgs = []
        for idx in range(batch_size):
            height = int(image_sizes_flat[idx, 0].item())
            width = int(image_sizes_flat[idx, 1].item())
            height_ratio = height // self.crop_size
            width_ratio = width // self.crop_size
            area_ratio = height_ratio * width_ratio

            global_img = img_features[idx, :1]
            global_img = global_img.reshape([1, base_feat_size, base_feat_size, self.image_dim_out])
            temporary_extensor = self._repeat_sub_img_feature_extensor(base_feat_size)
            global_img = paddle.concat([global_img, temporary_extensor], axis=2).reshape([1, -1, self.image_dim_out])

            sub_img = img_features[idx, 1:]
            sub_img = sub_img[:area_ratio]
            sub_img = (
                sub_img.reshape([height_ratio, width_ratio, base_feat_size, base_feat_size, self.image_dim_out])
                .transpose([0, 2, 1, 3, 4])
                .reshape([1, height_ratio * base_feat_size, width_ratio * base_feat_size, self.image_dim_out])
            )

            if image_attention_mask is not None:
                reshaped_image_attention_mask = (
                    image_attention_mask[idx, 1 : area_ratio + 1, 0::2, 0::2]
                    .reshape([height_ratio, width_ratio, base_feat_size, base_feat_size])
                    .transpose([0, 2, 1, 3])
                    .reshape([1, height_ratio * base_feat_size, width_ratio * base_feat_size])
                )
                reshaped_image_attention_mask_int = reshaped_image_attention_mask.astype("int64")
                useful_height = int(reshaped_image_attention_mask_int[0, :, 0].sum().item())
                useful_width = int(reshaped_image_attention_mask_int[0, 0, :].sum().item())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temporary_extensor = self._repeat_sub_img_feature_extensor(useful_height)
            else:
                temporary_extensor = self._repeat_sub_img_feature_extensor(height_ratio * base_feat_size)

            sub_img = paddle.concat([sub_img, temporary_extensor], axis=2).reshape([1, -1, self.image_dim_out])

            output_imgs.append(paddle.concat([sub_img, self.global_img_feature_extensor, global_img], axis=1))

        img_set_tensor = []
        for output_img in output_imgs:
            output_img = output_img.astype(target_dtype)
            img_feature_proj = self.img_projection_up(output_img)
            img_feature_proj = F.gelu(img_feature_proj)
            img_feature_proj = self.img_projection_down(img_feature_proj)
            img_set_tensor.append(img_feature_proj)

        merged_img_set_tensor = paddle.concat(img_set_tensor, axis=1).squeeze(0)
        merged_img_set_tensor = merged_img_set_tensor.astype(inputs_embeds.dtype)

        image_embeds = _merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            merged_img_set_tensor,
            self.config.vision_config.image_token_id,
            "Image",
        )

        image_embeds = self.drop(image_embeds)
        return image_embeds


# ======================= Audio Encoder =======================


class Phi4MultimodalAudioMLP(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.activation]
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        up_states = self.gate_up_proj(hidden_states)
        up_states, gate = up_states.chunk(2, axis=-1)
        up_states = up_states * self.act_fn(gate)
        up_states = self.dropout(up_states)
        hidden_states = self.down_proj(up_states)
        out = self.dropout(hidden_states)
        return out


class Phi4MultimodalAudioAttention(nn.Layer):
    # Upstream Conformer defaults to its explicit relative-position attention
    # path (use_pt_scaled_dot_product_attention=False). Keep the same eager
    # math and masked-softmax semantics instead of using text-model attention.
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.dropout_rate

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size)

    @staticmethod
    def _masked_softmax(attn_weights):
        blocked_positions = paddle.isinf(attn_weights) & (attn_weights < 0)
        attn_weights = F.softmax(attn_weights, axis=-1, dtype=paddle.float32)
        attn_weights = paddle.where(blocked_positions, paddle.zeros_like(attn_weights), attn_weights)
        normalizer = attn_weights.sum(axis=-1, keepdim=True)
        return attn_weights / paddle.clip(normalizer, min=1e-9)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
    ) -> paddle.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = list(input_shape) + [self.num_heads, self.head_dim]

        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose([0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = self._masked_softmax(attn_weights).astype(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape(list(input_shape) + [-1])
        attn_output = self.o_proj(attn_output)
        return attn_output


class Phi4MultimodalAudioDepthWiseSeparableConv1d(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig, padding: int = 0):
        super().__init__()
        self.dw_conv = nn.Conv1D(
            config.hidden_size,
            config.hidden_size * config.depthwise_multiplier,
            config.kernel_size,
            stride=1,
            padding=padding,
            groups=config.hidden_size,
        )
        self.pw_conv = nn.Conv1D(
            config.hidden_size * config.depthwise_multiplier, config.depthwise_separable_out_channel, 1, 1, 0
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return self.pw_conv(self.dw_conv(hidden_states))


class Phi4MultimodalAudioGluPointWiseConv(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.output_dim = config.ext_pw_out_channel

        self.ext_pw_conv_1d = nn.Conv1D(config.hidden_size, config.ext_pw_out_channel * 2, kernel_size=1, stride=1)
        self.glu_act = ACT2FN[config.conv_glu_type]
        self.b1 = self.create_parameter(
            shape=[1, config.ext_pw_out_channel, 1],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.b2 = self.create_parameter(
            shape=[1, config.ext_pw_out_channel, 1],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = hidden_states.transpose([0, 2, 1])
        hidden_states = self.ext_pw_conv_1d(hidden_states)
        out = hidden_states[:, 0 : self.output_dim, :] + self.b1
        out = out * self.glu_act(hidden_states[:, self.output_dim : self.output_dim * 2, :] + self.b2)
        return out.transpose([0, 2, 1])


class Phi4MultimodalAudioConvModule(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.kernel_size = config.kernel_size

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.glu = Phi4MultimodalAudioGluPointWiseConv(config)
        self.dw_sep_conv_1d = Phi4MultimodalAudioDepthWiseSeparableConv1d(config, padding=config.kernel_size - 1)
        self.act = ACT2FN[config.conv_activation]
        self.ext_pw_conv_1d = nn.Conv1D(config.hidden_size, config.ext_pw_out_channel, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.glu(self.layer_norm(hidden_states))
        hidden_states = self.dw_sep_conv_1d(hidden_states.transpose([0, 2, 1]))

        if self.kernel_size > 1:
            hidden_states = hidden_states[:, :, : -(self.kernel_size - 1)]

        hidden_states = self.act(hidden_states)
        hidden_states = self.ext_pw_conv_1d(hidden_states)
        out = self.dropout(hidden_states.transpose([0, 2, 1]))
        return out


class Phi4MultimodalAudioConformerEncoderLayer(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.feed_forward_in = Phi4MultimodalAudioMLP(config)
        self.self_attn = Phi4MultimodalAudioAttention(config)
        self.conv = Phi4MultimodalAudioConvModule(config)
        self.feed_forward_out = Phi4MultimodalAudioMLP(config)
        self.layer_norm_att = nn.LayerNorm(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
    ) -> paddle.Tensor:
        residual = hidden_states + 0.5 * self.feed_forward_in(hidden_states)
        hidden_states = self.layer_norm_att(residual)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask)
        hidden_states = hidden_states + self.conv(hidden_states)
        hidden_states = hidden_states + 0.5 * self.feed_forward_out(hidden_states)
        out = self.layer_norm(hidden_states)
        return out


class Phi4MultimodalAudioNemoConvSubsampling(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.subsampling_factor = config.time_reduction
        self.sampling_num = int(math.log2(self.subsampling_factor))
        self.act_fn = ACT2FN[config.nemo_activation]
        conv_channels = config.nemo_conv_channels

        layers = [
            nn.Conv2D(1, conv_channels, kernel_size=3, stride=2, padding=1),
            self.act_fn,
        ]
        for _ in range(self.sampling_num - 1):
            layers.extend(
                [
                    nn.Conv2D(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1, groups=conv_channels),
                    nn.Conv2D(conv_channels, conv_channels, kernel_size=1, stride=1, padding=0, groups=1),
                    self.act_fn,
                ]
            )

        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(conv_channels * config.nemo_final_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor, mask: Optional[paddle.Tensor]):
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = self.conv(hidden_states)

        b, _, t, _ = hidden_states.shape
        hidden_states = self.out(hidden_states.transpose([0, 2, 1, 3]).reshape([b, t, -1]))

        if mask is None:
            return hidden_states, None

        max_audio_length = hidden_states.shape[1]
        feature_lens = mask.sum(1)
        padding_length = paddle.ceil(feature_lens / self.subsampling_factor).astype("int64")
        arange_ = paddle.arange(0, max_audio_length, dtype="int64")
        pad_mask = arange_.expand([padding_length.shape[0], -1]) < padding_length.unsqueeze(1)
        return hidden_states, pad_mask.unsqueeze(1)


class Phi4MultimodalAudioRelativeAttentionBias(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.max_distance = config.bias_max_distance
        self.symmetric = config.bias_symmetric
        self.num_buckets = self.max_distance
        if not config.bias_symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, config.num_attention_heads)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        max_pos = x.shape[1]
        context_position = paddle.arange(max_pos, dtype="int64")[:, None]
        memory_position = paddle.arange(max_pos, dtype="int64")[None, :]
        relative_position = memory_position - context_position

        relative_position = paddle.clip(relative_position, min=-self.max_distance, max=self.max_distance - 1)

        if self.symmetric:
            bias_idx = paddle.abs(relative_position)
        else:
            bias_idx = relative_position + self.num_buckets // 2

        att_bias = self.bias_values(bias_idx)
        att_bias = att_bias.transpose([2, 0, 1]).unsqueeze(0)
        return att_bias


class Phi4MultimodalAudioMeanVarianceNormLayer(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.register_buffer("global_mean", paddle.zeros([config.input_size]))
        self.register_buffer("global_invstd", paddle.ones([config.input_size]))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return (x - self.global_mean) * self.global_invstd


def unfold_tensor(tensor: paddle.Tensor, max_seq_len: int) -> paddle.Tensor:
    _, T, D = tensor.shape
    n_chunks = T // max_seq_len
    tensor = tensor[:, : n_chunks * max_seq_len, :]
    tensor = tensor.reshape([-1, max_seq_len, D])
    return tensor


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    chunk_start_idx_t = paddle.to_tensor(chunk_start_idx, dtype="int64")
    start_pad = paddle.concat([paddle.zeros([1], dtype="int64"), chunk_start_idx_t])
    end_pad = paddle.concat([chunk_start_idx_t, paddle.full([1], x_len, dtype="int64")])

    seq_range = paddle.arange(x_len, dtype="int64").unsqueeze(-1)
    idx = paddle.nonzero((seq_range < end_pad) & (seq_range >= start_pad))[:, 1]
    seq_range_expand = paddle.arange(x_len, dtype="int64").unsqueeze(0).expand([x_len, x_len])

    idx_left = paddle.clip(idx - left_window, min=0)
    idx_right = paddle.clip(idx + right_window, max=len(chunk_start_idx))
    boundary_left = paddle.gather(start_pad, idx_left)
    boundary_right = paddle.gather(end_pad, idx_right)
    return (seq_range_expand >= boundary_left.unsqueeze(-1)) & (seq_range_expand < boundary_right.unsqueeze(-1))


class Phi4MultimodalAudioModel(nn.Layer):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.encoder_embedding = Phi4MultimodalAudioMeanVarianceNormLayer(config)
        self.embed = Phi4MultimodalAudioNemoConvSubsampling(config)
        self.relative_attention_bias_layer = Phi4MultimodalAudioRelativeAttentionBias(config)
        self.encoders = nn.LayerList(
            [Phi4MultimodalAudioConformerEncoderLayer(config) for _ in range(config.num_blocks)]
        )

    @paddle.jit.not_to_static
    def recompute_training_full(self, layer_module, hidden_states, attention_mask):
        return recompute(layer_module, hidden_states, attention_mask)

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_start_idx = np.arange(0, seq_len, chunk_size)
        if self.training and np.random.rand() > 0.5:
            chunk_start_idx = seq_len - chunk_start_idx
            chunk_start_idx = chunk_start_idx[::-1]
            chunk_start_idx = chunk_start_idx[:-1]
            chunk_start_idx = np.insert(chunk_start_idx, 0, 0)

        enc_streaming_mask = adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk)
        enc_streaming_mask = enc_streaming_mask.unsqueeze(0).expand([batch_size, -1, -1])
        return enc_streaming_mask

    def forward_embeddings(self, hidden_states, masks):
        seq_len = math.ceil(hidden_states.shape[1] / self.config.time_reduction)
        if seq_len <= 0:
            raise ValueError(
                f"The sequence length after time reduction is invalid: {seq_len}. Your input feature is too short."
            )
        batch_size = hidden_states.shape[0]
        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.config.chunk_size, self.config.left_chunk)

        hidden_states, masks = self.embed(hidden_states, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks.astype("bool") & streaming_mask.astype("bool")
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        return hidden_states, hs_mask, masks

    def calculate_hs_mask(self, hidden_states, mask):
        max_audio_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        enc_streaming_mask = self._streaming_mask(
            max_audio_length, batch_size, self.config.chunk_size, self.config.left_chunk
        )
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = paddle.arange(0, max_audio_length).expand([padding_length.shape[0], -1]) < padding_length.unsqueeze(
            1
        )
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask.astype("bool") & enc_streaming_mask.astype("bool")
        return pad_mask

    @staticmethod
    def _prepare_attention_mask(hs_mask, relative_attention_bias):
        if hs_mask is None:
            return relative_attention_bias

        hs_mask = hs_mask.unsqueeze(1)
        additive_mask = paddle.where(
            hs_mask,
            paddle.zeros_like(hs_mask, dtype=relative_attention_bias.dtype),
            paddle.full_like(
                hs_mask,
                float("-inf"),
                dtype=relative_attention_bias.dtype,
            ),
        )
        return additive_mask + relative_attention_bias

    def forward(self, hidden_states: paddle.Tensor, mask: Optional[paddle.Tensor] = None, **kwargs):
        hidden_states = self.encoder_embedding(hidden_states)
        hidden_states, hs_mask, mask = self.forward_embeddings(hidden_states, mask)

        unfolded = False
        bs, seq_len, _ = hidden_states.shape
        max_seq_len = 500
        if seq_len > max_seq_len:
            unfolded = True
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                hidden_states = F.pad(hidden_states, [0, 0, 0, chunk_pad_size], data_format="NLC")

            hidden_states = unfold_tensor(hidden_states, max_seq_len)
            masks_unfold = None
            if mask is not None:
                subsampled_pad_mask = mask.squeeze(1)
                extra_padded = F.pad(subsampled_pad_mask.astype("float32"), [0, chunk_pad_size])
                extra_padded = extra_padded.unsqueeze(-1)
                masks_unfold = unfold_tensor(extra_padded, max_seq_len)
                masks_unfold = masks_unfold.squeeze(-1).astype("bool")
            hs_mask = self.calculate_hs_mask(hidden_states, masks_unfold)

        relative_attention_bias = self.relative_attention_bias_layer(hidden_states)
        attention_mask = self._prepare_attention_mask(hs_mask, relative_attention_bias)

        for layer in self.encoders:
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(layer, hidden_states, attention_mask)
            else:
                hidden_states = layer(hidden_states, attention_mask)

        if unfolded:
            embed_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape([bs, -1, embed_dim])
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states


class Phi4MultimodalAudioEmbedding(nn.Layer):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.audio_config.feature_layer

        self.drop = nn.Dropout(config.embd_pdrop)
        self.encoder = Phi4MultimodalAudioModel(config.audio_config)
        self.up_proj_for_speech = nn.Linear(
            config.audio_config.hidden_size * config.audio_config.downsample_rate, config.hidden_size
        )
        self.down_proj_for_speech = nn.Linear(config.hidden_size, config.hidden_size)
        self.up_proj_for_vision_speech = nn.Linear(
            config.audio_config.hidden_size * config.audio_config.downsample_rate, config.hidden_size
        )
        self.down_proj_for_vision_speech = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: paddle.Tensor,
        inputs_embeds: paddle.Tensor,
        audio_input_features: paddle.Tensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
    ) -> paddle.Tensor:
        up_proj = self.up_proj_for_speech if audio_projection_mode == "speech" else self.up_proj_for_vision_speech
        down_proj = (
            self.down_proj_for_speech if audio_projection_mode == "speech" else self.down_proj_for_vision_speech
        )

        target_dtype = up_proj.bias.dtype
        audio_input_features = audio_input_features.astype(target_dtype)

        audio_encoder_hidden_states = self.encoder(audio_input_features, audio_attention_mask)
        audio_encoder_hidden_states = up_proj(audio_encoder_hidden_states)
        audio_encoder_hidden_states = F.gelu(audio_encoder_hidden_states)
        audio_embeds = down_proj(audio_encoder_hidden_states)

        merged_audio_embeds = paddle.concat(
            [audio_embeds[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))], axis=0
        )
        merged_audio_embeds = merged_audio_embeds.astype(inputs_embeds.dtype)

        audio_embeds_out = _merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            merged_audio_embeds,
            self.config.audio_config.audio_token_id,
            "Audio",
        )

        audio_embeds_out = self.drop(audio_embeds_out)
        return audio_embeds_out


# ======================= Text Decoder =======================


class Phi4MultimodalRMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        with paddle.amp.auto_cast(enable=False):
            hidden_states = hidden_states.astype(paddle.float32)
            variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
            hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
            hidden_states = hidden_states.astype(input_dtype)
        weight = self.weight.astype(input_dtype) if self.weight.dtype != input_dtype else self.weight
        return weight * hidden_states


class Phi4MultimodalMLP(nn.Layer):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.gate_up_proj = GeneralLinear.create(
            config.hidden_size,
            2 * config.intermediate_size,
            has_bias=config.mlp_bias,
            config=config,
            tp_plan="colwise",
        )
        self.down_proj = GeneralLinear.create(
            config.intermediate_size,
            config.hidden_size,
            has_bias=config.mlp_bias,
            config=config,
            tp_plan="rowwise",
        )
        self.activation_fn = ACT2FN[config.hidden_act]
        self._init_lora(config.hidden_size, 2 * config.intermediate_size, config.intermediate_size, config.hidden_size)

    def _init_lora(self, gate_in, gate_out, down_in, down_out):
        for adapter in ("vision", "speech"):
            rank = getattr(self.config, f"{adapter}_lora_rank", 0)
            alpha = getattr(self.config, f"{adapter}_lora_alpha", 1)
            if rank <= 0:
                continue
            setattr(self, f"{adapter}_gate_up_lora_A", _create_lora_parameter(self, [gate_in, rank]))
            setattr(self, f"{adapter}_gate_up_lora_B", _create_lora_parameter(self, [rank, gate_out]))
            setattr(self, f"{adapter}_down_lora_A", _create_lora_parameter(self, [down_in, rank]))
            setattr(self, f"{adapter}_down_lora_B", _create_lora_parameter(self, [rank, down_out]))
            setattr(self, f"{adapter}_lora_scaling", alpha / rank)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        adapter = _active_lora_adapter(self.config)
        if adapter is not None and hasattr(self, f"{adapter}_gate_up_lora_A"):
            up_states = up_states + _lora_delta(
                hidden_states,
                getattr(self, f"{adapter}_gate_up_lora_A"),
                getattr(self, f"{adapter}_gate_up_lora_B"),
                getattr(self, f"{adapter}_lora_scaling"),
            )
        gate, up_states = up_states.chunk(2, axis=-1)
        up_states = up_states * self.activation_fn(gate)
        hidden_states = self.down_proj(up_states)
        if adapter is not None and hasattr(self, f"{adapter}_down_lora_A"):
            hidden_states = hidden_states + _lora_delta(
                up_states,
                getattr(self, f"{adapter}_down_lora_A"),
                getattr(self, f"{adapter}_down_lora_B"),
                getattr(self, f"{adapter}_lora_scaling"),
            )
        return hidden_states


class Phi4MultimodalRotaryEmbedding(nn.Layer):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_type = config.rope_parameters.get("rope_type", "default")
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistable=False)

    def _match_torch_short_longrope_rounding(self, inv_freq):
        rope_parameters = self.config.rope_parameters
        short_factor = rope_parameters.get("short_factor", [])
        if (
            self.rope_type == "longrope"
            and inv_freq.shape[0] == 48
            and len(short_factor) == 48
            and all(float(factor) == 1.0 for factor in short_factor)
            and float(rope_parameters.get("rope_theta", 10000.0)) == 10000.0
        ):
            indices = paddle.to_tensor([7, 10, 14, 17, 20, 23, 25, 28, 31, 34, 37, 40, 43, 46], dtype="int64")
            selected = paddle.gather(inv_freq, indices)
            steps = [1, 1, 3, 3, 4, 4, 5, 5, 7, 7, 7, 9, 5, 6]
            updates = selected
            next_value = paddle.full_like(selected, float("inf"))
            for step in range(max(steps)):
                stepped = paddle.nextafter(updates, next_value)
                mask = paddle.to_tensor([step < count for count in steps], dtype="bool")
                updates = paddle.where(mask, stepped, updates)
            inv_freq = paddle.scatter(inv_freq, indices, updates, overwrite=True)
        return inv_freq

    def _update_longrope_inv_freq(self, position_ids, device):
        if self.rope_type != "longrope":
            return

        seq_len = int((paddle.max(position_ids) + 1).item())
        rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device, seq_len=seq_len)

        original_max_position_embeddings = self.config.rope_parameters.get(
            "original_max_position_embeddings",
            getattr(self.config, "original_max_position_embeddings", self.config.max_position_embeddings),
        )
        if seq_len <= original_max_position_embeddings:
            inv_freq = self._match_torch_short_longrope_rounding(inv_freq)
            self.register_buffer("original_inv_freq", inv_freq.clone(), persistable=False)
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    @staticmethod
    def compute_default_rope_parameters(config, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype="int64").astype("float32") / dim))
        return inv_freq, attention_factor

    def forward(self, x, position_ids):
        self._update_longrope_inv_freq(position_ids, x.place)
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq.astype("float32")[None, :, None].expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].astype("float32")
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose([0, 2, 1])
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class Phi4MultimodalAttention(nn.Layer):
    def __init__(self, config: Phi4MultimodalConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        op_size = config.num_attention_heads * self.head_dim + 2 * (config.num_key_value_heads * self.head_dim)
        self.qkv_proj = GeneralLinear.create(
            config.hidden_size,
            op_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="rowwise",
        )
        self._init_lora(config.hidden_size, op_size, config.num_attention_heads * self.head_dim, config.hidden_size)

    def _init_lora(self, qkv_in, qkv_out, o_in, o_out):
        for adapter in ("vision", "speech"):
            rank = getattr(self.config, f"{adapter}_lora_rank", 0)
            alpha = getattr(self.config, f"{adapter}_lora_alpha", 1)
            if rank <= 0:
                continue
            setattr(self, f"{adapter}_qkv_lora_A", _create_lora_parameter(self, [qkv_in, rank]))
            setattr(self, f"{adapter}_qkv_lora_B", _create_lora_parameter(self, [rank, qkv_out]))
            setattr(self, f"{adapter}_o_lora_A", _create_lora_parameter(self, [o_in, rank]))
            setattr(self, f"{adapter}_o_lora_B", _create_lora_parameter(self, [rank, o_out]))
            setattr(self, f"{adapter}_lora_scaling", alpha / rank)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        adapter = _active_lora_adapter(self.config)
        if adapter is not None and hasattr(self, f"{adapter}_qkv_lora_A"):
            qkv = qkv + _lora_delta(
                hidden_states,
                getattr(self, f"{adapter}_qkv_lora_A"),
                getattr(self, f"{adapter}_qkv_lora_B"),
                getattr(self, f"{adapter}_lora_scaling"),
            )
        query_pos = self.num_heads * self.head_dim
        kv_pos = self.num_key_value_heads * self.head_dim

        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + kv_pos]
        value_states = qkv[..., query_pos + kv_pos :]

        query_states = query_states.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        key_states = key_states.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        value_states = value_states.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
        )

        attn_output = attn_output.reshape([batch_size, seq_len, -1])
        o_input = attn_output
        attn_output = self.o_proj(o_input)
        if adapter is not None and hasattr(self, f"{adapter}_o_lora_A"):
            attn_output = attn_output + _lora_delta(
                o_input,
                getattr(self, f"{adapter}_o_lora_A"),
                getattr(self, f"{adapter}_o_lora_B"),
                getattr(self, f"{adapter}_lora_scaling"),
            )
        return attn_output, attn_weights


class Phi4MultimodalDecoderLayer(nn.Layer):
    def __init__(self, config: Phi4MultimodalConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Phi4MultimodalAttention(config=config, layer_idx=layer_idx)
        self.mlp = Phi4MultimodalMLP(config)
        self.input_layernorm = Phi4MultimodalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi4MultimodalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config
        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        output_attentions: bool = False,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + self.resid_attn_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ======================= Feature Embedding (bridge) =======================


class Phi4MultimodalFeatureEmbedding(nn.Layer):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.image_token_id = config.vision_config.image_token_id
        self.audio_token_id = config.audio_config.audio_token_id
        self.image_embed = Phi4MultimodalImageEmbedding(config)
        self.audio_embed = Phi4MultimodalAudioEmbedding(config)

    def forward(
        self,
        input_ids: paddle.Tensor,
        inputs_embeds: paddle.Tensor,
        image_pixel_values: Optional[paddle.Tensor] = None,
        audio_input_features: Optional[paddle.Tensor] = None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
    ) -> paddle.Tensor:
        image_position_mask = (input_ids == self.config.vision_config.image_token_id).unsqueeze(-1)
        non_image_position_mask = ~image_position_mask

        image_embeds = None
        audio_embeds = None
        if image_pixel_values is not None and (input_ids == self.image_token_id).any():
            image_embeds = self.image_embed(
                input_ids,
                inputs_embeds,
                image_pixel_values=image_pixel_values,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
            )
        if audio_input_features is not None and (input_ids == self.audio_token_id).any():
            audio_projection_mode = "vision" if image_pixel_values is not None else "speech"
            audio_embeds = self.audio_embed(
                input_ids,
                inputs_embeds,
                audio_input_features=audio_input_features,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
            )

        if image_embeds is not None and audio_embeds is not None:
            inputs_embeds = image_embeds * image_position_mask.astype(
                image_embeds.dtype
            ) + audio_embeds * non_image_position_mask.astype(audio_embeds.dtype)
        elif image_embeds is not None:
            inputs_embeds = image_embeds
        elif audio_embeds is not None:
            inputs_embeds = audio_embeds

        return inputs_embeds


# ======================= Main Model =======================


class Phi4MultimodalPreTrainedModel(PretrainedModel):
    config_class = Phi4MultimodalConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "qkv_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "img_projection_up",
        "img_projection_down",
        "up_proj_for_speech",
        "down_proj_for_speech",
        "up_proj_for_vision_speech",
        "down_proj_for_vision_speech",
        "out",
        "lm_head",
    ]

    def __init__(self, config: Phi4MultimodalConfig):
        if config.tensor_model_parallel_size > 1 or config.sequence_parallel:
            raise NotImplementedError(
                "Phi-4 multimodal does not currently support tensor parallel or sequence parallel."
            )
        super().__init__(config)

    @classmethod
    def _gen_aoa_config(cls, config: Phi4MultimodalConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_config = {"aoa_statements": []}
        stmts = aoa_config["aoa_statements"]

        # Embedding and norm
        stmts.append(f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight")
        stmts.append(f"model.norm.weight -> {model_prefix}norm.weight")
        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                stmts.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                stmts.append("lm_head.weight -> lm_head.weight")

        # Decoder layers
        for layer_id in range(config.num_hidden_layers):
            lp = f"model.layers.{layer_id}"
            tp = f"{model_prefix}layers.{layer_id}"
            stmts.append(f"{lp}.input_layernorm.weight -> {tp}.input_layernorm.weight")
            stmts.append(f"{lp}.post_attention_layernorm.weight -> {tp}.post_attention_layernorm.weight")
            stmts.append(f"{lp}.mlp.down_proj.base_layer.weight^T -> {tp}.mlp.down_proj.weight")
            stmts.append(f"{lp}.self_attn.o_proj.base_layer.weight^T -> {tp}.self_attn.o_proj.weight")
            stmts.append(
                f"{lp}.self_attn.qkv_proj.base_layer.weight^T -> {tp}.self_attn.qkv_proj.weight, "
                f"fused_qkv_old, num_heads={config.num_attention_heads}, "
                f"num_key_value_groups={config.num_key_value_heads}, axis=1"
            )
            stmts.append(f"{lp}.mlp.gate_up_proj.base_layer.weight^T -> {tp}.mlp.gate_up_proj.weight, fused_ffn")

            lora_specs = [
                ("self_attn", "qkv_proj", "qkv"),
                ("self_attn", "o_proj", "o"),
                ("mlp", "gate_up_proj", "gate_up"),
                ("mlp", "down_proj", "down"),
            ]
            for adapter in ("vision", "speech"):
                for block, src_proj, dst_proj in lora_specs:
                    rank = getattr(config, f"{adapter}_lora_rank", 0)
                    if rank > 0:
                        stmts.append(
                            f"{lp}.{block}.{src_proj}.lora_A.{adapter}.weight^T -> "
                            f"{tp}.{block}.{adapter}_{dst_proj}_lora_A"
                        )
                        stmts.append(
                            f"{lp}.{block}.{src_proj}.lora_B.{adapter}.weight^T -> "
                            f"{tp}.{block}.{adapter}_{dst_proj}_lora_B"
                        )

        # Vision encoder
        vis_prefix_src = "model.embed_tokens_extend.image_embed"
        vis_prefix_dst = f"{model_prefix}embed_tokens_extend.image_embed"

        stmts.append(f"{vis_prefix_src}.glb_GN -> {vis_prefix_dst}.global_img_feature_extensor")
        stmts.append(f"{vis_prefix_src}.sub_GN -> {vis_prefix_dst}.sub_img_feature_extensor")
        stmts.append(f"{vis_prefix_src}.img_projection.0.weight^T -> {vis_prefix_dst}.img_projection_up.weight")
        stmts.append(f"{vis_prefix_src}.img_projection.0.bias -> {vis_prefix_dst}.img_projection_up.bias")
        stmts.append(f"{vis_prefix_src}.img_projection.2.weight^T -> {vis_prefix_dst}.img_projection_down.weight")
        stmts.append(f"{vis_prefix_src}.img_projection.2.bias -> {vis_prefix_dst}.img_projection_down.bias")

        # Vision processor layers
        vp_src = f"{vis_prefix_src}.img_processor"
        vp_dst = f"{vis_prefix_dst}.img_processor"
        stmts.append(f"{vp_src}.embeddings.patch_embedding.weight -> {vp_dst}.embeddings.patch_embedding.weight")
        stmts.append(f"{vp_src}.embeddings.patch_embedding.bias -> {vp_dst}.embeddings.patch_embedding.bias")
        stmts.append(f"{vp_src}.embeddings.position_embedding.weight -> {vp_dst}.embeddings.position_embedding.weight")
        stmts.append(f"{vp_src}.post_layernorm.weight -> {vp_dst}.post_layernorm.weight")
        stmts.append(f"{vp_src}.post_layernorm.bias -> {vp_dst}.post_layernorm.bias")

        # Vision head (MultiheadAttentionPoolingHead)
        stmts.append(f"{vp_src}.head.probe -> {vp_dst}.head.probe")
        stmts.append(f"{vp_src}.head.layernorm.weight -> {vp_dst}.head.layernorm.weight")
        stmts.append(f"{vp_src}.head.layernorm.bias -> {vp_dst}.head.layernorm.bias")
        stmts.append(f"{vp_src}.head.mlp.fc1.weight^T -> {vp_dst}.head.mlp.fc1.weight")
        stmts.append(f"{vp_src}.head.mlp.fc1.bias -> {vp_dst}.head.mlp.fc1.bias")
        stmts.append(f"{vp_src}.head.mlp.fc2.weight^T -> {vp_dst}.head.mlp.fc2.weight")
        stmts.append(f"{vp_src}.head.mlp.fc2.bias -> {vp_dst}.head.mlp.fc2.bias")

        for i in range(config.vision_config.num_hidden_layers):
            vs = f"{vp_src}.encoder.layers.{i}"
            vd = f"{vp_dst}.encoder.layers.{i}"
            stmts.append(f"{vs}.layer_norm1.weight -> {vd}.layer_norm1.weight")
            stmts.append(f"{vs}.layer_norm1.bias -> {vd}.layer_norm1.bias")
            stmts.append(f"{vs}.layer_norm2.weight -> {vd}.layer_norm2.weight")
            stmts.append(f"{vs}.layer_norm2.bias -> {vd}.layer_norm2.bias")
            stmts.append(f"{vs}.self_attn.q_proj.weight^T -> {vd}.self_attn.q_proj.weight")
            stmts.append(f"{vs}.self_attn.q_proj.bias -> {vd}.self_attn.q_proj.bias")
            stmts.append(f"{vs}.self_attn.k_proj.weight^T -> {vd}.self_attn.k_proj.weight")
            stmts.append(f"{vs}.self_attn.k_proj.bias -> {vd}.self_attn.k_proj.bias")
            stmts.append(f"{vs}.self_attn.v_proj.weight^T -> {vd}.self_attn.v_proj.weight")
            stmts.append(f"{vs}.self_attn.v_proj.bias -> {vd}.self_attn.v_proj.bias")
            stmts.append(f"{vs}.self_attn.out_proj.weight^T -> {vd}.self_attn.out_proj.weight")
            stmts.append(f"{vs}.self_attn.out_proj.bias -> {vd}.self_attn.out_proj.bias")
            stmts.append(f"{vs}.mlp.fc1.weight^T -> {vd}.mlp.fc1.weight")
            stmts.append(f"{vs}.mlp.fc1.bias -> {vd}.mlp.fc1.bias")
            stmts.append(f"{vs}.mlp.fc2.weight^T -> {vd}.mlp.fc2.weight")
            stmts.append(f"{vs}.mlp.fc2.bias -> {vd}.mlp.fc2.bias")

        # Audio encoder
        aud_prefix_src = "model.embed_tokens_extend.audio_embed"
        aud_prefix_dst = f"{model_prefix}embed_tokens_extend.audio_embed"

        # Audio projections
        audio_projection_map = [
            ("audio_projection.speech.0", "up_proj_for_speech"),
            ("audio_projection.speech.2", "down_proj_for_speech"),
            ("audio_projection.vision.0", "up_proj_for_vision_speech"),
            ("audio_projection.vision.2", "down_proj_for_vision_speech"),
        ]
        for src_proj, dst_proj in audio_projection_map:
            stmts.append(f"{aud_prefix_src}.{src_proj}.weight^T -> {aud_prefix_dst}.{dst_proj}.weight")
            stmts.append(f"{aud_prefix_src}.{src_proj}.bias -> {aud_prefix_dst}.{dst_proj}.bias")

        # Audio encoder internals
        ae_src = f"{aud_prefix_src}.encoder"
        ae_dst = f"{aud_prefix_dst}.encoder"

        stmts.append(f"{ae_src}.encoder_embedding.global_mean -> {ae_dst}.encoder_embedding.global_mean")
        stmts.append(f"{ae_src}.encoder_embedding.global_invstd -> {ae_dst}.encoder_embedding.global_invstd")
        stmts.append(
            f"{ae_src}.relative_attention_bias_layer.bias_values.weight -> {ae_dst}.relative_attention_bias_layer.bias_values.weight"
        )

        # Nemo conv subsampling
        stmts.append(f"{ae_src}.embed.conv.0.weight -> {ae_dst}.embed.conv.0.weight")
        stmts.append(f"{ae_src}.embed.conv.0.bias -> {ae_dst}.embed.conv.0.bias")
        for conv_idx in [2, 3, 5, 6]:
            stmts.append(f"{ae_src}.embed.conv.{conv_idx}.weight -> {ae_dst}.embed.conv.{conv_idx}.weight")
            stmts.append(f"{ae_src}.embed.conv.{conv_idx}.bias -> {ae_dst}.embed.conv.{conv_idx}.bias")
        stmts.append(f"{ae_src}.embed.out.weight^T -> {ae_dst}.embed.out.weight")
        stmts.append(f"{ae_src}.embed.out.bias -> {ae_dst}.embed.out.bias")

        # Audio conformer encoder layers
        for i in range(config.audio_config.num_blocks):
            al_src = f"{ae_src}.encoders.{i}"
            al_dst = f"{ae_dst}.encoders.{i}"

            # feed_forward_in / feed_forward_out
            for ff in ["feed_forward_in", "feed_forward_out"]:
                stmts.append(f"{al_src}.{ff}.layer_norm.weight -> {al_dst}.{ff}.layer_norm.weight")
                stmts.append(f"{al_src}.{ff}.layer_norm.bias -> {al_dst}.{ff}.layer_norm.bias")
                stmts.append(f"{al_src}.{ff}.net.0.linear.weight^T -> {al_dst}.{ff}.gate_up_proj.weight")
                stmts.append(f"{al_src}.{ff}.net.0.linear.bias -> {al_dst}.{ff}.gate_up_proj.bias")
                stmts.append(f"{al_src}.{ff}.net.2.weight^T -> {al_dst}.{ff}.down_proj.weight")
                stmts.append(f"{al_src}.{ff}.net.2.bias -> {al_dst}.{ff}.down_proj.bias")

            # self_attn
            stmts.append(f"{al_src}.self_attn.linear_q.weight^T -> {al_dst}.self_attn.q_proj.weight")
            stmts.append(f"{al_src}.self_attn.linear_q.bias -> {al_dst}.self_attn.q_proj.bias")
            stmts.append(f"{al_src}.self_attn.linear_k.weight^T -> {al_dst}.self_attn.k_proj.weight")
            stmts.append(f"{al_src}.self_attn.linear_k.bias -> {al_dst}.self_attn.k_proj.bias")
            stmts.append(f"{al_src}.self_attn.linear_v.weight^T -> {al_dst}.self_attn.v_proj.weight")
            stmts.append(f"{al_src}.self_attn.linear_v.bias -> {al_dst}.self_attn.v_proj.bias")
            stmts.append(f"{al_src}.self_attn.linear_out.weight^T -> {al_dst}.self_attn.o_proj.weight")
            stmts.append(f"{al_src}.self_attn.linear_out.bias -> {al_dst}.self_attn.o_proj.bias")

            # conv module
            stmts.append(f"{al_src}.conv.layer_norm.weight -> {al_dst}.conv.layer_norm.weight")
            stmts.append(f"{al_src}.conv.layer_norm.bias -> {al_dst}.conv.layer_norm.bias")
            stmts.append(f"{al_src}.conv.glu.ext_pw_conv_1d.weight -> {al_dst}.conv.glu.ext_pw_conv_1d.weight")
            stmts.append(f"{al_src}.conv.glu.ext_pw_conv_1d.bias -> {al_dst}.conv.glu.ext_pw_conv_1d.bias")
            stmts.append(f"{al_src}.conv.glu.b1 -> {al_dst}.conv.glu.b1")
            stmts.append(f"{al_src}.conv.glu.b2 -> {al_dst}.conv.glu.b2")
            stmts.append(f"{al_src}.conv.dw_sep_conv_1d.dw_conv.weight -> {al_dst}.conv.dw_sep_conv_1d.dw_conv.weight")
            stmts.append(f"{al_src}.conv.dw_sep_conv_1d.dw_conv.bias -> {al_dst}.conv.dw_sep_conv_1d.dw_conv.bias")
            stmts.append(f"{al_src}.conv.dw_sep_conv_1d.pw_conv.weight -> {al_dst}.conv.dw_sep_conv_1d.pw_conv.weight")
            stmts.append(f"{al_src}.conv.dw_sep_conv_1d.pw_conv.bias -> {al_dst}.conv.dw_sep_conv_1d.pw_conv.bias")
            stmts.append(f"{al_src}.conv.ext_pw_conv_1d.weight -> {al_dst}.conv.ext_pw_conv_1d.weight")
            stmts.append(f"{al_src}.conv.ext_pw_conv_1d.bias -> {al_dst}.conv.ext_pw_conv_1d.bias")

            # layer norms
            stmts.append(f"{al_src}.layer_norm_att.weight -> {al_dst}.layer_norm_att.weight")
            stmts.append(f"{al_src}.layer_norm_att.bias -> {al_dst}.layer_norm_att.bias")
            stmts.append(f"{al_src}.layer_norm.weight -> {al_dst}.layer_norm.weight")
            stmts.append(f"{al_src}.layer_norm.bias -> {al_dst}.layer_norm.bias")

        # Vision head attention (nn.MultiHeadAttention has different weight names in Paddle)
        # The HF model uses torch.nn.MultiheadAttention with in_proj_weight/in_proj_bias
        # We'll handle this mapping for the vision pooling head attention
        head_src = f"{vp_src}.head.attention"
        head_dst = f"{vp_dst}.head.attention"
        q_weight_tmp = f"{head_src}.in_proj_weight.q"
        k_weight_tmp = f"{head_src}.in_proj_weight.k"
        v_weight_tmp = f"{head_src}.in_proj_weight.v"
        stmts.append(f"{head_src}.in_proj_weight -> {q_weight_tmp}, {k_weight_tmp}, {v_weight_tmp}, axis=0")
        stmts.append(f"{q_weight_tmp}^T -> {head_dst}.q_proj.weight")
        stmts.append(f"{k_weight_tmp}^T -> {head_dst}.k_proj.weight")
        stmts.append(f"{v_weight_tmp}^T -> {head_dst}.v_proj.weight")
        q_bias_tmp = f"{head_src}.in_proj_bias.q"
        k_bias_tmp = f"{head_src}.in_proj_bias.k"
        v_bias_tmp = f"{head_src}.in_proj_bias.v"
        stmts.append(f"{head_src}.in_proj_bias -> {q_bias_tmp}, {k_bias_tmp}, {v_bias_tmp}, axis=0")
        stmts.append(f"{q_bias_tmp} -> {head_dst}.q_proj.bias")
        stmts.append(f"{k_bias_tmp} -> {head_dst}.k_proj.bias")
        stmts.append(f"{v_bias_tmp} -> {head_dst}.v_proj.bias")
        stmts.append(f"{head_src}.out_proj.weight^T -> {head_dst}.out_proj.weight")
        stmts.append(f"{head_src}.out_proj.bias -> {head_dst}.out_proj.bias")

        return aoa_config


@register_base_model
class Phi4MultimodalModel(Phi4MultimodalPreTrainedModel):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Phi4MultimodalDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi4MultimodalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Phi4MultimodalRotaryEmbedding(config)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.embed_tokens_extend = Phi4MultimodalFeatureEmbedding(config)

    @paddle.jit.not_to_static
    def recompute_training_full(self, layer_module, hidden_states, *args):
        active_lora_adapter = getattr(self.config, "_active_lora_adapter", None)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                previous_adapter = getattr(self.config, "_active_lora_adapter", None)
                self.config._active_lora_adapter = active_lora_adapter
                try:
                    return module(*inputs)
                finally:
                    self.config._active_lora_adapter = previous_adapter

            return custom_forward

        hidden_states = recompute(create_custom_forward(layer_module), hidden_states, *args)
        return hidden_states

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        image_pixel_values: Optional[paddle.Tensor] = None,
        image_sizes: Optional[paddle.Tensor] = None,
        image_attention_mask=None,
        audio_input_features: Optional[paddle.Tensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        previous_adapter = getattr(self.config, "_active_lora_adapter", None)
        self.config._active_lora_adapter = _lora_adapter_from_input_mode(
            input_mode,
            image_pixel_values=image_pixel_values,
            audio_input_features=audio_input_features,
        )

        try:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
                inputs_embeds = self.embed_tokens_extend(
                    input_ids,
                    inputs_embeds,
                    image_pixel_values=image_pixel_values,
                    audio_input_features=audio_input_features,
                    image_sizes=image_sizes,
                    image_attention_mask=image_attention_mask,
                    audio_embed_sizes=audio_embed_sizes,
                    audio_attention_mask=audio_attention_mask,
                )

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

            if position_ids is None:
                position_ids = (
                    paddle.arange(seq_length, dtype="int64").unsqueeze(0).expand([batch_size, -1]) + cache_length
                )

            # Create causal mask
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "cache_length": cache_length,
                "attention_mask": attention_mask,
                "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
                "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
            }
            causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None

            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                has_gradient = not hidden_states.stop_gradient
                if (
                    self.config.recompute_granularity == "full"
                    and self.config.recompute_method == "uniform"
                    and self.config.recompute_num_layers == 1
                    and has_gradient
                ):
                    layer_outputs = self.recompute_training_full(
                        decoder_layer,
                        hidden_states,
                        causal_mask,
                        attn_mask_startend_row_indices,
                        position_ids,
                        past_key_values,
                        use_cache,
                        position_embeddings,
                        output_attentions,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if not return_dict:
                return tuple(
                    v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
                )

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        finally:
            self.config._active_lora_adapter = previous_adapter


class Phi4MultimodalForCausalLM(Phi4MultimodalPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__(config)
        self.model = Phi4MultimodalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self._apply_multimodal_freeze_config()

    def _apply_multimodal_freeze_config(self):
        freeze_prefixes = []
        if getattr(self.config, "freeze_vision_model", False):
            freeze_prefixes.extend(
                [
                    "model.embed_tokens_extend.image_embed.img_processor",
                    "model.embed_tokens_extend.audio_embed.encoder",
                ]
            )
        if getattr(self.config, "freeze_vision_projection", False):
            freeze_prefixes.extend(
                [
                    "model.embed_tokens_extend.image_embed.img_projection_up",
                    "model.embed_tokens_extend.image_embed.img_projection_down",
                    "model.embed_tokens_extend.image_embed.global_img_feature_extensor",
                    "model.embed_tokens_extend.image_embed.sub_img_feature_extensor",
                    "model.embed_tokens_extend.audio_embed.up_proj_for_speech",
                    "model.embed_tokens_extend.audio_embed.down_proj_for_speech",
                    "model.embed_tokens_extend.audio_embed.up_proj_for_vision_speech",
                    "model.embed_tokens_extend.audio_embed.down_proj_for_vision_speech",
                ]
            )
        if getattr(self.config, "freeze_language_model", False):
            freeze_prefixes.extend(["model.embed_tokens", "model.layers", "model.norm", "lm_head"])

        freeze_multimodal_adapters = (
            getattr(self.config, "freeze_vision_model", False)
            and getattr(self.config, "freeze_vision_projection", False)
            and not getattr(self.config, "freeze_language_model", False)
        )

        if not freeze_prefixes:
            return

        frozen = 0
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefixes) or (
                freeze_multimodal_adapters and "_lora_" in name
            ):
                param.stop_gradient = True
                frozen += 1
        logger.info(f"Phi-4 multimodal freeze_config applied. Frozen parameter tensors: {frozen}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model._apply_multimodal_freeze_config()
        return model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=True,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_pixel_values=None,
        image_sizes=None,
        image_attention_mask=None,
        audio_input_features=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        position_ids=None,
        **kwargs,
    ):
        if input_mode is None:
            has_image = image_pixel_values is not None
            has_audio = audio_input_features is not None
            if has_image and has_audio:
                input_mode = paddle.to_tensor([3], dtype="int64")
            elif has_image:
                input_mode = paddle.to_tensor([1], dtype="int64")
            elif has_audio:
                input_mode = paddle.to_tensor([2], dtype="int64")

        batch_size, seq_length = input_ids.shape
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).expand([batch_size, -1])
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(axis=-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "input_mode": input_mode,
            }
        )

        if past_key_values is None:
            model_inputs.update(
                {
                    "image_pixel_values": image_pixel_values,
                    "image_sizes": image_sizes,
                    "image_attention_mask": image_attention_mask,
                    "audio_input_features": audio_input_features,
                    "audio_embed_sizes": audio_embed_sizes,
                    "audio_attention_mask": audio_attention_mask,
                }
            )

        return model_inputs

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        image_pixel_values: Optional[paddle.Tensor] = None,
        image_sizes: Optional[paddle.Tensor] = None,
        image_attention_mask=None,
        audio_input_features: Optional[paddle.Tensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_pixel_values=image_pixel_values,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            audio_input_features=audio_input_features,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            input_mode=input_mode,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
            if isinstance(loss, tuple):
                loss = loss[0]

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


Phi4MultimodalForConditionalGeneration = Phi4MultimodalForCausalLM
Phi4MMForCausalLM = Phi4MultimodalForCausalLM
Phi4MMForConditionalGeneration = Phi4MultimodalForCausalLM
