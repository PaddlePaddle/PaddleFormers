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
"""Paddle ShieldGemma2 model."""

import os
from dataclasses import dataclass
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn

from ..activations import ACT2FN
from ..cache_utils import Cache
from ..configuration_utils import PretrainedConfig
from ..gemma3.modeling import Gemma3ForConditionalGeneration, _convert_hf_vision_tensor
from ..gemma3.multimodal_text_modeling import Gemma3RMSNorm, _iter_hf_tensors, load_hf_text_state_dict
from ..model_outputs import ImageClassifierOutputWithNoAttention
from ..model_utils import dtype_guard
from ...utils.log import logger
from .configuration import ShieldGemma2Config


@dataclass
class ShieldGemma2ImageClassifierOutputWithNoAttention(ImageClassifierOutputWithNoAttention):
    probabilities: Optional[paddle.Tensor] = None


class ShieldGemma2VisionEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.position_ids = paddle.arange(self.num_patches, dtype="int64").reshape([1, -1])

    def interpolate_pos_encoding(self, embeddings: paddle.Tensor, height: int, width: int) -> paddle.Tensor:
        num_patches = embeddings.shape[1]
        if num_patches == self.num_patches and height == width == self.image_size:
            return self.position_embedding(self.position_ids)

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(self.num_patches**0.5)

        patch_pos_embed = self.position_embedding.weight.reshape([1, sqrt_num_positions, sqrt_num_positions, dim])
        patch_pos_embed = patch_pos_embed.transpose([0, 3, 1, 2])
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=[new_height, new_width],
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.transpose([0, 2, 3, 1]).reshape([1, -1, dim])
        return patch_pos_embed

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        _, _, height, width = pixel_values.shape
        pixel_values = pixel_values.astype(self.patch_embedding.weight.dtype)
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(start_axis=2).transpose([0, 2, 1])
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        return embeddings


class ShieldGemma2VisionAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        batch_size, seq_length, embed_dim = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([batch_size, seq_length, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        key_states = key_states.reshape([batch_size, seq_length, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        value_states = value_states.reshape([batch_size, seq_length, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scale
        attn_weights = F.softmax(attn_weights, axis=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, seq_length, embed_dim])
        return self.out_proj(attn_output)


class ShieldGemma2VisionMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class ShieldGemma2VisionEncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.self_attn = ShieldGemma2VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = ShieldGemma2VisionMLP(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ShieldGemma2VisionTower(nn.Layer):
    def __init__(self, config: ShieldGemma2Config):
        super().__init__()
        vision_config = config.vision_config
        self.mm_tokens_per_image = config.mm_tokens_per_image
        self.embeddings = ShieldGemma2VisionEmbeddings(vision_config)
        self.encoder = nn.LayerList(
            [ShieldGemma2VisionEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(vision_config.hidden_size, epsilon=vision_config.layer_norm_eps)

    def _flatten_pixel_values(self, pixel_values: paddle.Tensor) -> tuple[paddle.Tensor, int, int]:
        if pixel_values.ndim == 4:
            batch_size = pixel_values.shape[0]
            num_images = 1
            flat_pixel_values = pixel_values
        elif pixel_values.ndim == 5:
            batch_size, num_images = pixel_values.shape[:2]
            flat_pixel_values = pixel_values.reshape([-1, *pixel_values.shape[-3:]])
        else:
            raise ValueError(
                "`pixel_values` must have shape [batch, channels, height, width] or [batch, images, channels, height, width]."
            )
        return flat_pixel_values, batch_size, num_images

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        flat_pixel_values, batch_size, num_images = self._flatten_pixel_values(pixel_values)
        hidden_states = self.embeddings(flat_pixel_values)
        for encoder_layer in self.encoder:
            hidden_states = encoder_layer(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states.reshape([batch_size, num_images, hidden_states.shape[1], hidden_states.shape[2]])


class ShieldGemma2MultiModalProjector(nn.Layer):
    def __init__(self, config: ShieldGemma2Config):
        super().__init__()
        self.vision_hidden_size = config.vision_config.hidden_size
        self.text_hidden_size = config.text_config.hidden_size
        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.mm_input_projection_weight = self.create_parameter(
            shape=[self.vision_hidden_size, self.text_hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Normal(std=config.initializer_range),
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            self.vision_hidden_size, eps=getattr(config.vision_config, "layer_norm_eps", 1e-6)
        )
        self.avg_pool = nn.AvgPool2D(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: paddle.Tensor) -> paddle.Tensor:
        batch_size, num_images, _, hidden_size = vision_outputs.shape
        reshaped_vision_outputs = vision_outputs.reshape(
            [batch_size * num_images, self.patches_per_image * self.patches_per_image, hidden_size]
        )
        reshaped_vision_outputs = reshaped_vision_outputs.transpose([0, 2, 1])
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            [batch_size * num_images, hidden_size, self.patches_per_image, self.patches_per_image]
        )

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(start_axis=2).transpose([0, 2, 1])
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)
        projected_vision_outputs = paddle.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        projected_vision_outputs = projected_vision_outputs.astype(vision_outputs.dtype)
        return projected_vision_outputs.reshape([batch_size, num_images * projected_vision_outputs.shape[1], -1])


class ShieldGemma2MultiModalModel(nn.Layer):
    def __init__(self, config: ShieldGemma2Config):
        super().__init__()
        self.config = config
        self.language_model = Gemma3ForConditionalGeneration(config)
        self.vision_tower = ShieldGemma2VisionTower(config)
        self.multi_modal_projector = ShieldGemma2MultiModalProjector(config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_image_features(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        image_features = self.vision_tower(pixel_values)
        return self.multi_modal_projector(image_features)

    def _merge_image_features(
        self,
        input_ids: paddle.Tensor,
        inputs_embeds: paddle.Tensor,
        image_features: paddle.Tensor,
    ) -> paddle.Tensor:
        image_mask = input_ids == self.config.image_token_index
        tokens_per_sample = image_features.shape[1]
        merged_embeds = paddle.clone(inputs_embeds)

        for batch_idx in range(input_ids.shape[0]):
            image_positions = paddle.nonzero(image_mask[batch_idx]).flatten()
            if image_positions.shape[0] != tokens_per_sample:
                raise ValueError(
                    "Image features and image placeholders do not match for sample "
                    f"{batch_idx}: found {image_positions.shape[0]} placeholders but expected {tokens_per_sample}."
                )
            if image_positions.shape[0] > 0:
                merged_embeds[batch_idx, image_positions, :] = image_features[batch_idx].astype(merged_embeds.dtype)

        return merged_embeds

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` is required when `inputs_embeds` is not provided.")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("`input_ids` is required when `pixel_values` are provided.")
            image_features = self.get_image_features(pixel_values).astype(inputs_embeds.dtype)
            inputs_embeds = self._merge_image_features(input_ids, inputs_embeds, image_features)

        return self.language_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=kwargs.pop("token_type_ids", None),
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )


class ShieldGemma2ForImageClassification(Gemma3ForConditionalGeneration):
    config_class = ShieldGemma2Config
    base_model_prefix = "backbone"
    input_modalities = ("image", "text")

    def __init__(self, config: ShieldGemma2Config):
        config.tie_word_embeddings = False
        config.text_config.tie_word_embeddings = False
        super().__init__(config)
        self.yes_token_index = getattr(config, "yes_token_index", 10784)
        self.no_token_index = getattr(config, "no_token_index", 3771)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        is_hf_safetensors = (
            isinstance(pretrained_model_name_or_path, str)
            and os.path.isdir(pretrained_model_name_or_path)
            and (
                os.path.exists(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
                or os.path.exists(os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json"))
            )
        )
        if not is_hf_safetensors:
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        config = kwargs.pop("config", None)
        dtype = kwargs.pop("dtype", None)
        if not isinstance(config, PretrainedConfig):
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        if dtype is not None:
            config.dtype = dtype
        with dtype_guard(dtype or paddle.get_default_dtype()):
            model = cls(config, *args)
        target_state_dict = model.state_dict()
        state_dict = load_hf_text_state_dict(
            pretrained_model_name_or_path,
            config.text_config,
            model_prefix="model.language_model.",
            # ShieldGemma2's HF checkpoint intentionally omits lm_head.  The
            # Torch implementation initializes this task head in post_init;
            # copying the tied embedding here would change its semantics.
            include_lm_head=False,
            source_prefix="model.language_model.",
        )

        for hf_key, tensor in _iter_hf_tensors(pretrained_model_name_or_path):
            if hf_key.startswith("model.vision_tower.vision_model."):
                target_key = "model.vision_tower." + hf_key[len("model.vision_tower.vision_model.") :]
            elif hf_key.startswith("model.multi_modal_projector."):
                target_key = "model." + hf_key[len("model.") :]
            else:
                continue
            if target_key not in target_state_dict:
                continue
            state_dict[target_key] = _convert_hf_vision_tensor(target_key, tensor)

        for name, tensor in list(state_dict.items()):
            if name in target_state_dict and tensor.dtype != target_state_dict[name].dtype:
                state_dict[name] = tensor.astype(target_state_dict[name].dtype)
        missing_keys, unexpected_keys = model.set_state_dict(state_dict)
        if missing_keys or unexpected_keys:
            logger.warning(
                f"HF ShieldGemma2 checkpoint load finished with missing keys {missing_keys} "
                f"and unexpected keys {unexpected_keys}"
            )
        return model

    @classmethod
    def _gen_aoa_config(cls, config: ShieldGemma2Config):
        aoa_config = super()._gen_aoa_config(config)
        statements = []
        for statement in aoa_config["aoa_statements"]:
            sources, target = statement.split(" -> ", 1)
            prefixed_sources = ", ".join(f"model.{source}" for source in sources.split(", "))
            statements.append(f"{prefixed_sources} -> {target}")
        return {"aoa_statements": statements}

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep=0,
        **kwargs,
    ) -> ShieldGemma2ImageClassifierOutputWithNoAttention:
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        last_token_logits = outputs.logits[:, -1, :]
        index_tensor = paddle.to_tensor([self.yes_token_index, self.no_token_index], dtype="int64")
        selected_logits = paddle.index_select(last_token_logits, index=index_tensor, axis=-1)
        probabilities = F.softmax(selected_logits, axis=-1)

        return ShieldGemma2ImageClassifierOutputWithNoAttention(
            loss=outputs.loss,
            logits=selected_logits,
            hidden_states=outputs.hidden_states,
            probabilities=probabilities,
        )


__all__ = ["ShieldGemma2ForImageClassification", "ShieldGemma2ImageClassifierOutputWithNoAttention"]
