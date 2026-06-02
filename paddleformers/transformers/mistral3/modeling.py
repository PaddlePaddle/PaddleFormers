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

"""Paddle Mistral3 multimodal model."""

from dataclasses import dataclass
from typing import Callable

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.activation import ACT2FN
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.norm import Norm as GeneralNorm
from ..cache_utils import Cache
from ..llama.modeling import (
    LLamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPretrainedModel,
    LlamaRotaryEmbedding,
    rotate_half,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel, register_base_model
from ..pixtral.modeling import PixtralRMSNorm, PixtralVisionModel
from .configuration import Mistral3Config


def _normalize_image_sizes(image_sizes):
    if isinstance(image_sizes, paddle.Tensor):
        image_sizes = image_sizes.numpy().tolist()
    return [(int(size[0]), int(size[1])) for size in image_sizes]


def apply_mistral3_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Mistral3TextAttention(LLamaAttention):
    def __init__(self, config, layer_idx: int):
        nn.Layer.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        assert config.num_attention_heads % config.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
            f"Found {config.num_attention_heads} and {config.num_key_value_heads}"
        )
        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert (
                self.num_key_value_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_key_value_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        q_hidden_size = self.head_dim * config.num_attention_heads
        kv_hidden_size = self.head_dim * config.num_key_value_heads

        self.q_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="rowwise",
        )

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

        query_states = self.q_proj(hidden_states).reshape(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_mistral3_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["sdpa"]
        if self.config._attn_implementation != "sdpa":
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
        )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Mistral3TextDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        nn.Layer.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = Mistral3TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )


class Mistral3TextModel(LlamaModel):
    def __init__(self, config):
        LlamaPretrainedModel.__init__(self, config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList(
            [Mistral3TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)


@dataclass
class Mistral3ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: paddle.Tensor | None = None


@dataclass
class Mistral3CausalLMOutputWithPast(CausalLMOutputWithPast):
    image_hidden_states: paddle.Tensor | None = None


@dataclass
class Mistral3ImageFeaturesOutput(ModelOutput):
    last_hidden_state: paddle.Tensor = None
    pooler_output: tuple[paddle.Tensor, ...] | None = None
    hidden_states: tuple[paddle.Tensor, ...] | None = None
    attentions: tuple[paddle.Tensor, ...] | None = None


class Mistral3PatchMerger(nn.Layer):
    """Learned merging of spatial_merge_size ** 2 neighboring vision patches."""

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config
        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias_attr=False)

    def forward(self, image_features: paddle.Tensor, image_sizes) -> paddle.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in _normalize_image_sizes(image_sizes)
        ]
        tokens_per_image = [height * width for height, width in image_sizes]
        hidden_size = image_features.shape[-1]

        merged_images = []
        offset = 0
        for height, width in image_sizes:
            image_tokens = image_features[offset : offset + height * width]
            offset += height * width

            image_grid = image_tokens.reshape([height, width, hidden_size]).transpose([2, 0, 1]).unsqueeze(0)
            grid = F.unfold(
                image_grid,
                kernel_sizes=self.spatial_merge_size,
                strides=self.spatial_merge_size,
            )
            grid = grid.squeeze(0).transpose([1, 0])
            merged_images.append(grid)

        if offset != sum(tokens_per_image):
            raise ValueError("Invalid image feature split while merging patches.")

        image_features = paddle.concat(merged_images, axis=0)
        return self.merging_layer(image_features)


class Mistral3MultiModalProjector(nn.Layer):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.norm = PixtralRMSNorm(config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.patch_merger = Mistral3PatchMerger(config)
        self.num_feature_layers = (
            1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * self.num_feature_layers,
            config.text_config.hidden_size,
            bias_attr=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias_attr=config.multimodal_projector_bias,
        )

    def forward(self, image_features: paddle.Tensor, image_sizes) -> paddle.Tensor:
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Mistral3PretrainedModel(PretrainedModel):
    config_class = Mistral3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "merging_layer",
        "linear_1",
        "linear_2",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Mistral3Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = []

        vision_src = "vision_tower"
        vision_dst = f"{model_prefix}vision_tower"
        aoa_statements.extend(
            [
                f"{vision_src}.patch_conv.weight -> {vision_dst}.patch_conv.weight",
                f"{vision_src}.ln_pre.weight -> {vision_dst}.ln_pre.weight",
            ]
        )
        for layer_name in ["attention_norm", "ffn_norm"]:
            aoa_statements.append(
                f"{vision_src}.transformer.layers.$LAYER_ID.{layer_name}.weight -> "
                f"{vision_dst}.transformer.layers.$LAYER_ID.{layer_name}.weight"
            )
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            aoa_statements.append(
                f"{vision_src}.transformer.layers.$LAYER_ID.attention.{proj_name}.weight^T -> "
                f"{vision_dst}.transformer.layers.$LAYER_ID.attention.{proj_name}.weight"
            )
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            aoa_statements.append(
                f"{vision_src}.transformer.layers.$LAYER_ID.feed_forward.{proj_name}.weight^T -> "
                f"{vision_dst}.transformer.layers.$LAYER_ID.feed_forward.{proj_name}.weight"
            )

        projector_src = "multi_modal_projector"
        projector_dst = f"{model_prefix}multi_modal_projector"
        aoa_statements.extend(
            [
                f"{projector_src}.norm.weight -> {projector_dst}.norm.weight",
                f"{projector_src}.patch_merger.merging_layer.weight^T -> "
                f"{projector_dst}.patch_merger.merging_layer.weight",
                f"{projector_src}.linear_1.weight^T -> {projector_dst}.linear_1.weight",
                f"{projector_src}.linear_2.weight^T -> {projector_dst}.linear_2.weight",
            ]
        )
        if config.multimodal_projector_bias:
            aoa_statements.extend(
                [
                    f"{projector_src}.linear_1.bias -> {projector_dst}.linear_1.bias",
                    f"{projector_src}.linear_2.bias -> {projector_dst}.linear_2.bias",
                ]
            )

        text_src = "language_model.model"
        text_dst = f"{model_prefix}language_model"
        aoa_statements.extend(
            [
                f"{text_src}.embed_tokens.weight -> {text_dst}.embed_tokens.weight",
                f"{text_src}.norm.weight -> {text_dst}.norm.weight",
                f"{text_src}.layers.$LAYER_ID.input_layernorm.weight -> "
                f"{text_dst}.layers.$LAYER_ID.input_layernorm.weight",
                f"{text_src}.layers.$LAYER_ID.post_attention_layernorm.weight -> "
                f"{text_dst}.layers.$LAYER_ID.post_attention_layernorm.weight",
            ]
        )
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            aoa_statements.append(
                f"{text_src}.layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> "
                f"{text_dst}.layers.$LAYER_ID.self_attn.{proj_name}.weight"
            )
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            aoa_statements.append(
                f"{text_src}.layers.$LAYER_ID.mlp.{proj_name}.weight^T -> "
                f"{text_dst}.layers.$LAYER_ID.mlp.{proj_name}.weight"
            )

        if cls != cls.base_model_class:
            if not config.tie_word_embeddings:
                aoa_statements.append("language_model.lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Mistral3Config):
        aoa_config = cls._gen_aoa_config(config)
        inv_statements = []
        for statement in aoa_config["aoa_statements"]:
            src, dst = statement.split(" -> ")
            if src.endswith("^T"):
                inv_statements.append(f"{dst}^T -> {src[:-2]}")
            else:
                inv_statements.append(f"{dst} -> {src}")
        return {"aoa_statements": inv_statements}


@register_base_model
class Mistral3Model(Mistral3PretrainedModel):
    base_model_prefix = ""

    def __init__(self, config: Mistral3Config):
        super().__init__(config)
        self.config = config
        self.vision_tower = PixtralVisionModel(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.language_model = Mistral3TextModel(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: paddle.Tensor,
        image_sizes,
        vision_feature_layer: int | list[int] | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
    ):
        if vision_feature_layer is None:
            vision_feature_layer = self.config.vision_feature_layer

        image_outputs = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
        )
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            selected_image_feature = paddle.concat(
                [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer],
                axis=-1,
            )

        image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
        downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
        split_sizes = [
            (image_size[0] // downsample_ratio) * (image_size[1] // downsample_ratio)
            for image_size in _normalize_image_sizes(image_sizes)
        ]
        image_features = tuple(paddle.split(image_features, split_sizes, axis=0))

        if not return_dict:
            return (image_outputs.last_hidden_state, image_features, image_outputs.hidden_states)
        return Mistral3ImageFeaturesOutput(
            last_hidden_state=image_outputs.last_hidden_state,
            pooler_output=image_features,
            hidden_states=image_outputs.hidden_states if output_hidden_states else None,
            attentions=image_outputs.attentions,
        )

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        if input_ids is None:
            image_token = paddle.to_tensor(self.config.image_token_index, dtype="int64")
            special_image_mask = inputs_embeds == self.get_input_embeddings()(image_token)
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index

        n_image_tokens = int(special_image_mask.astype("int64").sum().item())
        n_image_features = image_features.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                "Image features and image tokens do not match, "
                f"tokens: {n_image_tokens}, features: {n_image_features}"
            )
        return special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        pixel_values: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        use_cache: bool | None = None,
        image_sizes=None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            if image_sizes is None:
                batch_size, _, height, width = pixel_values.shape
                image_sizes = [(height, width)] * batch_size
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                return_dict=True,
            ).pooler_output
            image_features = paddle.concat(image_features, axis=0).astype(inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not return_dict:
            output = (outputs.last_hidden_state, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return output + (image_features,) if image_features is not None else output

        return Mistral3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class Mistral3ForConditionalGeneration(Mistral3PretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: Mistral3Config):
        super().__init__(config)
        self.config = config
        self.model = Mistral3Model(config)
        self.lm_head = GeneralLMHead(config.text_config)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def get_image_features(self, pixel_values, image_sizes, vision_feature_layer=None, **kwargs):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            **kwargs,
        )

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        pixel_values: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | paddle.Tensor = 0,
        image_sizes=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
        **kwargs,
    ):
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_sizes=image_sizes,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            hidden_states = hidden_states[:, -logits_to_keep:, :] if logits_to_keep > 0 else hidden_states
        else:
            hidden_states = hidden_states[:, logits_to_keep, :]
        if self.config.tie_word_embeddings:
            logits = paddle.matmul(hidden_states, self.model.get_input_embeddings().weight, transpose_y=True)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, logits.shape[-1]]), labels.reshape([-1]))

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output if loss is not None else output

        return Mistral3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_sizes=None,
        **kwargs,
    ):
        if cache_position is None:
            if past_key_values is None:
                cache_position = paddle.arange(input_ids.shape[1])
            else:
                cache_position = paddle.to_tensor([input_ids.shape[1] - 1])

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            use_cache=use_cache,
            **kwargs,
        )
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
        return model_inputs


Mistral3ForCausalLM = Mistral3ForConditionalGeneration


__all__ = [
    "Mistral3ForCausalLM",
    "Mistral3ForConditionalGeneration",
    "Mistral3Model",
    "Mistral3PretrainedModel",
    "Mistral3PatchMerger",
    "Mistral3MultiModalProjector",
]
