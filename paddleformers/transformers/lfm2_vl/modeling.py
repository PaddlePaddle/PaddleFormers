# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Paddle implementation of LFM2-VL and its SigLIP2 NaFlex vision tower."""

import inspect
import os
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
from paddle import nn
from safetensors import safe_open

from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead
from ...utils.log import logger
from ..activations import ACT2FN
from ..configuration_utils import PretrainedConfig
from ..model_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from ..model_utils import PretrainedModel, dtype_guard, register_base_model
from .configuration import Lfm2VlConfig
from .modeling_lfm2 import Lfm2Model


@dataclass
class Lfm2VlModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: paddle.Tensor | None = None


@dataclass
class Lfm2VlCausalLMOutputWithPast(ModelOutput):
    loss: paddle.Tensor | None = None
    logits: paddle.Tensor | None = None
    past_key_values: object | None = None
    hidden_states: tuple[paddle.Tensor] | None = None
    attentions: tuple[paddle.Tensor] | None = None
    image_hidden_states: paddle.Tensor | None = None


class Siglip2VisionEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size * config.patch_size,
            config.hidden_size,
        )
        self.position_embedding_size = int(config.num_patches**0.5)
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def forward(self, pixel_values, spatial_shapes):
        patch_embeds = self.patch_embedding(pixel_values.astype(self.patch_embedding.weight.dtype))
        position_embeddings = self.position_embedding.weight.reshape(
            [self.position_embedding_size, self.position_embedding_size, -1]
        )
        position_embeddings = position_embeddings.transpose([2, 0, 1]).unsqueeze(0)
        resized_batch = []
        max_length = pixel_values.shape[1]
        for batch_index in range(spatial_shapes.shape[0]):
            height, width = [int(item) for item in spatial_shapes[batch_index].tolist()]
            resized = F.interpolate(
                position_embeddings.astype("float32"),
                size=[height, width],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape([position_embeddings.shape[1], height * width]).transpose([1, 0])
            resized = resized.astype(patch_embeds.dtype)
            if height * width < max_length:
                resized = paddle.concat([resized, resized[0:1].expand([max_length - height * width, -1])], axis=0)
            resized_batch.append(resized)
        return patch_embeds + paddle.stack(resized_batch)


class Siglip2Attention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        shape = [batch_size, sequence_length, self.num_heads, self.head_dim]
        query = self.q_proj(hidden_states).reshape(shape).transpose([0, 2, 1, 3])
        key = self.k_proj(hidden_states).reshape(shape).transpose([0, 2, 1, 3])
        value = self.v_proj(hidden_states).reshape(shape).transpose([0, 2, 1, 3])
        scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = F.softmax(scores.astype("float32"), axis=-1).astype(query.dtype)
        output = paddle.matmul(probabilities, value).transpose([0, 2, 1, 3])
        return self.out_proj(output.reshape([batch_size, sequence_length, hidden_size]))


class Siglip2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.fc2(self.activation(self.fc1(hidden_states)))


class Siglip2EncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), attention_mask)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Siglip2Encoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.LayerList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class Siglip2VisionTransformer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes, return_dict=True, **kwargs):
        hidden_states = self.embeddings(pixel_values, spatial_shapes)
        attention_mask = None
        if pixel_attention_mask is not None:
            minimum = paddle.finfo(hidden_states.dtype).min
            attention_mask = (1 - pixel_attention_mask.astype(hidden_states.dtype))[:, None, None, :] * minimum
        hidden_states = self.encoder(hidden_states, attention_mask)
        hidden_states = self.post_layernorm(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=None)


class Siglip2VisionModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.vision_model = Siglip2VisionTransformer(config)

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)


class Lfm2VlMultiModalProjector(nn.Layer):
    def __init__(self, config):
        super().__init__()
        in_features = config.vision_config.hidden_size * config.downsample_factor**2
        self.factor = config.downsample_factor
        self.layer_norm = nn.LayerNorm(in_features) if config.projector_use_layernorm else None
        self.linear_1 = nn.Linear(in_features, config.projector_hidden_size, bias_attr=config.projector_bias)
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias_attr=config.projector_bias,
        )
        self.activation = ACT2FN[config.projector_hidden_act]

    def forward(self, hidden_states):
        batch_size, height, width, channels = hidden_states.shape
        factor = self.factor
        hidden_states = hidden_states.reshape([batch_size, height, width // factor, channels * factor])
        hidden_states = hidden_states.transpose([0, 2, 1, 3])
        hidden_states = hidden_states.reshape([batch_size, width // factor, height // factor, channels * factor**2])
        hidden_states = hidden_states.transpose([0, 2, 1, 3])
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        return self.linear_2(self.activation(self.linear_1(hidden_states)))


class Lfm2VlPreTrainedModel(PretrainedModel):
    config_class = Lfm2VlConfig
    base_model_prefix = "model"


@register_base_model
class Lfm2VlModel(Lfm2VlPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = Siglip2VisionModel(config.vision_config)
        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)
        self.language_model = Lfm2Model(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def get_image_features(self, pixel_values, spatial_shapes, pixel_attention_mask):
        outputs = self.vision_tower(pixel_values, pixel_attention_mask, spatial_shapes)
        feature_lengths = pixel_attention_mask.astype("int64").sum(axis=1)
        image_features = []
        for image_index in range(outputs.last_hidden_state.shape[0]):
            feature_length = int(feature_lengths[image_index])
            height, width = [int(item) for item in spatial_shapes[image_index].tolist()]
            features = outputs.last_hidden_state[image_index, :feature_length]
            features = features.reshape([1, height, width, -1])
            features = self.multi_modal_projector(features).reshape([-1, self.config.text_config.hidden_size])
            image_features.append(features)
        outputs.pooler_output = image_features
        return outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        spatial_shapes=None,
        pixel_attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        image_features = None
        if pixel_values is not None:
            image_features_list = self.get_image_features(
                pixel_values, spatial_shapes, pixel_attention_mask
            ).pooler_output
            image_features = paddle.concat(image_features_list, axis=0).astype(inputs_embeds.dtype)
            image_mask = input_ids == self.config.image_token_id
            if int(image_mask.astype("int64").sum()) != image_features.shape[0]:
                raise ValueError(
                    f"Image tokens and features do not match: {int(image_mask.sum())} versus {image_features.shape[0]}"
                )
            flat_embeddings = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            flat_mask = image_mask.flatten()
            flat_embeddings[flat_mask] = image_features
            inputs_embeds = flat_embeddings.reshape(inputs_embeds.shape)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return Lfm2VlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            image_hidden_states=image_features,
        )


class Lfm2VlForConditionalGeneration(Lfm2VlPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Lfm2VlModel(config)
        self.lm_head = LMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        checkpoint_path = (
            os.path.join(pretrained_model_name_or_path, "model.safetensors")
            if isinstance(pretrained_model_name_or_path, str)
            else ""
        )
        if not os.path.isfile(checkpoint_path):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        accepted_init_kwargs = {
            name for name in inspect.signature(cls.__init__).parameters if name not in {"self", "config"}
        }
        dtype = kwargs.pop("dtype", None)
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(config_path, return_unused_kwargs=True, **kwargs)
        else:
            model_kwargs = kwargs
        model_kwargs = {key: value for key, value in model_kwargs.items() if key in accepted_init_kwargs}
        if dtype is not None:
            config.dtype = dtype
            config.text_config.dtype = dtype
            config.vision_config.dtype = dtype
        with dtype_guard(dtype or paddle.get_default_dtype()):
            model = cls(config, *args, **model_kwargs)

        linear_weight_suffixes = (
            ".in_proj.weight",
            ".out_proj.weight",
            ".q_proj.weight",
            ".k_proj.weight",
            ".v_proj.weight",
            ".w1.weight",
            ".w2.weight",
            ".w3.weight",
            ".patch_embedding.weight",
            ".linear_1.weight",
            ".linear_2.weight",
            ".fc1.weight",
            ".fc2.weight",
        )
        state_dict = {}
        with safe_open(checkpoint_path, framework="np") as checkpoint:
            for name in checkpoint.keys():
                tensor = paddle.to_tensor(checkpoint.get_tensor(name))
                if name.endswith(linear_weight_suffixes):
                    tensor = tensor.transpose([1, 0]).contiguous()
                state_dict[name] = tensor
        if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.language_model.embed_tokens.weight"].clone()

        target_state_dict = model.state_dict()
        if set(state_dict) != set(target_state_dict):
            missing = sorted(set(target_state_dict) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(target_state_dict))
            raise ValueError(f"Incomplete LFM2-VL checkpoint conversion: missing={missing}, unexpected={unexpected}")
        for name, tensor in state_dict.items():
            target = target_state_dict[name]
            if list(tensor.shape) != list(target.shape):
                raise ValueError(f"LFM2-VL shape mismatch for {name}: source={tensor.shape}, target={target.shape}")
            if tensor.dtype != target.dtype:
                state_dict[name] = tensor.astype(target.dtype)
        missing_keys, unexpected_keys = model.set_state_dict(state_dict)
        if missing_keys or unexpected_keys:
            raise ValueError(f"LFM2-VL load failed: missing={missing_keys}, unexpected={unexpected_keys}")
        logger.info(f"Loaded and converted Hugging Face LFM2-VL checkpoint from {pretrained_model_name_or_path}")
        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, labels=None, **kwargs):
        outputs = self.model(**kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = self.criterion(logits, labels)[0] if labels is not None else None
        return Lfm2VlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "Lfm2VlForConditionalGeneration",
    "Lfm2VlModel",
    "Lfm2VlPreTrainedModel",
    "Siglip2VisionModel",
]
