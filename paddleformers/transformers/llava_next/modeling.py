# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Paddle Llava-NeXT model."""

from __future__ import annotations

import hashlib as _hashlib
import importlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as _np
import paddle
from paddle import nn
from safetensors import safe_open

from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead as GeneralLMHead
from ..activations import ACT2FN
from ..auto.configuration import config_class_to_model_type, model_type_to_module_name
from ..model_outputs import BaseModelOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel, dtype_guard, register_base_model
from .configuration import LlavaNextConfig
from .image_processor import select_best_resolution


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()
    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()
    height, width = select_best_resolution(image_size, grid_pinpoints)
    return (height // patch_size) * (width // patch_size) + 1


def _normalize_llava_next_position_ids(position_ids):
    if position_ids is not None and len(position_ids.shape) == 3 and position_ids.shape[0] == 1:
        return position_ids.squeeze(0)
    return position_ids


_GRANITE_TEXT_LINEAR_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _sync_sub_config_dtype(config, dtype=None):
    target_dtype = dtype or getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if target_dtype is None:
        return

    config.dtype = target_dtype
    config.torch_dtype = target_dtype
    for sub_config_name in ("text_config", "vision_config"):
        sub_config = getattr(config, sub_config_name, None)
        if sub_config is None:
            continue
        sub_config.dtype = target_dtype
        sub_config.torch_dtype = target_dtype


def _load_weight_map(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _iter_hf_tensors(model_dir):
    weight_map = _load_weight_map(model_dir)
    if weight_map is None:
        path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(path):
            return
        with safe_open(path, framework="np") as shard:
            for key in shard.keys():
                yield key, paddle.to_tensor(shard.get_tensor(key))
        return

    file_to_keys = defaultdict(list)
    for key, filename in weight_map.items():
        file_to_keys[filename].append(key)
    for filename in sorted(file_to_keys):
        with safe_open(os.path.join(model_dir, filename), framework="np") as shard:
            for key in sorted(file_to_keys[filename]):
                yield key, paddle.to_tensor(shard.get_tensor(key))


def _load_hf_granite_text_state_dict(model, model_dir):
    if not (
        isinstance(model_dir, str)
        and os.path.isdir(model_dir)
        and (
            os.path.exists(os.path.join(model_dir, "model.safetensors"))
            or os.path.exists(os.path.join(model_dir, "model.safetensors.index.json"))
        )
    ):
        return

    target_state_dict = model.state_dict()
    state_dict = {}
    tied_lm_head = None
    explicit_lm_head = False

    for hf_key, tensor in _iter_hf_tensors(model_dir):
        if hf_key == "lm_head.weight":
            if "lm_head.weight" in target_state_dict:
                state_dict["lm_head.weight"] = tensor
                explicit_lm_head = True
            continue
        if not hf_key.startswith("language_model."):
            continue

        text_key = hf_key[len("language_model.") :]
        if text_key == "model.embed_tokens.weight":
            target_key = "model.language_model.embed_tokens.weight"
            if target_key in target_state_dict:
                state_dict[target_key] = tensor
                tied_lm_head = tensor.clone()
            continue
        if text_key == "lm_head.weight":
            if "lm_head.weight" in target_state_dict:
                state_dict["lm_head.weight"] = tensor
                explicit_lm_head = True
            continue
        if text_key == "model.norm.weight":
            target_key = "model.language_model.norm.weight"
            if target_key in target_state_dict:
                state_dict[target_key] = tensor
            continue
        if not text_key.startswith("model.layers."):
            continue

        layer_suffix = text_key[len("model.layers.") :]
        layer_id, rest = layer_suffix.split(".", 1)
        target_key = f"model.language_model.layers.{layer_id}.{rest}"
        if target_key not in target_state_dict:
            continue
        if rest.endswith(_GRANITE_TEXT_LINEAR_WEIGHT_SUFFIXES):
            state_dict[target_key] = tensor.transpose([1, 0]).contiguous()
        else:
            state_dict[target_key] = tensor

    if tied_lm_head is not None and not explicit_lm_head and "lm_head.weight" in target_state_dict:
        state_dict["lm_head.weight"] = tied_lm_head

    for name, tensor in list(state_dict.items()):
        if name in target_state_dict and tensor.dtype != target_state_dict[name].dtype:
            state_dict[name] = tensor.astype(target_state_dict[name].dtype)

    with paddle.no_grad():
        for name, tensor in state_dict.items():
            target_state_dict[name].set_value(tensor)


def unpad_image(tensor, original_size):
    if not isinstance(original_size, (list, tuple)):
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        return tensor[:, padding : current_height - padding, :]
    scale_factor = current_height / original_height
    new_width = int(round(original_width * scale_factor, 7))
    padding = (current_width - new_width) // 2
    return tensor[:, :, padding : current_width - padding]


def _get_image_token_counts(input_ids, image_token_id):
    if input_ids is None:
        raise ValueError("`image_sizes` is required when `input_ids` is not provided.")
    image_token_counts = []
    for sample in input_ids.tolist():
        run_length = 0
        for token_id in sample:
            if int(token_id) == image_token_id:
                run_length += 1
            elif run_length > 0:
                image_token_counts.append(run_length)
                run_length = 0
        if run_length > 0:
            image_token_counts.append(run_length)
    return image_token_counts


def _infer_image_layout_from_token_count(
    token_count,
    image_seq_length,
    image_grid_pinpoints,
    image_size,
    patch_size,
    max_patches,
):
    extra_features = token_count - image_seq_length
    if extra_features <= 0:
        raise ValueError(f"Cannot infer LlavaNext image layout from {token_count} image tokens.")

    patches_height = image_size // patch_size
    patches_width = image_size // patch_size
    candidates = []
    for height, width in image_grid_pinpoints:
        grid_height = height // image_size
        grid_width = width // image_size
        num_patches = grid_height * grid_width + 1
        if num_patches > max_patches:
            continue

        current_height = patches_height * grid_height
        current_width = patches_width * grid_width
        for unpadded_height in range(1, current_height + 1):
            if extra_features % unpadded_height != 0:
                continue
            unpadded_width = extra_features // unpadded_height - 1
            if not (1 <= unpadded_width <= current_width):
                continue
            if unpadded_height != current_height and unpadded_width != current_width:
                continue
            wasted_features = current_height * current_width - unpadded_height * unpadded_width
            candidates.append((wasted_features, num_patches, grid_height, grid_width, unpadded_height, unpadded_width))

    if not candidates:
        raise ValueError(
            "`image_sizes` is required for this LlavaNext batch: cannot infer image layout from "
            f"{token_count} image tokens and {max_patches} visual patches."
        )

    _, num_patches, grid_height, grid_width, unpadded_height, unpadded_width = sorted(candidates)[0]
    return {
        "num_patches": num_patches,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "unpadded_height": unpadded_height,
        "unpadded_width": unpadded_width,
    }


@dataclass
class LlavaNextModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[paddle.Tensor] = None


@dataclass
class LlavaNextCausalLMOutputWithPast(ModelOutput):
    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple[paddle.Tensor]] = None
    attentions: Optional[tuple[paddle.Tensor]] = None
    image_hidden_states: Optional[paddle.Tensor] = None


class LlavaNextMultiModalProjector(nn.Layer):
    def __init__(self, config: LlavaNextConfig):
        super().__init__()
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias_attr=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias_attr=config.multimodal_projector_bias,
        )

    def forward(self, image_features):
        return self.linear_2(self.act(self.linear_1(image_features)))


def _language_model_from_config(config):
    model_type = config_class_to_model_type(config.__class__.__name__) or getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError(f"Cannot infer language model type from config class {config.__class__.__name__}.")
    module_name = model_type_to_module_name(model_type)
    module = importlib.import_module(f"paddleformers.transformers.{module_name}.modeling")
    class_name = config.__class__.__name__.removesuffix("Config") + "Model"
    if not hasattr(module, class_name):
        raise ValueError(f"Cannot find language model class {class_name} in paddleformers.transformers.{module_name}.")
    return getattr(module, class_name)._from_config(config)


def _disable_granite_llava_padding_mask(language_model, text_config):
    if getattr(text_config, "model_type", None) != "granite":
        return
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if embed_tokens is not None and getattr(embed_tokens, "_padding_idx", None) is not None:
        embed_tokens._padding_idx = None


def _vision_model_from_config(config):
    model_type = config_class_to_model_type(config.__class__.__name__) or getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError(f"Cannot infer vision model type from config class {config.__class__.__name__}.")
    module_name = (
        "siglip_vision_model" if model_type == "siglip_vision_model" else model_type_to_module_name(model_type)
    )
    module = importlib.import_module(f"paddleformers.transformers.{module_name}.modeling")
    class_name = config.__class__.__name__.removesuffix("Config") + "Model"
    if not hasattr(module, class_name):
        raise ValueError(f"Cannot find vision model class {class_name} in paddleformers.transformers.{module_name}.")
    return getattr(module, class_name)._from_config(config)


class LlavaNextPreTrainedModel(PretrainedModel):
    config_class = LlavaNextConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    transpose_weight_keys = [
        "linear_1",
        "linear_2",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
    ]

    def _init_weights(self, layer):
        std = getattr(self.config, "initializer_range", getattr(self.config.text_config, "initializer_range", 0.02))
        if isinstance(layer, (nn.Linear, nn.Conv2D)):
            layer.weight.set_value(paddle.normal(mean=0.0, std=std, shape=layer.weight.shape))
            if getattr(layer, "bias", None) is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))

    @classmethod
    def _gen_aoa_config(cls, config: LlavaNextConfig):
        is_conditional_generation = cls.__name__ == "LlavaNextForConditionalGeneration"
        # HF checkpoint keys carry no leading "model." prefix: the text tower is
        # "language_model.model.*" (GraniteForCausalLM wraps GraniteModel) and the
        # vision tower is "vision_tower.vision_model.*". Paddle params are instead
        # namespaced under "model." with no inner "model." on the text tower.
        llm_src_prefix = "language_model.model."
        vision_src_prefix = "vision_tower.vision_model."
        dst_model_prefix = "model."
        llm_dst_prefix = f"{dst_model_prefix}language_model."
        vision_dst_prefix = f"{dst_model_prefix}vision_tower."
        projector_dst_prefix = f"{dst_model_prefix}multi_modal_projector."

        aoa_statements = [
            f"{llm_src_prefix}embed_tokens.weight -> {llm_dst_prefix}embed_tokens.weight",
            f"{llm_src_prefix}norm.weight -> {llm_dst_prefix}norm.weight",
            f"{llm_src_prefix}layers.$LAYER_ID.input_layernorm.weight -> {llm_dst_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"{llm_src_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_dst_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"image_newline -> {dst_model_prefix}image_newline",
            f"multi_modal_projector.linear_1.weight^T -> {projector_dst_prefix}linear_1.weight",
            f"multi_modal_projector.linear_1.bias -> {projector_dst_prefix}linear_1.bias",
            f"multi_modal_projector.linear_2.weight^T -> {projector_dst_prefix}linear_2.weight",
            f"multi_modal_projector.linear_2.bias -> {projector_dst_prefix}linear_2.bias",
            f"{vision_src_prefix}embeddings.patch_embedding.weight -> {vision_dst_prefix}embeddings.patch_embedding.weight",
            f"{vision_src_prefix}embeddings.patch_embedding.bias -> {vision_dst_prefix}embeddings.patch_embedding.bias",
            f"{vision_src_prefix}embeddings.position_embedding.weight -> {vision_dst_prefix}embeddings.position_embedding.weight",
            f"{vision_src_prefix}post_layernorm.weight -> {vision_dst_prefix}post_layernorm.weight",
            f"{vision_src_prefix}post_layernorm.bias -> {vision_dst_prefix}post_layernorm.bias",
        ]
        aoa_statements.extend(
            [
                f"{llm_src_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> {llm_dst_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )
        aoa_statements.extend(
            [
                f"{llm_src_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight^T -> {llm_dst_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        for layer_id in range(config.vision_config.num_hidden_layers):
            src = f"{vision_src_prefix}encoder.layers.{layer_id}"
            dst = f"{vision_dst_prefix}encoder.layers.{layer_id}"
            aoa_statements += [
                f"{src}.layer_norm1.weight -> {dst}.layer_norm1.weight",
                f"{src}.layer_norm1.bias -> {dst}.layer_norm1.bias",
                f"{src}.layer_norm2.weight -> {dst}.layer_norm2.weight",
                f"{src}.layer_norm2.bias -> {dst}.layer_norm2.bias",
                f"{src}.self_attn.q_proj.weight^T -> {dst}.self_attn.q_proj.weight",
                f"{src}.self_attn.q_proj.bias -> {dst}.self_attn.q_proj.bias",
                f"{src}.self_attn.k_proj.weight^T -> {dst}.self_attn.k_proj.weight",
                f"{src}.self_attn.k_proj.bias -> {dst}.self_attn.k_proj.bias",
                f"{src}.self_attn.v_proj.weight^T -> {dst}.self_attn.v_proj.weight",
                f"{src}.self_attn.v_proj.bias -> {dst}.self_attn.v_proj.bias",
                f"{src}.self_attn.out_proj.weight^T -> {dst}.self_attn.out_proj.weight",
                f"{src}.self_attn.out_proj.bias -> {dst}.self_attn.out_proj.bias",
                f"{src}.mlp.fc1.weight^T -> {dst}.mlp.fc1.weight",
                f"{src}.mlp.fc1.bias -> {dst}.mlp.fc1.bias",
                f"{src}.mlp.fc2.weight^T -> {dst}.mlp.fc2.weight",
                f"{src}.mlp.fc2.bias -> {dst}.mlp.fc2.bias",
            ]
        if is_conditional_generation:
            if config.tie_word_embeddings:
                aoa_statements.append(f"{llm_src_prefix}embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: LlavaNextConfig):
        is_conditional_generation = cls.__name__ == "LlavaNextForConditionalGeneration"
        src_model_prefix = "model."
        llm_src_prefix = f"{src_model_prefix}language_model."
        vision_src_prefix = f"{src_model_prefix}vision_tower."
        projector_src_prefix = f"{src_model_prefix}multi_modal_projector."
        llm_dst_prefix = "language_model.model."
        vision_dst_prefix = "vision_tower.vision_model."

        aoa_statements = [
            f"{llm_src_prefix}embed_tokens.weight -> {llm_dst_prefix}embed_tokens.weight",
            f"{llm_src_prefix}norm.weight -> {llm_dst_prefix}norm.weight",
            f"{llm_src_prefix}layers.$LAYER_ID.input_layernorm.weight -> {llm_dst_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"{llm_src_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_dst_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{src_model_prefix}image_newline -> image_newline",
            f"{projector_src_prefix}linear_1.weight^T -> multi_modal_projector.linear_1.weight",
            f"{projector_src_prefix}linear_1.bias -> multi_modal_projector.linear_1.bias",
            f"{projector_src_prefix}linear_2.weight^T -> multi_modal_projector.linear_2.weight",
            f"{projector_src_prefix}linear_2.bias -> multi_modal_projector.linear_2.bias",
            f"{vision_src_prefix}embeddings.patch_embedding.weight -> {vision_dst_prefix}embeddings.patch_embedding.weight",
            f"{vision_src_prefix}embeddings.patch_embedding.bias -> {vision_dst_prefix}embeddings.patch_embedding.bias",
            f"{vision_src_prefix}embeddings.position_embedding.weight -> {vision_dst_prefix}embeddings.position_embedding.weight",
            f"{vision_src_prefix}post_layernorm.weight -> {vision_dst_prefix}post_layernorm.weight",
            f"{vision_src_prefix}post_layernorm.bias -> {vision_dst_prefix}post_layernorm.bias",
        ]
        aoa_statements.extend(
            [
                f"{llm_src_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> {llm_dst_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )
        aoa_statements.extend(
            [
                f"{llm_src_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight^T -> {llm_dst_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        for layer_id in range(config.vision_config.num_hidden_layers):
            src = f"{vision_src_prefix}encoder.layers.{layer_id}"
            dst = f"{vision_dst_prefix}encoder.layers.{layer_id}"
            aoa_statements += [
                f"{src}.layer_norm1.weight -> {dst}.layer_norm1.weight",
                f"{src}.layer_norm1.bias -> {dst}.layer_norm1.bias",
                f"{src}.layer_norm2.weight -> {dst}.layer_norm2.weight",
                f"{src}.layer_norm2.bias -> {dst}.layer_norm2.bias",
                f"{src}.self_attn.q_proj.weight^T -> {dst}.self_attn.q_proj.weight",
                f"{src}.self_attn.q_proj.bias -> {dst}.self_attn.q_proj.bias",
                f"{src}.self_attn.k_proj.weight^T -> {dst}.self_attn.k_proj.weight",
                f"{src}.self_attn.k_proj.bias -> {dst}.self_attn.k_proj.bias",
                f"{src}.self_attn.v_proj.weight^T -> {dst}.self_attn.v_proj.weight",
                f"{src}.self_attn.v_proj.bias -> {dst}.self_attn.v_proj.bias",
                f"{src}.self_attn.out_proj.weight^T -> {dst}.self_attn.out_proj.weight",
                f"{src}.self_attn.out_proj.bias -> {dst}.self_attn.out_proj.bias",
                f"{src}.mlp.fc1.weight^T -> {dst}.mlp.fc1.weight",
                f"{src}.mlp.fc1.bias -> {dst}.mlp.fc1.bias",
                f"{src}.mlp.fc2.weight^T -> {dst}.mlp.fc2.weight",
                f"{src}.mlp.fc2.bias -> {dst}.mlp.fc2.bias",
            ]
        if is_conditional_generation and not config.tie_word_embeddings:
            aoa_statements.append("lm_head.weight -> lm_head.weight")
        return {"aoa_statements": aoa_statements}


@register_base_model
class LlavaNextModel(LlavaNextPreTrainedModel):
    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)
        _sync_sub_config_dtype(config)
        self.vision_tower = _vision_model_from_config(config.vision_config)
        self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        self.image_newline = self.create_parameter(
            [config.text_config.hidden_size],
            default_initializer=nn.initializer.Normal(std=1 / math.sqrt(config.text_config.hidden_size)),
        )
        self.vocab_size = config.text_config.vocab_size
        self.language_model = _language_model_from_config(config.text_config)
        _disable_granite_llava_padding_mask(self.language_model, config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        image_sizes: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required to build LlavaNext position_ids.")
        if attention_mask is not None:
            attention_mask = attention_mask.astype("bool")
            position_ids = attention_mask.astype("int64").cumsum(-1) - 1
            position_ids = paddle.where(attention_mask, position_ids, paddle.ones_like(position_ids))
        else:
            position_ids = paddle.arange(input_ids.shape[1], dtype="int64").reshape([1, -1])
            position_ids = position_ids.expand([input_ids.shape[0], -1])
        position_ids = position_ids.astype(input_ids.dtype).unsqueeze(0)
        position_deltas = paddle.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
        return position_ids, position_deltas

    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.reshape([num_patch_height, num_patch_width, height, width, -1])
                image_feature = image_feature.transpose([4, 0, 2, 1, 3]).flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_newline = image_newline.astype(image_feature.dtype)
                    newline = image_newline[:, None, None].expand([image_feature.shape[0], image_feature.shape[1], 1])
                    image_feature = paddle.concat([image_feature, newline], axis=-1)
                image_feature = image_feature.flatten(1, 2).transpose([1, 0])
                image_feature = paddle.concat([base_image_feature, image_feature], axis=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = paddle.concat([image_feature, image_newline[None].astype(image_feature.dtype)], 0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.shape[0])
        return new_image_features, paddle.to_tensor(feature_lens, dtype="int64")

    def pack_image_features_without_image_sizes(self, image_features, image_layouts, image_newline=None):
        new_image_features = []
        feature_lens = []
        for image_feature, image_layout in zip(image_features, image_layouts):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                image_feature = image_feature.reshape(
                    [image_layout["grid_height"], image_layout["grid_width"], height, width, -1]
                )
                image_feature = image_feature.transpose([4, 0, 2, 1, 3]).flatten(1, 2).flatten(2, 3)

                current_height, current_width = image_feature.shape[1:]
                top = (current_height - image_layout["unpadded_height"]) // 2
                left = (current_width - image_layout["unpadded_width"]) // 2
                image_feature = image_feature[
                    :,
                    top : top + image_layout["unpadded_height"],
                    left : left + image_layout["unpadded_width"],
                ]
                if image_newline is not None:
                    image_newline = image_newline.astype(image_feature.dtype)
                    newline = image_newline[:, None, None].expand([image_feature.shape[0], image_feature.shape[1], 1])
                    image_feature = paddle.concat([image_feature, newline], axis=-1)
                image_feature = image_feature.flatten(1, 2).transpose([1, 0])
                image_feature = paddle.concat([base_image_feature, image_feature], axis=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = paddle.concat([image_feature, image_newline[None].astype(image_feature.dtype)], 0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.shape[0])
        return new_image_features, paddle.to_tensor(feature_lens, dtype="int64")

    def get_image_features(
        self,
        pixel_values,
        image_sizes,
        input_ids=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        output_hidden_states=None,
        **kwargs,
    ):
        vision_feature_layer = (
            self.config.vision_feature_layer if vision_feature_layer is None else vision_feature_layer
        )
        vision_feature_select_strategy = (
            self.config.vision_feature_select_strategy
            if vision_feature_select_strategy is None
            else vision_feature_select_strategy
        )
        image_layouts = None
        if image_sizes is None:
            image_token_counts = _get_image_token_counts(input_ids, self.config.image_token_id)
            max_patches = pixel_values.shape[1] if len(pixel_values.shape) == 5 else pixel_values.shape[0]
            image_layouts = [
                _infer_image_layout_from_token_count(
                    token_count,
                    self.config.image_seq_length - (1 if vision_feature_select_strategy == "default" else 0),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                    self.config.vision_config.patch_size,
                    max_patches,
                )
                for token_count in image_token_counts
            ]
            image_num_patches = [layout["num_patches"] for layout in image_layouts]
        else:
            image_num_patches = [
                image_size_to_num_patches(
                    imsize, self.config.image_grid_pinpoints, self.config.vision_config.image_size
                )
                for imsize in image_sizes
            ]
        if len(pixel_values.shape) == 5:
            pixel_values = paddle.concat(
                [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)], axis=0
            )
        elif len(pixel_values.shape) != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expected 4 or 5 dimensions")

        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            selected_image_feature = paddle.concat(
                [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer], axis=-1
            )
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = paddle.split(image_features, image_num_patches, axis=0)
        if image_sizes is None:
            image_features, _ = self.pack_image_features_without_image_sizes(
                image_features,
                image_layouts,
                image_newline=self.image_newline,
            )
        else:
            image_features, _ = self.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )
        return image_features

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        if input_ids is None:
            image_token = paddle.to_tensor(self.config.image_token_id, dtype="int64")
            special_image_mask = inputs_embeds == self.get_input_embeddings()(image_token)
            special_image_mask = special_image_mask.all(axis=-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
        n_image_tokens = int(special_image_mask.astype("int64").sum().item())
        if n_image_tokens != image_features.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                f"features: {image_features.shape[0]}"
            )
        return special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        image_grid_thw=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        use_cache=None,
        output_hidden_states=None,
        cache_position=None,
        is_first_iteration=None,
        return_dict=None,
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if image_sizes is None and image_grid_thw is not None:
            image_sizes = image_grid_thw
        position_ids = _normalize_llava_next_position_ids(position_ids)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None and pixel_values.shape[0] > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                input_ids=input_ids,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features = paddle.concat(image_features, axis=0).astype(inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if not return_dict:
            return (outputs.last_hidden_state, outputs.past_key_values, image_features)
        return LlavaNextModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            image_hidden_states=image_features,
        )


class LlavaNextForConditionalGeneration(LlavaNextPreTrainedModel):
    _checkpoint_conversion_mapping = {
        r"^model\.vision_tower\.vision_model": "model.vision_tower",
        r"^model\.language_model": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)
        self.model = LlavaNextModel(config)
        self.lm_head = GeneralLMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        dtype = kwargs.get("dtype", None)
        config = kwargs.get("config", None)
        if config is not None:
            _sync_sub_config_dtype(config, dtype)
        convert_from_hf = kwargs.get("convert_from_hf", True)
        load_checkpoint_format = kwargs.get("load_checkpoint_format", "flex_checkpoint")
        with dtype_guard(dtype or paddle.get_default_dtype()):
            model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        if (
            convert_from_hf
            and load_checkpoint_format == "flex_checkpoint"
            and getattr(text_config, "model_type", None) == "granite"
        ):
            _load_hf_granite_text_state_dict(model, pretrained_model_name_or_path)
        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def pack_image_features(self, *args, **kwargs):
        return self.model.pack_image_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def get_rope_index(self, *args, **kwargs):
        return self.model.get_rope_index(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        image_grid_thw=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=0,
        return_dict=None,
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if image_sizes is None and image_grid_thw is not None:
            image_sizes = image_grid_thw
        position_ids = _normalize_llava_next_position_ids(position_ids)
        _dump_dir = os.environ.get("DUMP_FIRST_FORWARD_DIR")
        _do_dump = bool(_dump_dir) and not getattr(LlavaNextForConditionalGeneration, "_dumped_first", False)
        if _do_dump:
            LlavaNextForConditionalGeneration._dumped_first = True

            os.makedirs(_dump_dir, exist_ok=True)

            def _save_small(_name, _tensor):
                if _tensor is None:
                    return
                try:
                    _np.save(os.path.join(_dump_dir, f"{_name}.npy"), _np.asarray(_tensor.numpy()))
                except Exception as _e:
                    with open(os.path.join(_dump_dir, "dump_errors.txt"), "a") as _f:
                        _f.write(f"{_name}: {repr(_e)}\n")

            def _summary(_name, _tensor):
                _arr = _np.asarray(_tensor.numpy())
                _head = _np.ascontiguousarray(_arr.reshape(-1)[: min(_arr.size, 4096)])
                return (
                    f"{_name}: dtype={_tensor.dtype}, shape={list(_tensor.shape)}, "
                    f"mean={float(_arr.mean()):.8f}, mean_abs={float(_np.abs(_arr).mean()):.8f}, "
                    f"std={float(_arr.std()):.8f}, "
                    f"sha_head={_hashlib.sha1(_head.view(_np.uint8)).hexdigest()[:16]}\n"
                )

            for _name, _tensor in [
                ("input_ids", input_ids),
                ("labels", labels),
                ("image_sizes", image_sizes),
                ("attention_mask", attention_mask),
                ("attn_mask_startend_row_indices", kwargs.get("attn_mask_startend_row_indices")),
                ("position_ids", position_ids),
            ]:
                _save_small(_name, _tensor)
            if pixel_values is not None:
                _save_small("pixel_values", pixel_values)
            with open(os.path.join(_dump_dir, "config_and_weight_summary.txt"), "w") as _f:
                for _obj_name, _obj in [
                    ("config", self.config),
                    ("text_config", getattr(self.config, "text_config", None)),
                    ("vision_config", getattr(self.config, "vision_config", None)),
                ]:
                    if _obj is None:
                        continue
                    for _attr in [
                        "_attn_implementation",
                        "attention_dropout",
                        "attention_probs_dropout_prob",
                        "hidden_dropout_prob",
                        "dtype",
                        "max_sequence_length",
                        "seq_length",
                        "recompute_granularity",
                        "recompute_method",
                        "recompute_num_layers",
                        "embedding_multiplier",
                        "residual_multiplier",
                        "logits_scaling",
                    ]:
                        if hasattr(_obj, _attr):
                            _f.write(f"{_obj_name}.{_attr}={getattr(_obj, _attr)}\n")
                for _name, _tensor in [
                    ("embed_tokens", self.model.language_model.embed_tokens.weight),
                    ("lm_head", self.lm_head.weight),
                    ("layer0.q_proj", self.model.language_model.layers[0].self_attn.q_proj.weight),
                    ("layer0.k_proj", self.model.language_model.layers[0].self_attn.k_proj.weight),
                    ("layer0.v_proj", self.model.language_model.layers[0].self_attn.v_proj.weight),
                    ("layer0.o_proj", self.model.language_model.layers[0].self_attn.o_proj.weight),
                    ("layer0.gate_proj", self.model.language_model.layers[0].mlp.gate_proj.weight),
                    ("layer0.up_proj", self.model.language_model.layers[0].mlp.up_proj.weight),
                    ("layer0.down_proj", self.model.language_model.layers[0].mlp.down_proj.weight),
                    ("layer1.q_proj", self.model.language_model.layers[1].self_attn.q_proj.weight),
                    ("layer1.k_proj", self.model.language_model.layers[1].self_attn.k_proj.weight),
                    ("layer1.v_proj", self.model.language_model.layers[1].self_attn.v_proj.weight),
                    ("layer1.o_proj", self.model.language_model.layers[1].self_attn.o_proj.weight),
                    ("layer1.gate_proj", self.model.language_model.layers[1].mlp.gate_proj.weight),
                    ("layer1.up_proj", self.model.language_model.layers[1].mlp.up_proj.weight),
                    ("layer1.down_proj", self.model.language_model.layers[1].mlp.down_proj.weight),
                    ("final_norm", self.model.language_model.norm.weight),
                    ("projector.linear_1", self.model.multi_modal_projector.linear_1.weight),
                    ("projector.linear_2", self.model.multi_modal_projector.linear_2.weight),
                ]:
                    _f.write(_summary(_name, _tensor))
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        loss = None
        if labels is not None:
            logits = self.lm_head(hidden_states)
            loss, _ = self.criterion(logits, labels)
            if _do_dump:

                def _summary(_name, _tensor):
                    _arr = _np.asarray(_tensor.numpy())
                    _head = _np.ascontiguousarray(_arr.reshape(-1)[: min(_arr.size, 4096)])
                    return (
                        f"{_name}: dtype={_tensor.dtype}, shape={list(_tensor.shape)}, "
                        f"mean={float(_arr.mean()):.8f}, mean_abs={float(_np.abs(_arr).mean()):.8f}, "
                        f"std={float(_arr.std()):.8f}, "
                        f"sha_head={_hashlib.sha1(_head.view(_np.uint8)).hexdigest()[:16]}\n"
                    )

                with open(os.path.join(_dump_dir, "forward_summary.txt"), "w") as _f:
                    _f.write(f"criterion_loss={float(loss.numpy()):.8f}\n")
                    _f.write(f"labels_valid={int((labels != -100).astype('int64').sum().numpy())}\n")
                    _f.write(_summary("last_hidden_state", hidden_states))
                    _f.write(_summary("logits", logits))
                    if getattr(outputs, "image_hidden_states", None) is not None:
                        _f.write(_summary("image_hidden_states", outputs.image_hidden_states))
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        if not return_dict:
            output = (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                outputs.image_hidden_states,
            )
            return (loss,) + output if loss is not None else output
        return LlavaNextCausalLMOutputWithPast(
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
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        image_grid_thw=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        is_first_iteration=False,
        use_cache=True,
        **kwargs,
    ):
        if cache_position is None:
            if past_key_values is None:
                cache_position = paddle.arange(input_ids.shape[1])
            else:
                cache_position = paddle.to_tensor([input_ids.shape[1] - 1])

        if logits_to_keep is None:
            logits_to_keep = 1

        if image_sizes is None and image_grid_thw is not None:
            image_sizes = image_grid_thw

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position[0] == 0 or is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes
        else:
            model_inputs["pixel_values"] = None
            model_inputs["image_sizes"] = None
        model_inputs["cache_position"] = cache_position
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep
        return model_inputs

    def _get_image_nums_and_patch_nums(self, input_ids, image_sizes):
        if image_sizes is None:
            raise ValueError("`image_sizes` must be provided when expanding LlavaNext visual inputs.")

        if not isinstance(image_sizes, (list, tuple)):
            image_sizes = image_sizes.tolist()
        image_counts = (input_ids == self.config.image_token_id).astype("int64").sum(axis=-1).tolist()
        image_nums = []
        patch_nums = []
        image_idx = 0
        for image_count in image_counts:
            remain_tokens = int(image_count)
            num_images = 0
            num_patches = 0
            while remain_tokens > 0:
                if image_idx >= len(image_sizes):
                    raise ValueError(
                        f"Image tokens imply more images than provided by `image_sizes`: {len(image_sizes)}."
                    )
                image_size = image_sizes[image_idx]
                height, width = select_best_resolution(image_size, self.config.image_grid_pinpoints)
                grid_h = height // self.config.vision_config.image_size
                grid_w = width // self.config.vision_config.image_size
                patches_h = self.config.vision_config.image_size // self.config.vision_config.patch_size
                patches_w = self.config.vision_config.image_size // self.config.vision_config.patch_size
                unpadded, newline = self._get_unpadded_features(
                    image_size[0], image_size[1], patches_h, patches_w, grid_h, grid_w
                )
                tokens = unpadded + newline + patches_h * patches_w
                if self.config.vision_feature_select_strategy == "default":
                    tokens -= 1
                if remain_tokens < tokens:
                    raise ValueError(
                        f"Image tokens and image sizes do not match, remaining tokens: {remain_tokens}, "
                        f"next image tokens: {tokens}."
                    )
                remain_tokens -= tokens
                num_images += 1
                num_patches += (grid_h * grid_w) + 1
                image_idx += 1
            image_nums.append(num_images)
            patch_nums.append(num_patches)
        if image_idx != len(image_sizes):
            raise ValueError(
                f"`image_sizes` contains {len(image_sizes)} images, but input_ids imply {image_idx} images."
            )
        return image_nums, patch_nums

    def _infer_image_nums_and_patch_nums(self, input_ids, pixel_values):
        image_nums = []
        patch_nums = []
        max_patches = pixel_values.shape[1] if len(pixel_values.shape) == 5 else pixel_values.shape[0]
        for sample in input_ids:
            sample_input_ids = sample.unsqueeze(0)
            image_token_counts = _get_image_token_counts(sample_input_ids, self.config.image_token_id)
            image_nums.append(len(image_token_counts))
            sample_patch_nums = 0
            for token_count in image_token_counts:
                layout = _infer_image_layout_from_token_count(
                    token_count,
                    self.config.image_seq_length
                    - (1 if self.config.vision_feature_select_strategy == "default" else 0),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                    self.config.vision_config.patch_size,
                    max_patches,
                )
                sample_patch_nums += layout["num_patches"]
            patch_nums.append(sample_patch_nums)
        return image_nums, patch_nums

    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width
        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2
        return current_height * current_width, current_height

    def expand_inputs_for_generation(self, input_ids, expand_size, attention_mask=None, **model_kwargs):
        source_input_ids = input_ids
        input_ids, model_kwargs = super().expand_inputs_for_generation(
            input_ids,
            expand_size=expand_size,
            attention_mask=attention_mask,
            **model_kwargs,
        )

        if expand_size == 1:
            return input_ids, model_kwargs

        pixel_values = model_kwargs.get("pixel_values")
        image_sizes = model_kwargs.get("image_sizes")
        if pixel_values is None:
            return input_ids, model_kwargs

        if image_sizes is None:
            image_nums, patch_nums = self._infer_image_nums_and_patch_nums(source_input_ids, pixel_values)
        else:
            image_nums, patch_nums = self._get_image_nums_and_patch_nums(source_input_ids, image_sizes)

        def _repeat_interleave_samples(x, lengths, repeat_times):
            samples = paddle.split(x, lengths, axis=0)
            out = []
            for sample in samples:
                reps = [repeat_times] + [1] * (len(sample.shape) - 1)
                out.append(paddle.tile(sample, reps))
            return paddle.concat(out, axis=0)

        pixel_lengths = image_nums if len(pixel_values.shape) == 5 else patch_nums
        if sum(pixel_lengths) != pixel_values.shape[0]:
            raise ValueError(
                f"Image tokens and pixel values do not match, tokens imply {sum(pixel_lengths)} "
                f"visual entries, but pixel_values has {pixel_values.shape[0]} entries."
            )
        model_kwargs["pixel_values"] = _repeat_interleave_samples(pixel_values, pixel_lengths, expand_size)
        if image_sizes is not None:
            model_kwargs["image_sizes"] = _repeat_interleave_samples(image_sizes, image_nums, expand_size)
        return input_ids, model_kwargs


__all__ = [
    "LlavaNextForConditionalGeneration",
    "LlavaNextPreTrainedModel",
    "LlavaNextModel",
    "LlavaNextMultiModalProjector",
]
