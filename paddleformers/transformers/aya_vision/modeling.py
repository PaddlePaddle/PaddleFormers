# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import paddle
from paddle import nn

from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead as GeneralLMHead
from ..activations import ACT2FN
from ..cache_utils import Cache
from ..cohere2.modeling import Cohere2Model
from ..model_outputs import BaseModelOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel, register_base_model
from ..siglip_vision_model.modeling import SiglipVisionModel
from .configuration import AyaVisionConfig


def _normalize_aoa_dtype(dtype):
    if dtype is None:
        return None
    dtype = str(dtype)
    return dtype.split(".")[-1]


class AyaVisionMultiModalProjector(nn.Layer):
    def __init__(self, config: AyaVisionConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        input_size = config.vision_config.hidden_size * (config.downsample_factor**2)
        self.alignment_intermediate_size = config.alignment_intermediate_size
        self.layernorm = nn.LayerNorm(input_size, epsilon=config.adapter_layer_norm_eps)
        self.linear_1 = nn.Linear(input_size, self.alignment_intermediate_size)
        self.act = ACT2FN["silu"]
        self.linear_2 = nn.Linear(self.alignment_intermediate_size // 2, config.text_config.hidden_size)

    def pixel_shuffle(self, image_features):
        batch_size, seq_length, _ = image_features.shape
        height = width = int(seq_length**0.5)
        image_features = image_features.reshape([batch_size, width, height, -1])
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            [batch_size, width, height // self.downsample_factor, channels * self.downsample_factor]
        )
        image_features = image_features.transpose([0, 2, 1, 3])
        image_features = image_features.reshape(
            [batch_size, height // self.downsample_factor, width // self.downsample_factor, -1]
        )
        image_features = image_features.transpose([0, 2, 1, 3])
        return image_features

    def forward(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)
        x, gate = paddle.chunk(hidden_states, chunks=2, axis=-1)
        hidden_states = self.act(gate) * x
        return self.linear_2(hidden_states)


@dataclass
class AyaVisionModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[paddle.Tensor] = None


@dataclass
class AyaVisionCausalLMOutputWithPast(ModelOutput):
    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    image_hidden_states: Optional[paddle.Tensor] = None


class AyaVisionCriterionLayer(CriterionLayer):
    def __init__(self, config: AyaVisionConfig, return_tuple: bool = True):
        criterion_config = copy.copy(config.text_config)
        for attr in (
            "ignored_index",
            "ignore_index",
            "use_filtered_label_loss",
            "sequence_parallel",
            "tensor_parallel_output",
        ):
            if hasattr(config, attr):
                setattr(criterion_config, attr, getattr(config, attr))
        if not hasattr(criterion_config, "ignored_index") and hasattr(criterion_config, "ignore_index"):
            criterion_config.ignored_index = criterion_config.ignore_index
        super().__init__(criterion_config, return_tuple=return_tuple)

    def forward(self, logits: paddle.Tensor, labels: paddle.Tensor, loss_mask: Optional[paddle.Tensor] = None):
        logits_seq_len = logits[0].shape[1] if isinstance(logits, tuple) else logits.shape[1]
        if labels.shape[1] != logits_seq_len:
            labels = labels[:, -logits_seq_len:]
        if loss_mask is not None and loss_mask.shape[1] != logits_seq_len:
            loss_mask = loss_mask[:, -logits_seq_len:]
        if loss_mask is not None:
            loss_mask = loss_mask.astype("bool") & (labels != self.ignored_index)
        return super().forward(logits, labels, loss_mask=loss_mask)


class AyaVisionPreTrainedModel(PretrainedModel):
    config_class = AyaVisionConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "linear_1",
        "linear_2",
    ]
    _keys_to_ignore_on_load_unexpected = [r"rotary_emb.inv_freq", r"position_ids"]

    @classmethod
    def _gen_aoa_config(cls, config: AyaVisionConfig):
        model_prefix = "" if cls == AyaVisionModel else "model."
        text_prefix = f"{model_prefix}language_model"
        vision_prefix = f"{model_prefix}vision_tower"
        projector_prefix = f"{model_prefix}multi_modal_projector"

        aoa_statements = [
            f"language_model.model.embed_tokens.weight -> {text_prefix}.embed_tokens.weight",
            f"language_model.model.norm.weight -> {text_prefix}.norm.weight",
            f"multi_modal_projector.layernorm.weight -> {projector_prefix}.layernorm.weight",
            f"multi_modal_projector.layernorm.bias -> {projector_prefix}.layernorm.bias",
            f"multi_modal_projector.linear_1.weight^T -> {projector_prefix}.linear_1.weight",
            f"multi_modal_projector.linear_1.bias -> {projector_prefix}.linear_1.bias",
            f"multi_modal_projector.linear_2.weight^T -> {projector_prefix}.linear_2.weight",
            f"multi_modal_projector.linear_2.bias -> {projector_prefix}.linear_2.bias",
            f"vision_tower.vision_model.embeddings.patch_embedding.weight -> {vision_prefix}.embeddings.patch_embedding.weight",
            f"vision_tower.vision_model.embeddings.patch_embedding.bias -> {vision_prefix}.embeddings.patch_embedding.bias",
            f"vision_tower.vision_model.embeddings.position_embedding.weight -> {vision_prefix}.embeddings.position_embedding.weight",
            f"vision_tower.vision_model.post_layernorm.weight -> {vision_prefix}.post_layernorm.weight",
            f"vision_tower.vision_model.post_layernorm.bias -> {vision_prefix}.post_layernorm.bias",
        ]
        if getattr(config.vision_config, "vision_use_head", False):
            SiglipVisionModel._append_head_aoa_statements(
                aoa_statements, "vision_tower.vision_model.head", f"{vision_prefix}.head"
            )
        if cls != AyaVisionModel:
            lm_head_stmt = "language_model.model.embed_tokens.weight -> lm_head.weight"
            target_dtype = _normalize_aoa_dtype(getattr(config, "dtype", None))
            if target_dtype is not None:
                lm_head_stmt += f', src_dtype="{target_dtype}",dst_dtype="{target_dtype}"'
            aoa_statements.append(lm_head_stmt)

        for layer_id in range(config.text_config.num_hidden_layers):
            src = f"language_model.model.layers.{layer_id}"
            dst = f"{text_prefix}.layers.{layer_id}"
            aoa_statements += [
                f"{src}.input_layernorm.weight -> {dst}.input_layernorm.weight",
                f"{src}.self_attn.q_proj.weight^T -> {dst}.self_attn.q_proj.weight",
                f"{src}.self_attn.k_proj.weight^T -> {dst}.self_attn.k_proj.weight",
                f"{src}.self_attn.v_proj.weight^T -> {dst}.self_attn.v_proj.weight",
                f"{src}.self_attn.o_proj.weight^T -> {dst}.self_attn.o_proj.weight",
                f"{src}.mlp.gate_proj.weight^T -> {dst}.mlp.gate_proj.weight",
                f"{src}.mlp.up_proj.weight^T -> {dst}.mlp.up_proj.weight",
                f"{src}.mlp.down_proj.weight^T -> {dst}.mlp.down_proj.weight",
            ]
            if config.text_config.use_qk_norm:
                aoa_statements += [
                    f"{src}.self_attn.q_norm.weight -> {dst}.self_attn.q_norm.weight",
                    f"{src}.self_attn.k_norm.weight -> {dst}.self_attn.k_norm.weight",
                ]
        for layer_id in range(config.vision_config.num_hidden_layers):
            src = f"vision_tower.vision_model.encoder.layers.{layer_id}"
            dst = f"{vision_prefix}.encoder.layers.{layer_id}"
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
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: AyaVisionConfig):
        model_prefix = "" if cls == AyaVisionModel else "model."
        text_prefix = f"{model_prefix}language_model"
        vision_prefix = f"{model_prefix}vision_tower"
        projector_prefix = f"{model_prefix}multi_modal_projector"

        aoa_statements = [
            f"{text_prefix}.embed_tokens.weight -> language_model.model.embed_tokens.weight",
            f"{text_prefix}.norm.weight -> language_model.model.norm.weight",
            f"{projector_prefix}.layernorm.weight -> multi_modal_projector.layernorm.weight",
            f"{projector_prefix}.layernorm.bias -> multi_modal_projector.layernorm.bias",
            f"{projector_prefix}.linear_1.weight^T -> multi_modal_projector.linear_1.weight",
            f"{projector_prefix}.linear_1.bias -> multi_modal_projector.linear_1.bias",
            f"{projector_prefix}.linear_2.weight^T -> multi_modal_projector.linear_2.weight",
            f"{projector_prefix}.linear_2.bias -> multi_modal_projector.linear_2.bias",
            f"{vision_prefix}.embeddings.patch_embedding.weight -> vision_tower.vision_model.embeddings.patch_embedding.weight",
            f"{vision_prefix}.embeddings.patch_embedding.bias -> vision_tower.vision_model.embeddings.patch_embedding.bias",
            f"{vision_prefix}.embeddings.position_embedding.weight -> vision_tower.vision_model.embeddings.position_embedding.weight",
            f"{vision_prefix}.post_layernorm.weight -> vision_tower.vision_model.post_layernorm.weight",
            f"{vision_prefix}.post_layernorm.bias -> vision_tower.vision_model.post_layernorm.bias",
        ]
        if getattr(config.vision_config, "vision_use_head", False):
            SiglipVisionModel._append_head_inv_aoa_statements(
                aoa_statements, f"{vision_prefix}.head", "vision_tower.vision_model.head"
            )
        if cls != AyaVisionModel:
            aoa_statements.append(f"lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}")

        for layer_id in range(config.text_config.num_hidden_layers):
            src = f"{text_prefix}.layers.{layer_id}"
            dst = f"language_model.model.layers.{layer_id}"
            aoa_statements += [
                f"{src}.input_layernorm.weight -> {dst}.input_layernorm.weight",
                f"{src}.self_attn.q_proj.weight^T -> {dst}.self_attn.q_proj.weight",
                f"{src}.self_attn.k_proj.weight^T -> {dst}.self_attn.k_proj.weight",
                f"{src}.self_attn.v_proj.weight^T -> {dst}.self_attn.v_proj.weight",
                f"{src}.self_attn.o_proj.weight^T -> {dst}.self_attn.o_proj.weight",
                f"{src}.mlp.gate_proj.weight^T -> {dst}.mlp.gate_proj.weight",
                f"{src}.mlp.up_proj.weight^T -> {dst}.mlp.up_proj.weight",
                f"{src}.mlp.down_proj.weight^T -> {dst}.mlp.down_proj.weight",
            ]
            if config.text_config.use_qk_norm:
                aoa_statements += [
                    f"{src}.self_attn.q_norm.weight -> {dst}.self_attn.q_norm.weight",
                    f"{src}.self_attn.k_norm.weight -> {dst}.self_attn.k_norm.weight",
                ]

        for layer_id in range(config.vision_config.num_hidden_layers):
            src = f"{vision_prefix}.encoder.layers.{layer_id}"
            dst = f"vision_tower.vision_model.encoder.layers.{layer_id}"
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
        return {"aoa_statements": aoa_statements}


@register_base_model
class AyaVisionModel(AyaVisionPreTrainedModel):
    _checkpoint_conversion_mapping = {
        r"^language_model\.model": "language_model",
        r"^vision_tower\.vision_model": "vision_tower",
        r"^multi_modal_projector": "multi_modal_projector",
    }

    def __init__(self, config: AyaVisionConfig):
        super().__init__(config)
        if config.vision_config.model_type != "siglip_vision_model":
            raise NotImplementedError("AyaVision currently supports SigLIP vision_config only.")
        if config.text_config.model_type != "cohere2":
            raise NotImplementedError("AyaVision currently supports Cohere2 text_config only.")
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = AyaVisionMultiModalProjector(config)
        self.language_model = Cohere2Model(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        output_hidden_states=None,
        **kwargs,
    ):
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        image_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
            **kwargs,
        )
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = paddle.concat(hs_pool, axis=-1)
        return self.multi_modal_projector(selected_image_feature)

    def _merge_image_features(self, input_ids, inputs_embeds, image_features):
        if input_ids is None:
            image_token_embed = self.get_input_embeddings()(
                paddle.to_tensor([self.config.image_token_index], dtype="int64")
            )[0]
            special_image_mask = (inputs_embeds == image_token_embed).all(axis=-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index
        n_image_tokens = int(special_image_mask.astype("int64").sum().item())
        flat_image_features = image_features.reshape([-1, image_features.shape[-1]])
        if n_image_tokens != flat_image_features.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                f"features: {flat_image_features.shape[0]}"
            )
        indices = paddle.nonzero(special_image_mask)
        scattered = paddle.scatter_nd(indices, flat_image_features.astype(inputs_embeds.dtype), inputs_embeds.shape)
        keep_mask = (~special_image_mask).unsqueeze(-1).astype(inputs_embeds.dtype)
        return inputs_embeds * keep_mask + scattered

    def get_rope_index(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            text_position_ids = paddle.arange(seq_length, dtype="int64").reshape([1, seq_length])
            text_position_ids = text_position_ids.expand([batch_size, seq_length])
        else:
            attention_mask = attention_mask.astype("int64")
            text_position_ids = paddle.cumsum(attention_mask, axis=-1) - 1
            text_position_ids = paddle.where(
                attention_mask == 1,
                text_position_ids,
                paddle.zeros_like(text_position_ids),
            )

        position_ids = text_position_ids.unsqueeze(0).expand([3, batch_size, seq_length])
        return position_ids, None

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        vision_feature_layer: Optional[Union[int, list]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            inputs_embeds = self._merge_image_features(input_ids, inputs_embeds, image_features)

        if position_ids is not None and len(position_ids.shape) == 3:
            position_ids = position_ids[0]

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            attn_mask_startend_row_indices=kwargs.get("attn_mask_startend_row_indices", None),
        )
        if not return_dict:
            return (
                outputs.last_hidden_state,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                image_features,
            )
        return AyaVisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class AyaVisionForConditionalGeneration(AyaVisionPreTrainedModel):
    _checkpoint_conversion_mapping = {
        r"^language_model\.model": "language_model",
        r"^vision_tower\.vision_model": "vision_tower",
        r"^multi_modal_projector": "multi_modal_projector",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: AyaVisionConfig):
        super().__init__(config)
        self.model = AyaVisionModel(config)
        for attr in (
            "sequence_parallel",
            "tensor_parallel_output",
            "use_filtered_label_loss",
        ):
            if hasattr(config, attr):
                setattr(config.text_config, attr, getattr(config, attr))
        self.lm_head = GeneralLMHead(config.text_config)
        self.criterion = AyaVisionCriterionLayer(config)
        if config.tie_word_embeddings:
            self.tie_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.tie_weights()
        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def get_image_features(
        self, pixel_values, vision_feature_layer=None, vision_feature_select_strategy=None, **kwargs
    ):
        return self.model.get_image_features(
            pixel_values, vision_feature_layer, vision_feature_select_strategy, **kwargs
        )

    def get_rope_index(self, input_ids, attention_mask=None, **kwargs):
        return self.model.get_rope_index(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        vision_feature_layer: Optional[Union[int, list]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[paddle.Tensor] = None,
        loss_mask: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            attn_mask_startend_row_indices=kwargs.get("attn_mask_startend_row_indices", None),
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None, None)
        else:
            slice_indices = logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels, loss_mask=loss_mask)
            if isinstance(loss, tuple):
                loss = loss[0]
        if not return_dict:
            output = (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                outputs.image_hidden_states,
            )
            return (loss,) + output if loss is not None else output
        return AyaVisionCausalLMOutputWithPast(
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
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if is_first_iteration or not kwargs.get("use_cache", True) or past_key_values is None:
            model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["pixel_values"] = None
        return model_inputs

    def _get_image_tile_nums(self, input_ids):
        image_tokens_per_tile = (
            self.config.vision_config.image_size
            // self.config.vision_config.patch_size
            // self.config.downsample_factor
        ) ** 2
        image_token_counts = (input_ids == self.config.image_token_index).astype("int64").sum(axis=-1)
        image_tile_nums = []
        for image_token_count in image_token_counts.tolist():
            image_token_count = int(image_token_count)
            if image_token_count % image_tokens_per_tile != 0:
                raise ValueError(
                    f"Image tokens per sample must be divisible by tokens per tile, got "
                    f"{image_token_count} and {image_tokens_per_tile}."
                )
            image_tile_nums.append(image_token_count // image_tokens_per_tile)
        return image_tile_nums

    def expand_inputs_for_generation(self, input_ids, expand_size, attention_mask=None, **model_kwargs):
        source_input_ids = input_ids
        input_ids, model_kwargs = super().expand_inputs_for_generation(
            input_ids,
            expand_size=expand_size,
            attention_mask=attention_mask,
            **model_kwargs,
        )

        def _repeat_interleave_samples(x, lengths, repeat_times):
            samples = paddle.split(x, lengths, axis=0)
            out = []
            for sample in samples:
                reps = [repeat_times] + [1] * (len(sample.shape) - 1)
                out.append(paddle.tile(sample, reps))
            return paddle.concat(out, axis=0)

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_tile_nums = self._get_image_tile_nums(source_input_ids)
            pixel_values = dict_to_expand.get("pixel_values")
            if pixel_values is None:
                return dict_to_expand
            if sum(image_tile_nums) != pixel_values.shape[0]:
                raise ValueError(
                    f"Image tokens and pixel values do not match, tokens imply {sum(image_tile_nums)} tiles, "
                    f"but pixel_values has {pixel_values.shape[0]} tiles."
                )
            dict_to_expand["pixel_values"] = _repeat_interleave_samples(
                pixel_values, lengths=image_tile_nums, repeat_times=expand_size
            )
            return dict_to_expand

        if expand_size > 1:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        return input_ids, model_kwargs


__all__ = [
    "AyaVisionConfig",
    "AyaVisionPreTrainedModel",
    "AyaVisionModel",
    "AyaVisionForConditionalGeneration",
]
