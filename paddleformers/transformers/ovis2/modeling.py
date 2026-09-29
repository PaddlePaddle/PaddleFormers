# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Paddle implementation of Ovis2."""

import math
from dataclasses import dataclass
from typing import Optional, Union

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...generation import GenerationMixin
from ...nn.lm_head import LMHead as GeneralLMHead
from ...utils.log import logger
from ..activations import ACT2FN
from ..cache_utils import Cache
from ..model_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from ..model_utils import PretrainedModel
from ..qwen2.modeling import Qwen2Model
from .configuration import Ovis2Config, Ovis2VisionConfig


def _use_high_precision_cublas_for_fp32(tensor):
    """Match Torch's default FP32 GEMM policy for numerical parity."""
    if tensor.dtype != paddle.float32 or not tensor.place.is_gpu_place():
        return
    from paddle.base import core

    if core.get_cublas_switch():
        core.set_cublas_switch(False)


@dataclass
class BaseModelOutputWithVisualIndicatorFeatures(BaseModelOutputWithPooling):
    visual_indicator_features: Optional[paddle.Tensor] = None


@dataclass
class Ovis2ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[paddle.Tensor] = None


@dataclass
class Ovis2CausalLMOutputWithPast(ModelOutput):
    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[paddle.Tensor]] = None
    attentions: Optional[tuple[paddle.Tensor]] = None
    image_hidden_states: Optional[paddle.Tensor] = None


class Ovis2RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size], default_initializer=nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.square().mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class Ovis2VisionMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Ovis2VisionEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2D(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, config.hidden_size)
        self.register_buffer(
            "position_ids",
            paddle.arange(self.num_patches, dtype="int64").reshape([1, -1]),
            persistable=False,
        )
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values.astype(self.patch_embedding.weight.dtype))
        embeddings = patch_embeds.flatten(start_axis=2).transpose([0, 2, 1])
        return self.rms_norm(embeddings) + self.position_embedding(self.position_ids)


class Ovis2VisionAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("Vision hidden size must be divisible by the number of attention heads.")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=config.qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=config.qkv_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=config.qkv_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=config.qkv_bias)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        target_shape = [batch_size, sequence_length, self.num_heads, self.head_dim]
        query = self.q_proj(hidden_states).reshape(target_shape).transpose([0, 2, 1, 3])
        key = self.k_proj(hidden_states).reshape(target_shape).transpose([0, 2, 1, 3])
        value = self.v_proj(hidden_states).reshape(target_shape).transpose([0, 2, 1, 3])
        weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * self.scale
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = F.softmax(weights, axis=-1, dtype="float32").astype(query.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = paddle.matmul(weights, value)
        output = output.transpose([0, 2, 1, 3]).reshape([batch_size, sequence_length, self.embed_dim])
        return self.out_proj(output), weights


class Ovis2VisionEncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = Ovis2VisionAttention(config)
        self.ffn = Ovis2VisionMLP(config)
        self.rms_norm1 = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rms_norm2 = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        attention_output, _ = self.attention(self.rms_norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + attention_output
        return hidden_states + self.ffn(self.rms_norm2(hidden_states))


class Ovis2VisionEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.LayerList([Ovis2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask=None):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Ovis2VisionTransformer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Ovis2VisionEmbeddings(config)
        self.encoder = Ovis2VisionEncoder(config)
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, pixel_values, attention_mask=None):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states, attention_mask).last_hidden_state
        return BaseModelOutput(last_hidden_state=self.rms_norm(hidden_states))


class Ovis2VisualEmbeddingTable(nn.Embedding):
    def forward(self, visual_tokens):
        if visual_tokens.dtype in (paddle.int8, paddle.int16, paddle.int32, paddle.int64):
            return super().forward(visual_tokens)
        return paddle.matmul(visual_tokens, self.weight)


class Ovis2PreTrainedModel(PretrainedModel):
    config_class = Ovis2Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["Ovis2VisionAttention"]
    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def _gen_aoa_config(cls, config):
        model_prefix = "" if cls.__name__ == "Ovis2Model" else "model."
        source_prefix = "model."
        language_prefix = f"{model_prefix}language_model."
        vision_prefix = f"{model_prefix}vision_tower."
        visual_table_prefix = f"{model_prefix}visual_embeddings_table."
        statements = [
            f"{source_prefix}language_model.embed_tokens.weight -> {language_prefix}embed_tokens.weight",
            f"{source_prefix}language_model.norm.weight -> {language_prefix}norm.weight",
            f"{source_prefix}language_model.layers.$LAYER_ID.input_layernorm.weight -> {language_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"{source_prefix}language_model.layers.$LAYER_ID.post_attention_layernorm.weight -> {language_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{source_prefix}language_model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {language_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{source_prefix}language_model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {language_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
            f"{source_prefix}vision_tower.transformer.embeddings.patch_embedding.weight -> {vision_prefix}transformer.embeddings.patch_embedding.weight",
            f"{source_prefix}vision_tower.transformer.embeddings.patch_embedding.bias -> {vision_prefix}transformer.embeddings.patch_embedding.bias",
            f"{source_prefix}vision_tower.transformer.embeddings.position_embedding.weight -> {vision_prefix}transformer.embeddings.position_embedding.weight",
            f"{source_prefix}vision_tower.transformer.embeddings.rms_norm.weight -> {vision_prefix}transformer.embeddings.rms_norm.weight",
            f"{source_prefix}vision_tower.transformer.encoder.layers.$LAYER_ID.rms_norm1.weight -> {vision_prefix}transformer.encoder.layers.$LAYER_ID.rms_norm1.weight",
            f"{source_prefix}vision_tower.transformer.encoder.layers.$LAYER_ID.rms_norm2.weight -> {vision_prefix}transformer.encoder.layers.$LAYER_ID.rms_norm2.weight",
            f"{source_prefix}vision_tower.transformer.rms_norm.weight -> {vision_prefix}transformer.rms_norm.weight",
            f"{source_prefix}vision_tower.head_linear.weight^T -> {vision_prefix}head_linear.weight",
            f"{source_prefix}vision_tower.head_norm.weight -> {vision_prefix}head_norm.weight",
            f"{source_prefix}vision_tower.head_norm.bias -> {vision_prefix}head_norm.bias",
            f"{source_prefix}visual_embeddings_table.weight -> {visual_table_prefix}weight",
        ]
        # Flex checkpoint's multi-input actions must be concrete.  Expanding
        # these rules here avoids treating the separate HF projections as
        # unexpected keys and leaving Paddle's fused parameters uninitialized.
        for layer_index in range(config.text_config.num_hidden_layers):
            source_layer = f"{source_prefix}language_model.layers.{layer_index}"
            target_layer = f"{language_prefix}layers.{layer_index}"
            statements.extend(
                [
                    (
                        f"{source_layer}.self_attn.q_proj.weight^T, "
                        f"{source_layer}.self_attn.k_proj.weight^T, "
                        f"{source_layer}.self_attn.v_proj.weight^T "
                        f"-> {target_layer}.self_attn.qkv_proj.weight, fused_qkv, "
                        f"num_heads={config.text_config.num_attention_heads}, "
                        f"num_key_value_groups={config.text_config.num_key_value_heads}"
                    ),
                    (
                        f"{source_layer}.self_attn.q_proj.bias, "
                        f"{source_layer}.self_attn.k_proj.bias, "
                        f"{source_layer}.self_attn.v_proj.bias "
                        f"-> {target_layer}.self_attn.qkv_proj.bias, fused_qkv, "
                        f"num_heads={config.text_config.num_attention_heads}, "
                        f"num_key_value_groups={config.text_config.num_key_value_heads}, axis=0"
                    ),
                    (
                        f"{source_layer}.mlp.gate_proj.weight^T, "
                        f"{source_layer}.mlp.up_proj.weight^T "
                        f"-> {target_layer}.mlp.up_gate_proj.weight, fused_ffn"
                    ),
                ]
            )
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            statements.append(
                f"{source_prefix}vision_tower.transformer.encoder.layers.$LAYER_ID.attention.{projection}.weight^T -> "
                f"{vision_prefix}transformer.encoder.layers.$LAYER_ID.attention.{projection}.weight"
            )
            if config.vision_config.qkv_bias:
                statements.append(
                    f"{source_prefix}vision_tower.transformer.encoder.layers.$LAYER_ID.attention.{projection}.bias -> "
                    f"{vision_prefix}transformer.encoder.layers.$LAYER_ID.attention.{projection}.bias"
                )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            statements.append(
                f"{source_prefix}vision_tower.transformer.encoder.layers.$LAYER_ID.ffn.{projection}.weight^T -> "
                f"{vision_prefix}transformer.encoder.layers.$LAYER_ID.ffn.{projection}.weight"
            )
            if config.vision_config.mlp_bias:
                statements.append(
                    f"{source_prefix}vision_tower.transformer.encoder.layers.$LAYER_ID.ffn.{projection}.bias -> "
                    f"{vision_prefix}transformer.encoder.layers.$LAYER_ID.ffn.{projection}.bias"
                )
        if cls.__name__ != "Ovis2Model":
            statements.append(f"{source_prefix}language_model.embed_tokens.weight -> lm_head.weight")
        return {"aoa_statements": statements}


def hard_softmax(logits, axis):
    soft = F.softmax(logits, axis=axis)
    hard = F.one_hot(soft.argmax(axis=axis), logits.shape[axis]).astype(logits.dtype)
    return hard - soft.detach() + soft


class Ovis2VisionModel(Ovis2PreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.transformer = Ovis2VisionTransformer(config)
        self.num_visual_indicator_tokens = config.num_visual_indicator_tokens
        self.vocab_size = config.vocab_size
        self.head_linear = nn.Linear(
            config.hidden_size * config.hidden_stride * config.hidden_stride,
            config.vocab_size - config.num_visual_indicator_tokens,
            bias_attr=False,
        )
        self.head_norm = nn.LayerNorm(config.vocab_size - config.num_visual_indicator_tokens)

    def forward(self, pixel_values, attention_mask=None, return_dict=True, **kwargs):
        _use_high_precision_cublas_for_fp32(pixel_values)
        last_hidden_state = self.transformer(pixel_values, attention_mask).last_hidden_state
        if self.config.hidden_stride > 1:
            num_images, sequence_length, hidden_size = last_hidden_state.shape
            stride = self.config.hidden_stride
            side = int(math.sqrt(sequence_length))
            if side * side != sequence_length:
                raise ValueError("Vision token sequence length must be a perfect square.")
            hidden = last_hidden_state.reshape([num_images, side, side, hidden_size])
            pad_size = (stride - side % stride) % stride
            if pad_size:
                hidden = paddle.concat(
                    [hidden, paddle.zeros([num_images, pad_size, side, hidden_size], dtype=hidden.dtype)], axis=1
                )
                hidden = paddle.concat(
                    [
                        hidden,
                        paddle.zeros([num_images, side + pad_size, pad_size, hidden_size], dtype=hidden.dtype),
                    ],
                    axis=2,
                )
                side += pad_size
            hidden = hidden.reshape(
                [num_images, side // stride, stride, side // stride, stride, hidden_size]
            ).transpose([0, 1, 3, 2, 4, 5])
            last_hidden_state = hidden.reshape([num_images, -1, stride * stride * hidden_size])
        logits = self.head_norm(self.head_linear(last_hidden_state))
        if self.config.tokenize_function == "gumbel_argmax":
            probabilities = F.gumbel_softmax(logits, axis=-1, hard=True)
        elif self.config.tokenize_function == "st_argmax":
            probabilities = hard_softmax(logits, -1)
        elif self.config.tokenize_function == "softmax":
            probabilities = F.softmax(logits, axis=-1)
        else:
            raise ValueError(f"Unknown visual tokenize_function: {self.config.tokenize_function}")
        output = BaseModelOutputWithVisualIndicatorFeatures(
            last_hidden_state=last_hidden_state, pooler_output=probabilities
        )
        return output if return_dict else output.to_tuple()


class Ovis2Model(Ovis2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = Ovis2VisionModel(config.vision_config)
        self.language_model = Qwen2Model(config.text_config)
        self.visual_embeddings_table = Ovis2VisualEmbeddingTable(
            config.vision_config.vocab_size, config.hidden_size
        )
        self.visual_vocab_size = config.vision_config.vocab_size
        self.vocab_size = config.vocab_size
        self.visual_indicator_token_ids = config.visual_indicator_token_ids

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_image_features(self, pixel_values, **kwargs):
        kwargs.pop("return_dict", None)
        outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        features = outputs.pooler_output
        padding = paddle.zeros(
            [features.shape[0], features.shape[1], self.vision_tower.num_visual_indicator_tokens],
            dtype=features.dtype,
        )
        features = self.visual_embeddings_table(paddle.concat([features, padding], axis=-1))
        indicators = paddle.arange(
            self.visual_vocab_size - self.vision_tower.num_visual_indicator_tokens,
            self.visual_vocab_size,
            dtype="int64",
        )
        outputs.pooler_output = features
        outputs.visual_indicator_features = self.visual_embeddings_table(indicators)
        return outputs

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        if input_ids is None:
            image_id = paddle.to_tensor([self.config.image_token_id], dtype="int64")
            image_embedding = self.get_input_embeddings()(image_id)
            mask = (inputs_embeds == image_embedding).all(axis=-1)
        else:
            mask = input_ids == self.config.image_token_id
        token_count = int(mask.astype("int64").sum().item())
        feature_count = image_features.shape[0] * image_features.shape[1]
        if token_count != feature_count:
            raise ValueError(
                f"Image features and image tokens do not match, tokens: {token_count}, features: {feature_count}"
            )
        return mask.unsqueeze(-1).expand_as(inputs_embeds)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ):
        del labels
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        _use_high_precision_cublas_for_fp32(inputs_embeds)

        image_features = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values)
            image_features = image_outputs.pooler_output.astype(inputs_embeds.dtype)
            indicator_features = image_outputs.visual_indicator_features.astype(inputs_embeds.dtype)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
            for index, indicator_id in enumerate(self.visual_indicator_token_ids):
                if input_ids is None:
                    token = paddle.to_tensor([indicator_id], dtype="int64")
                    indicator_mask = (inputs_embeds == self.get_input_embeddings()(token)).all(axis=-1)
                else:
                    indicator_mask = input_ids == indicator_id
                inputs_embeds = paddle.where(
                    indicator_mask.unsqueeze(-1),
                    indicator_features[index].reshape([1, 1, -1]),
                    inputs_embeds,
                )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        result = Ovis2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            image_hidden_states=image_features,
        )
        return result if return_dict else result.to_tuple()


class Ovis2ForConditionalGeneration(Ovis2PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = Ovis2Model(config)
        self.lm_head = GeneralLMHead(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_image_features(self, pixel_values, **kwargs):
        return self.model.get_image_features(pixel_values, **kwargs)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        return_dict=None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        if isinstance(logits_to_keep, int):
            indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        elif logits_to_keep is None:
            indices = slice(None)
        else:
            indices = logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.astype("float32").reshape([-1, logits.shape[-1]]),
                labels.reshape([-1]),
                ignore_index=-100,
            )
        result = Ovis2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
        return result if return_dict else result.to_tuple()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        model_inputs["pixel_values"] = (
            pixel_values if is_first_iteration or past_key_values is None or not use_cache else None
        )
        if labels is not None:
            logger.warning("`labels` are ignored during generation.")
        return model_inputs


__all__ = [
    "BaseModelOutputWithVisualIndicatorFeatures",
    "Ovis2ModelOutputWithPast",
    "Ovis2CausalLMOutputWithPast",
    "Ovis2PreTrainedModel",
    "Ovis2VisionModel",
    "Ovis2Model",
    "Ovis2ForConditionalGeneration",
]
