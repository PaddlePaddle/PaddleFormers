# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Paddle Idefics3 model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...generation import GenerationMixin
from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead as GeneralLMHead
from ..activations import ACT2FN
from ..auto.modeling import AutoModel
from ..llama.modeling import LlamaModel
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    ModelOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Idefics3Config, Idefics3VisionConfig


@dataclass
class Idefics3BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[paddle.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    image_hidden_states: Optional[paddle.Tensor] = None


@dataclass
class Idefics3CausalLMOutputWithPast(CausalLMOutputWithPast):
    image_hidden_states: Optional[paddle.Tensor] = None


def _bucketize_right(values: paddle.Tensor, boundaries: paddle.Tensor) -> paddle.Tensor:
    flat_values = values.reshape([-1])
    bucket_ids = paddle.searchsorted(boundaries, flat_values, right=True)
    return bucket_ids.reshape(values.shape)


class Idefics3VisionEmbeddings(nn.Layer):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_positions = self.num_patches_per_side**2
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: paddle.Tensor, patch_attention_mask: paddle.Tensor) -> paddle.Tensor:
        batch_size, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose([0, 2, 1])

        max_patches_h = height // self.patch_size
        max_patches_w = width // self.patch_size

        boundaries = paddle.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )  # default float32, matching HF
        position_ids = paddle.zeros([batch_size, max_patches_h * max_patches_w], dtype="int64")

        # Per-image position-id computation 鈥?matches the installed HF
        # implementation which loops over batch elements individually.
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            h_indices = paddle.arange(nb_patches_h, dtype=pixel_values.dtype)
            w_indices = paddle.arange(nb_patches_w, dtype=pixel_values.dtype)

            fractional_h = h_indices / nb_patches_h * (1 - 1e-6)
            fractional_w = w_indices / nb_patches_w * (1 - 1e-6)

            bucket_h = _bucketize_right(fractional_h, boundaries)
            bucket_w = _bucketize_right(fractional_w, boundaries)

            pos_ids = (bucket_h.unsqueeze(1) * self.num_patches_per_side + bucket_w.unsqueeze(0)).flatten()
            position_ids[batch_idx][p_attn_mask.reshape([-1])] = pos_ids

        return embeddings + self.position_embedding(position_ids)


class Idefics3VisionAttention(nn.Layer):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads, got {self.embed_dim} and {self.num_heads}."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, batch_size, seq_len):
        return tensor.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(self, hidden_states: paddle.Tensor, attention_mask: Optional[paddle.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), batch_size, seq_len)
        key = self._shape(self.k_proj(hidden_states), batch_size, seq_len)
        value = self._shape(self.v_proj(hidden_states), batch_size, seq_len)
        attn_weights = paddle.matmul(query, key, transpose_y=True) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights.astype("float32"), axis=-1).astype(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights, value)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, self.embed_dim])
        return self.out_proj(attn_output), attn_weights


class Idefics3VisionMLP(nn.Layer):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class Idefics3EncoderLayer(nn.Layer):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__()
        self.self_attn = Idefics3VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = Idefics3VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states: paddle.Tensor, attention_mask: Optional[paddle.Tensor] = None):
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.layer_norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + hidden_states


class Idefics3Encoder(nn.Layer):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__()
        self.layers = nn.LayerList([Idefics3EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask=None):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Idefics3SimpleMLP(nn.Layer):
    def __init__(self, config: Idefics3Config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias_attr=False)

    def forward(self, x):
        return self.proj(x)


class Idefics3Connector(nn.Layer):
    def __init__(self, config: Idefics3Config):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        batch_size, seq_len, embed_dim = x.shape
        height = width = int(math.sqrt(seq_len))
        if height * width != seq_len:
            raise ValueError(f"Idefics3 connector expects square image sequence, got sequence length {seq_len}.")
        x = x.reshape([batch_size, height, width, embed_dim])
        x = x.reshape([batch_size, height, width // scale_factor, embed_dim * scale_factor])
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape([batch_size, width // scale_factor, height // scale_factor, embed_dim * scale_factor**2])
        x = x.transpose([0, 2, 1, 3])
        return x.reshape([batch_size, seq_len // scale_factor**2, embed_dim * scale_factor**2])

    def forward(self, image_hidden_states):
        return self.modality_projection(self.pixel_shuffle(image_hidden_states, self.scale_factor))


class Idefics3PreTrainedModel(PretrainedModel):
    config_class = Idefics3Config
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["Idefics3VisionAttention", "Idefics3EncoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"rotary_emb\.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
        "proj",
        # Note: lm_head is GeneralLMHead (weight [vocab, hidden]) 閳?no transpose needed
    ]

    @classmethod
    def _gen_aoa_config(cls, config):
        prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        # HF weight paths use "model.text_model.*" directly (no intermediate "model")
        text_src = "model.text_model"
        text_dst = f"{prefix}text_model"
        text_layers = getattr(config.text_config, "num_hidden_layers", 0)

        statements = [
            # --- lm_head (GeneralLMHead weight [vocab, hidden], same as HF 鈫?no ^T) ---
            "lm_head.weight -> lm_head.weight",
            # --- connector ---
            f"model.connector.modality_projection.proj.weight^T -> {prefix}connector.modality_projection.proj.weight",
            # --- vision_model ---
            f"model.vision_model.embeddings.patch_embedding.weight -> {prefix}vision_model.embeddings.patch_embedding.weight",
            f"model.vision_model.embeddings.patch_embedding.bias -> {prefix}vision_model.embeddings.patch_embedding.bias",
            f"model.vision_model.embeddings.position_embedding.weight -> {prefix}vision_model.embeddings.position_embedding.weight",
            f"model.vision_model.post_layernorm.weight -> {prefix}vision_model.post_layernorm.weight",
            f"model.vision_model.post_layernorm.bias -> {prefix}vision_model.post_layernorm.bias",
            # --- text_model: embedding + final norm (no transpose needed) ---
            f"{text_src}.embed_tokens.weight -> {text_dst}.embed_tokens.weight",
            f"{text_src}.norm.weight -> {text_dst}.norm.weight",
        ]

        # --- vision encoder layers ---
        for layer_id in range(config.vision_config.num_hidden_layers):
            src = f"model.vision_model.encoder.layers.{layer_id}"
            dst = f"{prefix}vision_model.encoder.layers.{layer_id}"
            statements.extend(
                [
                    f"{src}.self_attn.q_proj.weight^T -> {dst}.self_attn.q_proj.weight",
                    f"{src}.self_attn.q_proj.bias -> {dst}.self_attn.q_proj.bias",
                    f"{src}.self_attn.k_proj.weight^T -> {dst}.self_attn.k_proj.weight",
                    f"{src}.self_attn.k_proj.bias -> {dst}.self_attn.k_proj.bias",
                    f"{src}.self_attn.v_proj.weight^T -> {dst}.self_attn.v_proj.weight",
                    f"{src}.self_attn.v_proj.bias -> {dst}.self_attn.v_proj.bias",
                    f"{src}.self_attn.out_proj.weight^T -> {dst}.self_attn.out_proj.weight",
                    f"{src}.self_attn.out_proj.bias -> {dst}.self_attn.out_proj.bias",
                    f"{src}.layer_norm1.weight -> {dst}.layer_norm1.weight",
                    f"{src}.layer_norm1.bias -> {dst}.layer_norm1.bias",
                    f"{src}.mlp.fc1.weight^T -> {dst}.mlp.fc1.weight",
                    f"{src}.mlp.fc1.bias -> {dst}.mlp.fc1.bias",
                    f"{src}.mlp.fc2.weight^T -> {dst}.mlp.fc2.weight",
                    f"{src}.mlp.fc2.bias -> {dst}.mlp.fc2.bias",
                    f"{src}.layer_norm2.weight -> {dst}.layer_norm2.weight",
                    f"{src}.layer_norm2.bias -> {dst}.layer_norm2.bias",
                ]
            )

        # --- text_model transformer layers (Linear weights need ^T for HF鈫扨addle) ---
        for layer_id in range(text_layers):
            src_l = f"{text_src}.layers.{layer_id}"
            dst_l = f"{text_dst}.layers.{layer_id}"
            statements.extend(
                [
                    # Attention: Q/K/V/O projections
                    f"{src_l}.self_attn.q_proj.weight^T -> {dst_l}.self_attn.q_proj.weight",
                    f"{src_l}.self_attn.k_proj.weight^T -> {dst_l}.self_attn.k_proj.weight",
                    f"{src_l}.self_attn.v_proj.weight^T -> {dst_l}.self_attn.v_proj.weight",
                    f"{src_l}.self_attn.o_proj.weight^T -> {dst_l}.self_attn.o_proj.weight",
                    # RMSNorm (1D, no transpose)
                    f"{src_l}.input_layernorm.weight -> {dst_l}.input_layernorm.weight",
                    f"{src_l}.post_attention_layernorm.weight -> {dst_l}.post_attention_layernorm.weight",
                    # MLP: gate/up/down projections
                    f"{src_l}.mlp.gate_proj.weight^T -> {dst_l}.mlp.gate_proj.weight",
                    f"{src_l}.mlp.up_proj.weight^T -> {dst_l}.mlp.up_proj.weight",
                    f"{src_l}.mlp.down_proj.weight^T -> {dst_l}.mlp.down_proj.weight",
                ]
            )

        return {"aoa_statements": statements}

    def _init_weights(self, layer):
        std = getattr(self.config, "initializer_range", None) or getattr(
            self.config.vision_config, "initializer_range", 0.02
        )
        if isinstance(layer, (nn.Linear, nn.Conv2D)):
            layer.weight.set_value(paddle.normal(mean=0.0, std=std, shape=layer.weight.shape))
            if getattr(layer, "bias", None) is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(paddle.normal(mean=0.0, std=std, shape=layer.weight.shape))


class Idefics3VisionTransformer(Idefics3PreTrainedModel):
    config_class = Idefics3VisionConfig

    def __init__(self, config: Idefics3VisionConfig):
        super().__init__(config)
        self.embeddings = Idefics3VisionEmbeddings(config)
        self.encoder = Idefics3Encoder(config)
        self.patch_size = config.patch_size
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(self, pixel_values, patch_attention_mask=None, return_dict=True, **kwargs):
        batch_size = pixel_values.shape[0]
        if patch_attention_mask is None:
            patch_attention_mask = paddle.ones(
                [batch_size, pixel_values.shape[2] // self.patch_size, pixel_values.shape[3] // self.patch_size],
                dtype="bool",
            )
        else:
            patch_attention_mask = patch_attention_mask.astype("bool")
        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        flat_mask = patch_attention_mask.reshape([batch_size, -1]).astype("bool")
        min_value = paddle.finfo(hidden_states.dtype).min
        attention_mask = paddle.where(
            flat_mask[:, None, None, :],
            paddle.zeros([1], dtype=hidden_states.dtype),
            paddle.full([1], min_value, dtype=hidden_states.dtype),
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask)
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        if not return_dict:
            return (last_hidden_state,)
        return BaseModelOutput(last_hidden_state=last_hidden_state)


@register_base_model
class Idefics3Model(Idefics3PreTrainedModel):
    def __init__(self, config: Idefics3Config):
        super().__init__(config)
        self.vision_model = Idefics3VisionTransformer(config.vision_config)
        self.connector = Idefics3Connector(config)
        if getattr(config.text_config, "model_type", None) == "llama":
            self.text_model = LlamaModel(config.text_config)
        else:
            self.text_model = AutoModel.from_config(config.text_config)
        self.image_token_id = config.image_token_id
        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(self, input_ids, inputs_embeds, image_hidden_states):
        if input_ids is None:
            special_image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(paddle.to_tensor(self.config.image_token_id, dtype="int64"))
            ).all(axis=-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = int(special_image_mask.astype("int64").sum().item())
        n_image_features = int(image_hidden_states.shape[0] * image_hidden_states.shape[1])
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"n_image_tokens={n_image_tokens}, "
                f"image_hidden_states.shape={image_hidden_states.shape}, "
                f"numel={image_hidden_states.numel()}"
            )
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(special_image_mask, image_hidden_states.astype(inputs_embeds.dtype))

    def get_image_features(self, pixel_values, pixel_attention_mask=None, **kwargs):
        if len(pixel_values.shape) != 5:
            raise ValueError("`pixel_values` must have shape [batch, num_images, channels, height, width].")
        batch_size, num_images = pixel_values.shape[:2]
        flat_pixel_values = pixel_values.reshape([batch_size * num_images, *pixel_values.shape[2:]])
        nb_values_per_image = math.prod(flat_pixel_values.shape[1:])
        real_images = (flat_pixel_values == 0.0).sum(axis=[1, 2, 3]) != nb_values_per_image
        real_indices = paddle.nonzero(real_images).flatten()
        if real_indices.numel() == 0:
            empty_shape = [0, self.image_seq_len, self.config.text_config.hidden_size]
            return BaseModelOutputWithPooling(pooler_output=paddle.empty(empty_shape, dtype=flat_pixel_values.dtype))

        flat_pixel_values = paddle.gather(flat_pixel_values, real_indices, axis=0)
        if pixel_attention_mask is None:
            pixel_attention_mask = paddle.ones(
                [flat_pixel_values.shape[0], flat_pixel_values.shape[2], flat_pixel_values.shape[3]], dtype="bool"
            )
        else:
            if pixel_attention_mask.shape[:2] != pixel_values.shape[:2]:
                raise ValueError("`pixel_attention_mask` batch and num_images dimensions must match `pixel_values`.")
            pixel_attention_mask = pixel_attention_mask.reshape(
                [batch_size * num_images, *pixel_attention_mask.shape[2:]]
            )
            pixel_attention_mask = paddle.gather(pixel_attention_mask, real_indices, axis=0).astype("bool")

        patch_size = self.config.vision_config.patch_size
        valid_h = (pixel_attention_mask.shape[1] // patch_size) * patch_size
        valid_w = (pixel_attention_mask.shape[2] // patch_size) * patch_size
        pixel_attention_mask = pixel_attention_mask[:, :valid_h, :valid_w]
        patch_attention_mask = (
            pixel_attention_mask.reshape(
                [pixel_attention_mask.shape[0], valid_h // patch_size, patch_size, valid_w // patch_size, patch_size]
            ).sum(axis=[2, 4])
            > 0
        )
        image_outputs = self.vision_model(
            pixel_values=flat_pixel_values, patch_attention_mask=patch_attention_mask, return_dict=True
        )
        image_features = self.connector(image_outputs.last_hidden_state)
        return BaseModelOutputWithPooling(
            last_hidden_state=image_outputs.last_hidden_state,
            pooler_output=image_features,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        cache_position=None,
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time.")
        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask).pooler_output
        if image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(input_ids, inputs_embeds, image_hidden_states)

        text_outputs = self.text_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if not return_dict:
            return (
                text_outputs.last_hidden_state,
                text_outputs.past_key_values,
                text_outputs.hidden_states,
                None,
                image_hidden_states,
            )
        return Idefics3BaseModelOutputWithPast(
            last_hidden_state=text_outputs.last_hidden_state,
            past_key_values=text_outputs.past_key_values,
            hidden_states=text_outputs.hidden_states,
            attentions=None,
            image_hidden_states=image_hidden_states,
        )


class Idefics3ForConditionalGeneration(Idefics3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: Idefics3Config):
        super().__init__(config)
        self.model = Idefics3Model(config)
        self.image_token_id = config.image_token_id
        # Use GeneralLMHead (weight [vocab, hidden]) for compatibility with
        # nn.Embedding weight tie. nn.Linear stores weight as [in, out] in
        # Paddle, which requires transposition when tied to embedding weights.
        self.lm_head = GeneralLMHead(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_image_features(self, pixel_values, pixel_attention_mask=None, **kwargs):
        return self.model.get_image_features(
            pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # logits_to_keep=None means keep all logits (same as 0)
        if logits_to_keep is None or logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        elif isinstance(logits_to_keep, int):
            logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])
        else:
            logits = self.lm_head(hidden_states[:, logits_to_keep, :])

        loss = None
        if labels is not None:
            if isinstance(logits_to_keep, int) and logits_to_keep != 0:
                labels = labels[:, -logits.shape[1] :]
            labels = paddle.where(labels == self.image_token_id, paddle.full_like(labels, -100), labels)
            # NOTE: Labels are already pre-shifted by the data pipeline (SFTDataset._process_sft_sequence).
            # The model MUST NOT shift here to avoid double-shifting.
            # This aligns with qwen3's pattern: self.criterion(logits, labels) without manual shift.
            loss, _ = self.criterion(logits, labels)

        if not return_dict:
            output = (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                outputs.image_hidden_states,
            )
            return ((loss,) + output) if loss is not None else output
        return Idefics3CausalLMOutputWithPast(
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
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        **kwargs,
    ):
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
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if image_hidden_states is not None or cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None
        if cache_position[0] != 0:
            model_inputs["image_hidden_states"] = None
        return model_inputs

    def expand_inputs_for_generation(self, input_ids, expand_size, attention_mask=None, **model_kwargs):
        if expand_size == 1:
            return input_ids, model_kwargs
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.repeat_interleave(expand_size, axis=0)
        for key, value in list(model_kwargs.items()):
            if (
                key == "attention_mask"
                or key == "cache_position"
                or value is None
                or not isinstance(value, paddle.Tensor)
            ):
                continue
            model_kwargs[key] = value.repeat_interleave(expand_size, axis=0)
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, axis=0)
        return input_ids, model_kwargs


__all__ = [
    "Idefics3ForConditionalGeneration",
    "Idefics3Model",
    "Idefics3PreTrainedModel",
    "Idefics3VisionTransformer",
]
