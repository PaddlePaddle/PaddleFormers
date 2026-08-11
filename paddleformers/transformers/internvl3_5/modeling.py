# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 OpenGVLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn

from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ..model_utils import PretrainedModel
from .configuration import InternVisionConfig, InternVLChatConfig

__all__ = ["InternVisionModel", "InternVLChatModel"]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, seq_len, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, seq_len, head_dim])


class Qwen3RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter([hidden_size], default_initializer=nn.initializer.Constant(1.0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight.astype(input_dtype) * hidden_states.astype(input_dtype)


class InternVLQwen3Attention(nn.Layer):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        inv_freq = 1.0 / (self.rope_theta ** (paddle.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def _get_cos_sin(self, position_ids, dtype):
        freqs = paddle.einsum("bi,j->bij", position_ids.astype("float32"), self.inv_freq)
        emb = paddle.concat([freqs, freqs], axis=-1)
        return paddle.cos(emb).astype(dtype), paddle.sin(emb).astype(dtype)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = paddle.arange(seq_len, dtype="int64").unsqueeze(0).expand([batch_size, seq_len])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        key_states = key_states.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        value_states = value_states.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = self._get_cos_sin(position_ids, query_states.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = paddle.matmul(query_states * self.scaling, key_states.transpose([0, 1, 3, 2]))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights.astype("float32"), axis=-1).astype(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape(
            [batch_size, seq_len, self.num_heads * self.head_dim]
        )
        return self.o_proj(attn_output), None, None


class InternVLQwen3MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class InternVLQwen3DecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = InternVLQwen3Attention(config, layer_idx)
        self.mlp = InternVLQwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class InternVLQwen3Model(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.LayerList(
            [InternVLQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _prepare_attention_mask(self, attention_mask, batch_size, seq_len, dtype):
        causal = paddle.triu(paddle.ones([seq_len, seq_len], dtype="bool"), diagonal=1)
        causal = paddle.where(
            causal,
            paddle.full([seq_len, seq_len], paddle.finfo(dtype).min, dtype=dtype),
            paddle.zeros([seq_len, seq_len], dtype=dtype),
        )
        causal = causal.reshape([1, 1, seq_len, seq_len]).expand([batch_size, 1, seq_len, seq_len])
        if attention_mask is not None:
            expanded = attention_mask[:, None, None, :].astype(dtype)
            padding = paddle.where(
                expanded > 0,
                paddle.zeros_like(expanded),
                paddle.full_like(expanded, paddle.finfo(dtype).min),
            )
            causal = causal + padding
        return causal

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = paddle.arange(seq_len, dtype="int64").unsqueeze(0).expand([batch_size, seq_len])
        causal_mask = self._prepare_attention_mask(attention_mask, batch_size, seq_len, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=None, hidden_states=all_hidden_states
        )


class InternVLQwen3ForCausalLM(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = InternVLQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            flat_logits = shift_logits.reshape([-1, self.config.vocab_size])
            flat_labels = shift_labels.reshape([-1])
            valid_mask = flat_labels != -100
            safe_labels = paddle.where(valid_mask, flat_labels, paddle.zeros_like(flat_labels))
            token_loss = F.cross_entropy(flat_logits, safe_labels, reduction="none")
            token_loss = token_loss * valid_mask.astype(token_loss.dtype)
            loss = token_loss.sum() / valid_mask.astype(token_loss.dtype).sum()
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
        )


class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        return x / keep_prob * paddle.floor(random_tensor)


class InternRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter([hidden_size], default_initializer=nn.initializer.Constant(1.0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = paddle.mean(hidden_states.pow(2), axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight.astype(input_dtype) * hidden_states.astype(input_dtype)


NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = self.create_parameter([1, 1, self.embed_dim])
        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = self.create_parameter([1, self.num_positions, self.embed_dim])

    def _get_pos_embed(self, pos_embed, height, width):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.astype("float32").reshape(
            [1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1]
        )
        pos_embed = pos_embed.transpose([0, 3, 1, 2])
        pos_embed = F.interpolate(pos_embed, size=[height, width], mode="bicubic", align_corners=False)
        return pos_embed.reshape([1, -1, height * width]).transpose([0, 2, 1]).astype(target_dtype)

    def forward(self, pixel_values):
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose([0, 2, 1])
        class_embeds = self.class_embedding.expand([batch_size, 1, self.embed_dim]).astype(target_dtype)
        embeddings = paddle.concat([class_embeds, patch_embeds], axis=1)
        position_embedding = paddle.concat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            axis=1,
        )
        return embeddings + position_embedding.astype(target_dtype)


class InternAttention(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias_attr=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)
        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, channels = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape([batch_size, seq_len, 3, self.num_heads, channels // self.num_heads])
        q, k, v = paddle.unbind(qkv.transpose([2, 0, 3, 1, 4]), axis=0)

        if self.qk_normalization:
            q = self.q_norm(q.transpose([0, 2, 1, 3]).flatten(-2, -1))
            q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
            k = self.k_norm(k.transpose([0, 2, 1, 3]).flatten(-2, -1))
            k = k.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        attn = paddle.matmul(q * self.scale, k.transpose([0, 1, 3, 2]))
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, channels])
        return self.proj_drop(self.proj(x))


class InternMLP(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.fc2(self.act(self.fc1(hidden_states)))


class InternVisionEncoderLayer(nn.Layer):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.ls1 = self.create_parameter(
            [self.embed_dim], default_initializer=nn.initializer.Constant(config.initializer_factor)
        )
        self.ls2 = self.create_parameter(
            [self.embed_dim], default_initializer=nn.initializer.Constant(config.initializer_factor)
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states).astype(hidden_states.dtype)) * self.ls1
        )
        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states).astype(hidden_states.dtype)) * self.ls2
        )
        return hidden_states


class InternVisionEncoder(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        dpr = paddle.linspace(0, config.drop_path_rate, config.num_hidden_layers).tolist()
        self.layers = nn.LayerList(
            [InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds, output_hidden_states=False, return_dict=True):
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        for layer in self.layers:
            if output_hidden_states:
                encoder_states += (hidden_states,)
            hidden_states = layer(hidden_states)
        if output_hidden_states:
            encoder_states += (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class InternVisionPretrainedModel(PretrainedModel):
    config_class = InternVisionConfig
    base_model_prefix = "vision_model"
    transpose_weight_keys = ["qkv", "proj", "fc1", "fc2"]


class InternVisionModel(InternVisionPretrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[paddle.Tensor] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = True if return_dict is None else return_dict
        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")
        hidden_states = pixel_embeds if pixel_embeds is not None else self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=None,
        )


class InternVLChatPretrainedModel(PretrainedModel):
    config_class = InternVLChatConfig
    base_model_prefix = "language_model"
    transpose_weight_keys = [
        "qkv",
        "proj",
        "fc1",
        "fc2",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
        "mlp1.1",
        "mlp1.3",
    ]


class InternVLChatModel(InternVLChatPretrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.img_context_token_id = config.img_context_token_id

        self.vision_model = vision_model if vision_model is not None else InternVisionModel(config.vision_config)
        self.language_model = (
            language_model if language_model is not None else InternVLQwen3ForCausalLM(config.llm_config)
        )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        shuffle_scale = int(1 / self.downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * shuffle_scale),
            nn.Linear(vit_hidden_size * shuffle_scale, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.shape
        x = x.reshape([n, w, int(h * scale_factor), int(c / scale_factor)])
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape([n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))])
        if self.ps_version != "v1":
            x = x.transpose([0, 2, 1, 3])
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(math.sqrt(vit_embeds.shape[1]))
        vit_embeds = vit_embeds.reshape([vit_embeds.shape[0], h, w, -1])
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape([vit_embeds.shape[0], -1, vit_embeds.shape[-1]])
        return self.mlp1(vit_embeds)

    def _merge_visual_embeds(self, input_ids, input_embeds, vit_embeds):
        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape([batch_size * seq_len, hidden_size])
        flat_input_ids = input_ids.reshape([batch_size * seq_len])
        selected = paddle.nonzero(flat_input_ids == self.img_context_token_id).flatten()
        vit_embeds = vit_embeds.reshape([-1, hidden_size]).astype(flat_embeds.dtype)
        if selected.shape[0] == 0:
            raise ValueError("No <IMG_CONTEXT> token found in input_ids.")
        if selected.shape[0] != vit_embeds.shape[0]:
            raise ValueError(
                f"The number of <IMG_CONTEXT> tokens ({selected.shape[0]}) does not match "
                f"visual tokens ({vit_embeds.shape[0]})."
            )
        flat_embeds = paddle.scatter(flat_embeds, selected, vit_embeds, overwrite=True)
        return flat_embeds.reshape([batch_size, seq_len, hidden_size])

    def _expand_visual_features_for_generation(self, visual_features, input_ids, expand_size):
        if visual_features is None:
            return None

        context_token_counts = paddle.sum((input_ids == self.img_context_token_id).astype("int64"), axis=-1)
        feature_counts = (context_token_counts // self.num_image_token).numpy().tolist()
        chunks = []
        offset = 0
        for feature_count in feature_counts:
            feature_count = int(feature_count)
            sample_features = visual_features[offset : offset + feature_count]
            offset += feature_count
            if feature_count == 0:
                continue
            chunks.extend([sample_features] * expand_size)
        return paddle.concat(chunks, axis=0) if chunks else visual_features

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_flags: Optional[paddle.Tensor] = None,
        visual_features: Optional[paddle.Tensor] = None,
        past_key_values=None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = True if return_dict is None else return_dict
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None or visual_features is not None:
            vit_embeds = visual_features if visual_features is not None else self.extract_feature(pixel_values)
            if image_flags is not None:
                image_flags = image_flags.squeeze(-1).astype("bool")
                vit_embeds = vit_embeds[image_flags]
            input_embeds = self._merge_visual_embeds(input_ids, input_embeds, vit_embeds)

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        visual_features=None,
        use_cache=True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_features=visual_features,
            use_cache=use_cache,
            **kwargs,
        )
        if past_key_values is not None and use_cache:
            model_inputs["visual_features"] = None
        return model_inputs

    def expand_inputs_for_generation(self, input_ids, expand_size, attention_mask=None, **model_kwargs):
        visual_features = model_kwargs.pop("visual_features", None)
        expanded_input_ids, model_kwargs = super().expand_inputs_for_generation(
            input_ids,
            expand_size,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        model_kwargs["visual_features"] = self._expand_visual_features_for_generation(
            visual_features,
            input_ids,
            expand_size,
        )
        return expanded_input_ids, model_kwargs

    def generate(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        image_flags: Optional[paddle.Tensor] = None,
        visual_features: Optional[paddle.Tensor] = None,
        **generate_kwargs,
    ):
        if visual_features is None and pixel_values is not None:
            visual_features = self.extract_feature(pixel_values)
            if image_flags is not None:
                image_flags = image_flags.squeeze(-1).astype("bool")
                visual_features = visual_features[image_flags]
        if "decode_strategy" not in generate_kwargs:
            if generate_kwargs.get("num_beams", 1) > 1:
                generate_kwargs["decode_strategy"] = "beam_search"
            elif generate_kwargs.get("do_sample", False):
                generate_kwargs["decode_strategy"] = "sampling"
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            **generate_kwargs,
        )

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        old_output_embeddings = self.get_output_embeddings()
        new_input_embeddings = super().resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return new_input_embeddings

        old_num_tokens = old_output_embeddings.weight.shape[1]
        hidden_size = old_output_embeddings.weight.shape[0]
        new_output_embeddings = nn.Linear(hidden_size, new_num_tokens, bias_attr=False)
        if new_output_embeddings.weight.dtype != old_output_embeddings.weight.dtype:
            new_output_embeddings.to(dtype=old_output_embeddings.weight.dtype)
        n = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            new_output_embeddings.weight[:, :n] = old_output_embeddings.weight[:, :n]
        self.set_output_embeddings(new_output_embeddings)
        self.config.vocab_size = new_num_tokens
        self.config.llm_config.vocab_size = new_num_tokens
        self.language_model.config.vocab_size = new_num_tokens
        return new_input_embeddings
