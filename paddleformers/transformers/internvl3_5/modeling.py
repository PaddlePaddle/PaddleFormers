# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 OpenGVLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math
from functools import partial
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.lm_head import LMHead as GeneralLMHead
from ..activations import ACT2FN
from ..conversion_utils import fuse_param_func
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ..model_utils import PretrainedModel
from ..qwen3.modeling import Qwen3ForCausalLMDeprecated
from ..qwen3_moe.configuration import Qwen3MoeConfig
from .configuration import InternVisionConfig, InternVLChatConfig

__all__ = ["InternVisionModel", "InternVLChatModel"]


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
        pixel_values = pixel_values.astype(target_dtype)
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
        "mlp1.1",
        "mlp1.3",
    ]

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: InternVLChatConfig, is_fuse=True):
        if not is_fuse:
            return {}
        mappings = {}
        llm_config = config.llm_config
        qkv_action = partial(
            fuse_param_func(),
            is_qkv=True,
            num_heads=llm_config.num_attention_heads,
            num_key_value_heads=llm_config.num_key_value_heads,
        )
        ffn_action = fuse_param_func()
        for layer_id in range(llm_config.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_id}"
            mappings[
                (
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.qkv_proj.weight",
                )
            ] = qkv_action
            if isinstance(llm_config, Qwen3MoeConfig) and cls._is_moe_layer(llm_config, layer_id):
                for expert_id in range(llm_config.num_experts):
                    mappings[
                        (
                            f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight",
                            f"{prefix}.mlp.experts.{expert_id}.up_proj.weight",
                            f"{prefix}.mlp.experts.{expert_id}.up_gate_proj.weight",
                        )
                    ] = ffn_action
            else:
                mappings[
                    (
                        f"{prefix}.mlp.gate_proj.weight",
                        f"{prefix}.mlp.up_proj.weight",
                        f"{prefix}.mlp.up_gate_proj.weight",
                    )
                ] = ffn_action
        return mappings

    @classmethod
    def _is_moe_layer(cls, llm_config, layer_id):
        return (
            isinstance(llm_config, Qwen3MoeConfig)
            and layer_id not in llm_config.mlp_only_layers
            and llm_config.num_experts > 0
            and (layer_id + 1) % llm_config.decoder_sparse_step == 0
        )

    @classmethod
    def _prefix_language_model_aoa_statement(cls, statement):
        return statement.replace("model.", "language_model.model.").replace("lm_head.", "language_model.lm_head.")

    @classmethod
    def _get_qwen3_moe_aoa_statements(cls, llm_config):
        from ..qwen3_moe.modeling import Qwen3MoeForCausalLMDeprecated

        aoa_config = Qwen3MoeForCausalLMDeprecated._gen_aoa_config(llm_config)
        return [
            cls._prefix_language_model_aoa_statement(statement) for statement in aoa_config.get("aoa_statements", [])
        ]

    @classmethod
    def _gen_aoa_config(cls, config: InternVLChatConfig):
        llm_config = config.llm_config
        statements = [
            "vision_model.embeddings.class_embedding -> vision_model.embeddings.class_embedding",
            "vision_model.embeddings.position_embedding -> vision_model.embeddings.position_embedding",
            "vision_model.embeddings.patch_embedding.weight -> vision_model.embeddings.patch_embedding.weight",
            "vision_model.embeddings.patch_embedding.bias -> vision_model.embeddings.patch_embedding.bias",
            "vision_model.encoder.layers.$LAYER_ID.attn.qkv.weight^T -> vision_model.encoder.layers.$LAYER_ID.attn.qkv.weight",
            "vision_model.encoder.layers.$LAYER_ID.attn.qkv.bias -> vision_model.encoder.layers.$LAYER_ID.attn.qkv.bias",
            "vision_model.encoder.layers.$LAYER_ID.attn.proj.weight^T -> vision_model.encoder.layers.$LAYER_ID.attn.proj.weight",
            "vision_model.encoder.layers.$LAYER_ID.attn.proj.bias -> vision_model.encoder.layers.$LAYER_ID.attn.proj.bias",
            "vision_model.encoder.layers.$LAYER_ID.mlp.fc1.weight^T -> vision_model.encoder.layers.$LAYER_ID.mlp.fc1.weight",
            "vision_model.encoder.layers.$LAYER_ID.mlp.fc1.bias -> vision_model.encoder.layers.$LAYER_ID.mlp.fc1.bias",
            "vision_model.encoder.layers.$LAYER_ID.mlp.fc2.weight^T -> vision_model.encoder.layers.$LAYER_ID.mlp.fc2.weight",
            "vision_model.encoder.layers.$LAYER_ID.mlp.fc2.bias -> vision_model.encoder.layers.$LAYER_ID.mlp.fc2.bias",
            "vision_model.encoder.layers.$LAYER_ID.norm1.weight -> vision_model.encoder.layers.$LAYER_ID.norm1.weight",
            "vision_model.encoder.layers.$LAYER_ID.norm1.bias -> vision_model.encoder.layers.$LAYER_ID.norm1.bias",
            "vision_model.encoder.layers.$LAYER_ID.norm2.weight -> vision_model.encoder.layers.$LAYER_ID.norm2.weight",
            "vision_model.encoder.layers.$LAYER_ID.norm2.bias -> vision_model.encoder.layers.$LAYER_ID.norm2.bias",
            "vision_model.encoder.layers.$LAYER_ID.ls1 -> vision_model.encoder.layers.$LAYER_ID.ls1",
            "vision_model.encoder.layers.$LAYER_ID.ls2 -> vision_model.encoder.layers.$LAYER_ID.ls2",
            "mlp1.0.weight -> mlp1.0.weight",
            "mlp1.0.bias -> mlp1.0.bias",
            "mlp1.1.weight^T -> mlp1.1.weight",
            "mlp1.1.bias -> mlp1.1.bias",
            "mlp1.3.weight^T -> mlp1.3.weight",
            "mlp1.3.bias -> mlp1.3.bias",
        ]
        if isinstance(llm_config, Qwen3MoeConfig):
            statements.extend(cls._get_qwen3_moe_aoa_statements(llm_config))
            return {"aoa_statements": statements}

        statements.extend(
            [
                "language_model.model.embed_tokens.weight -> language_model.model.embed_tokens.weight",
                "language_model.model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> language_model.model.layers.$LAYER_ID.self_attn.o_proj.weight",
                "language_model.model.layers.$LAYER_ID.self_attn.q_norm.weight -> language_model.model.layers.$LAYER_ID.self_attn.q_norm.weight",
                "language_model.model.layers.$LAYER_ID.self_attn.k_norm.weight -> language_model.model.layers.$LAYER_ID.self_attn.k_norm.weight",
                "language_model.model.layers.$LAYER_ID.input_layernorm.weight -> language_model.model.layers.$LAYER_ID.input_layernorm.weight",
                "language_model.model.layers.$LAYER_ID.post_attention_layernorm.weight -> language_model.model.layers.$LAYER_ID.post_attention_layernorm.weight",
                "language_model.model.layers.$LAYER_ID.mlp.down_proj.weight^T -> language_model.model.layers.$LAYER_ID.mlp.down_proj.weight",
                "language_model.model.norm.weight -> language_model.model.norm.weight",
                f"language_model.model.layers.$LAYER_ID.self_attn.q_proj.weight^T, language_model.model.layers.$LAYER_ID.self_attn.k_proj.weight^T, language_model.model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> language_model.model.layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={llm_config.num_attention_heads}, num_key_value_groups={llm_config.num_key_value_heads}",
                "language_model.model.layers.$LAYER_ID.mlp.gate_proj.weight^T, language_model.model.layers.$LAYER_ID.mlp.up_proj.weight^T -> language_model.model.layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            ]
        )
        if llm_config.attention_bias:
            statements.append(
                f"language_model.model.layers.$LAYER_ID.self_attn.q_proj.bias, language_model.model.layers.$LAYER_ID.self_attn.k_proj.bias, language_model.model.layers.$LAYER_ID.self_attn.v_proj.bias -> language_model.model.layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={llm_config.num_attention_heads}, num_key_value_groups={llm_config.num_key_value_heads}, axis=0"
            )
        if llm_config.tie_word_embeddings:
            statements.append("language_model.model.embed_tokens.weight -> language_model.lm_head.weight")
        else:
            statements.append("language_model.lm_head.weight -> language_model.lm_head.weight")
        return {"aoa_statements": statements}


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
        self.language_model = language_model if language_model is not None else self._build_language_model(config)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        shuffle_scale = int(1 / self.downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * shuffle_scale),
            nn.Linear(vit_hidden_size * shuffle_scale, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    @staticmethod
    def _build_language_model(config):
        if isinstance(config.llm_config, Qwen3MoeConfig):
            from ..qwen3_moe.modeling import Qwen3MoeForCausalLMDeprecated

            return Qwen3MoeForCausalLMDeprecated(config.llm_config)
        return Qwen3ForCausalLMDeprecated(config.llm_config)

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

        language_model_kwargs = {
            "input_ids": None,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": input_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "return_dict": return_dict,
        }
        if not isinstance(self.config.llm_config, Qwen3MoeConfig):
            language_model_kwargs["output_hidden_states"] = output_hidden_states
        outputs = self.language_model(**language_model_kwargs)

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

        old_num_tokens = old_output_embeddings.weight.shape[0]
        self.config.vocab_size = new_num_tokens
        self.config.llm_config.vocab_size = new_num_tokens
        self.language_model.config.vocab_size = new_num_tokens
        new_output_embeddings = GeneralLMHead(self.language_model.config)
        if new_output_embeddings.weight.dtype != old_output_embeddings.weight.dtype:
            new_output_embeddings.to(dtype=old_output_embeddings.weight.dtype)
        n = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            new_output_embeddings.weight[:n] = old_output_embeddings.weight[:n]
        self.set_output_embeddings(new_output_embeddings)
        return new_input_embeddings
