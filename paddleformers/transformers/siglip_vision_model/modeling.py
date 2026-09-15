# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Optional

import paddle
from paddle import nn

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ..activations import ACT2FN
from ..model_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ..model_utils import PretrainedModel, register_base_model
from .configuration import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
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
        self.register_buffer("position_ids", paddle.arange(self.num_patches).expand([1, -1]), persistable=False)

    def interpolate_pos_encoding(self, embeddings, height, width):
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]
        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0).reshape(
            [1, sqrt_num_positions, sqrt_num_positions, dim]
        )
        patch_pos_embed = patch_pos_embed.transpose([0, 3, 1, 2])
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=[new_height, new_width], mode="bicubic", align_corners=False
        )
        return patch_pos_embed.transpose([0, 2, 3, 1]).reshape([1, -1, dim])

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values.astype(self.patch_embedding.weight.dtype))
        embeddings = patch_embeds.flatten(2).transpose([0, 2, 1])
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        self.num_key_value_groups = 1
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _get_bidirectional_indices(self, batch_size, seq_len, dtype):
        """Generate flashmask startend_row_indices for full bidirectional attention.

        Each position attends to [0, seq_len) — the entire sequence.
        Follows Qwen3-VL's pattern for non-causal vision attention.
        """
        indices = paddle.to_tensor([0, seq_len, 0, seq_len], dtype="int32")
        return indices.reshape([1, 1, 1, 4]).expand([batch_size, self.num_heads, seq_len, 4])

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        target_shape = [bsz, seq_len, self.num_heads, self.head_dim]
        query = self.q_proj(hidden_states).reshape(target_shape).transpose([0, 2, 1, 3])
        key = self.k_proj(hidden_states).reshape(target_shape).transpose([0, 2, 1, 3])
        value = self.v_proj(hidden_states).reshape(target_shape).transpose([0, 2, 1, 3])

        attn_impl = self.config._attn_implementation
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        if attn_impl != "sdpa":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        # bidirectional 全注意力无需 mask。仅 flashmask 需要稀疏 mask 索引；
        # eager/sdpa 传入该索引会经 _gen_from_sparse_attn_mask_indices 生成全 -1e6 的
        # 加性 mask，softmax 虽平移不变但在 fp32 下引入 ~1e-3 量级精度损失。
        attn_kwargs = {}
        if attn_impl == "flashmask":
            attn_kwargs["attn_mask_startend_row_indices"] = self._get_bidirectional_indices(bsz, seq_len, query.dtype)

        attn_output, attn_weights = attention_interface(
            self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scale,
            **attn_kwargs,
        )

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class SiglipMLP(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class SiglipMultiheadAttentionPoolingHead(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.probe = self.create_parameter(shape=[1, 1, config.hidden_size])
        self.attention = nn.MultiHeadAttention(config.hidden_size, config.num_attention_heads)
        self.layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.expand([batch_size, -1, -1])
        hidden_state = self.attention(probe, hidden_state, hidden_state)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SiglipEncoderLayer(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        residual = hidden_states
        normed_hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(normed_hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        normed_hidden_states = self.layer_norm2(hidden_states)
        hidden_states = residual + self.mlp(normed_hidden_states)
        return (hidden_states, attn_weights) if output_attentions else (hidden_states, None)


class SiglipEncoder(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.LayerList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False, output_attentions=False):
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn = layer(hidden_states, attention_mask, output_attentions=output_attentions)
            if output_attentions:
                all_attentions += (attn,)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class SiglipVisionPretrainedModel(PretrainedModel):
    config_class = SiglipVisionConfig
    base_model_prefix = "vision_model"
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    @classmethod
    def _gen_aoa_config(cls, config: SiglipVisionConfig):
        aoa_statements = [
            "embeddings.patch_embedding.weight -> embeddings.patch_embedding.weight",
            "embeddings.patch_embedding.bias -> embeddings.patch_embedding.bias",
            "embeddings.position_embedding.weight -> embeddings.position_embedding.weight",
            "post_layernorm.weight -> post_layernorm.weight",
            "post_layernorm.bias -> post_layernorm.bias",
        ]
        for layer_id in range(config.num_hidden_layers):
            src = f"encoder.layers.{layer_id}"
            dst = f"encoder.layers.{layer_id}"
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
        if getattr(config, "vision_use_head", True):
            cls._append_head_aoa_statements(aoa_statements, "head", "head")
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: SiglipVisionConfig):
        aoa_statements = [
            "embeddings.patch_embedding.weight -> embeddings.patch_embedding.weight",
            "embeddings.patch_embedding.bias -> embeddings.patch_embedding.bias",
            "embeddings.position_embedding.weight -> embeddings.position_embedding.weight",
            "post_layernorm.weight -> post_layernorm.weight",
            "post_layernorm.bias -> post_layernorm.bias",
        ]
        for layer_id in range(config.num_hidden_layers):
            src = f"encoder.layers.{layer_id}"
            dst = f"encoder.layers.{layer_id}"
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
        if getattr(config, "vision_use_head", True):
            cls._append_head_inv_aoa_statements(aoa_statements, "head", "head")
        return {"aoa_statements": aoa_statements}

    @staticmethod
    def _append_head_aoa_statements(aoa_statements, src, dst):
        q_weight = f"{src}.attention._q_weight"
        k_weight = f"{src}.attention._k_weight"
        v_weight = f"{src}.attention._v_weight"
        q_bias = f"{src}.attention._q_bias"
        k_bias = f"{src}.attention._k_bias"
        v_bias = f"{src}.attention._v_bias"
        aoa_statements += [
            f"{src}.probe -> {dst}.probe",
            f"{src}.layernorm.weight -> {dst}.layernorm.weight",
            f"{src}.layernorm.bias -> {dst}.layernorm.bias",
            f"{src}.mlp.fc1.weight^T -> {dst}.mlp.fc1.weight",
            f"{src}.mlp.fc1.bias -> {dst}.mlp.fc1.bias",
            f"{src}.mlp.fc2.weight^T -> {dst}.mlp.fc2.weight",
            f"{src}.mlp.fc2.bias -> {dst}.mlp.fc2.bias",
            f"{src}.attention.in_proj_weight -> {q_weight}, {k_weight}, {v_weight}, axis=0",
            f"{q_weight}^T -> {dst}.attention.q_proj.weight",
            f"{k_weight}^T -> {dst}.attention.k_proj.weight",
            f"{v_weight}^T -> {dst}.attention.v_proj.weight",
            f"{src}.attention.in_proj_bias -> {q_bias}, {k_bias}, {v_bias}, axis=0",
            f"{q_bias} -> {dst}.attention.q_proj.bias",
            f"{k_bias} -> {dst}.attention.k_proj.bias",
            f"{v_bias} -> {dst}.attention.v_proj.bias",
            f"{src}.attention.out_proj.weight^T -> {dst}.attention.out_proj.weight",
            f"{src}.attention.out_proj.bias -> {dst}.attention.out_proj.bias",
        ]

    @staticmethod
    def _append_head_inv_aoa_statements(aoa_statements, src, dst):
        aoa_statements += [
            f"{src}.probe -> {dst}.probe",
            f"{src}.layernorm.weight -> {dst}.layernorm.weight",
            f"{src}.layernorm.bias -> {dst}.layernorm.bias",
            f"{src}.mlp.fc1.weight^T -> {dst}.mlp.fc1.weight",
            f"{src}.mlp.fc1.bias -> {dst}.mlp.fc1.bias",
            f"{src}.mlp.fc2.weight^T -> {dst}.mlp.fc2.weight",
            f"{src}.mlp.fc2.bias -> {dst}.mlp.fc2.bias",
            f"{src}.attention.q_proj.weight^T, {src}.attention.k_proj.weight^T, {src}.attention.v_proj.weight^T -> {dst}.attention.in_proj_weight, axis=0",
            f"{src}.attention.q_proj.bias, {src}.attention.k_proj.bias, {src}.attention.v_proj.bias -> {dst}.attention.in_proj_bias, axis=0",
            f"{src}.attention.out_proj.weight^T -> {dst}.attention.out_proj.weight",
            f"{src}.attention.out_proj.bias -> {dst}.attention.out_proj.bias",
        ]


@register_base_model
class SiglipVisionModel(SiglipVisionPretrainedModel):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=kwargs.get("attention_mask", None),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooler_output = self.head(last_hidden_state) if self.use_head else None
        hidden_states_tuple = encoder_outputs.hidden_states
        if not return_dict:
            return (last_hidden_state, pooler_output, hidden_states_tuple, encoder_outputs.attentions)
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states_tuple,
            attentions=encoder_outputs.attentions,
        )


__all__ = [
    "SiglipMultiheadAttentionPoolingHead",
    "SiglipVisionConfig",
    "SiglipVisionModel",
    "SiglipVisionPretrainedModel",
]
