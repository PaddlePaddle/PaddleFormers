# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Glm4vMoeTextMLP
from ...nn.mlp import MLP as Glm4vMoeVisionMLP
from ...nn.moe_deepep.moe_factory import QuickAccessMoEFactory
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..glm4_moe.modeling import Glm4MoeFlexMoE as Glm4vMoeFlexTextMoE
from ..glm4_moe.modeling import Glm4MoeMoE as Glm4vMoeTextMoE
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import ModelOutput
from ..model_utils import PretrainedModel
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import Glm4vMoeConfig, Glm4vMoeTextConfig, Glm4vMoeVisionConfig


@dataclass
class MoeModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """

    last_hidden_state: Optional[paddle.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[paddle.FloatTensor, ...]] = None
    attentions: Optional[tuple[paddle.FloatTensor, ...]] = None
    router_logits: Optional[tuple[paddle.FloatTensor]] = None


@dataclass
class Glm4vMoeModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`paddle.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[paddle.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[paddle.FloatTensor]] = None
    attentions: Optional[tuple[paddle.FloatTensor]] = None
    rope_deltas: Optional[paddle.LongTensor] = None


class Glm4vMoeTextRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Glm4vMoeTextConfig, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Glm4vMoeTextConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple[paddle.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`paddle.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # NOTE: Paddle's Automatic Mixed Precision (AMP) has a default op whitelist that may automatically cast
        # certain operations (like matmul) to FP16/BF16 for performance optimization. However, in scenarios where
        # numerical stability is critical (e.g., RoPE init/compute), this conversion can lead to precision loss.
        # Disabling auto_cast here ensures the matmul operation runs in the original precision (FP32) as intended.
        with paddle.amp.auto_cast(False):
            # In contrast to other models, GLM4V_MOE different position ids for the grids
            # So we expand the inv_freq to shape (3, ...)
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

            # freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = paddle.matmul(inv_freq_expanded, position_ids_expanded).transpose([0, 1, 3, 2])
            emb = paddle.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = paddle.cat(
        [m[i % 3] for i, m in enumerate(paddle.compat.split(cos, mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = paddle.cat(
        [m[i % 3] for i, m in enumerate(paddle.compat.split(sin, mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = paddle.cat([q_embed, q_pass], dim=-1)
    k_embed = paddle.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class Glm4vMoeTextAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.sequence_parallel = config.sequence_parallel

        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert (
                self.num_key_value_heads % config.tensor_model_parallel_size == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        kv_hidden_size = self.config.num_key_value_heads * self.head_dim
        q_hidden_size = self.config.num_attention_heads * self.head_dim

        self.qkv_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size + 2 * kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

        self.rope_parameters = config.rope_parameters

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[paddle.LongTensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[tuple[paddle.Tensor]]]:
        mix_layer = self.qkv_proj(hidden_states)
        if self.config.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
            q_len = max_sequence_length
            target_shape = [
                bsz,
                q_len,
                self.num_key_value_heads,
                (self.num_key_value_groups + 2) * self.head_dim,
            ]
        else:
            target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
        mix_layer = paddle.reshape_(mix_layer, target_shape)
        query_states, key_states, value_states = paddle.split(
            mix_layer,
            num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
            axis=-1,
        )
        query_states = query_states.reshape([0, 0, -1, self.head_dim])

        # b l h d -> b h l d
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(  # diff with Llama
            query_states, key_states, cos, sin, self.rope_parameters["mrope_section"]
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["eager"]
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Glm4vMoeTextDecoderLayer(nn.Layer):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Glm4vMoeTextAttention(config=config, layer_idx=layer_idx)

        try:
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        except:
            moe_group = None
        expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = (
                Glm4vMoeTextMoE(config)
                if expert_parallel_degree <= 1
                else (
                    QuickAccessMoEFactory.create_from_model_name(
                        pretrained_config=config,
                        expert_class=Glm4vMoeTextMLP,
                        gate_activation="sigmoid",
                        expert_activation="silu",
                        train_topk_method="noaux_tc",
                        inference_topk_method="noaux_tc",
                        drop_tokens=False,
                        transpose_gate_weight=True,
                    )
                    if config.use_unified_moe
                    else Glm4vMoeFlexTextMoE(config)
                )
            )
        else:
            self.mlp = Glm4vMoeTextMLP(config, fuse_up_gate=True)

        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.LongTensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Glm4vMoePreTrainedModel(PretrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["Glm4vMoeTextDecoderLayer", "Glm4vMoeVisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]
    config_class = Glm4vMoeConfig

    input_modalities = ("text", "image", "video")

    @classmethod
    def _gen_aoa_config(cls, config: Glm4vMoeConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                # do cast
                f"{llm_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias -> {llm_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias, dtype='float32'",
                f"{llm_prefix}layers.$LAYER_ID.mlp.gate.weight -> {llm_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                # do transpose
                f"{llm_prefix}embed_tokens.weight -> {llm_prefix}embed_tokens.weight",
                f"{llm_prefix}norm.weight -> {llm_prefix}norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                # moe
                f"{llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.weight"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.bias -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.bias"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.weight"
                for x in ("up", "gate", "down")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.bias"
                for x in ("up", "gate", "down")
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.proj.bias -> {visual_prefix}patch_embed.proj.bias",
            f"{visual_prefix}blocks.$LAYER_ID.norm1.weight -> {visual_prefix}blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm2.weight -> {visual_prefix}blocks.$LAYER_ID.norm2.weight",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.{p}_proj.weight^T -> {visual_prefix}merger.{p}_proj.weight"
            for p in ("gate", "up", "down")
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.proj.weight^T -> {visual_prefix}merger.proj.weight",
            f"{visual_prefix}merger.post_projection_norm.weight^T -> {visual_prefix}merger.post_projection_norm.weight",
            f"{visual_prefix}merger.post_projection_norm.bias -> {visual_prefix}merger.post_projection_norm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}embeddings.position_embedding.weight -> {visual_prefix}embeddings.position_embedding.weight",
            f"{visual_prefix}post_conv_layernorm.weight^T -> {visual_prefix}post_conv_layernorm.weight",
            f"{visual_prefix}post_layernorm.weight^T -> {visual_prefix}post_layernorm.weight",
            f"{visual_prefix}downsample.weight -> {visual_prefix}downsample.weight",
            f"{visual_prefix}downsample.bias -> {visual_prefix}downsample.bias",
        ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.self_attn.q_proj.weight^T, {llm_prefix}layers.$LAYER_ID.self_attn.k_proj.weight^T, {llm_prefix}layers.$LAYER_ID.self_attn.v_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}",
        ]
        if config.text_config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.self_attn.q_proj.bias, {llm_prefix}layers.$LAYER_ID.self_attn.k_proj.bias, {llm_prefix}layers.$LAYER_ID.self_attn.v_proj.bias -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}, axis=0",
            ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.mlp.gate_proj.weight^T, {llm_prefix}layers.$LAYER_ID.mlp.up_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            f"{llm_prefix}layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight^T, {llm_prefix}layers.$LAYER_ID.mlp.shared_experts.up_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
            f"{llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
        ]

        # without lm_head
        if cls.base_model_prefix:
            aoa_config["aoa_statements"] += [
                f"{f'{llm_prefix}embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'} -> lm_head.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Glm4vMoeConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                # do cast
                f"{llm_prefix}layers.$LAYER_ID.mlp.gate.weight -> {llm_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
                f"{llm_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias -> {llm_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias, dtype='bfloat16'",
                # do transpose
                f"{llm_prefix}embed_tokens.weight -> {llm_prefix}embed_tokens.weight",
                f"{llm_prefix}norm.weight -> {llm_prefix}norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                # moe
                f"{llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.weight"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.bias -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.bias"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.weight"
                for x in ("up", "gate", "down")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.{x}_proj.bias"
                for x in ("up", "gate", "down")
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.proj.bias -> {visual_prefix}patch_embed.proj.bias",
            f"{visual_prefix}blocks.$LAYER_ID.norm1.weight -> {visual_prefix}blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm2.weight -> {visual_prefix}blocks.$LAYER_ID.norm2.weight",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.{p}_proj.weight^T -> {visual_prefix}merger.{p}_proj.weight"
            for p in ("gate", "up", "down")
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.proj.weight^T -> {visual_prefix}merger.proj.weight",
            f"{visual_prefix}merger.post_projection_norm.weight^T -> {visual_prefix}merger.post_projection_norm.weight",
            f"{visual_prefix}merger.post_projection_norm.bias -> {visual_prefix}merger.post_projection_norm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}embeddings.position_embedding.weight -> {visual_prefix}embeddings.position_embedding.weight",
            f"{visual_prefix}post_conv_layernorm.weight^T -> {visual_prefix}post_conv_layernorm.weight",
            f"{visual_prefix}post_layernorm.weight^T -> {visual_prefix}post_layernorm.weight",
            f"{visual_prefix}downsample.weight -> {visual_prefix}downsample.weight",
            f"{visual_prefix}downsample.bias -> {visual_prefix}downsample.bias",
        ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.q_proj.weight, {llm_prefix}layers.$LAYER_ID.self_attn.k_proj.weight, {llm_prefix}layers.$LAYER_ID.self_attn.v_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}",
        ]
        if config.text_config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> {llm_prefix}layers.$LAYER_ID.self_attn.q_proj.bias, {llm_prefix}layers.$LAYER_ID.self_attn.k_proj.bias, {llm_prefix}layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}, axis=0",
            ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.gate_proj.weight, {llm_prefix}layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
            f"{llm_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> {llm_prefix}layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight, {llm_prefix}layers.$LAYER_ID.mlp.shared_experts.up_proj.weight, fused_ffn",
            f"{llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> {llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, {llm_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, fused_ffn",
        ]

        # without lm_head
        if cls.base_model_prefix:
            aoa_config["aoa_statements"] += [
                f"lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
            ]

        return aoa_config


@dataclass
class Glm4vMoeCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`paddle.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`paddle.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`paddle.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.FloatTensor] = None
    logits: Optional[paddle.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[paddle.FloatTensor]] = None
    attentions: Optional[tuple[paddle.FloatTensor]] = None
    rope_deltas: Optional[paddle.LongTensor] = None
    aux_loss: Optional[paddle.FloatTensor] = None


class Glm4vMoeVisionPatchEmbed(nn.Layer):
    def __init__(self, config: Glm4vMoeVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).reshape(-1, self.embed_dim)
        return hidden_states


class Glm4vMoeVisionRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype=paddle.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class Glm4vMoeVisionPatchMerger(nn.Layer):
    def __init__(self, config, dim: int, context_dim: int, hidden_act: str, bias: bool = False) -> None:
        super().__init__()
        # self.proj = nn.Linear(dim, dim, bias=bias)
        self.proj = GeneralLinear.create(dim, dim, has_bias=bias, linear_type="default")
        self.post_projection_norm = GeneralNorm.create(config, norm_type="layer_norm", hidden_size=dim)
        self.gate_proj = GeneralLinear.create(dim, context_dim, has_bias=bias, linear_type="default")
        self.up_proj = GeneralLinear.create(dim, context_dim, has_bias=bias, linear_type="default")
        self.down_proj = GeneralLinear.create(context_dim, dim, has_bias=bias, linear_type="default")
        self.act1 = nn.GELU()
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: paddle.Tensor) -> paddle.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vMoeVisionEmbeddings(nn.Layer):
    def __init__(self, config: Glm4vMoeVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = GeneralEmbedding.create(
            config=config,
            num_embeddings=self.num_positions,
            embedding_dim=self.embed_dim,
        )
        self.register_buffer("position_ids", paddle.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> paddle.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            embeddings: Input embeddings tensor
            lengths (paddle.Tensor): Sequence lengths for each image in the batch.
            image_shapes (paddle.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (paddle.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (paddle.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            paddle.Tensor: Embeddings with adapted position encoding added.
        """
        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = paddle.empty(0, hidden_size, dtype=pos_embed_weight.dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = paddle.to_tensor(lengths, dtype=paddle.long)
            if not isinstance(image_shapes, paddle.Tensor):
                image_shapes = paddle.to_tensor(image_shapes, dtype=paddle.long)

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d_base = pos_embed_weight.reshape(orig_size, orig_size, hidden_size)
            pos_embed_2d = pos_embed_2d_base.permute(2, 0, 1).unsqueeze(0).to(dtype=paddle.float32)

            # Calculate target dimensions for each patch
            target_h = paddle.cat(
                [paddle.full([lengths[i]], image_shapes[i, 1].item()) for i in range(len(lengths))]
            ).to(dtype=paddle.float32)
            target_w = paddle.cat(
                [paddle.full([lengths[i]], image_shapes[i, 2].item()) for i in range(len(lengths))]
            ).to(dtype=paddle.float32)

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(dtype=paddle.float32)
            w_coords = w_coords.to(dtype=paddle.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = paddle.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            # TODO: "bicubic" mode is not supported now, set "bilinear" temporarily
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d, grid, mode="bilinear", align_corners=False, padding_mode="border"
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype)

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


def apply_rotary_pos_emb_vision(
    q: paddle.Tensor, k: paddle.Tensor, cos: paddle.Tensor, sin: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    with paddle.amp.auto_cast(False):
        q, k = q.astype(dtype="float32"), k.astype(dtype="float32")
        cos, sin = cos.unsqueeze(-2).astype(dtype="float32"), sin.unsqueeze(-2).astype(dtype="float32")
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


class Glm4vMoeVisionAttention(nn.Layer):
    def __init__(self, config: Glm4vMoeVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = GeneralLinear.create(
            config.hidden_size,
            config.hidden_size * 3,
            has_bias=config.attention_bias,
            linear_type="default",
        )
        self.proj = GeneralLinear.create(
            config.hidden_size,
            config.hidden_size,
            has_bias=False,
            linear_type="default",
        )
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # TODO: flash_attention_2 is not supported now
        if self.config._attn_implementation == "flash_attention_2":
            logger.warning("'flash_attention_2' is currently unsupported. " "Switch to 'flashmask' automatically.")
            self.config._attn_implementation = "flashmask"

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["eager"]
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                paddle.compat.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    attn_mask_startend_row_indices=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = paddle.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Glm4vMoeVisionBlock(nn.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.norm1 = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.norm2 = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.attn = Glm4vMoeVisionAttention(config)
        self.mlp = Glm4vMoeVisionMLP(
            config,
            intermediate_size=config.out_hidden_size,
            has_bias=False,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Glm4vMoeVisionModel(Glm4vMoePreTrainedModel):
    input_modalities = ("image", "video")
    _no_split_modules = ["Glm4vMoeVisionBlock"]
    config_class = Glm4vMoeVisionConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = Glm4vMoeVisionEmbeddings(config)
        self.patch_embed = Glm4vMoeVisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm4vMoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList([Glm4vMoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Glm4vMoeVisionPatchMerger(
            config,
            dim=config.out_hidden_size,
            context_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.post_conv_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(paddle.stack([hpos_ids, wpos_ids], dim=-1).tile(repeat_times=[t, 1]))
        pos_ids = paddle.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: paddle.Tensor,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            cu_seqlens,
            position_embeddings,
        )
        return hidden_states

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`paddle.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `paddle.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])

        for blk in self.blocks:
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(
                    blk,
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.reshape(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).reshape(-1, self.config.out_hidden_size)

        hidden_states = self.merger(hidden_states)
        return hidden_states


class Glm4vMoeTextModel(Glm4vMoePreTrainedModel):
    input_modalities = ("text",)
    config_class = Glm4vMoeTextConfig

    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.layers = nn.LayerList(
            [Glm4vMoeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = Glm4vMoeTextRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]],
        attention_mask: Tensor,
        position_ids: Optional[paddle.Tensor],
        past_key_values: Optional[Cache],
        use_cache: Optional[bool] = None,
        cache_position: Optional[paddle.LongTensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_embeddings,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
            attn_mask_startend_row_indices,
        )

        return hidden_states

    def forward(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[paddle.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = paddle.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.reshape(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        # Create the causal mask and row indices
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for i, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    **kwargs,
                )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values] if v is not None)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Glm4vMoeModel(Glm4vMoePreTrainedModel):
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {
        "model.visual": "visual",
        "model.language_model": "language_model",
    }
    _no_split_modules = ["Glm4vMoeTextDecoderLayer", "Glm4vMoeVisionBlock"]
    config_class = Glm4vMoeConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual = Glm4vMoeVisionModel._from_config(config.vision_config)
        self.language_model = Glm4vMoeTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        image_grid_thw: Optional[paddle.LongTensor] = None,
        video_grid_thw: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`paddle.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`paddle.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`paddle.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`paddle.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`paddle.Tensor` of shape `(batch_size)`)
        """

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_start_token_id = self.config.video_start_token_id
        video_end_token_id = self.config.video_end_token_id

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = paddle.ones_like(total_input_ids)
            position_ids = paddle.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            video_group_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                input_tokens = input_ids.tolist()

                input_token_type = []
                video_check_flg = False
                for token in input_tokens:
                    if token == video_start_token_id:
                        video_check_flg = True
                    elif token == video_end_token_id:
                        video_check_flg = False

                    if token == image_token_id and not video_check_flg:
                        input_token_type.append("image")
                    elif token == image_token_id and video_check_flg:
                        input_token_type.append("video")
                    else:
                        input_token_type.append("text")

                input_type_group = []
                for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                    group = list(group)
                    start_index = group[0][0]
                    end_index = group[-1][0] + 1
                    input_type_group.append((key, start_index, end_index))

                llm_pos_ids_list = []
                video_frame_num = 1
                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    if modality_type == "image":
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        t_index = (
                            paddle.arange(llm_grid_t).reshape(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        )
                        h_index = (
                            paddle.arange(llm_grid_h).reshape(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        )
                        w_index = (
                            paddle.arange(llm_grid_w).reshape(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        )
                        llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + st_idx)

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t, h, w = (
                            video_frame_num,
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )

                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t,
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        for t_idx in range(llm_grid_t):
                            t_index = (
                                paddle.to_tensor(t_idx).reshape(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                            )
                            h_index = paddle.arange(llm_grid_h).reshape(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                            w_index = paddle.arange(llm_grid_w).reshape(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                            llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + st_idx)

                        video_group_index += 1

                        if video_group_index >= video_grid_thw[video_index][0]:
                            video_index += 1
                            video_group_index = 0

                        video_frame_num += 1

                    else:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(paddle.arange(text_len).reshape(1, -1).expand(3, -1) + st_idx)

                        video_frame_num = 1

                llm_positions = paddle.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1], device=input_ids.device)
                    .reshape(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: paddle.FloatTensor, video_grid_thw: Optional[paddle.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`paddle.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`paddle.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.astype(self.visual.patch_embed.proj.weight.dtype)
        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        temp_frames_hw = []
        for t, h, w in video_grid_thw:
            repeated_row = paddle.to_tensor([1, h.item(), w.item()]).unsqueeze(0).repeat([t, 1])
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = paddle.cat(temp_frames_hw, dim=0)
        video_embeds = self.visual(pixel_values_videos, grid_thw=flattened_video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = paddle.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: paddle.FloatTensor, image_grid_thw: Optional[paddle.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`paddle.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`paddle.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.astype(self.visual.patch_embed.proj.weight.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = paddle.split(image_embeds, split_sizes)
        return image_embeds

    def get_placeholder_mask(
        self,
        input_ids: paddle.LongTensor,
        inputs_embeds: paddle.FloatTensor,
        image_features: Optional[paddle.FloatTensor] = None,
        video_features: Optional[paddle.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                paddle.to_tensor(self.config.image_token_id, dtype=paddle.long)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                paddle.to_tensor(self.config.video_token_id, dtype=paddle.long)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            # GLM-4.1V and GLM-4.5V special_video_mask is special_image_mask
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.FloatTensor] = None,
        image_grid_thw: Optional[paddle.LongTensor] = None,
        video_grid_thw: Optional[paddle.LongTensor] = None,
        rope_deltas: Optional[paddle.LongTensor] = None,
        cache_position: Optional[paddle.LongTensor] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Glm4vMoeModelOutputWithPast]:
        r"""
        image_grid_thw (`paddle.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`paddle.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = paddle.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = paddle.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = paddle.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / paddle.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_stage = (input_ids is not None and input_ids.shape[1] != 1) or (
                inputs_embeds is not None and inputs_embeds.shape[1] != 1
            )

            if prefill_stage or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = paddle.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.reshape(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        return Glm4vMoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Glm4vMoeForConditionalGeneration(Glm4vMoePreTrainedModel):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        "^language_model": "model.language_model",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config_class = Glm4vMoeConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm4vMoeModel(config)
        self.lm_head = GeneralLMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        if config.tie_word_embeddings:
            self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(
        self, pixel_values_videos: paddle.FloatTensor, video_grid_thw: Optional[paddle.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: paddle.FloatTensor, image_grid_thw: Optional[paddle.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        labels: Optional[paddle.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.FloatTensor] = None,
        image_grid_thw: Optional[paddle.LongTensor] = None,
        video_grid_thw: Optional[paddle.LongTensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.LongTensor] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple, Glm4vMoeCausalLMOutputWithPast]:
        r"""
        labels (`paddle.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`paddle.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration

        >>> model = Glm4vMoeForConditionalGeneration.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
        >>> processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        # Fix: Check if outputs is a tuple or object and get last_hidden_state properly
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        return Glm4vMoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
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
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # NOTE: Due to differences in cache_position, it must be passed as an argument.
        _, seq_length = input_ids.shape
        if past_key_values is None:
            cache_position = paddle.arange(input_ids.shape[1])
        else:
            cache_position = paddle.to_tensor([seq_length - 1])

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # GLM-4.1V position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[paddle.LongTensor],
        inputs_embeds: Optional[paddle.Tensor] = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`paddle.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`paddle.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`paddle.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """

        if inputs_embeds is not None:
            is_image = (
                inputs_embeds
                == self.get_input_embeddings()(paddle.to_tensor(self.config.image_start_token_id, dtype=paddle.long))
            )[..., 0]
            is_video_start = (
                inputs_embeds
                == self.get_input_embeddings()(paddle.to_tensor(self.config.video_start_token_id, dtype=paddle.long))
            )[..., 0]
            is_video_end = (
                inputs_embeds
                == self.get_input_embeddings()(paddle.to_tensor(self.config.video_end_token_id, dtype=paddle.long))
            )[..., 0]
        else:
            is_image = input_ids == self.config.image_start_token_id
            is_video_start = input_ids == self.config.video_start_token_id
            is_video_end = input_ids == self.config.video_end_token_id

        # Cumulative sum to track if we're inside a video span
        # We'll assume well-formed video tags (i.e. matching starts and ends)
        video_level = paddle.cumsum(is_video_start.int() - is_video_end.int(), dim=1)
        inside_video = video_level > 0  # shape (batch_size, seq_length)

        # Mask out image tokens that are inside video spans
        standalone_images = is_image & (~inside_video)

        # Count per batch
        image_counts = standalone_images.sum(dim=1)
        video_counts = is_video_start.sum(dim=1)

        return image_counts, video_counts

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[paddle.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[paddle.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = paddle.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = paddle.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = paddle.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [paddle.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = paddle.split(video_grid_thw, list(video_nums))
                    lengths = [paddle.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], paddle.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "Glm4vMoeForConditionalGeneration",
    "Glm4vMoeModel",
    "Glm4vMoePreTrainedModel",
    "Glm4vMoeTextModel",
    "Glm4vMoeVisionModel",
]
