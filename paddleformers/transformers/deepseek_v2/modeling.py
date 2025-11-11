# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2023 DeepSeek. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Paddle DeepSeek model."""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as DeepseekV2MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.norm import RMSNorm
from ...nn.pp_model import EmbeddingPipe, GeneralModelForCausalLMPipe, parse_args
from ...utils.log import logger
from ...utils.masking_utils import (
    _expand_2d_mask,
    _make_causal_mask,
    get_use_casual_mask,
    is_casual_mask,
)
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..model_outputs import (
    BaseModelOutputWithPastAndMTP,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from ..moe_gate import PretrainedMoEGate
from ..moe_layer import MoEFlexTokenLayer
from .configuration import DeepseekV2Config

__all__ = [
    "DeepseekV2ForCausalLM",
    "DeepseekV2ForSequenceClassification",
    "DeepseekV2Model",
    "DeepseekV2PretrainedModel",
    "DeepseekV2ForCausalLMPipe",
]


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    attn_mask_startend_row_indices=None,
    softmax_scale=1.0,
    training=True,
    sequence_parallel=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, v_num_heads, v_head_dim = value_states.shape

    # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
    q_head_dim = query_states.shape[-1]
    softmax_scale = softmax_scale * (q_head_dim**0.5)
    query_states = query_states * softmax_scale
    value_padding = paddle.zeros(
        [bsz, kv_seq_len, v_num_heads, head_dim - v_head_dim],
        dtype=value_states.dtype,
    )
    value_states = paddle.cat([value_states, value_padding], axis=-1)

    attention_interface = ALL_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Placeholder: module unused but required by flashmask_attention_forward.
    attn_output, attn_weights = attention_interface(
        module=nn.Layer(),
        query=query_states,
        key=key_states,
        value=value_states,
        attention_mask=attention_mask,
        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        dropout=config.get("attention_dropout", 0.0) if training else 0.0,
        scaling=softmax_scale,
    )

    attn_output = attn_output.reshape([bsz, q_len, v_num_heads, head_dim])
    attn_output = attn_output[..., :v_head_dim]
    attn_output = attn_output.reshape([bsz, q_len, -1])

    if sequence_parallel:
        attn_output = attn_output.reshape([bsz * q_len, v_head_dim * num_heads])
    else:
        attn_output = attn_output.reshape([bsz, q_len, v_head_dim * num_heads])

    return (attn_output, attn_weights) if output_attentions else attn_output


def yarn_get_mscale(scale, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _compute_yarn_parameters(
    config,
    seq_len=None,
):
    base = config["rope_theta"]
    rope_parameters_dict = config["rope_scaling"]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "qk_rope_head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor", None)
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")

    # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if "original_max_position_embeddings" in rope_parameters_dict:
        original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = yarn_get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_parameters_dict.get("beta_fast") or 32
    beta_slow = rope_parameters_dict.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (paddle.arange(dim, dtype=paddle.float32) - min) / (max - min)
        ramp_func = paddle.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (paddle.arange(0, dim, 2).astype(paddle.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    # truncate = config.rope_parameters.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, True)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).astype(paddle.float32)

    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


class DeepseekV2YarnRotaryEmbedding(nn.Layer):
    def __init__(self, config: DeepseekV2Config, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.rope_type = self.config.rope_scaling["type"]
        assert self.rope_type == "yarn"

        self.inv_freq, self.attention_scaling = _compute_yarn_parameters(config)
        self.register_buffer("inv_freq", self.inv_freq, persistable=False)
        # self.original_inv_freq = self.inv_freq

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # NOTE: Paddle's Automatic Mixed Precision (AMP) has a default op whitelist that may automatically cast
        # certain operations (like matmul) to FP16/BF16 for performance optimization. However, in scenarios where
        # numerical stability is critical (e.g., RoPE init/compute), this conversion can lead to precision loss.
        # Disabling auto_cast here ensures the matmul operation runs in the original precision (FP32) as intended.
        with paddle.amp.auto_cast(False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = paddle.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden axiss of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, fuse_rope=False):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    b, s, h, d = q.shape
    q = q.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])

    b, s, h, d = k.shape
    k = k.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])

    if position_ids is None:
        # Note: Only for MixtralForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, axis]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, axis]
    else:
        cos = cos.squeeze().contiguous()  # [seq_len, axis]
        sin = sin.squeeze().contiguous()  # [seq_len, axis]
        if b == 1:
            cos = cos.unsqueeze(0).contiguous()
            sin = sin.unsqueeze(0).contiguous()
        cos = cos.unsqueeze(2).contiguous()  # [bs, seq_len, 1, axis]
        sin = sin.unsqueeze(2).contiguous()  # [bs, seq_len, 1, axis]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FakeGate(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, hidden_states, weight):
        expert_num = weight.shape[1]
        bsz, seq, _ = hidden_states.shape

        ctx.x_shape = hidden_states.shape
        ctx.x_dtype = hidden_states.dtype
        ctx.y_shape = weight.shape
        ctx.y_dtype = weight.dtype

        return paddle.randn([bsz, seq, expert_num]).cast(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return paddle.zeros(ctx.x_shape, dtype=ctx.x_dtype), paddle.zeros(ctx.y_shape, dtype=ctx.y_dtype)


class MoEGate(PretrainedMoEGate):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super().__init__(config, num_experts, expert_hidden_size, **kwargs)
        # [hidden_size, n_expert]

        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method

        self.weight = paddle.create_parameter(
            shape=[expert_hidden_size, num_experts],
            dtype=paddle.float32,
            is_bias=False,
        )

        self.config = config
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = paddle.create_parameter(
                shape=[num_experts],
                dtype=paddle.float32,
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.e_score_correction_bias.is_distributed = True
            self.e_score_correction_bias.stop_gradient = True
            self.expert_usage = paddle.zeros(
                shape=[num_experts],
                dtype=paddle.int64,
            )
            self.expert_usage.stop_gradient = True

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (_type_): [batch_size * seq_len, hidden_size]
        """

        # compute gating score
        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.cast(self.weight.dtype)

            if hasattr(self.config, "using_fake_gate") and self.config.using_fake_gate:
                logits = FakeGate.apply(hidden_states, self.weight)
            else:
                logits = F.linear(hidden_states, self.weight, None)
            scores = self.gate_score_func(logits=logits)
            scores = scores.cast(paddle.float32)

        scores, routing_map, exp_counts, l_aux, l_zloss = self.topkgating_nodrop(scores)
        with paddle.no_grad():
            self.expert_usage += exp_counts
        return scores, routing_map, l_aux, l_zloss


class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert paddle.numel(loss) == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = not loss.stop_gradient
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = paddle.ones(1, dtype=ctx.dtype)
        return grad_output, grad_loss


class DeepseekV2TopkRouter(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = paddle.create_parameter(
            shape=[config.hidden_size, self.n_routed_experts],
            dtype=paddle.float32,
            is_bias=False,
        )
        self.register_buffer("e_score_correction_bias", paddle.zeros(self.n_routed_experts))
        self._cast_to_low_precision = False

    @paddle.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = paddle.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = paddle.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = paddle.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.view(-1, self.config.hidden_size)
            router_logits = F.linear(hidden_states.astype(paddle.float32), self.weight.astype(paddle.float32))

            scores = router_logits.sigmoid().cast(paddle.float32)
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class DeepseekV2MoE(nn.Layer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        new_config = deepcopy(config)
        new_config.tensor_parallel_degree = 1

        self.experts = nn.LayerList(
            [
                DeepseekV2MLP(new_config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV2TopkRouter(config)
        self.shared_experts = DeepseekV2MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def moe(self, hidden_states: paddle.Tensor, topk_indices: paddle.Tensor, topk_weights: paddle.Tensor):
        final_hidden_states = paddle.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = paddle.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = paddle.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(index=token_indices, axis=0, value=weighted_output)

        return final_hidden_states.astype(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class DeepseekV2MoEFlexToken(MoEFlexTokenLayer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config):
        gate = MoEGate(
            config=config,
            num_experts=config.n_routed_experts,
            expert_hidden_size=config.hidden_size,
            top_k=config.num_experts_per_tok,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            drop_tokens=False,
        )

        hcg = fleet.get_hybrid_communicate_group()
        moe_group = hcg.get_expert_parallel_group()
        moe_grad_group = hcg.get_moe_sharding_parallel_group()
        new_config = deepcopy(config)
        new_config.tensor_parallel_degree = 1

        super().__init__(
            config=config,
            moe_num_experts=config.n_routed_experts,
            expert_class=DeepseekV2MLP,
            expert_kwargs={"config": new_config, "intermediate_size": config.moe_intermediate_size},
            gate=gate,
            moe_group=moe_group,
        )

        self.is_mp_moe = False
        self.is_ep_moe = True
        for p in self.experts.parameters():
            setattr(p, "is_moe_param", True)
            setattr(p, "color", {"color": "moe_expert", "group": moe_grad_group})
            p.no_sync = not self.is_mp_moe
            p.expert = not self.is_mp_moe
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if self.is_mp_moe or self.is_ep_moe:
                p.is_distributed = True

        self.alpha = config.aux_loss_alpha
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        final_hidden_states, l_aux, l_zloss = super().forward(hidden_states)
        if self.training and self.alpha > 0.0:
            l_aux = l_aux * self.alpha
            final_hidden_states = AddAuxiliaryLoss.apply(final_hidden_states, l_aux)

        if self.config.n_shared_experts is not None:
            shared_expert_output = self.shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states


class DeepseekV2Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads
        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"Attention head num ({self.num_heads}) is not divisible by tensor_parallel_degree ({config.tensor_parallel_degree})."
            self.num_local_heads = self.num_heads // config.tensor_parallel_degree

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True
        self.fuse_rope = config.use_fused_rope

        self.seq_length = config.seq_length
        self.tensor_parallel = config.tensor_parallel_degree > 1
        self.sequence_parallel = config.sequence_parallel

        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity

        # Note (@DrownFish19): For tensor parallel we consider that q_a_proj and kv_a_proj_with_mqa
        # are the small weight and cannot achieve performance gain. So we use the original
        # linear layers. We use the tensor parallel linear layers for q_proj，q_b_proj and kv_b_proj
        # for which are the large weight and can achieve performance gain.

        if self.q_lora_rank is None:
            self.q_proj = GeneralLinear.create(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                has_bias=False,
                config=config,
                fuse_matmul_bias=config.fuse_linear,
                tp_plan="colwise",
                gather_output=False,
            )
        else:
            self.q_a_proj = GeneralLinear.create(
                self.hidden_size,
                config.q_lora_rank,
                has_bias=config.attention_bias,
                config=config,
                fuse_matmul_bias=config.fuse_linear,
                linear_type="default",
                gather_output=False,
            )
            self.q_b_proj = GeneralLinear.create(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                has_bias=False,
                config=config,
                fuse_matmul_bias=config.fuse_linear,
                tp_plan="colwise",
                gather_output=False,
            )
        self.q_a_layernorm = GeneralNorm.create(
            config=config,
            hidden_size=config.q_lora_rank,
            norm_type="rms_norm",
            input_is_parallel=self.tensor_parallel,
        )

        self.kv_a_proj_with_mqa = GeneralLinear.create(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            has_bias=config.attention_bias,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            linear_type="default",
            gather_output=False,
        )

        self.kv_b_proj = GeneralLinear.create(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            has_bias=False,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            tp_plan="colwise",
            gather_output=False,
        )

        self.o_proj = GeneralLinear.create(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            tp_plan="rowwise",
            gather_output=False,
            input_is_parallel=True,
        )

        self.kv_a_layernorm = GeneralNorm.create(
            config=config,
            hidden_size=config.kv_lora_rank,
            norm_type="rms_norm",
            input_is_parallel=self.tensor_parallel and self.sequence_parallel,
        )

        if self.tensor_parallel and self.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.kv_a_proj_with_mqa.weight)
            mark_as_sequence_parallel_parameter(self.q_a_proj.weight)
            if config.attention_bias:
                mark_as_sequence_parallel_parameter(self.kv_a_proj_with_mqa.bias)
                mark_as_sequence_parallel_parameter(self.q_a_proj.bias)

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_func = scaled_dot_product_attention

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.v_head_dim]).transpose([1, 0, 2, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor]] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        ori_shape = hidden_states.shape
        # DeepSeekV2 q_lora_rank=1536
        # DeepSeekV2-lite q_lora_rank=None
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        if self.sequence_parallel:
            target_query_shape = [-1, self.seq_length, self.num_local_heads, self.q_head_dim]
            target_key_value_shape = [
                -1,
                self.seq_length,
                self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ]
        else:
            target_query_shape = [0, 0, self.num_heads, self.q_head_dim]
            target_key_value_shape = [0, 0, self.num_heads, self.qk_nope_head_dim + self.v_head_dim]

        q = q.reshape(shape=target_query_shape)
        q_nope, q_pe = paddle.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        # DeepSeekV2 kv_lora_rank+qk_rope_head_dim=512+64
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
        if self.sequence_parallel:
            k_pe = GatherOp.apply(k_pe)
        k_pe = k_pe.reshape([-1, self.seq_length, 1, self.qk_rope_head_dim]).expand(
            [-1, self.seq_length, self.num_local_heads, self.qk_rope_head_dim]
        )
        # self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim = 128+64
        # self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim) = config.qk_nope_head_dim + self.v_head_dim = 128+128
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).reshape(shape=target_key_value_shape)
        k_nope, value_states = paddle.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)
        kv_seq_len = value_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        cos, sin = position_embeddings[0], position_embeddings[1]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, self.fuse_rope)
        query_states = paddle.cat([q_nope, q_pe], axis=-1)
        key_states = paddle.cat([k_nope, k_pe], axis=-1)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.cat([past_key_value[0], key_states], axis=1)
            value_states = paddle.cat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if self.enable_recompute and has_gradient and self.recompute_granularity == "core_attn":
            outputs = recompute(
                self.attn_func,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                softmax_scale=self.softmax_scale,
                training=self.training,
                sequence_parallel=self.sequence_parallel,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = self.attn_func(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                softmax_scale=self.softmax_scale,
                training=self.training,
                sequence_parallel=self.sequence_parallel,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.o_proj(attn_output)
        if attn_output.shape != ori_shape:
            attn_output = attn_output.reshape(ori_shape)

        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class DeepseekV2DecoderLayer(nn.Layer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity
        self.tensor_parallel = config.tensor_parallel_degree > 1
        self.sequence_parallel = config.sequence_parallel
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV2Attention(config=config)

        try:
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        except:
            moe_group = None

        expert_paralled_degree = dist.get_world_size(moe_group) if moe_group is not None else 1
        MoELayerClass = DeepseekV2MoE if expert_paralled_degree <= 1 else DeepseekV2MoEFlexToken

        self.mlp = (
            MoELayerClass(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )

        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            input_is_parallel=self.tensor_parallel and self.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            input_is_parallel=self.tensor_parallel and self.sequence_parallel,
        )

    def subbatch_recompute_forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        offload_kwargs = {}
        offload_kwargs["offload_indices"] = [0]
        assert self.recompute_granularity != "full_attn"
        attn_outputs = recompute(
            self.attn,
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            position_embeddings,
            **offload_kwargs,
        )
        hidden_states = attn_outputs[0]
        residual = attn_outputs[1]
        self_attn_weights = attn_outputs[2] if output_attentions else None
        present_key_value = attn_outputs[3] if use_cache else None
        sub_seq_len = self.config.moe_subbatch_token_num
        seq_axis = 0 if self.config.sequence_parallel else 1
        seq_len = hidden_states.shape[seq_axis]
        assert seq_len % sub_seq_len == 0
        num_chunks = seq_len // sub_seq_len
        split_list = [sub_seq_len] * num_chunks
        input_list = paddle.split(hidden_states, split_list, axis=seq_axis)
        output_list = []

        for chunk in input_list:
            out = recompute(
                self.mlp.forward,
                chunk,
                **offload_kwargs,
            )
            output_list.append(out)
        hidden_states = paddle.concat(output_list, axis=seq_axis)
        outputs = recompute(
            self.post_process,
            hidden_states,
            residual,
            output_attentions,
            use_cache,
            self_attn_weights,
            present_key_value,
            **offload_kwargs,
        )
        return outputs

    def attn(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if self.enable_recompute and has_gradient and self.recompute_granularity == "full_attn":
            outputs = recompute(
                self.self_attn,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            outputs = self.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        attn_outputs = (hidden_states, residual)

        if output_attentions:
            self_attn_weights = outputs[1]
            attn_outputs += (self_attn_weights,)

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]
            attn_outputs += (present_key_value,)

        return attn_outputs

    def post_process(
        self,
        hidden_states,
        residual,
        output_attentions=False,
        use_cache=False,
        self_attn_weights=None,
        present_key_value=None,
    ):
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        attn_outputs = self.attn(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            position_embeddings,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        residual = attn_outputs[1]
        self_attn_weights = attn_outputs[2] if output_attentions else None
        present_key_value = attn_outputs[3] if use_cache else None
        hidden_states = self.mlp(hidden_states)
        outputs = self.post_process(
            hidden_states, residual, output_attentions, use_cache, self_attn_weights, present_key_value
        )
        return outputs


class DeepseekV2MTPLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
    ):
        super(DeepseekV2MTPLayer, self).__init__(config, layer_idx)

        self.enorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            input_is_parallel=self.tensor_parallel and self.sequence_parallel,
        )
        self.hnorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            input_is_parallel=self.tensor_parallel and self.sequence_parallel,
        )
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)

        if config.sequence_parallel and config.tensor_parallel_degree > 1:
            mark_as_sequence_parallel_parameter(self.eh_proj.weight)
            mark_as_sequence_parallel_parameter(self.eh_proj.bias)

    def subbatch_recompute_forward(
        self,
        hidden_states: paddle.Tensor,
        nextn_hidden_state: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        hidden_states = self.hnorm(hidden_states)
        nextn_hidden_state = self.enorm(nextn_hidden_state)

        hidden_states = self.eh_proj(paddle.concat([nextn_hidden_state, hidden_states], axis=-1))

        layer_outputs = super(DeepseekV2MTPLayer, self).subbatch_recompute_forward(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            position_embeddings,
            **kwargs,
        )

        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        return hidden_states

    def forward(
        self,
        hidden_states: paddle.Tensor,
        nextn_hidden_state: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        hidden_states = self.hnorm(hidden_states)
        nextn_hidden_state = self.enorm(nextn_hidden_state)

        hidden_states = self.eh_proj(paddle.cat([hidden_states, nextn_hidden_state], axis=-1))

        layer_outputs = super(DeepseekV2MTPLayer, self).forward(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            position_embeddings,
            **kwargs,
        )

        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        return hidden_states


class DeepseekV2PretrainedModel(PretrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV2DecoderLayer"]
    transpose_weight_keys = [
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
        "eh_proj",
    ]
    _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]

    @classmethod
    def _get_name_mappings(cls, config: DeepseekV2Config) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        # last one layer contains MTP (eagle) parameters for inference
        for layer_index in range(config.num_hidden_layers + config.num_nextn_predict_layers):
            layer_mappings = [
                [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.q_a_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.q_a_layernorm.weight"],
                [f"layers.{layer_index}.self_attn.q_b_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.kv_a_proj_with_mqa.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.kv_a_layernorm.weight"],
                [f"layers.{layer_index}.self_attn.kv_b_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

            # MoE parameters
            model_mappings.append([f"layers.{layer_index}.mlp.gate.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.gate.e_score_correction_bias"])
            for expert_idx in range(config.n_routed_experts):
                expert_mappings = [
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.gate_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.up_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.down_proj.weight", None, "transpose"],
                ]
                model_mappings.extend(expert_mappings)
            model_mappings.append([f"layers.{layer_index}.mlp.shared_experts.gate_proj.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.shared_experts.up_proj.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.shared_experts.down_proj.weight", None, "transpose"])

            # MTP (eagle) parameters for inference
            if layer_index >= config.num_hidden_layers:
                model_mappings.append([f"layers.{layer_index}.embed_tokens.weight"])
                model_mappings.append([f"layers.{layer_index}.enorm.weight"])
                model_mappings.append([f"layers.{layer_index}.hnorm.weight"])
                model_mappings.append([f"layers.{layer_index}.eh_proj.weight", None, "transpose"])
                model_mappings.append([f"layers.{layer_index}.shared_head.norm.weight"])
                model_mappings.append([f"layers.{layer_index}.shared_head.head.weight", None, "transpose"])

        init_name_mappings(mappings=model_mappings)
        if cls.base_model_class.__name__ not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = f"{cls.base_model_prefix}." + mapping[1]
            if not config.tie_word_embeddings:
                model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: DeepseekV2Config, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            base_actions["lm_head.weight"] = partial(fn, is_column=False)

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")

            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(fn, is_column=True)

            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.kv_b_proj.weight"] = partial(fn, is_column=True)

            # dense mlp
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.down_proj.weight"] = partial(fn, is_column=False)

            # moe unit shared experts
            base_actions["layers.0.mlp.shared_experts.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.down_proj.weight"] = partial(fn, is_column=False)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            # for MTP (eagle) parameters for inference
            base_actions.pop("embed_tokens.weight")
            base_actions.pop("lm_head.weight")
            base_actions["layers.0.embed_tokens.weight"] = partial(fn, is_column=False)
            base_actions["layers.0.shared_head.head.weight"] = partial(fn, is_column=True)
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(
                        config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers
                    ):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                else:
                    final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings


@register_base_model
class DeepseekV2Model(DeepseekV2PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )

        self.layers = nn.LayerList(
            [DeepseekV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        for layer_idx in range(config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers):
            self.layers.append(DeepseekV2MTPLayer(config, layer_idx))

        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            input_is_parallel=config.tensor_parallel_degree > 1 and config.sequence_parallel,
        )

        self.enable_recompute = False
        self.rotary_emb = DeepseekV2YarnRotaryEmbedding(config=config)

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Tensor,
        output_attentions: bool,
        past_key_value: Tensor,
        use_cache: bool,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndMTP]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if self.config.num_nextn_predict_layers > 0:
            seq_length -= self.config.num_nextn_predict_layers

            if attention_mask is not None:
                attention_mask = attention_mask[
                    :, :, : -self.config.num_nextn_predict_layers, : -self.config.num_nextn_predict_layers
                ].contiguous()

            # attn_mask_startend_row_indices: [b, num_head, seq_len] or [b, num_head, seq_len, C], C is 2 or 4
            if attn_mask_startend_row_indices is not None:
                if attn_mask_startend_row_indices.ndim == 3:
                    attn_mask_startend_row_indices = attn_mask_startend_row_indices[
                        :,
                        :,
                        : -self.config.num_nextn_predict_layers,
                    ].contiguous()
                elif attn_mask_startend_row_indices.ndim == 4:
                    attn_mask_startend_row_indices = attn_mask_startend_row_indices[
                        :, :, : -self.config.num_nextn_predict_layers, :
                    ].contiguous()
                else:
                    raise ValueError("attn_mask_startend_row_indices must be 3D or 4D tensor")

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[1]
            seq_length_with_past += past_key_values_length

        if position_ids is None and not self.config.fuse_rope:
            position_ids = (
                paddle.arange(
                    0,
                    self.config.seq_length,
                    dtype="int64",
                )
                .unsqueeze(0)
                .tile([input_ids.shape[0], 1])
            ).contiguous()

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

        if position_embeddings is None:
            position_embeddings = paddle.stack(self.rotary_emb(inputs_embeds, position_ids=position_ids))

        # embed positions
        if attn_mask_startend_row_indices is not None or get_use_casual_mask():
            attention_mask = None
        else:
            # [bs, seq_len]
            attention_mask = (
                paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                if attention_mask is None
                else attention_mask
            )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), past_key_values_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]
            if self.config.use_flash_attention:
                attention_mask = None if is_casual_mask(attention_mask) else attention_mask

        if self.config.num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds

        if self.config.sequence_parallel:
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape(inputs_embeds, [bs * seq_len, hidden_size])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds.contiguous()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        mtp_outputs = []

        moelayer_use_subbatch_recompute = self.config.moe_subbatch_token_num > 0

        for idx in range(self.config.num_hidden_layers):
            decoder_layer = self.layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if moelayer_use_subbatch_recompute:
                layer_outputs = decoder_layer.subbatch_recompute_forward(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                    position_embeddings,
                )
            elif self.enable_recompute and has_gradient and self.recompute_granularity == "full":
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                    position_embeddings,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.config.num_nextn_predict_layers > 0:
            mtp_outputs.append(hidden_states)

            for nextn in range(self.config.num_nextn_predict_layers):
                decoder_layer = self.layers[nextn + self.config.num_hidden_layers]

                if self.config.sequence_parallel:
                    hidden_states = GatherOp.apply(hidden_states)
                    hidden_states = hidden_states.reshape([-1, seq_length, hidden_states.shape[-1]])

                inputs_embeds_cur_depth = paddle.cat(
                    [inputs_embeds_ori[:, (nextn + 1) :, :], inputs_embeds_extra[:, : (nextn + 1), :]], axis=1
                )

                past_key_value = None
                layer_outputs = decoder_layer(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                    position_embeddings,
                )

                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                mtp_outputs.append(hidden_states)
            mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, mtp_outputs] if v is not None
            )
        return BaseModelOutputWithPastAndMTP(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mtp_outputs=mtp_outputs,
        )


class DeepseekV2PretrainingCriterion(nn.Layer):
    """
    Criterion for Mixtral.
    It calculates the final loss.
    """

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekV2PretrainingCriterion, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=self.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels, router_loss=None, mtp_logits=None):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                warnings.warn(
                    f"enable_parallel_cross_entropy, the vocab_size should be splitted: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        def subbatch_compute_loss(preds, labels, subbatch_token_num):
            seq_axis = 1
            seq_len = preds.shape[seq_axis]

            assert seq_len % subbatch_token_num == 0
            num_chunks = seq_len // subbatch_token_num
            preds_list = paddle.split(preds, num_chunks, axis=seq_axis)
            labels_list = paddle.split(labels, num_chunks, axis=seq_axis)

            loss_list = []
            offload_kwargs = {}
            for pred_chunk, label_chunk in zip(preds_list, labels_list):
                with paddle.amp.auto_cast(False):
                    offload_kwargs["offload_indices"] = [0]
                    sub_loss = recompute(
                        self.loss_func,
                        pred_chunk.astype("float32"),
                        label_chunk.unsqueeze(2),
                        **offload_kwargs,
                    )
                    loss_list.append(sub_loss)

            masked_lm_loss = paddle.concat(loss_list, axis=seq_axis)
            binary_sequence = paddle.where(
                masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count

            return loss

        def compute_loss(preds, labels):
            with paddle.amp.auto_cast(False):
                masked_lm_loss = self.loss_func(preds.astype("float32"), labels.unsqueeze(2))
                binary_sequence = paddle.where(
                    masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
                )
                count = paddle.sum(binary_sequence)
                if count == 0:
                    loss = paddle.sum(masked_lm_loss * binary_sequence)
                else:
                    loss = paddle.sum(masked_lm_loss * binary_sequence) / count
                return loss

        def add_loss(main_loss, loss):
            return main_loss + loss - loss.detach()

        if mtp_logits is not None and self.config.num_nextn_predict_layers > 0:
            assert len(mtp_logits) == self.config.num_nextn_predict_layers
            masked_lm_labels_ori = masked_lm_labels
            masked_lm_labels = masked_lm_labels[:, : -self.config.num_nextn_predict_layers]
            seq_length = masked_lm_labels.shape[1]
            if self.config.moe_subbatch_token_num > 0:
                loss = subbatch_compute_loss(prediction_scores, masked_lm_labels, self.config.moe_subbatch_token_num)
            else:
                loss = compute_loss(prediction_scores, masked_lm_labels)

            mtp_loss_res = []
            for depth in range(self.config.num_nextn_predict_layers):
                prediction_scores_cur_depth = mtp_logits[depth]
                masked_lm_labels_cur_depth = masked_lm_labels_ori[:, (depth + 1) : (depth + 1 + seq_length)]
                if self.config.moe_subbatch_token_num > 0:
                    res_cur_depth = subbatch_compute_loss(
                        prediction_scores_cur_depth, masked_lm_labels_cur_depth, self.config.moe_subbatch_token_num
                    )
                else:
                    res_cur_depth = compute_loss(prediction_scores_cur_depth, masked_lm_labels_cur_depth)
                mtp_loss_res.append(res_cur_depth)
            loss = add_loss(
                loss, self.config.num_nextn_predict_lambda * sum([x for x in mtp_loss_res]) / len(mtp_loss_res)
            )

        else:
            if self.config.moe_subbatch_token_num > 0:
                loss = subbatch_compute_loss(prediction_scores, masked_lm_labels, self.config.moe_subbatch_token_num)
            else:
                loss = compute_loss(prediction_scores, masked_lm_labels)

        if router_loss is not None and isinstance(router_loss, paddle.Tensor):
            loss = add_loss(loss, router_loss)

        return loss


class DeepseekV2ForCausalLM(DeepseekV2PretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.config = config
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)

    def get_input_embeddings(self):
        return self.deepseek_v2.embed_tokens

    def set_input_embeddings(self, value):
        self.deepseek_v2.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.deepseek_v2

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.deepseek_v2(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        if return_dict:
            hidden_states = outputs.hidden_states
            mtp_outputs = outputs.mtp_outputs
        else:
            hidden_states = outputs[0]
            mtp_outputs = outputs[-1]

        if labels is not None and self.config.use_fused_linear_cross_entropy:
            from paddlenlp_kernel.triton.cut_cross_entropy import linear_cross_entropy

            assert (
                self.config.tensor_parallel_degree <= 1
            ), "The argument `use_fused_linear_cross_entropy` is imcompatiable with tensor parallel "

            masked_lm_loss = linear_cross_entropy(hidden_states, self.lm_head.weight, targets=labels)

            binary_sequence = paddle.where(
                masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count
            logits = None
        else:
            # if labels is None，means we need full output, instead of tensor_parallel_output
            # tensor_parallel_output is together with ParallelCrossEntropy
            tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1
            logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)
            mtp_logits = (
                [
                    self.lm_head(_hidden_states, tensor_parallel_output=tensor_parallel_output)
                    for _hidden_states in mtp_outputs
                ]
                if len(mtp_outputs) > 0
                else []
            )

            loss = None
            if labels is not None:
                loss = self.criterion(logits, labels, mtp_logits=mtp_logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.cat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            # TODO: support attention mask for other models
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.cat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )
            elif len(attention_mask.shape) == 4:
                model_kwargs["attention_mask"] = paddle.cat(
                    [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )[:, :, -1:, :]

        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class DeepseekV2ForSequenceClassification(DeepseekV2PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias_attr=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = paddle.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int64):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class DeepseekV2MTPLayerPipe(DeepseekV2MTPLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, position_embeddings, nbatch_pack_offset = parse_args(args)

        if attention_mask is None:
            attn_mask = None
            attn_mask_startend_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            attn_mask = None
            attn_mask_startend_row_indices = attention_mask
        else:
            attn_mask = attention_mask
            attn_mask_startend_row_indices = None
            assert len(attn_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {attn_mask.shape}."

        hidden_states_list = paddle.split(hidden_states, self.config.num_nextn_predict_layers + 1, axis=-1)
        hidden_states_main_model = hidden_states_list[0]
        inputs_embeds_cur_depth_list = hidden_states_list[1:]
        has_gradient = not hidden_states_main_model.stop_gradient

        output_list = [hidden_states_main_model]
        hidden_states = hidden_states_main_model
        for depth in range(self.config.num_nextn_predict_layers):
            inputs_embeds_cur_depth = inputs_embeds_cur_depth_list[depth]

            moelayer_use_subbatch_recompute = self.config.moe_subbatch_token_num > 0
            if moelayer_use_subbatch_recompute:
                hidden_states = super().subbatch_recompute_forward(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_embeddings=position_embeddings,
                )
            elif self.enable_recompute and self.config.recompute_granularity == "full" and has_gradient:
                if attn_mask is not None or attn_mask_startend_row_indices is not None:
                    hidden_states = recompute(
                        super().forward,
                        hidden_states,
                        inputs_embeds_cur_depth,
                        position_ids=position_ids,
                        attention_mask=attn_mask,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        use_reentrant=self.config.recompute_use_reentrant,
                        position_embeddings=position_embeddings,
                    )
                else:
                    # for pretrain
                    hidden_states = recompute(
                        super().forward,
                        hidden_states,
                        inputs_embeds_cur_depth,
                        position_ids=position_ids,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        use_reentrant=self.config.recompute_use_reentrant,
                        position_embeddings=position_embeddings,
                    )
            else:
                hidden_states = super().forward(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_embeddings=position_embeddings,
                )
            output_list.append(hidden_states)

        hidden_states = paddle.concat(output_list, axis=-1)

        ret = (hidden_states,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if position_embeddings is not None:
            ret += (position_embeddings.clone(),)
        return ret


class DeepseekV2EmbeddingPipe(EmbeddingPipe):
    def __init__(self, config, embed_cls=None, rotary_emb_cls=None):
        rotary_emb_cls = DeepseekV2YarnRotaryEmbedding
        super().__init__(config, embed_cls, rotary_emb_cls)

    def forward(self, args):
        num_nextn_predict_layers = self.config.get("num_nextn_predict_layers", 0)
        input_ids, attention_mask, position_ids, position_embeddings, _ = parse_args(
            args, num_nextn_predict_layers > 0
        )
        inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        batch_size, max_seq_len = input_ids.shape
        max_seq_len -= self.config.num_nextn_predict_layers
        if attention_mask is None:
            attn_mask = None
            attn_mask_startend_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            attn_mask = None
            attn_mask_startend_row_indices = attention_mask[:, :, :max_seq_len]
        else:
            attn_mask = attention_mask[:, :, :max_seq_len, :max_seq_len]
            attn_mask_startend_row_indices = None
            assert len(attn_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {attn_mask.shape}."
        if attn_mask is not None:
            assert (
                attn_mask_startend_row_indices is None
            ), "attention_mask and attn_mask_startend_row_indices can not be set at same time"
            attn_mask = DeepseekV2Model._prepare_decoder_attention_mask(
                attn_mask, (batch_size, max_seq_len), 0, inputs_embeds.dtype
            )
        attn_mask = attn_mask_startend_row_indices if attn_mask_startend_row_indices is not None else attn_mask

        if position_ids is None and not self.config.fuse_rope:
            position_ids = (
                paddle.arange(
                    0,
                    max_seq_len,
                    dtype="int64",
                )
                .unsqueeze(0)
                .tile([input_ids.shape[0], 1])
            ).contiguous()

        if position_embeddings is None:
            position_embeddings = paddle.stack(self.rotary_emb(inputs_embeds, position_ids=position_ids))

        if num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds
            batch_size, seq_length, _ = inputs_embeds.shape

            if self.sequence_parallel:
                inputs_embeds = paddle.reshape(inputs_embeds, [-1, inputs_embeds.shape[-1]])
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            embeds_res = [inputs_embeds]
            for depth in range(num_nextn_predict_layers):
                inputs_embeds_mtp = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )
                if self.sequence_parallel:
                    inputs_embeds_mtp = paddle.reshape(inputs_embeds_mtp, [-1, inputs_embeds_mtp.shape[-1]])
                    inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)
                embeds_res.append(inputs_embeds_mtp)
            res = paddle.concat(embeds_res, axis=-1)
            ret = (res,)
        else:
            if self.sequence_parallel:
                inputs_embeds = paddle.reshape(inputs_embeds, [-1, inputs_embeds.shape[-1]])
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            ret = (inputs_embeds,)

        if attn_mask is not None:
            ret += (attn_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if position_embeddings is not None:
            ret += (position_embeddings.clone(),)
        return ret


class DeepseekV2DecoderLayerPipe(DeepseekV2DecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, position_embeddings, _ = parse_args(args)

        if self.config.num_nextn_predict_layers > 0:
            hidden_size = hidden_states.shape[-1]
            batch_size_mtp = hidden_size // (self.config.num_nextn_predict_layers + 1)
            inputs_embeds_mtp = hidden_states[..., -batch_size_mtp:].contiguous()
            hidden_states = hidden_states[..., :batch_size_mtp].contiguous()

        if attention_mask is None:
            attn_mask = None
            attn_mask_startend_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            attn_mask = None
            attn_mask_startend_row_indices = attention_mask
        else:
            attn_mask = attention_mask
            attn_mask_startend_row_indices = None
            assert len(attn_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {attn_mask.shape}."

        has_gradient = not hidden_states.stop_gradient

        moelayer_use_subbatch_recompute = self.config.moe_subbatch_token_num > 0
        if moelayer_use_subbatch_recompute:
            hidden_states = super().subbatch_recompute_forward(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=position_embeddings,
            )
        elif self.enable_recompute and self.config.recompute_granularity == "full" and has_gradient:
            hidden_states = recompute(
                super().forward,
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                use_reentrant=self.config.recompute_use_reentrant,
                position_embeddings=position_embeddings,
            )
        else:
            hidden_states = super().forward(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=position_embeddings,
            )

        if self.config.num_nextn_predict_layers > 0:
            hidden_states = paddle.concat([hidden_states, inputs_embeds_mtp], axis=-1)

        if isinstance(hidden_states, paddle.Tensor):
            ret = (hidden_states,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if len(ret) == 1:
            (ret,) = ret
        if position_embeddings is not None:
            ret += (position_embeddings.clone(),)
        return ret


class DeepseekV2LMHeadPipe(GeneralLMHead):
    def forward(self, args):
        if self.config.num_nextn_predict_layers > 0:
            logits = []
            for _hidden_states in args:
                logits.append(super().forward(_hidden_states))
            return logits

        hidden_states, _, _, _, _ = parse_args(args)
        logits = super().forward(hidden_states)
        return logits


class DeepseekV2PretrainingCriterionPipe(DeepseekV2PretrainingCriterion):
    def forward(self, logits, labels):

        # in GeneralModelForCausalLMPipe last_stage_keys = ["labels", "loss_mask"]
        labels = labels[0]
        if self.config.num_nextn_predict_layers > 0:
            mtp_logits = logits[1:]
            logits = logits[0]
            loss = super().forward(logits, labels, mtp_logits=mtp_logits)
        else:
            loss = super().forward(logits, labels)
        return loss


class DeepseekV2RMSNormLayerPipe(RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.sequence_parallel:
            self.enable_sequence_parallel()

    def forward(self, args):
        hidden_states, _, _, _, _ = parse_args(args)

        if self.config.num_nextn_predict_layers > 0:
            hidden_states_list = paddle.split(hidden_states, self.config.num_nextn_predict_layers + 1, axis=-1)
            hidden_states = hidden_states_list[0]
            hidden_states_mtp = hidden_states_list[-self.config.num_nextn_predict_layers :]

            output_list = [super().forward(hidden_states)]
            for hidden_states in hidden_states_mtp:
                output_list.append(super().forward(hidden_states))
            return output_list
        else:
            hidden_states = super().forward(hidden_states)
            return hidden_states


class DeepseekV2ForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = DeepseekV2Config
    _embedding_pipe_cls = DeepseekV2EmbeddingPipe
    _decoder_layer_cls = DeepseekV2DecoderLayer
    _criterion_pipe_cls = DeepseekV2PretrainingCriterionPipe
    _lmhead_pipe_cls = DeepseekV2LMHeadPipe
    _decoder_layer_pipe_cls = DeepseekV2DecoderLayerPipe
    _rms_norm_pipe_cls = DeepseekV2RMSNormLayerPipe
    _base_model = DeepseekV2PretrainedModel

    _get_tensor_parallel_mappings = DeepseekV2PretrainedModel._get_tensor_parallel_mappings
    _init_weights = DeepseekV2PretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = DeepseekV2PretrainedModel._keys_to_ignore_on_load_unexpected
    transpose_weight_keys = DeepseekV2PretrainedModel.transpose_weight_keys
    _keep_in_fp32_modules = DeepseekV2PretrainedModel._keep_in_fp32_modules

    _tied_weights_keys = ["lm_head.weight"]

    _mtp_layer_pipe_cls = DeepseekV2MTPLayerPipe
