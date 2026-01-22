# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp, ScatterOp
from paddle.nn import functional as F

from paddleformers.transformers.gpt_provider import GPTModelProvider

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.attention.utils import repeat_kv
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Glm4MoeMLP
from ...nn.moe_deepep.moe_factory import QuickAccessMoEFactory
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe, parse_args
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ..moe_gate import PretrainedMoEGate
from ..moe_layer import MoEFlexTokenLayer
from .configuration import Glm4MoeConfig


@dataclass
class GLMMoEModelProvider(GPTModelProvider):
    """Base provider for GLM MoE Models."""

    moe_router_load_balancing_type: str = "seq_aux_loss"

    gated_linear_unit: bool = True

    bias_activation_fusion: bool = True

    transform_rules = {
        "dtype": "params_dtype",
    }

    # (@peiziliang) hard code
    rotary_base: float = 1000000.0
    rotary_percent: float = 0.5
    moe_shared_expert_overlap: bool = True
    moe_router_pre_softmax: bool = False
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0
    persist_layer_norm: bool = True
    moe_router_force_load_balancing: bool = True
    share_embeddings_and_output_weights: bool = False

    apply_rope_fusion: bool = True
    mtp_loss_scaling_factor: float = 0.3
    recompute_granularity: str = None
    virtual_pipeline_model_parallel_size: int = None

    rope_scaling: float = 1.0
    bias_dropout_fusion: bool = True
    router_aux_loss_coef: float = 0.001
    moe_grouped_gemm: bool = False


def eager_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    perm = [0, 2, 1, 3]  # b l h d -> b h l d
    query = paddle.transpose(x=query, perm=perm)
    key = paddle.transpose(x=key, perm=perm)
    value = paddle.transpose(x=value, perm=perm)

    attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = paddle.matmul(attn_weights, value)
    attn_output = paddle.transpose(attn_output, perm=[0, 2, 1, 3])
    attn_output = paddle.reshape(x=attn_output, shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]])

    return attn_output, attn_weights


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = paddle.cat([q_embed, q_pass], axis=-1)
    k_embed = paddle.cat([k_embed, k_pass], axis=-1)

    return q_embed, k_embed


class Glm4MoeAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Glm4MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.rope_scaling = config.rope_scaling
        self.attention_dropout = config.attention_dropout

        self.tensor_parallel = config.tensor_model_parallel_size > 1
        self.sequence_parallel = config.sequence_parallel
        self.attention_bias = config.attention_bias
        self.fuse_attention_qkv = config.fuse_attention_qkv
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

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
        q_hidden_size = self.num_attention_heads * self.head_dim

        if not self.fuse_attention_qkv:
            self.q_proj = GeneralLinear.create(
                self.hidden_size,
                q_hidden_size,
                has_bias=self.attention_bias,
                config=config,
                tp_plan="colwise",
            )
            self.k_proj = GeneralLinear.create(
                self.hidden_size,
                kv_hidden_size,
                has_bias=self.attention_bias,
                config=config,
                tp_plan="colwise",
            )
            self.v_proj = GeneralLinear.create(
                self.hidden_size,
                kv_hidden_size,
                has_bias=self.attention_bias,
                config=config,
                tp_plan="colwise",
            )
        else:
            self.qkv_proj = GeneralLinear.create(
                self.hidden_size,
                q_hidden_size + 2 * kv_hidden_size,
                has_bias=self.attention_bias,
                config=config,
                tp_plan="colwise",
            )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            self.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = GeneralNorm.create(
                config=config,
                norm_type="rms_norm",
                hidden_size=self.head_dim,
                norm_eps=config.rms_norm_eps,
                input_is_parallel=self.tensor_parallel,
            )
            self.k_norm = GeneralNorm.create(
                config=config,
                norm_type="rms_norm",
                hidden_size=self.head_dim,
                norm_eps=config.rms_norm_eps,
                input_is_parallel=self.tensor_parallel,
            )

    def forward(
        self,
        hidden_states,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:

        if not self.fuse_attention_qkv:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.sequence_parallel:
                max_sequence_length = self.config.max_sequence_length
                bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
                q_len = max_sequence_length
            else:
                bsz, q_len, _ = hidden_states.shape
            query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
            key_states = key_states.reshape([bsz, q_len, -1, self.head_dim])
            value_states = value_states.reshape([bsz, q_len, -1, self.head_dim])
        else:
            mix_layer = self.qkv_proj(hidden_states)
            if self.sequence_parallel:
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
            if self.gqa_or_mqa:
                query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # b l h d -> b h l d
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.config.get("attention_dropout", 0.0) if self.training else 0.0,
            scaling=self.scaling,
        )

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Glm4MoeTopkFlexRouter(PretrainedMoEGate):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super().__init__(config, num_experts, expert_hidden_size, **kwargs)
        self.config = config

        self.weight = paddle.create_parameter(
            shape=[num_experts, expert_hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(),
        )

        self.register_buffer("e_score_correction_bias", paddle.zeros((num_experts,), dtype=paddle.float32))
        self.expert_usage = paddle.zeros(
            shape=[num_experts],
            dtype=paddle.int64,
        )
        self.expert_usage.stop_gradient = True

        # weight and e_score_correction_bias do not need to be cast to low precision
        self._cast_to_low_precision = False

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (_type_): [batch_size * seq_len, hidden_size]
        """
        # compute gating score
        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.cast(self.weight.dtype)
            logits = F.linear(hidden_states.cast("float32"), self.weight.cast("float32").t())
            scores = self.gate_score_func(logits=logits)
            scores = scores.cast(paddle.float32)

        scores, routing_map, exp_counts, l_aux, l_zloss = self.topkgating_nodrop(scores)
        with paddle.no_grad():
            self.expert_usage += exp_counts
        return scores, routing_map, l_aux, l_zloss


class Glm4MoeTopkRouter(nn.Layer):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = paddle.create_parameter(
            shape=[self.n_routed_experts, config.hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(),
        )

        self.register_buffer("e_score_correction_bias", paddle.zeros((self.n_routed_experts,), dtype=paddle.float32))

        # weight and e_score_correction_bias do not need to be cast to low precision
        self._cast_to_low_precision = False

    @paddle.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.reshape([-1, self.n_routed_experts]) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.reshape([-1, self.n_group, self.n_routed_experts // self.n_group])
            .topk(2, axis=-1)[0]
            .sum(axis=-1)
        )
        group_idx = paddle.topk(group_scores, k=self.topk_group, axis=-1, sorted=False)[1]
        group_mask = paddle.zeros_like(group_scores)
        group_mask = paddle.put_along_axis(group_mask, group_idx, 1, axis=1, broadcast=False)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand([-1, self.n_group, self.n_routed_experts // self.n_group])
            .reshape([-1, self.n_routed_experts])
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.cast("bool"), 0.0)
        topk_indices = paddle.topk(scores_for_choice, k=self.top_k, axis=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape([-1, self.config.hidden_size])
        router_logits = F.linear(hidden_states.cast("float32"), self.weight.cast("float32").t())
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = paddle.take_along_axis(scores, topk_indices, axis=1, broadcast=False)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(axis=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class Glm4MoeMoE(nn.Layer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        if getattr(config, "disable_ffn_model_parallel", False):
            config = deepcopy(config)
            config.tensor_model_parallel_size = 1
        super().__init__()
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        # if sequence_parallel is True, expert Linear will call ColumnParallelLinear instead of ColumnSequenceParallelLinear
        if self.sequence_parallel and config.tensor_model_parallel_size > 1:
            config = deepcopy(config)
            config.sequence_parallel = False
        self.experts = nn.LayerList(
            [
                Glm4MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size, fuse_up_gate=config.fuse_attention_ffn
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = Glm4MoeTopkRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            fuse_up_gate=config.fuse_attention_ffn,
        )

    def moe(self, hidden_states: paddle.Tensor, topk_indices: paddle.Tensor, topk_weights: paddle.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = paddle.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = paddle.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = paddle.transpose(expert_mask, perm=[2, 0, 1])

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

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.cast(hidden_states.dtype)

    def forward(self, hidden_states):
        if self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights)
        hidden_states = paddle.reshape(hidden_states, orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        if self.sequence_parallel:
            hidden_states = ScatterOp.apply(hidden_states)
        return hidden_states


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


class Glm4MoeFlexMoE(MoEFlexTokenLayer):
    """
    A mixed expert module containing shared experts for expert_model_parallel_size > 1 with deepep mode
    """

    def __init__(self, config):
        self.config = config
        gate = Glm4MoeTopkFlexRouter(
            config=config,
            num_experts=config.n_routed_experts,
            expert_hidden_size=config.hidden_size,
            top_k=config.num_experts_per_tok,
            topk_method="noaux_tc",
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
        )

        hcg = fleet.get_hybrid_communicate_group()
        moe_grad_group = None
        try:
            moe_group = hcg.get_expert_parallel_group()
        except:
            moe_group = None
        expert_model_parallel_size = dist.get_world_size(moe_group) if moe_group is not None else 1
        if hasattr(dist, "fleet") and dist.is_initialized() and expert_model_parallel_size > 1:
            moe_group = hcg.get_expert_parallel_group()
            moe_grad_group = hcg.get_moe_sharding_parallel_group()
        if expert_model_parallel_size > 1 and config.tensor_model_parallel_size >= 1:
            mlp_config = deepcopy(config)
            mlp_config.tensor_model_parallel_size = 1
        super().__init__(
            config=config,
            moe_num_experts=config.n_routed_experts,
            expert_class=Glm4MoeMLP,
            expert_kwargs={
                "config": mlp_config,
                "intermediate_size": mlp_config.moe_intermediate_size,
                "fuse_up_gate": config.fuse_attention_ffn,
            },
            gate=gate,
            moe_group=moe_group,
        )
        if hasattr(dist, "fleet") and dist.is_initialized() and expert_model_parallel_size > 1:
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

        self.shared_experts = Glm4MoeMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            fuse_up_gate=config.fuse_attention_ffn,
        )

    def forward(self, hidden_states):
        final_hidden_states, l_aux, _ = super().forward(hidden_states)
        if self.training and self.config.aux_loss_alpha > 0.0:
            l_aux = l_aux * self.config.aux_loss_alpha
            final_hidden_states = AddAuxiliaryLoss.apply(final_hidden_states, l_aux)
        final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)
        return final_hidden_states


class Glm4MoeDecoderLayer(nn.Layer):
    def __init__(self, config: Glm4MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Glm4MoeAttention(config=config, layer_idx=layer_idx)

        try:
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        except:
            moe_group = None
        expert_model_parallel_size = dist.get_world_size(moe_group) if moe_group is not None else 1
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = (
                Glm4MoeMoE(config)
                if expert_model_parallel_size <= 1
                else (
                    QuickAccessMoEFactory.create_from_model_name(
                        pretrained_config=config,
                        expert_class=Glm4MoeMLP,
                        gate_activation="sigmoid",
                        expert_activation="silu",
                        train_topk_method="noaux_tc",
                        inference_topk_method="noaux_tc",
                        transpose_gate_weight=True,
                    )
                    if config.use_unified_moe
                    else Glm4MoeFlexMoE(config)
                )
            )
        else:
            self.mlp = Glm4MoeMLP(config, fuse_up_gate=config.fuse_attention_ffn)

        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def subbatch_recompute_forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> paddle.Tensor:
        offload_kwargs = {}
        offload_kwargs["offload_indices"] = [0]

        has_gradient = not hidden_states.stop_gradient
        if (
            self.config.recompute_granularity is not None
            and self.config.recompute_modules is not None
            and "core_attn" in self.config.recompute_modules
            and has_gradient
        ):
            attn_outputs = recompute(
                self.attn,
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **offload_kwargs,
            )
        else:
            attn_outputs = self.attn(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

        hidden_states = attn_outputs[0]
        residual = attn_outputs[1]
        present_key_value = attn_outputs[2] if use_cache else None

        hidden_size = hidden_states.shape[-1]
        if self.config.sequence_parallel:
            # hidden_states shape:[b*s,h]
            seq_len = self.config.max_sequence_length // self.config.tensor_model_parallel_size
            batch_size = hidden_states.shape[0] // seq_len
            assert (
                batch_size > 0
            ), f"batch_size must larger than 0, but calulate batch_size:{batch_size}, hidden_states shape:{hidden_states.shape}"
            hidden_states = hidden_states.reshape([-1, batch_size, hidden_size])
        sub_seq_len = self.config.moe_subbatch_token_num_before_dispatch
        seq_axis = 0 if self.config.sequence_parallel else 1
        seq_len = hidden_states.shape[seq_axis]
        assert seq_len % sub_seq_len == 0
        num_chunks = seq_len // sub_seq_len
        split_list = [sub_seq_len] * num_chunks
        input_list = paddle.split(hidden_states, split_list, axis=seq_axis)
        output_list = []

        for chunk in input_list:
            if self.config.sequence_parallel:
                chunk = chunk.reshape([-1, hidden_size])
            has_gradient = not chunk.stop_gradient
            if (
                self.config.recompute_granularity is not None
                and self.config.recompute_modules is not None
                and "mlp" in self.config.recompute_modules
                and has_gradient
            ):
                out = recompute(
                    self.mlp.forward,
                    chunk,
                    **offload_kwargs,
                )
            else:
                out = self.mlp.forward(chunk)
            output_list.append(out)
        hidden_states = paddle.concat(output_list, axis=seq_axis)
        outputs = recompute(
            self.post_process,
            hidden_states,
            residual,
            use_cache,
            present_key_value,
            **offload_kwargs,
        )
        return outputs

    def attn(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "full_attn" in self.config.recompute_modules
            and has_gradient
        ):
            outputs = recompute(
                self.self_attn,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
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
                past_key_values=past_key_values,
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

        if use_cache:
            present_key_value = outputs[1]
            attn_outputs += (present_key_value,)

        return attn_outputs

    def post_process(
        self,
        hidden_states,
        residual,
        use_cache=False,
    ):
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:

        attn_outputs = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        residual = attn_outputs[1]

        hidden_states = self.mlp(hidden_states)
        outputs = self.post_process(hidden_states, residual, use_cache)
        return outputs


class Glm4MoePreTrainedModel(PretrainedModel):
    config: Glm4MoeConfig
    config_class = Glm4MoeConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: Glm4MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        is_fleet = getattr(cls, "is_fleet", False)
        using_sonic_moe = config.using_sonic_moe
        aoa_config = {
            "aoa_statements": [
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }
        if is_fleet:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
            ]
            if config.tie_word_embeddings:
                aoa_config["aoa_statements"] += [f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"]
            else:
                aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}lm_head.weight"]
        else:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            ]

        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )

        # layer 0
        aoa_config["aoa_statements"] += [
            f"model.layers.0.mlp.down_proj.weight^T -> {model_prefix}layers.{num_head_empty_layers}.mlp.down_proj.weight"
        ]
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += [
                f"model.layers.0.mlp.gate_proj.weight^T -> {model_prefix}layers.{num_head_empty_layers}.mlp.gate_proj.weight",
                f"model.layers.0.mlp.up_proj.weight^T -> {model_prefix}layers.{num_head_empty_layers}.mlp.up_proj.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.0.mlp.gate_proj.weight^T, model.layers.0.mlp.up_proj.weight^T -> {model_prefix}layers.{num_head_empty_layers}.mlp.up_gate_proj.weight, fused_ffn",
            ]

        # layer0 - layer_num_hidden_layers
        for layer_idx in reversed(range(0, num_hidden_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            aoa_config["aoa_statements"] += [
                f"{prefix}.input_layernorm.weight -> {prefix_offset}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight -> {prefix_offset}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.o_proj.weight^T -> {prefix_offset}.self_attn.o_proj.weight",
            ]
            # attention qkv
            if not config.fuse_attention_qkv:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.{x}_proj.weight^T -> {prefix_offset}.self_attn.{x}_proj.weight"
                    for x in ("q", "k", "v")
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.self_attn.q_proj.weight^T, {prefix}.self_attn.k_proj.weight^T, {prefix}.self_attn.v_proj.weight^T -> {prefix_offset}.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
                ]
                if config.attention_bias:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.self_attn.q_proj.bias, {prefix}.self_attn.k_proj.bias, {prefix}.self_attn.v_proj.bias -> {prefix_offset}.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                    ]
        # layer1 - layer_num_hidden_layers
        for layer_idx in reversed(range(1, num_hidden_layers)):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix = f"model.layers.{layer_idx}"
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            aoa_config["aoa_statements"] += [
                f"{prefix}.mlp.gate.e_score_correction_bias -> {prefix_offset}.mlp.gate.e_score_correction_bias",
                f"{prefix}.mlp.gate.weight -> {prefix_offset}.mlp.gate.weight, dtype='float32'",
                f"{prefix}.mlp.shared_experts.down_proj.weight^T -> {prefix_offset}.mlp.shared_experts.down_proj.weight",
            ]
            if using_sonic_moe:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.experts.$EXPERT_ID.down_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight",
                ]

            # FFN
            if not config.fuse_attention_ffn:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.shared_experts.{p}_proj.weight^T -> {prefix_offset}.mlp.shared_experts.{p}_proj.weight"
                    for p in ("gate", "up")
                ] + [
                    f"{prefix}.mlp.experts.$EXPERT_ID.{p}_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"{prefix}.mlp.shared_experts.gate_proj.weight^T, {prefix}.mlp.shared_experts.up_proj.weight^T -> {prefix_offset}.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                ]
                if is_fleet:
                    if using_sonic_moe:
                        aoa_config["aoa_statements"] += [
                            f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=0",
                        ]
                    else:
                        aoa_config["aoa_statements"] += [
                            f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
                        ]

                else:
                    aoa_config["aoa_statements"] += [
                        f"{prefix}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, {prefix}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {prefix_offset}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
                    ]

            if is_fleet and (config.moe_grouped_gemm or using_sonic_moe):
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(config.n_routed_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_config["aoa_statements"] += [
                    f"{group_gemm1} -> {prefix_offset}.mlp.grouped_gemm_experts.weight1, axis=0"
                    f"{group_gemm2} -> {prefix_offset}.mlp.grouped_gemm_experts.weight2, axis=0"
                ]

        return aoa_config

    # NOTE: These aoa_config items will be removed later. The subsequent AOA parsing module will automatically generate the reverse AOA based on the forward (from_pretrained) AOA.
    @classmethod
    def _gen_inv_aoa_config(cls, config: Glm4MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        using_sonic_moe = config.using_sonic_moe
        is_fleet = getattr(cls, "is_fleet", False)
        aoa_statements = [
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if is_fleet:
            aoa_statements += [
                "model.embedding.embed_tokens.weight -> model.embed_tokens.weight",
            ]
            if config.tie_word_embeddings:
                aoa_statements += [f"{model_prefix}lm_head.weight -> _"]
            else:
                aoa_statements += [f"{model_prefix}lm_head.weight -> lm_head.weight"]
        else:
            aoa_statements += [
                f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            ]
        num_hidden_layers = config.num_hidden_layers
        num_head_empty_layers = (
            config.num_empty_layers_add_in_head
            if hasattr(config, "num_empty_layers_add_in_head") and config.num_empty_layers_add_in_head
            else 0
        )

        # layer 0
        aoa_statements += [
            f"{model_prefix}layers.{num_head_empty_layers}.mlp.down_proj.weight^T -> model.layers.0.mlp.down_proj.weight",
        ]
        if not config.fuse_attention_ffn:
            aoa_statements += [
                f"{model_prefix}layers.{num_head_empty_layers}.mlp.gate_proj.weight^T -> model.layers.0.mlp.gate_proj.weight",
                f"{model_prefix}layers.{num_head_empty_layers}.mlp.up_proj.weight^T -> model.layers.0.mlp.up_proj.weight",
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.{num_head_empty_layers}.mlp.up_gate_proj.weight -> model.layers.{num_head_empty_layers}.mlp.gate_proj.weight, model.layers.{num_head_empty_layers}.mlp.up_proj.weight, fused_ffn",
                f"model.layers.{num_head_empty_layers}.mlp.gate_proj.weight^T -> model.layers.0.mlp.gate_proj.weight",
                f"model.layers.{num_head_empty_layers}.mlp.up_proj.weight^T -> model.layers.0.mlp.up_proj.weight",
            ]

        # layer 0 -> layer num_hidden_layers-1
        for layer_idx in range(0, num_hidden_layers):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            prefix = f"model.layers.{layer_idx}"

            aoa_statements += [
                f"{prefix_offset}.input_layernorm.weight -> {prefix}.input_layernorm.weight",
                f"{prefix_offset}.post_attention_layernorm.weight -> {prefix}.post_attention_layernorm.weight",
                f"{prefix_offset}.self_attn.o_proj.weight^T -> {prefix}.self_attn.o_proj.weight",
            ]
            if not config.fuse_attention_qkv:
                aoa_statements += [
                    f"{prefix_offset}.self_attn.{x}_proj.weight^T -> {prefix}.self_attn.{x}_proj.weight"
                    for x in ("q", "k", "v")
                ]
            else:
                aoa_statements += [
                    f"{prefix_offset}.self_attn.qkv_proj.weight -> {prefix}.self_attn.q_proj.weight, {prefix}.self_attn.k_proj.weight, {prefix}.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
                ]
                aoa_statements += [
                    f"{prefix}.self_attn.{x}_proj.weight^T -> {prefix}.self_attn.{x}_proj.weight"
                    for x in ("q", "k", "v")
                ]
                if config.attention_bias:
                    aoa_statements += [
                        f"{prefix_offset}.self_attn.qkv_proj.bias -> {prefix}.self_attn.q_proj.bias, {prefix}.self_attn.k_proj.bias, {prefix}.self_attn.v_proj.bias , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}, axis = 0",
                    ]

        # layer 1 -> layer num_hidden_layers-1
        for layer_idx in range(1, num_hidden_layers):
            layer_idx_offset = layer_idx + num_head_empty_layers
            prefix_offset = f"{model_prefix}layers.{layer_idx_offset}"
            prefix = f"model.layers.{layer_idx}"

            if is_fleet and (config.moe_grouped_gemm or using_sonic_moe):
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(config.n_routed_experts):
                    ep_weight1.append(f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight")
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_statements += [
                    f"{prefix_offset}.mlp.grouped_gemm_experts.weight1 -> {group_gemm1}, axis=0"
                    f"{prefix_offset}.mlp.grouped_gemm_experts.weight2 -> {group_gemm2}, axis=0"
                ]

            aoa_statements += [
                # do cast
                f"{prefix_offset}.mlp.gate.weight -> {prefix}.mlp.gate.weight, dtype='bfloat16'",
                # do transpose
                f"{prefix_offset}.mlp.gate.e_score_correction_bias -> {prefix}.mlp.gate.e_score_correction_bias",
                f"{prefix_offset}.mlp.shared_experts.down_proj.weight^T -> {prefix}.mlp.shared_experts.down_proj.weight",
            ]

            if not config.fuse_attention_ffn:
                aoa_statements += (
                    [
                        f"{prefix_offset}.mlp.shared_experts.{y}_proj.weight^T -> {prefix}.mlp.shared_experts.{y}_proj.weight"
                        for y in ("gate", "up")
                    ]
                    + [
                        f"{prefix_offset}.mlp.experts.$EXPERT_ID.{y}_proj.weight^T -> {prefix}.mlp.experts.$EXPERT_ID.{y}_proj.weight"
                        for y in ("gate", "up")
                    ]
                    + [
                        f"{prefix_offset}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {prefix}.mlp.experts.$EXPERT_ID.down_proj.weight"
                    ]
                )
            else:
                aoa_statements += [
                    f"{prefix_offset}.mlp.shared_experts.up_gate_proj.weight -> {prefix_offset}.mlp.shared_experts.gate_proj.weight, {prefix_offset}.mlp.shared_experts.up_proj.weight, fused_ffn",
                    f"{prefix_offset}.mlp.shared_experts.gate_proj.weight^T -> {prefix}.mlp.shared_experts.gate_proj.weight",
                    f"{prefix_offset}.mlp.shared_experts.up_proj.weight^T -> {prefix}.mlp.shared_experts.up_proj.weight",
                ]
                if is_fleet:
                    if using_sonic_moe:
                        aoa_statements += [
                            f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> {prefix_offset}.mlp.experts.{expert_id}.gate_proj.weight, {prefix_offset}.mlp.experts.{expert_id}.up_proj.weight, axis=0"
                            for expert_id in range(config.n_routed_experts)
                        ]
                    else:
                        aoa_statements += [
                            f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> {prefix_offset}.mlp.experts.{expert_id}.gate_proj.weight, {prefix_offset}.mlp.experts.{expert_id}.up_proj.weight, axis=1"
                            for expert_id in range(config.n_routed_experts)
                        ]
                else:
                    aoa_statements += [
                        f"{prefix_offset}.mlp.experts.{expert_id}.up_gate_proj.weight -> {prefix_offset}.mlp.experts.{expert_id}.gate_proj.weight, {prefix_offset}.mlp.experts.{expert_id}.up_proj.weight, fused_ffn"
                        for expert_id in range(config.n_routed_experts)
                    ]
                if not using_sonic_moe:
                    aoa_statements += (
                        [
                            f"{prefix_offset}.mlp.experts.{expert_id}.down_proj.weight^T -> {prefix}.mlp.experts.{expert_id}.down_proj.weight"
                            for expert_id in range(config.n_routed_experts)
                        ]
                        + [
                            f"{prefix_offset}.mlp.experts.{expert_id}.gate_proj.weight^T -> {prefix}.mlp.experts.{expert_id}.gate_proj.weight"
                            for expert_id in range(config.n_routed_experts)
                        ]
                        + [
                            f"{prefix_offset}.mlp.experts.{expert_id}.up_proj.weight^T -> {prefix}.mlp.experts.{expert_id}.up_proj.weight"
                            for expert_id in range(config.n_routed_experts)
                        ]
                    )

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class Glm4MoeRotaryEmbedding(nn.Layer):
    def __init__(self, config: Glm4MoeConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        rope_parameters = self.config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))

        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Glm4MoeConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`PreTrainedConfig`]):
                The model configuration.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @paddle.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@register_base_model
class Glm4MoeModel(Glm4MoePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.92.*", r"model\.layers\.46.*"]

    def __init__(self, config: Glm4MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.sequence_parallel = config.sequence_parallel
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )

        self.layers = nn.LayerList(
            [Glm4MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = Glm4MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Tensor,
        past_key_values: Cache,
        use_cache: bool,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices=None,
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
            past_key_values,
            use_cache,
            position_embeddings,
            attn_mask_startend_row_indices,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_length_with_past += cache_length

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

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

        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        moelayer_use_subbatch_recompute = (
            self.config.moe_subbatch_token_num_before_dispatch > 0
            if hasattr(self.config, "moe_subbatch_token_num_before_dispatch")
            else False
        )

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            has_gradient = not hidden_states.stop_gradient
            if moelayer_use_subbatch_recompute:
                layer_outputs = decoder_layer.subbatch_recompute_forward(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    past_key_values,
                    use_cache,
                    attn_mask_startend_row_indices,
                    position_embeddings,
                )
            elif (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training_full(
                    layer_module=decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Glm4MoeForCausalLM(Glm4MoePreTrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class Glm4MoeForCausalLMDecapitated(Glm4MoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = Glm4MoeModel(self.config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        loss_mask: Optional[paddle.Tensor] = None,
    ):
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

        outputs = self.model(
            input_ids=input_ids,  # [bs, seq_len]
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

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


class Glm4MoeDecoderLayerPipe(Glm4MoeDecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, position_embeddings, _ = parse_args(args)

        max_seq_len = hidden_states.shape[1]
        if self.config.sequence_parallel:
            # hidden_states shape:[b*s,h]
            max_seq_len = hidden_states.shape[0] * self.config.tensor_model_parallel_size
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

        position_ids_decoder = None
        if position_ids is not None:
            position_ids_decoder = position_ids[:, :max_seq_len]

        if position_embeddings is not None:
            position_embeddings = position_embeddings[..., :max_seq_len, :]
            tuple_position_embeddings = (position_embeddings[0], position_embeddings[1])
        else:
            tuple_position_embeddings = None

        has_gradient = not hidden_states.stop_gradient
        moelayer_use_subbatch_recompute = (
            self.config.moe_subbatch_token_num_before_dispatch > 0
            if hasattr(self.config, "moe_subbatch_token_num_before_dispatch")
            else False
        )
        if moelayer_use_subbatch_recompute:
            hidden_states = super().subbatch_recompute_forward(
                hidden_states,
                position_ids=position_ids_decoder,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=tuple_position_embeddings,
            )
        elif (
            self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and self.config.recompute_num_layers == 1
            and has_gradient
        ):
            hidden_states = recompute(
                super().forward,
                hidden_states,
                position_ids=position_ids_decoder,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=tuple_position_embeddings,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states = super().forward(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=tuple_position_embeddings,
            )

        if isinstance(hidden_states, paddle.Tensor):
            ret = (hidden_states,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if position_embeddings is not None:
            ret += (position_embeddings.clone(),)
        if len(ret) == 1:
            (ret,) = ret
        return ret


class Glm4MoeForCausalLMPipe(Glm4MoePreTrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = GLMMoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        if not hasattr(config, "architectures"):
            config.architectures = [cls.__name__.replace("Pipe", "")]
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class Glm4MoeForCausalLMPipeDecapitated(GeneralModelForCausalLMPipe):
    config_class = Glm4MoeConfig
    _decoder_layer_cls = Glm4MoeDecoderLayer
    _decoder_layer_pipe_cls = Glm4MoeDecoderLayerPipe
    _init_weights = Glm4MoeModel._init_weights
    _keep_in_fp32_modules = Glm4MoeModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Glm4MoeModel.transpose_weight_keys
    _rotary_emb_cls = Glm4MoeRotaryEmbedding
    _gen_aoa_config = Glm4MoeForCausalLMDecapitated._gen_aoa_config
    _gen_inv_aoa_config = Glm4MoeForCausalLMDecapitated._gen_inv_aoa_config


__all__ = [
    "Glm4MoeForCausalLMPipe",
    "Glm4MoeModel",
    "Glm4MoeForCausalLM",
    "Glm4MoeForCausalLMPipeDecapitated",
    "Glm4MoeForCausalLMDecapitated",
]
