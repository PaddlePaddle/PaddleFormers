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
from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp, ScatterOp
from paddle.nn import functional as F

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.attention.utils import repeat_kv
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Glm4MoeMLP
from ...nn.moe_deepep.moe_factory import QuickAccessMoEFactory
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe, parse_args
from ...utils.log import logger
from ..masking_utils import create_causal_masks_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..moe_gate import PretrainedMoEGate
from ..moe_layer import MoEFlexTokenLayer
from .configuration import Glm4MoeConfig


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


def _apply_rotary_emb(
    x: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
) -> paddle.Tensor:
    x = x.transpose([0, 2, 1, 3])
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.transpose([0, 2, 1, 3])


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = _apply_rotary_emb(q_rot, cos, sin)
    k_embed = _apply_rotary_emb(k_rot, cos, sin)

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

        self.tensor_parallel = config.tensor_parallel_degree > 1
        self.sequence_parallel = config.sequence_parallel
        self.attention_bias = config.attention_bias
        self.fuse_attention_qkv = config.fuse_attention_qkv
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree

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
                hidden_size=config.hidden_size,
                norm_eps=config.rms_norm_eps,
                input_is_parallel=self.tensor_parallel,
            )
            self.k_norm = GeneralNorm.create(
                config=config,
                norm_type="rms_norm",
                hidden_size=config.hidden_size,
                norm_eps=config.rms_norm_eps,
                input_is_parallel=self.tensor_parallel,
            )

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
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
                bsz = hidden_states.shape[0] * self.config.tensor_parallel_degree // max_sequence_length
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
                bsz = hidden_states.shape[0] * self.config.tensor_parallel_degree // max_sequence_length
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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = paddle.cat([past_key_value[0], key_states], axis=1)
            value_states = paddle.cat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

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

        return attn_output, past_key_value


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
            config.tensor_parallel_degree = 1
        super().__init__()
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        # if sequence_parallel is True, expert Linear will call ColumnParallelLinear instead of ColumnSequenceParallelLinear
        if self.sequence_parallel and config.tensor_parallel_degree > 1:
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
    A mixed expert module containing shared experts for expert_parallel_degree > 1 with deepep mode
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
            drop_tokens=False,
        )

        hcg = fleet.get_hybrid_communicate_group()
        moe_grad_group = None
        try:
            moe_group = hcg.get_expert_parallel_group()
        except:
            moe_group = None
        expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1
        if hasattr(dist, "fleet") and dist.is_initialized() and expert_parallel_degree > 1:
            moe_group = hcg.get_expert_parallel_group()
            moe_grad_group = hcg.get_moe_sharding_parallel_group()
        if expert_parallel_degree > 1 and config.tensor_parallel_degree >= 1:
            mlp_config = deepcopy(config)
            mlp_config.tensor_parallel_degree = 1
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
        if hasattr(dist, "fleet") and dist.is_initialized() and expert_parallel_degree > 1:
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
        expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = (
                Glm4MoeMoE(config)
                if expert_parallel_degree <= 1
                else (
                    QuickAccessMoEFactory.create_from_model_name(
                        pretrained_config=config,
                        expert_class=Glm4MoeMLP,
                        gate_activation="sigmoid",
                        expert_activation="silu",
                        train_topk_method="noaux_tc",
                        inference_topk_method="noaux_tc",
                        drop_tokens=False,
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
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> paddle.Tensor:
        offload_kwargs = {}
        offload_kwargs["offload_indices"] = [0]

        has_gradient = not hidden_states.stop_gradient
        if self.config.recompute and has_gradient and self.config.recompute_granularity != "full_attn":
            attn_outputs = recompute(
                self.attn,
                hidden_states,
                past_key_value=past_key_value,
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
                past_key_value=past_key_value,
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
            seq_len = self.config.max_sequence_length // self.config.tensor_parallel_degree
            batch_size = hidden_states.shape[0] // seq_len
            assert (
                batch_size > 0
            ), f"batch_size must larger than 0, but calulate batch_size:{batch_size}, hidden_states shape:{hidden_states.shape}"
            hidden_states = hidden_states.reshape([-1, batch_size, hidden_size])
        sub_seq_len = self.config.moe_subbatch_token_num
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
            if self.config.recompute and has_gradient and self.config.recompute_granularity != "full_attn":
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
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if self.config.recompute and has_gradient and self.config.recompute_granularity == "full_attn":
            outputs = recompute(
                self.self_attn,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
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

        if use_cache:
            present_key_value = outputs[1]
            attn_outputs += (present_key_value,)

        return attn_outputs

    def post_process(
        self,
        hidden_states,
        residual,
        use_cache=False,
        present_key_value=None,
    ):
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
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
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:

        attn_outputs = self.attn(
            hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        residual = attn_outputs[1]
        present_key_value = attn_outputs[2] if use_cache else None

        hidden_states = self.mlp(hidden_states)
        outputs = self.post_process(hidden_states, residual, use_cache, present_key_value)
        return outputs


class Glm4MoePreTrainedModel(PretrainedModel):
    config: Glm4MoeConfig
    config_class = Glm4MoeConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Glm4MoeConfig, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        LAYER_COLWISE = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ]
        FUSE_LAYER_COLWISE = [
            "self_attn.qkv_proj.weight",
        ]

        LAYER_ROWWISE = ["self_attn.o_proj.weight"]

        EXPERT_LAYER_COLWISE = [
            "up_proj.weight",
            "gate_proj.weight",
        ]
        FUSE_EXPERT_LAYER_COLWISE = [
            "up_gate_proj.weight",
        ]

        EXPERT_LAYER_ROWWISE = ["down_proj.weight"]

        BIAS_KEYS = [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]
        FUSE_BIAS_KEYS = [
            "self_attn.qkv_proj.bias",
        ]

        def make_base_actions():
            actions = {
                "lm_head.weight": partial(fn, is_column=False),
                "model.embed_tokens.weight": partial(fn, is_column=False),
            }
            for layer_idx in range(config.num_hidden_layers):
                if not config.fuse_attention_qkv:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                            for k in LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                            for k in FUSE_LAYER_COLWISE
                        }
                    )

                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
                        for k in LAYER_ROWWISE
                    }
                )
                try:
                    moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
                except:
                    moe_group = None
                expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1
                # TODO: merge disable_ffn_model_parallel and expert_parallel_degree
                if expert_parallel_degree <= 1:
                    # # if disable_ffn_model_parallel is True, disable expert layer tp plan
                    # if not config.disable_ffn_model_parallel:
                    if not config.fuse_attention_ffn:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                    fn, is_column=True
                                )
                                for e in range(config.n_routed_experts)
                                for k in EXPERT_LAYER_COLWISE
                            }
                        )
                    else:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                    fn, is_column=True, is_naive_2fuse=True
                                )
                                for e in range(config.n_routed_experts)
                                for k in FUSE_EXPERT_LAYER_COLWISE
                            }
                        )
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                fn, is_column=False
                            )
                            for e in range(config.n_routed_experts)
                            for k in EXPERT_LAYER_ROWWISE
                        }
                    )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(fn, is_column=False)
                        for k in EXPERT_LAYER_ROWWISE
                    }
                )
                if not config.fuse_attention_ffn:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(fn, is_column=True)
                            for k in EXPERT_LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            )
                            for k in FUSE_EXPERT_LAYER_COLWISE
                        }
                    )

                if not config.fuse_attention_ffn:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                                fn, is_column=True
                            )
                            for k in EXPERT_LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            )
                            for k in FUSE_EXPERT_LAYER_COLWISE
                        }
                    )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                            fn, is_column=False
                        )
                        for k in EXPERT_LAYER_ROWWISE
                    }
                )
                # bias
                if config.attention_bias:
                    if not config.fuse_attention_qkv:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                                for b in BIAS_KEYS
                            }
                        )
                    else:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                                for b in FUSE_BIAS_KEYS
                            }
                        )
            return actions

        mappings = make_base_actions()
        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: Glm4MoeConfig, is_fuse=False):
        # return parameter fuse utils
        from ..conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = [
            (
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.qkv_proj.weight",
            ),
            (
                "layers.0.self_attn.q_proj.bias",
                "layers.0.self_attn.k_proj.bias",
                "layers.0.self_attn.v_proj.bias",
                "layers.0.self_attn.qkv_proj.bias",
            ),
        ]
        fuse_gate_up_keys = [
            (
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.up_gate_proj.weight",
            ),
            (
                "layers.0.mlp.experts.0.gate_proj.weight",
                "layers.0.mlp.experts.0.up_proj.weight",
                "layers.0.mlp.experts.0.up_gate_proj.weight",
            ),
            (
                "layers.0.mlp.shared_experts.gate_proj.weight",
                "layers.0.mlp.shared_experts.up_proj.weight",
                "layers.0.mlp.shared_experts.up_gate_proj.weight",
            ),
        ]
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)
        num_experts = getattr(config, "n_routed_experts", 128)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )

            if fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_gate_up_keys:
                        keys = [key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys]
                        if "experts.0." in keys[0]:
                            for j in range(num_experts):
                                experts_keys = tuple([key.replace("experts.0.", f"experts.{j}.") for key in keys])
                                final_actions[experts_keys] = fn
                        else:
                            experts_keys = tuple(keys)
                            final_actions[experts_keys] = fn

        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn,
                            split_nums=3,
                            is_qkv=True,
                            num_heads=num_heads,
                            num_key_value_heads=num_key_value_heads,
                        )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_gate_up_keys:
                        keys = [key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys]
                        if "experts.0." in keys[0]:
                            for j in range(num_experts):
                                experts_keys = tuple([key.replace("experts.0.", f"experts.{j}.") for key in keys])
                                final_actions[experts_keys] = partial(fn, split_nums=2)
                        else:
                            experts_keys = tuple(keys)
                            final_actions[experts_keys] = partial(fn, split_nums=2)
        return final_actions

    @classmethod
    def _gen_aoa_config(cls, config: Glm4MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_config = {
            "aoa_statements": [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.layers.$LAYER_ID.mlp.gate.e_score_correction_bias -> {model_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias",
                f"model.layers.$LAYER_ID.mlp.gate.weight -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            ]
        }

        # attention qkv
        if not config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        # FFN
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += (
                [
                    f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
            )
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
            ]

        return aoa_config

    # NOTE: These aoa_config items will be removed later. The subsequent AOA parsing module will automatically generate the reverse AOA based on the forward (from_pretrained) AOA.
    @classmethod
    def _gen_inv_aoa_config(cls, config: Glm4MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_statements = [
            # do cast
            f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
            # do transpose
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias -> model.layers.$LAYER_ID.mlp.gate.e_score_correction_bias",
        ]

        if not config.fuse_attention_qkv:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}, axis = 0",
            ]
            aoa_statements += [
                f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                for layer_id in range(config.num_hidden_layers)
                for x in ("q", "k", "v")
            ]

        if not config.fuse_attention_ffn:
            aoa_statements += (
                [
                    f"{model_prefix}layers.$LAYER_ID.mlp.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
            )
        else:
            aoa_statements += [
                f"{model_prefix}layers.0.mlp.up_gate_proj.weight -> model.layers.0.mlp.gate_proj.weight, model.layers.0.mlp.up_proj.weight, fused_ffn",
                "model.layers.0.mlp.gate_proj.weight^T -> model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight^T -> model.layers.0.mlp.up_proj.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight, fused_ffn",
                f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, fused_ffn",
            ]
            aoa_statements += (
                [
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                    for expert_id in range(config.n_routed_experts)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                    for expert_id in range(config.n_routed_experts)
                ]
            )
        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class Glm4MoeRotaryEmbedding(nn.Layer):
    def __init__(self, config: Glm4MoeConfig, device=None):
        super().__init__()
        self.config = config
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    @paddle.no_grad()
    def forward(self, x, position_ids):
        # NOTE: Paddle's Automatic Mixed Precision (AMP) has a default op whitelist that may automatically cast
        # certain operations (like matmul) to FP16/BF16 for performance optimization. However, in scenarios where
        # numerical stability is critical (e.g., RoPE init/compute), this conversion can lead to precision loss.
        # Disabling auto_cast here ensures the matmul operation runs in the original precision (FP32) as intended.
        with paddle.amp.auto_cast(False):
            inv_freq_expanded = (
                self.inv_freq.unsqueeze(0)
                .unsqueeze(-1)
                .cast(paddle.float32)
                .expand([position_ids.shape[0], -1, 1])
                .to(x.place)
            )
            position_ids_expanded = position_ids.unsqueeze(1).cast(paddle.float32)

            freqs = paddle.matmul(inv_freq_expanded, position_ids_expanded).transpose([0, 2, 1])
            emb = paddle.cat((freqs, freqs), axis=-1)
            cos = paddle.cos(emb) * self.attention_scaling
            sin = paddle.sin(emb) * self.attention_scaling

        return cos.cast(dtype=x.dtype), sin.cast(dtype=x.dtype)


@register_base_model
class Glm4MoeModel(Glm4MoePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.92.*", r"model\.layers\.46.*"]

    def __init__(self, config: Glm4MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.sequence_parallel = config.sequence_parallel
        self.recompute_granularity = config.recompute_granularity
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
        past_key_value: Tensor,
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
            past_key_value,
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
        past_key_values: Optional[List[paddle.Tensor]] = None,
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
        cache_length = 0
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        else:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

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
            "return_mapping": False,
        }
        causal_mask, attn_mask_startend_row_indices = create_causal_masks_and_row_indices(**mask_kwargs)

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        moelayer_use_subbatch_recompute = (
            self.config.moe_subbatch_token_num > 0 if hasattr(self.config, "moe_subbatch_token_num") else False
        )

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if moelayer_use_subbatch_recompute:
                layer_outputs = decoder_layer.subbatch_recompute_forward(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                    position_embeddings,
                )
            elif self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
                layer_outputs = self.recompute_training_full(
                    layer_module=decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            # # NOTE: clear outdate cache after it has been used for memory saving
            # past_key_value = past_key_values[idx] = None
            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class Glm4MoeForCausalLM(Glm4MoePreTrainedModel):
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
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

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
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
            max_seq_len = hidden_states.shape[0] * self.config.tensor_parallel_degree
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
            self.config.moe_subbatch_token_num > 0 if hasattr(self.config, "moe_subbatch_token_num") else False
        )
        if moelayer_use_subbatch_recompute:
            hidden_states = super().subbatch_recompute_forward(
                hidden_states,
                position_ids=position_ids_decoder,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=tuple_position_embeddings,
            )
        elif self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
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


class Glm4MoeForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Glm4MoeConfig
    _decoder_layer_cls = Glm4MoeDecoderLayer
    _decoder_layer_pipe_cls = Glm4MoeDecoderLayerPipe
    _get_tensor_parallel_mappings = Glm4MoeModel._get_tensor_parallel_mappings
    _get_fuse_or_split_param_mappings = Glm4MoeModel._get_fuse_or_split_param_mappings
    _init_weights = Glm4MoeModel._init_weights
    _keep_in_fp32_modules = Glm4MoeModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Glm4MoeModel.transpose_weight_keys
    _rotary_emb_cls = Glm4MoeRotaryEmbedding
    _gen_aoa_config = Glm4MoeForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Glm4MoeForCausalLM._gen_inv_aoa_config


__all__ = ["Glm4MoeForCausalLMPipe", "Glm4MoeModel", "Glm4MoeForCausalLM"]
