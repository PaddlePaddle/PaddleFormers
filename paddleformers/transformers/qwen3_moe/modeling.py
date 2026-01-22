# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen3Moe model."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp, ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.moe_deepep.moe_factory import QuickAccessMoEFactory
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..gpt_provider import GPTModelProvider
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast
from ..model_utils import PretrainedModel, dtype_guard, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ..moe_gate import PretrainedMoEGate
from .configuration import Qwen3MoeConfig


@dataclass
class Qwen3MoEModelProvider(GPTModelProvider):
    """Base provider for Qwen3 MoE Models."""

    moe_router_load_balancing_type: str = "aux_loss"

    gated_linear_unit: bool = True

    bias_activation_fusion: bool = True

    transform_rules = {
        "tensor_parallel_degree": "tensor_model_parallel_size",
        "pipeline_parallel_degree": "pipeline_model_parallel_size",
        "context_parallel_degree": "context_parallel_size",
        "expert_parallel_degree": "expert_model_parallel_size",
        "dtype": "params_dtype",
        "num_experts": "n_routed_experts",
    }

    rotary_base: float = 1000000.0
    moe_router_pre_softmax: bool = False
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = False
    moe_router_bias_update_rate: float = 0
    persist_layer_norm: bool = True
    moe_router_force_load_balancing: bool = False
    share_embeddings_and_output_weights: bool = False

    apply_rope_fusion: bool = True
    recompute_granularity: str = None
    virtual_pipeline_model_parallel_size: int = None

    rope_scaling: float = 1.0
    bias_dropout_fusion: bool = True
    router_aux_loss_coef: float = 0.001
    moe_grouped_gemm: bool = True

    n_shared_experts: int = 0

    use_qk_norm: bool = True


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


class Qwen3MoeAttention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads

        self.tensor_parallel = config.tensor_model_parallel_size > 1
        self.sequence_parallel = config.sequence_parallel
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
        q_hidden_size = self.config.num_attention_heads * self.head_dim

        if not self.fuse_attention_qkv:
            self.q_proj = GeneralLinear.create(
                config.hidden_size,
                q_hidden_size,
                has_bias=config.attention_bias,
                config=config,
                tp_plan="colwise",
            )
            self.k_proj = GeneralLinear.create(
                config.hidden_size,
                kv_hidden_size,
                has_bias=config.attention_bias,
                config=config,
                tp_plan="colwise",
            )
            self.v_proj = GeneralLinear.create(
                config.hidden_size,
                kv_hidden_size,
                has_bias=config.attention_bias,
                config=config,
                tp_plan="colwise",
            )
        else:
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
            has_bias=config.attention_bias,
            config=config,
            tp_plan="rowwise",
        )
        self.q_norm = GeneralNorm.create(
            config,
            norm_type="rms_norm",
            hidden_size=self.head_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=self.tensor_parallel,
        )  # unlike olmo, only on the head dim!
        self.k_norm = GeneralNorm.create(
            config,
            norm_type="rms_norm",
            hidden_size=self.head_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=self.tensor_parallel,
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if not self.fuse_attention_qkv:
            # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.sequence_parallel:
                max_sequence_length = self.config.max_sequence_length
                bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
                q_len = max_sequence_length
            else:
                bsz, q_len, _ = hidden_states.shape
            # Add qk norm for Qwen3MoE model.
            query_states = self.q_norm(query_states.reshape([bsz, q_len, -1, self.head_dim]))
            key_states = self.k_norm(key_states.reshape([bsz, q_len, -1, self.head_dim]))
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
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # [bs, seq_len, num_head, head_dim] -> [bs, num_head, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # [bs, num_head, seq_len, head_dim]
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
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen3MoeMLP(MLP):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size=None, **kwargs):
        super().__init__(config, intermediate_size=intermediate_size, **kwargs)


class Qwen3MoeGate(PretrainedMoEGate):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super().__init__(config, num_experts, expert_hidden_size, **kwargs)
        # [hidden_size, n_expert]
        self.weight = paddle.create_parameter(
            shape=[expert_hidden_size, num_experts],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (_type_): [batch_size * seq_len, hidden_size]
        """
        # compute gating score
        logits = F.linear(hidden_states, self.weight, None)

        with paddle.amp.auto_cast(False):
            scores = self.gate_score_func(logits=logits)
            scores = scores.cast(paddle.get_default_dtype())

        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = self.topkgating(scores)

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss


class Qwen3MoeSparseMoeBlock(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and config.tensor_model_parallel_size > 1:
            config = copy.deepcopy(config)
            config.sequence_parallel = False

        # gating
        with dtype_guard("float32"):
            self.gate = GeneralLinear.create(
                config.hidden_size, config.num_experts, has_bias=False, linear_type="default"
            )
        self.experts = nn.LayerList(
            [
                Qwen3MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size, fuse_up_gate=config.fuse_attention_ffn
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """ """
        if self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        orig_shape = hidden_states.shape

        hidden_states = hidden_states.view([-1, hidden_states.shape[-1]])
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, axis=1, dtype=paddle.float32)
        # (batch * sequence_length, topk)
        routing_weights, selected_experts = paddle.topk(routing_weights, self.top_k, axis=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(axis=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = paddle.zeros(
            (hidden_states.shape[-2], hidden_states.shape[-1]), dtype=hidden_states.dtype
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).transpose([2, 1, 0])
        # [num_experts, topk, bs*seq]
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(axis=-1)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                if self.training and paddle.is_grad_enabled():
                    fake_top_x = paddle.zeros(1, dtype=paddle.int64)
                    fakse_current_state = hidden_states[fake_top_x, None].reshape([-1, hidden_states.shape[-1]])
                    fake_state = expert_layer(fakse_current_state * 0)
                    final_hidden_states.index_add_(index=fake_top_x, axis=0, value=fake_state.to(hidden_states.dtype))
                else:
                    continue
            else:
                current_state = hidden_states[idx, None].reshape([-1, hidden_states.shape[-1]])
                current_hidden_states = expert_layer(current_state) * routing_weights[idx, top_x].unsqueeze(-1)
                final_hidden_states.index_add_(
                    index=idx.reshape([-1]), axis=0, value=current_hidden_states.to(hidden_states.dtype)
                )

        final_hidden_states = paddle.reshape(final_hidden_states, orig_shape)

        if self.sequence_parallel:
            final_hidden_states = ScatterOp.apply(final_hidden_states)

        return final_hidden_states, router_logits


class Qwen3MoeDecoderLayer(nn.Layer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        try:
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        except:
            moe_group = None
        expert_model_parallel_size = dist.get_world_size(moe_group) if moe_group is not None else 1
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = (
                QuickAccessMoEFactory.create_from_model_name(
                    pretrained_config=config,
                    expert_class=Qwen3MoeMLP,
                    gate_activation="softmax",
                    expert_activation="silu",
                    train_topk_method="greedy",
                    inference_topk_method="greedy",
                    transpose_gate_weight=False,
                )
                if expert_model_parallel_size > 1
                else Qwen3MoeSparseMoeBlock(config)
            )
        else:
            # num_experts == 0 or this layer is not sparse layer
            self.mlp = Qwen3MoeMLP(config, fuse_up_gate=config.fuse_attention_ffn)

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
            if isinstance(out, tuple):
                out = out[0]
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
        """
        Args:
            hidden_states (`paddle.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`tuple[paddle.FloatTensor, paddle.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
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
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        outputs = self.post_process(hidden_states, residual, use_cache)
        return outputs


class Qwen3MoeRotaryEmbedding(nn.Layer):
    def __init__(self, config: Qwen3MoeConfig):
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
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Qwen3MoeConfig] = None,
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
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

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


class Qwen3MoePretrainedModel(PretrainedModel):
    config_class = Qwen3MoeConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen3MoeConfig):
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        model_prefix = "" if cls == cls.base_model_class else "model."
        using_sonic_moe = config.using_sonic_moe
        aoa_config = {
            "aoa_statements": [
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        if using_sonic_moe:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]

        if getattr(cls, "is_fleet", False):
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
                f"model.layers.$LAYER_ID.mlp.gate.weight -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.layers.$LAYER_ID.self_attn.q_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_layernorm.weight",
                f"model.layers.$LAYER_ID.self_attn.k_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_layernorm.weight",
                f"lm_head.weight -> {model_prefix}lm_head.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.layers.$LAYER_ID.mlp.gate.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.layers.$LAYER_ID.self_attn.q_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.layers.$LAYER_ID.self_attn.k_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            ]

        # attention qkv
        if not config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            ]
            if config.attention_bias:
                aoa_config["aoa_statements"] += [
                    f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        # FFN
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight"
                for p in ("gate", "up")
            ]
        else:
            if getattr(cls, "is_fleet", False):
                if using_sonic_moe:
                    aoa_config["aoa_statements"] += [
                        f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=0",
                    ]
                else:
                    aoa_config["aoa_statements"] += [
                        f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
                    ]

            else:
                aoa_config["aoa_statements"] += [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
                ]

        if getattr(cls, "is_fleet", False) and (config.moe_grouped_gemm or using_sonic_moe):
            for layer_idx in range(0, config.num_hidden_layers):
                src_prefix = f"model.layers.{layer_idx}"
                tgt_prefix = f"{model_prefix}layers.{layer_idx}"
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(num_experts):
                    ep_weight1.append(f"{src_prefix}.mlp.experts.{expert_id}.up_gate_proj.weight")
                    ep_weight2.append(f"{src_prefix}.mlp.experts.{expert_id}.down_proj.weight")
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_config["aoa_statements"] += [
                    f"{group_gemm1} -> {tgt_prefix}.mlp.grouped_gemm_experts.weight1, axis=0"
                    f"{group_gemm2} -> {tgt_prefix}.mlp.grouped_gemm_experts.weight2, axis=0"
                ]

        # lm_head
        if config.tie_word_embeddings:
            aoa_config["aoa_statements"] += ["model.embed_tokens.weight -> lm_head.weight"]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen3MoeConfig):
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        model_prefix = "" if cls == cls.base_model_class else "model."
        using_sonic_moe = config.using_sonic_moe
        aoa_statements = [
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if getattr(cls, "is_fleet", False):
            aoa_statements += [
                f"{model_prefix}embedding.embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_layernorm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_layernorm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
                f"{model_prefix}lm_head.weight -> lm_head.weight",
            ]
        else:
            aoa_statements += [
                f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight^T -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
            ]

        if not config.fuse_attention_qkv:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
            ]
            for layer_id in range(config.num_hidden_layers):
                for x in ("q", "k", "v"):
                    aoa_statements += [
                        f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                    ]
            if config.attention_bias:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        if not config.fuse_attention_ffn:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight"
                for y in ("gate", "up")
            ]
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]
        else:
            if getattr(cls, "is_fleet", False) and (config.moe_grouped_gemm or using_sonic_moe):
                for layer_id in range(config.num_hidden_layers):
                    ep_weight1 = []
                    ep_weight2 = []
                    for expert_id in range(num_experts):
                        ep_weight1.append(
                            f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight"
                        )
                        ep_weight2.append(f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight")
                    group_gemm1 = ",".join(ep_weight1)
                    group_gemm2 = ",".join(ep_weight2)
                    aoa_statements += [
                        f"{model_prefix}layers.{layer_id}.mlp.grouped_gemm_experts.weight1 -> {group_gemm1}, axis=0"
                        f"{model_prefix}layers.{layer_id}.mlp.grouped_gemm_experts.weight2 -> {group_gemm2}, axis=0"
                    ]

            for layer_id in range(config.num_hidden_layers):
                for expert_id in range(num_experts):
                    if getattr(cls, "is_fleet", False):
                        if using_sonic_moe:
                            aoa_statements += [
                                f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight, model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight, axis=0",
                            ]
                        else:
                            aoa_statements += [
                                f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight, model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight, axis=1",
                            ]
                    else:
                        aoa_statements += [
                            f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight, model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight, fused_ffn",
                        ]

                    if not using_sonic_moe:
                        aoa_statements += [
                            f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight",
                            f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight",
                            f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight",
                        ]

        if config.tie_word_embeddings:
            aoa_statements += ["lm_head.weight -> _"]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


@register_base_model
class Qwen3MoeModel(Qwen3MoePretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3MoeDecoderLayer`]
    Args:
        config: Qwen3MoeConfig
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
        )
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        past_key_values: Tensor,
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
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, MoEModelOutputWithPast]:

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

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

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
        if self.has_sliding_layers:
            causal_mask, attn_mask_startend_row_indices = create_sliding_window_causal_mask_and_row_indices(
                **mask_kwargs
            )
        else:
            causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        moelayer_use_subbatch_recompute = (
            self.config.moe_subbatch_token_num_before_dispatch > 0
            if hasattr(self.config, "moe_subbatch_token_num_before_dispatch")
            else False
        )

        for idx, (decoder_layer) in enumerate(self.layers):
            has_gradient = not hidden_states.stop_gradient
            if moelayer_use_subbatch_recompute:
                hidden_states = decoder_layer.subbatch_recompute_forward(
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
                hidden_states = self.recompute_training_full(
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
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values] if v is not None)
        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def load_balancing_loss_func(gate_logits, num_experts, top_k=2, attention_mask=None):
    """
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Paddle.
    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.
    Args:
        gate_logits (Union[`paddle.Tensor`, Tuple[paddle.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`):
            Number of experts.
        top_k (`int`):
            Number of top k experts to be considered for the loss computation.
        attention_mask (`paddle.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        concatenated_gate_logits = paddle.cat(
            gate_logits, axis=0
        )  # [num_hidden_layers X batch_size X sequence_length, num_experts]

    routing_weights = F.softmax(concatenated_gate_logits, axis=-1)
    _, selected_experts = paddle.topk(routing_weights, top_k, axis=-1)
    expert_mask = F.one_hot(
        selected_experts, num_classes=num_experts
    )  # [num_hidden_layers X batch_size X sequence_length, top_k, num_experts]

    if attention_mask is None or len(attention_mask.shape) == 4:
        # Only intokens strategy has 4-D attention_mask, we currently do not support excluding padding tokens.
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = paddle.mean(expert_mask.astype("float32"), axis=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = paddle.mean(routing_weights, axis=0)
    else:
        # Exclude the load balancing loss of padding tokens.
        if len(attention_mask.shape) == 2:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape([-1, top_k, num_experts])
            )  # [num_hidden_layers * batch_size * sequence_length, top_k, num_experts]

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = paddle.sum(expert_mask.astype("float32") * expert_attention_mask, axis=0) / paddle.sum(
                expert_attention_mask, axis=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape([-1, num_experts])
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = paddle.sum(
                routing_weights * router_per_expert_attention_mask, axis=0
            ) / paddle.sum(router_per_expert_attention_mask, axis=0)

    overall_loss = paddle.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Qwen3MoeForCausalLM(Qwen3MoePretrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = Qwen3MoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet

        return gpt_model


class Qwen3MoeForCausalLMDecapitated(Qwen3MoePretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
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
                "output_router_logits": output_router_logits,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        loss_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ):
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,  # [bs, seq_len]
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Qwen3MoeForCausalLMPipe(Qwen3MoePretrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = Qwen3MoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class Qwen3MoeForCausalLMPipeDecapitated(GeneralModelForCausalLMPipe):
    config_class = Qwen3MoeConfig
    _decoder_layer_cls = Qwen3MoeDecoderLayer
    _get_tensor_parallel_mappings = Qwen3MoeModel._get_tensor_parallel_mappings
    _init_weights = Qwen3MoeModel._init_weights
    _keep_in_fp32_modules = Qwen3MoeModel._keep_in_fp32_modules
    _rotary_emb_cls = Qwen3MoeRotaryEmbedding
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Qwen3MoeModel.transpose_weight_keys
    _gen_aoa_config = Qwen3MoeForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Qwen3MoeForCausalLM._gen_inv_aoa_config


__all__ = [
    "Qwen3MoeModel",
    "Qwen3MoePretrainedModel",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForCausalLMDecapitated",
    "Qwen3MoeForCausalLMPipe",
    "Qwen3MoeForCausalLMPipeDecapitated",
]
