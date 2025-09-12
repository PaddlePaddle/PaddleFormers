# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025 DeepSeek. All rights reserved.
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

import contextlib
import math
import os
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.jit import to_static
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from paddle.utils import try_import

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

from paddle import _C_ops
try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None
from paddleformers.transformers.model_utils import dtype_guard

from ...utils.initializer import kaiming_uniform_
from ...utils.log import logger
from ...utils.tools import get_env_device
from ..activations import ACT2FN
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..llama import fusion_ops
from ..llama.modeling import get_use_casual_mask
from ..model_outputs import (
    BaseModelOutputWithPastAndMTP,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from ..moe_gate import PretrainedMoEGate
from ..moe_layer import MoELayer
from ..utils import cast_if_needed, device_guard
from . import fp8_linear as linear_utils
from .configuration import DeepseekV2Config

FA_VERSION = int(os.getenv("FA_VERSION", 2))

from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import WeightGradStore

from ..fp8_utils import (
    FP8KeepXLinear,
    FP8Linear,
    FP8LinearFunction,
    FP8LinearFunctionBase,
    FP8Mlp,
    cache_fp8_weight,
    set_parameter_color,
)
from .fp8_linear import Linear

DSV3_USE_FP8_GEMM = os.getenv("DSV3_USE_FP8_GEMM", "False").lower() == "true"
DSV3_USE_ATTEN_RECOMPUTE = os.getenv("DSV3_USE_ATTEN_RECOMPUTE", "False").lower() == "true"

Linear = FP8Linear if DSV3_USE_FP8_GEMM else Linear

try:
    import fused_ln
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


try:
    from paddle.incubate.nn.functional import fused_partial_rope
except ImportError:
    fused_partial_rope = None

from .modeling import (set_global_step, scaled_dot_product_attention, is_casual_mask, _make_causal_mask, _expand_2d_mask, yarn_get_mscale, apply_rotary_pos_emb, DeepseekV2RMSNorm, DeepseekV2YarnRotaryEmbedding, FusedRMSLinear, MemroyRecomputeAttn, FusedNormGateFunc, FakeGate)

__all__ = [
    "DeepseekV2PretrainingCriterionFast",
    "DeepseekV2ModelFast",
    "DeepseekV2PretrainedModelFast",
]

def parallel_matmul(x: Tensor, y: Tensor, transpose_y=False, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except AttributeError:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=transpose_y)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = LMHeadFunction.apply(x, y, transpose_y=transpose_y)
        return logits


class DeepseekV2MLP(nn.Layer):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None, is_moe=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.fuse_attention_ffn = config.fuse_attention_ffn

        def linear_dtype_gaurd():
            if config.use_fp8:
                return dtype_guard("float8_e4m3fn")
            else:
                return contextlib.nullcontext()

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

        with linear_dtype_gaurd():
            if config.tensor_parallel_degree > 1 and not is_moe:
                self.gate_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )
                self.up_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )
                self.down_proj = RowParallelLinear(
                    self.intermediate_size,
                    self.hidden_size,
                    input_is_parallel=True,
                    has_bias=False,
                )
            else:
                if config.fuse_attention_ffn:
                    self.gate_up_fused_proj = Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=False)
                else:
                    self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
                    self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
                self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.fuse_attention_ffn:
            x = swiglu(self.gate_up_fused_proj(x))
        else:
            x = swiglu(self.gate_proj(x), self.up_proj(x))
        out = self.down_proj(x)
        return out


class MoEGate(PretrainedMoEGate):
    def __init__(
        self,
        config,
        num_experts,
        expert_hidden_size,
        using_post_norm_recompute=False,
        norm_weight=None,
        norm_eps=None,
        **kwargs
    ):
        super().__init__(config, num_experts, expert_hidden_size, **kwargs)
        # [hidden_size, n_expert]

        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method

        self.weight = paddle.create_parameter(
            shape=[expert_hidden_size, num_experts],
            dtype=paddle.float32,
            is_bias=False,
            # default_initializer=nn.initializer.Constant(1.0),
        )

        self.config = config
        self.using_post_norm_recompute = using_post_norm_recompute

        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = paddle.create_parameter(
                shape=[num_experts],
                dtype=paddle.float32,
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.e_score_correction_bias.is_distributed = True

        if self.using_post_norm_recompute:
            assert norm_weight is not None and norm_eps is not None
            self.norm_weight = norm_weight
            self.norm_eps = norm_eps
        self.using_flex_token = False

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (_type_): [batch_size * seq_len, hidden_size]
        """
        _, _, h_dim = hidden_states.shape

        # compute gating score
        if self.using_post_norm_recompute:
            logits, norm_out = FusedNormGateFunc.apply(hidden_states, self.norm_weight, self.weight, self.norm_eps)
            if hasattr(self.config, "using_fake_gate") and self.config.using_fake_gate:
                logits = FakeGate.apply(
                    hidden_states,
                    self.weight,
                    self.config.fakse_gate_restrict_balance,
                    self.config.num_experts_per_tok,
                )
        else:
            with paddle.amp.auto_cast(False):
                hidden_states = hidden_states.cast(self.weight.dtype)
                if hasattr(self.config, "using_fake_gate") and self.config.using_fake_gate:
                    logits = FakeGate.apply(
                        hidden_states,
                        self.weight,
                        self.config.fakse_gate_restrict_balance,
                        self.config.num_experts_per_tok,
                    )
                else:
                    logits = F.linear(hidden_states, self.weight, None)

        scores = self.gate_score_func(logits=logits)
        scores = scores.cast(paddle.float32)

        # Compute all possible return values
        if self.using_flex_token:
            scores, routing_map, exp_counts, l_aux, l_zloss = self.topkgating_nodrop(
                scores
            )  # (scores, routing_map, exp_counts, l_aux, l_zloss)
            ret = (scores, routing_map, l_aux, l_zloss)
        else:
            ret = self.topkgating(scores)  # (capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss)

        # Append norm_out if needed
        if self.using_post_norm_recompute:
            ret = (*ret, norm_out)

        return ret


class DeepseekV2MoE(MoELayer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config, norm_weight=None, norm_eps=None):
        assert config.tensor_parallel_degree <= 1, "tensor_parallel_degree should be 1"

        self.using_post_norm_recompute = config.using_post_norm_recompute
        if self.using_post_norm_recompute:
            assert norm_weight is not None and norm_eps is not None

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
            using_post_norm_recompute=self.using_post_norm_recompute,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        DeepseekV2MLPClass = FP8Mlp if DSV3_USE_FP8_GEMM else DeepseekV2MLP

        super().__init__(
            config=config,
            moe_num_experts=config.n_routed_experts,
            expert_class=DeepseekV2MLPClass,
            expert_kwargs={
                "config": config,
                "intermediate_size": config.moe_intermediate_size,
                "is_moe": True,
            },
            gate=gate,
            capacity=2.0,
            moe_group="expert",
            using_post_norm_recompute=self.using_post_norm_recompute,
        )

        if config.offline_quant_expert_weight and config.clear_origin_weight_when_offline_quant:
            moe_grad_group = fleet.get_hybrid_communicate_group().expert_grad_comm_group
            expert_w1_list = [expert.w1 for expert in self.experts if expert is not None]
            expert_w2_list = [expert.w2 for expert in self.experts if expert is not None]
            for p in expert_w1_list:
                setattr(p, "color", {"color": "moe_expert", "group": moe_grad_group})
            for p in expert_w2_list:
                setattr(p, "color", {"color": "moe_expert", "group": moe_grad_group})

        self.alpha = config.aux_loss_alpha
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            if self.using_post_norm_recompute:
                assert DeepseekV2MLPClass is FP8Mlp
                self.shared_experts = DeepseekV2MLPClass(
                    config=config,
                    intermediate_size=intermediate_size,
                    is_moe=False,
                    using_post_norm_recompute=self.using_post_norm_recompute,
                    norm_weight=norm_weight,
                    norm_eps=norm_eps,
                    recompute_fwd_gate_up=True,
                )
            else:
                self.shared_experts = DeepseekV2MLPClass(
                    config=config, intermediate_size=intermediate_size, is_moe=False
                )
            set_parameter_color([self.shared_experts.w1, self.shared_experts.w2], "shared_expert")

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=None):
        """Quantize weights in FP8 format.

        Args:
            batch_mode: If True, quantize all weights in batch mode using the first expert's weights.
                    If False, quantize each expert's weights individually.
        """

        def quantize_weights(weight_list, weight_obj=None, quant_transpose=None):
            """Helper function to quantize a list of weights."""
            if weight_obj is None:
                weight_obj = weight_list[0]
            if hasattr(weight_obj, "fp8_weight_stacked") or hasattr(weight_obj, "fp8_weight_stacked_transpose"):
                return

            if quant_transpose is None:
                fp8_weight, fp8_scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(
                    weight_list, transpose=False
                )
                setattr(weight_obj, "fp8_weight_stacked", fp8_weight)
                setattr(weight_obj, "fp8_scale_stacked", fp8_scale)

                fp8_weight_t, fp8_scale_t = paddle.incubate.nn.functional.fused_stack_transpose_quant(
                    weight_list, transpose=True
                )
                setattr(weight_obj, "fp8_weight_stacked_transpose", fp8_weight_t)
                setattr(weight_obj, "fp8_scale_stacked_transpose", fp8_scale_t)
            elif quant_transpose is False:
                # Only quantize without transpose
                fp8_weight, fp8_scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(
                    weight_list, transpose=False
                )
                setattr(weight_obj, "fp8_weight_stacked", fp8_weight)
                setattr(weight_obj, "fp8_scale_stacked", fp8_scale)
            elif quant_transpose is True:
                # Only quantize with transpose
                fp8_weight_t, fp8_scale_t = paddle.incubate.nn.functional.fused_stack_transpose_quant(
                    weight_list, transpose=True
                )
                setattr(weight_obj, "fp8_weight_stacked_transpose", fp8_weight_t)
                setattr(weight_obj, "fp8_scale_stacked_transpose", fp8_scale_t)
            else:
                raise ValueError("Invalid value for `quant_transpose`.")

        if batch_mode:
            # Batch mode: process all experts' weights together
            expert_w1_list = [expert.w1 for expert in self.experts if expert is not None]
            expert_w2_list = [expert.w2 for expert in self.experts if expert is not None]

            if expert_w1_list:
                quantize_weights(expert_w1_list, expert_w1_list[0], quant_transpose)
            if expert_w2_list:
                quantize_weights(expert_w2_list, expert_w2_list[0], quant_transpose)
        else:
            # Individual mode: process each expert's weights separately
            for expert in self.experts:
                if expert is not None:
                    quantize_weights([expert.w1], quant_transpose=quant_transpose)
                    quantize_weights([expert.w2], quant_transpose=quant_transpose)

        if self.config.n_shared_experts is not None:
            self.shared_experts.fp8_quant_weight(quant_transpose)

    def forward(self, hidden_states):
        if self.using_post_norm_recompute:
            super().update_flex_token()
            if self.using_flex_token:
                probs, routing_map, l_aux, l_zloss, norm_out = self.router(hidden_states)
                final_hidden_states, l_aux, l_zloss = super().forward(
                    norm_out, probs=probs, routing_map=routing_map, l_aux=l_aux, l_zloss=l_zloss
                )
            else:
                capacity, topk_weight, topk_ids, token_priority, l_aux, l_zloss, norm_out = self.gate(hidden_states)
                final_hidden_states, l_aux, l_zloss = super().forward(
                    norm_out,
                    capacity=capacity,
                    topk_weight=topk_weight,
                    topk_ids=topk_ids,
                    token_priority=token_priority,
                    l_aux=l_aux,
                    l_zloss=l_zloss,
                )
            final_hidden_states = self.post_process(hidden_states, final_hidden_states, l_aux)
        else:
            final_hidden_states, l_aux, l_zloss = super().forward(hidden_states)
            final_hidden_states = self.post_process(hidden_states, final_hidden_states, l_aux)
        return final_hidden_states

    def post_process(self, hidden_states, final_hidden_states, l_aux):
        if self.training and self.alpha > 0.0:
            l_aux = l_aux * self.alpha
            final_hidden_states = AddAuxiliaryLoss.apply(final_hidden_states, l_aux)

        if self.config.n_shared_experts is not None:
            shared_expert_output = self.shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layerwise_recompute: bool = False, recompute_fa3: bool = False):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

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

        if config.num_nextn_predict_layers > 0:
            self.seq_length = config.seq_length - config.num_nextn_predict_layers
        else:
            self.seq_length = config.seq_length
        self.sequence_parallel = config.sequence_parallel

        self.recompute_fa3 = recompute_fa3

        self.input_layernorm = DeepseekV2RMSNorm(config)

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

        def linear_dtype_gaurd():
            if config.use_fp8:
                return dtype_guard("float8_e4m3fn")
            else:
                return contextlib.nullcontext()

        # Note (@DrownFish19): For tensor parallel we consider that q_a_proj and kv_a_proj_with_mqa
        # are the small weight and cannot achieve performance gain. So we use the original
        # linear layers. We use the tensor parallel linear layers for q_proj，q_b_proj and kv_b_proj
        # for which are the large weight and can achieve performance gain.

        self._init_rope()
        self.softmax_scale = self.q_head_dim ** (-0.5)

        # fmt: off
        if self.config.tensor_parallel_degree > 1:
            # for tensor parallel
            if config.sequence_parallel:
                ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
                RowParallelLinear = linear_utils.RowSequenceParallelLinear
            else:
                ColumnParallelLinear = linear_utils.ColumnParallelLinear
                RowParallelLinear = linear_utils.RowParallelLinear

            if self.q_lora_rank is None:
                with linear_dtype_gaurd():
                    self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.q_head_dim, has_bias=False, gather_output=True)
            else:
                with linear_dtype_gaurd():
                    self.q_a_proj = Linear(self.hidden_size, config.q_lora_rank, bias_attr=config.attention_bias)
                    self.q_b_proj = ColumnParallelLinear(config.q_lora_rank, self.num_heads * self.q_head_dim, has_bias=False, gather_output=True)
                self.q_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.q_lora_rank, use_sequence_parallel=False)

            with linear_dtype_gaurd():
                self.kv_a_proj_with_mqa = paddle.nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias_attr=config.attention_bias)
                self.kv_b_proj = ColumnParallelLinear(config.kv_lora_rank, self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), has_bias=False, gather_output=True)
                self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size, has_bias=config.attention_bias, input_is_parallel=False)
            self.kv_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.kv_lora_rank, use_sequence_parallel=False)
        else:
            # for without tensor parallel
            if DSV3_USE_ATTEN_RECOMPUTE:
                self.fused_rms_norm_linear = FusedRMSLinear(self.hidden_size, config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim, 1e-6)
                kv_up_dim = self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim)
                self.memory_recompute_att = MemroyRecomputeAttn(config.q_lora_rank, config.kv_lora_rank, config.q_lora_rank, self.num_heads * self.q_head_dim, config.kv_lora_rank, kv_up_dim, self.rotary_emb, self.num_heads, self.q_head_dim, self.qk_nope_head_dim, self.v_head_dim, self.qk_rope_head_dim, 1e-6, self.kv_lora_rank, self.softmax_scale, recompute_fa3=self.recompute_fa3)
                self.o_proj = FP8KeepXLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias_attr=config.attention_bias)
            else:

                if self.q_lora_rank is None:
                    with linear_dtype_gaurd():
                        self.q_proj = Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias_attr=False)
                else:
                    with linear_dtype_gaurd():
                        self.q_a_proj = Linear(self.hidden_size, config.q_lora_rank, bias_attr=config.attention_bias)
                        self.q_b_proj = Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias_attr=False)
                    self.q_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.q_lora_rank)

                with linear_dtype_gaurd():
                    self.kv_a_proj_with_mqa = paddle.nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias_attr=config.attention_bias)
                    self.kv_b_proj = Linear(config.kv_lora_rank, self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), bias_attr=False)
                    self.o_proj = Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias_attr=config.attention_bias)
                self.kv_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.kv_lora_rank)

        # fmt: on
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_func = scaled_dot_product_attention

    def fp8_quant_weight(self, quant_transpose=None):

        if DSV3_USE_ATTEN_RECOMPUTE:
            self.o_proj.fp8_quant_weight(quant_transpose=quant_transpose)
            self.memory_recompute_att.fp8_quant_weight(quant_transpose=quant_transpose)
            self.fused_rms_norm_linear.fp8_quant_weight(quant_transpose=quant_transpose)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

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
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.shape

        # DeepSeekV2 q_lora_rank=1536
        # DeepSeekV2-lite q_lora_rank=None
        if DSV3_USE_ATTEN_RECOMPUTE:

            q_t1, compressed_kv = self.fused_rms_norm_linear(hidden_states)

            outputs = self.memory_recompute_att(q_t1, compressed_kv, position_ids)

            if self.v_head_dim * self.num_heads != outputs.shape[-1]:
                outputs = outputs.reshape([bsz, q_len, self.num_heads, -1])
                outputs = outputs[..., : self.v_head_dim]
                outputs = outputs.reshape([bsz, q_len, -1])
        else:
            hidden_states = self.input_layernorm(hidden_states)
            if self.q_lora_rank is None:
                q = self.q_proj(hidden_states)
            else:
                q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

            if self.sequence_parallel:
                target_query_shape = [-1, self.seq_length, self.num_heads, self.q_head_dim]
                target_key_value_shape = [-1, self.seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim]
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
            k_pe = k_pe.reshape([-1, q_len, 1, self.qk_rope_head_dim]).expand(
                [-1, q_len, self.num_heads, self.qk_rope_head_dim]
            )

            # self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim = 128+64
            # self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim) = config.qk_nope_head_dim + self.v_head_dim = 128+128
            kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).reshape(shape=target_key_value_shape)

            k_nope, value_states = paddle.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)
            kv_seq_len = value_states.shape[1]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-3]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, self.fuse_rope)

            query_states = paddle.concat([q_nope, q_pe], axis=-1)
            key_states = paddle.concat([k_nope, k_pe], axis=-1)

            # [bs, seq_len, num_head, head_dim]
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = paddle.concat([past_key_value[0], key_states], axis=1)
                value_states = paddle.concat([past_key_value[1], value_states], axis=1)
            past_key_value = (key_states, value_states) if use_cache else None

            has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
            if (
                self.enable_recompute
                and self.layerwise_recompute
                and has_gradient
                and self.recompute_granularity == "core_attn"
            ):
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
    def __init__(
        self, config: DeepseekV2Config, layer_idx: int, layerwise_recompute: bool = False, recompute_fa3: bool = False
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity
        self.using_post_norm_recompute = config.using_post_norm_recompute

        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV2Attention(
            config=config, layerwise_recompute=layerwise_recompute, recompute_fa3=recompute_fa3
        )

        DeepseekV2MLPClass = FP8Mlp if DSV3_USE_FP8_GEMM else DeepseekV2MLP

        self.input_layernorm = DeepseekV2RMSNorm(config)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config)

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = (
                DeepseekV2MoE(
                    config, self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon
                )
                if config.using_post_norm_recompute
                else DeepseekV2MoE(config)
            )
        else:
            self.mlp = DeepseekV2MLPClass(config, recompute_fwd_gate_up=True)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=None):
        """fp8_quant_weight"""
        if isinstance(self.mlp, DeepseekV2MoE):
            # logger.info(f"fp8 quant weight for mlp {type(self.mlp)}")
            self.mlp.fp8_quant_weight(batch_mode, quant_transpose=quant_transpose)
            self.self_attn.fp8_quant_weight(quant_transpose=quant_transpose)
        elif isinstance(self.mlp, FP8Mlp):
            self.self_attn.fp8_quant_weight(quant_transpose=quant_transpose)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_axis)`
            attention_mask (`paddle.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "full_attn"
        ):
            outputs = recompute(
                self.self_attn,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
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
                **kwargs,
            )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if output_attentions:
            self_attn_weights = outputs[1]

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        if not (self.using_post_norm_recompute and isinstance(self.mlp, DeepseekV2MoE)):
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def self_attn_compute(self, hidden_states, **kwargs):
        residual = hidden_states

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "full_attn"
        ):
            outputs = recompute(
                self.self_attn,
                hidden_states=hidden_states,
                position_ids=None,
                attention_mask=None,
                output_attentions=False,
                past_key_value=None,
                use_cache=False,
                attn_mask_startend_row_indices=None,
                **kwargs,
            )
        else:
            outputs = self.self_attn(
                hidden_states=hidden_states,
                position_ids=None,
                attention_mask=None,
                output_attentions=False,
                past_key_value=None,
                use_cache=False,
                attn_mask_startend_row_indices=None,
                **kwargs,
            )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        hidden_states = residual + hidden_states

        residual = hidden_states

        if not (self.using_post_norm_recompute and isinstance(self.mlp, DeepseekV2MoE)):
            hidden_states = self.post_attention_layernorm(hidden_states)

        return hidden_states, residual

    def pre_dispatch_compute(self, hidden_states):
        l_aux, l_zloss, intermediate_hidden_states, token_indices, token_probs = self.mlp.pre_dispatch_compute(
            hidden_states
        )

        return l_aux, l_zloss, intermediate_hidden_states, token_indices, token_probs

    def expert_forward_compute(self, intermediate_hidden_states, dispatched_indices, dispatched_probs):
        (global_input_tokens, token_permuted_indices, prob_permuted_indices) = self.mlp.post_dispatch_compute(
            intermediate_hidden_states, dispatched_indices, dispatched_probs
        )

        expert_output = self.mlp.expert_forward(global_input_tokens)

        expert_output = self.mlp.pre_combine_compute(
            expert_output, token_permuted_indices, prob_permuted_indices, dispatched_probs
        )

        return expert_output

    def post_combine_compute(self, residual, hidden_states, final_hidden_states, l_aux):
        final_hidden_states = self.mlp.post_combine_compute(final_hidden_states)

        final_hidden_states = self.mlp.post_process(hidden_states, final_hidden_states, l_aux)

        final_hidden_states = residual + final_hidden_states

        outputs = (final_hidden_states,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class DeepseekV2MTPLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        layerwise_recompute: bool = False,
    ):
        super(DeepseekV2MTPLayer, self).__init__(config, layer_idx, layerwise_recompute)

        self.enorm = DeepseekV2RMSNorm(config)
        self.hnorm = DeepseekV2RMSNorm(config)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias_attr=False)

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
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        hidden_states = self.hnorm(hidden_states)
        nextn_hidden_state = self.enorm(nextn_hidden_state)

        concat_h = paddle.concat([nextn_hidden_state, hidden_states], axis=-1)
        hidden_states = FP8LinearFunction.apply(concat_h, self.eh_proj)

        layer_outputs = super(DeepseekV2MTPLayer, self).forward(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            **kwargs,
        )

        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        return hidden_states


class DeepseekV2PretrainedModelFast(PretrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "deepseek_v2"
    _no_split_modules = ["DeepseekV2DecoderLayer"]

    def _get_model_flops(self, batch_size=1, seq_length=None, **kwargs):
        from .mfu_utils import DeepSeekProjection

        # self._
        mfu_cal_proj = DeepSeekProjection(self.config)
        if seq_length is None:
            if hasattr(self.config, "seq_length"):
                seq_length = self.config.seq_length
            else:
                seq_length = 2048

        return mfu_cal_proj.get_num_flop_per_token()

    def _get_hardware_flops(self, *args, **kwargs):
        return self._get_model_flops(*args, **kwargs)

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
        from paddleformers.transformers.conversion_utils import split_or_merge_func

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
            if config.use_fp8:
                base_actions["layers.0.self_attn.o_proj.weight.weight_scale_inv"] = partial(fn, is_column=False)

            if config.tie_word_embeddings:
                base_actions["lm_head.weight"] = partial(fn, is_column=False)
            else:
                base_actions["lm_head.weight"] = partial(fn, is_column=True)

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")

            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(fn, is_column=True)

            # if we have enough num_key_value_heads to split, then split it.
            # ???
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.kv_b_proj.weight"] = partial(fn, is_column=True)
                if config.use_fp8:
                    base_actions["layers.0.self_attn.kv_b_proj.weight.weight_scale_inv"] = partial(fn, is_column=True)

            # dense mlp
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.down_proj.weight"] = partial(fn, is_column=False)
            if config.use_fp8:
                base_actions["layers.0.mlp.up_proj.weight.weight_scale_inv"] = partial(fn, is_column=True)
                base_actions["layers.0.mlp.gate_proj.weight.weight_scale_inv"] = partial(fn, is_column=True)
                base_actions["layers.0.mlp.down_proj.weight.weight_scale_inv"] = partial(fn, is_column=False)

            # moe unit routed experts
            moe_group = dist.fleet.get_hybrid_communicate_group().get_data_parallel_group()
            expert_parallel_degree = dist.get_world_size(moe_group)
            if expert_parallel_degree <= 1:
                for e_i in range(config.n_routed_experts):
                    base_actions[f"layers.0.mlp.experts.{e_i}.up_proj.weight"] = partial(fn, is_column=True)
                    base_actions[f"layers.0.mlp.experts.{e_i}.gate_proj.weight"] = partial(fn, is_column=True)
                    base_actions[f"layers.0.mlp.experts.{e_i}.down_proj.weight"] = partial(fn, is_column=False)

            # moe unit shared experts
            base_actions["layers.0.mlp.shared_experts.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.down_proj.weight"] = partial(fn, is_column=False)
            if config.use_fp8:
                base_actions["layers.0.mlp.shared_experts.gate_proj.weight.weight_scale_inv"] = partial(
                    fn, is_column=True
                )
                base_actions["layers.0.mlp.shared_experts.up_proj.weight.weight_scale_inv"] = partial(
                    fn, is_column=True
                )
                base_actions["layers.0.mlp.shared_experts.down_proj.weight.weight_scale_inv"] = partial(
                    fn, is_column=False
                )

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

    def _init_weights(self, layer):
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state

        if isinstance(
            layer,
            (
                nn.Linear,
                nn.Embedding,
                mpu.VocabParallelEmbedding,
                mpu.RowParallelLinear,
                mpu.ColumnParallelLinear,
                linear_utils.RowSequenceParallelLinear,
                linear_utils.ColumnSequenceParallelLinear,
                Linear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                if layer.weight.is_distributed:
                    with rng_tracker():
                        layer.weight.set_value(
                            paddle.tensor.normal(
                                mean=0.0,
                                std=self.config.initializer_range
                                if hasattr(self.config, "initializer_range")
                                else self.config.initializer_range,
                                shape=layer.weight.shape,
                            )
                        )
                else:
                    layer.weight.set_value(
                        paddle.tensor.normal(
                            mean=0.0,
                            std=self.config.initializer_range
                            if hasattr(self.config, "initializer_range")
                            else self.config.initializer_range,
                            shape=layer.weight.shape,
                        )
                    )

                # set bias to zeros
                if getattr(layer, "bias", None) is not None:
                    layer.bias.set_value(paddle.zeros(shape=layer.bias.shape))

        if isinstance(layer, nn.Embedding):
            if layer._padding_idx is not None:
                layer.weight.data[layer._padding_idx].fill_(0)

        if isinstance(layer, MoEGate):
            kaiming_uniform_(layer.weight, a=math.sqrt(5))

        moe_grad_group = fleet.get_hybrid_communicate_group().expert_grad_comm_group
        if moe_grad_group is not None and moe_grad_group.nranks > 1:
            for p in layer.parameters():
                if hasattr(p, "color") and "color" in p.color:
                    if p.color["color"] == "moe_expert":
                        paddle.distributed.broadcast(p, src=moe_grad_group.ranks[0], group=moe_grad_group)

    def step_flex_token(self, cur_step):
        set_global_step(cur_step)


@register_base_model
class DeepseekV2ModelFast(DeepseekV2PretrainedModelFast):
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
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.embed_tokens = mpu.VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.LayerList(
            [
                DeepseekV2DecoderLayer(config, layer_idx, layer_idx not in self.no_recompute_layers)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        for layer_idx in range(config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers):
            self.layers.append(DeepseekV2MTPLayer(config, layer_idx, layer_idx not in self.no_recompute_layers))

        self.norm = DeepseekV2RMSNorm(config)

        self.enable_recompute = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        if get_env_device() == "xpu":
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(-1.7005809656952787e38, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask, x, y)
        else:
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(
                dtype
            )
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
                ]

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

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

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
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        mtp_outputs = []

        for idx in range(self.config.num_hidden_layers):
            decoder_layer = self.layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if (
                self.enable_recompute
                and idx not in self.no_recompute_layers
                and has_gradient
                and self.recompute_granularity == "full"
            ):
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
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

                inputs_embeds_cur_depth = paddle.concat(
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


class DeepseekV2PretrainingCriterionFast(nn.Layer):
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

        def compute_loss(preds, labels):
            with paddle.amp.auto_cast(False):
                masked_lm_loss = FastCrossEntropyFunction.apply(preds, labels.unsqueeze(2))
                binary_sequence = paddle.where(
                    masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
                )
                count = paddle.sum(binary_sequence)
                loss = paddle.where(
                    count == 0,
                    paddle.sum(masked_lm_loss * binary_sequence),
                    paddle.sum(masked_lm_loss * binary_sequence) / count,
                )
                return loss