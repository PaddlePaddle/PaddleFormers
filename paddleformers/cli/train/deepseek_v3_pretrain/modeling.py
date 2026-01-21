# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import WeightGradStore

from paddleformers.transformers.activations import ACT2FN
from paddleformers.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)
from paddleformers.transformers.deepseek_v3 import (
    rotate_half,
    scaled_dot_product_attention,
    yarn_get_mscale,
)
from paddleformers.transformers.fp8_utils import (
    FP8KeepXLinear,
    FP8Linear,
    FP8LinearFunction,
    FP8LinearFunctionBase,
    FP8Mlp,
    cache_fp8_weight,
    set_parameter_color,
)
from paddleformers.transformers.model_outputs import BaseModelOutputWithPastAndMTP
from paddleformers.transformers.model_utils import (
    PretrainedModel,
    dtype_guard,
    register_base_model,
)
from paddleformers.transformers.utils import device_guard
from paddleformers.utils.initializer import kaiming_uniform_
from paddleformers.utils.log import logger
from paddleformers.utils.masking_utils import (
    _expand_2d_mask,
    _make_causal_mask,
    is_casual_mask,
)

from . import fp8_linear as linear_utils
from .configuration import DeepseekV2FastConfig
from .fp8_linear import Linear as Linear_
from .moe_gate import PretrainedMoEGate
from .moe_layer import MoELayer
from .moe_utils import get_env_device

try:
    from paddle.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


try:
    from paddle.incubate.nn.functional import fused_partial_rope
except ImportError:
    fused_partial_rope = None


__all__ = [
    "DeepseekV2LMHead",
    "set_global_step",
    "get_global_step",
    "DeepseekV2PretrainingCriterionFast",
    "DeepseekV2ModelFast",
    "DeepseekV2PretrainedModelFast",
]

global_step = 0


def get_use_casual_mask():
    """Get the value of the 'USE_CASUAL_MASK' environment variable."""
    return os.getenv("USE_CASUAL_MASK", "False") == "True"


def set_global_step(cur_step):
    global global_step
    global_step = cur_step


def get_global_step():
    global global_step
    return global_step


def rms_norm_fused(x_in, w, eps, use_fast_ln=False):
    if use_fast_ln:
        fast_ln = try_import("fast_ln")
        return fast_ln.fast_rms_norm(x_in, w, eps)[0]
    else:
        return paddle.incubate.nn.functional.fused_rms_norm_ext(x_in, w, eps)[0]


def cast_if_needed(x, dtype):
    """
    cast_if_needed
    """
    return x.cast(dtype) if x.dtype != dtype else x


def fusion_rms_norm(hidden_states, weight, variance_epsilon, use_fast_ln=False):
    if get_env_device() == "npu":
        return paddle.base.core.eager._run_custom_op("rms_norm_npu", hidden_states, weight, variance_epsilon)[0]
    if get_env_device() == "mlu":
        return paddle.base.core.eager._run_custom_op("rms_norm_mlu", hidden_states, weight, variance_epsilon)[0]
    elif get_env_device() == "gcu":
        return paddle.base.core.eager._run_custom_op("rms_norm_gcu", hidden_states, weight, variance_epsilon)[0]
    elif get_env_device() == "intel_hpu":
        return paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states, weight, None, variance_epsilon, hidden_states.dim() - 1
        )[0]
    elif get_env_device() == "xpu":
        try:
            import paddle_xpu_nn  # noqa: F821

            return paddle_xpu_nn.xpu_rms_norm(hidden_states, weight, variance_epsilon)[0]
        except ImportError:
            raise NotImplementedError(
                f"Implementation of fused_rms_norm is not available on {get_env_device()}. Please install paddle_xpu to use this feature"
            )
    return rms_norm_fused(hidden_states, weight, variance_epsilon, use_fast_ln)


class LMHeadFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, transpose_y):
        out = paddle.matmul(x, weight, transpose_y=transpose_y)

        ctx.save_for_backward(x, weight, transpose_y)
        return out

    @staticmethod
    def backward(ctx, dout):
        if dout.dtype == paddle.float32:
            dout = dout.cast(paddle.bfloat16)

        x, weight, transpose_y = ctx.saved_tensor()

        dx = paddle.matmul(dout, weight, transpose_y=not transpose_y)
        if transpose_y:
            with paddle.amp.auto_cast(False):
                paddle._C_ops.fused_linear_param_grad_add(
                    dout.reshape([-1, dout.shape[-1]]),
                    x.reshape([-1, x.shape[-1]]),
                    weight.main_grad,
                    None,
                    True,
                    False,
                )
        else:
            with paddle.amp.auto_cast(False):
                paddle._C_ops.fused_linear_param_grad_add(
                    x.reshape([-1, x.shape[-1]]),
                    dout.reshape([-1, dout.shape[-1]]),
                    weight.main_grad,
                    None,
                    True,
                    False,
                )
        return dx, None


def parallel_matmul(x: Tensor, y: Tensor, transpose_y=False, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_model_parallel_size = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_model_parallel_size = hcg.get_model_parallel_world_size()
    except AttributeError:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_model_parallel_size > 1

    if is_fleet_init and tensor_model_parallel_size > 1 and y_is_distributed:
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
    def __init__(self, config: DeepseekV2FastConfig, hidden_size=None, intermediate_size=None, is_moe=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.fuse_attention_ffn = config.fuse_attention_ffn
        Linear = FP8Linear if self.config.dsv3_use_fp8_gemm else Linear_

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
            if config.tensor_model_parallel_size > 1 and not is_moe:
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

            self.expert_usage = paddle.zeros(
                shape=[num_experts],
                dtype=paddle.int64,
            )
            self.expert_usage.stop_gradient = True

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
            scores, routing_map, exp_counts, l_aux, l_zloss = self.topkgating_nodrop(scores)
            with paddle.no_grad():
                self.expert_usage += exp_counts
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

    def __init__(self, config: DeepseekV2FastConfig, norm_weight=None, norm_eps=None):
        assert config.tensor_model_parallel_size <= 1, "tensor_model_parallel_size should be 1"

        self.using_post_norm_recompute = config.using_post_norm_recompute
        if self.using_post_norm_recompute:
            assert norm_weight is not None and norm_eps is not None

        hcg = fleet.get_hybrid_communicate_group()
        moe_grad_group = hcg.get_moe_sharding_parallel_group()

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
            using_post_norm_recompute=self.using_post_norm_recompute,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        self.is_mp_moe = False
        self.is_ep_moe = True
        DeepseekV2MLPClass = FP8Mlp if config.dsv3_use_fp8_gemm else DeepseekV2MLP

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
            expert_w1_list = [expert.w1 for expert in self.experts if expert is not None]
            expert_w2_list = [expert.w2 for expert in self.experts if expert is not None]
            for p in expert_w1_list:
                setattr(p, "is_moe_param", True)
                setattr(p, "color", {"color": "moe_expert", "group": moe_grad_group})
                p.no_sync = not self.is_mp_moe
                p.expert = not self.is_mp_moe
                logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            for p in expert_w2_list:
                setattr(p, "is_moe_param", True)
                setattr(p, "color", {"color": "moe_expert", "group": moe_grad_group})
                p.no_sync = not self.is_mp_moe
                p.expert = not self.is_mp_moe
                logger.info(f"expert no-sync={p.no_sync}-{p.name}")

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


class DeepseekV2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        with device_guard("cpu"):
            self.inv_freq = 1.0 / (
                self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim)
            )
            self._set_cos_sin_cache(seq_len=max_position_embeddings)

        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, axis/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, axis]
        emb = paddle.cat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, axis]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2FastConfig, layerwise_recompute: bool = False, recompute_fa3: bool = False):
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
        self.apply_rope_fusion = config.apply_rope_fusion

        if config.num_nextn_predict_layers > 0:
            self.seq_length = config.seq_length - config.num_nextn_predict_layers
        else:
            self.seq_length = config.seq_length
        self.sequence_parallel = config.sequence_parallel

        self.recompute_fa3 = recompute_fa3
        self.fa_version = config.fa_version

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
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        Linear = FP8Linear if self.config.dsv3_use_fp8_gemm else Linear_

        # fmt: off
        if self.config.tensor_model_parallel_size > 1:
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
            if self.config.dsv3_use_atten_recompute:
                self.fused_rms_norm_linear = FusedRMSLinear(self.hidden_size, config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim, 1e-6)
                kv_up_dim = self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim)
                self.o_proj = FP8KeepXLinear(self.num_heads * self.v_head_dim, self.hidden_size, bias_attr=config.attention_bias)
                self.memory_recompute_att = MemroyRecomputeAttn(config.q_lora_rank, config.kv_lora_rank, config.q_lora_rank, self.num_heads * self.q_head_dim, config.kv_lora_rank, kv_up_dim, self.rotary_emb, self.num_heads, self.q_head_dim, self.qk_nope_head_dim, self.v_head_dim, self.qk_rope_head_dim, 1e-6, self.seq_length, self.o_proj.weight, self.kv_lora_rank, self.softmax_scale, recompute_fa3=self.recompute_fa3, fa_version=self.fa_version)
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
        self.attn_func = scaled_dot_product_attention

    def fp8_quant_weight(self, quant_transpose=None):

        if self.config.dsv3_use_atten_recompute:
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
            if scaling_type == "yarn":
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
        if self.config.dsv3_use_atten_recompute:

            q_t1, compressed_kv = self.fused_rms_norm_linear(hidden_states)

            outputs = self.memory_recompute_att(q_t1, compressed_kv, position_ids)

            if not self.recompute_fa3:
                if self.v_head_dim * self.num_heads != outputs.shape[-1]:
                    outputs = outputs.reshape([bsz, q_len, self.num_heads, -1])
                    outputs = outputs[..., : self.v_head_dim]
                    outputs = outputs.reshape([bsz, q_len, -1])
        else:
            assert self.recompute_fa3 is False
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
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, self.apply_rope_fusion)

            query_states = paddle.cat([q_nope, q_pe], axis=-1)
            key_states = paddle.cat([k_nope, k_pe], axis=-1)

            # [bs, seq_len, num_head, head_dim]
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = paddle.cat([past_key_value[0], key_states], axis=1)
                value_states = paddle.cat([past_key_value[1], value_states], axis=1)
            past_key_value = (key_states, value_states) if use_cache else None

            has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
            if (
                self.enable_recompute
                and self.layerwise_recompute
                and has_gradient
                and self.config.recompute_granularity == "selective"
                and self.config.recompute_modules is not None
                and "core_attn" in self.config.recompute_modules
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
            assert self.recompute_fa3 is False
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.

        if not self.recompute_fa3:
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
        self,
        config: DeepseekV2FastConfig,
        layer_idx: int,
        layerwise_recompute: bool = False,
        recompute_fa3: bool = False,
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

        DeepseekV2MLPClass = FP8Mlp if self.config.dsv3_use_fp8_gemm else DeepseekV2MLP

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
            and self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "full_attn" in self.config.recompute_modules
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
            and self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "full_attn" in self.config.recompute_modules
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
        config: DeepseekV2FastConfig,
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

        concat_h = paddle.cat([nextn_hidden_state, hidden_states], axis=-1)
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
    config_class = DeepseekV2FastConfig
    base_model_prefix = "deepseek_v2"
    _no_split_modules = ["DeepseekV2DecoderLayer"]

    def _get_model_flops(self, batch_size=1, seq_length=None, **kwargs):
        from paddleformers.transformers.deepseek_v3.mfu_utils import DeepSeekProjection

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
    def _get_name_mappings(cls, config: DeepseekV2FastConfig) -> list[StateDictNameMapping]:
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
    def _get_tensor_parallel_mappings(cls, config: DeepseekV2FastConfig, is_split=True):
        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
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

            if not config.vocab_size % config.tensor_model_parallel_size == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")

            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(fn, is_column=True)

            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_model_parallel_size == 0:
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
            expert_model_parallel_size = dist.get_world_size(moe_group)
            if expert_model_parallel_size <= 1:
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
        if self.config.tensor_model_parallel_size > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        Linear = FP8Linear if self.config.dsv3_use_fp8_gemm else Linear_

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
        config: DeepseekV2FastConfig
    """

    def __init__(self, config: DeepseekV2FastConfig):
        super().__init__(config)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        if config.tensor_model_parallel_size > 1 and config.vocab_size % config.tensor_model_parallel_size == 0:
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
                and self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
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

    def __init__(self, config: DeepseekV2FastConfig):
        super(DeepseekV2PretrainingCriterionFast, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = config.tensor_model_parallel_size > 1 and config.tensor_parallel_output

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

        def add_loss(main_loss, loss):
            return main_loss + loss - loss.detach()

        if mtp_logits is not None and self.config.num_nextn_predict_layers > 0:
            assert len(mtp_logits) == self.config.num_nextn_predict_layers
            masked_lm_labels_ori = masked_lm_labels
            masked_lm_labels = masked_lm_labels[:, : -self.config.num_nextn_predict_layers]
            seq_length = masked_lm_labels.shape[1]
            loss = compute_loss(prediction_scores, masked_lm_labels)

            mtp_loss_res = []
            for depth in range(self.config.num_nextn_predict_layers):
                prediction_scores_cur_depth = mtp_logits[depth]
                masked_lm_labels_cur_depth = masked_lm_labels_ori[:, (depth + 1) : (depth + 1 + seq_length)]
                res_cur_depth = compute_loss(prediction_scores_cur_depth, masked_lm_labels_cur_depth)
                mtp_loss_res.append(res_cur_depth)
            loss = add_loss(loss, self.config.num_nextn_predict_lambda * sum([x for x in mtp_loss_res]) / len(mtp_loss_res))  # fmt: skip

        else:
            loss = compute_loss(prediction_scores, masked_lm_labels)

        if router_loss is not None and isinstance(router_loss, paddle.Tensor):
            loss = add_loss(loss, router_loss)

        return loss


# Inverse axis formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find axis range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (paddle.arange(dim, dtype=paddle.float32) - min) / (max - min)
    ramp_func = paddle.clip(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len):
        with paddle.amp.auto_cast(False):
            self.max_seq_len_cached = seq_len
            dim = self.dim

            freq_extra = 1.0 / (self.base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
            freq_inter = 1.0 / (
                self.scaling_factor * self.base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim)
            )

            low, high = yarn_find_correction_range(
                self.beta_fast,
                self.beta_slow,
                dim,
                self.base,
                self.original_max_position_embeddings,
            )
            inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
            self.inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

            t = paddle.arange(seq_len, dtype=paddle.float32)

            freqs = paddle.outer(t, paddle.cast(self.inv_freq, dtype="float32"))

            _mscale = float(
                yarn_get_mscale(self.scaling_factor, self.mscale)
                / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
            )

            emb = paddle.cat((freqs, freqs), axis=-1)
            self.cos_cached = emb.cos() * _mscale
            self.sin_cached = emb.sin() * _mscale


class RmsNormFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, scale, epsilon):
        norm_output, invar = paddle.incubate.nn.functional.fused_rms_norm_ext(x, scale, epsilon)
        ctx.save_for_backward(x, scale, invar)
        ctx.epsilon = epsilon
        ctx.x_stop_gradient = x.stop_gradient
        ctx.scale_stop_gradient = scale.stop_gradient
        return norm_output

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, invar = ctx.saved_tensor()
        epsilon = ctx.epsilon
        dx, dscale = paddle._C_ops.fused_rms_norm_ext_grad(x, scale, invar, grad_output, epsilon)
        if ctx.x_stop_gradient and ctx.scale_stop_gradient:
            return None, None
        elif ctx.x_stop_gradient:
            return None, dscale
        elif ctx.scale_stop_gradient:
            return dx, None
        else:
            return dx, dscale


class DeepseekV2RMSNorm(nn.Layer):
    def __init__(self, config: DeepseekV2FastConfig, hidden_size=None, eps=1e-6, use_sequence_parallel=True):
        """DeepseekV2RMSNorm is equivalent to T5LayerNorm

        Args:
            config (DeepseekV2FastConfig): config dict of DeepseekV2
            hidden_size (_type_): history_states size
            eps (_type_, optional): eps value. Defaults to 1e-6.
            use_sequence_parallel (bool, optional): A switch to disable sequence parallelism for inputs that are not in tensor parallel mode.
                                                    By default, this is set to True.
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.variance_epsilon = eps

        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )

        if config.sequence_parallel and use_sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, hidden_states):
        if self.config.fuse_rms_norm:
            return RmsNormFunction.apply(hidden_states, self.weight, self.variance_epsilon)

        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.astype("float32")
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, dtype={self.weight.dtype}"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, apply_rope_fusion=False):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`):
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
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    b, s, h, d = q.shape
    q = q.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])

    b, s, h, d = k.shape
    k = k.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])

    if (get_env_device() == "xpu" or get_env_device() == "gpu") and apply_rope_fusion:
        q_embed, k_embed, _ = fused_rotary_position_embedding(
            q,
            k,
            None,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )
        return q_embed, k_embed

    if position_ids is None:
        # Note: Only for MixtralForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, axis]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, axis]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, axis]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, axis]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, axis]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, axis]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FusedNormGateFunc(paddle.autograd.PyLayer):
    """recompute of postnorm and gate"""

    _current_norm_output = None
    _current_invar = None

    @classmethod
    def set_temporary_vars(cls, norm_output, invar):
        FusedNormGateFunc._current_norm_output = norm_output
        FusedNormGateFunc._current_invar = invar

    @classmethod
    def clear_temporary_vars(cls):
        FusedNormGateFunc._current_norm_output = None
        FusedNormGateFunc._current_invar = None

    @staticmethod
    def forward(ctx, x, rms_norm_weight, moe_gate_weight, eps):
        ctx.dtype = paddle.float32
        norm_output, invar = paddle.incubate.nn.functional.fused_rms_norm_ext(x, rms_norm_weight, eps)
        with paddle.amp.auto_cast(False):
            gate_logits = F.linear(cast_if_needed(norm_output, ctx.dtype), cast_if_needed(moe_gate_weight, ctx.dtype))

        ctx.save_for_backward(x, rms_norm_weight, moe_gate_weight, eps)
        return gate_logits, norm_output

    @staticmethod
    def backward(ctx, d_gate_logits, d_norm_output):
        x, rms_norm_weight, moe_gate_weight, eps = ctx.saved_tensor()
        # recompute rmsnorm
        norm_output = FusedNormGateFunc._current_norm_output
        invar = FusedNormGateFunc._current_invar
        if norm_output is None or invar is None:
            norm_output, invar = paddle.incubate.nn.functional.fused_rms_norm_ext(x, rms_norm_weight, eps)
        d_norm_output_linear, d_moe_gate_weight = paddle._C_ops.matmul_grad(
            cast_if_needed(norm_output, ctx.dtype),
            cast_if_needed(moe_gate_weight, ctx.dtype),
            d_gate_logits,
            False,
            False,
        )
        d_norm_output_linear, d_moe_gate_weight = cast_if_needed(
            d_norm_output_linear, norm_output.dtype
        ), cast_if_needed(d_moe_gate_weight, moe_gate_weight.dtype)

        d_norm_output = d_norm_output + d_norm_output_linear
        dx, d_rms_norm_weight = paddle._C_ops.fused_rms_norm_ext_grad(x, rms_norm_weight, invar, d_norm_output, eps)

        return dx, d_rms_norm_weight, d_moe_gate_weight


class TemporaryVarContext:
    def __init__(self, norm_output, invar):
        self.norm_output = norm_output
        self.invar = invar

    def __enter__(self):
        FusedNormGateFunc.set_temporary_vars(self.norm_output, self.invar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        FusedNormGateFunc.clear_temporary_vars()


def balance_expert_assignment(n, m, k):
    assert k * n % m == 0
    matrix = paddle.zeros((n, m), dtype=paddle.int32)
    for row in range(n):
        start_col = row % m
        for i in range(k):
            col = (start_col + i) % m
            matrix[row, col] = 1
    return matrix


class FakeGate(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, hidden_states, weight, fakse_gate_restrict_balance=False, num_experts_per_tok=8):
        expert_num = weight.shape[1]
        bsz, seq, _ = hidden_states.shape

        ctx.x_shape = hidden_states.shape
        ctx.x_dtype = hidden_states.dtype
        ctx.y_shape = weight.shape
        ctx.y_dtype = weight.dtype
        if fakse_gate_restrict_balance:
            return paddle.reshape(
                balance_expert_assignment(bsz * seq, expert_num, num_experts_per_tok), [bsz, seq, expert_num]
            )
        else:
            return paddle.randn([bsz, seq, expert_num]).cast(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return paddle.zeros(ctx.x_shape, dtype=ctx.x_dtype), paddle.zeros(ctx.y_shape, dtype=ctx.y_dtype)


class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = not loss.stop_gradient
        return x.clone()  # clone to avoid inplace problem when using overlap

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = paddle.ones(1, dtype=ctx.dtype)
        return grad_output, grad_loss


@to_static(backend="CINN")
def qkv_pre_process_no_fuse(
    q, kv, k_pe, rotary_emb, num_heads, q_head_dim, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, position_ids
):
    bsz, q_len, _ = q.shape

    target_query_shape = [0, 0, num_heads, q_head_dim]
    target_key_value_shape = [0, 0, num_heads, qk_nope_head_dim + v_head_dim]

    q = q.reshape(shape=target_query_shape)
    q_nope = q[..., :qk_nope_head_dim]
    q_pe = q[..., qk_nope_head_dim:]

    # DeepSeekV2 kv_lora_rank+qk_rope_head_dim=512+64

    kv = kv.reshape(shape=target_key_value_shape)

    k_pe = k_pe.reshape([-1, q_len, 1, qk_rope_head_dim]).expand([-1, q_len, num_heads, qk_rope_head_dim])

    # self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim = 128+64
    # self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim) = config.qk_nope_head_dim + self.v_head_dim = 128+128
    k_nope = kv[..., :qk_nope_head_dim]
    value_states = kv[..., qk_nope_head_dim:]

    kv_seq_len = value_states.shape[1]

    cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, False)

    query_states = paddle.cat([q_nope, q_pe], axis=-1)
    key_states = paddle.cat([k_nope, k_pe], axis=-1)

    return query_states, key_states, value_states


@to_static(backend="CINN")
def rearrange_kv(kv, k_pe, qk_nope_head_dim, num_heads):
    k_nope = kv[..., :qk_nope_head_dim]
    value_states = kv[..., qk_nope_head_dim:]

    k_pe = k_pe.expand([k_pe.shape[0], k_pe.shape[1], num_heads, k_pe.shape[3]])
    key_states = paddle.cat([k_nope, k_pe], axis=-1)

    return key_states, value_states


def qkv_pre_process(
    q, kv, k_pe, rotary_emb, num_heads, q_head_dim, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, position_ids
):
    if (fused_partial_rope is None) or (position_ids is not None):
        return qkv_pre_process_no_fuse(
            q,
            kv,
            k_pe,
            rotary_emb,
            num_heads,
            q_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            qk_rope_head_dim,
            position_ids,
        )

    bsz, q_len, _ = q.shape

    target_query_shape = [0, 0, num_heads, q_head_dim]
    target_key_value_shape = [0, 0, num_heads, qk_nope_head_dim + v_head_dim]

    q = q.reshape(shape=target_query_shape)
    kv = kv.reshape(shape=target_key_value_shape)
    k_pe = k_pe.reshape([-1, q_len, 1, qk_rope_head_dim])

    value_states = kv[..., qk_nope_head_dim:]

    kv_seq_len = value_states.shape[1]

    cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    query_states = fused_partial_rope(q, cos, sin)
    k_pe = fused_partial_rope(k_pe, cos, sin)

    key_states, value_states = rearrange_kv(kv, k_pe, qk_nope_head_dim, num_heads)

    return query_states, key_states, value_states


def manul_fwd(
    q_init,
    kv_init,
    q_ln_weight,
    kv_ln_weight,
    q_up_weight,
    kv_up_weight,
    rotary_emb,
    num_heads,
    q_head_dim,
    qk_nope_head_dim,
    v_head_dim,
    qk_rope_head_dim,
    position_ids,
    eps,
    kv_lora_rank,
    softmax_scale,
):

    q_ln_t, q_ln_invar = paddle.incubate.nn.functional.fused_rms_norm_ext(q_init, q_ln_weight, eps)
    q = paddle.matmul(q_ln_t, q_up_weight)

    compressed_kv, k_pe = paddle.split(kv_init, [kv_lora_rank, qk_rope_head_dim], axis=-1)

    kv_ln_t, kv_ln_invar = paddle.incubate.nn.functional.fused_rms_norm_ext(compressed_kv, kv_ln_weight, eps)

    kv = paddle.matmul(kv_ln_t, kv_up_weight)

    query_states, key_states, value_states = qkv_pre_process(
        q, kv, k_pe, rotary_emb, num_heads, q_head_dim, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, position_ids
    )

    q_head_dim = query_states.shape[-1]
    softmax_scale = softmax_scale * (q_head_dim**0.5)
    query_states = query_states * softmax_scale

    attn_out, _, softmax_lse, seed_offset = _C_ops.flash_attn(
        query_states,
        key_states,
        query_states,
        None,
        None,
        0.0,
        True,
        False,
        False,
        "",
    )

    return attn_out


class MemroyRecomputeAttnFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        q_init,
        kv_init,
        q_ln_weight,
        kv_ln_weight,
        q_up_weight,
        kv_up_weight,
        rotary_emb,
        num_heads,
        q_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        qk_rope_head_dim,
        position_ids,
        eps,
        kv_lora_rank,
        softmax_scale,
        custom_map,
        recompute_fa3=False,
        fa_version=3,
    ):

        bsz = q_init.shape[0]
        q_ln_t, q_ln_invar = paddle.incubate.nn.functional.fused_rms_norm_ext(q_init, q_ln_weight, eps)
        # q = paddle.matmul(q_ln_t, q_up_weight)
        q_orig_shape = q_ln_t.shape
        q = FP8LinearFunctionBase.compute_fp8_linear(
            q_ln_t.reshape([-1, q_orig_shape[-1]]), q_up_weight, weight_transpose=True, return_transpose_only=True
        )
        q = q.reshape(q_orig_shape[:-1] + [q_up_weight.shape[-1]])

        compressed_kv, k_pe = paddle.split(kv_init, [kv_lora_rank, qk_rope_head_dim], axis=-1)

        kv_ln_t, kv_ln_invar = paddle.incubate.nn.functional.fused_rms_norm_ext(compressed_kv, kv_ln_weight, eps)
        # kv = paddle.matmul(kv_ln_t, kv_up_weight)
        kv_orig_shape = kv_ln_t.shape
        kv = FP8LinearFunctionBase.compute_fp8_linear(
            kv_ln_t.reshape([-1, kv_orig_shape[-1]]), kv_up_weight, weight_transpose=True, return_transpose_only=True
        )
        kv = kv.reshape(kv_orig_shape[:-1] + [kv_up_weight.shape[-1]])

        query_states, key_states, value_states = qkv_pre_process(
            q,
            kv,
            k_pe,
            rotary_emb,
            num_heads,
            q_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            qk_rope_head_dim,
            position_ids,
        )

        q_head_dim = query_states.shape[-1]

        if fa_version == 2:
            softmax_scale = softmax_scale * (q_head_dim**0.5)
            query_states = query_states * softmax_scale
            kv_seq_len = value_states.shape[1]
            v_num_heads = value_states.shape[2]
            value_padding = paddle.zeros(
                [bsz, kv_seq_len, v_num_heads, q_head_dim - v_head_dim],
                dtype=value_states.dtype,
            )
            value_states_pad = paddle.cat([value_states, value_padding], axis=-1)

            attn_out, _, softmax_lse, seed_offset = _C_ops.flash_attn(
                query_states,
                key_states,
                value_states_pad,
                None,
                None,
                0.0,
                True,
                False,
                False,
                "",
            )

        elif fa_version == 3:
            attn_out, softmax_lse = _C_ops.flash_attn_v3(
                query_states,
                key_states,
                value_states,
                None,  # q_v_
                None,  # q_descale_
                None,  # k_descale_
                None,  # v_descale_
                softmax_scale,
                True,
                -1,  # window_size_left
                -1,  # window_size_right
                0.0,  # softcap
                1,  # num_splits
                False,  # manual_set_pack_gqa
                False,  # pack_gqa_
                0,  # sm_margin
            )
        else:
            assert False, f"invalid {fa_version=}"

        if fa_version == 2:
            ctx.save_for_backward(
                q_init,
                kv_init,
                attn_out,
                softmax_lse,
                seed_offset,
                q_ln_weight,
                kv_ln_weight,
                q_up_weight,
                kv_up_weight,
                rotary_emb,
                num_heads,
                q_head_dim,
                qk_nope_head_dim,
                v_head_dim,
                qk_rope_head_dim,
                position_ids,
                eps,
                kv_lora_rank,
                softmax_scale,
            )
        elif fa_version == 3:
            if recompute_fa3:
                ctx.save_for_backward(
                    q_init,
                    kv_init,
                    None,
                    None,
                    q_ln_weight,
                    kv_ln_weight,
                    q_up_weight,
                    kv_up_weight,
                    rotary_emb,
                    num_heads,
                    q_head_dim,
                    qk_nope_head_dim,
                    v_head_dim,
                    qk_rope_head_dim,
                    position_ids,
                    eps,
                    custom_map.q_lens,
                    custom_map.out_proj_weight,
                    kv_lora_rank,
                    softmax_scale,
                    recompute_fa3,
                )
            else:
                ctx.save_for_backward(
                    q_init,
                    kv_init,
                    attn_out,
                    softmax_lse,
                    q_ln_weight,
                    kv_ln_weight,
                    q_up_weight,
                    kv_up_weight,
                    rotary_emb,
                    num_heads,
                    q_head_dim,
                    qk_nope_head_dim,
                    v_head_dim,
                    qk_rope_head_dim,
                    position_ids,
                    eps,
                    custom_map.q_lens,
                    custom_map.out_proj_weight,
                    kv_lora_rank,
                    softmax_scale,
                    recompute_fa3,
                )
        else:
            assert False, f"invalid {fa_version=}"

        ctx.fa_version = fa_version

        if recompute_fa3:
            attn_out_reshape_shape = [bsz, custom_map.q_lens, -1]
            # deep_gemm only support 2D
            attn_out_reshape = attn_out.reshape([bsz * custom_map.q_lens, -1]).contiguous()
            out = FP8LinearFunctionBase.compute_fp8_linear(
                attn_out_reshape,
                custom_map.out_proj_weight,
                weight_transpose=True,
                return_transpose_only=True,
            )
            out = out.reshape([attn_out_reshape_shape[0], -1, custom_map.out_proj_weight.shape[-1]])
            return out

        return attn_out

    @staticmethod
    def backward(ctx, dout):
        fa_version = ctx.fa_version
        if fa_version == 2:
            (
                q_init,
                kv_init,
                attn_out,
                softmax_lse,
                seed_offset,
                q_ln_weight,
                kv_ln_weight,
                q_up_weight,
                kv_up_weight,
                rotary_emb,
                num_heads,
                q_head_dim,
                qk_nope_head_dim,
                v_head_dim,
                qk_rope_head_dim,
                position_ids,
                eps,
                kv_lora_rank,
                softmax_scale,
            ) = ctx.saved_tensor()
        elif fa_version == 3:
            (
                q_init,
                kv_init,
                attn_out,
                softmax_lse,
                q_ln_weight,
                kv_ln_weight,
                q_up_weight,
                kv_up_weight,
                rotary_emb,
                num_heads,
                q_head_dim,
                qk_nope_head_dim,
                v_head_dim,
                qk_rope_head_dim,
                position_ids,
                eps,
                q_lens,
                out_proj_weight,
                kv_lora_rank,
                softmax_scale,
                recompute_fa3,
            ) = ctx.saved_tensor()
        else:
            assert False, f"invalid {fa_version=}"

        if fa_version == 2:
            assert "recompute_fa3" not in locals()
            assert attn_out is not None and softmax_lse is not None
        if fa_version == 3 and not recompute_fa3:
            assert attn_out is not None and softmax_lse is not None

        q_ln_t, q_ln_invar = paddle.incubate.nn.functional.fused_rms_norm_ext(q_init, q_ln_weight, eps)

        q_ln_fp8, q_ln_scale, q_ln_trans_fp8, q_ln_trans_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            q_ln_t.reshape([-1, q_ln_t.shape[-1]]),
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=True,
        )

        q_orig_shape = q_ln_t.shape
        q = FP8LinearFunctionBase.compute_fp8_linear(
            (q_ln_fp8, q_ln_scale), q_up_weight, weight_transpose=True, return_transpose_only=True
        )
        q = q.reshape(q_orig_shape[:-1] + [q_up_weight.shape[-1]])

        compressed_kv, k_pe = paddle.split(kv_init, [kv_lora_rank, qk_rope_head_dim], axis=-1)

        kv_ln_t, kv_ln_invar = paddle.incubate.nn.functional.fused_rms_norm_ext(compressed_kv, kv_ln_weight, eps)

        kv_ln_fp8, kv_ln_scale, kv_ln_trans_fp8, kv_ln_trans_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            kv_ln_t.reshape([-1, kv_ln_t.shape[-1]]),
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=True,
        )
        kv_orig_shape = kv_ln_t.shape
        kv = FP8LinearFunctionBase.compute_fp8_linear(
            (kv_ln_fp8, kv_ln_scale), kv_up_weight, weight_transpose=True, return_transpose_only=True
        )
        kv = kv.reshape(kv_orig_shape[:-1] + [kv_up_weight.shape[-1]])

        paddle.base.core._set_has_grad(True)
        q.stop_gradient = False
        kv.stop_gradient = False
        k_pe.stop_gradient = False
        query_states, key_states, value_states = qkv_pre_process(
            q,
            kv,
            k_pe,
            rotary_emb,
            num_heads,
            q_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            qk_rope_head_dim,
            position_ids,
        )

        if fa_version == 2:
            q_head_dim = query_states.shape[-1]
            query_states = query_states * softmax_scale

            bsz = value_states.shape[0]
            kv_seq_len = value_states.shape[1]
            v_num_heads = value_states.shape[2]
            value_padding = paddle.zeros(
                [bsz, kv_seq_len, v_num_heads, q_head_dim - v_head_dim],
                dtype=value_states.dtype,
            )
            value_states_pad = paddle.cat([value_states, value_padding], axis=-1)

            with paddle.no_grad():

                q_grad, k_grad, v_grad = _C_ops.flash_attn_grad(
                    query_states,
                    key_states,
                    value_states_pad,
                    attn_out,
                    softmax_lse.view("bfloat16"),
                    seed_offset,
                    None,
                    dout,
                    0.0,
                    True,
                )

                v_grad = v_grad[..., :v_head_dim]
                q_grad = q_grad * softmax_scale
        elif fa_version == 3:
            # recompute fa3
            if recompute_fa3:
                with paddle.no_grad():
                    attn_out, softmax_lse = _C_ops.flash_attn_v3(
                        query_states,
                        key_states,
                        value_states,
                        None,  # q_v_
                        None,  # q_descale_
                        None,  # k_descale_
                        None,  # v_descale_
                        softmax_scale,
                        True,
                        -1,  # window_size_left
                        -1,  # window_size_right
                        0.0,  # softcap
                        1,  # num_splits
                        False,  # manual_set_pack_gqa
                        False,  # pack_gqa_
                        0,  # sm_margin
                    )

                    # padding x and quant
                    bsz = value_states.shape[0]
                    attn_out_reshape = attn_out.reshape([bsz * q_lens, -1]).contiguous()
                    d_attn_out_reshape_shape = attn_out_reshape.shape
                    attn_out_reshape = FP8LinearFunctionBase.padding(attn_out_reshape, 0)
                    (
                        attn_out_reshape_t_fp8,
                        attn_out_reshape_t_scale,
                    ) = paddle.incubate.nn.functional.fp8_quant_blockwise(
                        attn_out_reshape,
                        output_scale_transpose=True,
                        quant_method="1x128",
                        input_transpose=True,
                        return_transpose_only=True,
                    )
                    dout_2d = dout.reshape([-1, dout.shape[-1]])

                    # ===== dx = deep_gemm(dout_fp8, w_fp8)
                    d_attn_out, dout_t_fp8, dout_t_scale = FP8LinearFunctionBase.compute_fp8_linear(
                        dout_2d, out_proj_weight, weight_transpose=False, return_mode="with_input_transpose_quant"
                    )
                    d_attn_out = d_attn_out.reshape(d_attn_out_reshape_shape)
                    FP8LinearFunctionBase.compute_expert_w_grad(
                        attn_out_reshape_t_fp8,
                        attn_out_reshape_t_scale,
                        dout_t_fp8,
                        dout_t_scale,
                        True,
                        True,
                        out_proj_weight,
                        paddle.float32,
                    )

                    dout = d_attn_out
                    dout = dout.reshape(attn_out.shape)

            with paddle.no_grad():
                q_grad, k_grad, v_grad = _C_ops.flash_attn_v3_grad(
                    query_states,
                    key_states,
                    value_states,
                    attn_out,
                    softmax_lse.view("bfloat16"),
                    dout,
                    softmax_scale,
                    True,
                    -1,
                    -1,
                    0.0,
                    0,
                )
        else:
            assert False, f"invalid {fa_version=}"

        d_q, d_kv, d_k_pe = paddle.grad(
            outputs=[query_states, key_states, value_states],
            inputs=[q, kv, k_pe],
            grad_outputs=[q_grad, k_grad, v_grad],
            create_graph=False,
            retain_graph=False,
        )

        paddle.base.core._set_has_grad(False)

        # call up proj
        if hasattr(kv_up_weight, "main_grad"):
            d_kv_fp8, d_kv_scale, d_kv_t_fp8, d_kv_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                d_kv.reshape([-1, d_kv.shape[-1]]),
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=True,
            )

            d_kv_ln_t = FP8LinearFunctionBase.compute_fp8_linear(
                (d_kv_fp8, d_kv_scale), kv_up_weight, weight_transpose=False
            )
            d_kv_ln_t = d_kv_ln_t.reshape(d_kv.shape[:-1] + [kv_up_weight.shape[0]])

            def kv_up_weight_grad(kv_ln_trans_fp8, kv_ln_trans_scale, d_kv_t_fp8, d_kv_t_scale, kv_up_weight):
                FP8LinearFunctionBase.kitchen_gemm(
                    kv_ln_trans_fp8,
                    kv_ln_trans_scale,
                    d_kv_t_fp8,
                    d_kv_t_scale,
                    True,
                    True,
                    kv_up_weight.main_grad,
                    paddle.float32,
                )

            if WeightGradStore.enabled:

                WeightGradStore.put(
                    partial(
                        kv_up_weight_grad, kv_ln_trans_fp8, kv_ln_trans_scale, d_kv_t_fp8, d_kv_t_scale, kv_up_weight
                    )
                )
            else:
                kv_up_weight_grad(kv_ln_trans_fp8, kv_ln_trans_scale, d_kv_t_fp8, d_kv_t_scale, kv_up_weight)

            d_kv_up_weight = None

        else:
            d_kv_ln_t, d_kv_up_weight = _C_ops.matmul_grad(kv_ln_t, kv_up_weight, d_kv, False, False)

        d_compressed_kv, d_kv_ln_weight = paddle._C_ops.fused_rms_norm_ext_grad(
            compressed_kv, kv_ln_weight, kv_ln_invar, d_kv_ln_t, eps
        )

        d_kv_init = paddle.cat([d_compressed_kv, d_k_pe], axis=-1)

        if hasattr(q_up_weight, "main_grad"):

            d_q_fp8, d_q_scale, d_q_t_fp8, d_q_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                d_q.reshape([-1, d_q.shape[-1]]),
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=True,
            )
            # d_q_ln_t = paddle.matmul(d_q, q_up_weight, transpose_y=True)

            d_q_ln_t = FP8LinearFunctionBase.compute_fp8_linear(
                (d_q_fp8, d_q_scale), q_up_weight, weight_transpose=False
            )
            d_q_ln_t = d_q_ln_t.reshape(d_q.shape[:-1] + [q_up_weight.shape[0]])

            def q_up_weight_grad(q_ln_trans_fp8, q_ln_trans_scale, d_q_t_fp8, d_q_t_scale, q_up_weight):
                FP8LinearFunctionBase.kitchen_gemm(
                    q_ln_trans_fp8,
                    q_ln_trans_scale,
                    d_q_t_fp8,
                    d_q_t_scale,
                    True,
                    True,
                    q_up_weight.main_grad,
                    paddle.float32,
                )

            if WeightGradStore.enabled:
                WeightGradStore.put(
                    partial(q_up_weight_grad, q_ln_trans_fp8, q_ln_trans_scale, d_q_t_fp8, d_q_t_scale, q_up_weight)
                )
            else:
                q_up_weight_grad(q_ln_trans_fp8, q_ln_trans_scale, d_q_t_fp8, d_q_t_scale, q_up_weight)

            d_q_up_weight = None

        else:
            d_q_ln_t, d_q_up_weight = _C_ops.matmul_grad(q_ln_t, q_up_weight, d_q, False, False)

        d_q_init, d_q_ln_weight = paddle._C_ops.fused_rms_norm_ext_grad(q_init, q_ln_weight, q_ln_invar, d_q_ln_t, eps)

        return d_q_init, d_kv_init, d_q_ln_weight, d_kv_ln_weight, d_q_up_weight, d_kv_up_weight


class MemroyRecomputeAttn(paddle.nn.Layer):
    def __init__(
        self,
        q_norm_hidden_size,
        kv_norm_hidden_size,
        q_up_in_dim,
        q_up_out_dim,
        kv_up_in_dim,
        kv_up_out_dim,
        rotary_emb,
        num_heads,
        q_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        qk_rope_head_dim,
        eps,
        q_lens,
        out_proj_weight,
        kv_lora_rank,
        softmax_scale,
        recompute_fa3=False,
        fa_version=3,
    ) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.q_ln_weight = paddle.create_parameter(
            shape=[q_norm_hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.kv_ln_weight = paddle.create_parameter(
            shape=[kv_norm_hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )

        self.q_up_weight = self.create_parameter(
            shape=[q_up_in_dim, q_up_out_dim],
            dtype=self._dtype,
            is_bias=False,
        )

        self.kv_up_weight = self.create_parameter(
            shape=[kv_up_in_dim, kv_up_out_dim],
            dtype=self._dtype,
            is_bias=False,
        )
        (
            self.rotary_emb,
            self.num_heads,
            self.q_head_dim,
            self.qk_nope_head_dim,
            self.v_head_dim,
            self.qk_rope_head_dim,
            self.eps,
            self.q_lens,
            self.out_proj_weight,
            self.kv_lora_rank,
            self.softmax_scale,
            self.recompute_fa3,
            self.fa_version,
        ) = (
            rotary_emb,
            num_heads,
            q_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            qk_rope_head_dim,
            eps,
            q_lens,
            out_proj_weight,
            kv_lora_rank,
            softmax_scale,
            recompute_fa3,
            fa_version,
        )
        set_parameter_color([self.q_up_weight, self.kv_up_weight], "memory_attn")

    def fp8_quant_weight(self, quant_transpose=None):
        cache_fp8_weight(self.q_up_weight, quant_transpose=quant_transpose)
        cache_fp8_weight(self.kv_up_weight, quant_transpose=quant_transpose)

    def forward(self, q_init, kv_init, position_ids):

        seq_len = q_init.shape[1]

        if self.rotary_emb.max_seq_len_cached is None or seq_len > self.rotary_emb.max_seq_len_cached:
            self.rotary_emb._set_cos_sin_cache(seq_len)

        return MemroyRecomputeAttnFunc.apply(
            q_init,
            kv_init,
            self.q_ln_weight,
            self.kv_ln_weight,
            self.q_up_weight,
            self.kv_up_weight,
            self.rotary_emb,
            self.num_heads,
            self.q_head_dim,
            self.qk_nope_head_dim,
            self.v_head_dim,
            self.qk_rope_head_dim,
            position_ids,
            self.eps,
            self.kv_lora_rank,
            self.softmax_scale,
            self,
            recompute_fa3=self.recompute_fa3,
            fa_version=self.fa_version,
        )


class FusedRMSLinearFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, rms_norm_weight, q_down_weight, kv_down_weight, eps):

        hidden_states, invar = paddle.incubate.nn.functional.fused_rms_norm_ext(x, rms_norm_weight, eps)

        h_fp8, h_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            hidden_states.reshape([-1, hidden_states.shape[-1]]), output_scale_transpose=True, quant_method="1x128"
        )

        h_orig_shape = hidden_states.shape
        q = FP8LinearFunctionBase.compute_fp8_linear(
            (h_fp8, h_scale), q_down_weight, weight_transpose=True, return_transpose_only=True
        )
        q = q.reshape(h_orig_shape[:-1] + [q_down_weight.shape[-1]])

        kv = paddle.matmul(hidden_states, kv_down_weight)

        ctx.save_for_backward(x, rms_norm_weight, q_down_weight, kv_down_weight)
        ctx.eps = eps
        return q, kv

    @staticmethod
    def backward(ctx, d_q, d_kv):
        x, rms_norm_weight, q_down_weight, kv_down_weight = ctx.saved_tensor()
        eps = ctx.eps
        hidden_states, invar = paddle.incubate.nn.functional.fused_rms_norm_ext(x, rms_norm_weight, eps)

        h_t_fp8, h_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            hidden_states.reshape([-1, hidden_states.shape[-1]]),
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=True,
            return_transpose_only=True,
        )

        h_grad, d_kv_down_weight = _C_ops.matmul_grad(hidden_states, kv_down_weight, d_kv, False, False)

        if hasattr(q_down_weight, "main_grad"):
            d_q_fp8, d_q_scale, d_q_t_fp8, d_q_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                d_q.reshape([-1, d_q.shape[-1]]),
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=True,
            )
            h_grad_0 = FP8LinearFunctionBase.compute_fp8_linear(
                (d_q_fp8, d_q_scale), q_down_weight, weight_transpose=False
            )
            h_grad = h_grad + h_grad_0

            def q_down_weight_grad(h_t_fp8, h_t_scale, d_q_t_fp8, d_q_t_scale, q_down_weight):
                FP8LinearFunctionBase.kitchen_gemm(
                    h_t_fp8, h_t_scale, d_q_t_fp8, d_q_t_scale, True, True, q_down_weight.main_grad, paddle.float32
                )

            if WeightGradStore.enabled:
                WeightGradStore.put(
                    partial(q_down_weight_grad, h_t_fp8, h_t_scale, d_q_t_fp8, d_q_t_scale, q_down_weight)
                )
            else:
                q_down_weight_grad(h_t_fp8, h_t_scale, d_q_t_fp8, d_q_t_scale, q_down_weight)

            d_q_down_weight = None

        else:
            h_grad_0, d_q_down_weight = _C_ops.matmul_grad(hidden_states, q_down_weight, d_q, False, False)
            h_grad = h_grad + h_grad_0

        dx, d_rms_norm_weight = paddle._C_ops.fused_rms_norm_ext_grad(x, rms_norm_weight, invar, h_grad, eps)

        return dx, d_rms_norm_weight, d_q_down_weight, d_kv_down_weight


class FusedRMSLinear(paddle.nn.Layer):
    def __init__(self, hidden_size, q_out_dim, kv_outdim, eps=1e-6) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.rms_norm_weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )

        self.q_down_weight = self.create_parameter(
            shape=[hidden_size, q_out_dim],
            dtype=self._dtype,
            is_bias=False,
        )

        self.kv_down_weight = self.create_parameter(
            shape=[hidden_size, kv_outdim],
            dtype=self._dtype,
            is_bias=False,
        )
        self.eps = eps
        set_parameter_color([self.q_down_weight], "rms_linear")

    def fp8_quant_weight(self, quant_transpose=None):
        cache_fp8_weight(self.q_down_weight, quant_transpose=quant_transpose)

    def forward(self, x):

        return FusedRMSLinearFunc.apply(x, self.rms_norm_weight, self.q_down_weight, self.kv_down_weight, self.eps)


class FusedRMSLinearSingleFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, rms_norm_weight, linear_weight, eps):

        hidden_states, invar = paddle.incubate.nn.functional.fused_rms_norm_ext(x, rms_norm_weight, eps)
        q = paddle.matmul(hidden_states, linear_weight)

        ctx.save_for_backward(x, rms_norm_weight, linear_weight, eps)
        return q

    @staticmethod
    def backward(ctx, d_q, d_kv):
        x, rms_norm_weight, linear_weight, eps = ctx.saved_tensor()
        hidden_states, invar = paddle.incubate.nn.functional.fused_rms_norm_ext(x, rms_norm_weight, eps)

        h_grad, d_linear_weight = _C_ops.matmul_grad(hidden_states, linear_weight, d_q, False, False)

        dx, d_rms_norm_weight = paddle._C_ops.fused_rms_norm_ext_grad(x, rms_norm_weight, invar, h_grad, eps)

        return dx, d_rms_norm_weight, d_linear_weight


class FusedRMSLinearSingle(paddle.nn.Layer):
    def __init__(self, hidden_size, q_out_dim, kv_outdim, eps=1e-6) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.rms_norm_weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )

        self.linear_weight = self.create_parameter(
            shape=[hidden_size, q_out_dim],
            dtype=self._dtype,
            is_bias=False,
        )
        self.eps = eps

    def forward(self, x):

        return FusedRMSLinearFunc.apply(x, self.rms_norm_weight, self.linear_weight, self.eps)


class FastCrossEntropyFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, preds, labels):
        preds = preds.cast(paddle.float32)
        softmax_val, loss = paddle._C_ops.cross_entropy_with_softmax(preds, labels, False, True, True, -100, -1)
        loss = loss.cast(paddle.float32)
        ctx.save_for_backward(labels, softmax_val)
        return loss

    @staticmethod
    def backward(ctx, dout):
        labels, softmax_val = ctx.saved_tensor()

        preds_grad = paddle.incubate.nn.functional.cross_entropy_with_softmax_bwd_w_downcast(
            labels, softmax_val.cast(paddle.float32), dout.cast(paddle.float32)
        )

        return preds_grad, None


class DeepseekV2LMHead(nn.Layer):
    def __init__(self, config: DeepseekV2FastConfig, embedding_weight=None):
        super(DeepseekV2LMHead, self).__init__()
        self.config = config

        if config.num_nextn_predict_layers > 0:
            self.seq_length = config.seq_length - config.num_nextn_predict_layers
        else:
            self.seq_length = config.seq_length

        if config.tensor_model_parallel_size > 1 and config.vocab_size % config.tensor_model_parallel_size == 0:
            vocab_size = config.vocab_size // config.tensor_model_parallel_size
        else:
            vocab_size = config.vocab_size

        if embedding_weight is not None:
            self.transpose_y = True
            self.weight = embedding_weight
        else:
            self.transpose_y = False
            self.weight = self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
                default_initializer=nn.initializer.XavierNormal(1.0),
            )
        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if get_env_device() == "xpu":
            try:
                from paddle_xpu.layers.nn import (  # noqa: F401
                    parallel_matmul as xpu_parallel_matmul,
                )

                self.xpu_parallel_matmul = xpu_parallel_matmul()
            except ImportError:
                self.xpu_parallel_matmul = None

    def forward(self, hidden_states, tensor_parallel_output=None):
        if self.config.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            hidden_states = paddle.reshape_(hidden_states, [-1, self.seq_length, self.config.hidden_size])

        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output

        if get_env_device() == "xpu" and self.xpu_parallel_matmul is not None:
            logits = self.xpu_parallel_matmul(
                hidden_states,
                self.weight,
                transpose_y=False,
                tensor_parallel_output=tensor_parallel_output,
                training=self.training,
            )
        else:
            logits = parallel_matmul(
                hidden_states, self.weight, transpose_y=self.transpose_y, tensor_parallel_output=tensor_parallel_output
            )
        return logits

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, vocab_size={self.weight.shape[1]}, dtype={self.weight.dtype}"
