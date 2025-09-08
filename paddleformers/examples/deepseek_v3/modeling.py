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

from functools import partial

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.jit import to_static
from paddle.utils import try_import

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

from paddle import _C_ops

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

from config.configuration import DeepseekV2FastConfig
from moe_utils import get_env_device
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import WeightGradStore

from paddleformers.transformers.deepseek_v2 import (
    DeepseekV2RotaryEmbedding,
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
)
from paddleformers.transformers.fp8_utils import (
    FP8LinearFunctionBase,
    cache_fp8_weight,
    set_parameter_color,
)
from paddleformers.transformers.utils import device_guard

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

from paddleformers.transformers.deepseek_v2 import rotate_half

__all__ = [
    "DeepseekV2LMHead",
    "set_global_step",
    "get_global_step",
]

global_step = 0


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
        fused_ln = try_import("fused_ln")
        return fused_ln.fused_rms_norm(x_in, w, eps)[0]


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

            emb = paddle.concat((freqs, freqs), axis=-1)
            self.cos_cached = emb.cos() * _mscale
            self.sin_cached = emb.sin() * _mscale


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
        if self.config.use_fused_rms_norm:
            return fusion_rms_norm(hidden_states, self.weight, self.variance_epsilon, self.config.use_fast_layer_norm)

        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.astype("float32")
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, dtype={self.weight.dtype}"


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
        emb = paddle.concat([freqs, freqs], axis=-1)
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

    if (get_env_device() == "xpu" or get_env_device() == "gpu") and fuse_rope:
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
        norm_output, invar = fused_ln.fused_rms_norm(x, rms_norm_weight, eps)
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
            norm_output, invar = fused_ln.fused_rms_norm(x, rms_norm_weight, eps)
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
        dx, d_rms_norm_weight = fused_ln.fused_rms_norm_grad_func(x, rms_norm_weight, invar, d_norm_output, eps)

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

    query_states = paddle.concat([q_nope, q_pe], axis=-1)
    key_states = paddle.concat([k_nope, k_pe], axis=-1)

    return query_states, key_states, value_states


@to_static(backend="CINN")
def rearrange_kv(kv, k_pe, qk_nope_head_dim, num_heads):
    k_nope = kv[..., :qk_nope_head_dim]
    value_states = kv[..., qk_nope_head_dim:]

    k_pe = k_pe.expand([k_pe.shape[0], k_pe.shape[1], num_heads, k_pe.shape[3]])
    key_states = paddle.concat([k_nope, k_pe], axis=-1)

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

    q_ln_t, q_ln_invar = fused_ln.fused_rms_norm(q_init, q_ln_weight, eps)
    q = paddle.matmul(q_ln_t, q_up_weight)

    compressed_kv, k_pe = paddle.split(kv_init, [kv_lora_rank, qk_rope_head_dim], axis=-1)

    kv_ln_t, kv_ln_invar = fused_ln.fused_rms_norm(compressed_kv, kv_ln_weight, eps)

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
        recompute_fa3=False,
        fa_version=3,
    ):

        bsz = q_init.shape[0]
        q_ln_t, q_ln_invar = fused_ln.fused_rms_norm(q_init, q_ln_weight, eps)
        # q = paddle.matmul(q_ln_t, q_up_weight)
        q_orig_shape = q_ln_t.shape
        q = FP8LinearFunctionBase.compute_fp8_linear(
            q_ln_t.reshape([-1, q_orig_shape[-1]]), q_up_weight, weight_transpose=True, return_transpose_only=True
        )
        q = q.reshape(q_orig_shape[:-1] + [q_up_weight.shape[-1]])

        compressed_kv, k_pe = paddle.split(kv_init, [kv_lora_rank, qk_rope_head_dim], axis=-1)

        kv_ln_t, kv_ln_invar = fused_ln.fused_rms_norm(compressed_kv, kv_ln_weight, eps)
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
            value_states_pad = paddle.concat([value_states, value_padding], axis=-1)

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
                    kv_lora_rank,
                    softmax_scale,
                    recompute_fa3,
                )
        else:
            assert False, f"invalid {fa_version=}"

        ctx.fa_version = fa_version

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
                kv_lora_rank,
                softmax_scale,
                recompute_fa3,
            ) = ctx.saved_tensor()
        else:
            assert False, f"invalid {fa_version=}"

        if fa_version == 2:
            assert not recompute_fa3
            assert attn_out is not None and softmax_lse is not None
        if fa_version == 3 and not recompute_fa3:
            assert attn_out is not None and softmax_lse is not None

        q_ln_t, q_ln_invar = fused_ln.fused_rms_norm(q_init, q_ln_weight, eps)

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

        kv_ln_t, kv_ln_invar = fused_ln.fused_rms_norm(compressed_kv, kv_ln_weight, eps)

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
            value_states_pad = paddle.concat([value_states, value_padding], axis=-1)

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

        d_compressed_kv, d_kv_ln_weight = fused_ln.fused_rms_norm_grad_func(
            compressed_kv, kv_ln_weight, kv_ln_invar, d_kv_ln_t, eps
        )

        d_kv_init = paddle.concat([d_compressed_kv, d_k_pe], axis=-1)

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

        d_q_init, d_q_ln_weight = fused_ln.fused_rms_norm_grad_func(q_init, q_ln_weight, q_ln_invar, d_q_ln_t, eps)

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
            recompute_fa3=self.recompute_fa3,
            fa_version=self.fa_version,
        )


class FusedRMSLinearFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, rms_norm_weight, q_down_weight, kv_down_weight, eps):

        hidden_states, invar = fused_ln.fused_rms_norm(x, rms_norm_weight, eps)

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
        hidden_states, invar = fused_ln.fused_rms_norm(x, rms_norm_weight, eps)

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
            FP8LinearFunctionBase.compute_fp8_linear(
                (d_q_fp8, d_q_scale), q_down_weight, weight_transpose=False, out=h_grad.view([-1, h_grad.shape[-1]])
            )

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

        dx, d_rms_norm_weight = fused_ln.fused_rms_norm_grad_func(x, rms_norm_weight, invar, h_grad, eps)

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

        hidden_states, invar = fused_ln.fused_rms_norm(x, rms_norm_weight, eps)
        q = paddle.matmul(hidden_states, linear_weight)

        ctx.save_for_backward(x, rms_norm_weight, linear_weight, eps)
        return q

    @staticmethod
    def backward(ctx, d_q, d_kv):
        x, rms_norm_weight, linear_weight, eps = ctx.saved_tensor()
        hidden_states, invar = fused_ln.fused_rms_norm(x, rms_norm_weight, eps)

        h_grad, d_linear_weight = _C_ops.matmul_grad(hidden_states, linear_weight, d_q, False, False)

        dx, d_rms_norm_weight = fused_ln.fused_rms_norm_grad_func(x, rms_norm_weight, invar, h_grad, eps)

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
        softmax_val, loss = paddle._C_ops.cross_entropy_with_softmax(preds, labels, False, True, False, -100, -1)

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

        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
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
