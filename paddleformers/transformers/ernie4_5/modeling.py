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

"""Paddle Ernie model."""

import contextlib
import functools
import math
from functools import partial
from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import incubate, nn, tensor
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import (
    ParallelCrossEntropy,
    VocabParallelEmbedding,
    get_rng_state_tracker,
)
from paddle.distributed.fleet.utils import recompute

from ...utils.log import logger
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Ernie4_5Config
from .distributed import (
    AllGatherVarlenOp,
    ColumnParallelLinear,
    ColumnSequenceParallelLinear,
    GatherOp,
    RowParallelLinear,
    RowSequenceParallelLinear,
    RRColumnSequenceParallelLinear,
    RRRowSequenceParallelLinear,
    mark_as_sequence_parallel_parameter,
    parallel_matmul,
    sequence_parallel_sparse_mask_labels,
)
from .fusion_ops import (
    Linear,
    fused_rms_norm_ext,
    fused_rope,
    fused_swiglu,
    fusion_flash_attention,
)

# from .loss.dpo import ErnieDPOCriterion
# from .refined_recompute.utils import (
#     RefinedRecomputeFunction,
#     create_skip_config_for_refined_recompute,
# )
from .sequence_parallel_utils import ScatterOp


def calc_lm_head_logits(config, hidden_states, weight, bias, tensor_parallel_output=None, training=True):
    """
    Calculate language model head logits with support for various parallelization strategies.

    This is the core function that computes the final output logits for a language model,
    handling sequence parallelism and tensor parallelism configurations.

    Args:
        config (Ernie4_5Config): Model configuration.
        hidden_states (Tensor): Hidden states from the transformer layers
        weight (Tensor): Weight matrix for the language model head
        bias (Tensor): Bias vector for the language model head
        tensor_parallel_output (bool, optional): Override for tensor parallel output behavior.
                                               If None, uses config.tensor_parallel_output.
                                               Defaults to None.
        training (bool, optional): Whether in training mode. Defaults to True.

    Returns:
        Tensor: The computed logits for language modeling.
    """
    if config.sequence_parallel:
        if not config.use_sparse_head_and_loss_fn:
            hidden_states = GatherOp.apply(hidden_states)
            max_sequence_length = config.max_sequence_length
            hidden_states = hidden_states.reshape([-1, max_sequence_length, hidden_states.shape[-1]])

    if tensor_parallel_output is None:
        tensor_parallel_output = config.tensor_parallel_output
    logits = parallel_matmul(
        hidden_states,
        weight,
        bias=bias,
        transpose_y=config.tie_word_embeddings,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
        fuse_linear=config.fuse_linear,
        training=training,
    )

    return logits


def subbatch(f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}):
    """
    Converts a function to one that applies to subbatch of an input dimension.
    This is useful for processing large tensors in smaller chunks to reduce memory usage.

    Args:
        f (Callable): Original function to be converted to subbatch processing.
        arg_idx ([int]): Indices of the inputs to be subbatched.
        axis ([int]): Indices of the dimensions to be subbatched for each input.
        bs (int): Subbatch size (number of elements to process at once).
        out_idx (int): Index of the output dimension that needs stacking.
        use_recompute (bool, optional): Whether to use recomputation for memory savings. Defaults to False.
        same_arg_idx (dict, optional): Mapping of argument indices that share the same tensor.
                                     e.g. {1: 0} means args[1] == args[0], avoiding duplicate slicing.

    Returns:
        Callable: Converted function that processes inputs in subbatches.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(axis), "Number of batching args and number of batching dims should match."

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = {inp: d for inp, d in zip(inps, axis)}

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert (
                        i > same_arg_idx[i]
                    ), f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice([inp_axis[inp]], [slice_at], [min(inp.shape[inp_axis[inp]], slice_at + bs)])
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(f, *_args, **kwargs)
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.concat(outs, out_idx)

    return wrapper


class FusedDropoutImpl(nn.Layer):
    """
    Fused dropout implementation with residual connection support.

    This layer combines dropout and residual addition in a single operation for better performance,
    particularly on GPU devices. The dropout is conditionally applied based on the probability.

    Args:
        prob (float): Dropout probability (between 0 and 1)
        mode (str): Dropout mode, either 'upscale_in_train' or 'downscale_in_infer'

    Attributes:
        prob (float): Stores the dropout probability
        mode (str): Stores the dropout mode
        dropout (nn.Dropout): The actual dropout layer instance
    """

    def __init__(self, prob, mode):
        """
        Initialize the fused dropout layer.

        Args:
            prob (float): Dropout probability (0 means no dropout)
            mode (str): Dropout mode ('upscale_in_train' or 'downscale_in_infer')
        """
        super().__init__()
        self.prob = prob
        self.mode = mode
        self.dropout = nn.Dropout(p=prob, mode=mode)

    def forward(self, x, y):
        """
        Forward pass of the fused dropout layer.

        Args:
            x (Tensor): Input tensor to potentially apply dropout on
            y (Tensor): Residual tensor to add to the (possibly dropped out) x

        Returns:
            Tensor: Result of x (with optional dropout) + y
        """
        if self.prob > 0:
            x = self.dropout(x)
        output = x + y

        return output


class RMSNorm(nn.Layer):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.

    RMSNorm is a simplified version of LayerNorm that focuses on the root mean square of inputs,
    omitting the mean-centering operation. This provides computational efficiency while maintaining
    good performance.

    """

    def __init__(self, config):
        """
        Initialize RMSNorm layer.

        Args:
            config (Ernie4_5Config): Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, hidden_states):
        """
        Apply RMS normalization to input hidden states.

        Args:
            hidden_states (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Normalized output tensor of same shape as input

        Note:
            - Uses fused kernel if config.fuse_rms_norm is True for better performance
            - Otherwise computes RMSNorm manually:
                1. Compute variance of features
                2. Apply reciprocal square root normalization
                3. Scale by learned weight parameter
            - Maintains original dtype for numerical stability during computation
        """
        if self.config.fuse_rms_norm:
            return fused_rms_norm_ext(hidden_states, self.weight, self.variance_epsilon)[0].astype(self.weight.dtype)
        with paddle.amp.auto_cast(False):
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        return hidden_states.astype(self.weight.dtype) * self.weight


class LayerNorm(nn.LayerNorm):
    """
    Layer Normalization (LayerNorm) implementation with optional optimizations.

    This extends PaddlePaddle's built-in LayerNorm with:
    1. Sequence parallelism support
    2. Fast fused kernel implementation option
    3. Configurable epsilon value

    """

    def __init__(self, config):
        """
        Initialize LayerNorm with configuration.

        Args:
            config (Ernie4_5Config): Model configuration contains normalization parameters and flags.
        """
        super().__init__(config.hidden_size, epsilon=config.rms_norm_eps)
        self.config = config
        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)
            mark_as_sequence_parallel_parameter(self.bias)


class RopeEmbedding(nn.Layer):
    """
    Rotary Position Embedding (RoPE) implementation for transformer models.

    RoPE encodes absolute positional information with rotation matrices and
    naturally incorporates relative position information in self-attention.

    Args:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float, optional): Sequence length compression ratio. Defaults to 1.0.
        base (int, optional): Base value for frequency calculation. Defaults to 10000.

    Attributes:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float): Sequence length compression factor
        base (int): Base value for frequency calculation
    """

    def __init__(self, head_dim, compression_ratio=1.0, base=10000, freq_allocation=0):
        """
        Initialize RoPE embedding layer.

        Args:
            head_dim: Dimension of each attention head
            compression_ratio: Scaling factor for position indices
            base: Base value for frequency calculation
        """
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.base = base

        # num of freq allocated to time
        self.freq_allocation = freq_allocation

    def forward(self, seq_length, position_ids=None):
        """
        Compute rotary position embeddings for given sequence length.

        Args:
            seq_length (int): Maximum sequence length
            position_ids (Tensor, optional): Custom position indices. Defaults to None.

        Returns:
            Tensor: Rotary position embeddings of shape [1, 1, seq_length, head_dim]
        """
        indices = paddle.arange(0, self.head_dim, 2, dtype="float32")
        indices = 1 / self.base ** (indices / self.head_dim)
        if position_ids is None:
            position_ids = paddle.arange(0, seq_length, 1, dtype="float32").unsqueeze(1)
            position_ids = position_ids / self.compression_ratio
            sinusoid_inp = position_ids * indices.unsqueeze(0)
        else:
            position_ids = position_ids / self.compression_ratio
            seq_length = position_ids.shape[-1]
            sinusoid_inp = position_ids.unsqueeze(-1).astype("float32") * indices.unsqueeze(
                0
            )  # [b, s, 1] * [1, d/2] -> [b, s, d/2]
        pos_emb = paddle.concat([paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1)
        pos_emb = paddle.reshape(pos_emb, (-1, 1, seq_length, self.head_dim))
        pos_emb.stop_gradient = True
        return pos_emb

    def apply_rotary(self, rp, q, k):
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            rp (Tensor): Rotary position embeddings
            q (Tensor): Query tensor [batch, heads, seq_len, dim]
            k (Tensor): Key tensor [batch, heads, seq_len, dim]

        Returns:
            Tuple[Tensor, Tensor]: Rotated queries and keys
        """
        # sin [sequence_length, embed_size_per_head//2]
        # cos [sequence_length, embed_size_per_head//2]
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), rp.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), rp.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = paddle.reshape(paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1), paddle.shape(q))
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos), paddle.multiply(rotate_half_q.astype("float32"), sin_pos)
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = paddle.reshape(paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1), paddle.shape(k))
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos), paddle.multiply(rotate_half_k.astype("float32"), sin_pos)
        )
        return query, key


class Ernie4_5MLP(nn.Layer):
    """
    Ernie4_5MLP - Gated Multi-Layer Perceptron module used in Ernie model.
    """

    def __init__(self, config, layer_idx=0):
        """
        Initialize the MLP module with configuration options.

        Args:
            config (Ernie4_5Config): Model configurations.
            layer_idx (int): Index of current layer (default: 0)
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if config.tensor_parallel_degree > 1:
            ColumnLN = ColumnSequenceParallelLinear if config.sequence_parallel else ColumnParallelLinear
            RowLN = RowSequenceParallelLinear if config.sequence_parallel else RowParallelLinear

            column_ln_configs = {}
            if (
                config.recompute
                and config.sequence_parallel
                and config.skip_recompute_ops[layer_idx].get("mlp_column_ln", False)
            ):
                ColumnLN = RRColumnSequenceParallelLinear
                column_ln_configs = {"use_rr": True}
            self.up_gate_proj = ColumnLN(
                self.hidden_size,
                self.intermediate_size * 2,
                gather_output=False,
                has_bias=config.use_bias,
                fuse_matmul_bias=config.fuse_linear,
                **column_ln_configs,
            )
        else:
            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
            self.up_gate_proj = LinearFN(self.hidden_size, self.intermediate_size * 2, bias_attr=config.use_bias)

        if config.tensor_parallel_degree > 1:
            row_ln_configs = {}
            if (
                config.recompute
                and config.sequence_parallel
                and config.skip_recompute_ops[layer_idx].get("mlp_row_ln", False)
            ):
                RowLN = RRRowSequenceParallelLinear
                row_ln_configs = {"use_rr": True}
            self.down_proj = RowLN(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=config.use_bias,
                fuse_matmul_bias=config.fuse_linear,
                **row_ln_configs,
            )
        else:
            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
            self.down_proj = LinearFN(self.intermediate_size, self.hidden_size, bias_attr=config.use_bias)

        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def forward(self, x):
        """
        Forward pass through the MLP module.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]

        Note:
            Implements SwiGLU activation: swish(Wx) * (Vx) where W and V are
            the first and second halves of up_gate_proj output respectively.
        """
        if self.fuse_swiglu:
            x = self.up_gate_proj(x)
            x = fused_swiglu(x)
        else:
            gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
            x = F.silu(gate) * x
        return self.down_proj(x)


class Ernie4_5Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        """Initialize the attention layer.

        Args:
            config (Ernie4_5Config): Model configuration.
            layer_idx (int, optional): Index in transformer stack. Defaults to 0.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if config.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = config.head_dim
        self.is_gqa = config.num_key_value_heads is not None and config.num_key_value_heads != self.num_heads
        if config.fuse_rope:
            assert fused_rope is not None, "fused_rope is not supported"
        self.fuse_rope = config.fuse_rope
        self.freq_allocation = getattr(config, "freq_allocation", 0)

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            if self.is_gqa:
                assert (
                    self.num_key_value_heads % config.tensor_parallel_degree == 0
                ), f"num_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
                self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree
        if self.is_gqa:
            logger.info(f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}")
            assert (
                self.num_heads % self.num_key_value_heads == 0
            ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
            if config.head_dim is None:
                kv_hidden_size = self.hidden_size // self.num_heads * self.num_key_value_heads
            else:
                kv_hidden_size = self.head_dim * config.num_key_value_heads
                q_hidden_size = self.head_dim * config.num_attention_heads
        else:
            q_hidden_size = kv_hidden_size = self.head_dim * config.num_attention_heads

        if config.tensor_parallel_degree > 1:
            column_ln_configs = {}
            ColumnLN = ColumnSequenceParallelLinear if config.sequence_parallel else ColumnParallelLinear
            RowLN = RowSequenceParallelLinear if config.sequence_parallel else RowParallelLinear
            if (
                config.recompute
                and config.sequence_parallel
                and config.skip_recompute_ops[layer_idx].get("attention_column_ln", False)
            ):
                ColumnLN = RRColumnSequenceParallelLinear
                column_ln_configs = {"use_rr": True}

            if config.head_dim is None:
                qkv_hidden_size = self.hidden_size * 3 if not self.is_gqa else self.hidden_size + kv_hidden_size * 2
            else:
                qkv_hidden_size = q_hidden_size + kv_hidden_size * 2
            self.qkv_proj = ColumnLN(
                self.hidden_size,
                qkv_hidden_size,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
                **column_ln_configs,
            )
        else:
            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
            if config.head_dim is None:
                qkv_hidden_size = self.hidden_size * 3 if not self.is_gqa else self.hidden_size + kv_hidden_size * 2
            else:
                qkv_hidden_size = q_hidden_size + kv_hidden_size * 2
            self.qkv_proj = LinearFN(
                self.hidden_size,
                qkv_hidden_size,
                bias_attr=config.use_bias,
            )

        if config.tensor_parallel_degree > 1:
            row_ln_configs = {}
            if (
                config.recompute
                and config.sequence_parallel
                and config.skip_recompute_ops[layer_idx].get("attention_row_ln", False)
            ):
                RowLN = RRRowSequenceParallelLinear
                row_ln_configs = {"use_rr": True}

            self.o_proj = RowLN(
                self.hidden_size if config.head_dim is None else q_hidden_size,
                self.hidden_size,
                has_bias=config.use_bias,
                input_is_parallel=True,
                fuse_matmul_bias=config.fuse_linear,
                **row_ln_configs,
            )
        else:
            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
            self.o_proj = LinearFN(
                self.hidden_size if config.head_dim is None else q_hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )
        self.rotary_emb = RopeEmbedding(
            self.head_dim,
            compression_ratio=config.compression_ratio,
            base=config.rope_theta,
            freq_allocation=self.freq_allocation,
        )
        self.config = config

        self._rr_flash_attn = None
        # if config.recompute and config.skip_recompute_ops[layer_idx].get("flash_attn", False):
        #     self._rr_flash_attn = RefinedRecomputeFunction()

        self.set_attn_func()

    def set_attn_func(self):
        """Configure attention function based on settings.

        Selects between flash/core attention.
        """
        config = self.config
        if config.use_flash_attention:
            self.attn_func = self._flash_attention_wrapper
        else:
            self.attn_func = self.core_attn

        if config.cachekv_quant:
            from paddleslim.common.wrapper_function import FuncWrapper

            self.attn_func = FuncWrapper(self.attn_func)

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_type_ids: Optional[Tuple[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Compute attention outputs.

        Args:
            hidden_states (paddle.Tensor): Input tensor [bsz, seq_len, hidden_size]
            past_key_value (Optional[Tuple[paddle.Tensor, paddle.Tensor]]): Cached key/value states
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            position_ids (Optional[paddle.Tensor]): Position indices for RoPE
            output_attentions (bool): Return attention weights if True
            use_cache (bool): Cache key/value states if True

        Returns:
            Tuple containing:
                - attention_output: [bsz, seq_len, hidden_size]
                - attention_weights: Optional attention probabilities
                - updated_key_value_cache: Optional updated cache
        """
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1]
        if self.config.sequence_parallel:
            if token_type_ids is not None:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_parallel_degree // max_sequence_length
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape
        query_states = key_states = value_states = mix_layer = None
        mix_layer = self.qkv_proj(hidden_states)
        if self.is_gqa:
            query_states, key_states, value_states = paddle.split(
                mix_layer.reshape([bsz, q_len, -1, self.head_dim]),
                [self.num_heads, self.num_key_value_heads, self.num_key_value_heads],
                axis=2,
            )
            mix_layer = None
        else:
            mix_layer = mix_layer.reshape([bsz, q_len, self.num_heads, 3 * self.head_dim])

        if mix_layer is not None:
            has_gradient = not mix_layer.stop_gradient
        else:
            has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if self.config.recompute and self.config.recompute_granularity == "core_attn" and has_gradient:
            assert past_key_value is None, "do not use kv cache in recompute"
            assert not use_cache
            attn_output, attn_weights, past_key_value = recompute(
                self.rope_attn,
                mix_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache,
                attn_mask_start_row_indices,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            attn_output, attn_weights, past_key_value = self.rope_attn(
                mix_layer=mix_layer,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
            )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_wrapper(
        self,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        seq_length=None,
    ):
        """Optimized flash attention implementation.

        Args:
            q (paddle.Tensor): Query tensor
            k (paddle.Tensor): Key tensor
            v (paddle.Tensor): Value tensor
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length indices
            seq_length (Optional[int]): Sequence length

        Returns:
            paddle.Tensor: Attention output tensor
        """
        return fusion_flash_attention(
            q,
            k,
            v,
            self.training,
            self.config.attention_probs_dropout_prob,
            self.config.use_sparse_flash_attn,
            attention_mask,
            attn_mask_start_row_indices,
            seq_length,
            self.config.use_var_len_flash_attn,
            self._rr_flash_attn if self.training else None,
        )

    def core_attn(
        self,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        seq_length=None,
    ):
        """Standard self-attention implementation.

        Args:
            q (paddle.Tensor): Query tensor
            k (paddle.Tensor): Key tensor
            v (paddle.Tensor): Value tensor
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length indices
            seq_length (Optional[int]): Sequence length

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: Attention output and weights
        """
        perm = [
            0,
            2,
            1,
            3,
        ]  # [1, 2, 0, 3] if self.sequence_parallel else [0, 2, 1, 3]
        origin_dtype = q.dtype

        q = tensor.transpose(x=q, perm=perm)
        k = tensor.transpose(x=k, perm=perm)
        v = tensor.transpose(x=v, perm=perm)

        scale_qk_coeff = self.config.scale_qk_coeff * self.head_dim**0.5

        product = paddle.matmul(x=q.scale(1.0 / scale_qk_coeff), y=k, transpose_y=True)

        product = product.cast(paddle.float32)
        if self.config.scale_qk_coeff != 1.0:
            product = product.scale(self.config.scale_qk_coeff)

        if attention_mask is not None:
            attention_mask = attention_mask.cast(paddle.float32)
            if self.config.fuse_softmax_mask:
                weights = incubate.softmax_mask_fuse(product, attention_mask)
            else:
                product = product + attention_mask
                weights = F.softmax(product)
        else:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)

        weights = weights.cast(origin_dtype)

        if self.config.attention_probs_dropout_prob:
            with get_rng_state_tracker().rng_state("local_seed"):
                weights = F.dropout(
                    weights,
                    self.config.attention_probs_dropout_prob,
                    training=self.training,
                    mode="upscale_in_train",
                )

        out = paddle.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        # If sequence_parallel is true, out shape is [s, b, h] after reshape
        # else out shape is [b, s, h]
        out = tensor.reshape(x=out, shape=[0, 0, -1])

        return out, weights

    def rope_attn(
        self,
        mix_layer,
        query_states,
        key_states,
        value_states,
        attention_mask,
        position_ids,
        output_attentions=False,
        past_key_value=None,
        use_cache=False,
        attn_mask_start_row_indices=None,
    ):
        """Attention computation with rotary embeddings.

        Args:
            mix_layer (Optional[paddle.Tensor]): Combined QKV projection
            query_states (paddle.Tensor): Query states
            key_states (paddle.Tensor): Key states
            value_states (paddle.Tensor): Value states
            attention_mask (Optional[paddle.Tensor]): Attention mask
            position_ids (Optional[paddle.Tensor]): Position indices
            output_attentions (bool): Return attention weights
            past_key_value (Optional[Tuple[paddle.Tensor, paddle.Tensor]]): Cached states
            use_cache (bool): Cache new states
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length indices

        Returns:
            Tuple containing:
                - attention_output: Result tensor
                - attention_weights: Optional weights
                - updated_key_value_cache: Optional cache
        """

        if mix_layer is not None:
            query_states, key_states, value_states = paddle.split(mix_layer, 3, axis=-1)
        query_states_dtype = query_states.dtype

        # don't get confused, kv_seq_len is just used to retrieve correct cos_sin
        kv_seq_len = key_states.shape[-3]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-3]
            kv_seq_len += offset

        if offset > 0 or position_ids is not None or not self.fuse_rope:
            cos_sin = self.rotary_emb(kv_seq_len, position_ids).transpose([0, 2, 1, 3])  # [b,h,s,d]->[b,s,h,d]
            if offset > 0 and position_ids is None:
                # position_ids has been sliced in prepare_inputs_for_generation
                cos_sin = cos_sin[:, offset:]
            query_states, key_states = self.rotary_emb.apply_rotary(cos_sin, query_states, key_states)

        else:
            _, _, num_heads, _ = query_states.shape
            _, kv_seq_len, num_key_value_heads, _ = key_states.shape
            if num_heads != num_key_value_heads:
                query_states, _, _ = fused_rope(query_states, None, None, rotary_emb_base=self.config.rope_theta)
                key_states, _, _ = fused_rope(key_states, None, None, rotary_emb_base=self.config.rope_theta)
            else:
                query_states, key_states, _ = fused_rope(
                    query_states, key_states, None, rotary_emb_base=self.config.rope_theta
                )

        query_states = query_states.astype(query_states_dtype)
        key_states = key_states.astype(query_states_dtype)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        # NOTE(for generation): use list instead of tuple to store the cache
        # tensors, so that we can clear the cache tensors for memory efficiency.
        past_key_value = [key_states, value_states] if use_cache else None
        seq_length = query_states.shape[1]
        attn_output, attn_weights = self.attn_func(
            query_states,
            key_states,
            value_states,
            attention_mask,
            attn_mask_start_row_indices,
            seq_length,
        )
        return attn_output, attn_weights, past_key_value


class FusedHeadParallelCrossEntropy(PyLayer):
    """Fused parallel cross-entropy loss computation for large sequence lengths.

    Combines head projection and loss computation with optimized memory usage for long sequences,
    supporting tensor parallel training.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight,
        bias,
        labels,
        tensor_parallel_degree,
        mp_group=None,
        ignore_index=-100,
        seq_chunk_size=8192,
        transpose_y=False,
        fuse_linear=False,
        training=True,
    ):
        """Forward pass for parallel cross-entropy computation.

        Args:
            ctx: Context object for saving tensors between forward/backward
            hidden_states (paddle.Tensor): Input tensor of shape [batch_size*seq_len, hidden_size]
            weight (paddle.Tensor): Weight matrix for projection
            bias (Optional[paddle.Tensor]): Optional bias vector
            labels (paddle.Tensor): Target labels tensor of shape [batch_size*seq_len]
            tensor_parallel_degree (int): Degree of tensor parallelism
            mp_group (Optional[dist.Group]): Model parallel group. Defaults to None (auto-detect)
            ignore_index (int): Index to ignore in loss computation. Defaults to -100
            seq_chunk_size (int): Chunk size for processing long sequences. Defaults to 8192
            transpose_y (bool): Whether to transpose weight matrix. Defaults to False
            fuse_linear (bool): Whether to use fused linear ops. Defaults to False
            training (bool): Whether in training mode. Defaults to True

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - loss: Computed loss tensor
                - gathered_labels: Concatenated labels from all parallel groups
        """

        ctx.tensor_parallel_degree = tensor_parallel_degree
        ctx.ignore_index = ignore_index
        ctx.seq_chunk_size = seq_chunk_size
        ctx.transpose_y = transpose_y
        ctx.fuse_linear = fuse_linear
        ctx.training = training

        ctx.hidden_states_shape = hidden_states.shape

        ctx.mp_group = (
            fleet.get_hybrid_communicate_group().get_model_parallel_group() if mp_group is None else mp_group
        )
        ctx.rank = ctx.mp_group.rank
        ctx.world_size = ctx.mp_group.nranks

        loss_all = []
        labels_all = []
        with paddle.no_grad():
            labels = labels.reshape_([-1])
            hidden_states = hidden_states.reshape_([-1, hidden_states.shape[-1]])

            num_tokens_per_rank = []
            dist.stream.all_gather(
                num_tokens_per_rank, paddle.to_tensor(hidden_states.shape[0], dtype=paddle.int32), group=ctx.mp_group
            )
            ctx.num_tokens_per_rank = num_tokens_per_rank

            for idx in range(ctx.world_size):
                if idx == ctx.rank:
                    hidden_states_recv = hidden_states
                    labels_recv = labels
                else:
                    hidden_states_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]], dtype=hidden_states.dtype
                    )
                    labels_recv = paddle.empty([ctx.num_tokens_per_rank[idx]], dtype=labels.dtype)

                dist.stream.broadcast(hidden_states_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group)
                dist.stream.broadcast(labels_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group)

                seq_len = hidden_states_recv.shape[0]
                num_chunk = (seq_len + ctx.seq_chunk_size - 1) // ctx.seq_chunk_size

                loss_chunk = []
                for chunk_idx in range(num_chunk):
                    start = chunk_idx * ctx.seq_chunk_size
                    end = min(start + ctx.seq_chunk_size, seq_len)
                    hidden_states_chunk = hidden_states_recv._slice(start, end)
                    labels_chunk = labels_recv._slice(start, end)

                    logits = parallel_matmul(
                        hidden_states_chunk,
                        weight,
                        bias=bias,
                        transpose_y=ctx.transpose_y,
                        tensor_parallel_degree=ctx.tensor_parallel_degree,
                        tensor_parallel_output=True,
                        fuse_linear=ctx.fuse_linear,
                        training=ctx.training,
                    )

                    with paddle.amp.auto_cast(False):
                        loss = mp_ops._c_softmax_with_cross_entropy(
                            logits.cast("float32"),
                            labels_chunk.unsqueeze(-1),
                            group=ctx.mp_group,
                            ignore_index=ctx.ignore_index,
                        )
                        loss_chunk.append(loss)
                loss_all.append(paddle.concat(loss_chunk, axis=0))
                labels_all.append(labels_recv)

            ctx.loss_concat_sections = [loss.shape[0] for loss in loss_all]
            loss_all = paddle.concat(loss_all, axis=0)
            labels_all = paddle.concat(labels_all, axis=0)

            tensor_inputs = [hidden_states, weight, bias, labels]
            ctx.save_for_backward(*tensor_inputs)

        return loss_all, labels_all

    @staticmethod
    def backward(ctx, loss_all_grad, labels_all_grad):
        """Backward pass for parallel cross-entropy computation.

        Args:
            ctx: Context object with saved tensors from forward
            loss_all_grad (paddle.Tensor): Gradient of loss
            labels_all_grad (paddle.Tensor): Gradient of labels (unused)

        Returns:
            Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[paddle.Tensor], None]:
                - hidden_states_grad: Gradient for input hidden states
                - weight_grad: Gradient for weight matrix (None if not trainable)
                - bias_grad: Gradient for bias vector (None if not trainable or not provided)
                - None: Placeholder for labels gradient
        """

        hidden_states, weight, bias, labels = ctx.saved_tensor()

        loss_all_grad_list = paddle.split(loss_all_grad, ctx.loss_concat_sections, axis=0)

        def detach_variable(inp):
            if inp is None:
                return None
            x = inp.detach()
            x.stop_gradient = inp.stop_gradient
            return x

        if weight.stop_gradient is False:
            weight_main_grad = paddle.zeros(weight.shape, dtype=paddle.float32)
        else:
            weight_main_grad = None
        if bias is not None and bias.stop_gradient is False:
            bias_main_grad = paddle.zeros(bias.shape, dtype=paddle.float32)
        else:
            bias_main_grad = None

        hidden_states = detach_variable(hidden_states)
        weight = detach_variable(weight)
        bias = detach_variable(bias)
        labels = detach_variable(labels)

        with paddle.base.dygraph.guard():
            tracer = paddle.base.framework._dygraph_tracer()
            tracer._has_grad = True

            for idx in range(ctx.world_size):
                if idx == ctx.rank:
                    hidden_states_recv = hidden_states
                    labels_recv = labels
                else:
                    hidden_states_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]], dtype=hidden_states.dtype
                    )
                    labels_recv = paddle.empty([ctx.num_tokens_per_rank[idx]], dtype=labels.dtype)

                dist.stream.broadcast(hidden_states_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group)
                dist.stream.broadcast(labels_recv, src=ctx.mp_group.ranks[idx], group=ctx.mp_group)
                hidden_states_recv.stop_gradient = False

                seq_len = hidden_states_recv.shape[0]
                num_chunk = (seq_len + ctx.seq_chunk_size - 1) // ctx.seq_chunk_size

                for chunk_idx in range(num_chunk):
                    start = chunk_idx * ctx.seq_chunk_size
                    end = min(start + ctx.seq_chunk_size, seq_len)
                    hidden_states_chunk = hidden_states_recv.slice(axes=[0], starts=[start], ends=[end])
                    labels_chunk = labels_recv._slice(start, end)
                    loss_grad_chunk = loss_all_grad_list[idx]._slice(start, end)

                    logits = parallel_matmul(
                        hidden_states_chunk,
                        weight,
                        bias=bias,
                        transpose_y=ctx.transpose_y,
                        tensor_parallel_degree=ctx.tensor_parallel_degree,
                        tensor_parallel_output=True,
                        fuse_linear=ctx.fuse_linear,
                        training=ctx.training,
                    )

                    with paddle.amp.auto_cast(False):
                        loss_chunk = mp_ops._c_softmax_with_cross_entropy(
                            logits.cast("float32"),
                            labels_chunk.unsqueeze(-1),
                            group=ctx.mp_group,
                            ignore_index=ctx.ignore_index,
                        )

                    with paddle.amp.auto_cast(enable=False):
                        paddle.autograd.backward(loss_chunk, loss_grad_chunk)

                    if weight_main_grad is not None:
                        weight_main_grad.add_(weight.grad.cast(paddle.float32))
                        weight.clear_gradient(True)
                    if bias_main_grad is not None:
                        bias_main_grad.add_(bias.grad.cast(paddle.float32))
                        bias.clear_gradient(True)

                if idx == ctx.rank:
                    hidden_states_grad = hidden_states_recv.grad
                    hidden_states_grad = hidden_states_grad.reshape(ctx.hidden_states_shape)

        if weight_main_grad is not None:
            weight_main_grad = weight_main_grad.astype(weight.dtype)
        if bias_main_grad is not None:
            bias_main_grad = bias_main_grad.astype(bias.dtype)

        return (
            hidden_states_grad,
            weight_main_grad,
            bias_main_grad,
            None,
        )


class ErniePretrainingCriterion(paddle.nn.Layer):
    """Criterion for ERNIE pretraining task."""

    def __init__(self, config, return_tuple=True):
        """Initialize the pretraining criterion.

        Args:
            config (Ernie4_5Config): Model configuration.
            return_tuple (bool): Whether to return loss as tuple (loss, loss_sum). Defaults to True.
        """
        super(ErniePretrainingCriterion, self).__init__()
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:
            logger.info("using parallel cross entroy, take care")
            self.loss_func = ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )
        self.token_balance_loss = config.token_balance_loss

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
        """Compute the pretraining loss.

        Args:
            prediction_scores (Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]):
                Either:
                - Direct logits tensor [batch_size, seq_len, vocab_size]
                - Tuple of (hidden_states, weight, bias) for sparse head computation
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]
            loss_mask (Optional[paddle.Tensor]): Optional mask for valid tokens. Defaults to None.

        Returns:
            Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
                - If return_tuple=False: Single loss tensor
                - If return_tuple=True: Tuple of (normalized_loss, sum_loss)
        """

        if self.config.use_sparse_head_and_loss_fn:
            hidden_states, outlinear_weight, outlinear_bias, _ = prediction_scores

            if self.config.sequence_parallel:
                masked_lm_labels, sparse_label_idx = sequence_parallel_sparse_mask_labels(
                    masked_lm_labels, self.ignored_index
                )
                sparse_label_idx = sparse_label_idx.reshape([-1, 1])
                hidden_states = paddle.gather(hidden_states, sparse_label_idx, axis=0)
                hidden_states = AllGatherVarlenOp.apply(hidden_states)
            else:
                masked_lm_labels = masked_lm_labels.flatten()
                sparse_label_idx = paddle.nonzero(masked_lm_labels != self.ignored_index).flatten()
                masked_lm_labels = paddle.take_along_axis(masked_lm_labels, sparse_label_idx, axis=0)

                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.take_along_axis(hidden_states, sparse_label_idx.reshape([-1, 1]), axis=0)

            # `loss_mask` must be reset to None and re-calculate it in ErnieBotPretrainingCriterion
            # when use use_sparse_head_and_loss_fn.
            loss_mask = None
            if self.config.use_recompute_loss_fn:
                offload_kwargs = {}
                if getattr(self.config, "offload_lm_head", False):
                    offload_kwargs["offload_indices"] = [1]
                res = recompute(
                    self.forward_impl_with_calc_logits,
                    masked_lm_labels,
                    loss_mask,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    **offload_kwargs,
                )
            else:
                logits = calc_lm_head_logits(
                    self.config, hidden_states, outlinear_weight, outlinear_bias, training=self.training
                )
                res = self.forward_impl(logits, masked_lm_labels, loss_mask)
        elif self.config.use_recompute_loss_fn:
            if self.config.use_fused_head_and_loss_fn:
                res = self.forward_impl_with_fused_head_loss_fn(masked_lm_labels, loss_mask, *prediction_scores)
            else:
                assert isinstance(prediction_scores, tuple) and len(prediction_scores) in [3, 4], prediction_scores
                res = recompute(self.forward_impl_with_calc_logits, masked_lm_labels, loss_mask, *prediction_scores)
        else:
            res = self.forward_impl(prediction_scores, masked_lm_labels, loss_mask)

        return res

    def forward_impl_with_fused_head_loss_fn(
        self, masked_lm_labels, loss_mask, hidden_states, outlinear_weight, outlinear_bias
    ):
        """Compute loss with fused head and parallel cross-entropy.

        Args:
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]
            loss_mask (Optional[paddle.Tensor]): Optional mask for valid tokens
            hidden_states (paddle.Tensor): Hidden states from transformer [batch_size, seq_len, hidden_size]
            outlinear_weight (paddle.Tensor): Weight matrix for output projection
            outlinear_bias (Optional[paddle.Tensor]): Optional bias for output projection

        Returns:
            Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
                Same return format as forward()
        """
        assert self.config.tensor_parallel_degree > 0, "use_fused_head_and_loss_fn require tensor_parallel_degree > 0"
        masked_lm_loss, masked_lm_labels_all = FusedHeadParallelCrossEntropy.apply(
            hidden_states,
            outlinear_weight,
            outlinear_bias,
            masked_lm_labels,
            self.config.tensor_parallel_degree,
            ignore_index=self.ignored_index,
            seq_chunk_size=getattr(self.config, "loss_subbatch_seqlen", 32768),
            transpose_y=self.config.tie_word_embeddings,
            fuse_linear=self.config.fuse_linear,
            training=self.training,
        )
        if loss_mask is None:
            loss_mask = masked_lm_labels_all != self.ignored_index
        if (~loss_mask).all():  # empty span
            logger.warning(f"encounter empty span when calculate loss, ignored_index={self.ignored_index}")
            loss = paddle.mean(masked_lm_loss) * 0.0
            loss_sum = masked_lm_loss.sum().detach()
        else:
            loss_mask = loss_mask.reshape([-1]).cast(paddle.float32)
            masked_lm_loss = paddle.sum(masked_lm_loss.cast(paddle.float32).reshape([-1]) * loss_mask)
            loss = masked_lm_loss / loss_mask.sum()
            if self.token_balance_loss:
                _loss = masked_lm_loss / self.config.token_balance_seqlen
                loss = _loss - _loss.detach() + loss.detach()
            loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum

    def forward_impl_with_calc_logits(
        self, masked_lm_labels, loss_mask, hidden_states, outlinear_weight, outlinear_bias
    ):
        """Compute logits then calculate loss.

        Args:
            Same as forward_impl_with_fused_head_loss_fn()

        Returns:
            Same return format as forward()
        """

        logits = calc_lm_head_logits(
            self.config, hidden_states, outlinear_weight, outlinear_bias, training=self.training
        )

        return self.forward_impl(logits, masked_lm_labels, loss_mask)

    def loss_impl(self, prediction_scores, masked_lm_labels):
        """Core loss computation without reduction.

        Args:
            prediction_scores (paddle.Tensor): Logits tensor [batch_size, seq_len, vocab_size]
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]

        Returns:
            paddle.Tensor: Unreduced loss tensor
        """
        prediction_scores = prediction_scores.cast("float32")
        masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(-1))
        return masked_lm_loss

    def forward_impl(self, prediction_scores, masked_lm_labels, loss_mask=None):
        """Standard loss computation with reduction and masking.

        Args:
            prediction_scores (paddle.Tensor): Logits tensor [batch_size, seq_len, vocab_size]
            masked_lm_labels (paddle.Tensor): Target labels tensor [batch_size, seq_len]
            loss_mask (Optional[paddle.Tensor]): Optional mask for valid tokens

        Returns:
            Same return format as forward()
        """
        if self.enable_parallel_cross_entropy:
            assert prediction_scores.shape[-1] != self.config.vocab_size, (
                f"enable_parallel_cross_entropy, the vocab_size should be splited:"
                f" {prediction_scores.shape[-1]}, {self.config.vocab_size}"
            )

        with paddle.amp.auto_cast(False):
            prediction_scores_dims = len(prediction_scores.shape)
            loss_subbatch_seqlen = getattr(self.config, "loss_subbatch_seqlen", 32768)
            if prediction_scores_dims == 2 and prediction_scores.shape[0] > loss_subbatch_seqlen:
                sb_loss_func = subbatch(self.loss_impl, [0, 1], [0, 0], loss_subbatch_seqlen, 0)
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            elif prediction_scores_dims == 3 and prediction_scores.shape[1] > loss_subbatch_seqlen:
                sb_loss_func = subbatch(self.loss_impl, [0, 1], [1, 1], loss_subbatch_seqlen, 1)
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            else:
                masked_lm_loss = self.loss_impl(prediction_scores, masked_lm_labels)

            if loss_mask is None:
                loss_mask = masked_lm_labels != self.ignored_index

            loss_mask = loss_mask.reshape([-1]).cast(paddle.float32)

            masked_lm_loss = paddle.sum(masked_lm_loss.cast(paddle.float32).reshape([-1]) * loss_mask)
            loss = masked_lm_loss / loss_mask.sum()
            if self.token_balance_loss:
                _loss = masked_lm_loss / self.config.token_balance_seqlen
                loss = _loss - _loss.detach() + loss.detach()
            loss_sum = masked_lm_loss.sum().detach()

        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


class Ernie4_5LMHead(nn.Layer):
    """Language model head for ERNIE with support for tensor parallelism."""

    def __init__(self, config):
        """Initialize the language model head.

        Args:
            config (Ernie4_5Config): Model configuration containing:
                - vocab_size: Size of vocabulary
                - hidden_size: Dimension of hidden states
                - tensor_parallel_degree: Degree of tensor parallelism
                - tie_word_embeddings: Whether to tie input/output embeddings
                - weight_share_add_bias: Whether to add bias when weight sharing
                - use_bias: Whether to use bias term
                - use_recompute_loss_fn: Whether to defer logits computation to loss function
                - use_sparse_head_and_loss_fn: Whether to use sparse head computation
        """

        super(Ernie4_5LMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.weight = self.create_parameter(
            shape=[vocab_size, config.hidden_size] if config.tie_word_embeddings else [config.hidden_size, vocab_size],
            dtype=paddle.get_default_dtype(),
        )
        logger.info(f"output-weight:{self.weight.shape} config.tie_word_embeddings={config.tie_word_embeddings}")
        if config.weight_share_add_bias and config.use_bias:
            self.bias = self.create_parameter(
                shape=[vocab_size],
                dtype=paddle.get_default_dtype(),
                attr=paddle.ParamAttr(initializer=paddle.nn.initializer.constant.Constant(0.0)),
            )
        else:
            self.bias = None

        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if config.weight_share_add_bias and config.use_bias:
            self.bias.is_distributed = True if (vocab_size != config.vocab_size) else False

        if self.weight.is_distributed:
            self.weight.split_axis = 1
        if config.weight_share_add_bias and config.use_bias and self.bias.is_distributed:
            self.bias.split_axis = 0

        if self.config.use_recompute_loss_fn:
            logger.info(
                "Using recompute_loss_fn, the calculation of logits will be moved into "
                "loss_fn for memory optimization"
            )

    def forward(self, hidden_states, tensor_parallel_output=None):
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states (paddle.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            tensor_parallel_output (Optional[bool]): Whether to output parallel results. Defaults to None.

        Returns:
            Union[
                Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
                    # When use_recompute_loss_fn or use_sparse_head_and_loss_fn
                    - hidden_states: Original input
                    - weight: Projection weights
                    - bias: Optional bias term
                Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor], bool]:  # With tensor_parallel_output
                    Same as above plus tensor_parallel_output flag
                paddle.Tensor:  # Normal case
                    Logits tensor of shape [batch_size, seq_len, vocab_size]
            ]
        """
        #  will enter this branch when:
        # 1. use_recompute_loss_fn or use_sparse_head_and_loss_fn
        # 2. dpo training
        if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
            return (hidden_states, self.weight, self.bias, self.config.tie_word_embeddings)

        return calc_lm_head_logits(
            self.config, hidden_states, self.weight, self.bias, tensor_parallel_output, training=self.training
        )


class Ernie4_5DecoderLayer(nn.Layer):
    """A single transformer decoder layer in ERNIE model.

    Contains self-attention and feed-forward components,
    support, residual connections, and layer normalization.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (Ernie4_5Config): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config

        self.self_attn = Ernie4_5Attention(config, layer_idx)
        self.mlp = Ernie4_5MLP(config)

        Norm = RMSNorm if config.use_rmsnorm else LayerNorm

        self.input_layernorm = Norm(config)
        self.post_attention_layernorm = Norm(config)

        self.residual_add1 = FusedDropoutImpl(config.hidden_dropout_prob, mode="upscale_in_train")
        self.residual_add2 = FusedDropoutImpl(config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.post_attention_layernorm.weight)
            if not hasattr(config, "disable_ffn_model_parallel"):
                mark_as_sequence_parallel_parameter(self.input_layernorm.weight)
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

            if not config.use_rmsnorm and config.use_bias:
                mark_as_sequence_parallel_parameter(self.post_attention_layernorm.bias)
                mark_as_sequence_parallel_parameter(self.input_layernorm.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (paddle.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Indices for variable length attention
            position_ids (Optional[paddle.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_value (Optional[Tuple[paddle.Tensor]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if self.config.recompute and self.config.recompute_granularity == "full_attn" and has_gradient:
            hidden_states, self_attn_weights, present_key_value = recompute(
                self.self_attn,
                hidden_states,
                past_key_value,
                attention_mask,
                attn_mask_start_row_indices,
                position_ids,
                output_attentions,
                use_cache,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )

        with self.model_parallel_dropout():
            hidden_states = self.residual_add1(hidden_states, residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        with self.model_parallel_dropout():
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def model_parallel_dropout(self):
        """Get context manager for model-parallel dropout with proper seed control.

        Returns:
            Context manager for dropout operation
        """
        if self.config.tensor_parallel_degree > 1 and self.config.hidden_dropout_prob > 0.0:
            current_seed = "local_seed" if self.config.sequence_parallel else "global_seed"
            return get_rng_state_tracker().rng_state(current_seed)
        return contextlib.nullcontext()


class Ernie4_5PretrainedModel(PretrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = Ernie4_5Config
    base_model_prefix = "ernie"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """Generate tensor parallel mappings for model conversion.

        Args:
            config (Ernie4_5Config): Model configuration.
            is_split (bool): Whether to generate split mappings (True)
                            or merge mappings (False). Defaults to True.

        Returns:
            Dict[str, Callable[[Any], Any]]: Dictionary mapping parameter names
                to their corresponding split/merge functions for tensor parallelism.
        """

        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def gqa_qkv_split_func(
            weight,
            tensor_parallel_degree,
            tensor_parallel_rank,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            is_quant=False,
            is_split=True,
        ):
            if is_quant:
                weight = weight.T

            def get_shape(tensor):
                return tensor.get_shape() if hasattr(tensor, "get_shape") else tensor.shape

            def slice_tensor(tensor, start, end):
                shape = get_shape(tensor)
                if len(shape) == 1:
                    return tensor[start:end]
                else:
                    return tensor[..., start:end]

            q_end = num_attention_heads * head_dim
            k_end = q_end + num_key_value_heads * head_dim
            v_end = k_end + num_key_value_heads * head_dim

            q = slice_tensor(weight, 0, q_end)
            k = slice_tensor(weight, q_end, k_end)
            v = slice_tensor(weight, k_end, v_end)

            def split_tensor(tensor, degree):
                shape = get_shape(tensor)
                size = shape[-1]
                block_size = size // degree
                if hasattr(tensor, "get_shape"):
                    return [slice_tensor(tensor, i * block_size, (i + 1) * block_size) for i in range(degree)]
                else:
                    return np.split(tensor, degree, axis=-1)

            q_list = split_tensor(q, tensor_parallel_degree)
            k_list = split_tensor(k, tensor_parallel_degree)
            v_list = split_tensor(v, tensor_parallel_degree)

            if tensor_parallel_rank is None:
                out = [np.concatenate([q_i, k_i, v_i], axis=-1) for q_i, k_i, v_i in zip(q_list, k_list, v_list)]
            else:
                out = np.concatenate(
                    [q_list[tensor_parallel_rank], k_list[tensor_parallel_rank], v_list[tensor_parallel_rank]], axis=-1
                )
            if is_quant:
                out = out.T
            return out

        def gqa_qkv_merge_func(
            weight_list, num_attention_heads, num_key_value_heads, head_dim, is_quant=False, is_split=False
        ):
            tensor_parallel_degree = len(weight_list)
            num_attention_heads = num_attention_heads // tensor_parallel_degree
            num_key_value_heads = num_key_value_heads // tensor_parallel_degree

            is_paddle_tensor = not isinstance(weight_list[0], np.ndarray)

            def get_shape(tensor):
                return tensor.get_shape() if hasattr(tensor, "get_shape") else tensor.shape

            def slice_tensor(tensor, start, end):
                if len(get_shape(tensor)) == 1:
                    return tensor[start:end]
                else:
                    return tensor[..., start:end]

            q_list, k_list, v_list = [], [], []

            for weight in weight_list:
                if is_quant:
                    weight = weight.T
                q_end = num_attention_heads * head_dim
                k_end = q_end + num_key_value_heads * head_dim
                v_end = k_end + num_key_value_heads * head_dim

                q = slice_tensor(weight, 0, q_end)
                k = slice_tensor(weight, q_end, k_end)
                v = slice_tensor(weight, k_end, v_end)

                q_list.append(q)
                k_list.append(k)
                v_list.append(v)

            merged = q_list + k_list + v_list

            if is_paddle_tensor:
                tensor = paddle.concat(merged, axis=-1)
                if tensor.place.is_gpu_place():
                    tensor = tensor._copy_to(paddle.CUDAPinnedPlace(), False)

            else:
                tensor = np.concatenate(merged, axis=-1)
            if is_quant:
                tensor = tensor.T
            return tensor

        if config.num_key_value_heads is not None and config.num_key_value_heads != config.num_attention_heads:
            if is_split:
                qkv_fn = partial(
                    gqa_qkv_split_func,
                    tensor_parallel_degree=config.tensor_parallel_degree,
                    tensor_parallel_rank=config.tensor_parallel_rank,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=(
                        config.hidden_size // config.num_attention_heads
                        if config.head_dim is None
                        else config.head_dim
                    ),
                    is_quant=False,
                    is_split=True,
                )
            else:
                qkv_fn = partial(
                    gqa_qkv_merge_func,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=(
                        config.hidden_size // config.num_attention_heads
                        if config.head_dim is None
                        else config.head_dim
                    ),
                    is_quant=False,
                    is_split=False,
                )
        else:
            qkv_fn = partial(fn, is_column=True)

        def get_tensor_parallel_split_mappings(num_hidden_layers):
            final_actions = {}

            base_actions = {
                # Column Linear
                "layers.0.self_attn.qkv_proj.weight": qkv_fn,
                "layers.0.mlp.up_gate_proj.weight": partial(fn, is_column=True, is_naive_2fuse=True),
                "lm_head.weight": partial(fn, is_column=not config.tie_word_embeddings),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            if config.use_bias:
                base_actions.update(
                    {
                        # Column Linear
                        "layers.0.self_attn.qkv_proj.bias": qkv_fn,
                        "layers.0.mlp.up_gate_proj.bias": partial(fn, is_column=True, is_naive_2fuse=True),
                        "layers.0.mlp.down_proj.bias": lambda x: x[:],  # convert PySafeSlice to ndarray.
                        "lm_head.bias": partial(fn, is_column=True),
                    }
                )

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_hidden_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                else:
                    final_actions[key] = action
            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        return mappings


@register_base_model
class Ernie4_5Model(Ernie4_5PretrainedModel):
    """The core ERNIE transformer model"""

    def __init__(self, config: Ernie4_5Config):
        """Initialize the ERNIE model architecture.

        Args:
            config (Ernie4_5Config): Model configuration.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [
                # Ernie4_5DecoderLayer(create_skip_config_for_refined_recompute(i, config), i)
                Ernie4_5DecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        self.norm = Norm(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            nn.Embedding: The embedding layer for input tokens
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set new input embeddings.

        Args:
            value (nn.Embedding): New embedding layer to use
        """
        self.embed_tokens = value

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_start_row_indices,
        position_ids,
        token_type_ids,
        output_attentions,
        past_key_value,
        use_cache,
    ):
        """Perform gradient checkpointing for memory-efficient training.

        Args:
            layer_module (nn.Layer): Transformer layer to recompute
            hidden_states (paddle.Tensor): Input hidden states
            attention_mask (paddle.Tensor): Attention mask
            attn_mask_start_row_indices (paddle.Tensor): Variable length indices
            position_ids (paddle.Tensor): Position indices
            output_attentions (bool): Whether to output attention weights
            past_key_value (Optional[Tuple[paddle.Tensor]]): Cached key/value states
            use_cache (bool): Whether to cache key/value states

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_gate_logits=False)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            attn_mask_start_row_indices,
            position_ids,
            token_type_ids,
            output_attentions,
            past_key_value,
            use_cache,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[paddle.Tensor]): Input token IDs
            position_ids (Optional[paddle.Tensor]): Position indices
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            inputs_embeds (Optional[paddle.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Tuple[Tuple[paddle.Tensor]]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            has_gradient = not hidden_states.stop_gradient
            if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


class Ernie4_5ForCausalLM(Ernie4_5PretrainedModel):
    """ERNIE model for causal language modeling."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the ERNIE model for causal language modeling.

        Args:
            config (Ernie4_5Config): Model configuration.
        """
        super().__init__(config)

        # initialize-trick for big model,
        # see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range
        self.config = config
        self.ernie = Ernie4_5Model(config)
        self.lm_head = Ernie4_5LMHead(config)
        # if self.config.dpo_config is not None:
        #     self.criterion = ErnieDPOCriterion(config)
        # else:
        #     self.criterion = ErniePretrainingCriterion(config)
        self.criterion = ErniePretrainingCriterion(config)

        self.tie_weights()

        if self.config.use_rmsnorm:
            if self.config.fuse_rms_norm:
                logger.info("Use fusedRMSNorm")
            else:
                logger.info("Use normal RMSNorm")
        else:
            if self.config.fuse_ln:
                logger.info("Use fusedLN")
            else:
                logger.info("Use normal LayerNorm")

    @paddle.no_grad()
    def set_state_dict(self, state_dict, *args, **kwargs):
        """
        Loads the model state dictionary.

        Args:
            state_dict (dict): Model state dictionary.
        """
        ret = super().set_state_dict(state_dict)
        return ret

    def get_input_embeddings(self):
        """Returns the input embeddings layer."""
        return self.ernie.embed_tokens

    def set_input_embeddings(self, value):
        """Sets the input embeddings layer."""
        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        """Returns the output embeddings (LM head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embeddings layer."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """Sets the ERNIE decoder model."""
        self.ernie = decoder

    def get_decoder(self):
        """Get the transformer decoder.

        Returns:
            nn.Layer: The decoder module
        """
        return self.ernie

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        """Avoid using attention_mask with flash_attn on generation."""
        if self.config.use_flash_attention:
            return None
        return super().prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Prepares model inputs for generation in PaddlePaddle models.

        Args:
            input_ids (paddle.Tensor):
                The input token IDs with shape [batch_size, sequence_length].
            use_cache (bool, optional):
                Whether to use cached key-value states for faster generation.
                Defaults to False.
            past_key_values (Optional[Tuple[paddle.Tensor]]):
                Cached past key-value states from previous generation steps.
                If provided, the input_ids will be truncated to only keep the last token.
            inputs_embeds (Optional[paddle.Tensor]):
                Precomputed embeddings instead of token IDs.
                Only used in the first generation step when past_key_values is None.
            **kwargs:
                Additional keyword arguments including:
                - attention_mask (paddle.Tensor): Attention mask tensor

        Returns:
            Dict[str, Union[paddle.Tensor, bool, Dict]]:
            A dictionary containing:
                - "input_ids" or "inputs_embeds": The main input tensors
                - "past_key_values": The cached key-value states
                - "use_cache": Flag indicating whether to use caching
                - "attention_mask": The attention mask tensor (if provided)
                - "return_dict": Always set to True for consistent output format

        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,  # use_cache,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
        )

        return model_inputs

    # @staticmethod
    def update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        """
        Updates model kwargs for generation.

        Args:
            outputs (Any): Model outputs.
            model_kwargs (dict): Current model kwargs.
            is_encoder_decoder (bool): Whether using encoder-decoder architecture.

        Returns:
            dict: Updated model kwargs.
        """
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        if not is_encoder_decoder and model_kwargs.get("attention_mask", None) is not None:
            # update attention mask
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [
                    attention_mask,
                    paddle.ones([attention_mask.shape[0], 1], dtype="int64"),
                ],
                axis=-1,
            )
        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        return model_kwargs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,  # true when decode, false when pretrain & eval
        **kwargs,
    ):
        """
        Forward pass for causal language modeling.

        Args:
            input_ids (paddle.Tensor): Input token IDs.
            position_ids (paddle.Tensor): Position IDs.
            attention_mask (paddle.Tensor): Attention mask.
            attn_mask_start_row_indices (paddle.Tensor): Attention mask start indices.
            inputs_embeds (paddle.Tensor): Optional embedded inputs.
            labels (paddle.Tensor): Target labels.
            loss_mask (paddle.Tensor): Loss mask.
            use_cache (bool): Whether to use cached hidden states.
            past_key_values (dict): Pre-computed hidden states.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return a dictionary.

        Returns:
            Union[tuple, CausalLMOutputWithCrossAttentions]: Model outputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        outputs = self.ernie(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # if isinstance(self.criterion, ErnieDPOCriterion):
        if False:
            logits = (hidden_states, self.lm_head.weight, None, self.config.tie_word_embeddings)
            chosen_labels = kwargs.get("chosen_labels", None)
            rejected_labels = kwargs.get("rejected_labels", None)
            response_indexs = kwargs.get("response_indexs", None)
            score_deltas = kwargs.get("score_deltas", None)
            reference_chosen_logps = kwargs.get("reference_chosen_logps", None)
            reference_rejected_logps = kwargs.get("reference_rejected_logps", None)
            labels = (
                chosen_labels,
                rejected_labels,
                response_indexs,
                score_deltas,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            return self.criterion(
                logits,
                labels,
            )

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        logits = self.lm_head(hidden_states)

        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(logits, labels, loss_mask)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Pretrain & Eval must have labels
        assert labels is not None

        return self.criterion(logits, labels, loss_mask)


__all__ = [
    "Ernie4_5Model",
    "Ernie4_5ForCausalLM",
]
