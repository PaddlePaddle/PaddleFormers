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
"""Paddle Ernie model"""
import functools
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
    get_rng_state_tracker,
)
from paddle.distributed.fleet.utils import recompute

from paddleformers.utils.log import logger

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
from .refined_recompute.utils import RefinedRecomputeFunction
from .sequence_parallel_utils import ScatterOp


def calc_lm_head_logits(config, hidden_states, weight, bias, tensor_parallel_output=None, training=True):
    """
    Calculate language model head logits with support for various parallelization strategies.

    This is the core function that computes the final output logits for a language model,
    handling sequence parallelism and tensor parallelism configurations.

    Args:
        config (Ernie4_5_Config): Model configuration.
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
        if config.use_sparse_head_and_loss_fn:
            pass  # Nothing needs to be done.
        else:
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
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
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
            config (Ernie4_5_Config): Model configuration.
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
            config (Ernie4_5_Config): Model configuration contains normalization parameters and flags.
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
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_q.astype("float32"), sin_pos),
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_k.astype("float32"), sin_pos),
        )
        return query, key

    def apply_rotary_3d(self, rp, q, k, position_ids):
        """
        rope 3d rotary

        args:
            rp: [1, max_seqlen, 1, head_dim]
            q: [bsz, seqlen, head, head_dim]
            k: [bsz, seqlen, head, head_dim]
            position_ids: [bsz, seqlen, 3]
        """
        sin, cos = paddle.chunk(rp, 2, axis=-1)
        assert position_ids.shape[:1] == q.shape[:1]
        batch_indices = paddle.arange(end=position_ids.shape[0])
        batch_indices = batch_indices[..., None]
        sin = sin.tile([position_ids.shape[0], 1, 1, 1])
        cos = cos.tile([position_ids.shape[0], 1, 1, 1])

        assert self.freq_allocation != 0
        sin_t = sin[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        sin_h = sin[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_w = sin[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_hw = paddle.stack([sin_h, sin_w], axis=-1).reshape(sin_h.shape[:-1] + [sin_h.shape[-1] * 2])
        sin_thw = paddle.concat([sin_hw, sin_t], axis=-1)

        cos_t = cos[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        cos_h = cos[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_w = cos[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_hw = paddle.stack([cos_h, cos_w], axis=-1).reshape(cos_h.shape[:-1] + [cos_h.shape[-1] * 2])
        cos_thw = paddle.concat([cos_hw, cos_t], axis=-1)

        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(
            paddle.stack([sin_thw, sin_thw], axis=-1),
            sin_thw.shape[:3] + [sin_thw.shape[-1] * 2],
        )
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(
            paddle.stack([cos_thw, cos_thw], axis=-1),
            cos_thw.shape[:3] + [cos_thw.shape[-1] * 2],
        )

        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = paddle.reshape(
            paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
            paddle.shape(q),
        )
        query = paddle.add(
            paddle.multiply(q.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_q.astype("float32"), sin_pos),
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = paddle.reshape(
            paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
            paddle.shape(k),
        )
        key = paddle.add(
            paddle.multiply(k.astype("float32"), cos_pos),
            paddle.multiply(rotate_half_k.astype("float32"), sin_pos),
        )
        return query, key

    def forward_single(self, position_ids):
        batch_size, seq_length = position_ids.shape[:2]
        rope_emb = paddle.zeros((2, batch_size, seq_length, 1, self.head_dim), dtype="float32")
        inv_freq = self.base ** (-paddle.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim)
        position_ids = position_ids.cast("float32")
        position_ids = position_ids / self.compression_ratio
        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, D]
        emb = paddle.stack([freqs, freqs], axis=-1).reshape((batch_size, seq_length, self.head_dim))
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rope_emb[0] = paddle.cos(emb)
        rope_emb[1] = paddle.sin(emb)
        return rope_emb

    @staticmethod
    def apply_rotary_single(x, rope_emb):
        rotate_half_x = paddle.reshape(
            paddle.stack([-x[:, :, :, 1::2], x[:, :, :, 0::2]], axis=-1),
            paddle.shape(x),
        )
        return x * rope_emb[0] + rotate_half_x * rope_emb[1]


class Ernie4_5_MLP(nn.Layer):
    """
    Ernie4_5_MLP - Gated Multi-Layer Perceptron module used in Ernie model.
    """

    def __init__(self, config, layer_idx=0):
        """
        Initialize the MLP module with configuration options.

        Args:
            config (Ernie4_5_Config): Model configurations.
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
            self.up_proj = ColumnLN(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=config.use_bias,
                fuse_matmul_bias=config.fuse_linear,
                **column_ln_configs,
            )
            self.gate_proj = ColumnLN(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=config.use_bias,
                fuse_matmul_bias=config.fuse_linear,
                **column_ln_configs,
            )
        else:
            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
            self.up_proj = LinearFN(self.hidden_size, self.intermediate_size, bias_attr=config.use_bias)
            self.gate_proj = LinearFN(self.hidden_size, self.intermediate_size, bias_attr=config.use_bias)

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
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.fuse_swiglu:
            x = paddle.concat([gate, up], axis=-1)
            x = fused_swiglu(x)
        else:
            x = F.silu(gate) * up
        return self.down_proj(x)


class Ernie4_5_Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        """Initialize the attention layer.

        Args:
            config (Ernie4_5_Config): Model configuration.
            layer_idx (int, optional): Index in transformer stack. Defaults to 0.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if getattr(config, "head_dim", None) is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = config.head_dim
        self.is_gqa = config.num_key_value_heads is not None and config.num_key_value_heads != self.num_heads
        if config.fuse_rope:
            assert fused_rope is not None, "fused_rope is not supported"
        self.fuse_rope = config.fuse_rope

        self.rope_3d = config.get("rope_3d", False)
        self.freq_allocation = config.get("freq_allocation", 0)
        if self.rope_3d:
            assert not self.fuse_rope, "does not support fuse rope when rope_3d is on for now."
            assert self.freq_allocation is not None, "freq_allocation must be provided if rope_3d is on."

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
            if getattr(config, "head_dim", None) is None:
                kv_hidden_size = self.hidden_size // self.num_heads * self.num_key_value_heads
                q_hidden_size = self.hidden_size
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

            self.q_proj = ColumnLN(
                self.hidden_size,
                q_hidden_size,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
                **column_ln_configs,
            )
            self.k_proj = ColumnLN(
                self.hidden_size,
                kv_hidden_size,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
                **column_ln_configs,
            )
            self.v_proj = ColumnLN(
                self.hidden_size,
                kv_hidden_size,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
                **column_ln_configs,
            )
        else:
            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
            self.q_proj = LinearFN(
                self.hidden_size,
                q_hidden_size,
                bias_attr=config.use_bias,
            )
            self.k_proj = LinearFN(
                self.hidden_size,
                kv_hidden_size,
                bias_attr=config.use_bias,
            )
            self.v_proj = LinearFN(
                self.hidden_size,
                kv_hidden_size,
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
                (self.hidden_size if getattr(config, "head_dim", None) is None else q_hidden_size),
                self.hidden_size,
                has_bias=config.use_bias,
                input_is_parallel=True,
                fuse_matmul_bias=config.fuse_linear,
                **row_ln_configs,
            )
        else:
            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else Linear
            self.o_proj = LinearFN(
                (self.hidden_size if getattr(config, "head_dim", None) is None else q_hidden_size),
                self.hidden_size,
                bias_attr=config.use_bias,
            )
        self.rotary_emb = RopeEmbedding(
            self.head_dim,
            compression_ratio=config.compression_ratio,
            base=config.rope_theta,
            freq_allocation=self.freq_allocation,  # 0 in LLM, 20 in MLLM
        )
        self.config = config

        self._rr_flash_attn = None
        if config.recompute and config.skip_recompute_ops[layer_idx].get("flash_attn", False):
            self._rr_flash_attn = RefinedRecomputeFunction()

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
        token_type_ids: Optional[Tuple[paddle.Tensor]] = None,  # MLLM
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
            bsz = 1
            q_len = hidden_states.shape[0] * self.config.tensor_parallel_degree
        else:
            bsz, q_len, _ = hidden_states.shape
        query_states = key_states = value_states = mix_layer = None
        query_states = self.q_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
        key_states = self.k_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
        value_states = self.v_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])

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

        replicate = self.config.num_attention_heads // self.config.num_key_value_heads
        k = paddle.repeat_interleave(k, replicate, axis=1)
        v = paddle.repeat_interleave(v, replicate, axis=1)

        scale_qk_coeff = self.config.scale_qk_coeff * self.head_dim**0.5
        attention_mask = paddle.where(
            attention_mask,
            paddle.to_tensor(0.0, dtype=q.dtype),
            paddle.finfo(q.dtype).min,
        )
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
        if self.rope_3d:
            assert position_ids is not None, "rope3d requires pos-id"
        kv_seq_len = key_states.shape[-3] if not self.rope_3d else position_ids.max() + 1
        offset = 0
        if past_key_value is not None:
            if not self.rope_3d:
                # LLM
                offset = past_key_value[0].shape[-3]
                kv_seq_len += offset
            else:
                # MLLM
                offset = position_ids.max()
                kv_seq_len = position_ids.max() + 1
                position_ids = position_ids[:, -1:, :]

        # TODO(daisiming): rope这块需重新考虑
        if offset > 0 or position_ids is not None or not self.fuse_rope:
            if not self.rope_3d:
                # LLM
                cos_sin = self.rotary_emb(kv_seq_len, position_ids).transpose([0, 2, 1, 3])  # [b,h,s,d]->[b,s,h,d]
                if offset > 0 and position_ids is None:
                    # position_ids has been sliced in prepare_inputs_for_generation
                    cos_sin = cos_sin[:, offset:]
                query_states, key_states = self.rotary_emb.apply_rotary(cos_sin, query_states, key_states)
            else:
                # MLLM
                cos_sin = self.rotary_emb(kv_seq_len).transpose([0, 2, 1, 3])  # [b,h,s,d]->[b,s,h,d]
                if offset > 0 and position_ids is None:
                    cos_sin = cos_sin[:, offset:]
                query_states, key_states = self.rotary_emb.apply_rotary_3d(
                    cos_sin, query_states, key_states, position_ids
                )
        else:
            _, _, num_heads, _ = query_states.shape
            _, kv_seq_len, num_key_value_heads, _ = key_states.shape
            if num_heads != num_key_value_heads:
                query_states, _, _ = fused_rope(query_states, None, None, rotary_emb_base=self.config.rope_theta)
                key_states, _, _ = fused_rope(key_states, None, None, rotary_emb_base=self.config.rope_theta)
            else:
                query_states, key_states, _ = fused_rope(
                    query_states,
                    key_states,
                    None,
                    rotary_emb_base=self.config.rope_theta,
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
                num_tokens_per_rank,
                paddle.to_tensor(hidden_states.shape[0], dtype=paddle.int32),
                group=ctx.mp_group,
            )
            ctx.num_tokens_per_rank = num_tokens_per_rank

            for idx in range(ctx.world_size):
                if idx == ctx.rank:
                    hidden_states_recv = hidden_states
                    labels_recv = labels
                else:
                    hidden_states_recv = paddle.empty(
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]],
                        dtype=hidden_states.dtype,
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
                        [ctx.num_tokens_per_rank[idx], hidden_states.shape[-1]],
                        dtype=hidden_states.dtype,
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
            config (Ernie4_5_Config): Model configuration.
            return_tuple (bool): Whether to return loss as tuple (loss, loss_sum). Defaults to True.
        """
        super(ErniePretrainingCriterion, self).__init__()
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
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
            hidden_states, outlinear_weight, outlinear_bias = prediction_scores[:3]

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
                if self.config.get("offload_lm_head", False):
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
                    self.config,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    training=self.training,
                )
                res = self.forward_impl(logits, masked_lm_labels, loss_mask)
        elif self.config.use_recompute_loss_fn:
            if self.config.use_fused_head_and_loss_fn:
                res = self.forward_impl_with_fused_head_loss_fn(masked_lm_labels, loss_mask, *prediction_scores)
            else:
                assert isinstance(prediction_scores, tuple) and len(prediction_scores) in [3, 4], prediction_scores
                res = recompute(
                    self.forward_impl_with_calc_logits,
                    masked_lm_labels,
                    loss_mask,
                    *prediction_scores,
                )
        else:
            res = self.forward_impl(prediction_scores, masked_lm_labels, loss_mask)

        return res

    def forward_impl_with_fused_head_loss_fn(
        self,
        masked_lm_labels,
        loss_mask,
        hidden_states,
        outlinear_weight,
        outlinear_bias,
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
            seq_chunk_size=self.config.get("loss_subbatch_seqlen", 32768),
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
            # 逐位对齐, 全精度聚合
            masked_lm_loss = paddle.sum(masked_lm_loss.cast(paddle.float32).reshape([-1]) * loss_mask)
            loss = masked_lm_loss / loss_mask.sum()
            if self.token_balance_loss:
                _loss = masked_lm_loss / self.config.token_balance_seqlen
                loss = _loss - _loss.detach() + loss.detach()  # for 对线
            loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum

    def forward_impl_with_calc_logits(
        self,
        masked_lm_labels,
        loss_mask,
        hidden_states,
        outlinear_weight,
        outlinear_bias,
    ):
        """Compute logits then calculate loss.

        Args:
            Same as forward_impl_with_fused_head_loss_fn()

        Returns:
            Same return format as forward()
        """

        logits = calc_lm_head_logits(
            self.config,
            hidden_states,
            outlinear_weight,
            outlinear_bias,
            training=self.training,
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
            if prediction_scores_dims == 2 and prediction_scores.shape[0] > self.config.get(
                "loss_subbatch_seqlen", 32768
            ):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [0, 0],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    0,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            elif prediction_scores_dims == 3 and prediction_scores.shape[1] > self.config.get(
                "loss_subbatch_seqlen", 32768
            ):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [1, 1],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    1,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            else:
                masked_lm_loss = self.loss_impl(prediction_scores, masked_lm_labels)

            if loss_mask is None:
                loss_mask = masked_lm_labels != self.ignored_index

            lossmask = masked_lm_labels != self.ignored_index
            if (~lossmask).all():  # empty span
                logger.warning(f"encounter empty span when calculate loss, ignored_index={self.ignored_index}")
                loss = paddle.mean(masked_lm_loss) * 0.0
                loss_sum = masked_lm_loss.sum().detach()
            else:
                loss_mask = loss_mask.reshape([-1]).cast(paddle.float32)
                # 逐位对齐, 全精度聚合
                masked_lm_loss = paddle.sum(masked_lm_loss.cast(paddle.float32).reshape([-1]) * loss_mask)
                loss = masked_lm_loss / loss_mask.sum()
                if self.token_balance_loss:
                    _loss = masked_lm_loss / self.config.token_balance_seqlen
                    loss = _loss - _loss.detach() + loss.detach()  # for 对线
                loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:  # only used in pp
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


class Ernie4_5_LMHead(nn.Layer):
    """Language model head for ERNIE with support for tensor parallelism."""

    def __init__(self, config):
        """Initialize the language model head.

        Args:
            config (Ernie4_5_Config): Model configuration containing:
                - vocab_size: Size of vocabulary
                - hidden_size: Dimension of hidden states
                - tensor_parallel_degree: Degree of tensor parallelism
                - tie_word_embeddings: Whether to tie input/output embeddings
                - weight_share_add_bias: Whether to add bias when weight sharing
                - use_bias: Whether to use bias term
                - use_recompute_loss_fn: Whether to defer logits computation to loss function
                - use_sparse_head_and_loss_fn: Whether to use sparse head computation
        """

        super(Ernie4_5_LMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.weight = self.create_parameter(
            shape=(
                [vocab_size, config.hidden_size] if config.tie_word_embeddings else [config.hidden_size, vocab_size]
            ),
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
            return (
                hidden_states,
                self.weight,
                self.bias,
                self.config.tie_word_embeddings,
            )

        return calc_lm_head_logits(
            self.config,
            hidden_states,
            self.weight,
            self.bias,
            tensor_parallel_output,
            training=self.training,
        )
