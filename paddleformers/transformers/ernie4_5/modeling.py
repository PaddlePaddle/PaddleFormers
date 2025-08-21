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

import math
from functools import partial
from typing import Optional, Tuple

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Ernie4_5MLP
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ..model_utils import PretrainedModel, register_base_model
from ..tensor_parallel_utils import model_parallel_dropout
from .configuration import Ernie4_5Config


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
        return query.astype(q.dtype), key.astype(k.dtype)


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
        self.head_dim = config.head_dim

        self.is_gqa = config.num_key_value_heads is not None and config.num_key_value_heads != self.num_heads
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
            kv_hidden_size = self.head_dim * config.num_key_value_heads
            q_hidden_size = self.head_dim * config.num_attention_heads
        else:
            q_hidden_size = kv_hidden_size = self.head_dim * config.num_attention_heads

        qkv_hidden_size = q_hidden_size + kv_hidden_size * 2

        self.qkv_proj = GeneralLinear.create(
            self.hidden_size,
            qkv_hidden_size,
            has_bias=config.use_bias,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            self.hidden_size,
            has_bias=config.use_bias,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            tp_plan="rowwise",
        )

        self.rotary_emb = RopeEmbedding(
            self.head_dim,
            compression_ratio=config.compression_ratio,
            base=config.rope_theta,
            freq_allocation=self.freq_allocation,
        )
        self.config = config
        self.scaling = self.head_dim**-0.5
        self.attn_implementation = config._attn_implementation

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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
        if self.config.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_parallel_degree // max_sequence_length
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape

        query_states = key_states = value_states = mix_layer = None

        mix_layer = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = paddle.split(
            mix_layer.reshape([bsz, q_len, -1, self.head_dim]),
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads],
            axis=2,
        )

        if attn_mask_start_row_indices is None and attention_mask is None:
            self.attn_implementation = "sdpa"
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        # apply rope
        kv_seq_len = key_states.shape[-3]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-3]
            kv_seq_len += offset

        cos_sin = self.rotary_emb(kv_seq_len, position_ids).transpose([0, 2, 1, 3])  # [b,h,s,d]->[b,s,h,d]
        if offset > 0 and position_ids is None:
            # position_ids has been sliced in prepare_inputs_for_generation
            cos_sin = cos_sin[:, offset:]

        query_states, key_states = self.rotary_emb.apply_rotary(cos_sin, query_states, key_states)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        # NOTE(for generation): use list instead of tuple to store the cache
        # tensors, so that we can clear the cache tensors for memory efficiency.
        past_key_value = [key_states, value_states] if use_cache else None

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            dropout=self.config.get("attention_dropout_prob", 0.0) if self.training else 0.0,
            scaling=self.scaling,
        )

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


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
        self.mlp = Ernie4_5MLP(config, fuse_up_gate=True)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
        )

        self.hidden_dropout = nn.Dropout(p=config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            self.post_attention_layernorm.enable_sequence_parallel()
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
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
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


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
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList([Ernie4_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
        )

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
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            attn_mask_start_row_indices,
            position_ids,
            output_attentions,
            past_key_value,
            use_cache,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
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
            kv_seq_len = 0
        else:
            kv_seq_len = past_key_values[0][0].shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

        if attention_mask is not None:
            causal_attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, hidden_states.shape[:2], kv_seq_len, hidden_states.dtype
            )
        else:
            causal_attention_mask = None

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
                    causal_attention_mask,
                    attn_mask_start_row_indices,
                    position_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    causal_attention_mask,
                    attn_mask_start_row_indices,
                    position_ids,
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
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

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

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
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
                    paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype),
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
        if self.criterion.loss_type == "dpo":
            logits = self.lm_head(hidden_states)
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
