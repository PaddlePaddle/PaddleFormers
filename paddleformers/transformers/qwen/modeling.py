# Copyright (c) 2023 Alibaba Cloud and PaddlePaddle Authors. All Rights Reserved.
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

import math
import os
import warnings
from functools import partial
from typing import List

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.recompute.recompute import recompute

from ..refined_recompute import (
    RRColumnParallelLinear,
    RRColumnSequenceParallelLinear,
    RRRowParallelLinear,
    RRRowSequenceParallelLinear,
    get_skip_recompute_ops,
    no_recompute,
)
from ..refined_recompute import recompute as rr_recompute

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


from ...utils.converter import StateDictNameMapping, init_name_mappings
from ...utils.log import logger
from .. import linear_utils
from ..linear_utils import Linear
from ..long_sequence_strategies import LongSequenceStrategies
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel
from ..utils import caculate_llm_per_token_flops
from .configuration import QWenConfig

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

__all__ = [
    "QWenBlock",
    "QWenForCausalLM",
    "QWenLMHeadModel",
    "QWenPretrainedModel",
    "QWenModel",
    "QWenLMHead",
    "QWenPretrainingCriterion",
]


MAX_NTK_SEQ_LENGTH = 32768

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except:
    fused_rotary_position_embedding = None


def get_use_casual_mask():
    """Get the value of the 'USE_CASUAL_MASK' environment variable."""
    return os.getenv("USE_CASUAL_MASK", "False") == "True"


def parallel_matmul(x: Tensor, y: Tensor, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if is_fleet_init and tensor_parallel_degree > 1 and y.is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=False)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=False)
        return logits


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


class QWenAttention(nn.Layer):
    def __init__(self, config, skip_recompute_ops=None):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.skip_recompute_ops = skip_recompute_ops
        self.config = config
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self.scale_attn_weights = True
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity

        self.projection_size = config.kv_channels * config.num_attention_heads

        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads

        self.sequence_parallel = config.sequence_parallel

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear

            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("attention_column_ln", False):
                    ColumnParallelLinear = RRColumnSequenceParallelLinear
                if skip_recompute_ops.get("attention_row_ln", False):
                    RowParallelLinear = RRRowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear
            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("attention_column_ln", False):
                    ColumnParallelLinear = RRColumnParallelLinear
                if skip_recompute_ops.get("attention_row_ln", False):
                    RowParallelLinear = RRRowParallelLinear

        if config.tensor_parallel_degree > 1:
            if config.num_attention_heads % config.tensor_parallel_degree != 0:
                raise ValueError("num_attention_heads has to be divisible by tensor_parallel_degree")
            self.num_heads = config.num_attention_heads // config.tensor_parallel_degree
            self.c_attn = ColumnParallelLinear(
                config.hidden_size,
                3 * self.projection_size,
                has_bias=True,
                gather_output=False,
            )
            self.c_proj = RowParallelLinear(
                config.hidden_size,
                self.projection_size,
                has_bias=not config.no_bias,
                input_is_parallel=True,
            )
        else:
            self.c_attn = Linear(config.hidden_size, 3 * self.projection_size, bias_attr=True)
            self.c_proj = Linear(
                config.hidden_size,
                self.projection_size,
                bias_attr=not config.no_bias,
            )

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(self.hidden_size_per_attention_head * config.rotary_pct)
        dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
        if config.use_long_sequence_strategies:
            self.rotary_emb = LongSequenceStrategies.build_long_sequence_strategy(
                config.long_sequence_strategy_type,
                config.long_sequence_strategy_name,
                **config.long_sequence_init_args,
            )
        else:
            self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [math.log(i, self.seq_length) if i > self.seq_length else 1 for i in range(1, MAX_NTK_SEQ_LENGTH)]
        self.logn_tensor = paddle.to_tensor(logn_list)[None, :, None, None]
        self._ntk_cached = 1.0

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

    def _attn(self, query, key, value, attention_mask=None):
        # Support the flash attention and normal attention
        bsz, q_len, num_heads, head_dim = query.shape
        _, kv_seq_len, _, _ = value.shape

        if self.config.use_flash_attention and flash_attention is not None:
            # Flash Attention now ignore attention mask
            # Current Flash Attention doesn't support attn maskt
            # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
            # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]
            version = paddle.version.full_version
            if version != "0.0.0" and version <= "2.5.2":
                attn_output, attn_weights = flash_attention(
                    query,
                    key,
                    value,
                    causal=query.shape[1] != 1,
                    dropout=self.config.attn_dropout_prob,
                    return_softmax=self.config.attn_dropout_prob > 0.0,
                )
            else:
                skip_recompute = (
                    self.config.recompute
                    and not self.config.recompute_use_reentrant
                    and self.skip_recompute_ops.get("flash_attn", False)
                )
                attn_output = no_recompute(
                    F.scaled_dot_product_attention,
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    is_causal=attention_mask is None,
                    enable=skip_recompute,
                )
                attn_weights = None

            if self.sequence_parallel:
                attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
            else:
                attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
            return attn_output, attn_weights
        else:
            # [bz, sql, nh, hid] ==> [bz, nh, sql hdim]
            query = query.transpose([0, 2, 1, 3])
            # [bz, sql, nh, hid] ==> [bz, nh, sql hdim]
            key = key.transpose([0, 2, 1, 3])
            # [bz, sql, nh, hid] ==> [bz, nh, sql hdim]
            value = value.transpose([0, 2, 1, 3])

            # Add pre divided factor to fix nan under float16.
            if paddle.in_dynamic_mode() and query.dtype == paddle.float16:
                pre_divided_factor = 32
            else:
                pre_divided_factor = 1

            attn_weights = paddle.matmul(
                query / (math.sqrt(head_dim) * pre_divided_factor), key.transpose([0, 1, 3, 2])
            )

            if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.shape}"
                )
            # If the attention mask is None, we need to construct the causal attention mask
            if attention_mask is None:
                attention_mask = get_triangle_upper_mask(attn_weights)
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights.astype("float32") * pre_divided_factor, axis=-1).astype(value.dtype)

            attn_weights = self.attn_dropout(attn_weights)
            attn_output = paddle.matmul(attn_weights, value)
            attn_output = attn_output.transpose([0, 2, 1, 3])

            if self.sequence_parallel:
                attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
            else:
                attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
            return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        use_cache=False,
    ):
        # [bz, sql, hid] ==> [bz, sql, 3*hid]
        mixed_x_layer = self.c_attn(hidden_states)

        if self.sequence_parallel:
            target_shape = [-1, self.seq_length, self.num_heads * 3 * self.head_dim]
            mixed_x_layer = paddle.reshape_(mixed_x_layer, target_shape)

        # [bz, sql, hid] ==> [bz, sql, nh, hdim]
        query, key, value = paddle.split(mixed_x_layer, num_or_sections=3, axis=-1)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        kv_seq_len = key.shape[-3]
        if layer_past:
            # layer past[0] shape: bs * seq_len * head_num * dim
            kv_seq_len += layer_past[0].shape[1]
        if self.use_dynamic_ntk and kv_seq_len == hidden_states.shape[1] and not self.training:
            context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
            ntk_alpha = 2 ** math.ceil(context_value) - 1
            ntk_alpha = max(ntk_alpha, 1)
            self._ntk_cached = ntk_alpha
        else:
            ntk_alpha = self._ntk_cached
        if self.config.use_long_sequence_strategies:
            cos, sin = self.rotary_emb(seq_len=kv_seq_len, ntk_alpha=ntk_alpha)
            rotary_pos_emb = (cos[None, :, None, :], sin[None, :, None, :])
        else:
            rotary_pos_emb = self.rotary_emb(value, kv_seq_len, ntk_alpha=ntk_alpha)

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2

        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            if self.config.use_fused_rope:
                query, key, _ = fused_rotary_position_embedding(
                    query,
                    key,
                    v=None,
                    sin=sin,
                    cos=cos,
                    position_ids=position_ids,
                    use_neox_rotary_style=False,
                )
            else:
                query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids=position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key = paddle.concat([past_key, key], axis=1)
            value = paddle.concat([past_value, value], axis=1)

        if use_cache:
            present = (key, value)
        else:
            present = None

        if self.use_logn_attn and not self.training:
            if self.logn_tensor.dtype != query.dtype:
                self.logn_tensor = self.logn_tensor.astype(query.dtype)
            seq_start = key.shape[1] - query.shape[1]
            seq_end = key.shape[1]
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
            query = query * logn_tensor.expand(query.shape)

        has_gradient = not (query.stop_gradient and key.stop_gradient and value.stop_gradient)
        if self.enable_recompute and self.training and has_gradient and self.recompute_granularity == "core_attn":
            recompute_fn = rr_recompute if any(self.skip_recompute_ops.values()) else recompute
            attn_output, attn_weight = recompute_fn(
                self._attn,
                query,
                key,
                value,
                attention_mask,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            attn_output, attn_weight = self._attn(query, key, value, attention_mask)

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.c_proj(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight,)
        return outputs


class QWenMLP(nn.Layer):
    def __init__(self, config, skip_recompute_ops=None):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        ff_dim_in = config.intermediate_size // 2
        self.fuse_attention_ffn = config.fuse_attention_ffn
        self.skip_recompute_ops = skip_recompute_ops

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear

            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("mlp_column_ln", False):
                    ColumnParallelLinear = RRColumnSequenceParallelLinear
                if skip_recompute_ops.get("mlp_row_ln", False):
                    RowParallelLinear = RRRowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear
            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("mlp_column_ln", False):
                    ColumnParallelLinear = RRColumnParallelLinear
                if skip_recompute_ops.get("mlp_row_ln", False):
                    RowParallelLinear = RRRowParallelLinear

        if config.tensor_parallel_degree > 1:
            if self.fuse_attention_ffn:
                self.gate_up_fused_proj = ColumnParallelLinear(
                    config.hidden_size,
                    ff_dim_in * 2,
                    gather_output=False,
                    has_bias=False,
                )
            else:
                self.w1 = ColumnParallelLinear(
                    config.hidden_size,
                    ff_dim_in,
                    gather_output=False,
                    has_bias=False,
                )
                self.w2 = ColumnParallelLinear(
                    config.hidden_size,
                    ff_dim_in,
                    gather_output=False,
                    has_bias=False,
                )
            self.c_proj = RowParallelLinear(
                ff_dim_in,
                config.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            if self.fuse_attention_ffn:
                self.gate_up_fused_proj = Linear(config.hidden_size, ff_dim_in * 2, bias_attr=not config.no_bias)
            else:
                self.w1 = Linear(config.hidden_size, ff_dim_in, bias_attr=not config.no_bias)
                self.w2 = Linear(config.hidden_size, ff_dim_in, bias_attr=not config.no_bias)
            self.c_proj = Linear(ff_dim_in, config.hidden_size, bias_attr=not config.no_bias)

    def forward(self, hidden_states):
        # up
        # a1 = self.w1(hidden_states)
        # # gate
        # a2 = self.w2(hidden_states)
        # intermediate_parallel = a1 * F.silu(a2)
        if self.fuse_attention_ffn:
            intermediate_parallel = swiglu(self.gate_up_fused_proj(hidden_states))
        else:
            intermediate_parallel = swiglu(self.w2(hidden_states), self.w1(hidden_states))
        output = self.c_proj(intermediate_parallel)
        return output


class QWenBlock(nn.Layer):
    def __init__(self, config, skip_recompute_ops=None):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.skip_recompute_ops = skip_recompute_ops
        self.sequence_parallel = config.sequence_parallel
        self.ln_1 = QWenRMSNorm(config)
        self.attn = QWenAttention(config, skip_recompute_ops=skip_recompute_ops)
        self.ln_2 = QWenRMSNorm(config)
        self.mlp = QWenMLP(config, skip_recompute_ops=skip_recompute_ops)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_input = attn_output + residual

        layernorm_output = self.ln_2(layernorm_input)

        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class QWenPretrainedModel(PretrainedModel):
    config_class = QWenConfig
    base_model_prefix = "qwen"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _get_model_flops(self):
        if hasattr(self.config, "seq_length"):
            seq_length = self.config.seq_length
        else:
            seq_length = 2048

        return caculate_llm_per_token_flops(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            layer_num=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            seq_length=seq_length,
            recompute=False,
        )

    def _get_hardware_flops(self):
        if hasattr(self.config, "seq_length"):
            seq_length = self.config.seq_length
        else:
            seq_length = 2048

        return caculate_llm_per_token_flops(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            layer_num=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            seq_length=seq_length,
            recompute=self.config.recompute,
            recompute_granularity=self.config.recompute_granularity,
        )

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_hidden_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                "lm_head.weight": partial(fn, is_column=True),
                "qwen.h.0.attn.c_attn.weight": partial(fn, is_column=True, is_naive_3fuse=True),
                "qwen.h.0.attn.c_attn.bias": partial(fn, is_column=True, is_naive_3fuse=True),
                # Row Linear
                "qwen.wte.weight": partial(fn, is_column=False),
                "qwen.h.0.mlp.c_proj.weight": partial(fn, is_column=False),
                "qwen.h.0.attn.c_proj.weight": partial(fn, is_column=False),
            }

            if config.fuse_attention_ffn:
                base_actions["layers.0.mlp.gate_up_fused_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                base_actions["qwen.h.0.mlp.w2.weight"] = partial(fn, is_column=True)
                base_actions["qwen.h.0.mlp.w1.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "h.0." in key:
                    for i in range(num_hidden_layers):
                        final_actions[key.replace("h.0.", f"h.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @classmethod
    def _get_name_mappings(cls, config: QWenConfig) -> List[StateDictNameMapping]:
        mappings = [
            "wte.weight",
            "ln_f.weight",
        ]

        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"h.{layer_index}.ln_1.weight",
                    f"h.{layer_index}.ln_1.weight",
                ],
                [
                    f"h.{layer_index}.attn.c_attn.weight",
                    f"h.{layer_index}.attn.c_attn.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.attn.c_attn.bias",
                    f"h.{layer_index}.attn.c_attn.bias",
                ],
                [
                    f"h.{layer_index}.attn.c_proj.weight",
                    f"h.{layer_index}.attn.c_proj.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.ln_2.weight",
                    f"h.{layer_index}.ln_2.weight",
                ],
                [
                    f"h.{layer_index}.mlp.w1.weight",
                    f"h.{layer_index}.mlp.w1.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.mlp.w2.weight",
                    f"h.{layer_index}.mlp.w2.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.mlp.c_proj.weight",
                    f"h.{layer_index}.mlp.c_proj.weight",
                    "transpose",
                ],
            ]
            mappings.extend(layer_mappings)

        init_name_mappings(mappings)
        for mapping in mappings:
            mapping[0] = "transformer." + mapping[0]
            if len(mapping) > 1 and mapping[1] is not None:
                mapping[1] = "qwen." + mapping[1]

        if config.architectures is not None:
            if "QWenForCausalLM" in config.architectures or "QWenLMHeadModel" in config.architectures:
                mappings.extend(
                    [
                        [
                            "lm_head.weight",
                            "lm_head.weight",
                            "transpose",
                        ]
                    ]
                )

        init_name_mappings(mappings)
        return [StateDictNameMapping(*mapping) for mapping in mappings]

    def _init_weights(self, module):
        """Initialize the weights."""
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        if isinstance(
            module,
            (
                nn.Linear,
                nn.Embedding,
                mpu.VocabParallelEmbedding,
                mpu.RowParallelLinear,
                mpu.ColumnParallelLinear,
                linear_utils.RowSequenceParallelLinear,
                linear_utils.ColumnSequenceParallelLinear,
                QWenLMHead,
            ),
        ):
            if isinstance(module.weight, paddle.Tensor):
                if module.weight.is_distributed:
                    with rng_tracker():
                        module.weight.set_value(
                            paddle.tensor.normal(
                                mean=0.0,
                                std=self.config.initializer_range,
                                shape=module.weight.shape,
                            )
                        )
            else:
                module.weight.set_value(
                    paddle.tensor.normal(mean=0.0, std=self.config.initializer_range, shape=module.weight.shape)
                )

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
                        shape=p.shape,
                    )
                )


class QWenModel(QWenPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity
        self.sequence_parallel = config.sequence_parallel

        if config.tensor_parallel_degree > 1:
            self.wte = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
            )
        else:
            self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.drop = nn.Dropout(config.emb_dropout_prob)
        self.h = nn.LayerList(
            [
                QWenBlock(
                    config=config,
                    skip_recompute_ops=get_skip_recompute_ops(config, layer_idx),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = QWenRMSNorm(config)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        block,
        hidden_states,
        layer_past,
        attention_mask,
        position_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        recompute_fn = rr_recompute if any(block.skip_recompute_ops.values()) else recompute
        hidden_states = recompute_fn(
            create_custom_forward(block),
            hidden_states,
            layer_past,
            attention_mask,
            position_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states

    def get_masks(self, batch_size, seq_length, past_length, padding_mask=None):
        # casual mask
        casual_mask = paddle.tril(paddle.ones([batch_size, 1, seq_length, seq_length], dtype="bool"))
        if past_length > 0:
            casual_mask = paddle.concat(
                [paddle.ones([batch_size, 1, seq_length, past_length], dtype="bool"), casual_mask], axis=-1
            )

        # seq_mask
        if padding_mask is None:
            padding_mask = paddle.ones((batch_size, 1, seq_length, seq_length + past_length), dtype="bool")
        if len(padding_mask.shape) == 2:
            # from Tokenizer
            padding_mask = (
                padding_mask.unsqueeze(axis=[1, 2])
                .expand([batch_size, 1, seq_length, seq_length + past_length])
                .astype("bool")
            )
        elif len(padding_mask.shape) == 3:
            # [batch_size,tgt_length, src_length] -> [batch_size, 1, tgt_length, src_length]
            padding_mask = padding_mask.unsqueeze(1).astype("bool")
        elif len(padding_mask.shape) == 4:
            padding_mask = padding_mask.astype("bool")

        casual_mask = casual_mask & padding_mask

        return casual_mask

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.sequence_parallel and use_cache:
            raise ValueError("We currently only support sequence parallel without cache.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[1]

        encoder_attention_mask = None
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds
        use_casual_mask = get_use_casual_mask()
        # bool 4D mask
        if use_casual_mask is True:
            attention_mask = None
        else:
            attention_mask = self.get_masks(input_shape[0], input_shape[1], past_length, padding_mask=attention_mask)
            zero = paddle.zeros(attention_mask.shape, dtype=hidden_states.dtype)
            neg_inf = paddle.full_like(
                attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype
            )
            # dtype 4D mask
            attention_mask = paddle.where(attention_mask, zero, neg_inf)

        hidden_states = self.drop(hidden_states)

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with recompute")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            has_gradient = not hidden_states.stop_gradient
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.enable_recompute and self.training and has_gradient and self.recompute_granularity == "full":
                outputs = self.recompute_training(
                    block,
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if type(outputs) is tuple:
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

            if use_cache is True:
                presents = presents + (outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QWenLMHead(nn.Layer):
    def __init__(self, config: QWenConfig):
        super(QWenLMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        if vocab_size != config.vocab_size:
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            self.weight = self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            )
        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if self.weight.is_distributed:
            self.weight.split_axis = 1

    def forward(self, hidden_states, tensor_parallel_output=None):
        if self.config.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            seq_length = self.config.seq_length
            hidden_states = paddle.reshape_(hidden_states, [-1, seq_length, self.config.hidden_size])

        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1

        logits = parallel_matmul(hidden_states, self.weight, tensor_parallel_output=tensor_parallel_output)
        return logits


class QWenPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for Llama.
    It calculates the final loss.
    """

    def __init__(self, config):

        super(QWenPretrainingCriterion, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=self.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                warnings.warn(
                    f"enable_parallel_cross_entropy, the vocab_size should be splited: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            # skip ignore_index which loss == 0
            masked_lm_loss = masked_lm_loss[masked_lm_loss > 0].astype("float32")
            loss = paddle.mean(masked_lm_loss)

        return loss


class QWenForCausalLM(QWenPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]

    def __init__(self, config):
        super().__init__(config)
        self.qwen = QWenModel(config)
        self.lm_head = QWenLMHead(config)
        self.criterion = QWenPretrainingCriterion(config)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["cache"] = outputs[1]
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, ModelOutput) and "past_key_values" in outputs:
            model_kwargs["cache"] = outputs.past_key_values
            model_kwargs["past_key_values"] = outputs.past_key_values

        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None and len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)], axis=-1
                )
            else:
                model_kwargs["attention_mask"] = None

        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        return model_inputs

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype(paddle.int64)
        else:
            attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        return attention_mask

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.qwen(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

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
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                loss = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_cos_sin_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (paddle.arange(0, self.dim, 2, dtype=paddle.float32) / self.dim))
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = paddle.arange(self._seq_len_cached)
            with paddle.amp.auto_cast(enable=False):
                freqs = paddle.outer(seq.astype(self.inv_freq.dtype), self.inv_freq)
            emb = paddle.concat([freqs, freqs], axis=-1)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_cos_sin_cache(max_seq_len, offset, ntk_alpha)
        cos = self.cos_cached[:, offset : offset + max_seq_len, :, ...]
        sin = self.sin_cached[:, offset : offset + max_seq_len, :, ...]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):

    if position_ids is None:
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QWenRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_epsilon
        self.weight = paddle.create_parameter(
            shape=[config.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.config.use_fused_rms_norm:
            return paddle.incubate.nn.functional.fused_rms_norm_ext(x, self.weight, self.eps)[0].astype(
                self.weight.dtype
            )

        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight


QWenLMHeadModel = QWenForCausalLM
