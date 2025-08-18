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
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

from ...utils.converter import StateDictNameMapping, init_name_mappings
from ...utils.log import logger
from ..model_outputs import BaseModelOutputWithPast
from ..model_utils import PretrainedModel
from .configuration import QWenConfig

__all__ = [
    "QWenBlockAuto",
    "QWenForCausalLM3DAuto",
    "QWenPretrainedModelAuto",
    "QWenModelAuto",
    "QWenLMHeadAuto",
    "QWenPretrainingCriterionAuto",
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

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


def get_mesh(pp_idx=0):
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp")[pp_idx]
    return mesh


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


def enable_fuse_ffn_qkv_pass():
    if os.getenv("FLAGS_enable_fused_ffn_qkv_pass") in [
        "True",
        "true",
        "1",
    ]:
        return True
    else:
        return False


def get_use_casual_mask():
    """Get the value of the 'USE_CASUAL_MASK' environment variable."""
    return os.getenv("USE_CASUAL_MASK", "False") == "True"


attention_cnt = 0


class QWenAttentionAuto(nn.Layer):
    def __init__(self, config, ipp=None):
        super().__init__()

        self.config = config
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self.scale_attn_weights = True
        self.enable_recompute = config.use_recompute
        self.recompute_granularity = config.recompute_granularity

        self.projection_size = config.kv_channels * config.num_attention_heads

        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads

        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size, bias_attr=True)
        self.c_proj = nn.Linear(config.hidden_size, self.projection_size, bias_attr=not config.no_bias)

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(self.hidden_size_per_attention_head * config.rotary_pct)
        dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
        self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [math.log(i, self.seq_length) if i > self.seq_length else 1 for i in range(1, MAX_NTK_SEQ_LENGTH)]
        self.logn_tensor = paddle.to_tensor(logn_list)[None, :, None, None]
        self._ntk_cached = 1.0

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.ipp = ipp
        global attention_cnt
        self.attention_cnt = attention_cnt
        attention_cnt += 1
        self.c_attn.weight = dist.shard_tensor(
            self.c_attn.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(1)]
        )
        self.c_attn.bias = dist.shard_tensor(self.c_attn.bias, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(0)])
        self.c_proj.weight = dist.shard_tensor(
            self.c_proj.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(0)]
        )

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
                attn_output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    is_causal=attention_mask is None,
                )
                attn_weights = None
            return attn_output, attn_weights
        else:
            # [bz, sql, nh, hid] ==> [bz, nh, sql hdim]
            query = query.transpose([0, 2, 1, 3])
            # [bz, sql, nh, hid] ==> [bz, nh, sql hdim]
            key = key.transpose([0, 2, 1, 3])
            # [bz, sql, nh, hid] ==> [bz, nh, sql hdim]
            value = value.transpose([0, 2, 1, 3])

            attn_weights = paddle.matmul(query / math.sqrt(head_dim), key.transpose([0, 1, 3, 2]))

            if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.shape}"
                )
            # If the attention mask is None, we need to construct the causal attention mask
            if attention_mask is None:
                attention_mask = get_triangle_upper_mask(attn_weights)
            attn_weights = attn_weights + attention_mask
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(value.dtype)

            attn_weights = self.attn_dropout(attn_weights)
            attn_output = paddle.matmul(attn_weights, value)
            attn_output = attn_output.transpose([0, 2, 1, 3])
            return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.shape[:-2] + [
            num_heads * attn_head_size,
        ]
        return tensor.reshape(new_shape)

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
        # # [bz, sql, hid] ==> [bz, sql, 3*hid]
        mixed_x_layer = self.c_attn(hidden_states)
        # [bz, sql, 3*hid] ==> [bz, sql, hid]
        target_shape = [0, 0, self.num_heads, 3 * self.head_dim]

        mixed_x_layer = paddle.reshape_(mixed_x_layer, target_shape)
        query, key, value = paddle.split(mixed_x_layer, num_or_sections=3, axis=-1)

        # [bz, sql, hid] ==> [bz, sql, nh, hdim]
        # query = self._split_heads(query, self.num_heads, self.head_dim)
        # key = self._split_heads(key, self.num_heads, self.head_dim)
        # value = self._split_heads(value, self.num_heads, self.head_dim)

        kv_seq_len = hidden_states.shape[1]
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
            attn_output, attn_weight = recompute(
                self._attn, query, key, value, attention_mask, use_reentrant=self.config.recompute_use_reentrant
            )
        else:
            attn_output, attn_weight = self._attn(query, key, value, attention_mask)
        context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(context_layer)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight,)

        return outputs


class QWenMLPAuto(nn.Layer):
    def __init__(self, config, ipp=None):
        super().__init__()
        ff_dim_in = config.intermediate_size // 2
        self.fuse_attention_ffn = config.fuse_attention_ffn
        self.ipp = ipp
        if config.fuse_attention_ffn and not enable_fuse_ffn_qkv_pass():
            self.gate_up_fused_proj = nn.Linear(config.hidden_size, ff_dim_in * 2, bias_attr=not config.no_bias)
            self.gate_up_fused_proj.weight = dist.shard_tensor(
                self.gate_up_fused_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
        else:
            self.w1 = nn.Linear(config.hidden_size, ff_dim_in, bias_attr=not config.no_bias)
            self.w2 = nn.Linear(config.hidden_size, ff_dim_in, bias_attr=not config.no_bias)
            self.w1.weight = dist.shard_tensor(self.w1.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(1)])
            self.w2.weight = dist.shard_tensor(self.w2.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(1)])
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias_attr=not config.no_bias)
        self.c_proj.weight = dist.shard_tensor(
            self.c_proj.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(0)]
        )

    def forward(self, hidden_states):
        # # up
        # a1 = self.w1(hidden_states)
        # # gate
        # a2 = self.w2(hidden_states)
        # intermediate_parallel = a1 * F.silu(a2)
        # down
        if self.fuse_attention_ffn and not enable_fuse_ffn_qkv_pass():
            intermediate_parallel = swiglu(self.gate_up_fused_proj(hidden_states))
        else:
            intermediate_parallel = swiglu(self.w2(hidden_states), self.w1(hidden_states))
        output = self.c_proj(intermediate_parallel)
        return output


class QWenBlockAuto(nn.Layer):
    def __init__(self, config, ipp=None, idx=None):
        super().__init__()
        self.config = config
        self.ipp = ipp
        self.ln_1 = QWenRMSNormAuto(config, self.ipp)
        self.attn = QWenAttentionAuto(config, ipp)
        self.ln_2 = QWenRMSNormAuto(config, self.ipp)
        self.mlp = QWenMLPAuto(config, ipp)
        self.ipp = ipp
        self.idx = idx

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
        layernorm_output = self.ln_1(hidden_states)
        attention_mask = (
            dist.reshard(attention_mask, get_mesh(self.ipp), [dist.Shard(0), dist.Replicate()])
            if attention_mask is not None
            else attention_mask
        )
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

        residual = hidden_states
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


class QWenPretrainedModelAuto(PretrainedModel):
    config_class = QWenConfig
    base_model_prefix = "qwen"

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
                "qwen.h.0.mlp.w2.weight": partial(fn, is_column=True),
                "qwen.h.0.mlp.w1.weight": partial(fn, is_column=True),
                "qwen.h.0.attn.c_attn.weight": partial(fn, is_column=True, is_naive_3fuse=True),
                "qwen.h.0.attn.c_attn.bias": partial(fn, is_column=True, is_naive_3fuse=True),
                # Row Linear
                "qwen.wte.weight": partial(fn, is_column=False),
                "qwen.h.0.mlp.c_proj.weight": partial(fn, is_column=False),
                "qwen.h.0.attn.c_proj.weight": partial(fn, is_column=False),
            }
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


class QWenModelAuto(QWenPretrainedModelAuto):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.enable_recompute = config.use_recompute
        self.recompute_granularity = config.recompute_granularity

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.wte.weight = dist.shard_tensor(self.wte.weight, get_mesh(), [dist.Replicate(), dist.Shard(1)])
        self.drop = nn.Dropout(config.emb_dropout_prob)

        self.h = nn.LayerList(
            [
                QWenBlockAuto(
                    config,
                    self.get_layer_ipp(i),
                    i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = QWenRMSNormAuto(config, self.get_last_layer_ipp())

    def get_layer_ipp(self, layer_index):
        mesh = fleet.auto.get_mesh()
        if "pp" not in mesh.dim_names:
            return None
        else:
            pp_degree = mesh.get_dim_size("pp")
            layer_per_stage = math.ceil(self.config.num_hidden_layers / pp_degree)
            return layer_index // layer_per_stage

    def get_last_layer_ipp(self):
        return self.get_layer_ipp(self.config.num_hidden_layers - 1)

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

        hidden_states = recompute(
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

    def get_masks(self, batch_size, seq_length, past_length, dtype, padding_mask=None):
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
            with paddle.amp.auto_cast(False):
                inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds
        hidden_states = dist.reshard(hidden_states, get_mesh(), [dist.Shard(0), dist.Replicate()])

        use_casual_mask = get_use_casual_mask()
        if use_casual_mask:
            attention_mask = None
        else:
            # bool 4D mask
            attention_mask = self.get_masks(
                input_shape[0], input_shape[1], past_length, dtype=hidden_states.dtype, padding_mask=attention_mask
            )
            # TODO(GhostScreaming): how to fix paddle.finfo?
            zero = paddle.zeros(attention_mask.shape, dtype=paddle.bfloat16)
            neg_inf = paddle.full_like(attention_mask, paddle.finfo(paddle.bfloat16).min, dtype=paddle.bfloat16)
            # dtype 4D mask
            attention_mask = paddle.where(attention_mask, zero, neg_inf)
            attention_mask = dist.shard_tensor(attention_mask, get_mesh(), [dist.Replicate(), dist.Replicate()])
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + [
            hidden_states.shape[-1],
        ]

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with recompute")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        pre_ipp = 0
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            has_gradient = not hidden_states.stop_gradient
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if block.ipp is not None and pre_ipp != block.ipp:
                hidden_states = dist.reshard(
                    hidden_states,
                    get_mesh(block.ipp),
                    [dist.Shard(0), dist.Replicate()],
                )
                if position_ids is not None:
                    position_ids = dist.reshard(
                        position_ids,
                        get_mesh(block.ipp),
                        [dist.Shard(0), dist.Replicate()],
                    )
                if attention_mask is not None:
                    attention_mask = dist.reshard(
                        attention_mask,
                        get_mesh(block.ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
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
            pre_ipp = block.ipp

            if type(outputs) is tuple:
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

            if use_cache is True:
                presents = presents + (outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.reshape(output_shape)
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


class QWenLMHeadAuto(nn.Layer):
    def __init__(self, config: QWenConfig, ipp=None):
        super(QWenLMHeadAuto, self).__init__()
        self.config = config
        vocab_size = config.vocab_size
        self.ipp = ipp
        self.weight = self.create_parameter(
            shape=[config.hidden_size, vocab_size],
            dtype=paddle.get_default_dtype(),
        )
        self.weight = dist.shard_tensor(self.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(1)])

    def forward(self, hidden_states, tensor_parallel_output=None):
        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output

        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits


loss_cnt = 0


class QWenPretrainingCriterionAuto(paddle.nn.Layer):
    """
    Criterion for Llama.
    It calculates the final loss.
    """

    def __init__(self, config):

        super(QWenPretrainingCriterionAuto, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels):
        global loss_cnt
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                warnings.warn(
                    f"enable_parallel_cross_entropy, the vocab_size should be splited: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            # skip ignore_index which loss == 0
            masked_lm_loss = paddle.masked_select(masked_lm_loss, masked_lm_loss > 0).astype("float32")
            loss = paddle.mean(masked_lm_loss)

        loss_cnt += 1
        return loss


class QWenForCausalLM3DAuto(QWenPretrainedModelAuto):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]

    def __init__(self, config):
        super().__init__(config)
        self.qwen = QWenModelAuto(config)
        self.lm_head = QWenLMHeadAuto(config, self.qwen.get_last_layer_ipp())

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

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        )
        lm_logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)

        return lm_logits


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
                freqs = paddle.outer(seq.astype(paddle.float32), self.inv_freq.astype(paddle.float32))
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


class QWenRMSNormAuto(nn.Layer):
    def __init__(self, config, ipp):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_epsilon
        self.weight = paddle.create_parameter(
            shape=[config.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.weight = dist.shard_tensor(self.weight, get_mesh(ipp), [dist.Replicate(), dist.Replicate()])

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.config.use_fused_rms_norm:
            return paddle.incubate.nn.functional.fused_rms_norm_ext(x, self.weight, self.eps)[0].astype(
                self.weight.dtype
            )

        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight
