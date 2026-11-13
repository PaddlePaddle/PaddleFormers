# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from copy import copy
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import VocabParallelEmbedding
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import fused_swiglu
from ...nn.norm import Norm as GeneralNorm
from ...nn.norm import RMSNorm
from ...nn.pp_model import EmbeddingPipe, GeneralModelForCausalLMPipe, parse_args
from ...utils.log import logger
from ...utils.masking_utils import _gen_from_sparse_attn_mask_indices
from ..cache_utils import Cache, DynamicCache
from ..linear_utils import (
    ColumnParallelLinear,
    ColumnSequenceParallelLinear,
    RowParallelLinear,
    RowSequenceParallelLinear,
)
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from .configuration import OpenELMConfig, make_divisible


class OpenELMPretrainedModel(PretrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OpenELMConfig
    base_model_prefix = "model"
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    def _init_weights(self, module: nn.Layer) -> None:
        """Initialize the weights."""
        if isinstance(
            module,
            (
                nn.Linear,
                ColumnParallelLinear,
                RowParallelLinear,
                ColumnSequenceParallelLinear,
                RowSequenceParallelLinear,
            ),
        ):
            module.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=getattr(module, "_openelm_init_std", module.weight.shape[0] ** -0.5),
                    shape=module.weight.shape,
                ).cast(module.weight.dtype)
            )
            if module.bias is not None:
                module.bias.set_value(paddle.zeros(module.bias.shape, dtype=module.bias.dtype))
        elif isinstance(module, (nn.Embedding, VocabParallelEmbedding)):
            module.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=module.weight.shape[1] ** -0.5,
                    shape=module.weight.shape,
                ).cast(module.weight.dtype)
            )
        elif isinstance(module, RMSNorm):
            module.weight.set_value(paddle.ones(module.weight.shape, dtype=module.weight.dtype))
        elif isinstance(module, GeneralLMHead) and not self.config.share_input_output_layers:
            module.weight.set_value(
                paddle.normal(mean=0.0, std=self.config.initializer_range, shape=module.weight.shape).cast(
                    module.weight.dtype
                )
            )

    @staticmethod
    def _validate_config(config):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        tp_size = config.tensor_model_parallel_size
        if config.sequence_parallel and tp_size == 1:
            raise ValueError("Sequence parallelism requires tensor_model_parallel_size > 1.")
        for layer_idx, (q_heads, kv_heads) in enumerate(zip(config.num_query_heads, config.num_kv_heads)):
            width = int(make_divisible(config.ffn_multipliers[layer_idx] * config.model_dim, config.ffn_dim_divisor))
            if q_heads % tp_size or kv_heads % tp_size or width % tp_size:
                raise ValueError(
                    f"Layer {layer_idx} query/KV heads and MLP width must be divisible by TP size {tp_size}."
                )
        if config._attn_implementation not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(f"Unknown attention backend: {config._attn_implementation}.")
        if config.apply_rope_fusion:
            raise ValueError("OpenELM currently requires apply_rope_fusion=False.")
        if config.fuse_swiglu and (not config.ffn_with_glu or config.activation_fn_name != "swish"):
            raise ValueError("fuse_swiglu requires a GLU FFN with swish activation.")

    @staticmethod
    def _prepare_attention_mask(
        config,
        inputs_embeds,
        batch_size,
        seq_length,
        cache_length,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
    ):
        if cache_length and attn_mask_startend_row_indices is not None:
            indices = attn_mask_startend_row_indices
            if indices.ndim == 3:
                indices = indices.unsqueeze(-1)
            dense = _gen_from_sparse_attn_mask_indices(indices, inputs_embeds.dtype, is_causal=indices.shape[-1] != 4)
            return dense[:, :, cache_length : cache_length + seq_length, :], None
        # Preserve complete masks and bottom-right causal alignment for chunked decode.
        mask_config = config
        if (attention_mask is not None and attention_mask.ndim >= 3) or (cache_length and seq_length > 1):
            mask_config = copy(config)
            mask_config._attn_implementation = "eager"

        def prepare_mask(attention_mask, **kwargs):
            if attention_mask is not None and attention_mask.ndim >= 3 and paddle.is_floating_point(attention_mask):
                return attention_mask.unsqueeze(1) if attention_mask.ndim == 3 else attention_mask
            return PretrainedModel._prepare_decoder_attention_mask(attention_mask=attention_mask, **kwargs)

        return create_causal_mask_and_row_indices(
            config=mask_config,
            inputs_embeds=inputs_embeds,
            batch_size=batch_size,
            seq_length=seq_length,
            cache_length=cache_length,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            prepare_decoder_attention_mask=prepare_mask,
        )

    @classmethod
    def _gen_aoa_config(cls, config: OpenELMConfig):
        """Map external names and contiguous fused projections to the model layout."""
        return cls._build_aoa_config(config, inverse=False)

    @classmethod
    def _gen_inv_aoa_config(cls, config: OpenELMConfig):
        """Restore external names and contiguous fused projection layout."""
        return cls._build_aoa_config(config, inverse=True)

    @classmethod
    def _build_aoa_config(cls, config: OpenELMConfig, inverse: bool):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        statements = []

        def map_weight(source, target, transpose=False):
            if inverse:
                source, target = target, source
            statements.append(f"{source}{'^T' if transpose else ''} -> {target}")

        map_weight("transformer.token_embeddings.weight", f"{model_prefix}embed_tokens.weight")
        map_weight("transformer.norm.weight", f"{model_prefix}norm.weight")
        for layer_idx in range(config.num_transformer_layers):
            source = f"transformer.layers.{layer_idx}"
            target = f"{model_prefix}layers.{layer_idx}"
            map_weight(f"{source}.attn_norm.weight", f"{target}.input_layernorm.weight")
            map_weight(f"{source}.ffn_norm.weight", f"{target}.post_attention_layernorm.weight")
            map_weight(f"{source}.attn.out_proj.weight", f"{target}.self_attn.o_proj.weight", transpose=True)
            if config.normalize_qk_projections:
                for norm in ("q_norm", "k_norm"):
                    map_weight(f"{source}.attn.{norm}.weight", f"{target}.self_attn.{norm}.weight")

            # Split into KV-width chunks: Q and KV widths differ under GQA.
            groups = config.num_query_heads[layer_idx] // config.num_kv_heads[layer_idx]
            q_chunks = [f"openelm_qkv_tmp.{layer_idx}.q_{i}" for i in range(groups)]
            k_weight = f"{target}.self_attn.k_proj.weight"
            v_weight = f"{target}.self_attn.v_proj.weight"
            q_weight = f"{target}.self_attn.q_proj.weight"
            chunks = ", ".join(q_chunks + [k_weight, v_weight])
            fused = f"{source}.attn.qkv_proj.weight"
            if inverse:
                statements.append(
                    f"{q_weight} -> {', '.join(q_chunks)}, axis=1" if groups > 1 else f"{q_weight} -> {q_chunks[0]}"
                )
                statements.append(f"{chunks} -> openelm_qkv_tmp.{layer_idx}.weight, axis=1")
                statements.append(f"openelm_qkv_tmp.{layer_idx}.weight^T -> {fused}")
            else:
                statements.append(f"{fused}^T -> {chunks}, axis=1")
                statements.append(
                    f"{', '.join(q_chunks)} -> {q_weight}, axis=1" if groups > 1 else f"{q_chunks[0]} -> {q_weight}"
                )

            map_weight(f"{source}.ffn.proj_2.weight", f"{target}.mlp.down_proj.weight", transpose=True)
            if config.ffn_with_glu:
                gate = f"{target}.mlp.gate_proj.weight"
                up = f"{target}.mlp.up_proj.weight"
                if inverse:
                    statements.append(f"{gate}^T, {up}^T -> {source}.ffn.proj_1.weight, axis=0")
                else:
                    statements.append(f"{source}.ffn.proj_1.weight^T -> {gate}, {up}, axis=1")
            else:
                map_weight(f"{source}.ffn.proj_1.weight", f"{target}.mlp.up_proj.weight", transpose=True)

        if cls != cls.base_model_class:
            if config.share_input_output_layers:
                if not inverse:
                    statements.append("transformer.token_embeddings.weight -> lm_head.weight")
                else:
                    statements.append("lm_head.weight -> _")
            else:
                map_weight("lm_head.weight", "lm_head.weight")
        return {"aoa_statements": statements}


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = paddle.chunk(x, chunks=2, axis=-1)
    return paddle.concat([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    return (x * pos_cos) + (_rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(nn.Layer):
    """OpenELM rotary frequencies using explicit per-sample positions."""

    def __init__(self, config: OpenELMConfig):
        super().__init__()
        self.config = config
        inv_freq = 1.0 / (
            config.rope_freq_constant ** (paddle.arange(0, config.head_dim, 2, dtype="float32") / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        with paddle.amp.auto_cast(enable=False):
            inv_freq = self.inv_freq
            if inv_freq.dtype != paddle.float32:
                # Layer.to(dtype) can cast non-persistent buffers as well.
                inv_freq = 1.0 / (
                    self.config.rope_freq_constant
                    ** (paddle.arange(0, self.config.head_dim, 2, dtype="float32") / self.config.head_dim)
                )
            freqs = position_ids.astype("float32").unsqueeze(-1) * inv_freq
            emb = paddle.concat([freqs, freqs], axis=-1)
            return emb.cos(), emb.sin()


class OpenELMAttention(nn.Layer):
    """Multi-head causal attention with optional Group Query Attention (GQA)."""

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]
        tp_size = max(config.tensor_model_parallel_size, 1)
        if q_heads % tp_size or k_heads % tp_size:
            raise ValueError("Query and KV head counts must be divisible by the tensor parallel size.")

        self.q_proj = GeneralLinear.create(
            config.model_dim, q_heads * head_dim, has_bias=False, config=config, tp_plan="colwise"
        )
        self.q_proj._openelm_init_std = config.model_dim**-0.5
        self.k_proj = GeneralLinear.create(
            config.model_dim, k_heads * head_dim, has_bias=False, config=config, tp_plan="colwise"
        )
        self.k_proj._openelm_init_std = config.model_dim**-0.5
        self.v_proj = GeneralLinear.create(
            config.model_dim, v_heads * head_dim, has_bias=False, config=config, tp_plan="colwise"
        )
        self.v_proj._openelm_init_std = config.model_dim**-0.5

        if config.normalize_qk_projections:
            self.q_norm = GeneralNorm.create(
                config,
                hidden_size=config.head_dim,
                norm_eps=config.rms_norm_eps,
                input_is_parallel=config.tensor_model_parallel_size > 1,
            )
            self.k_norm = GeneralNorm.create(
                config,
                hidden_size=config.head_dim,
                norm_eps=config.rms_norm_eps,
                input_is_parallel=config.tensor_model_parallel_size > 1,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.o_proj = GeneralLinear.create(
            q_heads * head_dim, config.model_dim, has_bias=False, config=config, tp_plan="rowwise"
        )

        self.o_proj._openelm_init_std = (config.model_dim**-0.5) * ((2 * config.num_transformer_layers) ** -0.5)

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads // tp_size
        self.num_k_heads = k_heads // tp_size
        self.num_v_heads = v_heads // tp_size
        self.model_dim = config.model_dim
        self.num_key_value_groups = self.num_q_heads // self.num_k_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        output_attentions: bool = False,
    ):
        if self.config.sequence_parallel:
            seq_length = self.config.max_sequence_length
            batch_size = hidden_states.shape[0] * self.config.tensor_model_parallel_size // seq_length
        else:
            batch_size, seq_length, _ = hidden_states.shape
        queries = (
            self.q_proj(hidden_states)
            .reshape([batch_size, seq_length, self.num_q_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        keys = (
            self.k_proj(hidden_states)
            .reshape([batch_size, seq_length, self.num_k_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        values = (
            self.v_proj(hidden_states)
            .reshape([batch_size, seq_length, self.num_v_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        cos, sin = position_embeddings
        queries = _apply_rotary_pos_emb(queries.astype("float32"), sin.unsqueeze(1), cos.unsqueeze(1)).astype(
            queries.dtype
        )
        keys = _apply_rotary_pos_emb(keys.astype("float32"), sin.unsqueeze(1), cos.unsqueeze(1)).astype(keys.dtype)
        if past_key_values is not None:
            keys, values = past_key_values.update(keys, values, self.layer_idx)
        backend = self.config._attn_implementation
        if backend == "flashmask" and attention_mask is not None:
            backend = "sdpa"
        if output_attentions and backend in ("sdpa", "flashmask"):
            raise ValueError("Attention weights are only available with the eager attention backend.")
        attention_interface = ALL_ATTENTION_FUNCTIONS[backend]
        attn_output, attn_weights = attention_interface(
            self,
            query=queries,
            key=keys,
            value=values,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            scaling=self.scaling,
            dropout=0.0,
        )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        return self.o_proj(attn_output), attn_weights


class OpenELMMLP(nn.Layer):
    """Feed-forward network with optional GLU."""

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        self.ffn_with_glu = config.ffn_with_glu
        self.fuse_swiglu = config.fuse_swiglu
        if self.ffn_with_glu:
            self.gate_proj = GeneralLinear.create(
                config.model_dim, intermediate_dim, has_bias=False, config=config, tp_plan="colwise"
            )
            self.gate_proj._openelm_init_std = config.model_dim**-0.5
        self.up_proj = GeneralLinear.create(
            config.model_dim, intermediate_dim, has_bias=False, config=config, tp_plan="colwise"
        )
        self.up_proj._openelm_init_std = config.model_dim**-0.5
        self.down_proj = GeneralLinear.create(
            intermediate_dim, config.model_dim, has_bias=False, config=config, tp_plan="rowwise"
        )
        self.down_proj._openelm_init_std = (config.model_dim**-0.5) * ((2 * config.num_transformer_layers) ** -0.5)

        if config.activation_fn_name == "swish":
            self.act = nn.Silu()
        elif config.activation_fn_name == "gelu":
            self.act = nn.GELU()
        elif config.activation_fn_name == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.activation_fn_name}")

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def forward(self, x: Tensor) -> Tensor:
        if self.ffn_with_glu:
            gate, up = self.gate_proj(x), self.up_proj(x)
            hidden = fused_swiglu(paddle.concat([gate, up], axis=-1)) if self.fuse_swiglu else self.act(gate) * up
            return self.down_proj(hidden)
        return self.down_proj(self.act(self.up_proj(x)))


class OpenELMDecoderLayer(nn.Layer):
    """Transformer decoder layer."""

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.self_attn = OpenELMAttention(config=config, layer_idx=layer_idx)
        self.mlp = OpenELMMLP(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = GeneralNorm.create(
            config,
            hidden_size=config.model_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.input_layernorm = GeneralNorm.create(
            config,
            hidden_size=config.model_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states, attn_weights) if output_attentions else hidden_states


@register_base_model
class OpenELMModel(OpenELMPretrainedModel):
    """OpenELM Model (base model without language modeling head)."""

    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.config = config
        self._validate_config(config)

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.model_dim, padding_idx=None
        )

        self.layers = nn.LayerList(
            [
                OpenELMDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_transformer_layers)
            ]
        )
        self.norm = GeneralNorm.create(
            config,
            hidden_size=config.model_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.num_transformer_layers = config.num_transformer_layers
        self.gradient_checkpointing = False

        self.rotary_emb = OpenELMRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: paddle.Tensor):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if output_attentions and self.config._attn_implementation in ("sdpa", "flashmask"):
            raise ValueError("Attention weights are only available with the eager attention backend.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        recompute_active = (
            self.training
            and paddle.is_grad_enabled()
            and (
                self.gradient_checkpointing
                or (
                    self.config.recompute_granularity == "full"
                    and self.config.recompute_method == "uniform"
                    and self.config.recompute_num_layers == 1
                )
            )
        )
        if recompute_active and (use_cache or past_key_values is not None):
            if past_key_values is not None:
                raise ValueError("A mutable cache cannot be passed during activation recomputation.")
            logger.warning("Disabling cache during activation recomputation.")
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length, _ = inputs_embeds.shape
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            positions = (
                cache_position
                if cache_position is not None
                else paddle.arange(cache_length, cache_length + seq_length, dtype="int64")
            )
            position_ids = positions.unsqueeze(0).tile([batch_size, 1])
        causal_mask, row_indices = self._prepare_attention_mask(
            self.config,
            inputs_embeds,
            batch_size,
            seq_length,
            cache_length,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        if self.config.sequence_parallel:
            if seq_length != self.config.max_sequence_length:
                raise ValueError("Sequence parallel input length must equal config.max_sequence_length.")
            hidden_states = ScatterOp.apply(hidden_states.reshape([-1, hidden_states.shape[-1]]))
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_kwargs = dict(
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                attn_mask_startend_row_indices=row_indices,
                output_attentions=output_attentions,
            )
            if recompute_active:
                from paddle.distributed.fleet.utils import recompute

                layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    position_embeddings,
                    None,
                    row_indices,
                    output_attentions,
                    use_reentrant=self.config.recompute_use_reentrant and not hidden_states.stop_gradient,
                )
            else:
                layer_outputs = decoder_layer(hidden_states, **layer_kwargs)
            if output_attentions:
                hidden_states, attn_weights = layer_outputs
                all_attentions.append(attn_weights)
            else:
                hidden_states = layer_outputs
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )
        return result if return_dict else result.to_tuple()


class OpenELMForCausalLM(OpenELMPretrainedModel):
    """OpenELM model with a language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.model = OpenELMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        loss_mask: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices", None)
        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning("Both dense and row-index masks were supplied; using row indices.")
            attention_mask = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels, loss_mask=loss_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else None,
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Prepare inputs for generation."""
        past_length = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, Cache):
                raise TypeError("past_key_values must use the shared Cache interface.")
            cache_length = past_key_values.get_seq_length()
            past_length = cache_length
            max_cache_length = past_key_values.get_max_cache_shape()

            # Keep only the unprocessed tokens:
            if (
                attention_mask is not None
                and attention_mask.ndim == 2
                and attention_mask.shape[-1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[-1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and max_cache_length > 0
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[..., -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and attention_mask.ndim == 2 and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.cast(paddle.int64).cumsum(axis=-1) - 1
            position_ids = paddle.where(
                attention_mask == 0,
                paddle.ones_like(position_ids),
                position_ids,
            )
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        query_length = (
            inputs_embeds.shape[1] if inputs_embeds is not None and past_key_values is None else input_ids.shape[-1]
        )
        if position_ids is None:
            position_ids = paddle.arange(past_length, past_length + query_length, dtype="int64").unsqueeze(0)
        elif past_key_values is not None:
            position_ids = position_ids[:, -query_length:]
        cache_position = paddle.arange(past_length, past_length + query_length, dtype="int64")
        if attention_mask is not None and attention_mask.ndim >= 3 and past_key_values is not None:
            attention_mask = attention_mask[..., -query_length:, :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class OpenELMEmbeddingPipe(EmbeddingPipe):
    """Prepare one mask and rotary tensor before pipeline communication."""

    def forward(self, args):
        input_ids, attention_mask, position_ids, _, _ = parse_args(args, is_embed=True)
        inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).tile([batch_size, 1])
        row_indices = None
        if attention_mask is not None and attention_mask.dtype == paddle.int32 and attention_mask.ndim >= 3:
            row_indices, attention_mask = attention_mask, None
        mask, row_indices = OpenELMPretrainedModel._prepare_attention_mask(
            self.config,
            inputs_embeds,
            batch_size,
            seq_length,
            0,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=row_indices,
        )
        positions = paddle.stack(self.rotary_emb(inputs_embeds, position_ids))
        if self.config.sequence_parallel:
            if seq_length != self.config.max_sequence_length:
                raise ValueError("Sequence parallel input length must equal config.max_sequence_length.")
            inputs_embeds = ScatterOp.apply(inputs_embeds.reshape([-1, inputs_embeds.shape[-1]]))
        if row_indices is not None:
            mask = row_indices
        if mask is None:
            return inputs_embeds, position_ids, positions
        return inputs_embeds, mask, position_ids, positions


class OpenELMCriterionPipe(CriterionLayer):
    """Preserve the SFT loss mask across the pipeline criterion boundary."""

    def __init__(self, config, **kwargs):
        super().__init__(config, return_tuple=False, **kwargs)

    def forward(self, logits, labels):
        loss_mask = None
        if isinstance(labels, tuple) and "sft" in self.loss_type:
            labels, loss_mask = labels
        return super().forward(logits, labels, loss_mask=loss_mask)


class OpenELMForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = OpenELMConfig
    _decoder_layer_cls = OpenELMDecoderLayer
    _rotary_emb_cls = OpenELMRotaryEmbedding
    _embedding_pipe_cls = OpenELMEmbeddingPipe
    _criterion_pipe_cls = OpenELMCriterionPipe
    _init_weights = OpenELMPretrainedModel._init_weights
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = OpenELMPretrainedModel.transpose_weight_keys
    _gen_aoa_config = OpenELMPretrainedModel._gen_aoa_config
    _gen_inv_aoa_config = OpenELMPretrainedModel._gen_inv_aoa_config

    def __init__(self, config):
        OpenELMPretrainedModel._validate_config(config)
        self._openelm_initializer_range = config.initializer_range
        super().__init__(config)

    def get_sequential_layers(self):
        self.config.initializer_range = self._openelm_initializer_range
        layers = super().get_sequential_layers()
        for layer in layers:
            # The shared embedding branch in the builder uses the base class directly.
            if getattr(layer, "layer_func", None) is EmbeddingPipe:
                layer.layer_func = OpenELMEmbeddingPipe
        return layers
