# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen3 model."""
from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Qwen3MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..contrastive_loss import SimpleContrastiveLoss
from ..embedding_utils import dist_gather_tensor_with_gradient
from ..masking_utils import create_causal_masks_and_row_indices
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Qwen3Config


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


def _apply_rotary_emb(
    x: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
) -> paddle.Tensor:
    x = x.transpose([0, 2, 1, 3])
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.transpose([0, 2, 1, 3])


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


class Qwen3Attention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen3Config, layer_idx: int = 0):
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

        self.tensor_parallel = config.tensor_parallel_degree > 1
        self.sequence_parallel = config.sequence_parallel

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree

        kv_hidden_size = self.config.num_key_value_heads * self.head_dim
        q_hidden_size = self.config.num_attention_heads * self.head_dim

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
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_parallel_degree // max_sequence_length
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape
        # Add qk norm for Qwen3 model.
        query_states = self.q_norm(query_states.reshape([bsz, q_len, -1, self.head_dim]))
        key_states = self.k_norm(key_states.reshape([bsz, q_len, -1, self.head_dim]))
        value_states = value_states.reshape([bsz, q_len, -1, self.head_dim])

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            key_states = paddle.cat([past_key_value[0], key_states], axis=1)
            value_states = paddle.cat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
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

        return attn_output, past_key_value


class Qwen3DecoderLayer(nn.Layer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config, layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.attention_type = config.layer_types[layer_idx]

        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            batch_size=batch_size,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return (
                hidden_states,
                present_key_value,
            )
        else:
            return hidden_states


class Qwen3PretrainedModel(PretrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Qwen3Config, is_split=True):
        """Generate tensor parallel mappings for model conversion."""
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        LAYER_COLWISE = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "mlp.up_proj.weight",
            "mlp.gate_proj.weight",
        ]

        LAYER_ROWWISE = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]

        BIAS_KEYS = [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]

        def make_base_actions():
            actions = {
                "lm_head.weight": partial(fn, is_column=False),
                "embed_tokens.weight": partial(fn, is_column=False),
            }
            for layer_idx in range(config.num_hidden_layers):
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                        for k in LAYER_COLWISE
                    }
                )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
                        for k in LAYER_ROWWISE
                    }
                )
                # bias
                if config.attention_bias:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                            for b in BIAS_KEYS
                        }
                    )

            return actions

        mappings = make_base_actions()
        return mappings


class Qwen3RotaryEmbedding(nn.Layer):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    def forward(self, x, position_ids):
        # NOTE: Paddle's Automatic Mixed Precision (AMP) has a default op whitelist that may automatically cast
        # certain operations (like matmul) to FP16/BF16 for performance optimization. However, in scenarios where
        # numerical stability is critical (e.g., RoPE init/compute), this conversion can lead to precision loss.
        # Disabling auto_cast here ensures the matmul operation runs in the original precision (FP32) as intended.
        with paddle.amp.auto_cast(False):
            inv_freq_expanded = (
                self.inv_freq.unsqueeze(0)
                .unsqueeze(-1)
                .cast(paddle.float32)
                .expand([position_ids.shape[0], -1, 1])
                .to(x.place)
            )
            position_ids_expanded = position_ids.unsqueeze(1).cast(paddle.float32)

            freqs = paddle.matmul(inv_freq_expanded, position_ids_expanded).transpose([0, 2, 1])
            emb = paddle.cat((freqs, freqs), axis=-1)
            cos = paddle.cos(emb) * self.attention_scaling
            sin = paddle.sin(emb) * self.attention_scaling

        return cos.cast(dtype=x.dtype), sin.cast(dtype=x.dtype)


@register_base_model
class Qwen3Model(Qwen3PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        past_key_value: Tensor,
        use_cache: bool,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices=None,
        batch_size: int = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            past_key_value,
            use_cache,
            position_embeddings,
            attn_mask_startend_row_indices,
            batch_size,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

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
            inputs_embeds = self.embed_tokens(input_ids)

        cache_length = 0
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        else:
            cache_length = past_key_values[0][0].shape[1]

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
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
        causal_mask_mapping, attn_mask_startend_row_indices_mapping = create_causal_masks_and_row_indices(
            **mask_kwargs
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        next_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            has_gradient = not hidden_states.stop_gradient
            if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    past_key_value,
                    use_cache,
                    position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    batch_size=batch_size,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    past_key_value,
                    use_cache,
                    position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    batch_size=batch_size,
                )

            if use_cache:
                hidden_states = layer_outputs[0]
                next_cache += (layer_outputs[1],)
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class Qwen3ForCausalLM(Qwen3PretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
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
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.cat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            # TODO: support attention mask for other models
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.cat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )
            elif len(attention_mask.shape) == 4:
                model_kwargs["attention_mask"] = paddle.cat(
                    [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )[:, :, -1:, :]

        return model_kwargs

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        loss_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3ForSequenceClassification(Qwen3PretrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        self.score = GeneralLinear.create(config.hidden_size, self.num_labels, has_bias=False, linear_type="default")

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = paddle.eq(input_ids, self.config.pad_token_id).astype("int32").argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths
            else:
                sequence_lengths = -1

        # pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]
        pooled_logits = logits.gather_nd(paddle.stack([paddle.arange(logits.shape[0]), sequence_lengths], axis=-1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Qwen3ForTokenClassification(Qwen3PretrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = GeneralLinear.create(config.hidden_size, config.num_labels, has_bias=False, linear_type="default")

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3SentenceEmbedding(Qwen3PretrainedModel):
    def __init__(
        self,
        config: Qwen3Config,
        embedding_temperature: float = 0.02,
    ):
        """Qwen3SentenceEmbedding
        For getting larger batch_size, we use tensor parallel to get larger batch_size.

        Args:
            config (Qwen3Config): _description_
            model (Qwen3Model): _description_
            embedding_temperature (float, optional): _description_. Defaults to 0.02.
        """
        super(Qwen3SentenceEmbedding, self).__init__(config)
        self.config = config
        self.model = Qwen3Model(config)
        self.in_batch_negative_loss = SimpleContrastiveLoss(embedding_temperature)
        self.world_size = dist.get_world_size()
        self.process_rank = dist.get_rank()
        self.embedding_negatives_cross_device = config.embedding_negatives_cross_device
        if self.world_size <= 1:
            self.embedding_negatives_cross_device = False

    def forward(
        self,
        query: Optional[Dict[str, paddle.Tensor]] = None,
        passages: Optional[Dict[str, paddle.Tensor]] = None,
        return_encode=False,
    ):
        """forward"""
        q_reps = self.encode(**query)
        p_reps = self.encode(**passages)

        q_reps = nn.functional.normalize(q_reps, axis=-1)
        p_reps = nn.functional.normalize(p_reps, axis=-1)

        if return_encode:
            return q_reps, p_reps

        if self.embedding_negatives_cross_device:
            q_reps = dist_gather_tensor_with_gradient(q_reps)
            p_reps = dist_gather_tensor_with_gradient(p_reps)

        loss = self.in_batch_negative_loss(q_reps, p_reps)
        return loss

    def encode(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        embedding_indices=None,
        return_dict=False,
        **kwargs,
    ):
        """encode"""
        input_type = type(input_ids)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
            **kwargs,
        )
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]
        last_hidden_states = hidden_states.gather_nd(embedding_indices)
        return last_hidden_states


class Qwen3ForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Qwen3Config
    _decoder_layer_cls = Qwen3DecoderLayer
    _get_tensor_parallel_mappings = Qwen3Model._get_tensor_parallel_mappings
    _init_weights = Qwen3Model._init_weights
    _keep_in_fp32_modules = Qwen3Model._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Qwen3Model.transpose_weight_keys


__all__ = [
    "Qwen3Model",
    "Qwen3PretrainedModel",
    "Qwen3ForCausalLM",
    "Qwen3ForCausalLMPipe",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
    "Qwen3SentenceEmbedding",
]
