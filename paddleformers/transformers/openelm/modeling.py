# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Adapted for PaddlePaddle / paddleformers from the original Apple OpenELM implementation.
# Original code: based on transformers/PyTorch. Migrated to PaddlePaddle.

from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from .configuration import OpenELMConfig, make_divisible


class OpenELMRMSNorm(nn.Layer):
    """RMS Normalization layer."""

    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.

        Args:
            num_features (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (paddle.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[num_features],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        return x * paddle.rsqrt(paddle.mean(paddle.pow(x, 2), axis=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight

    def extra_repr(self) -> str:
        return super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"


class OpenELMPreTrainedModel(PretrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OpenELMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    transpose_weight_keys = ["qkv_proj", "out_proj", "proj_1", "proj_2"]

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Layer) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=module.weight.shape,
                ).cast(module.weight.dtype)
            )
            if module.bias is not None:
                module.bias.set_value(paddle.zeros(module.bias.shape, dtype=module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            module.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=module.weight.shape,
                ).cast(module.weight.dtype)
            )
        elif isinstance(module, OpenELMRMSNorm):
            module.weight.set_value(paddle.ones(module.weight.shape, dtype=module.weight.dtype))


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = paddle.chunk(x, chunks=2, axis=-1)
    return paddle.concat([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    return (x * pos_cos) + (_rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(nn.Layer):
    """The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.

    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.

    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(self, model_dim: int, max_seq_length: int, freq_constant: int = 10000) -> None:
        inv_freq = 1.0 / (freq_constant ** (paddle.arange(0, model_dim, 2, dtype="float32") / model_dim))
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return (
            f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"
        )

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_place=None,
        key_dtype: paddle.dtype = paddle.float32,
    ) -> None:
        """Compute sine and cos embeddings.

        Recalculate if any of:
            1. key_len > cached seq length
            2. caches are empty
            3. device/dtype mismatch
        """
        need_recompute = key_len > self._cached_seq_length or self._cached_cos is None or self._cached_sin is None
        if not need_recompute and key_place is not None:
            need_recompute = self._cached_cos.place != key_place or self._cached_sin.place != key_place
        if not need_recompute and self._cached_cos is not None:
            need_recompute = self._cached_cos.dtype != key_dtype or self._cached_sin.dtype != key_dtype

        if need_recompute:
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            pos_index = paddle.arange(
                self._cached_seq_length,
                dtype="float32",
            )
            pos_index_theta = paddle.einsum("i,j->ij", pos_index, self.inv_freq)
            emb = paddle.concat((pos_index_theta, pos_index_theta), axis=-1)

            cos_emb = emb.cos().cast(key_dtype)
            sin_emb = emb.sin().cast(key_dtype)

            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Apply RoPE to query and key embeddings.

        Args:
            query: Query embeddings.
            key: Key embeddings.

        Returns:
            Tuple of query and key embeddings with positional information.
        """
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.place == query.place
        assert key.dtype == query.dtype

        assert key_len >= query_len, "Number of keys has to be greater than or equal to number of queries."

        query_float = query.astype(paddle.float32)
        key_float = key.astype(paddle.float32)

        self._compute_sin_cos_embeddings(key_len, key_place=key_float.place, key_dtype=key_float.dtype)
        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        return query_float.astype(query.dtype), key_float.astype(key.dtype)


class OpenELMMultiHeadCausalAttention(nn.Layer):
    """Multi-head causal attention with optional Group Query Attention (GQA)."""

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=(q_heads + k_heads + v_heads) * head_dim,
            bias_attr=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            in_features=q_heads * head_dim,
            out_features=config.model_dim,
            bias_attr=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Forward pass of multi-head self-attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, model_dim].
            past_key_value: Cached keys and values.
            output_attentions: Whether to output attention weights.
            use_cache: Whether to use kv-cache for generation.
            cache_position: Cache position indices.

        Returns:
            Output tensor, optionally with cached keys and values.
        """

        # scaled_dot_product_attention does not return attention weights, set output_attentions to False
        output_attentions = False
        batch_size, seq_length, d_model = hidden_states.shape

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            [
                batch_size,
                seq_length,
                self.num_q_heads + self.num_k_heads + self.num_v_heads,
                self.head_dim,
            ]
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose([0, 2, 1, 3])
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S, h], [B, k_h, S, h], [B, v_h, S, h]
        queries, keys, values = paddle.split(
            qkv,
            [self.num_q_heads, self.num_k_heads, self.num_v_heads],
            axis=1,
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(keys, values, self.layer_idx, cache_kwargs)

        # Add positional embedding
        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # GQA: [B, k_h, S, h] --> [B, q_h, S, h]
            keys = paddle.repeat_interleave(keys, self.num_groups, axis=1)
            values = paddle.repeat_interleave(values, self.num_groups, axis=1)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        scale = self.head_dim**-0.5
        # [B, q_h, S_q, h] x [B, q_h, h, S_k] -> [B, q_h, S_q, S_k]
        attn_weights = paddle.matmul(queries * scale, keys, transpose_y=True)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, axis=-1)
        attn_output = paddle.matmul(attn_weights, values)

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, seq_length, self.num_q_heads * self.head_dim])
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class OpenELMFeedForwardNetwork(nn.Layer):
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
        if config.ffn_with_glu:
            # FFN with Gated linear unit (https://arxiv.org/abs/2002.05202v1)
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=2 * intermediate_dim,
                bias_attr=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias_attr=False,
            )
            self.ffn_with_glu = True
        else:
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=intermediate_dim,
                bias_attr=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias_attr=False,
            )
            self.ffn_with_glu = False

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
            y_12 = self.proj_1(x)
            y_1, y_2 = paddle.chunk(y_12, chunks=2, axis=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class OpenELMDecoderLayer(nn.Layer):
    """Transformer decoder layer."""

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = OpenELMMultiHeadCausalAttention(config=config, layer_idx=layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config=config, layer_idx=layer_idx)
        self.ffn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )
        self.attn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass of decoder layer.

        Args:
            hidden_states: Input [batch, seq_len, embed_dim].
            attention_mask: Attention mask.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return cached key-value states.
            past_key_value: Cached past key and value projection states.
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@register_base_model
class OpenELMModel(OpenELMPreTrainedModel):
    """OpenELM Model (base model without language modeling head)."""

    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.config = config

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.model_dim,
        )

        self.layers = nn.LayerList(
            [
                OpenELMDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_transformer_layers)
            ]
        )
        self.norm = OpenELMRMSNorm(num_features=config.model_dim)
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Linear(
                in_features=config.model_dim,
                out_features=config.vocab_size,
                bias_attr=False,
            )
        self.num_transformer_layers = config.num_transformer_layers
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation.
        causal_mask = paddle.full(
            [config.max_context_length, config.max_context_length],
            fill_value=True,
            dtype="bool",
        )
        self.register_buffer("causal_mask", paddle.triu(causal_mask, diagonal=1), persistable=False)

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings: paddle.Tensor):
        self.token_embeddings = new_embeddings

    def init_weights(self):
        """Initialize weights for the model."""
        super().init_weights()
        self.reset_parameters(self.config)

    def reset_parameters(self, config: OpenELMConfig) -> None:
        """Initialize layers following OPT (https://arxiv.org/pdf/2205.01068.pdf)."""
        for module in self.sublayers():
            if isinstance(module, nn.Linear):
                std = module.in_features**-0.5
                module.weight.set_value(paddle.normal(mean=0.0, std=std, shape=module.weight.shape))
                if module.bias is not None:
                    module.bias.set_value(paddle.zeros(module.bias.shape))
            elif isinstance(module, nn.Embedding):
                std = module._embedding_dim**-0.5
                module.weight.set_value(paddle.normal(mean=0.0, std=std, shape=module.weight.shape))
            elif isinstance(module, OpenELMRMSNorm):
                if module.weight is not None:
                    module.weight.set_value(paddle.ones(module.weight.shape))

        model_dim = config.model_dim
        n_layers = config.num_transformer_layers
        std = (model_dim**-0.5) * ((2 * n_layers) ** -0.5)
        for param_name, param in self.named_parameters():
            if param_name.endswith("out_proj.weight") or param_name.endswith("ffn.proj_2.weight"):
                param.set_value(paddle.normal(mean=0.0, std=std, shape=param.shape))

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Forward pass of OpenELMModel.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            past_key_values: Past key-value cache.
            inputs_embeds: Input embeddings.
            use_cache: Whether to use KV cache.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dict.
            cache_position: Cache position indices.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = paddle.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                from paddle.distributed.fleet.utils import recompute

                layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        """Update the causal mask for the attention."""
        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = paddle.full(
                [2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]],
                fill_value=1,
            )
            self.register_buffer("causal_mask", paddle.triu(causal_mask, diagonal=1), persistable=False)

        min_dtype = paddle.finfo(dtype).min
        causal_mask = self.causal_mask[None, None, :, :].tile([batch_size, 1, 1, 1]).cast(dtype) * min_dtype

        if attention_mask is not None and len(attention_mask.shape) == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = (causal_mask[..., :mask_length] == 0.0) * (attention_mask[:, None, None, :] == 0.0)
            causal_mask[..., :mask_length] = paddle.where(
                padding_mask,
                paddle.full_like(causal_mask[..., :mask_length], min_dtype),
                causal_mask[..., :mask_length],
            )

        return causal_mask


class OpenELMForCausalLM(OpenELMPreTrainedModel):
    """OpenELM model with a language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size
        if config.share_input_output_layers:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
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
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
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
        )

        hidden_states = outputs[0]
        if self.lm_head is None:
            # shared embedding weights: Paddle matmul needs transpose_y=True for tied weights
            logits = paddle.matmul(hidden_states, self.transformer.token_embeddings.weight, transpose_y=True)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits[:, : self.config.vocab_size]
        loss = None
        if labels is not None:
            # Labels are pre-shifted by SFTDataset, do not shift again.
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits.reshape([-1, self.config.vocab_size])
            shift_labels = labels.reshape([-1])
            loss = loss_fct(shift_logits, shift_labels)

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
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = cache_length
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.cast(paddle.int64).cumsum(axis=-1) - 1
            position_ids = paddle.where(
                attention_mask == 0,
                paddle.ones_like(position_ids),
                position_ids,
            )
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        cache_position = paddle.arange(
            past_length,
            past_length + position_ids.shape[-1],
        )

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

    @classmethod
    def _gen_aoa_config(cls, config: OpenELMConfig):
        """Generate AOA (Auto-Optimized Architecture) config for flex_checkpoint loading.

        Maps HuggingFace safetensors weight names to PaddlePaddle weight names.
        Since OpenELM uses the same naming convention in both frameworks
        (base_model_prefix='transformer'), the source and target keys are identical.
        The ^T suffix indicates weights that need transposition (Linear layer weights).

        Args:
            config: OpenELMConfig instance.

        Returns:
            Dict with 'aoa_statements' list of mapping strings.
        """
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            # Embedding and final norm
            f"transformer.token_embeddings.weight -> {model_prefix}token_embeddings.weight",
            f"transformer.norm.weight -> {model_prefix}norm.weight",
            # Layer norms
            f"transformer.layers.$LAYER_ID.attn_norm.weight -> {model_prefix}layers.$LAYER_ID.attn_norm.weight",
            f"transformer.layers.$LAYER_ID.ffn_norm.weight -> {model_prefix}layers.$LAYER_ID.ffn_norm.weight",
        ]

        # Attention projections (need transposition for Linear weights)
        aoa_statements.extend(
            [
                f"transformer.layers.$LAYER_ID.attn.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.attn.{proj_name}.weight"
                for proj_name in ["qkv_proj", "out_proj"]
            ]
        )

        # Optional Q/K normalization (normalize_qk_projections=True for OpenELM-1_1B)
        if config.normalize_qk_projections:
            aoa_statements.extend(
                [
                    f"transformer.layers.$LAYER_ID.attn.{norm_name}.weight -> {model_prefix}layers.$LAYER_ID.attn.{norm_name}.weight"
                    for norm_name in ["q_norm", "k_norm"]
                ]
            )

        # FFN projections (need transposition for Linear weights)
        aoa_statements.extend(
            [
                f"transformer.layers.$LAYER_ID.ffn.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.ffn.{proj_name}.weight"
                for proj_name in ["proj_1", "proj_2"]
            ]
        )

        # LM head or tied embeddings
        if cls != cls.base_model_class:
            if config.share_input_output_layers:
                aoa_statements.append("transformer.token_embeddings.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: OpenELMConfig):
        """Generate inverse AOA config for saving PaddlePaddle weights to HuggingFace format.

        This is the reverse mapping of _gen_aoa_config.

        Args:
            config: OpenELMConfig instance.

        Returns:
            Dict with 'aoa_statements' list of mapping strings.
        """
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            # Embedding and final norm
            f"{model_prefix}token_embeddings.weight -> transformer.token_embeddings.weight",
            f"{model_prefix}norm.weight -> transformer.norm.weight",
            # Layer norms
            f"{model_prefix}layers.$LAYER_ID.attn_norm.weight -> transformer.layers.$LAYER_ID.attn_norm.weight",
            f"{model_prefix}layers.$LAYER_ID.ffn_norm.weight -> transformer.layers.$LAYER_ID.ffn_norm.weight",
        ]

        # Attention projections (need transposition)
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.attn.{proj_name}.weight^T -> transformer.layers.$LAYER_ID.attn.{proj_name}.weight"
                for proj_name in ["qkv_proj", "out_proj"]
            ]
        )

        # Optional Q/K normalization
        if config.normalize_qk_projections:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.attn.{norm_name}.weight -> transformer.layers.$LAYER_ID.attn.{norm_name}.weight"
                    for norm_name in ["q_norm", "k_norm"]
                ]
            )

        # FFN projections (need transposition)
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.ffn.{proj_name}.weight^T -> transformer.layers.$LAYER_ID.ffn.{proj_name}.weight"
                for proj_name in ["proj_1", "proj_2"]
            ]
        )

        # LM head or tied embeddings
        if not config.share_input_output_layers and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search."""
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.place)) for past_state in layer_past),
            )
        return reordered_past
