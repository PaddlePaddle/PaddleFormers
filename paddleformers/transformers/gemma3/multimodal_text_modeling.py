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

"""Gemma3 multimodal text backbone and Torch checkpoint conversion helpers.

The standalone ``gemma3_text`` implementation is maintained upstream.  This
module is intentionally private to the multimodal Gemma3 and ShieldGemma2
wrappers because those models require layer-specific RoPE and multimodal mask
handling that the upstream standalone path does not currently provide.
"""

import inspect
import json
import os
from collections import defaultdict
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from safetensors import safe_open

from ...generation import GenerationMixin
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as BaseMLP
from ...utils.log import logger
from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..configuration_utils import PretrainedConfig
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, dtype_guard
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import Gemma3TextConfig


def _restore_padding_query_rows(
    causal_mask: Optional[paddle.Tensor],
    attention_mask: Optional[paddle.Tensor],
    cache_length: int,
    seq_length: int,
):
    if causal_mask is None or attention_mask is None or attention_mask.ndim != 2:
        return causal_mask

    query_padding = attention_mask[:, cache_length : cache_length + seq_length] == 0
    if not paddle.any(query_padding):
        return causal_mask

    query_padding = query_padding.reshape([query_padding.shape[0], 1, query_padding.shape[1], 1])
    if causal_mask.dtype == paddle.bool:
        replacement = paddle.ones_like(causal_mask)
    else:
        replacement = paddle.zeros_like(causal_mask)
    return paddle.where(query_padding, replacement, causal_mask)


def _compute_causal_lm_loss(
    logits: paddle.Tensor,
    labels: paddle.Tensor,
    vocab_size: int,
    attention_mask: Optional[paddle.Tensor] = None,
    input_ids: Optional[paddle.Tensor] = None,
):
    logits = logits.cast("float32")

    labels_are_pre_shifted = False
    if input_ids is not None and labels.shape == input_ids.shape:
        # Pre-shifted labels should already have the last position masked out.
        # Without this guard, repetitive responses can falsely satisfy the
        # token-match heuristic and be treated as shifted when they are not.
        last_column_is_masked = bool(paddle.all(labels[:, -1] == -100).item())
        shifted_input_ids = paddle.concat(
            [input_ids[:, 1:], paddle.full([input_ids.shape[0], 1], -100, dtype=input_ids.dtype)],
            axis=1,
        )
        comparable_positions = labels != -100
        comparable_count = int(comparable_positions.astype("int64").sum().item())
        if last_column_is_masked and comparable_count > 0:
            matched = ((labels == shifted_input_ids) & comparable_positions).astype("int64").sum().item()
            labels_are_pre_shifted = (matched / comparable_count) > 0.98

    if labels_are_pre_shifted:
        effective_logits = logits
        effective_labels = labels
        if attention_mask is not None and attention_mask.ndim == 2:
            valid_positions = attention_mask != 0
            effective_logits = effective_logits[valid_positions]
            effective_labels = effective_labels[valid_positions]
    else:
        effective_logits = logits[:, :-1, :]
        effective_labels = labels[:, 1:]
        if attention_mask is not None and attention_mask.ndim == 2:
            shift_attention_mask = attention_mask[:, -effective_logits.shape[1] :]
            valid_positions = shift_attention_mask != 0
            effective_logits = effective_logits[valid_positions]
            effective_labels = effective_labels[valid_positions]

    shift_logits = effective_logits.reshape([-1, vocab_size])
    shift_labels = effective_labels.reshape([-1])
    per_token_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
    valid_mask = (shift_labels != -100).astype(per_token_loss.dtype)
    valid_count = paddle.clip(valid_mask.sum(), min=1.0)
    return (per_token_loss * valid_mask).sum() / valid_count


def _load_weight_map(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _iter_hf_tensors(model_dir):
    weight_map = _load_weight_map(model_dir)
    if weight_map is None:
        filename = "model.safetensors"
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No safetensors checkpoint found under {model_dir}")
        # The safetensors package does not always ship its optional Paddle
        # framework registration. NumPy is part of the core API and preserves
        # the checkpoint dtype before conversion to a Paddle tensor.
        with safe_open(path, framework="np") as shard:
            for key in shard.keys():
                yield key, paddle.to_tensor(shard.get_tensor(key))
        return

    file_to_keys = defaultdict(list)
    for key, filename in weight_map.items():
        file_to_keys[filename].append(key)
    for filename in sorted(file_to_keys):
        with safe_open(os.path.join(model_dir, filename), framework="np") as shard:
            for key in sorted(file_to_keys[filename]):
                yield key, paddle.to_tensor(shard.get_tensor(key))


def _fuse_qkv_weights(q_weight, k_weight, v_weight, num_heads, num_key_value_heads):
    q_splits = list(paddle.split(q_weight, num_heads, axis=1))
    k_splits = list(paddle.split(k_weight, num_key_value_heads, axis=1))
    v_splits = list(paddle.split(v_weight, num_key_value_heads, axis=1))
    num_query_heads_per_kv_head = num_heads // num_key_value_heads
    fused_parts = []
    for group_idx in range(num_key_value_heads):
        start = group_idx * num_query_heads_per_kv_head
        end = (group_idx + 1) * num_query_heads_per_kv_head
        fused_parts.extend(q_splits[start:end])
        fused_parts.append(k_splits[group_idx])
        fused_parts.append(v_splits[group_idx])
    return paddle.concat(fused_parts, axis=1)


def load_hf_text_state_dict(
    model_dir: str,
    config: Gemma3TextConfig,
    model_prefix: str = "",
    include_lm_head: bool = False,
    source_prefix: str = "",
):
    state_dict = {}
    qkv_buffers = defaultdict(dict)
    ffn_buffers = defaultdict(dict)
    tied_lm_head = None
    explicit_lm_head = False

    for hf_key, tensor in _iter_hf_tensors(model_dir):
        if source_prefix:
            if not hf_key.startswith(source_prefix):
                continue
            hf_key = hf_key[len(source_prefix) :]
        if hf_key == "model.embed_tokens.weight":
            state_dict[f"{model_prefix}embed_tokens.weight"] = tensor
            if include_lm_head:
                tied_lm_head = tensor.clone()
            continue

        if hf_key == "lm_head.weight":
            if include_lm_head:
                state_dict["lm_head.weight"] = tensor
                explicit_lm_head = True
            continue

        if hf_key == "model.norm.weight":
            state_dict[f"{model_prefix}norm.weight"] = tensor
            continue

        if not hf_key.startswith("model.layers."):
            raise ValueError(f"Unhandled text checkpoint key: {hf_key}")

        layer_suffix = hf_key[len("model.layers.") :]
        layer_id, rest = layer_suffix.split(".", 1)
        prefix = f"{model_prefix}layers.{layer_id}"

        if rest in {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        }:
            state_dict[f"{prefix}.{rest}"] = tensor
            continue

        if rest in {"self_attn.o_proj.weight", "mlp.down_proj.weight"}:
            state_dict[f"{prefix}.{rest}"] = tensor.transpose([1, 0]).contiguous()
            continue

        if rest in {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"}:
            proj_name = rest.split(".")[1][0]
            qkv_buffers[layer_id][proj_name] = tensor.transpose([1, 0]).contiguous()
            if len(qkv_buffers[layer_id]) == 3:
                state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = _fuse_qkv_weights(
                    qkv_buffers[layer_id]["q"],
                    qkv_buffers[layer_id]["k"],
                    qkv_buffers[layer_id]["v"],
                    num_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                )
                del qkv_buffers[layer_id]
            continue

        if rest in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
            proj_name = "gate" if "gate_proj" in rest else "up"
            ffn_buffers[layer_id][proj_name] = tensor.transpose([1, 0]).contiguous()
            if len(ffn_buffers[layer_id]) == 2:
                state_dict[f"{prefix}.mlp.up_gate_proj.weight"] = paddle.concat(
                    [ffn_buffers[layer_id]["gate"], ffn_buffers[layer_id]["up"]],
                    axis=1,
                )
                del ffn_buffers[layer_id]
            continue

        raise ValueError(f"Unhandled text checkpoint key: {hf_key}")

    if qkv_buffers:
        raise ValueError(f"Incomplete text QKV fusion buffers: {sorted(qkv_buffers.keys())}")
    if ffn_buffers:
        raise ValueError(f"Incomplete text FFN fusion buffers: {sorted(ffn_buffers.keys())}")
    if include_lm_head and not explicit_lm_head and tied_lm_head is not None:
        state_dict["lm_head.weight"] = tied_lm_head
    return state_dict


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, config):
        super().__init__(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.register_buffer("embed_scale", paddle.tensor(config.hidden_size**0.5), persistable=False)

    def forward(self, input_ids: paddle.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(BaseMLP):
    def __init__(self, config: Gemma3TextConfig, fuse_up_gate=False):
        super().__init__(config, fuse_up_gate=fuse_up_gate)
        self.act_fn = ACT2FN[config.hidden_activation]


class Gemma3RMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6, input_is_parallel=False):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )
        if input_is_parallel:
            self.enable_sequence_parallel()

    def _norm(self, x):
        with paddle.amp.auto_cast(False):
            return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class Gemma3RotaryEmbedding(nn.Layer):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = list(set(config.layer_types))
        self.rope_type = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters.get(layer_type)
            if rope_params is None:
                continue
            self.rope_type[layer_type] = rope_params.get("rope_type", rope_params.get("type", "default"))
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(self.config, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistable=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistable=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Gemma3TextConfig] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
    ):
        del seq_len
        rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
        base = rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(paddle.float32) / dim))
        return inv_freq, 1.0

    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        if layer_type is None:
            layer_type = self.layer_types[0]
        with paddle.amp.auto_cast(False):
            inv_freq = getattr(self, f"{layer_type}_inv_freq")
            attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
            inv_freq_expanded = inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


class Gemma3Attention(nn.Layer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not config.use_bidirectional_attention
        self.attn_implementation = config._attn_implementation
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads

        if config.tensor_model_parallel_size > 1:
            self.num_heads = self.num_heads // config.tensor_model_parallel_size
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        kv_hidden_size = config.num_key_value_heads * self.head_dim
        q_hidden_size = config.num_attention_heads * self.head_dim

        self.qkv_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size + 2 * kv_hidden_size,
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
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_norm = Gemma3RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps)

        if config.sequence_parallel:
            self.q_norm.enable_sequence_parallel()
            self.k_norm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ):
        del use_cache
        mix_layer = self.qkv_proj(hidden_states)
        if self.config.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
            q_len = max_sequence_length
            target_shape = [bsz, q_len, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
        else:
            target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
        mix_layer = paddle.reshape_(mix_layer, target_shape)
        query_states, key_states, value_states = paddle.split(
            mix_layer,
            num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
            axis=-1,
        )
        query_states = query_states.reshape([0, 0, -1, self.head_dim])
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class Gemma3DecoderLayer(nn.Layer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config, fuse_up_gate=True)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs[0] if len(outputs) == 1 else outputs


class Gemma3TextPreTrainedModel(PretrainedModel):
    config_class = Gemma3TextConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if (
            isinstance(pretrained_model_name_or_path, str)
            and os.path.isdir(pretrained_model_name_or_path)
            and os.path.exists(os.path.join(pretrained_model_name_or_path, "model_state.pdparams"))
        ):
            kwargs.setdefault("load_checkpoint_format", "naive")
            kwargs.setdefault("convert_from_hf", False)
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        if not (
            isinstance(pretrained_model_name_or_path, str)
            and os.path.isdir(pretrained_model_name_or_path)
            and (
                os.path.exists(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
                or os.path.exists(os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json"))
            )
        ):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        dtype = kwargs.pop("dtype", None)
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        accepted_init_kwargs = {
            name for name in inspect.signature(cls.__init__).parameters if name not in {"self", "config"}
        }
        model_kwargs = {key: value for key, value in model_kwargs.items() if key in accepted_init_kwargs}

        if dtype is not None:
            config.dtype = dtype
        with dtype_guard(dtype or paddle.get_default_dtype()):
            model = cls(config, *args, **model_kwargs)
        model_prefix = "" if cls == Gemma3TextModel else "model."
        include_lm_head = cls != Gemma3TextModel
        state_dict = load_hf_text_state_dict(
            pretrained_model_name_or_path,
            config,
            model_prefix=model_prefix,
            include_lm_head=include_lm_head,
        )
        target_state_dict = model.state_dict()
        for name, tensor in list(state_dict.items()):
            if name in target_state_dict and tensor.dtype != target_state_dict[name].dtype:
                state_dict[name] = tensor.astype(target_state_dict[name].dtype)
        missing_keys, unexpected_keys = model.set_state_dict(state_dict)
        if missing_keys or unexpected_keys:
            logger.warning(
                "HF text checkpoint load finished with missing keys %s and unexpected keys %s",
                missing_keys,
                unexpected_keys,
            )
        return model

    @classmethod
    def _gen_aoa_config(cls, config: Gemma3TextConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_config = {
            "aoa_statements": [
                "model.embed_tokens.weight -> lm_head.weight",
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.layers.$LAYER_ID.pre_feedforward_layernorm.weight -> {model_prefix}layers.$LAYER_ID.pre_feedforward_layernorm.weight",
                f"model.layers.$LAYER_ID.post_feedforward_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_feedforward_layernorm.weight",
                f"model.layers.$LAYER_ID.self_attn.q_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.layers.$LAYER_ID.self_attn.k_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                (
                    f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, "
                    f"model.layers.$LAYER_ID.self_attn.k_proj.weight^T, "
                    f"model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> "
                    f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, "
                    f"fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}"
                ),
                (
                    f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, "
                    f"model.layers.$LAYER_ID.mlp.up_proj.weight^T -> "
                    f"{model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn"
                ),
            ]
        }
        return aoa_config


class Gemma3TextModel(Gemma3TextPreTrainedModel):
    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.sequence_parallel = config.sequence_parallel
        self.embed_tokens = Gemma3TextScaledWordEmbedding(config)
        self.layers = nn.LayerList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])
        if config.sequence_parallel:
            self.norm.enable_sequence_parallel()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor],
        attention_mask: paddle.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        use_cache: bool,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        return recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            position_ids,
            output_attentions,
            use_cache,
            attn_mask_startend_row_indices,
        )

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[Union[paddle.Tensor, dict[str, Optional[paddle.Tensor]]]] = None,
        attn_mask_startend_row_indices: Optional[Union[paddle.Tensor, dict[str, Optional[paddle.Tensor]]]] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        del kwargs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        batch_size, seq_length = inputs_embeds.shape[:2]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = paddle.arange(cache_length, cache_length + seq_length, dtype="int64")

        if self.sequence_parallel:
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
            if isinstance(attn_mask_startend_row_indices, dict):
                attn_mask_startend_row_indices_mapping = attn_mask_startend_row_indices
            else:
                attn_mask_startend_row_indices_mapping = {
                    "full_attention": attn_mask_startend_row_indices,
                    "sliding_attention": attn_mask_startend_row_indices,
                }
        else:
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
            full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)
            full_mask = _restore_padding_query_rows(full_mask, attention_mask, cache_length, seq_length)
            causal_mask_mapping = {"full_attention": full_mask}
            attn_mask_startend_row_indices_mapping = {"full_attention": full_indices}
            if self.has_sliding_layers:
                sliding_mask, sliding_indices = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)
                sliding_mask = _restore_padding_query_rows(sliding_mask, attention_mask, cache_length, seq_length)
                causal_mask_mapping["sliding_attention"] = sliding_mask
                attn_mask_startend_row_indices_mapping["sliding_attention"] = sliding_indices

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        unique_layer_types = set(self.config.layer_types)
        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type) for layer_type in unique_layer_types
        }

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            has_gradient = not hidden_states.stop_gradient
            layer_attention_mask = causal_mask_mapping[decoder_layer.attention_type]
            layer_row_indices = attn_mask_startend_row_indices_mapping[decoder_layer.attention_type]
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    position_embeddings=position_embeddings[decoder_layer.attention_type],
                    attention_mask=layer_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=layer_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings[decoder_layer.attention_type],
                    attention_mask=layer_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=layer_row_indices,
                )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3ForCausalLM(Gemma3TextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[Union[paddle.Tensor, dict[str, Optional[paddle.Tensor]]]] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[Union[paddle.Tensor, dict[str, Optional[paddle.Tensor]]]] = None,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = paddle.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = _compute_causal_lm_loss(logits, labels, self.config.vocab_size, attention_mask, input_ids)

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


__all__ = [
    "Gemma3TextPreTrainedModel",
    "Gemma3TextScaledWordEmbedding",
    "Gemma3MLP",
    "Gemma3RMSNorm",
    "Gemma3RotaryEmbedding",
    "Gemma3Attention",
    "Gemma3DecoderLayer",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
    "load_hf_text_state_dict",
]
