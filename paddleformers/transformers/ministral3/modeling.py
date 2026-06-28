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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..model_outputs import BaseModelOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    dynamic_rope_update,
    standardize_rope_params,
)
from .configuration import Ministral3TextConfig, Mistral3Config

__all__ = [
    "Mistral3PreTrainedModel",
    "Mistral3Model",
    "Mistral3ForConditionalGeneration",
    "Mistral3RMSNorm",
    "Mistral3PatchMerger",
    "Mistral3MultiModalProjector",
    "Ministral3TextDecoder",
    "Ministral3DecoderLayer",
    "Ministral3Attention",
    "Ministral3MLP",
]


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv, slen, head_dim = hidden_states.shape
    return (
        hidden_states.unsqueeze(2)
        .expand([batch, num_kv, n_rep, slen, head_dim])
        .reshape([batch, num_kv * n_rep, slen, head_dim])
    )


def _get_llama4_attn_scale(cache_position: Tensor, beta: float, max_position_embeddings: int) -> Tensor:
    pos = cache_position.astype("float32")
    return (1.0 + beta * paddle.log(1.0 + paddle.floor(pos / max_position_embeddings))).reshape([1, 1, -1, 1])


class _RMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        hidden_states = hidden_states * paddle.rsqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * hidden_states.astype(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


Mistral3RMSNorm = _RMSNorm
Ministral3RMSNorm = _RMSNorm


class Ministral3RotaryEmbedding(nn.Layer):
    def __init__(self, text_cfg: "Ministral3TextConfig"):
        super().__init__()
        self.config = text_cfg
        self.max_seq_len_cached = text_cfg.max_position_embeddings
        self.original_max_seq_len = text_cfg.max_position_embeddings
        standardize_rope_params(text_cfg)

        rope_type = text_cfg.rope_parameters.get("rope_type", "default")
        self.rope_type = rope_type
        rope_init_fn = self._compute_default if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(text_cfg)
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def _compute_default(config, **kwargs):
        base = config.rope_parameters.get("rope_theta", 1000000.0)
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype("float32") / dim))
        return inv_freq, 1.0

    @dynamic_rope_update
    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])
            freqs = (inv_freq_expanded.float() @ position_ids[:, None, :].float()).transpose([0, 2, 1])
            emb = paddle.concat([freqs, freqs], axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.cast(x.dtype), sin.cast(x.dtype)


class Ministral3Attention(nn.Layer):
    def __init__(self, text_cfg: "Ministral3TextConfig", layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = text_cfg.head_dim
        self.num_heads = text_cfg.num_attention_heads
        self.num_kv_heads = text_cfg.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = text_cfg.attention_dropout

        rope_params = text_cfg.rope_parameters
        self.llama4_beta = float(rope_params.get("llama_4_scaling_beta", 0.1))
        self.original_max_pos = int(rope_params.get("original_max_position_embeddings", 16384))

        hidden = text_cfg.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias_attr=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias_attr=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias_attr=False)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor],
        cache_position: Tensor,
        past_key_values: Optional[Cache] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch, seq, _ = hidden_states.shape

        query = self.q_proj(hidden_states).reshape([batch, seq, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key = (
            self.k_proj(hidden_states).reshape([batch, seq, self.num_kv_heads, self.head_dim]).transpose([0, 2, 1, 3])
        )
        value = (
            self.v_proj(hidden_states).reshape([batch, seq, self.num_kv_heads, self.head_dim]).transpose([0, 2, 1, 3])
        )

        query, key = apply_rotary_pos_emb(query, key, cos.unsqueeze(1), sin.unsqueeze(1))
        query = query * _get_llama4_attn_scale(cache_position, self.llama4_beta, self.original_max_pos).cast(
            query.dtype
        )

        if past_key_values is not None:
            key, value = past_key_values.update(
                key,
                value,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)

        attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * self.scaling
        if attn_mask_startend_row_indices is not None:
            attn_weights = attn_weights + attn_mask_startend_row_indices.cast(query.dtype)
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").cast(query.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = paddle.matmul(attn_weights, value)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch, seq, self.num_heads * self.head_dim])
        return self.o_proj(attn_output), attn_weights


class Ministral3MLP(nn.Layer):
    def __init__(self, text_cfg: "Ministral3TextConfig"):
        super().__init__()
        self.gate_proj = nn.Linear(text_cfg.hidden_size, text_cfg.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(text_cfg.hidden_size, text_cfg.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(text_cfg.intermediate_size, text_cfg.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[text_cfg.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Ministral3DecoderLayer(nn.Layer):
    def __init__(self, text_cfg: "Ministral3TextConfig", layer_idx: int):
        super().__init__()
        self.self_attn = Ministral3Attention(text_cfg, layer_idx=layer_idx)
        self.mlp = Ministral3MLP(text_cfg)
        self.input_layernorm = Ministral3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
        self.post_attention_layernorm = Ministral3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor],
        cache_position: Tensor,
        past_key_values: Optional[Cache] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class Ministral3TextDecoder(nn.Layer):
    def __init__(self, text_cfg: "Ministral3TextConfig"):
        super().__init__()
        self.text_cfg = text_cfg
        self.embed_tokens = nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size)
        self.layers = nn.LayerList(
            [Ministral3DecoderLayer(text_cfg, layer_idx=i) for i in range(text_cfg.num_hidden_layers)]
        )
        self.norm = Ministral3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
        self.rotary_emb = Ministral3RotaryEmbedding(text_cfg)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        seq_len = inputs_embeds.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = paddle.arange(past_len, past_len + seq_len, dtype="int64")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        dtype = inputs_embeds.dtype
        total_len = seq_len + past_len
        causal_mask = paddle.full([seq_len, total_len], paddle.finfo(dtype).min, dtype=dtype)
        is_future = paddle.arange(total_len, dtype="int64").unsqueeze(0) > cache_position.unsqueeze(1)
        causal_mask = paddle.where(is_future, causal_mask, paddle.zeros_like(causal_mask)).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).cast(dtype)) * paddle.finfo(dtype).min
            causal_mask = causal_mask + pad_mask

        cos, sin = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=causal_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values if use_cache else None,
        )


class Mistral3PatchMerger(nn.Layer):
    def __init__(self, config: "Mistral3Config"):
        super().__init__()
        vision_cfg = config.vision_config
        hidden_size = vision_cfg.get("hidden_size", 1024) if isinstance(vision_cfg, dict) else vision_cfg.hidden_size
        patch_size = vision_cfg.get("patch_size", 14) if isinstance(vision_cfg, dict) else vision_cfg.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias_attr=False)

    def forward(self, image_features: Tensor, image_sizes: Tensor) -> Tensor:
        p = self.patch_size
        image_sizes_list = [(int(sz[0]) // p, int(sz[1]) // p) for sz in image_sizes.tolist()]
        tokens_per_image = [h * w for h, w in image_sizes_list]
        d = image_features.shape[-1]
        permuted = []
        for idx, img_tokens in enumerate(paddle.split(image_features, tokens_per_image, axis=0)):
            h, w = image_sizes_list[idx]
            grid = img_tokens.reshape([h, w, d]).transpose([2, 0, 1]).unsqueeze(0)
            grid = F.unfold(grid, kernel_sizes=self.spatial_merge_size, strides=self.spatial_merge_size, paddings=0)
            permuted.append(grid.reshape([d * self.spatial_merge_size**2, -1]).transpose([1, 0]))
        return self.merging_layer(paddle.concat(permuted, axis=0))


class Mistral3MultiModalProjector(nn.Layer):
    def __init__(self, config: "Mistral3Config"):
        super().__init__()
        vision_cfg = config.vision_config
        text_cfg = config.text_config
        vision_hidden = vision_cfg.get("hidden_size", 1024) if isinstance(vision_cfg, dict) else vision_cfg.hidden_size
        text_hidden = text_cfg.get("hidden_size", 4096) if isinstance(text_cfg, dict) else text_cfg.hidden_size
        rms_eps = text_cfg.get("rms_norm_eps", 1e-5) if isinstance(text_cfg, dict) else text_cfg.rms_norm_eps
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.norm = Mistral3RMSNorm(vision_hidden, eps=rms_eps)
        self.patch_merger = Mistral3PatchMerger(config)
        self.linear_1 = nn.Linear(
            vision_hidden * num_feature_layers, text_hidden, bias_attr=config.multimodal_projector_bias
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias_attr=config.multimodal_projector_bias)

    def forward(self, image_features: Tensor, image_sizes: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(self.patch_merger(self.norm(image_features), image_sizes))))


@dataclass
class Mistral3CausalLMOutputWithPast(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    past_key_values: Optional[Union[Tuple, Cache]] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None
    image_hidden_states: Optional[Tensor] = None


@dataclass
class Mistral3ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[Tensor] = None


class Mistral3PreTrainedModel(PretrainedModel):
    config_class = Mistral3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Mistral3Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
        ]

        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )

        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.mlp.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Mistral3Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
        ]

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> model.layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight^T -> model.layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )

        return {"aoa_statements": aoa_statements}

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.randn(shape=layer.weight.shape) * self.config.initializer_range)
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(paddle.randn(shape=layer.weight.shape) * self.config.initializer_range)
        elif isinstance(layer, _RMSNorm):
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class Mistral3Model(Mistral3PreTrainedModel):
    def __init__(self, config: Mistral3Config):
        super().__init__(config)
        text_cfg_raw = config.text_config
        if isinstance(text_cfg_raw, dict):
            self._text_cfg = Ministral3TextConfig.from_dict(text_cfg_raw)
        elif isinstance(text_cfg_raw, Ministral3TextConfig):
            self._text_cfg = text_cfg_raw
        else:
            # Mistral3TextConfig (PretrainedConfig) or other objects -> convert to dict then wrap
            from .configuration import Mistral3TextConfig as _PretrainedTextCfg

            if isinstance(text_cfg_raw, _PretrainedTextCfg):
                self._text_cfg = Ministral3TextConfig(
                    {
                        "attention_dropout": text_cfg_raw.attention_dropout,
                        "head_dim": text_cfg_raw.head_dim,
                        "hidden_act": text_cfg_raw.hidden_act,
                        "hidden_size": text_cfg_raw.hidden_size,
                        "initializer_range": text_cfg_raw.initializer_range,
                        "intermediate_size": text_cfg_raw.intermediate_size,
                        "max_position_embeddings": text_cfg_raw.max_position_embeddings,
                        "num_attention_heads": text_cfg_raw.num_attention_heads,
                        "num_hidden_layers": text_cfg_raw.num_hidden_layers,
                        "num_key_value_heads": text_cfg_raw.num_key_value_heads,
                        "rms_norm_eps": text_cfg_raw.rms_norm_eps,
                        "rope_parameters": getattr(
                            text_cfg_raw,
                            "rope_parameters",
                            {
                                "rope_type": "default",
                                "rope_theta": getattr(text_cfg_raw, "rope_theta", 1000000.0),
                            },
                        ),
                        "sliding_window": text_cfg_raw.sliding_window,
                        "use_cache": text_cfg_raw.use_cache,
                        "vocab_size": text_cfg_raw.vocab_size,
                    }
                )
            else:
                self._text_cfg = text_cfg_raw
        self.language_model = Ministral3TextDecoder(self._text_cfg)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.vision_tower = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Tensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        image_sizes: Optional[Tensor] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ) -> Union[Tuple, Mistral3ModelOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None and self.vision_tower is not None:
            raise NotImplementedError("Vision branch requires a full vision tower implementation")

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        return Mistral3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            image_hidden_states=None,
        )


class Mistral3ForConditionalGeneration(Mistral3PreTrainedModel):
    _checkpoint_conversion_mapping = {
        r"^language_model\.model\.": "model.language_model.",
        r"^language_model\.lm_head\.": "lm_head.",
        r"^vision_tower\.": "model.vision_tower.",
        r"^multi_modal_projector\.": "model.multi_modal_projector.",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _use_converted_weights = None

    @classmethod
    def _detect_converted_weights(cls, model_path):
        import os

        if isinstance(model_path, str) and model_path.startswith("~"):
            model_path = os.path.expanduser(model_path)

        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            if any(f.endswith(".pdparams") for f in files):
                return True
            marker_file = os.path.join(model_path, ".paddleformers_converted")
            if os.path.exists(marker_file):
                return True
            config_file = os.path.join(model_path, "config.json")
            if os.path.exists(config_file):
                try:
                    import json

                    with open(config_file, "r") as f:
                        config = json.load(f)
                        if config.get("_paddleformers_converted", False):
                            return True
                except Exception:
                    pass
            if any(f.endswith(".safetensors") for f in files):
                return False

        return False

    @classmethod
    def _gen_aoa_config(cls, config: Mistral3Config):
        aoa_statements = [
            "language_model.model.embed_tokens.weight -> model.language_model.embed_tokens.weight",
            "language_model.model.norm.weight -> model.language_model.norm.weight",
            "language_model.model.layers.$LAYER_ID.input_layernorm.weight -> model.language_model.layers.$LAYER_ID.input_layernorm.weight",
            "language_model.model.layers.$LAYER_ID.post_attention_layernorm.weight -> model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        aoa_statements.extend(
            [
                f"language_model.model.layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> model.language_model.layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )
        aoa_statements.extend(
            [
                f"language_model.model.layers.$LAYER_ID.mlp.{proj_name}.weight^T -> model.language_model.layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        return {"aoa_statements": aoa_statements}

    _HF_NAME_MAPPING = [
        (r"^language_model\.model\.", "model.language_model."),
        (r"^language_model\.lm_head\.", "lm_head."),
        (r"^vision_tower\.", "model.vision_tower."),
        (r"^multi_modal_projector\.", "model.multi_modal_projector."),
    ]

    _HF_TRANSPOSE_SUFFIXES = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "attention.q_proj.weight",
        "attention.k_proj.weight",
        "attention.v_proj.weight",
        "attention.o_proj.weight",
        "feed_forward.gate_proj.weight",
        "feed_forward.up_proj.weight",
        "feed_forward.down_proj.weight",
        "merging_layer.weight",
        "linear_1.weight",
        "linear_2.weight",
    )

    _HF_QUANT_SUFFIXES = (
        "activation_scale",
        "weight_scale_inv",
        "qscale_act",
        "qscale_weight",
    )

    @classmethod
    def _load_hf_safetensors_to_paddle(cls, model_path):
        """FP8 dequantization + HF name mapping + Linear weight transpose -> Paddle state_dict"""
        import glob
        import os
        import re

        import ml_dtypes
        import numpy as np
        import paddle

        from ...utils.safetensors import fast_safe_open

        model_path = os.path.expanduser(model_path)

        shard_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        if not shard_files:
            single_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(single_file):
                shard_files = [single_file]
            else:
                raise FileNotFoundError(f"Safetensors weight file not found: {model_path}")

        scale_inv_map = {}
        for sf in shard_files:
            with fast_safe_open(sf) as f:
                for key in f.keys():
                    if key.endswith("weight_scale_inv"):
                        weight_key = key[: -len("_scale_inv")]
                        scale_inv_map[weight_key] = f.get_tensor(key)

        state_dict = {}
        for sf in shard_files:
            with fast_safe_open(sf) as f:
                for hf_name in f.keys():
                    if any(hf_name.endswith(suf) for suf in cls._HF_QUANT_SUFFIXES):
                        continue

                    np_arr = f.get_tensor(hf_name)
                    if np_arr.dtype == ml_dtypes.float8_e4m3fn:
                        scale = scale_inv_map.get(hf_name)
                        if scale is not None:
                            np_arr = np_arr.astype(np.float32) * scale.astype(np.float32)
                        else:
                            np_arr = np_arr.astype(np.float32)
                    else:
                        np_arr = np_arr.astype(np.float32)

                    paddle_name = hf_name
                    for pattern, replacement in cls._HF_NAME_MAPPING:
                        paddle_name = re.sub(pattern, replacement, paddle_name)

                    if (
                        np_arr.ndim == 2
                        and any(hf_name.endswith(suf) for suf in cls._HF_TRANSPOSE_SUFFIXES)
                        and paddle_name != "lm_head.weight"
                    ):
                        np_arr = np.ascontiguousarray(np_arr.T)

                    state_dict[paddle_name] = paddle.to_tensor(np_arr, dtype=paddle.bfloat16)

        return state_dict

    @classmethod
    def _resolve_local_cache_path(cls, model_path, download_hub=None):
        """Resolve Hub model_id to local cache path. Returns (path, source)."""
        import os

        if os.path.isdir(model_path):
            return model_path, None

        hub_str = str(download_hub) if download_hub is not None else ""

        if hub_str == "aistudio":
            try:
                from aistudio_sdk.file_download import get_model_cache_root

                cache_candidate = os.path.join(get_model_cache_root(), model_path)
                if os.path.isdir(cache_candidate):
                    return cache_candidate, "aistudio"
            except ImportError:
                pass

        if hub_str == "modelscope":
            cache_root = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models")
            cache_candidate = os.path.join(cache_root, model_path)
            if os.path.isdir(cache_candidate):
                return cache_candidate, "modelscope"

        if hub_str in ("huggingface", ""):
            try:
                from huggingface_hub import scan_cache_dir

                cache = scan_cache_dir()
                for repo in cache.repos:
                    if repo.repo_id == model_path:
                        for rev in repo.revisions:
                            if os.path.isdir(rev.snapshot_path):
                                return rev.snapshot_path, "huggingface"
            except Exception:
                pass

        return model_path, None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load pretrained model, auto-detect three formats:
        1. Converted .pdparams (AIStudio): convert_from_hf=False, direct load
        2. Original HF safetensors (with FP8): in-memory dequant + mapping + transpose
        3. Converted safetensors (no FP8): flex_checkpoint + AOA load
        """
        import os

        use_converted_weights = kwargs.pop("use_converted_weights", None)
        load_checkpoint_format = kwargs.get("load_checkpoint_format", None)
        use_safetensors = kwargs.get("use_safetensors", None)

        model_path = pretrained_model_name_or_path
        if isinstance(model_path, str) and model_path.startswith("~"):
            model_path = os.path.expanduser(model_path)

        check_path, cache_source = cls._resolve_local_cache_path(model_path, kwargs.get("download_hub", None))
        if cache_source is not None and kwargs.get("download_hub") is None:
            kwargs["download_hub"] = cache_source

        has_pdparams = False
        has_safetensors = False
        if os.path.isdir(check_path):
            files = os.listdir(check_path)
            has_pdparams = any(f.endswith(".pdparams") for f in files)
            has_safetensors = any(f.endswith(".safetensors") for f in files)

        if has_pdparams and not has_safetensors:
            if load_checkpoint_format is None:
                kwargs["load_checkpoint_format"] = "legacy"
            if use_safetensors is None:
                kwargs["use_safetensors"] = False
            kwargs["convert_from_hf"] = False
            kwargs.setdefault("ignore_mismatched_sizes", True)
            cls._use_converted_weights = True

        elif has_safetensors:
            resolved_converted = use_converted_weights
            if resolved_converted is None:
                resolved_converted = cls._detect_converted_weights(check_path)
            cls._use_converted_weights = resolved_converted

            if not resolved_converted:
                state_dict = cls._load_hf_safetensors_to_paddle(check_path)
                kwargs["state_dict"] = state_dict
                kwargs["convert_from_hf"] = False
                kwargs["load_checkpoint_format"] = "legacy"
            else:
                if load_checkpoint_format is None:
                    kwargs["load_checkpoint_format"] = "flex_checkpoint"

        model = super(Mistral3ForConditionalGeneration, cls).from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        # Old AIStudio weights have lm_head.weight shape [hidden, vocab], skipped by
        # ignore_mismatched_sizes, need tie_weights() to rebind to embed_tokens.weight
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        return model

    def __init__(self, config: Mistral3Config):
        super().__init__(config)
        self.model = Mistral3Model(config)
        text_cfg_raw = config.text_config
        vocab_size = (
            text_cfg_raw.get("vocab_size", 131072) if isinstance(text_cfg_raw, dict) else text_cfg_raw.vocab_size
        )
        hidden_size = (
            text_cfg_raw.get("hidden_size", 4096) if isinstance(text_cfg_raw, dict) else text_cfg_raw.hidden_size
        )

        self.lm_head = nn.Embedding(vocab_size, hidden_size)
        self.lm_head.weight.is_persistable = True
        self.tie_weights()

    def _init_weights(self, layer):
        if layer is self.lm_head:
            return
        super()._init_weights(layer)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Tensor] = None,
        image_sizes: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        logits_to_keep: Union[int, Tensor] = 0,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ) -> Union[Tuple, Mistral3CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_sizes=image_sizes,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        hidden_states = outputs.last_hidden_state[:, slice_indices, :]
        logits = paddle.matmul(hidden_states, self.lm_head.weight, transpose_y=True)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape([-1, logits.shape[-1]]),
                labels.reshape([-1]),
                ignore_index=-100,
            )

        if return_dict is False:
            output = (logits,) + (outputs.past_key_values,)
            return (loss,) + output if loss is not None else output

        return Mistral3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if past_key_values is not None:
            model_inputs["pixel_values"] = None
        elif pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs
