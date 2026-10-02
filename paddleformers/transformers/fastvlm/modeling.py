# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
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

"""FastVLM model implementation."""

import inspect
import json
import os
from collections import defaultdict
from functools import partial
from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn
from safetensors import safe_open

from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead
from ...utils.download import resolve_file_path
from ...utils.log import logger
from ..cache_utils import Cache
from ..configuration_utils import PretrainedConfig
from ..model_outputs import CausalLMOutputWithPast
from ..model_utils import dtype_guard, register_base_model
from ..qwen2.modeling import (
    Qwen2ForCausalLMDeprecated,
    Qwen2Model,
    Qwen2PretrainedModel,
)
from ..utils import get_checkpoint_shard_files
from .configuration import FastVLMConfig
from .modeling_vision import FastVLMVisionModel

IGNORE_INDEX = -100


def _resolve_hf_checkpoint_dir(pretrained_model_name_or_path, **kwargs):
    """Resolve a local directory containing every Hugging Face safetensors shard."""
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    subfolder = kwargs.get("subfolder", "")
    resolved_file = resolve_file_path(
        pretrained_model_name_or_path,
        ["model.safetensors.index.json", "model.safetensors"],
        subfolder=subfolder,
        revision=kwargs.get("revision"),
        cache_dir=kwargs.get("cache_dir"),
        force_download=kwargs.get("force_download", False),
        token=kwargs.get("token"),
        local_files_only=kwargs.get("local_files_only", False),
        download_hub=kwargs.get("download_hub"),
    )
    if resolved_file.endswith(".index.json"):
        get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_file,
            cache_dir=kwargs.get("cache_dir"),
            subfolder=subfolder,
            download_hub=kwargs.get("download_hub"),
        )
    return os.path.dirname(resolved_file)


def _is_hf_fastvlm_checkpoint(model_dir):
    """Distinguish original FastVLM files from Paddle safetensors round trips."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    checkpoint_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as file:
            keys = json.load(file).get("weight_map", {}).keys()
    elif os.path.exists(checkpoint_path):
        with safe_open(checkpoint_path, framework="np") as shard:
            keys = list(shard.keys())
    else:
        return False
    return any(key.startswith("model.vision_tower.vision_tower.model.") for key in keys)


class FastVLMQKVLinear(nn.Linear):
    """QKV projection preserving the Transformers BF16 GEMM partitioning."""

    def __init__(self, config):
        head_dim = config.hidden_size // config.num_attention_heads
        output_size = config.hidden_size + 2 * config.num_key_value_heads * head_dim
        super().__init__(config.hidden_size, output_size, bias_attr=True)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = head_dim

    def forward(self, hidden_states):
        if hidden_states.dtype != paddle.bfloat16:
            return super().forward(hidden_states)

        queries_per_kv = self.num_attention_heads // self.num_key_value_heads
        group_width = (queries_per_kv + 2) * self.head_dim
        weight = self.weight.reshape([self.weight.shape[0], self.num_key_value_heads, group_width])
        bias = self.bias.reshape([self.num_key_value_heads, group_width])
        boundaries = [
            (0, queries_per_kv * self.head_dim),
            (queries_per_kv * self.head_dim, (queries_per_kv + 1) * self.head_dim),
            ((queries_per_kv + 1) * self.head_dim, group_width),
        ]
        projections = []
        for start, end in boundaries:
            part_weight = weight[:, :, start:end].reshape([self.weight.shape[0], -1]).contiguous()
            part_bias = bias[:, start:end].reshape([-1]).contiguous()
            projections.append(F.linear(hidden_states, part_weight, part_bias))

        query, key, value = projections
        input_shape = hidden_states.shape[:-1]
        query = query.reshape([*input_shape, self.num_key_value_heads, queries_per_kv * self.head_dim])
        key = key.reshape([*input_shape, self.num_key_value_heads, self.head_dim])
        value = value.reshape([*input_shape, self.num_key_value_heads, self.head_dim])
        return paddle.concat([query, key, value], axis=-1).reshape([*input_shape, -1])


def _iter_hf_tensors(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        file_to_keys = defaultdict(list)
        for key, filename in weight_map.items():
            file_to_keys[filename].append(key)
    else:
        checkpoint_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No safetensors checkpoint found under {model_dir}")
        file_to_keys = {"model.safetensors": None}

    for filename, keys in sorted(file_to_keys.items()):
        with safe_open(os.path.join(model_dir, filename), framework="np") as shard:
            for key in sorted(keys if keys is not None else shard.keys()):
                yield key, paddle.to_tensor(shard.get_tensor(key))


def _fuse_qkv(q_tensor, k_tensor, v_tensor, num_heads, num_key_value_heads, axis):
    q_splits = list(paddle.split(q_tensor, num_heads, axis=axis))
    k_splits = list(paddle.split(k_tensor, num_key_value_heads, axis=axis))
    v_splits = list(paddle.split(v_tensor, num_key_value_heads, axis=axis))
    queries_per_kv = num_heads // num_key_value_heads
    fused_parts = []
    for group_index in range(num_key_value_heads):
        start = group_index * queries_per_kv
        fused_parts.extend(q_splits[start : start + queries_per_kv])
        fused_parts.extend([k_splits[group_index], v_splits[group_index]])
    return paddle.concat(fused_parts, axis=axis)


def _load_hf_fastvlm_state_dict(model_dir, config):
    state_dict = {}
    qkv_weights = defaultdict(dict)
    qkv_biases = defaultdict(dict)
    ffn_weights = defaultdict(dict)

    for source_key, tensor in _iter_hf_tensors(model_dir):
        if source_key.startswith("model.vision_tower.vision_tower.model."):
            suffix = source_key[len("model.vision_tower.vision_tower.model.") :]
            if suffix.endswith(".num_batches_tracked"):
                continue
            suffix = suffix.replace(".running_mean", "._mean").replace(".running_var", "._variance")
            state_dict[f"model.vision_tower.vision_model.{suffix}"] = tensor
            continue

        if source_key.startswith("model.mm_projector."):
            state_dict[source_key] = tensor.transpose([1, 0]).contiguous() if source_key.endswith("weight") else tensor
            continue

        if source_key == "model.embed_tokens.weight":
            state_dict[source_key] = tensor
            if config.tie_word_embeddings:
                state_dict["lm_head.weight"] = tensor.clone()
            continue
        if source_key in {"model.norm.weight", "lm_head.weight"}:
            state_dict[source_key] = tensor
            continue
        if not source_key.startswith("model.layers."):
            raise ValueError(f"Unhandled FastVLM checkpoint key: {source_key}")

        layer_suffix = source_key[len("model.layers.") :]
        layer_id, suffix = layer_suffix.split(".", 1)
        target_prefix = f"model.layers.{layer_id}"
        if suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
            state_dict[f"{target_prefix}.{suffix}"] = tensor
        elif suffix in {"self_attn.o_proj.weight", "mlp.down_proj.weight"}:
            state_dict[f"{target_prefix}.{suffix}"] = tensor.transpose([1, 0]).contiguous()
        elif suffix in {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        }:
            projection = suffix.split(".")[1][0]
            qkv_weights[layer_id][projection] = tensor.transpose([1, 0]).contiguous()
            if len(qkv_weights[layer_id]) == 3:
                values = qkv_weights.pop(layer_id)
                state_dict[f"{target_prefix}.self_attn.qkv_proj.weight"] = _fuse_qkv(
                    values["q"],
                    values["k"],
                    values["v"],
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    axis=1,
                )
        elif suffix in {"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}:
            projection = suffix.split(".")[1][0]
            qkv_biases[layer_id][projection] = tensor
            if len(qkv_biases[layer_id]) == 3:
                values = qkv_biases.pop(layer_id)
                state_dict[f"{target_prefix}.self_attn.qkv_proj.bias"] = _fuse_qkv(
                    values["q"],
                    values["k"],
                    values["v"],
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    axis=0,
                )
        elif suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
            projection = "gate" if "gate_proj" in suffix else "up"
            ffn_weights[layer_id][projection] = tensor.transpose([1, 0]).contiguous()
            if len(ffn_weights[layer_id]) == 2:
                values = ffn_weights.pop(layer_id)
                state_dict[f"{target_prefix}.mlp.up_gate_proj.weight"] = paddle.concat(
                    [values["gate"], values["up"]], axis=1
                )
        else:
            raise ValueError(f"Unhandled FastVLM text checkpoint key: {source_key}")

    if qkv_weights or qkv_biases or ffn_weights:
        raise ValueError("Incomplete FastVLM fused text checkpoint parameters")
    return state_dict


@register_base_model
class FastVLMModel(Qwen2Model):
    """Qwen2 decoder augmented with the FastVLM MobileCLIP vision tower."""

    config_class = FastVLMConfig

    def __init__(self, config: FastVLMConfig):
        config.fuse_rms_norm = False
        super().__init__(config)
        # The custom layer reproduces the three Hugging Face BF16 GEMMs on one
        # device. Under tensor parallelism Qwen2's ColumnParallelLinear must be
        # retained so each rank owns only its QKV shard.
        if config.tensor_model_parallel_size == 1:
            for layer in self.layers:
                layer.self_attn.qkv_proj = FastVLMQKVLinear(config)
        image_size = int(config.mm_vision_tower.rsplit("_", 1)[-1])
        self.vision_tower = FastVLMVisionModel(
            image_size=image_size,
            trainable=config.unfreeze_mm_vision_tower,
            vision_config=config.vision_config,
        )
        if config.mm_projector_type != "mlp2x_gelu":
            raise ValueError(f"Unsupported FastVLM projector: {config.mm_projector_type}")
        self.mm_projector = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def encode_images(self, pixel_values):
        return self.mm_projector(self.vision_tower(pixel_values))


class FastVLMForConditionalGeneration(Qwen2PretrainedModel):
    """FastVLM image-text conditional generation model."""

    config_class = FastVLMConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FastVLMConfig):
        super().__init__(config)
        self.model = FastVLMModel(config)
        self.lm_head = LMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if not isinstance(pretrained_model_name_or_path, (str, os.PathLike)):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        is_native_checkpoint = os.path.isdir(pretrained_model_name_or_path) and os.path.exists(
            os.path.join(pretrained_model_name_or_path, "model_state.pdparams")
        )
        if is_native_checkpoint:
            dtype = kwargs.pop("dtype", None)
            config = kwargs.pop("config", None)
            if not isinstance(config, PretrainedConfig):
                config_path = config if config is not None else pretrained_model_name_or_path
                config, _ = cls.config_class.from_pretrained(config_path, return_unused_kwargs=True, **kwargs)
            if dtype is not None:
                config.dtype = dtype
            with dtype_guard(dtype or paddle.get_default_dtype()):
                model = cls(config, *args)
            state_dict = paddle.load(os.path.join(pretrained_model_name_or_path, "model_state.pdparams"))
            missing_keys, unexpected_keys = model.set_state_dict(state_dict)
            if missing_keys or unexpected_keys:
                raise ValueError(
                    f"Native FastVLM checkpoint is incomplete; missing={missing_keys}, unexpected={unexpected_keys}"
                )
            return model

        try:
            checkpoint_dir = _resolve_hf_checkpoint_dir(str(pretrained_model_name_or_path), **kwargs)
        except (EnvironmentError, FileNotFoundError, ValueError):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if not _is_hf_fastvlm_checkpoint(checkpoint_dir):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        accepted_init_kwargs = {
            name for name in inspect.signature(cls.__init__).parameters if name not in {"self", "config"}
        }
        dtype = kwargs.pop("dtype", None)
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(config_path, return_unused_kwargs=True, **kwargs)
        else:
            model_kwargs = kwargs
        model_kwargs = {key: value for key, value in model_kwargs.items() if key in accepted_init_kwargs}
        if dtype is not None:
            config.dtype = dtype
        with dtype_guard(dtype or paddle.get_default_dtype()):
            model = cls(config, *args, **model_kwargs)
        # Paddle's BatchNorm kernels keep parameters and running statistics in
        # FP32 even when convolution inputs use BF16/FP16.
        if dtype in {"bfloat16", "float16"}:
            for layer in model.model.vision_tower.sublayers():
                if isinstance(layer, paddle.compat.nn.BatchNorm2d):
                    layer.float()

        state_dict = _load_hf_fastvlm_state_dict(checkpoint_dir, config)
        if config.tensor_model_parallel_size > 1:
            state_dict = cls.convert_tensor_parallel(None, config, state_dict=state_dict)
        target_state_dict = model.state_dict()
        unknown_keys = sorted(set(state_dict) - set(target_state_dict))
        if unknown_keys:
            raise ValueError(f"Converted FastVLM checkpoint has unknown keys: {unknown_keys}")
        for name, tensor in state_dict.items():
            target = target_state_dict[name]
            if list(tensor.shape) != list(target.shape):
                raise ValueError(
                    f"FastVLM checkpoint shape mismatch for {name}: source {tensor.shape}, target {target.shape}"
                )
            if tensor.dtype != target.dtype:
                state_dict[name] = tensor.astype(target.dtype)
        missing_keys, unexpected_keys = model.set_state_dict(state_dict)
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"FastVLM checkpoint conversion is incomplete; missing={missing_keys}, unexpected={unexpected_keys}"
            )
        logger.info(f"Loaded and converted Hugging Face FastVLM checkpoint from {pretrained_model_name_or_path}")
        return model

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """Tensor-parallel actions for FastVLM's fused QKV and FFN layout."""
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )
        actions = {
            "embed_tokens.weight": partial(fn, is_column=False),
            "lm_head.weight": partial(fn, is_column=False),
        }
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layers.{layer_idx}"
            actions[f"{prefix}.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
            actions[f"{prefix}.self_attn.qkv_proj.bias"] = partial(fn, is_column=True)
            actions[f"{prefix}.self_attn.o_proj.weight"] = partial(fn, is_column=False)
            actions[f"{prefix}.mlp.up_gate_proj.weight"] = partial(fn, is_column=True, is_naive_2fuse=True)
            actions[f"{prefix}.mlp.down_proj.weight"] = partial(fn, is_column=False)
        return actions

    def save_pretrained(self, save_dir, *args, **kwargs):
        # FastVLM's vision tower does not use Hugging Face parameter names.
        # Default to the native Paddle checkpoint so a saved model is always
        # losslessly reloadable; callers can still explicitly request flex.
        kwargs.setdefault("save_checkpoint_format", "paddle")
        kwargs.setdefault("save_safetensors", False)
        return super().save_pretrained(save_dir, *args, **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _prepare_multimodal_inputs(self, input_ids, attention_mask, position_ids, labels, pixel_values):
        if pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, labels, None

        image_features = self.model.encode_images(pixel_values)
        original_attention_mask = attention_mask
        original_position_ids = position_ids
        original_labels = labels

        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids, dtype="bool")
        else:
            attention_mask = attention_mask.astype("bool")
        if labels is None:
            labels = paddle.full_like(input_ids, IGNORE_INDEX)

        new_embeddings = []
        new_labels = []
        for batch_index in range(input_ids.shape[0]):
            sample_ids = input_ids[batch_index][attention_mask[batch_index]]
            sample_labels = labels[batch_index][attention_mask[batch_index]]
            image_positions = paddle.nonzero(sample_ids == self.config.image_token_index).flatten().tolist()

            if not image_positions:
                new_embeddings.append(self.model.embed_tokens(sample_ids))
                new_labels.append(sample_labels)
                continue
            if len(image_positions) != 1:
                raise ValueError("FastVLM currently expects exactly one image token per sample.")

            image_position = image_positions[0]
            before_ids = sample_ids[:image_position]
            after_ids = sample_ids[image_position + 1 :]
            before_labels = sample_labels[:image_position]
            after_labels = sample_labels[image_position + 1 :]
            sample_image_features = image_features[batch_index]

            sample_embeddings = paddle.concat(
                [
                    self.model.embed_tokens(before_ids),
                    sample_image_features,
                    self.model.embed_tokens(after_ids),
                ],
                axis=0,
            )
            sample_new_labels = paddle.concat(
                [
                    before_labels,
                    paddle.full([sample_image_features.shape[0]], IGNORE_INDEX, dtype=sample_labels.dtype),
                    after_labels,
                ],
                axis=0,
            )
            new_embeddings.append(sample_embeddings[: self.config.tokenizer_model_max_length])
            new_labels.append(sample_new_labels[: self.config.tokenizer_model_max_length])

        max_length = max(item.shape[0] for item in new_embeddings)
        hidden_size = new_embeddings[0].shape[-1]
        padded_embeddings = []
        padded_labels = paddle.full([len(new_labels), max_length], IGNORE_INDEX, dtype=new_labels[0].dtype)
        new_attention_mask = paddle.zeros([len(new_labels), max_length], dtype="bool")
        new_position_ids = paddle.zeros([len(new_labels), max_length], dtype="int64")

        for batch_index, (sample_embeddings, sample_labels) in enumerate(zip(new_embeddings, new_labels)):
            sample_length = sample_embeddings.shape[0]
            pad = paddle.zeros([max_length - sample_length, hidden_size], dtype=sample_embeddings.dtype)
            if self.config.tokenizer_padding_side == "left":
                padded_embeddings.append(paddle.concat([pad, sample_embeddings], axis=0))
                padded_labels[batch_index, -sample_length:] = sample_labels
                new_attention_mask[batch_index, -sample_length:] = True
                new_position_ids[batch_index, -sample_length:] = paddle.arange(sample_length, dtype="int64")
            else:
                padded_embeddings.append(paddle.concat([sample_embeddings, pad], axis=0))
                padded_labels[batch_index, :sample_length] = sample_labels
                new_attention_mask[batch_index, :sample_length] = True
                new_position_ids[batch_index, :sample_length] = paddle.arange(sample_length, dtype="int64")

        inputs_embeds = paddle.stack(padded_embeddings, axis=0)
        if original_attention_mask is None:
            new_attention_mask = None
        else:
            new_attention_mask = new_attention_mask.astype(original_attention_mask.dtype)
        if original_position_ids is None:
            new_position_ids = None
        if original_labels is None:
            padded_labels = None
        return None, new_attention_mask, new_position_ids, padded_labels, inputs_embeds

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None and input_ids is not None:
            input_ids, attention_mask, position_ids, labels, inputs_embeds = self._prepare_multimodal_inputs(
                input_ids, attention_mask, position_ids, labels, pixel_values
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        if not return_dict:
            result = (logits,) + outputs[1:]
            return (loss,) + result if loss is not None else result
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            # The image placeholder expands to hundreds of embeddings, while
            # GenerationMixin tracks the unexpanded token mask. Recompute the
            # short decode sequence until multimodal cache metadata is carried
            # explicitly, otherwise the second-step cache mask is malformed.
            "use_cache": False if pixel_values is not None else kwargs.get("use_cache"),
            "pixel_values": pixel_values,
        }
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
        return model_inputs

    @classmethod
    def _gen_aoa_config(cls, config):
        aoa_config = Qwen2ForCausalLMDeprecated._gen_aoa_config(config)
        aoa_config["aoa_statements"].extend(
            [
                "model.mm_projector.0.weight^T -> model.mm_projector.0.weight",
                "model.mm_projector.0.bias -> model.mm_projector.0.bias",
                "model.mm_projector.2.weight^T -> model.mm_projector.2.weight",
                "model.mm_projector.2.bias -> model.mm_projector.2.bias",
            ]
        )
        return aoa_config


FastVLMForCausalLM = FastVLMForConditionalGeneration

__all__ = ["FastVLMForCausalLM", "FastVLMForConditionalGeneration", "FastVLMModel"]
