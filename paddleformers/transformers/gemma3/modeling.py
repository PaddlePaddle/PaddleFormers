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

import inspect
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn

if not hasattr(paddle.nn.functional, "swiglu"):

    def _compat_swiglu(x):
        gate, value = x.chunk(2, axis=-1)
        return paddle.nn.functional.silu(gate) * value

    paddle.nn.functional.swiglu = _compat_swiglu

from ...generation import GenerationMixin
from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead as GeneralLMHead
from ...utils.log import logger
from ..activations import ACT2FN
from ..cache_utils import Cache
from ..configuration_utils import PretrainedConfig
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from ..model_utils import PretrainedModel, dtype_guard
from .configuration import Gemma3Config, SiglipVisionConfig
from .multimodal_text_modeling import (
    Gemma3RMSNorm,
    Gemma3TextModel,
    _iter_hf_tensors,
    _compute_causal_lm_loss,
    _restore_padding_query_rows,
    load_hf_text_state_dict,
)


_HF_VISION_LINEAR_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)


def _convert_hf_vision_tensor(target_key: str, tensor: paddle.Tensor) -> paddle.Tensor:
    """Convert a Torch vision tensor to the layout used by Paddle layers."""
    if target_key.endswith(_HF_VISION_LINEAR_WEIGHT_SUFFIXES):
        return tensor.transpose([1, 0]).contiguous()
    return tensor


def _use_high_precision_cublas_for_fp32(tensor: paddle.Tensor) -> None:
    """Match Torch's default FP32 GEMM policy before multimodal computation."""
    if tensor.dtype != paddle.float32 or not tensor.place.is_gpu_place():
        return

    # Paddle enables TF32 cuBLAS kernels by default on Ampere and newer GPUs,
    # while Torch's default float32_matmul_precision="highest" does not.  The
    # switch is process-wide and must remain disabled for the following text
    # backbone and backward pass as well as for the vision tower.
    from paddle.base import core

    if core.get_cublas_switch():
        core.set_cublas_switch(False)


@dataclass
class Gemma3ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: paddle.Tensor | None = None


@dataclass
class Gemma3CausalLMOutputWithPast(ModelOutput):
    loss: paddle.Tensor | None = None
    logits: paddle.Tensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[paddle.Tensor] | None = None
    attentions: tuple[paddle.Tensor] | None = None
    image_hidden_states: paddle.Tensor | None = None


class SiglipVisionEmbeddings(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            paddle.arange(self.num_patches, dtype="int64").reshape([1, -1]),
            persistable=False,
        )

    def interpolate_pos_encoding(self, embeddings: paddle.Tensor, height: int, width: int) -> paddle.Tensor:
        if embeddings.shape[1] == self.num_patches and height == width == self.image_size:
            return self.position_embedding(self.position_ids)

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(self.num_patches**0.5)

        patch_pos_embed = self.position_embedding.weight.reshape([1, sqrt_num_positions, sqrt_num_positions, dim])
        patch_pos_embed = patch_pos_embed.transpose([0, 3, 1, 2])
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=[new_height, new_width],
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.transpose([0, 2, 3, 1]).reshape([1, -1, dim])
        return patch_pos_embed

    def forward(self, pixel_values: paddle.Tensor, interpolate_pos_encoding: bool = False) -> paddle.Tensor:
        _, _, height, width = pixel_values.shape
        pixel_values = pixel_values.astype(self.patch_embedding.weight.dtype)
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(start_axis=2).transpose([0, 2, 1])
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape([batch_size, seq_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        keys = keys.reshape([batch_size, seq_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        values = values.reshape([batch_size, seq_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        attn_weights = paddle.matmul(queries, keys.transpose([0, 1, 3, 2])) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(queries.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = paddle.matmul(attn_weights, values)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, seq_length, embed_dim])
        return self.out_proj(attn_output), attn_weights


class SiglipMLP(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class SiglipEncoderLayer(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: paddle.Tensor, attention_mask: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SiglipEncoder(nn.Layer):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.LayerList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: paddle.Tensor, attention_mask: Optional[paddle.Tensor] = None) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Gemma3VisionPretrainedModel(PretrainedModel):
    config_class = SiglipVisionConfig
    base_model_prefix = "vision_model"
    main_input_name = "pixel_values"


class Gemma3VisionModel(Gemma3VisionPretrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: paddle.Tensor,
        interpolate_pos_encoding: bool = False,
        attention_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        del kwargs
        _use_high_precision_cublas_for_fp32(pixel_values)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        outputs = self.encoder(hidden_states, attention_mask=attention_mask)
        last_hidden_state = self.post_layernorm(outputs.last_hidden_state)
        if not return_dict:
            return (last_hidden_state, None)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=None)


class Gemma3MultiModalProjector(paddle.nn.Layer):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.mm_input_projection_weight = self.create_parameter(
            shape=[config.vision_config.hidden_size, config.text_config.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Normal(std=config.initializer_range),
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size,
            eps=getattr(config.vision_config, "layer_norm_eps", 1e-6),
        )
        self.avg_pool = paddle.nn.AvgPool2D(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: paddle.Tensor) -> paddle.Tensor:
        batch_size, _, hidden_size = vision_outputs.shape
        reshaped_vision_outputs = vision_outputs.transpose([0, 2, 1]).reshape(
            [batch_size, hidden_size, self.patches_per_image, self.patches_per_image]
        )
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(start_axis=2).transpose([0, 2, 1])
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)
        projected_vision_outputs = paddle.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.astype(vision_outputs.dtype)


def token_type_ids_mask_function(token_type_ids: paddle.Tensor | None, image_group_ids: paddle.Tensor | None):
    def mask_fn(batch_idx: paddle.Tensor, head_idx: paddle.Tensor, q_idx: paddle.Tensor, kv_idx: paddle.Tensor):
        del head_idx
        if token_type_ids is None or image_group_ids is None:
            return paddle.zeros([1], dtype="bool")

        bsz, seq_len = token_type_ids.shape
        max_idx = paddle.full_like(q_idx, seq_len - 1)
        safe_q_idx = paddle.minimum(q_idx, max_idx)
        safe_kv_idx = paddle.minimum(kv_idx, paddle.full_like(kv_idx, seq_len - 1))

        token_type_ids_expanded = token_type_ids.reshape([bsz, 1, 1, seq_len])
        image_group_ids_expanded = image_group_ids.reshape([bsz, 1, 1, seq_len])

        safe_q_idx = safe_q_idx.expand([bsz, 1, safe_q_idx.shape[2], 1])
        safe_kv_idx = safe_kv_idx.expand([bsz, 1, 1, safe_kv_idx.shape[3]])

        token_type_ids_at_q_idx = paddle.take_along_axis(token_type_ids_expanded, safe_q_idx, axis=-1)
        token_type_ids_at_kv_idx = paddle.take_along_axis(token_type_ids_expanded, safe_kv_idx, axis=-1)
        image_group_ids_at_q_idx = paddle.take_along_axis(image_group_ids_expanded, safe_q_idx, axis=-1)
        image_group_ids_at_kv_idx = paddle.take_along_axis(image_group_ids_expanded, safe_kv_idx, axis=-1)

        valid_q = (q_idx < seq_len).expand([bsz, 1, q_idx.shape[2], 1])
        valid_kv = (kv_idx < seq_len).expand([bsz, 1, 1, kv_idx.shape[3]])
        is_image_block = (token_type_ids_at_q_idx == 1) & (token_type_ids_at_kv_idx == 1)
        same_image_block = image_group_ids_at_q_idx == image_group_ids_at_kv_idx
        return valid_q & valid_kv & is_image_block & same_image_block

    return mask_fn


def create_causal_mask_mapping(
    config: Gemma3Config,
    inputs_embeds: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor],
    cache_position: Optional[paddle.Tensor],
    past_key_values: Optional[Cache],
    position_ids: Optional[paddle.Tensor],
    token_type_ids: Optional[paddle.Tensor] = None,
    is_first_iteration: Optional[bool] = None,
    prepare_decoder_attention_mask: Optional[Callable] = None,
    return_row_indices: bool = False,
):
    batch_size, seq_length = inputs_embeds.shape[:2]
    cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    prepare_decoder_attention_mask = prepare_decoder_attention_mask or Gemma3TextModel._prepare_decoder_attention_mask
    or_mask_function = None
    if token_type_ids is not None and (is_first_iteration is not False):
        is_image = token_type_ids == 1
        previous_is_image = paddle.concat(
            [paddle.zeros([is_image.shape[0], 1], dtype=is_image.dtype), is_image[:, :-1]],
            axis=1,
        )
        new_image_start = is_image & ~previous_is_image.astype("bool")
        image_group_ids = paddle.cumsum(new_image_start.astype("int64"), axis=1) - 1
        image_group_ids = paddle.where(is_image, image_group_ids, paddle.full_like(image_group_ids, -1))
        or_mask_function = token_type_ids_mask_function(
            token_type_ids.astype("int64"), image_group_ids.astype("int64")
        )

    mask_kwargs = {
        "config": config.text_config,
        "inputs_embeds": inputs_embeds,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "cache_length": cache_length,
        "attention_mask": attention_mask,
        "attn_mask_startend_row_indices": None,
        "prepare_decoder_attention_mask": prepare_decoder_attention_mask,
        "or_mask_function": or_mask_function,
    }

    full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)
    full_mask = _restore_padding_query_rows(full_mask, attention_mask, cache_length, seq_length)
    causal_mask_mapping = {"full_attention": full_mask}
    attn_mask_startend_row_indices_mapping = {"full_attention": full_indices}
    if "sliding_attention" in getattr(config.text_config, "layer_types", []):
        sliding_mask, sliding_indices = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)
        sliding_mask = _restore_padding_query_rows(sliding_mask, attention_mask, cache_length, seq_length)
        causal_mask_mapping["sliding_attention"] = sliding_mask
        attn_mask_startend_row_indices_mapping["sliding_attention"] = sliding_indices

    if return_row_indices:
        return causal_mask_mapping, attn_mask_startend_row_indices_mapping
    return causal_mask_mapping


class Gemma3PreTrainedModel(PretrainedModel):
    config_class = Gemma3Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = [
        "Gemma3MultiModalProjector",
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
    ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        accepted_init_kwargs = {
            name for name in inspect.signature(cls.__init__).parameters if name not in {"self", "config"}
        }
        passthrough_kwargs = {key: value for key, value in kwargs.items() if key not in accepted_init_kwargs}
        init_kwargs = {key: value for key, value in kwargs.items() if key in accepted_init_kwargs}
        if (
            isinstance(pretrained_model_name_or_path, str)
            and os.path.isdir(pretrained_model_name_or_path)
            and os.path.exists(os.path.join(pretrained_model_name_or_path, "model_state.pdparams"))
        ):
            logger.info(
                f"Using local Paddle checkpoint direct load path for {cls.__name__} "
                f"from {pretrained_model_name_or_path}"
            )
            dtype = passthrough_kwargs.pop("dtype", None)
            config = passthrough_kwargs.pop("config", None)
            if not isinstance(config, PretrainedConfig):
                config_path = config if config is not None else pretrained_model_name_or_path
                config, model_kwargs = cls.config_class.from_pretrained(
                    config_path,
                    return_unused_kwargs=True,
                    **passthrough_kwargs,
                )
            else:
                model_kwargs = passthrough_kwargs

            model_kwargs = {key: value for key, value in model_kwargs.items() if key in accepted_init_kwargs}
            model_kwargs.update(init_kwargs)
            if dtype is not None:
                config.dtype = dtype
            with dtype_guard(dtype or paddle.get_default_dtype()):
                model = cls(config, *args, **model_kwargs)
            state_dict = paddle.load(os.path.join(pretrained_model_name_or_path, "model_state.pdparams"))
            target_state_dict = model.state_dict()
            for name, tensor in list(state_dict.items()):
                if name in target_state_dict and tensor.dtype != target_state_dict[name].dtype:
                    state_dict[name] = tensor.astype(target_state_dict[name].dtype)
            missing_keys, unexpected_keys = model.set_state_dict(state_dict)
            if missing_keys or unexpected_keys:
                logger.warning(
                    "Local Paddle checkpoint load finished with missing keys %s and unexpected keys %s",
                    missing_keys,
                    unexpected_keys,
                )
            return model

        is_hf_safetensors = (
            isinstance(pretrained_model_name_or_path, str)
            and os.path.isdir(pretrained_model_name_or_path)
            and (
                os.path.exists(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
                or os.path.exists(os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json"))
            )
        )
        if is_hf_safetensors:
            dtype = passthrough_kwargs.pop("dtype", None)
            config = passthrough_kwargs.pop("config", None)
            if not isinstance(config, PretrainedConfig):
                config_path = config if config is not None else pretrained_model_name_or_path
                config, model_kwargs = cls.config_class.from_pretrained(
                    config_path,
                    return_unused_kwargs=True,
                    **passthrough_kwargs,
                )
            else:
                model_kwargs = passthrough_kwargs

            model_kwargs = {key: value for key, value in model_kwargs.items() if key in accepted_init_kwargs}
            model_kwargs.update(init_kwargs)
            if dtype is not None:
                config.dtype = dtype
            with dtype_guard(dtype or paddle.get_default_dtype()):
                model = cls(config, *args, **model_kwargs)
            model_prefix = "" if cls.__name__ == "Gemma3Model" else "model."
            state_dict = load_hf_text_state_dict(
                pretrained_model_name_or_path,
                config.text_config,
                model_prefix=f"{model_prefix}language_model.",
                include_lm_head=cls.__name__ != "Gemma3Model",
                source_prefix="language_model.",
            )
            target_state_dict = model.state_dict()
            for hf_key, tensor in _iter_hf_tensors(pretrained_model_name_or_path):
                if hf_key.startswith("vision_tower.vision_model."):
                    target_key = f"{model_prefix}vision_tower." + hf_key[len("vision_tower.vision_model.") :]
                elif hf_key.startswith("multi_modal_projector."):
                    target_key = model_prefix + hf_key
                else:
                    continue
                if target_key not in target_state_dict:
                    continue
                state_dict[target_key] = _convert_hf_vision_tensor(target_key, tensor)

            for name, tensor in list(state_dict.items()):
                if name in target_state_dict and tensor.dtype != target_state_dict[name].dtype:
                    state_dict[name] = tensor.astype(target_state_dict[name].dtype)
            missing_keys, unexpected_keys = model.set_state_dict(state_dict)
            if missing_keys or unexpected_keys:
                logger.warning(
                    f"HF Gemma3 checkpoint load finished with missing keys {missing_keys} "
                    f"and unexpected keys {unexpected_keys}"
                )
            return model

        kwargs = passthrough_kwargs
        kwargs.update(init_kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def _gen_aoa_config(cls, config: Gemma3Config):
        model_prefix = "" if cls.__name__ == "Gemma3Model" else "model."
        llm_prefix = f"{model_prefix}language_model."
        vision_prefix = f"{model_prefix}vision_tower."
        projector_prefix = f"{model_prefix}multi_modal_projector."

        aoa_statements = [
            f"language_model.model.embed_tokens.weight -> {llm_prefix}embed_tokens.weight",
            f"language_model.model.norm.weight -> {llm_prefix}norm.weight",
            f"language_model.model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"language_model.model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"language_model.model.layers.$LAYER_ID.pre_feedforward_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.pre_feedforward_layernorm.weight",
            f"language_model.model.layers.$LAYER_ID.post_feedforward_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_feedforward_layernorm.weight",
            f"language_model.model.layers.$LAYER_ID.self_attn.q_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
            f"language_model.model.layers.$LAYER_ID.self_attn.k_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            f"language_model.model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
            f"language_model.model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
            f"vision_tower.vision_model.embeddings.patch_embedding.weight -> {vision_prefix}embeddings.patch_embedding.weight",
            f"vision_tower.vision_model.embeddings.patch_embedding.bias -> {vision_prefix}embeddings.patch_embedding.bias",
            f"vision_tower.vision_model.embeddings.position_embedding.weight -> {vision_prefix}embeddings.position_embedding.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.layer_norm1.weight -> {vision_prefix}encoder.layers.$LAYER_ID.layer_norm1.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.layer_norm1.bias -> {vision_prefix}encoder.layers.$LAYER_ID.layer_norm1.bias",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.layer_norm2.weight -> {vision_prefix}encoder.layers.$LAYER_ID.layer_norm2.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.layer_norm2.bias -> {vision_prefix}encoder.layers.$LAYER_ID.layer_norm2.bias",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.q_proj.weight^T -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.q_proj.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.k_proj.weight^T -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.k_proj.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.v_proj.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.out_proj.weight^T -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.out_proj.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.q_proj.bias -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.q_proj.bias",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.k_proj.bias -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.k_proj.bias",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.v_proj.bias -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.v_proj.bias",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.self_attn.out_proj.bias -> {vision_prefix}encoder.layers.$LAYER_ID.self_attn.out_proj.bias",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.mlp.fc1.weight^T -> {vision_prefix}encoder.layers.$LAYER_ID.mlp.fc1.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.mlp.fc2.weight^T -> {vision_prefix}encoder.layers.$LAYER_ID.mlp.fc2.weight",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.mlp.fc1.bias -> {vision_prefix}encoder.layers.$LAYER_ID.mlp.fc1.bias",
            f"vision_tower.vision_model.encoder.layers.$LAYER_ID.mlp.fc2.bias -> {vision_prefix}encoder.layers.$LAYER_ID.mlp.fc2.bias",
            f"vision_tower.vision_model.post_layernorm.weight -> {vision_prefix}post_layernorm.weight",
            f"vision_tower.vision_model.post_layernorm.bias -> {vision_prefix}post_layernorm.bias",
            f"multi_modal_projector.mm_input_projection_weight -> {projector_prefix}mm_input_projection_weight",
            f"multi_modal_projector.mm_soft_emb_norm.weight -> {projector_prefix}mm_soft_emb_norm.weight",
        ]

        aoa_statements += [
            (
                "language_model.model.layers.$LAYER_ID.self_attn.q_proj.weight^T, "
                "language_model.model.layers.$LAYER_ID.self_attn.k_proj.weight^T, "
                "language_model.model.layers.$LAYER_ID.self_attn.v_proj.weight^T "
                f"-> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, "
                f"fused_qkv, num_heads={config.text_config.num_attention_heads}, "
                f"num_key_value_groups={config.text_config.num_key_value_heads}"
            ),
            (
                "language_model.model.layers.$LAYER_ID.mlp.gate_proj.weight^T, "
                "language_model.model.layers.$LAYER_ID.mlp.up_proj.weight^T "
                f"-> {llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn"
            ),
        ]

        if cls.__name__ != "Gemma3Model":
            aoa_statements.append("language_model.model.embed_tokens.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}


class Gemma3Model(Gemma3PreTrainedModel):
    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.vision_tower = Gemma3VisionModel._from_config(config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.language_model = Gemma3TextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: paddle.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        kwargs.pop("return_dict", None)
        vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True, **kwargs)
        image_features = self.multi_modal_projector(vision_outputs.last_hidden_state)
        return BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=image_features,
            hidden_states=getattr(vision_outputs, "hidden_states", None),
            attentions=getattr(vision_outputs, "attentions", None),
        )

    def get_placeholder_mask(
        self,
        input_ids: Optional[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        image_features: paddle.Tensor,
    ) -> paddle.Tensor:
        if input_ids is None:
            image_token_embed = self.get_input_embeddings()(
                paddle.to_tensor([self.config.image_token_index], dtype="int64")
            )
            special_image_mask = (inputs_embeds == image_token_embed).all(axis=-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index

        n_image_tokens = int(special_image_mask.astype("int64").sum().item())
        n_image_features = int(image_features.shape[0] * image_features.shape[1])
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}"
            )
        return special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[Union[paddle.Tensor, dict[str, Optional[paddle.Tensor]]]] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, Gemma3ModelOutputWithPast]:
        del labels
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = paddle.where(special_image_mask, paddle.zeros_like(input_ids), input_ids)
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        _use_high_precision_cublas_for_fp32(inputs_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = paddle.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype="int64")

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, return_dict=True).pooler_output
            image_features = image_features.astype(inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
            attn_mask_startend_row_indices_mapping = attn_mask_startend_row_indices
        else:
            causal_mask_mapping, attn_mask_startend_row_indices_mapping = create_causal_mask_mapping(
                self.config,
                inputs_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
                token_type_ids=token_type_ids,
                is_first_iteration=bool(pixel_values is not None),
                prepare_decoder_attention_mask=self.language_model._prepare_decoder_attention_mask,
                return_row_indices=True,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        if not return_dict:
            result = (
                outputs.last_hidden_state,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                image_features,
            )
            return tuple(value for value in result if value is not None)

        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class Gemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.model = Gemma3Model(config)
        self.lm_head = GeneralLMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(self, pixel_values: paddle.Tensor, **kwargs):
        return self.model.get_image_features(pixel_values, **kwargs)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[Union[paddle.Tensor, dict[str, Optional[paddle.Tensor]]]] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int | paddle.Tensor = 0,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = paddle.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = _compute_causal_lm_loss(
                logits,
                labels,
                self.config.text_config.vocab_size,
                attention_mask,
                input_ids,
            )

        if not return_dict:
            result = (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                outputs.image_hidden_states,
            )
            return ((loss,) + result) if loss is not None else result

        return Gemma3CausalLMOutputWithPast(
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
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
        if labels is not None:
            logger.warning("`labels` are ignored during generation.")
        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: Gemma3Config,
        inputs_embeds: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor],
        cache_position: paddle.Tensor,
        past_key_values: Optional[Cache],
        position_ids: Optional[paddle.Tensor],
        token_type_ids: Optional[paddle.Tensor] = None,
        is_first_iteration: Optional[bool] = False,
        **kwargs,
    ) -> dict:
        return create_causal_mask_mapping(
            config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
        )


__all__ = [
    "SiglipVisionEmbeddings",
    "SiglipAttention",
    "SiglipMLP",
    "SiglipEncoderLayer",
    "SiglipEncoder",
    "SiglipVisionConfig",
    "Gemma3VisionModel",
    "Gemma3ModelOutputWithPast",
    "Gemma3CausalLMOutputWithPast",
    "Gemma3MultiModalProjector",
    "Gemma3PreTrainedModel",
    "Gemma3Model",
    "Gemma3ForConditionalGeneration",
]
