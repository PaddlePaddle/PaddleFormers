# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import paddle
from paddle.nn import functional as F
from paddlefleet import parallel_state
from paddlefleet.models.kimi_k25.kimi_k25_builders import kimi_k25_vision_builder
from paddlefleet.models.kimi_k25.kimi_k25_model import (
    KimiK25VisionModel,
    KimiK25VisionTransformerLayer,
)
from paddlefleet.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from paddlefleet.spec_utils import LayerSpec
from paddlefleet.transformer.enums import ModelType
from paddlefleet.transformer.transformer_config import TransformerConfig

from ...nn.criterion.interface import CriterionLayer
from ...utils.masking_utils import _expand_2d_mask, _make_causal_mask
from ..cache_utils import Cache
from ..gpt_provider import GPTModelProvider
from ..masking_utils import create_causal_masks_and_row_indices
from ..model_utils import PretrainedModel
from .configuration import KimiK25Config
from .modeling_base import KimiK25CausalLMOutputWithPast, KimiK25PretrainedModel


class KimiK25VisionProvider(TransformerConfig):
    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304
    embed_dim: int = (1152,)
    hidden_size: int = 1152
    out_hidden_size: int = 4096
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    # intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    hidden_act: Callable = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = KimiK25VisionTransformerLayer
    model_version: str = "kimi_k25"
    img_h: int = 336
    img_w: int = 336
    add_class_token: bool = False
    class_token_len: int = 1
    high_precision_rope: bool = True
    rotary_percent: float = 1.0
    transform_rules = {
        "dtype": "params_dtype",
        "vt_hidden_size": "hidden_size",
        "vt_intermediate_size": "intermediate_size",
        "vt_num_attention_heads": "num_attention_heads",
        "vt_num_hidden_layers": "num_hidden_layers",
    }

    def provide(self) -> "KimiK25VisionModel":
        pp_size = self.pipeline_model_parallel_size

        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        is_pipeline_asymmetric |= (
            getattr(self, "num_empty_layers_add_in_head", None) or getattr(self, "num_empty_layers_add_in_tail", None)
        ) is not None

        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(paddle.device, device="meta")

        with model_init_device_context():
            res_model = kimi_k25_vision_builder(
                self,
                seg_method="layer:TransformerLayer|EmptyLayer",
                num_stages=pp_size,
            )
        return res_model


@dataclass
class KimiK25TextProvider(GPTModelProvider):
    """
    Base config for Kimi-K25 Models.
    """

    transform_rules = {
        "dtype": "params_dtype",
    }

    def __post_init__(self):
        super().__post_init__()
        # self.mrope_section = self.rope_scaling.get("mrope_section", [24, 20, 20])


class KimiK25VisionModelFleet(KimiK25PretrainedModel):
    def __new__(cls, config):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = KimiK25VisionProvider
        model_provider = model_provider_class.from_config(config)

        vision_model = model_provider.provide()
        vision_model.config_to_save = config

        return vision_model


@dataclass
class KimiK25Provider(TransformerConfig):
    text_config: KimiK25TextProvider | None = None
    vision_config: KimiK25VisionProvider | None = None

    freeze_langurage_model: bool = False
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False

    def provide(self, tokenizer=None, vp_stage: int | None = None) -> "KimiK25ModelDist":
        self.text_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.text_config.sequence_parallel = self.sequence_parallel
        self.text_config.context_parallel_size = self.context_parallel_size
        self.text_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size

        self.vision_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.vision_config.pipeline_model_parallel_size = 1  # "ViT can only live on 1 pipeline stage."

        config_attrs = [
            "cross_entropy_loss_fusion",
            "gradient_accumulation_fusion",
            "bias_activation_fusion",
            "bias_dropout_fusion",
            "masked_softmax_fusion",
            "attention_softmax_in_fp32",
            "apply_rope_fusion",
            "overlap_p2p_comm",
            "batch_p2p_comm",
        ]

        for config in [
            self.text_config,
            self.vision_config,
        ]:
            for attr in config_attrs:
                setattr(config, attr, getattr(self, attr))

        self.text_config.tp_comm_overlap = self.tp_comm_overlap
        self.vision_config.tp_comm_overlap = False

        vp_stage = vp_stage or 0

        model = KimiK25ModelDist(
            config=self,
            tokenizer=tokenizer,
            pre_process=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() == self.pipeline_model_parallel_size,
            post_process=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_encoder=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_decoder=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() >= self.pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
            vp_stage=vp_stage,
        )

        return model

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.vision_config = KimiK25VisionProvider.from_config(config.vision_config)
        res.text_config = KimiK25TextProvider.from_config(config.text_config)
        # set text config params
        res.text_config.multi_latent_attention = True
        res.text_config.use_qk_norm = True

        res.vision_config.normalization = "LayerNorm"
        res.vision_config.gated_linear_unit = False

        return res


class KimiK25ModelDist(MCoreLLaVAModel):
    """KimiK25 Model Base Model Class."""

    def __init__(
        self,
        config: KimiK25Provider,
        tokenizer=None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
        vp_stage: int | None = None,
        model_version: str | None = None,
        criterion=False,
    ) -> None:
        super(MCoreLLaVAModel, self).__init__(config=config)

        language_transformer_config = config.text_config
        vision_transformer_config = config.vision_config

        self.model_version = vision_transformer_config.model_version if model_version is None else model_version
        assert self.model_version is not None

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage

        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.share_embeddings_and_output_weights = False
        self.rope_deltas = None

        if self.add_decoder:
            print("language_transformer_config ", language_transformer_config)
            self.language_model = language_transformer_config.provide(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )

            # self.language_model = DeepseekV3ForCausalLM(language_transformer_config)

        if add_encoder:
            self.vision_model = KimiK25VisionModelFleet(vision_transformer_config)

            if hasattr(self.language_model, "dtype"):
                target_dtype = self.language_model.dtype
                self.vision_model = self.vision_model.to(dtype=target_dtype)

        self.freeze(
            freeze_language_model=config.freeze_langurage_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

        self.model_type = ModelType.encoder_or_decoder

    def _merge_input_ids_with_image_features_training(
        self,
        image_features: list[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        input_ids: paddle.Tensor,
    ):
        """
        Merge image features into inputs_embeds by replacing placeholder tokens in-place.
        The number of placeholder tokens in input_ids must equal the total number of image
        feature tokens. attention_mask and position_ids are not modified.

        Args:
            image_features: List of tensors, each of shape (num_tokens_i, embed_dim).
            inputs_embeds: Shape (batch_size, sequence_length, embed_dim).
            input_ids: Shape (batch_size, sequence_length).
            attention_mask: Shape (batch_size, sequence_length). Passed through unchanged.
            labels: Shape (batch_size, sequence_length), optional. Passed through unchanged.
        """
        image_features = paddle.cat(image_features, dim=0)
        image_token_index: int = self.config.media_placeholder_token_id

        # Find all placeholder positions and replace with image features
        image_mask = input_ids == image_token_index
        num_placeholders = image_mask.sum().item()
        num_image_tokens = image_features.shape[0]

        if num_placeholders != num_image_tokens:
            raise ValueError(
                f"The number of image placeholder tokens ({num_placeholders}) does not match "
                f"the number of image feature tokens ({num_image_tokens})."
            )

        inputs_embeds[image_mask] = image_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        return inputs_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features: list[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        labels: paddle.Tensor | None = None,
    ):
        """
        Args:
            image_features (:obj:`paddle.Tensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The image features to merge with the input embeddings.
            inputs_embeds (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, embed_dim)`):
                The input embeddings.
            input_ids (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The input ids.
            attention_mask (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The attention mask.
            labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`, *optional*):
                The labels.
        """
        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        image_features = paddle.cat(image_features, dim=0)
        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id
        ignore_index: int = self.config.ignore_index

        batch_size, sequence_length = input_ids.shape
        left_padding = not paddle.sum(input_ids[:, -1] == paddle.tensor(pad_token_id))

        # 1. Create a mask to know where special image tokens are
        _token_occupation_table = paddle.ones_like(input_ids.flatten())

        _token_occupation_table[input_ids.flatten() == image_token_index] = paddle.tensor(
            feature_lengths, dtype=paddle.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        max_embed_dim = _token_occupation_table.sum(-1).max().item()
        assert (
            max_embed_dim >= sequence_length
        ), f"The maximum embedding dimension ({max_embed_dim}) is less than the sequence length ({sequence_length})"
        batch_indices, non_image_indices = paddle.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        new_token_positions = paddle.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = paddle.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = paddle.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = paddle.full(
                (batch_size, max_embed_dim),
                ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask.
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = paddle.full(
            (batch_size, max_embed_dim), True, dtype=paddle.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.to(paddle.int32).cumsum(-1) - 1 >= nb_image_pad[:, None].to(
            target_device
        )

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {image_to_overwrite.sum()} while"
                f" the number of image features given to the model is {image_features.shape[:-1].numel()}. "
                "This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite.to(paddle.int64)
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = paddle.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]
        if indices_to_mask.size != 0:
            final_embedding[batch_indices, indices_to_mask] = 0
        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _extract_image_features(self, pixel_values: paddle.Tensor, grid_thws: paddle.Tensor) -> list[paddle.Tensor]:
        """
        Args:
            pixel_values (:obj:`paddle.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
                The pixel values of the images processed by image processor.
            grid_thws (:obj:`paddle.Tensor` of shape :obj:`(batch_size, 3)`):
                The grid, height, width of the images.
        Returns:
            selected_image_feature (:obj:`paddle.FloatTensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The selected image features to use as input to the projector head.
        """

        def get_attn_mask_startend_row_indices(
            grid_thws: paddle.Tensor,
        ):
            lengths = paddle.cat(
                (
                    paddle.zeros(1, dtype=grid_thws.dtype),
                    grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
                )
            )

            cu_seqlens = lengths.cumsum(dim=0, dtype=paddle.int32)
            cu_seqlens_rm_first = cu_seqlens[1:]
            cu_seqlens_rm_last = cu_seqlens[:-1]
            repeats = cu_seqlens_rm_first - cu_seqlens_rm_last

            startend_row_indices_lts = paddle.repeat_interleave(cu_seqlens_rm_first, repeats).reshape([1, 1, -1, 1])
            startend_row_indices_ute = paddle.repeat_interleave(cu_seqlens_rm_last, repeats).reshape([1, 1, -1, 1])
            startend_row_indices = paddle.concat([startend_row_indices_lts, startend_row_indices_ute], axis=-1)

            return startend_row_indices

        attn_mask_startend_row_indices = get_attn_mask_startend_row_indices(grid_thws)

        pixel_values = pixel_values.to(self.config.params_dtype)

        input_dict = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }
        image_features = self.vision_model(input_dict)
        return image_features

    # copy from DeepseekV3Model
    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

    def get_inputs_embeds(
        self,
        input_ids: paddle.LongTensor,
        pixel_values: paddle.Tensor | None = None,
        grid_thws: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        past_key_values: list[paddle.FloatTensor] | None = None,
        labels: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
    ):
        # 1. Extra the input embeddings
        inputs_embeds = self.language_model.run_function[0]({"input_ids": input_ids})["hidden_states"]
        print("inputs_embeds ", inputs_embeds)
        # 2. Merge text and images
        if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
            image_features = self._extract_image_features(pixel_values, grid_thws)
            image_features = image_features["hidden_states"]

            num_image_tokens = sum([image.shape[0] for image in image_features])
            image_token_index: int = self.config.media_placeholder_token_id

            # Find all placeholder positions and replace with image features
            image_mask = input_ids == image_token_index
            num_placeholders = image_mask.sum().item()
            if num_placeholders == num_image_tokens:
                inputs_embeds = self._merge_input_ids_with_image_features_training(
                    image_features, inputs_embeds, input_ids
                )

            else:
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )
                batch_size, seq_length = inputs_embeds.shape[:2]
                past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
                mask_kwargs = {
                    "config": self.config,
                    "inputs_embeds": inputs_embeds,
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "cache_length": past_key_values_length,
                    "attention_mask": attention_mask,
                    "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
                    "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
                    "return_mapping": False,
                }

                # if attention_mask is not None or attn_mask_startend_row_indices is not None:
                attention_mask, attn_mask_startend_row_indices = create_causal_masks_and_row_indices(**mask_kwargs)
        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = paddle.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = paddle.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = paddle.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = paddle.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        input_dict = {
            "input_ids": None,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "decoder_input": inputs_embeds,
            "labels": labels,
            "past_key_values": past_key_values,
        }

        return input_dict

    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.LongTensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        inference_params=None,
        pixel_values: paddle.Tensor | None = None,
        grid_thws=None,  # image grid thws
        runtime_gather_output: bool | None = None,
        cache_position: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        **kwargs,
    ) -> paddle.Tensor:
        assert loss_mask is None, "loss_mask is not supported yet"
        input_dict = self.get_inputs_embeds(
            input_ids, pixel_values, grid_thws, attention_mask, None, labels, position_ids
        )
        input_dict["attn_mask_startend_row_indices"] = attn_mask_startend_row_indices
        labels = input_dict.get("labels", labels)

        output = self.language_model(input_dict)
        return output


class KimiK25PretrainedModelFleet(PretrainedModel):
    config_class = KimiK25Config

    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "gate_proj",
        "up_proj",
        "down_proj",
        "proj",
        "linear_fc\d+",
        "up_gate_proj",
        "qkv_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: KimiK25Config):
        # vision model
        aoa_config = {"aoa_statements": []}
        # qkv
        aoa_config["aoa_statements"] += [
            stmt
            for i in range(config.vision_config.vt_num_hidden_layers)
            for stmt in (
                f"vision_tower.encoder.blocks.{i}.wqkv.weight -> vision_tower.encoder.blocks.layers.{i}.self_attn.q.weight, vision_tower.encoder.blocks.layers.{i}.self_attn.k.weight, vision_tower.encoder.blocks.layers.{i}.self_attn.v.weight,axis=0",
                f"vision_tower.encoder.blocks.layers.{i}.self_attn.q.weight^T, vision_tower.encoder.blocks.layers.{i}.self_attn.k.weight^T, vision_tower.encoder.blocks.layers.{i}.self_attn.v.weight^T -> vision_tower.encoder.blocks.layers.{i}.self_attn.qkv_proj.weight,fused_qkv, num_heads={config.vision_config.vt_num_attention_heads}, num_key_value_groups={config.vision_config.vt_num_attention_heads}",
                f"vision_tower.encoder.blocks.{i}.wqkv.bias ->vision_tower.encoder.blocks.layers.{i}.self_attn.q.bias, vision_tower.encoder.blocks.layers.{i}.self_attn.k.bias, vision_tower.encoder.blocks.layers.{i}.self_attn.v.bias,axis=0",
                f"vision_tower.encoder.blocks.layers.{i}.self_attn.q.bias, vision_tower.encoder.blocks.layers.{i}.self_attn.k.bias, vision_tower.encoder.blocks.layers.{i}.self_attn.v.bias -> vision_tower.encoder.blocks.layers.{i}.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.vision_config.vt_num_attention_heads}, num_key_value_groups={config.vision_config.vt_num_attention_heads}, axis=0",
                f"vision_tower.encoder.blocks.{i}.wo.weight^T -> vision_tower.encoder.blocks.layers.{i}.self_attn.o_proj.weight",
                f"vision_tower.encoder.blocks.{i}.wo.bias -> vision_tower.encoder.blocks.layers.{i}.self_attn.o_proj.bias",
            )
        ]
        for i in range(config.vision_config.vt_num_hidden_layers):
            for last_prefix in ["bias", "weight"]:
                aoa_config["aoa_statements"] += [
                    f"vision_tower.encoder.blocks.{i}.norm0.{last_prefix} -> vision_tower.encoder.blocks.layers.{i}.input_layernorm.{last_prefix}",
                    f"vision_tower.encoder.blocks.{i}.norm1.{last_prefix} -> vision_tower.encoder.blocks.layers.{i}.post_attention_layernorm.{last_prefix}",
                ]
            for o_last_prefix, n_last_prefix in zip(["bias", "weight^T"], ["bias", "weight"]):
                aoa_config["aoa_statements"] += [
                    f"vision_tower.encoder.blocks.{i}.mlp.fc0.{o_last_prefix} -> vision_tower.encoder.blocks.layers.{i}.mlp.up_gate_proj.{n_last_prefix}",
                    f"vision_tower.encoder.blocks.{i}.mlp.fc1.{o_last_prefix} -> vision_tower.encoder.blocks.layers.{i}.mlp.down_proj.{n_last_prefix}",
                ]

        for last_prefix in ["bias", "weight"]:
            aoa_config["aoa_statements"] += [
                f"vision_tower.encoder.final_layernorm.{last_prefix} -> vision_tower.encoder.blocks.final_layernorm.norm.{last_prefix}",
                f"mm_projector.pre_norm.{last_prefix} -> vision_tower.encoder.blocks.mm_projector.pre_norm.{last_prefix}",
                f"vision_tower.patch_embed.proj.{last_prefix} -> vision_tower.encoder.blocks.patch_embed.proj.{last_prefix}",
            ]
        for o_last_prefix, n_last_prefix in zip(["bias", "weight^T"], ["bias", "weight"]):
            aoa_config["aoa_statements"] += [
                f"mm_projector.proj.0.{o_last_prefix} -> vision_tower.encoder.blocks.mm_projector.proj.up_gate_proj.{n_last_prefix}",
                f"mm_projector.proj.2.{o_last_prefix} -> vision_tower.encoder.blocks.mm_projector.proj.down_proj.{n_last_prefix}",
            ]

        aoa_config["aoa_statements"] += [
            "vision_tower.patch_embed.pos_emb.weight -> vision_tower.encoder.blocks.patch_embed.pos_emb.weight",
        ]

        # language model

        aoa_config["aoa_statements"] += [
            "language_model.model.embed_tokens.weight -> language_model.model.embedding.embed_tokens.weight",
            "language_model.lm_head.weight -> language_model.model.lm_head.weight ",
            "language_model.model.layers.1.mlp.gate.weight -> language_model.model.layers.1.mlp.gate.weight, src_dtype='bfloat16',dst_dtype='float32'",
        ]
        # MLA
        for layer_id in range(config.text_config.num_hidden_layers):
            for mla_atten in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]:
                aoa_config["aoa_statements"] += [
                    f"language_model.model.layers.{layer_id}.self_attn.{mla_atten}.weight^T -> language_model.model.layers.{layer_id}.self_attn.{mla_atten}.weight",
                ]
        # MLP
        # layer 0
        aoa_config["aoa_statements"] += [
            "language_model.model.layers.0.mlp.down_proj.weight^T -> language_model.model.layers.0.mlp.down_proj.weight",
            "language_model.model.layers.0.mlp.up_proj.weight^T ,language_model.model.layers.0.mlp.gate_proj.weight^T ->  language_model.model.layers.0.mlp.up_gate_proj.weight, axis=1",
        ]
        # layer 1 -> num_hidden_layers
        for layer_id in range(1, config.text_config.num_hidden_layers):
            aoa_config["aoa_statements"] += [
                f"language_model.model.layers.{layer_id}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> language_model.model.layers.{layer_id}.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"language_model.model.layers.{layer_id}.mlp.experts.$EXPERT_ID.up_proj.weight^T, language_model.model.layers.{layer_id}.mlp.experts.$EXPERT_ID.gate_proj.weight^T -> language_model.model.layers.{layer_id}.mlp.experts.$EXPERT_ID.up_gate_proj.weight , axis=1",
                f"language_model.model.layers.{layer_id}.mlp.shared_experts.down_proj.weight^T -> language_model.model.layers.{layer_id}.mlp.shared_experts.down_proj.weight",
                f"language_model.model.layers.{layer_id}.mlp.shared_experts.up_proj.weight^T, language_model.model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight^T -> language_model.model.layers.{layer_id}.mlp.shared_experts.up_gate_proj.weight , axis=1",
            ]

        return aoa_config


class KimiK25Model(KimiK25PretrainedModelFleet):
    config_class = KimiK25Config

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = KimiK25Provider
        model_provider = model_provider_class.from_config(config)

        KimiK25_model = KimiK25ModelDist(model_provider, model_version=config.model_type)
        KimiK25_model._gen_aoa_config = cls._gen_aoa_config

        KimiK25_model.config_to_save = config

        return KimiK25_model


class KimiK25ForConditionalGeneration(KimiK25PretrainedModelFleet):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config_class = KimiK25Config

    def __init__(self, config):
        super().__init__(config)
        self.model_type = config.model_type
        self.model = KimiK25Model(config, have_criterion=False)
        self.criterion = CriterionLayer(config.text_config)

    def state_dict(self, *args, **kwargs):
        # all state_dict should be replace
        state_dict = {}

        if self.model.language_model is not None:
            # Get language_model's state_dict
            lm_state_dict = self.model.language_model.state_dict(*args, **kwargs)

            # Merge language_model parameters into main state_dict
            for key, value in lm_state_dict.items():
                state_dict[key.replace("model.", "language_model.model.")] = value
        if self.model.vision_model is not None:
            # Get vision_model's state_dict
            vision_state_dict = self.model.vision_model.state_dict(*args, **kwargs)

            # Merge vision_model parameters into main state_dict
            for key, value in vision_state_dict.items():
                state_dict[key.replace("model.vision_model.", "vision_tower.encoder.blocks.")] = value
        return state_dict

    def sharded_state_dict(self, *args, **kwargs):
        # all sharded_state_dict should be replace
        sharded_state_dict = {}

        if self.model.language_model is not None:
            # Get language_model's sharded_state_dict
            lm_state_dict = self.model.language_model.sharded_state_dict(*args, **kwargs)

            # Merge language_model parameters into main sharded_state_dict
            for key, value in lm_state_dict.items():
                new_key = key.replace("model.", "language_model.model.")
                value.key = new_key
                sharded_state_dict[new_key] = value

        if self.model.vision_model is not None:
            # Get vision_model's sharded_state_dict
            vision_state_dict = self.model.vision_model.sharded_state_dict(*args, **kwargs)

            # Merge vision_model parameters into main sharded_state_dict
            for key, value in vision_state_dict.items():
                new_key = key.replace("model.vision_model.", "vision_tower.encoder.blocks.")
                value.key = new_key
                sharded_state_dict[new_key] = value

        return sharded_state_dict

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple, KimiK25CausalLMOutputWithPast]:
        """ """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=None,
            labels=labels,
            inference_params=None,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            runtime_gather_output=None,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        logits = outputs

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        return KimiK25CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
        )
