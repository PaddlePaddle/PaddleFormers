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

from dataclasses import dataclass
from typing import Optional, Union

import paddle
from paddlefleet import parallel_state
from paddlefleet.models.kimi_k25 import KimiK25VisionProvider
from paddlefleet.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from paddlefleet.transformer.enums import ModelType
from paddlefleet.transformer.transformer_config import TransformerConfig

from ...nn.criterion.interface import CriterionLayer
from ..cache_utils import Cache
from ..model_utils import PretrainedModel
from .configuration import KimiK25Config
from .modeling_base import (
    DeepseekV3ForCausalLM,
    KimiK25CausalLMOutputWithPast,
    KimiK25PretrainedModel,
)

'''
class KimiK25TextTransformerLayer(TransformerLayer):
    """KimiK25 text model for adapt deepstack process"""

    def forward(
        self,
        dict_args: dict,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        dict_args.pop("dynamic_inference_decode_only", None)
        dict_args.pop("position_ids", None)
        if self.full_recompute:
            hidden_states = dict_args["hidden_states"]
            attention_mask = dict_args.get("attention_mask", None)
            attn_mask_startend_row_indices = dict_args.get("attn_mask_startend_row_indices", None)
            context = dict_args.get("context", None)
            context_mask = dict_args.get("context_mask", None)
            rotary_pos_emb = dict_args.get("rotary_pos_emb", None)
            rotary_pos_cos = dict_args.get("rotary_pos_cos", None)
            rotary_pos_sin = dict_args.get("rotary_pos_sin", None)
            attention_bias = dict_args.get("attention_bias", None)
            packed_seq_params = dict_args.get("packed_seq_params", None)
            deepstack_visual_emb = dict_args.get("deepstack_visual_emb", None)
            visual_pos_masks = dict_args.get("visual_pos_masks", None)

            assert (rotary_pos_sin is None) == (rotary_pos_cos is None)

            if rotary_pos_cos is not None and rotary_pos_sin is not None:
                rotary_pos_cos = rotary_pos_cos.clone()
                rotary_pos_sin = rotary_pos_sin.clone()
                if self.config.apply_rope_fusion:
                    rotary_pos_cos = rotary_pos_cos[0, ...]
                    rotary_pos_sin = rotary_pos_sin[0, ...]
                    if rotary_pos_cos.ndim == 2:
                        rotary_pos_cos = rotary_pos_cos.reshape(
                            [1, rotary_pos_cos.shape[0], 1, rotary_pos_cos.shape[1]]
                        )
                        rotary_pos_sin = rotary_pos_sin.reshape(
                            [1, rotary_pos_sin.shape[0], 1, rotary_pos_sin.shape[1]]
                        )

            outputs = recompute(
                self._forward_impl,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices.clone()  # Clone is necessary!
                if attn_mask_startend_row_indices is not None
                else None,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb.clone() if rotary_pos_emb is not None else None,  # Clone is necessary!
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                deepstack_visual_emb=deepstack_visual_emb,
                visual_pos_masks=visual_pos_masks,
            )
        else:
            outputs = self._forward_impl(**dict_args)

        if isinstance(outputs, tuple):
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

        rst = OrderedDict()
        rst = {"hidden_states": output}
        if context is not None:
            rst["context"] = context
        rst = {**dict_args, **rst}
        return rst

    def _forward_impl(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
        attn_mask_startend_row_indices: paddle.Tensor = None,
        context: paddle.Tensor = None,
        context_mask: paddle.Tensor = None,
        rotary_pos_emb: paddle.Tensor = None,
        rotary_pos_cos: paddle.Tensor = None,
        rotary_pos_sin: paddle.Tensor = None,
        attention_bias: paddle.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        deepstack_visual_emb: list[paddle.Tensor] = None,
        visual_pos_masks: paddle.Tensor = None,
    ):
        hidden_states, context = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self._forward_mlp(hidden_states)
        if deepstack_visual_emb and self.layer_number in range(len(deepstack_visual_emb)):
            # print("process _deepstack_process ",hidden_states.shape,visual_pos_masks.shape,deepstack_visual_emb[self.layer_number].shape)
            hidden_states = self._deepstack_process(
                hidden_states=hidden_states,
                visual_embeds=deepstack_visual_emb[self.layer_number],
                visual_pos_masks=visual_pos_masks,
            )
        if context is not None:
            return hidden_states, context
        return hidden_states

    def _deepstack_process(
        self, hidden_states: paddle.Tensor, visual_pos_masks: paddle.Tensor, visual_embeds: paddle.Tensor
    ):
        # Store original shape and flatten hidden_states to 2D [B*S, D]
        original_shape = hidden_states.shape
        if hidden_states.ndim > 2:
            hidden_states = hidden_states.flatten(start_axis=0, stop_axis=1)

        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        # complicated logic for squential parallelism
        if visual_pos_masks.ndim > 1:
            visual_pos_masks = visual_pos_masks.flatten()

        # This block handles Sequence Parallelism (Row Slicing)
        if visual_pos_masks.shape[0] > hidden_states.shape[0]:
            try:
                from paddle.distributed.fleet import get_hybrid_communicate_group

                hcg = get_hybrid_communicate_group()
                mp_rank = hcg.get_model_parallel_rank()
                mp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                mp_size = visual_pos_masks.shape[0] // hidden_states.shape[0]
                mp_rank = paddle.distributed.get_rank() % mp_size
            total_len = visual_pos_masks.shape[0]
            chunk_size = total_len // mp_size
            start_idx = mp_rank * chunk_size
            end_idx = start_idx + chunk_size
            if start_idx > 0:
                pre_mask = visual_pos_masks[:start_idx]
                visual_offset = paddle.sum(paddle.cast(pre_mask, "int32")).item()
            else:
                visual_offset = 0
            local_mask = visual_pos_masks[start_idx:end_idx]
            local_visual_count = paddle.sum(paddle.cast(local_mask, "int32")).item()

            visual_embeds = visual_embeds[visual_offset : visual_offset + local_visual_count]
            visual_pos_masks = local_mask

        # If TP is enabled, hidden_states has shape [..., Hidden_Dim / TP_Size],
        # but visual_embeds usually has full [Hidden_Dim]. We need to slice visual_embeds column-wise.
        if hidden_states.shape[-1] != visual_embeds.shape[-1]:
            try:
                from paddle.distributed.fleet import get_hybrid_communicate_group

                hcg = get_hybrid_communicate_group()
                tp_rank = hcg.get_model_parallel_rank()
                tp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                # Fallback simple estimation
                tp_size = visual_embeds.shape[-1] // hidden_states.shape[-1]
                tp_rank = paddle.distributed.get_rank() % tp_size

            if tp_size > 1:
                embed_dim = visual_embeds.shape[-1]
                slice_width = embed_dim // tp_size
                start_col = tp_rank * slice_width
                end_col = start_col + slice_width
                visual_embeds = visual_embeds[:, start_col:end_col]

        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this  # 这个操作可能会导致paddle转静态图或推理时出问题，建议使用 scatter

        # [Supplement 3] Restore original shape [B*S, D] -> [B, S, D] if necessary
        if len(original_shape) > 2:
            hidden_states = hidden_states.reshape(original_shape)

        return hidden_states


@dataclass
class KimiK25TextProvider(GPTModelProvider):
    """
    Base config for Qwen3 Models.
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    use_bias: bool = False
    add_qkv_bias: bool = True
    seq_length: int = 4096
    init_method_std: int = 0.02
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: bool | None = False
    rms_norm_eps: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"
    use_qk_norm: bool = True
    specific_layer: type = KimiK25TextTransformerLayer
    max_sequence_length: int = 262144
    multimodal_embedding: bool = False
    _save_to_hf: bool = False
    use_fused_linear_cross_entropy: bool = True
    high_precision_rope: bool = True
    moe_grouped_gemm: bool = True

    n_shared_experts: int = 0
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
        "num_experts": "n_routed_experts",
    }

    def __post_init__(self):
        super().__post_init__()
        self.mrope_section = self.rope_scaling.get("mrope_section", [24, 20, 20])

'''


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
    text_config: TransformerConfig | None = None
    vision_config: KimiK25VisionProvider | None = None

    drop_vision_class_token: bool = False
    vision_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1

    seq_length: int = 1024

    language_model_from_pretrained: str | None = None
    vision_model_from_pretrained: str | None = None

    freeze_langurage_model: bool = False
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False

    def provide(self, tokenizer=None, vp_stage: int | None = None) -> "KimiK25ModelDist":
        self.text_config.scatter_embedding_sequence_parallel = False
        self.text_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.text_config.sequence_parallel = self.sequence_parallel
        self.text_config.context_parallel_size = self.context_parallel_size
        self.vision_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        # self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.text_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size

        if self.encoder_pipeline_model_parallel_size > 0:
            assert self.encoder_pipeline_model_parallel_size == 1, "ViT can only live on 1 pipeline stage."
            self.vision_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            # self.vision_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.text_config.encoder_pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            if self.encoder_tensor_model_parallel_size > 0:
                self.vision_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                # self.vision_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size

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
            # self.vision_projection_config,
        ]:
            for attr in config_attrs:
                setattr(config, attr, getattr(self, attr))

        self.text_config.tp_comm_overlap = self.tp_comm_overlap
        self.vision_config.tp_comm_overlap = False
        # self.vision_projection_config.tp_comm_overlap = False

        vp_stage = vp_stage or 0

        model = KimiK25ModelDist(
            config=self,
            tokenizer=tokenizer,
            pre_process=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() == self.encoder_pipeline_model_parallel_size,
            post_process=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_encoder=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_decoder=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
            vp_stage=vp_stage,
        )

        return model

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.vision_config = KimiK25VisionProvider.from_config(config.vision_config)
        # res.text_config = KimiK25TextProvider.from_config(config.text_config)
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
            # self.language_model = language_transformer_config.provide(
            #     pre_process=pre_process,
            #     post_process=post_process,
            #     vp_stage=vp_stage,
            # )
            # self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
            self.language_model = DeepseekV3ForCausalLM(language_transformer_config)

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

    def get_inputs_embeds(
        self,
        input_ids: paddle.LongTensor,
        pixel_values: paddle.Tensor | None = None,
        grid_thws: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        past_key_values: list[paddle.FloatTensor] | None = None,
        labels: paddle.Tensor | None = None,
    ):
        # 1. Extra the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
            image_features = self._extract_image_features(pixel_values, grid_thws)
            image_features = image_features["hidden_states"]

            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
            )

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
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
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
        input_dict = self.get_inputs_embeds(input_ids, pixel_values, grid_thws, attention_mask, None, labels)
        labels = input_dict.get("labels", labels)

        output = self.language_model(**input_dict)

        return output


class KimiK25PretrainedModelFleet(PretrainedModel):
    config_class = KimiK25Config
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    _no_split_modules = ["KimiK25TextTransformerLayer", "KimiK25VisionTransformerBlock"]
    # _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
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
        pass

    @classmethod
    def _gen_inv_aoa_config(cls, config: KimiK25Config):
        pass


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
        KimiK25_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
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
        # model_provider = KimiK25Provider.from_config(config)
        self.model_type = config.model_type
        self.model = KimiK25Model(
            config, have_criterion=False
        )  # KimiK25Model(model_provider, model_version=config.model_type)
        self.criterion = CriterionLayer(config.text_config)

    def state_dict(self, *args, **kwargs):
        # Override state_dict method to handle language_model's custom state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Remove existing language_model keys to avoid duplicates
        delete_key = []
        for key in state_dict.keys():
            if key.startswith("model.language_model."):
                delete_key.append(key)
        for key in delete_key:
            state_dict.pop(key)
        if self.model.language_model is not None:
            # Get language_model's state_dict
            language_state_dict = self.model.language_model.state_dict(*args, **kwargs)

            # Merge language_model parameters into main state_dict
            for key, value in language_state_dict.items():
                state_dict[key] = value
        return state_dict

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

        return KimiK25CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=None,
        )
