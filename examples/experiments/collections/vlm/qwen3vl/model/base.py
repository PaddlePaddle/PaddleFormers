# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from abc import Callable
from contextlib import nullcontext

from dataclasses import dataclass
from doctest import REPORT_NDIFF
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlefleet import parallel_state, tensor_parallel
from paddlefleet.packed_seq_params import PackedSeqParams
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.spec_utils import LayerSpec
from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from paddlefleet.transformer.enums import ModelType
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.utils import WrappedTensor, deprecate_inference_params
from paddlefleet.transformer.transformer_block import TransformerBlock, TransformerBlockSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from paddlefleet.models.vision.multimodal_projector import MultimodalProjector as MCoreMultiModalProjector

from PaddleFormers.paddleformers.transformers.gpt_provider import GPTModelProvider

from .layer_spec import get_layer_spec
from .vision import Qwen3VisionModel


MODEL_CONFIG_ATTR = [
    'num_hidden_layers',
    'hidden_size',
    'num_attention_heads',
    'num_key_value_heads',
    'intermediate_size',
    'head_dim',
    'hidden_dropout_prob',
    'attention_dropout',
    'fp32_residual_connection',
    'apply_residual_connection_post_layernorm',
    'init_method_std',
    'rms_norm_eps',
    'use_bias',
    'add_qkv_bias',
    'gated_linear_unit',
    'activation_func',
    'n_routed_experts',
    'rotary_interleaved',
    'sliding_window',
    'normalization',
    'use_qk_norm',
    'position_embedding_type',
    'rotary_base',
    'calculate_per_token_loss',
    'seq_length',
    'share_embeddings_and_output_weights',
    'topk_method',
]


@dataclass
class MultimodalProjectorProvider(TransformerConfig):
    projector_type: str = "mlp2x_gelu"
    layer_spec = None
    input_size: int | None = 1024
    hidden_size: int = 1024
    intermediate_size: int = 1024
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    num_hidden_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!
    
    def provide(self) -> "MCoreMultiModalProjector":
        if self.projector_type.startswith("mcore") and self.layer_spec is None:
            self.use_bias = self.bias
            if self.projector_type == "mcore_mlp":
                self.projector_type = "mlp"
                self.layer_spec = LayerSpec(
                    layer=MLP,
                    sublayers_spec=MLPSublayersSpec(
                        up_gate_proj=ColumnParallelLinear,
                        down_proj=RowParallelLinear,
                    )
                )
                self.layer_spec = self.layer_spec.sublayers_spec
            elif self.projector_type == "mcore_affine":
                self.projector_type = "affine"
                self.layer_spec = MLPSublayersSpec(up_gate_proj=ColumnParallelLinear, down_proj=None)
            else:
                raise NotImplementedError(f"Not supported projector type: {self.projector_type}")
            
            return MCoreMultiModalProjector(
                self,
                self.layer_spec,
                projector_type=self.projector_type,
                input_size=self.input_size,
            )
        
        if self.projector_type == "vila_downsample_mlp":
            model = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(self.input_size * 4, dtype=self.params_dtype),
                nn.Linear(self.input_size * 4, self.hidden_size, bias=True, dtype=self.params_dtype),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True, dtype=self.params_dtype),
            )
            from types import MethodType
            
            model.set_input_tensor = MethodType(set_input_tensor, model)
            return model
        
        mlp_gelu_match = REPORT_NDIFF.match(r"mlp(\d+)x_gelu$", self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.groups(1))
            modules = [nn.Linear(self.input_size, self.intermediate_size, bias=True, dtype=self.params_dtype)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(self.intermediate_size, self.hidden_size, bias=True, dtype=self.params_dtype)
                )
            model = nn.Sequential(*modules)
            from types import MethodType
            
            model.set_input_tensor = MethodType(set_input_tensor, model)
        else:
            raise NotImplementedError(f"Not supported projector type: {self.projector_type}")
        
        return model


@dataclass
class Qwen3Provider(GPTModelProvider):
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


def qwen3vl_data_step(dataloader_iter) -> dict[str, paddle.Tensor]:
    from paddlefleet import parallel_state
    
    batch = next(dataloader_iter)
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 1:
        _batch = batch[0]
    else:
        _batch = batch
    
    required_keys = {"input_ids", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    
    if parallel_state.is_pipeline_first_stage():
        required_keys.add("position_ids")
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))
    
    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    output = _batch
    return _batch


def qwen3vl_forward_step(model, batch) -> paddle.Tensor:
    forward_args = {
        "input_ids": batch["input_ids"],
        "pixel_values": batch["pixel_values"],
        "image_grid_thw": batch["image_grid_thw"],
        "pixel_values_videos": batch["pixel_values_videos"],
        "video_grid_thw": batch["video_grid_thw"],
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
    }
    return model(**forward_args)


def set_input_tensor(self, tensor):
    pass


@dataclass
class Qwen3VLVisionProvider(TransformerConfig):
    """Qwen3VL Vidion Model Configuration."""
    patch_size: int = 16,
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304,
    embed_dim: int = 1152,
    hidden_size: int = 1152,
    out_hidden_size: int = 4096,
    in_channels: int = 3,
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = True
    activation_func: Callable = F.gelu
    num_key_value_heads: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = None
    model_version: str = "qwen3_vl"
    img_h: int = 336
    img_w: int = 336
    add_class_token: bool = False
    class_token_len: int = 1

    
    def provide(self) -> "Qwen3VisionModel":
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, LayerSpec):
            transformer_layer_spec = get_layer_spec(is_vit=True, normalization=self.normalization)
        
        model = Qwen3VisionModel(
            transformer_config=self,
            transformer_layer_spec=transformer_layer_spec,
        )
        
        return model


@dataclass
class Qwen3VLProvider(TransformerConfig):
    """Qwen3VL model base configuration."""
    
    language_transformer_config: Qwen3Provider | None = None
    vision_transformer_config: Qwen3VLVisionProvider | None = None
    vision_projection_config: MultimodalProjectorProvider | None = None
    
    drop_vision_class_token: bool = False
    vision_feature_layer: int = -2
    
    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1
    num_hidden_layers: int = 1
    num_attention_heads: int = 16
    
    seq_length: int = 1024
    
    language_model_from_pretrained: str | None = None
    vision_model_from_pretrained: str | None = None
    vision_projection_from_preatrained: str | None = None
    
    freeze_langurage_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    
    forward_step_fn: Callable = qwen3vl_forward_step
    data_step_fn: Callable = qwen3vl_data_step
    
    def provide(self, tokenizer=None, vp_stage: int | None = None) -> "McoreQwen3VLModel":
        self.language_transformer_config.scatter_embedding_sequence_parallel = False
        self.language_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.sequence_parallel = self.sequence_parallel
        self.language_transformer_config.context_parallel_size = self.context_parallel_size
        self.vision_transformer_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_transformer_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        
        if self.encoder_pipeline_model_parallel_size > 0:
            assert self.encoder_pipeline_model_parallel_size == 1, "ViT can only live on 1 pipeline stage."
            self.vision_transformer_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.vision_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.language_transformer_config.encoder_pipeline_model_parallel_size = (
                self.encoder_pipeline_model_parallel_size
            )
            if self.encoder_tensor_model_parallel_size > 0:
                self.vision_transformer_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                self.vision_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
        
        config_attrs = [
            "cross_entropy_loss_function",
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
            self.language_transformer_config,
            self.vision_transformer_config,
            self.vision_projection_config,
        ]:
            for attr in config_attrs:
                setattr(config, attr, getattr(self, attr))
        
        self.language_transformer_config.tp_comm_overlap = self.tp_comm_overlap
        self.vision_transformer_config.tp_comm_overlap = False
        self.vision_projection_config.tp_comm_overlap = False
        
        vp_stage = vp_stage or 0
        
        model = MCoreQwen3VLModel(
            config,
            tokenizer=tokenizer,
            pre_process=parallel_state.is_pipline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() == self.encoder_pipeline_model_parallel_size,
            post_process=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_encoder=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_decoder=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
            vp_stage=vp_stage,
        )
        
        return model
    
    def __post_init__(self):
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))
            self.language_transformer_config.position_embedding_type = "mrope"
            self.language_transformer_config.mrope_section = [24, 20, 20]


class MCoreQwen3VLModel(MCoreLLaVAModel):
    """Qwen3VL Model Base Model Class."""
    
    def __init__(
        self,
        config: Qwen3VLProvider,
        tokenizer=None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
        vp_stage: int | None = None,
    ) -> None:
        super(MCoreLLaVAModel, self).__init__(config=config)
        
        language_transformer_config = config.language_transformer_config
        vision_transformer_config = config.vision_transformer_config
        vision_projection_config = config.vision_projection_config
        self.model_version = vision_transformer_config.model_version
        assert self.model_version is not None     

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage
        
        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None
        
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        assert not (self.sequence_parallel_lm or self.context_parallel_lm > 1), \
            f"qwenvl donnot support sequence parallel {self.sequence_parallel_lm} "\
            f"or context parallel {self.context_parallel_lm}"
        self.share_embeddings_and_output_weights = False
        
        if self.add_decoder:
            self.language_model = language_transformer_config.provide(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
            self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
        
        if add_encoder:
            self.vision_model = vision_transformer_config.provide()
            self.vision_projection = vision_projection_config.provide()
            self._drop_vision_class_token = drop_vision_class_token
        
        self.freeze(
            freeze_language_model=config.freeze_langurage_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )
        
        self.model_type = ModelType.encoder_or_decoder
        
        self._img_seq_len = get_image_sequence_length(
            img_h=vision_transformer_config.img_h,
            img_w=vision_transformer_config.img_w,
            patch_dim=vision_transformer_config.patch_size,
            add_class_token=not drop_vision_class_token,
            class_token_len=vision_transformer_config.class_token_len,
        )
    
    def get_rope_index(
        self,
        input_ids: paddle.LongTensor | None = None,
        image_grid_thw: paddle.LongTensor | None = None,
        video_grid_thw: paddle.LongTensor | None = None,
        attention_mask: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = paddle.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_transformer_config.spatial_merge_size
        
        # TODO when implemented data file.
        image_token_id = IMAGE_TOKEN_INDEX
        video_token_id = VIDEO_TOKEN_INDEX
        vision_start_token_id = 151652
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = paddle.ones_like(total_input_ids)
            position_ids = paddle.ones(
                [3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    
                    t_index = paddle.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = paddle.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = paddle.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = paddle.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1])
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas
    def get_video_features(
        self, pixel_values_videos: paddle.FloatTensor, video_grid_thw: paddle.LongTensor | None = None,
    ):
        return self.get_image_features(pixel_values_videos, video_grid_thw)
    
    def get_image_features(
        self, pixel_values: paddle.FloatTensor, image_grid_thw: paddle.LongTensor | None = None
    ):
        pixel_values = pixel_values.to(self.vision_model._dtype)
        image_embeds, deepstack_image_embeds = self.vision_model(pixel_values, image_grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size ** 2).tolist()
        image_embeds = paddle.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds
    
    def get_placehodler_mask(
        self,
        input_ids: paddle.LongTensor,
        inputs_embeds: paddle.FloatTensor,
        image_features: paddle.FloatTensor | None = None,
        video_features: paddle.FloatTensor | None = None,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                paddle.to_tensor(IMAGE_TOKEN_INDEX, dtype="long")
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                paddle.to_tensor(VIDEO_TOKEN_INDEX, dtype="long")
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == IMAGE_TOKEN_INDEX
            special_video_mask = input_ids == VIDEO_TOKEN_INDEX
        
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features: {image_features.shape[0]}"
            )
        
        n_video_tokens = special_video_mask.sum()
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.LongTensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        inference_params = None,
        pixel_values: paddle.Tensor | None = None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        runtime_gather_output: bool | None = None,
    ) -> paddle.Tensor:
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if self.add_encoder and pixel_values is not None:
            pixel_values.to(self.vision_model.parameters()[0].dtype)
            if self.config.freeze_vision_model:
                with paddle.no_grad():
                    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            else:
                image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        
        if self.add_encoder and pixel_values_videos is not None:
            pixel_values_videos.to(next(self.vision_model.parameters()).dtype)
            if self.config.freeze_vision_model:
                with paddle.no_grad():
                    video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            else:
                video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            

        
        language_embeds = None
        
        language_seq_len = input_ids.shape[1]
        if language_seq_len > self._language_max_sequence_length:
            input_ids = input_ids[:, : self._language_max_sequence_length]
            if position_ids is not None:
                position_ids = position_ids[:, : self._language_max_sequence_length]
            
            if labels is not None and labels.shape[1] > self._language_max_sequence_length:
                labels = labels[:, : self._language_max_sequence_length]
                loss_mask = loss_mask[:, : self._language_max_sequence_length]
        
        if self._language_is_pipeline_parallel and language_seq_len < self._language_max_sequence_length:
            padded_seq_len = self._language_max_sequence_length - language_seq_len
            input_ids = F.pad(input_ids, (0, padded_seq_len))
            if position_ids is not None:
                position_ids = F.pad(position_ids, (0, padded_seq_len))
        
        if position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
        
        if self.pre_process:

            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            input_ids_text = input_ids.clone()
            # MultiModal Token indices are assumed to be values
            input_ids_text[input_ids_text < 0] = 0

            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=None
            )  # [decoder_seq_len, b, h_language]

            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]
        
        image_mask, video_mask = self.get_placehodler_mask(input_ids, language_embeddings, image_embeds, video_embeds)
        input_embeds = language_embeds.masked_scatter(image_mask)
        input_embeds = input_embeds.masked_scatter(video_mask)
        
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds
        
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=input_embeds,
            labels=labels,
            runtime_gather_output=runtime_gather_output,
        )
        
        if labels is None or loss_mask is None:
            return output
        else:
            return output, loss_mask.contiguous()
    
    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])


class Qwen3VLTextTransformerBlock(TransformerBlock):
    def __init__(
        config: TransformerConfig,
        spec: TransformerBlockSublayersSpec | LayerSpec,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(
            config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
    
    def forward(
        self,
        hidden_states: paddle.Tensor | WrappedTensor,
        attention_mask: paddle.Tensor | None,
        context: paddle.Tensor | None = None,
        context_mask: paddle.Tensor | None = None,
        rotary_pos_emb: paddle.Tensor | None = None,
        rotaty_pos_cos: paddle.Tensor | None = None,
        rotary_pos_sin: paddle.Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        inference_context = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: paddle.Tensor | None = None,
        visual_pos_masks: paddle.Tensor | None = None,
        deepstack_visual_embeds: paddle.Tensor | None = None,
        *,
        inference_params = None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)
        
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        
        if not self.pre_process:
            hidden_states = self.input_tensor
        
        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
    
        # hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        print("fleet vision 0 hidden_states", hidden_states._md5sum())
        
        with rng_context:
            if self.recompute_granularity == "full" and self.training:
                pass
            else:
                packed_seq_params_now = packed_seq_params
                for l_no, layer in self.layers:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotaty_pos_cos=rotaty_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params_now,
                    )
                    if deepstack_visual_embeds is not None and l_no in range(len(deepstack_visual_embeds)):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[l_no],
                        )
                    print(f"fleet vision {l_no} hidden_states", hidden_states._md5sum())
                if self.norm is not None:
                    hidden_states = self.norm(hidden_states)
                
                return hidden_states
    
    def _deepstack_process(
        self, hidden_states: paddle.Tensor, visual_pos_masks: paddle.Tensor, visual_embeds: paddle.Tensor
    ):
        visual_embeds = visual_embeds.to(hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states
    
    def _checkpoint_forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        context: paddle.Tensor,
        context_mask: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor,
        attentioin_bias: paddle.Tensor,
        packed_seq_params: PackedSeqParams,
        deepstack_visual_embeds: paddle.Tensor,
        visual_pos_masks: paddle.Tensor,
    ):
        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
                deepstack_visual_embeds, visual_pos_masks
            ):
                for index in range(start, end):
                    packed_seq_params_now = packed_seq_params
                    layer = self._get_layer(index)
                    
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attentioin_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attentioin_bias=attentioin_bias,
                        inference_params=None,
                        packed_seq_params=packed_seq_params_now,
                    )
                    if deepstack_visual_embeds is not None and index in range(len(deepstack_visual_embeds)):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[index],
                        )
                return hidden_states, context
            return custom_forward
        
        def checkpoint_handler(forward_func):
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    deepstack_visual_embeds,
                    visual_pos_masks
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    deepstack_visual_embeds,
                    visual_pos_masks,
                )
        
        if self.config.recompute_method == "uniform":
            layer_index = 0
            while layer_index < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_index, layer_index + self.config.recompute_num_layers),
                )
        
        elif self.config.recompute_method == "block":
            recompute_skip_num_layers = 0
            for layer_index in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_index >= recompute_skip_num_layers
                    and layer_index < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_index, layer_index + 1))
                else:
                    hidden_states, context = custom(layer_index, layer_index + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                    )
        
        else:
            raise ValueError(f"Invalid activation recompute method: {self.config.recompute_method}")


class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(
        self, config: TransformerConfig,
    ):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        
        self.config = config
        
        self.rope_type = self.config.rope_parameters["rope_type"]
        


class Qwen3VLTextModel(FleetLayer):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: LayerSpec,
    ):
        super.__init__(config)
        self.vocab_size = config.vocab_size
        self.padding_idx = PAD_TOKEN_INDEX
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # self.rotary_emb =
        self.decoder = Qwen3VLTextTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )
