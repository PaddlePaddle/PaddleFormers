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

from dataclasses import dataclass
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlefleet import parallel_state
from paddlefleet.spec_utils import LayerSpec
from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from paddlefleet.transformer.enums import ModelType
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec

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
        
        mlp_gelu_match = re.match(r"mlp(\d+)x_gelu$", self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.groups(1))
            modules = [nn.Linear(self.input_size, self.intermediate_size, bias=True, dtype=self.params_dtype)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(self.intermediate_size, self.hidden_sizes, bias=True, dtype=self.params_dtype)
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
