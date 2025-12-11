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
import contextlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Literal, Union
import paddle
import paddle.distributed
from paddle import nn
import paddle.nn.functional as F

from paddlefleet import parallel_state as ps
from paddlefleet.transformer.enums import ModelType,AttnMaskType
from paddlefleet.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from paddlefleet.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from paddlefleet.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from paddlefleet.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel

from paddlefleet.tensor_parallel.layers import ColumnParallelLinear,RowParallelLinear
from paddlefleet.spec_utils import LayerSpec
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.mlp import MLP,MLPSublayersSpec
from paddleformers.transformers.gpt_provider import GPTModelProvider

from ..data.multimodal_tokens import IGNORE_INDEX, IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX
from .vision import Qwen2VisionModel, Qwen25VisionModel
from .layer_spec import get_layer_spec

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
    
def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    """Get image sequence length given image size, patch size, and class token."""
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)

@dataclass
class MultimodalProjectorProvider(TransformerConfig):
    """
    For MLP, fc1 in shape of input_size, intermediate_size, fc2 in shape of intermediate_size, hidden_size
    """

    projector_type: str = "mlp2x_gelu"
    layer_spec = None
    input_size: Optional[int] = 1024
    hidden_size: int = 1024
    intermediate_size: int = 1024
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    num_hidden_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!

    def provide(self) -> "MCoreMultimodalProjector":
        # pylint: disable=C0115,C0116
        if self.projector_type.startswith("mcore") and self.layer_spec is None:
            self.use_bias = self.bias
            if self.projector_type == "mcore_mlp":
                self.projector_type = "mlp"  # strip "mcore_" for mcore init
                self.layer_spec = LayerSpec(
                    layer=MLP,
                    sublayers_spec=MLPSublayersSpec(
                        up_gate_proj=ColumnParallelLinear,
                        down_proj=RowParallelLinear,
                    ),
                )
                self.layer_spec = self.layer_spec.sublayers_spec
            elif self.projector_type == "mcore_affine":
                self.projector_type = "affine"  # strip "mcore_" for mcore init
                self.layer_spec = MLPSublayersSpec(up_gate_proj=ColumnParallelLinear, down_proj=None)
            else:
                raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

            return MCoreMultimodalProjector(
                self,
                self.layer_spec,
                projector_type=self.projector_type,
                input_size=self.input_size,
            )

        # if using vila's downsample + mlp projector
        if self.projector_type == "vila_downsample_mlp":
            model = paddle.nn.Sequential(
                DownSampleBlock(),
                paddle.nn.LayerNorm(self.input_size * 4, dtype=self.params_dtype),
                paddle.nn.Linear(self.input_size * 4, self.hidden_size, bias=True, dtype=self.params_dtype),
                paddle.nn.GELU(),
                paddle.nn.Linear(self.hidden_size, self.hidden_size, bias=True, dtype=self.params_dtype),
            )
            from types import MethodType

            model.set_input_tensor = MethodType(set_input_tensor, model)
            return model

        # e.g. "mlp2x_gelu"
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [paddle.nn.Linear(self.input_size, self.intermediate_size, bias=True, dtype=self.params_dtype)]
            for _ in range(1, mlp_depth):
                modules.append(paddle.nn.GELU())
                modules.append(
                    paddle.nn.Linear(self.intermediate_size, self.hidden_size, bias=True, dtype=self.params_dtype)
                )
            model = paddle.nn.Sequential(*modules)
            from types import MethodType

            model.set_input_tensor = MethodType(set_input_tensor, model)
        else:
            raise NotImplementedError(f"Not supported projector type `{self.projector_type}`")

        return model
@dataclass
class Qwen2Provider(GPTModelProvider):
    """
    Base config for Qwen 2 Models
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
    share_embeddings_and_output_weights: Optional[bool] = False
    rms_norm_eps: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"

def qwen2vl_data_step(dataloader_iter, model_version) -> Dict[str, paddle.Tensor]:
    """Qwen2VL Data Step"""
    from paddlefleet import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/experiments.collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    batch = next(dataloader_iter)
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    if model_version == "qwen2-vl":
        required_keys.update(("input_ids", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"))
    elif model_version == "qwen25-vl":
        required_keys.update(
            (
                "input_ids",
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "second_per_grid_ts",
            )
        )
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("position_ids",))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(
            (
                "labels",
                "loss_mask",
            )
        )

    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    # slice batch along sequence dimension for context parallelism
    output = _batch
    return output


def qwen2vl_forward_step(model, batch) -> paddle.Tensor:
    # pylint: disable=C0115,C0116
    forward_args = {
        "input_ids": batch["input_ids"],
        "pixel_values": batch.get("pixel_values", None),
        "image_grid_thw": batch.get("image_grid_thw", None),
        "pixel_values_videos": batch.get("pixel_values_videos", None),
        "video_grid_thw": batch.get("video_grid_thw", None),
        "second_per_grid_ts": batch.get("second_per_grid_ts", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
    }
    return model(**forward_args)


def set_input_tensor(self, tensor):
    # pylint: disable=C0115,C0116
    pass


@dataclass
class Qwen2VLVisionProvider(TransformerConfig):
    """Qwen2VL Vision Model Config"""

    add_class_token: bool = False
    class_token_len: int = 1
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    embed_dim: int = 1280
    hidden_size: int = 1280
    spatial_merge_size: int = 2
    spatial_patch_size: int = 14
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 5120  # 1280 * 4
    gated_linear_unit: bool = True
    activation_func: Callable = paddle.nn.functional.gelu
    head_dim: int = 80
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
    model_version: str = "qwen2-vl"

    def provide(self) -> "Qwen2VisionModel":
        # pylint: disable=C0115,C0116
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, LayerSpec):
            transformer_layer_spec = get_layer_spec(is_vit=True,normalization=self.normalization)
            # raise ValueError(f"transformer_layer_spec is not a LayerSpec {transformer_layer_spec}")

        model = Qwen2VisionModel(
            self,
            transformer_layer_spec,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size,
            spatial_patch_size=self.spatial_patch_size,
            img_h=self.img_h,
            img_w=self.img_w,
        )

        return model


@dataclass
class Qwen25VLVisionProvider(TransformerConfig):
    """Qwen2.5VL Vision Model Config"""

    add_class_token: bool = False
    class_token_len: int = 1
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    embed_dim: int = 1280
    hidden_size: int = 1280
    spatial_merge_size: int = 2
    spatial_patch_size: int = 14
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 3420
    gated_linear_unit: bool = True
    activation_func: Callable = paddle.nn.functional.silu  # Qwen 2.5-VL uses swiGLU as activation function
    head_dim: int = 80
    num_key_value_heads: int = 16
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'RMSNorm'  # set the normalization to RMSNorm for Qwen2.5-VL
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = None
    fullatt_block_indexes: List[int] = field(default_factory=lambda: [7, 15, 23, 31])
    model_version: str = "qwen25-vl"
    fp8:bool = False
    apply_vision_rope: bool = True

    def provide(self) -> "Qwen25VisionModel":
        # pylint: disable=C0115,C0116
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, LayerSpec):
            transformer_layer_spec = get_layer_spec(is_vit=True,normalization=self.normalization)
            # raise ValueError(f"transformer_layer_spec is not a LayerSpec {transformer_layer_spec}")

        model = Qwen25VisionModel(
            self,
            transformer_layer_spec,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size,
            spatial_patch_size=self.spatial_patch_size,
            img_h=self.img_h,
            img_w=self.img_w,
        )

        return model


@dataclass
class Qwen2VLProvider(TransformerConfig):
    """Qwen2VL Model Base Config"""

    language_transformer_config: Optional[Qwen2Provider] = None
    vision_transformer_config: Optional[Qwen2VLVisionProvider | Qwen25VLVisionProvider] = None
    vision_projection_config: Optional[MultimodalProjectorProvider] = None

    drop_vision_class_token: bool = False
    vision_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1
    num_hidden_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    seq_length: int = 1024

    language_model_from_pretrained: Optional[str] = None
    vision_model_from_pretrained: Optional[str] = None  # TODO
    vision_projection_from_pretrained: Optional[str] = None  # TODO

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = qwen2vl_forward_step
    data_step_fn: Callable = qwen2vl_data_step

    def provide(self, tokenizer=None, vp_stage: Optional[int] = None) -> "MCoreQwen2VLModel":
        # pylint: disable=C0115,C0116
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

        # Define common config attributes to set
        config_attrs = [
            'cross_entropy_loss_fusion',
            'gradient_accumulation_fusion',
            'bias_activation_fusion',
            'bias_dropout_fusion',
            'masked_softmax_fusion',
            'attention_softmax_in_fp32',
            'apply_rope_fusion',
            'overlap_p2p_comm',
            'batch_p2p_comm',
        ]

        # Set common configs for all transformer components
        for config in [
            self.language_transformer_config,
            self.vision_transformer_config,
            self.vision_projection_config,
        ]:
            for attr in config_attrs:
                setattr(config, attr, getattr(self, attr))

        # Set tp_comm_overlap only for language transformer only
        self.language_transformer_config.tp_comm_overlap = self.tp_comm_overlap
        self.vision_transformer_config.tp_comm_overlap = False
        self.vision_projection_config.tp_comm_overlap = False

        # During fake lightning initialization, pass 0 to bypass the assertion that vp_stage must be
        # non-None when using virtual pipeline model parallelism
        vp_stage = vp_stage or 0
        model = MCoreQwen2VLModel(
            config=self,
            tokenizer=tokenizer,
            pre_process=ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
            or ps.get_pipeline_model_parallel_rank() == self.encoder_pipeline_model_parallel_size,
            post_process=ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_encoder=ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_decoder=ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
            or ps.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
            vp_stage=vp_stage,
        )

        return model

    def __post_init__(self):
        # pylint: disable=C0115,C0116
        if self.language_transformer_config is not None:
            for attr in MODEL_CONFIG_ATTR:
                setattr(self, attr, getattr(self.language_transformer_config, attr))
            # must have this setting to use MultimodalRotaryEmbedding in GPTModel
            self.language_transformer_config.position_embedding_type = "mrope"
            # See Qwen2-VL 2B/7B/72B share the same mrope_section config, see below for details:
            # https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/blob/main/config.json
            # https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json
            # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/config.json
            self.language_transformer_config.mrope_section = [16, 24, 24]

class MCoreQwen2VLModel(MCoreLLaVAModel):
    """Qwen2VL Model Base Model Class"""

    def __init__(
        self,
        config: Qwen2VLProvider,
        tokenizer = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
        vp_stage: Optional[int] = None,
    ) -> None:
        # pylint: disable=C0115,C0116
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
        assert not (self.sequence_parallel_lm or self.context_parallel_lm>1),f"qwenvl donnot support sequence parallel {self.sequence_parallel_lm} or context parallel {self.context_parallel_lm}"
        self.share_embeddings_and_output_weights = False

        if self.add_decoder:
            self.language_model = language_transformer_config.provide(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
            # self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
            # self._language_max_sequence_length = self.language_model.max_sequence_length
            self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1

        if self.add_encoder:
            self.vision_model = vision_transformer_config.provide()
            self.vision_projection = vision_projection_config.provide()
            self._drop_vision_class_token = drop_vision_class_token

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.

        self._img_seq_len = get_image_sequence_length(
            img_h=vision_transformer_config.img_h,
            img_w=vision_transformer_config.img_w,
            patch_dim=vision_transformer_config.patch_dim,
            add_class_token=not drop_vision_class_token,
            class_token_len=vision_transformer_config.class_token_len,
        )

    def get_rope_index(
        self,
        input_ids= None,
        image_grid_thw = None,
        video_grid_thw = None,
        second_per_grid_ts = None,
        attention_mask = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Qwen2-VL and Qwen25-VL has differnt type:
            Qwen2-VL Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
            Qwen25-VL Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each
                    second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens"
                    are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens
                    per second. So each second of the video will be represented with 25 separate time points. It
                    essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as
                    tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each
                    temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`paddle.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
            image_grid_thw (`paddle.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`paddle.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`paddle.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`paddle.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`paddle.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = 2  # self.config.vision_config.spatial_merge_size
        image_token_id = IMAGE_TOKEN_INDEX
        video_token_id = VIDEO_TOKEN_INDEX
        vision_start_token_id = 151652  # self.config.vision_start_token_id
        tokens_per_second = 2
        if second_per_grid_ts is not None:
            second_per_grid_ts = second_per_grid_ts.cpu()

        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids.clone()
            if attention_mask is None:
                attention_mask = paddle.ones_like(total_input_ids)
            position_ids = paddle.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype,
            )
            image_index, video_index = 0, 0
            for i, input_ids_item in enumerate(total_input_ids):
                _input_ids = input_ids_item[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.argwhere(_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = _input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = _input_ids.tolist()
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
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if self.model_version == "qwen25-vl":
                            if second_per_grid_ts is not None:
                                second_per_grid_t = second_per_grid_ts[video_index]
                            else:
                                second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    if self.model_version == "qwen2-vl":
                        t_index = paddle.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    elif self.model_version == "qwen25-vl":
                        range_tensor = paddle.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()
                    h_index = paddle.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = paddle.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = paddle.concat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.place)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.place)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1], place=input_ids.place)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1],
                    place=input_ids.place,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids = None,
        loss_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        inference_params = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos= None,
        image_grid_thw= None,
        video_grid_thw = None,
        runtime_gather_output: Optional[bool] = None,
        second_per_grid_ts = None,
    ) -> paddle.Tensor:
        """Forward function of the Qwen2VL model.

        Args:
            input_ids (paddle.Tensor): input text ids [batch, decoder_seq_len].
            attention_mask (paddle.Tensor): Attention mask for the language model [batch, 1, combined_seq_len,
            combined_seq_len].
            position_ids (paddle.Tensor): input text position ids [batch, decoder_seq_len].
            loss_mask (paddle.Tensor): Text loss mask [batch, decoder_seq_len].
            labels (paddle.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            pixel_values (paddle.Tensor): input image of shape [images_total_patches,
            num_channels * temporal_size * patch_size * patch_size].
            pixel_values_videos (paddle.Tensor): input video of shape [videos_total_patches,
            num_channels * temporal_size * patch_size * patch_size].
            image_grid_thw (paddle.Tensor): The temporal, height and width of feature shape of each image.
            Shape [num_images, 3].
            video_grid_thw (paddle.Tensor): The temporal, height and width of feature shape of each video.
            Shape [num_videos, 3].
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        Returns:
            output (paddle.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (paddle.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )

        has_images = pixel_values is not None
        has_videos = pixel_values_videos is not None

        image_embeddings = None
        if use_inference_kv_cache:
            # If running inference, we can skip media token computation if they were computed already earlier
            # for this sample.
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = None
        elif self.add_encoder and has_images:
            pixel_values = pixel_values.cast(self.vision_model.parameters()[0].dtype)
            if self.config.freeze_vision_model:
                with paddle.no_grad():
                    image_embeddings = self.vision_model(
                        pixel_values, grid_thw=image_grid_thw
                    )  # [bs, img_seq_len, h_vision]
            else:
                image_embeddings = self.vision_model(
                    pixel_values, grid_thw=image_grid_thw
                )  # [bs, img_seq_len, h_vision]

            window_index = self.vision_model.window_index if self.model_version == "qwen25-vl" else None

            # if self._drop_vision_class_token:
            #     class_token_len = getattr(self.vision_model, "class_token_len", 1)
            #     image_embeddings = image_embeddings[:, class_token_len:, :]
            #     if self.model_version == "qwen25-vl":
            #         window_index = [idx - class_token_len for idx in window_index if idx >= class_token_len]
            #     else:
            #         window_index = None

            image_embeddings = self.vision_projection(image_embeddings)
            if self.model_version == "qwen25-vl":
                reverse_indices = paddle.argsort(window_index)
                image_embeddings = image_embeddings[reverse_indices, :]
        else:
            image_embeddings = self.encoder_hidden_state

        video_embeddings = None
        if self.add_encoder and has_videos:
            pixel_values_videos = pixel_values_videos.to(next(self.vision_model.parameters()).dtype)
            if self.config.freeze_vision_model:
                with paddle.no_grad():
                    video_embeddings = self.vision_model(
                        pixel_values_videos, grid_thw=video_grid_thw
                    )  # [bs, img_seq_len, h_vision]
            else:
                video_embeddings = self.vision_model(
                    pixel_values_videos, grid_thw=video_grid_thw
                )  # [bs, img_seq_len, h_vision]

            video_embeddings = self.vision_projection(video_embeddings)
        if not self.add_decoder:
            return image_embeddings

        # language_embeddings is a container for text, image and video embeddings; to feed to decoder
        language_embeddings = None

        language_seq_len = input_ids.shape[1]
        # chunk if input seq_len > _language_max_sequence_length
        if language_seq_len > self._language_max_sequence_length:
            input_ids = input_ids[:, : self._language_max_sequence_length]
            if position_ids is not None:
                position_ids = position_ids[:, :, : self._language_max_sequence_length]

            if labels is not None and labels.shape[1] > self._language_max_sequence_length:
                labels = labels[:, : self._language_max_sequence_length]
                loss_mask = loss_mask[:, : self._language_max_sequence_length]

        # Pipeline parallel expects fixed input size. Check if we need to pad.
        if self._language_is_pipeline_parallel and language_seq_len < self._language_max_sequence_length:
            padded_seq_len = self._language_max_sequence_length - language_seq_len
            input_ids = paddle.nn.functional.pad(input_ids, (0, padded_seq_len))
            if position_ids is not None:
                position_ids = paddle.nn.functional.pad(position_ids, (0, padded_seq_len))

        if position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask
            )

        # Create the language_embeddings (if this is the first language model stage).
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
        # # Preprocess input, labels and loss mask.
        # combined_embeddings, final_labels, final_loss_mask, final_attention_mask = self._preprocess_data(
        #     input_ids,
        #     loss_mask=loss_mask,
        #     labels=labels,
        #     language_embeddings=language_embeddings,
        #     image_embeddings=image_embeddings,
        #     video_embeddings=video_embeddings,
        #     attention_mask=attention_mask,
        # )  # [decoder_seq_len, b, h_language], [b, decoder_seq_len], [b, decoder_seq_len]
        combined_embeddings = self.combine_embedding(input_ids,language_embeddings,image_embeddings,video_embeddings)

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            runtime_gather_output=runtime_gather_output,
        )  # output shape: [batch_size, seq length, vocab_size]

        if labels is None or loss_mask is None:
            return output
        else:
            return output, loss_mask.contiguous()

    def get_placeholder_mask(
        self,
        input_ids: paddle.Tensor,
        inputs_embeds: paddle.Tensor,
        image_features: Optional[paddle.Tensor] = None,
        video_features: Optional[paddle.Tensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.language_model.embedding((
                paddle.to_tensor(IMAGE_TOKEN_INDEX, dtype="int64")
            ))
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.language_model.embedding((
                paddle.to_tensor(VIDEO_TOKEN_INDEX, dtype="int64")
            ))
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == IMAGE_TOKEN_INDEX
            special_video_mask = input_ids == VIDEO_TOKEN_INDEX
        special_image_mask=special_image_mask.transpose(1,0).unsqueeze(-1)
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.expand_as(inputs_embeds)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )
        special_video_mask=special_video_mask.transpose(1,0).unsqueeze(-1)
        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.expand_as(inputs_embeds)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def combine_embedding(
            self,
            input_ids: paddle.Tensor,
            language_embeddings: Optional[paddle.Tensor] = None,
            image_embeddings: Optional[paddle.Tensor] = None,
            video_embeddings: Optional[paddle.Tensor] = None,
    ):
        
        if image_embeddings is not None:
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=language_embeddings, image_features=image_embeddings
            )
            combine_embeds = language_embeddings.masked_scatter(image_mask, image_embeddings)

        if video_embeddings is not None:
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=language_embeddings, video_features=video_embeddings
            )
            combine_embeds  = language_embeddings.masked_scatter(video_mask, video_embeddings)

        return combine_embeds
    # override _preprocess_data() in megatron-lm/megatron/core/models/multimodal/llava_model.py
    def _preprocess_data(
        self,
        input_ids: paddle.Tensor,
        loss_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        language_embeddings: Optional[paddle.Tensor] = None,
        image_embeddings: Optional[paddle.Tensor] = None,
        video_embeddings: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        use_inference_kv_cache: Optional[bool] = False,
        attention_mask: Optional[paddle.Tensor] = None,
    ):
        """
        MCoreQwen2VLModel uses its own version of _preprocess_data instead of MCoreLLaVAModel's (in
        megatron-lm/megatron/core/models/multimodal/llava_model.py)

        This function handles several data preprocess requirements:
            - merge image and/or video embeddings into language embedding
            - padding inputs variables (e.g. labels/loss masks) for pipeline_parallel case
            - truncate inputs variables (e.g. labels/loss masks) if exceeding max seq length

        This function won't shift labels as forward() and _preprocess_data() in MCoreQwen2VLModel
        expect labels from input arguments already handle this shift.

        About merging image/video embeddings: language_embeddings may include num of imgage_token
        placeholders, and this function will put each imgage_token from image_embeddings into
        placeholder within language_embeddings(1:1 mapping), when image_embeddings/video_embeddings
        is available and it's the 1st pipeline_parallel stage
        """

        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return None, None, None, None

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels, attention_mask

        # img_seq_len = self._img_seq_len
        batch_size, language_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        has_images = image_embeddings is not None
        has_videos = video_embeddings is not None

        #
        # Create the final input embedding (if this is the first language model stage).
        #
        final_embedding = None
        if self.pre_process:
            final_embedding = language_embeddings

            # merge image embeddings into language_embeddings
            if has_images:
                # has images, merge image_embeddings into final_embedding
                n_image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
                n_image_features = image_embeddings.shape[0]
                print("n_image_tokens ",n_image_tokens)
                print("input_ids ",input_ids,input_ids == IMAGE_TOKEN_INDEX)
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                        f"features {n_image_features}"
                    )

                image_mask = (
                    (input_ids == IMAGE_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(final_embedding)
                    .to(final_embedding.place)
                )
                image_embeddings = image_embeddings.to(final_embedding.place, final_embedding.dtype)
                final_embedding = final_embedding.masked_scatter(
                    image_mask, image_embeddings
                )  #  [b, seq_len, h_language]

            # merge video embeddings into final_embedding
            if has_videos:
                # has images, merge image_embeddings into final_embedding
                n_video_tokens = (input_ids == VIDEO_TOKEN_INDEX).sum().item()
                n_video_features = video_embeddings.shape[0]

                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, "
                        f"features {n_video_features}"
                    )

                video_mask = (
                    (input_ids == VIDEO_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(final_embedding)
                    .to(final_embedding.place)
                )
                video_embeddings = video_embeddings.to(final_embedding.place, final_embedding.dtype)
                final_embedding = final_embedding.masked_scatter(video_mask, video_embeddings)

        #
        # Create the final labels and loss mask (if this is the last language model stage).
        #
        final_labels, final_loss_mask = None, None

        if self.post_process and has_labels:

            # Pipeline parallel expects fixed input size. Check if we need to pad
            if self._language_is_pipeline_parallel and labels.shape[1] < self._language_max_sequence_length:
                max_seq_len = self._language_max_sequence_length
                final_labels = paddle.full(
                    (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, place=labels.place
                )
                final_loss_mask = paddle.full(
                    (batch_size, max_seq_len), 0, dtype=loss_mask.dtype, place=loss_mask.place
                )
                final_labels[:, : labels.shape[1]] = labels[:, :]
                final_loss_mask[:, : labels.shape[1]] = loss_mask[:, :]
            else:
                final_labels, final_loss_mask = labels, loss_mask

        if final_embedding is not None and final_labels is not None:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        if final_embedding is not None:
            # Truncate if exceeding the language model's max sequence length.
            if final_embedding.shape[1] > self._language_max_sequence_length:
                final_embedding = final_embedding[:, : self._language_max_sequence_length]

            # TODO: check and add self.context_parallel_lm to MCoreQwen2VLModel
            # # Transpose to [s,b,h] if not using CP because CP Sharding expects seq in dim=1
            final_embedding = final_embedding.transpose(1, 0).contiguous()  #  [seq_len, bs, h_language]

        truncate_labels = final_labels is not None and final_labels.shape[1] > self._language_max_sequence_length
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]
        return final_embedding, final_labels, final_loss_mask, attention_mask

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


class Qwen2VLModel(paddle.nn.Layer):
    """Lightning Wrapper for Qwen2VL Model"""

    def __init__(
        self,
        config: Qwen2VLProvider,
        model_version: str,
        optim = None,
        tokenizer = None,
        model_transform: Optional[Callable[[nn.Layer], nn.Layer]] = None,
    ):
        # pylint: disable=C0115,C0116
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        self.model_version = model_version
        assert self.model_version in ["qwen2-vl", "qwen25-vl"], "model_version only supports qwen2-vl and qwen25-vl."

    def provide(self, vp_stage: Optional[int] = None) -> None:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "module"):
            self.module = self.config.provide(self.tokenizer, vp_stage=vp_stage)

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids = None,
        loss_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        inference_params = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.FloatTensor] = None,
        image_grid_thw = None,
        video_grid_thw = None,
        second_per_grid_ts: Optional[paddle.FloatTensor] = None,
    ) -> paddle.Tensor:
        # pylint: disable=C0115,C0116
        output_tensor = self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            labels=labels,
            inference_params=inference_params,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

        return output_tensor
    def forward_step(self, batch) -> paddle.Tensor:
        # pylint: disable=C0115,C0116
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> paddle.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> paddle.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)


__all__ = [
    "Qwen2VLModel",
    "Qwen2VLProvider",
    "qwen2vl_data_step",
    "qwen2vl_forward_step",
]
