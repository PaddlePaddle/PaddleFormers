# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import copy
import itertools
import random
import unittest
from dataclasses import dataclass

import numpy as np
import paddle
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    NoPipelineParallel,
    build_spec_layer,
)
from paddle.nn import functional as F

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.models.common.empty_layer import EmptyLayer
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.models.gpt.gpt_embedding import GPTEmbedding
from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_spec,
)
from paddleformers.fleet.models.gpt.lm_head import GPTLMHead
from paddleformers.fleet.models.qwen3_5.layer_specs import (
    get_qwen3_5_vision_spec,
)
from paddleformers.fleet.tensor_parallel.mappings import (
    scatter_to_sequence_parallel_region,
)
from paddleformers.fleet.transformer import TransformerConfig
from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.paddle_norm import (
    WrappedPaddleNorm,
    WrappedPaddleNormPipe,
)
from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none

# ======================================================================
# Qwen3_5VisionProvider (inlined from deleted qwen3_5_provider.py)
# ======================================================================


@dataclass
class Qwen3_5VisionProvider(TransformerConfig):
    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304
    embed_dim: int = (1152,)
    hidden_size: int = 1152
    out_hidden_size: int = 3584
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    activation_func: object = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    model_version: str = "qwen3_5"

    def provide(self):
        spec = get_qwen3_5_vision_spec(self)
        return build_spec_layer(
            spec,
            seg_method="layer:TransformerLayer|EmptyLayer",
            num_stages=self.pipeline_model_parallel_size,
        )


# ======================================================================
# get_qwen3_5_language_spec (inlined from deleted layer_specs function)
# ======================================================================


from paddleformers.fleet.models.qwen3_5.qwen3_5_model import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormPipe,
)


def get_qwen3_5_language_spec(config):
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * config.num_hidden_layers

    empty_layer_spec = LayerSpec(
        layer=EmptyLayer, extra_kwargs={"config": config}
    )
    head_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_head
    tail_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_tail

    head_offset = getattr(config, "num_empty_layers_add_in_head", 0)

    LAYER_TYPE_MAP = {
        "full_attention": "self_attention",
        "linear_attention": "gated_delta_net",
    }

    transformer_layers_spec = []
    for i, lt in enumerate(layer_types):
        attn_type = LAYER_TYPE_MAP.get(lt)
        if attn_type is None:
            raise ValueError(f"Unknown layer type: {lt!r} at index {i}")
        spec = get_gpt_layer_local_spec(
            config=config,
            normalization=config.normalization,
            layer_number=i + head_offset,
            attention_layer_type=attn_type,
            num_experts=config.n_routed_experts,
            moe_expert_fusion=config.moe_expert_fusion,
            multi_latent_attention=config.multi_latent_attention,
        )

        sub = spec.sublayers_spec
        if sub.input_layernorm is WrappedPaddleNorm:
            sub.input_layernorm = Qwen3_5RMSNorm
        if sub.post_attention_layernorm is WrappedPaddleNorm:
            sub.post_attention_layernorm = Qwen3_5RMSNorm

        attn_spec = sub.self_attn
        if hasattr(attn_spec, "sublayers_spec"):
            attn_sub = attn_spec.sublayers_spec
            if (
                hasattr(attn_sub, "q_norm")
                and attn_sub.q_norm is WrappedPaddleNorm
            ):
                attn_sub.q_norm = Qwen3_5RMSNorm
            if (
                hasattr(attn_sub, "k_norm")
                and attn_sub.k_norm is WrappedPaddleNorm
            ):
                attn_sub.k_norm = Qwen3_5RMSNorm

        transformer_layers_spec.append(spec)

    full_spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=transformer_layers_spec,
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        head_empty_layers_spec=head_empty_layers,
        tail_empty_layers_spec=tail_empty_layers,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rotary_base,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    final_norm_spec = full_spec.sublayers_spec.layer_norm
    if final_norm_spec.layer is WrappedPaddleNormPipe:
        final_norm_spec.layer = Qwen3_5RMSNormPipe

    return full_spec


# ======================================================================
# Qwen3_5Model (inlined from deleted qwen3_5_model.py class)
# ======================================================================


class Qwen3_5Model(FleetLayer):
    def __init__(
        self,
        config,
        vision_model=None,
        language_model=None,
        spatial_merge_size=2,
        image_token_id=None,
        video_token_id=None,
    ):
        assert isinstance(language_model, NoPipelineParallel)
        assert isinstance(vision_model, NoPipelineParallel)
        super().__init__(config=config)
        self.visual = vision_model
        self.language_model = language_model
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.rope_deltas = None

        self.language_embedding = self._find_language_embedding()
        self.language_backbone = self._find_language_backbone()
        self.language_lm_head = self._find_lm_head()

        self.tp_group = get_tensor_model_parallel_group_if_none(None)

        if self.language_embedding is not None:
            embed_tokens = self.language_embedding.embedding.embed_tokens
            embed_tokens.reduce_scatter_embeddings = False

    def _find_language_embedding(self):
        for layer in self.language_model._layers.run_function:
            if isinstance(layer, GPTEmbedding):
                return layer
        return None

    def _find_language_backbone(self):
        return [
            layer
            for layer in self.language_model._layers.run_function
            if not isinstance(layer, (GPTEmbedding, GPTLMHead))
        ]

    def _find_lm_head(self):
        for layer in self.language_model._layers.run_function:
            if isinstance(layer, GPTLMHead):
                return layer
        return None

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        dict_input = {
            "pixel_values": pixel_values,
            "grid_thw": image_grid_thw,
        }
        output = self.visual._layers.forward(dict_input)
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_video_features(
        self, pixel_values_videos, video_grid_thw=None, **kwargs
    ):
        return self.get_image_features(
            pixel_values_videos, video_grid_thw, **kwargs
        )

    def get_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        if input_ids is None:
            embed_fn = self.get_input_embeddings()
            special_image_mask = (
                inputs_embeds
                == embed_fn(
                    paddle.to_tensor(self.image_token_id, dtype="int64")
                )
            ).all(-1)
            special_video_mask = (
                inputs_embeds
                == embed_fn(
                    paddle.to_tensor(self.video_token_id, dtype="int64")
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.image_token_id
            special_video_mask = input_ids == self.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        if image_features is not None:
            assert int(inputs_embeds[special_image_mask].numel()) == int(
                image_features.numel()
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        if video_features is not None:
            assert int(inputs_embeds[special_video_mask].numel()) == int(
                video_features.numel()
            )

        return special_image_mask, special_video_mask

    def get_vision_position_ids(
        self,
        start_position,
        grid_thw,
        spatial_merge_size=1,
        device=None,
    ):
        if isinstance(grid_thw, Tensor):
            t = int(grid_thw[0].item())
            h = int(grid_thw[1].item())
            w = int(grid_thw[2].item())
        else:
            t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])

        llm_t = t
        llm_h = h // spatial_merge_size
        llm_w = w // spatial_merge_size
        seq_len = llm_t * llm_h * llm_w

        pos_w = paddle.arange(start_position, start_position + llm_w).tile(
            [llm_h * llm_t]
        )
        pos_h = paddle.arange(
            start_position, start_position + llm_h
        ).repeat_interleave(llm_w * llm_t)
        pos_t = paddle.full([seq_len], start_position, dtype="int64")

        return paddle.stack([pos_t, pos_h, pos_w], axis=0)

    def get_rope_index(
        self,
        input_ids,
        mm_token_type_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        spatial_merge_size = self.spatial_merge_size
        mrope_position_deltas = []
        position_ids = paddle.zeros(
            [3, input_ids.shape[0], input_ids.shape[1]],
            dtype=input_ids.dtype,
        )

        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx in range(input_ids.shape[0]):
            current_input_ids = input_ids[batch_idx]
            input_token_type = mm_token_type_ids[batch_idx]

            if attention_mask is not None:
                mask = attention_mask[batch_idx].astype("bool")
                current_input_ids = current_input_ids[mask]
                input_token_type = input_token_type[mask]

            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        paddle.arange(text_len).reshape([1, -1]).expand([3, -1])
                        + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        spatial_merge_size,
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    t_val = (
                        int(grid_thw[0].item())
                        if isinstance(grid_thw, Tensor)
                        else int(grid_thw[0])
                    )
                    h_val = (
                        int(grid_thw[1].item())
                        if isinstance(grid_thw, Tensor)
                        else int(grid_thw[1])
                    )
                    w_val = (
                        int(grid_thw[2].item())
                        if isinstance(grid_thw, Tensor)
                        else int(grid_thw[2])
                    )
                    current_pos += max(h_val, w_val) // spatial_merge_size

            llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape(
                [3, -1]
            )

            if attention_mask is not None:
                mask = attention_mask[batch_idx].astype("bool")
                position_ids[:, batch_idx, mask] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions

            mrope_position_deltas.append(
                int(llm_positions.max().item()) + 1 - len(current_input_ids)
            )

        mrope_position_deltas = paddle.to_tensor(
            mrope_position_deltas, dtype="int64"
        ).unsqueeze(1)

        return position_ids, mrope_position_deltas

    def compute_3d_position_ids(
        self,
        input_ids=None,
        inputs_embeds=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        past_key_values=None,
        mm_token_type_ids=None,
    ):
        past_key_values_length = (
            0
            if past_key_values is None
            else past_key_values.get_seq_length()
            if hasattr(past_key_values, "get_seq_length")
            else 0
        )
        can_compute_mrope = (
            input_ids is not None
            and mm_token_type_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        )

        if can_compute_mrope and (
            self.rope_deltas is None or past_key_values_length == 0
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids

        # Handle text-only case: generate 3D position_ids with identical values across all three dimensions
        if input_ids is not None and (
            image_grid_thw is None and video_grid_thw is None
        ):
            batch_size, seq_length = input_ids.shape
            if attention_mask is not None:
                position_ids = attention_mask.astype("int64").cumsum(-1) - 1
                position_ids = paddle.where(
                    attention_mask == 0,
                    paddle.zeros_like(position_ids),
                    position_ids,
                )
                position_ids = position_ids.reshape([1, batch_size, -1]).tile(
                    [3, 1, 1]
                )
            else:
                position_ids = (
                    paddle.arange(seq_length)
                    .reshape([1, 1, -1])
                    .expand([3, batch_size, -1])
                )
            return position_ids

        if self.rope_deltas is not None and inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.astype("int64").cumsum(-1) - 1
                position_ids = paddle.where(
                    attention_mask == 0,
                    paddle.zeros_like(position_ids),
                    position_ids,
                )
                position_ids = position_ids.reshape([1, batch_size, -1]).tile(
                    [3, 1, 1]
                )
            else:
                position_ids = (
                    paddle.arange(
                        past_key_values_length,
                        past_key_values_length + seq_length,
                    )
                    .reshape([1, 1, -1])
                    .expand([3, batch_size, -1])
                )

            delta = self.rope_deltas
            if delta.shape[0] != batch_size:
                delta = delta.tile([batch_size // delta.shape[0], 1])
            position_ids = position_ids + delta.unsqueeze(0)
            return position_ids

        return None

    def forward(self, dict_args):
        input_ids = dict_args.get("input_ids", None)
        inputs_embeds = dict_args.get("inputs_embeds", None)
        pixel_values = dict_args.get("pixel_values", None)
        pixel_values_videos = dict_args.get("pixel_values_videos", None)
        image_grid_thw = dict_args.get("image_grid_thw", None)
        video_grid_thw = dict_args.get("video_grid_thw", None)
        attention_mask = dict_args.get("attention_mask", None)
        position_ids = dict_args.get("position_ids", None)
        mm_token_type_ids = dict_args.get("mm_token_type_ids", None)
        past_key_values = dict_args.get("past_key_values", None)

        if (
            inputs_embeds is None
            and input_ids is not None
            and self.language_model is not None
        ):
            inputs_embeds = self.language_embedding.embedding.embed_tokens(
                input_ids
            )

        if pixel_values is not None and self.visual is not None:
            image_features = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_features = image_features.astype(inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_features
            )

        if pixel_values_videos is not None and self.visual is not None:
            video_features = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_features = video_features.astype(inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                video_features=video_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_features
            )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.transpose([1, 0, 2]).contiguous()
            inputs_embeds = scatter_to_sequence_parallel_region(
                inputs_embeds, group=self.tp_group
            )

        dict_args["position_ids"] = position_ids
        dict_args["input_ids"] = None
        dict_args["decoder_input"] = inputs_embeds

        lm_dict_args = self.language_embedding(
            dict_args, decoder_input=inputs_embeds
        )

        for layer in self.language_backbone:
            lm_dict_args = layer(lm_dict_args)

        if self.language_lm_head is not None:
            logits = self.language_lm_head(lm_dict_args)
            return logits

        return lm_dict_args


# ======================================================================
# Test dimensions
# ======================================================================

# ---- Test dimensions (small for fast unit testing) ----
HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 16
NUM_LAYERS = 2
OUT_HIDDEN_SIZE = 96
INTERMEDIATE_SIZE = 128
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
IN_CHANNELS = 3
NUM_POSITION_EMBEDDINGS = 256  # 16 * 16

# Test image: 1 image, 2 temporal frames, 64x64 spatial
IMAGE_H = 64
IMAGE_W = 64
GRID_T = 1
GRID_H = IMAGE_H // PATCH_SIZE  # 4
GRID_W = IMAGE_W // PATCH_SIZE  # 4
SEQ_LEN = GRID_T * GRID_H * GRID_W  # 16
MERGED_TOKENS = SEQ_LEN // (SPATIAL_MERGE_SIZE**2)  # 4

# ---- Qwen3_5Model test dimensions ----
VL_HIDDEN_SIZE = HIDDEN_SIZE  # 64, vision hidden size
VL_LM_HIDDEN_SIZE = OUT_HIDDEN_SIZE  # 96, must match vision output dim
VL_VOCAB_SIZE = 256
VL_IMAGE_TOKEN_ID = 200
VL_VIDEO_TOKEN_ID = 201
VL_NUM_LM_LAYERS = 2
VL_TEXT_BEFORE = 5
VL_TEXT_AFTER = 3
VL_NUM_IMAGE_TOKENS = MERGED_TOKENS  # 4
VL_SEQ_LEN = VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS + VL_TEXT_AFTER  # 12


# ======================================================================
# Tests
# ======================================================================


class TestQwen3_5Model(unittest.TestCase):
    """Test Qwen3_5Model (VL composite model) forward and backward.

    Uses real Qwen3_5VisionModel and GPTModel sub-models to verify:
    - Vision encoder (ViT + patch merger) produces correct features
    - Language decoder (GPT with transformer layers) processes embeddings
    - Vision-language feature merging via masked_scatter
    - 3D MRoPE position ID computation
    - Gradient flow through the entire computation graph
    """

    def setUp(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
            "pp_configs": {"delay_scale_loss": False},
        }
        strategy.pipeline_configs = {
            "micro_batch_size": 1,
            "accumulate_steps": 1,
        }
        self.strategy = strategy

        if not ps.have_global_memory_buffer():
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()
            ps.initialize_model_parallel(hcg)

        # Step 1: Create vision_config and Qwen3_5VisionModel
        vision_config = Qwen3_5VisionProvider(
            num_hidden_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            out_hidden_size=OUT_HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            patch_size=PATCH_SIZE,
            spatial_merge_size=SPATIAL_MERGE_SIZE,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            in_channels=IN_CHANNELS,
            num_position_embeddings=NUM_POSITION_EMBEDDINGS,
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            normalization="LayerNorm",
            use_qk_norm=False,
            gated_linear_unit=False,
            apply_rope_fusion=False,
        )
        self.vision_config = vision_config
        vision_model = vision_config.provide()

        # Step 2: Create language_config and GPTModel
        # head_dim = VL_LM_HIDDEN_SIZE // NUM_HEADS = 24
        # mrope_section: [T, H, W] sections for interleaved MRoPE
        # Sum of sections should equal head_dim // 2 = 12
        language_config = GPTConfig(
            num_hidden_layers=VL_NUM_LM_LAYERS,
            hidden_size=VL_LM_HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            head_dim=VL_LM_HIDDEN_SIZE // NUM_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            normalization="LayerNorm",
            gated_linear_unit=False,
            apply_rope_fusion=False,
            vocab_size=VL_VOCAB_SIZE,
            max_sequence_length=1024,
            position_embedding_type="mrope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=False,
            parallel_output=False,
            tie_word_embeddings=False,
            layer_types=["full_attention", "linear_attention"],
            gated_attention=True,
        )
        # Set mrope_section after creation since GPTConfig doesn't define this field
        language_config.mrope_section = [8, 2, 2]
        self.language_config = language_config
        self.language_config.model_type = "qwen3_5"

        language_spec = get_qwen3_5_language_spec(
            config=language_config,
        )
        language_model = build_spec_layer(
            language_spec,
            seg_method="layer:TransformerLayer|EmptyLayer",
            num_stages=1,
        )

        # Step 3: Create Qwen3_5Model with vision and language models
        model = Qwen3_5Model(
            config=language_config,
            vision_model=NoPipelineParallel(vision_model, strategy),
            language_model=NoPipelineParallel(language_model, strategy),
            spatial_merge_size=SPATIAL_MERGE_SIZE,
            image_token_id=VL_IMAGE_TOKEN_ID,
            video_token_id=VL_VIDEO_TOKEN_ID,
        )

        # Convert model to bf16 (attention requires fp16/bf16 with packed_seq_params)
        self.model = paddle.amp.decorate(
            models=model, level="O2", dtype="bfloat16"
        )

    def _clear_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.clear_gradient()
        self.model.rope_deltas = None

    def test_forward_backward_with_image(self):
        """Test full VL forward and backward with image input.

        Exercises the complete computation flow:
          1. Embed text tokens via language model embedding layer
          2. Encode image via real vision encoder (Conv3D + Transformer + PatchMerger)
          3. Merge image features into embedding sequence (masked_scatter)
          4. Compute 3D MRoPE position IDs
          5. Forward through language model transformer layers
          6. Backward through the entire graph
        """
        self._clear_gradients()
        batch_size = 1

        # ---- Construct multimodal input ----
        # input_ids: [text ... image_tokens ... text]
        input_ids = paddle.randint(0, 100, [batch_size, VL_SEQ_LEN])
        input_ids[0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS] = (
            VL_IMAGE_TOKEN_ID
        )

        # mm_token_type_ids: 0=text, 1=image
        mm_token_type_ids = paddle.zeros(
            [batch_size, VL_SEQ_LEN], dtype="int64"
        )
        mm_token_type_ids[
            0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS
        ] = 1

        image_grid_thw = paddle.to_tensor(
            [[GRID_T, GRID_H, GRID_W]], dtype="int32"
        )
        pixel_values = paddle.randn(
            [GRID_T, IN_CHANNELS, TEMPORAL_PATCH_SIZE, IMAGE_H, IMAGE_W]
        )

        dict_args = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }

        # ---- Forward (bf16) ----
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = self.model.forward(dict_args)

            # ---- Backward ----
            loss = output.sum()
            loss.backward()

        # ---- Verify gradients ----
        params_with_grad = 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(f"  [NO GRAD] {name}: shape={list(param.shape)}")
                continue

            params_with_grad += 1
            assert list(param.shape) == list(param.grad.shape), (
                f"Gradient shape mismatch for {name}: "
                f"param={list(param.shape)}, grad={list(param.grad.shape)}"
            )
            assert paddle.isfinite(param.grad).all().item(), (
                f"Non-finite gradients for {name}"
            )
            grad_norm = param.grad.detach().norm().item()
            print(
                f"  {name}: shape={list(param.shape)}, grad_norm={grad_norm:.6f}"
            )

        assert params_with_grad > 0, "No parameters received gradients"

        # Both vision and language model parameters should receive gradients
        vision_has_grad = any(
            p.grad is not None for p in self.model.visual.parameters()
        )
        lm_has_grad = any(
            p.grad is not None for p in self.model.language_model.parameters()
        )
        assert vision_has_grad, (
            "Vision model parameters did not receive gradients"
        )
        assert lm_has_grad, (
            "Language model parameters did not receive gradients"
        )

    def test_forward_backward_text_only(self):
        """Test forward and backward with text-only input (no vision).

        When no pixel_values are provided, the model should:
        - Embed text via language model embedding layer
        - Skip vision encoding entirely
        - Forward through language model transformer layers
        - Gradient flows only through language model
        """
        self._clear_gradients()
        batch_size = 1
        text_seq_len = 10

        input_ids = paddle.randint(0, 100, [batch_size, text_seq_len])
        dict_args = {"input_ids": input_ids}

        # ---- Forward (bf16) ----
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = self.model.forward(dict_args)

            # ---- Backward ----
            loss = output.sum()
            loss.backward()

        # Language model params should have gradients
        lm_params_with_grad = 0
        for name, param in self.model.language_model.named_parameters():
            if param.grad is not None:
                lm_params_with_grad += 1
                assert paddle.isfinite(param.grad).all().item(), (
                    f"Non-finite gradients for language_model.{name}"
                )
        assert lm_params_with_grad > 0, (
            "Language model parameters did not receive gradients"
        )

        # Vision model params should NOT have gradients (not used)
        for name, param in self.model.visual.named_parameters():
            assert param.grad is None, (
                f"Vision param {name} should not have gradient in text-only mode"
            )

    def test_get_rope_index(self):
        """Test 3D MRoPE position ID computation for mixed text+image tokens.

        Verifies that get_rope_index correctly computes 3D (temporal, height, width)
        position IDs for a sequence with interleaved text and image tokens.
        """
        batch_size = 1

        # Construct input_ids with image placeholders
        input_ids = paddle.randint(0, 100, [batch_size, VL_SEQ_LEN])
        input_ids[0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS] = (
            VL_IMAGE_TOKEN_ID
        )

        mm_token_type_ids = paddle.zeros(
            [batch_size, VL_SEQ_LEN], dtype="int64"
        )
        mm_token_type_ids[
            0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS
        ] = 1

        image_grid_thw = paddle.to_tensor(
            [[GRID_T, GRID_H, GRID_W]], dtype="int32"
        )

        # ---- Compute rope index ----
        position_ids, mrope_deltas = self.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
        )

        # position_ids: [3, batch_size, seq_len] for (temporal, height, width)
        assert list(position_ids.shape) == [3, batch_size, VL_SEQ_LEN], (
            f"Expected position_ids shape [3, {batch_size}, {VL_SEQ_LEN}], "
            f"got {list(position_ids.shape)}"
        )
        # mrope_position_deltas: [batch_size, 1]
        assert list(mrope_deltas.shape) == [batch_size, 1], (
            f"Expected mrope_deltas shape [{batch_size}, 1], "
            f"got {list(mrope_deltas.shape)}"
        )
        # All position IDs should be non-negative
        assert (position_ids >= 0).all().item(), (
            "Position IDs contain negative values"
        )

        # Text tokens before image should have monotonically increasing position IDs
        # and the 3 axes should be identical for text tokens
        text_before_pos = position_ids[:, 0, :VL_TEXT_BEFORE]
        for axis in range(3):
            for j in range(1, VL_TEXT_BEFORE):
                assert (
                    text_before_pos[axis, j].item()
                    > text_before_pos[axis, j - 1].item()
                ), f"Text positions not monotonically increasing on axis {axis}"

    def test_get_placeholder_mask(self):
        """Test that placeholder masks correctly identify image/video tokens."""
        batch_size = 1

        input_ids = paddle.randint(0, 100, [batch_size, VL_SEQ_LEN])
        input_ids[0, VL_TEXT_BEFORE : VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS] = (
            VL_IMAGE_TOKEN_ID
        )

        inputs_embeds = paddle.randn(
            [batch_size, VL_SEQ_LEN, VL_LM_HIDDEN_SIZE]
        )

        image_mask, video_mask = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds,
        )

        # Masks should be broadcastable to inputs_embeds shape
        assert list(image_mask.shape) == [
            batch_size,
            VL_SEQ_LEN,
            VL_LM_HIDDEN_SIZE,
        ]
        assert list(video_mask.shape) == [
            batch_size,
            VL_SEQ_LEN,
            VL_LM_HIDDEN_SIZE,
        ]

        # image_mask should be True at image token positions (expanded across hidden dim)
        image_mask_1d = image_mask[0, :, 0]  # [seq_len]
        for i in range(VL_SEQ_LEN):
            if VL_TEXT_BEFORE <= i < VL_TEXT_BEFORE + VL_NUM_IMAGE_TOKENS:
                assert image_mask_1d[i].item(), (
                    f"Position {i} should be masked as image"
                )
            else:
                assert not image_mask_1d[i].item(), (
                    f"Position {i} should NOT be masked as image"
                )

        # No video tokens in this input
        assert not video_mask.any().item(), "No video tokens should be detected"

    def test_create_mla(self):
        language_config = copy.deepcopy(self.language_config)
        language_config.multi_latent_attention = True

        language_spec = get_qwen3_5_language_spec(
            config=language_config,
        )

    def test_sharded_state_dict(self):
        state_dict = self.model.state_dict()


if __name__ == "__main__":
    unittest.main()
