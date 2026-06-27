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

"""Layer specs for Qwen3.5 vision encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.transformer.enums import AttnMaskType

from ..backends import LocalSpecProvider
from ..common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from ..common.empty_layer import EmptyLayer
from ..gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from ..qwen3_vl.embedding import VisionEmbedding, VisionEmbeddingSpec
from ..qwen3_vl.patch_merger import (
    Qwen3VLVisionPatchMergerSpec,
    Qwen3VLVisionPathMerger,
)
from .qwen3_5_model import (
    Qwen3_5VisionModel,
    Qwen3_5VisionSublayersSpec,
)

if TYPE_CHECKING:
    from ...transformer.transformer_config import TransformerConfig


# ======================================================================
# Vision model specs
# ======================================================================


def get_qwen3_5_vision_spec(config: TransformerConfig) -> LayerSpec:
    """Build the complete Qwen3.5 vision model spec."""
    backend = LocalSpecProvider()

    # --- Empty layers for pipeline parallel padding ---
    empty_layer_spec = LayerSpec(
        layer=EmptyLayer, extra_kwargs={"config": config}
    )
    head_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_head
    tail_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_tail

    # --- Transformer encoder layers ---
    head_offset = config.num_empty_layers_add_in_head
    transformer_layers = [
        get_gpt_layer_local_spec(
            config=config,
            layer_number=i + head_offset,
            attn_mask_type=AttnMaskType.no_mask,
        )
        for i in range(config.num_hidden_layers)
    ]

    # --- Vision embedding with rotary position embedding ---
    embedding_spec = LayerSpec(
        layer=VisionEmbedding,
        sublayers_spec=VisionEmbeddingSpec(
            rope_embedding=LayerSpec(
                layer=RotaryEmbedding,
                extra_kwargs={
                    "head_dim": config.head_dim // 2,
                    "rotary_base": config.rope_theta,
                    "rope_scaling": config.rope_scaling,
                    "rotary_percent": config.rotary_percent,
                },
            )
        ),
        extra_kwargs={"config": config},
    )

    # --- Patch merger ---
    config.merger_hidden_size = config.hidden_size * (
        config.spatial_merge_size**2
    )
    merger_spec = LayerSpec(
        layer=Qwen3VLVisionPathMerger,
        sublayers_spec=Qwen3VLVisionPatchMergerSpec(
            norm=backend.layer_norm(
                rms_norm=(config.normalization == "RMSNorm"), for_qk=False
            ),
        ),
        extra_kwargs={
            "config": config,
            "dim": config.out_hidden_size,
            "context_dim": config.hidden_size,
        },
    )

    # --- Assemble full vision model spec ---
    return LayerSpec(
        layer=Qwen3_5VisionModel,
        extra_kwargs={"config": config, "modal": "vision_model"},
        sublayers_spec=Qwen3_5VisionSublayersSpec(
            embedding=embedding_spec,
            head_empty_layers=head_empty_layers,
            transformer_layers=transformer_layers,
            tail_empty_layers=tail_empty_layers,
            merger=merger_spec,
        ),
    )
