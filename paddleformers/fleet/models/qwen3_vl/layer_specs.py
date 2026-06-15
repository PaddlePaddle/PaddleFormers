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
from functools import partial

from paddle.distributed.fleet.meta_parallel import LayerSpec

from ...fusions.fused_bias_dropout import get_bias_dropout_add
from ...transformer.attention import SelfAttention, SelfAttentionSublayersSpec
from ...transformer.identity_op import IdentityOp
from ...transformer.transformer_config import TransformerConfig
from ..backends import LocalSpecProvider
from ..common.embeddings.rotary_pos_embedding import RotaryEmbedding
from ..gpt.gpt_layer_specs import get_mlp_layer_spec_for_backend
from .embedding import VisionEmbedding, VisionEmbeddingSpec
from .patch_merger import Qwen3VLVisionPatchMergerSpec, Qwen3VLVisionPathMerger
from .qwen3_vl_model import (
    Qwen3VLVisionModel,
    Qwen3VLVisionSublayersSpec,
    Qwen3VLVisionTransformerLayer,
    Qwen3VLVsisionTransformerSubLayerSpec,
)


def get_qwen3_vl_vision_layer_local_spec(
    config: TransformerConfig = None,
    use_qk_norm: bool = False,
    layer_number: int = 1,
    append_deepstack: bool = False,
) -> LayerSpec:
    backend = LocalSpecProvider()
    layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
    qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)
    mlp = get_mlp_layer_spec_for_backend(
        backend=backend,
    )
    transformer_cls = Qwen3VLVisionTransformerLayer
    merger_spec = LayerSpec(
        layer=Qwen3VLVisionPathMerger,
        sublayers_spec=Qwen3VLVisionPatchMergerSpec(
            backend.layer_norm(rms_norm=(config.normalization == "RMSNorm"), for_qk=False)
        ),
        extra_kwargs={"config": config, "use_postshuffle_norm": True},
    )
    return LayerSpec(
        layer=transformer_cls,
        sublayers_spec=Qwen3VLVsisionTransformerSubLayerSpec(
            input_layernorm=layer_norm,
            self_attn=LayerSpec(
                layer=SelfAttention,
                sublayers_spec=SelfAttentionSublayersSpec(
                    qkv_proj=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    o_proj=backend.row_parallel_linear(),
                    q_norm=qk_norm if use_qk_norm else IdentityOp,
                    k_norm=qk_norm if use_qk_norm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_attention_layernorm=layer_norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            deepstack_merger=merger_spec if append_deepstack else None,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attn.qkv_proj.layer_norm_",
                "post_attention_layernorm.": "mlp.up_gate_proj.layer_norm_",
            },
        ),
        extra_kwargs={
            "config": config,
            "layer_number": layer_number,
            "hidden_dropout_prob": config.hidden_dropout_prob if config is not None else None,
            "modal": "vision_model",
        },
    )


def get_qwen3vl_vision_encoder_layers_spec(
    config: TransformerConfig,
) -> list[LayerSpec]:
    layer_spec_func = partial(
        get_qwen3_vl_vision_layer_local_spec,
        config=config,
        use_qk_norm=config.use_qk_norm,
    )
    layer_specs = []
    append_deepstack = False
    for layer_number in range(config.num_hidden_layers):
        real_layer_number = layer_number + config.num_empty_layers_add_in_head
        if layer_number in config.deepstack_visual_indexes:
            append_deepstack = True
        layer_specs.append(
            layer_spec_func(
                layer_number=real_layer_number,
                append_deepstack=append_deepstack,
            )
        )

    return layer_specs


def get_qwen3_vl_vision_spec(
    config: TransformerConfig,
    transformer_layers_spec: list[LayerSpec],
    head_empty_layers_spec: list[LayerSpec] | None = None,
    tail_empty_layer_spec: list[LayerSpec] | None = None,
    rotary_percent: float = 1.0,
    rotary_base: int = 10000,
    rope_scaling: bool = False,
):
    backend = LocalSpecProvider()
    embedding_extra_kwargs = {"config": config}
    rotary_emb_extra_kwargs = {
        "head_dim": config.head_dim // 2,
        "rotary_base": rotary_base,
        "rope_scaling": rope_scaling,
        "rotary_percent": rotary_percent,
    }
    embedding_spec = VisionEmbeddingSpec(
        rope_embedding=LayerSpec(
            layer=RotaryEmbedding,
            extra_kwargs=rotary_emb_extra_kwargs,
        )
    )
    merger_norm = backend.layer_norm(rms_norm=(config.normalization == "RMSNorm"), for_qk=False)
    merger_spec = LayerSpec(
        layer=Qwen3VLVisionPathMerger,
        sublayers_spec=Qwen3VLVisionPatchMergerSpec(
            norm=merger_norm,
        ),
        extra_kwargs={
            "config": config,
            "dim": config.out_hidden_size,
            "context_dim": config.hidden_size,
        },
    )

    return LayerSpec(
        layer=Qwen3VLVisionModel,
        extra_kwargs={"config": config, "modal": "vision_model"},
        sublayers_spec=Qwen3VLVisionSublayersSpec(
            embedding=LayerSpec(
                layer=VisionEmbedding,
                sublayers_spec=embedding_spec,
                extra_kwargs=embedding_extra_kwargs,
            ),
            head_empty_layers=head_empty_layers_spec,
            transformer_layers=transformer_layers_spec,
            tail_empty_layers=tail_empty_layer_spec,
            merger=merger_spec,
        ),
    )
