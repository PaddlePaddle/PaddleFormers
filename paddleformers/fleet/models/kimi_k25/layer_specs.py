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
from ...transformer.paddle_norm import WrappedPaddleNormPipe
from ...transformer.transformer_config import TransformerConfig
from ...transformer.transformer_layer import TransformerLayerSublayersSpec
from ..backends import LocalSpecProvider
from ..common.embeddings.rotary_pos_embedding import Rope2DPosEmbRepeated
from ..gpt.gpt_layer_specs import get_mlp_layer_spec_for_backend
from .embedding import MoonVision3dPatchEmbed, VisionEmbeddingSpec
from .kimi_k25_model import (
    KimiK25VisionModel,
    KimiK25VisionSublayersSpec,
    KimiK25VisionTransformerLayer,
)
from .sd2_tpool_merge import (
    KimiK25VisionPatchMergerSpec,
    KimiK25VisionPathMerger,
    KimiK25VisionSd2TpoolMerger,
)


def get_kimi_k25_vision_layer_local_spec(
    config: TransformerConfig = None,
    use_qk_norm: bool = False,
    layer_number: int = 1,
) -> LayerSpec:
    backend = LocalSpecProvider()
    layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
    qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)
    mlp = get_mlp_layer_spec_for_backend(
        backend=backend,
    )
    transformer_cls = KimiK25VisionTransformerLayer

    return LayerSpec(
        layer=transformer_cls,
        sublayers_spec=TransformerLayerSublayersSpec(
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
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attn.qkv_proj.layer_norm_",
                "post_attention_layernorm.": "mlp.up_gate_proj.layer_norm_",
            },
        ),
        extra_kwargs={
            "config": config,
            "layer_number": layer_number,
            "hidden_dropout_prob": config.hidden_dropout_prob
            if config is not None
            else None,
            "modal": "vision_model",
        },
    )


def get_kimi_k25_vision_encoder_layers_spec(
    config: TransformerConfig,
) -> list[LayerSpec]:
    layer_spec_func = partial(
        get_kimi_k25_vision_layer_local_spec,
        config=config,
        use_qk_norm=config.use_qk_norm,
    )
    layer_specs = []
    for layer_number in range(config.num_hidden_layers):
        real_layer_number = layer_number + config.num_empty_layers_add_in_head
        layer_specs.append(
            layer_spec_func(
                layer_number=real_layer_number,
            )
        )

    return layer_specs


def get_kimi_k25_vision_spec(
    config: TransformerConfig,
    transformer_layers_spec: list[LayerSpec],
    head_empty_layers_spec: list[LayerSpec] | None = None,
    tail_empty_layer_spec: list[LayerSpec] | None = None,
):
    backend = LocalSpecProvider()
    embedding_extra_kwargs = {"config": config}
    rotary_emb_extra_kwargs = {
        "head_dim": config.hidden_size // config.num_attention_heads,
        "max_height": getattr(config, "max_height", 512),
        "max_width": getattr(config, "max_width", 512),
    }
    embedding_spec = VisionEmbeddingSpec(
        rope_embedding=LayerSpec(
            layer=Rope2DPosEmbRepeated,
            extra_kwargs=rotary_emb_extra_kwargs,
        )
    )

    sdtpool_merger_spec = LayerSpec(
        layer=KimiK25VisionSd2TpoolMerger, extra_kwargs={"config": config}
    )
    merger_spec = LayerSpec(
        layer=KimiK25VisionPathMerger,
        sublayers_spec=KimiK25VisionPatchMergerSpec(
            backend.layer_norm(
                rms_norm=(config.normalization == "RMSNorm"), for_qk=False
            )
        ),
        extra_kwargs={
            "config": config,
        },
    )

    return LayerSpec(
        layer=KimiK25VisionModel,
        sublayers_spec=KimiK25VisionSublayersSpec(
            embedding=LayerSpec(
                layer=MoonVision3dPatchEmbed,
                sublayers_spec=embedding_spec,
                extra_kwargs=embedding_extra_kwargs,
            ),
            head_empty_layers=head_empty_layers_spec,
            transformer_layers=transformer_layers_spec,
            tail_empty_layers=tail_empty_layer_spec,
            final_layernorm=LayerSpec(
                layer=WrappedPaddleNormPipe,
                extra_kwargs={
                    "config": config,
                    "hidden_size": config.hidden_size,
                },
            ),
            sdtpool_merger=sdtpool_merger_spec,
            merger=merger_spec,
        ),
        extra_kwargs={"config": config, "modal": "vision_model"},
    )
