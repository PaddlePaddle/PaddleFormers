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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, Union

import paddle
import transformers
from paddlefleet.transformer.transformer_config import TransformerConfig
from transformers import AutoConfig as HFAutoConfig
from transformers import AutoModelForImageTextToText
from transformers import Qwen2_5_VLConfig as HFQwen25VLConfig
from transformers import Qwen2VLConfig as HFQwen2VLConfig
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig as HFQwen25VLVisionConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig as HFQwen2VLVisionConfig

from .qwen2 import (
    Qwen2Provider,
    Qwen2Provider1P5B,
    Qwen2Provider7B,
    Qwen2Provider72B,
    Qwen25Provider3B,
    Qwen25Provider7B,
    Qwen25Provider32B,
    Qwen25Provider72B,
)
from .base import (
    Qwen2VLProvider,
    Qwen2VLModel,
    Qwen2VLVisionProvider,
    Qwen25VLVisionProvider,
)
from .base import MultimodalProjectorProvider
# Note: these Qwen2VL Providers are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs
@dataclass
class Qwen2VLProvider2B(Qwen2VLProvider):
    """Qwen2VL Config 2B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen2Provider1P5B(share_embeddings_and_output_weights=True)
    )
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionProvider(num_hidden_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(input_size=5120, hidden_size=1536, intermediate_size=5120)
    )


@dataclass
class Qwen2VLProvider7B(Qwen2VLProvider):
    """Qwen2VL Config 7B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Provider7B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionProvider(num_hidden_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(input_size=5120, hidden_size=3584, intermediate_size=5120)
    )


@dataclass
class Qwen2VLProvider72B(Qwen2VLProvider):
    """Qwen2VL Provider 72B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Provider72B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen2VLVisionProvider(num_hidden_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(input_size=5120, hidden_size=8192, intermediate_size=5120)
    )


@dataclass
class Qwen25VLProvider3B(Qwen2VLProvider):
    """Qwen2.5VL Config 3B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider3B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionProvider(num_hidden_layers=32, num_attention_heads=16,fullatt_block_indexes=[7, 15, 23, 31])
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(
            projector_type="mcore_mlp", input_size=5120, hidden_size=2048, intermediate_size=5120
        )
    )


@dataclass
class Qwen25VLProvider7B(Qwen2VLProvider):
    """Qwen2.5VL Config 7B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider7B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionProvider(num_hidden_layers=32, num_attention_heads=16)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(
            projector_type="mcore_mlp", input_size=5120, hidden_size=3584, intermediate_size=5120
        )
    )


@dataclass
class Qwen25VLProvider32B(Qwen2VLProvider):
    """Qwen2.5VL Config 32B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider32B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionProvider(num_hidden_layers=32, num_attention_heads=16, intermediate_size=3456)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(
            projector_type="mcore_mlp", input_size=5120, hidden_size=5120, intermediate_size=5120
        )
    )


@dataclass
class Qwen25VLProvider72B(Qwen2VLProvider):
    """Qwen2.5VL Config 72B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider72B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionProvider(num_hidden_layers=32, num_attention_heads=16, intermediate_size=3456)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(
            projector_type="mcore_mlp", input_size=5120, hidden_size=8192, intermediate_size=5120
        )
    )


# def import_qkv(q, k, v, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
#     # pylint: disable=C0115,C0116
#     old_tensor_shape = q.size()
#     new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
#     new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

#     q = q.view(*new_q_tensor_shape)
#     k = k.view(*new_kv_tensor_shape)
#     v = v.view(*new_kv_tensor_shape)

#     qkv_weights_l = []
#     for i in range(num_query_groups):
#         qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
#         qkv_weights_l.append(k[i : i + 1, :, :])
#         qkv_weights_l.append(v[i : i + 1, :, :])
#     qkv_weights = paddle.concat(qkv_weights_l)
#     assert qkv_weights.ndim == 3, qkv_weights.shape
#     assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
#     assert qkv_weights.shape[1] == head_size, qkv_weights.shape
#     assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

#     qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

#     return qkv_weights


# @io.state_transform(
#     source_key=("visual.blocks.*.attn.qkv.weight",),
#     target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
# )
# def _import_vision_qkv(ctx: io.TransformCTX, hf_qkv_weights):
#     # pylint: disable=C0115,C0116
#     megatron_config = ctx.target.config.vision_transformer_config

#     slice = int(hf_qkv_weights.shape[0] / 3)
#     assert slice == megatron_config.hidden_size
#     q = hf_qkv_weights[:slice, :]
#     k = hf_qkv_weights[slice : slice * 2, :]
#     v = hf_qkv_weights[slice * 2 :, :]

#     return import_qkv(
#         q,
#         k,
#         v,
#         head_num=megatron_config.num_attention_heads,
#         num_query_groups=megatron_config.num_query_groups,
#         heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
#         hidden_size=megatron_config.hidden_size,
#         head_size=megatron_config.kv_channels,
#     )


# @io.state_transform(
#     source_key=("visual.blocks.*.attn.qkv.bias",),
#     target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
# )
# def _import_vision_qkv_bias(ctx: io.TransformCTX, hf_qkv_bias):
#     # pylint: disable=C0115,C0116
#     megatron_config = ctx.target.config.vision_transformer_config

#     slice = int(hf_qkv_bias.shape[0] / 3)
#     assert slice == megatron_config.hidden_size

#     q_bias = hf_qkv_bias[:slice]
#     k_bias = hf_qkv_bias[slice : slice * 2]
#     v_bias = hf_qkv_bias[slice * 2 :]

#     return import_qkv(
#         q_bias.unsqueeze(-1),
#         k_bias.unsqueeze(-1),
#         v_bias.unsqueeze(-1),
#         head_num=megatron_config.num_attention_heads,
#         num_query_groups=megatron_config.num_query_groups,
#         heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
#         hidden_size=1,
#         head_size=megatron_config.kv_channels,
#     ).squeeze(-1)


# @io.state_transform(
#     source_key=(
#         "model.layers.*.self_attn.q_proj.weight",
#         "model.layers.*.self_attn.k_proj.weight",
#         "model.layers.*.self_attn.v_proj.weight",
#     ),
#     target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
# )
# def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
#     # pylint: disable=C0115,C0116
#     megatron_config = ctx.target.config.language_transformer_config
#     return import_qkv(
#         q,
#         k,
#         v,
#         head_num=megatron_config.num_attention_heads,
#         num_query_groups=megatron_config.num_query_groups,
#         heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
#         hidden_size=megatron_config.hidden_size,
#         head_size=megatron_config.kv_channels,
#     )


# @io.state_transform(
#     source_key=(
#         "model.layers.*.self_attn.q_proj.bias",
#         "model.layers.*.self_attn.k_proj.bias",
#         "model.layers.*.self_attn.v_proj.bias",
#     ),
#     target_key="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
# )
# def _import_language_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
#     # pylint: disable=C0115,C0116
#     megatron_config = ctx.target.config.language_transformer_config
#     return import_qkv(
#         q_bias.unsqueeze(-1),
#         k_bias.unsqueeze(-1),
#         v_bias.unsqueeze(-1),
#         head_num=megatron_config.num_attention_heads,
#         num_query_groups=megatron_config.num_query_groups,
#         heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
#         hidden_size=1,
#         head_size=megatron_config.kv_channels,
#     ).squeeze(-1)


# @io.state_transform(
#     source_key=("vision_model.embeddings.class_embedding",),
#     target_key="vision_model.class_token",
# )
# def _import_cls_token(ctx: io.TransformCTX, cls_token):
#     # pylint: disable=C0115,C0116
#     return cls_token.reshape(1, 1, -1)


# @io.state_transform(
#     source_key=(
#         "model.layers.*.mlp.gate_proj.weight",
#         "model.layers.*.mlp.up_proj.weight",
#     ),
#     target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
# )
# def _import_linear_fc1(down, gate):
#     # pylint: disable=C0115,C0116
#     return paddle.concat((down, gate), axis=0)


# @io.state_transform(
#     source_key=("visual.blocks.*.mlp.gate_proj.weight", "visual.blocks.*.mlp.up_proj.weight"),
#     target_key="vision_model.decoder.layers.*.mlp.linear_fc1.weight",
# )
# def _import_vision_linear_fc1_weight(down, gate):
#     # pylint: disable=C0115,C0116
#     return paddle.concat((down, gate), axis=0)


# @io.state_transform(
#     source_key=("visual.blocks.*.mlp.gate_proj.bias", "visual.blocks.*.mlp.up_proj.bias"),
#     target_key="vision_model.decoder.layers.*.mlp.linear_fc1.bias",
# )
# def _import_vision_linear_fc1_bias(down, gate):
#     # pylint: disable=C0115,C0116
#     return paddle.concat((down, gate), axis=0)


# def export_qkv(linear_qkv, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
#     # pylint: disable=C0115,C0116
#     qkv_total_dim = head_num + 2 * num_query_groups

#     linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
#     hidden_size = linear_qkv.size(-1)
#     q_slice = paddle.concat(
#         [
#             paddle.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
#             for i in range(num_query_groups)
#         ]
#     )
#     k_slice = paddle.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
#     v_slice = paddle.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

#     q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
#     k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
#     v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

#     return q_proj, k_proj, v_proj


# def export_qkv_bias(qkv_bias: paddle.Tensor, head_num, num_query_groups, heads_per_group, head_size):
#     """
#     Split interleave-concatenated qkv bias to separate q, k, v bias

#     Example: export layer linear_qkv bias to HF {q|k|v}_proj bias
#     """
#     qkv_total_dim = head_num + 2 * num_query_groups

#     qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])
#     q_slice = paddle.concat(
#         [
#             paddle.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
#             for i in range(num_query_groups)
#         ]
#     )
#     k_slice = paddle.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
#     v_slice = paddle.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

#     q_bias = qkv_bias[q_slice].reshape(-1).cpu()
#     k_bias = qkv_bias[k_slice].reshape(-1).cpu()
#     v_bias = qkv_bias[v_slice].reshape(-1).cpu()

#     return q_bias, k_bias, v_bias


# @io.state_transform(
#     source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
#     target_key="visual.blocks.*.attn.qkv.weight",
# )
# def _export_vision_qkv(ctx: io.TransformCTX, qkv):
#     # pylint: disable=C0115,C0116
#     hf_config = ctx.target.config.vision_config
#     hidden_size = hf_config.embed_dim if hf_config.model_type == "qwen2_vl" else hf_config.hidden_size
#     return paddle.concat(
#         export_qkv(
#             qkv,
#             head_num=hf_config.num_heads,
#             num_query_groups=hf_config.num_heads,
#             heads_per_group=hf_config.num_heads // hf_config.num_heads,
#             hidden_size=hidden_size,
#             head_size=hidden_size // hf_config.num_heads,
#         ),
#         axis=0,
#     )


# @io.state_transform(
#     source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
#     target_key="visual.blocks.*.attn.qkv.bias",
# )
# def _export_vision_qkv_bias(ctx: io.TransformCTX, qkv_bias):
#     # pylint: disable=C0115,C0116
#     hf_config = ctx.target.config.vision_config
#     hidden_size = hf_config.embed_dim if hf_config.model_type == "qwen2_vl" else hf_config.hidden_size
#     return paddle.concat(
#         export_qkv_bias(
#             qkv_bias,
#             head_num=hf_config.num_heads,
#             num_query_groups=hf_config.num_heads,
#             heads_per_group=hf_config.num_heads // hf_config.num_heads,
#             head_size=hidden_size // hf_config.num_heads,
#         ),
#         axis=0,
#     )


# @io.state_transform(
#     source_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
#     target_key=(
#         "model.layers.*.self_attn.q_proj.weight",
#         "model.layers.*.self_attn.k_proj.weight",
#         "model.layers.*.self_attn.v_proj.weight",
#     ),
# )
# def _export_language_qkv(ctx: io.TransformCTX, qkv):
#     # pylint: disable=C0115,C0116
#     hf_config = ctx.target.config
#     return export_qkv(
#         qkv,
#         head_num=hf_config.num_attention_heads,
#         num_query_groups=hf_config.num_key_value_heads,
#         heads_per_group=hf_config.num_attention_heads // hf_config.num_key_value_heads,
#         hidden_size=hf_config.hidden_size,
#         head_size=hf_config.hidden_size // hf_config.num_attention_heads,
#     )


# @io.state_transform(
#     source_key="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
#     target_key=(
#         "model.layers.*.self_attn.q_proj.bias",
#         "model.layers.*.self_attn.k_proj.bias",
#         "model.layers.*.self_attn.v_proj.bias",
#     ),
# )
# def _export_language_qkv_bias(ctx: io.TransformCTX, qkv_bias):
#     # pylint: disable=C0115,C0116
#     hf_config = ctx.target.config
#     return export_qkv_bias(
#         qkv_bias,
#         head_num=hf_config.num_attention_heads,
#         num_query_groups=hf_config.num_key_value_heads,
#         heads_per_group=hf_config.num_attention_heads // hf_config.num_key_value_heads,
#         head_size=hf_config.hidden_size // hf_config.num_attention_heads,
#     )


# @io.state_transform(
#     source_key="vision_model.class_token",
#     target_key="vision_model.embeddings.class_embedding",
# )
# def _export_cls_token(ctx: io.TransformCTX, cls_token):
#     # pylint: disable=C0115,C0116
#     return cls_token.squeeze()


# @io.state_transform(
#     source_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
#     target_key=(
#         "model.layers.*.mlp.gate_proj.weight",
#         "model.layers.*.mlp.up_proj.weight",
#     ),
# )
# def _export_linear_fc1(linear_fc1):
#     # pylint: disable=C0115,C0116
#     gate_proj, up_proj = paddle.chunk(linear_fc1, 2, dim=0)
#     return gate_proj, up_proj


# @io.state_transform(
#     source_key="vision_model.decoder.layers.*.mlp.linear_fc1.weight",
#     target_key=(
#         "visual.blocks.*.mlp.gate_proj.weight",
#         "visual.blocks.*.mlp.up_proj.weight",
#     ),
# )
# def _export_vision_linear_fc1_weight(vision_fc1_weight):
#     # pylint: disable=C0115,C0116
#     gate_proj, up_proj = paddle.chunk(vision_fc1_weight, 2, dim=0)
#     return gate_proj, up_proj


# @io.state_transform(
#     source_key="vision_model.decoder.layers.*.mlp.linear_fc1.bias",
#     target_key=(
#         "visual.blocks.*.mlp.gate_proj.bias",
#         "visual.blocks.*.mlp.up_proj.bias",
#     ),
# )
# def _export_vision_linear_fc1_bias(vision_fc1_bias):
#     # pylint: disable=C0115,C0116
#     gate_proj, up_proj = paddle.chunk(vision_fc1_bias, 2, dim=0)
#     return gate_proj, up_proj
