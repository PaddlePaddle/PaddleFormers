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
import types
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Union

import paddle
import paddle.nn.functional as F
from paddlefleet import parallel_state
from paddlefleet.models.qwen3_vl import (
    Qwen3VLModelDist,
    Qwen3VLTextEmbedding,
    Qwen3VLTextTransformerLayer,
    Qwen3VLVisionModel,
    Qwen3VLVisionTransformerLayer,
)
from paddlefleet.models.qwen3_vl.qwen3_vl_builders import qwen3_vl_vision_builder
from paddlefleet.transformer import TransformerConfig

from ...nn.criterion.interface import CriterionLayer
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ..cache_utils import Cache
from ..gpt_provider import GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import Qwen3VLConfig
from .modeling import Qwen3VLCausalLMOutputWithPast, Qwen3VLPretrainedModel


@dataclass
class Qwen3VLVisionProvider(TransformerConfig):
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
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec = Qwen3VLVisionTransformerLayer
    model_version: str = "qwen3_vl"
    img_h: int = 336
    img_w: int = 336
    add_class_token: bool = False
    class_token_len: int = 1
    high_precision_rope: bool = True
    rotary_percent: float = 1.0
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
    }

    def provide(self) -> "Qwen3VLVisionModel":
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
            fleet_model = qwen3_vl_vision_builder(
                self,
                seg_method="layer:TransformerLayer|EmptyLayer",
                num_stages=pp_size,
            )
            model = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)

            for attr_name in dir(fleet_model):
                if not attr_name.startswith("__"):
                    try:
                        attr_value = getattr(fleet_model, attr_name)
                        setattr(model, attr_name, attr_value)
                    except:
                        pass
        return model


@dataclass
class Qwen3VLTextProvider(GPTModelProvider):
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
    specific_embedding: type = Qwen3VLTextEmbedding
    specific_transformer_layer: type = Qwen3VLTextTransformerLayer
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


@dataclass
class Qwen3VLProvider(TransformerConfig):
    text_config: Qwen3VLTextProvider | None = None
    vision_config: Qwen3VLVisionProvider | None = None

    drop_vision_class_token: bool = False
    vision_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1

    seq_length: int = 1024

    language_model_from_pretrained: str | None = None
    vision_model_from_pretrained: str | None = None

    def provide(self, tokenizer=None, vp_stage: int | None = None) -> "Qwen3VLModelDist":
        self.text_config.scatter_embedding_sequence_parallel = False
        self.text_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.text_config.tensor_parallel_output = self.tensor_parallel_output
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

        model = Qwen3VLModelDist(
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
            criterion=loss_fn,
        )

        return model

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.vision_config = Qwen3VLVisionProvider.from_config(config.vision_config)
        res.text_config = Qwen3VLTextProvider.from_config(config.text_config)
        res.vision_config.normalization = "LayerNorm"
        res.vision_config.gated_linear_unit = False
        res.text_config.multimodal_embedding = True
        res.text_config.position_embedding_type = "mrope"
        res.text_config.image_token_id = config.image_token_id
        res.text_config.video_token_id = config.video_token_id
        return res


class Qwen3VLVisionModelFleet(Qwen3VLPretrainedModel):
    def __new__(cls, config):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = Qwen3VLVisionProvider
        model_provider = model_provider_class.from_config(config)
        vision_model = model_provider.provide()
        vision_model._gen_aoa_config = cls._gen_aoa_config
        vision_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        vision_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        vision_model.config_to_save = config

        return vision_model


class Qwen3VLPretrainedModelFleet(PretrainedModel):
    config_class = Qwen3VLConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["Qwen3VLTextTransformerLayer", "Qwen3VLVisionTransformerLayer"]
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
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
    def _gen_aoa_config(cls, config: Qwen3VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        # visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        visual_target = "model.vision_model"
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"model.language_model.embed_tokens.weight -> {llm_prefix}embedding.embed_tokens.weight",
                f"model.language_model.norm.weight -> {llm_prefix}norm.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(config.vision_config.depth)
            for stmt in (
                f"model.visual.blocks.{layer_id}.attn.qkv.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight,model.visual.blocks.{layer_id}.attn.v.weight,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> {visual_prefix}layers.{layer_id}.self_attn.qkv_proj.weight,fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.qkv.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> {visual_prefix}layers.{layer_id}.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads},axis=0",
            )
        ]
        aoa_config["aoa_statements"] += (
            [
                f"model.visual.blocks.$LAYER_ID.attn.proj.weight^T -> {visual_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.attn.proj.bias -> {visual_prefix}layers.$LAYER_ID.self_attn.o_proj.bias"
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.weight^T -> {visual_prefix}layers.$LAYER_ID.mlp.{y}.weight"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.bias -> {visual_prefix}layers.$LAYER_ID.mlp.{y}.bias"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
        )
        aoa_config["aoa_statements"] += [
            f"model.visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.weight",
            f"model.visual.patch_embed.proj.bias -> {visual_prefix}patch_embed.bias",
            f"model.visual.pos_embed.weight -> {visual_prefix}pos_embed.weight",
            f"model.visual.merger.norm.weight -> {visual_prefix}merger.norm.weight",
            f"model.visual.merger.norm.bias -> {visual_prefix}merger.norm.bias",
            f"model.visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm1.bias -> {visual_prefix}layers.$LAYER_ID.input_layernorm.bias",
            f"model.visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm2.bias -> {visual_prefix}layers.$LAYER_ID.post_attention_layernorm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"model.visual.merger.linear_fc1.weight^T -> {visual_prefix}merger.linear_fc1.weight",
            f"model.visual.merger.linear_fc1.bias -> {visual_prefix}merger.linear_fc1.bias",
            f"model.visual.merger.linear_fc2.weight^T -> {visual_prefix}merger.linear_fc2.weight",
            f"model.visual.merger.linear_fc2.bias -> {visual_prefix}merger.linear_fc2.bias",
        ]
        for i, deepstack_idx in enumerate(config.vision_config.deepstack_visual_indexes):
            aoa_config["aoa_statements"] += [
                f"model.visual.deepstack_merger_list.{i}.linear_fc1.weight^T -> {visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc1.weight",
                f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias -> {visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc1.bias",
                f"model.visual.deepstack_merger_list.{i}.linear_fc2.weight^T -> {visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc2.weight",
                f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias -> {visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc2.bias",
                f"model.visual.deepstack_merger_list.{i}.norm.weight -> {visual_prefix}layers.{deepstack_idx}.deepstack_merger.norm.weight",
                f"model.visual.deepstack_merger_list.{i}.norm.bias -> {visual_prefix}layers.{deepstack_idx}.deepstack_merger.norm.bias",
            ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{layer_id}.self_attn.q_proj.weight^T, model.language_model.layers.{layer_id}.self_attn.k_proj.weight^T, model.language_model.layers.{layer_id}.self_attn.v_proj.weight^T -> {llm_prefix}layers.{layer_id}.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{layer_id}.mlp.gate_proj.weight^T, model.language_model.layers.{layer_id}.mlp.up_proj.weight^T -> {llm_prefix}layers.{layer_id}.mlp.up_gate_proj.weight, fused_ffn"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]

        # Qwen3_VLModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{'model.language_model.embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'} -> {llm_prefix}lm_head.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen3VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        # visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        visual_target = "model.vision_model"
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"{llm_prefix}embedding.embed_tokens.weight -> model.language_model.embed_tokens.weight",
                f"{llm_prefix}norm.weight -> model.language_model.norm.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(config.vision_config.depth)
            for stmt in (
                f"{visual_prefix}layers.{layer_id}.self_attn.qkv_proj.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight, model.visual.blocks.{layer_id}.attn.v.weight, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> model.visual.blocks.{layer_id}.attn.qkv.weight, axis=0",
                f"{visual_prefix}layers.{layer_id}.self_attn.qkv_proj.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads},axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> model.visual.blocks.{layer_id}.attn.qkv.bias, axis=0",
            )
        ]
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.visual.blocks.$LAYER_ID.attn.proj.weight"
            ]
            + [
                f"{visual_prefix}layers.$LAYER_ID.self_attn.o_proj.bias -> model.visual.blocks.$LAYER_ID.attn.proj.bias"
            ]
            + [
                f"{visual_prefix}layers.$LAYER_ID.mlp.{y}.weight^T -> model.visual.blocks.$LAYER_ID.mlp.{x}.weight"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
            + [
                f"{visual_prefix}layers.$LAYER_ID.mlp.{y}.bias -> model.visual.blocks.$LAYER_ID.mlp.{x}.bias"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.weight -> model.visual.patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.bias -> model.visual.patch_embed.proj.bias",
            f"{visual_prefix}pos_embed.weight -> model.visual.pos_embed.weight",
            f"{visual_prefix}merger.norm.weight -> model.visual.merger.norm.weight",
            f"{visual_prefix}merger.norm.bias -> model.visual.merger.norm.bias",
            f"{visual_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}layers.$LAYER_ID.input_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm1.bias",
            f"{visual_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm2.weight",
            f"{visual_prefix}layers.$LAYER_ID.post_attention_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.linear_fc1.weight^T -> model.visual.merger.linear_fc1.weight",
            f"{visual_prefix}merger.linear_fc1.bias -> model.visual.merger.linear_fc1.bias",
            f"{visual_prefix}merger.linear_fc2.weight^T -> model.visual.merger.linear_fc2.weight",
            f"{visual_prefix}merger.linear_fc2.bias -> model.visual.merger.linear_fc2.bias",
        ]
        for i, deepstack_idx in enumerate(config.vision_config.deepstack_visual_indexes):
            aoa_config["aoa_statements"] += [
                f"{visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc1.weight^T -> model.visual.deepstack_merger_list.{i}.linear_fc1.weight",
                f"{visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc1.bias -> model.visual.deepstack_merger_list.{i}.linear_fc1.bias",
                f"{visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc2.weight^T -> model.visual.deepstack_merger_list.{i}.linear_fc2.weight",
                f"{visual_prefix}layers.{deepstack_idx}.deepstack_merger.linear_fc2.bias -> model.visual.deepstack_merger_list.{i}.linear_fc2.bias",
                f"{visual_prefix}layers.{deepstack_idx}.deepstack_merger.norm.weight -> model.visual.deepstack_merger_list.{i}.norm.weight",
                f"{visual_prefix}layers.{deepstack_idx}.deepstack_merger.norm.bias -> model.visual.deepstack_merger_list.{i}.norm.bias",
            ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.self_attn.qkv_proj.weight  -> model.language_model.layers.{layer_id}.self_attn.q_proj.weight, model.language_model.layers.{layer_id}.self_attn.k_proj.weight, model.language_model.layers.{layer_id}.self_attn.v_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.language_model.layers.{layer_id}.self_attn.{x}_proj.weight"
            for layer_id in range(config.text_config.num_hidden_layers)
            for x in ("q", "k", "v")
        ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.mlp.up_gate_proj.weight -> model.language_model.layers.{layer_id}.mlp.gate_proj.weight, model.language_model.layers.{layer_id}.mlp.up_proj.weight, fused_ffn"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.mlp.{x}_proj.weight^T -> model.language_model.layers.{layer_id}.mlp.{x}_proj.weight"
            for layer_id in range(config.text_config.num_hidden_layers)
            for x in ("gate", "up")
        ]

        # Qwen3VLModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
            ]

        return aoa_config


class Qwen3VLModel(Qwen3VLPretrainedModelFleet):
    config_class = Qwen3VLConfig

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)
        model_provider_class = Qwen3VLProvider
        model_provider = model_provider_class.from_config(config)
        qwen3vl_model = Qwen3VLModelDist(model_provider, model_version=config.model_type, criterion=criterion)
        qwen3vl_model._gen_aoa_config = cls._gen_aoa_config
        qwen3vl_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3vl_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        qwen3vl_model.config_to_save = config

        return qwen3vl_model


class Qwen3VLForConditionalGeneration(Qwen3VLPretrainedModelFleet):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config_class = Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        # model_provider = Qwen3VLProvider.from_config(config)
        self.model = Qwen3VLModel(
            config, have_criterion=False
        )  # Qwen3VLModel(model_provider, model_version=config.model_type)
        self.criterion = CriterionLayer(config.text_config)
        # self.tie_weights()

    def state_dict(self, *args, **kwargs):
        # Override state_dict method to handle language_model's custom state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Remove existing language_model keys to avoid duplicates
        delete_key = []
        for key in state_dict.keys():
            if key.startswith("model.language_model."):
                delete_key.append(key)
            if key.startswith("model.vision_model."):
                delete_key.append(key)
        for key in delete_key:
            state_dict.pop(key)
        if self.model.language_model is not None:
            # Get language_model's state_dict
            language_state_dict = self.model.language_model.state_dict(*args, **kwargs)

            # Merge language_model parameters into main state_dict
            for key, value in language_state_dict.items():
                state_dict[key] = value

        if self.model.vision_model is not None:
            vision_state_dict = self.model.vision_model.state_dict(*args, **kwargs)
            for key, value in vision_state_dict.items():
                state_dict[key] = value

        return state_dict

    # def get_input_embeddings(self):
    #     return self.model.get_input_embeddings()
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
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.

        Example:

        ```python
        >>> from paddleformers.transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        >>> model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pd"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        logits = outputs

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
        )


class Qwen3VLForCausalLMPipe(Qwen3VLPretrainedModelFleet, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)
        model_provider_class = Qwen3VLProvider
        model_provider = model_provider_class.from_config(config)
        qwen3vl_model = Qwen3VLModelDist(model_provider, model_version=config.model_type, criterion=criterion)
        qwen3vl_model._gen_aoa_config = cls._gen_aoa_config
        qwen3vl_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3vl_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        qwen3vl_model.config_to_save = config

        return qwen3vl_model


class Qwen3VLModelPipe(Qwen3VLPretrainedModelFleet, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)
        model_provider_class = Qwen3VLProvider
        model_provider = model_provider_class.from_config(config)
        qwen3vl_model = Qwen3VLModelDist(model_provider, model_version=config.model_type, criterion=criterion)
        qwen3vl_model._gen_aoa_config = cls._gen_aoa_config
        qwen3vl_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3vl_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        qwen3vl_model.get_hardware_flops = types.MethodType(cls.get_hardware_flops, qwen3vl_model)
        qwen3vl_model.config_to_save = config

        return qwen3vl_model


__all__ = [
    "Qwen3VLModel",
    "Qwen3VLForCausalLMPipe",
    "Qwen3VLModelPipe",
    "Qwen3VLForConditionalGeneration",
]
