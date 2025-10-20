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

""" Ernie4_5VL model configuration """
import json
from typing import Optional, Union

from ...utils.log import logger
from ..configuration_utils import PretrainedConfig


class Ernie4_5_VLVisionConfig(PretrainedConfig):
    r"""
    Configuration class for Ernie4_5_VLVariableResolutionResamplerModel model.

    This class stores the configuration of an Ernie4_5_VLVariableResolutionResamplerModel model, defining the model architecture.
    It inherits from PretrainedConfig and can be used to control model outputs.
    """

    model_type = "ernie4_5_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=1280,
        hidden_act="quick_gelu",
        intermediate_size=4 * 1280,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_merge_size=2,
        text_hidden_size=2560,
        rms_norm_eps=1e-5,
        vision_rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # vision projection
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size

        # resampler
        self.text_hidden_size = text_hidden_size
        self.temporal_merge_size = temporal_merge_size
        self.rms_norm_eps = rms_norm_eps
        self.vision_rms_norm_eps = vision_rms_norm_eps

        self.initializer_range = initializer_range


class Ernie4_5_VLTextConfig(PretrainedConfig):
    r"""
    Configuration class for Ernie4_5_VLTextModel model.

    This class stores the configuration of an Ernie4_5_VLTextModel model, defining the model architecture.
    It inherits from PretrainedConfig and can be used to control model outputs.
    """

    model_type = "ernie4_5_vl_text"
    base_config_key = "text_config"
    attribute_map = {"num_experts": "moe_num_experts", "num_experts_per_tok": "moe_k"}

    def __init__(
        self,
        vocab_size=103424,
        hidden_size=2560,
        intermediate_size=12288,
        num_hidden_layers=28,
        num_attention_heads=20,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        use_flash_attention=True,
        use_sparse_flash_attn=True,
        use_var_len_flash_attn=False,
        use_bias=False,
        tie_word_embeddings=True,
        rope_theta=500_000.0,
        freq_allocation=20,
        rope_scaling=None,
        multimodel_experts=True,
        moe_gate="topk",
        moe_intermediate_size=[1536, 512],
        moe_k=6,
        moe_num_experts=64,
        moe_num_shared_experts=2,
        moe_layer_start_index=1,
        moe_layer_end_index=29,
        moe_layer_interval=1,
        moe_norm_min=1e-12,
        moe_use_hard_gate=True,
        moe_reverse_token_drop=False,
        moe_dense_experts_token_type_id=3,
        moe_all_to_all_dropout=0.0,
        moe_group_experts=False,
        num_nextn_predict_layers=0,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        rope_3d=True,
        compression_ratio=1.0,
        fuse_linear=False,
        fuse_rms_norm=False,
        fuse_swiglu=False,
        fuse_rope=False,
        fuse_ln=False,
        cachekv_quantFalse,
        attention_probs_dropout_prob=0.0,
        ignored_index=-100,
        weight_share_add_bias=True,
        use_rmsnorm=True,
        multi_token_pred_lambda=0.1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_flash_attention = use_flash_attention
        self.use_sparse_flash_attn = use_sparse_flash_attn
        self.use_var_len_flash_attn = use_var_len_flash_attn
        self.use_bias = use_bias
        self.rope_theta = rope_theta
        self.freq_allocation = freq_allocation
        self.multimodel_experts = multimodel_experts
        self.moe_gate = moe_gate
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_k = moe_k
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = moe_layer_end_index
        self.moe_layer_interval = moe_layer_interval
        self.moe_norm_min = moe_norm_min
        self.moe_use_hard_gate = moe_use_hard_gate
        self.moe_reverse_token_drop = moe_reverse_token_drop
        self.moe_dense_experts_token_type_id = moe_dense_experts_token_type_id
        self.moe_all_to_all_dropout = moe_all_to_all_dropout
        self.moe_group_experts = moe_group_experts
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.rope_3d = rope_3d
        self.compression_ratio = compression_ratio
        self.fuse_linear = fuse_linear
        self.fuse_rms_norm = fuse_rms_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_rope = fuse_rope
        self.fuse_ln = fuse_ln
        self.cachekv_quant = cachekv_quant
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ignored_index = ignored_index
        self.weight_share_add_bias = weight_share_add_bias
        self.use_rmsnorm = use_rmsnorm
        self.multi_token_pred_lambda = multi_token_pred_lambda

        self.register_unsavable_keys(
            [
                "recompute",
                "recompute_use_reentrant",
                "refined_recompute",
                "recompute_granularity",
                "use_recompute_lm_head",
                "use_recompute_loss_fn",
                "pp_seg_method",
                "skip_recompute_ops",
                "use_sparse_flash_attn",
                "use_var_len_flash_attn",
                "use_sparse_head_and_loss_fn",
                "loss_subbatch_seqlen",
                "micro_batch_size",
                "fuse_softmax_mask",
                "cachekv_quant",
                "use_fused_head_and_loss_fn",
                "max_sequence_length",
                "moe_group",
                "dpo_config",
                "use_recompute_moe",
                "enable_delay_scale_loss",
                "moe_dropout_prob",
                "moe_all_to_all_dropout",
                "num_acc_steps",
                "disable_ffn_model_parallel",
                "moe_group_origin",
                "moe_multimodal_dispatch_use_allgather",
                "moe_rank",
                "moe_world_size",
                "sequence_parallel",
            ]
        )

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def use_moe(self) -> bool:
        """
        Check if model is using MoE architecture.

        Returns:
            bool: True if moe_num_experts > 0, False otherwise
        """
        return (
            sum(self.moe_num_experts) > 0
            if self.multimodel_experts
            else self.moe_num_experts > 0
        )

class Ernie4_5_VLConfig(PretrainedConfig):
    """
    Configuration class for Ernie4_5VL model.

    This class stores the configuration of an Ernie4_5VL model, defining the model architecture.
    It inherits from PretrainedConfig and can be used to control model outputs.
    """

    model_type = "ernie4_5_vl"
    sub_configs = {"vision_config": Ernie4_5_VLVisionConfig, "text_config": Ernie4_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        im_patch_id=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_start_token_id=101304,
        image_end_token_id=101305,
        image_token_id=100295,
        video_start_token_id=101306,
        video_end_token_id=101307,
        video_token_id=100296,
        dpo_config=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.im_patch_id = im_patch_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id
        self.dpo_config = dpo_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.register_unsavable_keys(
            [
                "recompute",
                "recompute_use_reentrant",
                "refined_recompute",
                "recompute_granularity",
                "use_recompute_lm_head",
                "use_recompute_loss_fn",
                "pp_seg_method",
                "skip_recompute_ops",
                "use_sparse_flash_attn",
                "use_var_len_flash_attn",
                "use_sparse_head_and_loss_fn",
                "loss_subbatch_seqlen",
                "micro_batch_size",
                "fuse_softmax_mask",
                "cachekv_quant",
                "use_fused_head_and_loss_fn",
                "max_sequence_length",
                "moe_group",
                "dpo_config",
                "use_recompute_moe",
                "enable_delay_scale_loss",
                "moe_dropout_prob",
                "moe_all_to_all_dropout",
                "num_acc_steps",
                "disable_ffn_model_parallel",
                "moe_group_origin",
                "moe_multimodal_dispatch_use_allgather",
                "moe_rank",
                "moe_world_size",
                "sequence_parallel",
            ]
        )

__all__ = [
    "Ernie4_5_VLConfig",
    "Ernie4_5_VLTextConfig",
    "Ernie4_5_VLVisionConfig",
]
