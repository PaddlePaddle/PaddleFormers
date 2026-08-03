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

from ..configuration_utils import PretrainedConfig


class Gemma4MoeConfig(PretrainedConfig):
    """Gemma4 26B-A4B text-only MoE model configuration.

    NOTE: Gemma4 currently only has VL checkpoints (Gemma4ForConditionalGeneration),
    no standalone text-only checkpoint exists. The config.json has top-level
    model_type="gemma4" (VL wrapper) with text params nested in text_config
    (model_type="gemma4_text").

    Both "gemma4" and "gemma4_text" are registered in CONFIG_MAPPING to this class,
    and from_dict() automatically extracts text_config from the VL wrapper.

    TODO(VL): When implementing full multimodal Gemma4:
      1. Create Gemma4Config (VL wrapper with text_config + vision_config)
      2. Change CONFIG_MAPPING "gemma4" to point to Gemma4Config
      3. Keep only "gemma4_text" -> Gemma4MoeConfig
      4. Remove the text_config extraction logic in from_dict() below
    """

    model_type = "gemma4_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Load text config from VL wrapper config.

        Gemma4 only ships VL checkpoints, so config.json outer model_type="gemma4"
        with text params in text_config. This extracts it automatically.

        TODO(VL): Move this extraction to Gemma4Config when VL model is implemented.
        """
        if config_dict.get("model_type") == "gemma4" and "text_config" in config_dict:
            config_dict = config_dict["text_config"]
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=2816,
        intermediate_size=2112,
        moe_intermediate_size=704,
        num_hidden_layers=30,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation=None,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        # RoPE
        rope_parameters=None,
        sliding_window_rope_base=10000.0,
        full_attention_rope_base=1000000.0,
        full_attention_rope_partial_factor=0.25,
        # Attention pattern
        layer_types=None,
        sliding_window=1024,
        interleaved_attn_pattern=(5, 1),
        # MoE
        num_experts=128,
        top_k_experts=8,
        scoring_func="sigmoid",
        enable_moe_block=True,
        # Gemma4-specific
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        scale_embeddings_by_hidden_size=True,
        # Pipeline
        pp_seg_method="layer:Gemma4TransformerLayer",
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_global_key_value_heads = num_global_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.hidden_act = hidden_activation or hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.enable_moe_block = enable_moe_block
        self.sliding_window = sliding_window
        self.interleaved_attn_pattern = interleaved_attn_pattern
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.scoring_func = scoring_func
        self.attention_k_eq_v = attention_k_eq_v
        self.final_logit_softcapping = final_logit_softcapping
        self.scale_embeddings_by_hidden_size = scale_embeddings_by_hidden_size
        self.pp_seg_method = pp_seg_method

        # Parse rope_parameters from HF config format
        if rope_parameters is not None:
            sliding_rope = rope_parameters.get("sliding_attention", {})
            full_rope = rope_parameters.get("full_attention", {})
            self.sliding_window_rope_base = sliding_rope.get("rope_theta", sliding_window_rope_base)
            self.full_attention_rope_base = full_rope.get("rope_theta", full_attention_rope_base)
            self.full_attention_rope_partial_factor = full_rope.get(
                "partial_rotary_factor", full_attention_rope_partial_factor
            )
        else:
            self.sliding_window_rope_base = sliding_window_rope_base
            self.full_attention_rope_base = full_attention_rope_base
            self.full_attention_rope_partial_factor = full_attention_rope_partial_factor

        # Build layer_types from interleaved pattern if not provided
        if layer_types is None:
            sliding_count, global_count = self.interleaved_attn_pattern
            pattern = ["sliding_attention"] * sliding_count + ["full_attention"] * global_count
            self.layer_types = (pattern * ((num_hidden_layers // len(pattern)) + 1))[:num_hidden_layers]
        else:
            self.layer_types = layer_types
