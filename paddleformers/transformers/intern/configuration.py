# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
InternLM2 Common Configuration

This module provides a unified configuration for both InternLM2 2.0 and 2.5 models.
It detects the version based on the configuration fields and routes accordingly.
"""

from paddleformers.transformers.configuration_utils import PretrainedConfig


class InternLM2Config(PretrainedConfig):
    """
    InternLM2 configuration. This is a unified config that handles both 2.0 and 2.5 versions.

    When loading from HuggingFace, the `model_type` will be "internlm2" (not "internlm2_5").
    This config detects the actual version and routes to the appropriate implementation.
    """

    model_type = "internlm2"  # Important: must match HuggingFace config
    _auto_class = "AutoConfig"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=92550,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        bias=True,
        rope_theta=10000,
        rope_scaling=None,
        attn_implementation=None,
        dtype="bfloat16",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.bias = bias

        import paddle

        if isinstance(dtype, str):
            dtype_map = {
                "float32": paddle.float32,
                "float16": paddle.float16,
                "bfloat16": paddle.bfloat16,
            }
            self.dtype = dtype_map.get(dtype.lower(), paddle.float32)
        else:
            self.dtype = dtype

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attn_implementation = attn_implementation
        if self.attn_implementation is None:
            self.attn_implementation = "eager"

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise ValueError(f"`rope_scaling` must be a dictionary, got {self.rope_scaling}")
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_factor is None:
            raise ValueError("`rope_scaling` must contain 'type' and 'factor' keys, " f"got {self.rope_scaling}")
        if rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(f"`rope_scaling` type must be 'linear' or 'dynamic', got '{rope_scaling_type}'")
        if not isinstance(rope_scaling_factor, (int, float)) or rope_scaling_factor < 1.0:
            raise ValueError(f"`rope_scaling` factor must be a number >= 1, got {rope_scaling_factor}")

    @property
    def is_version_2_5(self):
        if hasattr(self, "auto_map") and self.auto_map is not None:
            if "AutoModelForSequenceClassification" in self.auto_map:
                return True
        return False
