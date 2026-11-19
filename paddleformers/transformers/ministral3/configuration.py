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

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params

__all__ = ["Ministral3Config"]


def _default_rope_parameters(max_position_embeddings):
    return {
        "rope_type": "yarn",
        "rope_theta": 1000000.0,
        "factor": 16.0,
        "original_max_position_embeddings": 16384,
        "max_position_embeddings": max_position_embeddings,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale_all_dim": 1.0,
        "mscale": 1.0,
        "llama_4_scaling_beta": 0.1,
    }


class Ministral3Config(PretrainedConfig):
    r"""
    Configuration of a [`Ministral3ForCausalLM`] model.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden states.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the MLP intermediate states.
        num_hidden_layers (`int`, *optional*, defaults to 34):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads used for grouped-query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function of the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            Maximum sequence length the model may ever attend to.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon of the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return a key/value cache.
        rope_parameters (`dict`, *optional*):
            RoPE parameters. `rope_scaling` from older checkpoints is accepted as an alias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied to the attention probabilities.
    """

    model_type = "ministral3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=11,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        rope_theta=1000000.0,
        sliding_window=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta

        if rope_parameters is None:
            rope_parameters = kwargs.pop("rope_scaling", None)
        if rope_parameters is None:
            rope_parameters = _default_rope_parameters(max_position_embeddings)
        else:
            rope_parameters = dict(rope_parameters)
            if "rope_type" not in rope_parameters and "type" in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters["type"]
            rope_parameters.pop("type", None)
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        standardize_rope_params(self, rope_theta=self.rope_theta)
        self.rope_theta = self.rope_parameters["rope_theta"]
        rope_config_validation(self, ignore_keys={"llama_4_scaling_beta", "max_position_embeddings"})
