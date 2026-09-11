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

from __future__ import annotations

from typing import Any, Mapping

from ..configuration_utils import PretrainedConfig
from ..llama.configuration import LlamaConfig

_COMPUTE_DTYPE_CHOICES = ("checkpoint", "float16", "bfloat16", "float32", "float64")


def _validate_compute_dtype(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if value not in _COMPUTE_DTYPE_CHOICES:
        choices = ", ".join(_COMPUTE_DTYPE_CHOICES)
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


class JanusConfig(PretrainedConfig):
    """Configuration for the Janus understanding path and image branch."""

    model_type = "multi_modality"
    is_composition = True
    sub_configs = {"language_config": LlamaConfig}

    _language_defaults = {
        "vocab_size": 102400,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 30,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "max_position_embeddings": 16384,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }

    def __init__(
        self,
        language_config: LlamaConfig | Mapping[str, Any] | None = None,
        vision_config: Mapping[str, Any] | None = None,
        aligner_config: Mapping[str, Any] | None = None,
        gen_vision_config: Mapping[str, Any] | None = None,
        gen_aligner_config: Mapping[str, Any] | None = None,
        gen_head_config: Mapping[str, Any] | None = None,
        language_compute_dtype: str | None = None,
        vision_compute_dtype: str | None = None,
        **kwargs,
    ):
        if isinstance(language_config, LlamaConfig):
            normalized_language_config = language_config
        elif language_config is None:
            language_values = {}
            normalized_language_config = None
        elif isinstance(language_config, Mapping):
            language_values = dict(language_config)
            normalized_language_config = None
        else:
            raise TypeError("language_config must be a LlamaConfig, mapping, or None")

        if normalized_language_config is None:
            if language_values.get("model_type", "llama") != "llama":
                raise ValueError("Janus language_config must use model_type='llama'")
            for name, default in self._language_defaults.items():
                language_values.setdefault(name, default)
            normalized_language_config = LlamaConfig(**language_values)

        self.language_config = normalized_language_config
        self.vision_config = dict(vision_config or {})
        self.aligner_config = dict(aligner_config or {})
        self.gen_vision_config = dict(gen_vision_config or {})
        self.gen_aligner_config = dict(gen_aligner_config or {})
        self.gen_head_config = dict(gen_head_config or {})
        self.language_compute_dtype = _validate_compute_dtype("language_compute_dtype", language_compute_dtype)
        self.vision_compute_dtype = _validate_compute_dtype("vision_compute_dtype", vision_compute_dtype)

        kwargs.setdefault("architectures", ["JanusForCausalLM"])
        kwargs.setdefault("torch_dtype", getattr(self.language_config, "dtype", "bfloat16") or "bfloat16")
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


__all__ = ["JanusConfig"]
