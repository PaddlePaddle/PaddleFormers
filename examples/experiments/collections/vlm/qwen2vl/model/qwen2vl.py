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
import paddle.nn.functional as F
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
        default_factory=lambda: Qwen2Provider1P5B(share_embeddings_and_output_weights=True,multimodal_embedding=True)
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

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Provider7B(multimodal_embedding=True))
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

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen2Provider72B(multimodal_embedding=True))
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

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider3B(multimodal_embedding=True))
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionProvider(num_hidden_layers=32, num_attention_heads=16,fullatt_block_indexes=[7, 15, 23, 31],hidden_act=F.silu)
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

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider7B(multimodal_embedding=True))
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

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider32B(multimodal_embedding=True))
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

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Qwen25Provider72B(multimodal_embedding=True))
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: Qwen25VLVisionProvider(num_hidden_layers=32, num_attention_heads=16, intermediate_size=3456)
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorProvider(
            projector_type="mcore_mlp", input_size=5120, hidden_size=8192, intermediate_size=5120
        )
    )