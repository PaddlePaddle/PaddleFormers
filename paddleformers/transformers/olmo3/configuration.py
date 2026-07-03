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

"""OLMo3 model configuration.

Migrated from transformers.models.olmo3.configuration_olmo3.Olmo3Config

OLMo3 extends OLMo2 with:
- Sliding window attention for 3 out of 4 layers
- Full attention for every 4th layer
- Separate RoPE embeddings for sliding and full attention
"""

from ..configuration_utils import layer_type_validation
from ..olmo2.configuration import Olmo2Config


class Olmo3Config(Olmo2Config):
    r"""
    This is the configuration class to store the configuration of a [`Olmo3Model`].
    It is used to instantiate an OLMo3 model according to the specified arguments,
    defining the model architecture.

    OLMo3 extends OLMo2 with sliding window attention support.

    Args:
        sliding_window (`int`, *optional*, defaults to 4096):
            Size of the sliding window for sliding window attention.
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Defaults to sliding window attention
            for 3 out of 4 layers, and full attention for every 4th layer.
    """

    model_type = "olmo3"

    def __init__(
        self,
        sliding_window=4096,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sliding_window = sliding_window
        self.layer_types = layer_types
        if self.layer_types is None:
            # Default: 3 out of 4 layers use sliding_attention, every 4th uses full_attention
            self.layer_types = [
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)
