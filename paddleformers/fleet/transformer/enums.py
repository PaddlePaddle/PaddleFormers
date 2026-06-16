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

import enum


class ModelType(enum.Enum):
    """Model Type

    encoder_or_decoder for bert, gpt etc
    encoder_and_decoder for multimodal , T5 etc
    """

    encoder_or_decoder = 1

    @property
    def encoder_and_decoder(self):
        """Deprecated property - use encoder_or_decoder instead."""
        raise ValueError(
            "ModelType.encoder_and_decoder is deprecated. Please use ModelType.encoder_or_decoder "
            "instead."
        )


class LayerType(enum.Enum):
    """Layer type
    embedding: embedding layer
    loss: loss layer
    encoder: encoder layer, not implemented yet, expect to be used in MLLM models
    decoder: decoder layer
    mtp: multi-token prediction layer, not implemented yet
    """

    embedding = 1
    loss = 2
    encoder = 3
    decoder = 4
    mtp = 5


class AttnType(enum.Enum):
    """Attention type"""

    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    """Attention Mask Type"""

    padding = 1
    causal = 2
    no_mask = 3
    padding_causal = 4
    arbitrary = 5
    causal_bottom_right = 6


class AttnBackend(enum.Enum):
    """Attention Backend"""

    flash = 1
    fused = 2
    unfused = 3
    local = 4
    auto = 5
