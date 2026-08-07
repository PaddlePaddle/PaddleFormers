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

from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


# Note: This is only a stub at the moment. This will be expanded in follow-up changes.
class VisionLayer(FleetLayer):
    """Base vision module that has common helper functions used across CLIP, ViT, etc.

    Args:
        config (TransformerConfig): Input transformer config for the model
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
