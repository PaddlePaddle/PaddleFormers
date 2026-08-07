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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig


from paddle.distributed.fleet.meta_parallel import ScheduleNode

from paddleformers.fleet.transformer.layer import FleetLayer


class EmptyLayer(FleetLayer):
    """
    A pass-through layer that performs no operation on its input.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="EmptyLayer")

    def forward(self, x):
        return x
