# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2023 DeepSeek. All rights reserved.
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
""" DeepSeekV3 model configuration"""
from ..deepseek_v2.configuration import DeepseekV2Config

__all__ = [
    "DeepseekV3Config",
]


class DeepseekV3Config(DeepseekV2Config):
    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
