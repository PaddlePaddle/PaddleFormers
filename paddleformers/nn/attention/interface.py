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
from ..general import GeneralInterface
from .eager_attention import eager_attention_forward
from .flashmask_attention import flashmask_attention_forward
from .sdpa_attention import sdpa_attention_forward

__all__ = ["AttentionInterface"]


class AttentionInterface(GeneralInterface):
    _global_mapping = {
        "eager": eager_attention_forward,
        "sdpa": sdpa_attention_forward,
        "flashmask": flashmask_attention_forward,
    }


ALL_ATTENTION_FUNCTIONS = AttentionInterface()
