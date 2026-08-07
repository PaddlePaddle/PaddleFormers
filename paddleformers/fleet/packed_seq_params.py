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

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle import Tensor


@dataclass
class PackedSeqParams:
    """
    parameters to DotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    """

    qkv_format: str | None = None
    cu_seqlens_q: Tensor | None = None
    cu_seqlens_kv: Tensor | None = None
    cu_seqlens_q_padded: Tensor | None = None
    cu_seqlens_kv_padded: Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_kv: int | None = None
    total_seqlen_q: int | None = None
    total_seqlen_kv: int | None = None
