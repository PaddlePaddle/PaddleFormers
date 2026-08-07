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

"""cuDNN frontend ops bridged into PaddleFleet."""

from .attn.csa_sparse_attn_bwd_cudnn import csa_sparse_attn_bwd_cudnn
from .block_sparse_mqa_dsa import (
    block_sparse_mqa_attention_dsa,
    is_dsa_available,
)
from .indexer.csa_indexer_bwd_cudnn import csa_indexer_bwd
from .indexer.csa_indexer_fwd_cudnn import (
    cudnn_indexer_forward,
    cudnn_indexer_topk,
    cudnn_indexer_topk_fwd,
)

__all__ = [
    "block_sparse_mqa_attention_dsa",
    "csa_indexer_bwd",
    "csa_sparse_attn_bwd_cudnn",
    "cudnn_indexer_forward",
    "cudnn_indexer_topk",
    "cudnn_indexer_topk_fwd",
    "is_dsa_available",
]
