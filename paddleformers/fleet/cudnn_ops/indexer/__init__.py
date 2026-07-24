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

from .csa_indexer_bwd_cudnn import csa_indexer_bwd
from .csa_indexer_fwd_cudnn import (
    cudnn_indexer_forward,
    cudnn_indexer_topk,
    cudnn_indexer_topk_fwd,
)
from .docmask_utils import (
    bshd_b1_to_thd,
    shift_scores_to_local_window,
    thd_to_bshd_b1,
    topk_global_to_local,
    topk_local_to_global,
    valid_range_to_counts,
)

__all__ = [
    "csa_indexer_bwd",
    "cudnn_indexer_forward",
    "cudnn_indexer_topk",
    "cudnn_indexer_topk_fwd",
    "bshd_b1_to_thd",
    "shift_scores_to_local_window",
    "thd_to_bshd_b1",
    "topk_global_to_local",
    "topk_local_to_global",
    "valid_range_to_counts",
]
