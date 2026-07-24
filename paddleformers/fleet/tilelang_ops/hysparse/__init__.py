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

"""HySparse block-attention TileLang operators (paper arXiv 2602.03560).

MQA/MLA variant: K/V are a single shared head; block selection is aggregated
across the query group by a group-wise maximum; the sparse branch gathers only
the selected blocks.

Public entry points (enable_hy_sparse_attention=True):
* full layers -> block_score_fa4_attn_fwd (FA4-fused block scoring) +
                 select_topk_blocks
* SWA layers  -> sliding_window_mqa_attention (windowed MQA main path);
                 the sparse gather branch runs on the cuDNN DSA op
                 (paddleformers.fleet.cudnn_ops.block_sparse_mqa_attention_dsa).
"""

from .pipeline import select_topk_blocks
from .swa_attn import sliding_window_mqa_attention


def block_score_fa4_attn_fwd(*args, **kwargs):
    """Lazy wrapper for the FA4-fused block scorer.

    ``block_score_fa4`` imports ``paddlefleet_ops.flash_mask.cute.*``, which is
    only available on SM10/Blackwell (guarded by ``is_flash_mask_available()``).
    Importing it eagerly at package init would break ``from ... import
    sliding_window_mqa_attention`` / ``select_topk_blocks`` on non-Blackwell
    hardware (H20/A100), even for callers that never touch the FA4 full path.
    Defer the FA4-only import to the actual call site.
    """
    from .block_score_fa4 import block_score_fa4_attn_fwd as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "block_score_fa4_attn_fwd",
    "select_topk_blocks",
    "sliding_window_mqa_attention",
]
