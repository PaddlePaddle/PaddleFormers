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


from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


def csa_sparse_attn_bwd_cudnn(
    q,  # (total_sq, H, D) bf16
    kv,  # (total_skv, D) bf16
    out,  # (total_sq, H, D) bf16
    dout,  # (total_sq, H, D) bf16
    lse,  # (total_sq, H) fp32
    attn_sink,  # (H,) fp32
    topk_idxs,  # (total_sq, topk) int32, global flat indices
    softmax_scale=None,
    topk_length=None,
):
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.sparse_attention_backward.api import (
        sparse_attention_backward_wrapper,
    )

    # print(f"get here.....")
    result = sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    return result["dq"], result["dkv"], result["d_sink"]
