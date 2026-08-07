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

"""TileLang sparse MQA forward for the "tilelang" CSA sparse-attn backend."""

import paddle

from . import sparse_mqa_fwd


def sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale=None):
    out, lse = sparse_mqa_fwd.sparse_mqa_fwd_interface(
        q, kv, attn_sink, topk_idxs, sm_scale=sm_scale
    )
    if not isinstance(out, paddle.Tensor) or not isinstance(lse, paddle.Tensor):
        raise RuntimeError(
            f"TileLang must return Paddle tensors, got output={type(out)!r}, lse={type(lse)!r}. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    return out, lse
