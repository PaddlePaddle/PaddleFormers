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

__all__ = ["csa_indexer_bwd", "csa_sparse_attn_bwd_cudnn"]


def __getattr__(name):
    if name == "csa_indexer_bwd":
        from .indexer.csa_indexer_bwd_cudnn import csa_indexer_bwd

        globals()[name] = csa_indexer_bwd
        return csa_indexer_bwd
    if name == "csa_sparse_attn_bwd_cudnn":
        from .attn.csa_sparse_attn_bwd_cudnn import csa_sparse_attn_bwd_cudnn

        globals()[name] = csa_sparse_attn_bwd_cudnn
        return csa_sparse_attn_bwd_cudnn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
