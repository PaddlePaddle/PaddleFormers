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

"""
XPU distributed utils
"""

try:
    from paddle_xpu.layers.nn import (
        ColumnParallelLinear,
        RowParallelLinear,
        parallel_matmul,
    )
    from paddle_xpu.layers.nn.sequence_parallel import (
        ColumnSequenceParallelLinear,
        GatherOp,
        RowSequenceParallelLinear,
        ScatterOp,
    )
except ImportError:
    from paddle.distributed.fleet.meta_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        ColumnSequenceParallelLinear,
        GatherOp,
        RowSequenceParallelLinear,
        ScatterOp,
    )

    parallel_matmul = None

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ColumnSequenceParallelLinear",
    "RowSequenceParallelLinear",
    "GatherOp",
    "ScatterOp",
]
