# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
This file is used for replacing Paddle's native Linear implementations with vendors' customized implementations
"""

import paddle.distributed.fleet.meta_parallel as mpu
from paddle import nn

try:
    from paddle.distributed.fleet.utils import sequence_parallel_utils
except:
    sequence_parallel_utils = None

from ..utils.tools import get_env_device
from .mc2_parallel_linear import MC2ColumnSeqParallelLinear, MC2RowSeqParallelLinear

Linear = nn.Linear
ColumnParallelLinear = mpu.ColumnParallelLinear
RowParallelLinear = mpu.RowParallelLinear

__all__ = [
    "Linear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ColumnSequenceParallelLinear",
    "RowSequenceParallelLinear",
]

try:
    ColumnSequenceParallelLinear = sequence_parallel_utils.ColumnSequenceParallelLinear
    RowSequenceParallelLinear = sequence_parallel_utils.RowSequenceParallelLinear
except:

    class ColumnSequenceParallelLinearPass(object):
        """
        A dummy class for ColumnSequenceParallelLinear, used when the actual class
        cannot be imported from sequence_parallel_utils.
        """

        pass

    class RowSequenceParallelLinearPass(object):
        """
        A dummy class for RowSequenceParallelLinear, used when the actual class
        cannot be imported from sequence_parallel_utils.
        """

        pass

    ColumnSequenceParallelLinear = ColumnSequenceParallelLinearPass
    RowSequenceParallelLinear = RowSequenceParallelLinearPass

if get_env_device() == "npu":
    if MC2ColumnSeqParallelLinear is not None and MC2RowSeqParallelLinear is not None:
        ColumnSequenceParallelLinear = MC2ColumnSeqParallelLinear
        RowSequenceParallelLinear = MC2RowSeqParallelLinear
elif get_env_device() == "xpu":
    try:
        from paddle_xpu.layers.nn import ColumnParallelLinear as XPUColumnParallelLinear
        from paddle_xpu.layers.nn import Linear as XPULinear
        from paddle_xpu.layers.nn import RowParallelLinear as XPURowParallelLinear
        from paddle_xpu.layers.nn.sequence_parallel import (
            XPUColumnSequenceParallelLinear,
            XPURowSequenceParallelLinear,
        )

        Linear = XPULinear
        ColumnParallelLinear = XPUColumnParallelLinear
        RowParallelLinear = XPURowParallelLinear
        ColumnSequenceParallelLinear = XPUColumnSequenceParallelLinear
        RowSequenceParallelLinear = XPURowSequenceParallelLinear
    except ImportError:
        # If paddle_xpu is not installed, just use Paddle's native Linear implementations
        pass
else:
    # By default, use Paddle's native Linear implementations
    pass
