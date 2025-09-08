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

import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "abstract": ["MOELayerBase"],
    "all_gather": ["allgather_async", "reduce_scatter_async", "AlltoAllSmart", "AllGatherAsync"],
    "all_to_all": ["AlltoAll", "AlltoAllAsync"],
    "moe_allgather_layer": ["ReshardCombineWeight", "MOEAllGatherLayerV2"],
    "moe_alltoall_layer": ["GateCombine", "combining"],
    "moe_block": ["create_moe_block", "MoEStatics"],
    "top_gate": [
        "masked_fill",
        "compute_optimal_transport",
        "cast_if_needed",
        "FusedGateDetachMatmul",
        "gate_detach_matmul",
        "TopKGate",
    ],
    "utils": [
        "ReduceScatterGroupOp",
        "AllGatherGroupOp",
        "get_async_loader",
        "hack_offload_wait",
        "all_gather_group",
        "reduce_scatter_group",
        "detach_and_requires_grad_",
        "FakeClone",
        "manual_backward",
        "_parse_moe_group",
    ],
}

if TYPE_CHECKING:
    from .abstract import *
    from .all_gather import *
    from .all_to_all import *
    from .moe_allgather_layer import *
    from .moe_alltoall_layer import *
    from .moe_block import *
    from .topk_gate import *
    from .utils import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
