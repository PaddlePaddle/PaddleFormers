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
MoE (Mixture of Experts) module for PaddleFleet.

This module provides implementations for Mixture of Experts layers,
including communication mechanisms, routers, experts, and dispatchers.
"""

# Fused A2A operations
from .fused_a2a import (
    CombineNode,
    DeepEPCombine,
    DeepEPDispatch,
    DispatchNode,
    fused_combine,
    fused_dispatch,
)

# MoE experts
from .moe_expert import StandardMLPExpert

# MoE layer and router
from .moe_layer import MoELayer, MoESublayers
from .moe_router import StandardMoERouter
from .moe_shared_expert import StandardMLPSharedExpert

# MoE utilities
from .moe_utils import AddAuxiliaryLoss, _AllToAll

# MoE token dispatcher
from .token_dispatcher import (
    AllToAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
    _DeepEPManager,
    _DispatchManager,
)

__all__ = [
    # Fused A2A
    "DeepEPDispatch",
    "DeepEPCombine",
    "DispatchNode",
    "CombineNode",
    "fused_dispatch",
    "fused_combine",
    # Experts
    "StandardMLPExpert",
    "StandardMLPSharedExpert",
    # Layer and sublayers
    "MoELayer",
    "MoESublayers",
    # Router
    "StandardMoERouter",
    # Utilities
    "AddAuxiliaryLoss",
    "_AllToAll",
    # Token Dispatcher
    "MoETokenDispatcher",
    "MoEFlexTokenDispatcher",
    "AllToAllTokenDispatcher",
    "_DispatchManager",
    "_DeepEPManager",
]
