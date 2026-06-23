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

from .fused_mla_yarn_rope_apply import (
    fused_apply_mla_rope_for_kv,
    fused_apply_mla_rope_for_q,
)
from .mla_rope_inplace_fusion import fused_apply_mla_rope_inplace
from .moe_topk_fusion import MoETopkFusion, routing_map_fusion_forward
from .rms_norm_fusion import RMSNormFusionTriton
from .sigmoid_gate_fusion import SigmoidGateFusionTriton
from .ue8m0_scale_transpose_fusion import (
    FuseStackUe8m0ScaleTransposeTriton,
    fuse_stack_ue8m0_scale_transpose,
)

__all__ = [
    "RMSNormFusionTriton",
    "MoETopkFusion",
    "routing_map_fusion_forward",
    "SigmoidGateFusionTriton",
    "FuseStackUe8m0ScaleTransposeTriton",
    "fuse_stack_ue8m0_scale_transpose",
    "fused_apply_mla_rope_for_kv",
    "fused_apply_mla_rope_for_q",
    "fused_apply_mla_rope_inplace",
]
