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

from ..utils.lazy_import import _LazyModule

import_structure = {
    "attention": ["AttentionInterface", "ALL_ATTENTION_FUNCTIONS"],
    "criterion": ["LossInterface", "ALL_LOSS_FUNCTIONS", "CriterionLayer"],
    "attention.eager_attention": ["eager_attention_forward"],
    "attention.flashmask_attention": ["flashmask_attention_forward"],
    "attention.interface": ["AttentionInterface", "ALL_ATTENTION_FUNCTIONS"],
    "attention.sdpa_attention": ["sdpa_attention_forward"],
    "attention.utils": ["repeat_kv"],
    "criterion.dpo_loss": ["dpo_preprocess_inputs", "dpo_logps", "cal_dpo_loss", "dpo_loss_forward"],
    "criterion.interface": ["LossInterface", "ALL_LOSS_FUNCTIONS", "CriterionLayer"],
    "criterion.kto_loss": ["kto_preprocess_inputs", "_nested_gather", "kto_logps", "kto_loss", "kto_loss_forward"],
    "criterion.loss_utils": ["calc_lm_head_logits", "subbatch"],
    "criterion.sft_loss": [
        "sft_preprocess_inputs",
        "sft_postprocess_loss",
        "sft_loss_forward",
    ],
    "moe.abstract": ["MOELayerBase"],
    "moe.all_gather": ["allgather_async", "reduce_scatter_async", "AlltoAllSmart", "AllGatherAsync"],
    "moe.all_to_all": ["AlltoAll", "AlltoAllAsync"],
    "moe.moe_allgather_layer": ["ReshardCombineWeight", "MOEAllGatherLayerV2"],
    "moe.moe_alltoall_layer": ["GateCombine", "combining"],
    "moe.moe_block": ["create_moe_block", "MoEStatics"],
    "moe.top_gate": [
        "masked_fill",
        "compute_optimal_transport",
        "cast_if_needed",
        "FusedGateDetachMatmul",
        "gate_detach_matmul",
        "TopKGate",
    ],
    "moe.utils": [
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
    "activation": ["ACT2FN", "ClassInstantier", "ACT2CLS"],
    "embedding": ["Embedding"],
    "general": ["GeneralInterface"],
    "linear": ["Linear"],
    "lm_head": ["LMHead"],
    "mlp": ["MLP"],
    "norm": ["Norm", "LayerNorm", "RMSNorm"],
    "pp_model": ["GeneralModelForCausalLMPipe"],
}

if TYPE_CHECKING:
    from .activation import *
    from .attention import *
    from .criterion import *
    from .embedding import *
    from .general import *
    from .linear import *
    from .lm_head import *
    from .mlp import *
    from .moe import *
    from .norm import *
    from .pp_model import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
