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

import paddle.distributed as dist


def get_dist_config(model, prefix=""):
    if prefix != "":
        assert prefix.endswith(".")
    config = {
        "sp_config": {
            "parallelize_plan": {
                f"{prefix}qwen.wte": [
                    dist.RowWiseParallel(),
                    dist.SequenceParallelBegin(),
                ],
                f"{prefix}qwen.h.*.attn.c_attn": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.attn.c_proj": dist.RowWiseParallel(),
                f"{prefix}qwen.h.*.attn": dist.SequenceParallelDisable(),
                f"{prefix}qwen.h.*.mlp.gate_up_fused_proj": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.mlp.w1": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.mlp.w2": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.mlp.c_proj": dist.RowWiseParallel(),
                f"{prefix}qwen.h.*.mlp": dist.SequenceParallelDisable(need_transpose=False),
                f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                f"{prefix}lm_head": dist.SequenceParallelEnd(),
            }
        },
        "mp_config": {
            "parallelize_plan": {
                f"{prefix}qwen.wte": dist.RowWiseParallel(),
                f"{prefix}qwen.h.*.attn.c_attn": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.attn.c_proj": dist.RowWiseParallel(),
                f"{prefix}qwen.h.*.mlp.gate_up_fused_proj": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.mlp.w1": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.mlp.w2": dist.ColWiseParallel(),
                f"{prefix}qwen.h.*.mlp.c_proj": dist.RowWiseParallel(),
                f"{prefix}lm_head.weight": dist.ColWiseParallel(),
            }
        },
        "pp_config": {
            "split_spec": f"{prefix}qwen.h",
            "global_spec": f"{prefix}qwen.global_layer",
        },
    }

    return config
