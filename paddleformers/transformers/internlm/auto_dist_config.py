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
    """Generate distributed configuration for InternLM model"""
    if prefix != "":
        assert prefix.endswith(".")

    config = {
        "mp_config": {
            "parallelize_plan": {
                f"{prefix}model.embed_tokens": dist.ColWiseParallel(gather_output=True),
                f"{prefix}model.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                f"{prefix}model.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                f"{prefix}model.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                f"{prefix}model.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                f"{prefix}model.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                f"{prefix}model.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                f"{prefix}model.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                f"{prefix}lm_head.weight": dist.ColWiseParallel(),
            }
        },
    }
    return config
