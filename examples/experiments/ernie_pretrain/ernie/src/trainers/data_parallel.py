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

import paddle
from paddle import framework
from paddle.distributed import fleet
from paddle.framework import base as imperative_base
from paddle.framework import core, in_dynamic_mode


class DataParallel(paddle.DataParallel):
    def init_reducer(self):
        layers_param = []
        params_set = set()
        for sublayer in self.sublayers():
            for _, param in sublayer.named_parameters(include_sublayers=False):
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                if not isinstance(param, self.var_dtype):
                    raise TypeError("The data type of '%s' must be '%s'" % (param.name, self.var_dtype))
                if param.trainable:
                    layers_param.append((sublayer, param))

        trainable_parameters = list(
            filter(
                lambda x: not getattr(x, "no_sync", False),
                [param for _, param in layers_param],
            )
        )

        assert len(trainable_parameters) > 0, (
            "This model does not have any parameters to train, and " "does not need to use DataParallel"
        )

        def check_layer_sparse(sublayer):
            if isinstance(sublayer, paddle.nn.layer.common.Embedding):
                return sublayer._sparse
            return False

        is_sparse_gradient = [
            check_layer_sparse(sublayer) for sublayer, param in layers_param if not getattr(param, "no_sync", False)
        ]

        if in_dynamic_mode():
            self.group_indices = core.eager_assign_group_by_size(
                trainable_parameters,
                is_sparse_gradient,
                [self.last_comm_buffer_size, self.comm_buffer_size],
            )
            self._reducer = core.EagerReducer(
                trainable_parameters,
                list(reversed(self.group_indices)),
                is_sparse_gradient,
                self.group.process_group,
                [self.last_comm_buffer_size, self.comm_buffer_size],
                self.find_unused_parameters,
            )


@imperative_base.no_grad
@framework.dygraph_only
def sync_dp_moe_params_across_sharding(model: paddle.nn.Layer) -> None:
    hcg = fleet.fleet._hcg
    sharding_parallel_group = hcg.get_sharding_parallel_group()
    src_rank = hcg.get_sharding_parallel_group_src_rank()
    model_vars = []
    for _, param in model._obtain_parameters_buffers().items():
        if not isinstance(param, core.eager.Tensor):
            raise TypeError(f"The data type of '{param.name}' must be core.eager.Tensor")

        if param.type == core.VarDesc.VarType.VOCAB:
            continue

        if getattr(param, "no_sync", False):
            model_vars.append(param.detach())

    if len(model_vars) == 0:
        return

    for var in model_vars:
        var = var.contiguous()
        paddle.distributed.broadcast(var, src=src_rank, group=sharding_parallel_group, sync_op=True)
