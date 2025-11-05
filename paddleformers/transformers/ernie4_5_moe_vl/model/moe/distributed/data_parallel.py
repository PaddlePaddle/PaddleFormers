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
"""data parallel file for moe"""

import paddle

# (TODO: GhostScreaming) It will be removed later.
from paddle.framework import core, in_dynamic_mode


class DataParallel(paddle.DataParallel):
    """Wrap DataParallel for dp moe"""

    def init_reducer(self):
        """
        Initialize the reducer for data parallel training.

        This method:
        1. Collects all trainable parameters across sublayers
        2. Validates parameter types and trainability
        3. Handles sparse gradient cases (e.g., Embedding layers)
        4. Creates parameter groups and initializes the reducer
        """
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

        # NOTE(shenliang03): Here we can only use the attributes to judge whether
        # parameter is sparse(or SelectedRows). The reason is that the sparse message
        # can't be obtained when bp hasn't happened yet. So if layer supports sparse parameter,
        # we should add the layer here like "paddle.nn.layer.common.Embedding".
        def check_layer_sparse(sublayer):
            """check_layer_sparse"""
            if isinstance(sublayer, paddle.nn.layer.common.Embedding):
                return sublayer._sparse
            return False

        is_sparse_gradient = [
            check_layer_sparse(sublayer)
            for sublayer, param in layers_param
            if not getattr(param, "no_sync", False)  # here
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
