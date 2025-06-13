# Copyright 2020-present the HuggingFace Inc. team.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import types

import numpy as np
import paddle
from paddle.common_ops_import import LayerHelper

from ...utils.log import logger


def npu_accelerate_plugin(optimizer):
    """npu_accelerate_plugin uses the flatten_param_grads method to speed up the performance of the model on NPU devices.
    flatten_param_grads method will be added to `step` function of optimizer.

    Args:
        optimizer (`paddle.optimizer.Optimizer`):
            The Optimizer whose `step` method will be modified.
    """
    optimizer.step = types.MethodType(_optimizer_step_with_flatten_param_grads, optimizer)


def _optimizer_step_with_flatten_param_grads(optimizer):
    if not isinstance(optimizer._param_groups[0], dict):
        params_grads = []
        for param in optimizer._param_groups:
            if param.stop_gradient:
                continue
            if param._grad_ivar() is not None:
                grad_var = param._grad_ivar()
                params_grads.append((param, grad_var))

        # currently, only support ClipGradByGlobalNorm and without regularization.
        if isinstance(params_grads, list) and optimizer.regularization is None:
            if optimizer._grad_clip is None or isinstance(optimizer._grad_clip, paddle.nn.ClipGradByGlobalNorm):
                params_grads = _flatten_param_grads(optimizer, params_grads)

        optimizer._apply_optimize(
            loss=None,
            startup_program=None,
            params_grads=params_grads,
            param_group_idx=0,
        )
    else:
        raise RuntimeError("flatten_param_grads is not supported when _param_groups[0] is dict.")


def _flatten_param_grads(optimizer, params_grads):
    optimizer.helper = LayerHelper(optimizer.__class__.__name__)
    need_flatten_params = []
    need_flatten_grads = []
    for p, g in params_grads:
        if g is None:
            continue
        g.persistable = True
        if getattr(p, "need_clip", True) is False or getattr(p, "regularizer", None) is not None:
            logger.warning(
                f"flatten_param_grads=True will be discarded since parameter {p.name}'s need_clip is False or "
                "the regularizer is set."
            )
            return params_grads

        need_flatten_params.append(p)
        need_flatten_grads.append(g)

    shape = [np.prod(p.shape) for p in need_flatten_params]

    flatten_param = optimizer.helper.create_global_variable(
        name="flatten_param",
        persistable=True,
        dtype=need_flatten_params[0].dtype,
        shape=[np.sum(shape)],
        belong_to_optimizer=True,
    )

    flatten_grad = optimizer.helper.create_global_variable(
        name="flatten_grad",
        persistable=True,
        dtype=need_flatten_grads[0].dtype,
        shape=[np.sum(shape)],
        belong_to_optimizer=True,
    )

    flatten_param.stop_gradient = False
    # In the final state of the dynamic graph, the `coalesce_tensor` op
    # does not support passing the output as an input into the op in
    # temporary, so _legacy_C_ops is temporarily used here.
    # `use_align` is set to false, which is different from the behavior
    # under static graphs. `use_align` can be set to true after calling
    # the coalesce_tensor op of the final state (_C_ops).
    paddle._legacy_C_ops.coalesce_tensor(
        need_flatten_params,
        need_flatten_params,
        flatten_param,
        "copy_data",
        True,
        "use_align",
        False,
        "dtype",
        need_flatten_params[0].dtype,
    )

    paddle._legacy_C_ops.coalesce_tensor(
        need_flatten_grads,
        need_flatten_grads,
        flatten_grad,
        "copy_data",
        True,
        "use_align",
        False,
        "dtype",
        need_flatten_grads[0].dtype,
    )
    return [(flatten_param, flatten_grad)]
