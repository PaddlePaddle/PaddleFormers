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
from paddle import _C_ops
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.optimizer import Optimizer

from .sharding_io import to_device


def offload(tensor):
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPinnedPlace()
    else:
        place = paddle.CPUPlace()

    new_tensor = to_device(tensor, place)
    assert new_tensor is tensor, "to_device must be inplace operation"


def reload(tensor):
    new_tensor = to_device(tensor)
    assert new_tensor is tensor, "to_device must be inplace operation"


def hack_offload_optimizer(mode=None):
    if mode == "eb5":
        return hack_offload_optimizer_eb5()

    # Step 1: mock _add_accumulator
    origin_add_accumulator = getattr(Optimizer, "_add_accumulator")

    def new_add_accumulator(self, *args, **kwargs):
        x = origin_add_accumulator(self, *args, **kwargs)
        offload(x)
        return x

    setattr(Optimizer, "_add_accumulator", new_add_accumulator)

    # Step 2: mock _C_ops.adamw_ and _C_ops.adamw
    for name in ["adam_", "adamw_"]:
        origin_op = getattr(_C_ops, name)

        def new_opt_op(*args):
            for arg in args:
                if isinstance(arg, paddle.Tensor):
                    reload(arg)

            ret = origin_op(*args)

            is_offload_opt = getattr(args[0], "is_offload_opt", True)
            for i, arg in enumerate(args):
                if (
                    i >= 2 and isinstance(arg, paddle.Tensor) and is_offload_opt
                ):  # do not offload parameter and gradient
                    offload(arg)
            return ret

        setattr(_C_ops, name, new_opt_op)

    # Step 3: mock _insert_sync
    opt_type = HybridParallelOptimizer
    origin_insert_sync = getattr(opt_type, "_insert_sync")

    def new_insert_sync(self, sync_var, *args, **kwargs):
        origin_place = sync_var.place
        reload(sync_var)
        ret = origin_insert_sync(self, sync_var, *args, **kwargs)
        is_offload_opt = getattr(sync_var, "is_offload_opt", True)
        if is_offload_opt:
            new_sync_var = to_device(sync_var, origin_place)
        else:
            new_sync_var = sync_var
        assert new_sync_var is sync_var, "to_device must be inplace operation"
        return ret

    setattr(opt_type, "_insert_sync", new_insert_sync)


def hack_offload_optimizer_eb5():
    # Step 1: mock _add_accumulator
    origin_add_accumulator = getattr(Optimizer, "_add_accumulator")

    def new_add_accumulator(self, *args, **kwargs):
        x = origin_add_accumulator(self, *args, **kwargs)
        offload(x)
        return x

    setattr(Optimizer, "_add_accumulator", new_add_accumulator)

    # Step 2: mock _C_ops.adamw_ and _C_ops.adamw
    for name in ["adam_", "adamw_"]:
        origin_op = getattr(_C_ops, name)

        def new_opt_op(*args):
            for arg in args:
                if isinstance(arg, paddle.Tensor):
                    reload(arg)

            ret = origin_op(*args)
            is_offload_opt = getattr(args[0], "is_offload_opt", False)
            for i, arg in enumerate(args):
                if (
                    i >= 2 and isinstance(arg, paddle.Tensor) and is_offload_opt
                ):  # do not offload parameter and gradient
                    offload(arg)
            return ret

        setattr(_C_ops, name, new_opt_op)

    # Step 3: mock _insert_sync
    opt_type = HybridParallelOptimizer
    origin_insert_sync = getattr(opt_type, "_insert_sync")

    def new_insert_sync(self, sync_var, *args, **kwargs):
        origin_place = sync_var.place
        reload(sync_var)
        ret = origin_insert_sync(self, sync_var, *args, **kwargs)
        is_offload_opt = getattr(sync_var, "is_offload_opt", False)
        if is_offload_opt:
            new_sync_var = to_device(sync_var, origin_place)
        else:
            new_sync_var = sync_var
        assert new_sync_var is sync_var, "to_device must be inplace operation"
        return ret

    setattr(opt_type, "_insert_sync", new_insert_sync)
