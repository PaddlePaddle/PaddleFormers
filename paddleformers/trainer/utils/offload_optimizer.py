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
    elif paddle.is_compiled_with_xpu():
        place = paddle.XPUPinnedPlace()
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

    # Step 4: mock Muon._muon_update and Muon._apply_optimize
    # Muon's _muon_update is pure Python (paddle.lerp + paddle.assign),
    # so it bypasses the _C_ops.adamw_ patch above. We need explicit
    # reload/offload for Muon's momentum_buffer and master_weights.
    try:
        from paddle.optimizer.muon import Muon

        # 4a: Patch _muon_update — per-param momentum offload
        # Note: _muon_update is an instance method, not a staticmethod.
        # It requires self as the first argument.
        origin_muon_update = Muon._muon_update

        def new_muon_update(
            self,
            param,
            grad,
            lr,
            momentum_buffer,
            momentum_beta,
            ns_steps,
            nesterov,
            epsilon,
            weight_decay,
            version,
        ):
            reload(momentum_buffer)
            ret = origin_muon_update(
                self,
                param,
                grad,
                lr,
                momentum_buffer,
                momentum_beta,
                ns_steps,
                nesterov,
                epsilon,
                weight_decay,
                version,
            )
            is_offload_opt = getattr(param, "is_offload_opt", True)
            if is_offload_opt:
                offload(momentum_buffer)
            return ret

        Muon._muon_update = new_muon_update

        # 4b: Patch _apply_optimize — reload/offload master_weights around Muon updates
        origin_muon_apply = Muon._apply_optimize

        def new_muon_apply(self, loss, startup_program, params_grads):
            # Reload master_weights to GPU before Muon update
            # (needed after checkpoint restore where master_weights may be on CPU/pinned)
            mw_dict = getattr(self, "_master_weights", None)
            if mw_dict:
                for param, grad in params_grads:
                    if grad is None:
                        continue
                    mw = mw_dict.get(param.name)
                    if mw is not None and isinstance(mw, paddle.Tensor):
                        reload(mw)

            ret = origin_muon_apply(self, loss, startup_program, params_grads)

            # Offload master_weights back to CPU pinned after Muon update
            if mw_dict:
                for param, grad in params_grads:
                    if grad is None:
                        continue
                    mw = mw_dict.get(param.name)
                    if mw is not None and isinstance(mw, paddle.Tensor):
                        is_offload_opt = getattr(param, "is_offload_opt", True)
                        if is_offload_opt:
                            offload(mw)
            return ret

        Muon._apply_optimize = new_muon_apply

    except ImportError:
        pass


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

    # Step 4: mock Muon._muon_update and Muon._apply_optimize
    # Muon's _muon_update is pure Python (paddle.lerp + paddle.assign),
    # so it bypasses the _C_ops.adamw_ patch above. We need explicit
    # reload/offload for Muon's momentum_buffer and master_weights.
    try:
        from paddle.optimizer.muon import Muon

        # 4a: Patch _muon_update — per-param momentum offload
        # Note: _muon_update is an instance method, not a staticmethod.
        # It requires self as the first argument.
        origin_muon_update = Muon._muon_update

        def new_muon_update(
            self,
            param,
            grad,
            lr,
            momentum_buffer,
            momentum_beta,
            ns_steps,
            nesterov,
            epsilon,
            weight_decay,
            version,
        ):
            reload(momentum_buffer)
            ret = origin_muon_update(
                self,
                param,
                grad,
                lr,
                momentum_buffer,
                momentum_beta,
                ns_steps,
                nesterov,
                epsilon,
                weight_decay,
                version,
            )
            is_offload_opt = getattr(param, "is_offload_opt", True)
            if is_offload_opt:
                offload(momentum_buffer)
            return ret

        Muon._muon_update = new_muon_update

        # 4b: Patch _apply_optimize — reload/offload master_weights around Muon updates
        origin_muon_apply = Muon._apply_optimize

        def new_muon_apply(self, loss, startup_program, params_grads):
            # Reload master_weights to GPU before Muon update
            # (needed after checkpoint restore where master_weights may be on CPU/pinned)
            mw_dict = getattr(self, "_master_weights", None)
            if mw_dict:
                for param, grad in params_grads:
                    if grad is None:
                        continue
                    mw = mw_dict.get(param.name)
                    if mw is not None and isinstance(mw, paddle.Tensor):
                        reload(mw)

            ret = origin_muon_apply(self, loss, startup_program, params_grads)

            # Offload master_weights back to CPU pinned after Muon update
            if mw_dict:
                for param, grad in params_grads:
                    if grad is None:
                        continue
                    mw = mw_dict.get(param.name)
                    if mw is not None and isinstance(mw, paddle.Tensor):
                        is_offload_opt = getattr(param, "is_offload_opt", True)
                        if is_offload_opt:
                            offload(mw)
            return ret

        Muon._apply_optimize = new_muon_apply

    except ImportError:
        pass
