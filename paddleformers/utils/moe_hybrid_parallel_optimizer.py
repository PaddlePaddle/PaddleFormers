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
import paddle.distributed as dist
from paddle.autograd import no_grad
from paddle.distributed.fleet.base.topology import ParallelMode
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer as HPBase,
)
from paddle.distributed.fleet.utils import timer_helper as timer
from paddle.distributed.fleet.utils.hybrid_parallel_util import unwrap_optimizer
from paddle.distributed.fleet.utils.log_util import logger
from paddle.distributed.fleet.utils.mix_precision_utils import MixPrecisionOptimizer
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm, clip

__all__ = [
    "MoEHybridParallelOptimizer",
]


class MoEHybridParallelClipGrad:
    def __init__(self, clip, hcg, timers=None):
        self._clip = clip
        self._hcg = hcg
        if hasattr(hcg, "get_moe_sharding_parallel_world_size") and hcg.get_moe_sharding_parallel_world_size() > 0:
            # hybrid expert parallel
            self.moe_group = hcg.get_expert_parallel_group()
            self.moe_sharding_group = hcg.get_moe_sharding_parallel_group()

        self.stat = {}  # for logging
        self._timers = timers
        self.processed_steps = 0

    def _global_norm(
        self, global_norm_var_dist, global_norm_var_not_dist, global_norm_var_dist_moe, global_norm_var_not_dist_moe
    ):
        # sharding first
        sharding_flag = self._hcg.get_sharding_parallel_world_size() > 1
        mp_flag = self._hcg.get_model_parallel_world_size() > 1
        pp_flag = self._hcg.get_pipe_parallel_world_size() > 1

        """do comm"""
        logger.info(
            f"before reduce: dist-moe-grad-norm={global_norm_var_dist_moe.item()} "
            f"before reduce: non-dist-moe-grad-norm={global_norm_var_not_dist_moe.item()}"
        )

        if self.moe_sharding_group:
            dist.all_reduce(
                global_norm_var_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_sharding_group,
            )
            dist.all_reduce(
                global_norm_var_not_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_sharding_group,
            )

        if self.moe_group:
            dist.all_reduce(
                global_norm_var_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_group,
            )
            dist.all_reduce(
                global_norm_var_not_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_group,
            )

        if pp_flag:
            paddle.distributed.all_reduce(
                global_norm_var_dist_moe,
                group=self._hcg.get_pipe_parallel_group(),
            )
            paddle.distributed.all_reduce(
                global_norm_var_not_dist_moe,
                group=self._hcg.get_pipe_parallel_group(),
            )

        # add all reduce to get global norm of distributed params_and_grads
        if sharding_flag:
            # norm of mp distributed variable
            if mp_flag:
                # dist should reduce among sharding group、mp group、pp group
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_sharding_parallel_group(),
                )
            # not dist only reduce among sharding group and pp group later
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_sharding_parallel_group(),
            )

        # norm of mp distributed variable
        if mp_flag:
            # dist should reduce among sharding group、mp group、pp group
            paddle.distributed.all_reduce(
                global_norm_var_dist,
                group=self._hcg.get_model_parallel_group(),
            )
            if pp_flag:
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_pipe_parallel_group(),
                )

        # add all reduce to get global norm of non-distributed params_and_grads in groups of pp
        if pp_flag:
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_pipe_parallel_group(),
            )

        logger.info(
            f"after reduce: dist-grad-norm={global_norm_var_dist.item()} "
            f"after reduce: non-dist-grad-norm={global_norm_var_not_dist.item()}"
        )

    @no_grad()
    def _dygraph_clip(self, params_grads):
        if self._timers:
            self._timers("dygraph-clip").start()
        sum_square_dist_fp16 = []
        sum_square_dist_bf16 = []
        sum_square_dist_fp32 = []

        sum_square_dist_moe_fp16 = []
        sum_square_dist_moe_bf16 = []
        sum_square_dist_moe_fp32 = []

        sum_square_not_dist_fp16 = []
        sum_square_not_dist_bf16 = []
        sum_square_not_dist_fp32 = []

        sum_square_not_dist_moe_fp16 = []
        sum_square_not_dist_moe_bf16 = []
        sum_square_not_dist_moe_fp32 = []

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, "need_clip", True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = clip.merge_selected_rows(g)
                merge_grad = clip.get_tensor_from_selected_rows(merge_grad)
            sum_square = clip._squared_l2_norm(merge_grad)

            not_shared_enable = (not hasattr(p, "is_firstly_shared")) or (
                hasattr(p, "is_firstly_shared") and getattr(p, "is_firstly_shared", True)
            )

            is_moe_param = getattr(p, "is_moe_param", False)

            if is_moe_param:
                assert 0
            if not_shared_enable:
                if getattr(p, "no_sync", False):
                    if p.is_distributed:
                        if g.dtype == paddle.float16:
                            sum_square_dist_moe_fp16.append(sum_square)
                        elif g.dtype == paddle.bfloat16:
                            sum_square_dist_moe_bf16.append(sum_square)
                        elif g.dtype == paddle.float32:
                            sum_square_dist_moe_fp32.append(sum_square)
                    else:
                        if g.dtype == paddle.float16:
                            sum_square_not_dist_moe_fp16.append(sum_square)
                        elif g.dtype == paddle.bfloat16:
                            sum_square_not_dist_moe_bf16.append(sum_square)
                        elif g.dtype == paddle.float32:
                            sum_square_not_dist_moe_fp32.append(sum_square)

                elif p.is_distributed:
                    if g.dtype == paddle.float16:
                        sum_square_dist_fp16.append(sum_square)
                    elif g.dtype == paddle.bfloat16:
                        sum_square_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_dist_fp32.append(sum_square)
                else:
                    assert not getattr(
                        p, "no_sync", False
                    ), f"moe param shoud be distributed, got: {p.name}, shape={p.shape}"
                    if g.dtype == paddle.float16:
                        sum_square_not_dist_fp16.append(sum_square)
                    if g.dtype == paddle.bfloat16:
                        sum_square_not_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_not_dist_fp32.append(sum_square)
            else:
                assert not getattr(p, "no_sync", False), "MoE cannot handle shared param"

        def add_n_list(tensor_list):
            if not tensor_list:
                return paddle.zeros((1,), dtype=paddle.float32)
            return paddle.add_n(tensor_list).cast(paddle.float32)

        # moe global norm of distributed FP16 params_and_grads
        global_norm_dist_moe_fp16 = add_n_list(
            sum_square_dist_moe_fp16,
        )
        global_norm_not_dist_moe_fp16 = add_n_list(
            sum_square_not_dist_moe_fp16,
        )
        global_norm_dist_fp16 = add_n_list(
            sum_square_dist_fp16,
        )
        global_norm_not_dist_fp16 = add_n_list(
            sum_square_not_dist_fp16,
        )

        global_norm_dist_moe_bf16 = add_n_list(
            sum_square_dist_moe_bf16,
        )
        global_norm_not_dist_moe_bf16 = add_n_list(
            sum_square_not_dist_moe_bf16,
        )
        global_norm_dist_bf16 = add_n_list(
            sum_square_dist_bf16,
        )
        global_norm_not_dist_bf16 = add_n_list(
            sum_square_not_dist_bf16,
        )

        global_norm_dist_moe_fp32 = add_n_list(
            sum_square_dist_moe_fp32,
        )
        global_norm_not_dist_moe_fp32 = add_n_list(
            sum_square_not_dist_moe_fp32,
        )
        global_norm_dist_fp32 = add_n_list(
            sum_square_dist_fp32,
        )
        global_norm_not_dist_fp32 = add_n_list(
            sum_square_not_dist_fp32,
        )

        global_norm_var_dist_moe = global_norm_dist_moe_fp16 + global_norm_dist_moe_bf16 + global_norm_dist_moe_fp32

        global_norm_var_not_dist_moe = (
            global_norm_not_dist_moe_fp16 + global_norm_not_dist_moe_bf16 + global_norm_not_dist_moe_fp32
        )

        global_norm_var_dist = global_norm_dist_fp16 + global_norm_dist_bf16 + global_norm_dist_fp32
        global_norm_var_not_dist = global_norm_not_dist_fp16 + global_norm_not_dist_bf16 + global_norm_not_dist_fp32
        result = self._comm_and_clip(
            params_grads,
            global_norm_var_dist,
            global_norm_var_not_dist,
            global_norm_var_dist_moe,
            global_norm_var_not_dist_moe,
        )
        if self._timers:
            self._timers("dygraph-clip").stop()

        return result

    def _comm_and_clip(
        self,
        params_grads,
        global_norm_var_dist,
        global_norm_var_not_dist,
        global_norm_var_dist_moe,
        global_norm_var_not_dist_moe,
    ):

        self._global_norm(
            global_norm_var_dist, global_norm_var_not_dist, global_norm_var_dist_moe, global_norm_var_not_dist_moe
        )

        global_norm_var_fp32 = paddle.sqrt(
            global_norm_var_dist + global_norm_var_not_dist + global_norm_var_dist_moe + global_norm_var_not_dist_moe
        )
        self.stat["global_grad_norm"] = global_norm_var_fp32.astype("float32").item()

        max_global_norm = paddle.full(
            shape=[],
            dtype=global_norm_var_fp32.dtype,
            fill_value=self.clip_norm,
        )
        clip_var = paddle.divide(
            x=max_global_norm,
            y=paddle.maximum(x=global_norm_var_fp32, y=max_global_norm)
            + paddle.full(shape=[], dtype=paddle.float32, fill_value=1.0e-6),
        )
        logger.info(f"hybrid-moe-clip, var={clip_var.item()}, global_norm:{global_norm_var_fp32.item()}")
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        if (
            not isinstance(paddle.framework._current_expected_place(), paddle.CustomPlace)
            or paddle.framework._current_expected_place().get_device_type() == "npu"
        ):
            clip_var_bf16 = paddle.cast(clip_var, paddle.bfloat16)
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, "need_clip", True) is False:
                continue
            if g.dtype == paddle.float16:
                g.multiply_(clip_var_fp16)
            elif g.dtype == paddle.bfloat16:
                if paddle.is_compiled_with_xpu():
                    raise NotImplementedError("BF16 is not supported on XPU now")
                g.multiply_(clip_var_bf16)
            else:
                g.multiply_(clip_var)
            p._reset_grad_inplace_version(True)

        return params_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return params_grads
        return self._dygraph_clip(params_grads)


class MoEHybridParallelOptimizer(HPBase):
    # adapter wrapper for optimizer
    def __init__(self, optimizer, hcg, strategy):
        # Note: Only sharding stage 1 is considered in HybridParallelOptimizer.
        # The sharding stage2 and stage3 optimizers are invoked in other api.
        print(
            f"moe_sharding_degree:{hcg.get_moe_sharding_parallel_world_size()}, sharding_degree:{hcg.get_sharding_parallel_world_size()}, ep_degree:{hcg.get_expert_parallel_world_size()}"
        )
        if hcg.get_moe_sharding_parallel_world_size() > 0:
            split_param = strategy.hybrid_configs["sharding_configs"].split_param
            assert (
                hcg.get_sharding_parallel_world_size() >= 1 and split_param is True
            ), "Hybrid expert parallel only supports ShardingV2 now"
        if hcg.get_sharding_parallel_world_size() > 1:
            split_param = strategy.hybrid_configs["sharding_configs"].split_param
            ShardingOptimizer = DygraphShardingOptimizerV2 if split_param else DygraphShardingOptimizer
            optimizer = ShardingOptimizer(optimizer, hcg)

        self._enable_timer = strategy.hybrid_configs["enable_optimizer_timer"]

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self._timers = timer.get_timers()
        else:
            self._timers = None

        self._inner_opt = optimizer
        self._strategy = strategy
        self._hcg = hcg

        self._use_dp_mode = self._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL

        self._need_dp = self._hcg.get_data_parallel_world_size() > 1

        self._dp_enable = not self._use_dp_mode and self._need_dp

        self._sharding_enable = self._hcg.get_sharding_parallel_world_size() > 1

        self._sep_enable = self._hcg.get_sep_parallel_world_size() > 1

        if isinstance(self._inner_opt._grad_clip, ClipGradByGlobalNorm) and not self._use_dp_mode:
            logger.warning(
                "While using ClipGradByGlobalNorm in TensorParallel, PipelineParallel "
                "or Sharding, the grad clip of original optimizer will be changed."
            )

            inner_opt = unwrap_optimizer(
                self._inner_opt,
                (
                    MixPrecisionOptimizer,
                    DygraphShardingOptimizer,
                    DygraphShardingOptimizerV2,
                ),
            )

            if (
                inner_opt._parameter_list
                and not isinstance(inner_opt._parameter_list[0], dict)
                and len([p for p in inner_opt._parameter_list if hasattr(p, "main_grad")]) > 0
            ):
                inner_opt._grad_clip = MoEHybridParallelClipGrad(inner_opt._grad_clip, hcg, self._timers)
            else:
                inner_opt._grad_clip = MoEHybridParallelClipGrad(inner_opt._grad_clip, hcg, self._timers)
                if inner_opt._parameter_list and isinstance(inner_opt._parameter_list[0], dict):
                    for item in inner_opt._param_groups:
                        if "grad_clip" in item.keys():
                            item["grad_clip"] = MoEHybridParallelClipGrad(inner_opt._grad_clip, hcg, self._timers)
        self.processed_steps = 0
        self._minimax_gate_adamw_state = {}
        self._minimax_wrap_gate_fp32_wgrad_optimizer()

    def _minimax_moe_router(self):
        try:
            import paddlefleet.transformer.moe.moe_router as moe_router
        except Exception:
            return None
        return moe_router

    def _minimax_compute_gate_shard(self, full_wg, param_numel):
        total_numel = int(full_wg.numel().item())
        rank = paddle.distributed.get_rank()
        nranks = max(total_numel // param_numel, 1)
        shard_size = total_numel // nranks
        start = rank * shard_size
        end = start + shard_size
        return full_wg.reshape([-1])[start:end].contiguous()

    def _minimax_manual_gate_adamw_step(self, param, grad):
        key = getattr(param, "name", "_gate")
        state = self._minimax_gate_adamw_state.setdefault(
            key,
            {"m": paddle.zeros_like(param), "v": paddle.zeros_like(param), "step": 0},
        )
        state["step"] += 1
        step_t = state["step"]
        beta1, beta2, eps, lr, wd = 0.9, 0.95, 1e-8, 1e-5, 0.1
        m = state["m"]
        v = state["v"]

        m_new = paddle.add(
            paddle.multiply(m, paddle.full([], beta1, dtype="float32")),
            paddle.multiply(grad, paddle.full([], 1.0 - beta1, dtype="float32")),
        )
        v_new = paddle.add(
            paddle.multiply(v, paddle.full([], beta2, dtype="float32")),
            paddle.multiply(
                paddle.multiply(grad, grad),
                paddle.full([], 1.0 - beta2, dtype="float32"),
            ),
        )
        m_hat = paddle.divide(m_new, paddle.full([], 1.0 - beta1 ** step_t, dtype="float32"))
        v_hat = paddle.divide(v_new, paddle.full([], 1.0 - beta2 ** step_t, dtype="float32"))
        denom = paddle.add(paddle.sqrt(v_hat), paddle.full([], eps, dtype="float32"))
        p_decayed = paddle.multiply(param, paddle.full([], 1.0 - lr * wd, dtype="float32"))
        p_new = paddle.subtract(
            p_decayed,
            paddle.multiply(
                paddle.full([], lr, dtype="float32"),
                paddle.divide(m_hat, denom),
            ),
        )
        paddle.assign(m_new, m)
        paddle.assign(v_new, v)
        paddle.assign(p_new, param)

    def _minimax_wrap_gate_fp32_wgrad_optimizer(self):
        moe_router = self._minimax_moe_router()
        if moe_router is None:
            return
        inner_apply_opt = getattr(self._inner_opt, "_apply_optimize", None)
        if inner_apply_opt is None or getattr(inner_apply_opt, "_minimax_gate_fp32_wgrad", False):
            return

        import functools
        import types

        @functools.wraps(inner_apply_opt)
        def _apply_optimize_with_gate_fp32(
            inner_self,
            loss,
            startup_program,
            params_grads,
            param_group_idx=0,
        ):
            gate_param = None
            gate_grad = None
            if isinstance(params_grads, list):
                full_wg = moe_router._minimax_peek_router_gate_fp32_wgrad()
                if full_wg is not None:
                    dist.all_reduce(full_wg)
                    full_wg = full_wg / dist.get_world_size()
                    full_numel = int(full_wg.numel().item())
                    for i, (param, grad) in enumerate(params_grads):
                        param_numel = 1
                        for dim in param.shape:
                            param_numel *= int(dim)
                        if param_numel != full_numel and (
                            full_numel % param_numel != 0 or full_numel // param_numel not in (1, 2, 4, 8)
                        ):
                            continue
                        gate_param = param
                        gate_grad = (
                            full_wg
                            if param_numel == full_numel
                            else self._minimax_compute_gate_shard(full_wg, param_numel)
                        )
                        params_grads = params_grads[:i] + params_grads[i + 1 :]
                        break
            try:
                result = inner_apply_opt(
                    loss,
                    startup_program,
                    params_grads,
                    param_group_idx=param_group_idx,
                )
            finally:
                moe_router._minimax_clear_router_gate_fp32_wgrad()
            if gate_param is not None and gate_grad is not None:
                self._minimax_manual_gate_adamw_step(gate_param, gate_grad)
            return result

        _apply_optimize_with_gate_fp32._minimax_gate_fp32_wgrad = True
        self._inner_opt._apply_optimize = types.MethodType(_apply_optimize_with_gate_fp32, self._inner_opt)
