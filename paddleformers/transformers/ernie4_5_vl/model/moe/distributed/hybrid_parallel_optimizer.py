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
"""Hybrid parallel optimizer wrapped for MoE"""

import os

import numpy as np
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
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    unwrap_optimizer,
)
from paddle.distributed.fleet.utils.log_util import get_sync_logger
from paddle.distributed.fleet.utils.mix_precision_utils import MixPrecisionOptimizer
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm, clip
from paddleformers.utils.log import logger

g_profile_optimizer_details_steps = int(os.getenv("FLAGS_profile_optimizer_details_steps", "0"))

__all__ = []


class HybridParallelClipGrad:
    """HybridParallelClipGrad"""

    def __init__(self, clip, hcg, timers=None):
        """
        Initialize gradient clipping for hybrid parallel training.

        Args:
            clip: The base gradient clipping method
            hcg: Hybrid communication group for parallel training
            timers: Optional timer utility for profiling
        """
        self._clip = clip
        self._hcg = hcg
        self.not_sharding_stage1 = True
        # 当 moe-group 为 dp-group 时，需要额外做 gradnorm 通信，其他情况下 `HybridParallelClipGrad` 已经自带了 gradnorm 的通信了
        self.moe_group = hcg.get_data_parallel_group() if hcg.get_data_parallel_world_size() else None
        self.stat = {}  # for logging
        self._timers = timers
        self.processed_steps = 0

    def _global_norm(self, global_norm_var_dist, global_norm_var_not_dist):
        """
        Compute global norm across distributed parameters.

        Args:
            global_norm_var_dist: Norm of distributed parameters
            global_norm_var_not_dist: Norm of non-distributed parameters
        """
        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting to calculate global norm.")
        # sharding first
        sharding_flag = self._hcg.get_sharding_parallel_world_size() > 1
        dp_flag = self._hcg.get_data_parallel_world_size() > 1
        mp_flag = self._hcg.get_model_parallel_world_size() > 1
        pp_flag = self._hcg.get_pipe_parallel_world_size() > 1

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

            # the else branch would suffice, but this branch remains here for number precision backward compatibility
            if not (dp_flag and sharding_flag):
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_check_parallel_group(sharding_flag),
                )
            else:
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

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished calculating global norm.")

        self.processed_steps += 1

    @no_grad()
    def _dygraph_clip(self, params_grads):
        """
        Perform gradient clipping in dynamic graph mode.

        Args:
            params_grads: List of (parameter, gradient) pairs

        Returns:
            List of clipped (parameter, gradient) pairs
        """
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
                    elif g.dtype == paddle.bfloat16:
                        sum_square_not_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_not_dist_fp32.append(sum_square)
            else:
                assert not getattr(p, "no_sync", False), "moe don't know share param"

        def add_n_list(tensor_list):
            if not tensor_list:
                return paddle.to_tensor(np.array(0.0), dtype=paddle.float32)
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
        """
        Perform communication and gradient clipping.

        Args:
            params_grads: List of (parameter, gradient) pairs
            global_norm_var_dist: Norm of distributed parameters
            global_norm_var_not_dist: Norm of non-distributed parameters
            global_norm_var_dist_moe: Norm of distributed MoE parameters
            global_norm_var_not_dist_moe: Norm of non-distributed MoE parameters

        Returns:
            List of clipped (parameter, gradient) pairs
        """
        logger.info(
            f"reducing: dist-moe-grad-norm={global_norm_var_dist_moe.item()} "
            f"non-dist-moe-grad-norm={global_norm_var_not_dist_moe.item()}"
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
        logger.info(
            f"reducing: dist-moe-grad-norm={global_norm_var_dist_moe.item()} "
            f"non-dist-moe-grad-norm={global_norm_var_not_dist_moe.item()}"
        )

        global_norm_var_dist.add_(global_norm_var_dist_moe)
        global_norm_var_not_dist.add_(global_norm_var_not_dist_moe)

        self._global_norm(global_norm_var_dist, global_norm_var_not_dist)

        global_norm_var_fp32 = paddle.sqrt(global_norm_var_dist + global_norm_var_not_dist)
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

        if not isinstance(paddle.framework._current_expected_place(), paddle.CustomPlace):
            clip_var_bf16 = paddle.cast(clip_var, paddle.bfloat16)
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, "need_clip", True) is False:
                continue
            if g.dtype == paddle.float16:
                g.multiply_(clip_var_fp16)
            elif g.dtype == paddle.bfloat16:
                g.multiply_(clip_var_bf16)
            else:
                g.multiply_(clip_var)
            p._reset_grad_inplace_version(True)

        return params_grads

    def __getattr__(self, item):
        """
        Fallback attribute access to the base clip method.

        Args:
            item: Attribute name to access
        """
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        """
        Callable interface for gradient clipping.

        Args:
            params_grads: List of (parameter, gradient) pairs

        Returns:
            List of clipped (parameter, gradient) pairs
        """
        return self._dygraph_clip(params_grads)


class HybridParallelOptimizer(HPBase):
    """HybridParallelOptimizer"""

    # adapter wrapper for optimizer
    def __init__(self, optimizer, hcg, strategy):
        """
        Initialize hybrid parallel optimizer.

        Args:
            optimizer: Base optimizer to wrap
            hcg: Hybrid communication group
            strategy: Training strategy configuration
        """
        # Note: Only sharding stage 1 is considered in HybridParallelOptimizer.
        # The sharding stage2 and stage3 optimizers are invoked in other api.
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

        # NOTE(shenliang03): Because of the pure DataParallel mode, the gradient synchronization
        # is achieved through reducer, so there is no need to call fuse_allreduce in optimizer.
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
                inner_opt._grad_clip = HybridParallelClipGrad(inner_opt._grad_clip, hcg, self._timers)
            else:
                inner_opt._grad_clip = HybridParallelClipGrad(inner_opt._grad_clip, hcg, self._timers)
                if inner_opt._parameter_list and isinstance(inner_opt._parameter_list[0], dict):
                    for item in inner_opt._param_groups:
                        if "grad_clip" in item.keys():
                            item["grad_clip"] = HybridParallelClipGrad(inner_opt._grad_clip, hcg, self._timers)
        self.processed_steps = 0
