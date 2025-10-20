# !/usr/bin/env python3

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

"""
optimizer callback
"""

import importlib.util

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddleformers.trainer.trainer_callback import TrainerCallback


def is_tensorboard_available():
    """
    Check if tensorboard is available.
    """
    return (
        importlib.util.find_spec("tensorboard") is not None
        or importlib.util.find_spec("tensorboardX") is not None
    )


@paddle.no_grad()
def gather_sharding_optimizer_state_stat(args, optimizer, scaler=None):
    """
    Gather sharding optimizer statistics.

    Args:
        optimier.
    """
    eps = 1e-8
    # moment1
    embedding_num_badcases = paddle.to_tensor([0.0])
    outlinear_num_badcases = paddle.to_tensor([0.0])
    intermediate_num_badcases = paddle.to_tensor([0.0])
    gnorm = paddle.to_tensor([0.0])
    gnumel = paddle.to_tensor([0.0])

    if scaler is not None:
        scale_coef = scaler._scale

    # Get gnorm
    for param in optimizer._inner_opt._parameter_list:
        name = param.name
        m_name = name + "_fp32_master_0"
        if m_name not in optimizer._inner_opt._accumulators["moment1"]:
            continue
        if hasattr(param, "main_grad") and param.main_grad is not None:
            assert param.grad is None
            grad = param.main_grad
        else:
            grad = param.grad
        if scaler is not None:
            gnorm += paddle.norm(grad / scale_coef) ** 2
        else:
            gnorm += paddle.norm(grad) ** 2

    if args.use_hybrid_parallel:
        hcg = fleet.get_hybrid_communicate_group()
        # Sharding communication
        sharding_size = hcg.get_sharding_parallel_world_size()
        sharding_group = hcg.get_sharding_parallel_group()
        if sharding_size > 1:
            dist.all_reduce(gnorm, group=sharding_group)
        # MP communication
        mp_size = hcg.get_model_parallel_world_size()
        mp_group = hcg.get_model_parallel_group()
        if mp_size > 1:
            dist.all_reduce(gnorm, group=mp_group)
        # PP communication
        pp_size = hcg.get_pipe_parallel_world_size()
        pp_group = hcg.get_pipe_parallel_group()
        if pp_size > 1:
            dist.all_reduce(gnorm, group=pp_group)

    gnorm = paddle.sqrt(gnorm)

    for param in optimizer._inner_opt._parameter_list:
        name = param.name
        m_name = name + "_fp32_master_0"
        if hasattr(param, "main_grad") and param.main_grad is not None:
            assert param.grad is None
            grad = param.main_grad
        else:
            grad = param.grad
        if name in optimizer._inner_opt._accumulators["moment1"]:
            m_name = name
        if m_name not in optimizer._inner_opt._accumulators["moment1"]:
            continue
        assert grad.shape == optimizer._inner_opt._accumulators["moment1"][m_name].shape
        abs_clipped_grad = paddle.abs(grad) / max(args.max_grad_norm, gnorm)
        if scaler is not None:
            abs_clipped_grad = abs_clipped_grad / scale_coef
        badcases = paddle.logical_and(
            abs_clipped_grad > 100 * eps,
            abs_clipped_grad
            > 50 * paddle.abs(optimizer._inner_opt._accumulators["moment1"][m_name]),
        )
        badcases = paddle.logical_and(
            badcases,
            abs_clipped_grad
            > 10 * paddle.sqrt(optimizer._inner_opt._accumulators["moment2"][m_name]),
        )
        if "ernie_lm_head" in m_name:
            outlinear_num_badcases += badcases.sum()
        elif "embedding" in m_name:
            embedding_num_badcases += badcases.sum()
        else:
            intermediate_num_badcases += badcases.sum()
            gnumel += paddle.numel(grad)

    if args.use_hybrid_parallel:
        # Sharding group communication
        if sharding_size > 1:
            dist.all_reduce(embedding_num_badcases, group=sharding_group)
            dist.all_reduce(intermediate_num_badcases, group=sharding_group)
            dist.all_reduce(outlinear_num_badcases, group=sharding_group)
            dist.all_reduce(gnumel, group=sharding_group)
        # MP communication
        if mp_size > 1:
            dist.all_reduce(embedding_num_badcases, group=mp_group)
            dist.all_reduce(intermediate_num_badcases, group=mp_group)
            dist.all_reduce(outlinear_num_badcases, group=mp_group)
            dist.all_reduce(gnumel, group=mp_group)
        # PP communication
        if pp_size > 1:
            dist.all_reduce(embedding_num_badcases, group=pp_group)
            dist.all_reduce(intermediate_num_badcases, group=pp_group)
            dist.all_reduce(outlinear_num_badcases, group=pp_group)
            dist.all_reduce(gnumel, group=pp_group)

    res = {
        "embedding_num_badcases": embedding_num_badcases.item(),
        "intermediate_num_badcases": intermediate_num_badcases.item(),
        "outlinear_num_badcases": outlinear_num_badcases.item(),
        "gnorm": gnorm.item(),
        "gnumel": gnumel.item(),
    }

    return res


@paddle.no_grad()
def gather_optimizer_state_stat(args, optimizer, scaler=None):
    """
    Gather optimizer statistics.

    Args:
        optimier.
    """
    eps = 1e-8
    # moment1
    embedding_num_badcases = paddle.to_tensor([0.0])
    outlinear_num_badcases = paddle.to_tensor([0.0])
    intermediate_num_badcases = paddle.to_tensor([0.0])
    gnorm = paddle.to_tensor([0.0])
    gnumel = paddle.to_tensor([0.0])

    if scaler is not None:
        scale_coef = scaler._scale

    # Get gnorm
    for param in optimizer._parameter_list:
        name = param.name
        if hasattr(param, "main_grad") and param.main_grad is not None:
            assert param.grad is None
            grad = param.main_grad
        else:
            grad = param.grad
        if scaler is not None:
            gnorm += paddle.norm(grad / scale_coef) ** 2
        else:
            gnorm += paddle.norm(grad) ** 2

    if args.use_hybrid_parallel:
        hcg = fleet.get_hybrid_communicate_group()
        # MP communication
        mp_size = hcg.get_model_parallel_world_size()
        mp_group = hcg.get_model_parallel_group()
        if mp_size > 1:
            dist.all_reduce(gnorm, group=mp_group)
        # PP communication
        pp_size = hcg.get_pipe_parallel_world_size()
        pp_group = hcg.get_pipe_parallel_group()
        if pp_size > 1:
            dist.all_reduce(gnorm, group=pp_group)

    gnorm = paddle.sqrt(gnorm)

    # Get bad case statistics.
    for param in optimizer._parameter_list:
        name = param.name
        m_name = name + "_fp32_master_0"
        if hasattr(param, "main_grad") and param.main_grad is not None:
            assert param.grad is None
            grad = param.main_grad
        else:
            grad = param.grad
        if m_name not in optimizer._accumulators["moment1"]:
            continue
        assert grad.shape == optimizer._accumulators["moment1"][m_name].shape
        abs_clipped_grad = paddle.abs(grad) / max(args.max_grad_norm, gnorm)
        if scaler is not None:
            abs_clipped_grad = abs_clipped_grad / scale_coef
        badcases = paddle.logical_and(
            abs_clipped_grad > 100 * eps,
            abs_clipped_grad
            > 50 * paddle.abs(optimizer._accumulators["moment1"][m_name]),
        )
        badcases = paddle.logical_and(
            badcases,
            abs_clipped_grad
            > 10 * paddle.sqrt(optimizer._accumulators["moment2"][m_name]),
        )
        if "ernie_lm_head" in m_name:
            outlinear_num_badcases += badcases.sum()
        elif "embedding" in m_name:
            embedding_num_badcases += badcases.sum()
        else:
            intermediate_num_badcases += badcases.sum()
            gnumel += paddle.numel(grad)

    if args.use_hybrid_parallel:
        # MP communication
        if mp_size > 1:
            dist.all_reduce(embedding_num_badcases, group=mp_group)
            dist.all_reduce(intermediate_num_badcases, group=mp_group)
            dist.all_reduce(outlinear_num_badcases, group=mp_group)
            dist.all_reduce(gnumel, group=mp_group)
        # PP communication
        if pp_size > 1:
            dist.all_reduce(embedding_num_badcases, group=pp_group)
            dist.all_reduce(intermediate_num_badcases, group=pp_group)
            dist.all_reduce(outlinear_num_badcases, group=pp_group)
            dist.all_reduce(gnumel, group=pp_group)

    res = {
        "embedding_num_badcases": embedding_num_badcases.item(),
        "intermediate_num_badcases": intermediate_num_badcases.item(),
        "outlinear_num_badcases": outlinear_num_badcases.item(),
        "gnorm": gnorm.item(),
        "gnumel": gnumel.item(),
    }

    return res


class OptimizerCallback(TrainerCallback):
    """
    A [`OptimizerCallback`] that gather and save optimizer momentum statistics.

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, args, optimizer, tb_writer=None):
        self.optimizer = optimizer
        self.skip_optimizer_badcases = args.skip_optimizer_badcases
        self.stat = {}
        self.skip_step = False
        self.base_lr = (
            optimizer._learning_rate
            if isinstance(optimizer._learning_rate, float)
            else optimizer._learning_rate.base_lr
        )
        assert isinstance(self.base_lr, float)

    def on_optimizer_begin(self, args, state, control, **kwargs):
        """
        Callback when optimizer begins.
        """
        # Only support amp_master_grad and bf16.
        if not args.amp_master_grad or not args.bf16:
            return

        if args.sharding_parallel_degree > 1:
            stat = gather_sharding_optimizer_state_stat(
                args, self.optimizer, kwargs["scaler"]
            )
        else:
            stat = gather_optimizer_state_stat(args, self.optimizer, kwargs["scaler"])
        self.stat = stat

        if (
            args.skip_optimizer_badcases
            and stat["intermediate_num_badcases"] > 1e-3 * stat["gnumel"]
            and state.global_step > 1000
        ):  # Set threshold be to 1e-3.
            self.skip_step = True
            badcase_ratio = stat["intermediate_num_badcases"] / stat["gnumel"]
            scale_ratio = min(1e-4 / (badcase_ratio + 1e-8), 1.0)
            if args.sharding_parallel_degree > 1:
                if isinstance(self.optimizer._inner_opt._learning_rate, float):
                    self.optimizer._inner_opt._learning_rate = (
                        scale_ratio * self.optimizer._inner_opt._learning_rate
                    )
                else:
                    self.optimizer._inner_opt._learning_rate.base_lr = (
                        scale_ratio * self.optimizer._inner_opt._learning_rate.base_lr
                    )
            else:
                if isinstance(self.optimizer._learning_rate, float):
                    self.optimizer._learning_rate = (
                        scale_ratio * self.optimizer._learning_rate
                    )
                else:
                    self.optimizer._learning_rate.base_lr = (
                        scale_ratio * self.optimizer._learning_rate.base_lr
                    )
        self.stat["skip_step"] = int(self.skip_step)

    def on_optimizer_end(self, args, state, control, **kwargs):
        """
        Callback when optimizer ends.
        """
        # Set `base_lr` back to default value.
        if self.skip_step:
            if args.sharding_parallel_degree > 1:
                if isinstance(self.optimizer._inner_opt._learning_rate, float):
                    self.optimizer._inner_opt._learning_rate = self.base_lr
                else:
                    self.optimizer._inner_opt._learning_rate.base_lr = self.base_lr
            else:
                if isinstance(self.optimizer._learning_rate, float):
                    self.optimizer._learning_rate = self.base_lr
                else:
                    self.optimizer._learning_rate.base_lr = self.base_lr

        self.skip_step = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Write to logs.
        """
        assert logs is not None, "logs should not be None."
        for k, v in self.stat.items():
            if k not in logs:
                logs[k] = v
