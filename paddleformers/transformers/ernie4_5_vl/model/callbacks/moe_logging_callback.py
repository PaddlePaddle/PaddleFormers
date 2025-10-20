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
record the grad-norm during MoE training.
"""

import hashlib
import os
import random
from collections import OrderedDict

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddleformers.trainer.trainer import PREFIX_CHECKPOINT_DIR
from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.utils.log import logger

try:
    from paddleformers.trainer.trainer import (
        PADDLE_WEIGHT_FILE_NAME as PADDLE_WEIGHTS_NAME,
    )
except ImportError:
    from paddleformers.utils.env import PADDLE_WEIGHTS_NAME
from paddleformers.transformers.model_utils import _add_variant

from ernie.modeling_moe import Ernie4_5_MoeForCausalLM
from ernie.moe.moe_layer import MOELayer

__all__ = ["GlobalRNGCallback", "MoeLoggingCallback"]


def tensor_md5(tensor):
    """debug use"""
    numpy_array = tensor.numpy()
    array_bytes = numpy_array.tobytes()
    return hashlib.md5(array_bytes).hexdigest()


class GlobalRNGCallback(TrainerCallback):
    """
    Insert the correct global random number generator into the network.
    """

    def on_step_end(self, args, state, control, model, **kwargs):
        """Perform MD5 checksum of parameters between DP and Sharding"""
        isinstance(model, Ernie4_5_MoeForCausalLM), type(model)
        rng = random.Random(state.global_step)

        def _set_global_rng(layer):
            if isinstance(layer, MOELayer):
                layer.rng = rng

        model.apply(_set_global_rng)


class MoeLoggingCallback(TrainerCallback):
    """
    Record grad-norm during MoE training
    """

    def __init__(self, optimizer):
        if isinstance(optimizer, GroupShardedOptimizerStage2):
            optimizer = optimizer._optim
        if optimizer._grad_clip is not None:
            assert hasattr(
                optimizer._grad_clip, "stat"
            ), f"expect clip type to be `ClipGradForMOEByGlobalNorm` or `HybridParallelClipGrad`,\
            got grad-clip-type: {type(optimizer._grad_clip)} optimizer-type:{type(optimizer)}"
        self.optimizer = optimizer
        self.check_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """on log"""
        if self.optimizer._grad_clip is not None:
            for k, v in self.optimizer._grad_clip.stat.items():
                if k not in logs:
                    logs[k] = v

    def on_step_end(self, args, state, control, model, **kwargs):
        """Perform MD5 checksum of parameters between DP and Sharding"""
        self.check_step += 1
        if self.check_step != 5:
            return
        if not dist.is_initialized():
            return
        sharding_group, dp_group = None, None
        if args.use_hybrid_parallel:
            _hcg = fleet.get_hybrid_communicate_group()
            sharding_group = _hcg.get_sharding_parallel_group()
            dp_group = _hcg.get_data_parallel_group()

        p_md5_info = [
            (name, tensor_md5(param)[-5:], getattr(param, "no_sync", False))
            for name, param in model.named_parameters()
        ]

        check_error = False
        if args.use_hybrid_parallel and args.sharding_parallel_degree > 1:
            sd_md5_lst = []
            dist.all_gather_object(sd_md5_lst, p_md5_info, sharding_group)
            for idx, (name, pmd5, no_sync) in enumerate(p_md5_info):
                if set([info[idx][1] for info in sd_md5_lst]) != {pmd5}:
                    logger.error(
                        f"param: {name} md5 is not equal between sharding-group"
                    )
                    check_error = True

        if not args.use_hybrid_parallel or args.data_parallel_degree > 1:
            dp_md5_lst = []
            dist.all_gather_object(dp_md5_lst, p_md5_info, dp_group)
            for idx, (name, pmd5, no_sync) in enumerate(p_md5_info):
                if no_sync:
                    if set([info[idx][1] for info in dp_md5_lst]) == {pmd5}:
                        logger.error(
                            f"param: {name} md5 is not different between dp-group"
                        )
                        check_error = True
                else:
                    if set([info[idx][1] for info in dp_md5_lst]) != {pmd5}:
                        logger.error(f"param: {name} md5 is not equal between dp-group")
                        check_error = True
        assert not check_error, "params md5 check failed"
        logger.info("params md5 check pass")

    def on_save(self, args, state, control, model, **kwargs):
        """Shard-reshard stores additional model state for inference."""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        if (
            getattr(args, "should_save_sharding_stage1_model", False)
            or getattr(args, "save_sharding_stage1_model", False)
        ) and args.sharding_parallel_rank == 0:
            logger.info("save extra moe model weights")
            state_dict = model.state_dict()
            if args.data_parallel_rank > 0:
                filter_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if getattr(v, "no_sync", False):
                        filter_state_dict[k] = v
                state_dict = filter_state_dict
                del filter_state_dict
            paddle.save(
                state_dict,
                os.path.join(
                    output_dir,
                    _add_variant(
                        PADDLE_WEIGHTS_NAME,
                        args.weight_name_suffix.replace("shard00_", ""),
                    ),
                ),
            )
