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

import paddle
from paddleformers.trainer.trainer_callback import TrainerCallback

from ernie.modeling_moe import Ernie4_5_DecoderLayer

try:
    from ernie.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}


class OrthogonalCallback(TrainerCallback):
    """
    A [`OptimizerCallback`] that gather and save optimizer momentum statistics.

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, ortho_loss_lambda):
        self.ortho_loss_lambda = ortho_loss_lambda

    def on_optimizer_end(self, args, state, control, **kwargs):
        """
        Callback when optimizer ends.
        """
        # Set `base_lr` back to default value.
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        lr = optimizer.get_lr()

        def _varbase_help(param, tmp_tensor):
            tmp_tensor._share_buffer_to(param)
            tmp_tensor._clear()

        def update_(layer):
            if isinstance(layer, Ernie4_5_DecoderLayer):

                log = {}
                gate = layer.mlp.gate
                if (
                    hasattr(gate, "weight") and not gate.weight.stop_gradient
                ):  # 文本gate
                    assert (
                        gate.weight.dtype == paddle.float32
                    ), f"got unexpected dtype: {gate.weight.dtype}"
                    oloss = gate._cal_orthogonal_loss_opt_each_weight(
                        gate.weight, model.config.moe_group_experts
                    )
                    (oloss_grad,) = paddle.autograd.grad(oloss, gate.weight)
                    with paddle.no_grad():
                        gate.weight.data.add_(-oloss_grad * lr * self.ortho_loss_lambda)

                    gate.weight.stop_gradient = False
                    log[f"orthogonal_loss_layer_{layer.layer_idx}"] = oloss
                    prefix = "lm"

                    global_training_logs.update(
                        **log,
                        **{
                            k.replace(f"_layer_{layer.layer_idx}", ""): v
                            for k, v in log.items()
                        },
                    )
                    global_training_logs.update(
                        **{
                            prefix + "_" + k.replace(f"_layer_{layer.layer_idx}", ""): v
                            for k, v in log.items()
                        }
                    )
                if hasattr(gate, "weight_1") and not gate.weight_1.stop_gradient:
                    assert (
                        gate.weight_1.dtype == paddle.float32
                    ), f"got unexpected dtype: {gate.weight_1.dtype}"
                    oloss = gate._cal_orthogonal_loss_opt_each_weight(
                        gate.weight_1, False
                    )
                    (oloss_grad,) = paddle.autograd.grad(oloss, gate.weight_1)
                    with paddle.no_grad():
                        gate.weight_1.data.add_(
                            -oloss_grad * lr * self.ortho_loss_lambda
                        )
                    gate.weight_1.stop_gradient = False
                    log[f"orthogonal_loss_layer_{layer.layer_idx}"] = oloss
                    prefix = "mm"

                    global_training_logs.update(
                        **log,
                        **{
                            k.replace(f"_layer_{layer.layer_idx}", ""): v
                            for k, v in log.items()
                        },
                    )
                    global_training_logs.update(
                        **{
                            prefix + "_" + k.replace(f"_layer_{layer.layer_idx}", ""): v
                            for k, v in log.items()
                        }
                    )

        model.apply(update_)
