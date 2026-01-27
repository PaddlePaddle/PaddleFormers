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

from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling_moe import (
    ErnieDecoderLayer,
)
from paddleformers.trainer.trainer_callback import TrainerCallback


class OrthogonalCallback(TrainerCallback):
    def __init__(self, ortho_loss_lambda):
        self.ortho_loss_lambda = ortho_loss_lambda

    def on_optimizer_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]

        def update_(layer):
            if isinstance(layer, ErnieDecoderLayer):

                if not hasattr(layer.mlp, "gate"):
                    return
                gate = layer.mlp.gate
                if hasattr(gate, "weight") and not gate.weight.stop_gradient:
                    if getattr(optimizer, "_all_gather_overlap_forward", None):
                        assert len(gate._forward_pre_hooks) == 1
                        hook_id = list(gate._forward_pre_hooks.keys())[0]
                        gate._forward_pre_hooks[hook_id](gate, inputs=None)
                    assert gate.weight.dtype == paddle.float32, f"got unexpected dtype: {gate.weight.dtype}"
                    weight = gate.weight.detach()
                    weight.stop_gradient = False
                    oloss = gate._cal_orthogonal_loss_opt_each_weight(weight, model.config.moe_group_experts)
                    (oloss_grad,) = paddle.autograd.grad(oloss, weight)
                    with paddle.no_grad():
                        gate.weight.data.add_(-oloss_grad * self.ortho_loss_lambda)
                    gate.weight.stop_gradient = False

        model.apply(update_)
