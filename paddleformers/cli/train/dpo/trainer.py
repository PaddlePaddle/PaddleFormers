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
DPO Trainer for Ernie-MoE model with enhanced distributed training support.
"""

from functools import partial

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet.utils.sequence_parallel_utils import register_sequence_parallel_allreduce_hooks
from paddleformers.trainer import Trainer
from paddleformers.trl import DPOTrainer
from paddleformers.utils.log import logger

from ernie.callbacks import SPGradSyncCallback
from ernie.moe.distributed.hybrid_parallel_optimizer import (
    HybridParallelClipGrad as MoEHybridParallelClipGrad,
)
from ernie.moe.moe_clip import ClipGradForMOEByGlobalNorm


class ErnieMoEDPOTrainer(DPOTrainer):
    """
    Custom DPO trainer class for Ernie-MoE model with enhanced distributed training support.
    """

    def _wrap_model(self, model, training=True):
        """Wrap model."""
        model = super()._wrap_model(model, training)

        def enable_sequence_parallel(_model):
            if self.args.tensor_parallel_degree > 1 and self.args.sequence_parallel:
                if self.args.use_sp_callback:
                    self.add_callback(SPGradSyncCallback(_model._layers))
                else:
                    register_sequence_parallel_allreduce_hooks(
                        _model, self.args.gradient_accumulation_steps, self.args.fuse_sequence_parallel_allreduce
                    )

        enable_sequence_parallel(model)
        return model

    def create_optimizer(self, lr_scheduler=None):
        """
        Create and configure the optimizer for training.

        Args:
            lr_scheduler (Optional): Learning rate scheduler for adjusting the learning rate during training.

        Returns:
            paddle.optimizer.Optimizer: The configured optimizer instance with specified parameters and settings.
        """
        self.static_name_to_dyg_name = {p.name: n for n, p in self.model.named_parameters()}

        if self.optimizer is None:
            if self.optimizer_grouped_parameters is not None:
                optimizer_params = self.optimizer_grouped_parameters
            else:
                optimizer_params = self.model.parameters()

            decay_parameters = [
                p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
            ]

            def apply_decay_param_fun(x):
                return x in decay_parameters

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            if hasattr(optimizer_cls, "_create_master_weight") and self.args.fp16_opt_level == "O2":
                optimizer_kwargs["multi_precision"] = True

            def _get_layer_lrs(x, lr_lower_bound, n_layers):
                """
                Calculate layer-wise learning rates with depth-based scaling.

                Implements a learning rate schedule where layers closer to the input (lower depth)
                get smaller learning rates, while deeper layers get progressively higher rates.
                This follows the common practice that earlier layers typically need finer tuning.

                Args:
                    x (Parameter): The model parameter to calculate learning rate for
                    lr_lower_bound (float): Minimum learning rate (for depth=0 layers)
                    n_layers (int): Total number of transformer layers in the model

                Returns:
                    float: Computed learning rate for the given parameter

                Note:
                    - Special layers (embedding and head) get fixed positions in the depth hierarchy
                    - The depth-to-LR mapping follows a linear interpolation between lower bound and 1.0
                    - TODO: Needs to consider LoRA (Low-Rank Adaptation) parameters in future
                """
                name = self.static_name_to_dyg_name[x.name]
                if "lm_head" in name or "ernie.norm" in name:
                    depth = n_layers + 2
                elif "embed_tokens" in name:
                    depth = 0
                elif self.dpo_config.lora and "lora" in name:
                    if "ernie.layers" in name:
                        depth = int(name.split(".")[3])
                    else:
                        depth = int(name.split(".")[1])
                else:
                    if name.startswith("ernie.layers."):
                        depth = int(name.split(".")[2])
                    else:
                        depth = int(name.split(".")[0])
                return lr_lower_bound + depth / (n_layers + 2) * (1 - lr_lower_bound)

            lr_ratio_func = None
            layerwise_lr_decay_bound = self.args.layerwise_lr_decay_bound
            assert (
                layerwise_lr_decay_bound > 0 and layerwise_lr_decay_bound <= 1
            ), f"layerwise_lr_decay_bound: {layerwise_lr_decay_bound} out of range. should be in (0, 1]"
            if layerwise_lr_decay_bound < 1:
                lr_ratio_func = partial(
                    _get_layer_lrs,
                    lr_lower_bound=layerwise_lr_decay_bound,
                    n_layers=self.model.config.num_hidden_layers,
                )

            if self.args.max_grad_norm <= 0:
                grad_clip = None
            elif self.args.use_expert_parallel and not self.args.use_hybrid_parallel:

                def expert_fn(p):
                    return getattr(p, "no_sync", False)

                grad_clip = ClipGradForMOEByGlobalNorm(
                    self.args.max_grad_norm,
                    is_expert_param_func=expert_fn,
                    moe_group=_get_global_group(),
                )
            else:
                grad_clip = nn.ClipGradByGlobalNorm(self.args.max_grad_norm)

            self.optimizer = optimizer_cls(
                learning_rate=(self.lr_scheduler if lr_scheduler is None else lr_scheduler),
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=optimizer_params,
                weight_decay=self.args.weight_decay,
                grad_clip=grad_clip,
                lr_ratio=lr_ratio_func,
                **optimizer_kwargs,
            )

            if self.args.use_expert_parallel and self.args.use_hybrid_parallel:
                logger.debug('using moe-hybrid-clip under hybrid parallel')
                hcg = fleet.get_hybrid_communicate_group()
                self.optimizer._grad_clip = MoEHybridParallelClipGrad(
                    self.optimizer._grad_clip,
                    hcg,
                    moe_group=hcg.get_data_parallel_group(),
                )

            self.optimizer._dtype = paddle.get_default_dtype()
        return self.optimizer
