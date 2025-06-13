# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed import fleet

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        register_sequence_parallel_allreduce_hooks,
    )
except:
    pass

from ...trainer import Trainer, TrainingArguments
from ...trainer.trainer_utils import ShardingOption
from ...transformers.model_utils import PretrainedModel, unwrap_model
from ...utils.log import logger
from .lora_layers import (
    ColumnParallelLoRALinear,
    ColumnSequenceParallelLoRALinear,
    LoRALinear,
    RowParallelLoRALinear,
    RowSequenceParallelLoRALinear,
)
from .lora_model import LoRAModel


class LoRAGATrainer(Trainer):
    """A Trainer class for Lora-GA gradient estimation."""

    def __init__(self, loraga_init_iters: int, gradient_offload: bool, **kwargs):
        """
        Initialize the Trainer class for Lora-GA gradient estimation.

        Args:
        loraga_init_iters (int): The number of forward and backward process in initializing Lora-GA.
        gradient_offload (bool): Whether to offload gradients to CPU memory.

        """
        super().__init__(**kwargs)
        logger.info(f"Initialization iterations for LoraGA: {loraga_init_iters}")
        self.loraga_init_iters = loraga_init_iters
        self.gradient_offload = gradient_offload

    def estimate_gradient(self, model: PretrainedModel):
        """
        Estimate the gradient of the model on the given dataset
        Args:
            model (PretrainedModel): The base model to be trained.

        Returns:
            dict: A dictionary containing the estimated gradients for each named layer.
                  Note: In tensor parallel mode, the gradients in the dict are not gathered.
        """
        gradient_dict = {}
        logger.info("Estimating gradient for LoraGA.")

        model = self._wrap_model(model)
        dataloader = self.get_train_dataloader()
        iters = 0

        with GradientOffloadHookContext(
            model=model,
            gradient_dict=gradient_dict,
            local_rank=self.args.local_rank,
            loraga_init_iters=self.loraga_init_iters,
            gradient_offload=self.gradient_offload,
        ):
            for batch in dataloader:
                iters += 1
                # Pipeline parallel not supported currently
                self.training_step(model, batch)

                if iters == self.loraga_init_iters:
                    break
        return gradient_dict

    def _wrap_model(self, model):
        """Wrap Model without optimizer, support dp, tp and sharding"""

        if self.args.tensor_parallel_degree > 1 and self.args.sequence_parallel:
            register_sequence_parallel_allreduce_hooks(
                model, self.args.gradient_accumulation_steps, self.args.fuse_sequence_parallel_allreduce
            )

        in_pipeline_parallel_mode = self.args.pipeline_parallel_degree > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_mode = self.args.tensor_parallel_degree > 1
        in_sep_parallel_mode = self.args.sep_parallel_degree > 1
        in_cp_parallel_mode = self.args.context_parallel_degree > 1

        if in_pipeline_parallel_mode:
            raise ValueError("LoRA-GA do not supported pipeline parallel currently.")

        # Multi-gpu training
        if self.args.world_size > 1 and (not self.args.use_hybrid_parallel):
            # MOE use DDP to broadcaset parameters.
            ddp_kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                ddp_kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PretrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                ddp_kwargs["find_unused_parameters"] = not any(
                    hasattr(m, "enable_recompute") and m.enable_recompute for m in model.sublayers(include_self=True)
                )
            else:
                ddp_kwargs["find_unused_parameters"] = True
            model = paddle.DataParallel(model, **ddp_kwargs)

        # sharding
        if in_sharding_parallel_mode:
            # Sharded DDP!
            if self.args.tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)
            if ShardingOption.SHARD_OP in self.args.sharding:
                model = fleet.distributed_model(model)

        if not in_sharding_parallel_mode and (in_tensor_parallel_mode or in_sep_parallel_mode or in_cp_parallel_mode):
            model = fleet.distributed_model(model)

        return model


def get_module_gradient(
    grad_name,
    base_model_prefix,
    gradient_dict,
    base_model_split_mappings,
    tp_degree,
    sharding_degree,
    dp_degree,
    local_rank,
):
    """
    Gather modules gradient in tensor parallel mode.
    Average module gradient in data parallel mode and sharding parallel mode.

    Args:
        grad_name (str): The name of the gradient parameter.
        base_model_prefix (str): The prefix of the base model's parameter names.
        gradient_dict (dict): A dictionary containing the estimated gradients for each named layer.
        base_model_split_mappings (dict): A mapping of model keys to merge functions.
        sharding_degree (int): The sharding parallel degree.
        dp_degree (int): The data parallel degree.
        local_rank (int): The local rank of the current process.

    Returns:
        Tensor: The processed gradient tensor.
    """

    rank_suffix = "_" + str(local_rank)
    local_grad_name = ".".join(grad_name.split(".")[1:]) + ".weight" + rank_suffix
    gradient = gradient_dict.pop(local_grad_name)._copy_to(paddle.framework._current_expected_place(), False)

    is_fleet_init = True
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        sharding_parallel_group = hcg.get_sharding_parallel_group()
        data_parallel_group = hcg.get_data_parallel_group()
    except:
        is_fleet_init = False

    if tp_degree > 1:
        # remove prefix and suffix in name
        model_split_key = local_grad_name.split(base_model_prefix)[-1].rsplit(rank_suffix, 1)[0]
        if model_split_key in base_model_split_mappings:
            merge_func = base_model_split_mappings[model_split_key]
            output_tensors = []
            dist.all_gather(output_tensors, gradient, group=model_parallel_group)

            output_tensors = [t if len(t.shape) > 0 else t.reshape_([-1]) for t in output_tensors]
            gradient = merge_func(output_tensors)._copy_to(paddle.framework._current_expected_place(), False)

    # sharding
    if sharding_degree > 1:
        if sharding_parallel_group.nranks > 1:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=sharding_parallel_group)
            gradient /= sharding_parallel_group.nranks

    # dp
    if dp_degree > 1:
        if data_parallel_group.nranks > 1:
            if is_fleet_init:
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=data_parallel_group)
            else:
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient /= data_parallel_group.nranks
    return gradient


def loraga_svd_reinit(
    model: LoRAModel, gradient_dict: dict, stable_gamma: int, training_args: TrainingArguments, **kwargs
) -> None:
    """
    Perform SVD to gradients and reinitialize base model weight and lora adapter weight.

    Args:
        model (LoRAModel): The LoRAModel containing LoRA layers.
        gradient_dict (dict): A dictionary containing the estimated gradients for each named layer.
        stable_gamma (int): A scaling factor for LoRA-GA initialization.
        training_args (TrainingArguments): Training arguments.

    Returns:
        None: Updates the model's weights and LoRA adapter weights in place.
    """
    tensor_parallel_degree = training_args.tensor_parallel_degree
    in_tensor_parallel_mode = tensor_parallel_degree > 1
    lora_split_mapping = None
    base_model_split_mappings = None
    if in_tensor_parallel_mode:
        base_model_split_mappings = model.model._get_tensor_parallel_mappings(config=model.config, is_split=False)

    base_model_prefix = unwrap_model(model).base_model_prefix + "."
    if in_tensor_parallel_mode:
        lora_split_mapping = model._get_tensor_parallel_mappings(model.config)
    loraga_init_dict = {}
    for name, module in model.named_sublayers():
        if isinstance(
            module,
            (
                LoRALinear,
                RowSequenceParallelLoRALinear,
                ColumnSequenceParallelLoRALinear,
                RowParallelLoRALinear,
                ColumnParallelLoRALinear,
            ),
        ):
            # gather gradient if in tensor parallel mode, average gradient if in data parallel mode
            module_gradient = get_module_gradient(
                name,
                base_model_prefix,
                gradient_dict,
                base_model_split_mappings,
                training_args.tensor_parallel_degree,
                training_args.sharding_parallel_degree,
                training_args.data_parallel_degree,
                training_args.local_rank,
            )
            # perform SVD to reinit base model weight and lora adapter weight
            loraga_svd_module(
                name,
                module,
                module_gradient,
                stable_gamma,
                loraga_init_dict,
                in_tensor_parallel_mode,
                lora_split_mapping,
                **kwargs,
            )
    model.reinit_base_model = True
    model.loraga_init_dict = loraga_init_dict


def loraga_svd_module(
    name,
    module,
    grads,
    stable_gamma,
    loraga_init_dict,
    in_tensor_parallel_mode=False,
    lora_split_mapping=None,
    **kwargs
):
    with paddle.no_grad():
        lora_r = module.r

        loraA_name = ".".join(name.split(".")[1:]) + ".lora_A"
        loraB_name = ".".join(name.split(".")[1:]) + ".lora_B"
        # Perform SVD to gradients
        U, S, V = paddle.linalg.svd_lowrank(grads.astype("float32"), q=4 * lora_r, niter=4)

        V = V.T
        # get new low-rank adapter after SVD
        A = U[:, lora_r : 2 * lora_r]
        B = V[:lora_r, :]

        m, n = grads.shape
        # If stable_gamma is not -1, scale the matrices A and B by the square root of the stable_gamma
        if stable_gamma != -1:
            A = A * m**0.25 / stable_gamma**0.5
            B = B * m**0.25 / stable_gamma**0.5
        else:
            A = A / module.scaling
            B = B / module.scaling

        if in_tensor_parallel_mode:
            # split lora adapter weight if in tensor parallel mode
            if module.lora_A.is_distributed and lora_split_mapping is not None:
                split_function = lora_split_mapping[loraA_name]
                A = paddle.to_tensor(split_function(A))
            if module.lora_B.is_distributed and lora_split_mapping is not None:
                split_function = lora_split_mapping[loraB_name]
                B = paddle.to_tensor(split_function(B))
        A = A.astype(module.lora_A.dtype)
        B = B.astype(module.lora_B.dtype)
        loraga_init_dict[loraA_name] = A
        loraga_init_dict[loraB_name] = B
        # reinit lora adapter weight
        module.lora_A.set_value(A)
        module.lora_B.set_value(B)

        offset = module.lora_A @ module.lora_B
        # reinit base model weight
        module.weight.data -= module.scaling * offset


def set_hook_enable(value=False):
    global ENABLE_HOOK
    ENABLE_HOOK = value


def get_hook_enable():
    global ENABLE_HOOK
    return ENABLE_HOOK


class GradientOffloadHookContext:
    """Context manager for offloading gradient memory to CPU."""

    def __init__(
        self,
        model,
        gradient_dict: dict,
        local_rank: int = 0,
        loraga_init_iters: int = 4,
        gradient_offload: bool = False,
        *args,
        **kwargs,
    ):
        self.model = model
        self.gradient_dict = gradient_dict
        self.local_rank = local_rank
        self.loraga_init_iters = loraga_init_iters
        self.gradient_offload = gradient_offload

    def __enter__(self):
        set_hook_enable(True)
        self.register_gradient_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_hook_enable(False)

    def register_gradient_hook(self):
        """Register gradient hooks for all model parameters."""
        for grad_name, param in self.model.named_parameters():
            param._register_backward_hook(
                self.get_record_gradient_hook(self.model, self.gradient_dict, grad_name, param)
            )

    def get_record_gradient_hook(self, model, gradient_dict, grad_name, param):
        """Create a gradient recording hook for a parameter."""

        def record_gradient_hook(*_):
            if get_hook_enable():
                grad = param.grad
                local_grad_name = grad_name.split("_layers.")[-1] + "_" + str(self.local_rank)
                if not param.stop_gradient and grad is not None:
                    if local_grad_name not in gradient_dict:
                        if self.gradient_offload:
                            gradient_dict[local_grad_name] = (grad / self.loraga_init_iters).cpu()
                        else:
                            gradient_dict[local_grad_name] = grad.clone() / self.loraga_init_iters
                    else:
                        if self.gradient_offload:
                            new_grad = (
                                gradient_dict[local_grad_name]._copy_to(
                                    paddle.framework._current_expected_place(), False
                                )
                                + grad / self.loraga_init_iters
                            )
                            gradient_dict[local_grad_name] = new_grad.cpu()
                        else:
                            gradient_dict[local_grad_name] += grad / self.loraga_init_iters
                param.clear_gradient(False)  # release gradient memory

        return record_gradient_hook
