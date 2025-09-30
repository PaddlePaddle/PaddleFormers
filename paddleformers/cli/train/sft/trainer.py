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
Trainer for Ernie-MoE model with enhanced distributed training support.
"""

import inspect
import os
import random
from collections import OrderedDict
from functools import partial
from typing import Dict

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fleet.utils.sequence_parallel_utils import register_sequence_parallel_allreduce_hooks
from paddleformers.peft import LoRAModel
from paddleformers.trainer import Trainer
from paddleformers.trainer.trainer_utils import OptimizerNames, ShardingOption, has_length
from paddleformers.transformers.model_utils import _add_variant, unwrap_model
from paddleformers.utils import infohub
from paddleformers.utils.batch_sampler import DistributedBatchSampler
from paddleformers.utils.env import PADDLE_OPTIMIZER_NAME, PADDLE_WEIGHTS_NAME
from paddleformers.utils.log import logger

try:
    from paddleformers.quantization.quantization_linear import QuantizationLinear
except:
    QuantizationLinear = None

# moe hack
from ernie.callbacks import SPGradSyncCallback
from ernie.moe.distributed.data_parallel import DataParallel as MoEDDP
from ernie.moe.distributed.hybrid_parallel_optimizer import (
    HybridParallelClipGrad as MoEHybridParallelClipGrad,
)
from ernie.moe.moe_clip import ClipGradForMOEByGlobalNorm
from ernie.utils.moe_utils import distributed_optimizer_for_moe


def is_dp_group_support_in_group_sharded_parallel():
    """
    Check if 'dp_group' parameter is supported in group_sharded_parallel function.

    Returns:
        bool: True if 'dp_group' is a valid parameter, False otherwise.
    """
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())


class ErnieMoETrainer(Trainer):
    """
    Custom trainer class for Ernie-MoE model with enhanced distributed training support.
    """

    def __init__(self, data_args, do_generation: bool, **kwargs):
        """
        Initialize ErnieMoETrainer.

        Args:
            data_args: Dataset configuration arguments.
            do_generation (bool): Flag to enable generation mode.
            **kwargs: Additional keyword arguments for base Trainer class.
        """
        super().__init__(**kwargs)
        self.data_args = data_args
        self.do_generation = do_generation
        self.data_seed = kwargs.pop("data_seed", None)

    def prediction_pipeline_step_with_logits_acc(
        self,
        *args,
        **kwargs,
    ):
        """
        Pipeline step for prediction with logits accuracy calculation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: Contains:
                - loss (Tensor): Computed loss value.
                - preds (Tensor): Predicted indices from logits.
                - weight (Tensor): Weights from logits.
                - labels (Tensor): Ground truth labels.
        """
        loss, _, labels = self.prediction_pipeline_step(*args, **kwargs)
        if "pp_preds" in infohub:
            preds = paddle.concat(infohub["pp_preds"], axis=0)
            weight = paddle.concat(infohub["pp_preds_w"], axis=0)
            infohub["pp_preds"] = []
            infohub["pp_preds_w"] = []

            return (loss, (preds, weight), labels)
        return (loss, None, labels)

    def _wrap_model(self, model, training=True):
        """
        Wrap model with distributed training components.

        Args:
            model: Model to wrap.
            training (bool): Whether in training mode. Defaults to True.

        Returns:
            Model: Wrapped model with distributed training components.
        """
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Note: in paddle.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Mixed precision training
        if training and self.do_grad_scaling:  # self.args.fp16_opt_level=="O2":
            # model, self.optimizer
            decorated = paddle.amp.decorate(
                models=model,
                optimizers=self.optimizer,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                excluded_layers=QuantizationLinear,
            )

            if self.optimizer is None:
                model = decorated
            else:
                model, self.optimizer = decorated

        def enable_sequence_parallel(_model):
            if self.args.tensor_parallel_degree > 1 and self.args.sequence_parallel:
                if self.args.use_sp_callback:
                    self.add_callback(SPGradSyncCallback(_model._layers))
                else:
                    register_sequence_parallel_allreduce_hooks(
                        _model, self.args.gradient_accumulation_steps, self.args.fuse_sequence_parallel_allreduce
                    )

        if self.args.world_size == 1:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)
                assert self.optimizer is not None, "optimizer is empty!"
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

        # Multi-gpu training
        if self.args.use_expert_parallel:
            logger.debug("using moe ddp, hack Paddle")  # TODO move this into paddle
            paddle.DataParallel = MoEDDP

        in_pipeline_parallel_mode = self.args.pipeline_parallel_degree > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_mode = self.args.tensor_parallel_degree > 1

        # Multi-gpu training
        if (
            self.args.world_size > 1
            and not self.args.use_hybrid_parallel
            and not (in_pipeline_parallel_mode or in_sharding_parallel_mode or in_tensor_parallel_mode)
        ):
            model = paddle.DataParallel(model)
            # Distributed training (should be after fp16 initialization)

            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)
                assert self.optimizer is not None, "optimizer is empty!"
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

        # Pipeline mode
        if in_pipeline_parallel_mode:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
            # hack for pipeline model mini batch to batch
            # need batter solution @ZHUI
            # make batch_fn compatible for fleet.distributed_model decorate.
            prepare_pipeline_inputs_func = (
                model._prepare_pipeline_inputs_func if hasattr(model, "_prepare_pipeline_inputs_func") else None
            )
            if isinstance(model, LoRAModel):
                model = model.model
            model = fleet.distributed_model(model)
            if prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func
            else:

                def _prepare_pipeline_inputs_func(inputs):
                    first_stage_keys = [
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                    ]
                    last_stage_keys = ["labels"]

                    def get_expected_keys(inputs, keys):
                        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
                        if len(ret) == 1:
                            ret = ret[0]
                        return ret

                    if type(inputs) is dict or type(inputs) is OrderedDict:
                        return [
                            get_expected_keys(inputs, first_stage_keys),
                            get_expected_keys(inputs, last_stage_keys),
                        ]

                    keys = list(inputs[0].keys())
                    inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
                    return [
                        get_expected_keys(inputs_batch, first_stage_keys),
                        get_expected_keys(inputs_batch, last_stage_keys),
                    ]

                logger.warning(
                    "Using default prepare pipeline inputs func, only support input_ids and labels as inputs."
                )
                model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func

            enable_sequence_parallel(model)

            assert self.optimizer is not None, "Pipeline mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = distributed_optimizer_for_moe(self.optimizer, self.args.use_expert_parallel)

        # No pipeline mode, sharding only
        if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
            # Sharded DDP!
            if self.args.tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)
                enable_sequence_parallel(model)

            if ShardingOption.SHARD_OP in self.args.sharding:
                if self.args.amp_master_grad:
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                model = fleet.distributed_model(model)
                if self.args.amp_master_grad:
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer = distributed_optimizer_for_moe(self.optimizer, self.args.use_expert_parallel)
            else:
                # sync params (broadcast) buffers in dp group, no quite understanding here.
                if (
                    not is_dp_group_support_in_group_sharded_parallel() or self.args.use_expert_parallel
                ) and self.args.data_parallel_degree > 1:
                    from paddle.distributed.parallel import sync_params_buffers

                    hcg = fleet.get_hybrid_communicate_group()
                    dp_group = hcg.get_data_parallel_group()
                    sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

                cpu_offload = ShardingOption.OFFLOAD in self.args.sharding
                assert self.optimizer is not None, "optimizer is empty!"
                level = None
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                    level = "os_g"
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    level = "p_g_os"

                from paddle.distributed.sharding import group_sharded_parallel

                # add dp_group and exclude_layer params
                # https://www.paddlepaddle.org.cn/
                # documentation/docs/zh/develop/
                # api/paddle/distributed/sharding/group_sharded_parallel_cn.html#group-sharded-parallel
                extra_kwargs = {}
                if is_dp_group_support_in_group_sharded_parallel() and not self.args.use_expert_parallel:
                    extra_kwargs["dp_group"] = self.dp_group
                    extra_kwargs["exclude_layer"] = ["GroupNorm"]

                if self.args.amp_master_grad:
                    assert (
                        self.args.data_parallel_degree == 1
                    ), "Sharding stage 2 / Sharding stage 3 main grad is not compatible with dp for now."
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

                model, optimizer, _ = group_sharded_parallel(
                    model,
                    self.optimizer,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding and self.args.amp_master_grad:
                    assert hasattr(optimizer, "use_main_grad"), (
                        "Current installed paddle doesn't support sharding stage 2 with main grad, "
                        "please upgrade your paddle (using nightly version)."
                    )

                sharding_parallel_config = set(self.args.sharding_parallel_config.split(" "))
                if level == "os_g" and "enable_stage2_overlap" in sharding_parallel_config:
                    model._set_reduce_overlap(True)
                    optimizer._set_broadcast_overlap(True, model)
                self.optimizer = optimizer

        # pure tesnor parallel mode, no pipeline_parallel, no sharding.
        if not in_pipeline_parallel_mode and not in_sharding_parallel_mode and in_tensor_parallel_mode:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
            model = fleet.distributed_model(model)
            model.accumulate_steps = self.args.gradient_accumulation_steps
            assert self.optimizer is not None, "Tensor parallel mode need decorate optimizer, pelease init optimizer."
            enable_sequence_parallel(model)
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = distributed_optimizer_for_moe(self.optimizer, self.args.use_expert_parallel)

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
            if self.args.optim == OptimizerNames.ADAMW_CUSTOM:
                optimizer_kwargs["quantization_config"] = self.model.config.quantization_config
                optimizer_kwargs["use_lowprecision_moment"] = self.args.use_lowprecision_moment
                optimizer_kwargs["tensorwise_offload_optimizer"] = self.args.tensorwise_offload_optimizer

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

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Perform a single prediction step with model inference.

        Args:
            model: The neural network model used for prediction.
            inputs: Input data for the prediction step.
            prediction_loss_only (bool): Flag indicating whether to return only loss.
            ignore_keys (Optional): Keys to ignore during the prediction process.

        Returns:
            Tuple: A tuple containing loss values, predictions, and labels.
        """
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            try:
                hcg = fleet.get_hybrid_communicate_group()
                model_parallel_group = hcg.get_model_parallel_group()
                tensor_parallel_degree = hcg.get_model_parallel_world_size()
                pipeline_parallel_group = hcg.get_pipe_parallel_group()
                pipeline_parallel_degree = hcg.get_pipe_parallel_world_size()
            except:
                model_parallel_group = None
                tensor_parallel_degree = 1
                pipeline_parallel_group = None
                pipeline_parallel_degree = 1

            # register `pp_accuracy` flag
            if pipeline_parallel_degree > 1:
                infohub["pp_accuracy"] = True
                inputs = self._prepare_inputs(inputs)
                loss, logits, labels = self.prediction_pipeline_step_with_logits_acc(
                    model, inputs, prediction_loss_only, ignore_keys
                )
            else:
                loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            is_tensor_parallel_output = (
                model.config.tensor_parallel_output
                if hasattr(model, "config")
                else model._layers.config.tensor_parallel_output
            )

            if pipeline_parallel_degree > 1:
                if logits is None:
                    preds = None
                    preds_shape = [[]]
                else:
                    # preds: [bz, seq], logits: [bz, seq, part_hidden]
                    vocab_size_part = model._layers.config.vocab_size // tensor_parallel_degree
                    # logits were already argmax in `modeling.py` in pp mode.
                    preds = logits[0]
                    weight = logits[1]
                    # tp group concat
                    if tensor_parallel_degree > 1 and is_tensor_parallel_output:
                        # extract maximum `weight`
                        # weight: [bz, seq], logits: [bz, seq, part_hidden]

                        batch_size, seq_len = preds.shape

                        # indices offset
                        offset = (
                            paddle.arange(tensor_parallel_degree)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .expand([batch_size, seq_len, tensor_parallel_degree])
                            * vocab_size_part
                        )
                        preds = paddle.distributed.collective._c_concat(preds, group=model_parallel_group)
                        preds = preds.reshape([batch_size, -1, seq_len]).transpose([0, 2, 1])
                        preds = preds + offset
                        # preds: [bz, seq, tp_size], weight: [bz, seq, tp_size]
                        weight = (
                            paddle.distributed.collective._c_concat(weight, group=model_parallel_group)
                            .reshape([batch_size, -1, seq_len])
                            .transpose([0, 2, 1])
                        )

                        # weight: [bz, seq]
                        # concat and argmax again to get true maximum
                        weight = weight.argmax(axis=-1)

                        # preds: [bz, seq, tp_size], weight: [bz, seq]
                        # extract maximum indices `preds`
                        preds = preds[
                            paddle.arange(preds.shape[0]).unsqueeze(1),
                            paddle.arange(preds.shape[1]).unsqueeze(0),
                            weight,
                        ]

                    if len(preds.shape) == 1:
                        # NOTE(hehuang): Adapt evaluation with use_sparse_head_and_loss_fn.
                        # logits' shape is [num_predictions, vocab_size] when use_sparse_flash_attn is on,
                        # and need to add virtual batch dim for Trainer._pad_across_processes.
                        preds = preds[None]

                    preds_shape = [preds.shape]
            else:
                if logits is None:
                    preds = None
                    preds_shape = [[]]
                else:
                    # NOTE(hehuang): Decrease the communication cost of nested_gather.
                    if tensor_parallel_degree > 1 and is_tensor_parallel_output:
                        logits = paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
                    preds = logits.argmax(axis=-1)
                    if len(preds.shape) == 1:
                        # NOTE(hehuang): Adapt evaluation with use_sparse_head_and_loss_fn.
                        # logits' shape is [num_predictions, vocab_size] when use_sparse_flash_attn is on,
                        # and need to add virtual batch dim for Trainer._pad_across_processes.
                        preds = preds[None]

                    preds_shape = [preds.shape]

            if pipeline_parallel_group and pipeline_parallel_group.nranks > 1:
                # broadcast logits from pp last rank to others.
                paddle.distributed.broadcast_object_list(
                    preds_shape,
                    src=pipeline_parallel_group.ranks[-1],
                    group=pipeline_parallel_group,
                )
                if not model.is_pipeline_last_stage():
                    preds = paddle.empty(shape=preds_shape[0], dtype=paddle.int64)
                task = dist.stream.broadcast(
                    preds, src=pipeline_parallel_group.ranks[-1], group=pipeline_parallel_group, sync_op=False
                )
                task.wait()

            return (loss, preds, labels)
        loss = None
        model.eval()
        with paddle.no_grad():
            generated_tokens = model.generate(
                **inputs,
                decoding_strategy="sampling",
                top_k=1,
                max_length=self.data_args.max_seq_len,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.cls_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )[0]

            all_preds = []
            for pred_tokens in generated_tokens:
                pred_tokens = pred_tokens[pred_tokens != self.tokenizer.pad_token_id]
                all_preds.append(pred_tokens)
            max_pred_length = max([len(x) for x in all_preds])
            for index, preds in enumerate(all_preds):
                all_preds[index] = paddle.to_tensor(preds.tolist() + [-100] * (max_pred_length - len(preds)))
            all_preds = paddle.to_tensor(all_preds)

            if "labels" in inputs:
                all_labels = inputs["labels"]
                all_labels = paddle.to_tensor(all_labels)
            else:
                all_labels = None
        return (loss, all_preds, all_labels)

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """
        Log training metrics and calculate perplexity where applicable.

        Args:
            logs (Dict[str, float]): Dictionary containing training/evaluation metrics.
            **kwargs: Additional keyword arguments for logging.
        """
        if hasattr(self.model, "ranking_loss"):
            logs["ranking_loss"] = self.model.ranking_loss
        else:
            if "loss" in logs:
                logs["ppl"] = np.exp(logs["loss"])
            if "eval_loss" in logs:
                logs["eval_ppl"] = np.exp(logs["eval_loss"])

        train_eval = "train" if "loss" in logs else "eval"
        if self.state.epoch is not None and train_eval == "train":
            self.state.epoch *= self.args.num_train_epochs
        super().log(logs, **kwargs)

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        """
        Load random number generator states from checkpoint.

        Args:
            checkpoint: Checkpoint containing saved RNG states for reproducibility.
        """
        super()._load_rng_state(checkpoint)
        if self.data_seed is not None:
            random.setstate(random.getstate())
            np.random.set_state(np.random.get_state())

    def _save_moe_weights(self, output_dir):
        """
        Save model weights and optimizer states for Mixture-of-Experts (MoE) models.

        Args:
            output_dir (str): Directory path to save the model and optimizer checkpoints.
        """
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        filtered_state_dict = OrderedDict()
        filter_optimizer_state_dict = OrderedDict()

        param_names_in_master_weights = (
            list(optimizer_state_dict["master_weights"].keys()) if self.args.bf16 or self.args.fp16 else []
        )
        filter_optimizer_state_dict["master_weights"] = OrderedDict()

        for k, v in state_dict.items():
            if getattr(v, 'no_sync', False):
                if v.name in param_names_in_master_weights:
                    filter_optimizer_state_dict["master_weights"][v.name] = optimizer_state_dict["master_weights"][
                        v.name
                    ]

                filtered_state_dict[k] = v

                for op_k, op_v in optimizer_state_dict.items():
                    if op_k.startswith(v.name):
                        filter_optimizer_state_dict[op_k] = op_v

        filter_optimizer_state_dict['LR_Scheduler'] = optimizer_state_dict['LR_Scheduler']

        self._save_ckpt_func(
            filtered_state_dict,
            os.path.join(
                output_dir,
                _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix),
            ),
        )
        if not self.args.ignore_save_lr_and_optim:
            self._save_ckpt_func(
                filter_optimizer_state_dict,
                os.path.join(
                    output_dir,
                    _add_variant(PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix),
                ),
            )

    def _get_train_sampler(self):
        """
        Create and return appropriate data sampler for distributed training.

        Returns:
            DistributedBatchSampler: Configured sampler for distributing training data across devices.
        """
        if self._is_iterable_dataset(self.train_dataset):
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None

            if self.args.world_size <= 1:
                return paddle.io.BatchSampler(
                    dataset=self.train_dataset,
                    shuffle=True,
                    batch_size=self.args.per_device_train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                )

            return DistributedBatchSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                num_replicas=self.args.dataset_world_size,
                rank=self.args.dataset_rank,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            return DistributedBatchSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=False,
                num_replicas=self.args.dataset_world_size,
                rank=self.args.dataset_rank,
                drop_last=self.args.dataloader_drop_last,
            )
