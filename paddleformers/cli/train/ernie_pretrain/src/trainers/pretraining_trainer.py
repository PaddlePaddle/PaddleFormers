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

__all__ = [
    "PretrainingTrainer",
]


import contextlib
import json
import logging
import math
import os
import pickle
import random
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from types import MethodType
from typing import Optional

import numpy as np
import paddle
import paddle.amp.auto_cast as autocast
from paddle import framework, nn
from paddle.base import core
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet.utils import mix_precision_utils

from paddleformers.trainer import Trainer, TrainingArguments, speed_metrics
from paddleformers.utils.tools import get_env_device

try:
    from paddleformers.trainer import TRAINING_ARGS_NAME
except ImportError:
    TRAINING_ARGS_NAME = "training_args.bin"

try:
    from paddleformers.utils.env import PADDLE_OPTIMIZER_NAME
except ImportError:
    from paddleformers.trainer.trainer import OPTIMIZER_NAME

    PADDLE_OPTIMIZER_NAME = OPTIMIZER_NAME

try:
    from paddleformers.trainer.trainer import (
        PADDLE_WEIGHT_FILE_NAME as PADDLE_WEIGHTS_NAME,
    )
except ImportError:
    from paddleformers.utils.env import PADDLE_WEIGHTS_NAME

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)

from paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)
from paddleformers.cli.train.ernie_pretrain.models.utils import (
    global_training_logs_enabled,
)
from paddleformers.cli.train.ernie_pretrain.src.callbacks import (
    FP8QuantWeightCallback,
    GCCallback,
    LoggingCallback,
    SPGradSyncCallback,
    TensorBoardCallback,
)
from paddleformers.cli.train.ernie_pretrain.src.callbacks.moe_logging_callback import (
    MoeLoggingCallback,
)
from paddleformers.cli.train.ernie_pretrain.src.clip import ClipGradForMOEByGlobalNorm
from paddleformers.cli.train.ernie_pretrain.src.lr_schedulers import (
    get_wsd_schedule_with_warmup,
)
from paddleformers.cli.train.ernie_pretrain.src.trainers.data_parallel import (
    sync_dp_moe_params_across_sharding,
)
from paddleformers.cli.train.ernie_pretrain.src.utils.misc import global_training_logs
from paddleformers.cli.train.ernie_pretrain.src.utils.training_utils import (
    reset_per_device_batch_size,
)
from paddleformers.datasets import MapDataset
from paddleformers.trainer.trainer_callback import PrinterCallback
from paddleformers.trainer.trainer_utils import ShardingOption
from paddleformers.trainer.utils import add_start_docstrings
from paddleformers.transformers.model_utils import _add_variant, unwrap_model
from paddleformers.utils.batch_sampler import (
    DistributedBatchSampler as PaddleNLPDistributedBatchSampler,
)

logger = logging.getLogger(__name__)


def distributed_optimizer_maybe_overwrite(
    optimizer,
    use_moe,
):
    if use_moe:
        from paddleformers.cli.train.ernie_pretrain.src.trainers.dygraph_optimizer.hybrid_parallel_optimizer import (
            HybridParallelOptimizer as MoEHybridParallelOptimizer,
        )

        fleet_env = fleet.fleet
        fleet_env.user_defined_optimizer = optimizer
        hp_optim = MoEHybridParallelOptimizer(optimizer, fleet_env._hcg, fleet_env._user_defined_strategy)

        if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].dp_comm_overlap:
            hp_optim._dp_enable = False

        if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].sharding_comm_overlap:
            hp_optim._sharding_enable = False
        return hp_optim
    else:
        return fleet.distributed_optimizer(optimizer)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
    vocab_path: str = field(default=None, metadata={"help": "eb35 streaming data vocab"})
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "https://paddleformers.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "global random seed factor."},
    )
    eval_iters: int = field(
        default=-1,
        metadata={"help": "eval iteration for every evaluation."},
    )
    num_consecutive: int = field(
        default=1,
        metadata={"help": "H5 file consecutive num."},
    )
    min_lr: float = field(
        default=0.0,
        metadata={"help": "minus learning rate"},
    )
    dataset: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    input_dir: str = field(default=None, metadata={"help": "data path"})
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split ratio"})

    max_seq_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    global_batch_size: int = field(
        default=-1,
        metadata={
            "help": "if `global_batch_size` and `per_device_train_batch_size` is provied, "
            "`gradient_accumulation_steps` will be ignored"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    sequence_parallel: Optional[int] = field(
        default=0,
        metadata={},
    )
    virtual_pipeline_model_parallel_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "vpp",
        },
    )
    from_scratch: Optional[int] = field(default=1, metadata={"help": "train from scratch"})
    same_data: Optional[bool] = field(
        default=None,
        metadata={"help": "when resume from checkpoint, keey data same with the ckpt"},
    )
    base_seq_length: Optional[int] = field(default=4096, metadata={"help": "reeao min seq_length"})
    shuffle_consecutive: Optional[bool] = field(
        default=False,
        metadata={"help": "shuffle num_consecutive or not"},
    )
    global_shuffle_num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "max num of shuffling among different parts"},
    )
    use_async_save: Optional[bool] = field(default=False, metadata={"help": "use async save or not"})
    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )
    enable_global_training_logs: bool = field(default=False, metadata={"help": "use global_training_logs or not"})
    moe_group: Optional[str] = field(default="dp", metadata={"help": "moe comm group"})
    use_moe: Optional[bool] = field(default=False, metadata={"help": "enable expert parallel"})
    log_global_grad_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "print global grad-norm"},
    )
    multi_token_pred_depth: Optional[int] = field(
        default=0,
        metadata={},
    )
    enable_mtp_magic_send: Optional[bool] = field(default=False, metadata={"help": ""})
    enable_optimizer_timer: Optional[bool] = field(default=False, metadata={"help": "enable timer in zero-1"})
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. support linear, cosine, constant, constant_with_warmup"},
    )
    decay_function: str = field(
        default="half_life",
        metadata={"help": "The decay function for WSD LR scheduler. support half_life(default), 1-sqrt"},
    )
    moe_gate_lr_ratio: float = field(
        default=None,
        metadata={"help": ("special handle the lr for gate/router")},
    )

    gc_interval: int = field(default=0, metadata={"help": "gc time"})
    use_sp_callback: int = field(
        default=True,
        metadata={"help": "use callback for sequence parallel"},
    )
    moe_use_aux_free_update_coef: float = field(
        default=1.0e-3,
        metadata={"help": "moe aux free update coef,"},
    )
    use_fp8: bool = field(
        default=False,
        metadata={"help": "whether to use fp8 training"},
    )
    global_logging_interval: int = field(
        default=1,
        metadata={"help": "the logging interval of global_training_logs"},
    )
    train_moe_only: int = field(default=None, metadata={"help": "train moe params only"})
    use_ortho_loss_callback: bool = field(default=False, metadata={"help": "Use orthogonal loss callback or not"})

    recompute: bool = field(
        default=False,
        metadata={
            "help": "Recompute the forward pass to calculate gradients. Used for saving memory. "
            "Only support for networks with transformer blocks."
        },
    )

    refined_recompute: str = field(
        default="",
        metadata={
            "help": "The refined recompute parameter is designed to optimize the balance between GPU memory usage and computational speed.\n"
            "An example configuration could be: `attention_column_ln:-1,attention_row_ln:-1,flash_attn:-1,mlp_column_ln:5,mlp_row_ln:-1`.\n"
            "The supported parameters for refining recompute are `attention_column_ln`, `attention_row_ln`, `flash_attn`, `mlp_column_ln`, and `mlp_row_ln`.\n"
            "The associated number, `skip_num`, determines how many times to bypass recomputation for the specified operation.\n"
            "A `skip_num` of `-1` indicates no recomputation across all stages, maximizing memory usage;\n"
            "A `skip_num` of `0` enforces recomputation at every stage, minimizing memory usage.\n"
            "You can also set `skip_num` to a value within the range [1, ..., num_layers]. If `skip_num` exceeds `num_layers`, it will behave as if set to `-1`.\n"
            "If a parameter is omitted, it defaults to `xxx:0`."
        },
    )

    @property
    def use_moe(self):  # noqa: F811
        return getattr(self, "use_expert_parallel", self._use_moe)

    @use_moe.setter
    def use_moe(self, value):
        self.use_expert_parallel = value
        self._use_moe = value

    @property
    def need_data(self):
        return self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0

    @property
    def combine_batch(self):
        return self.max_seq_len // self.base_seq_length

    @property
    def reeao_dataset_rank(self):
        return super().dataset_rank

    @property
    def reeao_dataset_world_size(self):
        return super().dataset_world_size

    def __post_init__(self):
        super().__post_init__()

        if self.global_batch_size > 0:
            micro_bsz, acc_steps = reset_per_device_batch_size(
                self.global_batch_size,
                self.per_device_train_batch_size,
                self.dataset_world_size,
            )
            logger.info(f"global_batch={self.global_batch_size} micro-bsz:{micro_bsz}, accumulate_steps:{acc_steps}")
            if (
                acc_steps != 1
                and self.gradient_accumulation_steps != 1
                and acc_steps != self.gradient_accumulation_steps
            ):
                raise ValueError(
                    f"global_accumulation_steps={self.gradient_accumulation_steps}"
                    f"& global_batch={self.global_batch_size} are both set"
                )
            self.per_device_train_batch_size, self.gradient_accumulation_steps = (
                micro_bsz,
                acc_steps,
            )

        self.max_gradient_accumulation_steps = self.gradient_accumulation_steps

        if self.pipeline_model_parallel_size > 1:
            self.per_device_eval_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps
            logger.warn(f"eval_batch_size set to {self.per_device_eval_batch_size} in Pipeline Parallel!")
            user_defined_strategy = fleet.fleet._user_defined_strategy
            user_defined_strategy.strategy.pipeline_configs.accumulate_steps = self.gradient_accumulation_steps
            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps
            logger.info(f"fixing pp configs: {user_defined_strategy.pipeline_configs}")
        else:
            self.per_device_eval_batch_size = self.per_device_train_batch_size
            logger.warn(f"eval_batch_size set to {self.per_device_eval_batch_size}")

        if self.sharding_parallel_size > 1:
            sharding_comm_overlap_non_pp = (
                True if self.sd_shardingv1_comm_overlap or self.sd_sharding_comm_overlap else False
            )
            if sharding_comm_overlap_non_pp:
                assert hasattr(fleet.fleet, "_user_defined_strategy")
                user_defined_strategy = fleet.fleet._user_defined_strategy
                user_defined_strategy.hybrid_configs[
                    "sharding_configs"
                ].accumulate_steps = self.gradient_accumulation_steps

        if hasattr(fleet.fleet, "_user_defined_strategy"):
            user_defined_strategy = fleet.fleet._user_defined_strategy
            if (
                hasattr(user_defined_strategy, "hybrid_configs")
                and "sharding_configs" in user_defined_strategy.hybrid_configs
            ):
                sd_configs = user_defined_strategy.hybrid_configs["sharding_configs"]
                if sd_configs.comm_overlap:
                    assert self.global_batch_size % self.dataset_world_size == 0, (
                        f"global_batch_size[{self.global_batch_size}] should be divisible by "
                        f"dataset_world_size[{self.dataset_world_size}]"
                    )
                    lbs = self.global_batch_size // self.dataset_world_size
                    assert lbs % self.per_device_train_batch_size == 0, (
                        f"local_batch_size[{lbs}] should be divisible by "
                        f"per_device_train_batch_size[{self.per_device_train_batch_size}]"
                    )
                    assert lbs // self.per_device_train_batch_size == sd_configs.accumulate_steps, (
                        f"local_batch_size[{lbs}] should be equal to "
                        f"accumulate_steps[{sd_configs.accumulate_steps}] * "
                        f"per_device_train_batch_size[{self.per_device_train_batch_size}]"
                    )

        if ShardingOption.SHARD_GRAD_OP in self.sharding:
            logger.info("disabling `sp_callback` b/c using sharding stage2")
            self.use_sp_callback = False

        # arse_refined_recompute string to dict
        if self.refined_recompute in [None, ""]:
            self.refined_recompute = dict()
        else:
            refined_recompute_dict = {
                "mlp_row_ln": 0,
                "attention_row_ln": 0,
                "attention_column_ln": 0,
                "mlp_column_ln": 0,
                "flash_attn": 0,
                "global": 0,
            }
            ops = self.refined_recompute.split(",")
            enable_rr = False
            for op in ops:
                op = op.strip()
                if ":" not in op:
                    raise ValueError("Illegal refined_recompute input, please check.")
                op_name, skip_num = op.split(":")[0], int(op.split(":")[1])
                if op_name not in refined_recompute_dict:
                    raise ValueError(f"Refined recompute do not support {op_name}, please check.")
                if (
                    op_name in ["mlp_row_ln", "attention_row_ln", "attention_column_ln", "mlp_column_ln"]
                    and self.tensor_model_parallel_size <= 1
                ):
                    logger.warning(
                        f"Refined recompute is only supported for the `{op_name}` operation when `tensor_model_parallel_size` is greater than 1. \
                            This refined recompute operation will be ignored."
                    )
                    continue

                refined_recompute_dict[op_name] = skip_num
                if skip_num != 0:
                    enable_rr = True
            if not enable_rr:
                refined_recompute_dict = dict()
            self.refined_recompute = refined_recompute_dict


class WeightedDistributedSampler(PaddleNLPDistributedBatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        output_dir,
        dp_rank,
        dp_size,
        num_consecutive=1,
        seed=0,
        gradient_accumulation_steps=None,
        max_gradient_accumulation_steps=None,
        per_device_train_batch_size=None,
        combine_batch: int = 1,
        shuffle_consecutive: bool = False,
        global_shuffle_num_examples: int = 0,
        same_data: bool = False,
        **kwargs,
    ):
        self.num_consecutive = num_consecutive
        self.seed = seed
        super().__init__(dataset, batch_size, **kwargs)
        self.weights = None
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.rng = random.Random(self.seed + self.epoch)
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_gradient_accumulation_steps = max_gradient_accumulation_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.combine_batch = combine_batch
        self.shuffle_consecutive = shuffle_consecutive
        self.global_shuffle_seed = 0
        self.global_shuffle_num_examples = global_shuffle_num_examples
        self.same_data = same_data
        self.load_data_seq = False
        if isinstance(self.dataset, MapDataset):
            self.inner_dataset = self.dataset.data
        else:
            self.inner_dataset = self.dataset
        assert self.inner_dataset._load

        self.max_part_id = self.inner_dataset.global_max_part_id

        self.set_epoch(0)

    def set_epoch(self, epoch=0, consumed_samples=0):
        consumed_samples = consumed_samples // self.dp_size
        logger.info(f"set consumed samples={consumed_samples}, epoch={epoch}")
        super().set_epoch(epoch, consumed_samples)

    def gen_data_seq(self):
        total = []
        for ex in self.inner_dataset.exs:
            total.extend([(ex.part, 0, i) for i in range(ex.data_status, len(ex))])
        assert len(total) > self.num_consecutive, f"total={total} < num_consecutive={self.num_consecutive}"
        indices = np.array_split(np.array(total), len(total) // self.num_consecutive)
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = np.concatenate(indices)
        indices = self.roundup_and_shard(indices)
        logger.info(indices[:10])
        return indices

    def load_data_seq_from_cache(self):
        indices_file = os.path.join(
            self.output_dir,
            f"data_seq.epoch{self.epoch}.dp_{self.dp_rank}_of_{self.dp_size}"
            f"_shard_{self.local_rank}_of_{self.nranks}.pth",
        )
        if self.same_data and os.path.exists(indices_file):
            logger.info(f"load data seq from file - {indices_file}")
            self.load_data_seq = True
            with open(indices_file, "rb") as of:
                return pickle.load(of)
        return None

    def gen_data_seq_weighted(self, num_examples, data_type=None):
        assert (
            self.load_data_seq is False
        ), "Ensure that the data_seq for all epochs is loaded from the file; otherwise, the randomness cannot be controlled when deleting data_seq next time."
        logger.info(
            f"generating data sequence... #non_consecutive_data_chunks={num_examples},"
            f" num_consecutive={self.num_consecutive}"
        )

        if num_examples > 1e5:
            logger.info("generating data sequence for very large data, consider use large `num_consecutive`")
        if data_type is not None:
            weights = [ex.weights for ex in self.inner_dataset.exs if ex.data_type == data_type]
            exs = [ex for ex in self.inner_dataset.exs if ex.data_type == data_type]
        else:
            weights = [ex.weights for ex in self.inner_dataset.exs]
            exs = self.inner_dataset.exs
        assert len(exs) > 0, f"data_type={data_type}, no data found"
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        logger.info(
            f"using weighted sampler, num_consecutive={self.num_consecutive}:\n"
            + "\n".join(["%-100s...%.3e" % (e.path, w) for w, e in zip(weights, exs)])
        )

        part_indices_gen = {}
        indices = []
        for i, ex in enumerate(exs):
            sample_size = int(weights[i] * num_examples)
            logger.info(f"part_data_pre_sampling--[part-{ex.part}]-[sampler-size-{sample_size}]")
            assert ex.combine_batch == self.combine_batch
            part_indices_gen[ex.part] = ex.sampler()
            indices.extend([ex.part] * sample_size)

        logger.info(f"shuffle part placeholder index, size={len(indices)}, exmaple={indices[0]}")
        if self.shuffle:
            self.rng.shuffle(indices)
        logger.info("shuffle done")
        indices_ret = []
        logger.info("build_index from shuffled placeholder")

        for part_id in indices:
            epoch, _index = next(part_indices_gen[part_id])
            if len(_index) % self.combine_batch != 0:
                _index += [-1] * (self.combine_batch - len(_index) % self.combine_batch)
            indices_ret += [(part_id, epoch, i) for i in _index]

        if self.shuffle_consecutive and self.combine_batch >= 1:
            part_data_gen = defaultdict(lambda: [])
            logger.info("consecutive placeholder 2 shuffle")
            for item in indices_ret:
                part_data_gen[item[0]].append(item)
            logger.info("consecutive placeholder 2 shuffle...")
            part_data_gen_iter = {}
            for key in part_data_gen.keys():
                part_data_gen_iter[key] = iter(part_data_gen[key])
            logger.info("consecutive placeholder 2 shuffle......")
            placeholder_indices = [i[0] for i in indices_ret]
            placeholder_indices = [
                placeholder_indices[i : i + self.combine_batch]
                for i in range(0, len(placeholder_indices), self.combine_batch)
            ]
            logger.info("consecutive placeholder 2 shuffle..........")
            self.rng.shuffle(placeholder_indices)
            logger.info("consecutive placeholder 2 shuffle.............")
            placeholder_indices = [item for sublist in placeholder_indices for item in sublist]
            logger.info("consecutive placeholder 2 shuffle................")
            indices_ret = [next(part_data_gen_iter[i]) for i in placeholder_indices]
            logger.info("consecutive placeholder 2 shuffle done")

        logger.info("build index done")
        indices = np.array(indices_ret)
        del indices_ret
        logger.info(f"num_data_seq={len(indices)}, example={indices[:10]}")
        indices = self.roundup_and_shard(indices)
        return indices

    def roundup_and_shard(self, indices):
        if self.nranks == 1:
            return indices

        padding_size = self.total_size - len(indices)
        logger.info(f"padding-size={padding_size}, total_size={self.total_size} shard={self.local_rank}/{self.nranks}")
        if padding_size < 0:
            indices = indices[:padding_size]
        else:
            indices = np.concatenate(
                [
                    indices,
                    np.tile(indices, math.ceil(padding_size / len(indices)))[:padding_size],
                ]
            )

        assert len(indices) == self.total_size, (len(indices), self.total_size)

        indices = indices[self.local_rank : self.total_size : self.nranks]
        assert len(indices) == self.num_samples
        return indices

    def __len__(self):
        raise TypeError

    def __iter__(self):
        self.rng = random.Random(self.seed + self.epoch + self.global_shuffle_seed)
        logger.info(f"seed={self.seed + self.epoch + self.global_shuffle_seed}")
        weights = [e.weights for e in self.inner_dataset.exs]
        if any(w is None for w in weights) or sum(weights) == 0.0:
            logger.info(f"using normal sampler, num_consecutive={self.num_consecutive}")
            indices = self.gen_data_seq()
            self.weights = None
        else:
            self.weights = weights
            num_examples = sum([ex.num_examples for ex in self.inner_dataset.exs])

            if self.global_shuffle_num_examples > 0:
                num_examples = min([self.global_shuffle_num_examples, num_examples])
                logger.info(f"using global shuffle num examples: {self.global_shuffle_num_examples}")
            indices = self.load_data_seq_from_cache()
            if indices is None:
                indices = self.gen_data_seq_weighted(num_examples)

        if self.output_dir:
            with open(
                os.path.join(
                    self.output_dir,
                    f"data_seq.epoch{self.epoch}.dp_{self.dp_rank}_of_{self.dp_size}"
                    f"_shard_{self.local_rank}_of_{self.nranks}.pth",
                ),
                "wb",
            ) as of:
                pickle.dump(indices, of, protocol=4)

        def ret():
            nonlocal indices
            buf = []
            logger.info(f"start training sequence, data-sequence: {indices[:10]}")
            while 1:
                if self.consumed_samples >= len(indices):
                    self.consumed_samples -= len(indices)
                else:
                    for i in range(self.consumed_samples, len(indices)):
                        if len(buf) == self.batch_size:
                            yield buf
                            buf = []
                        buf.append(indices[i].tolist())
                    self.consumed_samples = 0
                self.epoch += 1
                logger.info(f"epoch done, #data={self.total_size}, reshuffle-sequence: epoch={self.epoch}")

                self.rng = random.Random(self.seed + self.epoch)
                if self.weights:
                    indices = self.load_data_seq_from_cache()
                    if indices is None:
                        indices = self.gen_data_seq_weighted(num_examples)
                else:
                    indices = self.gen_data_seq()
                if self.output_dir:
                    with open(
                        os.path.join(
                            self.output_dir,
                            f"data_seq.epoch{self.epoch}.dp_{self.dp_rank}_of_{self.dp_size}"
                            f"_shard_{self.local_rank}_of_{self.nranks}.pth",
                        ),
                        "wb",
                    ) as of:
                        pickle.dump(indices, of, protocol=4)

        return ret()


class DummySampler(PaddleNLPDistributedBatchSampler):
    def __init__(self, dataset, batch_size=1, **kwargs):
        super().__init__(dataset, batch_size=batch_size, **kwargs)

    def __len__(self):
        raise TypeError

    def __iter__(self):
        while True:
            yield [0] * self.batch_size


class PretrainingTrainer(Trainer):
    def __init__(self, args=None, model=None, callbacks=[], **kwargs):
        callbacks = [
            FP8QuantWeightCallback(),
            LoggingCallback(),
            TensorBoardCallback(args, model=model, log_tokens_per_step=True, log_flops_per_step=False),
            GCCallback(),
        ] + callbacks

        args.use_async_save = args.use_async_save and args.save_sharded_model and args.load_sharded_model
        super().__init__(args=args, model=model, callbacks=callbacks, **kwargs)
        self.pop_callback(PrinterCallback)
        self.pp_data_buffer = []
        self._tokens_per_sec_per_card_buffer = []
        self._start_save_time = time.time()
        self._end_save_time = time.time()
        self._first_end_save_time = time.time()
        self.resume_global_step = -1
        self.first_skip_step = 5 if self.args.save_steps > 5 else self.args.save_steps / 2
        global_training_logs.enable_skip_zero([r".*aux_loss.*"])
        global_training_logs.set_trainer_interval(self, self.args.global_logging_interval)

    def autocast_smart_context_manager(self):
        if self.enable_autocast_context_manager:
            black = [
                "reduce_sum",
                "c_softmax_with_cross_entropy",
                "elementwise_div",
                "sin",
                "cos",
            ]
            white = [
                "lookup_table",
                "lookup_table_v2",
                "flash_attn",
                "flash_attn_v1",
                "matmul",
                "matmul_v2",
                "fused_gemm_epilogue",
            ]
            if self.args.bf16 and self.args.fp16_opt_level == "O2":
                black.append("c_embedding")

            ctx_manager = autocast(
                True,
                custom_black_list=black,
                custom_white_list=white,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
            )
        else:
            ctx_manager = contextlib.nullcontext()
        return ctx_manager

    def _load_optimizer_state(self, checkpoint):
        def _broadcast_moe_optimizer_state(state_dict):
            base_state_dict = {"master_weights": {}}
            buf = [
                {i: j.shape for i, j in state_dict.items() if i not in ["master_weights", "LR_Scheduler"]},
                {i: j.shape for i, j in state_dict["master_weights"].items()},
                {"LR_Scheduler": state_dict.get("LR_Scheduler", {})},
            ]

            if self.args.use_hybrid_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                src_rank = hcg.get_data_parallel_group_src_rank()
                group = hcg.get_data_parallel_group()
            else:
                src_rank = 0
                group = None

            dist.broadcast_object_list(buf, src=src_rank, group=group)
            for k, s in buf[0].items():
                v = state_dict.get(k, paddle.zeros(s, "float32")).to(get_env_device())
                v.name = k
                dist.broadcast(v, src=src_rank, group=group)
                logger.info(f"broadcast moe optimizer {k} from {src_rank}")
                base_state_dict[k] = v.cpu()
            for k, s in buf[1].items():
                v = state_dict["master_weights"].get(k, paddle.zeros(s, "float32")).to(get_env_device())
                v.name = k
                dist.broadcast(v, src=src_rank, group=group)
                logger.info(f"broadcast moe optimizer-master_weights {k} from {src_rank}")
                base_state_dict["master_weights"][k] = v.cpu()
            base_state_dict.update(buf[2])
            return base_state_dict

        state_dict = super()._load_optimizer_state(checkpoint)

        if self.args.use_moe:
            base_state_dict = _broadcast_moe_optimizer_state(state_dict)
            if self.args.data_parallel_rank > 0:
                master_weight = state_dict.pop("master_weights", {})
                base_state_dict.update(state_dict)
                if master_weight:
                    if "master_weights" in base_state_dict:
                        base_state_dict["master_weights"].update(master_weight)
                    else:
                        base_state_dict["master_weights"] = master_weight
                state_dict = base_state_dict
                del base_state_dict
        return state_dict

    def _save_moe_weights(self, output_dir):
        optimizer_name = _add_variant(PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
        saved_signal_path = os.path.join(output_dir, f"saved_signal_{dist.get_rank()}")

        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.model.state_dict()
        optimzier_state_dict = self.optimizer.state_dict()

        filtered_state_dict = OrderedDict()
        filter_optimzier_state_dict = OrderedDict()

        param_names_in_master_weights = list(optimzier_state_dict["master_weights"].keys()) if self.args.bf16 else []
        filter_optimzier_state_dict["master_weights"] = OrderedDict()

        for k, v in state_dict.items():
            if getattr(v, "no_sync", False):

                if v.name in param_names_in_master_weights:
                    filter_optimzier_state_dict["master_weights"][v.name] = optimzier_state_dict["master_weights"][
                        v.name
                    ]
                if not (
                    getattr(self.args, "should_save_sharding_stage1_model", False)
                    or getattr(self.args, "save_sharding_stage1_model", False)
                ):
                    filtered_state_dict[k] = v
                for op_k, op_v in optimzier_state_dict.items():
                    if op_k.startswith(v.name):
                        filter_optimzier_state_dict[op_k] = op_v

        if getattr(self.args, "should_save_sharding_stage1_model", False) or getattr(
            self.args, "save_sharding_stage1_model", False
        ):
            self._save(output_dir=output_dir)
        else:
            if self.args.sharding_parallel_rank == 0:
                paddle.save(
                    filtered_state_dict,
                    os.path.join(
                        output_dir,
                        _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix),
                    ),
                )
        paddle.save(filter_optimzier_state_dict, os.path.join(output_dir, optimizer_name))
        with open(saved_signal_path, mode="w+") as f:
            f.write("1")

    def _wrap_model(self, model, training=True):
        if unwrap_model(model) is not model:
            return model
        if not training:
            return model
        if self.args.fp16 or self.args.bf16:
            model = paddle.amp.decorate(models=model, level=self.args.fp16_opt_level, dtype=self.amp_dtype)

        if self.args.use_moe:
            from paddleformers.cli.train.ernie_pretrain.src.trainers.data_parallel import (
                DataParallel as MoEDDP,
            )

            paddle.DataParallel = MoEDDP

        if self.args.world_size > 1 and not self.args.use_hybrid_parallel:
            model = paddle.DataParallel(model)

        in_pipeline_parallel_mode = self.args.pipeline_model_parallel_size > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_model = self.args.tensor_model_parallel_size > 1

        def enable_sequence_parallel(_model):
            if self.args.tensor_model_parallel_size > 1 and self.args.sequence_parallel:
                if self.args.use_sp_callback:
                    self.add_callback(SPGradSyncCallback(_model._layers))
                else:
                    register_sequence_parallel_allreduce_hooks(_model)

        is_dp_moe = self.args.use_moe and self.args.moe_group in {"data", "dp"}

        if in_pipeline_parallel_mode:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(
                    model,
                    dtype=self.amp_dtype if hasattr(self, "amp_dtype") else "float16",
                )
            prepare_pipeline_inputs_func = (
                model._prepare_pipeline_inputs_func if hasattr(model, "_prepare_pipeline_inputs_func") else None
            )
            model = fleet.distributed_model(model)
            if is_dp_moe:
                logger.info("start broadcast dp moe parameters across sharding group")
                sync_dp_moe_params_across_sharding(model._layers)
            if prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func
            else:

                def _prepare_pipeline_inputs_func(inputs):
                    first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
                    last_stage_keys = ["labels"]

                    def get_expected_keys(inputs, keys):
                        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
                        if len(ret) == 1:
                            ret = ret[0]
                        return ret

                    if type(inputs) is dict:
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
            self.optimizer = distributed_optimizer_maybe_overwrite(self.optimizer, self.args.use_moe)

        if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
            if self.args.tensor_model_parallel_size > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)
                model.accumulate_steps = self.args.gradient_accumulation_steps
                enable_sequence_parallel(model)

            if ShardingOption.SHARD_OP in self.args.sharding:
                if self.args.amp_master_grad:
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)
                model = fleet.distributed_model(model)
                if is_dp_moe:
                    logger.info("start broadcast dp moe parameters across sharding group")
                    sync_dp_moe_params_across_sharding(model._layers)
                if self.args.amp_master_grad:
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer = distributed_optimizer_maybe_overwrite(self.optimizer, self.args.use_moe)

            else:
                if (self.args.use_moe) and self.args.data_parallel_size > 1:
                    try:
                        from paddle.fluid.dygraph.parallel import sync_params_buffers
                    except ImportError:
                        from paddle.distributed.parallel import sync_params_buffers

                    hcg = fleet.get_hybrid_communicate_group()
                    dp_group = hcg.get_data_parallel_group()
                    sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

                if is_dp_moe:
                    logger.info("start broadcast dp moe parameters across sharding group")
                    sync_dp_moe_params_across_sharding(model)

                cpu_offload = ShardingOption.OFFLOAD in self.args.sharding
                assert self.optimizer is not None, "optimizer is empty!"
                level = None
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                    level = "os_g"
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    level = "p_g_os"

                from paddle.distributed.sharding import group_sharded_parallel

                extra_kwargs = {}
                if not self.args.use_moe:
                    extra_kwargs["dp_group"] = self.dp_group
                    extra_kwargs["exclude_layer"] = ["GroupNorm"]

                model, optimizer, _ = group_sharded_parallel(
                    model,
                    self.optimizer,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                self.optimizer = optimizer

        if not in_pipeline_parallel_mode and not in_sharding_parallel_mode and in_tensor_parallel_model:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)

            model = fleet.distributed_model(model)
            model.accumulate_steps = self.args.gradient_accumulation_steps
            enable_sequence_parallel(model)
            assert self.optimizer is not None, "Tensor parallel mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

            self.optimizer = distributed_optimizer_maybe_overwrite(self.optimizer, self.args.use_moe)

        if self.args.use_moe:
            self.callback_handler.callbacks.insert(0, MoeLoggingCallback(self.optimizer))

        try:
            from paddle.fluid.dygraph.parallel import sync_params_buffers
        except ImportError:
            from paddle.distributed.parallel import sync_params_buffers

        self._new_gradclip()
        return model

    def _new_gradclip(self):
        if (
            isinstance(self.optimizer, HybridParallelOptimizer)
            and self.args.log_global_grad_norm
            and self.args.max_grad_norm > 0
        ):
            gradclip = self.optimizer._inner_opt._grad_clip
            oldcomm = gradclip._comm_and_clip
            oldclip = gradclip._dygraph_clip
            hcg = fleet.get_hybrid_communicate_group()
            num_pp = hcg.get_pipe_parallel_world_size()

            @paddle.no_grad()
            def newcomm(
                self,
                params_grads,
                global_norm_var_dist,
                global_norm_var_not_dist,
                *args,
            ):
                if num_pp > 1:
                    for p, g in params_grads:
                        if getattr(p, "need_clip", True) == "pp_non_distributed":
                            g.scale_(np.sqrt(num_pp))
                ret = oldcomm(params_grads, global_norm_var_dist, global_norm_var_not_dist, *args)
                global_norm_var_fp32 = paddle.sqrt(global_norm_var_dist + global_norm_var_not_dist)
                if global_training_logs_enabled():
                    global_training_logs.update(global_grad_norm=global_norm_var_fp32.item())
                return ret

            @paddle.no_grad()
            def new_dygraph_clip(self, params_grads):
                if num_pp > 1:
                    for p, g in params_grads:
                        if getattr(p, "need_clip", True) == "pp_non_distributed":
                            g.scale_(1 / np.sqrt(num_pp))
                ret = oldclip(params_grads)
                return ret

            self.optimizer._inner_opt._grad_clip._comm_and_clip = MethodType(
                newcomm, self.optimizer._inner_opt._grad_clip
            )
            self.optimizer._inner_opt._grad_clip._dygraph_clip = MethodType(
                new_dygraph_clip, self.optimizer._inner_opt._grad_clip
            )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self.model_wrapped.accumulate_steps = self.args.gradient_accumulation_steps
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()
        compute_metrics = self.compute_metrics
        eval_loop = self.evaluation_loop

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            max_eval_iters=self.args.eval_iters,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return output.metrics

    def prediction_pipeline_step(self, model, inputs, prediction_loss_only, ignore_keys):
        loss, _, labels = super().prediction_pipeline_step(model, inputs, prediction_loss_only, ignore_keys)
        num_tokens = (labels != self.tokenizer.ignored_index).sum().item()
        loss_avg = loss * self.model_wrapped.accumulate_steps / num_tokens
        return loss_avg, loss, labels

    def restore_dataloader_status(self):
        if self.args.same_data is None or self.args.same_data == "":
            if self.args.resume_from_checkpoint is not None:
                train_bin_file = os.path.join(self.args.resume_from_checkpoint, TRAINING_ARGS_NAME)
                assert os.path.exists(train_bin_file), f"{train_bin_file} not found."
                train_bin = paddle.load(train_bin_file)
                old_data_filelist = train_bin.data_filelist
                old_data_weights = train_bin.data_weights
                old_sharding_degree = train_bin.sharding_parallel_size
                old_data_parallel_size = train_bin.data_parallel_size
                old_reeao_data_world_size = getattr(train_bin, "reeao_data_world_size", None)
                new_data_filelist = self.args.data_filelist
                new_data_weights = self.args.data_weights
                new_sharding_degree = self.args.sharding_parallel_size
                new_data_parallel_size = self.args.data_parallel_size
                self.args.same_data = (
                    (old_data_filelist == new_data_filelist)
                    and (old_data_weights == new_data_weights)
                    and (old_sharding_degree == new_sharding_degree)
                    and (old_data_parallel_size == new_data_parallel_size)
                    and (not self.args.multimodal)
                    and (
                        old_reeao_data_world_size is None
                        or old_reeao_data_world_size == self.args.reeao_data_world_size
                    )
                )
                logger.info(f"Automatically setting same_data value: {self.args.same_data}")
            else:
                self.args.same_data = False
                logger.info(f"Training from scratch, setting same_data value: {self.args.same_data}")
        else:
            logger.info(f"User has defined same_data value: {self.args.same_data}")

        if self.args.same_data:
            logger.warning(
                "same_data has been set to True. \
                            Carefully check whether the data, population proportion, "
                "and DP count are completely consistent with those before."
            )
        else:
            logger.warning(
                "same_data has been set to False. \
                            which will regenerate the global shuffle domain."
            )

    def _get_eval_sampler(self, eval_dataset) -> Optional[paddle.io.Sampler]:
        return PaddleNLPDistributedBatchSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        return PaddleNLPDistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        flag_log = self.control.should_log
        if self.control.should_log:
            logs = {}
            tr_loss_single_dp_scalar = tr_loss.item()
            dist.all_reduce(tr_loss, dist.ReduceOp.SUM)
            tr_loss_scalar = tr_loss.item() / dist.get_world_size()
            tr_loss.zero_()

            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs["loss_cur_dp"] = tr_loss_single_dp_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs["learning_rate"] = float(self._get_learning_rate())
            logs["global_step"] = int(self.state.global_step)

            divisor = 2**30

            current_device = framework._current_expected_place_()
            device_id = current_device.get_device_id()
            current_memory_allocated = core.device_memory_stat_current_value("Allocated", device_id)
            current_memory_reserved = core.device_memory_stat_current_value("Reserved", device_id)
            max_memory_allocated = core.device_memory_stat_peak_value("Allocated", device_id)
            max_memory_reserved = core.device_memory_stat_peak_value("Reserved", device_id)
            logs["mem_allocated_gb"] = current_memory_allocated / divisor
            logs["max_mem_allocated_gb"] = max_memory_allocated / divisor
            logs["mem_reserved_gb"] = current_memory_reserved / divisor
            logs["max_mem_reserved_gb"] = max_memory_reserved / divisor

            if not self.args.enable_global_training_logs:
                global_training_logs.global_meters_keys = []

            if get_env_device() == "gpu":
                info_callback = global_training_logs.dict(use_async=True)

            if hasattr(self, "scaler"):
                logs["loss_scale"] = float(f"{self.scaler._scale.item():.3e}")

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.reeao_dataset_world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "global",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )
            if not hasattr(self, "model_numel"):
                model_numel = sum(
                    p.numel().item()
                    for n, p in model.named_parameters()
                    if not p.stop_gradient and "embeddings" not in n and "embed_tokens" not in n
                )
                numel_tensor = paddle.to_tensor(model_numel)
                dist.all_reduce(numel_tensor)
                self.model_numel = numel_tensor.item() // self.args.dataset_world_size

            tokens_per_steps = self.args.max_seq_len * total_train_batch_size
            logs["tokens_trained_current_step"] = tokens_per_steps
            logs["timestamp"] = int(time.time() * 1000)
            logs["TFLOPS_per_sec_per_card"] = round(
                6
                * tokens_per_steps
                * self.model_numel
                * logs["global_steps_per_second"]
                / 1e12
                / self.args.world_size,
                3,
            )
            logs["tokens_per_sec_per_card"] = round(
                tokens_per_steps * logs["global_steps_per_second"] / self.args.world_size,
                1,
            )
            self._tokens_per_sec_per_card_buffer.append(logs["tokens_per_sec_per_card"])
            logs["tokens_per_sec_per_card_average"] = round(np.mean(self._tokens_per_sec_per_card_buffer), 1)
            if self.resume_global_step == -1:
                self.resume_global_step = self.state.global_step - 1
            if self.state.global_step <= self.resume_global_step + self.first_skip_step:
                self._tokens_per_sec_per_card_buffer = []
                self._end_save_time = time.time()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            info, gathered_info = info_callback()
            global_training_logs.reset()
            logs.update({f"{k}_cur_dp": v for k, v in info.items()})
            logs.update(gathered_info)
            if self.args.enable_global_training_logs:
                info_list = []
                dist.all_gather_object(info_list, info)
                logs.update(
                    {
                        k: np.mean([v[k] for v in info_list if k in v])
                        for k in {key for item in info_list for key in item.keys()}
                    }
                )

            self.log(logs, **kwargs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            if hasattr(self.args, "flash_device_save_steps") and self.args.flash_device_save_steps > 0:
                is_persistent_ckpt = 1 if self.state.global_step % self.args.save_steps == 0 else 0
            else:
                is_persistent_ckpt = 1

            if is_persistent_ckpt:
                self._start_save_time = time.time()
            else:
                zcc_start_save_time = time.time()
            self._save_checkpoint(model, metrics=metrics)
            paddle.distributed.barrier()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            if flag_log:
                logs = {"is_persistent_ckpt": is_persistent_ckpt}
                tbk = self._start_save_time - self._end_save_time
                if (self.state.global_step == self.resume_global_step + self.args.save_steps) or (
                    hasattr(self.args, "flash_device_save_steps")
                    and (self.state.global_step == self.resume_global_step + self.args.flash_device_save_steps)
                ):
                    actual_tbk = self._start_save_time - self._first_end_save_time
                    actual_avg_speed_step = self.args.save_steps * tokens_per_steps / actual_tbk / self.args.world_size
                    tbk = tbk / (self.args.save_steps - self.first_skip_step) * self.args.save_steps
                if is_persistent_ckpt:
                    ts = time.time() - self._start_save_time
                else:
                    ts = time.time() - zcc_start_save_time
                logs["save_ckpt_time_sec"] = ts
                logs["global_save_step"] = self.state.global_step
                if is_persistent_ckpt:
                    tokens_per_steps = self.args.max_seq_len * total_train_batch_size
                    avg_speed_step = self.args.save_steps * tokens_per_steps / tbk / self.args.world_size
                    logs["train_time_sec_without_save"] = tbk
                    logs["average_tokens_per_sec_per_card_without_save"] = round(avg_speed_step, 1)
                    logs["average_tokens_per_sec_per_card_with_save"] = round(
                        self.args.save_steps * tokens_per_steps / (tbk + ts) / self.args.world_size,
                        2,
                    )
                    if self.state.global_step == self.resume_global_step + self.args.save_steps:
                        logs["actual_average_tokens_per_sec_per_card_without_save"] = round(actual_avg_speed_step, 1)
                        logs["actual_average_tokens_per_sec_per_card_with_save"] = round(
                            self.args.save_steps * tokens_per_steps / (actual_tbk + ts) / self.args.world_size,
                            2,
                        )
                    logs["one_day_billion_tokens_without_save"] = round(
                        0.0000864 * self.args.save_steps * tokens_per_steps / tbk, 2
                    )
                    logs["one_day_billion_tokens_with_save"] = round(
                        0.0000864 * self.args.save_steps * tokens_per_steps / (tbk + ts),
                        2,
                    )
                self.log(logs, **kwargs)
                if is_persistent_ckpt:
                    self._globalstep_last_start_time = time.time()
                    self._tokens_per_sec_per_card_buffer = []
            if is_persistent_ckpt:
                self._end_save_time = time.time()

    def create_scheduler(self, num_training_steps):
        if self.args.warmup_steps > 0:
            warmup = self.args.warmup_steps
        else:
            warmup = int(self.args.warmup_ratio * num_training_steps)

        assert self.args.lr_scheduler.startswith("wsd")
        scheduler = self.args.lr_scheduler.split(":")
        if len(scheduler) == 2:
            num_steady_steps = int(scheduler[1])
        else:
            num_steady_steps = None
        logger.info(f"using wsd lr scheduler, num_steady_steps={num_steady_steps}")
        self.lr_scheduler = get_wsd_schedule_with_warmup(
            self.args.learning_rate,
            warmup,
            self.args.max_steps,
            decay_function=self.args.decay_function,
            min_lr=self.args.min_lr if self.args.min_lr else 0.0,
            num_steady_steps=num_steady_steps,
        )

        return self.lr_scheduler

    def create_optimizer(self, lr_scheduler=None):
        optimizer_params = [p for n, p in self.model.named_parameters() if p.stop_gradient is False]
        if self.args.train_moe_only:
            optimizer_params = (
                [p for n, p in self.model.named_parameters() if "mlp.experts" in n or "mlp.gate" in n]
                if self.args.train_moe_only
                else [p for n, p in self.model.named_parameters() if p.stop_gradient is False]
            )
            logger.info(f"using `train_moe-only`, #moe params={len(optimizer_params)}")
        elif len(optimizer_params) < len(self.model.parameters()):
            logger.info(
                f"some params are not optimized, #totally={len(self.model.parameters())}, \
                  #optimized={len(optimizer_params)}"
            )
        if self.optimizer is None:
            decay_parameters = [
                p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
            ]

            def apply_decay_param_fun(x):
                return x in decay_parameters

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.args.use_moe and not self.args.use_hybrid_parallel:
                logger.info("using moe Global clip")

                def expert_fn(p):
                    return getattr(p, "no_sync", False)

                grad_clip = ClipGradForMOEByGlobalNorm(
                    self.args.max_grad_norm,
                    is_expert_param_func=expert_fn,
                    moe_group=_get_global_group(),
                    local_clip=False,
                )
            else:
                grad_clip = nn.ClipGradByGlobalNorm(self.args.max_grad_norm) if self.args.max_grad_norm > 0 else None

            self.static_name_to_dyg_name = {p.name: n for n, p in self.model.state_dict().items()}
            gate_pattern = re.compile(r"ernie\.layers\.0\.mlp\.gate\.weight")

            def lr_ratio_fn(param):
                name = self.static_name_to_dyg_name[param.name]
                if self.args.moe_gate_lr_ratio is not None and gate_pattern.match(name):
                    logger.info(f"apply moe_gate_lr_ratio to {name}, ratio={self.args.moe_gate_lr_ratio}")
                    return float(self.args.moe_gate_lr_ratio)
                return 1.0

            if isinstance(optimizer_kwargs["epsilon"], str):
                optimizer_kwargs["epsilon"] = float(optimizer_kwargs["epsilon"])
            self.optimizer = optimizer_cls(
                learning_rate=(self.lr_scheduler if lr_scheduler is None else lr_scheduler),
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=optimizer_params,
                weight_decay=self.args.weight_decay,
                grad_clip=grad_clip,
                multi_precision=True,
                lr_ratio=(lr_ratio_fn if (self.args.moe_gate_lr_ratio is not None) else None),
                **optimizer_kwargs,
            )

        return self.optimizer

    def save_model(self, output_dir=None):
        super().save_model(output_dir)
        if self.args.should_save:
            with open(os.path.join(output_dir, "static_name_to_dyg_name.json"), "w") as of:
                of.write(json.dumps(self.static_name_to_dyg_name))

    def _load_rng_state(self, checkpoint):
        pass
