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
GPT/Llama auto parallel pretraining scripts.
"""
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist

from paddleformers.ops import Topology
from paddleformers.trainer import PdArgumentParser, get_last_checkpoint
from paddleformers.trainer.auto_trainer import AutoTrainer
from paddleformers.trainer.auto_training_args import AutoTrainingArguments
from paddleformers.trainer.trainer_utils import IntervalStrategy, _get_distributed_seeds
from paddleformers.transformers import (
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    LlamaConfig,
    LlamaForCausalLM3DAuto,
    LlamaForCausalLMNet,
    LlamaPretrainingCriterion3DAuto,
    LlamaPretrainingCriterionNet,
)
from paddleformers.utils.log import logger

MODEL_CLASSES = {
    "llama": (LlamaConfig, LlamaForCausalLM3DAuto, LlamaPretrainingCriterion3DAuto),
    "llama_network": (LlamaConfig, LlamaForCausalLMNet, LlamaPretrainingCriterionNet),
}


from paddleformers.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
    print_rank_0,
)
from paddleformers.trainer.utils.doc import add_start_docstrings
from paddleformers.utils.tools import get_env_device


@dataclass
@add_start_docstrings(AutoTrainingArguments.__doc__)
class PreTrainingArguments(AutoTrainingArguments):
    min_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate deacyed to."},
    )
    decay_steps: float = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, will use the min_learning_rate."
        },
    )
    enable_linear_fused_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise add for grad accumulation in the backward of nn.Linear ."
        },
    )
    pipeline_schedule_mode: str = field(
        default="1F1B", metadata={"help": "The pipeline schedule mode, support FThenB, 1F1B, VPP and Eager-1F1B."}
    )
    sr: Optional[int] = field(default=0, metadata={"help": "The count of chunks without recompute."})
    virtual_pipeline_seg_method: str = field(
        default="LlamaDecoderLayerAuto",
        metadata={"help": "The seg method of splitting pp layer for virtual pipeline."},
    )
    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    n_microbatches: int = field(
        default=1,
        metadata={"help": "Control the num of microbatches in one pp step."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.enable_auto_parallel

        # NOTE(gongenlei): new add autotuner_benchmark
        if self.autotuner_benchmark:
            self.max_steps = 5
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO

        logger.info(self.strategy)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluating.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    input_dir: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split."})

    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    share_folder: bool = field(
        default=False,
        metadata={"help": "Use share folder for data dir and output dir on multi machine."},
    )

    data_impl: str = field(default="mmap", metadata={"help": "The format of the preprocessed data."})
    skip_warmup: bool = field(
        default=True,
        metadata={"help": "Whether to skip the warmup process of mmap files."},
    )
    data_cache: str = field(default=None, metadata={"help": "The path of the cached dataset."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(
        default="llama", metadata={"help": "Only support for llama pre-training for now."}
    )
    model_name_or_path: str = field(
        default="__internal_testing__/tiny-random-llama",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    use_fast_layer_norm: bool = field(
        default=False,
        metadata={"help": "GPT3 model, use fast layernorm"},
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    vocab_size: Optional[int] = field(
        default=None,
        metadata={
            "help": ".Vocabulary size of the Llama model. Defines the number of different tokens that can be represented by the `inputs_ids`"
        },
    )
    hidden_size: Optional[int] = field(default=None, metadata={"help": "Dimension of the hidden representations."})
    intermediate_size: Optional[int] = field(default=None, metadata={"help": "Dimension of the MLP representations."})
    num_hidden_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of hidden layers in the Transformer encoder."}
    )
    num_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of attention heads for each attention layer in the Transformer encoder."},
    )
    fuse_rms_norm: bool = field(
        default=False,
        metadata={"help": "llama, fuse_rms_norm"},
    )
    recompute_granularity: str = field(
        default="full",
        metadata={"help": "Choose among ['full', 'core_attn', 'full_attn']"},
    )
    virtual_pipeline_model_parallel_size: int = field(
        default=1,
        metadata={"help": "virtual_pipeline_model_parallel_size"},
    )
    continue_training: bool = field(
        default=False,
        metadata={
            "help": "Pre-training from existing paddlenlp model weights. Default False and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models."
        },
    )
    apply_rope_fusion: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable rope fusion or not."},
    )
    no_recompute_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Specify the full transformer layers that should not be recomputed."},
    )
    pp_recompute_interval: int = field(
        default=1,
        metadata={
            "help": "The interval for the number of layers at which recomputation occurs. A value of 0 indicates no recomputation. Default is 0."
        },
    )
    recompute_use_reentrant: bool = field(
        default=False,
        metadata={"help": "recompute_use_reentrant"},
    )


def create_pretrained_dataset(
    data_args,
    training_args,
    data_file,
    tokenizer,
    need_data=True,
):

    check_data_split(data_args.split, training_args.do_train, training_args.do_eval, training_args.do_predict)

    train_val_test_num_samples = [
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps,
        training_args.per_device_eval_batch_size
        * training_args.dataset_world_size
        * training_args.eval_iters
        * (training_args.max_steps // training_args.eval_steps + 1),
        training_args.per_device_eval_batch_size * training_args.dataset_world_size * training_args.test_iters,
    ]

    print_rank_0(" > datasets target sizes (minimum size):")
    if training_args.do_train:
        print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))
    if training_args.do_eval:
        print_rank_0("    validation: {}".format(train_val_test_num_samples[1]))
    if training_args.do_predict:
        print_rank_0("    test:       {}".format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_file,
        data_impl=data_args.data_impl,
        splits_string=data_args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=data_args.max_seq_length,
        seed=training_args.seed,
        skip_warmup=data_args.skip_warmup,
        share_folder=data_args.share_folder,
        data_cache_path=data_args.data_cache,
        need_data=need_data,
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode.")
        # input_ids, loss_mask, attention_mask, position_ids, labels = data
        input_ids = data["text"]

        logger.info(tokenizer._decode(input_ids))

    from paddleformers.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in data])

        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]

        return {
            "input_ids": tokens,
            "labels": labels,
        }

    if need_data:
        if training_args.do_train:
            print_dataset(train_dataset[0], "train")
        if training_args.do_eval:
            print_dataset(valid_dataset[0], "valid")
        if training_args.do_predict:
            print_dataset(test_dataset[0], "test")

    return train_dataset, valid_dataset, test_dataset, _collate_data


def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return args.input_dir.split()
    else:
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and ("_idx.npz" in str(f) or ".idx" in str(f)))
        ]
        files = [x.replace("_idx.npz", "") for x in files]
        files = [x.replace(".idx", "") for x in files]

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret

    return files


class PretrainingTrainer(AutoTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_pretraining = True

    def _wrap_for_dist_loader(self, train_dataloader):
        dist_loader = super()._wrap_for_dist_loader(train_dataloader)
        dist_loader._input_keys = ["input_ids", "labels"]
        return dist_loader

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None:
            return None

        total_batch_size_per_acc_step = self.args.per_device_train_batch_size * self.args.dataset_world_size
        total_batch_size = total_batch_size_per_acc_step

        # In llm/llama/run_pretrain.py, it uses paddleformers.utils.batch_sampler.DistributedBatchSampler,
        # which does no shuffle when shuffle is set True.
        sampler = paddle.io.BatchSampler(
            dataset=self.train_dataset,
            shuffle=False,
            batch_size=total_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )
        sampler._acc_steps = self.args.gradient_accumulation_steps
        return sampler


def init_seed(seed: int = 1234, args=None):
    if args is None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
    else:
        assert not args.use_hybrid_parallel and args.enable_auto_parallel
        if dist.get_world_size() > 1:
            if args.hybrid_parallel_topo_order is None or args.hybrid_parallel_topo_order == "pp_first":
                order = ["pp", "dp", "sharding", "mp", "sep"]
            elif args.hybrid_parallel_topo_order == "sharding_first":
                order = ["dp", "sharding", "pp", "mp", "sep"]
            if args.context_parallel_size is not None and args.context_parallel_size > 1:
                sep_degree = args.context_parallel_size
            elif args.sep_parallel_size is not None and args.sep_parallel_size > 1:
                sep_degree = args.sep_parallel_size
            else:
                sep_degree = 1
            sep_degree = args.sep_parallel_size if args.sep_parallel_size > 1 else args.context_parallel_size
            topo = Topology(
                dist.get_rank(),
                dist.get_world_size(),
                dp_degree=args.dataset_world_size,
                pp_degree=args.pipeline_model_parallel_size,
                mp_degree=args.tensor_model_parallel_size,
                sep_degree=sep_degree,
                sharding_degree=1,  # auto_parallel's sharding is not orthogonal with dp, mp and pp
                order=order,
            )

            global_seed, local_seed, random_seed = _get_distributed_seeds(args.seed, topo)

            paddle.seed(local_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            logger.info(
                "The global seed is set to {}, local seed is set to {} and "
                "random seed is set to {}.".format(global_seed, local_seed, random_seed)
            )
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    do_enable_linear_fused_grad_add = training_args.enable_linear_fused_grad_add
    do_enable_mp_async_allreduce = (
        training_args.enable_auto_parallel
        and training_args.tensor_model_parallel_size > 1
        and training_args.mp_async_allreduce
        and not training_args.sequence_parallel
    )
    do_enable_sp_async_reduce_scatter = (
        training_args.enable_auto_parallel
        and training_args.tensor_model_parallel_size > 1
        and training_args.sequence_parallel
        and training_args.sp_async_reduce_scatter
    )
    if (
        do_enable_linear_fused_grad_add or do_enable_mp_async_allreduce or do_enable_sp_async_reduce_scatter
    ) and not training_args.to_static:
        from llm.utils.fused_layers import mock_layers

        mock_layers(do_enable_linear_fused_grad_add, do_enable_mp_async_allreduce, do_enable_sp_async_reduce_scatter)

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    init_seed(args=training_args)
    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    training_args.eval_iters = 10
    training_args.test_iters = training_args.eval_iters * 10

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    config_class, model_class, criterion_class = MODEL_CLASSES[model_args.model_type]

    config = config_class.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    config.use_fast_layer_norm = model_args.use_fast_layer_norm

    config.seq_length = data_args.max_seq_length
    # There are some technique extend RotaryEmbedding context. so don't change max_position_embeddings
    if not model_args.continue_training:
        config.max_position_embeddings = max(config.max_position_embeddings, data_args.max_seq_length)

    if not model_args.continue_training:
        config.vocab_size = max(config.vocab_size, ((tokenizer.vocab_size - 1) // 128 + 1) * 128)
        logger.info(f"Reset vocab size to {config.vocab_size} for batter amp performance.")

    if model_args.no_recompute_layers is not None:
        model_args.no_recompute_layers.sort()

    config.vocab_size = model_args.vocab_size if model_args.vocab_size is not None else config.vocab_size
    config.hidden_size = model_args.hidden_size if model_args.hidden_size is not None else config.hidden_size
    config.intermediate_size = (
        model_args.intermediate_size if model_args.intermediate_size is not None else config.intermediate_size
    )
    config.num_hidden_layers = (
        model_args.num_hidden_layers if model_args.num_hidden_layers is not None else config.num_hidden_layers
    )
    config.num_attention_heads = (
        model_args.num_attention_heads if model_args.num_attention_heads is not None else config.num_attention_heads
    )

    config.fuse_rms_norm = model_args.fuse_rms_norm
    config.recompute_granularity = model_args.recompute_granularity
    config.virtual_pipeline_model_parallel_size = model_args.virtual_pipeline_model_parallel_size
    config.sequence_parallel = training_args.sequence_parallel

    config.fuse_sequence_parallel_allreduce = training_args.fuse_sequence_parallel_allreduce

    config.apply_rope_fusion = model_args.apply_rope_fusion
    config.no_recompute_layers = model_args.no_recompute_layers
    config.pp_recompute_interval = model_args.pp_recompute_interval
    config.recompute_use_reentrant = model_args.recompute_use_reentrant

    config.use_recompute = training_args.recompute
    config.tensor_model_parallel_size = training_args.tensor_model_parallel_size
    config.tensor_parallel_rank = training_args.tensor_parallel_rank
    config.sharding_parallel_size = training_args.sharding_parallel_size
    config.to_static = training_args.to_static
    config.sep_parallel_size = training_args.sep_parallel_size
    config.context_parallel_size = training_args.context_parallel_size

    if training_args.strategy.pipeline.enable and config.virtual_pipeline_model_parallel_size > 1:
        pipeline = training_args.strategy.pipeline
        pipeline.vpp_degree = config.virtual_pipeline_model_parallel_size
        pipeline.vpp_seg_method = training_args.virtual_pipeline_seg_method
    if get_env_device() == "xpu" and training_args.gradient_accumulation_steps > 1:
        try:
            from paddle_xpu.layers.nn.linear import LinearConfig  # noqa: F401

            LinearConfig.enable_accumulate_steps_opt()
            LinearConfig.set_accumulate_steps(training_args.gradient_accumulation_steps)
        except ImportError:
            # It's OK, not use accumulate_steps optimization
            pass

    print("Final pre-training config:", config)

    if (
        training_args.replace_with_parallel_cross_entropy
        and config.tensor_model_parallel_size > 1
        and config.to_static is False
    ):
        from llm.utils.replace_ops import replace_cross_entropy

        replace_cross_entropy()

    # # Set the dtype for loading model
    # dtype = "float32"
    # if training_args.fp16_opt_level == "O2":
    #     if training_args.fp16:
    #         dtype = "float16"
    #     if training_args.bf16:
    #         dtype = "bfloat16"

    with paddle.LazyGuard():
        model = model_class.from_config(config, dtype="float32")
        criterion = criterion_class(config)

    if training_args.recompute:

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True

        model.apply(fn)

    # Create the learning_rate scheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    if training_args.warmup_steps > 0:
        warmup_steps = training_args.warmup_steps
    else:
        warmup_steps = training_args.warmup_ratio * training_args.max_steps

    lr_scheduler = None
    if training_args.lr_scheduler_type.value == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    elif training_args.lr_scheduler_type.value == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )

    data_file = get_train_data_file(data_args)
    train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
        data_args,
        training_args,
        data_file,
        tokenizer,
        need_data=training_args.should_load_dataset,
    )
    trainer = PretrainingTrainer(
        model=model,
        model_type=model_args.model_type,
        criterion=criterion,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(None, lr_scheduler),
        tokenizer=tokenizer,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # NOTE(gongenlei): new add
        if not training_args.autotuner_benchmark:
            metrics = train_result.metrics
            if not int(os.getenv("test_ci_no_save_model", 0)):
                trainer.save_model()
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    if training_args.do_predict:
        test_ret = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_ret.metrics)


if __name__ == "__main__":
    main()
