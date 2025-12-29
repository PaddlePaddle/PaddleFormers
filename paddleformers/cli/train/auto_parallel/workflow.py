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

import paddle

from paddleformers.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
    print_rank_0,
)
from paddleformers.trainer import get_last_checkpoint
from paddleformers.trainer.trainer import Trainer
from paddleformers.trainer.trainer_utils import set_seed
from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
)
from paddleformers.transformers.configuration_utils import LlmMetaConfig
from paddleformers.utils.log import logger
from paddleformers.utils.tools import get_env_device


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
        seq_length=data_args.max_seq_len,
        seed=training_args.seed,
        skip_warmup=data_args.skip_warmup,
        share_folder=data_args.share_folder,
        data_cache_path=data_args.data_cache,
        need_data=need_data,
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode.")
        input_ids = data["text"]

        logger.info(tokenizer._decode(list(input_ids)))

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


class PretrainingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_pretraining = True


def run_auto_parallel(model_args, data_args, generating_args, training_args):

    do_enable_linear_fused_grad_add = training_args.enable_linear_fused_grad_add
    # do_enable_mp_async_allreduce = (
    #     training_args.enable_auto_parallel
    #     and training_args.tensor_model_parallel_size > 1
    #     and training_args.mp_async_allreduce
    #     and not training_args.sequence_parallel
    # )
    # do_enable_sp_async_reduce_scatter = (
    #     training_args.enable_auto_parallel
    #     and training_args.tensor_model_parallel_size > 1
    #     and training_args.sequence_parallel
    #     and training_args.sp_async_reduce_scatter
    # )
    if (
        do_enable_linear_fused_grad_add
        # do_enable_linear_fused_grad_add or do_enable_mp_async_allreduce or do_enable_sp_async_reduce_scatter
    ) and not training_args.to_static:
        from llm.utils.fused_layers import mock_layers

        # mock_layers(do_enable_linear_fused_grad_add, do_enable_mp_async_allreduce, do_enable_sp_async_reduce_scatter)
        mock_layers(do_enable_linear_fused_grad_add)

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    paddle.set_device(training_args.device)
    set_seed(seed=training_args.seed)

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

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    LlmMetaConfig.set_llm_config(config, training_args)
    config.use_fast_layer_norm = model_args.use_fast_layer_norm

    config.seq_length = data_args.max_seq_len
    config.max_sequence_length = data_args.max_seq_len
    # There are some technique extend RotaryEmbedding context. so don't change max_position_embeddings
    if not model_args.continue_training:
        config.max_position_embeddings = max(config.max_position_embeddings, data_args.max_seq_len)

    if not model_args.continue_training:
        config.vocab_size = max(config.vocab_size, ((tokenizer.vocab_size - 1) // 128 + 1) * 128)
        logger.info(f"Reset vocab size to {config.vocab_size} for batter amp performance.")

    config.num_hidden_layers = (
        model_args.num_hidden_layers if model_args.num_hidden_layers is not None else config.num_hidden_layers
    )

    # Config for model using dropout, such as GPT.
    if hasattr(config, "use_dualpipev"):
        # NOTE(zhangyuqin): In Paddle, the segmentation and scheduling of pipeline parallel
        # models are separate. Therefore, first we need to set the flag in the model config
        # to perform V-shape segmentation. Second, we need to set the flag in the training_args
        # to configure strategy.hybrid_configs to choose the DualPipeV schedule.
        config.use_dualpipev = training_args.use_dualpipev
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob

    if config.sequence_parallel:
        assert (
            config.tensor_model_parallel_size > 1
        ), "tensor_model_parallel_size must be larger than 1 for sequence parallel."
    assert (
        config.num_attention_heads % config.sep_parallel_size == 0
    ), f"num_attention_heads:{config.num_attention_heads} must be divisible by sep_parallel_size {config.sep_parallel_size}"
    assert (
        config.seq_length % config.context_parallel_size == 0
    ), f"seq_length:{config.seq_length} must be divisible by context_parallel_size {config.context_parallel_size}"

    # for stage1 overlap optimization
    if training_args.stage1_allgather_overlap or training_args.stage1_broadcast_overlap:
        from paddle.io.reader import use_pinned_memory

        use_pinned_memory(False)

    if get_env_device() == "xpu" and training_args.gradient_accumulation_steps > 1:
        try:
            from paddle_xpu.layers.nn.linear import LinearConfig  # noqa: F401

            LinearConfig.enable_accumulate_steps_opt()
            LinearConfig.set_accumulate_steps(training_args.gradient_accumulation_steps)
        except ImportError:
            # It's OK, not use accumulate_steps optimization
            pass

    if training_args.no_recompute_layers is not None:
        training_args.no_recompute_layers.sort()

    if training_args.use_intermediate_api:
        config.use_single_model_implementation = True
        config.tensor_model_parallel_size = 1
        config.sharding_parallel_size = 1
        config.sep_parallel_size = 1
        config.context_parallel_size = 1

    print("Final pre-training config:", config)

    # Set the dtype for loading model
    dtype = "float32"
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    model_class = AutoModelForCausalLM

    if not training_args.enable_auto_parallel and training_args.pipeline_model_parallel_size > 1:
        model_class = AutoModelForCausalLMPipe

    architectures_to_check = {"Qwen2Moe", "DeepseekV2", "DeepseekV3"}
    if (
        any(architecture in str(config.architectures) for architecture in architectures_to_check)
        and training_args.data_parallel_size > 1
    ):
        training_args.use_expert_parallel = True

    if model_args.continue_training:
        if training_args.autotuner_benchmark:
            model = model_class.from_config(config, dtype=dtype)
        else:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                dtype=dtype,
            )
    else:
        if training_args.enable_auto_parallel:
            with paddle.LazyGuard():
                model = model_class.from_config(config, dtype=dtype)
        else:
            model = model_class.from_config(config, dtype=dtype)

    if training_args.recompute_granularity is not None:

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
            min_lr=training_args.min_lr,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    elif training_args.lr_scheduler_type.value == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_lr,
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

    total_effective_tokens = (
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps
        * data_args.max_seq_len
    )

    trainer = PretrainingTrainer(
        model=model,
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

    if training_args.do_train and training_args.should_load_dataset:
        effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
        print(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")
        print(f"ips: {effective_tokens_per_second:.2f} tokens/s")
