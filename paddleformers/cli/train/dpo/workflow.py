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

"""Training DPO"""

import os
from functools import partial

import paddle

from paddleformers.datasets.dpo import collate_fn, create_dataset
from paddleformers.nn.attention import AttentionInterface
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.trainer import (
    IntervalStrategy,
    MoECorrectionBiasAdjustCallback,
    MoeExpertsGradScaleCallback,
    MoEGateSpGradSyncCallBack,
    get_last_checkpoint,
    set_seed,
)
from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
)
from paddleformers.transformers.configuration_utils import LlmMetaConfig
from paddleformers.trl import DPOTrainer
from paddleformers.trl.llm_utils import get_lora_target_modules
from paddleformers.utils.log import logger

from ...hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
)
from .dpo_argument import DPOConfig
from .dpo_estimate_training import dpo_estimate_training


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    generating_args: "GeneratingArguments",
    training_args: "FinetuningArguments",
):
    """main"""
    paddle.set_device(training_args.device)
    set_seed(training_args.seed)

    avaible_attn_impl = AttentionInterface._global_mapping.keys()
    if model_args.attn_impl not in avaible_attn_impl:
        raise ValueError(f"Invalid attn_impl: {model_args.attn_impl}, available attn_impl: {avaible_attn_impl}")

    if training_args.loss_type == "orpo":
        training_args.reference_free = True
        training_args.sft_loss_ratio = 1.0
        training_args.loss_type = "or"
        logger.info("orpo loss_type is equal to sft_loss + pref_loss_ratio * or_loss.")
    if training_args.loss_type in ["or", "simpo"] and not training_args.reference_free:
        training_args.reference_free = True
        logger.warning(
            f"{training_args.loss_type} loss_type only supports reference_free. Set reference_free to True."
        )
    if training_args.pipeline_parallel_degree > 1:
        assert (
            hasattr(training_args, "pipeline_parallel_config")
            and "enable_clear_every_step_cache" in training_args.pipeline_parallel_config
        ), "Should set '--pipeline_parallel_config enable_clear_every_step_cache' in bash script for pp."
    if training_args.sequence_parallel:
        if training_args.pipeline_parallel_degree > 1:
            assert (
                hasattr(training_args, "pipeline_parallel_config")
                and "disable_partial_send_recv" in training_args.pipeline_parallel_config
            ), "Should set '--pipeline_parallel_config disable_partial_send_recv' in bash script for pp with sp."
        if training_args.tensor_parallel_degree <= 1:
            training_args.sequence_parallel = False
            logger.info("Tensor_parallel_degree = 1. Set sequence_parallel to False.")
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(training_args, "Train")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: "
        f"{training_args.world_size}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model & tokenizer.")

    dpo_config = DPOConfig(
        beta=training_args.beta,
        offset_alpha=training_args.offset_alpha,
        simpo_gamma=training_args.simpo_gamma,
        normalize_logps=training_args.normalize_logps,
        ignore_eos_token=training_args.ignore_eos_token,
        label_smoothing=training_args.label_smoothing,
        loss_type=training_args.loss_type,
        pref_loss_ratio=training_args.pref_loss_ratio,
        sft_loss_ratio=training_args.sft_loss_ratio,
        dpop_lambda=training_args.dpop_lambda,
        ref_model_update_steps=training_args.ref_model_update_steps,
        reference_free=training_args.reference_free,
        lora=model_args.lora,
    )

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
    )
    model_config._attn_implementation = model_args.attn_impl
    model_config.pp_seg_method = model_args.pp_seg_method
    model_config.max_sequence_length = data_args.max_seq_len
    model_config.seq_length = data_args.max_seq_len

    LlmMetaConfig.set_llm_config(model_config, training_args)

    if not training_args.reference_free and not model_args.lora:
        ref_model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            dtype=dtype,
        )
        ref_model_config.pp_seg_method = model_args.pp_seg_method
        ref_model_config.max_sequence_length = data_args.max_seq_len
        ref_model_config.seq_length = data_args.max_seq_len
        ref_model_config._attn_implementation = model_args.attn_impl

        LlmMetaConfig.set_llm_config(ref_model_config, training_args)

    if training_args.pipeline_parallel_degree > 1:
        model_class = AutoModelForCausalLMPipe
    else:
        model_class = AutoModelForCausalLM
    if not training_args.reference_free and not model_args.lora:
        ref_model_config.dpo_config = dpo_config
    model_config.dpo_config = dpo_config
    if not training_args.autotuner_benchmark or training_args.weight_quantize_algo is not None:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            convert_from_hf=training_args.convert_from_hf,
            load_via_cpu=training_args.load_via_cpu,
            load_checkpoint_format=training_args.load_checkpoint_format,
        )
        # for DPO save
        if not training_args.reference_free and not model_args.lora:
            ref_model = model_class.from_config(ref_model_config)
            ref_model.set_state_dict(model.state_dict())
        else:
            ref_model = None
    else:
        model = model_class.from_config(model_config)
        if not training_args.reference_free and not model_args.lora:
            ref_model = model_class.from_config(ref_model_config)
        else:
            ref_model = None
    if training_args.pipeline_parallel_degree > 1:
        model.config.dpo_config = None

    if model_args.tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    logger.info("Loading model & tokenizer successfully !")

    if model_args.lora:
        if training_args.sharding_parallel_degree > 1:
            assert (
                "enable_stage1_overlap" not in training_args.sharding_parallel_config
            ), "Currently not support enabling sharding_stage1_overlap in lora mode."
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            if model_args.rslora_plus:
                model_args.rslora = True
                model_args.lora_plus_scale = 4
                model_args.lora_alpha = 4
            if training_args.weight_quantize_algo is not None:
                if model_args.rslora or model_args.lora_plus_scale != 1.0:
                    logger.info("Weight quantization is not supported in LoRA+ and RsLoRA.")
            if model_args.lora_alpha == -1:
                if model_args.rslora:
                    model_args.lora_alpha = 4
                else:
                    model_args.lora_alpha = 2 * model_args.lora_rank
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank if not model_args.rslora else 4,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
                use_quick_lora=model_args.use_quick_lora,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)

        model.print_trainable_parameters()

    logger.info("Start to create dataset")
    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": data_args.max_seq_len,
        "max_prompt_len": data_args.max_prompt_len,
        "random_seed": training_args.seed,
        "num_replicas": training_args.dataset_world_size,
        "rank": training_args.dataset_rank,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "buffer_size": data_args.buffer_size,
        "use_attn_mask_startend_row_indices": model_args.use_attn_mask_startend_row_indices,
        "mask_out_eos_token": data_args.mask_out_eos_token,
        "random_shuffle": data_args.random_shuffle,
        "greedy_intokens": data_args.greedy_intokens,
        "packing": data_args.packing,
        "mix_strategy": data_args.mix_strategy,
        "encode_one_turn": data_args.encode_one_turn,
    }
    if training_args.max_steps == -1:
        if data_args.mix_strategy == "random":
            raise ValueError(
                "When using 'random' mix_strategy, max_steps must be explicitly set (cannot be -1). "
                "Random mixing requires a fixed number of training steps to properly sample data."
            )
        if training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            training_args, _ = dpo_estimate_training(tokenizer, data_args, training_args, config=model.config)

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            pd_max_steps = paddle.to_tensor([training_args.max_steps])
            paddle.distributed.broadcast(pd_max_steps, src=0)
            training_args.max_steps = int(pd_max_steps.item())
        logger.info(
            f"Re-setting training_args.max_steps to {training_args.max_steps} ({training_args.num_train_epochs})"
        )
        if training_args.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {training_args.max_steps}. Please check your dataset")
    if training_args.save_strategy == IntervalStrategy.EPOCH:
        training_args.save_strategy = IntervalStrategy.STEPS
        training_args.save_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.evaluation_strategy == IntervalStrategy.EPOCH:
        training_args.evaluation_strategy = IntervalStrategy.STEPS
        training_args.eval_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.logging_strategy == IntervalStrategy.EPOCH:
        training_args.logging_strategy = IntervalStrategy.STEPS
        training_args.logging_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.do_train and training_args.should_load_dataset:
        train_dataset = create_dataset(
            task_group=data_args.train_dataset_path,
            task_group_prob=data_args.train_dataset_prob,
            sub_dataset_type=data_args.train_dataset_type,
            **dataset_config,
        )
    else:
        train_dataset = None

    if training_args.do_eval and training_args.should_load_dataset:
        eval_dataset = create_dataset(
            task_group=data_args.eval_dataset_path,
            task_group_prob=data_args.eval_dataset_prob,
            sub_dataset_type=data_args.eval_dataset_type,
            is_valid=True,
            **dataset_config,
        )
    else:
        eval_dataset = None
    logger.info("Creating dataset successfully ...")

    callbacks = []
    if getattr(model_config, "topk_method", None) == "noaux_tc":
        callbacks += [MoECorrectionBiasAdjustCallback(lr=0)]

    if training_args.use_expert_parallel:
        callbacks += [MoeExpertsGradScaleCallback(training_args)]

    if training_args.sequence_parallel and not model_args.lora:
        callbacks += [MoEGateSpGradSyncCallBack()]

    logger.info(f"callbacks: {callbacks}")
    # padding to the maximum seq length in batch data when max_seq_len is None
    max_seq_len = data_args.max_seq_len if (data_args.packing or training_args.sequence_parallel) else None
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        dpo_config=dpo_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            use_sparse_head_and_loss_fn=model_config.use_sparse_head_and_loss_fn,
            use_fused_head_and_loss_fn=model_config.use_fused_head_and_loss_fn,
        ),
        ignore_eos_token=dpo_config.ignore_eos_token,
        model_with_dpo_criterion=model_args.model_with_dpo_criterion,
        callbacks=callbacks,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        if not training_args.autotuner_benchmark and not training_args.benchmark:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1, last_fc_to_hf=True)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
