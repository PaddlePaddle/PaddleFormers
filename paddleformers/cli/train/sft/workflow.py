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

"""Training Ernie Model."""

import gc
import math
import os
from functools import partial

import numpy as np
import paddle

from paddleformers.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
)
from paddleformers.datasets.collate import collate_fn, mm_collate_fn
from paddleformers.datasets.data_utils import estimate_training
from paddleformers.datasets.loader import create_dataset as create_dataset_sft
from paddleformers.datasets.template.template import get_template_and_fix_tokenizer
from paddleformers.nn.attention import AttentionInterface
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.trainer import (
    IntervalStrategy,
    MoECorrectionBiasAdjustCallback,
    MoeExpertsGradScaleCallback,
    MoEGateSpGradSyncCallBack,
    get_last_checkpoint,
    set_random_seed,
    set_seed,
)
from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoModelForConditionalGeneration,
    AutoModelForConditionalGenerationPipe,
    AutoProcessor,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
)
from paddleformers.transformers.configuration_utils import LlmMetaConfig
from paddleformers.utils.import_utils import is_paddlefleet_available
from paddleformers.utils.log import logger

from .sft_trainer import SFTTrainer

# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"

from paddleformers.cli.hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
)
from paddleformers.cli.utils import (
    freeze_model_parameters,
    get_lora_target_modules,
    get_multimodel_lora_target_modules,
)


def create_pretrained_dataset(training_args, data_args, model_args):
    assert data_args.input_dir is not None and len(data_args.input_dir.split()) > 1

    check_data_split(
        data_args.split,
        training_args.do_train,
        training_args.do_eval,
        training_args.do_predict,
    )

    if training_args.max_steps < 0:
        raise ValueError(
            f"max_steps mush be larger than 0 when using pretrain offline dataset, but get {training_args.max_steps}."
        )

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

    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_args.input_dir.split(),
        data_impl="mmap",
        splits_string=data_args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=data_args.max_seq_len + training_args.num_nextn_predict_layers,
        seed=training_args.seed,
        skip_warmup=True,
        data_cache_path=None,
    )

    from paddleformers.data import Stack

    def _collate_data(batch, stack_fn=Stack()):
        input_keys = ["input_ids", "labels", "position_ids", "attn_mask_startend_row_indices"]
        return_list = []
        for batch_sequence in batch:
            # tokens
            padded_token_ids = np.array([batch_sequence["text"][:-1]])
            # labels
            padded_labels = np.array([batch_sequence["text"][1:]])
            # position_ids
            padded_position_ids = np.array([sum(batch_sequence["position_ids"], [])[:-1]])
            return_list.append(
                [
                    padded_token_ids,
                    padded_labels,
                    padded_position_ids,
                ]
            )
            # attn mask
            oral_position_ids = batch_sequence["position_ids"]
            from paddleformers.datasets.collate import (
                gen_attn_mask_startend_row_indices,
            )

            return_list[-1].append(
                gen_attn_mask_startend_row_indices(
                    oral_position_ids,
                    data_args.max_seq_len + training_args.num_nextn_predict_layers,
                    model_args.use_global_causal_attn,
                )[:, :, :-1, :]
            )

        return_list = [np.concatenate(tensor_list) for tensor_list in zip(*return_list)]
        input_dict = dict(zip(input_keys, return_list))
        return input_dict

    return train_dataset, valid_dataset, test_dataset, _collate_data


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    generating_args: "GeneratingArguments",
    finetuning_args: "FinetuningArguments",
):
    """_summary_

    Args:
        model_args (ModelArguments): _description_
        data_args (DataArguments): _description_
        generating_args (GeneratingArguments): _description_
        finetuning_args (FinetuningArguments): _description_
        callbacks (Optional[list[&quot;TrainerCallback&quot;]], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """

    training_args = finetuning_args
    training_args.max_seq_len = data_args.max_seq_len
    if is_paddlefleet_available() and model_args.lora and training_args.moe_token_dispatcher_type == "deepep":
        logger.warning("For PaddleFleet, moe_use_fusion_node should False when using LoRA.")
        training_args.moe_use_fusion_node = False

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(training_args, "Train")

    if training_args.pre_alloc_memory > 0:
        memory_size = int(training_args.pre_alloc_memory * 1024 * 1024 * 1024)
        x = paddle.empty([memory_size], dtype=paddle.uint8)
        logger.info(f"pre_alloc_memory size {x.shape}")
        del x

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    set_random_seed(seed_=training_args.seed)
    set_seed(seed=training_args.seed)
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

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
    )

    architectures_to_check = {"Qwen2Moe", "DeepseekV2", "DeepseekV3"}
    if (
        any(architecture in str(model_config.architectures) for architecture in architectures_to_check)
        and training_args.data_parallel_size > 1
        and not training_args.use_expert_parallel
    ):
        raise ValueError("Please set use_expert_parallel to true in expert parallel mode.")

    # (Liuting) Not support acc calculation now due to MTP.
    if "DeepseekV3" in str(model_config.architectures):
        training_args.prediction_loss_only = True

    if "qwen3_vl" in model_config.model_type and not model_args.lora:
        if training_args.sequence_parallel:
            logger.warning("Qwen3VL model do not support `sequence_parallel` yet, temporarily set to False")
        training_args.sequence_parallel = False

    LlmMetaConfig.set_llm_config(model_config, training_args)
    model_config.use_fast_layer_norm = model_args.use_fast_layer_norm

    # Config for model using dropout, such as GPT.
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = finetuning_args.hidden_dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = finetuning_args.attention_probs_dropout_prob
    if hasattr(model_config, "ignore_index"):
        model_config.ignore_index = -100

    avaible_attn_impl = AttentionInterface._global_mapping.keys()
    if model_args.attn_impl not in avaible_attn_impl:
        raise ValueError(f"Invalid attn_impl: {model_args.attn_impl}, available attn_impl: {avaible_attn_impl}")

    model_config.pp_seg_method = model_args.pp_seg_method
    model_config.seq_length = data_args.max_seq_len
    model_config.max_sequence_length = data_args.max_seq_len
    model_config._attn_implementation = model_args.attn_impl
    model_config.is_lora = model_args.lora

    # Sync arguments to MLLM sub_config
    if getattr(model_config, "text_config", None) is not None:
        model_config.text_config.max_sequence_length = data_args.max_seq_len
    if getattr(model_config, "vision_config", None) is not None:
        model_config.vision_config._attn_implementation = model_args.attn_impl
        model_config.vision_config.recompute_granularity = model_config.recompute_granularity
        model_config.vision_config.recompute_method = model_config.recompute_method
        model_config.vision_config.recompute_num_layers = model_config.recompute_num_layers

    logger.info(f"Final model config: {model_config}")
    logger.info("Creating model")

    if model_args.stage == "VL-SFT":
        model_class = AutoModelForConditionalGeneration
        if training_args.pipeline_model_parallel_size > 1:
            if data_args.eval_with_do_generation and training_args.do_eval:
                raise ValueError("Please set eval_with_do_generation to false in pipeline parallel mode.")
            model_class = AutoModelForConditionalGenerationPipe
    else:
        model_class = AutoModelForCausalLM
        if training_args.pipeline_model_parallel_size > 1:
            if data_args.eval_with_do_generation and training_args.do_eval:
                raise ValueError("Please set eval_with_do_generation to false in pipeline parallel mode.")
            model_class = AutoModelForCausalLMPipe

    if model_args.continue_training and not training_args.autotuner_benchmark:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            convert_from_hf=training_args.convert_from_hf,
            load_via_cpu=training_args.load_via_cpu,
            load_checkpoint_format=training_args.load_checkpoint_format,
        )
    else:
        model = model_class.from_config(model_config, dtype=dtype)

    if training_args.do_train and model_args.neftune:
        # Inspired by https://github.com/neelsjain/NEFTune
        if hasattr(model, "get_input_embeddings"):

            def neft_post_hook(module, input, output):
                if module.training:
                    mag_norm = model_args.neftune_noise_alpha / paddle.sqrt(
                        paddle.to_tensor(output.shape[0] * output.shape[1], dtype="float32")
                    )
                    output = output + paddle.uniform(
                        shape=output.shape, dtype=output.dtype, min=-mag_norm, max=mag_norm
                    )
                return output

            neft_post_hook_handle = model.get_input_embeddings().register_forward_post_hook(neft_post_hook)
        else:
            raise NotImplementedError("Only support neftune for model with get_input_embeddings")

    # Load tokenizer & processor & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    dataset_config = {
        "tokenizer": tokenizer,
        "processor": processor,
        "max_seq_len": data_args.max_seq_len,
        "random_seed": training_args.seed,
        "num_replicas": training_args.dataset_world_size,
        "rank": training_args.dataset_rank,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "random_shuffle": data_args.random_shuffle,
        "greedy_intokens": data_args.greedy_intokens,
        "packing": data_args.packing,
        "mix_strategy": data_args.mix_strategy,
        "encode_one_turn": data_args.encode_one_turn,
        "use_template": data_args.use_template,
        "is_pretraining": True if model_args.stage.lower() == "pt" else False,
        "truncate_packing": data_args.truncate_packing,
        "stage": model_args.stage,
        "template_backend": data_args.template_backend,
        "split_multi_turn": data_args.split_multi_turn,
    }

    dataset_config.update(
        {
            "template": data_args.template,
            "tool_format": None,
            "default_system": None,
        }
    )

    if dataset_config["template_backend"] == "custom":
        template_instance = get_template_and_fix_tokenizer(dataset_config)
    else:
        template_instance = None
    dataset_config.update(
        {
            "template_instance": template_instance,
        }
    )

    if data_args.dataset_type == "pretrain":
        training_args.test_iters = training_args.eval_iters * 10
        train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
            training_args, data_args, model_args
        )
    else:
        if training_args.should_load_dataset:
            train_dataset = create_dataset_sft(
                task_group=data_args.train_dataset_path,
                task_group_prob=data_args.train_dataset_prob,
                sub_dataset_type=data_args.train_dataset_type,
                **dataset_config,
            )
        if training_args.do_eval and training_args.should_load_dataset:
            eval_dataset = create_dataset_sft(
                task_group=data_args.eval_dataset_path,
                task_group_prob=data_args.eval_dataset_prob,
                sub_dataset_type=data_args.eval_dataset_type,
                is_valid=True,
                **dataset_config,
            )

    # Freeze model based on training args (Supports for MLLM Full training)
    if not model_args.lora and getattr(training_args, "freeze_config", ""):
        freeze_model_parameters(model, training_args.freeze_config)

    model = create_peft_model(model_args, training_args, dtype, model)
    # Create trainer

    # padding to the maximum seq length in batch data when max_seq_len is None
    if getattr(model, "is_fleet", False) and not model_args.lora:
        if training_args.per_device_train_batch_size > 1:
            max_seq_len = data_args.max_seq_len + model_config.num_nextn_predict_layers
            logger.warning(f"Setting max_seq_len to {max_seq_len} for mbs > 1 using PaddleFleet model.")
        else:
            max_seq_len = None
            logger.warning("Setting max_seq_len to None for mbs = 1 using PaddleFleet Model.")
    else:
        max_seq_len = (
            data_args.max_seq_len + model_config.num_nextn_predict_layers
            if (data_args.packing or training_args.sequence_parallel or training_args.context_parallel_size > 1)
            else None
        )
        logger.info(f"Setting max_seq_len to {max_seq_len} using PaddleFormers Model.")
    if data_args.dataset_type != "pretrain":
        if model_args.stage == "VL-SFT":
            data_collator = partial(
                mm_collate_fn,
                template=template_instance,
                processor=processor,
                tokenizer=tokenizer,
                training_args=training_args,
                model_args=model_args,
                max_seq_len=max_seq_len,
                padding_free=data_args.padding_free,
                model=model,
            )
        else:
            data_collator = partial(
                collate_fn,
                tokenizer=tokenizer,
                training_args=training_args,
                model_args=model_args,
                max_seq_len=max_seq_len,
                padding_free=data_args.padding_free,
            )

    if training_args.max_steps == -1:
        if data_args.mix_strategy == "random":
            raise ValueError(
                "When using 'random' mix_strategy, max_steps must be explicitly set (cannot be -1). "
                "Random mixing requires a fixed number of training steps to properly sample data."
            )
        if training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            if data_args.dataset_type != "pretrain":
                training_args.max_steps = estimate_training(train_dataset, data_args, training_args, model_args)
                del train_dataset
                gc.collect()
                train_dataset = create_dataset_sft(
                    task_group=data_args.train_dataset_path,
                    task_group_prob=data_args.train_dataset_prob,
                    sub_dataset_type=data_args.train_dataset_type,
                    **dataset_config,
                )
            else:
                global_batch_size = (
                    training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                    * training_args.dataset_world_size
                )
                training_args.max_steps = math.ceil(len(train_dataset) / global_batch_size)

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            max_steps = paddle.to_tensor([training_args.max_steps])
            paddle.distributed.broadcast(max_steps, src=0)
            training_args.max_steps = int(max_steps.item())
        if training_args.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {training_args.max_steps}. Please check your dataset")

        logger.info(f"Re-setting training_args.max_steps to {training_args.max_steps}.")
    # Create the learning_rate sheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    if training_args.save_strategy == IntervalStrategy.EPOCH:
        training_args.save_strategy = IntervalStrategy.STEPS
        training_args.save_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.evaluation_strategy == IntervalStrategy.EPOCH:
        training_args.evaluation_strategy = IntervalStrategy.STEPS
        training_args.eval_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.logging_strategy == IntervalStrategy.EPOCH:
        training_args.logging_strategy = IntervalStrategy.STEPS
        training_args.logging_steps = int(training_args.max_steps / training_args.num_train_epochs)

    callbacks = []
    if getattr(model_config, "topk_method", None) == "noaux_tc":
        callbacks += [MoECorrectionBiasAdjustCallback(lr=0)]

    if training_args.use_expert_parallel:
        callbacks += [MoeExpertsGradScaleCallback(training_args)]

    if training_args.sequence_parallel and not model_args.lora:
        callbacks += [MoEGateSpGradSyncCallBack()]

    print("callbacks:", callbacks, flush=True)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=(train_dataset if training_args.do_train and training_args.should_load_dataset else None),
        eval_dataset=(eval_dataset if training_args.do_eval and training_args.should_load_dataset else None),
        tokenizer=tokenizer,
        processing_class=processor,
        data_collator=data_collator,
        do_generation=data_args.eval_with_do_generation,
        data_args=data_args,
        callbacks=callbacks,
    )
    trainable_parameters = [
        p for p in model.parameters() if not p.stop_gradient or ("quantization_linear" in p.name and "w_1" in p.name)
    ]
    trainer.set_optimizer_grouped_parameters(trainable_parameters)

    # Train
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if model_args.neftune:
            neft_post_hook_handle.remove()
        if training_args.benchmark:
            total_tokens = (
                data_args.max_seq_len
                * training_args.per_device_train_batch_size
                * training_args.dataset_world_size
                * training_args.gradient_accumulation_steps
                * training_args.max_steps
            )
            total_tokens_per_second_per_gpu = (
                total_tokens / train_result.metrics["train_runtime"] / training_args.world_size
            )
            logger.info(f"Total_Tokens_per_second_per_gpu: {total_tokens_per_second_per_gpu} ")
            logger.info("Benchmark done.")
        else:
            if not training_args.autotuner_benchmark:
                trainer.save_model(
                    merge_tensor_parallel=training_args.tensor_model_parallel_size > 1, last_fc_to_hf=True
                )
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
                trainer.save_state()


def create_peft_model(model_args, training_args, dtype, model):
    if model_args.lora:
        if training_args.sharding_parallel_size > 1:
            assert (
                not training_args.stage1_overlap
            ), "Currently not support enabling sharding_stage1_overlap in lora mode."
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)

            # Freeze model based on training args (Supports for MLLM LoRA training)
            if getattr(training_args, "freeze_config", ""):
                target_modules = get_multimodel_lora_target_modules(model, target_modules, training_args.freeze_config)

            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank if not model_args.rslora else 4,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                pissa=model_args.pissa,
                merge_weights=False,
                tensor_model_parallel_size=training_args.tensor_model_parallel_size,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
                use_quick_lora=model_args.use_quick_lora,
                lora_use_mixer=model_args.lora_use_mixer,
                use_mora=model_args.use_mora,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(
                model=model,
                lora_path=model_args.lora_path,
                load_checkpoint_format=training_args.load_checkpoint_format,
            )
        if hasattr(model, "_set_pipeline_name_mapping"):
            model._set_pipeline_name_mapping()
        model.print_trainable_parameters()

    return model
