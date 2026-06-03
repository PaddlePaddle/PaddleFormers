# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""DPO workflow using datasets_v2 pipeline.

Uses the new datasets_v2 module for data loading, encoding, and collation,
while reusing the existing DPOTrainer and DPO loss functions.
"""

import logging

logging.getLogger("paddleformers.transformers.modeling_rope_utils").setLevel(logging.ERROR)

import math
import os
from functools import partial

import paddle

from paddleformers.cli.hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
)
from paddleformers.cli.utils.process import add_new_special_tokens
from paddleformers.datasets_v2 import LazyEncodeDataset, get_template
from paddleformers.datasets_v2 import load_dataset as v2_load_dataset
from paddleformers.datasets_v2.datapipe.collate import collate_dpo
from paddleformers.datasets_v2.datapipe.encode import DPOEncodeConfig, encode_dpo
from paddleformers.nn.attention import AttentionInterface
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.trainer import (
    IntervalStrategy,
    get_last_checkpoint,
    set_random_seed,
    set_seed,
)
from paddleformers.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddleformers.transformers.configuration_utils import (
    LlmMetaConfig,
    QuantizationConfig,
)
from paddleformers.utils.log import logger

from .dpo_argument import DPOConfig
from .dpo_trainer import DPOTrainer

os.environ["USE_CASUAL_MASK"] = "False"


def _detect_template(tokenizer, data_args) -> str:
    """Detect which template to use based on config and tokenizer."""
    if hasattr(data_args, "use_template") and not data_args.use_template:
        return "empty"

    if data_args.template:
        return data_args.template

    model_path = getattr(data_args, "_model_name_or_path", "")
    model_lower = model_path.lower()
    if "qwen" in model_lower:
        return "chatml"
    elif "llama" in model_lower:
        return "llama3"
    elif "deepseek" in model_lower:
        return "deepseek3"
    elif "glm" in model_lower:
        return "chatml"

    if tokenizer.chat_template is not None:
        return "__jinja__"
    return "chatml"


def run_dpo_v2(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    generating_args: "GeneratingArguments",
    finetuning_args: "FinetuningArguments",
):
    """Run DPO training using datasets_v2 pipeline."""

    training_args = finetuning_args
    training_args.max_seq_len = data_args.max_seq_len
    training_args.model_name_or_path = model_args.model_name_or_path
    training_args.download_hub = model_args.download_hub
    training_args.copy_custom_file_list = model_args.copy_custom_file_list

    # DPO-specific loss type adjustments
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

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(training_args, "Train")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    set_random_seed(seed_=training_args.seed)
    set_seed(seed=training_args.seed)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"world_size: {training_args.world_size}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # ====== DPO Config ======
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

    # ====== Model Setup ======
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    if finetuning_args.weight_quantize_algo is not None:
        quantization_config = dict(
            weight_quantize_algo=finetuning_args.weight_quantize_algo,
            ignore_modules=[".*out_linear.*"],
        )
    else:
        quantization_config = dict(weight_quantize_algo=finetuning_args.weight_quantize_algo)
    quantization_config = QuantizationConfig.from_dict(quantization_config)

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        quantization_config=quantization_config,
    )

    LlmMetaConfig.set_llm_config(model_config, training_args)

    avaible_attn_impl = AttentionInterface._global_mapping.keys()
    if model_args._attn_implementation not in avaible_attn_impl:
        raise ValueError(
            f"Invalid _attn_implementation: {model_args._attn_implementation}, available: {avaible_attn_impl}"
        )

    model_config.seq_length = data_args.max_seq_len
    model_config.max_sequence_length = data_args.max_seq_len
    model_config._attn_implementation = model_args._attn_implementation
    model_config.is_lora = model_args.lora
    model_config.dpo_config = dpo_config

    logger.info(f"Final model config: {model_config}")
    logger.info("Loading model...")

    model_class = AutoModelForCausalLM

    # Reference model setup
    ref_model = None
    if not training_args.reference_free and not model_args.lora:
        ref_model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, dtype=dtype)
        ref_model_config.max_sequence_length = data_args.max_seq_len
        ref_model_config.seq_length = data_args.max_seq_len
        ref_model_config._attn_implementation = model_args._attn_implementation
        ref_model_config.dpo_config = dpo_config
        LlmMetaConfig.set_llm_config(ref_model_config, training_args)

    if model_args.continue_training and not training_args.autotuner_benchmark:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            convert_from_hf=training_args.convert_from_hf,
            load_via_cpu=training_args.load_via_cpu,
            load_checkpoint_format=training_args.load_checkpoint_format,
        )
        if not training_args.reference_free and not model_args.lora:
            ref_model = model_class.from_config(ref_model_config)
            ref_model.set_state_dict(model.state_dict())
    else:
        model = model_class.from_config(model_config, dtype=dtype)
        if not training_args.reference_free and not model_args.lora:
            ref_model = model_class.from_config(ref_model_config)
            ref_model.set_state_dict(model.state_dict())

    # ====== Tokenizer ======
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    add_new_special_tokens(tokenizer, data_args.new_special_tokens_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ====== Dataset (datasets_v2) ======
    logger.info("[datasets_v2] Loading DPO dataset...")

    data_args._model_name_or_path = model_args.model_name_or_path

    # Determine template
    template_name = _detect_template(tokenizer, data_args)
    use_jinja = template_name == "__jinja__"

    if use_jinja:
        template = None
        logger.info("[datasets_v2] Using tokenizer's built-in chat_template (jinja)")
    else:
        template = get_template(template_name)
        logger.info(f"[datasets_v2] Using template: {template_name}")

    # DPO Encode config
    use_filtered_label_loss = getattr(model_config, "use_filtered_label_loss", True)
    encode_config = DPOEncodeConfig(
        max_seq_len=data_args.max_seq_len,
        truncation="right",
        label_shift=False,  # DPO does NOT shift labels (response_labels are direct token IDs)
        use_filtered_label_loss=use_filtered_label_loss,
    )

    encode_fn = partial(
        encode_dpo,
        tokenizer=tokenizer,
        template=template,
        config=encode_config,
    )

    # Load train dataset
    train_dataset = None
    eval_dataset = None
    num_proc = getattr(data_args, "dataset_num_proc", 1)
    train_format = getattr(data_args, "train_dataset_type", None)
    eval_format = getattr(data_args, "eval_dataset_type", None)

    # DPO always uses map mode (no streaming packing needed, needs random access)
    if training_args.do_train and training_args.should_load_dataset:
        hf_ds = v2_load_dataset(
            data_args.train_dataset_path, streaming=False, num_proc=num_proc, dataset_format=train_format
        )
        train_dataset = LazyEncodeDataset(hf_ds, encode_fn, seed=training_args.seed)
        logger.info(f"[datasets_v2] DPO train dataset loaded: {len(train_dataset)} samples")

    if training_args.do_eval and training_args.should_load_dataset:
        hf_eval_ds = v2_load_dataset(
            data_args.eval_dataset_path, streaming=False, num_proc=num_proc, dataset_format=eval_format
        )
        eval_dataset = LazyEncodeDataset(hf_eval_ds, encode_fn, seed=training_args.seed)
        logger.info(f"[datasets_v2] DPO eval dataset loaded: {len(eval_dataset)} samples")

    # ====== Collate Function ======
    max_seq_len = (
        data_args.max_seq_len if training_args.sequence_parallel or training_args.context_parallel_size > 1 else None
    )
    logger.info(f"[datasets_v2] max_seq_len for collate: {max_seq_len}")

    use_startend = getattr(model_args, "use_attn_mask_startend_row_indices", True)

    data_collator = partial(
        collate_dpo,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=data_args.max_seq_len,
        use_attn_mask_startend_row_indices=use_startend,
        use_filtered_label_loss=use_filtered_label_loss,
    )

    # ====== LoRA (optional) ======
    if model_args.lora:
        from paddleformers.cli.utils import get_lora_target_modules

        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank if not model_args.rslora else 4,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                merge_weights=False,
                tensor_model_parallel_size=training_args.tensor_model_parallel_size,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)

    # ====== max_steps calculation ======
    if training_args.max_steps == -1:
        if training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            training_args.max_steps = math.ceil(len(train_dataset) / training_args.global_batch_size)
            training_args.max_steps *= training_args.num_train_epochs
            logger.info(
                f"len(train_dataset): {len(train_dataset)}, "
                f"global_batch_size: {training_args.global_batch_size}, "
                f"num_train_epochs: {training_args.num_train_epochs}, "
                f"max_steps: {training_args.max_steps}"
            )

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            max_steps = paddle.to_tensor([training_args.max_steps])
            paddle.distributed.broadcast(max_steps, src=0)
            training_args.max_steps = int(max_steps.item())

        if training_args.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {training_args.max_steps}. Please check your dataset")
        logger.info(f"Re-setting training_args.max_steps to {training_args.max_steps}.")

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

    # ====== Create DPO Trainer ======
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        dpo_config=dpo_config,
        args=training_args,
        train_dataset=(train_dataset if training_args.do_train and training_args.should_load_dataset else None),
        eval_dataset=(eval_dataset if training_args.do_eval and training_args.should_load_dataset else None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_with_dpo_criterion=getattr(model_args, "model_with_dpo_criterion", False),
    )

    trainable_parameters = [
        p for p in model.parameters() if not p.stop_gradient or ("quantization_linear" in p.name and "w_1" in p.name)
    ]
    trainer.set_optimizer_grouped_parameters(trainable_parameters)

    # ====== Train ======
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if not training_args.autotuner_benchmark and not getattr(training_args, "benchmark", False):
            trainer.save_model(merge_tensor_parallel=training_args.tensor_model_parallel_size > 1, last_fc_to_hf=True)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    # ====== Eval ======
    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
