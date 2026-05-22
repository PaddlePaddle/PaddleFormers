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

"""SFT workflow using datasets_v2 pipeline.

Simplified version of workflow.py that uses the new datasets_v2 module
for data loading, encoding, and collation. Supports packing + flashmask.
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
from paddleformers.datasets_v2 import (
    EncodeConfig,
    LazyEncodeDataset,
    StreamingDataset,
    encode_pt,
    encode_sft,
    get_template,
)
from paddleformers.datasets_v2 import load_dataset as v2_load_dataset
from paddleformers.datasets_v2.datapipe.collate import collate_sft
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

from .sft_trainer import SFTTrainer

os.environ["USE_CASUAL_MASK"] = "False"


def _detect_template(tokenizer, data_args) -> str:
    """Detect which template to use based on config and tokenizer."""
    if hasattr(data_args, "use_template") and not data_args.use_template:
        return "empty"

    if data_args.template:
        return data_args.template

    # Auto-detect from model name
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

    # Default: use jinja if tokenizer has chat_template, else chatml
    if tokenizer.chat_template is not None:
        return "__jinja__"
    return "chatml"


def run_sft_v2(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    generating_args: "GeneratingArguments",
    finetuning_args: "FinetuningArguments",
):
    """Run SFT training using datasets_v2 pipeline."""

    training_args = finetuning_args
    training_args.max_seq_len = data_args.max_seq_len
    training_args.model_name_or_path = model_args.model_name_or_path
    training_args.download_hub = model_args.download_hub
    training_args.copy_custom_file_list = model_args.copy_custom_file_list

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

    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = finetuning_args.hidden_dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = finetuning_args.attention_probs_dropout_prob
    if hasattr(model_config, "ignore_index"):
        model_config.ignore_index = -100

    avaible_attn_impl = AttentionInterface._global_mapping.keys()
    if model_args._attn_implementation not in avaible_attn_impl:
        raise ValueError(
            f"Invalid _attn_implementation: {model_args._attn_implementation}, " f"available: {avaible_attn_impl}"
        )

    model_config.seq_length = data_args.max_seq_len
    model_config.max_sequence_length = data_args.max_seq_len
    model_config._attn_implementation = model_args._attn_implementation
    model_config.is_lora = model_args.lora

    # MTP: extend model seq_length to accommodate extra prediction tokens
    num_nextn_predict_layers = getattr(training_args, "num_nextn_predict_layers", 0)
    if num_nextn_predict_layers > 0:
        model_config.seq_length = data_args.max_seq_len + num_nextn_predict_layers
        model_config.max_sequence_length = data_args.max_seq_len + num_nextn_predict_layers

    # Autoregressive MTP training: swap mtp_num_layers and num_nextn_predict_layers
    if getattr(model_config, "mtp_num_layers", 0) > 1:
        tmp = model_config.mtp_num_layers
        model_config.mtp_num_layers = model_config.num_nextn_predict_layers
        model_config.num_nextn_predict_layers = tmp
        tmp = training_args.mtp_num_layers
        training_args.mtp_num_layers = training_args.num_nextn_predict_layers
        training_args.num_nextn_predict_layers = tmp
        num_nextn_predict_layers = training_args.num_nextn_predict_layers
        logger.info(
            f"MTP args swapped for autoregressive mtp training: "
            f"mtp_num_layers={model_config.mtp_num_layers}, "
            f"num_nextn_predict_layers={model_config.num_nextn_predict_layers}"
        )

    logger.info(f"Final model config: {model_config}")
    logger.info("Loading model...")

    model_class = AutoModelForCausalLM
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

    # ====== Tokenizer ======
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    add_new_special_tokens(tokenizer, data_args.new_special_tokens_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ====== Dataset (datasets_v2) ======
    logger.info("[datasets_v2] Loading dataset...")

    # Stash model path for template detection
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

    # Encode config
    encode_config = EncodeConfig(
        max_seq_len=data_args.max_seq_len,
        truncation="right",
        label_shift=True,
    )

    # Build encode function
    is_pretraining = "pt" in model_args.stage.lower()

    if is_pretraining:
        encode_fn = partial(
            encode_pt,
            tokenizer=tokenizer,
            config=encode_config,
        )
        logger.info("[datasets_v2] Pretrain mode: using encode_pt")
    else:
        encode_fn = partial(
            encode_sft,
            tokenizer=tokenizer,
            template=template,
            config=encode_config,
        )

    # Load train dataset (and optionally split for eval)
    train_dataset = None
    eval_dataset = None
    num_proc = getattr(data_args, "dataset_num_proc", 1)
    # dataset_type 控制数据集加载模式:
    #   iterable → streaming=True, 不下载到本地, 逐条从网络拉取(IterableDataset)
    #   map      → streaming=False, 下载到HF cache后全量加载(MapDataset, 支持随机访问)
    # V2 管道默认使用 map 模式（支持 packing），除非用户显式指定 iterable
    dataset_type = getattr(data_args, "dataset_type", "iterable")
    use_packing = getattr(data_args, "packing", False) or getattr(data_args, "binpacking", False)

    # 如果用户未显式设置 dataset_type（仍为默认值 iterable）且开启了 packing，V2 自动用 map
    if dataset_type == "iterable" and use_packing:
        dataset_type = "map"
        logger.info("V2 pipeline: auto-switching dataset_type to 'map' (packing requires map mode).")

    is_iterable = dataset_type == "iterable"

    # 获取数据格式 hint（兼容旧配置的 train_dataset_type）
    train_format = getattr(data_args, "train_dataset_type", None)
    eval_format = getattr(data_args, "eval_dataset_type", None)

    if training_args.do_train and training_args.should_load_dataset:
        if is_iterable:
            if training_args.max_steps <= 0:
                raise ValueError("dataset_type=iterable requires --max_steps to be explicitly set (no len available).")

            # dataset_type=iterable → HF streaming=True，无论在线离线都返回 IterableDataset
            hf_ds = v2_load_dataset(
                data_args.train_dataset_path, streaming=True, num_proc=1, dataset_format=train_format
            )

            # Use HF IterableDataset.map() for lazy encoding
            def _streaming_encode(row):
                result = encode_fn(row)
                if result is None:
                    return {"input_ids": [], "labels": [], "seq_len": 0}
                return {"input_ids": result.input_ids, "labels": result.labels, "seq_len": result.seq_len}

            hf_ds = hf_ds.map(_streaming_encode, remove_columns=hf_ds.column_names or [])
            # Filter out empty samples
            hf_ds = hf_ds.filter(lambda x: x["seq_len"] > 0)
            # Shuffle with buffer
            hf_ds = hf_ds.shuffle(buffer_size=getattr(data_args, "buffer_size", 500))

            train_dataset = StreamingDataset(hf_ds)
            # Force single-thread DataLoader for iterable mode
            training_args.dataloader_num_workers = 0
            logger.info("[datasets_v2] Train dataset ready (iterable mode, streaming=True)")
        else:
            # dataset_type=map → HF streaming=False，全量加载（在线则先下载到缓存）
            hf_ds = v2_load_dataset(
                data_args.train_dataset_path, streaming=False, num_proc=num_proc, dataset_format=train_format
            )

            # Auto-split: if eval path is same as train or not set, split from train
            need_auto_split = training_args.do_eval and (
                not data_args.eval_dataset_path or data_args.eval_dataset_path == data_args.train_dataset_path
            )

            if need_auto_split:
                from paddleformers.datasets_v2 import split_dataset

                hf_ds, hf_eval_ds = split_dataset(hf_ds, test_ratio=0.1, shuffle=True, seed=training_args.seed)
                eval_dataset = LazyEncodeDataset(hf_eval_ds, encode_fn, seed=training_args.seed)
                logger.info(f"[datasets_v2] Auto-split: train={len(hf_ds)}, eval={len(hf_eval_ds)}")

            train_dataset = LazyEncodeDataset(hf_ds, encode_fn, seed=training_args.seed)
            logger.info(f"[datasets_v2] Train dataset loaded: {len(train_dataset)} samples")

    # Load eval dataset (only if not auto-split above)
    if training_args.do_eval and training_args.should_load_dataset and eval_dataset is None:
        if is_iterable:
            # eval 也走流式，不下载到本地
            hf_eval_ds = v2_load_dataset(
                data_args.eval_dataset_path, streaming=True, num_proc=1, dataset_format=eval_format
            )

            def _streaming_eval_encode(row):
                result = encode_fn(row)
                if result is None:
                    return {"input_ids": [], "labels": [], "seq_len": 0}
                return {"input_ids": result.input_ids, "labels": result.labels, "seq_len": result.seq_len}

            hf_eval_ds = hf_eval_ds.map(_streaming_eval_encode, remove_columns=hf_eval_ds.column_names or [])
            hf_eval_ds = hf_eval_ds.filter(lambda x: x["seq_len"] > 0)
            eval_dataset = StreamingDataset(hf_eval_ds)
            logger.info("[datasets_v2] Eval dataset ready (iterable mode, streaming=True)")
        else:
            hf_eval_ds = v2_load_dataset(data_args.eval_dataset_path, num_proc=num_proc, dataset_format=eval_format)
            eval_dataset = LazyEncodeDataset(hf_eval_ds, encode_fn, seed=training_args.seed)
            logger.info(f"[datasets_v2] Eval dataset loaded: {len(eval_dataset)} samples")

    # ====== Collate Function ======
    max_seq_len = (
        data_args.max_seq_len
        if (data_args.packing or training_args.sequence_parallel or training_args.context_parallel_size > 1)
        else None
    )
    logger.info(f"[datasets_v2] max_seq_len for collate: {max_seq_len}")

    # Determine whether to use flashmask compact format
    use_startend = getattr(model_args, "use_attn_mask_startend_row_indices", True)
    use_global_causal_attn = getattr(model_args, "use_global_causal_attn", False)

    data_collator = partial(
        collate_sft,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=data_args.max_seq_len,
        packing=data_args.packing,
        packing_method="binpacking" if getattr(data_args, "binpacking", False) else "greedy",
        packing_interval=getattr(data_args, "packing_interval", 1000),
        use_attn_mask_startend_row_indices=use_startend,
        num_nextn_predict_layers=num_nextn_predict_layers,
        eos_token_id=tokenizer.eos_token_id if num_nextn_predict_layers > 0 else None,
        use_global_causal_attn=use_global_causal_attn,
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
        if is_iterable:
            raise ValueError("Streaming (iterable) mode requires --max_steps to be explicitly set.")
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

    # ====== Create Trainer ======
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=(train_dataset if training_args.do_train and training_args.should_load_dataset else None),
        eval_dataset=(eval_dataset if training_args.do_eval and training_args.should_load_dataset else None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        do_generation=False,
        data_args=data_args,
    )

    # MTP-only training: freeze everything except MTP layers
    if getattr(training_args, "train_mtp_only", False):
        from paddleformers.cli.train.sft.workflow import freeze_param_except_mtp

        freeze_param_except_mtp(model, model_config)

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
            logger.info(f"Total_Tokens_per_second_per_gpu: {total_tokens_per_second_per_gpu}")
            logger.info("Benchmark done.")
        else:
            if not training_args.autotuner_benchmark:
                trainer.save_model(
                    merge_tensor_parallel=training_args.tensor_model_parallel_size > 1, last_fc_to_hf=True
                )
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
                trainer.save_state()
