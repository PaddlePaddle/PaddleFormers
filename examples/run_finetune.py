# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
from functools import partial

import paddle
from utils.data import get_convert_example

from paddleformers.data import DataCollatorForSeq2Seq
from paddleformers.datasets import (
    ZeroPaddingIterableDataset,
    ZeroPaddingMapDataset,
    load_dataset,
)
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.peft.reft import ReftDataCollator
from paddleformers.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddleformers.trainer.trainer_callback import TrainerState
from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
    DeepseekV2ForCausalLM,
    DeepseekV2ForCausalLMPipe,
    DeepseekV3ForCausalLM,
    DeepseekV3ForCausalLMPipe,
    Llama3Tokenizer,
    LlamaForCausalLM,
    LlamaForCausalLMPipe,
    LlamaTokenizer,
    Qwen2ForCausalLM,
    Qwen2ForCausalLMPipe,
    Qwen2MoeForCausalLM,
    Qwen2MoeForCausalLMPipe,
)
from paddleformers.transformers.configuration_utils import LlmMetaConfig
from paddleformers.trl import DataConfig, ModelConfig, SFTConfig, SFTTrainer
from paddleformers.trl.llm_utils import (
    ZeroPaddingIterDatasetCallback,
    compute_metrics,
    get_lora_target_modules,
    init_chat_template,
)
from paddleformers.utils.log import logger

# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"

flash_mask_support_list = [
    DeepseekV2ForCausalLM,
    DeepseekV2ForCausalLMPipe,
    DeepseekV3ForCausalLM,
    DeepseekV3ForCausalLMPipe,
    LlamaForCausalLM,
    LlamaForCausalLMPipe,
    Qwen2ForCausalLM,
    Qwen2ForCausalLMPipe,
    Qwen2MoeForCausalLM,
    Qwen2MoeForCausalLMPipe,
]


def main():
    parser = PdArgumentParser((ModelConfig, DataConfig, SFTConfig))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    elif len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file_and_cmd_lines()
    elif len(sys.argv) >= 2 and sys.argv[1].endswith(".py"):
        model_args, data_args, training_args = parser.parse_python_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
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
        download_hub=model_args.download_hub,
    )

    architectures_to_check = {"Qwen2Moe", "DeepseekV2", "DeepseekV3"}
    if (
        any(architecture in str(model_config.architectures) for architecture in architectures_to_check)
        and training_args.data_parallel_degree > 1
        and not training_args.use_expert_parallel
    ):
        raise ValueError("Please set use_expert_parallel to true in expert parallel mode.")

    # (Liuting) Not support acc calculation now due to MTP.
    if "DeepseekV3" in str(model_config.architectures):
        training_args.prediction_loss_only = True

    LlmMetaConfig.set_llm_config(model_config, training_args)
    model_config.use_fast_layer_norm = model_args.use_fast_layer_norm

    # Config for model using dropout, such as GPT.
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = model_args.hidden_dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
    if hasattr(model_config, "ignore_index"):
        model_config.ignore_index = -100

    if model_args.fuse_attention_qkv is not None:
        model_config.fuse_attention_qkv = model_args.fuse_attention_qkv
    if model_args.fuse_attention_ffn is not None:
        model_config.fuse_attention_ffn = model_args.fuse_attention_ffn

    model_config.seq_length = data_args.max_length
    logger.info(f"Final model config: {model_config}")

    logger.info("Creating model")

    model_class = AutoModelForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        if data_args.eval_with_do_generation and training_args.do_eval:
            raise ValueError("Please set eval_with_do_generation to false in pipeline parallel mode.")

        model_class = AutoModelForCausalLMPipe

    if model_args.continue_training and not training_args.autotuner_benchmark:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            download_hub=model_args.download_hub,
            convert_from_hf=False,  # run paddle weights
        )
    else:
        model = model_class.from_config(model_config, dtype=dtype)

    if model_args.flash_mask and (not data_args.zero_padding or not model.config.use_flash_attention):
        logger.warning("`flash_mask` must use with zero padding and flash attention.")
        data_args.zero_padding = True
        model.config.use_flash_attention = True

    if model_args.flash_mask and not any(isinstance(model, cls) for cls in flash_mask_support_list):
        raise NotImplementedError(f"{model.__class__} not support flash mask.")

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

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, download_hub=model_args.download_hub)
    tokenizer.chat_template = None

    # init chat_template for tokenizer
    init_chat_template(tokenizer, model_args.model_name_or_path, data_args.chat_template)

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds, dev_ds, test_ds = create_dataset(data_args, training_args)

    if training_args.resume_from_checkpoint is not None and data_args.lazy:
        logger.info(
            f"Loading from '{training_args.resume_from_checkpoint}' with `lazy=True`, manually skipping dataset and setting `ignore_data_skip` to True."
        )
        training_args.ignore_data_skip = True
        state = TrainerState.load_from_json(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"))
        if state.trial_params is not None and "zero_padding_global_step" in state.trial_params:
            consumed_samples = state.trial_params["zero_padding_global_step"]
        else:
            consumed_samples = (
                state.global_step
                * training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.dataset_world_size
            )
        logger.info(
            f"Skipping the first {consumed_samples} samples to warmup the dataset from checkpoint '{training_args.resume_from_checkpoint}'."
        )
        train_ds = train_ds.skip(consumed_samples)

    if training_args.pipeline_parallel_degree > 1:
        from utils.data import convert_example_common

        trans_func = partial(convert_example_common, tokenizer=tokenizer, data_args=data_args)
    else:
        trans_func = partial(get_convert_example(model), tokenizer=tokenizer, data_args=data_args)

    eval_zero_padding = data_args.zero_padding
    if data_args.zero_padding and data_args.eval_with_do_generation:
        logger.warning(
            "`zero_padding` conflicts with `eval_with_do_generation`. Setting zero_padding to False for the eval_dataset."
        )
        eval_zero_padding = False

    logger.info("Trans the dataset text into token ids, please wait for a moment.")
    train_ds, dev_ds, test_ds = trans_dataset_to_ids(
        train_ds, dev_ds, test_ds, model_args, data_args, trans_func, eval_zero_padding
    )

    if data_args.zero_padding:
        if data_args.lazy:
            intoken_dataset = ZeroPaddingIterableDataset
        else:
            intoken_dataset = ZeroPaddingMapDataset
        logger.info("Creating Zero Padding Data Stream. This may take a few minutes.")
        if train_ds is not None:
            train_ds = intoken_dataset(
                train_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
                greedy_zero_padding=data_args.greedy_zero_padding,
            )
        if eval_zero_padding and dev_ds is not None:
            dev_ds = intoken_dataset(dev_ds, tokenizer=tokenizer, max_length=data_args.max_length)
        if eval_zero_padding and test_ds is not None:
            test_ds = intoken_dataset(test_ds, tokenizer=tokenizer, max_length=data_args.max_length)

    model = create_peft_model(model_args, training_args, dtype, model)

    # Create trainer

    if (
        training_args.pipeline_parallel_degree > 1
        or training_args.sequence_parallel
        or training_args.autotuner_benchmark
        or data_args.zero_padding
        or data_args.pad_to_max_length
    ):
        max_length = data_args.max_length
        padding = "max_length"
    else:
        max_length = None
        padding = True

    if training_args.pipeline_parallel_degree > 1:
        metrics = None
    else:
        metrics = compute_metrics

    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        max_length=max_length,
        padding=padding,
        max_label_length=max_length,
        return_tensors="np",
        return_attention_mask=not model_args.flash_mask,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        data_collator=data_collator_fn if not model_args.reft else ReftDataCollator(data_collator=data_collator_fn),
        do_generation=data_args.eval_with_do_generation,
        callbacks=[ZeroPaddingIterDatasetCallback()] if isinstance(train_ds, ZeroPaddingIterableDataset) else None,
        data_args=data_args,
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
            total_effective_tokens = (
                sum([len(i["input_ids"]) for i in trainer.train_dataset]) * train_result.metrics["progress_or_epoch"]
            )
            effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
            logger.info(f"Effective_Tokens_per_second: {effective_tokens_per_second} ")
            logger.info("Benchmark done.")
        else:
            if not training_args.autotuner_benchmark:
                trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
                trainer.save_state()

    # Evaluation test set
    if training_args.do_predict:
        eval_result = trainer.predict(test_ds).metrics
        trainer.log_metrics("test", eval_result)
    # Evaluation dev set
    if training_args.do_eval:
        logger.info("*** Evaluate result after train ***")
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)


def create_peft_model(model_args, training_args, dtype, model):
    if model_args.lora:
        if training_args.sharding_parallel_degree > 1:
            assert (
                "enable_stage1_overlap" not in training_args.sharding_parallel_config
            ), "Currently not support enabling sharding_stage1_overlap in lora mode."
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank if not model_args.rslora else 4,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                pissa=model_args.pissa,
                merge_weights=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
                use_quick_lora=model_args.use_quick_lora,
                lora_use_mixer=model_args.lora_use_mixer,
                use_mora=model_args.use_mora,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)

        model.print_trainable_parameters()

    return model


def trans_dataset_to_ids(train_ds, dev_ds, test_ds, model_args, data_args, trans_func, eval_zero_padding):
    if train_ds is not None:
        train_ds = train_ds.map(
            partial(
                trans_func,
                is_test=False,
                zero_padding=data_args.zero_padding,
                flash_mask=model_args.flash_mask,
            )
        )
    if dev_ds is not None:
        dev_ds = dev_ds.map(
            partial(
                trans_func,
                is_test=data_args.eval_with_do_generation,
                zero_padding=eval_zero_padding,
                flash_mask=model_args.flash_mask,
            )
        )
    if test_ds is not None:
        test_ds = test_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation))

    return train_ds, dev_ds, test_ds


def create_dataset(data_args, training_args):
    if data_args.dataset_name_or_path is None:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")

    train_ds = None
    dev_ds = None
    test_ds = None
    if os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")) or os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev.json")
    ):
        logger.info("load train")
        if training_args.do_train:
            train_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                lazy=data_args.lazy,
            )[0]
        logger.info("load eval")
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "dev.json"),
                lazy=data_args.lazy,
            )[0]
        logger.info("load test")
        if training_args.do_predict:
            test_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "test.json"),
                lazy=data_args.lazy,
            )[0]

    elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train")) or os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev")
    ):
        import glob

        if training_args.do_train:
            train_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "train", "*.json")),
                lazy=data_args.lazy,
            )[0]
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "dev", "*.json")),
                lazy=data_args.lazy,
            )[0]
        if training_args.do_predict:
            test_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "test", "*.json")),
                lazy=data_args.lazy,
            )[0]
    else:
        if training_args.do_train:
            train_ds = load_dataset(data_args.dataset_name_or_path, splits=["train"])[0]

        if training_args.do_eval:
            dev_ds = load_dataset(data_args.dataset_name_or_path, splits=["dev"])[0]

        if training_args.do_predict:
            test_ds = load_dataset(data_args.dataset_name_or_path, splits=["test"])[0]

    return train_ds, dev_ds, test_ds


if __name__ == "__main__":
    main()
