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

import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import paddle

from paddleformers.data import DataCollatorForSeq2Seq
from paddleformers.datasets import load_dataset
from paddleformers.peft import (
    LoKrConfig,
    LoKrModel,
    LoRAAutoConfig,
    LoRAAutoModel,
    PrefixConfig,
    VeRAConfig,
    VeRAModel,
)
from paddleformers.peft.reft import (
    ReFTConfig,
    ReftDataCollator,
    ReFTModel,
    intervention_mapping,
)
from paddleformers.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddleformers.trainer.trainer_callback import TrainerState
from paddleformers.trainer.utils.doc import add_start_docstrings
from paddleformers.transformers import (
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaConfig,
    LlamaForCausalLM3DAuto,
    LlamaForCausalLMNet,
    LlamaPretrainingCriterion3DAuto,
    LlamaPretrainingCriterionNet,
    LlamaTokenizer,
)

MODEL_CLASSES = {
    "llama": (LlamaConfig, LlamaForCausalLM3DAuto, LlamaPretrainingCriterion3DAuto),
    "llama_network": (LlamaConfig, LlamaForCausalLMNet, LlamaPretrainingCriterionNet),
}

import random

import numpy as np

from paddleformers.peft import LoRAModel, PrefixModelForCausalLM
from paddleformers.trl import DataConfig, ModelConfig, SFTAutoTrainer, SFTConfig
from paddleformers.trl.llm_utils import (
    compute_metrics,
    get_lora_target_modules,
    init_chat_template,
)
from paddleformers.utils.log import logger
from paddleformers.utils.tools import get_env_device


def convert_multi_rounds_to_single_round(example, tokenizer):
    # 1. convert multi-rounds to single-round data format with chat_template
    example["src"] = example["src"] if isinstance(example["src"], list) else [example["src"]]
    example["tgt"] = example["tgt"] if isinstance(example["tgt"], list) else [example["tgt"]]

    src = tokenizer.chat_template.render_system()
    conversations = list(zip(example["src"], example["tgt"]))

    for index, conversation in enumerate(conversations[:-1]):
        src += "".join(tokenizer.chat_template.render_conversation(conversation, index=index))

    last_user, last_bot = tokenizer.chat_template.render_conversation(conversations[-1], index=len(conversations) - 1)

    example["src"] = [src + last_user]
    example["tgt"] = [last_bot]
    return example


def get_convert_example(model):
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        base_model_prefix = model.model.base_model_prefix
    else:
        base_model_prefix = model.base_model_prefix

    if base_model_prefix == "chatglm":
        return convert_example_chatglm
    elif base_model_prefix in [
        "chatglm_v2",
        "llama",
        "bloom",
        "opt",
        "qwen",
        "mixtral",
        "mistral",
        "gemma",
        "qwen2",
        "qwen2_moe",
        "gpt",
        "yuan",
        "jamba",
        "deepseek_v2",
        "deepseek_v3",
    ]:
        return convert_example_common
    else:
        raise ValueError(
            f"Unknown base_model_prefix: {model.base_model_prefix}. Supported base_model_prefix list: chatglm, bloom, llama, qwen, mixtral, gemma, qwen2, qwen2_moe, yuan, jamba,deepseek_v2, deepseek_v3",
        )


class DataFormatError(ValueError):
    pass


def tokenize_unsupervised_example(tokenizer, example, data_args, is_test=True, zero_padding=False, flash_mask=False):
    if "src" in example:
        source = example["src"][0] if isinstance(example["src"], list) else example["src"]
    else:
        raise DataFormatError(
            f"Example format is wrong, please check: {example} or rewrite tokenize_example in data.py "
        )
    tokenized_source = tokenizer(
        source,
        truncation=False,
        padding=True,
        max_length=data_args.src_length,
        add_special_tokens=True,
    )
    if data_args.use_pose_convert:
        tokenized_source = get_example_pose(tokenized_source, tokenizer, data_args)

    return tokenized_source


def tokenize_example(tokenizer, example, data_args):
    if "src" in example and "tgt" in example:
        source = example["src"][0] if isinstance(example["src"], list) else example["src"]
        target = example["tgt"][0] if isinstance(example["tgt"], list) else example["tgt"]
    else:
        raise DataFormatError(
            f"Example format is wrong, please check: {example} or rewrite tokenize_example in data.py "
        )
    tokenized_source = tokenizer(
        source,
        max_length=data_args.src_length,
        truncation=True,
        truncation_side="left",
        add_special_tokens=True,
    )

    tgt_max_length = data_args.max_length - len(tokenized_source["input_ids"])
    tokenized_target = tokenizer(
        target,
        max_length=tgt_max_length,
        truncation=True,
        truncation_side="right",
        add_special_tokens=False,
    )

    tokenized_target_input_ids = tokenized_target["input_ids"]
    # Add eos_token_id at the end of sequence if the sentence is not truncated.
    # Attention! In some cases(ex. ChatGLMv2), tokenized eos_token is not equal to eos_token_id.
    if len(tokenized_target_input_ids) < tgt_max_length:
        tokenized_target_input_ids += [tokenizer.eos_token_id]

    return tokenized_source, tokenized_target_input_ids


def tokenize_rounds_example(tokenizer, example, data_args, **kwargs):
    """tokenize multi-rounds examples with chat_template.json

    Args:
        tokenizer (PretrainedTokenizer): the instance of tokenizer
        example (dict[str, str | list[str]]):
                the example instance, which can be: {"src": "src-sentence", "tgt": "tgt-sentence"}
                or {"src": ["src-sentence-1", ..., "src-sentence-N"], "tgt": ["tgt-sentence-1", ..., "tgt-sentence-N"]}
        data_args (DataArgument): the data_argument instance of data processing

    Returns:
        dict[str, list[int]]: return input_ids and labels fields
    """

    # 0. prepare data
    context_data = example.get("context", {})
    context_data["is_training"] = True

    example["src"] = example["src"] if isinstance(example["src"], list) else [example["src"]]
    example["tgt"] = example["tgt"] if isinstance(example["tgt"], list) else [example["tgt"]]

    assert len(example["src"]) == len(example["tgt"]), "the length of `src` and `tgt` field must be same."

    conversations = [[src, tgt] for src, tgt in zip(example["src"], example["tgt"])]

    # 1. only tokenize input_ids
    conversation_result: list[tuple[list[int], list[int]]] = tokenizer.encode_chat_inputs(
        conversations, context_data=context_data, **kwargs
    )
    system_ids = conversation_result.pop("system", []) or []

    # 2. truncate conversations based on conversation unit
    input_ids, labels = [], []
    conversations_ids = conversation_result.pop("conversations")

    assert (
        len(system_ids) < data_args.max_length
    ), f"the length of system_ids<{len(system_ids)}> should be smaller than max_length<{data_args.max_length}>."
    max_length = data_args.max_length - len(system_ids)

    should_break = False
    for index in range(len(conversations_ids) - 1, -1, -1):
        user_input_ids, bot_input_ids = conversations_ids[index][0], conversations_ids[index][1]

        # break when the length of current conversations is greater than max_length
        if len(input_ids) + len(user_input_ids) + len(bot_input_ids) > max_length:

            # when the length of last conversation is lager than max_length, we should not break: at least one round
            if index < len(conversations_ids) - 1:
                break

            user_input_ids = user_input_ids[: data_args.src_length - len(system_ids)]
            bot_input_ids = bot_input_ids[: max_length - len(user_input_ids)]

            should_break = True

        input_ids = user_input_ids + bot_input_ids + input_ids
        labels = len(user_input_ids) * [-100] + bot_input_ids + labels

        if should_break:
            break

    input_ids = system_ids + input_ids
    labels = [-100] * len(system_ids) + labels
    tokenized_source = {"input_ids": input_ids}
    sequence_length = len(input_ids)

    if "position_ids" in tokenizer.model_input_names:
        tokenized_source["position_ids"] = list(range(sequence_length))

    return tokenized_source, labels


def convert_example_common(example, tokenizer, data_args, is_test=True, zero_padding=False, flash_mask=False):
    if data_args.autoregressive:
        tokenized_source = tokenize_unsupervised_example(
            tokenizer, example, data_args, is_test=True, zero_padding=False, flash_mask=False
        )
        input_ids = tokenized_source["input_ids"]
        if "labels" in tokenized_source:
            labels = tokenized_source["labels"]
        else:
            labels = input_ids
            input_ids = input_ids[:-1] + [tokenizer.eos_token_id]
            labels = labels[1:] + [-100]
        features = {"input_ids": input_ids, "labels": labels}
        if "position_ids" in tokenized_source:
            features["position_ids"] = tokenized_source["position_ids"]
    else:
        if tokenizer.chat_template is not None:
            return convert_rounds_example_common(example, tokenizer, data_args, is_test, zero_padding, flash_mask)
        else:
            tokenized_source, tokenized_target_input_ids = tokenize_example(tokenizer, example, data_args)

            if is_test:
                return {
                    **tokenized_source,
                    "labels": tokenized_target_input_ids,
                }
            else:
                input_ids = tokenized_source["input_ids"] + tokenized_target_input_ids
                source_length = len(tokenized_source["input_ids"])
                labels = [-100] * source_length + input_ids[source_length:]
                # shift input_ids and labels
                input_ids, labels = input_ids[:-1], labels[1:]
                seq_length = len(input_ids)
                features = {"input_ids": input_ids, "labels": labels}
                if "position_ids" in tokenized_source:
                    features["position_ids"] = list(range(seq_length))
    # maybe change here to suit flash_mask with longlora
    if zero_padding:
        if flash_mask:
            features["attn_mask_startend_row_indices"] = [seq_length] * seq_length
        else:
            features["attention_mask"] = np.tri(seq_length, seq_length, dtype=bool)
    return features


def convert_rounds_example_common(example, tokenizer, data_args, is_test=True, zero_padding=False, flash_mask=False):
    """convert multi-rounds conversation example

    Args:
        example (dict): the source of example
        tokenizer (PretrainedTokenizer): the instance of tokenizer
        data_args (DataArgument): data argument for data preprocessing
        is_test (bool, optional): whether is testing stage. Defaults to True.
        zero_padding (bool, optional): whether use in_tokens. Defaults to False.

    Returns:
        dict[str, np.ndarray]: the features of example
    """
    rounds_inputs, labels = tokenize_rounds_example(tokenizer, example, data_args)

    if is_test:
        return {
            **rounds_inputs,
            "labels": labels,
        }

    input_ids = rounds_inputs.pop("input_ids")
    # shift input_ids and labels
    input_ids, labels = input_ids[:-1], labels[1:]

    seq_length = len(input_ids)
    features = {"input_ids": input_ids, "labels": labels}
    if zero_padding:
        if flash_mask:
            features["attn_mask_startend_row_indices"] = [seq_length] * seq_length
        else:
            features["attention_mask"] = np.tri(seq_length, seq_length, dtype=bool)

    if "position_ids" in rounds_inputs:
        rounds_inputs["position_ids"] = rounds_inputs["position_ids"][:-1]

    rounds_inputs.update(features)
    return rounds_inputs


def convert_example_chatglm(example, tokenizer, data_args, is_test=True, zero_padding=False, flash_mask=False):
    if flash_mask:
        raise ValueError("chatglm does not support flash mask for now!")
    if tokenizer.chat_template is not None:
        # chatglm only support single-round finetune
        example = convert_multi_rounds_to_single_round(example, tokenizer)

    tokenized_source, tokenized_target_input_ids = tokenize_example(tokenizer, example, data_args)

    if is_test:
        return {
            **tokenized_source,
            "labels": tokenized_target_input_ids,
        }
    else:
        input_ids = tokenized_source["input_ids"] + tokenized_target_input_ids
        bos_position = len(tokenized_source["input_ids"]) - 1
        labels = [-100] * bos_position + input_ids[bos_position:]
        # shift input_ids and labels
        input_ids, labels = input_ids[:-1], labels[1:]
        features = {
            "input_ids": input_ids,
            "labels": labels,
        }

        if zero_padding:
            seq_length = len(input_ids)
            # attention_mask
            attention_mask = np.tri(seq_length, seq_length, dtype=bool)
            attention_mask[:, :bos_position] = 1
            features["attention_mask"] = attention_mask
            # 2d position_ids
            position_ids = np.arange(seq_length, dtype=np.int64)
            position_ids[:bos_position] = bos_position - 1
            block_position_ids = np.concatenate(
                [
                    np.zeros(bos_position, dtype=np.int64),
                    np.arange(1, seq_length - bos_position + 1, dtype=np.int64),
                ]
            )
            features["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)

        return features


def get_example_pose(tokenized_source, tokenizer, data_args):

    ids = tokenized_source["input_ids"]
    len_chunk = min(len(ids), data_args.max_length)
    if len(tokenized_source["input_ids"]) <= data_args.max_length:
        tokenized_source["input_ids"] += [tokenizer.eos_token_id]

    len_input = len(ids)

    lt1 = 0  # chunk1 start pos
    rt1 = random.randint(1, (len_chunk) // 2)  # chunk1 end pos

    rt2 = random.randint(lt1 + len_chunk, len_input - 1)  # chunk2 end pos
    lt2 = rt2 - (len_chunk - (rt1 - lt1))  # chunk2 start pos
    chunked_ids = ids[lt1:rt1] + ids[lt2:rt2]
    labels = ids[lt1 + 1 : rt1 + 1] + ids[lt2 + 1 : rt2 + 1]

    pos_ids = range(len(chunked_ids))
    pos_ids = [x + lt1 if i < rt1 - lt1 else x + (lt2 - (rt1 - lt1)) for i, x in enumerate(pos_ids)]

    features = {"input_ids": chunked_ids, "labels": labels, "position_ids": pos_ids}

    return features


# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"

flash_mask_support_list = [LlamaForCausalLM3DAuto, LlamaForCausalLMNet]


@dataclass
@add_start_docstrings(SFTConfig.__doc__)
class SFTAutoConfig(SFTConfig):
    enable_linear_fused_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise add for grad accumulation in the backward of nn.Linear ."
        },
    )
    job_schedule_profiler_start: int = field(
        default=-1,
        metadata={"help": "The step to start job_schedule_profiler."},
    )
    job_schedule_profiler_end: int = field(
        default=-1,
        metadata={"help": "The step to end job_schedule_profiler."},
    )
    pipeline_schedule_mode: str = field(
        default="1F1B", metadata={"help": "The pipeline schedule mode, support FThenB, 1F1B, VPP and Eager-1F1B."}
    )
    sr: Optional[int] = field(default=0, metadata={"help": "The count of chunks without recompute."})
    refined_ops_patterns: Optional[List[str]] = field(
        default=None, metadata={"help": "The pattern of refined recompute."}
    )
    virtual_pipeline_seg_method: str = field(
        default="LlamaDecoderLayerAuto",
        metadata={"help": "The seg method of splitting pp layer for virtual pipeline."},
    )
    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    use_intermediate_api: bool = field(
        default=False,
        metadata={"help": "Weather to use auto_parallel intermediate api"},
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

        logger.info(self.strategy)


@dataclass
class ModelAutoConfig(ModelConfig):
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(
        default="llama", metadata={"help": "Only support for llama pre-training for now."}
    )
    num_hidden_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of hidden layers in the Transformer encoder."}
    )


@dataclass
class GenerateArgument:
    top_k: int = field(
        default=1,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"
        },
    )
    top_p: float = field(
        default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."}
    )


@dataclass
class ReftArgument:
    layers: str = field(default="all", metadata={"help": "Layer configuration for the model."})
    position: str = field(default="f7+l7", metadata={"help": "Position parameter for model."})
    intervention_type: str = field(default="LoreftIntervention", metadata={"help": "Type of intervention."})
    rank: int = field(default=8, metadata={"help": "Rank parameter for model."})
    act_fn: str = field(default="linear", metadata={"help": "Activation function."})
    add_bias: bool = field(default=False, metadata={"help": "Flag indicating whether to add bias."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate."})


def main():
    parser = PdArgumentParser((GenerateArgument, ModelAutoConfig, ReftArgument, DataConfig, SFTAutoConfig))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        gen_args, model_args, reft_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        gen_args, model_args, reft_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(gen_args, "Generation")

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

    if get_env_device() == "xpu" and training_args.gradient_accumulation_steps > 1:
        try:
            from paddle_xpu.layers.nn.linear import LinearConfig  # noqa: F401

            LinearConfig.enable_accumulate_steps_opt()
            LinearConfig.set_accumulate_steps(training_args.gradient_accumulation_steps)
        except ImportError:
            # It's OK, not use accumulate_steps optimization
            pass

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
    quantization_config = dict(
        weight_quantize_algo=model_args.weight_quantize_algo,
        qlora_weight_blocksize=model_args.qlora_weight_blocksize,
        qlora_weight_double_quant=model_args.qlora_weight_double_quant,
        qlora_weight_double_quant_block_size=model_args.qlora_weight_double_quant_block_size,
    )
    config_class, model_class, criterion_class = MODEL_CLASSES[model_args.model_type]
    model_config = config_class.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        from_aistudio=model_args.from_aistudio,
        quantization_config=quantization_config,
    )
    model_config.use_flash_attention = training_args.use_flash_attention
    model_config.use_fast_layer_norm = model_args.use_fast_layer_norm
    model_config.num_hidden_layers = (
        model_args.num_hidden_layers if model_args.num_hidden_layers is not None else model_config.num_hidden_layers
    )
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

    # Config for model using long sequence strategy
    if model_args.use_long_sequence_strategies:
        data_args.scaled_max_length = int(data_args.max_length * model_args.rope_scaling_factor)
        model_config.use_long_sequence_strategies = True
        model_config.long_sequence_strategy_type = model_args.strategy_type
        model_config.long_sequence_strategy_name = model_args.strategy_name
        model_config.rope_scaling_factor = model_args.rope_scaling_factor
        model_config.long_sequence_init_args = {
            "dim": int(model_config.hidden_size / model_config.num_attention_heads),
            "max_position_embeddings": data_args.scaled_max_length,  # extended context window
            "base": model_config.rope_theta,
            "scaling_factor": model_args.rope_scaling_factor,
        }
        if model_args.strategy_name == "YaRNScalingRotaryEmbedding":
            model_config.long_sequence_init_args["original_max_position_embeddings"] = data_args.max_length
    model_config.sequence_parallel = training_args.sequence_parallel
    model_config.pipeline_parallel_degree = training_args.pipeline_parallel_degree
    model_config.tensor_parallel_degree = training_args.tensor_parallel_degree

    logger.info(f"Final model config: {model_config}")

    if model_args.continue_training and not training_args.autotuner_benchmark:
        criterion = criterion_class(model_config)
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            from_aistudio=model_args.from_aistudio,
        )
    else:
        with paddle.LazyGuard():
            criterion = criterion_class(model_config)
            # NOTE(gongenlei): new add autotuner_benchmark
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
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, from_aistudio=model_args.from_aistudio)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    reft_layers = None
    if model_args.reft:
        # reft requires padding side right
        tokenizer.padding_side = "right"
        layers = reft_args.layers
        if reft_args.layers != "all":
            layers = [int(l) for l in layers.split(";")]
        else:
            layers = [l for l in range(model_config.num_hidden_layers)]
        reft_layers = layers
        logging.info("Using ReFT with layers: ", reft_layers)
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

    else:
        trans_func = partial(get_convert_example(model), tokenizer=tokenizer, data_args=data_args)

    eval_zero_padding = data_args.zero_padding
    if data_args.zero_padding and data_args.eval_with_do_generation:
        logger.warning(
            "`zero_padding` conflicts with `eval_with_do_generation`. Setting zero_padding to False for the eval_dataset."
        )
        eval_zero_padding = False

    train_ds, dev_ds, test_ds = trans_dataset_to_ids(
        train_ds, dev_ds, test_ds, model_args, data_args, trans_func, eval_zero_padding
    )

    model = create_peft_model(model_args, reft_args, training_args, dtype, model_config, model, reft_layers)

    if (
        training_args.pipeline_parallel_degree > 1
        or training_args.sequence_parallel
        or training_args.autotuner_benchmark
        or data_args.zero_padding
        or data_args.pad_to_max_length
    ):
        max_length = data_args.max_length
        padding = "max_length"
    elif max(training_args.sharding_parallel_degree, training_args.data_parallel_degree) == 1:
        max_length = None
        padding = True
    else:
        max_length = data_args.max_length
        padding = "max_length"
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
    trainer = SFTAutoTrainer(
        model=model,
        criterion=criterion,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        data_collator=data_collator_fn if not model_args.reft else ReftDataCollator(data_collator=data_collator_fn),
        do_generation=data_args.eval_with_do_generation,
        callbacks=None,
        gen_args=gen_args,
        data_args=data_args,
    )

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
            if model_args.save_to_aistudio:
                save_to_aistudio(model_args, training_args, trainer)

            if not training_args.autotuner_benchmark:
                trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
                trainer.save_state()

    # Evaluation test set
    if training_args.do_predict:
        eval_result = trainer.predict(test_ds).metrics
        trainer.log_metrics("test", eval_result)
    training_args.do_eval = False
    # Evaluation dev set
    if training_args.do_eval:
        logger.info("*** Evaluate result after train ***")
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)


def save_to_aistudio(model_args, training_args, trainer):
    kwargs = {}
    if model_args.aistudio_token is not None:
        kwargs["token"] = model_args.aistudio_token
        # PEFT Model only save PEFT parameters, if pretrained model obtains from aistudio
    if model_args.from_aistudio and (model_args.lora or model_args.prefix_tuning):
        kwargs["base_model"] = model_args.model_name_or_path
    else:
        trainer.tokenizer.save_to_aistudio(
            repo_id=model_args.aistudio_repo_id,
            private=model_args.aistudio_repo_private,
            license=model_args.aistudio_repo_license,
            exist_ok=True,
            **kwargs,
        )
    trainer.model.save_to_aistudio(
        repo_id=model_args.aistudio_repo_id,
        private=model_args.aistudio_repo_private,
        license=model_args.aistudio_repo_license,
        merge_tensor_parallel=training_args.tensor_parallel_degree > 1,
        exist_ok=True,
        **kwargs,
    )


def get_prefix_tuning_params(model):
    if model.base_model_prefix == "llama":
        from paddlenlp.peft.prefix import llama_postprocess_past_key_value

        num_attention_heads = model.config.n_head
        num_hidden_layers = model.config.n_layer
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = llama_postprocess_past_key_value
        multi_query_group_num = None
    else:
        raise ValueError(f"Unknown base_model_prefix: {model.base_model_prefix}. ")
    return dict(
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        postprocess_past_key_value=postprocess_past_key_value,
        multi_query_group_num=multi_query_group_num,
    )


def create_peft_model(model_args, reft_args, training_args, dtype, model_config, model, reft_layers):
    if model_args.prefix_tuning:
        if training_args.pipeline_parallel_degree > 1:
            raise NotImplementedError("Prefix tuning is not implemented for pipeline parallelism.")

        prefix_tuning_params = get_prefix_tuning_params(model)
        prefix_config = PrefixConfig(
            num_prefix_tokens=model_args.num_prefix_tokens,
            num_attention_heads=prefix_tuning_params["num_attention_heads"],
            num_hidden_layers=prefix_tuning_params["num_hidden_layers"],
            hidden_size=prefix_tuning_params["hidden_size"],
            multi_query_group_num=prefix_tuning_params["multi_query_group_num"],
            dtype=dtype,
        )
        if model_args.prefix_path is None:
            model = PrefixModelForCausalLM(
                model=model,
                prefix_config=prefix_config,
                postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
            )
        else:
            model = PrefixModelForCausalLM.from_pretrained(
                model=model,
                prefix_path=model_args.prefix_path,
                postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
            )
        model.print_trainable_parameters()

    if model_args.lora:
        if training_args.sharding_parallel_degree > 1:
            assert (
                "enable_stage1_overlap" not in training_args.sharding_parallel_config
            ), "Currently not support enabling sharding_stage1_overlap in lora mode."
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAAutoConfig(
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
                use_intermediate_api=training_args.use_intermediate_api,
                pipeline_parallel_degree=training_args.pipeline_parallel_degree,
            )
            model = LoRAAutoModel(model, lora_config)
        else:
            model = LoRAAutoModel.from_pretrained(model=model, lora_path=model_args.lora_path)

        model.print_trainable_parameters()

    if model_args.lokr:
        if model_args.lokr_path is None:
            target_modules = get_lora_target_modules(model)
            lokr_config = LoKrConfig(
                target_modules=target_modules,
                lokr_dim=model_args.lokr_dim,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
            )
            model = LoKrModel(model, lokr_config)
        else:
            model = LoKrModel.from_pretrained(model=model, lokr_path=model_args.lokr_path)

    if model_args.reft:
        intervention_dtype = dtype
        intervention_params = {
            "embed_dim": model_config.hidden_size,
            "low_rank_dimension": reft_args.rank,
            "dropout": reft_args.dropout,
            "dtype": intervention_dtype,
            "act_fn": reft_args.act_fn,
            "device": "gpu",
            "add_bias": reft_args.add_bias,
        }
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": reft_args.rank,
                "intervention": intervention_mapping[reft_args.intervention_type](**intervention_params),
            }
            for l in reft_layers
        ]
        reft_config = ReFTConfig(
            representations=representations, intervention_params=intervention_params, position=reft_args.position
        )
        # get reft model
        model = ReFTModel(reft_config, model)
        # disable original model gradients
        model.disable_model_gradients()
        model.print_trainable_parameters()

    if model_args.vera:
        target_modules = get_lora_target_modules(model)
        vera_config = VeRAConfig(
            target_modules=target_modules,
            r=model_args.vera_rank,
            vera_alpha=model_args.vera_rank,
            dtype=dtype,
            base_model_name_or_path=model_args.model_name_or_path,
            pissa_init=True,
        )
        model = VeRAModel(model, vera_config)
        model.mark_only_vera_as_trainable(notfreezeB=True)
        model.print_trainable_parameters()

    return model


def trans_dataset_to_ids(train_ds, dev_ds, test_ds, model_args, data_args, trans_func, eval_zero_padding):
    if train_ds is not None:
        train_ds = train_ds.map(
            partial(trans_func, is_test=False, zero_padding=data_args.zero_padding, flash_mask=model_args.flash_mask)
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
        if training_args.do_train:
            train_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                lazy=data_args.lazy,
            )[0]
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "dev.json"),
                lazy=data_args.lazy,
            )[0]
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
