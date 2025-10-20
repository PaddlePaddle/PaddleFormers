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

import json
import os

import numpy as np
import paddle
from paddleformers.utils.log import logger

MODEL_LIB_NAMES = [
    "ernie.modeling",
    "ernie.modeling_moe",
    "ernie.modeling_moe_pp",
]

MAX_BSZ = 256
MAX_DRAFT_TOKENS = 6
LOCAL_RANK = int(os.getenv("PADDLE_RANK_IN_NODE", 0))


def calculate_effective_tokens(training_args, train_dataset, max_seq_len):
    """
    Calculate the effective tokens during training.

    Args:
        training_args: Configuration object containing training parameters.
        train_dataset: Dataset used for training.
        max_seq_len: Maximum sequence length of the model.

    Returns:
        tuple: Contains total_effective_tokens (int) and total_tokens (int).
    """
    total_effective_tokens = 0
    try:
        data_parallel_degree = training_args.data_parallel_degree
    except:
        data_parallel_degree = 1
    if training_args.sharding_parallel_degree > 1:
        sharding_parallel_degree = training_args.sharding_parallel_degree
    else:
        sharding_parallel_degree = 1

    total_batch = (
        training_args.max_steps
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * sharding_parallel_degree
        * data_parallel_degree
    )
    for i, data in enumerate(train_dataset):
        if i == total_batch:
            break
        for dd in data:
            total_effective_tokens += len(dd.token_ids)
    total_tokens = total_batch * max_seq_len

    return total_effective_tokens, total_tokens


def estimate_training(train_dataset, data_args, training_args, model_args):
    """
    Estimate required training steps based on dataset.

    Args:
        train_dataset: Dataset used for training estimation.
        data_args: Configuration object containing data parameters.
        training_args: Configuration object containing training parameters.
        model_args: Configuration object containing model parameters.

    Returns:
        dict: Contains estimated training steps and related parameters.
    """
    train_dataset.estimate = True
    logger.info("Start to estimate max training steps...")

    train_dataset_path_list = [path for path in str(data_args.train_dataset_path).replace(" ", "").split(',')]
    if len(train_dataset_path_list) > 1:
        logger.warning("Suggest to use max_steps instead of num_train_epochs for multi source dataset.")
        logger.info(
            "Multi source dataset detected, number of samples will be estimated by following rule. "
            "num_samples = (source1_num_samples * prob1 + source2_num_samples * prob2 + ...) * epochs"
        )

    max_samples = train_dataset.max_estimate_samples

    if training_args.max_estimate_samples != -1:
        # Set estimate samples to max_estimate_samples
        logger.warning("The results between sampling and non-sampling methods may differ.")
        train_dataset.max_estimate_samples = min(
            training_args.max_estimate_samples, train_dataset.max_estimate_samples
        )

    if train_dataset.max_estimate_samples > 0:
        train_batches = 0
        train_tokens = 0
        for sequences in train_dataset:
            if not train_dataset.estimate:
                break
            train_batches += 1
            for sequence in sequences:
                train_tokens += len(sequence.token_ids)

        train_tokens *= training_args.num_train_epochs
        train_batches *= training_args.num_train_epochs
        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * max(training_args.data_parallel_degree, 1)
            * max(training_args.sharding_parallel_degree, 1)
        )
        max_steps = train_batches / global_batch_size

        if max_samples != train_dataset.max_estimate_samples:
            max_steps *= max_samples / train_dataset.max_estimate_samples
            train_tokens *= max_samples / train_dataset.max_estimate_samples
            train_dataset.used_samples *= max_samples / train_dataset.max_estimate_samples
            train_dataset.unused_samples *= max_samples / train_dataset.max_estimate_samples

        max_steps = int(np.ceil(max_steps))

        res = {
            "num_train_epochs": int(training_args.num_train_epochs),
            "max_steps": max_steps,
            "train_tokens": int(train_tokens),
            "global_batch_size": int(global_batch_size),
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "warmup_steps": int(np.ceil(0.1 * max_steps)),
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "tensor_parallel_degree": int(training_args.tensor_parallel_degree),
            "pipeline_parallel_degree": int(training_args.pipeline_parallel_degree),
            "sharding_parallel_degree": int(training_args.sharding_parallel_degree),
            "seed": training_args.seed,
            "num_samples_each_epoch": data_args.num_samples_each_epoch,
            "max_seq_len": int(data_args.max_seq_len),
            "valid": True,
            "train_samples": int(max_samples * training_args.num_train_epochs),
            "estimate_samples": int(train_dataset.max_estimate_samples),
            "actual_train_samples": int(train_dataset.used_samples * training_args.num_train_epochs),
            "skip_samples": int(train_dataset.unused_samples * training_args.num_train_epochs),
        }
        if hasattr(training_args, "num_of_gpus"):
            res["num_of_gpus"] = training_args.num_of_gpus

        if train_batches / training_args.num_train_epochs / global_batch_size < 1:
            logger.warning("This dataset is too small, you'd better enlarge your dataset.")
            res["valid"] = False

        if getattr(training_args, "estimation_output_file", None):
            with open(training_args.estimation_output_file, "w", encoding="utf-8") as f:
                json.dump(res, f)

        return max_steps
    else:
        res = {
            "num_train_epochs": int(training_args.num_train_epochs),
            "max_steps": 0,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "train_tokens": 0,
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "tensor_parallel_degree": int(training_args.tensor_parallel_degree),
            "pipeline_parallel_degree": int(training_args.pipeline_parallel_degree),
            "sharding_parallel_degree": int(training_args.sharding_parallel_degree),
            "num_samples_each_epoch": data_args.num_samples_each_epoch,
            "max_seq_len": int(data_args.max_seq_len),
            "seed": data_args.seed,
            "valid": False,
            "train_samples": 0,
        }
        if hasattr(training_args, "num_of_gpus"):
            res["num_of_gpus"] = training_args.num_of_gpus

        if getattr(training_args, "estimation_output_file", None):
            with open(training_args.estimation_output_file, "w", encoding="utf-8") as f:
                json.dump(res, f)

        logger.error("No valid data found, please check your dataset format.")
        return 0


def check_refined_recompute(rr, sequence_parallel, lora=False):
    """
    Update refined recompute configuration.

    Args:
        rr: Original recompute configuration (dict).
        sequence_parallel: Boolean indicating if sequence parallel is enabled.
        lora: Boolean indicating if LoRA is used.

    Returns:
        dict: Updated recompute configuration.
    """
    if len(rr) > 0:
        rr = {}
        logger.error("Currently do not support refine recompute; to be supported soon.")

    for op_name in rr.keys():
        if op_name in ["mlp_row_ln", "attention_row_ln", "attention_column_ln", "mlp_column_ln"]:
            if not sequence_parallel:
                logger.warning(
                    f"Currently, the `{op_name}` op is only supported "
                    "when `sequence_parallel=True`. This refined recompute op will be ignored."
                )
                continue
            if lora:
                logger.warning(
                    "Currently, LoRA does not support refined recompute "
                    f"for the `{op_name}` op. This refined recompute op will be ignored."
                )
                continue


def save_stop_info(args, stop_step, outside_eval, outside_predict):
    """
    Save training stop information to JSON file.

    Args:
        args: Command line arguments.
        stop_step: Step number when training stopped.
        outside_eval: Number of external evaluations performed.
        outside_predict: Number of external predictions made.

    Returns:
        None
    """

    process_index = paddle.distributed.get_rank() if LOCAL_RANK != -1 else 0
    if process_index != 0:
        return

    output_path = args.logging_dir
    eval_turns = 0 + outside_eval
    predict_turns = 0 + outside_predict
    if args.do_eval:
        eval_turns += stop_step // args.eval_steps

    data = {
        "stop_step": stop_step,
        "eval_turns": eval_turns,
        "predict_turns": predict_turns,
    }
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, "stop_step.json")
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
    logger.info(f"Saving stop info into {file_path}")
    return


def add_start_docstrings(*docstr):
    """
    Decorator to prepend docstrings to function documentation.

    Args:
        *docstr: Variable length argument list of docstrings.

    Returns:
        function: Decorator function.
    """

    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def infer_save_test_case(cases: list[list[dict]], file: str):
    """save test to result file

    Args:
        cases (list[list[dict]]): the content of case
        file (str): the path of saved file
    """
    with open(file, "a+", encoding="utf-8") as f:
        for case in cases:
            raw = json.dumps(case, ensure_ascii=False)
            f.write(raw + "\n")
