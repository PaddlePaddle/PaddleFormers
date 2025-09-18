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
"""Model Merge Tools."""

import argparse
import json
import os
import shutil
import time

import paddle

from paddleformers.mergekit import MergeConfig, MergeModel
from paddleformers.trainer.argparser import strtobool
from paddleformers.utils.log import logger


def parse_arguments():
    """
    Parse command line arguments for model merging configuration.

    This function sets up and configures all available command line arguments
    for the model merging process, including paths, device selection, and optional
    tokenizer handling.

    Returns:
        argparse.Namespace: An object containing all parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mergekit_task_config", type=str, help="The merge config path.")
    parser.add_argument("--output_path", required=True, type=str, help="The merge config path.")
    parser.add_argument("--lora_model_path", default=None, type=str, help="The lora model path.")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="The base model path.")
    parser.add_argument("--device", default="gpu", type=str, help="Device")
    parser.add_argument("--copy_tokenizer", default="True", type=strtobool, help="Copy tokenizer file")
    parser.add_argument(
        "--convert_from_hf", default="True", type=strtobool, help="Load base model from HuggingFace safetensors."
    )
    parser.add_argument("--save_to_hf", default="True", type=strtobool, help="Save model to HuggingFace safetensors.")
    return parser.parse_args()


def logger_merge_config(merge_config, lora_merge):
    """
    Logs the merge configuration details to debug output, with different formatting
    for LoRA merges versus standard model merges.

    Args:
        merge_config (object): Configuration object containing merge parameters.
                              Expected to have attributes accessible via __dict__.
        lora_merge (bool): Flag indicating whether this is a LoRA merge operation.
                           When True, logs only LoRA-specific parameters.
                           When False, logs standard merge parameters.

    Outputs:
        Writes formatted configuration details to the logger at DEBUG level.
        For LoRA merges: Displays centered "LoRA Merge Info" header and specific paths.
        For standard merges: Displays centered "Mergekit Config Info" header and all
        parameters except excluded ones.
    """
    if lora_merge:
        logger.debug("{:^40}".format("LoRA Merge Info"))
        for k, v in merge_config.__dict__.items():
            if k in ["lora_model_path", "base_model_path"]:
                logger.debug(f"{k:30}: {v}")
    else:
        logger.debug("{:^40}".format("Mergekit Config Info"))
        for k, v in merge_config.__dict__.items():
            if k in ["model_path_str", "device", "tensor_type", "merge_preifx"]:
                continue
            logger.debug(f"{k:30}: {v}")


def merge():
    """
    Main function for merging models, supporting both LoRA adapter merging and standard model merging.

    Handles the complete merging workflow including:
    - Argument parsing
    - Device configuration
    - Configuration setup for different merge types
    - Model merging execution
    - Progress logging and timing

    The function has two main execution paths:
    1. LoRA Merge: When lora_model_path is specified
    2. Standard Merge: When mergekit_task_config is specified

    Returns:
        None: Outputs are written to specified paths and logged to console
    """
    args = parse_arguments()

    paddle.set_device(args.device)

    lora_merge = args.lora_model_path is not None
    if lora_merge:
        start = time.time()
        logger.info("***** Start merging LoRA model *****")
        config = {}
        config["output_path"] = args.output_path
        config["lora_model_path"] = args.lora_model_path
        config["base_model_path"] = args.model_name_or_path
        config["convert_from_hf"] = args.convert_from_hf
        config["save_to_hf"] = args.save_to_hf
        if args.copy_tokenizer:
            config["copy_file_list"] = [
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]
        merge_config = MergeConfig(**config)
        mergekit = MergeModel(merge_config)
        logger_merge_config(merge_config, lora_merge)
        mergekit.merge_model()
        src_file = os.path.join(args.model_name_or_path, "config.json")
        dst_file = os.path.join(args.output_path, "config.json")
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            logger.debug(f"Copy failed: 'config.json' not found in {args.model_name_or_path}")
        logger.info(f"***** Successfully finished merging LoRA model. Time cost: {time.time() - start} s *****")
    else:
        with open(args.mergekit_task_config, "r", encoding="utf-8") as f:
            config_list = json.load(f)
        if not (isinstance(config_list, list) and all(isinstance(config, dict) for config in config_list)):
            raise ValueError("The mergekit_task_config must be a list of dict. Please check config.")

        for i, config in enumerate(config_list):
            logger.info("=" * 30)
            start = time.time()
            logger.info(f"***** Start merging model id: {i} *****")
            config["output_path"] = os.path.join(args.output_path, config.pop("output_folder_name"))
            config["tensor_type"] = "np" if args.device == "cpu" else "pd"
            if args.copy_tokenizer:
                config["copy_file_list"] = [
                    "tokenizer.model",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ]
            merge_config = MergeConfig(**config)
            mergekit = MergeModel(merge_config)
            logger_merge_config(merge_config, lora_merge)
            mergekit.merge_model()
            logger.info(f"***** Successfully finished merging model id: {i}. Time cost: {time.time() - start} s *****")


if __name__ == "__main__":
    merge()
