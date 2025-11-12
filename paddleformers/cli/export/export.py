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
import shutil
import time
from typing import Any, Optional

import paddle

from paddleformers.mergekit import MergeConfig, MergeModel
from paddleformers.trainer import get_last_checkpoint
from paddleformers.utils.download import check_repo, resolve_file_path
from paddleformers.utils.env import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from paddleformers.utils.log import logger

from ..hparams import get_export_args, read_args
from ..utils.process import is_valid_model_dir


def check_download_repo(model_name_or_path, download_hub=None):
    # Detect torch model.
    is_local = os.path.isfile(model_name_or_path) or os.path.isdir(model_name_or_path)
    if is_local:
        config_path = os.path.join(model_name_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            if "torch_dtype" in config_dict:
                print("Loading local model which contains torch dtype.")
    else:
        model_name_or_path = check_repo(model_name_or_path, download_hub)

    return model_name_or_path


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


def run_export(args: Optional[dict[str, Any]] = None) -> None:
    """_summary_

    Args:
        args (Optional[dict[str, Any]], optional): _description_. Defaults to None.
    """

    args = read_args(args)
    model_args, data_args, generating_args, finetuning_args, export_args = get_export_args(args)

    paddle.set_device(finetuning_args.device)

    last_checkpoint = None
    if os.path.isdir(finetuning_args.output_dir):
        # Check if the output directory is a valid model directory (contains .safetensors or .pdparams files)
        if is_valid_model_dir(finetuning_args.output_dir):
            last_checkpoint = finetuning_args.output_dir
        # If not a model directory but still a valid path, try to find the latest checkpoint
        else:
            last_checkpoint = get_last_checkpoint(finetuning_args.output_dir)
    if last_checkpoint is not None:
        logger.info(f"Starting model export from checkpoint: {last_checkpoint}")
    else:
        raise FileNotFoundError(f"No valid checkpoint found in: {finetuning_args.output_dir}")

    if model_args.lora:
        start = time.time()
        logger.info("***** Start merging LoRA model *****")

        model_args.model_name_or_path = check_download_repo(
            model_args.model_name_or_path,
            download_hub=model_args.download_hub,
        )

        download_source_kwargs = {}
        download_source_kwargs["download_hub"] = model_args.download_hub

        resolve_result = resolve_file_path(
            model_args.model_name_or_path,
            [SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME],
            **download_source_kwargs,
        )

        if resolve_result is not None:
            resolve_path = os.path.dirname(resolve_result)
            logger.info(f"base model path parsed:{resolve_path}")
        else:
            logger.error(f"{model_args.model_name_or_path} does not found.")

        config = {}
        config["base_model_path"] = resolve_path
        config["lora_model_path"] = last_checkpoint
        config["output_path"] = os.path.join(finetuning_args.output_dir, "export")
        config["convert_from_hf"] = finetuning_args.convert_from_hf
        config["save_to_hf"] = finetuning_args.save_to_hf

        if export_args.copy_tokenizer:
            config["copy_file_list"] = [
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                # "config.json",
            ]

        merge_config = MergeConfig(**config)
        mergekit = MergeModel(merge_config)
        logger_merge_config(merge_config, model_args.lora)
        mergekit.merge_model()
        src_file = os.path.join(config["base_model_path"], "config.json")
        dst_file = os.path.join(config["output_path"], "config.json")
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            logger.debug(f'Copy failed: "config.json" not found in {config["base_model_path"]}')
        src_file = os.path.join(config["base_model_path"], "preprocessor_config.json")
        dst_file = os.path.join(config["output_path"], "preprocessor_config.json")
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            logger.debug(f'Copy failed: "preprocessor_config.json" not found in {config["base_model_path"]}')
        logger.info(f"***** Successfully finished merging LoRA model. Time cost: {time.time() - start} s *****")
    else:
        raise ValueError("Only support merge lora checkpoint, but get model_args.lora is False.")
