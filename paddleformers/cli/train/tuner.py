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

from typing import Any, Optional

import paddle

from ..hparams import get_train_args, read_args
from .dpo import run_dpo
from .pretrain import run_dsv3_pretrain
from .sft import run_sft


def check_path(path):
    """_summary_"""
    if path is None:
        raise ValueError("Dataset Path is None. Please set dataset path firstly.")
    else:
        pass


def _training_function(config: dict[str, Any]) -> None:
    """_summary_

    Args:
        config (dict[str, Any]): _description_

    Raises:
        ValueError: _description_
    """
    args = config.get("args")
    model_args, data_args, preprocess_args, generating_args, finetuning_args = get_train_args(args)

    if "VL" in model_args.stage or model_args.stage == "dsv3_pretrain":
        pass
    elif data_args.dataset_type != "pretrain":
        check_path(data_args.train_dataset_path)
        check_path(data_args.eval_dataset_path)

    if model_args.stage == "SFT" or model_args.stage == "PT":
        with paddle.amp.auto_cast(enable=False):
            run_sft(model_args, data_args, generating_args, finetuning_args)
    elif model_args.stage == "DPO":
        with paddle.amp.auto_cast(enable=False):
            run_dpo(model_args, data_args, generating_args, finetuning_args)
    elif model_args.stage == "dsv3_pretrain":
        run_dsv3_pretrain(model_args, data_args, generating_args, finetuning_args)
    else:
        raise ValueError(f"Unknown task: {model_args.stage}.")


def run_tuner(args: Optional[dict[str, Any]] = None) -> None:
    """_summary_

    Args:
        args (Optional[dict[str, Any]], optional): _description_. Defaults to None.
    """
    args = read_args(args)

    _training_function(config={"args": args})
