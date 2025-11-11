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

try:
    from paddleformers.utils.download import MODEL_MAPPINGS, check_repo
except ImportError:
    # for old paddleformers
    import re

    from paddleformers.utils.log import logger

    check_repo = None

    MODEL_MAPPINGS = {
        "ERNIE-4.5-300B-A47B-Base": {
            "huggingface": "baidu/ERNIE-4.5-300B-A47B-Base-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-300B-A47B-Base-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-300B-A47B-Base-Paddle",
        },
        "ERNIE-4.5-300B-A47B": {
            "huggingface": "baidu/ERNIE-4.5-300B-A47B-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle",
        },
        "ERNIE-4.5-21B-A3B-Base": {
            "huggingface": "baidu/ERNIE-4.5-21B-A3B-Base-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-21B-A3B-Base-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-21B-A3B-Base-Paddle",
        },
        "ERNIE-4.5-21B-A3B": {
            "huggingface": "baidu/ERNIE-4.5-21B-A3B-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle",
        },
        "ERNIE-4.5-0.3B-Base": {
            "huggingface": "baidu/ERNIE-4.5-0.3B-Base-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-0.3B-Base-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-0.3B-Base-Paddle",
        },
        "ERNIE-4.5-0.3B": {
            "huggingface": "baidu/ERNIE-4.5-0.3B-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-0.3B-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-0.3B-Paddle",
        },
        "ERNIE-4.5-VL-424B-A47B-Base": {
            "huggingface": "baidu/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
        },
        "ERNIE-4.5-VL-424B": {
            "huggingface": "baidu/ERNIE-4.5-VL-424B-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-VL-424B-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-VL-424B-Paddle",
        },
        "ERNIE-4.5-VL-28B-A3B-Base": {
            "huggingface": "baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
        },
        "ERNIE-4.5-VL-28B-A3B": {
            "huggingface": "baidu/ERNIE-4.5-VL-28B-A3B-Paddle",
            "aistudio": "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle",
            "modelscope": "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle",
        },
    }


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
        # check remote repo
        if check_repo is not None:
            model_name_or_path = check_repo(model_name_or_path, download_hub)
        else:
            # check remote repo
            model_name = model_name_or_path.split("/")[-1].rstrip("-Paddle")
            if model_name in MODEL_MAPPINGS.keys():
                if re.match(
                    r"^(baidu|PaddlePaddle)/ERNIE-4\.5-.+-Paddle$", model_name_or_path
                ):  # model download from baidu
                    download_repo = MODEL_MAPPINGS[model_name]
                    if download_hub == "huggingface":
                        if model_name_or_path != download_repo["huggingface"]:
                            logger.warning(
                                f"The repo id of baidu's model in the huggingface should be 'baidu', model_name_or_path has changed to {download_repo['huggingface']}"
                            )
                        return download_repo["huggingface"]
                    elif download_hub == "aistudio":
                        if model_name_or_path != download_repo["aistudio"]:
                            logger.warning(
                                f"The repo id of baidu's model in the aistudio should be 'PaddlePaddle', model_name_or_path has changed to {download_repo['aistudio']}"
                            )
                        return download_repo["aistudio"]
                    elif download_hub == "modelscope":
                        if model_name_or_path != download_repo["modelscope"]:
                            logger.warning(
                                f"The repo id of baidu's model in the modelscope should be 'PaddlePaddle', model_name_or_path has changed to {download_repo['modelscope']}"
                            )
                        return download_repo["modelscope"]
                    else:
                        raise ValueError(
                            "please select a model downloading source by setting `download_hub`: `huggingface`, `aistudio`, `modelscope`"
                        )

    return model_name_or_path
