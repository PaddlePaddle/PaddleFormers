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

from paddleformers.utils.download import resolve_file_path
from paddleformers.utils.env import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from paddleformers.utils.log import logger


def resolve_weight_source(model_name_or_path, download_source_kwargs={}):
    convert_from_hf = False
    save_to_hf = False
    resolve_result = resolve_file_path(
        model_name_or_path,
        [SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME],
        **download_source_kwargs,
    )
    if resolve_result is not None:
        resolve_path = os.path.dirname(resolve_result)
        config_json = os.path.join(resolve_path, "config.json")
        with open(config_json) as f:
            config_dict = json.load(f)
        if "torch_dtype" in config_dict:
            convert_from_hf = True
            save_to_hf = True
        logger.info(f"base model path parsed:{resolve_path}")
    else:
        logger.error(f"{model_name_or_path} does not found.")
    return {
        "convert_from_hf": convert_from_hf,
        "save_to_hf": save_to_hf,
    }
