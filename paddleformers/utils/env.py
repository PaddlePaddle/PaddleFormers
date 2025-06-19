# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""
This module is used to store environmental variables in PaddleFormers.
PPNLP_HOME              -->  the root directory for storing PaddleFormers related data. Default to ~/.paddleformers. Users can change the
├                            default value through the PPNLP_HOME environment variable.
├─ MODEL_HOME              -->  Store model files.
└─ DATA_HOME         -->  Store automatically downloaded datasets.
"""
import os
import re

try:
    from paddle.base.framework import use_pir_api

    pir_enabled = use_pir_api()
except ImportError:
    pir_enabled = False


def _get_user_home():
    return os.path.expanduser("~")


def _get_ppnlp_home():
    if "PPNLP_HOME" in os.environ:
        home_path = os.environ["PPNLP_HOME"]
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError("The environment variable PPNLP_HOME {} is not a directory.".format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), ".paddleformers")


def _get_sub_home(directory, parent_home=_get_ppnlp_home()):
    home = os.path.join(parent_home, directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


def _get_bool_env(env_key: str, default_value: str) -> bool:
    """get boolean environment variable, which can be "true", "True", "1"

    Args:
        env_key (str): key of env variable
    """
    value = os.getenv(env_key, default_value).lower()
    return value in ["true", "1"]


USER_HOME = _get_user_home()
PPNLP_HOME = _get_ppnlp_home()
MODEL_HOME = _get_sub_home("models")
HF_CACHE_HOME = os.environ.get("HUGGINGFACE_HUB_CACHE", MODEL_HOME)
DATA_HOME = _get_sub_home("datasets")
PACKAGE_HOME = _get_sub_home("packages")
DOWNLOAD_SERVER = "http://paddlepaddle.org.cn/paddlehub"
FAILED_STATUS = -1
SUCCESS_STATUS = 0

SPECIAL_TOKENS_MAP_NAME = "special_tokens_map.json"
ADDED_TOKENS_NAME = "added_tokens.json"
LEGACY_CONFIG_NAME = "model_config.json"
CONFIG_NAME = "config.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
CHAT_TEMPLATE_CONFIG_NAME = "chat_template.json"
NONE_CHAT_TEMPLATE = "{% for message in messages -%} {% if message['role'] == 'user' or message['role'] == 'assistant' -%} {{ message['content'] }} {% endif -%}{% endfor -%}"
GENERATION_CONFIG_NAME = "generation_config.json"

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
MODEL_META_NAME = "model_meta.json"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"
SHARDING_META_NAME = "shard_meta.json"

# checkpoint dir name and regex
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
FULL_TOKENIZER_NAME = "tokenizer.json"
TIKTOKEN_VOCAB_FILE = "tokenizer.model"
MERGE_CONFIG_NAME = "merge_config.json"

LORA_CONFIG_NAME = "lora_config.json"
LORA_WEIGHTS_NAME = "lora_model_state.pdparams"

VERA_CONFIG_NAME = "vera_config.json"
VERA_WEIGHTS_NAME = "vera_model_state.pdparams"

PREFIX_CONFIG_NAME = "prefix_config.json"
PREFIX_WEIGHTS_NAME = "prefix_model_state.pdparams"
PADDLE_PEFT_WEIGHTS_INDEX_NAME = "peft_model.pdparams.index.json"

LOKR_WEIGHTS_NAME = "lokr_model_state.pdparams"
LOKR_CONFIG_NAME = "lokr_config.json"

PAST_KEY_VALUES_FILE_NAME = "pre_caches.npy"

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
PADDLE_WEIGHTS_INDEX_NAME = "model_state.pdparams.index.json"

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

PADDLE_OPTIMIZER_NAME = "optimizer.pdopt"
PADDLE_OPTIMIZER_INDEX_NAME = "optimizer.pdopt.index.json"

SAFE_OPTIMIZER_NAME = "optimizer.safetensors"
SAFE_OPTIMIZER_INDEX_NAME = "optimizer.safetensors.index.json"

PADDLE_MASTER_WEIGHTS_NAME = "master_weights.pdparams"
PADDLE_MASTER_WEIGHTS_INDEX_NAME = "master_weights.pdparams.index.json"

SAFE_MASTER_WEIGHTS_NAME = "master_weights.safetensors"
SAFE_MASTER_WEIGHTS_INDEX_NAME = "master_weights.safetensors.index.json"

SAFE_PEFT_WEIGHTS_NAME = "peft_model.safetensors"
SAFE_PEFT_WEIGHTS_INDEX_NAME = "peft_model.safetensors.index.json"

# Checkpoint quantization
MOMENT1_KEYNAME = "moment1_0"
MOMENT2_KEYNAME = "moment2_0"
BETA1_KEYNAME = "beta1_pow_acc_0"
BETA2_KEYNAME = "beta2_pow_acc_0"
SYMMETRY_QUANT_SCALE = "@scales"
ASYMMETRY_QUANT_SCALE_MIN = "@min_scales"
ASYMMETRY_QUANT_SCALE_MAX = "@max_scales"
MAX_QUANTIZATION_TIMES = 1

# LLM Inference related environment variables
# Note(@Wanglongzhi2001): MAX_BSZ must be the same as definition in get_output / save_output
# SPECULATE_MAX_BSZ, MAX_DRAFT_TOKENS must be the same as definition in speculate_get_output / speculate_save_output
MAX_BSZ = 512
SPECULATE_MAX_BSZ = 256
MAX_DRAFT_TOKENS = 6

if pir_enabled:
    PADDLE_INFERENCE_MODEL_SUFFIX = ".json"
    PADDLE_INFERENCE_WEIGHTS_SUFFIX = ".pdiparams"
else:
    PADDLE_INFERENCE_MODEL_SUFFIX = ".pdmodel"
    PADDLE_INFERENCE_WEIGHTS_SUFFIX = ".pdiparams"

USE_FAST_TOKENIZER: bool = _get_bool_env("USE_FAST_TOKENIZER", "false")
PREFILL_USE_SAGE_ATTN: bool = _get_bool_env("PREFILL_USE_SAGE_ATTN", "false")
