# Copyright 2025 the LlamaFactory team.
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
from collections import OrderedDict, defaultdict
from enum import Enum, unique
from typing import Optional

# from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
# from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
# from transformers.utils import (
#     SAFE_WEIGHTS_INDEX_NAME,
#     SAFE_WEIGHTS_NAME,
#     WEIGHTS_INDEX_NAME,
#     WEIGHTS_NAME,
# )

AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")

# CHECKPOINT_NAMES = {
#     SAFE_ADAPTER_WEIGHTS_NAME,
#     ADAPTER_WEIGHTS_NAME,
#     SAFE_WEIGHTS_INDEX_NAME,
#     SAFE_WEIGHTS_NAME,
#     WEIGHTS_INDEX_NAME,
#     WEIGHTS_NAME,
# }

CHOICES = ["A", "B", "C", "D"]

DATA_CONFIG = "dataset_info.json"

DEFAULT_TEMPLATE = defaultdict(str)

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100

IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")

LAYERNORM_NAMES = {"norm", "ln"}

LLAMABOARD_CONFIG = "llamaboard_config.yaml"

METHODS = ["full", "freeze", "lora", "oft"]

MOD_SUPPORTED_MODELS = {"bloom", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "starcoder2"}

MULTIMODAL_SUPPORTED_MODELS = set()

PEFT_METHODS = {"lora", "oft"}

RUNNING_LOG = "running_log.txt"

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINER_LOG = "trainer_log.jsonl"

TRAINING_ARGS = "training_args.yaml"

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "KTO": "kto",
    "Pre-Training": "pt",
}

STAGES_USE_PAIR_DATA = {"rm", "dpo"}

SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}

SWANLAB_CONFIG = "swanlab_public_config.json"

VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"


class AttentionFunction(str, Enum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"


class EngineName(str, Enum):
    HF = "huggingface"
    VLLM = "vllm"
    SGLANG = "sglang"


class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"


@unique
class QuantizationMethod(str, Enum):
    r"""Borrowed from `transformers.utils.quantization_config.QuantizationMethod`."""

    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"
    MXFP4 = "mxfp4"


class RopeScaling(str, Enum):
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"


def register_model_group(
    models: dict[str, dict[DownloadSource, str]],
    template: Optional[str] = None,
    multimodal: bool = False,
) -> None:
    for name, path in models.items():
        SUPPORTED_MODELS[name] = path
        if template is not None and (
            any(suffix in name for suffix in ("-Chat", "-Distill", "-Instruct", "-Thinking")) or multimodal
        ):
            DEFAULT_TEMPLATE[name] = template

        if multimodal:
            MULTIMODAL_SUPPORTED_MODELS.add(name)


register_model_group(
    models={
        "Aya-23-8B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-8B",
        },
        "Aya-23-35B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-35B",
        },
    },
    template="cohere",
)


register_model_group(
    models={
        "Baichuan-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-7B",
            DownloadSource.MODELSCOPE: "baichuan-inc/baichuan-7B",
        },
        "Baichuan-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Base",
        },
        "Baichuan-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Chat",
        },
    },
    template="baichuan",
)


register_model_group(
    models={
        "Baichuan2-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Base",
        },
        "Baichuan2-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Base",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_13b_base_pt",
        },
        "Baichuan2-7B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Chat",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_7b_chat_pt",
        },
        "Baichuan2-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Chat",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_13b_chat_pt",
        },
    },
    template="baichuan2",
)


register_model_group(
    models={
        "BLOOM-560M": {
            DownloadSource.DEFAULT: "bigscience/bloom-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-560m",
        },
        "BLOOM-3B": {
            DownloadSource.DEFAULT: "bigscience/bloom-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-3b",
        },
        "BLOOM-7B1": {
            DownloadSource.DEFAULT: "bigscience/bloom-7b1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-7b1",
        },
    },
)


register_model_group(
    models={
        "BLOOMZ-560M": {
            DownloadSource.DEFAULT: "bigscience/bloomz-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-560m",
        },
        "BLOOMZ-3B": {
            DownloadSource.DEFAULT: "bigscience/bloomz-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-3b",
        },
        "BLOOMZ-7B1-mt": {
            DownloadSource.DEFAULT: "bigscience/bloomz-7b1-mt",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-7b1-mt",
        },
    },
)


register_model_group(
    models={
        "BlueLM-7B-Base": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Base",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Base",
        },
        "BlueLM-7B-Chat": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Chat",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Chat",
        },
    },
    template="bluelm",
)


register_model_group(
    models={
        "Breeze-7B": {
            DownloadSource.DEFAULT: "MediaTek-Research/Breeze-7B-Base-v1_0",
        },
        "Breeze-7B-Instruct": {
            DownloadSource.DEFAULT: "MediaTek-Research/Breeze-7B-Instruct-v1_0",
        },
    },
    template="breeze",
)


register_model_group(
    models={
        "ChatGLM2-6B-Chat": {
            DownloadSource.DEFAULT: "zai-org/chatglm2-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm2-6b",
        }
    },
    template="chatglm2",
)


register_model_group(
    models={
        "ChatGLM3-6B-Base": {
            DownloadSource.DEFAULT: "zai-org/chatglm3-6b-base",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b-base",
        },
        "ChatGLM3-6B-Chat": {
            DownloadSource.DEFAULT: "zai-org/chatglm3-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b",
        },
    },
    template="chatglm3",
)


register_model_group(
    models={
        "Chinese-Llama-2-1.3B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-1.3b",
        },
        "Chinese-Llama-2-7B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-7b",
        },
        "Chinese-Llama-2-13B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-13b",
        },
        "Chinese-Alpaca-2-1.3B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-1.3b",
        },
        "Chinese-Alpaca-2-7B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-7b",
        },
        "Chinese-Alpaca-2-13B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-13b",
        },
    },
    template="llama2_zh",
)


register_model_group(
    models={
        "CodeGeeX4-9B-Chat": {
            DownloadSource.DEFAULT: "zai-org/codegeex4-all-9b",
            DownloadSource.MODELSCOPE: "ZhipuAI/codegeex4-all-9b",
        },
    },
    template="codegeex4",
)


register_model_group(
    models={
        "CodeGemma-7B": {
            DownloadSource.DEFAULT: "google/codegemma-7b",
        },
        "CodeGemma-7B-Instruct": {
            DownloadSource.DEFAULT: "google/codegemma-7b-it",
            DownloadSource.MODELSCOPE: "AI-ModelScope/codegemma-7b-it",
        },
        "CodeGemma-1.1-2B": {
            DownloadSource.DEFAULT: "google/codegemma-1.1-2b",
        },
        "CodeGemma-1.1-7B-Instruct": {
            DownloadSource.DEFAULT: "google/codegemma-1.1-7b-it",
        },
    },
    template="gemma",
)


register_model_group(
    models={
        "Codestral-22B-v0.1-Chat": {
            DownloadSource.DEFAULT: "mistralai/Codestral-22B-v0.1",
            DownloadSource.MODELSCOPE: "swift/Codestral-22B-v0.1",
        },
    },
    template="mistral",
)


register_model_group(
    models={
        "CommandR-35B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-v01",
            DownloadSource.MODELSCOPE: "AI-ModelScope/c4ai-command-r-v01",
        },
        "CommandR-Plus-104B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-plus",
            DownloadSource.MODELSCOPE: "AI-ModelScope/c4ai-command-r-plus",
        },
        "CommandR-35B-4bit-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-v01-4bit",
            DownloadSource.MODELSCOPE: "mirror013/c4ai-command-r-v01-4bit",
        },
        "CommandR-Plus-104B-4bit-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-plus-4bit",
        },
    },
    template="cohere",
)


register_model_group(
    models={
        "DBRX-132B-Base": {
            DownloadSource.DEFAULT: "databricks/dbrx-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/dbrx-base",
        },
        "DBRX-132B-Instruct": {
            DownloadSource.DEFAULT: "databricks/dbrx-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/dbrx-instruct",
        },
    },
    template="dbrx",
)


register_model_group(
    models={
        "DeepSeek-LLM-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-base",
        },
        "DeepSeek-LLM-67B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-base",
        },
        "DeepSeek-LLM-7B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-chat",
        },
        "DeepSeek-LLM-67B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-chat",
        },
        "DeepSeek-Math-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-math-7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-base",
        },
        "DeepSeek-Math-7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-math-7b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-instruct",
        },
        "DeepSeek-MoE-16B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-moe-16b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-base",
        },
        "DeepSeek-MoE-16B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-moe-16b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-chat",
        },
        "DeepSeek-V2-16B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Lite",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite",
        },
        "DeepSeek-V2-236B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2",
        },
        "DeepSeek-V2-16B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Lite-Chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite-Chat",
        },
        "DeepSeek-V2-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Chat",
        },
        "DeepSeek-Coder-V2-16B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        },
        "DeepSeek-Coder-V2-236B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Base",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Base",
        },
        "DeepSeek-Coder-V2-16B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        },
        "DeepSeek-Coder-V2-236B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        },
    },
    template="deepseek",
)


register_model_group(
    models={
        "DeepSeek-Coder-6.7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-base",
        },
        "DeepSeek-Coder-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-7b-base-v1.5",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-7b-base-v1.5",
        },
        "DeepSeek-Coder-33B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-base",
        },
        "DeepSeek-Coder-6.7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-instruct",
        },
        "DeepSeek-Coder-7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        },
        "DeepSeek-Coder-33B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-instruct",
        },
    },
    template="deepseekcoder",
)


register_model_group(
    models={
        "DeepSeek-V2-0628-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Chat-0628",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Chat-0628",
        },
        "DeepSeek-V2.5-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2.5",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2.5",
        },
        "DeepSeek-V2.5-1210-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2.5-1210",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2.5-1210",
        },
        "DeepSeek-V3-671B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V3-Base",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3-Base",
        },
        "DeepSeek-V3-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V3",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3",
        },
        "DeepSeek-V3-0324-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V3-0324",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3-0324",
        },
    },
    template="deepseek3",
)


register_model_group(
    models={
        "DeepSeek-R1-1.5B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        },
        "DeepSeek-R1-7B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        },
        "DeepSeek-R1-8B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        },
        "DeepSeek-R1-14B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        },
        "DeepSeek-R1-32B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        },
        "DeepSeek-R1-70B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        },
        "DeepSeek-R1-671B-Chat-Zero": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Zero",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Zero",
        },
        "DeepSeek-R1-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1",
        },
        "DeepSeek-R1-0528-8B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        },
        "DeepSeek-R1-0528-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-0528",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-0528",
        },
    },
    template="deepseekr1",
)


register_model_group(
    models={
        "Devstral-Small-2507-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Devstral-Small-2507",
            DownloadSource.MODELSCOPE: "mistralai/Devstral-Small-2507",
        },
    },
    template="mistral_small",
)


register_model_group(
    models={
        "dots.ocr": {
            DownloadSource.DEFAULT: "rednote-hilab/dots.ocr",
            DownloadSource.MODELSCOPE: "rednote-hilab/dots.ocr",
        },
    },
    template="dots_ocr",
    multimodal=True,
)


register_model_group(
    models={
        "ERNIE-4.5-21B-A3B-Thinking": {
            DownloadSource.DEFAULT: "baidu/ERNIE-4.5-21B-A3B-Thinking",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-21B-A3B-Thinking",
        },
    },
    template="ernie",
)


register_model_group(
    models={
        "ERNIE-4.5-0.3B-PT": {
            DownloadSource.DEFAULT: "baidu/ERNIE-4.5-0.3B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-0.3B-PT",
        },
        "ERNIE-4.5-21B-A3B-PT": {
            DownloadSource.DEFAULT: "baidu/ERNIE-4.5-21B-A3B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-21B-A3B-PT",
        },
        "ERNIE-4.5-300B-A47B-PT": {
            DownloadSource.DEFAULT: "baidu/ERNIE-4.5-300B-A47B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-300B-A47B-PT",
        },
    },
    template="ernie_nothink",
)

register_model_group(
    models={
        "Phi-4-14B-Instruct": {
            DownloadSource.DEFAULT: "microsoft/phi-4",
            DownloadSource.MODELSCOPE: "LLM-Research/phi-4",
        },
    },
    template="phi4",
)


register_model_group(
    models={
        "Pixtral-12B": {
            DownloadSource.DEFAULT: "mistral-community/pixtral-12b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/pixtral-12b",
        }
    },
    template="pixtral",
    multimodal=True,
)


register_model_group(
    models={
        "Qwen-1.8B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B",
        },
        "Qwen-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B",
        },
        "Qwen-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B",
        },
        "Qwen-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B",
        },
        "Qwen-1.8B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B-Chat",
        },
        "Qwen-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B-Chat",
        },
        "Qwen-14B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B-Chat",
        },
        "Qwen-72B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B-Chat",
        },
        "Qwen-1.8B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B-Chat-Int8",
        },
        "Qwen-1.8B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B-Chat-Int4",
        },
        "Qwen-7B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B-Chat-Int8",
        },
        "Qwen-7B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B-Chat-Int4",
        },
        "Qwen-14B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B-Chat-Int8",
        },
        "Qwen-14B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B-Chat-Int4",
        },
        "Qwen-72B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B-Chat-Int8",
        },
        "Qwen-72B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B-Chat-Int4",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen1.5-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B",
        },
        "Qwen1.5-1.8B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B",
        },
        "Qwen1.5-4B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B",
        },
        "Qwen1.5-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B",
        },
        "Qwen1.5-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B",
        },
        "Qwen1.5-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-32B",
        },
        "Qwen1.5-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B",
        },
        "Qwen1.5-110B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-110B",
        },
        "Qwen1.5-MoE-A2.7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-MoE-A2.7B",
        },
        "Qwen1.5-0.5B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B-Chat",
        },
        "Qwen1.5-1.8B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B-Chat",
        },
        "Qwen1.5-4B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B-Chat",
        },
        "Qwen1.5-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B-Chat",
        },
        "Qwen1.5-14B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B-Chat",
        },
        "Qwen1.5-32B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-32B-Chat",
        },
        "Qwen1.5-72B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B-Chat",
        },
        "Qwen1.5-110B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-110B-Chat",
        },
        "Qwen1.5-MoE-A2.7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        },
        "Qwen1.5-0.5B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-0.5B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B-Chat-AWQ",
        },
        "Qwen1.5-1.8B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-1.8B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B-Chat-AWQ",
        },
        "Qwen1.5-4B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-4B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B-Chat-AWQ",
        },
        "Qwen1.5-7B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-7B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B-Chat-AWQ",
        },
        "Qwen1.5-14B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-14B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B-Chat-AWQ",
        },
        "Qwen1.5-32B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-32B-Chat-AWQ",
        },
        "Qwen1.5-72B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-72B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B-Chat-AWQ",
        },
        "Qwen1.5-110B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-110B-Chat-AWQ",
        },
        "Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        },
        "CodeQwen1.5-7B": {
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B",
            DownloadSource.MODELSCOPE: "Qwen/CodeQwen1.5-7B",
        },
        "CodeQwen1.5-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/CodeQwen1.5-7B-Chat",
        },
        "CodeQwen1.5-7B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/CodeQwen1.5-7B-Chat-AWQ",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen2-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B",
        },
        "Qwen2-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B",
        },
        "Qwen2-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B",
        },
        "Qwen2-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B",
        },
        "Qwen2-MoE-57B-A14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B",
        },
        "Qwen2-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-0.5B-Instruct",
        },
        "Qwen2-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-1.5B-Instruct",
        },
        "Qwen2-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-7B-Instruct",
        },
        "Qwen2-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct",
        },
        "Qwen2-MoE-57B-A14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B-Instruct",
        },
        "Qwen2-0.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2-0.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2-0.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct-AWQ",
        },
        "Qwen2-1.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2-1.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2-1.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct-AWQ",
        },
        "Qwen2-7B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2-7B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct-AWQ",
        },
        "Qwen2-72B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2-72B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct-AWQ",
        },
        "Qwen2-57B-A14B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
        },
        "Qwen2-Math-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-1.5B",
        },
        "Qwen2-Math-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-7B",
        },
        "Qwen2-Math-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-72B",
        },
        "Qwen2-Math-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-1.5B-Instruct",
        },
        "Qwen2-Math-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-7B-Instruct",
        },
        "Qwen2-Math-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-72B-Instruct",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen2.5-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B",
        },
        "Qwen2.5-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B",
        },
        "Qwen2.5-3B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B",
        },
        "Qwen2.5-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B",
        },
        "Qwen2.5-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B",
        },
        "Qwen2.5-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B",
        },
        "Qwen2.5-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B",
        },
        "Qwen2.5-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "Qwen2.5-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct",
        },
        "Qwen2.5-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct",
        },
        "Qwen2.5-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct",
        },
        "Qwen2.5-14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct",
        },
        "Qwen2.5-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct",
        },
        "Qwen2.5-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct",
        },
        "Qwen2.5-7B-Instruct-1M": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-1M",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-1M",
        },
        "Qwen2.5-14B-Instruct-1M": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-1M",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-1M",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-0.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-1.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-3B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-AWQ",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-AWQ",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-14B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-AWQ",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-32B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-AWQ",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-AWQ",
        },
        "Qwen2.5-Coder-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B",
        },
        "Qwen2.5-Coder-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B",
        },
        "Qwen2.5-Coder-3B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B",
        },
        "Qwen2.5-Coder-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B",
        },
        "Qwen2.5-Coder-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B",
        },
        "Qwen2.5-Coder-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B",
        },
        "Qwen2.5-Coder-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        },
        "Qwen2.5-Coder-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Coder-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "Qwen2.5-Coder-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Coder-14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B-Instruct",
        },
        "Qwen2.5-Coder-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B-Instruct",
        },
        "Qwen2.5-Math-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-1.5B",
        },
        "Qwen2.5-Math-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-7B",
        },
        "Qwen2.5-Math-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-72B",
        },
        "Qwen2.5-Math-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Math-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Math-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-72B-Instruct",
        },
        "QwQ-32B-Preview-Instruct": {
            DownloadSource.DEFAULT: "Qwen/QwQ-32B-Preview",
            DownloadSource.MODELSCOPE: "Qwen/QwQ-32B-Preview",
        },
        "QwQ-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/QwQ-32B",
            DownloadSource.MODELSCOPE: "Qwen/QwQ-32B",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen3-0.6B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-0.6B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B-Base",
        },
        "Qwen3-1.7B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-1.7B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B-Base",
        },
        "Qwen3-4B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Base",
        },
        "Qwen3-8B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-8B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B-Base",
        },
        "Qwen3-14B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-14B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B-Base",
        },
        "Qwen3-30B-A3B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Base",
        },
        "Qwen3-0.6B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-0.6B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B",
        },
        "Qwen3-1.7B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-1.7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B",
        },
        "Qwen3-4B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B",
        },
        "Qwen3-4B-Thinking-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B-Thinking-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Thinking-2507",
        },
        "Qwen3-8B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-8B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B",
        },
        "Qwen3-14B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B",
        },
        "Qwen3-32B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-32B",
        },
        "Qwen3-30B-A3B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B",
        },
        "Qwen3-30B-A3B-Thinking-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-Thinking-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Thinking-2507",
        },
        "Qwen3-235B-A22B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B",
        },
        "Qwen3-235B-A22B-Thinking-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B-Thinking-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Thinking-2507",
        },
        "Qwen3-0.6B-Thinking-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-0.6B-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B-GPTQ-Int8",
        },
        "Qwen3-1.7B-Thinking-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-1.7B-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B-GPTQ-Int8",
        },
        "Qwen3-4B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-AWQ",
        },
        "Qwen3-8B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-8B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B-AWQ",
        },
        "Qwen3-14B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-14B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B-AWQ",
        },
        "Qwen3-32B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-32B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-32B-AWQ",
        },
        "Qwen3-30B-A3B-Thinking-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        },
        "Qwen3-235B-A22B-Thinking-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
        },
        "Qwen/Qwen3-Next-80B-A3B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-Next-80B-A3B-Thinking",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-Next-80B-A3B-Thinking",
        },
    },
    template="qwen3",
)


register_model_group(
    models={
        "Qwen3-4B-Instruct-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B-Instruct-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Instruct-2507",
        },
        "Qwen3-30B-A3B-Instruct-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-Instruct-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Instruct-2507",
        },
        "Qwen3-235B-A22B-Instruct-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B-Instruct-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Instruct-2507",
        },
        "Qwen3-Next-80B-A3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-Next-80B-A3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-Next-80B-A3B-Instruct",
        },
    },
    template="qwen3_nothink",
)

register_model_group(
    models={
        "Qwen2-VL-2B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B",
        },
        "Qwen2-VL-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B",
        },
        "Qwen2-VL-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B",
        },
        "Qwen2-VL-2B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-VL-2B-Instruct",
        },
        "Qwen2-VL-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-VL-7B-Instruct",
        },
        "Qwen2-VL-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct",
        },
        "Qwen2-VL-2B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-2B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-2B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct-AWQ",
        },
        "Qwen2-VL-7B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-7B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        },
        "Qwen2-VL-72B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-72B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct-AWQ",
        },
        "QVQ-72B-Preview": {
            DownloadSource.DEFAULT: "Qwen/QVQ-72B-Preview",
            DownloadSource.MODELSCOPE: "Qwen/QVQ-72B-Preview",
        },
        "Qwen2.5-VL-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-3B-Instruct",
        },
        "Qwen2.5-VL-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "Qwen2.5-VL-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-32B-Instruct",
        },
        "Qwen2.5-VL-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-72B-Instruct",
        },
        "Qwen2.5-VL-3B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        },
        "Qwen2.5-VL-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        },
        "Qwen2.5-VL-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        },
    },
    template="qwen2_vl",
    multimodal=True,
)


register_model_group(
    models={
        "Qwen3-VL-4B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-4B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-4B-Instruct",
        },
        "Qwen3-VL-8B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-8B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-8B-Instruct",
        },
        "Qwen3-VL-30B-A3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-30B-A3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-30B-A3B-Instruct",
        },
        "Qwen3-VL-235B-A22B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-235B-A22B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-235B-A22B-Instruct",
        },
    },
    template="qwen3_vl_nothink",
    multimodal=True,
)


register_model_group(
    models={
        "Qwen3-VL-4B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-4B-Thinking",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-4B-Thinking",
        },
        "Qwen3-VL-8B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-8B-Thinking",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-8B-Thinking",
        },
        "Qwen3-VL-30B-A3B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-30B-A3B-Thinking",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-30B-A3B-Thinking",
        },
        "Qwen3-VL-235B-A22B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-VL-235B-A22B-Thinking",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-VL-235B-A22B-Thinking",
        },
    },
    template="qwen3_vl",
    multimodal=True,
)
