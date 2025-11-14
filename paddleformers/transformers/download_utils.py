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

from ..utils.download import DownloadSource, register_model_group

# qwen2
register_model_group(
    models={
        "Qwen2-0.5B": {
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-0.5B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B",
        },
        "Qwen2-1.5B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-1.5B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-1.5B",
        },
        "Qwen2-7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-7B",
        },
        "Qwen2-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-72B",
        },
        "Qwen2-0.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-0.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-0.5B-Instruct",
        },
        "Qwen2-1.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-1.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-1.5B-Instruct",
        },
        "Qwen2-7B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-7B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-7B-Instruct",
        },
        "Qwen2-72B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-72B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-72B-Instruct",
        },
        "Qwen2-Math-1.5B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-1.5B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-Math-1.5B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-Math-1.5B",
        },
        "Qwen2-Math-7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-7B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-Math-7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-Math-7B",
        },
        "Qwen2-Math-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-72B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-Math-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-Math-72B",
        },
        "Qwen2-Math-1.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-1.5B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-Math-1.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-Math-1.5B-Instruct",
        },
        "Qwen2-Math-7B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-7B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-Math-7B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-Math-7B-Instruct",
        },
        "Qwen2-Math-72B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-72B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-Math-72B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-Math-72B-Instruct",
        },
        "Qwen2-Math-RM-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-RM-72B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-Math-RM-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-Math-RM-72B",
        },
    }
)


# qwen2_5_vl
register_model_group(
    models={
        "Qwen2.5-VL-3B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-3B-Instruct",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-VL-3B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-VL-3B-Instruct",
        },
        "Qwen2.5-VL-7B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-7B-Instruct",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-VL-7B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "Qwen2.5-VL-32B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-32B-Instruct",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-VL-32B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-VL-32B-Instruct",
        },
        "Qwen2.5-VL-72B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-72B-Instruct",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-VL-72B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-VL-72B-Instruct",
        },
    }
)


# qwen2_moe
register_model_group(
    models={
        "Qwen2-MoE-57B-A14B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-57B-A14B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-57B-A14B",
        },
        "Qwen2-MoE-57B-A14B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2-57B-A14B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-57B-A14B-Instruct",
        },
    }
)


# qwen2_vl
register_model_group(
    models={
        "Qwen2-VL-2B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2-VL-2B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-VL-2B",
        },
        "Qwen2-VL-7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2-VL-7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-VL-7B",
        },
        "Qwen2-VL-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2-VL-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-VL-72B",
        },
        "Qwen2-VL-2B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2-VL-2B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-VL-2B-Instruct",
        },
        "Qwen2-VL-7B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2-VL-7B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-VL-7B-Instruct",
        },
        "Qwen2-VL-72B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct",
            # DownloadSource.AISTUDIO: "ModelHub/Qwen2-VL-72B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2-VL-72B-Instruct",
        },
    }
)


# qwen2.5
register_model_group(
    models={
        "Qwen2.5-0.5B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-0.5B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-0.5B",
        },
        "Qwen2.5-1.5B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-1.5B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-1.5B",
        },
        "Qwen2.5-3B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-3B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-3B",
        },
        "Qwen2.5-7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-7B",
        },
        "Qwen2.5-14B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-14B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-14B",
        },
        "Qwen2.5-32B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-32B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-32B",
        },
        "Qwen2.5-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-72B",
        },
        "Qwen2.5-0.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-0.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "Qwen2.5-1.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-1.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-1.5B-Instruct",
        },
        "Qwen2.5-3B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-3B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-3B-Instruct",
        },
        "Qwen2.5-7B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-7B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-7B-Instruct",
        },
        "Qwen2.5-14B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-14B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-14B-Instruct",
        },
        "Qwen2.5-32B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-32B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-32B-Instruct",
        },
        "Qwen2.5-72B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-72B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-72B-Instruct",
        },
        "Qwen2.5-7B-Instruct-1M": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-1M",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-7B-Instruct-1M",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-7B-Instruct-1M",
        },
        "Qwen2.5-14B-Instruct-1M": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-1M",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-14B-Instruct-1M",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-14B-Instruct-1M",
        },
        "Qwen2.5-Coder-0.5B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-0.5B",
        },
        "Qwen2.5-Coder-1.5B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-1.5B",
        },
        "Qwen2.5-Coder-3B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-3B",
        },
        "Qwen2.5-Coder-7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-7B",
        },
        "Qwen2.5-Coder-14B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-14B",
        },
        "Qwen2.5-Coder-32B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-32B",
        },
        "Qwen2.5-Coder-0.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        },
        "Qwen2.5-Coder-1.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Coder-3B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "Qwen2.5-Coder-7B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Coder-14B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-14B-Instruct",
        },
        "Qwen2.5-Coder-32B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Coder-32B-Instruct",
        },
        "Qwen2.5-Math-1.5B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-1.5B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-1.5B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-1.5B",
        },
        "Qwen2.5-Math-7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-7B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-7B",
        },
        "Qwen2.5-Math-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-72B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-72B",
        },
        "Qwen2.5-Math-1.5B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-1.5B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-1.5B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-1.5B-Instruct",
        },
        "Qwen2.5-Math-7B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-7B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-7B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-7B-Instruct",
        },
        "Qwen2.5-Math-72B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-72B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-72B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-72B-Instruct",
        },
        "Qwen2.5-Math-RM-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-RM-72B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-RM-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-RM-72B",
        },
        "Qwen2.5-Math-PRM-7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-PRM-7B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-PRM-7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-PRM-7B",
        },
        "Qwen2.5-Math-PRM-72B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-PRM-72B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen2.5-Math-PRM-72B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen2.5-Math-PRM-72B",
        },
        "QwQ-32B-Preview-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/QwQ-32B-Preview",
            DownloadSource.HUGGINGFACE: "Qwen/QwQ-32B-Preview",
        },
        "QwQ-32B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/QwQ-32B",
            DownloadSource.HUGGINGFACE: "Qwen/QwQ-32B",
        },
    }
)


# qwen3
register_model_group(
    models={
        "Qwen3-0.6B-Base": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B-Base",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-0.6B-Base",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-0.6B-Base",
        },
        "Qwen3-1.7B-Base": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B-Base",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-1.7B-Base",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-1.7B-Base",
        },
        "Qwen3-4B-Base": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Base",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-4B-Base",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-4B-Base",
        },
        "Qwen3-8B-Base": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B-Base",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-8B-Base",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-8B-Base",
        },
        "Qwen3-14B-Base": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B-Base",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-14B-Base",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-14B-Base",
        },
        "Qwen3-0.6B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-0.6B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-0.6B",
        },
        "Qwen3-1.7B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-1.7B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-1.7B",
        },
        "Qwen3-4B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-4B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-4B",
        },
        "Qwen3-8B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-8B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-8B",
        },
        "Qwen3-14B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-14B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-14B",
        },
        "Qwen3-32B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-32B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-32B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-32B",
        },
        "Qwen3-4B-Instruct-2507": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Instruct-2507",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-4B-Instruct-2507",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-4B-Instruct-2507",
        },
        "Qwen3-4B-Thinking-2507": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Thinking-2507",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-4B-Thinking-2507",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-4B-Thinking-2507",
        },
        "Qwen3-0.6B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-0.6B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-0.6B-FP8",
        },
        "Qwen3-1.7B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-1.7B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-1.7B-FP8",
        },
        "Qwen3-4B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-4B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-4B-FP8",
        },
        "Qwen3-8B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-8B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-8B-FP8",
        },
        "Qwen3-14B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-14B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-14B-FP8",
        },
        "Qwen3-32B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-32B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-32B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-32B-FP8",
        },
        "Qwen3-4B-Instruct-2507-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Instruct-2507-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-4B-Instruct-2507-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-4B-Instruct-2507-FP8",
        },
        "Qwen3-4B-Thinking-2507-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Thinking-2507-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-4B-Thinking-2507-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-4B-Thinking-2507-FP8",
        },
    }
)


# qwen3_moe
register_model_group(
    models={
        "Qwen3-30B-A3B-Base": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Base",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-30B-A3B-Base",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-30B-A3B-Base",
        },
        "Qwen3-30B-A3B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-30B-A3B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-30B-A3B",
        },
        "Qwen3-235B-A22B": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-235B-A22B",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-235B-A22B",
        },
        "Qwen3-30B-A3B-Instruct-2507": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Instruct-2507",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-30B-A3B-Instruct-2507",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-30B-A3B-Instruct-2507",
        },
        "Qwen3-235B-A22B-Instruct-2507": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Instruct-2507",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-235B-A22B-Instruct-2507",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-235B-A22B-Instruct-2507",
        },
        "Qwen3-30B-A3B-Thinking-2507": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Thinking-2507",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-30B-A3B-Thinking-2507",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-30B-A3B-Thinking-2507",
        },
        "Qwen3-235B-A22B-Thinking-2507": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Thinking-2507",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-235B-A22B-Thinking-2507",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-235B-A22B-Thinking-2507",
        },
        "Qwen3-30B-A3B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-30B-A3B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-30B-A3B-FP8",
        },
        "Qwen3-235B-A22B-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-235B-A22B-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-235B-A22B-FP8",
        },
        "Qwen3-30B-A3B-Instruct-2507-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-30B-A3B-Instruct-2507-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        },
        "Qwen3-235B-A22B-Instruct-2507-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-235B-A22B-Instruct-2507-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        },
        "Qwen3-30B-A3B-Thinking-2507-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-30B-A3B-Thinking-2507-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        },
        "Qwen3-235B-A22B-Thinking-2507-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-235B-A22B-Thinking-2507-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        },
        "Qwen3-Coder-30B-A3B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-Coder-30B-A3B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        },
        "Qwen3-Coder-480B-A35B-Instruct": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-Coder-480B-A35B-Instruct",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        },
        "Qwen3-Coder-30B-A3B-Instruct-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-Coder-30B-A3B-Instruct-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        },
        "Qwen3-Coder-480B-A35B-Instruct-FP8": {
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            DownloadSource.AISTUDIO: "ModelHub/Qwen3-Coder-480B-A35B-Instruct-FP8",
            DownloadSource.HUGGINGFACE: "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        },
    }
)


# deepseek-v2
register_model_group(
    models={
        "DeepSeek-Math-7B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-base",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-math-7b-base",
        },
        "DeepSeek-Math-7B-Instruct": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-instruct",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-math-7b-instruct",
        },
        "DeepSeek-MoE-16B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-base",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-moe-16b-base",
        },
        "DeepSeek-MoE-16B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-chat",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-moe-16b-chat",
        },
        "DeepSeek-V2-16B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V2-Lite",
        },
        "DeepSeek-V2-236B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V2",
        },
        "DeepSeek-V2-16B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite-Chat",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V2-Lite-Chat",
        },
        "DeepSeek-V2-236B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Chat",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V2-Chat",
        },
        "DeepSeek-Coder-V2-16B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        },
        "DeepSeek-Coder-V2-236B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Base",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-Coder-V2-Base",
        },
        "DeepSeek-Coder-V2-16B-Instruct": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        },
        "DeepSeek-Coder-V2-236B-Instruct": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        },
        "DeepSeek-V2-0628-236B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Chat-0628",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V2-Chat-0628",
        },
        "DeepSeek-V2.5-236B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2.5",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V2.5",
        },
        "DeepSeek-V2.5-1210-236B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2.5-1210",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V2.5-1210",
        },
    }
)

register_model_group(
    models={
        "DeepSeek-Coder-6.7B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-base",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-coder-6.7b-base",
        },
        "DeepSeek-Coder-7B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-7b-base-v1.5",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-coder-7b-base-v1.5",
        },
        "DeepSeek-Coder-33B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-base",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-coder-33b-base",
        },
        "DeepSeek-Coder-6.7B-Instruct": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-instruct",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-coder-6.7b-instruct",
        },
        "DeepSeek-Coder-7B-Instruct": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        },
        "DeepSeek-Coder-33B-Instruct": {
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-instruct",
            DownloadSource.HUGGINGFACE: "deepseek-ai/deepseek-coder-33b-instruct",
        },
    }
)


# deepseek-v3
register_model_group(
    models={
        "DeepSeek-V3-671B-Base": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3-Base",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V3-Base",
        },
        "DeepSeek-V3-671B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V3",
        },
        "DeepSeek-V3-0324-671B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3-0324",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-V3-0324",
        },
    }
)


# deepseek-r1
register_model_group(
    models={
        "DeepSeek-R1-1.5B-Distill": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        },
        "DeepSeek-R1-7B-Distill": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        },
        "DeepSeek-R1-8B-Distill": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        },
        "DeepSeek-R1-14B-Distill": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        },
        "DeepSeek-R1-32B-Distill": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        },
        "DeepSeek-R1-70B-Distill": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        },
        "DeepSeek-R1-671B-Chat-Zero": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Zero",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-Zero",
        },
        "DeepSeek-R1-671B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1",
        },
        "DeepSeek-R1-0528-8B-Distill": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        },
        "DeepSeek-R1-0528-671B-Chat": {
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-0528",
            DownloadSource.HUGGINGFACE: "deepseek-ai/DeepSeek-R1-0528",
        },
    },
)


# llama
register_model_group(
    models={
        "Llama-7B": {
            DownloadSource.MODELSCOPE: "skyline2006/llama-7b",
            DownloadSource.AISTUDIO: "PaddleNLP/llama-7b",
            DownloadSource.HUGGINGFACE: "huggyllama/llama-7b",
        },
        "Llama-13B": {
            DownloadSource.MODELSCOPE: "skyline2006/llama-13b",
            DownloadSource.AISTUDIO: "PaddleNLP/llama-13b",
            DownloadSource.HUGGINGFACE: "huggyllama/llama-13b",
        },
        "Llama-30B": {
            DownloadSource.MODELSCOPE: "skyline2006/llama-30b",
            DownloadSource.AISTUDIO: "PaddleNLP/llama-30b",
            DownloadSource.HUGGINGFACE: "huggyllama/llama-30b",
        },
        "Llama-65B": {
            DownloadSource.MODELSCOPE: "skyline2006/llama-65b",
            DownloadSource.AISTUDIO: "PaddleNLP/llama-65b",
            DownloadSource.HUGGINGFACE: "huggyllama/llama-65b",
        },
    }
)


# llama2
register_model_group(
    models={
        "Llama-2-7B": {
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-ms",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-2-7b",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-2-7b-hf",
        },
        "Llama-2-13B": {
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-ms",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-2-13b",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-2-13b-hf",
        },
        "Llama-2-70B": {
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-ms",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-2-70b",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-2-70b-hf",
        },
        "Llama-2-7B-Chat": {
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-chat-ms",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-2-7b-chat",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-2-7b-chat-hf",
        },
        "Llama-2-13B-Chat": {
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-chat-ms",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-2-13b-chat",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-2-13b-chat-hf",
        },
        "Llama-2-70B-Chat": {
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-chat-ms",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-2-70b-chat",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-2-70b-chat-hf",
        },
    }
)


# llama3
register_model_group(
    models={
        "Llama-3-8B": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3-8B",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3-8B",
        },
        "Llama-3-70B": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3-70B",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3-70B",
        },
        "Llama-3-8B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3-8B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3-8B-Instruct",
        },
        "Llama-3-70B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3-70B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3-70B-Instruct",
        },
        "Llama-3-8B-Chinese-Chat": {
            DownloadSource.MODELSCOPE: "LLM-Research/Llama3-8B-Chinese-Chat",
            DownloadSource.HUGGINGFACE: "shenzhi-wang/Llama3-8B-Chinese-Chat",
        },
        "Llama-3-70B-Chinese-Chat": {
            DownloadSource.MODELSCOPE: "LLM-Research/Llama3-70B-Chinese-Chat",
            DownloadSource.HUGGINGFACE: "shenzhi-wang/Llama3-70B-Chinese-Chat",
        },
        "Llama-3.1-8B": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3.1-8B",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3.1-8B",
        },
        "Llama-3.1-70B": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3.1-70B",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3.1-70B",
        },
        "Llama-3.1-405B": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3.1-405B",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3.1-405B",
        },
        "Llama-3.1-8B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3.1-8B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        },
        "Llama-3.1-70B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3.1-70B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3.1-70B-Instruct",
        },
        "Llama-3.1-405B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Meta-Llama-3.1-405B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Meta-Llama-3.1-405B-Instruct",
        },
        "Llama-3.1-8B-Chinese-Chat": {
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-8B-Chinese-Chat",
            DownloadSource.HUGGINGFACE: "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
        },
        "Llama-3.1-70B-Chinese-Chat": {
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-70B-Chinese-Chat",
            DownloadSource.HUGGINGFACE: "shenzhi-wang/Llama3.1-70B-Chinese-Chat",
        },
        "Llama-3.2-1B": {
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-3.2-1B",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-3.2-1B",
        },
        "Llama-3.2-3B": {
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-3.2-3B",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-3.2-3B",
        },
        "Llama-3.2-1B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-3.2-1B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-3.2-1B-Instruct",
        },
        "Llama-3.2-3B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-3.2-3B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-3.2-3B-Instruct",
        },
        "Llama-3.3-70B-Instruct": {
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.3-70B-Instruct",
            DownloadSource.AISTUDIO: "PaddleNLP/Llama-3.3-70B-Instruct",
            DownloadSource.HUGGINGFACE: "meta-llama/Llama-3.3-70B-Instruct",
        },
    }
)

# ernie
register_model_group(
    models={
        "ERNIE-4.5-300B-A47B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-300B-A47B-Base-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-300B-A47B-Base-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-300B-A47B-Base-PT",
        },
        "ERNIE-4.5-300B-A47B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-300B-A47B-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-300B-A47B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-300B-A47B-PT",
        },
        "ERNIE-4.5-21B-A3B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-21B-A3B-Base-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-21B-A3B-Base-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-21B-A3B-Base-PT",
        },
        "ERNIE-4.5-21B-A3B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-21B-A3B-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-21B-A3B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-21B-A3B-PT",
        },
        "ERNIE-4.5-0.3B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-0.3B-Base-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-0.3B-Base-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-0.3B-Base-PT",
        },
        "ERNIE-4.5-0.3B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-0.3B-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-0.3B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-0.3B-PT",
        },
        "ERNIE-4.5-VL-424B-A47B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-PT",
        },
        "ERNIE-4.5-VL-424B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-424B-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-424B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-424B-PT",
        },
        "ERNIE-4.5-VL-28B-A3B-Base": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-28B-A3B-Base-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-PT",
        },
        "ERNIE-4.5-VL-28B-A3B": {
            DownloadSource.HUGGINGFACE: "baidu/ERNIE-4.5-VL-28B-A3B-PT",
            DownloadSource.AISTUDIO: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT",
            DownloadSource.MODELSCOPE: "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT",
        },
    }
)
