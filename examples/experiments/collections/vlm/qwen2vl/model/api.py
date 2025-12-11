# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from experiments.collections.vlm.qwen2vl.model import (
    Qwen2VLProvider2B,
    Qwen2VLProvider7B,
    Qwen2VLProvider72B,
    Qwen2VLModel,
    Qwen25VLProvider3B,
    Qwen25VLProvider7B,
    Qwen25VLProvider32B,
    Qwen25VLProvider72B,
)


def qwen2vl_2b():
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen2VLProvider2B(), model_version="qwen2-vl")


def qwen2vl_7b() :
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen2VLProvider7B(), model_version="qwen2-vl")


def qwen2vl_72b() :
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen2VLProvider72B(), model_version="qwen2-vl")


def qwen25vl_3b() :
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen25VLProvider3B(), model_version="qwen25-vl")


def qwen25vl_7b() :
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen25VLProvider7B(), model_version="qwen25-vl")


def qwen25vl_32b() :
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen25VLProvider32B(), model_version="qwen25-vl")


def qwen25vl_72b() :
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen25VLProvider72B(), model_version="qwen25-vl")


__all__ = ["qwen2vl_2b", "qwen2vl_7b", "qwen2vl_72b", "qwen25vl_3b", "qwen25vl_7b", "qwen25vl_32b", "qwen25vl_72b"]
