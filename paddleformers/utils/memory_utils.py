# coding:utf-8
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from .log import logger
from .tools import get_env_device

__all__ = [
    "empty_device_cache",
]


def empty_device_cache():
    device = get_env_device()
    if device == "gpu":
        paddle.device.cuda.empty_cache()
    elif device == "xpu":
        paddle.device.xpu.empty_cache()
    else:
        if not getattr(empty_device_cache, "has_warned", False):
            logger.warning(
                "The current device ({}) does not support empty cache, calling empty_device_cache() will have no effect.".format(
                    device
                )
            )
            setattr(empty_device_cache, "has_warned", True)
