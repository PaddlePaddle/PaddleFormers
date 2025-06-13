# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib

import paddle

from .batch_sampler import *
from .env import CONFIG_NAME, GENERATION_CONFIG_NAME, LEGACY_CONFIG_NAME
from .import_utils import *
from .infohub import infohub
from .initializer import to
from .log import logger
from .memory_utils import empty_device_cache

try:
    from .optimizer import *
except:
    logger.info("Not support custom optimizer")

from .paddle_patch import *
from .serialization import load_torch

# hack impl for EagerParamBase to function
# https://github.com/PaddlePaddle/Paddle/blob/fa44ea5cf2988cd28605aedfb5f2002a63018df7/python/paddle/nn/layer/layers.py#L2077
paddle.framework.io.EagerParamBase.to = to


@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)
