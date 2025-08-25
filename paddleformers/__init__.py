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

import os
import sys
from datetime import datetime

PADDLEFORMERS_STABLE_VERSION = "PADDLEFORMERS_STABLE_VERSION"

# this version is used for develop and test.
# release version will be added fixed version by setup.py.
__version__ = "0.1.3.post"
if os.getenv(PADDLEFORMERS_STABLE_VERSION):
    __version__ = __version__.replace(".post", "")
else:
    formatted_date = datetime.now().date().strftime("%Y%m%d")
    __version__ = __version__.replace(".post", ".post{}".format(formatted_date))

# the next line will be replaced by setup.py for release version.
# [VERSION_INFO]

if "datasets" in sys.modules.keys():
    from paddleformers.utils.log import logger

    logger.warning(
        "Detected that datasets module was imported before paddleformers. "
        "This may cause PaddleFormers datasets to be unavailable in intranet. "
        "Please import paddleformers before datasets module to avoid download issues"
    )
import paddle

from . import (
    data,
    datasets,
    mergekit,
    ops,
    peft,
    quantization,
    trainer,
    transformers,
    trl,
    utils,
    version,
)

paddle.disable_signal_handler()
