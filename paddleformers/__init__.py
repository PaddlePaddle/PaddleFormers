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
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING

from .utils.lazy_import import _LazyModule

PADDLEFORMERS_STABLE_VERSION = "PADDLEFORMERS_STABLE_VERSION"

with suppress(Exception):
    import paddle

    from .utils.paddle_patch import *

    paddle.disable_signal_handler()

# this version is used for develop and test.
# release version will be added fixed version by setup.py.
__version__ = "0.2.4.post"
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

# module index
modules = [
    "data",
    "datasets",
    "nn",
    "mergekit",
    "ops",
    "peft",
    "quantization",
    "trainer",
    "trl",
    "utils",
    "version",
    "transformers",
]
import_structure = {module: [] for module in modules}
import_structure["transformers.tokenizer_utils"] = ["PreTrainedTokenizer"]

if TYPE_CHECKING:
    from . import (
        data,
        datasets,
        mergekit,
        nn,
        ops,
        peft,
        quantization,
        trainer,
        transformers,
        trl,
        utils,
        version,
    )
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
