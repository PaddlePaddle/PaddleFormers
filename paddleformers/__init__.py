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
from paddleformers.utils.log import logger

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


def compare_version(v1, v2):
    for a, b in zip(v1.split("."), v2.split(".")):
        if a.isnumeric() and b.isnumeric():
            if a != b:
                return 1 if int(a) > int(b) else -1
        else:
            return 1 if a.isnumeric() else -1
    return 0


def _check_dependency_versions():
    for pkg_names, min_version in [(["paddlepaddle-gpu", "paddlepaddle"], "3.3"), (["paddlefleet"], "0.1")]:
        for pkg_name in pkg_names:
            try:
                _version = metadata.version(pkg_name)
                if compare_version(_version, min_version) < 0:
                    logger.warning(
                        "Version check warning:\n" + f"{pkg_name} version {version}, recommended >= {min_version}"
                    )
            except:
                pass


_check_dependency_versions()


with suppress(Exception):
    import paddle

    from .utils.paddle_patch import *

    paddle.disable_signal_handler()

# this version is used for develop and test.
# release version will be added fixed version by setup.py.
__version__ = "1.0.1.post"
if os.getenv(PADDLEFORMERS_STABLE_VERSION):
    __version__ = __version__.replace(".post", "")
else:
    formatted_date = datetime.now().date().strftime("%Y%m%d")
    __version__ = __version__.replace(".post", ".post{}".format(formatted_date))

# the next line will be replaced by setup.py for release version.
# [VERSION_INFO]

import os

PADDLEFORMERS_TESTING = os.environ.get("PADDLEFORMERS_TESTING", False)
if "torch" not in sys.modules and not PADDLEFORMERS_TESTING:
    sys.modules["torch"] = None
    sys.modules["torchvision"] = None
    import transformers  # qa

    del sys.modules["torch"]
else:
    import transformers  # qa

logger.warning(
    """Due to potential compatibility issues between PaddlePaddle and PyTorch in PaddleFormers, PaddleFormers defaults `transformers.utils.import_utils.is_torch_available` and `transformers.utils.import_utils.is_torchvision_available` to False. If you need to use PyTorch in transformers or torchvision, please add `del sys.modules['transformers']` before using them."""
)

if "datasets" in sys.modules.keys():

    logger.warning(
        "Detected that datasets module was imported before paddleformers. "
        "This may cause PaddleFormers datasets to be unavailable in intranet. "
        "Please import paddleformers before datasets module to avoid download issues"
    )

# module index
modules = [
    "cli",
    "data",
    "datasets",
    "generation",
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
    from . import datasets  # noqa
    from . import transformers  # noqa
    from . import (
        cli,
        data,
        generation,
        mergekit,
        nn,
        ops,
        peft,
        quantization,
        trainer,
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
