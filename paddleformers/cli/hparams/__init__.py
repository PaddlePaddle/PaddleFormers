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

import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "data_args": ["DataArguments"],
    "export_args": ["ExportArguments"],
    "finetuning_args": ["FinetuningArguments"],
    "generating_args": ["GeneratingArguments"],
    "model_args": ["ModelArguments"],
    "parser": ["get_eval_args", "get_train_args", "get_server_args", "get_export_args", "read_args"],
    "server_args": ["ServerArguments"],
}

if TYPE_CHECKING:
    from .data_args import *
    from .export_args import *
    from .finetuning_args import *
    from .generating_args import *
    from .model_args import *
    from .parser import *
    from .preprocess_args import *
    from .server_args import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
