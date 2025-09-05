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
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "tokenizer_utils": [
        "PretrainedTokenizer",
        "BPETokenizer",
        "tokenize_chinese_chars",
        "is_chinese_char",
        "normalize_chars",
        "tokenize_special_chars",
        "convert_to_unicode",
    ],
    "tokenizer_utils_base": [
        "import_protobuf_decode_error",
        "ExplicitEnum",
        "PaddingStrategy",
        "TensorType",
        "to_py_obj",
        "_is_numpy",
        "TruncationStrategy",
        "CharSpan",
        "TokenSpan",
        "BatchEncoding",
        "SpecialTokensMixin",
        "PretrainedTokenizerBase",
        "EncodingFast",
    ],
}

if TYPE_CHECKING:
    from .tokenizer_utils import *
    from .tokenizer_utils_base import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
