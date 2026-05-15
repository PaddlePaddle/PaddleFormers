# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from .loaders import load_dataset, load_datasets
from .ops import (
    concat_datasets,
    interleave,
    sample_dataset,
    shuffle_dataset,
    split_dataset,
)
from .preprocessors import (
    AlpacaPreprocessor,
    AutoPreprocessor,
    BasePreprocessor,
    MessagesPreprocessor,
    ResponsePreprocessor,
)

# Load built-in dataset registry on import
from .registry import (
    DatasetMeta,
    DatasetSpec,
    _load_builtin_registry,
    get_dataset_meta,
    list_datasets,
    parse_dataset_string,
    register_dataset,
    register_dataset_info,
)
from .schema import (
    PAIR_KEYS,
    ROLES,
    STANDARD_KEYS,
    GroundingObjects,
    ImageMedia,
    Message,
    StandardRow,
    cast_images,
    cast_media_list,
    check_messages,
    remove_non_standard_keys,
)

_load_builtin_registry()

from .datapipe import (
    EncodeConfig,
    EncodedSample,
    TemplateMeta,
    VLEncodedSample,
    collate_sft,
    collate_vl_sft,
    encode_multiturn,
    encode_pt,
    encode_sft,
    encode_vl_sft,
    get_template,
    greedy_pack,
    list_templates,
    register_template,
)
from .dataset import LazyEncodeDataset
