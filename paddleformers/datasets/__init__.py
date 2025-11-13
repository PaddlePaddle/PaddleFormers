# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ..utils.lazy_import import _LazyModule

import_structure = {
    "dataset": [
        "load_from_ppnlp",
        "DatasetTuple",
        "import_main_class",
        "load_from_hf",
        "load_dataset",
        "MapDataset",
        "IterDataset",
        "DatasetBuilder",
        "SimpleBuilder",
    ],
    "base": [
        "open_file",
        "FileDataset",
        "FileListDataset",
        "MultiSourceDataset",
        "InfiniteDataset",
    ],
    "data_utils": [
        "pad_batch_data",
        "convert_to_tokens_for_pt",
        "convert_to_tokens_for_sft",
        "convert_to_input_ids",
        "function_call_chat_template",
        "postprocess_fc_sequence",
        "estimate_training",
        "round_up_to_multiple_of_8",
    ],
    "dpo": [
        "create_dataset",
        "process_fc",
        "collate_fn",
        "process_session_example",
        "SequenceDataset",
    ],
    "finetuning": [
        "create_dataset",
        "create_indexed_dataset",
        "collate_fn",
        "process_fc",
        "process_example",
        "process_pretraining_example",
        "SequenceDataset",
        "gen_self_attn_mask",
        "gen_attn_mask_startend_row_indices",
    ],
    "mix_datasets": [
        "BaseMixDataset",
        "RandomDataset",
        "ConcatDataset",
        "InterLeaveDataset",
        "create_dataset_instance",
    ],
}

if TYPE_CHECKING:
    from .base import *
    from .data_utils import *
    from .dataset import *
    from .dpo import *
    from .finetuning import *
    from .mix_datasets import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
