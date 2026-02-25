# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
    "blendable_dataset": [
        "BlendableDataset",
    ],
    "causal_dataset": [
        "check_data_split",
        "get_datasets_weights_and_num_samples",
        "build_train_valid_test_datasets",
        "GPTDataset",
    ],
    "sampler": ["SamplerHelper"],
    "data_collator": [
        "DataCollatorForSeq2Seq",
        "default_data_collator",
        "DataCollator",
        "DataCollatorWithPadding",
        "InputDataClass",
        "DataCollatorMixin",
        "paddle_default_data_collator",
        "numpy_default_data_collator",
        "DefaultDataCollator",
        "DataCollatorForTokenClassification",
        "DataCollatorForEmbedding",
        "_paddle_collate_batch",
        "_numpy_collate_batch",
        "tolist",
        "DataCollatorForLanguageModeling",
    ],
    "dist_dataloader": ["DummyDataset", "IterableDummyDataset", "DistDataLoader", "init_dataloader_comm_group"],
    "collate": ["Dict", "Pad", "Stack", "Tuple"],
    "vocab": ["Vocab"],
    "tokenizer": ["BaseTokenizer"],
    "indexed_dataset": [
        "print_rank_0",
        "get_available_dataset_impl",
        "make_dataset",
        "make_sft_dataset",
        "dataset_exists",
        "read_longs",
        "write_longs",
        "read_shorts",
        "write_shorts",
        "dtypes",
        "code",
        "index_file_path",
        "sft_index_file_path",
        "sft_data_file_path",
        "data_file_path",
        "loss_mask_file_path",
        "create_doc_idx",
        "IndexedDataset",
        "IndexedDatasetBuilder",
        "_warmup_mmap_file",
        "MMapIndexedDataset",
        "SFTMMapIndexedDataset",
        "make_builder",
        "SFTMMapIndexedDatasetBuilder",
        "MMapIndexedDatasetBuilder",
        "get_indexed_dataset_",
        "CompatibleIndexedDataset",
    ],
}


if TYPE_CHECKING:
    from .collate import *
    from .data_collator import *
    from .dist_dataloader import *
    from .sampler import *
    from .vocab import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
