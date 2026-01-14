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
    "collate": [
        "dpo_collate_fn",
        "collate_fn",
        "mm_collate_fn",
        "pad_batch_data",
        "gen_self_attn_mask",
        "gen_attn_mask_startend_row_indices",
    ],
    "data_utils": [
        "convert_to_tokens_for_pt",
        "convert_to_tokens_for_sft",
        "convert_to_input_ids",
        "function_call_chat_template",
        "postprocess_fc_sequence",
        "estimate_training",
        "get_worker_sliced_iterator",
        "print_debug_info",
        "round_up_to_multiple_of_8",
    ],
    "loader": [
        "create_dataset",
        "create_indexed_dataset",
    ],
    "DPODataset": ["DPODataSet"],
    "SFTDataset": ["SFTDataSet"],
    "reader.convertor": [
        "convert_dpo_txt_data",
        "convert_txt_data",
        "convert_mm_data",
        "convert_pretraining_data",
        "erniekit_convertor",
        "messages_convertor",
    ],
    "reader.download_manager": ["HuggingFaceDownload"],
    "reader.file_reader": ["BaseReader", "FileReader", "FileListReader", "get_hf_dataset_config", "HuggingFaceReader"],
    "reader.io": ["load_json", "load_txt", "load_parquet", "load_csv"],
    "reader.mix_datasets": [
        "BaseMixDataset",
        "RandomDataset",
        "ConcatDataset",
        "InterLeaveDataset",
        "create_dataset_instance",
    ],
    "reader.multi_source_datasets": ["InfiniteDataset", "MultiSourceDataset"],
    "template.formatter": [
        "Formatter",
        "EmptyFormatter",
        "StringFormatter",
        "FunctionFormatter",
        "ToolFormatter",
    ],
    "template.grounding_plugin": [
        "BaseGroundingPlugin",
        "register_grounding_plugin",
        "get_grounding_plugin",
    ],
    "template.mm_plugin": [
        "_make_batched_images",
        "_check_video_is_nested_images",
        "MMPluginMixin",
        "BasePlugin",
        "ErnieVLPlugin",
        "PaddleOCRVLPlugin",
        "Qwen2VLPlugin",
        "Qwen3VLPlugin",
        "GLM4VPlugin",
        "Gemma3Plugin",
        "register_mm_plugin",
        "get_mm_plugin",
    ],
    "template.template": [
        "Role",
        "Template",
        "ReasoningTemplate",
        "Llama2Template",
        "register_template",
        "parse_template",
        "get_template_and_fix_tokenizer",
    ],
    "template.tool_utils": [
        "FunctionCall",
        "ToolUtils",
        "DefaultToolUtils",
        "QwenToolUtils",
        "GLM4ToolUtils",
        "GLM4MOEToolUtils",
        "Llama3ToolUtils",
        "ERNIEToolUtils",
        "get_tool_utils",
    ],
    "template.augment_utils": [
        "RandomApply",
        "RandomDiscreteRotation",
        "JpegCompression",
        "RandomScale",
        "RandomSingleSidePadding",
    ],
}

if TYPE_CHECKING:
    from .collate import *
    from .data_utils import *
    from .dataset import *
    from .DPODataset import *
    from .loader import *
    from .reader import *
    from .rlhf_datasets import *
    from .sampler import *
    from .SFTDataset import *
    from .template import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
