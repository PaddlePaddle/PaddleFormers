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

from .convertor import erniekit_convertor
from .file_reader import BaseReader, FileListReader, FileReader, HuggingFaceReader
from .mix_datasets import create_dataset_instance
from .multi_source_datasets import MultiSourceDataset

# def _get_dataset_processor(
#     data_args: "DataArguments"
# ) -> "DatasetProcessor":
#     r"""Return the corresponding dataset processor."""
#     if stage == "pt":
#         dataset_processor_class = PretrainDatasetProcessor
#     elif stage == "sft" and not do_generate:


#     elif stage == "rm":
#         dataset_processor_class = PairwiseDatasetProcessor
#     elif stage == "kto":
#         dataset_processor_class = FeedbackDatasetProcessor
#     else:
#         dataset_processor_class = UnsupervisedDatasetProcessor

#     return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)

__all__ = [
    BaseReader,
    FileReader,
    FileListReader,
    HuggingFaceReader,
    MultiSourceDataset,
    create_dataset_instance,
    erniekit_convertor,
]
