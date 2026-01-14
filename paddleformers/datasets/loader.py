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

from typing import Any, Dict

from .DPODataset import DPODataSet
from .SFTDataset import Sequence, SFTDataSet


def create_dataset(**dataset_config: Dict[str, Any]):
    """Create dataset based on configuration parameters.

    Args:
        dataset_config (dict): Configuration dictionary, required keys:
            - stage: 'dpo', 'sft', 'pt' (case-insensitive).
            - Other keys passed to dataset constructors.

    Returns:
        SequenceDataset: Configured sequence dataset
    """
    if dataset_config["stage"].lower() == "dpo":
        train_dataset = DPODataSet(**dataset_config)
    else:
        train_dataset = SFTDataSet(**dataset_config)

    return train_dataset


def create_indexed_dataset(data_file_prefix):
    """Create indexed dataset from raw data files.

    Args:
        data_file_prefix (str): Path prefix for raw data files

    Returns:
        IndexedDataset: Preprocessed dataset with memory-efficient indexing
    """
    from paddleformers.data.indexed_dataset import (
        make_sft_dataset as make_sft_indexed_dataset,
    )

    indexed_dataset = make_sft_indexed_dataset(
        path=data_file_prefix,
        dataclass=Sequence,
    )
    return indexed_dataset
