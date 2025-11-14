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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .base_processor import DatasetProcessor
from paddle.io import IterableDataset


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def encode_example(self, example: dict) -> dict:
        messages = example.get("messages", [])
        images = example.get("images", [])
        videos = example.get("videos", [])

        image_inputs, video_inputs = self.vision_loader(images=images, videos=videos)
        model_input = self.encoder(messages=messages, image_inputs=image_inputs, video_inputs=video_inputs, processor=self.processor)

        return model_input

    def preprocess_dataset(self, example) -> list[dict]:
        return self.encode_example(example)

    def print_data_example(self, example: list[dict]) -> None:
        print("Example:", example)

@dataclass
class ProcessDataset(IterableDataset):

    def __init__(self, mix_datasets, dataset_processor):
        self.mix_datasets = mix_datasets
        self.processor = dataset_processor

    def __iter__(self):
        for item in self.mix_datasets:
            res = self.processor.preprocess_dataset(item)
            yield res

    def __len__(self):
        return self.mix_datasets.__len__()