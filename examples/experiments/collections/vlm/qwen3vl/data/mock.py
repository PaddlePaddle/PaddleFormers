# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from PIL import Image
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Qwen3VLMockDataModule(pl.LightningDataModule):
    """
    A mock data module for Qwen3VL training, validation and testing.
    """
    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer = None,
        image_processor = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: list[int] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False
    ):
        super().__init__()
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        assert tokenizer is not None and image_processor is not None, \
            "please assign tokenizer and image processor"
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def setup(self, stage: str = "") -> None:
        self._train_ds = _Qwen3VLMockDataset(
            self.tokenizer, self.image_processor, "train", self.num_train_samples, self.seq_length
        )
        self._validation_ds = _Qwen3VLMockDataset(
            self.tokenizer, self.image_processor, "valid", self.num_val_samples, self.seq_length
        )
        self._test_ds = _Qwen3VLMockDataset(
            self.tokenizer, self.image_processor, "test", self.num_test_samples, self.seq_length
        )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)
    
    def _create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs
        )

def prepare_image_inputs(num_channels: np.uint8 = 3, width=1024, height=1024):
    image_inputs = [np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8)]
    image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
    return image_inputs


class _Qwen3VLMockDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        image_processor,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.seq_length = seq_length
        
        self.vocab_size = tokenizer.vocab_size
        
        self.image_processor = image_processor
        self.image_width, self.image_height = np.random.choice(np.arange(56, 1024), 2)
        
        self.length = num_samples
        self.seed = seed
        
        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)
        self.image_token = "<|image_pad|>"
        self.spatial_merge_size = 2
    
    def __len__(self):
        return self.length
    
    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=self.seed + idx)
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        # Processor image input
        image_inputs = prepare_image_inputs(3, self.image_width, self.image_height)
        process_out = self.image_processor(image_inputs, return_tensors="pt")