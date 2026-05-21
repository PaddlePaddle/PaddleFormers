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

"""Streaming dataset wrapper: adapts HF IterableDataset to paddle.io.IterableDataset."""

import paddle


class StreamingDataset(paddle.io.IterableDataset):
    """Wraps a HuggingFace IterableDataset as a paddle.io.IterableDataset.

    This allows Trainer to recognize the dataset as iterable and use the
    appropriate DataLoader path (no sampler, direct iteration).

    Args:
        hf_iterable: A HuggingFace IterableDataset instance (e.g. from
            load_dataset(..., streaming=True) with .map()/.filter() applied).
    """

    def __init__(self, hf_iterable):
        super().__init__()
        self._hf = hf_iterable

    def __iter__(self):
        for item in self._hf:
            yield item
