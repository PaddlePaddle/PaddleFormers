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

from paddleformers.datasets.SFTDataset import IteratorSFTDataset, Sequence


def test_non_packing_iterator_does_not_repeat_last_sequence():
    sequences = [
        Sequence(token_ids=[1, 2], position_ids=[0, 1], labels=[-100, 2], num_examples=1),
        Sequence(token_ids=[3, 4], position_ids=[0, 1], labels=[-100, 4], num_examples=1),
    ]
    dataset = IteratorSFTDataset.__new__(IteratorSFTDataset)
    dataset.mix_datasets = sequences
    dataset.is_pretraining = False
    dataset.packing = False
    dataset.estimate = False
    dataset.dataset_num_proc = 1
    dataset.mem_debug = False
    dataset.iter_all_examples = False
    dataset._current_processor_func = lambda sequence, _: sequence

    batches = list(dataset._generate_sequences())

    assert [[sequence.token_ids for sequence in batch] for batch in batches] == [[[1, 2]], [[3, 4]]]
    assert dataset.iter_all_examples is True
