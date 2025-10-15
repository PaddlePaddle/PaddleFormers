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


import numpy as np

from paddle.io import IterableDataset

class PackingDataset(IterableDataset):

    def __init__(self, processed_dataset, **dataset_config):
        self.processed_dataset = processed_dataset
        self.packing = dataset_config['packing']
        self.greedy_intokens = dataset_config['greedy_intokens']

        self.estimate = False
        # The number of valid samples and skipped samples in estimation
        self.unused_samples = 0
        self.used_samples = 0
        # If used_estimate_samples exceeds max_estimate_samples,stop estimating.
        self.used_estimate_samples = 0
        self.max_estimate_samples = 0
        # set max estimate samples
        # if not self.is_valid:
        #     self.max_estimate_samples = len(self.mix_datasets)

        self.max_seq_len = dataset_config["max_seq_len"]
    
    def __iter__(self):

        dataset_iterator = iter(self.processed_dataset)

        if not self.packing:
            for _ in range(len(self.processed_dataset)):
                example = next(dataset_iterator)
                actual_example_num = 1
                sequence = example
                # unused_samples and used_samples are used to calculate skip_samples and actual_train_samples
                if sequence is None:
                    if self.estimate:
                        self.unused_samples += actual_example_num
                    continue
                if self.estimate:
                    self.used_samples += actual_example_num
                batch_sequence, cur_len = [sequence], len(sequence.token_ids)
                yield batch_sequence

                if self.estimate:
                    self.used_estimate_samples += actual_example_num
                    if self.used_estimate_samples >= self.max_estimate_samples:
                        self.used_estimate_samples = 0
                        # Set flag to False and yield empty list to signal the end of estimation
                        self.estimate = False
                        yield []
            if len(batch_sequence) > 0:
                yield batch_sequence
        else:
            if not self.greedy_intokens:
                # base
                for _ in range(len(self.processed_dataset)):
                    example = next(dataset_iterator)
                    actual_example_num = 1
                    sequence = example
                    if sequence is None:
                        if self.estimate:
                            self.unused_samples += actual_example_num
                        continue
                    if self.estimate:
                        self.used_samples += actual_example_num
                    if cur_len + len(sequence.token_ids) <= self.max_seq_len:
                        batch_sequence.append(sequence)
                        cur_len += len(sequence.token_ids)
                    else:
                        yield batch_sequence
                        batch_sequence, cur_len = [sequence], len(sequence.token_ids)

                    if self.estimate:
                        self.used_estimate_samples += actual_example_num
                        if self.used_estimate_samples >= self.max_estimate_samples:
                            # Yield left batch sequence before estimation ends
                            if len(batch_sequence) > 0:
                                yield batch_sequence
                            self.used_estimate_samples = 0
                            # Set flag to False and yield empty list to signal the end of estimation
                            self.estimate = False
                            yield []
                if len(batch_sequence) > 0:
                    yield batch_sequence
            else:
                # Pseudo multiple rounds + group greedy intokens.
                buffer_size = 500
                examples = []
                actual_example_num_list = []
                i = 0
                for _ in range(len(self.processed_dataset)):
                    example = next(dataset_iterator)
                    actual_example_num = 1
                    if i < buffer_size:
                        examples.append(example)
                        actual_example_num_list.append(actual_example_num)
                        i += 1
                    else:
                        # Running greedy strategy in examples.
                        generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                        for pack in generate_packs:
                            if len(pack) > 0:
                                yield pack
                        examples = [example]
                        i = 1

                    if self.estimate:
                        self.used_estimate_samples += actual_example_num
                        # Stop estimation if the number of samples used in estimation is larger than max_estimate_samples
                        if self.used_estimate_samples >= self.max_estimate_samples:
                            # Yield left packs before estimation ends
                            if len(examples) > 0:
                                generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                                for pack in generate_packs:
                                    if len(pack) > 0:
                                        yield pack
                            # Set flag to False and yield empty list to signal the end of estimation
                            self.estimate = False
                            yield []

                if len(examples) > 0:
                    generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                    for pack in generate_packs:
                        if len(pack) > 0:
                            yield pack

    def _generate_greedy_packs(self, examples, actual_example_num_list):
        """Generate packed sequences using greedy strategy.

        Args:
            examples: List of examples to pack.
            actual_example_num_list: List of example counts.

        Returns:
            list: List of packed sequences.
        """

        left_len = np.zeros([len(examples)]) - 1
        left_len[0] = self.max_seq_len  # At the beginning, only the first pack is valid.
        generate_packs = [[] for i in range(len(examples))]
        index = 0
        left_index = 0

        while index < len(examples):
            sequence = examples[index]
            if sequence is None:
                if self.estimate:
                    self.unused_samples += actual_example_num_list[index]
                index += 1
                continue

            max_left_index = left_len.argmax()
            # Put the current sequence into the largest left space valid pack.
            if len(sequence.token_ids) <= left_len[max_left_index]:
                generate_packs[max_left_index].append(sequence)
                left_len[max_left_index] -= len(sequence.token_ids)
                if self.estimate:
                    self.used_samples += actual_example_num_list[index]
                index += 1
            else:
                left_index += 1
                left_len[left_index] = self.max_seq_len

        return generate_packs