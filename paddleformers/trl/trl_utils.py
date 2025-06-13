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


def calculate_effective_tokens(training_args, train_dataset, max_seq_len):
    """
    Calculate the effective tokens during training.
    """
    total_effective_tokens = 0

    try:
        data_parallel_degree = training_args.data_parallel_degree
    except:
        data_parallel_degree = 1
    if training_args.sharding_parallel_degree > 1:
        sharding_parallel_degree = training_args.sharding_parallel_degree
    else:
        sharding_parallel_degree = 1
    if training_args.max_steps > 0:
        total_batch = (
            training_args.max_steps
            * training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * sharding_parallel_degree
            * data_parallel_degree
        )
        for i, data in enumerate(train_dataset):
            if i == total_batch:
                break
            total_effective_tokens += len(data["input_ids"])
        total_tokens = total_batch * max_seq_len
    else:
        for i, data in enumerate(train_dataset):
            total_effective_tokens += len(data["input_ids"])
        total_tokens = (i + 1) * max_seq_len
        total_effective_tokens *= training_args.num_train_epochs
        total_tokens *= training_args.num_train_epochs
    return total_effective_tokens, total_tokens
