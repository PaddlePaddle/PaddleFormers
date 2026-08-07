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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from .cross_entropy import vocab_parallel_cross_entropy
from .layers import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from .random import (
    RecomputeWithoutOutput,
    checkpoint,
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
    model_parallel_cuda_manual_seed,
)

__all__ = [
    # cross_entropy.py
    "vocab_parallel_cross_entropy",
    # layers.py
    "ColumnParallelLinear",
    "Linear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    # random.py
    "checkpoint",
    "get_cuda_rng_tracker",
    "model_parallel_cuda_manual_seed",
    "get_expert_parallel_rng_tracker_name",
    "RecomputeWithoutOutput",
]
