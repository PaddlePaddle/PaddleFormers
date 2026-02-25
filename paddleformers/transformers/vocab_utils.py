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

# Refer to NVIDIA Megatron-Bridge https://github.com/NVIDIA-NeMo/Megatron-Bridge
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import math
from functools import lru_cache

import paddle


def print_rank_0(*args, **kwargs):
    if paddle.distributed.get_rank() == 0:
        print(*args, **kwargs)


def calculate_padded_vocab_size(
    vocab_size: int,
    make_vocab_size_divisible_by: int,
    tensor_model_parallel_size: int,
    logging_enabled: bool = True,
) -> int:
    """Calculate padded vocab size for tensor parallelism.

    This function pads the vocabulary size to ensure it's divisible by the required
    multiple for efficient tensor parallel operations.

    Args:
        vocab_size: The original (unpadded) vocabulary size
        make_vocab_size_divisible_by: Base divisibility requirement (e.g., 128)
        tensor_model_parallel_size: Number of tensor parallel ranks
        logging_enabled: Whether to log the padding information

    Returns:
        int: The padded vocabulary size
    """
    padded_size = _calculate_padded_vocab_size_cached(
        vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size
    )

    # Handle logging separately to avoid affecting cache behavior
    if logging_enabled:
        print_rank_0(
            " > padded vocab (size: {}) with {} dummy tokens (new size: {})".format(
                vocab_size, padded_size - vocab_size, padded_size
            )
        )

    return padded_size


@lru_cache(maxsize=128)
def _calculate_padded_vocab_size_cached(
    vocab_size: int,
    make_vocab_size_divisible_by: int,
    tensor_model_parallel_size: int,
) -> int:
    """Cached computation of padded vocab size."""
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if make_vocab_size_divisible_by <= 0:
        raise ValueError(f"make_vocab_size_divisible_by must be positive, got {make_vocab_size_divisible_by}")
    if tensor_model_parallel_size <= 0:
        raise ValueError(f"tensor_model_parallel_size must be positive, got {tensor_model_parallel_size}")

    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    return int(math.ceil(vocab_size / multiple) * multiple)
