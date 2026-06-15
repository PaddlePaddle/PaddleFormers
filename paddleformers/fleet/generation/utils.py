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

"""Utility functions for text generation."""

import paddle
import paddle.nn.functional as F


def apply_repetition_penalty(logits, input_ids, penalty):
    """Apply repetition penalty to logits.

    This function penalizes tokens that have already appeared in the input sequence.
    Tokens with positive logits are divided by the penalty, tokens with negative
    logits are multiplied by the penalty.

    Args:
        logits: Logits tensor with shape [batch_size, vocab_size]
        input_ids: Input token ids with shape [batch_size, seq_len]
        penalty: Repetition penalty factor (1.0 = no penalty, >1.0 = penalize repeats)

    Returns:
        Logits with repetition penalty applied, same shape as input
    """
    if penalty == 1.0:
        return logits

    batch_size, seq_len = input_ids.shape
    vocab_size = logits.shape[-1]

    # Create mask for tokens in input_ids
    # [batch_size, seq_len, vocab_size]
    input_ids_expanded = input_ids.unsqueeze(-1).expand([-1, -1, vocab_size])

    # Create position mask
    positions = paddle.arange(vocab_size).unsqueeze(0).unsqueeze(0).expand([batch_size, seq_len, -1])
    token_mask = (positions == input_ids_expanded).astype("float32")

    # Get logits for tokens that appear in input
    # [batch_size, vocab_size]
    token_appearance = token_mask.sum(axis=1)  # How many times each token appears
    mask = token_appearance > 0

    # Apply penalty: divide positive logits, multiply negative logits
    # We only apply to tokens that actually appeared
    affected_logits = paddle.where(
        mask,  # [batch_size, vocab_size] - no unsqueeze needed
        paddle.where(
            logits > 0,
            logits / penalty,
            logits * penalty,
        ),
        logits,
    )

    return affected_logits


def sample_with_top_k(logits, top_k):
    """Sample from logits with top-k filtering.

    This function filters out all but the top-k highest logit values and samples
    from the remaining distribution.

    Args:
        logits: Logits tensor with shape [batch_size, vocab_size]
        top_k: Number of top tokens to keep (>= 1)

    Returns:
        Sampled token ids with shape [batch_size, 1]
    """
    if top_k >= logits.shape[-1]:
        # No filtering needed
        probs = F.softmax(logits)
        return paddle.multinomial(probs)  # [batch_size, 1]

    # Get top-k logits and their threshold
    top_k_logits, _ = paddle.topk(logits, k=top_k, axis=-1)
    threshold = top_k_logits[:, -1].unsqueeze(-1)

    # Filter: set non-top-k logits to -inf
    filter_mask = logits < threshold
    logits = paddle.where(
        filter_mask,
        paddle.full_like(logits, float("-inf")),
        logits,
    )

    # Sample from filtered distribution
    probs = F.softmax(logits)
    sampled = paddle.multinomial(probs)
    return sampled  # [batch_size, 1]


def sample_with_top_p(logits, top_p):
    """Sample from logits with top-p (nucleus) filtering.

    This function filters out tokens with cumulative probability less than top-p
    and samples from the remaining distribution. Following the Megatron implementation
    which includes a shift to ensure at least one token is kept.

    Args:
        logits: Logits tensor with shape [batch_size, vocab_size]
        top_p: Cumulative probability threshold (0, 1.0]

    Returns:
        Sampled token ids with shape [batch_size, 1]
    """
    batch_size, vocab_size = logits.shape

    # Sort logits in descending order
    sorted_indices = paddle.argsort(logits, descending=True, axis=-1)
    # Use gather_nd instead of gather (Paddle's gather has index shape restrictions)
    batch_indices = paddle.arange(batch_size).unsqueeze(-1).expand([-1, vocab_size])
    indices = paddle.stack([batch_indices, sorted_indices], axis=-1)
    sorted_logits = paddle.gather_nd(logits, indices)

    # Calculate cumulative probabilities
    sorted_probs = F.softmax(sorted_logits)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Create filter mask: tokens with cumulative probability > top_p
    filter_mask = cumulative_probs > top_p

    # Shift filter by 1 position (from Megatron implementation)
    # This ensures at least one token is kept and the first token is never filtered
    filter_mask = paddle.concat(
        [
            paddle.zeros([filter_mask.shape[0], 1], dtype="bool"),
            filter_mask[:, :-1],
        ],
        axis=-1,
    )

    # Scatter the filter mask back to original order
    # Flatten indices and filter_mask for scatter_nd
    flat_indices = indices.reshape([-1, 2])
    flat_filter = filter_mask.reshape([-1]).astype("int32")
    scattered_filter = paddle.scatter_nd(
        flat_indices,
        flat_filter,
        [batch_size, vocab_size],
    ).astype("bool")

    # Apply filter: set filtered logits to -inf
    filtered_logits = paddle.where(
        scattered_filter,
        paddle.full_like(logits, float("-inf")),
        logits,
    )

    # Sample from filtered distribution
    probs = F.softmax(filtered_logits)
    sampled = paddle.multinomial(probs)

    # sampled is already in the original vocabulary index space,
    # no need to remap through sorted_indices
    return sampled  # [batch_size, 1]
