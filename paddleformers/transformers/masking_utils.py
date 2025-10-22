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

from typing import Callable, Optional

import paddle

from .configuration_utils import PretrainedConfig


def prepare_sliding_window_startend_row_indices(
    startend_row_indices: paddle.Tensor,
    window_size: int = 5,
):
    """
    Restrict start-end row indices to a sliding window range.

    Args:
        startend_row_indices (`paddle.Tensor`, optional):
            Tensor of shape (batch_size, num_heads, seq_length, bound_num) containing start/end row indices.
            If None, returns None.
        window_size (`int`, optional, defaults to 5):
            Sliding window size to restrict the start indices.

    Returns:
        `paddle.Tensor` or None:
            Modified start-end row indices with sliding window applied.
    """
    if startend_row_indices is None:
        return None
    batch_size, num_head, seq_length, bound_num = startend_row_indices.shape
    assert bound_num <= 2, f"bound_num should be <= 2 when using sliding window, but got {bound_num}"
    sliding_window_startend_row_indices = startend_row_indices.clone()
    for bi in range(batch_size):
        for hi in range(num_head):
            for j in range(seq_length):
                sliding_window_startend_row_indices[bi, hi, j, 0] = min(
                    startend_row_indices[bi, hi, j, 0], window_size + j
                )
    return sliding_window_startend_row_indices


def create_causal_masks_and_row_indices(
    config: PretrainedConfig,
    inputs_embeds: paddle.Tensor,
    batch_size: int,
    seq_length: int,
    cache_length: int,
    attention_mask: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    prepare_decoder_attention_mask: Optional[Callable] = None,
    return_mapping: bool = True,
):
    """
    Prepare causal attention masks and optional start/end row indices for full and sliding attention.

    This function handles both:
    1. Pre-computed start/end row indices for optimized attention.
    2. Standard causal masks, optionally supporting sliding-window attention.

    Args:
        config (`PretrainedConfig`):
            Model configuration. Must include attributes like `sliding_window`.
        inputs_embeds (`paddle.Tensor`):
            Input embeddings of shape `(batch_size, seq_length, hidden_dim)`.
        batch_size (`int`):
            Current batch size.
        seq_length (`int`):
            Sequence length **excluding** past key-values.
        cache_length (`int`):
            Length of cached key-values (past sequence length).
        attention_mask (`paddle.Tensor`, *optional*):
            Attention mask of shape `(batch_size, seq_length + cache_length)`. If `None`, a mask of ones is used.
        attn_mask_startend_row_indices (`paddle.Tensor`, *optional*):
            Pre-computed start and end row indices for efficient attention. If provided, causal masks are skipped.
        prepare_decoder_attention_mask (`Callable`, *optional*):
            Function that creates causal attention masks, similar to
            `transformers.models.llama.modeling_llama._prepare_decoder_attention_mask`.
        return_mapping (`bool`, *optional*, defaults to True):
            - If True, returns dicts mapping `"full_attention"` and `"sliding_attention"`.
            - If False, returns a tuple `(causal_mask, attn_mask_startend_row_indices)` for single-mode attention.

    Returns:
        Tuple[Dict[str, paddle.Tensor], Dict[str, paddle.Tensor]] or Tuple[paddle.Tensor, paddle.Tensor]:
            - causal_mask_mapping (`Dict[str, paddle.Tensor]` or `paddle.Tensor`/None):
                Attention masks for `"full_attention"` and `"sliding_attention"`.
            - attn_mask_startend_row_indices_mapping (`Dict[str, paddle.Tensor]` or `paddle.Tensor`/None):
                Start/end row indices mapping for full and sliding attention.
    """

    has_sliding_layers = config.sliding_window is not None and "sliding_attention" in config.layer_types

    if attn_mask_startend_row_indices is not None:
        attention_mask = None
        causal_mask = None

        if return_mapping:
            causal_mask_mapping = {"full_attention": None, "sliding_attention": None}
            attn_mask_startend_row_indices_mapping = {
                "full_attention": attn_mask_startend_row_indices,
                "sliding_attention": (
                    prepare_sliding_window_startend_row_indices(
                        attn_mask_startend_row_indices, window_size=config.sliding_window
                    )
                    if has_sliding_layers
                    else None
                ),
            }
            return causal_mask_mapping, attn_mask_startend_row_indices_mapping
        else:
            if has_sliding_layers:
                attn_mask_startend_row_indices = prepare_sliding_window_startend_row_indices(
                    attn_mask_startend_row_indices, window_size=config.sliding_window
                )
            return causal_mask, attn_mask_startend_row_indices

    seq_length_with_past = seq_length + cache_length
    attention_mask = (
        paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
        if attention_mask is None
        else attention_mask
    )

    if return_mapping:
        causal_mask_mapping = {
            "full_attention": prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=cache_length,
                dtype=inputs_embeds.dtype,
            ),
            "sliding_attention": (
                prepare_decoder_attention_mask(
                    attention_mask=attention_mask,
                    input_shape=(batch_size, seq_length),
                    past_key_values_length=cache_length,
                    dtype=inputs_embeds.dtype,
                    sliding_window_size=config.sliding_window,
                )
                if has_sliding_layers
                else None
            ),
        }
        attn_mask_startend_row_indices_mapping = {"full_attention": None, "sliding_attention": None}
        return causal_mask_mapping, attn_mask_startend_row_indices_mapping
    else:
        causal_mask = (
            prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=cache_length,
                dtype=inputs_embeds.dtype,
                sliding_window_size=config.sliding_window,
            )
            if has_sliding_layers
            else prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=cache_length,
                dtype=inputs_embeds.dtype,
            )
        )
        attn_mask_startend_row_indices = None
        return causal_mask, attn_mask_startend_row_indices
