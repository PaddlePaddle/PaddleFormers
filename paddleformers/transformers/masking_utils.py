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
    This method is retained for compatibility and will be deprecated later

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

    sliding_window_val = getattr(config, "sliding_window", None)
    layer_types_val = getattr(config, "layer_types", [])

    has_sliding_layers = (sliding_window_val is not None) and ("sliding_attention" in layer_types_val)

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

    # Enables the efficient built-in causal mode (is_causal=True)
    # for FA backends (sdpa/flashmask), bypassing manual mask generation.
    # for third-party attention registered via _attn_implementation, default to bypass mask generation.
    attn_impl = getattr(config, "_attn_implementation", "eager")
    is_flash_backend = attn_impl != "eager"
    is_fully_attended = attention_mask is None or (attention_mask is not None and attention_mask.cast("bool").all())
    if is_flash_backend and is_fully_attended:
        if return_mapping:
            causal_mask_mapping = {"full_attention": None, "sliding_attention": None}
            attn_mask_startend_row_indices_mapping = {"full_attention": None, "sliding_attention": None}
            return causal_mask_mapping, attn_mask_startend_row_indices_mapping
        else:
            return None, None
    # We only return an actual mask if there is at least 1 padding token,
    # otherwise we return `None` and use `is_causal` in FA2
    if attention_mask is not None and attention_mask.cast("bool").all():
        attention_mask = None

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


def create_causal_mask_and_row_indices(
    config: PretrainedConfig,
    inputs_embeds: paddle.Tensor,
    batch_size: int,
    seq_length: int,
    cache_length: int,
    attention_mask: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    prepare_decoder_attention_mask: Optional[Callable] = None,
    or_mask_function: Optional[Callable] = None,
):
    """
        Prepare causal attention mask and optional start/end row indices for full attention.

        Args:
            config (`PretrainedConfig`):
                Model configuration.
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
                Pre-computed start and end row indices for efficient attention. If provided, causal mask is skipped.
            prepare_decoder_attention_mask (`Callable`, *optional*):
                Function that creates causal attention masks.
            or_mask_function (`Callable`, optional):
                An optional mask function to combine with the causal mask function (by doing the union of both). This is
                useful to easily overlay another mask on top of the causal one, for example for image tokens handling.

    Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - causal_mask: The attention mask for full attention.
                - attn_mask_startend_row_indices: The row indices for full attention (if applicable).
    """
    if attn_mask_startend_row_indices is not None:
        causal_mask = None
        row_indices = attn_mask_startend_row_indices
    else:
        attn_impl = getattr(config, "_attn_implementation", "eager")
        is_flash_backend = attn_impl != "eager"

        # Check if the mask can be safely skipped
        # Condition: Must be Flash Backend AND No extra mask func AND No padding (mask is None or all True)
        # For third-party attention registered via _attn_implementation, default to bypass mask generation.
        is_fully_attended = attention_mask is None or (
            attention_mask is not None and attention_mask.cast("bool").all()
        )

        if is_flash_backend and or_mask_function is None and is_fully_attended:
            causal_mask = None
            row_indices = None
        else:
            seq_length_with_past = seq_length + cache_length
            attention_mask = (
                paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                if attention_mask is None
                else attention_mask
            )

            causal_mask = prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=cache_length,
                dtype=inputs_embeds.dtype,
                or_mask_function=or_mask_function,
            )
            row_indices = None

    return causal_mask, row_indices


def create_sliding_window_causal_mask_and_row_indices(
    config: PretrainedConfig,
    inputs_embeds: paddle.Tensor,
    batch_size: int,
    seq_length: int,
    cache_length: int,
    attention_mask: Optional[paddle.Tensor] = None,
    attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    prepare_decoder_attention_mask: Optional[Callable] = None,
    or_mask_function: Optional[Callable] = None,
):
    """
        Prepare causal attention mask and optional start/end row indices for sliding window attention.

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
                Pre-computed start and end row indices. If provided, they are adapted for sliding window.
            prepare_decoder_attention_mask (`Callable`, *optional*):
                Function that creates causal attention masks.
            or_mask_function (`Callable`, optional):
                An optional mask function to combine with the causal mask function (by doing the union of both). This is
                useful to easily overlay another mask on top of the causal one, for example for image tokens handling.

    Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - causal_mask: The attention mask for sliding attention.
                - attn_mask_startend_row_indices: The row indices adjusted for sliding window.
    """
    sliding_window_val = getattr(config, "sliding_window", None)

    if attn_mask_startend_row_indices is not None:
        causal_mask = None
        row_indices = prepare_sliding_window_startend_row_indices(
            attn_mask_startend_row_indices, window_size=sliding_window_val
        )
    else:
        seq_length_with_past = seq_length + cache_length
        attention_mask = (
            paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            if attention_mask is None
            else attention_mask
        )

        causal_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=cache_length,
            dtype=inputs_embeds.dtype,
            sliding_window_size=sliding_window_val,
            or_mask_function=or_mask_function,
        )
        row_indices = None

    return causal_mask, row_indices
