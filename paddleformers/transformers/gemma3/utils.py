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


import paddle


def prepare_sliding_window_startend_row_indices(startend_row_indices, window_size=5):
    if startend_row_indices is None:
        return None
    batch_size, num_head, seq_length, bound_num = startend_row_indices.shape
    assert bound_num <= 2, f"bound_num should be less than or equal to 2 when use sling window, but got {bound_num}"
    sliding_window_startend_row_indices = startend_row_indices.clone()
    for bi in range(batch_size):
        for hi in range(num_head):
            for j in range(seq_length):
                sliding_window_startend_row_indices[bi, hi, j, 0] = min(
                    startend_row_indices[bi, hi, j, 0], window_size + j
                )
    return sliding_window_startend_row_indices


def ignore_causal_mask_sdpa(
    attention_mask,
    input_shape,
    past_key_values_length,
    sliding_window_size=None,
    **kwargs,
) -> bool:
    """
    Detects whether the causal mask can be ignored when using Paddle's SDPA,
    allowing reliance on the `is_causal` argument for performance optimization.

    If no token is masked in the 2D `padding_mask`, and if `query_length == 1` or
    `kv_length == query_length`, we can safely use SDPA's built-in causal mechanism.
    This enables dispatch to optimized kernels (e.g., Flash Attention).
    """
    # Check if we are in static graph mode (equivalent to torch.jit.is_tracing)
    def is_in_jit_mode():
        try:
            return paddle.jit.dy2static.globals._in_declarative_mode()
        except AttributeError:
            return False

    is_tracing = is_in_jit_mode()

    # Adjust padding_mask to current kv range if necessary
    batch_size, query_length = input_shape
    kv_length = query_length + past_key_values_length
    kv_offset = past_key_values_length

    if attention_mask is not None and attention_mask.shape[-1] > kv_length:
        mask_indices = paddle.arange(kv_length, dtype="int64") + kv_offset
        attention_mask = paddle.index_select(attention_mask, axis=-1, index=mask_indices)

    # Skip mask only if not tracing and conditions allow
    if (
        not is_tracing
        and (query_length == 1 or kv_length == query_length)
        and (sliding_window_size is None or kv_length < sliding_window_size)
        and (
            attention_mask is None
            or (attention_mask.all() if query_length == 1 else attention_mask[:, :query_length].all())
        )
    ):
        return True

    return False
