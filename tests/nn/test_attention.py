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

import unittest
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn

from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS


def flashmask_to_densemask(startend_row_indices, num_key_value_groups, dtype, causal=True):
    """
    Helper function to convert the sparse `startend_row_indices` format, used by FlashMask,
    into a dense attention mask tensor that can be used by naive attention implementations.
    """
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)

    # Iterate through batch, heads, and sequence length to build the dense mask
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):  # j represents the key/column index
                downstart = startend_row_indices[bi, hi, j, 0].item()
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1].item()
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf

                if causal:
                    # For causal attention, mask out all future tokens
                    m[bi, hi, j + 1 :, j] = -np.inf
                else:
                    # For non-causal, use the provided upper bounds
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2].item()
                        upend = startend_row_indices[bi, hi, j, 3].item()
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1].item()
                        m[bi, hi, :upend, j] = -np.inf

    # If using Grouped-Query Attention (GQA), the mask for KV heads must be
    # expanded to match the number of Query heads.
    if num_key_value_groups > 1:
        m = m.unsqueeze(2).expand([bz, num_head, num_key_value_groups, seq_len, seq_len])
        num_q_heads = num_head * num_key_value_groups
        m = m.reshape([bz, num_q_heads, seq_len, seq_len])

    # The final mask shape is [B, H, S, S] to match the attention weights matrix.
    return m


class TestAttentionInterface(unittest.TestCase):
    """
    Unit tests for the high-level attention function interface in PaddleFormers.
    This class tests both the callability and numerical correctness of different
    attention implementations (e.g., sdpa, flashmask) against a naive reference.
    """

    def gen_random_flashmask(self, bz, num_head, seqlen, has_end, causal):
        """Generates a random sparse mask in the FlashMask format [start, end]."""
        mask_num = 1
        if not causal:
            mask_num *= 2
        if has_end:
            mask_num *= 2

        m = np.random.randint(0, seqlen, (bz, num_head, seqlen, mask_num))
        diag = np.arange(seqlen).reshape((1, 1, seqlen))

        # Ensure start index is not after the diagonal
        m[:, :, :, 0] = np.maximum(diag, m[:, :, :, 0])

        if not causal:
            if has_end:
                raise NotImplementedError
            # Ensure end index is after the start index
            m[:, :, :, 1] = np.minimum(diag + 1, m[:, :, :, 1])
        else:
            m[:, :, :, 0] = diag  # For causal, start is always the diagonal
            if has_end:
                m[:, :, :, 1] = m[:, :, :, 0] + np.random.randint(1, seqlen, m[:, :, :, 0].shape)
                m[:, :, :, 1] = np.minimum(seqlen, m[:, :, :, 1])

        return paddle.to_tensor(m, dtype="int32")

    def setUp(self):
        """Set up common parameters and tensors for all tests in this class."""
        paddle.seed(92)  # Set a fixed seed for reproducibility
        self.batch_size = 1
        self.seq_len = 512
        self.num_heads = 64
        self.head_dim = 64

        self.scaling = self.head_dim**-0.5
        self.training = True
        self.dtype = "bfloat16"

        # Tensors are created in the [batch, seq_len, num_heads, head_dim] layout.
        # This setup configures a Multi-Head Attention (MHA) scenario because the
        # number of heads for key and value is the same as for query.
        self.query = paddle.rand([self.batch_size, self.seq_len, self.num_heads, self.head_dim], dtype=self.dtype)
        self.key = paddle.rand([self.batch_size, self.seq_len, self.num_heads, self.head_dim], dtype=self.dtype)
        self.value = paddle.rand([self.batch_size, self.seq_len, self.num_heads, self.head_dim], dtype=self.dtype)
        self.sink = paddle.rand([self.num_heads], dtype=self.dtype)

        # Flashmask is generated based on the number of attention heads.
        self.startend_row_indices = self.gen_random_flashmask(
            self.batch_size, self.num_heads, self.seq_len, has_end=False, causal=False
        )

    def assert_tensor_close(self, a, b, atol=1e-2, rtol=1e-2):
        """
        Assert that two tensors are close within specified tolerances.
        Converts tensors to float32 before comparison for better stability.
        """
        # Cast to float32 to avoid precision issues with bfloat16 during comparison
        a = a.to("float32")
        b = b.to("float32")
        self.assertTrue(
            paddle.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True),
            f"Tensors are not close.\n"
            f"Max Abs Error: {paddle.max(paddle.abs(a - b)).item()}\n"
            f"Max Rel Error: {paddle.max(paddle.abs(a - b) / (paddle.abs(b) + 1e-9)).item()}",
        )

    def naive_attn_sink(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        sink: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor],
        scaling: float,
        dropout: float = 0.0,
        num_key_value_groups: int = 1,
        **kwargs,
    ):
        """
        A naive reference implementation of attention with a 'sink' mechanism.
        This serves as the ground truth for correctness validation. It follows the
        standard attention formula step-by-step.
        """
        # Step 1: Reshape tensors from [B, S, H, D] to [B, H, S, D] for matrix multiplication
        query_states = paddle.transpose(query, perm=[0, 2, 1, 3])
        key_states = paddle.transpose(key, perm=[0, 2, 1, 3])
        value_states = paddle.transpose(value, perm=[0, 2, 1, 3])

        # Step 2: Transpose key for matmul: [B, H, S, D] -> [B, H, D, S]
        key_states = paddle.transpose(key_states, perm=[0, 1, 3, 2])

        # Step 3: Calculate attention scores (Query @ Key^T) and apply scaling
        attn_weights = paddle.matmul(query_states, key_states) * scaling

        # Step 4: Apply the attention mask if provided
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # Step 5: Prepare and concatenate the sink logits. The sink is a special token
        # that every other token can attend to, preventing the attention from collapsing.
        sinks = sink.reshape(shape=[1, -1, 1, 1]).expand(shape=[query_states.shape[0], -1, query_states.shape[-2], -1])
        combined_logits = paddle.cat(x=[attn_weights, sinks], axis=-1)

        # Step 6: Apply softmax over the combined logits (scores + sink)
        combined_logits = combined_logits - paddle.max(combined_logits, axis=-1, keepdim=True)
        probs = nn.functional.softmax(combined_logits, axis=-1, dtype=combined_logits.dtype)

        # Step 7: Separate the attention probabilities from the sink probabilities
        scores = probs[..., :-1]

        # Step 8: Apply dropout to the scores
        attn_weights = nn.functional.dropout(scores, p=dropout, training=True)

        # Step 9: Compute the weighted sum of values (Scores @ Value)
        attn_output = paddle.matmul(attn_weights, value_states)

        # Step 10: Reshape the output back to [B, S, H, D] and flatten the head dimension
        attn_output = paddle.transpose(attn_output, perm=[0, 2, 1, 3]).contiguous()
        attn_output = paddle.reshape(x=attn_output, shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]])

        return attn_output

    def test_forward_calls_correct_function(self):
        """
        A simple 'smoke test' to ensure that all attention interfaces can be
        called with the configured tensors without raising an error.
        """
        # Test the basic eager implementation
        eager_interface = ALL_ATTENTION_FUNCTIONS["eager"]
        eager_interface(self, self.query, self.key, self.value, scaling=self.scaling)

        # Test the SDPA implementation (without and with sink)
        sdpa_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        sdpa_interface(self, self.query, self.key, self.value, scaling=self.scaling)
        sdpa_interface(self, self.query, self.key, self.value, sink=self.sink, scaling=self.scaling)

        # Test the FlashMask implementation with its specific arguments
        flashmask_interface = ALL_ATTENTION_FUNCTIONS["flashmask"]
        flashmask_interface(
            self,
            self.query,
            self.key,
            self.value,
            scaling=self.scaling,
            attn_mask_startend_row_indices=self.startend_row_indices,
            sink=self.sink,
        )

    def test_correctness(self):
        """
        Verifies the numerical correctness of optimized attention implementations
        against the naive reference implementation.
        """
        # --- Test 1: SDPA with Causal Mask and Sink ---

        # Get the output from the optimized SDPA implementation
        sdpa_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        sdpa_output, _ = sdpa_interface(
            self, self.query, self.key, self.value, sink=self.sink, scaling=self.scaling, is_causal=True
        )

        # Create the ground truth dense causal mask for the naive implementation
        causal_mask = paddle.triu(
            paddle.full(shape=[self.seq_len, self.seq_len], fill_value=float("-inf"), dtype=self.dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(shape=[self.batch_size, self.num_heads, -1, -1])

        # Get the output from the naive reference implementation
        eager_output_causal = self.naive_attn_sink(
            self.query, self.key, self.value, self.sink, causal_mask, self.scaling
        )

        # Compare the results from the optimized and naive implementations
        self.assert_tensor_close(sdpa_output, eager_output_causal)

        # --- Test 2: FlashMask with Non-Causal Mask and Sink ---

        # Get the output from the optimized FlashMask implementation
        flashmask_interface = ALL_ATTENTION_FUNCTIONS["flashmask"]
        flashmask_output, _ = flashmask_interface(
            self,
            self.query,
            self.key,
            self.value,
            scaling=self.scaling,
            attn_mask_startend_row_indices=self.startend_row_indices,
            sink=self.sink,
            is_causal=False,
        )

        # Create the ground truth dense mask from the FlashMask sparse format
        dense_mask = flashmask_to_densemask(
            self.startend_row_indices, num_key_value_groups=1, dtype=self.dtype, causal=False
        )
        # Get the output from the naive reference implementation
        eager_output_flashmask = self.naive_attn_sink(
            self.query, self.key, self.value, self.sink, dense_mask, self.scaling
        )

        # Compare the results
        self.assert_tensor_close(flashmask_output, eager_output_flashmask)


# Standard entry point to run the tests when the script is executed directly
if __name__ == "__main__":
    unittest.main()
