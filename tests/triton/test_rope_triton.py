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

import paddle


def apply_rotary_ref(x, cos, sin, conjugate=False):

    orig_dtype = x.dtype
    x = x.astype("float32")
    cos = cos.astype("float32")
    sin = sin.astype("float32")

    if conjugate:
        sin = -sin

    rotary_dim = cos.shape[-1] * 2
    x0 = x[..., : rotary_dim // 2]
    x1 = x[..., rotary_dim // 2 : rotary_dim]

    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos

    out = paddle.concat([o0, o1], axis=-1)
    if rotary_dim < x.shape[-1]:
        out = paddle.concat([out, x[..., rotary_dim:]], axis=-1)

    return out.astype(orig_dtype)


def apply_rotary_pos_emb_vision_ref(q, k, cos, sin):
    """Reference implementation for q, k pair."""
    q_out = apply_rotary_ref(q, cos, sin, conjugate=False)
    k_out = apply_rotary_ref(k, cos, sin, conjugate=False)
    return q_out, k_out


class TestTritonRoPE(unittest.TestCase):
    """Test cases for Triton Rotary Position Embedding."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_shapes = [
            (1, 1024, 16, 72),  # Standard shape
            (2, 512, 16, 72),  # Batch > 1
        ]
        self.dtype = "bfloat16"

    def test_forward_correctness(self):
        """Test forward pass correctness against reference implementation."""
        from paddleformers.triton_kernels import apply_rotary_pos_emb_vision

        for batch, seq_len, num_heads, head_dim in self.test_shapes:
            with self.subTest(shape=(batch, seq_len, num_heads, head_dim)):
                q = paddle.randn([batch, seq_len, num_heads, head_dim], dtype=self.dtype)
                k = paddle.randn([batch, seq_len, num_heads, head_dim], dtype=self.dtype)
                cos = paddle.randn([seq_len, head_dim // 2], dtype=self.dtype)
                sin = paddle.randn([seq_len, head_dim // 2], dtype=self.dtype)

                # Reference
                q_ref, k_ref = apply_rotary_pos_emb_vision_ref(q, k, cos, sin)

                # Triton
                q_tri, k_tri = apply_rotary_pos_emb_vision(q, k, cos, sin)

                # Check correctness
                q_close = paddle.allclose(
                    q_ref.astype("float32"), q_tri.astype("float32"), rtol=1e-2, atol=1e-2
                ).item()
                k_close = paddle.allclose(
                    k_ref.astype("float32"), k_tri.astype("float32"), rtol=1e-2, atol=1e-2
                ).item()

                self.assertTrue(q_close, f"Q mismatch for shape {(batch, seq_len, num_heads, head_dim)}")
                self.assertTrue(k_close, f"K mismatch for shape {(batch, seq_len, num_heads, head_dim)}")

    def test_backward_correctness(self):
        """Test backward pass correctness."""
        from paddleformers.triton_kernels import apply_rotary_pos_emb_vision

        batch, seq_len, num_heads, head_dim = 1, 1024, 16, 72

        q = paddle.randn([batch, seq_len, num_heads, head_dim], dtype=self.dtype)
        k = paddle.randn([batch, seq_len, num_heads, head_dim], dtype=self.dtype)
        cos = paddle.randn([seq_len, head_dim // 2], dtype=self.dtype)
        sin = paddle.randn([seq_len, head_dim // 2], dtype=self.dtype)

        # Triton forward + backward
        q_tri = q.clone()
        k_tri = k.clone()
        q_tri.stop_gradient = False
        k_tri.stop_gradient = False

        q_embed, k_embed = apply_rotary_pos_emb_vision(q_tri, k_tri, cos, sin)
        loss = q_embed.sum() + k_embed.sum()
        loss.backward()

        # Reference gradient (conjugate=True means sin = -sin)
        grad_output = paddle.ones_like(q)
        grad_ref = apply_rotary_ref(grad_output, cos, sin, conjugate=True)

        # Compare
        grad_q_close = paddle.allclose(
            grad_ref.astype("float32"), q_tri.grad.astype("float32"), rtol=1e-2, atol=1e-2
        ).item()
        grad_k_close = paddle.allclose(
            grad_ref.astype("float32"), k_tri.grad.astype("float32"), rtol=1e-2, atol=1e-2
        ).item()

        self.assertTrue(grad_q_close, "grad_q mismatch")
        self.assertTrue(grad_k_close, "grad_k mismatch")

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        from paddleformers.triton_kernels import apply_rotary_pos_emb_vision

        batch, seq_len, num_heads, head_dim = 1, 1024, 16, 72
        dtypes = ["bfloat16", "float16", "float32"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                q = paddle.randn([batch, seq_len, num_heads, head_dim], dtype=dtype)
                k = paddle.randn([batch, seq_len, num_heads, head_dim], dtype=dtype)
                cos = paddle.randn([seq_len, head_dim // 2], dtype=dtype)
                sin = paddle.randn([seq_len, head_dim // 2], dtype=dtype)

                q_out, k_out = apply_rotary_pos_emb_vision(q, k, cos, sin)

                self.assertEqual(q_out.dtype, q.dtype, f"Q dtype not preserved for {dtype}")
                self.assertEqual(k_out.dtype, k.dtype, f"K dtype not preserved for {dtype}")

    def test_3d_input(self):
        """Test that 3D input (without batch dim) works correctly."""
        from paddleformers.triton_kernels import apply_rotary_pos_emb_vision

        seq_len, num_heads, head_dim = 1024, 16, 72

        q = paddle.randn([seq_len, num_heads, head_dim], dtype=self.dtype)
        k = paddle.randn([seq_len, num_heads, head_dim], dtype=self.dtype)
        cos = paddle.randn([seq_len, head_dim // 2], dtype=self.dtype)
        sin = paddle.randn([seq_len, head_dim // 2], dtype=self.dtype)

        q_out, k_out = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Check output shape is same as input
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

        # Check correctness
        q_ref, k_ref = apply_rotary_pos_emb_vision_ref(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q_ref = q_ref.squeeze(0)
        k_ref = k_ref.squeeze(0)

        q_close = paddle.allclose(q_ref.astype("float32"), q_out.astype("float32"), rtol=1e-2, atol=1e-2).item()
        k_close = paddle.allclose(k_ref.astype("float32"), k_out.astype("float32"), rtol=1e-2, atol=1e-2).item()

        self.assertTrue(q_close, "Q mismatch for 3D input")
        self.assertTrue(k_close, "K mismatch for 3D input")


if __name__ == "__main__":
    unittest.main()
