# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for the in-place Triton-fused MLA RoPE.

Verifies, against the unfused PaddleFleet reference path, that
`fused_apply_mla_rope_inplace`:

  - Forward output equivalent (bit-exact match) to the slice + rope +
    concat baseline used in dsv4_hybrid_attention.py.
  - Truly in-place: q.data_ptr() preserved.
  - Backward grads match the autograd reference.
"""

import unittest

import paddle

from paddleformers.fleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddleformers.fleet.triton_ops import fused_apply_mla_rope_inplace

# Shapes from DeepSeek-V4-Flash
B, S, H, D = 1, 4096, 64, 512
NOPE_DIM = 448
ROPE_DIM = D - NOPE_DIM  # 64


def _reference_forward(
    q: paddle.Tensor,
    freqs: paddle.Tensor,
    inverse: bool = False,
) -> paddle.Tensor:
    """Slice + rope + concat baseline (current implementation)."""
    q_nope = q[..., :NOPE_DIM]
    q_pe = q[..., NOPE_DIM:]
    q_pe = _apply_rotary_pos_emb_bshd(
        q_pe,
        freqs,
        mscale=1.0,
        rotary_interleaved=False,
        multi_latent_attention=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    return paddle.concat([q_nope, q_pe], axis=-1)


def _check_equal(a: paddle.Tensor, b: paddle.Tensor):
    """Binary-exact equality check (no tolerance allowed)."""
    assert paddle.all(a == b), f"tensor not equal:\nA: {a}\nB: {b}"


class TestFusedMLARopeQPeInplace(unittest.TestCase):
    def setUp(self) -> None:
        paddle.seed(0)

    def _run_case(
        self,
        b: int,
        s: int,
        freqs: paddle.Tensor,
        inverse: bool = False,
    ) -> None:
        """Shared driver: q is contiguous bf16; freqs supplied by caller."""
        x = paddle.randn([b, s, H, D], "bfloat16")
        x.stop_gradient = False

        # ---- reference path ----
        x_ref = x.detach()
        x_ref.stop_gradient = False
        q_ref = x_ref.clone()  # non-leaf
        out_ref = _reference_forward(q_ref, freqs, inverse=inverse)
        out_grad = paddle.randn_like(out_ref)
        out_ref.backward(out_grad)
        grad_ref = x_ref.grad

        # ---- fused path ----
        x_fused = x.detach()
        x_fused.stop_gradient = False
        q_fused = x_fused.clone()  # non-leaf — safe target for in-place kernel
        self.assertTrue(q_fused.is_contiguous())
        nope_before = q_fused[..., :NOPE_DIM].clone()
        ptr_before = q_fused.data_ptr()
        out_fused = fused_apply_mla_rope_inplace(
            q_fused, freqs, NOPE_DIM, inverse=inverse
        )

        # ---- in-place storage invariants ----
        self.assertIs(out_fused, q_fused)
        self.assertEqual(out_fused.data_ptr(), ptr_before)
        _check_equal(q_fused[..., :NOPE_DIM], nope_before)

        # ---- forward parity ----
        _check_equal(out_fused, out_ref)

        # ---- backward parity ----
        out_fused.backward(out_grad)
        grad_fused = x_fused.grad

        _check_equal(grad_fused[..., :NOPE_DIM], grad_ref[..., :NOPE_DIM])
        _check_equal(grad_fused[..., NOPE_DIM:], grad_ref[..., NOPE_DIM:])

    def test_forward_backward(self) -> None:
        """Test the normal case."""
        freqs = paddle.randn([B, S, 1, ROPE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, freqs)

    def test_freqs_noncontiguous_b_gt_1(self) -> None:
        """Test multi-batch and non-contiguous freqs."""
        b = 2
        s = 128  # smaller to keep test fast
        # Build oversize freqs and slice along seq (non-contig stride),
        # then unsqueeze the singleton head dim from a slice as well.
        rope_len = s + 17  # arbitrary position_offset
        freqs_full = paddle.randn([b, rope_len, ROPE_DIM])
        freqs_full.stop_gradient = True
        freqs = freqs_full[:, 17 : 17 + s, :].unsqueeze(2)  # [b, s, 1, D]
        # Sanity: this slice is non-contiguous.
        self.assertFalse(freqs.is_contiguous())
        self._run_case(b, s, freqs)

    def test_inverse(self) -> None:
        """Test inverse rope."""
        freqs = paddle.randn([B, S, 1, ROPE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, freqs, inverse=True)


if __name__ == "__main__":
    unittest.main()
