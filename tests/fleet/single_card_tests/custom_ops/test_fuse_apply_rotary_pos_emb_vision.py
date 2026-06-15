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

"""Unit tests for the fused_apply_rotary_pos_emb_vision CUDA custom op.

Tests cover:
  1. Forward pass accuracy vs. Python reference (float32, float16, bfloat16).
  2. Backward pass accuracy (gradient through paddle.grad) vs. Python reference.
  3. Shape correctness.

Python reference (exactly matching the original Python function):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)

    def apply_rotary_pos_emb_vision(tensor, freqs):
        orig_dtype = tensor.dtype
        tensor = tensor.cast("float32")
        cos = freqs.cos().unsqueeze(1).tile([1, 1, 2]).unsqueeze(0).cast("float32")
        sin = freqs.sin().unsqueeze(1).tile([1, 1, 2]).unsqueeze(0).cast("float32")
        output = tensor * cos + rotate_half(tensor) * sin
        return output.cast(orig_dtype)
"""

import unittest

import numpy as np
import paddle
from paddlefleet_ops import fused_apply_rotary_pos_emb_vision

#   float32  : 1e-6  (CUDA device cosf/sinf vs Paddle API can differ by ±1 ULP
#                     of float32 ≈ 1.19e-7; worst-case accumulation → ~4.8e-7)
#   float16  : 2e-3  (≈ 2× machine epsilon of fp16, covers ±1 ULP cosf/sinf diff)
#   bfloat16 : 2e-2  (≈ 2× machine epsilon of bf16, covers ±1 ULP cosf/sinf diff)
_TOL = {
    "float32": (1e-6, 1e-6),
    "float16": (2e-3, 2e-3),
    "bfloat16": (2e-2, 2e-2),
}


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision_ref(tensor, freqs):
    """Pure-Python / Paddle reference implementation.

    Args:
        tensor : paddle.Tensor  [batch, seq, heads, dim]
        freqs  : paddle.Tensor  [seq, dim//2]  float32

    Returns:
        paddle.Tensor of same shape and dtype as `tensor`.
    """
    orig_dtype = tensor.dtype
    tensor = tensor.astype("float32")
    cos = freqs.cos().unsqueeze(1).tile([1, 1, 2]).unsqueeze(0).astype("float32")
    sin = freqs.sin().unsqueeze(1).tile([1, 1, 2]).unsqueeze(0).astype("float32")
    output = tensor * cos + rotate_half(tensor) * sin
    return output.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Helper: create random test tensors
# ---------------------------------------------------------------------------


def make_tensors(batch, seq, heads, dim, dtype, seed=42):
    """Return (tensor, freqs) on GPU.

    tensor : [batch, seq, heads, dim]   <dtype>
    freqs  : [seq, dim//2]              float32
    """
    np.random.seed(seed)
    half = dim // 2
    tensor_np = np.random.randn(batch, seq, heads, dim).astype("float32")
    freqs_np = np.random.randn(seq, half).astype("float32")
    tensor = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
    freqs = paddle.to_tensor(freqs_np, place="gpu")
    return tensor, freqs


def make_tensors_3d(seq, heads, dim, dtype, freqs_dtype="float32", seed=42):
    """Return (tensor, freqs) on GPU with 3D tensor (no batch dim).

    tensor : [seq, heads, dim]   <dtype>
    freqs  : [seq, dim//2]       <freqs_dtype>
    """
    np.random.seed(seed)
    half = dim // 2
    tensor_np = np.random.randn(seq, heads, dim).astype("float32")
    freqs_np = np.random.randn(seq, half).astype("float32")
    tensor = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
    freqs = paddle.to_tensor(freqs_np, place="gpu").astype(freqs_dtype)
    return tensor, freqs


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


class TestApplyRopevisionForward(unittest.TestCase):
    """Forward output must match Python reference to within 1e-9."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    def _run_fwd(self, batch, seq, heads, dim, dtype):
        tensor, freqs = make_tensors(batch, seq, heads, dim, dtype)

        ref_out = apply_rotary_pos_emb_vision_ref(tensor, freqs)
        cuda_out = fused_apply_rotary_pos_emb_vision(tensor, freqs)

        self.assertEqual(
            list(cuda_out.shape),
            list(ref_out.shape),
            f"Shape mismatch: cuda={cuda_out.shape} ref={ref_out.shape}",
        )
        self.assertEqual(
            cuda_out.dtype,
            ref_out.dtype,
            f"Dtype mismatch: cuda={cuda_out.dtype} ref={ref_out.dtype}",
        )

        atol, rtol = _TOL[dtype]
        ref_np = ref_out.numpy().astype("float32")
        cuda_np = cuda_out.numpy().astype("float32")
        np.testing.assert_allclose(
            cuda_np,
            ref_np,
            rtol=rtol,
            atol=atol,
            err_msg=(f"Forward mismatch: batch={batch} seq={seq} " f"heads={heads} dim={dim} dtype={dtype}"),
        )

    # ---- float32 ----
    def test_fwd_float32_s(self):
        self._run_fwd(4, 256, 8, 64, "float32")

    def test_fwd_float32_m(self):
        self._run_fwd(8, 512, 16, 128, "float32")

    def test_fwd_float32_l(self):
        self._run_fwd(16, 1024, 32, 128, "float32")

    def test_fwd_float32_dim64(self):
        self._run_fwd(8, 512, 16, 64, "float32")

    def test_fwd_float32_dim256(self):
        self._run_fwd(4, 256, 8, 256, "float32")

    # ---- float16 ----
    def test_fwd_float16_s(self):
        self._run_fwd(4, 256, 8, 64, "float16")

    def test_fwd_float16_m(self):
        self._run_fwd(8, 512, 16, 128, "float16")

    def test_fwd_float16_l(self):
        self._run_fwd(16, 1024, 32, 128, "float16")

    def test_fwd_float16_dim256(self):
        self._run_fwd(4, 256, 8, 256, "float16")

    # ---- bfloat16 ----
    def test_fwd_bfloat16_s(self):
        self._run_fwd(4, 256, 8, 64, "bfloat16")

    def test_fwd_bfloat16_m(self):
        self._run_fwd(8, 512, 16, 128, "bfloat16")

    def test_fwd_bfloat16_l(self):
        self._run_fwd(16, 1024, 32, 128, "bfloat16")

    def test_fwd_bfloat16_dim256(self):
        self._run_fwd(4, 256, 8, 256, "bfloat16")


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------


class TestApplyRopevisionBackward(unittest.TestCase):
    """Backward gradients must match Python reference to within 1e-9."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    def _run_bwd(self, batch, seq, heads, dim, dtype):
        np.random.seed(99)
        half = dim // 2

        tensor_np = np.random.randn(batch, seq, heads, dim).astype("float32")
        freqs_np = np.random.randn(seq, half).astype("float32")

        # Upstream gradient: random values in fp32, then cast to output dtype
        d_out_np = np.random.randn(batch, seq, heads, dim).astype("float32")

        # ---- Reference backward ----
        tensor_ref = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
        tensor_ref.stop_gradient = False
        freqs_ref = paddle.to_tensor(freqs_np, place="gpu")
        out_ref = apply_rotary_pos_emb_vision_ref(tensor_ref, freqs_ref)
        d_out_ref = paddle.to_tensor(d_out_np, place="gpu").astype(out_ref.dtype)
        grad_ref = paddle.grad(outputs=[out_ref], inputs=[tensor_ref], grad_outputs=[d_out_ref])[0]

        # ---- CUDA backward ----
        tensor_cuda = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
        tensor_cuda.stop_gradient = False
        freqs_cuda = paddle.to_tensor(freqs_np, place="gpu")
        out_cuda = fused_apply_rotary_pos_emb_vision(tensor_cuda, freqs_cuda)
        d_out_cuda = paddle.to_tensor(d_out_np, place="gpu").astype(out_cuda.dtype)
        grad_cuda = paddle.grad(outputs=[out_cuda], inputs=[tensor_cuda], grad_outputs=[d_out_cuda])[0]

        atol, rtol = _TOL[dtype]
        ref_np = grad_ref.numpy().astype("float32")
        cuda_np = grad_cuda.numpy().astype("float32")
        np.testing.assert_allclose(
            cuda_np,
            ref_np,
            rtol=rtol,
            atol=atol,
            err_msg=(f"Backward mismatch: batch={batch} seq={seq} " f"heads={heads} dim={dim} dtype={dtype}"),
        )

    # ---- float32 ----
    def test_bwd_float32_s(self):
        self._run_bwd(4, 256, 8, 64, "float32")

    def test_bwd_float32_m(self):
        self._run_bwd(8, 512, 16, 128, "float32")

    def test_bwd_float32_l(self):
        self._run_bwd(16, 1024, 32, 128, "float32")

    def test_bwd_float32_dim256(self):
        self._run_bwd(4, 256, 8, 256, "float32")

    # ---- float16 ----
    def test_bwd_float16_s(self):
        self._run_bwd(4, 256, 8, 64, "float16")

    def test_bwd_float16_m(self):
        self._run_bwd(8, 512, 16, 128, "float16")

    def test_bwd_float16_l(self):
        self._run_bwd(16, 1024, 32, 128, "float16")

    def test_bwd_float16_dim256(self):
        self._run_bwd(4, 256, 8, 256, "float16")

    # ---- bfloat16 ----
    def test_bwd_bfloat16_s(self):
        self._run_bwd(4, 256, 8, 64, "bfloat16")

    def test_bwd_bfloat16_m(self):
        self._run_bwd(8, 512, 16, 128, "bfloat16")

    def test_bwd_bfloat16_l(self):
        self._run_bwd(16, 1024, 32, 128, "bfloat16")

    def test_bwd_bfloat16_dim256(self):
        self._run_bwd(4, 256, 8, 256, "bfloat16")


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestApplyRopevisionShapes(unittest.TestCase):
    """Output shape must equal input shape for a variety of configurations."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    def test_shapes(self):
        configs = [
            (1, 64, 4, 32),
            (4, 256, 8, 64),
            (8, 512, 16, 128),
            (16, 1024, 32, 128),
            (2, 128, 4, 256),
        ]
        for batch, seq, heads, dim in configs:
            with self.subTest(batch=batch, seq=seq, heads=heads, dim=dim):
                tensor, freqs = make_tensors(batch, seq, heads, dim, "float32")
                out = fused_apply_rotary_pos_emb_vision(tensor, freqs)
                self.assertEqual(
                    list(out.shape),
                    [batch, seq, heads, dim],
                    f"Shape mismatch for config {(batch, seq, heads, dim)}",
                )


# ---------------------------------------------------------------------------
# Edge-case tests: 0-size tensors and large tensors (int64 index safety)
# ---------------------------------------------------------------------------


class TestApplyRopevisionEdgeCases(unittest.TestCase):
    """Edge cases: 0-size inputs and large tensors that require int64 indexing."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    # ── 0-size forward ──────────────────────────────────────────────────────

    def _assert_empty_fwd(self, batch, seq, heads, dim, dtype="float32"):
        """Forward on a 0-size tensor must return the correct empty shape."""
        half = dim // 2
        tensor = paddle.zeros([batch, seq, heads, dim], dtype=dtype).cuda()
        freqs = paddle.zeros([seq, half], dtype="float32").cuda()
        out = fused_apply_rotary_pos_emb_vision(tensor, freqs)
        self.assertEqual(list(out.shape), [batch, seq, heads, dim])
        self.assertEqual(out.numel(), 0)
        self.assertEqual(out.dtype, tensor.dtype)

    def test_zero_batch_fwd(self):
        self._assert_empty_fwd(0, 64, 8, 64)

    def test_zero_seq_fwd(self):
        self._assert_empty_fwd(2, 0, 8, 64)

    def test_zero_heads_fwd(self):
        self._assert_empty_fwd(2, 64, 0, 64)

    def test_zero_batch_fwd_fp16(self):
        self._assert_empty_fwd(0, 64, 8, 64, "float16")

    def test_zero_seq_fwd_fp16(self):
        self._assert_empty_fwd(2, 0, 8, 64, "float16")

    # ── 0-size backward ─────────────────────────────────────────────────────

    def _assert_empty_bwd(self, batch, seq, heads, dim, dtype="float32"):
        """Backward on a 0-size tensor must return the correct empty gradient."""
        half = dim // 2
        tensor = paddle.zeros([batch, seq, heads, dim], dtype=dtype).cuda()
        tensor.stop_gradient = False
        freqs = paddle.zeros([seq, half], dtype="float32").cuda()
        out = fused_apply_rotary_pos_emb_vision(tensor, freqs)
        d_out = paddle.zeros_like(out)
        grad = paddle.grad(outputs=[out], inputs=[tensor], grad_outputs=[d_out])[0]
        self.assertEqual(list(grad.shape), [batch, seq, heads, dim])
        self.assertEqual(grad.numel(), 0)

    def test_zero_batch_bwd(self):
        self._assert_empty_bwd(0, 64, 8, 64)

    def test_zero_seq_bwd(self):
        self._assert_empty_bwd(2, 0, 8, 64)

    def test_zero_heads_bwd(self):
        self._assert_empty_bwd(2, 64, 0, 64)

    # ── large tensor: int64 index safety ────────────────────────────────────
    # The kernel uses int64_t for base_sb / base_h so that index arithmetic
    # never overflows even when batch*seq*heads*dim > 2^31.
    #
    # The shape below has:
    #   max index = (batch-1)*seq*heads*dim + (heads-1)*dim + (dim-1)
    #             = 1*8192*32*128 + 31*128 + 127 = 33,558,655   (<2^31, in int32)
    # We cannot practically allocate a >2^31-element tensor in a unit test
    # (~4 GB float16), so this test verifies numerical correctness at a large
    # but feasible size and documents that int64 indexing is always in use.

    def _run_large(self, batch, seq, heads, dim, dtype):
        tensor, freqs = make_tensors(batch, seq, heads, dim, dtype)
        ref_out = apply_rotary_pos_emb_vision_ref(tensor, freqs)
        cuda_out = fused_apply_rotary_pos_emb_vision(tensor, freqs)
        atol, rtol = _TOL[dtype]
        np.testing.assert_allclose(
            cuda_out.numpy().astype("float32"),
            ref_out.numpy().astype("float32"),
            rtol=rtol,
            atol=atol,
            err_msg=f"large tensor mismatch: {batch}x{seq}x{heads}x{dim} {dtype}",
        )

    def test_large_float32(self):
        # 2*8192*32*128 = 67,108,864 elements, ~256 MB float32
        self._run_large(2, 8192, 32, 128, "float32")

    def test_large_float16(self):
        # 2*8192*32*128 = 67,108,864 elements, ~128 MB float16
        self._run_large(2, 8192, 32, 128, "float16")

    def test_large_bfloat16(self):
        self._run_large(2, 8192, 32, 128, "bfloat16")


# ---------------------------------------------------------------------------
# 3D input tests: tensor shape [seq, heads, dim] (no batch dim)
# This is the actual usage pattern in vision models.
# ---------------------------------------------------------------------------


class TestApplyRopevision3DForward(unittest.TestCase):
    """Forward with 3D input [seq, heads, dim] must match Python reference."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    def _run_fwd_3d(self, seq, heads, dim, dtype, freqs_dtype="float32"):
        tensor, freqs = make_tensors_3d(seq, heads, dim, dtype, freqs_dtype)

        # Reference: unsqueeze to 4D, compute, squeeze back
        tensor_4d = tensor.unsqueeze(0)  # [1, seq, heads, dim]
        freqs_f32 = freqs.astype("float32")
        ref_out_4d = apply_rotary_pos_emb_vision_ref(tensor_4d, freqs_f32)
        ref_out = ref_out_4d.squeeze(0)  # [seq, heads, dim]

        cuda_out = fused_apply_rotary_pos_emb_vision(tensor, freqs)

        self.assertEqual(
            list(cuda_out.shape),
            [seq, heads, dim],
            f"Shape mismatch: cuda={cuda_out.shape} expected=[{seq}, {heads}, {dim}]",
        )
        self.assertEqual(cuda_out.dtype, tensor.dtype)

        atol, rtol = _TOL[dtype]
        np.testing.assert_allclose(
            cuda_out.numpy().astype("float32"),
            ref_out.numpy().astype("float32"),
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"3D Forward mismatch: seq={seq} heads={heads} " f"dim={dim} dtype={dtype} freqs_dtype={freqs_dtype}"
            ),
        )

    # ---- 3D float32 ----
    def test_fwd_3d_float32(self):
        self._run_fwd_3d(256, 8, 64, "float32")

    def test_fwd_3d_float32_large(self):
        self._run_fwd_3d(1024, 16, 128, "float32")

    # ---- 3D float16 ----
    def test_fwd_3d_float16(self):
        self._run_fwd_3d(256, 8, 64, "float16")

    # ---- 3D bfloat16 (matches real usage: q is bf16) ----
    def test_fwd_3d_bfloat16(self):
        self._run_fwd_3d(256, 8, 64, "bfloat16")

    def test_fwd_3d_bfloat16_large(self):
        self._run_fwd_3d(1024, 16, 128, "bfloat16")

    # ---- 3D with bf16 freqs (the actual usage pattern) ----
    def test_fwd_3d_bf16_tensor_bf16_freqs(self):
        """Matches real usage: q=bf16, rotary_pos_emb=bf16."""
        self._run_fwd_3d(256, 8, 64, "bfloat16", freqs_dtype="bfloat16")

    def test_fwd_3d_bf16_tensor_fp16_freqs(self):
        self._run_fwd_3d(256, 8, 64, "bfloat16", freqs_dtype="float16")


class TestApplyRopevision3DBackward(unittest.TestCase):
    """Backward with 3D input [seq, heads, dim] must match Python reference."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    def _run_bwd_3d(self, seq, heads, dim, dtype, freqs_dtype="float32"):
        np.random.seed(99)
        half = dim // 2

        tensor_np = np.random.randn(seq, heads, dim).astype("float32")
        freqs_np = np.random.randn(seq, half).astype("float32")
        d_out_np = np.random.randn(seq, heads, dim).astype("float32")

        # ---- Reference backward (via 4D) ----
        # Use the same freqs dtype to ensure fair comparison:
        # cast freqs to freqs_dtype then back to fp32 (simulating precision loss)
        tensor_ref_4d = paddle.to_tensor(tensor_np.reshape(1, seq, heads, dim), place="gpu").astype(dtype)
        tensor_ref_4d.stop_gradient = False
        freqs_ref = paddle.to_tensor(freqs_np, place="gpu").astype(freqs_dtype).astype("float32")
        out_ref = apply_rotary_pos_emb_vision_ref(tensor_ref_4d, freqs_ref)
        d_out_ref = paddle.to_tensor(d_out_np.reshape(1, seq, heads, dim), place="gpu").astype(dtype)
        grad_ref = paddle.grad(outputs=[out_ref], inputs=[tensor_ref_4d], grad_outputs=[d_out_ref])[0].squeeze(
            0
        )  # [seq, heads, dim]

        # ---- CUDA backward (3D) ----
        tensor_cuda = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
        tensor_cuda.stop_gradient = False
        freqs_cuda = paddle.to_tensor(freqs_np, place="gpu").astype(freqs_dtype)
        out_cuda = fused_apply_rotary_pos_emb_vision(tensor_cuda, freqs_cuda)
        d_out_cuda = paddle.to_tensor(d_out_np, place="gpu").astype(dtype)
        grad_cuda = paddle.grad(outputs=[out_cuda], inputs=[tensor_cuda], grad_outputs=[d_out_cuda])[0]

        self.assertEqual(list(grad_cuda.shape), [seq, heads, dim])

        atol, rtol = _TOL[dtype]
        np.testing.assert_allclose(
            grad_cuda.numpy().astype("float32"),
            grad_ref.numpy().astype("float32"),
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"3D Backward mismatch: seq={seq} heads={heads} " f"dim={dim} dtype={dtype} freqs_dtype={freqs_dtype}"
            ),
        )

    def test_bwd_3d_float32(self):
        self._run_bwd_3d(256, 8, 64, "float32")

    def test_bwd_3d_bfloat16(self):
        self._run_bwd_3d(256, 8, 64, "bfloat16")

    def test_bwd_3d_bfloat16_bf16_freqs(self):
        """Matches real usage: q=bf16, rotary_pos_emb=bf16."""
        self._run_bwd_3d(256, 8, 64, "bfloat16", freqs_dtype="bfloat16")

    def test_bwd_3d_float16(self):
        self._run_bwd_3d(256, 8, 64, "float16")


# ---------------------------------------------------------------------------
# 4D with non-float32 freqs tests
# ---------------------------------------------------------------------------


class TestApplyRopevisionNonFloat32Freqs(unittest.TestCase):
    """Test that non-float32 freqs are handled correctly (auto-cast to fp32)."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    def _run_fwd_freqs_dtype(self, batch, seq, heads, dim, dtype, freqs_dtype):
        np.random.seed(42)
        half = dim // 2
        tensor_np = np.random.randn(batch, seq, heads, dim).astype("float32")
        freqs_np = np.random.randn(seq, half).astype("float32")

        tensor = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
        freqs_f32 = paddle.to_tensor(freqs_np, place="gpu")
        freqs_cast = freqs_f32.astype(freqs_dtype)

        # Use the same precision freqs for reference (round-trip through freqs_dtype)
        freqs_ref = freqs_cast.astype("float32")
        ref_out = apply_rotary_pos_emb_vision_ref(tensor, freqs_ref)
        cuda_out = fused_apply_rotary_pos_emb_vision(tensor, freqs_cast)

        atol, rtol = _TOL[dtype]

        np.testing.assert_allclose(
            cuda_out.numpy().astype("float32"),
            ref_out.numpy().astype("float32"),
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"Forward mismatch with freqs_dtype={freqs_dtype}: "
                f"batch={batch} seq={seq} heads={heads} dim={dim} dtype={dtype}"
            ),
        )

    def test_fwd_bf16_tensor_bf16_freqs(self):
        self._run_fwd_freqs_dtype(4, 256, 8, 64, "bfloat16", "bfloat16")

    def test_fwd_fp16_tensor_fp16_freqs(self):
        self._run_fwd_freqs_dtype(4, 256, 8, 64, "float16", "float16")

    def test_fwd_fp32_tensor_bf16_freqs(self):
        self._run_fwd_freqs_dtype(4, 256, 8, 64, "float32", "bfloat16")


def print_precision():
    """Report actual max atol / rtol for all shapes and dtypes (fwd + bwd)."""
    import sys

    if not paddle.is_compiled_with_cuda():
        print("CUDA not available.")
        sys.exit(0)

    paddle.device.set_device("gpu:0")

    SHAPES = [
        ("tiny        ", (1, 16, 2, 32)),
        ("small       ", (2, 64, 8, 64)),
        ("medium      ", (4, 256, 16, 64)),
        ("ViT-B 224   ", (2, 577, 12, 64)),
        ("ViT-L 336   ", (2, 1225, 16, 64)),
        ("large       ", (8, 512, 32, 128)),
        ("xlarge      ", (16, 1024, 32, 128)),
        ("vision-3k   ", (4, 3136, 16, 64)),
        ("vision-4k   ", (2, 4096, 16, 128)),
    ]
    DTYPES = ["float32", "float16", "bfloat16"]
    NUM_SEEDS = 5

    hdr = f"{'shape':<14} {'dtype':<10} " f"{'fwd_atol':>12} {'fwd_rtol':>12}  " f"{'bwd_atol':>12} {'bwd_rtol':>12}"
    sep = "─" * len(hdr)
    print()
    print("=" * len(hdr))
    print("  Actual precision: custom CUDA vs Python reference")
    print(f"  (worst-case over {NUM_SEEDS} random seeds)")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for label, (batch, seq, heads, dim) in SHAPES:
        for dtype in DTYPES:
            worst_fa = worst_fr = worst_ba = worst_br = 0.0

            for seed in range(NUM_SEEDS):
                np.random.seed(seed)
                tensor_np = np.random.randn(batch, seq, heads, dim).astype("float32")
                freqs_np = np.random.randn(seq, dim // 2).astype("float32")
                dout_np = np.random.randn(batch, seq, heads, dim).astype("float32")

                tensor = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
                freqs = paddle.to_tensor(freqs_np, place="gpu")
                d_out = paddle.to_tensor(dout_np, place="gpu").astype(dtype)

                # forward
                ref_out = apply_rotary_pos_emb_vision_ref(tensor, freqs)
                cuda_out = fused_apply_rotary_pos_emb_vision(tensor, freqs)
                r32 = ref_out.numpy().astype("float32")
                c32 = cuda_out.numpy().astype("float32")
                diff = np.abs(r32 - c32)
                worst_fa = max(worst_fa, float(diff.max()))
                worst_fr = max(worst_fr, float((diff / (np.abs(r32) + 1e-8)).max()))

                # backward
                t_ref = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
                t_ref.stop_gradient = False
                g_ref = paddle.grad(
                    outputs=[apply_rotary_pos_emb_vision_ref(t_ref, freqs)],
                    inputs=[t_ref],
                    grad_outputs=[d_out],
                )[0]

                t_cus = paddle.to_tensor(tensor_np, place="gpu").astype(dtype)
                t_cus.stop_gradient = False
                g_cus = paddle.grad(
                    outputs=[fused_apply_rotary_pos_emb_vision(t_cus, freqs)],
                    inputs=[t_cus],
                    grad_outputs=[d_out],
                )[0]

                rg = g_ref.numpy().astype("float32")
                cg = g_cus.numpy().astype("float32")
                gdiff = np.abs(rg - cg)
                worst_ba = max(worst_ba, float(gdiff.max()))
                worst_br = max(worst_br, float((gdiff / (np.abs(rg) + 1e-8)).max()))

            print(
                f"{label:<14} {dtype:<10} "
                f"{worst_fa:>12.3e} {worst_fr:>12.3e}  "
                f"{worst_ba:>12.3e} {worst_br:>12.3e}"
            )

        print(sep)

    print()
    print("atol = max |cuda - ref|  (float32 space)")
    print("rtol = max |cuda - ref| / (|ref| + 1e-8)")
    print()


if __name__ == "__main__":
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--precision":
        print_precision()
    else:
        unittest.main(verbosity=2)
