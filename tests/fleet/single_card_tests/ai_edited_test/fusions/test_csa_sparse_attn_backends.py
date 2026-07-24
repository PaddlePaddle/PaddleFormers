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

"""Verify TileLang and cuDNN CSA sparse attention backends agree."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import paddle

try:
    import paddlefleet_ops
    from paddleformers.fleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

    _HAS_FLASH_MLA = (
        paddlefleet_ops.is_flash_mla_available()
        and csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is not None
    )
except (ImportError, RuntimeError, AttributeError):
    _HAS_FLASH_MLA = False

try:
    import cudnn  # noqa: F401

    from paddleformers.fleet.cudnn_ops import csa_sparse_attn_bwd_cudnn

    _HAS_CUDNN_SPARSE_BWD = callable(csa_sparse_attn_bwd_cudnn)
except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError):
    _HAS_CUDNN_SPARSE_BWD = False


TEST_CASES = [
    (1, 128, 256, 64, 512, 64, "small-basic"),
    (2, 128, 256, 64, 512, 64, "batch2"),
    (4, 128, 256, 64, 512, 64, "batch4"),
    (1, 64, 128, 64, 512, 64, "tiny-seq"),
    (1, 256, 512, 64, 512, 128, "medium-seq"),
    (1, 512, 1024, 64, 512, 128, "large-seq"),
    (1, 1, 256, 64, 512, 64, "single-token"),
    (2, 1, 128, 64, 512, 64, "single-token-batch2"),
    (2, 256, 512, 64, 512, 192, "large-topk-192"),
    (1, 128, 512, 64, 512, 256, "large-topk-256"),
    (8, 64, 128, 64, 512, 64, "batch8"),
    (1, 1, 64, 64, 512, 64, "minimal"),
]

COS_THRESHOLDS = {
    "out": 0.99,
    "dq": 0.95,
    "dkv": 0.95,
    "d_sink": 0.95,
}

# Magnitude-sensitive relative-L2 ceilings (||tl-cu|| / ||cu||). Cosine is
# scale-invariant, so it passes even if the two backends diverge by a constant
# factor (exactly the class of bug -- e.g. a 128x gradient scale -- that shipped
# undetected). Since this compares TileLang vs cuDNN (two kernels, no fp32 ref)
# the bound guards against a scale divergence between the backends. Ceilings are
# ~2x the worst observed bf16-level error across all TEST_CASES.
REL_L2_THRESHOLDS = {
    "out": 5e-3,
    "dq": 3e-3,
    "dkv": 7e-4,
    "d_sink": 6e-3,
}


def cosine_sim(a, b):
    a_f = a.flatten().cast("float32")
    b_f = b.flatten().cast("float32")
    return float(
        paddle.nn.functional.cosine_similarity(
            a_f.unsqueeze(0), b_f.unsqueeze(0)
        )
    )


def max_abs_diff(a, b):
    return float((a.cast("float32") - b.cast("float32")).abs().max())


def rel_l2(a, b):
    a_f = a.flatten().cast("float32")
    b_f = b.flatten().cast("float32")
    return float(
        paddle.linalg.norm(a_f - b_f) / (paddle.linalg.norm(b_f) + 1e-12)
    )


def make_inputs(batch_size, seq_len, kv_seq_len, num_heads, head_dim, topk):
    q = paddle.randn([batch_size, seq_len, num_heads, head_dim]).cast(
        "bfloat16"
    )
    q.stop_gradient = False

    kv = paddle.randn([batch_size, kv_seq_len, head_dim]).cast("bfloat16")
    kv.stop_gradient = False

    attn_sink = paddle.randn([num_heads]).cast("float32") * 0.1
    attn_sink.stop_gradient = False

    topk_idxs = paddle.randint(0, kv_seq_len, [batch_size, seq_len, topk]).cast(
        "int32"
    )
    softmax_scale = 1.0 / (head_dim**0.5)
    return q, kv, attn_sink, topk_idxs, softmax_scale


def run_forward_backward(q, kv, attn_sink, topk_idxs, softmax_scale, backend):
    from paddleformers.fleet.fusions.csa_sparse_attn import csa_sparse_attn

    q_c = q.detach().clone()
    q_c.stop_gradient = False
    kv_c = kv.detach().clone()
    kv_c.stop_gradient = False
    attn_sink_c = attn_sink.detach().clone()
    attn_sink_c.stop_gradient = False

    out = csa_sparse_attn(
        q_c, kv_c, attn_sink_c, topk_idxs, softmax_scale, backend=backend
    )
    out.sum().backward()

    return out, q_c.grad, kv_c.grad, attn_sink_c.grad


def run_single_shape(
    batch_size, seq_len, kv_seq_len, num_heads, head_dim, topk
):
    q, kv, attn_sink, topk_idxs, softmax_scale = make_inputs(
        batch_size, seq_len, kv_seq_len, num_heads, head_dim, topk
    )

    out_tl, dq_tl, dkv_tl, dsink_tl = run_forward_backward(
        q, kv, attn_sink, topk_idxs, softmax_scale, backend="tilelang"
    )
    out_cu, dq_cu, dkv_cu, dsink_cu = run_forward_backward(
        q, kv, attn_sink, topk_idxs, softmax_scale, backend="cudnn"
    )

    if dsink_tl is None or dsink_cu is None:
        return False, {"d_sink": None}

    metrics = {
        "out": (
            cosine_sim(out_tl, out_cu),
            max_abs_diff(out_tl, out_cu),
            rel_l2(out_tl, out_cu),
        ),
        "dq": (
            cosine_sim(dq_tl, dq_cu),
            max_abs_diff(dq_tl, dq_cu),
            rel_l2(dq_tl, dq_cu),
        ),
        "dkv": (
            cosine_sim(dkv_tl, dkv_cu),
            max_abs_diff(dkv_tl, dkv_cu),
            rel_l2(dkv_tl, dkv_cu),
        ),
        "d_sink": (
            cosine_sim(dsink_tl, dsink_cu),
            max_abs_diff(dsink_tl, dsink_cu),
            rel_l2(dsink_tl, dsink_cu),
        ),
    }
    # Gate on BOTH a cosine floor (direction) and a rel-L2 ceiling (magnitude);
    # cosine alone is scale-blind and would pass a constant-factor divergence.
    passed = all(
        metrics[name][0] > COS_THRESHOLDS[name] for name in COS_THRESHOLDS
    ) and all(
        metrics[name][2] < REL_L2_THRESHOLDS[name] for name in REL_L2_THRESHOLDS
    )
    return passed, metrics


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(),
    "CSA sparse attention backend comparison requires CUDA",
)
@unittest.skipUnless(
    _HAS_FLASH_MLA and _HAS_CUDNN_SPARSE_BWD,
    "CSA sparse attention backend comparison requires FlashMLA and cuDNN sparse backward",
)
class TestCSASparseAttentionBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_device("gpu:0")
        except Exception as exc:
            raise unittest.SkipTest(f"gpu:0 is not available: {exc}")
        paddle.seed(2026)

    def test_tilelang_and_cudnn_backends_match(self):
        for case in TEST_CASES:
            (
                batch_size,
                seq_len,
                kv_seq_len,
                num_heads,
                head_dim,
                topk,
                label,
            ) = case
            with self.subTest(label=label):
                passed, metrics = run_single_shape(
                    batch_size,
                    seq_len,
                    kv_seq_len,
                    num_heads,
                    head_dim,
                    topk,
                )
                self.assertTrue(passed, f"{label} metrics={metrics}")
                paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Unit tests for import-time patches in csa_sparse_attn_bwd_cudnn.py
# (No GPU kernel dependency — always runs regardless of cudnn availability)
# ---------------------------------------------------------------------------

# Load module under test from source tree.
# CI does editable install so coverage tracks the source file automatically.
# We use spec_from_file_location + register in sys.modules so coverage
# (source=paddlefleet) recognizes the executed file path.
_bwd_mod = None
_BWD_CUDNN_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../paddleformers/fleet/cudnn_ops/attn/csa_sparse_attn_bwd_cudnn.py",
)
_BWD_CUDNN_PATH = os.path.abspath(_BWD_CUDNN_PATH)
_BWD_MODULE_NAME = (
    "paddleformers.fleet.cudnn_ops.attn.csa_sparse_attn_bwd_cudnn"
)


def _load_bwd_module():
    """Load the bwd_cudnn module, mocking paddlefleet_ops if needed."""
    spec = importlib.util.spec_from_file_location(
        _BWD_MODULE_NAME, _BWD_CUDNN_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_BWD_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    _bwd_mod = _load_bwd_module()
except ImportError:
    try:
        _mock_pflops = MagicMock()
        _mock_pflops.CUDNN_FRONTEND_HINT = "cudnn frontend not available"
        _mock_pflops.is_cudnn_frontend_available = MagicMock(return_value=False)
        with patch.dict(sys.modules, {"paddlefleet_ops": _mock_pflops}):
            _bwd_mod = _load_bwd_module()
    except (RuntimeError, AttributeError, ModuleNotFoundError, ImportError):
        _bwd_mod = None
except (RuntimeError, AttributeError, ModuleNotFoundError):
    _bwd_mod = None

if _bwd_mod is not None:
    _patch_cutlass_nvgpu = _bwd_mod._patch_cutlass_nvgpu
    _patch_paddle_nvtx_range = _bwd_mod._patch_paddle_nvtx_range
    _patch_paddle_stream_cuda_stream = _bwd_mod._patch_paddle_stream_cuda_stream
    _HAS_BWD_CUDNN_MODULE = True
else:
    _HAS_BWD_CUDNN_MODULE = False
    _patch_cutlass_nvgpu = None
    _patch_paddle_nvtx_range = None
    _patch_paddle_stream_cuda_stream = None


@unittest.skipUnless(
    _HAS_BWD_CUDNN_MODULE, "csa_sparse_attn_bwd_cudnn module unavailable"
)
class TestMemoryFormatPatch(unittest.TestCase):
    """Test _MemoryFormat sentinel objects (lines 42/45/48)."""

    def test_repr(self):
        fmt = paddle.contiguous_format
        self.assertEqual(repr(fmt), "paddle.contiguous_format")

    def test_hash(self):
        fmt = paddle.contiguous_format
        self.assertEqual(hash(fmt), hash("contiguous_format"))

    def test_eq_identity(self):
        fmt = paddle.contiguous_format
        self.assertEqual(fmt, fmt)

    def test_eq_different_object(self):
        self.assertNotEqual(paddle.contiguous_format, paddle.preserve_format)

    def test_all_formats_exist(self):
        for name in (
            "contiguous_format",
            "preserve_format",
            "channels_last",
            "channels_last_3d",
        ):
            self.assertTrue(hasattr(paddle, name), f"paddle.{name} missing")


@unittest.skipUnless(
    _HAS_BWD_CUDNN_MODULE, "csa_sparse_attn_bwd_cudnn module unavailable"
)
class TestPatchCutlassNvgpuException(unittest.TestCase):
    """Test _patch_cutlass_nvgpu exception path (lines 81-82)."""

    def test_import_error_is_silenced(self):
        with patch.dict(
            sys.modules,
            {
                "cutlass": None,
                "cutlass.cute": None,
                "cutlass.cute.nvgpu": None,
            },
        ):
            _patch_cutlass_nvgpu()

    def test_attribute_error_is_silenced(self):
        mock_nvgpu = MagicMock(spec=[])
        mock_cute = MagicMock()
        mock_cute.nvgpu = mock_nvgpu
        mock_cutlass = MagicMock()
        mock_cutlass.cute = mock_cute

        with patch.dict(
            sys.modules,
            {
                "cutlass": mock_cutlass,
                "cutlass.cute": mock_cute,
                "cutlass.cute.nvgpu": mock_nvgpu,
                "cutlass.cute.nvgpu.warpgroup": None,
            },
        ):
            _patch_cutlass_nvgpu()


@unittest.skipUnless(
    _HAS_BWD_CUDNN_MODULE, "csa_sparse_attn_bwd_cudnn module unavailable"
)
class TestPatchPaddleNvtxRange(unittest.TestCase):
    """Test _patch_paddle_nvtx_range logic (lines 103/107-109/111)."""

    def test_skip_when_range_exists(self):
        class FakeNvtx:
            range = staticmethod(lambda msg: None)
            range_push = staticmethod(lambda msg: None)
            range_pop = staticmethod(lambda: None)

        original_range = FakeNvtx.range
        with (
            patch.object(paddle.device, "nvtx", FakeNvtx, create=True),
            patch.object(paddle.cuda, "nvtx", None, create=True),
        ):
            _patch_paddle_nvtx_range()
        self.assertIs(FakeNvtx.range, original_range)

    def test_inject_range_formats_and_pushpop(self):
        class FakeNvtx:
            pushed = []
            popped = 0

            @staticmethod
            def range_push(msg):
                FakeNvtx.pushed.append(msg)

            @staticmethod
            def range_pop():
                FakeNvtx.popped += 1

        with (
            patch.object(paddle.device, "nvtx", FakeNvtx, create=True),
            patch.object(paddle.cuda, "nvtx", None, create=True),
        ):
            _patch_paddle_nvtx_range()

        with FakeNvtx.range("layer_{}", 3):
            pass

        self.assertEqual(FakeNvtx.pushed, ["layer_3"])
        self.assertEqual(FakeNvtx.popped, 1)

    def test_range_pop_on_exception(self):
        class FakeNvtx:
            popped = 0

            @staticmethod
            def range_push(msg):
                pass

            @staticmethod
            def range_pop():
                FakeNvtx.popped += 1

        with (
            patch.object(paddle.device, "nvtx", FakeNvtx, create=True),
            patch.object(paddle.cuda, "nvtx", None, create=True),
        ):
            _patch_paddle_nvtx_range()

        with self.assertRaises(RuntimeError), FakeNvtx.range("test"):
            raise RuntimeError("boom")

        self.assertEqual(FakeNvtx.popped, 1)


@unittest.skipUnless(
    _HAS_BWD_CUDNN_MODULE, "csa_sparse_attn_bwd_cudnn module unavailable"
)
class TestPatchPaddleStreamCudaStream(unittest.TestCase):
    """Test _patch_paddle_stream_cuda_stream assert guard (lines 136-140)."""

    def test_cuda_stream_property_returns_int(self):
        _patch_paddle_stream_cuda_stream()
        s = paddle.device.Stream()
        val = s.cuda_stream
        self.assertIsInstance(val, int)

    def test_non_cuda_stream_raises_assertion(self):
        from paddle.base import core

        _patch_paddle_stream_cuda_stream()
        s = paddle.device.Stream()

        mock_base = MagicMock(spec=[])
        self.assertNotIsInstance(mock_base, core.CUDAStream)

        with patch.object(
            type(s),
            "stream_base",
            new=property(lambda self: mock_base),
            create=True,
        ):
            with self.assertRaises(AssertionError) as ctx:
                _ = s.cuda_stream
            self.assertIn("only available for CUDA streams", str(ctx.exception))


@unittest.skipUnless(
    _HAS_BWD_CUDNN_MODULE, "csa_sparse_attn_bwd_cudnn module unavailable"
)
class TestCsaSparseAttnBwdCudnnFunction(unittest.TestCase):
    """Test csa_sparse_attn_bwd_cudnn function body (lines 148-176)."""

    def test_raises_when_cudnn_unavailable(self):
        """Cover _require_cudnn_frontend raising ImportError (line 159-160)."""
        csa_fn = _bwd_mod.csa_sparse_attn_bwd_cudnn
        with (
            patch.object(
                _bwd_mod, "is_cudnn_frontend_available", return_value=False
            ),
            self.assertRaises(ImportError),
        ):
            csa_fn(None, None, None, None, None, None, None)

    def test_calls_wrapper_and_returns_tuple(self):
        """Cover the import + call + return path (lines 160-176)."""
        fake_result = {
            "dq": "mock_dq",
            "dkv": "mock_dkv",
            "d_sink": "mock_d_sink",
        }
        csa_fn = _bwd_mod.csa_sparse_attn_bwd_cudnn

        with (
            patch.object(
                _bwd_mod, "is_cudnn_frontend_available", return_value=True
            ),
            patch.dict(
                sys.modules,
                {
                    "paddlefleet_ops.cudnn.deepseek_sparse_attention.sparse_attention_backward.api": MagicMock(
                        sparse_attention_backward_wrapper=MagicMock(
                            return_value=fake_result
                        )
                    ),
                },
            ),
        ):
            dq, dkv, d_sink = csa_fn(
                "q",
                "kv",
                "out",
                "dout",
                "lse",
                "attn_sink",
                "topk",
                softmax_scale=0.1,
                topk_length=64,
            )

        self.assertEqual(dq, "mock_dq")
        self.assertEqual(dkv, "mock_dkv")
        self.assertEqual(d_sink, "mock_d_sink")
