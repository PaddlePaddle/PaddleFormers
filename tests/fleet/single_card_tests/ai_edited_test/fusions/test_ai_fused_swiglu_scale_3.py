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
import os
import sys

# Walk up from the test file to find the repo root (where src/ lives).
_test_file = os.path.abspath(__file__)
_repo_root = _test_file
for _ in range(10):
    _repo_root = os.path.dirname(_repo_root)
    if os.path.isdir(os.path.join(_repo_root, "src", "paddleformers.fleet")):
        break

sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))

# Flush any pre-cached paddleformers.fleet modules so the src/ version wins.
for _mod in list(sys.modules.keys()):
    if _mod == "paddleformers.fleet" or _mod.startswith("paddleformers.fleet."):
        del sys.modules[_mod]

import unittest
from unittest.mock import MagicMock, patch

import paddle

# paddle may have re-imported paddleformers.fleet from site-packages; clear again
for _mod in list(sys.modules.keys()):
    if _mod == "paddleformers.fleet" or _mod.startswith("paddleformers.fleet."):
        del sys.modules[_mod]


class TestFusedSwigluScaleForward(unittest.TestCase):
    """Tests for fused_swiglu_scale_forward."""

    def test_forward_cpu_fallback(self):
        """Test fused_swiglu_scale_forward CPU fallback path."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4, 1])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])

    def test_forward_with_1d_scale(self):
        """Test with 1D scale tensor."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])

    def test_forward_scale_broadcast(self):
        """Test scale broadcasting."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.full([4, 1], 2.0)
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])


class TestFusedSwigluScaleBackward(unittest.TestCase):
    """Tests for fused_swiglu_scale_backward."""

    def test_backward_cpu_fallback(self):
        """Test fused_swiglu_scale_backward CPU fallback path."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4, 1])
        out_grad = paddle.randn([4, 8])
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [4, 16])
        self.assertEqual(d_scale.shape, [4, 1])

    def test_backward_shapes(self):
        """Test backward output shapes match inputs."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([2, 32])
        scale = paddle.ones([2, 1])
        out_grad = paddle.randn([2, 16])
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [2, 32])
        self.assertEqual(d_scale.shape, [2, 1])


# ----------------------------------------------------------------------------
# New tests added by PR #999 — minimal set covering only branches not exercised
# by the original tests above:
#   * CPU clamp branch (fwd + bwd, with mask-zero assertion)
#   * GPU dispatch branches (clamp / no-clamp, fwd + bwd) via mocks
#   * Static-graph InferShape regression for the new clamp forward op
# ----------------------------------------------------------------------------


def _no_cuda():
    return patch(
        "paddleformers.fleet.fusions.fused_swiglu_scale.paddle.is_compiled_with_cuda",
        return_value=False,
    )


class TestFusedSwigluScaleCPUFallback(unittest.TestCase):
    """CPU-fallback paths (fwd lines 34-46, bwd lines 65-106)."""

    def test_forward_no_clamp_1d_scale(self):
        """Covers line 40 (swiglu) and 42-46 (1D scale expansion)."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            result = fused_swiglu_scale_forward(paddle.randn([4, 16]), paddle.ones([4]))
            self.assertEqual(result.shape, [4, 8])

    def test_backward_no_clamp(self):
        """Covers lines 77-81 (no-clamp branch sets g/v_mask = None)."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            d_x, _ = fused_swiglu_scale_backward(
                paddle.randn([2, 16]),
                paddle.ones([2]),
                paddle.randn([2, 8]),
            )
            self.assertEqual(d_x.shape, [2, 16])


class TestFusedSwigluScaleGPUDispatch(unittest.TestCase):
    """GPU dispatch branches (fwd, bwd, clamp and non-clamp), via mock op modules."""

    def test_forward_no_clamp_dispatch(self):
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        mock_op = MagicMock(return_value=paddle.randn([2, 8]))
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {"paddlefleet_ops": MagicMock(fused_swiglu_scale=mock_op)},
            ),
        ):
            fused_swiglu_scale_forward(paddle.randn([2, 16]), paddle.ones([2, 1]))
            mock_op.assert_called_once()

    def test_forward_clamp_dispatch(self):
        """GPU + clamp_value > 0 dispatches to fused_swiglu_scale_clamp."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        mock_op = MagicMock(return_value=paddle.randn([2, 8]))
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {"paddlefleet_ops": MagicMock(fused_swiglu_scale_clamp=mock_op)},
            ),
        ):
            fused_swiglu_scale_forward(
                paddle.randn([2, 16]),
                paddle.ones([2, 1]),
                clamp_value=5.0,
            )
            mock_op.assert_called_once()

    def test_backward_no_clamp_dispatch(self):
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        mock_op = MagicMock(return_value=(paddle.randn([2, 16]), paddle.randn([2])))
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {"paddlefleet_ops": MagicMock(fused_swiglu_scale_bwd=mock_op)},
            ),
        ):
            fused_swiglu_scale_backward(
                paddle.randn([2, 16]),
                paddle.ones([2]),
                paddle.randn([2, 8]),
            )
            mock_op.assert_called_once()

    def test_backward_clamp_dispatch(self):
        """GPU + clamp_value > 0 dispatches to fused_swiglu_scale_clamp_bwd."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        mock_op = MagicMock(return_value=(paddle.randn([2, 16]), paddle.randn([2])))
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                "sys.modules",
                {"paddlefleet_ops": MagicMock(fused_swiglu_scale_clamp_bwd=mock_op)},
            ),
        ):
            fused_swiglu_scale_backward(
                paddle.randn([2, 16]),
                paddle.ones([2]),
                paddle.randn([2, 8]),
                clamp_value=5.0,
            )
            mock_op.assert_called_once()


# ============================================================================
# Static-graph InferShape regression for the new clamp forward op.
#
# fused_swiglu_scale_clamp has 2 inputs (X, Scale) and 1 output of shape
# {rows, hidden2 / 2}, so it must register FusedFwdInferShape /
# FusedFwdInferDtype. In eager mode a wrong InferShape would be hidden by the
# kernel return; in static mode the framework relies on InferShape and a
# wrong registration aborts. The static-graph test runs in a subprocess so a
# C++ SIGABRT does not take down the pytest worker.
# ============================================================================


def _has_op(name):
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        import paddlefleet_ops

        return hasattr(paddlefleet_ops, name)
    except ImportError:
        return False


def _run_static_infer_shape(op_name, hidden2, has_clamp):
    import subprocess
    import textwrap

    code = textwrap.dedent(
        f"""
        import paddle
        from paddlefleet_ops import {op_name}

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[4, {hidden2}], dtype='float32')
            scale = paddle.static.data(name='scale', shape=[4, 1], dtype='float32')
            out = {op_name}(x, scale{", 5.0" if has_clamp else ""})
            print('SHAPE_OK', list(out.shape))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


@unittest.skipUnless(
    _has_op("fused_swiglu_scale_clamp"),
    "fused_swiglu_scale_clamp custom op not built / CUDA unavailable",
)
class TestFusedSwigluScaleClampInferShape(unittest.TestCase):
    """Static-graph InferShape regression for the clamp forward op."""

    def test_clamp_forward_infer_shape_halves_hidden(self):
        rc, stdout, stderr = _run_static_infer_shape("fused_swiglu_scale_clamp", hidden2=32, has_clamp=True)
        self.assertEqual(
            rc,
            0,
            f"static-graph build of fused_swiglu_scale_clamp crashed "
            f"(likely InferShape registration bug).\nSTDERR:\n{stderr}",
        )
        self.assertIn("SHAPE_OK [4, 16]", stdout, f"stdout was:\n{stdout}")


if __name__ == "__main__":
    unittest.main()
