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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle


def _make_node(use_accuracy_compatible):
    from paddleformers.fleet.transformer.moe.fp8_utils import (
        ExpertsGroupGemmContiguousNode,
    )

    custom_map = MagicMock()
    custom_map.experts = [MagicMock()]
    return ExpertsGroupGemmContiguousNode(
        custom_map,
        use_fp8_mlp=False,
        moe_expert_fusion=False,
        use_accuracy_compatible=use_accuracy_compatible,
    )


def _make_node_activation(use_accuracy_compatible, activation_type):
    from paddleformers.fleet.transformer.moe.fp8_utils import (
        ExpertsGroupGemmContiguousNode,
    )

    custom_map = MagicMock()
    custom_map.experts = [MagicMock()]
    return ExpertsGroupGemmContiguousNode(
        custom_map,
        use_fp8_mlp=False,
        moe_expert_fusion=False,
        use_accuracy_compatible=use_accuracy_compatible,
        activation_type=activation_type,
    )


def _make_grouped_node(use_accuracy_compatible, activation_type="swiglu"):
    """Grouped-gemm bf16 node: moe_expert_fusion=True + use_fp8_mlp=False, so
    is_split_group_gemm is False (a public MoELayer config)."""
    from paddleformers.fleet.transformer.moe.fp8_utils import (
        ExpertsGroupGemmContiguousNode,
    )

    custom_map = MagicMock()
    custom_map.grouped_gemm_experts = MagicMock()
    return ExpertsGroupGemmContiguousNode(
        custom_map,
        use_fp8_mlp=False,
        moe_expert_fusion=True,
        use_accuracy_compatible=use_accuracy_compatible,
        activation_type=activation_type,
    )


class TestExpertsGroupGemmUseAccuracyCompatibleInit(unittest.TestCase):
    """The flag must be stored on the node and default to False."""

    def test_flag_stored_true(self):
        node = _make_node(True)
        self.assertTrue(node.use_accuracy_compatible)

    def test_flag_stored_false(self):
        node = _make_node(False)
        self.assertFalse(node.use_accuracy_compatible)

    def test_flag_default_false(self):
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map, use_fp8_mlp=False, moe_expert_fusion=False
        )
        self.assertFalse(node.use_accuracy_compatible)


class TestBwdGateUpInputBf16UseAccuracyCompatible(unittest.TestCase):
    """Cover the per-expert matmul vs F.linear branch in
    bwd_gate_up_input_bf16."""

    def test_branches_match(self):
        """matmul (compatible) and F.linear (default) must give identical
        dx for the split-group bf16 backward."""
        paddle.seed(0)
        do1 = paddle.randn([3, 4], dtype=paddle.bfloat16)
        # expert_w1[i] is [K, N]; the code uses expert_w1[i].T -> [N, K]
        expert_w1 = [
            paddle.randn([5, 4], dtype=paddle.bfloat16),
            paddle.randn([5, 4], dtype=paddle.bfloat16),
        ]

        node_compat = _make_node(True)
        node_compat.tokens_per_expert = [2, 1]
        node_default = _make_node(False)
        node_default.tokens_per_expert = [2, 1]

        dx_compat = node_compat.bwd_gate_up_input_bf16(do1, expert_w1)
        dx_default = node_default.bwd_gate_up_input_bf16(do1, expert_w1)

        self.assertEqual(dx_compat.shape, [3, 5])
        np.testing.assert_array_equal(
            dx_compat.astype("float32").numpy(),
            dx_default.astype("float32").numpy(),
        )

    def test_compatible_matches_reference_matmul(self):
        """The compatible branch must equal a manual per-expert matmul."""
        paddle.seed(1)
        do1 = paddle.randn([3, 4], dtype=paddle.bfloat16)
        expert_w1 = [
            paddle.randn([5, 4], dtype=paddle.bfloat16),
            paddle.randn([5, 4], dtype=paddle.bfloat16),
        ]

        node = _make_node(True)
        node.tokens_per_expert = [2, 1]
        dx = node.bwd_gate_up_input_bf16(do1, expert_w1)

        ref0 = paddle.matmul(do1[:2], expert_w1[0].T.contiguous())
        ref1 = paddle.matmul(do1[2:], expert_w1[1].T.contiguous())
        ref = paddle.concat([ref0, ref1], axis=0)
        np.testing.assert_array_equal(
            dx.astype("float32").numpy(), ref.astype("float32").numpy()
        )

    def test_skips_zero_token_expert(self):
        """An expert with zero tokens is skipped; remaining tokens still
        produce the correct dx in the compatible branch."""
        paddle.seed(2)
        do1 = paddle.randn([2, 4], dtype=paddle.bfloat16)
        expert_w1 = [
            paddle.randn([5, 4], dtype=paddle.bfloat16),
            paddle.randn([5, 4], dtype=paddle.bfloat16),
        ]

        node = _make_node(True)
        node.tokens_per_expert = [0, 2]
        dx = node.bwd_gate_up_input_bf16(do1, expert_w1)

        ref = paddle.matmul(do1, expert_w1[1].T.contiguous())
        np.testing.assert_array_equal(
            dx.astype("float32").numpy(), ref.astype("float32").numpy()
        )


class TestBwdDownInputBf16UseAccuracyCompatible(unittest.TestCase):
    """Cover the per-expert matmul vs F.linear branch in
    bwd_down_input_bf16. The downstream fused swiglu ops are mocked so we can
    isolate and assert the ``do2_s`` produced by the branch under test."""

    def _inputs(self):
        paddle.seed(3)
        total, hidden, inter = 3, 4, 2
        unzipped_grad = paddle.randn([total, hidden], dtype=paddle.bfloat16)
        # expert_w2[i] is [inter, hidden]; code uses expert_w2[i].T -> [hidden, inter]
        expert_w2 = [
            paddle.randn([inter, hidden], dtype=paddle.bfloat16),
            paddle.randn([inter, hidden], dtype=paddle.bfloat16),
        ]
        o1 = paddle.randn([total, 2 * inter], dtype=paddle.bfloat16)
        unzipped_probs = paddle.ones([total, 1], dtype=paddle.bfloat16)
        return unzipped_grad, expert_w2, o1, unzipped_probs

    def _run_capture_do2s(self, flag, unzipped_grad, expert_w2, o1, probs):
        """Run bwd_down_input_bf16 with the swiglu ops mocked, returning the
        ``do2_s`` tensor consumed by the backward.

        With use_accuracy_compatible=True the split-group backward no longer
        routes through ``fused_swiglu_scale_backward``; instead it rebuilds the
        SwiGLU graph in fp32 and calls ``paddle.autograd.backward`` with
        ``do2_s`` as the upstream gradient. Capture from both sinks so the
        helper works for the compatible (autograd) and default (fused kernel)
        paths alike."""
        node = _make_node(flag)
        node.tokens_per_expert = [2, 1]

        captured = {}

        def fake_backward(x, scale, out_grad):
            captured["do2_s"] = out_grad
            return (
                paddle.zeros_like(o1),
                paddle.zeros_like(probs),
            )

        real_autograd_backward = paddle.autograd.backward

        def fake_autograd_backward(tensors, grad_tensors=None, *args, **kwargs):
            if grad_tensors:
                captured["do2_s"] = grad_tensors[0]
            return real_autograd_backward(
                tensors, grad_tensors, *args, **kwargs
            )

        with (
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.fused_swiglu_scale_forward",
                MagicMock(return_value=paddle.zeros([o1.shape[0], 2])),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.fused_swiglu_scale_backward",
                MagicMock(side_effect=fake_backward),
            ),
            patch(
                "paddle.autograd.backward",
                MagicMock(side_effect=fake_autograd_backward),
            ),
        ):
            node.bwd_down_input_bf16(expert_w2, unzipped_grad, o1, probs)
        return captured["do2_s"]

    def test_do2s_branches_match(self):
        """matmul (compatible) and F.linear (default) must give identical
        do2_s for the split-group bf16 backward."""
        unzipped_grad, expert_w2, o1, probs = self._inputs()

        do2s_compat = self._run_capture_do2s(
            True, unzipped_grad, expert_w2, o1, probs
        )
        do2s_default = self._run_capture_do2s(
            False, unzipped_grad, expert_w2, o1, probs
        )

        self.assertEqual(do2s_compat.shape, [3, 2])
        np.testing.assert_array_equal(
            do2s_compat.astype("float32").numpy(),
            do2s_default.astype("float32").numpy(),
        )

    def test_do2s_matches_reference_matmul(self):
        """The compatible branch must equal a manual per-expert matmul."""
        unzipped_grad, expert_w2, o1, probs = self._inputs()

        do2s = self._run_capture_do2s(True, unzipped_grad, expert_w2, o1, probs)

        ref0 = paddle.matmul(unzipped_grad[:2], expert_w2[0].T.contiguous())
        ref1 = paddle.matmul(unzipped_grad[2:], expert_w2[1].T.contiguous())
        ref = paddle.concat([ref0, ref1], axis=0)
        np.testing.assert_array_equal(
            do2s.astype("float32").numpy(), ref.astype("float32").numpy()
        )


class TestGeGLUAccuracyCompatibleActivation(unittest.TestCase):
    """use_accuracy_compatible must NOT force the SwiGLU (silu) formula onto
    GeGLU experts; GeGLU must keep using gelu even when the flag is set."""

    def test_geglu_uses_gelu_not_silu(self):
        import paddle.nn.functional as F

        paddle.seed(7)
        tokens, hidden = 3, 4
        # o1 last dim = 2 * hidden so chunk halves are [tokens, hidden];
        # identity down-proj weight makes o3 == o2 (the activation output).
        o1 = paddle.randn([tokens, 2 * hidden], dtype=paddle.bfloat16)
        probs = paddle.rand([tokens, 1], dtype=paddle.bfloat16)
        expert_w2 = [paddle.eye(hidden, dtype=paddle.bfloat16)]

        node = _make_node_activation(True, "geglu")
        node.tokens_per_expert = [tokens]
        o3 = node.fwd_down_bf16(o1, probs, expert_w2)

        gate, up = paddle.chunk(o1, 2, axis=-1)
        gelu_ref = ((F.gelu(gate, approximate=True) * up) * probs).cast(
            o1.dtype
        )
        silu_ref = (
            F.silu(gate.astype("float32"))
            * up.astype("float32")
            * probs.astype("float32")
        ).astype(o1.dtype)

        np.testing.assert_array_equal(
            o3.astype("float32").numpy(), gelu_ref.astype("float32").numpy()
        )
        # gelu and silu differ, so the fix must not collapse to silu.
        self.assertFalse(
            np.array_equal(
                o3.astype("float32").numpy(),
                silu_ref.astype("float32").numpy(),
            )
        )


class TestFwdDownBf16ActivationBranches(unittest.TestCase):
    """Cover the remaining fwd_down_bf16 activation branches:
    - accuracy_compatible=True + SwiGLU -> the silu fp32 path
    - accuracy_compatible=False + GeGLU -> the gelu bf16 path
    An identity down-proj weight (eye) makes o3 == o2, so o3 equals the
    activation output and can be checked against a hand-written reference."""

    def test_swiglu_accuracy_compatible_uses_silu_fp32(self):
        import paddle.nn.functional as F

        paddle.seed(9)
        tokens, hidden = 3, 4
        o1 = paddle.randn([tokens, 2 * hidden], dtype=paddle.bfloat16)
        probs = paddle.rand([tokens, 1], dtype=paddle.bfloat16)
        expert_w2 = [paddle.eye(hidden, dtype=paddle.bfloat16)]

        node = _make_node_activation(True, "swiglu")
        node.tokens_per_expert = [tokens]
        o3 = node.fwd_down_bf16(o1, probs, expert_w2)

        gate, up = paddle.chunk(o1, 2, axis=-1)
        # SwiGLU accuracy-compatible: promote to fp32, round once.
        silu_ref = (
            F.silu(gate.astype("float32"))
            * up.astype("float32")
            * probs.astype("float32")
        ).astype(o1.dtype)
        np.testing.assert_array_equal(
            o3.astype("float32").numpy(), silu_ref.astype("float32").numpy()
        )

    def test_geglu_non_accuracy_compatible_uses_gelu(self):
        import paddle.nn.functional as F

        paddle.seed(9)
        tokens, hidden = 3, 4
        o1 = paddle.randn([tokens, 2 * hidden], dtype=paddle.bfloat16)
        # This branch scales with unzipped_probs.unsqueeze(-1), so probs is 1-D.
        probs = paddle.rand([tokens], dtype=paddle.bfloat16)
        expert_w2 = [paddle.eye(hidden, dtype=paddle.bfloat16)]

        node = _make_node_activation(False, "geglu")
        node.tokens_per_expert = [tokens]
        o3 = node.fwd_down_bf16(o1, probs, expert_w2)

        gate, up = paddle.chunk(o1, 2, axis=-1)
        gelu_ref = (
            (F.gelu(gate, approximate=True) * up) * probs.unsqueeze(-1)
        ).cast(o1.dtype)
        silu_ref = (
            F.silu(gate.astype("float32"))
            * up.astype("float32")
            * probs.unsqueeze(-1).astype("float32")
        ).astype(o1.dtype)

        np.testing.assert_array_equal(
            o3.astype("float32").numpy(), gelu_ref.astype("float32").numpy()
        )
        # gelu and silu differ, so this branch must not collapse to silu.
        self.assertFalse(
            np.array_equal(
                o3.astype("float32").numpy(),
                silu_ref.astype("float32").numpy(),
            )
        )


class TestGroupedGemmBf16AccuracyCompatibleForwardBackwardPairing(
    unittest.TestCase
):
    """Grouped-gemm (moe_expert_fusion=True, use_fp8_mlp=False) bf16 forward
    and backward must use the SAME math path when use_accuracy_compatible=True.

    The backward fp32-autograd branch is gated on is_split_group_gemm (False
    here), so grouped-gemm stays on the fused backward. The forward must
    therefore also keep the fused SwiGLU kernel instead of the hand-written
    fp32 silu path, otherwise forward (fp32 round-once) and backward (fused
    kernel) would disagree."""

    def _inputs(self):
        paddle.seed(4)
        tokens, inter = 3, 2
        o1 = paddle.randn([tokens, 2 * inter], dtype=paddle.bfloat16)
        probs = paddle.rand([tokens], dtype=paddle.bfloat16)
        return tokens, inter, o1, probs

    def test_forward_uses_fused_kernel_not_fp32_silu(self):
        tokens, inter, o1, probs = self._inputs()
        hidden = 4

        node = _make_grouped_node(True, "swiglu")
        node.tokens_per_expert = [tokens]
        self.assertFalse(node.is_split_group_gemm)

        fused_out = paddle.randn([tokens, inter], dtype=paddle.bfloat16)
        mock_fwd = MagicMock(return_value=fused_out)

        with (
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.fused_swiglu_scale_forward",
                mock_fwd,
            ),
            patch(
                "paddle.incubate.nn.functional.batched_gemm",
                MagicMock(
                    return_value=paddle.zeros(
                        [tokens, hidden], dtype=paddle.bfloat16
                    )
                ),
            ),
        ):
            node.fwd_down_bf16(o1, probs, MagicMock())

        # The fused forward must be invoked; the fp32 silu branch never calls it.
        mock_fwd.assert_called_once()

    def test_backward_uses_fused_kernel(self):
        tokens, inter, o1, probs = self._inputs()

        node = _make_grouped_node(True, "swiglu")
        node.tokens_per_expert = [tokens]

        # do2_s handed to the down-proj backward.
        unzipped_grad = paddle.randn([tokens, inter], dtype=paddle.bfloat16)
        # grouped-gemm bf16 backward computes do2_s via batched_gemm.
        do2_s = paddle.randn([tokens, inter], dtype=paddle.bfloat16)

        fused_backward = MagicMock(
            return_value=(
                paddle.zeros_like(o1),
                paddle.zeros([tokens, 1], dtype=paddle.bfloat16),
            )
        )

        with (
            patch(
                "paddle.incubate.nn.functional.batched_gemm",
                MagicMock(return_value=do2_s),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.fused_swiglu_scale_forward",
                MagicMock(
                    return_value=paddle.zeros(
                        [tokens, inter], dtype=paddle.bfloat16
                    )
                ),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.fused_swiglu_scale_backward",
                fused_backward,
            ),
        ):
            node.bwd_down_input_bf16(MagicMock(), unzipped_grad, o1, probs)

        # grouped-gemm must reach the fused backward (fp32-autograd branch is
        # gated on is_split_group_gemm), matching the fused forward above.
        fused_backward.assert_called_once()


class TestGeGLUBwdDownInputBf16AccuracyCompatible(unittest.TestCase):
    """GeGLU backward must NOT be captured by the SwiGLU fp32-autograd branch
    in bwd_down_input_bf16: do1 / probs_grad must follow the GeGLU (gelu)
    formula, not the SwiGLU (silu) one, even when use_accuracy_compatible=True
    and the split-group path is active."""

    def _inputs(self):
        paddle.seed(5)
        total, hidden, inter = 3, 4, 2
        unzipped_grad = paddle.randn([total, hidden], dtype=paddle.bfloat16)
        # expert_w2[i] is [inter, hidden]; code uses expert_w2[i].T -> [hidden, inter]
        expert_w2 = [
            paddle.randn([inter, hidden], dtype=paddle.bfloat16),
            paddle.randn([inter, hidden], dtype=paddle.bfloat16),
        ]
        o1 = paddle.randn([total, 2 * inter], dtype=paddle.bfloat16)
        probs = paddle.rand([total, 1], dtype=paddle.bfloat16)
        return unzipped_grad, expert_w2, o1, probs

    def _reference_do2s(self, unzipped_grad, expert_w2, split):
        # The accuracy-compatible split-group backward derives do2_s via a
        # per-expert matmul; reproduce it so the reference gradients below use
        # exactly the same upstream gradient the branch under test consumes.
        r0 = paddle.matmul(unzipped_grad[:split], expert_w2[0].T.contiguous())
        r1 = paddle.matmul(unzipped_grad[split:], expert_w2[1].T.contiguous())
        return paddle.concat([r0, r1], axis=0)

    def _autograd_grads(self, o1, probs, do2_s, activation):
        import paddle.nn.functional as F

        gate, up = paddle.chunk(o1, 2, axis=-1)
        g = gate.astype("float32").detach()
        u = up.astype("float32").detach()
        s = probs.astype("float32").detach()
        for t in (g, u, s):
            t.stop_gradient = False
        if activation == "geglu":
            o2 = (F.gelu(g, approximate=True) * u) * s
        else:
            o2 = F.silu(g) * u * s
        paddle.autograd.backward([o2], [do2_s.astype("float32").detach()])
        do1 = paddle.concat([g.grad, u.grad], axis=-1).astype(o1.dtype)
        probs_grad = s.grad.reshape(probs.shape).astype(probs.dtype)
        return do1, probs_grad

    def test_backward_uses_gelu_not_silu(self):
        unzipped_grad, expert_w2, o1, probs = self._inputs()

        node = _make_node_activation(True, "geglu")
        node.tokens_per_expert = [2, 1]
        do1, _o2_s, probs_grad = node.bwd_down_input_bf16(
            expert_w2, unzipped_grad, o1, probs
        )

        do2_s = self._reference_do2s(unzipped_grad, expert_w2, 2)
        do1_gelu, pg_gelu = self._autograd_grads(o1, probs, do2_s, "geglu")
        do1_silu, _ = self._autograd_grads(o1, probs, do2_s, "silu")

        # do1 / probs_grad must match the GeGLU reference.
        np.testing.assert_allclose(
            do1.astype("float32").numpy(),
            do1_gelu.astype("float32").numpy(),
            atol=3e-2,
            rtol=3e-2,
        )
        np.testing.assert_allclose(
            probs_grad.astype("float32").numpy(),
            pg_gelu.astype("float32").numpy(),
            atol=3e-2,
            rtol=3e-2,
        )
        # gelu and silu gradients differ, so a regression that routes GeGLU
        # through the SwiGLU branch would make do1 match the silu reference.
        self.assertFalse(
            np.allclose(
                do1.astype("float32").numpy(),
                do1_silu.astype("float32").numpy(),
                atol=3e-2,
                rtol=3e-2,
            )
        )


class TestSwiGLUAccuracyCompatibleClamp(unittest.TestCase):
    """The accuracy-compatible SwiGLU fp32 forward/backward (split-group) must
    honor activation_func_clamp_value with the same semantics as the fused
    kernel: clamp gate to max=clamp_value, value to [-clamp_value, clamp_value]
    before silu. Otherwise forward/backward diverge from the clamped fused path."""

    def _make_clamp_node(self, clamp_value):
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_expert_fusion=False,
            use_accuracy_compatible=True,
            activation_type="swiglu",
            clamp_value=clamp_value,
        )
        node.tokens_per_expert = [3]
        return node

    def _inputs(self):
        paddle.seed(2)
        tokens, hidden = 3, 4
        # scale up so some entries saturate the clamp bound.
        o1 = paddle.randn([tokens, 2 * hidden], dtype=paddle.bfloat16) * 3
        probs = paddle.rand([tokens, 1], dtype=paddle.bfloat16)
        expert_w2 = [paddle.eye(hidden, dtype=paddle.bfloat16)]
        return tokens, hidden, o1, probs, expert_w2

    def test_forward_applies_clamp(self):
        import paddle.nn.functional as F

        cv = 1.0
        tokens, hidden, o1, probs, expert_w2 = self._inputs()
        node = self._make_clamp_node(cv)
        o3 = node.fwd_down_bf16(o1, probs, expert_w2)

        gate = o1.astype("float32")[..., :hidden]
        val = o1.astype("float32")[..., hidden:]
        # round-once clamped reference (fp32 math, cast to bf16 once).
        clamp_ref = (
            F.silu(paddle.clip(gate, max=cv))
            * paddle.clip(val, min=-cv, max=cv)
            * probs.astype("float32")
        ).astype(o1.dtype)
        unclamped_ref = (F.silu(gate) * val * probs.astype("float32")).astype(
            o1.dtype
        )

        np.testing.assert_array_equal(
            o3.astype("float32").numpy(), clamp_ref.astype("float32").numpy()
        )
        # clamp must actually change the result for these saturated inputs.
        self.assertFalse(
            np.array_equal(
                o3.astype("float32").numpy(),
                unclamped_ref.astype("float32").numpy(),
            )
        )

    def test_backward_applies_clamp(self):
        import paddle.nn.functional as F

        cv = 1.0
        tokens, hidden, o1, probs, expert_w2 = self._inputs()
        node = self._make_clamp_node(cv)

        unzipped_grad = paddle.randn([tokens, hidden], dtype=paddle.bfloat16)
        do1, _o2_s, probs_grad = node.bwd_down_input_bf16(
            expert_w2, unzipped_grad, o1, probs
        )

        # accuracy-compatible split-group backward computes do2_s via matmul.
        do2_s = paddle.matmul(unzipped_grad, expert_w2[0].T.contiguous())

        def grads(clamp):
            gate = o1.astype("float32")[..., :hidden].detach()
            val = o1.astype("float32")[..., hidden:].detach()
            scale = probs.astype("float32").detach()
            for t in (gate, val, scale):
                t.stop_gradient = False
            if clamp:
                o2 = (
                    F.silu(paddle.clip(gate, max=cv))
                    * paddle.clip(val, min=-cv, max=cv)
                    * scale
                )
            else:
                o2 = F.silu(gate) * val * scale
            paddle.autograd.backward([o2], [do2_s.astype("float32").detach()])
            d1 = paddle.concat([gate.grad, val.grad], axis=-1).astype(o1.dtype)
            dp = scale.grad.reshape(probs.shape).astype(probs.dtype)
            return d1, dp

        do1_clamp, pg_clamp = grads(True)
        do1_unclamp, _ = grads(False)

        np.testing.assert_array_equal(
            do1.astype("float32").numpy(),
            do1_clamp.astype("float32").numpy(),
        )
        np.testing.assert_array_equal(
            probs_grad.astype("float32").numpy(),
            pg_clamp.astype("float32").numpy(),
        )
        # saturated gate/value gradients are masked by the clamp, so the
        # unclamped gradient must differ.
        self.assertFalse(
            np.array_equal(
                do1.astype("float32").numpy(),
                do1_unclamp.astype("float32").numpy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
