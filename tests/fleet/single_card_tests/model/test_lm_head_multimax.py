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

"""Tests for GPTLMHead with multimax feature."""

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

import paddlefleet_ops

# sonicmoe ecosystem op is not always loadable in CI envs; the multimax
# feature does not depend on it, so neutralize the gating before importing
# anything from paddleformers.fleet.models.
paddlefleet_ops.is_sonic_moe_available = lambda: False

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig


class TestMultimaxLMHead(unittest.TestCase):
    """Tests for GPTLMHead multimax initialization and forward paths."""

    @classmethod
    def setUpClass(cls):
        seed = 48
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)
        cls.strategy = strategy

        # Config with multimax_modules=[lm_head]
        cls.config_multimax = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            multimax_modules=["lm_head"],
        )
        cls.model_multimax = gpt_builder(cls.config_multimax, num_stages=1)

        # Config without multimax
        cls.config_no_multimax = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            multimax_modules=None,
        )
        cls.model_no_multimax = gpt_builder(
            cls.config_no_multimax, num_stages=1
        )

        # Config with multimax + fused path
        cls.config_fused = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            multimax_modules=["lm_head"],
            fused_linear_ce_loss_chunk=1,
        )
        cls.model_fused = gpt_builder(cls.config_fused, num_stages=1)

    def _find_lm_head(self, model):
        from paddleformers.fleet.models.gpt.lm_head import GPTLMHead

        for layer in model.run_function:
            if isinstance(layer, GPTLMHead):
                return layer
        return None

    def test_multimax_lmhead_creates_params(self):
        """When multimax_modules=[lm_head], GPTLMHead should create multimax_ranges/ts params."""
        lm_head = self._find_lm_head(self.model_multimax)
        self.assertIsNotNone(lm_head)
        self.assertTrue(hasattr(lm_head, "use_multimax_lmhead"))
        self.assertTrue(lm_head.use_multimax_lmhead)
        self.assertTrue(hasattr(lm_head, "multimax_ranges"))
        self.assertTrue(hasattr(lm_head, "multimax_ts"))
        self.assertEqual(lm_head.multimax_ranges.shape, [4])
        self.assertEqual(lm_head.multimax_ts.shape, [4])

    def test_no_multimax_lmhead_no_params(self):
        """When multimax_modules=None, GPTLMHead should NOT create multimax params."""
        lm_head = self._find_lm_head(self.model_no_multimax)
        self.assertIsNotNone(lm_head)
        self.assertFalse(getattr(lm_head, "use_multimax_lmhead", False))
        self.assertFalse(hasattr(lm_head, "multimax_ranges"))
        self.assertFalse(hasattr(lm_head, "multimax_ts"))

    def test_multimax_params_init_zero(self):
        """multimax params should init to zero (SegLU is identity at step 0)."""
        lm_head = self._find_lm_head(self.model_multimax)
        zeros = paddle.zeros_like(lm_head.multimax_ranges)
        self.assertTrue(paddle.allclose(lm_head.multimax_ranges, zeros).item())
        self.assertTrue(paddle.allclose(lm_head.multimax_ts, zeros).item())

    def test_fused_path_returns_5tuple(self):
        """With fused_linear_ce_loss_chunk>0 and multimax_modules, forward returns 5-tuple.
        Also exercises the rank-0 [MULTIMAX-LMHEAD-APPLIED] one-shot warn banner
        at the top of the fused branch.
        """
        lm_head = self._find_lm_head(self.model_fused)
        self.assertIsNotNone(lm_head)
        # Reset the one-shot flag so the warn block executes.
        if hasattr(lm_head, "_multimax_applied_logged"):
            delattr(lm_head, "_multimax_applied_logged")

        # Create dummy input
        batch_size, seq_len, hidden_size = 2, 8, 256
        hidden_states = paddle.randn([seq_len, batch_size, hidden_size])

        # Call forward with dict args (expected signature)
        output = lm_head.forward({"hidden_states": hidden_states})

        # Should return 5-tuple: (hidden, weight, bias, multimax_ranges, multimax_ts)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 5)
        # The one-shot flag must be set after first call.
        self.assertTrue(getattr(lm_head, "_multimax_applied_logged", False))

    def test_unfused_path_applies_seglu_and_logs(self):
        """With multimax_modules + fused_linear_ce_loss_chunk=0 (default), forward
        returns logits (not a tuple) and applies SegLU via recompute.
        Exercises the unfused-path warn block + SegLU recompute branch.
        """
        lm_head = self._find_lm_head(self.model_multimax)
        self.assertIsNotNone(lm_head)
        # Reset the one-shot flag so the warn block executes.
        if hasattr(lm_head, "_multimax_applied_logged"):
            delattr(lm_head, "_multimax_applied_logged")

        batch_size, seq_len, hidden_size = 2, 8, 256
        hidden_states = paddle.randn([seq_len, batch_size, hidden_size])

        output = lm_head.forward({"hidden_states": hidden_states})
        # Unfused path returns logits Tensor, not a tuple.
        self.assertIsInstance(output, paddle.Tensor)
        # Logits shape: [B, S, V] (transposed from [S, B, V] by sequence_parallel
        # when enabled, otherwise produced directly).
        self.assertEqual(output.shape[-1], self.config_multimax.vocab_size)
        self.assertTrue(getattr(lm_head, "_multimax_applied_logged", False))

    def test_sharded_state_dict_with_multimax(self):
        """sharded_state_dict on multimax lm_head: non-sharded params (the
        multimax scalars) are intentionally absent from shard_rules — only
        the sharded weight/bias appear when world_size > 1.
        We patch `build_sharded_state_dict` to capture the shard_rules dict
        so we can assert directly without depending on the flex-checkpoint
        backend.
        """
        from unittest import mock

        from paddleformers.fleet.models.gpt import lm_head as lm_head_mod

        lm_head = self._find_lm_head(self.model_multimax)
        self.assertIsNotNone(lm_head)

        captured = {}

        def fake_build(state_dict, shard_rules, prefix):
            captured["shard_rules"] = shard_rules
            captured["prefix"] = prefix
            return {"state_dict": state_dict, "shard_rules": shard_rules}

        # world_size==1 branch: shard_rules is None.
        with mock.patch.object(
            lm_head_mod, "build_sharded_state_dict", side_effect=fake_build
        ):
            lm_head.sharded_state_dict()
        self.assertIsNone(captured["shard_rules"])

        # Force world_size>1: only weight/bias appear in shard_rules.
        captured.clear()
        with (
            mock.patch.object(lm_head, "world_size", 2),
            mock.patch.object(
                lm_head_mod, "build_sharded_state_dict", side_effect=fake_build
            ),
        ):
            lm_head.sharded_state_dict()
        rules = captured["shard_rules"]
        self.assertEqual(rules.get("weight"), 0)
        self.assertEqual(rules.get("bias"), 0)
        self.assertNotIn("multimax_ranges", rules)
        self.assertNotIn("multimax_ts", rules)


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "fused multimax CE kernel requires CUDA"
)
class TestFusedMultimaxNumerical(unittest.TestCase):
    """Numerical + backward parity for the fused multimax CE kernel.

    Compares `LigerFusedLinearCrossEntropyFunction` (multimax branch) against
    a Python reference: F.linear -> SegLU -> cross_entropy, including a sample
    masked by `ignore_index`. Verifies loss and grads on all four trainable
    inputs: `_input`, `weight`, `multimax_ranges`, `multimax_ts`.
    """

    def _seglu(self, x, ranges, ts):
        """Python ref SegLU matching the Triton kernel formula."""
        r0, r1, r2, r3 = [ranges[i] for i in range(4)]
        t0, t1, t2, t3 = [ts[i] for i in range(4)]
        m0 = paddle.nn.functional.relu(r0 - x)
        m1 = paddle.nn.functional.relu(x - r1)
        m2 = paddle.nn.functional.relu(r2 - x)
        m3 = paddle.nn.functional.relu(x - r3)
        return x + t0 * m0 + t1 * m1 + t2 * m2 * m2 + t3 * m3 * m3

    def _make_inputs(self, BT=6, H=8, V=12, ignore_index=-100):
        paddle.seed(123)
        x = paddle.randn([BT, H], dtype="float32")
        w = paddle.randn([V, H], dtype="float32") * 0.1
        targets = paddle.to_tensor([0, 3, 5, ignore_index, 7, 1], dtype="int64")
        # Non-zero multimax params so SegLU is non-trivial.
        ranges = paddle.to_tensor([0.5, -0.5, 0.2, -0.2], dtype="float32")
        ts = paddle.to_tensor([0.3, 0.4, 0.1, 0.2], dtype="float32")
        return x, w, targets, ranges, ts, ignore_index

    def _run_reference(self, x, w, targets, ranges, ts, ignore_index):
        x_ref = x.detach().clone()
        x_ref.stop_gradient = False
        w_ref = w.detach().clone()
        w_ref.stop_gradient = False
        r_ref = ranges.detach().clone()
        r_ref.stop_gradient = False
        t_ref = ts.detach().clone()
        t_ref.stop_gradient = False

        logits = paddle.nn.functional.linear(x_ref, w_ref.T)
        modulated = self._seglu(logits, r_ref, t_ref)
        loss = paddle.nn.functional.cross_entropy(
            modulated,
            targets,
            ignore_index=ignore_index,
            reduction="sum",
        )
        loss.backward()
        return loss, x_ref.grad, w_ref.grad, r_ref.grad, t_ref.grad

    def _run_fused(self, x, w, targets, ranges, ts, ignore_index):
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        x_f = x.detach().clone()
        x_f.stop_gradient = False
        w_f = w.detach().clone()
        w_f.stop_gradient = False
        r_f = ranges.detach().clone()
        r_f.stop_gradient = False
        t_f = ts.detach().clone()
        t_f.stop_gradient = False

        loss_1d = LigerFusedLinearCrossEntropyFunction.apply(
            x_f,
            w_f,
            targets,
            None,  # bias
            ignore_index,
            "none",
            1,  # num_chunks
            False,  # ec_align
            r_f,
            t_f,
        )
        loss = loss_1d.sum()
        loss.backward()
        return loss, x_f.grad, w_f.grad, r_f.grad, t_f.grad

    def test_fused_multimax_matches_reference(self):
        x, w, targets, ranges, ts, ig = self._make_inputs()

        loss_ref, gx_ref, gw_ref, gr_ref, gt_ref = self._run_reference(
            x, w, targets, ranges, ts, ig
        )
        loss_fused, gx_f, gw_f, gr_f, gt_f = self._run_fused(
            x, w, targets, ranges, ts, ig
        )

        # Loss (reduction='sum' on ref vs sum-of-1d on fused).
        self.assertTrue(
            paddle.allclose(loss_ref, loss_fused, atol=1e-3, rtol=1e-3).item(),
            f"loss mismatch: ref={loss_ref.item()} fused={loss_fused.item()}",
        )
        # Input grad.
        self.assertTrue(
            paddle.allclose(gx_ref, gx_f, atol=1e-3, rtol=1e-3).item(),
            f"grad_input mismatch (max abs diff="
            f"{(gx_ref - gx_f).abs().max().item()})",
        )
        # multimax_ranges grad.
        self.assertTrue(
            paddle.allclose(gr_ref, gr_f, atol=1e-3, rtol=1e-3).item(),
            f"grad_ranges mismatch: ref={gr_ref.numpy().tolist()} "
            f"fused={gr_f.numpy().tolist()}",
        )
        # multimax_ts grad.
        self.assertTrue(
            paddle.allclose(gt_ref, gt_f, atol=1e-3, rtol=1e-3).item(),
            f"grad_ts mismatch: ref={gt_ref.numpy().tolist()} "
            f"fused={gt_f.numpy().tolist()}",
        )

    def test_fused_multimax_grads_when_input_frozen(self):
        """Freeze backbone (input.stop_gradient=True) but keep multimax
        params trainable: kernel must still emit non-zero grad_ranges/ts.
        Regression for the prior `HAS_GRADIENTS=input_requires_grad` bug
        that gated multimax param grads on the input grad.
        """
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        x, w, targets, ranges, ts, ig = self._make_inputs()
        x_f = x.detach().clone()
        x_f.stop_gradient = True  # frozen backbone
        w_f = w.detach().clone()
        w_f.stop_gradient = True  # also freeze weight (head-only training)
        r_f = ranges.detach().clone()
        r_f.stop_gradient = False
        t_f = ts.detach().clone()
        t_f.stop_gradient = False

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x_f, w_f, targets, None, ig, "none", 1, False, r_f, t_f
        ).sum()
        loss.backward()

        # Reference grads (from the full-grad path, then ignore x/w).
        _, _, _, gr_ref, gt_ref = self._run_reference(
            x, w, targets, ranges, ts, ig
        )
        self.assertIsNotNone(
            r_f.grad, "multimax_ranges.grad is None when frozen"
        )
        self.assertIsNotNone(t_f.grad, "multimax_ts.grad is None when frozen")
        self.assertTrue(
            paddle.allclose(gr_ref, r_f.grad, atol=1e-3, rtol=1e-3).item(),
            f"frozen-input grad_ranges mismatch: ref={gr_ref.numpy().tolist()} "
            f"got={r_f.grad.numpy().tolist()}",
        )
        self.assertTrue(
            paddle.allclose(gt_ref, t_f.grad, atol=1e-3, rtol=1e-3).item(),
            f"frozen-input grad_ts mismatch: ref={gt_ref.numpy().tolist()} "
            f"got={t_f.grad.numpy().tolist()}",
        )

    def test_fused_multimax_grads_when_only_input_frozen(self):
        """Freeze only the backbone input; keep weight + multimax trainable.
        This is the head-only-training regime flagged in CR: the kernel must
        still write grad_logits (so weight grad is populated) and emit
        non-zero multimax param grads, even though x.stop_gradient=True.
        Regression for the `HAS_GRADIENTS=input_requires_grad` gating bug
        and for the `grad_weight` allocation being conditioned on
        `input_requires_grad and weight_requires_grad`.
        """
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        x, w, targets, ranges, ts, ig = self._make_inputs()
        x_f = x.detach().clone()
        x_f.stop_gradient = True  # frozen backbone
        w_f = w.detach().clone()
        w_f.stop_gradient = False  # LM head weight is trainable
        r_f = ranges.detach().clone()
        r_f.stop_gradient = False
        t_f = ts.detach().clone()
        t_f.stop_gradient = False

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x_f, w_f, targets, None, ig, "none", 1, False, r_f, t_f
        ).sum()
        loss.backward()

        _, _, gw_ref, gr_ref, gt_ref = self._run_reference(
            x, w, targets, ranges, ts, ig
        )
        self.assertIsNotNone(
            w_f.grad, "weight.grad is None when only input is frozen"
        )
        self.assertTrue(
            paddle.allclose(gw_ref, w_f.grad, atol=1e-3, rtol=1e-3).item(),
            f"frozen-input grad_weight max abs diff="
            f"{(gw_ref - w_f.grad).abs().max().item()}",
        )
        self.assertIsNotNone(r_f.grad)
        self.assertIsNotNone(t_f.grad)
        self.assertTrue(
            paddle.allclose(gr_ref, r_f.grad, atol=1e-3, rtol=1e-3).item()
        )
        self.assertTrue(
            paddle.allclose(gt_ref, t_f.grad, atol=1e-3, rtol=1e-3).item()
        )

    def test_fused_multimax_partial_freeze_respects_per_param_stop_gradient(
        self,
    ):
        """Freeze only one of multimax_ranges/multimax_ts; the other stays
        trainable. The kernel emits both grads (multimax_requires_grad is the
        OR of the two), but the backward must respect each param's individual
        stop_gradient: the frozen param must NOT receive a grad (its .grad
        stays None and main_grad is untouched), while the trainable param
        gets the correct grad.

        Regression for: backward loop only checked `param is None or g is
        None` and silently wrote grads to frozen multimax params.
        """
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        # Case A: freeze multimax_ranges, keep multimax_ts trainable.
        x, w, targets, ranges, ts, ig = self._make_inputs()
        x_f = x.detach().clone()
        x_f.stop_gradient = False
        w_f = w.detach().clone()
        w_f.stop_gradient = False
        r_f = ranges.detach().clone()
        r_f.stop_gradient = True  # frozen
        t_f = ts.detach().clone()
        t_f.stop_gradient = False

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x_f, w_f, targets, None, ig, "none", 1, False, r_f, t_f
        ).sum()
        loss.backward()

        _, _, _, _, gt_ref = self._run_reference(x, w, targets, ranges, ts, ig)
        self.assertIsNone(
            r_f.grad,
            "frozen multimax_ranges received a grad despite stop_gradient=True",
        )
        self.assertIsNotNone(t_f.grad)
        self.assertTrue(
            paddle.allclose(gt_ref, t_f.grad, atol=1e-3, rtol=1e-3).item()
        )

        # Case B: freeze multimax_ts, keep multimax_ranges trainable.
        x, w, targets, ranges, ts, ig = self._make_inputs()
        x_f = x.detach().clone()
        x_f.stop_gradient = False
        w_f = w.detach().clone()
        w_f.stop_gradient = False
        r_f = ranges.detach().clone()
        r_f.stop_gradient = False
        t_f = ts.detach().clone()
        t_f.stop_gradient = True  # frozen

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x_f, w_f, targets, None, ig, "none", 1, False, r_f, t_f
        ).sum()
        loss.backward()

        _, _, _, gr_ref, _ = self._run_reference(x, w, targets, ranges, ts, ig)
        self.assertIsNone(
            t_f.grad,
            "frozen multimax_ts received a grad despite stop_gradient=True",
        )
        self.assertIsNotNone(r_f.grad)
        self.assertTrue(
            paddle.allclose(gr_ref, r_f.grad, atol=1e-3, rtol=1e-3).item()
        )


if __name__ == "__main__":
    unittest.main()
