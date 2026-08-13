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

"""Unit tests for IndexerBiasAdjustCallback (MoH load balancing)."""

import types
import unittest
from unittest.mock import MagicMock, patch

import paddle
from paddle import nn

from paddleformers.trainer.trainer_callback import IndexerBiasAdjustCallback

# ---------------------------------------------------------------------------
# Helpers: minimal fake model that carries the MoH buffers.
# ---------------------------------------------------------------------------


class _FakeLinearWqB(nn.Layer):
    """Minimal stand-in for CSAIndexer.linear_wq_b."""

    def __init__(self, in_size=16, out_size=16):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[in_size, out_size], default_initializer=nn.initializer.Constant(0.0)
        )


class _FakeCSAIndexer(nn.Layer):
    """Fakes the two MoH buffers and ``linear_wq_b`` for callback testing."""

    def __init__(self, n_heads=8):
        super().__init__()
        self.register_buffer(
            "indexer_moh_bias",
            paddle.zeros([n_heads], dtype="float32"),
            persistable=True,
        )
        self.register_buffer(
            "local_tokens_per_indexer_moh",
            paddle.zeros([n_heads], dtype="float32"),
            persistable=False,
        )
        self.linear_wq_b = _FakeLinearWqB()


class _NoMoHLayer(nn.Layer):
    """Layer with no MoH buffers -- the callback should skip it entirely."""

    def __init__(self):
        super().__init__()
        self.linear = _FakeLinearWqB()


class _FakeModel(nn.Layer):
    """Model containing a mix of MoH-carrying indexers and unrelated layers."""

    def __init__(self, n_indexers=2, n_heads=8, include_no_moh_layer=True):
        super().__init__()
        self.indexers = nn.LayerList([_FakeCSAIndexer(n_heads=n_heads) for _ in range(n_indexers)])
        if include_no_moh_layer:
            self.other = _NoMoHLayer()


def _run_callback(callback, model, freeze_training=False, optimizer=None):
    """Invoke ``on_optimizer_end`` with the arg shape the callback expects."""
    args = types.SimpleNamespace(freeze_training=freeze_training)
    state = MagicMock()
    control = MagicMock()

    # Bypass the distributed all_reduce path: ``fleet._hcg`` is absent in the
    # test env, so the callback would fall through to ``dist.all_reduce``.
    # Patch it into an identity op so the callback can run single-process.
    with patch(
        "paddleformers.trainer.trainer_callback.dist.all_reduce",
        side_effect=lambda t, group=None: t,
    ):
        callback.on_optimizer_end(args, state, control, model=model, optimizer=optimizer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIndexerBiasAdjustCallbackNoMoHModel(unittest.TestCase):
    """A model without MoH buffers is a silent no-op (no error, no writes)."""

    def test_no_moh_module_is_noop(self):
        model = nn.LayerList([_NoMoHLayer(), _NoMoHLayer()])
        callback = IndexerBiasAdjustCallback(lr=0.001)
        # Should return cleanly without raising -- no all_reduce call needed.
        _run_callback(callback, model)


class TestIndexerBiasAdjustCallbackUpdatesBias(unittest.TestCase):
    """Standard update path: bias shifts toward mean, counter is zeroed."""

    def test_sign_based_update_and_counter_reset(self):
        paddle.seed(0)
        model = _FakeModel(n_indexers=1, n_heads=4, include_no_moh_layer=True)
        indexer = model.indexers[0]
        # Head 0 got the most tokens, head 3 the fewest.
        indexer.local_tokens_per_indexer_moh.set_value(paddle.to_tensor([10.0, 5.0, 5.0, 0.0], dtype="float32"))
        # Snapshot pre-update bias.
        before = indexer.indexer_moh_bias.numpy().copy()

        lr = 0.01
        callback = IndexerBiasAdjustCallback(lr=lr)
        _run_callback(callback, model)

        after = indexer.indexer_moh_bias.numpy()
        # mean = 5.0; sign(mean - usage) = [-1, 0, 0, +1] -> update = [-lr, 0, 0, +lr]
        expected_delta = [-lr, 0.0, 0.0, lr]
        for i in range(4):
            self.assertAlmostEqual(float(after[i] - before[i]), expected_delta[i], places=6)
        # Counter must be zeroed for the next accumulation window.
        self.assertEqual(float(indexer.local_tokens_per_indexer_moh.sum().item()), 0.0)

    def test_multiple_indexers_updated_independently(self):
        model = _FakeModel(n_indexers=3, n_heads=4)
        for i, indexer in enumerate(model.indexers):
            # Give each indexer a distinct imbalance.
            indexer.local_tokens_per_indexer_moh.set_value(paddle.to_tensor([i + 1.0, 0.0, 0.0, 0.0], dtype="float32"))
        callback = IndexerBiasAdjustCallback(lr=0.001)
        _run_callback(callback, model)
        # Every indexer's head-0 (over-used) bias must have decreased and the
        # others increased, independently per indexer.
        for indexer in model.indexers:
            bias = indexer.indexer_moh_bias.numpy()
            self.assertLess(float(bias[0]), 0.0)
            for k in (1, 2, 3):
                self.assertGreater(float(bias[k]), 0.0)


class TestIndexerBiasAdjustCallbackFreezeTraining(unittest.TestCase):
    """``freeze_training`` skips both the bias update and the counter reset."""

    def test_freeze_training_skips_everything(self):
        model = _FakeModel(n_indexers=1, n_heads=4)
        indexer = model.indexers[0]
        indexer.local_tokens_per_indexer_moh.set_value(paddle.to_tensor([10.0, 0.0, 0.0, 0.0], dtype="float32"))
        before_bias = indexer.indexer_moh_bias.numpy().copy()
        before_counter = indexer.local_tokens_per_indexer_moh.numpy().copy()

        callback = IndexerBiasAdjustCallback(lr=0.01)
        _run_callback(callback, model, freeze_training=True)

        # Neither the bias nor the counter changes when freeze_training is on.
        self.assertTrue((indexer.indexer_moh_bias.numpy() == before_bias).all())
        self.assertTrue((indexer.local_tokens_per_indexer_moh.numpy() == before_counter).all())


class TestIndexerBiasAdjustCallbackFrozenIndexer(unittest.TestCase):
    """A frozen indexer (linear_wq_b.weight.stop_gradient) is skipped."""

    def test_stop_gradient_skips_update_but_still_zeroes_counter(self):
        model = _FakeModel(n_indexers=1, n_heads=4)
        indexer = model.indexers[0]
        indexer.linear_wq_b.weight.stop_gradient = True
        indexer.local_tokens_per_indexer_moh.set_value(paddle.to_tensor([10.0, 0.0, 0.0, 0.0], dtype="float32"))
        before_bias = indexer.indexer_moh_bias.numpy().copy()

        callback = IndexerBiasAdjustCallback(lr=0.01)
        _run_callback(callback, model)

        # Bias untouched, but counter still reset (mirrors MoECorrectionBias
        # callback: frozen layers should not accumulate stale usage counts).
        self.assertTrue((indexer.indexer_moh_bias.numpy() == before_bias).all())
        self.assertEqual(float(indexer.local_tokens_per_indexer_moh.sum().item()), 0.0)


class TestIndexerBiasAdjustCallbackExports(unittest.TestCase):
    """The callback must be importable via the top-level trainer package."""

    def test_top_level_import(self):
        from paddleformers.trainer import IndexerBiasAdjustCallback as cb  # noqa: F401

        self.assertIs(cb, IndexerBiasAdjustCallback)

    def test_all_export(self):
        from paddleformers.trainer.trainer_callback import __all__

        self.assertIn("IndexerBiasAdjustCallback", __all__)


if __name__ == "__main__":
    unittest.main()
