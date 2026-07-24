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
"""Multi-card (EP>1) tests for AllGatherTokenDispatcher and AllToAllTokenDispatcher.

Run with:
  python -m paddle.distributed.launch --gpus=0,1 \
      tests/fleet/multi_card_tests/moe/test_allgather_dispatcher_ep.py
"""

import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddlefleet_ops
from paddle.distributed import fleet

from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.initialize import initialize_fleet

# ═══════════════════════════════════════════════════════════════════════════════
#  Shared setup — fleet initialised ONCE for the whole process
# ═══════════════════════════════════════════════════════════════════════════════

_fleet_initialised = False
_pg_collection = None


def _ensure_fleet():
    global _fleet_initialised, _pg_collection
    if _fleet_initialised:
        return _pg_collection
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 2,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 2,
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
    initialize_fleet(strategy=strategy)
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    _fleet_initialised = True
    return _pg_collection


class _EPTestBase(unittest.TestCase):
    """Base class: initialises fleet once, provides ep_group / rank."""

    @classmethod
    def setUpClass(cls):
        cls.pg_collection = _ensure_fleet()

    def setUp(self):
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        paddle.manual_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        self.ep_group = self.__class__.pg_collection.ep
        self.ep_size = self.ep_group.nranks
        self.rank = dist.get_rank(self.ep_group)
        self.num_experts = 4
        self.hidden_size = 64
        self.num_experts_per_tok = 2
        self.T_local = 4

    def _make_routing(self):
        """Create routing data. Same on all ranks (deterministic seed)."""
        T, K, E = self.T_local, self.num_experts_per_tok, self.num_experts
        topk_indices = paddle.to_tensor(
            [[i % E, (i + 1) % E] for i in range(T)], dtype="int32"
        )
        topk_weights = paddle.randn([T, K]).abs()
        topk_weights = topk_weights / topk_weights.sum(axis=1, keepdim=True)
        probs = paddle.zeros([T, E], dtype="float32")
        for i in range(T):
            probs[i, i % E] = float(topk_weights[i, 0])
            probs[i, (i + 1) % E] = float(topk_weights[i, 1])
        mask = (probs > 0).cast("float32")
        return topk_indices, topk_weights, probs, mask


# ═══════════════════════════════════════════════════════════════════════════════
#  ReduceScatterGroupOp
# ═══════════════════════════════════════════════════════════════════════════════


class TestReduceScatterGroupOpEP(_EPTestBase):
    def test_forward_scatters_and_sums(self):
        from paddleformers.fleet.transformer.moe.moe_utils import ReduceScatterGroupOp

        T_local, H = 4, 8
        x = paddle.randn([T_local, H]) * (self.rank + 1)
        out = ReduceScatterGroupOp.apply(x, self.ep_group)
        self.assertEqual(out.shape[0], T_local // self.ep_size)
        self.assertEqual(out.shape[1], H)

    def test_backward_all_gathers_grad(self):
        from paddleformers.fleet.transformer.moe.moe_utils import ReduceScatterGroupOp

        T_local, H = 4, 8
        x = paddle.randn([T_local, H], stop_gradient=False)
        out = ReduceScatterGroupOp.apply(x, self.ep_group)
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [T_local, H])


# ═══════════════════════════════════════════════════════════════════════════════
#  _RouterAllGather
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouterAllGatherEP(_EPTestBase):
    def test_forward_global_shape(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        T_local, K = 4, 2
        x = paddle.randn([T_local, K])
        out = _RouterAllGather.apply(x, self.ep_group)
        self.assertEqual(out.shape[0], T_local * self.ep_size)
        self.assertEqual(out.shape[1], K)

    def test_forward_identical_across_ranks(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        T_local, K = 4, 2
        x = paddle.randn([T_local, K]) * (self.rank + 1)
        out = _RouterAllGather.apply(x, self.ep_group)
        # AllGather output must be identical on every rank
        all_outs = [paddle.empty_like(out) for _ in range(self.ep_size)]
        dist.all_gather(all_outs, out, group=self.ep_group)
        for other in all_outs:
            np.testing.assert_allclose(out.numpy(), other.numpy(), rtol=1e-5)

    def test_backward_reduce_scatter(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        T_local, K = 4, 2
        x = paddle.randn([T_local, K], stop_gradient=False)
        out = _RouterAllGather.apply(x, self.ep_group)
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, [T_local, K])


# ═══════════════════════════════════════════════════════════════════════════════
#  _tokens_per_expert_histogram
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokensPerExpertHistogramEP(_EPTestBase):
    def test_histogram_with_global_indices(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _tokens_per_expert_histogram,
        )

        num_experts, T_local, K = 4, 3, 2
        if self.rank == 0:
            indices_local = paddle.to_tensor(
                [[0, 1], [2, 3], [0, -1]], dtype="int32"
            )
        else:
            indices_local = paddle.to_tensor(
                [[1, 2], [3, -1], [0, 1]], dtype="int32"
            )
        recv = [paddle.empty_like(indices_local) for _ in range(self.ep_size)]
        dist.all_gather(recv, indices_local, group=self.ep_group)
        indices_global = paddle.concat(recv, axis=0)

        counts = _tokens_per_expert_histogram(indices_global, num_experts)
        # R0: [0,1],[2,3],[0,-1] → 0:2,1:1,2:1,3:1
        # R1: [1,2],[3,-1],[0,1] → 0:1,1:2,2:1,3:1
        # Total: [3,3,2,2]
        np.testing.assert_array_equal(counts.numpy(), [3, 3, 2, 2])


# ═══════════════════════════════════════════════════════════════════════════════
#  AllGatherTokenDispatcher
# ═══════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available",
)
class TestAllGatherDispatcherEP(_EPTestBase):
    def _make_dispatcher(self, fp8=False):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllGatherTokenDispatcher,
        )

        return AllGatherTokenDispatcher(
            moe_group=self.ep_group,
            expert_model_parallel_size=self.ep_size,
            num_experts=self.num_experts,
            fp8_dispatch=fp8,
            use_ue8m0=False,
        )

    def test_dispatch_preprocess_global_hidden(self):
        dispatcher = self._make_dispatcher()
        ti, tw, probs, mask = self._make_routing()
        x_local = paddle.randn([self.T_local, self.hidden_size])
        global_x = dispatcher.dispatch_preprocess(
            x_local,
            probs,
            mask,
            topk_weights=tw,
            topk_indices=ti,
        )
        self.assertEqual(
            global_x.shape,
            [self.T_local * self.ep_size, self.hidden_size],
        )

    def test_get_dispatched_routing_after_preprocess(self):
        dispatcher = self._make_dispatcher()
        ti, tw, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]),
            probs,
            mask,
            topk_weights=tw,
            topk_indices=ti,
        )
        indices, weights, counts = dispatcher.get_dispatched_routing()
        T_global = self.T_local * self.ep_size
        self.assertEqual(indices.shape, [T_global, self.num_experts_per_tok])
        self.assertEqual(weights.shape, [T_global, self.num_experts_per_tok])
        self.assertEqual(counts.shape, [self.num_experts])

    def test_token_dispatch_pass_through(self):
        dispatcher = self._make_dispatcher()
        ti, tw, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]),
            probs,
            mask,
            topk_weights=tw,
            topk_indices=ti,
        )
        x = paddle.randn([self.T_local * self.ep_size, self.hidden_size])
        result, handle = dispatcher.token_dispatch(x, using_sonic_moe=True)
        np.testing.assert_allclose(result.numpy(), x.numpy())
        self.assertIsNone(handle)

    def test_token_combine_reducescatter(self):
        dispatcher = self._make_dispatcher()
        T_global = self.T_local * self.ep_size
        x = paddle.randn([T_global, self.hidden_size])
        combined = dispatcher.token_combine(x, combine_overlap_handle=None)
        self.assertEqual(combined.shape[0], T_global // self.ep_size)

    def test_combine_postprocess_cached(self):
        dispatcher = self._make_dispatcher()
        T_global = self.T_local * self.ep_size
        x = paddle.randn([T_global, self.hidden_size])
        dispatcher.token_combine(x, combine_overlap_handle=None)
        result = dispatcher.combine_postprocess(x)
        self.assertEqual(result.shape[0], T_global // self.ep_size)

    def test_combine_preprocess_noop(self):
        dispatcher = self._make_dispatcher()
        x = paddle.randn([4, 8])
        result = dispatcher.combine_preprocess(x)
        np.testing.assert_allclose(result.numpy(), x.numpy())


# ═══════════════════════════════════════════════════════════════════════════════
#  AllToAllTokenDispatcher
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllToAllDispatcherEP(_EPTestBase):
    def setUp(self):
        super().setUp()
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_indices = list(
            range(
                self.rank * self.num_local_experts,
                (self.rank + 1) * self.num_local_experts,
            )
        )

    def _make_dispatcher(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        return AllToAllTokenDispatcher(
            moe_group=self.ep_group,
            expert_model_parallel_size=self.ep_size,
            num_experts_per_device=self.num_local_experts,
            local_expert_indices=self.local_expert_indices,
        )

    def test_dispatch_preprocess_sets_metadata(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        self.assertIsNotNone(dispatcher.tokens_per_expert)
        self.assertEqual(
            dispatcher.tokens_per_expert.shape[0], self.num_local_experts
        )

    def test_token_dispatch_returns_tokens(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        result, handle = dispatcher.token_dispatch(permuted)
        self.assertIsNotNone(result)
        self.assertIsNone(handle)

    def test_dispatch_postprocess_sorts_tokens(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        dispatched, _ = dispatcher.token_dispatch(permuted)
        sorted_tokens, tpe = dispatcher.dispatch_postprocess(dispatched)
        self.assertIsNotNone(sorted_tokens)
        self.assertEqual(tpe.shape[0], self.num_local_experts)

    def test_get_dispatched_routing(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        indices, probs_out, tpe = dispatcher.get_dispatched_routing()
        self.assertIsNone(indices)
        self.assertIsNone(probs_out)
        self.assertEqual(tpe.shape[0], self.num_local_experts)

    def test_combine_preprocess_preserves_shape(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        dispatched, _ = dispatcher.token_dispatch(permuted)
        sorted_tokens, _ = dispatcher.dispatch_postprocess(dispatched)
        expert_out = paddle.randn_like(sorted_tokens)
        restored = dispatcher.combine_preprocess(expert_out)
        self.assertEqual(restored.shape, expert_out.shape)

    def test_full_round_trip_shape(self):
        dispatcher = self._make_dispatcher()
        _, _, probs, mask = self._make_routing()
        permuted = dispatcher.dispatch_preprocess(
            paddle.randn([self.T_local, self.hidden_size]), probs, mask
        )
        dispatched, _ = dispatcher.token_dispatch(permuted)
        sorted_tokens, _ = dispatcher.dispatch_postprocess(dispatched)
        # Identity expert output
        restored = dispatcher.combine_preprocess(sorted_tokens)
        combined = dispatcher.token_combine(restored)
        output = dispatcher.combine_postprocess(combined)
        self.assertEqual(output.shape[0], self.T_local)
        self.assertEqual(output.shape[1], self.hidden_size)


# ═══════════════════════════════════════════════════════════════════════════════
#  MoELayer.combine() routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestMoELayerCombineEP(_EPTestBase):
    def test_combine_allgather_path(self):
        from unittest.mock import MagicMock

        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "allgather"
        layer.token_dispatcher = MagicMock()
        MoELayer.combine(
            layer, paddle.randn([4, 8]), combine_overlap_handle=None
        )
        layer.token_dispatcher.token_combine.assert_called_once()
        layer.token_dispatcher.combine_postprocess.assert_called_once()

    def test_combine_alltoall_path(self):
        from unittest.mock import MagicMock

        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "alltoall"
        layer.token_dispatcher = MagicMock()
        MoELayer.combine(layer, paddle.randn([4, 8]))
        layer.token_dispatcher.token_combine.assert_called_once()
        layer.token_dispatcher.combine_postprocess.assert_called_once()
