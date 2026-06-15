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

"""
Multi-card integration test for DeepEP Combine Refined Recompute.

Verifies gradient consistency between:
- Recompute WITHOUT RR (re-executes combine communication in second forward)
- Recompute WITH RR (caches first forward combine result, reuses in second forward)

Run with: paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 test_rr_deepep_combine.py
"""

import os
import random
import sys
import unittest

# Enable coverage in subprocess when WITH_COVERAGE is set
if os.environ.get("WITH_COVERAGE") == "ON":
    import coverage

    cov = coverage.Coverage(
        data_file=os.environ.get("COVERAGE_FILE", ".coverage"),
        config_file=os.environ.get("COVERAGE_RCFILE", True),
    )
    cov.start()

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

try:
    from paddleformers.fleet.refined_recompute.queue_check import global_rr_queue_log
    from paddleformers.fleet.transformer.moe.fused_a2a import (
        DeepEPCombineAsyncRefinedRecompute,
        fused_combine,
        fused_dispatch,
    )
except ImportError:
    print(
        "Failed to import from paddleformers.fleet.transformer.moe.fused_a2a.",
        file=sys.stderr,
    )
    fused_combine = None
    DeepEPCombineAsyncRefinedRecompute = None
    fused_dispatch = None

# ----- MoE Test Parameters -----
batch_size = 32
hidden_size = 128
num_experts = 64
topk = 2
# ---------------------------


class SharedExpertSimulator(paddle.nn.Layer):
    """Simulates shared expert computation overlapped with combine."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(hidden_size, hidden_size, bias_attr=False)

    def forward(self, x):
        return (self.linear(x),)


class TestCombineLayer(paddle.nn.Layer):
    """
    Simulates a MoE communication cycle (dispatch -> combine) with combine-overlap.
    The combine operation overlaps with shared expert computation via combine_overlap_handle.
    """

    def __init__(self, use_rr=False):
        super().__init__()
        self.use_rr = use_rr
        if self.use_rr:
            self._rr_fusedcombined = DeepEPCombineAsyncRefinedRecompute()
        else:
            self._rr_fusedcombined = None
        self.group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        self.linear = paddle.nn.Linear(hidden_size, hidden_size, bias_attr=False)
        self.shared_expert = SharedExpertSimulator()
        self.deepep_dtype = paddle.bfloat16

    def forward(self, x, mock_token_indices, mock_token_probs):
        # 1. Initial computation
        processed_x = self.linear(x)
        origin_dtype = processed_x.dtype
        processed_x_bf16 = processed_x.astype(self.deepep_dtype)

        # 2. Dispatch
        dispatched_x, _, states, _ = fused_dispatch(
            x=processed_x_bf16,
            token_indices=mock_token_indices,
            token_probs=mock_token_probs,
            num_experts=num_experts,
            group=self.group,
        )

        handle = states["handle"]

        # 3. Combine with overlap (shared expert runs concurrently)
        combine_overlap_handle = {
            "fn": self.shared_expert,
            "fn_args": (processed_x,),
        }

        combined_output = fused_combine(
            x=dispatched_x,
            group=self.group,
            handle=handle,
            _rr_fusedcombined=self._rr_fusedcombined,
            combine_overlap_handle=combine_overlap_handle,
            use_rr_deepep_combine=self.use_rr,
        )

        # 4. Add shared expert output (same as real MoE layer)
        shared_output = combine_overlap_handle["fn_out"][0]
        output = combined_output.astype(origin_dtype) + shared_output

        return output


class TestRecomputeLayer(paddle.nn.Layer):
    """Wrapper that applies recompute to the core TestCombineLayer."""

    def __init__(self, use_rr=False):
        super().__init__()
        self.combine_layer = TestCombineLayer(use_rr=use_rr)

    def forward(self, x, mock_token_indices, mock_token_probs):
        out = recompute(self.combine_layer, x, mock_token_indices, mock_token_probs)
        return out


class TestRRDeepEPCombine(unittest.TestCase):
    """
    Compares backward gradients of recompute without RR vs recompute with RR.
    They should be numerically identical since RR only caches and reuses the
    first forward's combine result without changing the computation.
    """

    @classmethod
    def setUpClass(cls):
        if fused_combine is None or fused_dispatch is None:
            raise unittest.SkipTest("DeepEP fused_combine/fused_dispatch not available.")

        cls.original_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 8,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        seed = 12345
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        cls.x = paddle.randn([batch_size, hidden_size], dtype="float32")

        num_tokens = paddle.shape(cls.x)[0]
        cls.mock_token_indices = paddle.randint(0, num_experts, shape=[num_tokens, topk])
        cls.mock_token_indices.stop_gradient = True

        cls.mock_token_probs_raw = paddle.rand([num_tokens, topk], dtype="float32")
        cls.mock_token_probs_raw.stop_gradient = False

    @classmethod
    def tearDownClass(cls):
        paddle.set_default_dtype(cls.original_dtype)

    def run_test_case(self, layer, x, mock_token_indices, mock_token_probs_raw):
        """Run forward + backward, return gradients."""
        x_copy = x.clone()
        x_copy.stop_gradient = False

        # Clear accumulated gradients before each run
        mock_token_probs_raw.clear_gradient()
        layer.clear_gradients()

        mock_token_probs = F.softmax(mock_token_probs_raw, axis=-1)

        out = layer(x_copy, mock_token_indices, mock_token_probs)
        loss = out.sum()
        loss.backward()

        return (
            x_copy.grad.detach().numpy(),
            layer.combine_layer.linear.weight.grad.detach().numpy(),
            mock_token_probs_raw.grad.detach().numpy(),
        )

    def test_rr_deepep_combine(self):
        """Gradient consistency: recompute without RR vs recompute with RR."""
        seed = 42
        paddle.seed(seed)
        layer_without_rr = TestRecomputeLayer(use_rr=False)

        paddle.seed(seed)
        layer_with_rr = TestRecomputeLayer(use_rr=True)

        # Ensure identical initial weights
        state_dict = layer_without_rr.state_dict()
        layer_with_rr.set_state_dict(state_dict)

        # Run baseline (recompute without RR)
        ori_x_grad, ori_weight_grad, ori_probs_grad = self.run_test_case(
            layer_without_rr,
            self.x,
            self.mock_token_indices,
            self.mock_token_probs_raw,
        )

        # Run RR (recompute with RR)
        rr_x_grad, rr_weight_grad, rr_probs_grad = self.run_test_case(
            layer_with_rr,
            self.x,
            self.mock_token_indices,
            self.mock_token_probs_raw,
        )

        # Assert bit-exact gradient consistency
        np.testing.assert_array_equal(ori_x_grad, rr_x_grad)
        np.testing.assert_array_equal(ori_weight_grad, rr_weight_grad)
        np.testing.assert_array_equal(ori_probs_grad, rr_probs_grad)

        # Verify all RR queues are fully consumed
        global_rr_queue_log.check()

        print("\nTest passed: All gradients with and without Refined Recompute are consistent.")


if __name__ == "__main__":
    filename = os.path.basename(__file__)
    result = unittest.main(exit=False)
    failed_flag = f"{filename}.failed"
    if not result.result.wasSuccessful():
        with open(failed_flag, "w") as f:
            f.write(f"{filename} unittest failed")

    # Save coverage data for subprocess
    if os.environ.get("WITH_COVERAGE") == "ON":
        cov.stop()
        cov.save()
