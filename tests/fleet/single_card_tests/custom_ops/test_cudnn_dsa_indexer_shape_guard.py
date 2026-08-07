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

"""Host-side shape guard for the cuDNN CSA indexer forward.

The cuDNN CSA indexer forward kernel does not reliably support short
compressed-KV shapes: ``S_k == 1`` (n_compressed == 1) crashes inside the CUDA
kernel with ``cudaErrorIllegalInstruction`` (715) regardless of ``S_q`` —
verified by a per-process sweep before the v1.26 integration.

``_check_cudnn_indexer_shape_support`` converts that unsupported short-sequence
case into a readable ``ValueError`` so it fails clearly instead of poisoning the
CUDA context. These checks are pure host-side and need no GPU.
"""

import unittest

import paddle

from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
    _check_cudnn_indexer_shape_support,
)


def _qkw(sq, sk, h=64, d=128):
    index_q = paddle.zeros([1, sq, h, d], dtype="bfloat16")
    index_k = paddle.zeros([1, sk, d], dtype="bfloat16")
    weights = paddle.zeros([1, sq, h], dtype="bfloat16")
    return index_q, index_k, weights


class TestCudnnIndexerShapeGuard(unittest.TestCase):
    """_check_cudnn_indexer_shape_support rejects unsupported short S_k."""

    def test_sk1_raises_value_error(self):
        # S_k == 1 (n_compressed == 1) crashes the CUDA kernel; guard rejects it.
        for sq in (1, 4, 5, 9):
            iq, ik, w = _qkw(sq, 1)
            with self.assertRaises(ValueError) as cm:
                _check_cudnn_indexer_shape_support(iq, ik, ratio=4)
            self.assertIn("compressed KV length >= 2", str(cm.exception))

    def test_sq_gt_sk_ratio_passes_on_v126(self):
        # v1.26 clamps ratio-causal block count to S_k; tail query rows are valid.
        iq, ik, w = _qkw(9, 2)  # 9 > 2 * 4
        _check_cudnn_indexer_shape_support(iq, ik, ratio=4)

    def test_seq_offset_past_ratio_bound_passes_on_v126(self):
        # CP causal-only mode may have local chunks whose q offset exceeds S_k*ratio.
        iq, ik, w = _qkw(4, 2)  # 4 + 5 > 2 * 4
        _check_cudnn_indexer_shape_support(iq, ik, ratio=4, seq_offset=5)

    def test_valid_shapes_pass(self):
        # S_k >= 2: no host-side shape error.
        for sq, sk in [(8, 2), (4, 2), (16, 4), (1, 2), (32, 8)]:
            iq, ik, w = _qkw(sq, sk)
            _check_cudnn_indexer_shape_support(iq, ik, ratio=4)  # no raise

    def test_boundary_sq_equals_sk_ratio_passes(self):
        iq, ik, w = _qkw(8, 2)  # 8 == 2 * 4
        _check_cudnn_indexer_shape_support(iq, ik, ratio=4)  # no raise

    def test_ratio_no_longer_restricts_host_shape(self):
        iq, ik, w = _qkw(8, 2)
        _check_cudnn_indexer_shape_support(iq, ik, ratio=4)
        _check_cudnn_indexer_shape_support(iq, ik, ratio=2)


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer forward requires Blackwell GPU (SM100)",
)
class TestCudnnIndexerForwardGuardIntegration(unittest.TestCase):
    """The guard fires through the public forward / topk_fwd entry points,
    turning the CUDA-715 crash into a clean ValueError without poisoning the
    CUDA context (a subsequent valid call still succeeds)."""

    def test_forward_rejects_sk1_then_valid_call_works(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_forward,
        )

        iq, ik, w = _qkw(5, 1)
        with self.assertRaises(ValueError):
            cudnn_indexer_forward(iq, ik, w, ratio=4)

        # CUDA context must still be usable (guard ran before any kernel).
        iq2, ik2, w2 = _qkw(8, 2)
        scores = cudnn_indexer_forward(iq2, ik2, w2, ratio=4)
        paddle.device.synchronize()
        self.assertEqual(list(scores.shape), [1, 8, 2])

    def test_topk_fwd_rejects_sk1(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        iq, ik, w = _qkw(5, 1)
        with self.assertRaises(ValueError):
            cudnn_indexer_topk_fwd(iq, ik, w, ratio=4, topk_effective=4)


if __name__ == "__main__":
    unittest.main()
