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
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    ),
)

import numpy as np
import paddle

from paddleformers.fleet.context_parallel_utils import (
    ContextParallelGatherOp,
    ContextParallelScatterOp,
)
from paddleformers.fleet.models.common.language_loss.language_loss import (
    LanguageLoss,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from tests.fleet.multi_card_tests.tensor_parallel.test_utilities import Utils


class TestMTPDistillationLossCP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        world_size = 4
        Utils.initialize_model_parallel(
            context_parallel_size=world_size,
            expert_parallel_size=world_size,
            sharding_parallel_size=world_size,
        )

    def setUp(self):
        paddle.seed(Utils.rank)

    def test_forward_backward(self):
        batch_size = 2
        seq_len = 256
        vocab_size = 128
        nextn = 2

        config = TransformerConfig(
            context_parallel_size=Utils.world_size,
            num_nextn_predict_layers=nextn,
            params_dtype=paddle.bfloat16,
            mtp_distillation_loss=True,
            experimental_dataflow=True,
            cp_balance_mode="dualchunk_allgather",
        )

        model = LanguageLoss(config)

        logits_ref, logits_tgt = [], []
        for i in range(1 + nextn):
            t = paddle.randn(
                [batch_size, seq_len, vocab_size], dtype="bfloat16"
            )
            t.stop_gradient = False
            logits_ref.append(t)
            t = t.detach()
            t.stop_gradient = False
            logits_tgt.append(t)

        labels = paddle.randint(
            vocab_size,
            shape=[batch_size, seq_len * Utils.world_size + nextn],
            dtype="int64",
        )

        # Run reference (Gather-Shift-Scatter)
        # 注：下面实际走的是 dualchunk_allgather 的路径，但是通过 mock 让分布式通信使用
        # contiguous_allgather 的布局，从而与 target 对齐
        try:
            gather_apply = ContextParallelGatherOp.apply
            scatter_apply = ContextParallelScatterOp.apply

            ContextParallelGatherOp.apply = (
                lambda tensor, axis, mode: gather_apply(
                    tensor, axis, mode="contiguous_allgather"
                )
            )
            ContextParallelScatterOp.apply = (
                lambda tensor, axis, mode: scatter_apply(
                    tensor, axis, mode="contiguous_allgather"
                )
            )

            loss_ref = model(logits_ref, labels)
        finally:
            del ContextParallelGatherOp.apply
            del ContextParallelScatterOp.apply

        loss_ref.backward()

        # Run target (MTPDistillationLossShift)
        model.config.cp_balance_mode = "contiguous_allgather"
        loss_tgt = model(logits_tgt, labels)
        loss_tgt.backward()

        # Compare
        np.testing.assert_allclose(loss_ref, loss_tgt, atol=0, rtol=0)
        for ref, tgt in zip(logits_ref, logits_tgt):
            np.testing.assert_allclose(
                ref.grad.float(), tgt.grad.float(), atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
