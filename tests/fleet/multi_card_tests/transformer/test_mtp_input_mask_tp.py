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

"""Multi-card test for MTP mtp_hidden_inputs_mask when sequence_parallel is enabled.

Verifies that MultiTokenPredictionLayer._concat_embeddings correctly
handles the mask with scatter_to_sequence_parallel_region when TP > 1.

Run with:
    python -m paddle.distributed.launch --gpus 0,1 \
        tests/multi_card_tests/transformer/test_mtp_input_mask_tp.py
"""

import functools
import unittest

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)

# ---------------------------------------------------------------------------
# Module-level initialization
# ---------------------------------------------------------------------------
TP_SIZE = None


def setUpModule():
    global TP_SIZE
    TP_SIZE = dist.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": TP_SIZE,
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
    model_parallel_cuda_manual_seed(42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMTPInputMaskTP(unittest.TestCase):
    """Test MTP mask handling through real MultiTokenPredictionLayer with TP > 1."""

    @classmethod
    def setUpClass(cls):
        cls.config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=100,
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
            sequence_parallel=True,
            tensor_model_parallel_size=TP_SIZE,
            init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            tie_word_embeddings=True,
            num_nextn_predict_layers=2,
        )
        cls.model = gpt_builder(cls.config, num_stages=1)

    def _find_mtp_layer(self):
        for layer in self.model.run_function:
            if isinstance(layer, MultiTokenPredictionLayer):
                return layer
        return None

    def test_concat_embeddings_mask_with_sp(self):
        """Call _concat_embeddings with a mask to exercise the sequence_parallel branch.

        The fix uses scatter_to_sequence_parallel_region to shard the mask
        when sequence_parallel is enabled, matching hidden_states [local_seq, B, H].
        """
        mtp_layer = self._find_mtp_layer()
        self.assertIsNotNone(mtp_layer)
        self.assertTrue(mtp_layer.sequence_parallel)

        B, S, H = 2, 32, self.config.hidden_size
        local_seq = S // TP_SIZE

        # hidden_states: [local_seq, B, H] (seq-first, SP sharded)
        hidden_states = paddle.randn([local_seq, B, H])
        decoder_input = paddle.randn([local_seq, B, H])

        # mask: [B, 1, S] with position-dependent values
        mask_data = paddle.arange(S, dtype="float32").reshape([1, 1, S]).expand([B, 1, S])
        dist.broadcast(mask_data, src=0)

        result = mtp_layer._concat_embeddings(hidden_states, decoder_input, mtp_hidden_inputs_mask=mask_data)

        # Result shape: [local_seq, B, H] after scatter
        self.assertEqual(result.shape[1], B)
        self.assertFalse(paddle.isnan(result).any().item())
        self.assertFalse(paddle.isinf(result).any().item())

    def test_concat_embeddings_mask_none(self):
        """When mask is None, _concat_embeddings should work without error."""
        mtp_layer = self._find_mtp_layer()
        self.assertIsNotNone(mtp_layer)

        B, H = 2, self.config.hidden_size
        local_seq = 32 // TP_SIZE

        hidden_states = paddle.randn([local_seq, B, H])
        decoder_input = paddle.randn([local_seq, B, H])

        result = mtp_layer._concat_embeddings(hidden_states, decoder_input, mtp_hidden_inputs_mask=None)

        self.assertEqual(result.shape[1], B)
        self.assertFalse(paddle.isnan(result).any().item())

    def test_forward_backward_with_mask(self):
        """End-to-end forward-backward with mtp_hidden_inputs_mask_all."""
        sequence_length = self.config.max_sequence_length
        num_nextn = self.config.num_nextn_predict_layers
        main_seq_len = sequence_length - num_nextn
        micro_batch_size = 1

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).reshape((micro_batch_size, -1))
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).reshape((micro_batch_size, -1))
        labels = paddle.to_tensor(list(range(1, sequence_length + 1)), dtype=paddle.int64).reshape(
            (micro_batch_size, -1)
        )

        # mtp_hidden_inputs_mask_all: [B, num_nextn, main_seq_len]
        mask_all = paddle.ones(
            [micro_batch_size, num_nextn, main_seq_len],
            dtype="float32",
        )
        mask_all[:, :, -2:] = 0.0

        # mtp_startend_row_indices_all: [B, num_nextn, main_seq_len, 1]
        startend_all = paddle.zeros(
            [micro_batch_size, num_nextn, main_seq_len, 1],
            dtype="int32",
        )

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": TP_SIZE,
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
        gpt_pipe_model = NoPipelineParallel(self.model, strategy)
        pipe_data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "mtp_startend_row_indices_all": startend_all,
                "mtp_hidden_inputs_mask_all": mask_all,
            },
            [labels],
        )
        loss = gpt_pipe_model.forward_backward_pipeline(pipe_data)
        self.assertIsNotNone(loss)
        self.assertFalse(paddle.isnan(loss).any().item())
        self.assertFalse(paddle.isinf(loss).any().item())


class TestMTPInputMaskTPWithMHC(unittest.TestCase):
    """Test MTP mask + SP through the mHC (hyper-connections) path."""

    @classmethod
    def setUpClass(cls):
        cls.config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=100,
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
            sequence_parallel=True,
            tensor_model_parallel_size=TP_SIZE,
            init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            tie_word_embeddings=True,
            num_nextn_predict_layers=2,
            enable_hyper_connections=True,
            num_residual_streams=2,
        )
        cls.model = gpt_builder(cls.config, num_stages=1)

    def _find_mtp_layer(self):
        for layer in self.model.run_function:
            if isinstance(layer, MultiTokenPredictionLayer):
                return layer
        return None

    def test_mhc_concat_embeddings_mask_with_sp(self):
        """mHC path: mask scatter via sequence_parallel (line 460-470)."""
        mtp_layer = self._find_mtp_layer()
        self.assertIsNotNone(mtp_layer)
        self.assertTrue(mtp_layer.mhc_enabled)
        self.assertTrue(mtp_layer.sequence_parallel)

        B, S = 2, 32
        H = self.config.hidden_size
        n = self.config.num_residual_streams
        local_seq = S // TP_SIZE

        # mHC hidden_states: [local_seq, B, n*H]
        hidden_states = paddle.randn([local_seq, B, n * H])
        decoder_input = paddle.randn([local_seq, B, H])

        # mask: [B, 1, S] — S must be divisible by TP for scatter
        mask_data = paddle.ones([B, 1, S], dtype="float32")
        mask_data[:, :, -2:] = 0.0
        dist.broadcast(mask_data, src=0)

        result = mtp_layer._concat_embeddings(hidden_states, decoder_input, mtp_hidden_inputs_mask=mask_data)

        self.assertEqual(result.shape[1], B)
        self.assertFalse(paddle.isnan(result).any().item())
        self.assertFalse(paddle.isinf(result).any().item())

    def test_mhc_concat_embeddings_no_mask(self):
        """mHC path without mask still exercises TP gather + SP scatter."""
        mtp_layer = self._find_mtp_layer()
        self.assertIsNotNone(mtp_layer)

        B, H = 2, self.config.hidden_size
        n = self.config.num_residual_streams
        local_seq = 32 // TP_SIZE

        hidden_states = paddle.randn([local_seq, B, n * H])
        decoder_input = paddle.randn([local_seq, B, H])

        result = mtp_layer._concat_embeddings(hidden_states, decoder_input, mtp_hidden_inputs_mask=None)

        self.assertEqual(result.shape[1], B)
        self.assertFalse(paddle.isnan(result).any().item())


if __name__ == "__main__":
    unittest.main()
