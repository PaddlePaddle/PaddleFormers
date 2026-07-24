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


import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.transformer.multi_token_prediction import (
    WeightOnlyMTPLayer,
)


class TestWeightOnlyMTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 46
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

        cls.config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=1024,
            max_sequence_length=64,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
            num_nextn_predict_layers=1,
            mtp_load_weight_only=True,
        )

        cls.gpt_model = gpt_builder(cls.config, num_stages=1)

    def test_weight_only_mtp_layer_exists(self):
        """Test that the model contains a WeightOnlyMTPLayer."""
        found_weight_only_mtp = False
        for layer in self.gpt_model.run_function:
            if isinstance(layer, WeightOnlyMTPLayer):
                found_weight_only_mtp = True
                break
        assert found_weight_only_mtp, (
            "GPTModel with mtp_load_weight_only=True should contain a WeightOnlyMTPLayer"
        )

    def test_weight_only_mtp_params_marked(self):
        """Test that all params in WeightOnlyMTPLayer have is_weight_only_mtp=True."""
        weight_only_params = self.gpt_model._get_weight_only_params()
        assert len(weight_only_params) > 0, (
            "Model with mtp_load_weight_only=True should have weight-only params"
        )

        for param in weight_only_params:
            assert getattr(param, "is_weight_only_mtp", False), (
                "Weight-only MTP param should have is_weight_only_mtp=True"
            )

    def test_weight_only_mtp_forward_is_noop(self):
        """Test that WeightOnlyMTPLayer.forward is a no-op (returns input unchanged)."""
        mtp_layer = None
        for layer in self.gpt_model.run_function:
            if isinstance(layer, WeightOnlyMTPLayer):
                mtp_layer = layer
                break
        assert mtp_layer is not None

        test_dict = {
            "hidden_states": paddle.randn([2, 64, 512]),
            "some_key": "some_value",
        }
        output = mtp_layer.forward(test_dict)
        assert output is test_dict, (
            "WeightOnlyMTPLayer.forward should return the input dict unchanged"
        )

    def test_offload_weight_only_params(self):
        """Test offloading weight-only MTP params from GPU to CPU pinned memory."""
        weight_only_params = self.gpt_model._get_weight_only_params()
        assert len(weight_only_params) > 0

        # First reload to make sure all params are on GPU
        self.gpt_model.reload_weight_only_params()
        for param in self.gpt_model._get_weight_only_params():
            assert param.place.is_gpu_place(), (
                f"After reload, param should be on GPU but is on {param.place}"
            )

        # Offload to CPU
        self.gpt_model.offload_weight_only_params()
        for param in self.gpt_model._get_weight_only_params():
            assert not param.place.is_gpu_place(), (
                f"After offload, param should be on CPU but is on {param.place}"
            )

    def test_reload_weight_only_params(self):
        """Test reloading weight-only MTP params from CPU back to GPU."""
        # Offload first
        self.gpt_model.offload_weight_only_params()
        for param in self.gpt_model._get_weight_only_params():
            assert not param.place.is_gpu_place(), (
                f"After offload, param should be on CPU but is on {param.place}"
            )

        # Reload back to GPU
        self.gpt_model.reload_weight_only_params()
        for param in self.gpt_model._get_weight_only_params():
            assert param.place.is_gpu_place(), (
                f"After reload, param should be on GPU but is on {param.place}"
            )

    def test_offload_reload_preserves_values(self):
        """Test that offload and reload preserves parameter values."""
        # Make sure params are on GPU first
        self.gpt_model.reload_weight_only_params()

        # Record original values
        original_values = {}
        for name, param in self.gpt_model.state_dict().items():
            if getattr(param, "is_weight_only_mtp", False):
                original_values[name] = param.numpy().copy()

        assert len(original_values) > 0

        # Offload -> Reload
        self.gpt_model.offload_weight_only_params()
        self.gpt_model.reload_weight_only_params()

        # Verify values are preserved
        for name, param in self.gpt_model.state_dict().items():
            if name in original_values:
                np.testing.assert_allclose(
                    param.numpy(),
                    original_values[name],
                    rtol=0,
                    atol=0,
                    err_msg=f"Parameter {name} values changed after offload/reload",
                )

    def test_offload_idempotent(self):
        """Test that calling offload multiple times is safe."""
        self.gpt_model.reload_weight_only_params()
        self.gpt_model.offload_weight_only_params()
        self.gpt_model.offload_weight_only_params()  # second offload should be no-op

        for param in self.gpt_model._get_weight_only_params():
            assert not param.place.is_gpu_place()

    def test_reload_idempotent(self):
        """Test that calling reload multiple times is safe."""
        self.gpt_model.offload_weight_only_params()
        self.gpt_model.reload_weight_only_params()
        self.gpt_model.reload_weight_only_params()  # second reload should be no-op

        for param in self.gpt_model._get_weight_only_params():
            assert param.place.is_gpu_place()

    def test_forward_backward_with_weight_only_mtp(self):
        """Test that model forward/backward works with weight-only MTP (MTP layer is no-op)."""
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        sequence_length = self.config.max_sequence_length
        micro_batch_size = 1

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        attention_mask = paddle.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        labels = paddle.to_tensor(
            list(range(1, sequence_length + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))

        # Make sure params are on GPU for forward pass
        self.gpt_model.reload_weight_only_params()

        gpt_pipe_model = NoPipelineParallel(self.gpt_model, self.strategy)
        data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "attention_mask": [attention_mask],
            },
            [labels],
        )
        loss = gpt_pipe_model.forward_backward_pipeline(data)

        assert loss is not None, "Loss should not be None"
        assert not paddle.isnan(loss).any(), "Loss should not contain NaN"
        assert not paddle.isinf(loss).any(), "Loss should not contain Inf"
        print(f"Loss with weight_only_mtp: {loss.item()}")


if __name__ == "__main__":
    unittest.main()
