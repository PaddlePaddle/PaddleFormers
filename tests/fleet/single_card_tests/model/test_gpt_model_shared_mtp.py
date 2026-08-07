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
    MultiTokenPredictionLayer,
)
from paddleformers.fleet.transformer.transformer_layer import TransformerLayer


class TestMTPSharedLastLayer(unittest.TestCase):
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
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            moe_expert_fusion=False,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
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
            use_qk_norm=True,
            num_nextn_predict_layers=1,
            mtp_shared_last_layer=True,
        )
        cls.model = gpt_builder(cls.config, num_stages=1)

    def _find_last_transformer_layer(self, model):
        layers = []
        for layer in model.run_function:
            if isinstance(layer, TransformerLayer):
                layers.append(layer)
        assert layers, "No TransformerLayer found"
        return layers[-1]

    def _find_mtp_layer(self, model):
        for layer in model.run_function:
            if isinstance(layer, MultiTokenPredictionLayer):
                return layer
        return None

    def test_shared_parameters(self):
        """MTP layer and last backbone TransformerLayer should share parameters."""
        last_tl = self._find_last_transformer_layer(self.model)
        mtp_layer = self._find_mtp_layer(self.model)
        assert mtp_layer is not None, "MTP layer not found"

        backbone_params = dict(last_tl.transformer_layer_weights)
        mtp_params = dict(mtp_layer.transformer_layer_weights)

        assert len(backbone_params) > 0, "Backbone has no parameters"
        assert len(mtp_params) > 0, "MTP has no parameters"

        # MTP内部transformer的参数应该和backbone最后一层是同一个对象
        for name, mtp_param in mtp_params.items():
            assert name in backbone_params, (
                f"Parameter {name} in MTP not found in backbone last layer"
            )
            assert mtp_param.data_ptr() == backbone_params[name].data_ptr(), (
                f"Parameter {name} not shared (different data_ptr)"
            )

    def test_forward_backward(self):
        """Forward/backward pass should work with mtp_shared_last_layer=True."""
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        seq_len = self.config.max_sequence_length
        micro_batch_size = 1

        data = list(range(seq_len))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        labels = paddle.to_tensor(
            list(range(1, seq_len + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))

        gpt_pipe_model = NoPipelineParallel(self.model, self.strategy)
        data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
            },
            [labels],
        )
        loss = gpt_pipe_model.forward_backward_pipeline(data)

        assert loss is not None, "Loss should not be None"
        assert not paddle.isnan(loss).any(), "Loss contains NaN"
        assert not paddle.isinf(loss).any(), "Loss contains Inf"
        print(f"Loss with mtp_shared_last_layer: {loss.item()}")

    def test_config_incompatible_with_use_dense_mtp(self):
        """mtp_shared_last_layer=True + use_dense_mtp=True should raise AssertionError."""
        with self.assertRaises(AssertionError) as ctx:
            GPTConfig(
                num_hidden_layers=2,
                hidden_size=512,
                vocab_size=100,
                max_sequence_length=64,
                num_attention_heads=4,
                intermediate_size=1024,
                normalization="RMSNorm",
                num_nextn_predict_layers=1,
                mtp_shared_last_layer=True,
                use_dense_mtp=True,
            )
        self.assertIn("mtp_shared_last_layer", str(ctx.exception))

    def test_not_shared_when_disabled(self):
        """With mtp_shared_last_layer=False, MTP and backbone should NOT share params."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            moe_expert_fusion=False,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
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
            use_qk_norm=True,
            num_nextn_predict_layers=1,
            mtp_shared_last_layer=False,
        )
        model = gpt_builder(config, num_stages=1)
        last_tl = self._find_last_transformer_layer(model)
        mtp_layer = self._find_mtp_layer(model)
        assert mtp_layer is not None

        backbone_params = dict(last_tl.transformer_layer_weights)
        mtp_params = dict(mtp_layer.transformer_layer_weights)
        shared_count = sum(
            1
            for name in mtp_params
            if name in backbone_params
            and mtp_params[name].data_ptr() == backbone_params[name].data_ptr()
        )
        self.assertEqual(shared_count, 0)

    def test_non_last_layer_not_shared(self):
        """Non-last transformer layers should not share params with MTP."""
        layers = []
        for layer in self.model.run_function:
            if isinstance(layer, TransformerLayer):
                layers.append(layer)
        assert len(layers) > 1
        first_tl = layers[0]
        mtp_layer = self._find_mtp_layer(self.model)

        first_params = dict(first_tl.transformer_layer_weights)
        mtp_params = dict(mtp_layer.transformer_layer_weights)
        shared_count = sum(
            1
            for name in mtp_params
            if name in first_params
            and mtp_params[name].data_ptr() == first_params[name].data_ptr()
        )
        self.assertEqual(shared_count, 0)

    def test_mtp_weights_delegate_to_inner_transformer(self):
        """MTP.transformer_layer_weights should equal its inner transformer_layer.named_parameters."""
        mtp_layer = self._find_mtp_layer(self.model)
        assert mtp_layer is not None
        mtp_weights = dict(mtp_layer.transformer_layer_weights)
        inner_weights = dict(mtp_layer.transformer_layer.named_parameters())
        self.assertEqual(set(mtp_weights.keys()), set(inner_weights.keys()))


if __name__ == "__main__":
    unittest.main()
