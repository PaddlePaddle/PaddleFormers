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

# from tests.unit_tests.test_utilities import Utils
# from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig


class TestGPTModelStateDict(unittest.TestCase):
    """Test cases for GPTModel state_dict and sharded_state_dict methods."""

    def setUp(self):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
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
        # ps.initialize_model_parallel(hcg)
        self.strategy = strategy

        config = GPTConfig(
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
        )
        self.gpt_model = gpt_builder(config, num_stages=1)
        self.config = config

    def test_state_dict_structure(self):
        """Test that state_dict returns parameters with correct naming structure."""
        # Create model instance
        self.gpt_model._set_pipeline_name_mapping()
        # Call state_dict directly without mocking
        state_dict = self.gpt_model.state_dict()
        print("state_dict: ", state_dict)
        # Verify all keys start with expected prefixes
        valid_prefixes = ("model.", "model.layers.")
        for key in state_dict.keys():
            self.assertTrue(
                key.startswith(valid_prefixes),
                f"Key '{key}' does not start with any of {valid_prefixes}",
            )

    def test_sharded_state_dict_structure(self):
        """Test that sharded_state_dict remaps parameter keys correctly."""
        self.gpt_model._set_pipeline_name_mapping()

        sharded_state_dict = self.gpt_model.sharded_state_dict()
        print("sharded_state_dict: ", sharded_state_dict)
        # Verify all keys start with expected prefixes
        valid_prefixes = ("model.", "model.layers.")
        for key in sharded_state_dict.keys():
            self.assertTrue(
                key.startswith(valid_prefixes),
                f"Key '{key}' does not start with any of {valid_prefixes}",
            )

    def test_check_shared_model_state(self):
        """Test _check_shared_model_state method."""
        self.gpt_model._check_shared_model_state()


if __name__ == "__main__":
    unittest.main()
