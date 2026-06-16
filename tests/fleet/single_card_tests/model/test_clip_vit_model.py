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

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddleformers.fleet.models.vision.clip_vit_model import CLIPViTModel
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestCLIPViTModel(unittest.TestCase):
    """Test CLIP ViT model."""

    def setUp(self):
        if not ps.have_global_memory_buffer():
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
            ps.initialize_model_parallel(hcg)

        transformer_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        transformer_layer_spec = get_gpt_layer_local_spec()
        self.model = CLIPViTModel(
            transformer_config,
            transformer_layer_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        )

    def test_constructor(self):
        assert isinstance(self.model, CLIPViTModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])

        assert num_weights == 173312

    def test_set_input_tensor(self):
        # [s, b, h] expected to the transformer.
        expected_shape = (577, 2, 64)
        input_tensor = paddle.zeros(expected_shape)

        self.model.set_input_tensor(input_tensor)
        assert self.model.decoder.input_tensor.shape == list(expected_shape)

    def test_forward(self):
        img = paddle.zeros((2, 3, 336, 336))

        out = self.model.forward(img)
        assert out.shape == [2, 577, 64]


if __name__ == "__main__":
    unittest.main()
