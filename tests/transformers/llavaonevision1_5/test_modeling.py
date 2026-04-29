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

import unittest

import paddle

from paddleformers.transformers import RiceConfig, RiceTransformerPretrainedModel
from tests.testing_utils import gpu_device_initializer


class RiceTransformerModelTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="RiceTransformerModelTest", gpu_id=0)
    def setUp(self):
        self.config = RiceConfig(
            depth=2,
            hidden_size=32,
            embed_dim=32,
            intermediate_size=64,
            num_heads=4,
            in_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=1,
            text_hidden_size=48,
            layer_norm_eps=1e-5,
            _attn_implementation="eager",
        )

    def test_forward_shape(self):
        model = RiceTransformerPretrainedModel(self.config)
        model.eval()
        pixel_values = paddle.randn([4, 3 * 14 * 14], dtype="float32")
        grid_thw = paddle.to_tensor([[1, 2, 2]], dtype="int64")

        with paddle.no_grad():
            output = model(pixel_values, grid_thw)

        self.assertEqual(output.shape, [1, 48])

    def test_verify_forward_shape_before_merger(self):
        model = RiceTransformerPretrainedModel(self.config)
        model.eval()
        pixel_values = paddle.randn([4, 3 * 14 * 14], dtype="float32")
        grid_thw = paddle.to_tensor([[1, 2, 2]], dtype="int64")

        with paddle.no_grad():
            output = model(pixel_values, grid_thw, is_verifying=True)

        self.assertEqual(output.shape, [4, 32])


if __name__ == "__main__":
    unittest.main()
