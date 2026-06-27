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

import unittest

from paddleformers.fleet.recompute_utils import need_full_recompute
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestNeedRescompute(unittest.TestCase):
    def test_recompute_no_pp(self):
        config_1 = TransformerConfig(
            num_hidden_layers=8,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        res_1 = {
            0: True,
            1: True,
            2: True,
            3: True,
            4: True,
            5: True,
            6: True,
            7: True,
        }
        for layer_id, res in res_1.items():
            assert need_full_recompute(layer_id, config_1) == res

        config_2 = TransformerConfig(
            num_hidden_layers=8,
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=4,
        )
        res_2 = {
            0: True,
            1: True,
            2: True,
            3: True,
            4: False,
            5: False,
            6: False,
            7: False,
        }
        for layer_id, res in res_2.items():
            assert need_full_recompute(layer_id, config_2) == res

        config_3 = TransformerConfig(
            num_hidden_layers=8,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=4,
        )
        res_3 = {
            0: True,
            1: True,
            2: True,
            3: True,
            4: False,
            5: False,
            6: False,
            7: False,
        }
        for layer_id, res in res_3.items():
            assert need_full_recompute(layer_id, config_3) == res

    def test_recompute_with_pp(self):
        config_1 = TransformerConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        res_1 = {
            0: True,
            1: True,
            2: True,
            3: True,
            4: True,
            5: True,
            6: True,
            7: True,
        }
        for layer_id, res in res_1.items():
            assert need_full_recompute(layer_id, config_1) == res

        config_2 = TransformerConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=2,
        )
        res_2 = {
            0: True,
            1: True,
            2: False,
            3: False,
            4: True,
            5: True,
            6: False,
            7: False,
        }
        for layer_id, res in res_2.items():
            assert need_full_recompute(layer_id, config_2) == res

        config_3 = TransformerConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=4,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=1,
        )
        res_3 = {
            0: True,
            1: False,
            2: True,
            3: False,
            4: True,
            5: False,
            6: True,
            7: False,
        }
        for layer_id, res in res_3.items():
            assert need_full_recompute(layer_id, config_3) == res

        config_4 = TransformerConfig(
            num_hidden_layers=9,
            pipeline_model_parallel_size=4,
            num_empty_layers_add_in_head=1,
            num_empty_layers_add_in_tail=2,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=1,
        )
        res_4 = {
            1: False,
            2: False,
            3: True,
            4: False,
            5: False,
            6: True,
            7: False,
            8: True,
            9: False,
        }
        for layer_id, res in res_3.items():
            assert need_full_recompute(layer_id, config_3) == res

    def test_recompute_with_vpp(self):
        config_1 = TransformerConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=4,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        res_1 = {
            0: True,
            1: True,
            2: True,
            3: True,
            4: True,
            5: True,
            6: True,
            7: True,
        }
        for layer_id, res in res_1.items():
            assert need_full_recompute(layer_id, config_1) == res

        config_2 = TransformerConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=4,
            recompute_granularity="full",
            recompute_method="first_n",
            recompute_num_layers=1,
        )
        res_2 = {
            0: True,
            1: True,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
        }
        for layer_id, res in res_2.items():
            assert need_full_recompute(layer_id, config_2) == res

        config_3 = TransformerConfig(
            num_hidden_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=4,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=1,
        )
        res_3 = {
            0: True,
            1: True,
            2: True,
            3: True,
            4: True,
            5: True,
            6: True,
            7: True,
        }
        for layer_id, res in res_3.items():
            assert need_full_recompute(layer_id, config_3) == res


if __name__ == "__main__":
    unittest.main()
