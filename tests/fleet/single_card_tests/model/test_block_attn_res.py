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


class TestBlockAttnRes(unittest.TestCase):
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
        ps.initialize_model_parallel(hcg)
        self.strategy = strategy

    def test_block_attn_res(self):
        config = GPTConfig(
            num_hidden_layers=4,
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
            block_attention_residuals=True,
            attn_res_block_size=2,
        )
        gpt_model = gpt_builder(config, num_stages=1)

        sequence_length = config.max_sequence_length
        micro_batch_size = 1

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        attention_mask = paddle.ones(
            (micro_batch_size, 1, sequence_length, sequence_length),
            dtype=bool,
        )
        labels = paddle.to_tensor(
            list(range(1, sequence_length + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))

        gpt_pipe_model = NoPipelineParallel(gpt_model, self.strategy)
        data = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "attention_mask": [attention_mask],
            },
            [labels],
        )

        loss = gpt_pipe_model.forward_backward_pipeline(data)

        # Verify loss is finite
        assert paddle.isfinite(loss).item(), (
            f"loss is not finite: {loss.item()}"
        )
        print("block_attn_res loss", loss.item())

        # Verify gradients exist and are finite
        for name, param in gpt_model.named_parameters():
            assert param.grad is not None, f"param {name} has no gradient"
            grad_norm = param.grad.detach().norm().item()
            assert np.isfinite(grad_norm), (
                f"param {name} has non-finite gradient: {grad_norm}"
            )

        # Verify block_attn_res parameters have gradients
        has_block_attn_res_param = False
        for name, param in gpt_model.named_parameters():
            if "block_attn_res" in name:
                has_block_attn_res_param = True
                assert param.grad is not None, (
                    f"block_attn_res param {name} has no gradient"
                )
        assert has_block_attn_res_param, (
            "No block_attn_res parameters found in model"
        )


if __name__ == "__main__":
    unittest.main()
