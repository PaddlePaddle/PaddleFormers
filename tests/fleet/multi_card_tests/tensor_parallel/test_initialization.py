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

# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle
import pytest

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


class Test:
    transformer_config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    def get_tp1_weights(self):
        paddle.manual_seed(42)
        vocab_tp1 = VocabParallelEmbedding(
            num_embeddings=16,
            embedding_dim=4,
            init_method=self.transformer_config.init_method,
            config=self.transformer_config,
        ).weight

        paddle.manual_seed(42)
        row_tp1 = RowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=False,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        paddle.manual_seed(42)
        col_tp1 = ColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        return vocab_tp1, row_tp1, col_tp1

    def get_tp4_weights(self):
        Utils.initialize_model_parallel(4, 1)

        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        vocab_tp4 = VocabParallelEmbedding(
            num_embeddings=16,
            embedding_dim=4,
            init_method=self.transformer_config.init_method,
            config=self.transformer_config,
        ).weight

        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        row_tp4 = RowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=False,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        col_tp4 = ColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        return vocab_tp4, row_tp4, col_tp4

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="CUDA not available"
    )
    def test_init(self):
        vocab_tp1, row_tp1, col_tp1 = self.get_tp1_weights()
        vocab_tp4, row_tp4, col_tp4 = self.get_tp4_weights()

        rank = ps.get_tensor_model_parallel_rank()
        assert vocab_tp4.shape[0] * 4 == vocab_tp1.shape[0]
        assert paddle.equal_all(vocab_tp1[rank * 4 : (rank + 1) * 4], vocab_tp4)

        assert row_tp4.shape[0] * 4 == row_tp1.shape[0]
        assert paddle.equal_all(row_tp1[rank * 4 : (rank + 1) * 4, :], row_tp4)

        assert col_tp4.shape[1] * 4 == col_tp1.shape[1]
        assert paddle.equal_all(col_tp1[:, rank * 4 : (rank + 1) * 4], col_tp4)


if __name__ == "__main__":
    test_obj = Test()
    test_obj.test_init()
