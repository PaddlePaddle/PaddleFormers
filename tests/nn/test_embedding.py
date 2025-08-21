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

import sys
import unittest
from pathlib import Path

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet

from paddleformers.nn.embedding import Embedding
from paddleformers.transformers import LlamaConfig
from tests.parallel_launch import TestMultipleGpus
from tests.testing_utils import require_paddle_at_least_2_gpu

sys.path.append(str(Path(__file__).parent.parent))

tp_size = paddle.distributed.get_world_size()
tp_rank = 0
if tp_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tp_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    tp_rank = hcg.get_model_parallel_rank()
    mp_group = hcg.get_model_parallel_group()


def _test_create_vocab_parallel_embedding():
    # Test creating default embedding
    config = LlamaConfig()
    config.tensor_parallel_degree = tp_size
    embedding = Embedding.create(config=config)
    assert isinstance(embedding, mpu.VocabParallelEmbedding)
    assert embedding.weight.shape == [config.vocab_size // tp_size, config.hidden_size]
    print("paddleformers.nn.Embedding: _test_create_vocab_parallel_embedding: success")


@require_paddle_at_least_2_gpu
class TestEmbedding(TestMultipleGpus):
    def setUp(self):
        super().setUp()
        self.config = LlamaConfig()
        self.num_embeddings = self.config.vocab_size
        self.embedding_dim = self.config.hidden_size

    def test_create_default(self):
        # Test creating default embedding
        embedding = Embedding.create(self.config)
        self.assertIsInstance(embedding, nn.Embedding)

        self.config.vocab_size = None
        self.config.hidden_size = None
        embedding = Embedding.create(self.config, num_embeddings=40, embedding_dim=38)
        self.assertIsInstance(embedding, nn.Embedding)
        print("paddleformers.nn.Embedding: test_create_default: success")

    def test_create_with_optional_params(self):
        # Test with optional parameters
        name = "test_embedding"

        embedding = Embedding.create(
            config=self.config,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            name=name,
            padding_idx=0,
            sparse=True,
        )

        self.assertIsInstance(embedding, nn.Embedding)
        self.assertEqual(embedding._name, name)
        print("paddleformers.nn.Embedding: test_create_with_optional_params: success")

    def test_process_kwargs_default(self):
        # Test kwargs processing for default embedding
        kwargs = {"mp_group": "dummy", "padding_idx": 0, "sparse": True, "other_param": "value"}

        processed = Embedding.process_kwargs("default", **kwargs)

        self.assertNotIn("mp_group", processed)
        self.assertIn("padding_idx", processed)
        self.assertIn("sparse", processed)
        self.assertIn("other_param", processed)
        print("paddleformers.nn.Embedding: test_process_kwargs_default: success")

    def test_process_kwargs_vocab_parallel(self):
        # Test kwargs processing for vocab parallel embedding
        kwargs = {"mp_group": "dummy", "padding_idx": 0, "sparse": True, "other_param": "value"}

        processed = Embedding.process_kwargs("vocab_parallel", **kwargs)

        self.assertIn("mp_group", processed)
        self.assertNotIn("padding_idx", processed)
        self.assertNotIn("sparse", processed)
        self.assertIn("other_param", processed)
        print("paddleformers.nn.Embedding: test_process_kwargs_vocab_parallel: success")

    # def test_create_vocab_parallel_embedding(self):
    #     self.run_2gpu(__file__)

    def test_register_new_embedding(self):
        class MyEmbedding(nn.Layer):
            def __init__(self, num_embeddings, embedding_dim, **kwargs):
                super().__init__()
                self.weight = self.create_parameter(shape=[num_embeddings, embedding_dim], is_bias=False)

            def forward(self, x):
                return F.embedding(x, weight=self.weight)

        Embedding.register("my_embedding", MyEmbedding)
        my_embedding = Embedding.create(
            config=self.config,
            embedding_type="my_embedding",
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
        self.assertIsInstance(my_embedding, MyEmbedding)
        my_embedding(paddle.to_tensor([0]))
        print("paddleformers.nn.Embedding: test_register_new_embedding: success")


if __name__ == "__main__":
    # _test_create_vocab_parallel_embedding()
    unittest.main()
