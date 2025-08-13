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

import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn as nn

from ..transformers.configuration_utils import PretrainedConfig
from .general import GeneralInterface

__all__ = ["Embedding"]


class Embedding(GeneralInterface):
    _global_mapping = {
        "default": nn.Embedding,
        "vocab_parallel": mpu.VocabParallelEmbedding,
    }

    @classmethod
    def create(self, num_embeddings, embedding_dim, weight_attr=None, name=None, **kwargs):
        embedding_type = kwargs.pop("embedding_type", "default")
        embdding_cls = self._global_mapping[embedding_type]
        kwargs = self.process_kwargs(embedding_type, **kwargs)
        return embdding_cls(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, weight_attr=weight_attr, name=name, **kwargs
        )

    @classmethod
    def from_config(self, config: PretrainedConfig):
        return self.create(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            embedding_type=self.get_embedding_type(config),
        )

    @classmethod
    def process_kwargs(self, embedding_type, **kwargs):
        if embedding_type == "default":
            kwargs.pop("mp_group", None)
        elif embedding_type == "vocab_parallel":
            pop_keys = ["padding_idx", "sparse"]
            for key in pop_keys:
                kwargs.pop(key, None)
        return kwargs

    @classmethod
    def get_embedding_type(self, config: PretrainedConfig):
        if config.tensor_parallel_degree <= 1:
            return "default"
        return "vocab_parallel"
