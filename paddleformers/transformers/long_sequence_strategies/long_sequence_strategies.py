# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import importlib

all_strategy_types = ["embedding_strategies", "attention_strategies"]


class LongSequenceStrategies:
    @classmethod
    def build_long_sequence_strategy(cls, strategy_type=None, stratety_name=None, **init_args):
        """

        **init_args:   head_dim,
                       max_position_embeddings,
                       rope_scaling_type,
                       rope_scaling_factor,
                       ...

        strategy_type: "None" ---------------走原始的built-in模块
                       "embedding_strategies"、
                       "attention_strategies"
                       ...

        stratety_name: "RotaryEmbedding"、
                       "LinearScalingRotaryEmbedding"、
                       "NTKScalingRotaryEmbedding"、
                       "DynamicNTKScalingRotaryEmbedding"、
                       "AttentionWithLinearBias"
                       ...

        """

        """
        paddleformers.transformers.long_sequence_strategies.{strategy_type<->import_class)}.{stratety_name<->strategy_class)}
        paddleformers.transformers.long_sequence_strategies.{embedding_strategies}.{RoPE,...}
        paddleformers.transformers.long_sequence_strategies.{attention_strategies}.{ALiBi,...}
        """
        try:
            import_class = importlib.import_module(
                f"paddleformers.transformers.long_sequence_strategies.{strategy_type}"
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Wrong strategy type {strategy_type}. module only supports the following types: "
                + ", ".join(m for m in all_strategy_types)
            )
        try:
            strategy_class = getattr(import_class, stratety_name)
        except:
            all_strategy_classes = import_class.__all__
            raise LookupError(
                f"module '{import_class.__name__}' only supports the following classes: "
                + ", ".join(m for m in all_strategy_classes)
            )
        strategy_instance = strategy_class(**init_args)
        return strategy_instance
