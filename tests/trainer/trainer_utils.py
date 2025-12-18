# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy

import numpy as np
import paddle
import paddle.nn as nn

from paddleformers.transformers import PretrainedConfig, PretrainedModel


def get_pretrain_arguments(pretrain_arguments):

    configs = {}

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 8
    train_args["pipeline_model_parallel_size"] = 1
    configs["TP8"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 2
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding"] = ""
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 4
    configs["TP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 4
    train_args["pipeline_model_parallel_size"] = 2
    configs["TP4PP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 4
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding"] = ""
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["TP4DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 4
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["TP4Sharding2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 2
    train_args["pipeline_model_parallel_size"] = 4
    configs["TP2PP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 2
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 4
    configs["TP2Sharding4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 8
    configs["PP8"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 4
    train_args["sharding"] = ""
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["PP4DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 4
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["PP4Sharding2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding8S1"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding8S2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding_parallel_degree"] = 4
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding4S1DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding_parallel_degree"] = 4
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding4S2DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding_parallel_degree"] = 2
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding2S1DP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding_parallel_degree"] = 2
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding2S2DP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 1
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding_parallel_degree"] = 1
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["DP8"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_model_parallel_size"] = 2
    train_args["pipeline_model_parallel_size"] = 1
    train_args["sharding_parallel_degree"] = 2
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 4
    configs["TP2DP2Sharding2"] = train_args

    return configs


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


class RegressionModelConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


class RegressionPretrainedModel(PretrainedModel):
    config_class = RegressionModelConfig
    base_model_prefix = "regression"

    def __init__(self, config):
        super().__init__(config)
        self.a = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.b = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.a.set_value(paddle.to_tensor(config.a, paddle.float32))
        self.b.set_value(paddle.to_tensor(config.b, paddle.float32))
        self.double_output = config.double_output

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y, y) if self.double_output else (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y, y) if self.double_output else (loss, y)
