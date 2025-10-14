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
import glob
import os
import tempfile
import unittest

import paddle

input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])


def prepare_default_config(config):
    config = copy.deepcopy(config)
    config.hidden_size = 512
    config.num_layers = 2
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    config.intermediate_size = config.hidden_size
    config.word_embed_proj_dim = 512
    return config


def prepare_split_config(config):
    config = prepare_default_config(config)
    config = copy.deepcopy(config)
    config.fuse_attention_qkv = False
    config.fuse_attention_ffn = False
    return config


def prepare_fuse_config(config):
    config = prepare_default_config(config)
    config = copy.deepcopy(config)
    config.fuse_attention_qkv = True
    config.fuse_attention_ffn = True
    return config


def common_test_load(model_class, model_first, config_second, tempdir):
    model_first.eval()
    with paddle.no_grad():
        first = model_first(input_ids)[0]

    model_second = model_class.from_pretrained(tempdir, config=config_second)
    model_second.eval()
    with paddle.no_grad():
        second = model_second(input_ids)[0]

    assert paddle.allclose(paddle.mean(first), paddle.mean(second), atol=1e-5)
    # assert paddle.allclose(first, second, atol=1e-4)

    files = glob.glob(tempdir + "/*")
    for f in files:
        os.remove(f)


def common_test_save_and_load(config_first, config_second, model_class):
    model_first = model_class.from_config(config_first)

    with tempfile.TemporaryDirectory() as tempdir:
        # test load pdparams: model.pdparams
        model_first.save_pretrained(save_dir=tempdir)
        common_test_load(model_class, model_first, config_second, tempdir)

        # test load shard pdparams: model-001-0f-008.pdparams
        model_first.save_pretrained(tempdir, max_shard_size="5MB")
        common_test_load(model_class, model_first, config_second, tempdir)

        # test save safetensors: model.safetensors
        model_first.save_pretrained(tempdir, safe_serialization=True)
        common_test_load(model_class, model_first, config_second, tempdir)

        # test load shard safetensors: model-001-0f-008.safetensors
        model_first.save_pretrained(tempdir, max_shard_size="5MB", safe_serialization=True)
        common_test_load(model_class, model_first, config_second, tempdir)


def _test_split_to_fuse(config_class, model_class):
    config = config_class()

    config_split = prepare_split_config(config)
    config_fuse = prepare_fuse_config(config)

    # Test from splitted weights to fused weight
    common_test_save_and_load(config_split, config_fuse, model_class)


def _test_fuse_to_split(config_class, model_class):
    config = config_class()

    config_split = prepare_split_config(config)
    config_fuse = prepare_fuse_config(config)

    # Test from fused weight to splitted weights
    common_test_save_and_load(config_fuse, config_split, model_class)


def _test_fast_ffn():
    from functools import partial

    import paddle
    from paddle import nn

    from paddleformers.transformers import PretrainedModel
    from paddleformers.transformers.configuration_utils import PretrainedConfig

    class TestConfig(PretrainedConfig):
        def __init__(self, fast_ffn_state=False, convert_fast_ffn=False):
            self.fast_ffn_state = fast_ffn_state
            self.convert_fast_ffn = convert_fast_ffn
            super().__init__()

    class TestMLP(nn.Layer):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.hidden_size = config.hidden_size
            self.gate_up_fused_proj = nn.Linear(self.hidden_size, self.hidden_size * 2, bias_attr=True)

        def forward(self, hidden_state):
            hidden_state = self.gate_up_fused_proj(hidden_state)
            if self.config.use_fast_ffn:
                x, y = paddle.chunk(hidden_state, chunks=2, axis=-1)
            else:
                x, y = hidden_state[..., ::2], hidden_state[..., 1::2]

            return nn.functional.silu(x) * y

    class TestPretrainedModel(PretrainedModel):
        config_class = TestConfig

        @classmethod
        def _get_fuse_or_split_param_mappings(cls, config: TestConfig, is_fuse=False):

            #  user defined function to get convert param mappings
            def convert_fast_ffn_fn(fuse_params, convert_fast_ffn=False):
                import numpy as np

                concat_fn = np.concatenate
                if isinstance(fuse_params[0], paddle.Tensor):
                    concat_fn = paddle.cat

                if convert_fast_ffn:
                    # fast_ffn
                    first = fuse_params[0][..., ::2]
                    second = fuse_params[0][..., 1::2]
                    return concat_fn([first, second], axis=-1)

            fn = convert_fast_ffn_fn

            convert_fast_ffn_keys = (
                "layers.0.gate_up_fused_proj.weight",
                "layers.0.gate_up_fused_proj.weight",
            )
            convert_fast_ffn_bias_keys = (
                "layers.0.gate_up_fused_proj.bias",
                "layers.0.gate_up_fused_proj.bias",
            )
            fast_ffn_state = getattr(config, "fast_ffn_state", False)
            convert_fast_ffn = getattr(config, "convert_fast_ffn", False)
            convert_fast_ffn &= not fast_ffn_state

            final_actions = {}
            if is_fuse:
                # for_get_fuse_or_split_param_mappings, is_fuse have two conditions, true and false,
                # to fit partial fuse or split conditions, is_fuse will called twice(True and False).
                # thus, for this func, we only use one condition.

                # use_fast_ffn only in one condition
                # convert when use_fast_ffn is False
                if convert_fast_ffn:
                    for i in range(config.num_hidden_layers):
                        for keys in [convert_fast_ffn_keys, convert_fast_ffn_bias_keys]:
                            keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in keys])
                            final_actions[keys] = partial(fn, convert_fast_ffn=convert_fast_ffn)
            return final_actions

        def _init_weights(self, layer):
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                if isinstance(layer.weight, paddle.Tensor):
                    layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=1.0, shape=layer.weight.shape))
                if hasattr(layer, "bias") and isinstance(layer.bias, paddle.Tensor):
                    layer.bias.set_value(paddle.tensor.normal(mean=0.0, std=1.0, shape=layer.bias.shape))

    class TestModel(TestPretrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.LayerList([TestMLP(config=config) for i in range(config.num_hidden_layers)])

        def forward(self, hidden_state):
            for idx, (decoder_layer) in enumerate(self.layers):
                hidden_state = decoder_layer(hidden_state)
            return hidden_state

    class TestForCausalLM(TestPretrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.embedding_layer = nn.Embedding(65535, self.config.hidden_size)
            self.test = TestModel(config=config)

        def forward(self, input_ids):
            hidden_state = self.embedding_layer(input_ids)
            return self.test(hidden_state)

    config = TestConfig()
    config = prepare_default_config(config)
    config_no_fast_ffn = copy.deepcopy(config)
    config_fast_ffn = copy.deepcopy(config)

    config_no_fast_ffn.use_fast_ffn = False

    config_fast_ffn.use_fast_ffn = True
    config_fast_ffn.fast_ffn_state = False
    config_fast_ffn.convert_fast_ffn = True

    common_test_save_and_load(config_no_fast_ffn, config_fast_ffn, TestForCausalLM)


from paddleformers.transformers import LlamaConfig, LlamaForCausalLM


class TestFuseOrSplit(unittest.TestCase):
    def test_model_split_to_fuse(self):
        _test_split_to_fuse(LlamaConfig, LlamaForCausalLM)

    def test_model_fuse_to_split(self):
        _test_fuse_to_split(LlamaConfig, LlamaForCausalLM)

    def test_model_convert_fast_ffn(self):
        _test_fast_ffn()
