# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import tempfile
import unittest
from typing import Dict, Optional

from paddleformers.transformers import Qwen3Config
from paddleformers.transformers.configuration_utils import (
    PretrainedConfig,
    attribute_map,
)
from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.utils import CONFIG_NAME
from paddleformers.utils.env import LEGACY_CONFIG_NAME
from tests.testing_utils import slow


class FakeSimplePretrainedModelConfig(PretrainedConfig):
    """simple fake Pretrained Model Config"""

    def __init__(self, a=0, b=1, c=2):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()


class FakePretrainedModelConfig(PretrainedConfig):
    """Fake Pretrained Model which is similar with actual situation"""

    attribute_map: Dict[str, str] = {
        "num_classes": "num_labels",
    }

    def __init__(self, hidden_dropout_prob: float, **kwargs):
        attribute_map(self, kwargs=kwargs)
        super().__init__(**kwargs)
        self.hidden_dropout_prob = hidden_dropout_prob


class FakeLayer:
    def __init__(self, config: Optional[FakeSimplePretrainedModelConfig] = None, *args, **kwargs):
        super(FakeLayer, self).__init__()

        self.a = config.a
        self.b = config.b
        self.c = config.c


class FakeModel(PretrainedModel):
    def __init__(self, config: FakeSimplePretrainedModelConfig):
        """fake `__init__`, the source of parameters is:
            def __init__(self, model, a, b):
                self.model = model
                self.a = a
                self.b = b
        Args:
            config_or_model (Optional[Union[FakeLayer, FakeSimplePretrainedModelConfig]], optional): config or model instance. Defaults to None.
        """
        super().__init__()

        self.model: FakeLayer = FakeLayer(config)
        self.a = config.a
        self.b = config.b


class ConfigurationUtilsTest(unittest.TestCase):
    def test_parse_config_with_single_config(self):
        # 1. single config
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        model = FakeLayer(config)
        assert model.a == 10
        assert model.b == 11

    def test_model_config_save(self):
        # 1. single config
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        config.fuse_attention_qkv = True
        config.fuse_rms_norm = True
        config.tensor_model_parallel_size = 8
        config.tensor_parallel_output = True

        config.quantization_config.quant_type = "weight_only_int8"
        str_config = str(config)
        assert "tensor_model_parallel_size" in str_config

        config.test_nonsave = "test"
        config.test_nonsave_2 = "test"
        config.register_unsavable_keys(["test_nonsave"])

        with tempfile.TemporaryDirectory() as tp:
            config.save_pretrained(tp)
            import json

            loaded_config = json.load(open(os.path.join(tp, "config.json"), "r"))
            assert "fuse_attention_qkv" in loaded_config, "fuse qkv is need to save"
            assert "fuse_rms_norm" not in loaded_config, "fuse_rms_norm don't need to save"
            assert "tensor_model_parallel_size" in loaded_config, "tensor_model_parallel_size need to save"
            assert "paddleformers_version" in loaded_config, "always save paddleformers_version"
            assert (
                "quantization_config" in loaded_config and "quant_type" in loaded_config["quantization_config"]
            ), "missing quantization_config"
            assert "test_nonsave" not in loaded_config
            assert "test_nonsave_2" in loaded_config

    def test_parse_config_and_model_with_single_config(self):
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        model = FakeModel(config)
        assert model.a == 10
        assert model.b == 11

    def test_get_value_with_default_from_config(self):
        config = FakeSimplePretrainedModelConfig(a=10)
        assert config.get("a", None) == 10
        assert config.get("a", None) == config.a
        assert config.get("no_name", 0) == 0


class StandardConfigMappingTest(unittest.TestCase):
    def test_qwen3_config_mapping(self):
        # create new fake-qwen3 class to prevent static-attributed modified by this test
        class FakeQwen3Config(Qwen3Config):
            pass

        config = FakeQwen3Config.from_pretrained("Paddleformers/tiny-random-qwen3")
        hidden_size = config.hidden_size

        FakeQwen3Config.attribute_map = {"fake_field": "hidden_size"}

        loaded_config = FakeQwen3Config.from_pretrained("Paddleformers/tiny-random-qwen3")
        fake_field = loaded_config.fake_field
        self.assertEqual(fake_field, hidden_size)

    @slow
    def test_from_pretrained_cache_dir(self):
        model_id = "Paddleformers/tiny-random-qwen3"
        with tempfile.TemporaryDirectory() as tempdir:
            Qwen3Config.from_pretrained(model_id, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_id, CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_id, model_id)))

    def test_load_from_hf(self):
        """test load config from hf"""
        config = Qwen3Config.from_pretrained("Paddleformers/tiny-random-qwen3", download_hub="huggingface")
        self.assertEqual(config.hidden_size, 4096)

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            self.assertTrue(os.path.exists(os.path.join(tempdir, CONFIG_NAME)))

            loaded_config = Qwen3Config.from_pretrained(tempdir)
            self.assertEqual(loaded_config.hidden_size, 4096)

    def test_config_mapping(self):
        # create new fake-qwen3 class to prevent static-attributed modified by this test
        class FakeQwen3Config(Qwen3Config):
            pass

        with tempfile.TemporaryDirectory() as tempdir:
            config = FakeQwen3Config.from_pretrained("PaddleFormers/tiny-random-qwen3")
            config.save_pretrained(tempdir)

            # rename `config.json` -> `model_config.json`
            shutil.move(os.path.join(tempdir, CONFIG_NAME), os.path.join(tempdir, LEGACY_CONFIG_NAME))

            FakeQwen3Config.attribute_map = {"fake_field": "hidden_size"}

            loaded_config = FakeQwen3Config.from_pretrained(tempdir)
            self.assertEqual(loaded_config.fake_field, config.hidden_size)


class TestTensorParallelConveter(unittest.TestCase):
    def test_qkv_convertor(self):
        """test_qkv_convertor"""
        hidden_size = 8
        tensor_model_parallel_size = 4
        num_attention_heads = 4
        # head_dim = hidden_size // num_attention_heads
        import numpy as np

        from paddleformers.transformers.conversion_utils import (
            naive_merged_qkv_to_tensor_parallel_qkv,
            normal_fuse_merge_tp,
            normal_fuse_split_tp,
            tensor_parallel_qkv_to_naive_merged_qkv,
        )

        naive_merged_qkv = np.arange(3 * hidden_size * hidden_size).reshape([hidden_size, -1])
        tensor_parallel_qkv = naive_merged_qkv_to_tensor_parallel_qkv(naive_merged_qkv, num_attention_heads)
        new_naive_merged_qkv = tensor_parallel_qkv_to_naive_merged_qkv(tensor_parallel_qkv, num_attention_heads)
        np.testing.assert_equal(new_naive_merged_qkv, naive_merged_qkv)
        # print("tensor_parallel_qkv", tensor_parallel_qkv)
        np.testing.assert_equal(
            tensor_parallel_qkv[0],
            [0, 1, 8, 9, 16, 17, 2, 3, 10, 11, 18, 19, 4, 5, 12, 13, 20, 21, 6, 7, 14, 15, 22, 23],
        )

        mp_qkv_splited = normal_fuse_split_tp(tensor_parallel_qkv, tensor_model_parallel_size)
        new_tensor_parallel_qkv = normal_fuse_merge_tp(mp_qkv_splited)
        # print("mp_qkv_splited", mp_qkv_splited[0])
        np.testing.assert_equal(new_tensor_parallel_qkv, tensor_parallel_qkv)
        np.testing.assert_equal(mp_qkv_splited[0][0], [0, 1, 8, 9, 16, 17])
