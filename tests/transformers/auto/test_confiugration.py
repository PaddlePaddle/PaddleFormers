# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Hugging Face inc.
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
from __future__ import annotations

import json
import os
import random
import tempfile
import unittest

from paddlenlp.transformers import AutoConfig
from paddlenlp.transformers.auto.configuration import CONFIG_MAPPING
from paddlenlp.transformers.bert.configuration import BertConfig
from paddlenlp.utils.env import CONFIG_NAME

from ...utils.test_module.custom_configuration import CustomConfig


class AutoConfigTest(unittest.TestCase):
    def test_built_in_model_class_config(self):
        config = AutoConfig.from_pretrained("bert-base-uncased")
        number = random.randint(0, 10000)
        self.assertEqual(config.hidden_size, 768)

        config.hidden_size = number

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            # there is no architectures in config.json
            with open(os.path.join(tempdir, AutoConfig.config_file), "r", encoding="utf-8") as f:
                config_data = json.load(f)

            self.assertNotIn("architectures", config_data)

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            self.assertEqual(auto_config.hidden_size, number)

    def test_community_model_class(self):
        # OPT model do not support PretrainedConfig, but can load it as the AutoConfig object
        config = AutoConfig.from_pretrained("facebook/opt-125m")

        self.assertEqual(config.hidden_size, 768)

        number = random.randint(0, 10000)
        config.hidden_size = number

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            self.assertEqual(auto_config.hidden_size, number)

    @unittest.skip("skipping due to connection error!")
    def test_from_hf_hub(self):
        config = AutoConfig.from_pretrained("facebook/opt-66b", from_hf_hub=True)
        self.assertEqual(config.hidden_size, 9216)

    @unittest.skip("skipping due to connection error!")
    def test_from_aistudio(self):
        config = AutoConfig.from_pretrained("PaddleNLP/tiny-random-bert", from_aistudio=True)
        self.assertEqual(config.hidden_size, 32)

    # def test_subfolder(self):
    #     config = AutoConfig.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    #     self.assertEqual(config.hidden_size, 768)

    def test_load_from_legacy_config(self):
        number = random.randint(0, 10000)
        legacy_config = {"init_class": "BertModel", "hidden_size": number}
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, AutoConfig.legacy_config_file), "w", encoding="utf-8") as f:
                json.dump(legacy_config, f, ensure_ascii=False)

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            self.assertEqual(auto_config.hidden_size, number)

    def test_new_config_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            # Wrong model type will raise an error
            with self.assertRaises(ValueError):
                AutoConfig.register("model", CustomConfig)
            # Trying to register something existing in the PaddleNLP library will raise an error
            with self.assertRaises(ValueError):
                AutoConfig.register("bert", BertConfig)

            # Now that the config is registered, it can be used as any other config with the auto-API
            config = CustomConfig()
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir)
                new_config = AutoConfig.from_pretrained(tmp_dir)
                self.assertIsInstance(new_config, CustomConfig)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]

    def test_from_pretrained_cache_dir(self):
        model_id = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            AutoConfig.from_pretrained(model_id, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_id, CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_id, model_id)))

    def test_load_from_custom_arch(self):
        config_dict = {
            "alibi": False,
            "architectures": ["LlamaModelForScore"],
            "bias": False,
            "bos_token_id": 1,
            "do_normalize": False,
            "eos_token_id": 2,
            "fuse_attention_ffn": False,
            "fuse_attention_qkv": False,
            "fuse_sequence_parallel_allreduce": False,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "no_recompute_layers": None,
            "normalizer_type": None,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 32000,
            "paddlenlp_version": None,
            "pp_recompute_interval": 1,
            "recompute_granularity": "full",
            "rms_norm_eps": 1e-06,
            "rope_scaling_factor": 1.0,
            "rope_scaling_type": None,
            "score_dim": 1,
            "score_type": "reward",
            "seq_length": 2048,
            "sequence_parallel": False,
            "tensor_parallel_output": True,
            "tie_word_embeddings": False,
            "transformers_version": "4.28.1",
            "use_flash_attention": False,
            "use_fused_rms_norm": False,
            "use_fused_rope": False,
            "use_recompute": False,
            "virtual_pp_degree": 1,
            "vocab_size": 32001,
        }
        config_str = json.dumps(config_dict, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with tempfile.TemporaryDirectory() as tempdir:
            cache_dir = os.path.join(tempdir, "cache_dir")
            model_dir = os.path.join(tempdir, "custom_model")
            os.mkdir(cache_dir)
            os.mkdir(model_dir)
            json_file_path = os.path.join(model_dir, AutoConfig.config_file)
            with open(json_file_path, "w", encoding="utf-8") as writer:
                writer.write(config_str)
            config = AutoConfig.from_pretrained(model_dir, cache_dir=cache_dir)
            self.assertTrue(config.__class__.__name__ == "LlamaConfig")
