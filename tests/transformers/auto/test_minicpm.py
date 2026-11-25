# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import gc
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import paddle
from safetensors.numpy import save_file

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    MiniCPM4_1Config,
    MiniCPM4_1ForCausalLM,
    MiniCPM4_1Model,
    MiniCPMConfig,
    MiniCPMForCausalLM,
    MiniCPMModel,
)
from tests.testing_utils import slow

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "minicpm"
FAMILIES = {
    "legacy": (MiniCPMConfig, MiniCPMModel, MiniCPMForCausalLM),
    "minicpm4_1": (MiniCPM4_1Config, MiniCPM4_1Model, MiniCPM4_1ForCausalLM),
}


class MiniCPMAutoLoadingTest(unittest.TestCase):
    def _config(self, family, tiny=False):
        with (FIXTURES / family / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        if tiny:
            config.update(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                rope_scaling=None,
                tie_word_embeddings=False,
                eos_token_id=2,
                pad_token_id=0,
            )
        return config

    def _write_config(self, directory, config):
        with (Path(directory) / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle)

    def _write_hf_checkpoint(self, directory, config):
        # Upstream HF names/layouts, including non-square K/V and MLP weights.
        # Keep fusion fields absent, as in the real checkpoint metadata.
        self._write_config(directory, config)
        rng = np.random.default_rng(23)
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        kv = hidden // config["num_attention_heads"] * config["num_key_value_heads"]

        def weight(shape):
            return rng.normal(0, 0.02, shape).astype("float32")

        state = {
            "model.embed_tokens.weight": weight((config["vocab_size"], hidden)),
            "model.norm.weight": np.ones(hidden, dtype="float32"),
            "lm_head.weight": weight((config["vocab_size"], hidden)),
        }
        for index in range(config["num_hidden_layers"]):
            prefix = f"model.layers.{index}."
            for projection, width in [("q", hidden), ("k", kv), ("v", kv), ("o", hidden)]:
                state[prefix + f"self_attn.{projection}_proj.weight"] = weight((width, hidden))
            for projection in ["gate", "up"]:
                state[prefix + f"mlp.{projection}_proj.weight"] = weight((intermediate, hidden))
            state[prefix + "mlp.down_proj.weight"] = weight((hidden, intermediate))
            for norm in ["input_layernorm", "post_attention_layernorm"]:
                state[prefix + norm + ".weight"] = np.ones(hidden, dtype="float32")
        save_file(state, str(Path(directory) / "model.safetensors"), metadata={"format": "pt"})
        return state

    def _assert_forward(self, model, causal_lm):
        model.eval()
        with paddle.no_grad():
            output = model(paddle.to_tensor([[1, 10, 20]], dtype="int64"), use_cache=False, return_dict=True)
        tensor = output.logits if causal_lm else output.last_hidden_state
        width = model.config.vocab_size if causal_lm else model.config.hidden_size
        self.assertEqual(list(tensor.shape), [1, 3, width])
        self.assertTrue(bool(paddle.isfinite(tensor).all().item()))

    def test_auto_config_from_upstream_metadata(self):
        for family, (config_class, _, _) in FAMILIES.items():
            with self.subTest(family=family):
                config = AutoConfig.from_pretrained(str(FIXTURES / family))
                self.assertIs(type(config), config_class)
                self.assertFalse(config.fuse_attention_qkv)
                self.assertFalse(config.fuse_attention_ffn)

    def test_legacy_registration_without_name_metadata(self):
        raw = self._config("legacy")
        raw.pop("_name_or_path", None)
        raw["model_type"] = "minicpm"
        with tempfile.TemporaryDirectory() as directory:
            self._write_config(directory, raw)
            self.assertIs(type(AutoConfig.from_pretrained(directory)), MiniCPMConfig)
        raw = self._config("legacy", tiny=True)
        raw.pop("_name_or_path", None)
        raw["model_type"] = "minicpm"
        self._assert_hf_loading("legacy", raw)

    def test_from_config_preserves_minicpm4_1_class(self):
        # AutoConfig resolves upstream model_type="minicpm" to MiniCPM4_1Config,
        # while the upstream architectures entry still says MiniCPMForCausalLM.
        with tempfile.TemporaryDirectory() as directory:
            self._write_config(directory, self._config("minicpm4_1", tiny=True))
            config = AutoConfig.from_pretrained(directory)
            for auto_class, expected in [(AutoModel, MiniCPM4_1Model), (AutoModelForCausalLM, MiniCPM4_1ForCausalLM)]:
                with self.subTest(auto_class=auto_class.__name__):
                    model = auto_class.from_config(config)
                    self.assertIs(type(model), expected)
                    self._assert_forward(model, auto_class is AutoModelForCausalLM)

    def test_explicit_minicpm4_1_type_without_name_metadata(self):
        raw = self._config("minicpm4_1", tiny=True)
        raw.pop("_name_or_path", None)
        raw["model_type"] = "minicpm4_1"
        self._assert_hf_loading("minicpm4_1", raw)

    def _assert_hf_loading(self, family, raw, remove_fusion_attributes=False):
        config_class, base_class, causal_class = FAMILIES[family]
        with tempfile.TemporaryDirectory() as directory:
            state = self._write_hf_checkpoint(directory, raw)
            for auto_class, expected in [(AutoModel, base_class), (AutoModelForCausalLM, causal_class)]:
                with self.subTest(family=family, auto_class=auto_class.__name__):
                    config = AutoConfig.from_pretrained(directory)
                    self.assertIs(type(config), config_class)
                    config._attn_implementation = "eager"
                    if remove_fusion_attributes:
                        del config.fuse_attention_qkv
                        del config.fuse_attention_ffn
                    model = auto_class.from_pretrained(
                        directory,
                        config=config,
                        dtype="float32",
                        convert_from_hf=True,
                        load_checkpoint_format="flex_checkpoint",
                        load_via_cpu=True,
                    )
                    self.assertIs(type(model), expected)
                    self.assertFalse(model.config.fuse_attention_qkv)
                    self.assertFalse(model.config.fuse_attention_ffn)
                    backbone = model.model if auto_class is AutoModelForCausalLM else model
                    np.testing.assert_array_equal(
                        backbone.layers[0].self_attn.k_proj.weight.numpy(),
                        state["model.layers.0.self_attn.k_proj.weight"].T,
                    )
                    self._assert_forward(model, auto_class is AutoModelForCausalLM)
                    del model

    def test_hf_checkpoint_loading_both_families(self):
        for family in FAMILIES:
            self._assert_hf_loading(family, self._config(family, tiny=True))

    def test_legacy_checkpoint_without_fusion_attributes(self):
        self._assert_hf_loading("legacy", self._config("legacy", tiny=True), remove_fusion_attributes=True)


class MiniCPMRealCheckpointLoadingTest(unittest.TestCase):
    def _assert_checkpoint(self, checkpoint, config_class, base_class, causal_class):
        config = AutoConfig.from_pretrained(checkpoint)
        self.assertIs(type(config), config_class)
        for auto_class, expected in [(AutoModel, base_class), (AutoModelForCausalLM, causal_class)]:
            with self.subTest(checkpoint=checkpoint, auto_class=auto_class.__name__):
                model = auto_class.from_pretrained(
                    checkpoint,
                    dtype="float32",
                    convert_from_hf=True,
                    load_checkpoint_format="flex_checkpoint",
                    load_via_cpu=True,
                    _attn_implementation="eager",
                )
                try:
                    self.assertIs(type(model), expected)
                    MiniCPMAutoLoadingTest()._assert_forward(model, auto_class is AutoModelForCausalLM)
                finally:
                    del model
                    gc.collect()
                    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                        paddle.device.cuda.empty_cache()

    @slow
    def test_legacy_real_checkpoint(self):
        self._assert_checkpoint(
            os.environ.get("MINICPM_LEGACY_CHECKPOINT", "openbmb/MiniCPM4-0.5B"), *FAMILIES["legacy"]
        )

    @slow
    def test_minicpm4_1_real_checkpoint(self):
        self._assert_checkpoint(
            os.environ.get("MINICPM4_1_CHECKPOINT", "openbmb/MiniCPM4.1-8B"), *FAMILIES["minicpm4_1"]
        )
