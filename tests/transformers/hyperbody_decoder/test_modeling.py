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

"""Unit tests for the HyperBody decoder PaddleFormers wrapper.

The forward path builds a fleet ``GPTModel`` (a ``PipelineLayer``) which needs
a distributed launcher and packed pipeline inputs, so end-to-end parity is
validated separately by the bitwise-alignment scripts. These tests cover the
wrapper contract that runs single-process: config round-trip, Auto* mapping,
the provider's pinned fields, the dense/MoE layer split, and ``_gen_aoa_config``.
"""

import json
import os
import tempfile
import unittest

from paddleformers.transformers import AutoConfig
from paddleformers.transformers.hyperbody_decoder.configuration import (
    HyperBodyDecoderConfig,
)
from paddleformers.transformers.hyperbody_decoder.modeling import (
    HyperBodyDecoderForCausalLM,
    HyperBodyDecoderModelProvider,
)


def tiny_hyperbody_decoder_config(**kwargs):
    config_kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=128,
        n_routed_experts=4,
        moe_intermediate_size=64,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_layer_freq=[0, 1, 1, 1],
        n_group=1,
        topk_group=1,
        max_position_embeddings=64,
    )
    config_kwargs.update(kwargs)
    return HyperBodyDecoderConfig(**config_kwargs)


class HyperBodyDecoderConfigTest(unittest.TestCase):
    def test_config_round_trip(self):
        config = tiny_hyperbody_decoder_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = HyperBodyDecoderConfig.from_pretrained(tmpdir)

        for field in [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "n_routed_experts",
            "moe_layer_freq",
            "num_experts_per_tok",
        ]:
            self.assertEqual(getattr(config, field), getattr(loaded, field))
        self.assertEqual(loaded.model_type, "hyperbody_decoder")

    def test_auto_config_mapping(self):
        config = tiny_hyperbody_decoder_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            # Drop architectures so AutoConfig has to route via model_type.
            config_dict.pop("architectures", None)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)

            auto_config = AutoConfig.from_pretrained(tmpdir)

        self.assertEqual(type(auto_config).__name__, "HyperBodyDecoderConfig")
        self.assertEqual(auto_config.model_type, "hyperbody_decoder")

    def test_first_k_dense_and_moe_layer_freq_mutually_exclusive(self):
        # Both knobs describe the same dense/MoE pattern; supplying both is
        # ambiguous and must be rejected when the provider is materialized.
        config = tiny_hyperbody_decoder_config(first_k_dense_replace=1)
        with self.assertRaises(ValueError):
            HyperBodyDecoderModelProvider.from_config(config)


class HyperBodyDecoderProviderTest(unittest.TestCase):
    def test_provider_pins(self):
        provider = HyperBodyDecoderModelProvider.from_config(tiny_hyperbody_decoder_config())
        self.assertFalse(provider.multi_latent_attention)
        self.assertFalse(provider.use_qk_norm)
        self.assertEqual(provider.normalization, "RMSNorm")
        self.assertTrue(provider.gated_linear_unit)
        self.assertEqual(provider.moe_token_dispatcher_type, "alltoall")
        self.assertFalse(provider.tie_word_embeddings)

    def test_provider_builds_layer_specs(self):
        from paddlefleet.models.hyperbody_decoder.layer_specs import (
            get_hyperbody_decoder_layer_specs,
        )

        provider = HyperBodyDecoderModelProvider.from_config(tiny_hyperbody_decoder_config())
        specs = get_hyperbody_decoder_layer_specs(provider)
        mlp_classes = [s.sublayers_spec.mlp.layer.__name__ for s in specs]
        self.assertEqual(mlp_classes[0], "MLP")
        self.assertTrue(all(name == "MoELayer" for name in mlp_classes[1:]))


class HyperBodyDecoderAoAConfigTest(unittest.TestCase):
    def test_gen_aoa_config_produces_statements(self):
        aoa = HyperBodyDecoderForCausalLM._gen_aoa_config(tiny_hyperbody_decoder_config())
        self.assertIn("aoa_statements", aoa)
        self.assertGreater(len(aoa["aoa_statements"]), 0)


if __name__ == "__main__":
    unittest.main()
