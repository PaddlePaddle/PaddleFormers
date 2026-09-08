# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Focused tests for GLM MoE DSA config needles used by the formal YAML."""

import inspect
import unittest

from paddleformers.transformers.glm_moe_dsa.configuration import GlmMoeDsaConfig


class TestGlmMoeDsaRopeParameters(unittest.TestCase):
    def test_init_pops_nested_partial_rotary_factor(self):
        source = inspect.getsource(GlmMoeDsaConfig.__init__)
        self.assertIn('self.rope_parameters.pop("partial_rotary_factor", None)', source)
        self.assertIn("isinstance(self.rope_scaling, dict)", source)
        self.assertIn('self.rope_scaling.pop("partial_rotary_factor", None)', source)

        cfg = GlmMoeDsaConfig()
        self.assertEqual(cfg.partial_rotary_factor, 0.5)
        self.assertNotIn("partial_rotary_factor", cfg.rope_parameters or {})
        self.assertNotIn("partial_rotary_factor", cfg.rope_scaling or {})

    def test_init_registers_derived_rope_fields_unsavable(self):
        source = inspect.getsource(GlmMoeDsaConfig.__init__)
        self.assertIn("self.register_unsavable_keys(", source)
        self.assertIn('"rotary_base", "rope_type"', source)
        self.assertNotIn('register_unsavable_keys(["rope_parameters"', source)

    def test_save_pretrained_keeps_official_rope_parameters(self):
        import json
        import tempfile

        cfg = GlmMoeDsaConfig(rope_theta=8000000)
        with tempfile.TemporaryDirectory() as tmp:
            cfg.save_pretrained(tmp)
            with open(f"{tmp}/config.json") as handle:
                saved = json.load(handle)
            loaded = GlmMoeDsaConfig.from_pretrained(tmp)
        self.assertIn("rope_parameters", saved)
        self.assertEqual(saved["rope_parameters"]["rope_theta"], 8000000)
        self.assertEqual(loaded.rope_parameters["rope_theta"], 8000000)
        self.assertIsNone(loaded.rope_scaling)

    def test_json_roundtrip_keeps_rope_scaling_none(self):
        import tempfile

        cfg = GlmMoeDsaConfig(vocab_size=256, hidden_size=24)
        self.assertIsNone(cfg.rope_scaling)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/config.json"
            cfg.to_json_file(path)
            loaded = GlmMoeDsaConfig.from_json_file(path)
        self.assertIsNone(loaded.rope_scaling)
        self.assertEqual(loaded.to_dict(), cfg.to_dict())
