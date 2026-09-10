# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Focused tests for GLM MoE DSA config needles used by the formal YAML."""

import unittest

from paddleformers.transformers.glm_moe_dsa.configuration import GlmMoeDsaConfig


class TestGlmMoeDsaRopeParameters(unittest.TestCase):
    def test_nested_rotary_fraction_is_normalized_without_serializing_derived_fields(self):
        cfg = GlmMoeDsaConfig(
            rope_parameters={"rope_theta": 8000000, "partial_rotary_factor": 0.5},
        )
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            cfg.save_pretrained(directory)
            saved = json.loads((Path(directory) / "config.json").read_text())
        self.assertEqual(cfg.partial_rotary_factor, 0.5)
        self.assertNotIn("partial_rotary_factor", saved["rope_parameters"])
        self.assertNotIn("rotary_base", saved)
        self.assertNotIn("rope_type", saved)
        self.assertEqual(saved["rope_parameters"]["rope_theta"], 8000000)

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
