# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Focused tests for GLM MoE DSA config needles used by the formal YAML."""

import inspect
import unittest

from paddleformers.transformers.glm_moe_dsa.configuration import GlmMoeDsaConfig


class TestGlmMoeDsaRopeParameters(unittest.TestCase):
    def test_init_pops_nested_partial_rotary_factor(self):
        source = inspect.getsource(GlmMoeDsaConfig.__init__)
        self.assertIn('self.rope_parameters.pop("partial_rotary_factor", None)', source)
        self.assertIn('self.rope_scaling.pop("partial_rotary_factor", None)', source)

        cfg = GlmMoeDsaConfig()
        self.assertEqual(cfg.partial_rotary_factor, 0.5)
        self.assertNotIn("partial_rotary_factor", cfg.rope_parameters or {})
        self.assertNotIn("partial_rotary_factor", cfg.rope_scaling or {})

    def test_init_registers_derived_rope_fields_unsavable(self):
        source = inspect.getsource(GlmMoeDsaConfig.__init__)
        self.assertIn("self.register_unsavable_keys(", source)
        self.assertIn('"rope_parameters", "rotary_base", "rope_type"', source)
