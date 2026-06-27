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

"""Targeted unit tests for TransformerConfig.multimax_modules field validation."""

import importlib.util
import unittest
import warnings

_HAS_OMEGACONF = importlib.util.find_spec("omegaconf") is not None


class TestMultimaxConfig(unittest.TestCase):
    """TransformerConfig.multimax_modules accepts None or a list of submodule names.

    Mirrors Megatron's ``recompute_modules`` style. Currently the only
    implemented entry is ``"lm_head"``; ``"attention"`` is reserved and
    triggers a not-implemented warning.

    YAML/JSON ergonomics:
    - unset key, ``null``, empty string, or empty list ``[]`` are all
      coerced to ``None`` (feature disabled).
    - a bare string (``multimax_modules: lm_head``) is auto-promoted to a
      single-element list for back-compat with older configs.
    """

    @classmethod
    def setUpClass(cls):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        cls.TransformerConfig = TransformerConfig

    def _build(self, **overrides):
        defaults = {
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 4,
        }
        defaults.update(overrides)
        return self.TransformerConfig(**defaults)

    def test_default_is_none(self):
        cfg = self._build()
        self.assertIsNone(cfg.multimax_modules)

    def test_lm_head_list_accepted(self):
        cfg = self._build(multimax_modules=["lm_head"])
        self.assertEqual(cfg.multimax_modules, ["lm_head"])

    def test_bare_string_promoted_to_list(self):
        """Back-compat: ``multimax_modules: lm_head`` -> ``["lm_head"]``."""
        cfg = self._build(multimax_modules="lm_head")
        self.assertEqual(cfg.multimax_modules, ["lm_head"])

    def test_attention_accepted_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(multimax_modules=["attention"])
        self.assertEqual(cfg.multimax_modules, ["attention"])
        # 'attention' branch is unimplemented -> a banner warning must mention it.
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("not implemented" in m and "attention" in m for m in msgs),
            f"expected unimplemented-attention warning, got: {msgs}",
        )

    def test_combined_lm_head_and_attention(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(multimax_modules=["lm_head", "attention"])
        self.assertEqual(cfg.multimax_modules, ["lm_head", "attention"])
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("not implemented" in m and "attention" in m for m in msgs),
            f"expected unimplemented-attention warning, got: {msgs}",
        )

    def test_empty_string_coerced_to_none(self):
        """YAML `multimax_modules:` parses to '' -- must be canonicalized to None."""
        cfg = self._build(multimax_modules="")
        self.assertIsNone(cfg.multimax_modules)

    def test_empty_list_coerced_to_none(self):
        """YAML `multimax_modules: []` -- must be canonicalized to None."""
        cfg = self._build(multimax_modules=[])
        self.assertIsNone(cfg.multimax_modules)

    def test_invalid_entry_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(multimax_modules=["not_a_real_mode"])
        self.assertIn(
            "multimax_modules entries must each be one of", str(ctx.exception)
        )
        self.assertIn("not_a_real_mode", str(ctx.exception))

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(multimax_modules=123)
        self.assertIn(
            "multimax_modules must be None or a list[str]", str(ctx.exception)
        )

    def test_grep_friendly_banner_emitted(self):
        """[MULTIMAX-CONFIG] tag must be present so operators can grep train logs."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._build(multimax_modules=["lm_head"])
        msgs = [str(x.message) for x in w]
        self.assertTrue(
            any("[MULTIMAX-CONFIG]" in m for m in msgs),
            f"expected [MULTIMAX-CONFIG] banner, got: {msgs}",
        )

    def test_none_explicit_no_unimplemented_warning(self):
        """multimax_modules=None explicitly: no warnings fire (default disabled);
        the [MULTIMAX-CONFIG] banner is only emitted when the feature is on."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._build(multimax_modules=None)
        msgs = [str(x.message) for x in w]
        self.assertFalse(
            any("not implemented" in m for m in msgs),
            f"unexpected unimplemented warning for None: {msgs}",
        )
        self.assertFalse(
            any("[MULTIMAX-CONFIG]" in m for m in msgs),
            f"unexpected [MULTIMAX-CONFIG] banner for default None: {msgs}",
        )

    def test_invalid_entry_message_lists_choices(self):
        """ValueError message must enumerate the valid options so users can
        self-correct without reading the code."""
        with self.assertRaises(ValueError) as ctx:
            self._build(multimax_modules=["lm-head"])  # hyphen, not underscore
        msg = str(ctx.exception)
        for choice in ("lm_head", "attention"):
            self.assertIn(choice, msg, f"missing choice {choice!r} in: {msg}")
        self.assertIn("lm-head", msg, "offending value not echoed")

    def test_empty_string_does_not_warn_unimplemented(self):
        """Empty string is canonicalized to None; must not emit an
        'attention not implemented' warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self._build(multimax_modules="")
        self.assertIsNone(cfg.multimax_modules)
        msgs = [str(x.message) for x in w]
        self.assertFalse(
            any("not implemented" in m for m in msgs),
            f"unexpected unimplemented warning for empty string: {msgs}",
        )

    @unittest.skipUnless(_HAS_OMEGACONF, "omegaconf not installed in this env")
    def test_yaml_listconfig_accepted(self):
        """Regression: YAML entry path returns OmegaConf ListConfig, not a
        builtin list. The validator must normalize ListConfig to list so
        the recommended ``multimax_modules: [lm_head]`` form works through
        the --configs YAML pipeline."""
        from omegaconf import OmegaConf

        listcfg = OmegaConf.create(["lm_head"])
        cfg = self._build(multimax_modules=listcfg)
        self.assertEqual(cfg.multimax_modules, ["lm_head"])
        # After validation the field must be a plain list (not ListConfig)
        # so downstream `isinstance(..., list)` checks behave as expected.
        self.assertIsInstance(cfg.multimax_modules, list)

    @unittest.skipUnless(_HAS_OMEGACONF, "omegaconf not installed in this env")
    def test_yaml_pipeline_end_to_end(self):
        """End-to-end regression from load_yaml() through
        core_transformer_config_from_args() with multimax_modules: [lm_head]."""
        import os
        import tempfile

        from paddleformers.fleet.training.arguments import (
            core_transformer_config_from_args,
        )
        from paddleformers.fleet.training.yaml_arguments import load_yaml

        yaml_text = (
            "model:\n"
            "  num_hidden_layers: 4\n"
            "  hidden_size: 64\n"
            "  num_attention_heads: 4\n"
            "  multimax_modules: [lm_head]\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_text)
            yaml_path = f.name
        try:
            args = load_yaml(yaml_path)
            cfg = core_transformer_config_from_args(
                args, config_class=self.TransformerConfig
            )
        finally:
            os.unlink(yaml_path)
        self.assertEqual(cfg.multimax_modules, ["lm_head"])
        self.assertIsInstance(cfg.multimax_modules, list)


if __name__ == "__main__":
    unittest.main()
