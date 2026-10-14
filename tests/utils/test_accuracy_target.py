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

"""Tests for ``use_accuracy_compatible`` normalization.

The field is truthiness-tested in about a dozen places across the trainer and
the model definitions, several of which change weight dtypes at load time
(``glm4_moe`` sets ``_keep_in_fp32_modules`` and injects ``dtype='bfloat16'``
AOA statements from it). A non-canonical spelling therefore does not merely look
untidy -- it silently selects a different set of kernels and a different set of
FP32 parameters. These tests pin the two properties that keep that from
happening: the default is falsy, and every spelling a config layer can produce
resolves to a canonical value or raises.
"""

import dataclasses
import types
import unittest

from paddleformers.cli.hparams.finetuning_args import FinetuningArguments
from paddleformers.transformers.configuration_utils import LlmMetaConfig
from paddleformers.utils.accuracy_target import (
    ACCURACY_TARGET_HF,
    ACCURACY_TARGET_MEGATRON,
    normalize_accuracy_target,
    targets_hf,
)


class TestNormalizeAccuracyTarget(unittest.TestCase):
    """Every spelling the YAML/CLI/dataclass layers produce must canonicalize."""

    def test_falsy_inputs_become_real_false(self):
        # ``bool("false") is True``, so returning the input unchanged would turn
        # the default path into the accuracy-compatible path. The type check is
        # the point of this test: a truthy string must not survive.
        for value in [False, None, "", 0, "false", "False", "FALSE", "no", "off", "none", "null", "0"]:
            with self.subTest(value=value):
                result = normalize_accuracy_target(value)
                self.assertIs(result, False)
                self.assertFalse(result)

    def test_true_spellings_become_megatron(self):
        """``True`` predates the "hf" target, so it keeps its historical meaning."""
        for value in [True, 1, "true", "True", "yes", "on", "1"]:
            with self.subTest(value=value):
                self.assertEqual(normalize_accuracy_target(value), ACCURACY_TARGET_MEGATRON)

    def test_target_names_are_case_and_space_insensitive(self):
        for value, expected in [
            ("megatron", ACCURACY_TARGET_MEGATRON),
            ("MEGATRON", ACCURACY_TARGET_MEGATRON),
            ("  megatron  ", ACCURACY_TARGET_MEGATRON),
            ("hf", ACCURACY_TARGET_HF),
            ("HF", ACCURACY_TARGET_HF),
            ("  hf  ", ACCURACY_TARGET_HF),
        ]:
            with self.subTest(value=value):
                self.assertEqual(normalize_accuracy_target(value), expected)

    def test_unknown_target_raises_instead_of_degrading(self):
        """A typo must fail loudly, not quietly select the default kernels."""
        for value in ["megatron_lm", "megatron-lm", "huggingface", "torch", "mg"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_accuracy_target(value)

    def test_error_message_names_the_valid_targets(self):
        with self.assertRaisesRegex(ValueError, "megatron.*hf|hf.*megatron"):
            normalize_accuracy_target("huggingface")

    def test_non_bool_non_str_raises_type_error(self):
        """Truthy values of an unexpected type are a programming error."""
        for value in [2, 1.5, [1], {"a": 1}, object()]:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                normalize_accuracy_target(value)

    def test_any_falsy_value_is_off_regardless_of_type(self):
        """The falsy short-circuit runs before the type check, by design.

        It is what turns ``None``, ``""`` and ``0`` into "off", and an empty
        container means the same thing. Documented here so the ordering is not
        "fixed" into a raise; ``paddlefleet.accuracy_target`` behaves the same and
        the two must not diverge.
        """
        for value in [[], {}, set(), 0.0]:
            with self.subTest(value=value):
                self.assertIs(normalize_accuracy_target(value), False)

    def test_normalization_is_idempotent(self):
        for value in [False, ACCURACY_TARGET_MEGATRON, ACCURACY_TARGET_HF]:
            with self.subTest(value=value):
                once = normalize_accuracy_target(value)
                self.assertEqual(normalize_accuracy_target(once), once)


class TestTargetsHF(unittest.TestCase):
    """``targets_hf`` is the only discriminator between the two references."""

    def test_true_only_for_hf(self):
        self.assertTrue(targets_hf(ACCURACY_TARGET_HF))
        for value in [False, True, ACCURACY_TARGET_MEGATRON, "", None]:
            with self.subTest(value=value):
                self.assertFalse(targets_hf(value))


class TestFinetuningArgumentsDefault(unittest.TestCase):
    """The declared default must be falsy on its own, before normalization."""

    def test_default_is_falsy(self):
        fields = {f.name: f for f in dataclasses.fields(FinetuningArguments)}
        self.assertIn("use_accuracy_compatible", fields)
        default = fields["use_accuracy_compatible"].default
        # Guards the regression directly: a default of "false" is truthy and
        # would enable the accuracy-compatible kernels for every run.
        self.assertFalse(default, f"default {default!r} is truthy")

    def test_default_normalizes_to_false(self):
        fields = {f.name: f for f in dataclasses.fields(FinetuningArguments)}
        default = fields["use_accuracy_compatible"].default
        self.assertIs(normalize_accuracy_target(default), False)


class TestFinetuningArgumentsCLI(unittest.TestCase):
    """The CLI spellings that existed when the field was a ``bool`` must keep working.

    ``PdArgumentParser`` synthesizes ``nargs="?"`` / ``const=True`` only for
    ``bool`` fields, so widening the field to a string would have made the
    valueless ``--use_accuracy_compatible`` fail to parse. The field declares both
    in its metadata instead; these cases pin that.
    """

    def _parse(self, argv):
        from paddleformers.trainer.argparser import PdArgumentParser

        parser = PdArgumentParser(FinetuningArguments)
        (args,) = parser.parse_args_into_dataclasses(
            ["--output_dir", "/tmp/accuracy_target_cli"] + argv, look_for_args_file=False
        )
        return args.use_accuracy_compatible

    def test_valueless_flag_still_means_megatron(self):
        value = self._parse(["--use_accuracy_compatible"])
        self.assertEqual(normalize_accuracy_target(value), ACCURACY_TARGET_MEGATRON)

    def test_omitted_flag_is_off(self):
        self.assertIs(normalize_accuracy_target(self._parse([])), False)

    def test_explicit_targets_round_trip(self):
        for spelling, expected in (
            ("hf", ACCURACY_TARGET_HF),
            ("megatron", ACCURACY_TARGET_MEGATRON),
            ("true", ACCURACY_TARGET_MEGATRON),
            ("false", False),
        ):
            with self.subTest(spelling=spelling):
                value = self._parse(["--use_accuracy_compatible", spelling])
                self.assertEqual(normalize_accuracy_target(value), expected)


class TestSetLlmConfigNormalizes(unittest.TestCase):
    """``set_llm_config`` is the single funnel from args to config."""

    def test_default_when_args_omit_the_field(self):
        config = types.SimpleNamespace()
        LlmMetaConfig.set_llm_config(config, types.SimpleNamespace())
        self.assertIs(config.use_accuracy_compatible, False)

    def test_truthy_false_string_is_normalized_away(self):
        """The exact regression: a stringified ``false`` must not enable the mode."""
        config = types.SimpleNamespace()
        args = types.SimpleNamespace(use_accuracy_compatible="false")
        LlmMetaConfig.set_llm_config(config, args)
        self.assertIs(config.use_accuracy_compatible, False)
        self.assertFalse(config.use_accuracy_compatible)

    def test_bool_true_becomes_megatron(self):
        config = types.SimpleNamespace()
        args = types.SimpleNamespace(use_accuracy_compatible=True)
        LlmMetaConfig.set_llm_config(config, args)
        self.assertEqual(config.use_accuracy_compatible, ACCURACY_TARGET_MEGATRON)

    def test_hf_target_survives(self):
        config = types.SimpleNamespace()
        args = types.SimpleNamespace(use_accuracy_compatible="hf")
        LlmMetaConfig.set_llm_config(config, args)
        self.assertEqual(config.use_accuracy_compatible, ACCURACY_TARGET_HF)
        self.assertTrue(targets_hf(config.use_accuracy_compatible))

    def test_unknown_value_raises_at_config_time(self):
        config = types.SimpleNamespace()
        args = types.SimpleNamespace(use_accuracy_compatible="megatron_lm")
        with self.assertRaises(ValueError):
            LlmMetaConfig.set_llm_config(config, args)

    def test_stored_value_is_json_serializable_as_a_bool(self):
        """A serialized config.json must hold ``false``, not the string."""
        import json

        config = types.SimpleNamespace()
        LlmMetaConfig.set_llm_config(config, types.SimpleNamespace(use_accuracy_compatible="false"))
        restored = json.loads(json.dumps({"use_accuracy_compatible": config.use_accuracy_compatible}))
        self.assertIs(restored["use_accuracy_compatible"], False)
        # A round trip must not re-introduce a truthy spelling.
        self.assertIs(normalize_accuracy_target(restored["use_accuracy_compatible"]), False)


if __name__ == "__main__":
    unittest.main()
