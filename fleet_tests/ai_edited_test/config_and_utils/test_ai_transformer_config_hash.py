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
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


class TestTransformerConfigHashRoutingValidation(unittest.TestCase):
    """Cover hash routing validation in TransformerConfig.__post_init__
    (lines 990-1027)."""

    def _make_config_kwargs(self, **overrides):
        """Minimal kwargs to construct a valid TransformerConfig."""
        defaults = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "num_hidden_layers": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_n_hash_layers": 1,
            "actual_vocab_size": 128,
            "scoring_func": "softmax",
        }
        defaults.update(overrides)
        return defaults

    def test_hash_missing_actual_vocab_size_raises(self):
        """Lines 990-991: actual_vocab_size is None with moe_n_hash_layers > 0."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        kwargs = self._make_config_kwargs()
        del kwargs["actual_vocab_size"]
        with self.assertRaises(ValueError):
            TransformerConfig(**kwargs)

    def test_hash_negative_actual_vocab_size_raises(self):
        """Lines 995-996: actual_vocab_size <= 0."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(**self._make_config_kwargs(actual_vocab_size=-1))

    def test_hash_too_many_hash_layers_raises(self):
        """Lines 1000-1001: moe_n_hash_layers > num_hidden_layers."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(
                **self._make_config_kwargs(
                    moe_n_hash_layers=100, num_hidden_layers=8
                )
            )

    def test_hash_invalid_scoring_func_raises(self):
        """Lines 1005-1006: scoring_func not in allowed set."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(**self._make_config_kwargs(scoring_func="relu"))

    def test_hash_no_topk_raises(self):
        """Lines 1011-1015: num_experts_per_tok is None or <= 0."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(**self._make_config_kwargs(num_experts_per_tok=0))

    def test_hash_too_few_routed_experts_raises(self):
        """Lines 1019-1023: n_routed_experts < num_experts_per_tok."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError):
            TransformerConfig(
                **self._make_config_kwargs(
                    n_routed_experts=1, num_experts_per_tok=2
                )
            )

    def test_hash_valid_config_passes(self):
        """Valid hash routing config should not raise."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(**self._make_config_kwargs())
        self.assertEqual(config.moe_n_hash_layers, 1)
        self.assertEqual(config.actual_vocab_size, 128)

    def test_no_hash_layers_skips_validation(self):
        """moe_n_hash_layers=0 should skip hash validation entirely."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        # This would fail hash validation if it were checked, but
        # moe_n_hash_layers=0 means no hash routing, so no validation.
        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=256,
            num_hidden_layers=8,
            moe_n_hash_layers=0,
        )
        self.assertEqual(config.moe_n_hash_layers, 0)


class TestTransformerConfigHashPostInit(unittest.TestCase):
    """Cover __post_init__ hash validation paths directly.

    Python's coverage tool sometimes loses line-tracking for code executed
    inside the dataclass-generated ``__init__``.  Calling ``__post_init__``
    explicitly ensures the validation lines are attributed correctly.
    """

    def _make_minimal_obj(self, **overrides):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        defaults = {
            "moe_n_hash_layers": 1,
            "actual_vocab_size": 128,
            "num_hidden_layers": 8,
            "scoring_func": "softmax",
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
        }
        defaults.update(overrides)
        obj = TransformerConfig.__new__(TransformerConfig)
        for k, v in defaults.items():
            setattr(obj, k, v)
        return obj

    def test_post_init_missing_vocab_size(self):
        """__post_init__ raises when actual_vocab_size is None."""
        obj = self._make_minimal_obj(actual_vocab_size=None)
        with self.assertRaises(ValueError):
            obj.__post_init__()

    def test_post_init_negative_vocab_size(self):
        """__post_init__ raises when actual_vocab_size <= 0."""
        obj = self._make_minimal_obj(actual_vocab_size=-1)
        with self.assertRaises(ValueError):
            obj.__post_init__()

    def test_post_init_too_many_hash_layers(self):
        """__post_init__ raises when moe_n_hash_layers > num_hidden_layers."""
        obj = self._make_minimal_obj(moe_n_hash_layers=100, num_hidden_layers=8)
        with self.assertRaises(ValueError):
            obj.__post_init__()

    def test_post_init_invalid_scoring_func(self):
        """__post_init__ raises for unsupported scoring_func."""
        obj = self._make_minimal_obj(scoring_func="relu")
        with self.assertRaises(ValueError):
            obj.__post_init__()

    def test_post_init_no_topk(self):
        """__post_init__ raises when num_experts_per_tok <= 0."""
        obj = self._make_minimal_obj(num_experts_per_tok=0)
        with self.assertRaises(ValueError):
            obj.__post_init__()

    def test_post_init_too_few_routed_experts(self):
        """__post_init__ raises when n_routed_experts < num_experts_per_tok."""
        obj = self._make_minimal_obj(n_routed_experts=1, num_experts_per_tok=2)
        with self.assertRaises(ValueError):
            obj.__post_init__()

    def test_post_init_valid_config(self):
        """__post_init__ passes for a valid hash routing config."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        obj = self._make_minimal_obj()
        # Should not raise — but __post_init__ for TransformerConfig
        # may need additional fields. Use the constructor instead.
        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=256,
            num_hidden_layers=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=128,
            scoring_func="softmax",
        )
        self.assertEqual(config.moe_n_hash_layers, 1)


if __name__ == "__main__":
    unittest.main()
