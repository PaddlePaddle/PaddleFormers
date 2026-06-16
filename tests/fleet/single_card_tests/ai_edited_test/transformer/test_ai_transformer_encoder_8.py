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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.transformer_encoder import (
    TransformerEncoder,
)


class TestTransformerEncoderSetStateDict(unittest.TestCase):
    """Tests for TransformerEncoder.set_state_dict."""

    def test_set_state_dict_remaps_keys(self):
        """set_state_dict should remap keys using pipeline_name_mapping."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder._pipeline_name_mapping = {
                "model.embed.weight": "0.weight",
                "model.layer.0.weight": "1.weight",
            }
            input_sd = {
                "model.embed.weight": paddle.randn([4, 4]),
                "model.layer.0.weight": paddle.randn([4, 4]),
            }
            with patch.object(
                type(encoder).__mro__[1], "set_state_dict", return_value=True
            ) as mock_super:
                result = encoder.set_state_dict(input_sd)
                self.assertTrue(result)
                # Verify keys were remapped
                call_args = mock_super.call_args[0][0]
                self.assertIn("0.weight", call_args)
                self.assertIn("1.weight", call_args)

    def test_set_state_dict_skips_unknown_keys(self):
        """set_state_dict should skip keys not in pipeline_name_mapping."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder._pipeline_name_mapping = {"model.embed.weight": "0.weight"}
            input_sd = {
                "model.embed.weight": paddle.randn([4, 4]),
                "unknown_key": paddle.randn([4, 4]),
            }
            with patch.object(
                type(encoder).__mro__[1], "set_state_dict", return_value=True
            ):
                encoder.set_state_dict(input_sd)
                # unknown_key should be skipped


class TestTransformerEncoderCheckSharedModelState(unittest.TestCase):
    """Tests for TransformerEncoder._check_shared_model_state."""

    def test_check_shared_model_state_no_missing_keys(self):
        """When pp_to_single_mapping and pipeline_name_mapping are consistent, no missing keys."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder._pipeline_name_mapping = {"a": "0.a", "b": "1.b"}
            encoder._pp_to_single_mapping = {"0.a": "a", "1.b": "b"}
            with patch.object(
                type(encoder).__mro__[1], "state_dict", return_value={}
            ):
                result = encoder._check_shared_model_state()
                # No missing shared keys since mapping is consistent
                self.assertEqual(len(result), 0)


class TestTransformerEncoderFP8QuantWeight(unittest.TestCase):
    """Tests for TransformerEncoder.fp8_quant_weight."""

    def test_fp8_quant_weight_single_stage(self):
        """fp8_quant_weight iterates over run_function in single virtual pipeline stage."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder._num_virtual_pipeline_stages = 1
            from paddleformers.fleet.transformer.transformer_layer import (
                TransformerLayer,
            )

            mock_layer = MagicMock(spec=TransformerLayer)
            other_layer = MagicMock()
            encoder.run_function = [mock_layer, other_layer]
            encoder.fp8_quant_weight(batch_mode=True, quant_transpose=False)
            mock_layer.fp8_quant_weight.assert_called_once_with(
                batch_mode=True, quant_transpose=False
            )
            other_layer.fp8_quant_weight.assert_not_called()


class TestTransformerEncoderUseFP8(unittest.TestCase):
    """Tests for TransformerEncoder.use_fp8."""

    def test_use_fp8_returns_false_when_no_fp8_layer(self):
        """use_fp8 should return False when no TransformerLayer uses fp8."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder._num_virtual_pipeline_stages = 1
            mock_layer = MagicMock()
            mock_layer.use_fp8.return_value = False
            encoder.run_function = [mock_layer]
            result = encoder.use_fp8()
            self.assertFalse(result)

    def test_use_fp8_returns_true_when_fp8_layer_exists(self):
        """use_fp8 should return True when a TransformerLayer uses fp8."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder._num_virtual_pipeline_stages = 1
            from paddleformers.fleet.transformer.transformer_layer import (
                TransformerLayer,
            )

            mock_layer = MagicMock(spec=TransformerLayer)
            mock_layer.use_fp8.return_value = True
            encoder.run_function = [mock_layer]
            result = encoder.use_fp8()
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
