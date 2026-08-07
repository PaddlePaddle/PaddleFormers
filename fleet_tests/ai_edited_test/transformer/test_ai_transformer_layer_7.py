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

import paddle

from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
    tensors_clone,
)


class TestTensorsCloneNested(unittest.TestCase):
    """Tests for tensors_clone with nested structures."""

    def test_clone_nested_dict_in_list(self):
        """Test cloning a list containing dicts with tensors."""
        data = [{"a": paddle.randn([4, 8])}, 42]
        cloned = tensors_clone(data)
        self.assertIsInstance(cloned, list)
        self.assertEqual(len(cloned), 2)
        self.assertIsInstance(cloned[0], dict)

    def test_clone_preserves_tensor_values(self):
        """Test that cloned tensors have the same values."""
        x = paddle.randn([4, 8])
        cloned = tensors_clone(x)
        self.assertTrue(paddle.allclose(x, cloned).item())

    def test_clone_empty_tuple(self):
        """Test cloning an empty tuple."""
        cloned = tensors_clone(())
        self.assertIsInstance(cloned, tuple)
        self.assertEqual(len(cloned), 0)

    def test_clone_empty_dict(self):
        """Test cloning an empty dict."""
        cloned = tensors_clone({})
        self.assertIsInstance(cloned, dict)
        self.assertEqual(len(cloned), 0)


class TestTransformerLayerSublayersSpecCustom(unittest.TestCase):
    """Tests for TransformerLayerSublayersSpec with custom fields."""

    def test_with_sharded_state_dict_keys_map(self):
        """Test TransformerLayerSublayersSpec with custom sharded_state_dict_keys_map."""
        spec = TransformerLayerSublayersSpec(
            sharded_state_dict_keys_map={"old_key": "new_key"}
        )
        self.assertEqual(
            spec.sharded_state_dict_keys_map, {"old_key": "new_key"}
        )


class TestTransformerLayerSkipMtpProbes(unittest.TestCase):
    """Tests for TransformerLayer._skip_mtp_probes flag."""

    def test_set_and_unset_skip_mtp_probes(self):
        """Test setting and unsetting _skip_mtp_probes."""
        original = TransformerLayer._skip_mtp_probes
        TransformerLayer._skip_mtp_probes = True
        self.assertTrue(TransformerLayer._skip_mtp_probes)
        TransformerLayer._skip_mtp_probes = False
        self.assertFalse(TransformerLayer._skip_mtp_probes)
        TransformerLayer._skip_mtp_probes = original


class TestTransformerLayerExperimentalVersion(unittest.TestCase):
    """Tests for TransformerLayer._gpt_model_use_experimental_version."""

    def test_set_experimental_version(self):
        """Test setting _gpt_model_use_experimental_version."""
        original = TransformerLayer._gpt_model_use_experimental_version
        TransformerLayer._gpt_model_use_experimental_version = True
        self.assertTrue(TransformerLayer._gpt_model_use_experimental_version)
        TransformerLayer._gpt_model_use_experimental_version = original


class TestTransformerLayerLogMD5Enabled(unittest.TestCase):
    """Tests for TransformerLayer._log_md5 when enabled."""

    def test_log_md5_enabled(self):
        """Test _log_md5 when enabled (should print but not raise)."""
        original_log = TransformerLayer._LOG_LAYER_MD5
        original_exp = TransformerLayer._gpt_model_use_experimental_version
        original_skip = TransformerLayer._skip_mtp_probes

        TransformerLayer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False

        try:
            # Should not raise
            TransformerLayer._log_md5(paddle.randn([2, 4]), "test_tensor", 1)
        finally:
            TransformerLayer._LOG_LAYER_MD5 = original_log
            TransformerLayer._gpt_model_use_experimental_version = original_exp
            TransformerLayer._skip_mtp_probes = original_skip

    def test_log_md5_disabled(self):
        """Test _log_md5 when disabled (should do nothing)."""
        original_log = TransformerLayer._LOG_LAYER_MD5
        TransformerLayer._LOG_LAYER_MD5 = False

        try:
            # Should not raise and should do nothing
            TransformerLayer._log_md5(paddle.randn([2, 4]), "test_tensor", 1)
        finally:
            TransformerLayer._LOG_LAYER_MD5 = original_log


if __name__ == "__main__":
    unittest.main()
