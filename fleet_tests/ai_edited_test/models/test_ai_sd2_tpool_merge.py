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
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.models.kimi_k25.sd2_tpool_merge import (
    KimiK25VisionPatchMergerSpec,
    KimiK25VisionPathMerger,
    KimiK25VisionSd2TpoolMerger,
)


class TestKimiK25VisionSd2TpoolMerger(unittest.TestCase):
    """Test KimiK25VisionSd2TpoolMerger forward method."""

    def test_forward_basic(self):
        """Test forward with basic input."""
        config = MagicMock()
        config.merge_kernel_size = (2, 2)

        merger = KimiK25VisionSd2TpoolMerger(config)
        # t=1, h=4, w=4, kernel=(2,2) => new_h=2, new_w=2
        hidden_states = paddle.randn([1, 16, 64])  # [batch=1, seq=16, dim=64]
        grid_thws = paddle.to_tensor([[1, 4, 4]])

        result = merger.forward(
            {
                "hidden_states": hidden_states,
                "grid_thws": grid_thws,
            }
        )
        self.assertIn("hidden_states", result)
        # Output should be a list of [new_h*new_w, kernel_h*kernel_w, dim]
        outputs = result["hidden_states"]
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape[0], 4)  # 2*2
        self.assertEqual(outputs[0].shape[1], 4)  # 2*2
        self.assertEqual(outputs[0].shape[2], 64)

    def test_forward_multiple_grids(self):
        """Test forward with multiple grids."""
        config = MagicMock()
        config.merge_kernel_size = (2, 2)

        merger = KimiK25VisionSd2TpoolMerger(config)
        # Two grids: (1, 4, 4) and (1, 4, 4)
        hidden_states = paddle.randn([1, 32, 64])
        grid_thws = paddle.to_tensor([[1, 4, 4], [1, 4, 4]])

        result = merger.forward(
            {
                "hidden_states": hidden_states,
                "grid_thws": grid_thws,
            }
        )
        outputs = result["hidden_states"]
        self.assertEqual(len(outputs), 2)

    def test_forward_preserves_dict_args(self):
        """Test that forward preserves other dict_args."""
        config = MagicMock()
        config.merge_kernel_size = (2, 2)

        merger = KimiK25VisionSd2TpoolMerger(config)
        hidden_states = paddle.randn([1, 16, 64])
        grid_thws = paddle.to_tensor([[1, 4, 4]])

        result = merger.forward(
            {
                "hidden_states": hidden_states,
                "grid_thws": grid_thws,
                "extra_key": "value",
            }
        )
        self.assertIn("extra_key", result)
        self.assertEqual(result["extra_key"], "value")


class TestKimiK25VisionPatchMergerSpec(unittest.TestCase):
    """Test KimiK25VisionPatchMergerSpec dataclass."""

    def test_default_norm(self):
        """Test default norm field."""
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        spec = KimiK25VisionPatchMergerSpec()
        self.assertEqual(spec.norm, IdentityOp)

    def test_custom_norm(self):
        """Test custom norm field."""
        mock_norm = MagicMock()
        spec = KimiK25VisionPatchMergerSpec(norm=mock_norm)
        self.assertEqual(spec.norm, mock_norm)


class TestKimiK25VisionPathMerger(unittest.TestCase):
    """Test KimiK25VisionPathMerger class."""

    def test_forward_with_list_hidden_states(self):
        """Test forward when hidden_states is a list."""
        merger = KimiK25VisionPathMerger.__new__(KimiK25VisionPathMerger)
        merger.__dict__.setdefault("_parameters", {})
        merger.__dict__.setdefault("_buffers", {})
        merger.__dict__.setdefault("_sub_layers", {})
        merger.__dict__.setdefault("_loaddict_holder", {})
        merger.__dict__.setdefault("_non_persistable_buffers", set())
        merger.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        merger.hidden_size = 64

        # Mock pre_norm and proj
        merger.pre_norm = MagicMock(return_value=paddle.randn([2, 4, 64]))
        merger.proj = MagicMock(
            return_value=(paddle.randn([2, 32]), paddle.randn([32]))
        )

        dict_args = {
            "hidden_states": [
                paddle.randn([2, 4, 64]),
                paddle.randn([2, 4, 64]),
            ],
        }
        result = merger.forward(dict_args)
        self.assertIn("hidden_states", result)
        self.assertIsInstance(result["hidden_states"], list)

    def test_forward_with_tensor_hidden_states(self):
        """Test forward when hidden_states is a tensor."""
        merger = KimiK25VisionPathMerger.__new__(KimiK25VisionPathMerger)
        merger.__dict__.setdefault("_parameters", {})
        merger.__dict__.setdefault("_buffers", {})
        merger.__dict__.setdefault("_sub_layers", {})
        merger.__dict__.setdefault("_loaddict_holder", {})
        merger.__dict__.setdefault("_non_persistable_buffers", set())
        merger.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        merger.hidden_size = 64

        merger.pre_norm = MagicMock(return_value=paddle.randn([2, 4, 64]))
        merger.proj = MagicMock(return_value=(paddle.randn([8, 32]), None))

        dict_args = {
            "hidden_states": paddle.randn([2, 4, 64]),
        }
        result = merger.forward(dict_args)
        self.assertIn("hidden_states", result)

    def test_forward_preserves_dict_args(self):
        """Test that forward preserves other dict_args."""
        merger = KimiK25VisionPathMerger.__new__(KimiK25VisionPathMerger)
        merger.__dict__.setdefault("_parameters", {})
        merger.__dict__.setdefault("_buffers", {})
        merger.__dict__.setdefault("_sub_layers", {})
        merger.__dict__.setdefault("_loaddict_holder", {})
        merger.__dict__.setdefault("_non_persistable_buffers", set())
        merger.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        merger.hidden_size = 64

        merger.pre_norm = MagicMock(return_value=paddle.randn([2, 4, 64]))
        merger.proj = MagicMock(return_value=(paddle.randn([8, 32]), None))

        dict_args = {
            "hidden_states": paddle.randn([2, 4, 64]),
            "grid_thws": paddle.to_tensor([[1, 4, 4]]),
        }
        result = merger.forward(dict_args)
        self.assertIn("grid_thws", result)


if __name__ == "__main__":
    unittest.main()
