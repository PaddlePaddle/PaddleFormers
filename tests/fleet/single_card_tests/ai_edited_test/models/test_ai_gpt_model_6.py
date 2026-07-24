# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
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

from paddle.distributed.fleet.meta_parallel import PipelineLayer

from paddleformers.fleet.models.gpt.gpt_model import GPTModel


class TestGPTModelSetStateDict(unittest.TestCase):
    """Tests for GPTModel.set_state_dict."""

    def test_set_state_dict_remaps_keys(self):
        """set_state_dict should remap keys via _pipeline_name_mapping."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._pipeline_name_mapping = {"model.weight": "0.weight"}
            mock_super = MagicMock()
            mock_super.return_value = True
            with patch.object(PipelineLayer, "set_state_dict", mock_super):
                sd = {"model.weight": MagicMock()}
                model.set_state_dict(sd)
                mock_super.assert_called_once()


class TestGPTModelCheckSharedModelState(unittest.TestCase):
    """Tests for GPTModel._check_shared_model_state."""

    def test_check_shared_model_state_calls_super_state_dict(self):
        """_check_shared_model_state should call super().state_dict()."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._pipeline_name_mapping = {}
            model._pp_to_single_mapping = {}
            mock_super = MagicMock()
            mock_super.return_value = {}
            with patch.object(PipelineLayer, "state_dict", mock_super):
                result = model._check_shared_model_state()
                self.assertIsInstance(result, dict)


class TestGPTModelStateDictNonQwen(unittest.TestCase):
    """Tests for GPTModel.state_dict with non-qwen model."""

    def test_state_dict_no_prefix_for_gpt(self):
        """state_dict should not strip prefix when model_type is gpt."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model.config = MagicMock()
            model.config.model_type = "gpt"
            model._pipeline_name_mapping = {}
            model._pp_to_single_mapping = {}

            mock_super_sd = MagicMock()
            mock_val = MagicMock()
            mock_super_sd.return_value = {"0.weight": mock_val}
            with patch.object(PipelineLayer, "state_dict", mock_super_sd):
                result = model.state_dict()
                self.assertIn("0.weight", result)


if __name__ == "__main__":
    unittest.main()
