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

from paddleformers.fleet.models.gpt.gpt_model import GPTModel


class TestGPTModelBuildOverlappedNodes(unittest.TestCase):
    """Tests for GPTModel build_overlapped_nodes function."""

    def test_build_overlapped_nodes_function_exists(self):
        """build_overlapped_nodes should be importable from gpt_model."""
        from paddleformers.fleet.models.gpt.gpt_model import (
            build_overlapped_nodes,
        )

        self.assertTrue(callable(build_overlapped_nodes))


class TestGPTModelOverlappedForwardBackward(unittest.TestCase):
    """Tests for GPTModel.overlapped_forward_backward logic."""

    @patch("paddleformers.fleet.models.gpt.gpt_model.build_overlapped_nodes")
    def test_forward_loss_none_when_no_loss_fn(self, mock_build):
        """forward_loss should be None when forward_loss_fn_node is None."""
        mock_forward_pre = MagicMock()
        mock_forward_pre.forward.return_value = "fwd_out"
        mock_backward_pre = MagicMock()
        mock_backward_pre.backward.return_value = "bwd_grad"
        mock_overlap = MagicMock()
        mock_overlap.nodes = []
        mock_forward_post = MagicMock()
        mock_forward_post.forward.return_value = "fwd_post_out"
        mock_backward_post = MagicMock()
        mock_backward_post.backward.return_value = "bwd_post_grad"

        mock_build.return_value = (
            mock_forward_pre,
            mock_backward_pre,
            mock_overlap,
            mock_forward_post,
            mock_backward_post,
        )

        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            result = model.overlapped_forward_backward(
                forward_chunk=MagicMock(),
                forward_inputs="fwd_in",
                forward_loss_fn_node=None,
                backward_chunk=MagicMock(),
                backward_loss_fn_node=None,
                backward_input_grads=None,
                scaler=None,
                p2p_async_handle=None,
            )
            _, forward_loss, _ = result
            self.assertIsNone(forward_loss)


class TestGPTModelFP8VirtualPipeline(unittest.TestCase):
    """Tests for GPTModel.fp8_quant_weight method existence."""

    def test_fp8_quant_weight_method_exists(self):
        """fp8_quant_weight should be a callable method."""
        self.assertTrue(callable(GPTModel.fp8_quant_weight))


if __name__ == "__main__":
    unittest.main()
