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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock, patch


class TestGPTSublayersSpec(unittest.TestCase):
    """Tests for GPTSublayersSpec dataclass."""

    def test_default_values(self):
        from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec

        spec = GPTSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.mtp)
        self.assertIsNone(spec.layer_norm)
        self.assertIsNone(spec.lm_head)

    def test_custom_values(self):
        from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec

        mock_embedding = MagicMock()
        mock_layer_norm = MagicMock()
        mock_lm_head = MagicMock()
        spec = GPTSublayersSpec(
            embedding=mock_embedding,
            layer_norm=mock_layer_norm,
            lm_head=mock_lm_head,
        )
        self.assertEqual(spec.embedding, mock_embedding)
        self.assertEqual(spec.layer_norm, mock_layer_norm)
        self.assertEqual(spec.lm_head, mock_lm_head)

    def test_with_transformer_layers(self):
        from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec

        layers = [MagicMock() for _ in range(3)]
        spec = GPTSublayersSpec(transformer_layers=layers)
        self.assertEqual(len(spec.transformer_layers), 3)

    def test_with_head_and_tail_empty_layers(self):
        from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec

        head = [MagicMock()]
        tail = [MagicMock(), MagicMock()]
        spec = GPTSublayersSpec(head_empty_layers=head, tail_empty_layers=tail)
        self.assertEqual(len(spec.head_empty_layers), 1)
        self.assertEqual(len(spec.tail_empty_layers), 2)

    def test_with_mtp_layers(self):
        from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec

        mtp_layers = [MagicMock(), MagicMock()]
        spec = GPTSublayersSpec(mtp=mtp_layers)
        self.assertEqual(len(spec.mtp), 2)


class TestBuildOverlappedNodes(unittest.TestCase):
    """Tests for build_overlapped_nodes function."""

    def test_build_overlapped_nodes_empty_chunks(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        from paddleformers.fleet.models.gpt.gpt_model import build_overlapped_nodes

        forward_chunk = ScheduleChunk([])
        backward_chunk = ScheduleChunk([])

        with patch("paddleformers.fleet.models.gpt.gpt_model.TransformerLayerNode"):  # noqa: SIM117
            with patch("paddleformers.fleet.models.gpt.gpt_model.TransformerLayerOverlappedScheduleNode"):
                (
                    fwd_pre,
                    bwd_pre,
                    overlap,
                    fwd_post,
                    bwd_post,
                ) = build_overlapped_nodes(forward_chunk, backward_chunk)

                self.assertEqual(len(overlap.nodes), 0)
                self.assertEqual(len(fwd_pre.nodes), 0)
                self.assertEqual(len(bwd_pre.nodes), 0)


class TestGPTModelHelperMethods(unittest.TestCase):
    """Tests for GPTModel helper methods with mocking."""

    def test_add_sequential_layer(self):
        from paddleformers.fleet.models.gpt.gpt_model import GPTSublayersSpec

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        spec = GPTSublayersSpec(
            embedding=MagicMock(),
            transformer_layers=[MagicMock() for _ in range(2)],
            layer_norm=MagicMock(),
            lm_head=MagicMock(),
        )

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config
            model._sequential_layers = []

            layers = []
            model.add_sequential_layer(layers, MagicMock(), "model.layers.0")
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0]["name_prefix"], "model.layers.0")

    def test_get_sequential_layers(self):
        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config
            model._sequential_layers = [
                {"layer": MagicMock(), "name_prefix": "model"},
                {"layer": MagicMock(), "name_prefix": "model.layers.0"},
            ]

            layers = model.get_sequential_layers()
            self.assertEqual(len(layers), 2)

    def test_get_sequential_name_prefixes(self):
        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config
            model._sequential_layers = [
                {"layer": MagicMock(), "name_prefix": "model"},
                {"layer": MagicMock(), "name_prefix": "model.layers.0"},
            ]

            prefixes = model.get_sequential_name_prefixes()
            self.assertEqual(prefixes["0"], "model")
            self.assertEqual(prefixes["1"], "model.layers.0")

    def test_get_hardware_flops(self):
        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config

            flops = model.get_hardware_flops()
            self.assertEqual(flops, 989e3)

    def test_fp8_quant_weight_no_virtual_stages(self):
        """Test fp8_quant_weight when no virtual pipeline stages."""

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config
            model._num_virtual_pipeline_stages = 1
            model.run_function = []
            # Should not raise
            model.fp8_quant_weight()

    def test_use_fp8_no_virtual_stages(self):
        """Test use_fp8 returns False when no virtual pipeline stages."""

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config
            model._num_virtual_pipeline_stages = 1
            model.run_function = []

            result = model.use_fp8()
            self.assertFalse(result)

    def test_offload_weight_only_params_empty(self):
        """Test offload_weight_only_params with no weight-only params."""

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config
            model.state_dict = MagicMock(return_value={})
            # Should not raise
            model.offload_weight_only_params()

    def test_reload_weight_only_params_empty(self):
        """Test reload_weight_only_params with no weight-only params."""

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1
        mock_config.tie_word_embeddings = False
        mock_config.model_type = ""

        with patch(
            "paddleformers.fleet.models.gpt.gpt_model.GPTModel.__init__",
            lambda self, *a, **kw: None,
        ):
            from paddleformers.fleet.models.gpt.gpt_model import GPTModel

            model = GPTModel.__new__(GPTModel)
            model.config = mock_config
            model.state_dict = MagicMock(return_value={})
            # Should not raise
            model.reload_weight_only_params()


if __name__ == "__main__":
    unittest.main()
