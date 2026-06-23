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
import unittest.mock
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.models.gpt.gpt_embedding import (
    GPTEmbedding,
    GPTEmbeddingSpec,
)


class TestGPTEmbeddingSpec(unittest.TestCase):
    """Test GPTEmbeddingSpec dataclass."""

    def test_spec_with_both_fields(self):
        """Test spec with both language and rope embedding."""
        mock_lang = MagicMock()
        mock_rope = MagicMock()
        spec = GPTEmbeddingSpec(
            language_embedding=mock_lang,
            rope_embedding=mock_rope,
        )
        self.assertEqual(spec.language_embedding, mock_lang)
        self.assertEqual(spec.rope_embedding, mock_rope)

    def test_spec_with_no_rope(self):
        """Test spec with None rope embedding."""
        mock_lang = MagicMock()
        spec = GPTEmbeddingSpec(
            language_embedding=mock_lang,
            rope_embedding=None,
        )
        self.assertEqual(spec.language_embedding, mock_lang)
        self.assertIsNone(spec.rope_embedding)


class TestGPTEmbeddingGetPlaceholderMask(unittest.TestCase):
    """Test GPTEmbedding.get_placeholder_mask method."""

    def _make_embedding(self):
        """Create a GPTEmbedding-like object with get_placeholder_mask."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        # Initialize Paddle Layer internals
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        return emb

    def test_image_mask_with_matching_tokens(self):
        """Test get_placeholder_mask when image tokens match features."""
        emb = self._make_embedding()
        # Mock the config
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2
        # Mock the embedding sublayer
        emb.embedding = MagicMock()

        input_ids = paddle.to_tensor([[1, 1, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        image_features = paddle.randn([2, 8])  # 2 image tokens, hidden=8

        image_mask, video_mask = emb.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features=image_features,
        )
        self.assertIsNotNone(image_mask)
        self.assertIsNotNone(video_mask)
        # Image mask should be True at positions 0, 1
        self.assertTrue(image_mask.shape == inputs_embeds.shape)

    def test_video_mask_with_matching_tokens(self):
        """Test get_placeholder_mask when video tokens match features."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[2, 2, 2, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        video_features = paddle.randn([3, 8])

        image_mask, video_mask = emb.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            video_features=video_features,
        )
        self.assertIsNotNone(image_mask)
        self.assertIsNotNone(video_mask)

    def test_mismatched_image_tokens_raises(self):
        """Test ValueError when image tokens don't match features."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[1, 1, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        # 2 image tokens but only 1 feature vector
        image_features = paddle.randn([1, 8])

        with self.assertRaises(ValueError):
            emb.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
            )

    def test_mismatched_video_tokens_raises(self):
        """Test ValueError when video tokens don't match features."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[2, 2, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        # 2 video tokens but only 1 feature vector
        video_features = paddle.randn([1, 8])

        with self.assertRaises(ValueError):
            emb.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                video_features=video_features,
            )

    def test_no_image_or_video_features(self):
        """Test get_placeholder_mask with no features (just masks)."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[1, 2, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])

        image_mask, video_mask = emb.get_placeholder_mask(
            input_ids,
            inputs_embeds,
        )
        self.assertIsNotNone(image_mask)
        self.assertIsNotNone(video_mask)


class TestGPTEmbeddingForwardPaths(unittest.TestCase):
    """Test GPTEmbedding forward method paths."""

    def test_forward_with_decoder_input(self):
        """Test forward with decoder_input provided."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.num_nextn_predict_layers = 0
        emb.config.apply_rope_fusion = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = False

        mock_decoder_input = paddle.randn([2, 8, 64])
        result = emb.forward(
            dict_args={"input_ids": None},
            decoder_input=mock_decoder_input,
        )
        self.assertIn("hidden_states", result)
        self.assertTrue(
            paddle.allclose(result["hidden_states"], mock_decoder_input)
        )

    def test_forward_removes_none_values(self):
        """Test that forward removes None values from output dict."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.num_nextn_predict_layers = 0
        emb.config.apply_rope_fusion = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = False

        mock_decoder_input = paddle.randn([2, 8, 64])
        result = emb.forward(
            dict_args={"input_ids": None, "attention_mask": None},
            decoder_input=mock_decoder_input,
        )
        # None values should be removed
        self.assertNotIn("attention_mask", result)
        self.assertIn("hidden_states", result)

    def test_forward_mtp_assertion(self):
        """Test forward raises when mtp params are inconsistent."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.num_nextn_predict_layers = 0
        emb.config.apply_rope_fusion = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = False

        mock_decoder_input = paddle.randn([2, 8, 64])
        # Only one of the two mtp params is set - should raise
        with self.assertRaises(AssertionError):
            emb.forward(
                dict_args={
                    "input_ids": None,
                    "mtp_startend_row_indices_all": paddle.randn([2, 8]),
                },
                decoder_input=mock_decoder_input,
            )


class TestGPTEmbeddingBuildScheduleNode(unittest.TestCase):
    """Test GPTEmbedding.build_schedule_node method."""

    def test_build_schedule_node(self):
        """Test build_schedule_node returns ScheduleNode."""
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        emb = GPTEmbedding.__new__(GPTEmbedding)
        node = emb.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)


class TestGPTEmbeddingSWARotaryPosEmb(unittest.TestCase):
    """Tests for SWA rotary position embedding branches in forward."""

    def _make_embedding(self):
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.gpt_model_use_experimental_version = False
        emb.config.num_nextn_predict_layers = 0
        emb.config.apply_rope_fusion = False
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = False
        emb.config.gpt_model_use_experimental_version = False
        emb.config.pad_token_id = 0
        return emb

    def test_swa_rope_path(self):
        """Test SWA rotary pos emb via rope path (L502-515)."""
        emb = self._make_embedding()
        emb.position_embedding_type = "rope"
        mock_swa = MagicMock()
        mock_swa.get_rotary_seq_len = MagicMock(return_value=4)
        mock_swa.return_value = paddle.randn([1, 4, 1, 16])
        emb.swa_rotary_pos_emb = mock_swa

        decoder_input = paddle.randn([2, 4, 64])
        result = emb.forward(
            dict_args={"input_ids": None},
            decoder_input=decoder_input,
        )
        mock_swa.assert_called_once()
        self.assertIn("swa_rotary_pos_emb", result)
        self.assertIsNotNone(result["swa_rotary_pos_emb"])

    def test_swa_mrope_path(self):
        """Test SWA rotary pos emb via mrope path (L517-523)."""
        emb = self._make_embedding()
        emb.position_embedding_type = "mrope"
        mock_swa = MagicMock()
        mock_swa.return_value = paddle.randn([2, 4, 16])
        emb.swa_rotary_pos_emb = mock_swa
        emb.mrope_section = [4, 4, 8]

        decoder_input = paddle.randn([2, 4, 64])
        result = emb.forward(
            dict_args={
                "input_ids": None,
                "position_ids": paddle.zeros([2, 4], dtype="int64"),
            },
            decoder_input=decoder_input,
        )
        mock_swa.assert_called_once()
        self.assertIsNotNone(result["swa_rotary_pos_emb"])

    def test_swa_apply_rope_fusion(self):
        """Test apply_rope_fusion cos/sin computation (L525-528)."""
        emb = self._make_embedding()
        emb.position_embedding_type = "rope"
        emb.config.apply_rope_fusion = True
        mock_swa = MagicMock()
        mock_swa.get_rotary_seq_len = MagicMock(return_value=4)
        mock_swa.return_value = paddle.randn([1, 4, 1, 16])
        emb.swa_rotary_pos_emb = mock_swa

        decoder_input = paddle.randn([2, 4, 64])
        result = emb.forward(
            dict_args={"input_ids": None},
            decoder_input=decoder_input,
        )
        self.assertIsNotNone(result.get("swa_rotary_pos_cos"))
        self.assertIsNotNone(result.get("swa_rotary_pos_sin"))

    def test_swa_sequence_parallel_rope(self):
        """Test sequence_parallel transpose for rope SWA (L535-539)."""
        emb = self._make_embedding()
        emb.position_embedding_type = "rope"
        emb.config.sequence_parallel = True
        mock_swa = MagicMock()
        mock_swa.get_rotary_seq_len = MagicMock(return_value=4)
        # Shape: [1, S, 1, head_dim]
        mock_swa.return_value = paddle.randn([1, 4, 1, 16])
        emb.swa_rotary_pos_emb = mock_swa

        decoder_input = paddle.randn([2, 4, 64])
        result = emb.forward(
            dict_args={"input_ids": None},
            decoder_input=decoder_input,
        )
        # After transpose: [S, 1, 1, head_dim]
        swa_emb = result["swa_rotary_pos_emb"]
        self.assertEqual(list(swa_emb.shape), [4, 1, 1, 16])

    def test_swa_sequence_parallel_mrope(self):
        """Test sequence_parallel transpose for mrope SWA (L530-534)."""
        emb = self._make_embedding()
        emb.position_embedding_type = "mrope"
        emb.config.sequence_parallel = True
        mock_swa = MagicMock()
        # Shape: [B, S, head_dim]
        mock_swa.return_value = paddle.randn([2, 4, 16])
        emb.swa_rotary_pos_emb = mock_swa
        emb.mrope_section = [4, 4, 8]

        decoder_input = paddle.randn([2, 4, 64])
        result = emb.forward(
            dict_args={
                "input_ids": None,
                "position_ids": paddle.zeros([2, 4], dtype="int64"),
            },
            decoder_input=decoder_input,
        )
        # After transpose: [S, B, head_dim]
        swa_emb = result["swa_rotary_pos_emb"]
        self.assertEqual(list(swa_emb.shape), [4, 2, 16])


class TestGPTEmbeddingCPScatterSPAssert(unittest.TestCase):
    """Test assertion: sequence_parallel not supported with CP scatter in plain path."""

    def _make_embedding(self):
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.tensor_model_parallel_size = 1
        emb.config.num_nextn_predict_layers = 0
        emb.config.mtp_load_weight_only = False
        emb.config.apply_rope_fusion = False
        emb.config.experimental_dataflow = True
        emb.config.cp_balance_mode = "padding"
        emb.config.clone_scatter_output_in_embedding = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = True
        emb.multimodal_embedding = False
        mock_embedding = MagicMock()
        mock_embedding.return_value = paddle.randn([2, 8, 64])
        emb.embedding = mock_embedding
        emb.config.gpt_model_use_experimental_version = False
        emb.config.pad_token_id = 0
        return emb

    @unittest.mock.patch(
        "paddleformers.fleet.models.gpt.gpt_embedding.get_context_parallel_world_size",
        return_value=2,
    )
    def test_sp_with_cp_scatter_plain_path_raises(self, mock_cp_ws):
        """sequence_parallel + CP scatter in plain path should raise AssertionError."""
        emb = self._make_embedding()
        input_ids = paddle.ones([2, 8], dtype="int64")
        with self.assertRaises(AssertionError) as ctx:
            emb.forward(dict_args={"input_ids": input_ids})
        self.assertIn("sequence_parallel is not supported", str(ctx.exception))

    @unittest.mock.patch(
        "paddleformers.fleet.models.gpt.gpt_embedding.ContextParallelScatterOp"
    )
    @unittest.mock.patch(
        "paddleformers.fleet.models.gpt.gpt_embedding.get_context_parallel_world_size",
        return_value=2,
    )
    def test_no_sp_with_cp_scatter_plain_path_passes(
        self, mock_cp_ws, mock_cp_op
    ):
        """Without sequence_parallel, CP scatter in plain path should not raise."""
        emb = self._make_embedding()
        emb.sequence_parallel = False
        mock_cp_op.apply.return_value = paddle.randn([2, 8, 64])
        input_ids = paddle.ones([2, 8], dtype="int64")
        # Should not raise
        result = emb.forward(dict_args={"input_ids": input_ids})
        self.assertIn("hidden_states", result)


if __name__ == "__main__":
    unittest.main()
