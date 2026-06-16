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

from paddleformers.fleet.models.gpt.lm_head import (
    GPTLMHead,
    GPTMainLMHead,
    GPTMTPLMHead,
)


def _make_head(cls, **attrs):
    """Create a head instance bypassing __init__ but safely setting attributes."""
    head = cls.__new__(cls)
    # Initialize Paddle Layer internals so __setattr__ works
    head.__dict__.setdefault("_parameters", {})
    head.__dict__.setdefault("_buffers", {})
    head.__dict__.setdefault("_sub_layers", {})
    head.__dict__.setdefault("_loaddict_holder", {})
    head.__dict__.setdefault("_non_persistable_buffers", set())
    for k, v in attrs.items():
        object.__setattr__(head, k, v)
    return head


class TestGPTLMHeadForward(unittest.TestCase):
    """Test GPTLMHead forward method."""

    def test_forward_basic(self):
        """Test basic forward pass with dict_args."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                block_attention_residuals=False,
                num_nextn_predict_layers=None,
                mtp_load_weight_only=False,
            ),
        )
        head._forward = MagicMock(return_value=paddle.randn([2, 10, 100]))

        dict_args = {"hidden_states": paddle.randn([2, 10, 64])}
        result = head.forward(dict_args)
        self.assertIsNotNone(result)

    def test_forward_with_block_attn_res(self):
        """Test forward with block_attention_residuals enabled."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                block_attention_residuals=True,
                num_nextn_predict_layers=None,
                mtp_load_weight_only=False,
            ),
        )
        head._forward = MagicMock(return_value=paddle.randn([2, 10, 100]))
        head.block_attn_res = MagicMock(return_value=paddle.randn([2, 10, 64]))

        dict_args = {
            "hidden_states": paddle.randn([2, 10, 64]),
            "blocks": ["block1"],
        }
        result = head.forward(dict_args)
        head.block_attn_res.assert_called_once()
        self.assertIsNotNone(result)

    def test_forward_with_mtp(self):
        """Test forward with MTP layers."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                block_attention_residuals=False,
                num_nextn_predict_layers=2,
                mtp_load_weight_only=False,
            ),
        )
        head._forward = MagicMock(
            side_effect=lambda x: paddle.randn([*list(x.shape[:-1]), 100])
        )

        # 3 * 10 = 30 rows (main + 2 MTP)
        dict_args = {"hidden_states": paddle.randn([30, 64])}
        result = head.forward(dict_args)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # main + 2 MTP

    def test_forward_without_mtp(self):
        """Test forward without MTP layers calls _forward once."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                block_attention_residuals=False,
                num_nextn_predict_layers=None,
                mtp_load_weight_only=False,
            ),
        )
        head._forward = MagicMock(return_value=paddle.randn([2, 10, 100]))

        dict_args = {"hidden_states": paddle.randn([2, 10, 64])}
        result = head.forward(dict_args)
        head._forward.assert_called_once()


class TestGPTLMHeadEmbeddingWeight(unittest.TestCase):
    """Test GPTLMHead embedding_weight property."""

    def test_embedding_weight_returns_weight(self):
        head = _make_head(GPTLMHead)
        weight = paddle.randn([100, 64])
        object.__setattr__(head, "weight", weight)
        self.assertIs(head.embedding_weight, weight)


class TestGPTLMHeadBuildScheduleNode(unittest.TestCase):
    """Test GPTLMHead build_schedule_node."""

    def test_returns_schedule_node(self):
        head = _make_head(GPTLMHead)
        node = head.build_schedule_node()
        self.assertIsNotNone(node)


class TestGPTLMHeadForwardImpl(unittest.TestCase):
    """Test GPTLMHead._forward method."""

    def test_fused_linear_ce_path(self):
        """Test _forward with fused_linear_ce_loss_chunk > 0."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                fused_linear_ce_loss_chunk=128,
                sequence_parallel=False,
            ),
        )
        object.__setattr__(head, "weight", paddle.randn([100, 64]))
        object.__setattr__(head, "bias", None)

        hidden_states = paddle.randn([2, 10, 64])
        result = head._forward(hidden_states)
        # Should return tuple (hidden_states, weight, bias)
        self.assertIsInstance(result, tuple)

    def test_fused_linear_ce_path_with_sequence_parallel(self):
        """Test _forward with fused_linear_ce_loss_chunk and sequence_parallel."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                fused_linear_ce_loss_chunk=128,
                sequence_parallel=True,
            ),
        )
        object.__setattr__(head, "weight", paddle.randn([100, 64]))
        object.__setattr__(head, "bias", None)

        hidden_states = paddle.randn([10, 2, 64])  # time-major for SP
        result = head._forward(hidden_states)
        self.assertIsInstance(result, tuple)

    def test_forward_impl_basic(self):
        """Test basic _forward without fused CE or recompute."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                fused_linear_ce_loss_chunk=0,
                recompute_modules=None,
                sequence_parallel=False,
            ),
        )
        object.__setattr__(head, "weight", paddle.randn([100, 64]))

        # Mock the parent forward
        with patch.object(
            type(head).__bases__[0],
            "forward",
            return_value=(paddle.randn([2, 10, 100]), None),
        ):
            hidden_states = paddle.randn([2, 10, 64])
            result = head._forward(hidden_states)
            self.assertIsNotNone(result)

    def test_forward_impl_with_recompute(self):
        """Test _forward with recompute_modules containing 'lm_head'."""
        head = _make_head(
            GPTLMHead,
            config=MagicMock(
                fused_linear_ce_loss_chunk=0,
                recompute_modules=["lm_head"],
                sequence_parallel=False,
            ),
        )
        object.__setattr__(head, "weight", paddle.randn([100, 64]))

        with patch.object(
            type(head).__bases__[0],
            "forward",
            return_value=(paddle.randn([2, 10, 100]), None),
        ):
            hidden_states = paddle.randn([2, 10, 64])
            result = head._forward(hidden_states)
            self.assertIsNotNone(result)


class TestGPTLMHeadShardedStateDict(unittest.TestCase):
    """Test GPTLMHead.sharded_state_dict method."""

    def test_sharded_state_dict_single_rank(self):
        """Test sharded_state_dict with world_size=1."""
        head = _make_head(GPTLMHead, world_size=1)
        object.__setattr__(head, "weight", paddle.randn([100, 64]))
        head.bias = paddle.create_parameter(shape=[100], dtype="float32")
        head.state_dict = MagicMock(
            return_value={"weight": MagicMock(), "bias": MagicMock()}
        )

        with patch(
            "paddleformers.fleet.models.gpt.lm_head.build_sharded_state_dict",
            return_value={},
        ) as mock_build:
            result = head.sharded_state_dict(structured_name_prefix="model.")
            mock_build.assert_called_once()
            # shard_rules should be None for single rank
            call_kwargs = mock_build.call_args
            self.assertIsNone(call_kwargs[0][1])


class TestGPTMTPLMHeadForward(unittest.TestCase):
    """Test GPTMTPLMHead forward method."""

    def test_forward_basic(self):
        """Test basic GPTMTPLMHead forward."""
        head = _make_head(
            GPTMTPLMHead,
            config=MagicMock(num_nextn_predict_layers=2),
        )
        head._forward = MagicMock(
            side_effect=lambda x: paddle.randn([*list(x.shape[:-1]), 50])
        )

        # 3*10 = 30 rows total (1 main + 2 MTP)
        hidden = paddle.randn([30, 64])
        dict_args = {
            "hidden_states": hidden,
            "labels": paddle.zeros([2, 12], dtype="int64"),
        }
        result = head.forward(dict_args)
        self.assertIn("mtp_logits", result)
        self.assertEqual(len(result["mtp_logits"]), 2)

    def test_forward_only_mtp_splits(self):
        """Test GPTMTPLMHead only processes MTP splits (not main)."""
        head = _make_head(
            GPTMTPLMHead,
            config=MagicMock(num_nextn_predict_layers=1),
        )
        head._forward = MagicMock(return_value=paddle.randn([2, 10, 50]))

        # 2*10 = 20 rows (1 main + 1 MTP)
        hidden = paddle.randn([20, 64])
        dict_args = {"hidden_states": hidden}
        result = head.forward(dict_args)
        # _forward should be called once (for MTP layer 1)
        self.assertEqual(head._forward.call_count, 1)


class TestGPTMainLMHeadForward(unittest.TestCase):
    """Test GPTMainLMHead forward method."""

    def test_forward_basic(self):
        """Test basic GPTMainLMHead forward."""
        head = _make_head(
            GPTMainLMHead,
            config=MagicMock(
                block_attention_residuals=False,
                num_nextn_predict_layers=1,
            ),
        )
        head._forward = MagicMock(return_value=paddle.randn([2, 10, 100]))
        head.block_attn_res = MagicMock()

        # 2*10 = 20 rows (1 main + 1 MTP)
        hidden = paddle.randn([20, 64])
        mtp_loss = [paddle.to_tensor(1.0)]
        dict_args = {"hidden_states": hidden, "mtp_loss": mtp_loss}
        result = head.forward(dict_args)
        self.assertIn("logits", result)
        self.assertIn("mtp_loss", result)

    def test_forward_removes_none_values(self):
        """Test forward removes None values from return dict."""
        head = _make_head(
            GPTMainLMHead,
            config=MagicMock(
                block_attention_residuals=False,
                num_nextn_predict_layers=1,
            ),
        )
        head._forward = MagicMock(return_value=paddle.randn([2, 10, 100]))
        head.block_attn_res = MagicMock()

        hidden = paddle.randn([20, 64])
        dict_args = {"hidden_states": hidden}
        result = head.forward(dict_args)
        # mtp_loss not in dict_args, so should not appear in result
        self.assertNotIn("mtp_loss", result)


if __name__ == "__main__":
    unittest.main()
