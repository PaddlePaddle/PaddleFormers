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

"""Smoke tests for OLMo2 tensor-parallel weight splitting.

Covers the load path exercised when sharding a non-TP checkpoint onto
``tensor_model_parallel_size > 1``. The lm_head is vocab-parallel: its weight
has shape ``[vocab_size / tp, hidden_size]`` with ``split_axis = 0``, so the
split action must slice the vocabulary dimension (``is_column=False``), not the
hidden dimension. This test guards against that regression.
"""

import unittest

import numpy as np

from paddleformers.transformers import Olmo2Config, Olmo2ForCausalLM


def _make_tp_config(tp_size, tensor_parallel_rank):
    config = Olmo2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        dtype="float32",
    )
    config.tensor_model_parallel_size = tp_size
    config.tensor_parallel_rank = tensor_parallel_rank
    config.tensor_parallel_output = True
    config.sequence_parallel = False
    return config


def _get_split_actions(config):
    return Olmo2ForCausalLM._get_tensor_parallel_mappings(config, is_split=True)


class Olmo2LmHeadTPSplitTest(unittest.TestCase):
    """Verify lm_head.weight is split along the vocab dimension for TP."""

    def test_lm_head_weight_split_along_vocab(self):
        tp_size = 2
        vocab_size = 64
        hidden_size = 32

        full_weight = np.arange(vocab_size * hidden_size, dtype=np.float32).reshape(
            vocab_size, hidden_size
        )

        shards = []
        for rank in range(tp_size):
            config = _make_tp_config(tp_size, rank)
            actions = _get_split_actions(config)
            self.assertIn("lm_head.weight", actions)
            action = actions["lm_head.weight"]

            shard = action(full_weight)
            # vocab-parallel shard must keep hidden_size intact and halve vocab.
            self.assertEqual(
                list(shard.shape),
                [vocab_size // tp_size, hidden_size],
                f"rank {rank}: expected vocab-parallel shard "
                f"[{vocab_size // tp_size}, {hidden_size}], "
                f"got {list(shard.shape)}",
            )
            shards.append(np.asarray(shard))

        merged = np.concatenate(shards, axis=0)
        np.testing.assert_allclose(merged, full_weight)

    def test_lm_head_shard_matches_param_shape(self):
        """Shards produced by the split action must match LMHead's param shape."""
        from unittest.mock import MagicMock

        from paddleformers.nn.lm_head import LMHead

        tp_size = 2
        vocab_size = 64
        hidden_size = 32

        full_weight = np.arange(vocab_size * hidden_size, dtype=np.float32).reshape(
            vocab_size, hidden_size
        )

        for rank in range(tp_size):
            tp_config = _make_tp_config(tp_size, rank)
            actions = _get_split_actions(tp_config)
            shard = np.asarray(actions["lm_head.weight"](full_weight))

            head_config = MagicMock()
            head_config.vocab_size = vocab_size
            head_config.hidden_size = hidden_size
            head_config.tensor_model_parallel_size = tp_size
            head_config.lm_head_bias = False
            head_config.use_fused_head_and_loss_fn = False
            head_config.sequence_parallel = False
            head_config.max_sequence_length = 128
            head_config.tensor_parallel_output = False
            head_config.get = lambda key, default=None: default
            head = LMHead(head_config)

            self.assertEqual(
                list(head.weight.shape),
                list(shard.shape),
                f"rank {rank}: LMHead param {list(head.weight.shape)} "
                f"does not match split shard "
                f"{list(shard.shape)}",
            )


if __name__ == "__main__":
    unittest.main()
