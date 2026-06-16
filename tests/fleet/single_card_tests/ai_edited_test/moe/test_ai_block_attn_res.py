# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def _make_block_config(**overrides):
    """Helper to create config for BlockAttnRes testing."""
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "rms_norm_eps": 1e-5,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_norm_mock():
    """Create a mock that acts like a norm layer (callable returning transformed input)."""
    mock = MagicMock()
    # When called as a layer, it should return the input transformed
    mock.side_effect = lambda x: x * 0.5 + 0.1
    return mock


class TestBlockAttnRes(unittest.TestCase):
    """Unit tests for block_attn_res module."""

    def test_block_attn_res_sublayers_spec(self):
        """Test BlockAttnResSublayersSpec defaults."""
        from paddleformers.fleet.transformer.block_attn_res import (
            BlockAttnResSublayersSpec,
        )
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        spec = BlockAttnResSublayersSpec()
        self.assertIs(spec.norm, IdentityOp)


if __name__ == "__main__":
    unittest.main()
