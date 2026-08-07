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

from paddleformers.fleet.transformer.moe.moe_layer import (
    GradDtypeGuard,
    GradDtypeUnguard,
    MoELayer,
    MoESublayers,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "moe_token_dispatcher_type": "alltoall",
        "moe_shared_expert_overlap": False,
        "fp8": False,
        "fp8_wgrad": False,
        "router_aux_loss_coef": 0.01,
        "moe_expert_fusion": False,
        "moe_deep_gemm": False,
        "moe_ep_barrier": True,
        "moe_use_fusion_node": False,
        "n_shared_experts": 0,
        "recompute_granularity": None,
        "recompute_modules": None,
        "using_sonic_moe": False,
        "gated_linear_unit": True,
        "hidden_act": "silu",
        "use_bias": False,
        "intermediate_size": 256,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestMoESublayers(unittest.TestCase):
    """Test MoESublayers dataclass."""

    def test_default_values(self):
        sublayers = MoESublayers()
        self.assertIsNone(sublayers.mlp_spec)

    def test_custom_mlp_spec(self):
        sublayers = MoESublayers(mlp_spec=MagicMock())
        self.assertIsNotNone(sublayers.mlp_spec)


class TestGradDtypeGuard(unittest.TestCase):
    """Test GradDtypeGuard PyLayer."""

    def test_forward_returns_empty_and_dict(self):
        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False
        empty, status = GradDtypeGuard.apply(x, paddle.float32)
        self.assertEqual(empty.shape[0], 0)
        self.assertIn("x", status)


class TestGradDtypeUnguard(unittest.TestCase):
    """Test GradDtypeUnguard PyLayer."""

    def test_forward_returns_original_tensor(self):
        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False
        empty, status = GradDtypeGuard.apply(x, paddle.float32)
        result = GradDtypeUnguard.apply(empty, status)
        self.assertEqual(result.shape, [2, 3])


class TestMoELayerInitExpertParallel(unittest.TestCase):
    """Test _init_expert_parallel method."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_layer.utils.get_pg_size",
        return_value=4,
    )
    @patch(
        "paddleformers.fleet.transformer.moe.moe_layer.utils.get_pg_rank",
        return_value=0,
    )
    def test_expert_parallel_setup(self, mock_rank, mock_size):
        config = _make_config()
        pg = MagicMock()
        pg.ep = MagicMock()
        pg.expt_dp = MagicMock()

        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.pg_collection = pg
        layer.moe_group = pg.ep
        layer.expert_model_parallel_size = 4
        layer.num_experts = 8
        layer._init_expert_parallel()

        self.assertEqual(layer.moe_rank, 0)
        self.assertEqual(layer.num_experts_per_device, 2)
        self.assertEqual(layer.expert_model_parallel_size, 4)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_layer.utils.get_pg_rank",
        return_value=-1,
    )
    def test_moe_rank_floor_zero(self, mock_rank):
        config = _make_config()
        pg = MagicMock()
        pg.ep = MagicMock()
        pg.expt_dp = MagicMock()

        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.pg_collection = pg
        layer.moe_group = pg.ep
        layer.expert_model_parallel_size = 2
        layer.num_experts = 4
        layer._init_expert_parallel()

        self.assertEqual(layer.moe_rank, 0)

    def test_no_expert_parallel(self):
        config = _make_config()
        pg = MagicMock()

        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.pg_collection = pg
        layer.expert_model_parallel_size = 1
        layer.num_experts = 4
        layer._init_expert_parallel()

        self.assertEqual(layer.moe_rank, 0)
        self.assertEqual(layer.num_experts_per_device, 4)
        self.assertEqual(layer.expert_model_parallel_size, 1)


class TestMoELayerExpertForward(unittest.TestCase):
    """Test expert_forward method."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_layer.utils.get_pg_size",
        return_value=1,
    )
    def test_expert_forward(self, mock_size):
        config = _make_config()
        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.moe_rank = 0
        layer.num_experts_per_device = 4
        layer.experts = []
        for i in range(4):
            expert = MagicMock()
            expert.return_value = (paddle.randn([2, 64]), None)
            layer.experts.append(expert)

        dispatched = paddle.randn([8, 64])
        tokens_per_expert = [2, 2, 2, 2]
        result = layer.expert_forward(
            dispatched, paddle.to_tensor(tokens_per_expert)
        )
        self.assertEqual(result.shape[0], 8)

    def test_expert_forward_no_tokens(self):
        config = _make_config()
        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.num_experts = 4
        layer.moe_rank = 0
        layer.num_experts_per_device = 4
        layer.experts = [MagicMock() for _ in range(4)]

        dispatched = paddle.randn([0, 64])
        tokens_per_expert = [0, 0, 0, 0]
        result = layer.expert_forward(
            dispatched, paddle.to_tensor(tokens_per_expert)
        )
        # When no tokens, should return dispatched_input
        self.assertEqual(result.shape[0], 0)


class TestMoELayerUseFp8(unittest.TestCase):
    """Test use_fp8 method."""

    def test_use_fp8_disabled(self):
        config = _make_config(fp8=False, moe_use_fusion_node=False)
        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.moe_use_fusion_node = False
        layer.fp8 = False
        self.assertFalse(layer.use_fp8())

    def test_use_fp8_enabled(self):
        config = _make_config(fp8=True, moe_use_fusion_node=True)
        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.moe_use_fusion_node = True
        layer.fp8 = True
        self.assertTrue(layer.use_fp8())


class TestMoELayerSetLayerNumber(unittest.TestCase):
    """Test set_layer_number method."""

    def test_set_layer_number(self):
        layer = MoELayer.__new__(MoELayer)
        layer.gate = MagicMock()
        layer.set_layer_number(5)
        self.assertEqual(layer.layer_number, 5)
        layer.gate.set_layer_number.assert_called_once_with(
            5, is_mtp_layer=False
        )

    def test_set_layer_number_no_set_method(self):
        layer = MoELayer.__new__(MoELayer)
        layer.gate = MagicMock()
        del layer.gate.set_layer_number
        with self.assertRaises(AssertionError):
            layer.set_layer_number(5)


class TestMoELayerDispatchPermuteUnpermute(unittest.TestCase):
    """Test dispatch, permute, unpermute, combine delegation."""

    def test_permute_delegates(self):
        layer = MoELayer.__new__(MoELayer)
        layer.token_dispatcher = MagicMock()
        hidden = paddle.randn([4, 64])
        layer.token_dispatcher.dispatch_postprocess.return_value = (
            hidden,
            paddle.to_tensor([2, 2]),
        )
        global_tokens, tokens_per_expert = layer.permute(hidden)
        self.assertEqual(global_tokens.shape, [4, 64])
        layer.token_dispatcher.dispatch_postprocess.assert_called_once()

    def test_unpermute_delegates(self):
        layer = MoELayer.__new__(MoELayer)
        layer.token_dispatcher = MagicMock()
        hidden = paddle.randn([4, 64])
        layer.token_dispatcher.combine_preprocess.return_value = hidden
        result = layer.unpermute(hidden)
        self.assertEqual(result.shape, [4, 64])
        layer.token_dispatcher.combine_preprocess.assert_called_once()

    def test_combine_delegates(self):
        layer = MoELayer.__new__(MoELayer)
        layer.token_dispatcher = MagicMock()
        hidden = paddle.randn([4, 64])
        layer.token_dispatcher.token_combine.return_value = hidden
        layer.token_dispatcher.combine_postprocess.return_value = hidden
        result = layer.combine(hidden)
        self.assertEqual(result.shape, [4, 64])


class TestMoELayerFp8QuantWeight(unittest.TestCase):
    """Test fp8_quant_weight early return."""

    def test_early_return_when_not_fp8(self):
        config = _make_config(fp8=False, moe_use_fusion_node=False)
        layer = MoELayer.__new__(MoELayer)
        layer.config = config
        layer.moe_use_fusion_node = False
        layer.fp8 = False
        # Should return early without errors
        layer.fp8_quant_weight()


class TestMoELayerForwardLogging(unittest.TestCase):
    """Test forward loss logging hooks."""

    @patch("paddleformers.fleet.transformer.moe.moe_layer.framework._dygraph_tracer")
    @patch("paddleformers.fleet.transformer.moe.moe_layer.log_moe_losses")
    @patch("paddleformers.fleet.transformer.moe.moe_layer._log_moe_md5")
    def test_forward_logs_aux_and_zloss_when_has_grad(
        self, mock_log_md5, mock_log_moe_losses, mock_tracer
    ):
        del mock_log_md5
        hidden_states = paddle.randn([4, 64], dtype="float32")
        aux_loss = paddle.to_tensor(1.5, dtype="float32")
        z_loss = paddle.to_tensor(2.5, dtype="float32")

        layer = MoELayer.__new__(MoELayer)
        layer.sequence_parallel = False
        layer.expert_model_parallel_size = 1
        layer.shared_experts = None
        layer.moe_shared_expert_overlap = False
        layer.moe_use_fusion_node = False
        layer.moe_expert_fusion = False
        layer.training = False
        layer.router_aux_loss_coef = None
        layer.use_latent_moe = False
        layer.layer_number = 7
        layer.gate = MagicMock(
            return_value=(
                None,
                paddle.randn([4, 2], dtype="float32"),
                paddle.randint(0, 4, [4, 2], dtype="int64"),
                paddle.randn([4, 4], dtype="float32"),
                paddle.randint(0, 2, [4, 4], dtype="int64"),
                None,
                aux_loss,
                z_loss,
            )
        )
        layer._forward_single_card_moe = MagicMock(return_value=hidden_states)
        tracer = MagicMock()
        tracer._has_grad = True
        mock_tracer.return_value = tracer

        output, bias = layer.forward(hidden_states)

        mock_log_moe_losses.assert_called_once_with(
            7, aux_loss=aux_loss, z_loss=z_loss
        )
        self.assertEqual(list(output.shape), [4, 64])
        self.assertIsNone(bias)


if __name__ == "__main__":
    unittest.main()
