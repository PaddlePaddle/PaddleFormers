import types
import unittest
from unittest.mock import patch

import numpy as np

from paddleformers.trainer.trainer import Trainer


class FakeParam:
    def __init__(self, shape, stop_gradient=False, expert=False, is_moe_param=False):
        self.shape = shape
        self.stop_gradient = stop_gradient
        self.expert = expert
        self.is_moe_param = is_moe_param


class FakeModel:
    def __init__(self, named_parameters, config):
        self._named_parameters = named_parameters
        self.config = config

    def named_parameters(self):
        return list(self._named_parameters)

    def parameters(self):
        return [param for _, param in self._named_parameters]


class TestMoeTrainableNumel(unittest.TestCase):
    def setUp(self):
        self.trainer = object.__new__(Trainer)
        self.trainer.args = types.SimpleNamespace(
            enable_auto_parallel=False,
            use_hybrid_parallel=False,
            world_size=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=1,
        )

    def test_log_moe_trainable_numel_allreduces_before_moe_check(self):
        model = FakeModel(
            [("layers.0.mlp.weight", FakeParam((2, 3)))],
            types.SimpleNamespace(num_hidden_layers=1, n_routed_experts=8, num_experts_per_tok=2),
        )
        reduced_numels = np.array([6, 0, 8, 0], dtype="int64")

        def fake_all_reduce(tensor):
            tensor.set_value(reduced_numels)

        with patch("paddleformers.trainer.trainer.unwrap_model", return_value=model), patch(
            "paddleformers.trainer.trainer.paddle.distributed.is_initialized", return_value=True
        ), patch("paddleformers.trainer.trainer.paddle.distributed.all_reduce", side_effect=fake_all_reduce) as all_reduce, patch(
            "paddleformers.trainer.trainer.paddle.get_device", return_value="gpu:0"
        ), patch("paddleformers.trainer.trainer.logger.debug") as debug:
            self.trainer.log_moe_trainable_numel(model)

        all_reduce.assert_called_once()
        messages = [call.args[0] for call in debug.call_args_list]
        self.assertIn("  Number of trainable parameters = 14 (whole model, expert parallel restored)", messages)
        self.assertIn("  Number of trainable parameters = 8 (activated per token)", messages)

    def test_log_moe_trainable_numel_skips_dense_model_after_reduction(self):
        model = FakeModel(
            [("layers.0.mlp.weight", FakeParam((2, 3)))],
            types.SimpleNamespace(num_hidden_layers=1, n_routed_experts=8, num_experts_per_tok=2),
        )

        with patch("paddleformers.trainer.trainer.unwrap_model", return_value=model), patch(
            "paddleformers.trainer.trainer.paddle.distributed.is_initialized", return_value=True
        ), patch("paddleformers.trainer.trainer.paddle.distributed.all_reduce") as all_reduce, patch(
            "paddleformers.trainer.trainer.paddle.get_device", return_value="gpu:0"
        ), patch("paddleformers.trainer.trainer.logger.debug") as debug:
            self.trainer.log_moe_trainable_numel(model)

        all_reduce.assert_called_once()
        debug.assert_not_called()


if __name__ == "__main__":
    unittest.main()
