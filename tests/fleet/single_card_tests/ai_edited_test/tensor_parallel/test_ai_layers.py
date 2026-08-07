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
from unittest.mock import MagicMock, patch

import numpy as np
import paddle


def _make_mock_group(world_size=1, rank=0):
    """Create a mock process group."""
    mock = MagicMock()
    mock.world_size = world_size
    mock.nranks = world_size
    mock.rank = rank
    mock.ranks = list(range(world_size))
    return mock


class TestParamIsNotTensorParallelDuplicate(unittest.TestCase):
    """Tests for param_is_not_tensor_parallel_duplicate."""

    def test_with_tp_attribute_true(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            param_is_not_tensor_parallel_duplicate,
        )

        param = MagicMock()
        param.tensor_model_parallel = True

        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
            return_value=1,
        ):
            result = param_is_not_tensor_parallel_duplicate(param)
        self.assertTrue(result)

    def test_without_tp_attribute_rank0(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            param_is_not_tensor_parallel_duplicate,
        )

        param = MagicMock()
        del param.tensor_model_parallel

        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            result = param_is_not_tensor_parallel_duplicate(param)
        self.assertTrue(result)

    def test_without_tp_attribute_rank1(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            param_is_not_tensor_parallel_duplicate,
        )

        param = MagicMock()
        del param.tensor_model_parallel

        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
            return_value=1,
        ):
            result = param_is_not_tensor_parallel_duplicate(param)
        self.assertFalse(result)


class TestSetTensorModelParallelAttributes(unittest.TestCase):
    """Tests for set_tensor_model_parallel_attributes."""

    def test_set_attributes(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            set_tensor_model_parallel_attributes,
        )

        tensor = paddle.randn([4, 8])
        set_tensor_model_parallel_attributes(tensor, True, 1, 2)

        self.assertTrue(tensor.tensor_model_parallel)
        self.assertEqual(tensor.partition_dim, 1)
        self.assertEqual(tensor.partition_stride, 2)

    def test_already_set_raises(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            set_tensor_model_parallel_attributes,
        )

        tensor = paddle.randn([4, 8])
        set_tensor_model_parallel_attributes(tensor, True, 1, 2)

        with self.assertRaises(AssertionError):
            set_tensor_model_parallel_attributes(tensor, False, 0, 1)


class TestSetDefaultsIfNotSetTensorModelParallelAttributes(unittest.TestCase):
    """Tests for set_defaults_if_not_set_tensor_model_parallel_attributes."""

    def test_set_defaults(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            set_defaults_if_not_set_tensor_model_parallel_attributes,
        )

        tensor = paddle.randn([4, 8])
        set_defaults_if_not_set_tensor_model_parallel_attributes(tensor)

        self.assertFalse(tensor.tensor_model_parallel)
        self.assertEqual(tensor.partition_dim, -1)
        self.assertEqual(tensor.partition_stride, 1)

    def test_does_not_override(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            set_defaults_if_not_set_tensor_model_parallel_attributes,
            set_tensor_model_parallel_attributes,
        )

        tensor = paddle.randn([4, 8])
        set_tensor_model_parallel_attributes(tensor, True, 0, 1)
        set_defaults_if_not_set_tensor_model_parallel_attributes(tensor)

        self.assertTrue(tensor.tensor_model_parallel)
        self.assertEqual(tensor.partition_dim, 0)
        self.assertEqual(tensor.partition_stride, 1)


class TestCopyTensorModelParallelAttributes(unittest.TestCase):
    """Tests for copy_tensor_model_parallel_attributes."""

    def test_copy_attributes(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            copy_tensor_model_parallel_attributes,
            set_tensor_model_parallel_attributes,
        )

        src = paddle.randn([4, 8])
        dst = paddle.randn([4, 8])
        set_tensor_model_parallel_attributes(src, True, 0, 2)

        copy_tensor_model_parallel_attributes(dst, src)

        self.assertTrue(dst.tensor_model_parallel)
        self.assertEqual(dst.partition_dim, 0)
        self.assertEqual(dst.partition_stride, 2)

    def test_copy_from_non_parallel(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            copy_tensor_model_parallel_attributes,
            set_defaults_if_not_set_tensor_model_parallel_attributes,
        )

        src = paddle.randn([4, 8])
        dst = paddle.randn([4, 8])
        set_defaults_if_not_set_tensor_model_parallel_attributes(src)
        set_defaults_if_not_set_tensor_model_parallel_attributes(dst)

        copy_tensor_model_parallel_attributes(dst, src)
        self.assertFalse(dst.tensor_model_parallel)


class TestInitializeAffineWeightCPU(unittest.TestCase):
    """Tests for _initialize_affine_weight_cpu."""

    def test_basic_init(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            _initialize_affine_weight_cpu,
        )

        weight = paddle.empty([4, 8], dtype=paddle.float32)
        init_method = paddle.nn.initializer.Constant(1.0)

        with (
            patch(
                "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
            ),
        ):
            result = _initialize_affine_weight_cpu(
                weight,
                4,
                8,
                8,
                1,  # partition_dim
                init_method,
                rank=0,
                world_size=1,
            )
        self.assertIsNone(result)

    def test_with_return_master_weight(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            _initialize_affine_weight_cpu,
        )

        weight = paddle.empty([4, 8], dtype=paddle.float32)
        init_method = paddle.nn.initializer.Normal()

        with (
            patch(
                "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
            ),
        ):
            master = _initialize_affine_weight_cpu(
                weight,
                4,
                8,
                8,
                1,
                init_method,
                return_master_weight=True,
                rank=0,
                world_size=1,
            )
        self.assertIsNotNone(master)
        self.assertEqual(master.shape, [4, 8])

    def test_skip_tp_attributes(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            _initialize_affine_weight_cpu,
        )

        weight = paddle.empty([4, 8], dtype=paddle.float32)
        init_method = paddle.nn.initializer.Constant(0.0)

        result = _initialize_affine_weight_cpu(
            weight,
            4,
            8,
            8,
            1,
            init_method,
            skip_set_tensor_parallel_attributes=True,
            rank=0,
            world_size=1,
        )
        self.assertIsNone(result)
        # Should not have tensor_model_parallel attribute
        self.assertFalse(hasattr(weight, "tensor_model_parallel"))


class TestLinearWithFrozenWeight(unittest.TestCase):
    """Tests for LinearWithFrozenWeight autograd function."""

    def test_forward_with_bias(self):
        from paddleformers.fleet.tensor_parallel.layers import LinearWithFrozenWeight

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = True
        bias = paddle.randn([16], dtype=paddle.float32)

        output = LinearWithFrozenWeight.apply(
            input_tensor, weight, bias, False, None
        )
        self.assertEqual(output.shape, [4, 16])

    def test_forward_without_bias(self):
        from paddleformers.fleet.tensor_parallel.layers import LinearWithFrozenWeight

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = True

        output = LinearWithFrozenWeight.apply(
            input_tensor, weight, None, False, None
        )
        self.assertEqual(output.shape, [4, 16])


class TestLinearWithFrozenWeightFunc(unittest.TestCase):
    """Tests for linear_with_frozen_weight wrapper function."""

    def test_basic_call(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_frozen_weight,
        )

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = True
        bias = paddle.randn([16], dtype=paddle.float32)

        output = linear_with_frozen_weight(
            input_tensor,
            weight,
            bias,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
            tp_group=None,
        )
        self.assertEqual(output.shape, [4, 16])

    def test_with_sequence_parallel(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_frozen_weight,
        )

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = True

        with patch(
            "paddleformers.fleet.tensor_parallel.layers.gather_from_sequence_parallel_region",
            return_value=input_tensor,
        ):
            output = linear_with_frozen_weight(
                input_tensor,
                weight,
                None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=True,
                tp_group=None,
            )
        self.assertEqual(output.shape, [4, 16])

    def test_grad_output_buffer_raises(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_frozen_weight,
        )

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = True

        with self.assertRaises(AssertionError):
            linear_with_frozen_weight(
                input_tensor,
                weight,
                None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=False,
                tp_group=None,
                grad_output_buffer=[],
            )

    def test_wgrad_deferral_limit_raises(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_frozen_weight,
        )

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = True

        with self.assertRaises(AssertionError):
            linear_with_frozen_weight(
                input_tensor,
                weight,
                None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=False,
                tp_group=None,
                wgrad_deferral_limit=5,
            )

    def test_async_grad_allreduce_warns(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_frozen_weight,
        )

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = True

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            linear_with_frozen_weight(
                input_tensor,
                weight,
                None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=False,
                tp_group=None,
                async_grad_allreduce=True,
            )
            self.assertTrue(len(w) > 0)


class TestLinearLayer(unittest.TestCase):
    """Tests for the Linear layer class (no tensor parallelism)."""

    def _make_config(self, **kwargs):
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        defaults = {
            "params_dtype": paddle.float32,
            "perform_initialization": True,
            "use_cpu_initialization": True,
            "sequence_parallel": False,
            "deterministic_mode": False,
            "gradient_accumulation_fusion": False,
            "defer_embedding_wgrad_compute": False,
            "wgrad_deferral_limit": 0,
            "expert_model_parallel_size": 1,
        }
        defaults.update(kwargs)
        config = ModelParallelConfig(**defaults)
        return config

    def test_linear_no_bias(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(
            8, 16, config=config, init_method=init_method, bias=False
        )
        input_tensor = paddle.randn([2, 4, 8])
        output, output_bias = layer(input_tensor)
        self.assertEqual(output.shape, [2, 4, 16])
        self.assertIsNone(output_bias)

    def test_linear_skip_bias_add(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(
            8,
            16,
            config=config,
            init_method=init_method,
            bias=True,
            skip_bias_add=True,
        )
        input_tensor = paddle.randn([2, 4, 8])
        output, output_bias = layer(input_tensor)
        self.assertEqual(output.shape, [2, 4, 16])
        # bias should be returned separately when skip_bias_add is True
        self.assertIsNotNone(output_bias)

    def test_linear_repr(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(8, 16, config=config, init_method=init_method, bias=True)
        repr_str = repr(layer)
        self.assertIn("Linear", repr_str)
        self.assertIn("in_features=8", repr_str)
        self.assertIn("out_features=16", repr_str)

    def test_linear_get_extra_state(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(8, 16, config=config, init_method=init_method, bias=True)
        self.assertIsNone(layer.get_extra_state())

    def test_linear_set_extra_state(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(8, 16, config=config, init_method=init_method, bias=True)
        # Should not raise
        layer.set_extra_state(None)

    def test_linear_skip_weight_allocation_raises(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(
            8,
            16,
            config=config,
            init_method=init_method,
            skip_weight_param_allocation=True,
        )
        input_tensor = paddle.randn([2, 4, 8])
        with self.assertRaises(RuntimeError):
            layer(input_tensor)

    def test_linear_wrong_weight_shape_raises(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(
            8,
            16,
            config=config,
            init_method=init_method,
            skip_weight_param_allocation=True,
        )
        wrong_weight = paddle.randn([4, 8])  # Should be [8, 16]
        input_tensor = paddle.randn([2, 4, 8])
        with self.assertRaises(RuntimeError):
            layer(input_tensor, weight=wrong_weight)

    def test_linear_skip_weight_with_correct_weight(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(
            8,
            16,
            config=config,
            init_method=init_method,
            skip_weight_param_allocation=True,
        )
        correct_weight = paddle.randn([8, 16])
        input_tensor = paddle.randn([2, 4, 8])
        output, output_bias = layer(input_tensor, weight=correct_weight)
        self.assertEqual(output.shape, [2, 4, 16])

    def test_linear_with_frozen_weight_forward(self):
        from paddleformers.fleet.tensor_parallel.layers import Linear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = Linear(8, 16, config=config, init_method=init_method, bias=True)
        # Stop gradient on weight to trigger frozen weight path
        layer.weight.stop_gradient = True
        input_tensor = paddle.randn([2, 4, 8])
        output, output_bias = layer(input_tensor)
        self.assertEqual(output.shape, [2, 4, 16])


class TestColumnParallelLinearBasic(unittest.TestCase):
    """Tests for ColumnParallelLinear basics."""

    def _make_config(self, **kwargs):
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        defaults = {
            "params_dtype": paddle.float32,
            "perform_initialization": True,
            "use_cpu_initialization": True,
            "sequence_parallel": False,
            "deterministic_mode": False,
            "gradient_accumulation_fusion": False,
            "defer_embedding_wgrad_compute": False,
            "wgrad_deferral_limit": 0,
            "expert_model_parallel_size": 1,
        }
        defaults.update(kwargs)
        return ModelParallelConfig(**defaults)

    def _make_mock_group(self, world_size=1, rank=0):
        mock = MagicMock()
        mock.world_size = world_size
        mock.nranks = world_size
        mock.rank = rank
        mock.ranks = list(range(world_size))
        return mock

    def test_column_linear_basic(self):
        from paddleformers.fleet.tensor_parallel.layers import ColumnParallelLinear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        mock_group = self._make_mock_group(world_size=1, rank=0)
        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            layer = ColumnParallelLinear(
                8, 16, config=config, init_method=init_method, bias=True
            )
        input_tensor = paddle.randn([2, 4, 8])
        output, output_bias = layer(input_tensor)
        self.assertEqual(output.shape, [2, 4, 16])

    def test_column_linear_repr(self):
        from paddleformers.fleet.tensor_parallel.layers import ColumnParallelLinear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        mock_group = self._make_mock_group(world_size=1, rank=0)
        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            layer = ColumnParallelLinear(
                8, 16, config=config, init_method=init_method, bias=True
            )
        repr_str = repr(layer)
        self.assertIn("ColumnParallelLinear", repr_str)
        self.assertIn("in_features=8", repr_str)
        self.assertIn("out_features=16", repr_str)

    def test_column_linear_get_extra_state(self):
        from paddleformers.fleet.tensor_parallel.layers import ColumnParallelLinear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        mock_group = self._make_mock_group(world_size=1, rank=0)
        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            layer = ColumnParallelLinear(
                8, 16, config=config, init_method=init_method, bias=True
            )
        self.assertIsNone(layer.get_extra_state())

    def test_column_linear_set_extra_state(self):
        from paddleformers.fleet.tensor_parallel.layers import ColumnParallelLinear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        mock_group = self._make_mock_group(world_size=1, rank=0)
        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            layer = ColumnParallelLinear(
                8, 16, config=config, init_method=init_method, bias=True
            )
        layer.set_extra_state(None)


class TestRowParallelLinearBasic(unittest.TestCase):
    """Tests for RowParallelLinear basics."""

    def _make_config(self, **kwargs):
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        defaults = {
            "params_dtype": paddle.float32,
            "perform_initialization": True,
            "use_cpu_initialization": True,
            "sequence_parallel": False,
            "deterministic_mode": False,
            "gradient_accumulation_fusion": False,
            "defer_embedding_wgrad_compute": False,
            "wgrad_deferral_limit": 0,
            "expert_model_parallel_size": 1,
        }
        defaults.update(kwargs)
        return ModelParallelConfig(**defaults)

    def _make_mock_group(self, world_size=1, rank=0):
        mock = MagicMock()
        mock.world_size = world_size
        mock.nranks = world_size
        mock.rank = rank
        mock.ranks = list(range(world_size))
        return mock

    def test_row_linear_basic(self):
        from paddleformers.fleet.tensor_parallel.layers import RowParallelLinear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        mock_group = self._make_mock_group(world_size=1, rank=0)
        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            layer = RowParallelLinear(
                8,
                16,
                config=config,
                init_method=init_method,
                bias=True,
                input_is_parallel=False,
                skip_bias_add=False,
            )
        input_tensor = paddle.randn([2, 4, 8])
        output, output_bias = layer(input_tensor)
        self.assertEqual(output.shape, [2, 4, 16])

    def test_row_linear_skip_bias_add(self):
        from paddleformers.fleet.tensor_parallel.layers import RowParallelLinear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        mock_group = self._make_mock_group(world_size=1, rank=0)
        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            layer = RowParallelLinear(
                8,
                16,
                config=config,
                init_method=init_method,
                bias=True,
                skip_bias_add=True,
                input_is_parallel=False,
            )
        input_tensor = paddle.randn([2, 4, 8])
        output, output_bias = layer(input_tensor)
        self.assertEqual(output.shape, [2, 4, 16])
        self.assertIsNotNone(output_bias)

    def test_row_linear_repr(self):
        from paddleformers.fleet.tensor_parallel.layers import RowParallelLinear

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        mock_group = self._make_mock_group(world_size=1, rank=0)
        with patch(
            "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            layer = RowParallelLinear(
                8,
                16,
                config=config,
                init_method=init_method,
                bias=True,
                input_is_parallel=False,
                skip_bias_add=False,
            )
        repr_str = repr(layer)
        self.assertIn("RowParallelLinear", repr_str)


class TestVocabParallelEmbeddingBasic(unittest.TestCase):
    """Tests for VocabParallelEmbedding basics."""

    def _make_config(self, **kwargs):
        from paddleformers.fleet.model_parallel_config import ModelParallelConfig

        defaults = {
            "params_dtype": paddle.float32,
            "perform_initialization": True,
            "use_cpu_initialization": True,
            "deterministic_mode": False,
            "sequence_parallel": False,
            "gradient_accumulation_fusion": False,
            "expert_model_parallel_size": 1,
        }
        defaults.update(kwargs)
        return ModelParallelConfig(**defaults)

    def test_vocab_embedding_basic(self):
        from paddleformers.fleet.tensor_parallel.layers import VocabParallelEmbedding

        config = self._make_config()
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = VocabParallelEmbedding(
            100,
            32,
            init_method=init_method,
            reduce_scatter_embeddings=False,
            config=config,
        )
        input_ids = paddle.randint(0, 100, shape=[4, 8])
        output = layer(input_ids)
        self.assertEqual(output.shape, [4, 8, 32])

    def test_vocab_embedding_deterministic_mode(self):
        from paddleformers.fleet.tensor_parallel.layers import VocabParallelEmbedding

        config = self._make_config(deterministic_mode=True)
        init_method = paddle.nn.initializer.Constant(1.0)
        layer = VocabParallelEmbedding(
            100,
            32,
            init_method=init_method,
            reduce_scatter_embeddings=False,
            config=config,
        )
        input_ids = paddle.randint(0, 100, shape=[4, 8])
        output = layer(input_ids)
        self.assertEqual(output.shape, [4, 8, 32])


class TestLinearWithGradAccumulationAndAsyncCommunication(unittest.TestCase):
    """Tests for LinearWithGradAccumulationAndAsyncCommunication."""

    def test_forward_basic(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            LinearWithGradAccumulationAndAsyncCommunication,
        )

        mock_group = _make_mock_group(world_size=1)
        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        bias = paddle.randn([16], dtype=paddle.float32)

        output = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_tensor,
            weight,
            bias,
            False,  # gradient_accumulation_fusion
            False,  # allreduce_dgrad
            False,  # sequence_parallel
            None,  # grad_output_buffer
            0,  # wgrad_deferral_limit
            mock_group,  # tp_group
        )
        self.assertEqual(output.shape, [4, 16])

    def test_forward_no_bias(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            LinearWithGradAccumulationAndAsyncCommunication,
        )

        mock_group = _make_mock_group(world_size=1)
        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)

        output = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_tensor,
            weight,
            None,
            False,
            False,
            False,
            None,
            0,
            mock_group,
        )
        self.assertEqual(output.shape, [4, 16])


class TestLinearWithGradAccumulationFunc(unittest.TestCase):
    """Tests for linear_with_grad_accumulation_and_async_allreduce function."""

    def test_basic_call(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_grad_accumulation_and_async_allreduce,
        )

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        bias = paddle.randn([16], dtype=paddle.float32)

        output = linear_with_grad_accumulation_and_async_allreduce(
            input_tensor,
            weight,
            bias,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
            tp_group=None,
        )
        self.assertEqual(output.shape, [4, 16])

    def test_async_grad_allreduce_warns(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_grad_accumulation_and_async_allreduce,
        )

        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        bias = paddle.randn([16], dtype=paddle.float32)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            linear_with_grad_accumulation_and_async_allreduce(
                input_tensor,
                weight,
                bias,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=False,
                tp_group=None,
                async_grad_allreduce=True,
            )
            self.assertTrue(len(w) > 0)


class TestLinearWithGradAccumUseAccuracyCompatible(unittest.TestCase):
    """Tests for the use_accuracy_compatible branch in
    LinearWithGradAccumulationAndAsyncCommunication.forward (no-bias path)."""

    def _run(self, use_accuracy_compatible, input_tensor, weight):
        from paddleformers.fleet.tensor_parallel.layers import (
            LinearWithGradAccumulationAndAsyncCommunication,
        )

        mock_group = _make_mock_group(world_size=1)
        return LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_tensor,
            weight,
            None,  # bias -> exercises the matmul / F.linear branch
            False,  # gradient_accumulation_fusion
            False,  # allreduce_dgrad
            False,  # sequence_parallel
            None,  # grad_output_buffer
            0,  # wgrad_deferral_limit
            mock_group,  # tp_group
            use_accuracy_compatible,  # use_accuracy_compatible
        )

    def test_accuracy_compatible_true_matches_default(self):
        """F.linear (compatible) and matmul (default) must give the same
        result for the no-bias forward."""
        paddle.seed(2026)
        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)

        out_compat = self._run(True, input_tensor, weight)
        out_default = self._run(False, input_tensor, weight)

        self.assertEqual(out_compat.shape, [4, 16])
        np.testing.assert_allclose(
            out_compat.numpy(),
            out_default.numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_accuracy_compatible_matches_reference_linear(self):
        """The compatible branch must equal paddle.nn.functional.linear."""
        paddle.seed(0)
        input_tensor = paddle.randn([3, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 5], dtype=paddle.float32)

        out_compat = self._run(True, input_tensor, weight)
        ref = paddle.nn.functional.linear(input_tensor, weight)
        np.testing.assert_allclose(
            out_compat.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_default_flag_is_matmul(self):
        """When the flag is omitted it defaults to False (matmul branch)."""
        from paddleformers.fleet.tensor_parallel.layers import (
            LinearWithGradAccumulationAndAsyncCommunication,
        )

        mock_group = _make_mock_group(world_size=1)
        input_tensor = paddle.randn([2, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 4], dtype=paddle.float32)

        # No use_accuracy_compatible argument -> defaults to False.
        out = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_tensor,
            weight,
            None,
            False,
            False,
            False,
            None,
            0,
            mock_group,
        )
        ref = paddle.matmul(input_tensor, weight)
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_wrapper_passes_flag_through(self):
        """linear_with_grad_accumulation_and_async_allreduce forwards
        use_accuracy_compatible into the forward and gives a valid result."""
        from paddleformers.fleet.tensor_parallel.layers import (
            linear_with_grad_accumulation_and_async_allreduce,
        )

        paddle.seed(7)
        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        weight = paddle.randn([8, 16], dtype=paddle.float32)

        out = linear_with_grad_accumulation_and_async_allreduce(
            input_tensor,
            weight,
            None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
            tp_group=None,
            use_accuracy_compatible=True,
        )
        ref = paddle.nn.functional.linear(input_tensor, weight)
        self.assertEqual(out.shape, [4, 16])
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_compatible_backward_grads(self):
        """Gradients must flow through the F.linear (compatible) branch."""
        paddle.seed(11)
        input_tensor = paddle.randn([4, 8], dtype=paddle.float32)
        input_tensor.stop_gradient = False
        weight = paddle.randn([8, 16], dtype=paddle.float32)
        weight.stop_gradient = False

        out = self._run(True, input_tensor, weight)
        out.sum().backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(weight.grad)
        self.assertEqual(input_tensor.grad.shape, [4, 8])
        self.assertEqual(weight.grad.shape, [8, 16])


class TestGradAccumFusionAvailable(unittest.TestCase):
    """Tests for _grad_accum_fusion_available flag."""

    def test_flag_exists(self):
        from paddleformers.fleet.tensor_parallel.layers import (
            _grad_accum_fusion_available,
        )

        self.assertIsInstance(_grad_accum_fusion_available, bool)


class TestHaveTE(unittest.TestCase):
    """Tests for HAVE_TE flag."""

    def test_have_te(self):
        from paddleformers.fleet.tensor_parallel.layers import HAVE_TE

        self.assertIsInstance(HAVE_TE, bool)
