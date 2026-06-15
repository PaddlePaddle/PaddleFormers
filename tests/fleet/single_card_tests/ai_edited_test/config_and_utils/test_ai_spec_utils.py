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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest

import paddle


class TestLayerSpec(unittest.TestCase):
    """Tests for the LayerSpec dataclass in paddleformers.fleet.spec_utils."""

    def test_layer_spec_creation_with_tuple(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        spec = LayerSpec(layer=("os.path", "join"))
        self.assertEqual(spec.layer, ("os.path", "join"))
        self.assertEqual(spec.extra_kwargs, {})

    def test_layer_spec_creation_with_class(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        spec = LayerSpec(layer=paddle.nn.Linear, extra_kwargs={"in_features": 10})
        self.assertEqual(spec.layer, paddle.nn.Linear)
        self.assertEqual(spec.extra_kwargs, {"in_features": 10})

    def test_layer_spec_default_extra_kwargs(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        spec = LayerSpec(layer=paddle.nn.ReLU)
        self.assertEqual(spec.extra_kwargs, {})

    def test_layer_spec_repr_with_tuple(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        spec = LayerSpec(layer=("os.path", "join"))
        r = repr(spec)
        self.assertIsInstance(r, str)
        self.assertIn("join", r)

    def test_layer_spec_repr_with_class(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        spec = LayerSpec(layer=paddle.nn.ReLU, extra_kwargs={"a": 1})
        r = repr(spec)
        self.assertIsInstance(r, str)

    def test_layer_spec_sublayers_spec(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        spec = LayerSpec(layer=paddle.nn.Linear, sublayers_spec=paddle.nn.Sequential)
        self.assertEqual(spec.sublayers_spec, paddle.nn.Sequential)

    def test_layer_spec_default_sublayers_spec_none(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        spec = LayerSpec(layer=paddle.nn.Linear)
        self.assertIsNone(spec.sublayers_spec)


class TestImportLayer(unittest.TestCase):
    """Tests for the import_spec_layer function in paddleformers.fleet.spec_utils."""

    def test_import_layer_success(self):
        from paddle.distributed.fleet.meta_parallel import import_spec_layer

        result = import_spec_layer(("os.path", "join"))
        import os.path

        self.assertEqual(result, os.path.join)

    def test_import_layer_import_error(self):
        from paddle.distributed.fleet.meta_parallel import import_spec_layer

        result = import_spec_layer(("nonexistent_module_xyz", "SomeClass"))
        self.assertIsNone(result)


class TestGetLayer(unittest.TestCase):
    """Tests for the get_layer function in paddleformers.fleet.spec_utils."""

    def test_get_layer_with_type(self):
        from paddle.distributed.fleet.meta_parallel import get_spec_layer

        result = get_spec_layer(paddle.nn.ReLU)
        self.assertEqual(result, paddle.nn.ReLU)

    def test_get_layer_with_function(self):
        from paddle.distributed.fleet.meta_parallel import get_spec_layer

        def my_fn():
            pass

        result = get_spec_layer(my_fn)
        self.assertEqual(result, my_fn)

    def test_get_layer_with_spec_containing_type(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec, get_spec_layer

        spec = LayerSpec(layer=paddle.nn.ReLU)
        result = get_spec_layer(spec)
        self.assertEqual(result, paddle.nn.ReLU)

    def test_get_layer_with_spec_containing_function(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec, get_spec_layer

        def my_fn():
            pass

        spec = LayerSpec(layer=my_fn)
        result = get_spec_layer(spec)
        self.assertEqual(result, my_fn)

    def test_get_layer_with_spec_containing_tuple(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec, get_spec_layer

        spec = LayerSpec(layer=("os.path", "join"))
        result = get_spec_layer(spec)
        import os.path

        self.assertEqual(result, os.path.join)


class TestBuildLayer(unittest.TestCase):
    """Tests for the build_spec_layer function in paddleformers.fleet.spec_utils."""

    def test_build_layer_with_function(self):
        from paddle.distributed.fleet.meta_parallel import build_spec_layer

        def my_fn():
            return 42

        result = build_spec_layer(my_fn)
        self.assertEqual(result, my_fn)

    def test_build_layer_with_spec_containing_function(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

        def my_fn():
            return 42

        spec = LayerSpec(layer=my_fn)
        result = build_spec_layer(spec)
        self.assertEqual(result, my_fn)

    def test_build_layer_with_class(self):
        from paddle.distributed.fleet.meta_parallel import build_spec_layer

        result = build_spec_layer(paddle.nn.ReLU)
        self.assertIsInstance(result, paddle.nn.ReLU)

    def test_build_layer_with_spec_containing_class(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

        spec = LayerSpec(layer=paddle.nn.ReLU)
        result = build_spec_layer(spec)
        self.assertIsInstance(result, paddle.nn.ReLU)

    def test_build_layer_with_import_path(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

        spec = LayerSpec(layer=("paddle.nn", "ReLU"))
        result = build_spec_layer(spec)
        self.assertIsInstance(result, paddle.nn.ReLU)

    def test_build_layer_extra_kwargs_conflict_warning(self):
        """Test that a warning is raised when extra_kwargs and kwargs have same key."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

        spec = LayerSpec(
            layer=paddle.nn.Linear,
            extra_kwargs={"in_features": 4, "out_features": 8},
        )
        with self.assertWarns(UserWarning):
            result = build_spec_layer(spec, in_features=10)

    def test_build_layer_exception_with_improved_message(self):
        """Test that build_spec_layer re-raises with improved error message."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

        spec = LayerSpec(layer=paddle.nn.Linear, extra_kwargs={})
        # Missing required arguments should produce an error mentioning Linear
        with self.assertRaises(Exception) as ctx:
            build_spec_layer(spec)
        self.assertIn("Linear", str(ctx.exception))
