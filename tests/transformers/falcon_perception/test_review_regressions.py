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

"""Regression checks for initialization, CPU normalization, and optional RLE dependencies."""
import subprocess
import sys

import numpy as np
import paddle
import pytest

from paddleformers.transformers.falcon_perception.anyup import (
    AnyUp,
    AnyUpRoPE,
    AttentionWrapper,
)
from paddleformers.transformers.falcon_perception.norm import rms_norm


@pytest.fixture(autouse=True)
def cpu_device():
    previous = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(previous)


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(4,), (3, 4)])
def test_cpu_rms_norm_matches_numpy(dtype, shape):
    values = paddle.to_tensor(np.linspace(-2, 2, 24).reshape(2, 3, 4), dtype=dtype)
    weight = paddle.to_tensor(np.linspace(0.5, 1.5, np.prod(shape)).reshape(shape), dtype=dtype)
    actual = rms_norm(values, shape, weight)
    x = values.astype("float64").numpy()
    w = weight.astype("float64").numpy()
    eps = np.finfo(np.float64 if dtype == "float64" else np.float32).eps
    axes = tuple(range(x.ndim - len(shape), x.ndim))
    expected = x / np.sqrt(np.mean(x**2, axis=axes, keepdims=True) + eps) * w
    expected = paddle.to_tensor(expected, dtype=dtype).astype("float64").numpy()
    assert actual.dtype == values.dtype
    np.testing.assert_allclose(actual.astype("float64").numpy(), expected, atol=1e-6, rtol=1e-6)
    assert bool(paddle.isfinite(rms_norm(paddle.zeros([2, 4]), [4])).all())


def test_anyup_initialization_is_reproducible_and_forward_is_finite():
    paddle.seed(42)
    first = AnyUpRoPE(16)
    attention = AttentionWrapper(16)
    paddle.seed(42)
    second = AnyUpRoPE(16)
    np.testing.assert_array_equal(first.freqs.numpy(), second.freqs.numpy())
    np.testing.assert_array_equal(attention.in_proj_bias.numpy(), np.zeros(48))
    assert float(paddle.abs(first.freqs).sum()) > 0
    assert float(paddle.abs(attention.in_proj_weight).sum()) > 0
    from paddleformers.transformers.falcon_perception.attention import WindowMask

    model = AnyUp(qk_dim=16, num_heads=4).eval()
    with paddle.no_grad():
        output = model(paddle.randn([1, 3, 8, 8]), paddle.randn([1, 8, 2, 2]), WindowMask(8, 8, 2, 2))
    assert output.shape == [1, 8, 8, 8]
    assert bool(paddle.isfinite(output).all())


def test_model_import_does_not_require_pycocotools():
    code = """
import importlib.abc
import sys
class BlockCoco(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pycocotools' or fullname.startswith('pycocotools.'):
            raise ModuleNotFoundError('Blocked optional dependency')
sys.meta_path.insert(0, BlockCoco())
import paddle
from paddleformers.transformers import FalconPerceptionForSegmentation
assert 'pycocotools' not in sys.modules
try:
    FalconPerceptionForSegmentation._mask_to_coco_rle(paddle.ones([1, 2, 2], dtype='bool'))
except ImportError as error:
    assert 'pip install pycocotools' in str(error)
else:
    raise AssertionError('Expected actionable optional dependency error')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
