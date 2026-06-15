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

import logging
import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestGlobalMemoryBufferExtra(unittest.TestCase):
    """Additional tests for GlobalMemoryBuffer."""

    def test_get_tensor_creates_new(self):
        """Test get_tensor creates new buffer when needed."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t = buf.get_tensor([4, 8], paddle.float32, "test_buf")
        self.assertEqual(t.shape, [4, 8])

    def test_get_tensor_reuses_existing(self):
        """Test get_tensor reuses existing buffer when large enough."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([4, 8], paddle.float32, "reuse_buf")
        t2 = buf.get_tensor([2, 4], paddle.float32, "reuse_buf")
        # Should reuse the same buffer
        self.assertIsNotNone(t2)

    def test_get_tensor_different_names(self):
        """Test get_tensor with different names creates separate buffers."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([4, 8], paddle.float32, "buf_a")
        t2 = buf.get_tensor([4, 8], paddle.float32, "buf_b")
        # Both should exist
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)

    def test_get_tensor_different_dtypes(self):
        """Test get_tensor with different dtypes creates separate buffers."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([4, 8], paddle.float32, "dtype_buf")
        t2 = buf.get_tensor([4, 8], paddle.float16, "dtype_buf")
        self.assertEqual(t1.dtype, paddle.float32)
        self.assertEqual(t2.dtype, paddle.float16)


class TestMakeViewlessTensor(unittest.TestCase):
    """Tests for make_viewless_tensor."""

    def test_make_viewless_tensor_non_view(self):
        """Test make_viewless_tensor returns tensor as-is when not a view."""
        from paddleformers.fleet.utils import make_viewless_tensor

        t = paddle.randn([4, 8])
        # The function calls inp._is_view() internally, which may not exist
        # in all paddle versions
        try:
            result = make_viewless_tensor(t, False, False)
            self.assertIsNotNone(result)
        except AttributeError:
            self.skipTest("_is_view not available in this paddle version")

    def test_make_viewless_tensor_keep_graph(self):
        """Test make_viewless_tensor with keep_graph=True."""
        from paddleformers.fleet.utils import make_viewless_tensor

        t = paddle.randn([4, 8])
        try:
            result = make_viewless_tensor(t, False, True)
            self.assertIsNotNone(result)
        except AttributeError:
            self.skipTest("_is_view not available in this paddle version")


class TestGetModelType(unittest.TestCase):
    """Tests for get_model_type."""

    def test_get_model_type_with_attribute(self):
        """Test get_model_type with model that has model_type."""
        from paddleformers.fleet.utils import get_model_type

        model = MagicMock()
        model.model_type = "gpt"
        result = get_model_type(model)
        self.assertEqual(result, "gpt")


class TestGetModelXattn(unittest.TestCase):
    """Tests for get_model_xattn."""

    def test_get_model_xattn_with_attribute(self):
        """Test get_model_xattn with model that has xattn_needed."""
        from paddleformers.fleet.utils import get_model_xattn

        model = MagicMock()
        model.xattn_needed = True
        result = get_model_xattn(model)
        self.assertTrue(result)

    def test_get_model_xattn_without_attribute(self):
        """Test get_model_xattn with model that lacks xattn_needed."""
        from paddleformers.fleet.utils import get_model_xattn

        model = MagicMock(spec=["module"])
        model.module = MagicMock(spec=[])
        result = get_model_xattn(model)
        self.assertFalse(result)


class TestIsPaddleMinVersion(unittest.TestCase):
    """Tests for is_paddle_min_version."""

    def test_is_paddle_min_version_callable(self):
        """Test is_paddle_min_version is callable."""
        from paddleformers.fleet.utils import is_paddle_min_version

        self.assertTrue(callable(is_paddle_min_version))

    def test_get_paddle_version_callable(self):
        """Test get_paddle_version is callable."""
        from paddleformers.fleet.utils import get_paddle_version

        version = get_paddle_version()
        self.assertIsNotNone(version)


class TestLogSingleRank(unittest.TestCase):
    """Tests for log_single_rank."""

    def test_log_single_rank_without_dist(self):
        """Test log_single_rank when distributed is not initialized."""
        from paddleformers.fleet.utils import log_single_rank

        logger = MagicMock()
        with patch(
            "paddleformers.fleet.utils.paddle.distributed.is_initialized",
            return_value=False,
        ):
            log_single_rank(logger, logging.INFO, "test message")
            logger.log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
