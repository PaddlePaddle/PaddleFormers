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
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddleformers.fleet.transformer.utils import (
    get_doc_lens,
    get_doc_starts,
    inspect_tensor,
)


class TestInspectTensor(unittest.TestCase):
    """Tests for inspect_tensor."""

    def _make_tensor(self, shape=(2, 4), dtype="float32"):
        arr = np.random.randn(*shape).astype(np.float32)
        return paddle.to_tensor(arr)

    def test_returns_original_tensor_when_inspect_disabled(self):
        """inspect_flag=0, should return tensor unchanged without printing."""
        t = self._make_tensor()
        with patch("builtins.print") as mock_print:
            result = inspect_tensor("q", 0, t, save=False, load=True)
        self.assertIs(result, t)
        mock_print.assert_not_called()

    def test_none_tensor_returns_none(self):
        """tensor is None should return None immediately."""
        result = inspect_tensor("q", 0, None, save=False, load=True)
        self.assertIsNone(result)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_none_tensor_with_inspect_enabled(self):
        """inspect_flag=1 but tensor is None, should return None without crash."""
        with patch("builtins.print") as mock_print:
            result = inspect_tensor("q", 0, None, save=False, load=False)
        self.assertIsNone(result)
        mock_print.assert_not_called()

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_info_prints_with_rank_abssum_md5(self):
        """When inspect enabled, should print info with rank, abssum, md5."""
        t = self._make_tensor()
        with patch("builtins.print") as mock_print:
            result = inspect_tensor("attn_out", 2, t, save=False, load=False)
        self.assertIs(result, t)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("[ABLATION_train]", output)
        self.assertIn("tag=attn_out", output)
        self.assertIn("layer=2", output)
        self.assertIn("rank=", output)
        self.assertIn("abssum=", output)
        self.assertIn("md5=", output)
        self.assertIn("shape=", output)
        self.assertIn("dtype=", output)

    @patch.dict(
        os.environ,
        {"ABLATION_INSPECT_TENSOR": "1", "ABLATION_INFO_SKIP_TAGS": "q,k"},
    )
    def test_info_skip_tags(self):
        """Tags in ABLATION_INFO_SKIP_TAGS should not be printed."""
        t = self._make_tensor()
        with patch("builtins.print") as mock_print:
            inspect_tensor("q", 0, t, save=False, load=False)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertNotIn("[ABLATION_train]", output)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_info_exception_handling(self):
        """If tensor info computation fails, should print info_failed without crashing."""
        t = self._make_tensor()
        with (
            patch.object(t, "astype", side_effect=RuntimeError("cast fail")),
            patch("builtins.print") as mock_print,
        ):
            result = inspect_tensor("q", 0, t, save=False, load=False)
        self.assertIs(result, t)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("info_failed", output)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_save_tensor_to_disk(self):
        """save=True should dump tensor to rank_0/layer_X/{tag}.npy."""
        t = self._make_tensor((3, 5))
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.dict(os.environ, {"ABLATION_SAVE_TENSOR_PATH": tmpdir}),
                patch("builtins.print"),
            ):
                inspect_tensor("q", 2, t, save=True, load=False)
            saved_path = os.path.join(tmpdir, "rank_0", "layer_2", "q.npy")
            self.assertTrue(os.path.exists(saved_path))
            loaded_arr = np.load(saved_path)
            np.testing.assert_allclose(
                loaded_arr, t.numpy().astype(np.float32), rtol=1e-5
            )

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_save_prints_dump_info(self):
        """save=True should print [ABLATION_dump_tensor] info."""
        t = self._make_tensor()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"ABLATION_SAVE_TENSOR_PATH": tmpdir}),
            patch("builtins.print") as mock_print,
        ):
            inspect_tensor("q", 0, t, save=True, load=False)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("[ABLATION_dump_tensor]", output)
        self.assertIn("saved q", output)

    @patch.dict(
        os.environ,
        {"ABLATION_INSPECT_TENSOR": "1", "ABLATION_DUMP_SKIP_TAGS": "q"},
    )
    def test_save_skipped_by_dump_skip_tags(self):
        """Tags in ABLATION_DUMP_SKIP_TAGS should not be saved."""
        t = self._make_tensor()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.dict(os.environ, {"ABLATION_SAVE_TENSOR_PATH": tmpdir}),
                patch("builtins.print"),
            ):
                inspect_tensor("q", 0, t, save=True, load=False)
            saved_path = os.path.join(tmpdir, "rank_0", "layer_0", "q.npy")
            self.assertFalse(os.path.exists(saved_path))

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_save_not_triggered_when_no_path(self):
        """save=True but no ABLATION_SAVE_TENSOR_PATH should not crash."""
        t = self._make_tensor()
        with patch("builtins.print"):
            result = inspect_tensor("q", 0, t, save=True, load=False)
        self.assertIs(result, t)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_load_tensor_from_rank_layer_dir(self):
        """load=True should load from rank_0/layer_X/{tag}.npy."""
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_layer_dir = os.path.join(tmpdir, "rank_0", "layer_0")
            os.makedirs(rank_layer_dir)
            np.save(os.path.join(rank_layer_dir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                with patch("builtins.print"):
                    result = inspect_tensor("q", 0, t, save=False, load=True)
        np.testing.assert_allclose(result.numpy(), arr, rtol=1e-5)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_load_nonzero_layer(self):
        """load should work for any layer, not just layer 0."""
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_layer_dir = os.path.join(tmpdir, "rank_0", "layer_3")
            os.makedirs(rank_layer_dir)
            np.save(os.path.join(rank_layer_dir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                with patch("builtins.print"):
                    result = inspect_tensor("q", 3, t, save=False, load=True)
        np.testing.assert_allclose(result.numpy(), arr, rtol=1e-5)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_load_returns_original_when_file_missing(self):
        """If .npy file does not exist, should return original tensor."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}),
        ):
            t = paddle.zeros([2, 4])
            with patch("builtins.print"):
                result = inspect_tensor("q", 0, t, save=False, load=True)
        self.assertIs(result, t)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_load_auto_reshape(self):
        """Loaded array with different shape should be reshaped to match tensor."""
        arr = np.arange(8, dtype=np.float32)  # shape (8,)
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_layer_dir = os.path.join(tmpdir, "rank_0", "layer_0")
            os.makedirs(rank_layer_dir)
            np.save(os.path.join(rank_layer_dir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                with patch("builtins.print"):
                    result = inspect_tensor("q", 0, t, save=False, load=True)
        self.assertEqual(list(result.shape), [2, 4])
        np.testing.assert_allclose(result.numpy(), arr.reshape(2, 4), rtol=1e-5)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_load_dtype_cast(self):
        """Loaded tensor should be cast to target tensor's dtype."""
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_layer_dir = os.path.join(tmpdir, "rank_0", "layer_0")
            os.makedirs(rank_layer_dir)
            np.save(os.path.join(rank_layer_dir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4], dtype="float16")
                with patch("builtins.print"):
                    result = inspect_tensor("q", 0, t, save=False, load=True)
        self.assertEqual(result.dtype, paddle.float16)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_load_prints_diff_info(self):
        """Load should print [ABLATION_load_tensor] with diff stats."""
        arr = np.ones((2, 4), dtype=np.float32) * 3.0
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_layer_dir = os.path.join(tmpdir, "rank_0", "layer_0")
            os.makedirs(rank_layer_dir)
            np.save(os.path.join(rank_layer_dir, "v.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                with patch("builtins.print") as mock_print:
                    inspect_tensor("v", 0, t, save=False, load=True)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("[ABLATION_load_tensor]", output)
        self.assertIn("max_abs_diff", output)
        self.assertIn("mean_abs_diff", output)
        self.assertIn("relative_diff", output)

    @patch.dict(
        os.environ,
        {"ABLATION_INSPECT_TENSOR": "1", "ABLATION_DUMP_SKIP_TAGS": "q"},
    )
    def test_load_skipped_by_dump_skip_tags(self):
        """Tags in ABLATION_DUMP_SKIP_TAGS should not be loaded."""
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_layer_dir = os.path.join(tmpdir, "rank_0", "layer_0")
            os.makedirs(rank_layer_dir)
            np.save(os.path.join(rank_layer_dir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                with patch("builtins.print"):
                    result = inspect_tensor("q", 0, t, save=False, load=True)
        self.assertIs(result, t)

    @patch.dict(os.environ, {"ABLATION_INSPECT_TENSOR": "1"})
    def test_returns_original_when_load_false(self):
        """load=False should not load even if file exists."""
        arr = np.random.randn(2, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_layer_dir = os.path.join(tmpdir, "rank_0", "layer_0")
            os.makedirs(rank_layer_dir)
            np.save(os.path.join(rank_layer_dir, "q.npy"), arr)
            with patch.dict(os.environ, {"ABLATION_LOAD_TENSOR_PATH": tmpdir}):
                t = paddle.zeros([2, 4])
                with patch("builtins.print"):
                    result = inspect_tensor("q", 0, t, save=False, load=False)
        self.assertIs(result, t)


class TestGetDocLens(unittest.TestCase):
    """Tests for get_doc_lens."""

    def test_single_doc(self):
        # single doc of length 4: all positions point to end=4
        indices = paddle.to_tensor([4, 4, 4, 4], dtype="int32").reshape(
            [1, 1, 4, 1]
        )
        doc_lens = get_doc_lens(indices)
        self.assertEqual(doc_lens.numpy().tolist(), [4])

    def test_two_docs(self):
        # doc1 length 2 (ends at 2), doc2 length 2 (ends at 4)
        indices = paddle.to_tensor([2, 2, 4, 4], dtype="int32").reshape(
            [1, 1, 4, 1]
        )
        doc_lens = get_doc_lens(indices)
        self.assertEqual(doc_lens.numpy().tolist(), [2, 2])

    def test_three_docs(self):
        # doc1=1, doc2=2, doc3=3
        indices = paddle.to_tensor([1, 3, 3, 6, 6, 6], dtype="int32").reshape(
            [1, 1, 6, 1]
        )
        doc_lens = get_doc_lens(indices)
        self.assertEqual(doc_lens.numpy().tolist(), [1, 2, 3])


class TestGetDocStarts(unittest.TestCase):
    """Tests for get_doc_starts."""

    def test_single_doc(self):
        doc_lens = paddle.to_tensor([5], dtype="int32")
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0])

    def test_multiple_docs(self):
        doc_lens = paddle.to_tensor([2, 3, 4], dtype="int32")
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0, 2, 5])

    def test_single_token_docs(self):
        doc_lens = paddle.to_tensor([1, 1, 1], dtype="int32")
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0, 1, 2])
