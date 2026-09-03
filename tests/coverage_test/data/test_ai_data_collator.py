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

import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle

from paddleformers.data.data_collator import (
    DataCollatorMixin,
    DefaultDataCollator,
    _numpy_collate_batch,
    _paddle_collate_batch,
    default_data_collator,
    numpy_default_data_collator,
    paddle_default_data_collator,
    tolist,
)


class TestDataCollatorMixin(unittest.TestCase):
    """Tests for DataCollatorMixin class."""

    def test_import(self):
        """Test that DataCollatorMixin can be imported."""
        self.assertIsNotNone(DataCollatorMixin)

    def test_call_with_pd_tensors(self):
        """Test __call__ with return_tensors='pd' dispatches to paddle_call."""

        class DummyCollator(DataCollatorMixin):
            return_tensors = "pd"

            def paddle_call(self, features):
                return {"called": "paddle"}

        collator = DummyCollator()
        result = collator([{"a": 1}], return_tensors="pd")
        self.assertEqual(result["called"], "paddle")

    def test_call_with_np_tensors(self):
        """Test __call__ with return_tensors='np' dispatches to numpy_call."""

        class DummyCollator(DataCollatorMixin):
            return_tensors = "np"

            def numpy_call(self, features):
                return {"called": "numpy"}

        collator = DummyCollator()
        result = collator([{"a": 1}], return_tensors="np")
        self.assertEqual(result["called"], "numpy")

    def test_call_default_return_tensors(self):
        """Test __call__ uses self.return_tensors when return_tensors is None."""

        class DummyCollator(DataCollatorMixin):
            return_tensors = "pd"

            def paddle_call(self, features):
                return {"called": "paddle_default"}

        collator = DummyCollator()
        result = collator([{"a": 1}])
        self.assertEqual(result["called"], "paddle_default")

    def test_call_invalid_framework_raises(self):
        """Test __call__ raises ValueError for unrecognized framework."""

        class DummyCollator(DataCollatorMixin):
            return_tensors = "pd"

            def paddle_call(self, features):
                return {}

        collator = DummyCollator()
        with self.assertRaises(ValueError):
            collator([{"a": 1}], return_tensors="invalid")


class TestDefaultDataCollator(unittest.TestCase):
    """Tests for DefaultDataCollator class."""

    def test_import(self):
        """Test that DefaultDataCollator can be imported."""
        self.assertIsNotNone(DefaultDataCollator)

    def test_call_with_dict_features(self):
        """Test that DefaultDataCollator collates dict features."""
        collator = DefaultDataCollator()
        features = [{"input_ids": [1, 2, 3], "label": 0}, {"input_ids": [4, 5, 6], "label": 1}]
        batch = collator(features)
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertEqual(batch["input_ids"].shape, [2, 3])
        self.assertEqual(batch["labels"].shape, [2])

    def test_call_with_int_labels(self):
        """Test that int labels produce int64 dtype."""
        collator = DefaultDataCollator()
        features = [{"input_ids": [1, 2], "label": 0}, {"input_ids": [3, 4], "label": 1}]
        batch = collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.int64)

    def test_call_with_float_labels(self):
        """Test that float labels produce float32 dtype."""
        collator = DefaultDataCollator()
        features = [{"input_ids": [1, 2], "label": 0.5}, {"input_ids": [3, 4], "label": 1.5}]
        batch = collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.float32)

    def test_call_with_label_ids(self):
        """Test that label_ids are collated correctly."""
        collator = DefaultDataCollator()
        features = [
            {"input_ids": [1, 2], "label_ids": [0, 1]},
            {"input_ids": [3, 4], "label_ids": [1, 0]},
        ]
        batch = collator(features)
        self.assertIn("labels", batch)

    def test_default_return_tensors_pd(self):
        """Test default return_tensors is 'pd'."""
        collator = DefaultDataCollator()
        self.assertEqual(collator.return_tensors, "pd")

    def test_custom_return_tensors(self):
        """Test custom return_tensors setting."""
        collator = DefaultDataCollator(return_tensors="np")
        self.assertEqual(collator.return_tensors, "np")


class TestDefaultDataCollatorFunction(unittest.TestCase):
    """Tests for default_data_collator function."""

    def test_pd_return(self):
        """Test default_data_collator with pd return."""
        features = [{"input_ids": [1, 2, 3], "label": 0}]
        batch = default_data_collator(features, return_tensors="pd")
        self.assertIn("input_ids", batch)

    def test_np_return(self):
        """Test default_data_collator with np return."""
        features = [{"input_ids": [1, 2, 3], "label": 0}]
        batch = default_data_collator(features, return_tensors="np")
        self.assertIn("input_ids", batch)


class TestPaddleDefaultDataCollator(unittest.TestCase):
    """Tests for paddle_default_data_collator function."""

    def test_basic_dict(self):
        """Test collating basic dict features."""
        features = [{"a": [1, 2], "b": [3, 4]}, {"a": [5, 6], "b": [7, 8]}]
        batch = paddle_default_data_collator(features)
        self.assertIn("a", batch)
        self.assertIn("b", batch)

    def test_with_int_label(self):
        """Test collating features with int label."""
        features = [{"input_ids": [1, 2], "label": 0}, {"input_ids": [3, 4], "label": 1}]
        batch = paddle_default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.int64)

    def test_with_float_label(self):
        """Test collating features with float label."""
        features = [{"input_ids": [1, 2], "label": 0.5}, {"input_ids": [3, 4], "label": 1.5}]
        batch = paddle_default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.float32)

    def test_with_tensor_label(self):
        """Test collating features with paddle.Tensor label."""
        features = [
            {"input_ids": [1, 2], "label": paddle.to_tensor(0)},
            {"input_ids": [3, 4], "label": paddle.to_tensor(1)},
        ]
        batch = paddle_default_data_collator(features)
        self.assertIn("labels", batch)

    def test_string_values_excluded(self):
        """Test that string values are excluded from batch."""
        features = [{"a": [1, 2], "text": "hello"}, {"a": [3, 4], "text": "world"}]
        batch = paddle_default_data_collator(features)
        self.assertNotIn("text", batch)
        self.assertIn("a", batch)

    def test_tensor_values_stacked(self):
        """Test that Tensor values are stacked."""
        features = [
            {"input_ids": paddle.to_tensor([1, 2])},
            {"input_ids": paddle.to_tensor([3, 4])},
        ]
        batch = paddle_default_data_collator(features)
        self.assertEqual(batch["input_ids"].shape, [2, 2])


class TestNumpyDefaultDataCollator(unittest.TestCase):
    """Tests for numpy_default_data_collator function."""

    def test_basic_dict(self):
        """Test collating basic dict features with numpy."""
        features = [{"a": [1, 2], "b": [3, 4]}, {"a": [5, 6], "b": [7, 8]}]
        batch = numpy_default_data_collator(features)
        self.assertIn("a", batch)
        self.assertIn("b", batch)
        self.assertIsInstance(batch["a"], np.ndarray)

    def test_with_int_label(self):
        """Test collating features with int label using numpy."""
        features = [{"input_ids": [1, 2], "label": 0}, {"input_ids": [3, 4], "label": 1}]
        batch = numpy_default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, np.int64)

    def test_with_float_label(self):
        """Test collating features with float label using numpy."""
        features = [{"input_ids": [1, 2], "label": 0.5}, {"input_ids": [3, 4], "label": 1.5}]
        batch = numpy_default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, np.float32)


class TestTolist(unittest.TestCase):
    """Tests for tolist helper function."""

    def test_list_input(self):
        """Test that list input is returned as-is."""
        result = tolist([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_paddle_tensor_input(self):
        """Test that paddle.Tensor is converted to list."""
        tensor = paddle.to_tensor([1, 2, 3])
        result = tolist(tensor)
        self.assertEqual(result, [1, 2, 3])

    def test_numpy_array_input(self):
        """Test that numpy array with .numpy() attribute is handled."""
        arr = np.array([4, 5, 6])
        # numpy arrays don't have .numpy() method, but do have .tolist()
        # tolist checks hasattr(x, "numpy") which is False for ndarray
        result = tolist(arr)
        self.assertEqual(result, [4, 5, 6])


class TestPaddleCollateBatch(unittest.TestCase):
    """Tests for _paddle_collate_batch function."""

    def test_same_length_stacked(self):
        """Test that same-length examples are stacked."""
        examples = [paddle.to_tensor([1, 2, 3]), paddle.to_tensor([4, 5, 6])]
        mock_tokenizer = MagicMock()
        result = _paddle_collate_batch(examples, mock_tokenizer)
        self.assertEqual(result.shape, [2, 3])

    def test_list_inputs_converted(self):
        """Test that list inputs are converted to tensors."""
        examples = [[1, 2, 3], [4, 5, 6]]
        mock_tokenizer = MagicMock()
        result = _paddle_collate_batch(examples, mock_tokenizer)
        self.assertEqual(result.shape, [2, 3])

    def test_different_lengths_padded(self):
        """Test that different-length examples are padded."""
        examples = [paddle.to_tensor([1, 2, 3]), paddle.to_tensor([4, 5])]
        mock_tokenizer = MagicMock()
        mock_tokenizer._pad_token = MagicMock()  # Not None, so padding check passes
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.padding_side = "right"
        result = _paddle_collate_batch(examples, mock_tokenizer)
        self.assertEqual(result.shape, [2, 3])

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of specified value."""
        examples = [paddle.to_tensor([1, 2, 3]), paddle.to_tensor([4, 5, 6])]
        mock_tokenizer = MagicMock()
        mock_tokenizer._pad_token = MagicMock()  # Not None
        mock_tokenizer.pad_token_id = 0  # Must be int, not MagicMock (used as fill_value)
        mock_tokenizer.padding_side = "right"
        result = _paddle_collate_batch(examples, mock_tokenizer, pad_to_multiple_of=4)
        self.assertEqual(result.shape[1], 4)


class TestNumpyCollateBatch(unittest.TestCase):
    """Tests for _numpy_collate_batch function."""

    def test_same_length_stacked(self):
        """Test that same-length examples are stacked."""
        examples = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        mock_tokenizer = MagicMock()
        result = _numpy_collate_batch(examples, mock_tokenizer)
        self.assertEqual(result.shape, (2, 3))

    def test_list_inputs_converted(self):
        """Test that list inputs are converted to numpy arrays."""
        examples = [[1, 2, 3], [4, 5, 6]]
        mock_tokenizer = MagicMock()
        result = _numpy_collate_batch(examples, mock_tokenizer)
        self.assertEqual(result.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
