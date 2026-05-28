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

import numpy as np
import paddle

from paddleformers.datasets.rlhf_datasets.protocol import (
    DataProto,
    TensorDict,
    list_of_dict_to_dict_of_list,
    pad_dataproto_to_divisor,
    union_numpy_dict,
    union_tensor_dict,
    union_two_dict,
    unpad_dataproto,
)


class TestTensorDict(unittest.TestCase):
    """Tests for TensorDict."""

    def test_set_and_get(self):
        td = TensorDict(source={}, batch_size=None)
        tensor = paddle.randn([2, 3])
        td["x"] = tensor
        self.assertTrue(
            paddle.equal(td["x"], tensor).item()
            if td["x"].numel() == 1
            else paddle.equal(td["x"], tensor).all().item()
        )

    def test_keys_and_items(self):
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 4])
        td = TensorDict(source={"a": t1, "b": t2}, batch_size=(2,))
        self.assertIn("a", td.keys())
        self.assertIn("b", td.keys())
        self.assertEqual(len(list(td.items())), 2)

    def test_batch_size_mismatch(self):
        t1 = paddle.randn([2, 3])
        with self.assertRaises(AssertionError):
            TensorDict(source={"a": t1}, batch_size=(5,))

    def test_to(self):
        td = TensorDict(source={"a": paddle.randn([2, 3])}, batch_size=(2,))
        result = td.to("cpu")
        self.assertIsInstance(result, TensorDict)


class TestUnionTwoDict(unittest.TestCase):
    """Tests for union_two_dict."""

    def test_basic_union(self):
        d1 = {"a": 1}
        d2 = {"b": 2}
        result = union_two_dict(d1, d2)
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_conflict_raises(self):
        d1 = {"a": 1}
        d2 = {"a": 2}
        with self.assertRaises(AssertionError):
            union_two_dict(d1, d2)

    def test_same_object_no_error(self):
        shared = [1, 2, 3]
        d1 = {"a": shared}
        d2 = {"a": shared}
        result = union_two_dict(d1, d2)
        self.assertEqual(result, {"a": shared})


class TestListODictToDictOfList(unittest.TestCase):
    """Tests for list_of_dict_to_dict_of_list."""

    def test_basic(self):
        list_of_dict = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = list_of_dict_to_dict_of_list(list_of_dict)
        self.assertEqual(result, {"a": [1, 3], "b": [2, 4]})

    def test_empty_list(self):
        result = list_of_dict_to_dict_of_list([])
        self.assertEqual(result, {})


class TestUnionTensorDict(unittest.TestCase):
    """Tests for union_tensor_dict."""

    def test_basic_union(self):
        t1 = TensorDict(source={"a": paddle.randn([2, 3])}, batch_size=(2,))
        t2 = TensorDict(source={"b": paddle.randn([2, 3])}, batch_size=(2,))
        result = union_tensor_dict(t1, t2)
        self.assertIn("a", result.keys())
        self.assertIn("b", result.keys())

    def test_batch_size_mismatch(self):
        t1 = TensorDict(source={"a": paddle.randn([2, 3])}, batch_size=(2,))
        t2 = TensorDict(source={"b": paddle.randn([3, 3])}, batch_size=(3,))
        with self.assertRaises(AssertionError):
            union_tensor_dict(t1, t2)

    def test_no_conflict_keys(self):
        # No conflict keys should work fine
        t1 = TensorDict(source={"a": paddle.randn([2, 3])}, batch_size=(2,))
        t2 = TensorDict(source={"b": paddle.randn([2, 3])}, batch_size=(2,))
        result = union_tensor_dict(t1, t2)
        self.assertIn("a", result.keys())
        self.assertIn("b", result.keys())

    def test_conflict_keys_raises(self):
        # Different tensors with same key should raise due to .equal() check
        t1 = TensorDict(source={"a": paddle.randn([2, 3])}, batch_size=(2,))
        t2 = TensorDict(source={"a": paddle.randn([2, 3])}, batch_size=(2,))
        with self.assertRaises((AssertionError, Exception)):
            union_tensor_dict(t1, t2)


class TestUnionNumpyDict(unittest.TestCase):
    """Tests for union_numpy_dict."""

    def test_basic_union(self):
        d1 = {"a": np.array([1, 2])}
        d2 = {"b": np.array([3, 4])}
        result = union_numpy_dict(d1, d2)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_conflict_raises(self):
        d1 = {"a": np.array([1, 2])}
        d2 = {"a": np.array([3, 4])}
        with self.assertRaises(AssertionError):
            union_numpy_dict(d1, d2)


class TestDataProto(unittest.TestCase):
    """Tests for DataProto."""

    def _make_data_proto(self, batch_size=4):
        tensors = {"input_ids": paddle.randint(0, 100, [batch_size, 10])}
        non_tensors = {"labels": np.array(["a", "b", "c", "d"][:batch_size], dtype=object)}
        return DataProto(batch=TensorDict(source=tensors, batch_size=(batch_size,)), non_tensor_batch=non_tensors)

    def test_len(self):
        dp = self._make_data_proto(batch_size=4)
        self.assertEqual(len(dp), 4)

    def test_from_single_dict(self):
        data = {
            "input_ids": paddle.randint(0, 100, [4, 10]),
            "labels": np.array(["a", "b", "c", "d"], dtype=object),
        }
        dp = DataProto.from_single_dict(data)
        self.assertEqual(len(dp), 4)

    def test_from_single_dict_invalid_type(self):
        data = {"input_ids": "invalid"}
        with self.assertRaises(ValueError):
            DataProto.from_single_dict(data)

    def test_from_dict(self):
        dp = DataProto.from_dict(tensors={"x": paddle.randn([4, 3])}, non_tensors={}, meta_info={})
        self.assertEqual(len(dp), 4)

    def test_from_dict_non_tensors(self):
        non_tensors = {"labels": np.array(["a", "b", "c", "d"], dtype=object)}
        dp = DataProto.from_dict(tensors={"x": paddle.randn([4, 3])}, non_tensors=non_tensors, meta_info={})
        self.assertEqual(len(dp), 4)

    def test_repeat_interleave(self):
        dp = self._make_data_proto(batch_size=2)
        result = dp.repeat(repeat_times=3, interleave=True)
        self.assertEqual(len(result), 6)

    def test_repeat_no_interleave(self):
        dp = self._make_data_proto(batch_size=2)
        result = dp.repeat(repeat_times=2, interleave=False)
        self.assertEqual(len(result), 4)

    def test_check_consistency(self):
        dp = self._make_data_proto()
        dp.check_consistency()

    def test_check_consistency_bad_non_tensor(self):
        # non_tensor_batch values must be np.ndarray with dtype=object
        tensors = {"input_ids": paddle.randint(0, 100, [2, 10])}
        non_tensors = {"labels": [1, 2]}  # not an ndarray
        with self.assertRaises(AssertionError):
            DataProto(
                batch=TensorDict(source=tensors, batch_size=(2,)),
                non_tensor_batch=non_tensors,
            )

    def test_to_device(self):
        dp = self._make_data_proto()
        result = dp.to("cpu")
        self.assertIs(result, dp)

    def test_union(self):
        dp1 = DataProto.from_dict(tensors={"x": paddle.randn([2, 3])}, non_tensors={}, meta_info={"m1": 1})
        dp2 = DataProto.from_dict(tensors={"y": paddle.randn([2, 3])}, non_tensors={}, meta_info={"m2": 2})
        dp1.union(dp2)
        self.assertIn("x", dp1.batch.keys())
        self.assertIn("y", dp1.batch.keys())
        self.assertIn("m1", dp1.meta_info)
        self.assertIn("m2", dp1.meta_info)


class TestPadDataProtoToDivisor(unittest.TestCase):
    """Tests for pad_dataproto_to_divisor."""

    def test_already_divisible(self):
        """Test when batch size is already divisible."""
        dp = DataProto.from_dict(tensors={"x": paddle.randn([4, 3])}, non_tensors={}, meta_info={})
        # When already divisible, no slicing is needed
        result, pad_size = pad_dataproto_to_divisor(dp, 2)
        self.assertEqual(pad_size, 0)
        self.assertEqual(len(result), 4)


class TestUnpadDataProto(unittest.TestCase):
    """Tests for unpad_dataproto."""

    def test_no_padding(self):
        """Test with pad_size=0 returns same object."""
        dp = DataProto.from_dict(tensors={"x": paddle.randn([4, 3])}, non_tensors={}, meta_info={})
        result = unpad_dataproto(dp, pad_size=0)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
