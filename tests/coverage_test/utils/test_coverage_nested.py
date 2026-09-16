# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest


class TestNestedUtils(unittest.TestCase):
    """Tests for utils/nested.py"""

    def test_nested_reduce_tensor_scalar(self):
        from paddleformers.utils.nested import nested_reduce_tensor

        result = nested_reduce_tensor(42)
        self.assertEqual(result, 42)

    def test_nested_reduce_tensor_dict(self):
        import paddle

        from paddleformers.utils.nested import TensorHolder, nested_reduce_tensor

        t = paddle.randn([4, 8], dtype="float32")
        d = {"a": t}
        result = nested_reduce_tensor(d)
        self.assertIsInstance(result["a"], TensorHolder)
        self.assertEqual(result["a"].shape, [4, 8])

    def test_nested_reduce_tensor_list(self):
        import paddle

        from paddleformers.utils.nested import TensorHolder, nested_reduce_tensor

        t = paddle.randn([4, 8], dtype="float32")
        result = nested_reduce_tensor([t, 42])
        self.assertIsInstance(result[0], TensorHolder)
        self.assertEqual(result[1], 42)

    def test_nested_empty_tensor(self):
        import paddle

        from paddleformers.utils.nested import TensorHolder, nested_empty_tensor

        holder = TensorHolder(shape=[4, 8], dtype="float32", name="test")
        result = nested_empty_tensor(holder)
        self.assertIsInstance(result, paddle.Tensor)
        self.assertEqual(result.shape, [4, 8])

    def test_nested_copy_dict(self):
        from paddleformers.utils.nested import nested_copy

        d = {"a": 1, "b": [2, 3]}
        result = nested_copy(d)
        self.assertEqual(result, d)

    def test_nested_copy_non_dict(self):
        from paddleformers.utils.nested import nested_copy

        result = nested_copy(42)
        self.assertEqual(result, 42)

    def test_flatten_list(self):
        from paddleformers.utils.nested import flatten_list

        result = flatten_list([[1, 2], [3, [4, 5]], 6])
        self.assertEqual(result, [1, 2, 3, 4, 5, 6])

    def test_flatten_list_empty(self):
        from paddleformers.utils.nested import flatten_list

        result = flatten_list([])
        self.assertEqual(result, [])

    def test_flatten_list_flat(self):
        from paddleformers.utils.nested import flatten_list

        result = flatten_list([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_nested_copy_place_non_tensor(self):
        from paddleformers.utils.nested import nested_copy_place

        d = {"a": 1}
        result = nested_copy_place(d)
        self.assertEqual(result["a"], 1)

    def test_nested_copy_place_dict(self):
        from paddleformers.utils.nested import nested_copy_place

        d = {"a": 1, "b": 2}
        result = nested_copy_place(d, place=None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["a"], 1)
