# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import patch


class TestDistributed(unittest.TestCase):
    """Tests for utils/distributed.py"""

    def test_convert_file_size_to_int_int(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int(1024), 1024)

    def test_convert_file_size_to_int_gib(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int("1GiB"), 2**30)

    def test_convert_file_size_to_int_mib(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int("2MiB"), 2 * 2**20)

    def test_convert_file_size_to_int_kib(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int("1KiB"), 2**10)

    def test_convert_file_size_to_int_gb(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int("1GB"), 10**9)
        self.assertEqual(convert_file_size_to_int("1Gb"), 10**9 // 8)

    def test_convert_file_size_to_int_mb(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int("1MB"), 10**6)
        self.assertEqual(convert_file_size_to_int("1Mb"), 10**6 // 8)

    def test_convert_file_size_to_int_kb(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int("1KB"), 10**3)
        self.assertEqual(convert_file_size_to_int("1Kb"), 10**3 // 8)

    def test_convert_file_size_to_int_invalid(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        with self.assertRaises(ValueError):
            convert_file_size_to_int("invalid")

    def test_convert_file_size_to_int_large(self):
        from paddleformers.utils.distributed import convert_file_size_to_int

        self.assertEqual(convert_file_size_to_int("10GiB"), 10 * 2**30)

    def test_dtype_byte_size_bool(self):
        import paddle

        from paddleformers.utils.distributed import dtype_byte_size

        result = dtype_byte_size(paddle.bool)
        self.assertEqual(result, 1 / 8)

    def test_dtype_byte_size_float32(self):
        import paddle

        from paddleformers.utils.distributed import dtype_byte_size

        result = dtype_byte_size(paddle.float32)
        self.assertEqual(result, 4)

    def test_dtype_byte_size_float16(self):
        import paddle

        from paddleformers.utils.distributed import dtype_byte_size

        result = dtype_byte_size(paddle.float16)
        self.assertEqual(result, 2)

    def test_dtype_byte_size_int64(self):
        import paddle

        from paddleformers.utils.distributed import dtype_byte_size

        result = dtype_byte_size(paddle.int64)
        self.assertEqual(result, 8)

    def test_reduce_tensor(self):
        import paddle

        from paddleformers.utils.distributed import reduce_tensor

        x = paddle.randn([4, 8], dtype="float32")
        with patch("paddleformers.utils.distributed.convert_file_size_to_int", return_value=1024):
            parts = list(reduce_tensor(x, buffer_size="32MiB"))
        # Should yield parts based on buffer size
        self.assertTrue(len(parts) >= 1)

    def test_reduce_tensor_int8(self):
        import paddle

        from paddleformers.utils.distributed import reduce_tensor

        x = paddle.randint(0, 10, [4, 8], dtype="int32")
        with patch("paddleformers.utils.distributed.convert_file_size_to_int", return_value=1024):
            parts = list(reduce_tensor(x, buffer_size="32MiB"))
        self.assertTrue(len(parts) >= 1)

    def test_dtype_byte_size_invalid(self):
        with patch("paddleformers.utils.distributed.re") as mock_re:
            mock_re.search.return_value = None
            with self.assertRaises(ValueError):
                pass

                from paddleformers.utils.distributed import dtype_byte_size

                dtype_byte_size("not_a_real_dtype")
