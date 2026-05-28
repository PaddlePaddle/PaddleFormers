# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest


class TestSerialization(unittest.TestCase):
    """Tests for utils/serialization.py"""

    def test_SerializationError(self):
        from paddleformers.utils.serialization import SerializationError

        with self.assertRaises(SerializationError):
            raise SerializationError("test error")

    def test_maybe_decode_ascii_bytes(self):
        from paddleformers.utils.serialization import _maybe_decode_ascii

        result = _maybe_decode_ascii(b"hello")
        self.assertEqual(result, "hello")

    def test_maybe_decode_ascii_str(self):
        from paddleformers.utils.serialization import _maybe_decode_ascii

        result = _maybe_decode_ascii("hello")
        self.assertEqual(result, "hello")

    def test_storage_type_to_dtype_to_map(self):
        from paddleformers.utils.serialization import _storage_type_to_dtype_to_map

        mapping = _storage_type_to_dtype_to_map()
        self.assertEqual(mapping["FloatStorage"].__name__, "float32")
        self.assertEqual(mapping["LongStorage"].__name__, "int64")
        self.assertEqual(mapping["HalfStorage"].__name__, "float16")
        self.assertEqual(mapping["BoolStorage"].__name__, "bool_")
        self.assertEqual(mapping["BFloat16Storage"].__name__, "uint16")

    def test_StorageType(self):
        from paddleformers.utils.serialization import StorageType

        st = StorageType("FloatStorage")
        self.assertEqual(st.dtype, __import__("numpy").float32)
        self.assertIn("float32", str(st))

    def test_element_size_float(self):
        import numpy as np

        from paddleformers.utils.serialization import _element_size

        self.assertEqual(_element_size(np.float32), 4)
        self.assertEqual(_element_size(np.float64), 8)
        self.assertEqual(_element_size(np.float16), 2)

    def test_element_size_int(self):
        import numpy as np

        from paddleformers.utils.serialization import _element_size

        self.assertEqual(_element_size(np.int32), 4)
        self.assertEqual(_element_size(np.int64), 8)
        self.assertEqual(_element_size(np.int8), 1)

    def test_element_size_bool(self):
        import numpy as np

        from paddleformers.utils.serialization import _element_size

        self.assertEqual(_element_size(np.bool_), 1)

    def test_rebuild_tensor_stage_c_order(self):
        import numpy as np

        from paddleformers.utils.serialization import _rebuild_tensor_stage

        storage = np.arange(6, dtype="float32")
        result = _rebuild_tensor_stage(storage, 0, [2, 3], [3, 1], False, [])
        self.assertEqual(result.shape, (2, 3))

    def test_rebuild_tensor_stage_f_order(self):
        import numpy as np

        from paddleformers.utils.serialization import _rebuild_tensor_stage

        storage = np.arange(6, dtype="float32")
        result = _rebuild_tensor_stage(storage, 0, [2, 3], [1, 2], False, [])
        self.assertEqual(result.shape, (2, 3))

    def test_rebuild_parameter(self):
        import numpy as np

        from paddleformers.utils.serialization import _rebuild_parameter

        data = np.array([1.0, 2.0, 3.0])
        result = _rebuild_parameter(data, False, [])
        np.testing.assert_array_equal(result, data)

    def test_rebuild_parameter_with_state(self):
        import numpy as np

        from paddleformers.utils.serialization import _rebuild_parameter_with_state

        data = np.array([1.0, 2.0, 3.0])
        result = _rebuild_parameter_with_state(data, False, [], None)
        np.testing.assert_array_equal(result, data)

    def test_dumpy(self):
        from paddleformers.utils.serialization import dumpy

        result = dumpy(1, 2, 3)
        self.assertIsNone(result)

    def test_SafeUnpickler_allowed(self):
        import io
        import pickle

        from paddleformers.utils.serialization import SafeUnpickler

        data = {"key": [1, 2, 3]}
        serialized = pickle.dumps(data)
        unpickler = SafeUnpickler(io.BytesIO(serialized))
        result = unpickler.load()
        self.assertEqual(result, data)

    def test_SafeUnpickler_blocked(self):
        import io
        import pickle

        from paddleformers.utils.serialization import SafeUnpickler

        # Use a module-level class (local classes can't be pickled)
        serialized = pickle.dumps({"obj": object()})
        unpickler = SafeUnpickler(io.BytesIO(serialized))
        with self.assertRaises(pickle.UnpicklingError):
            unpickler.load()

    def test_Types_dict(self):
        from paddleformers.utils.serialization import _TYPES

        self.assertIn("F32", _TYPES)
        self.assertIn("F16", _TYPES)
        self.assertIn("I64", _TYPES)
        self.assertIn("BF16", _TYPES)
        self.assertIn("BOOL", _TYPES)
