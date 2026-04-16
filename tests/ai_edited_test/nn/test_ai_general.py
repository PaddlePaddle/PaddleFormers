# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest


class TestGeneralInterface(unittest.TestCase):
    """Tests for paddleformers.nn.general.GeneralInterface."""

    def setUp(self):
        from paddleformers.nn.general import GeneralInterface

        # Use a fresh subclass to avoid cross-test contamination of _global_mapping
        class TestInterface(GeneralInterface):
            _global_mapping = {}

        self.cls = TestInterface

    def tearDown(self):
        # Clean up class-level global_mapping to avoid polluting other tests
        self.cls._global_mapping = {}

    def test_getitem_local_override(self):
        """Local mapping should take precedence over global mapping."""
        obj = self.cls()
        self.cls.register("key_a", lambda: "global_value")
        obj["key_a"] = lambda: "local_value"
        self.assertEqual(obj["key_a"](), "local_value")

    def test_getitem_global_fallback(self):
        """If no local override exists, fall back to global mapping."""
        obj = self.cls()
        self.cls.register("key_b", lambda: "global_value")
        self.assertEqual(obj["key_b"](), "global_value")

    def test_getitem_key_error(self):
        """Accessing a non-existent key should raise KeyError."""
        obj = self.cls()
        with self.assertRaises(KeyError):
            _ = obj["nonexistent"]

    def test_setitem(self):
        """__setitem__ should store value in local mapping."""
        obj = self.cls()
        obj["my_key"] = 42
        self.assertEqual(obj["my_key"], 42)

    def test_delitem(self):
        """__delitem__ should remove key from local mapping."""
        obj = self.cls()
        obj["my_key"] = 42
        del obj["my_key"]
        # After deletion, accessing should check global (which doesn't have it)
        with self.assertRaises(KeyError):
            _ = obj["my_key"]

    def test_delitem_missing(self):
        """Deleting a non-existent key should raise KeyError."""
        obj = self.cls()
        with self.assertRaises(KeyError):
            del obj["nonexistent"]

    def test_iter_combined_keys(self):
        """__iter__ should yield keys from both global and local mappings."""
        obj = self.cls()
        self.cls.register("g1", None)
        self.cls.register("g2", None)
        obj["l1"] = None
        keys = set(iter(obj))
        self.assertIn("g1", keys)
        self.assertIn("g2", keys)
        self.assertIn("l1", keys)

    def test_len_combined(self):
        """__len__ should count unique keys from both mappings."""
        obj = self.cls()
        self.cls.register("g1", None)
        obj["l1"] = None
        self.assertEqual(len(obj), 2)

    def test_len_with_overlap(self):
        """__len__ should not double-count overlapping keys."""
        obj = self.cls()
        self.cls.register("shared", None)
        obj["shared"] = None
        self.assertEqual(len(obj), 1)

    def test_register_classmethod(self):
        """register should update the class-level global_mapping."""
        self.cls.register("fn_key", lambda x: x * 2)
        self.assertIn("fn_key", self.cls._global_mapping)

    def test_valid_keys(self):
        """valid_keys should return a list of all available keys."""
        obj = self.cls()
        self.cls.register("k1", None)
        self.cls.register("k2", None)
        obj["k3"] = None
        vk = obj.valid_keys()
        self.assertIsInstance(vk, list)
        self.assertIn("k1", vk)
        self.assertIn("k2", vk)
        self.assertIn("k3", vk)

    def test_multiple_instances_share_global(self):
        """Multiple instances should share the global mapping."""
        self.cls.register("shared_fn", lambda: "hello")
        obj1 = self.cls()
        obj2 = self.cls()
        self.assertEqual(obj1["shared_fn"](), "hello")
        self.assertEqual(obj2["shared_fn"](), "hello")

    def test_local_does_not_affect_global(self):
        """Setting a local key should not affect global mapping or other instances."""
        self.cls.register("shared_fn", lambda: "global")
        obj1 = self.cls()
        obj2 = self.cls()
        obj1["shared_fn"] = lambda: "local"
        self.assertEqual(obj1["shared_fn"](), "local")
        self.assertEqual(obj2["shared_fn"](), "global")

    def test_empty_mappings(self):
        """Empty mappings should yield empty iteration and zero length."""
        obj = self.cls()
        self.assertEqual(len(obj), 0)
        self.assertEqual(list(iter(obj)), [])
        self.assertEqual(obj.valid_keys(), [])
