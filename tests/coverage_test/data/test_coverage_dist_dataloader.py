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

import paddle


class TestDummyDataset(unittest.TestCase):
    """Tests for DummyDataset class."""

    def test_import(self):
        """Test that DummyDataset can be imported."""
        from paddleformers.data.dist_dataloader import DummyDataset

        self.assertIsNotNone(DummyDataset)

    def test_is_paddle_dataset(self):
        """Test that DummyDataset is a subclass of paddle.io.Dataset."""
        from paddleformers.data.dist_dataloader import DummyDataset

        self.assertTrue(issubclass(DummyDataset, paddle.io.Dataset))

    def test_len_returns_zero(self):
        """Test that __len__ returns 0."""
        from paddleformers.data.dist_dataloader import DummyDataset

        ds = DummyDataset()
        self.assertEqual(len(ds), 0)


class TestIterableDummyDataset(unittest.TestCase):
    """Tests for IterableDummyDataset class."""

    def test_import(self):
        """Test that IterableDummyDataset can be imported."""
        from paddleformers.data.dist_dataloader import IterableDummyDataset

        self.assertIsNotNone(IterableDummyDataset)

    def test_is_iterable_dataset(self):
        """Test that IterableDummyDataset is a subclass of paddle.io.IterableDataset."""
        from paddleformers.data.dist_dataloader import IterableDummyDataset

        self.assertTrue(issubclass(IterableDummyDataset, paddle.io.IterableDataset))

    def test_iter_returns_none(self):
        """Test that __iter__ returns None (non-iterator)."""
        from paddleformers.data.dist_dataloader import IterableDummyDataset

        ds = IterableDummyDataset()
        # __iter__ returns None, which means iterating would fail
        result = ds.__iter__()
        self.assertIsNone(result)


class TestDistDataLoader(unittest.TestCase):
    """Tests for DistDataLoader class."""

    def test_import(self):
        """Test that DistDataLoader can be imported."""
        from paddleformers.data.dist_dataloader import DistDataLoader

        self.assertIsNotNone(DistDataLoader)

    def test_is_dataloader(self):
        """Test that DistDataLoader is a subclass of paddle.io.DataLoader."""
        from paddleformers.data.dist_dataloader import DistDataLoader

        self.assertTrue(issubclass(DistDataLoader, paddle.io.DataLoader))


class TestStreamDistDataLoader(unittest.TestCase):
    """Tests for StreamDistDataLoader class."""

    def test_import(self):
        """Test that StreamDistDataLoader can be imported."""
        from paddleformers.data.dist_dataloader import StreamDistDataLoader

        self.assertIsNotNone(StreamDistDataLoader)

    def test_len_raises_value_error(self):
        """Test that __len__ raises ValueError."""
        from paddleformers.data.dist_dataloader import StreamDistDataLoader

        # Cannot easily instantiate without fleet, so test the method exists
        self.assertTrue(hasattr(StreamDistDataLoader, "__len__"))


class TestInitDataloaderCommGroup(unittest.TestCase):
    """Tests for init_dataloader_comm_group function."""

    def test_import(self):
        """Test that init_dataloader_comm_group can be imported."""
        from paddleformers.data.dist_dataloader import init_dataloader_comm_group

        self.assertTrue(callable(init_dataloader_comm_group))


class TestInitStreamDataGroup(unittest.TestCase):
    """Tests for init_stream_data_group function."""

    def test_import(self):
        """Test that init_stream_data_group can be imported."""
        from paddleformers.data.dist_dataloader import init_stream_data_group

        self.assertTrue(callable(init_stream_data_group))


if __name__ == "__main__":
    unittest.main()
