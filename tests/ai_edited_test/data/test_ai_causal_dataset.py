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

from paddleformers.data.causal_dataset import (
    check_data_split,
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)


class TestCheckDataSplit(unittest.TestCase):
    """Tests for check_data_split function."""

    def test_valid_split_comma(self):
        """Test check_data_split with valid comma-separated splits."""
        # Should not raise
        check_data_split("0.8,0.1,0.1", do_train=True, do_eval=True, do_predict=True)

    def test_valid_split_slash(self):
        """Test check_data_split with valid slash-separated splits."""
        check_data_split("8/1/1", do_train=True, do_eval=True, do_predict=True)

    def test_single_split(self):
        """Test check_data_split with single value."""
        check_data_split("1.0", do_train=True, do_eval=False, do_predict=False)

    def test_zero_train_split_raises(self):
        """Test that zero train split with do_train=True raises ValueError."""
        with self.assertRaises(ValueError):
            check_data_split("0,0.5,0.5", do_train=True, do_eval=True, do_predict=True)

    def test_zero_eval_split_raises(self):
        """Test that zero eval split with do_eval=True raises ValueError."""
        with self.assertRaises(ValueError):
            check_data_split("0.5,0,0.5", do_train=True, do_eval=True, do_predict=True)

    def test_zero_predict_split_raises(self):
        """Test that zero predict split with do_predict=True raises ValueError."""
        with self.assertRaises(ValueError):
            check_data_split("0.5,0.5,0", do_train=True, do_eval=True, do_predict=True)

    def test_zero_sum_raises(self):
        """Test that sum of splits being 0 raises AssertionError."""
        with self.assertRaises(AssertionError):
            check_data_split("0,0,0", do_train=False, do_eval=False, do_predict=False)


class TestGetTrainValidTestSplit(unittest.TestCase):
    """Tests for get_train_valid_test_split_ function."""

    def test_basic_split(self):
        """Test basic data split."""
        result = get_train_valid_test_split_("0.8,0.1,0.1", 1000)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 1000)

    def test_two_way_split(self):
        """Test two-way split (train and eval only)."""
        result = get_train_valid_test_split_("0.9,0.1", 1000)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[-1], 1000)

    def test_single_split(self):
        """Test single value split."""
        result = get_train_valid_test_split_("1.0", 1000)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[-1], 1000)

    def test_slash_split(self):
        """Test slash-separated split string."""
        result = get_train_valid_test_split_("8/1/1", 1000)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[-1], 1000)

    def test_split_sums_to_size(self):
        """Test that split indices sum to the total size."""
        for size in [100, 500, 1000]:
            result = get_train_valid_test_split_("0.7,0.2,0.1", size)
            self.assertEqual(result[-1], size, f"Failed for size={size}")


class TestGetDatasetsWeightsAndNumSamples(unittest.TestCase):
    """Tests for get_datasets_weights_and_num_samples function."""

    def test_single_dataset(self):
        """Test with a single dataset prefix."""
        prefixes, weights, num_samples = get_datasets_weights_and_num_samples(
            ["1.0", "/path/to/data"], [1000, 100, 50]
        )
        self.assertEqual(len(prefixes), 1)
        self.assertEqual(len(weights), 1)
        self.assertEqual(len(num_samples), 1)
        self.assertAlmostEqual(weights[0], 1.0)

    def test_two_datasets(self):
        """Test with two dataset prefixes."""
        prefixes, weights, num_samples = get_datasets_weights_and_num_samples(
            ["0.7", "/path/a", "0.3", "/path/b"], [1000, 100, 50]
        )
        self.assertEqual(len(prefixes), 2)
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_asserts_even_length(self):
        """Test that odd-length data_prefix raises AssertionError."""
        with self.assertRaises(AssertionError):
            get_datasets_weights_and_num_samples(["0.7", "/path/a", "0.3"], [1000, 100, 50])

    def test_num_samples_include_margin(self):
        """Test that num_samples include 0.5% margin and +20."""
        _, _, num_samples = get_datasets_weights_and_num_samples(["1.0", "/path/data"], [1000, 100, 50])
        # For weight=1.0: ceil(1000 * 1.0 * 1.005) + 20
        expected_train = int(1000 * 1.0 * 1.005) + 20  # ceil is applied per val
        # The actual formula uses math.ceil(val * weight * 1.005) + 20
        import math

        expected_train = math.ceil(1000 * 1.0 * 1.005) + 20
        self.assertEqual(num_samples[0][0], expected_train)

    def test_weights_are_normalized(self):
        """Test that weights sum to approximately 1.0."""
        _, weights, _ = get_datasets_weights_and_num_samples(["2.0", "/path/a", "3.0", "/path/b"], [1000, 100, 50])
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertAlmostEqual(weights[0], 0.4, places=5)
        self.assertAlmostEqual(weights[1], 0.6, places=5)

    def test_zero_weight_raises(self):
        """Test that zero total weight raises AssertionError."""
        with self.assertRaises(AssertionError):
            get_datasets_weights_and_num_samples(["0.0", "/path/a"], [1000, 100, 50])


if __name__ == "__main__":
    unittest.main()
