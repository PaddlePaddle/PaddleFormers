# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import importlib.metadata
import unittest

import numpy as np
from parameterized import parameterized


def build_blending_indices_python(dataset_index, dataset_sample_index, weights, num_datasets, size, verbose):
    """
    Given multiple datasets and a weighting array, build samples such that it follows those weights.

    Parameters:
    - dataset_index: NumPy array to store the dataset index for each sample.
    - dataset_sample_index: NumPy array to store the sample index within each dataset.
    - weights: NumPy array of weights for each dataset.
    - num_datasets: Integer, the number of datasets.
    - size: Integer, the total number of samples to generate.
    - verbose: Boolean, whether to print verbose output.
    """
    if verbose:
        print("> building indices for blendable datasets ...")

    # Initialize buffer for number of samples used for each dataset.
    current_samples = np.zeros(num_datasets, dtype=np.int64)

    # For each sample:
    for sample_idx in range(size):
        # Determine where the max error in sampling is happening.
        sample_idx_double = max(sample_idx, 1)
        max_error_index = 0
        max_error = weights[0] * sample_idx_double - current_samples[0]
        for dataset_idx in range(1, num_datasets):
            error = weights[dataset_idx] * sample_idx_double - current_samples[dataset_idx]
            if error > max_error:
                max_error = error
                max_error_index = dataset_idx

        # Populate the indices.
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index]

        # Update the total samples.
        current_samples[max_error_index] += 1

    # Print info
    if verbose:
        print(" > sample ratios:")
        for dataset_idx in range(num_datasets):
            ratio = current_samples[dataset_idx] / size
            print(f"   dataset {dataset_idx}, input: {weights[dataset_idx]}, achieved: {ratio}")


def skip_if_version_not_equal(version="0.1.1", package_name="fast_dataindex"):
    try:
        importlib.import_module(package_name)
    except ImportError:
        return True, f"package<{package_name}> not found, so to skip this test"
    package_version = importlib.metadata.version(package_name)
    if package_version != version:
        return True, f"{package_name} version must be equal to {version}, but got {package_version}!"
    return False, f"{package_name} version is ok!"


class TestToolHelpers(unittest.TestCase):
    def _test_build_blending_indices(
        self, num_datasets=128, size=8192, dataset_index_dtype="uint8", verbose=False, seed=42, assert_true=True
    ):
        if isinstance(dataset_index_dtype, str):
            dataset_index_dtype = np.dtype(dataset_index_dtype)
        assert dataset_index_dtype in [np.uint8, np.int16], "dataset_index_dtype must be uint8 or int16!"

        np.random.seed(seed)
        random_numbers = np.random.rand(num_datasets)
        random_numbers[0] = 200
        weights = random_numbers / random_numbers.sum()
        weights = weights.astype(np.float64)

        # for ground truth, so we use np.int32
        python_dataset_index = np.zeros(size, dtype=np.int32)
        python_dataset_sample_index = np.zeros(size, dtype=np.int64)
        build_blending_indices_python(
            python_dataset_index, python_dataset_sample_index, weights, num_datasets, size, verbose
        )

        from fast_dataindex import helpers

        c_dataset_index = np.zeros(size, dtype=dataset_index_dtype)
        c_dataset_sample_index = np.zeros(size, dtype=np.int64)
        helpers.build_blending_indices(c_dataset_index, c_dataset_sample_index, weights, num_datasets, size, verbose)

        assert_func = self.assertTrue if assert_true else self.assertFalse
        assert_func(np.all(python_dataset_index == c_dataset_index.astype(python_dataset_index.dtype)))
        self.assertTrue(
            np.all(python_dataset_sample_index == c_dataset_sample_index.astype(python_dataset_sample_index.dtype))
        )

    @parameterized.expand(
        [
            (128, 8192, "uint8", False, 42, True),
            (1024, 8192, "uint8", False, 42, False),
            (128, 8192, "int16", False, 42, False),
            (1024, 8192, "int16", False, 42, False),
        ]
    )
    @unittest.skipIf(*skip_if_version_not_equal(version="0.1.1", package_name="fast_dataindex"))
    def test_build_blending_indices_version_0_1_1(
        self, num_datasets=128, size=8192, dataset_index_dtype="uint8", verbose=False, seed=42, assert_true=True
    ):
        self._test_build_blending_indices(num_datasets, size, dataset_index_dtype, verbose, seed, assert_true)

    @parameterized.expand(
        [
            (128, 8192, "uint8", False, 42, True),
            (1024, 8192, "uint8", False, 42, False),
            (128, 8192, "int16", False, 42, True),
            (1024, 8192, "int16", False, 42, True),
        ]
    )
    @unittest.skipIf(*skip_if_version_not_equal(version="0.1.2", package_name="fast_dataindex"))
    def test_build_blending_indices_version_0_1_2(
        self, num_datasets=128, size=8192, dataset_index_dtype="uint8", verbose=False, seed=42, assert_true=True
    ):
        self._test_build_blending_indices(num_datasets, size, dataset_index_dtype, verbose, seed, assert_true)
