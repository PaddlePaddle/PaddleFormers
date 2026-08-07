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
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest


class TestPaddleFleetInit(unittest.TestCase):
    """Tests for paddlefleet top-level __init__.py."""

    def test_import_paddlefleet(self):
        """Test that paddlefleet can be imported."""
        import paddleformers.fleet as paddlefleet

        self.assertIsNotNone(paddlefleet)

    def test_all_exports(self):
        """Test that __all__ contains expected entries."""
        from paddleformers.fleet import __all__

        expected_exports = [
            "training",
            "parallel_state",
            "Timers",
            "__version__",
            "__package_name__",
            "__description__",
            "__license__",
            "__contact_names__",
            "__contact_emails__",
            "__homepage__",
            "__repository_url__",
            "__download_url__",
            "__keywords__",
        ]
        for export in expected_exports:
            self.assertIn(export, __all__)

    def test_mpu_alias(self):
        """Test that mpu is an alias for parallel_state."""
        import paddleformers.fleet as paddlefleet

        self.assertIs(paddleformers.fleet.mpu, paddleformers.fleet.parallel_state)

    def test_timers_import(self):
        """Test that Timers can be imported from top-level package."""
        from paddleformers.fleet import Timers

        self.assertIsNotNone(Timers)

    def test_spec_utils_import(self):
        """Test that spec_utils module is importable."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        self.assertIsNotNone(LayerSpec)


if __name__ == "__main__":
    unittest.main()
