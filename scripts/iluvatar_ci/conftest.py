# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
Pytest configuration and fixtures for iluvatar CI tests.
"""
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from ci_utils.training_utils import run_command_and_validate  # noqa: F401

# Define directory paths
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent


@pytest.fixture(scope="session")
def test_dir():
    """Return the test directory path."""
    return TEST_DIR


@pytest.fixture(scope="session")
def project_root():
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_dir(test_dir):
    """Return the config directory path."""
    return test_dir / "config"


@pytest.fixture(scope="session")
def base_value_dir(test_dir):
    """Return the base_value directory path."""
    return test_dir / "base_value"


@pytest.fixture
def log_file(tmp_path, request):
    """Create a unique log file for each test."""
    test_name = request.node.name
    return tmp_path / f"{test_name}.log"
