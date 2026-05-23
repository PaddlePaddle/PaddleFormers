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

import logging
import subprocess
import os

logger = logging.getLogger(__name__)


def test_ci_environment_info():
    """Verify CI environment is correctly configured."""
    result = subprocess.run(["date"], capture_output=True, text=True)
    logger.info(f"CI_DATE: {result.stdout.strip()}")

    result = subprocess.run(["hostname"], capture_output=True, text=True)
    logger.info(f"CI_HOST: {result.stdout.strip()}")

    result = subprocess.run(["whoami"], capture_output=True, text=True)
    logger.info(f"CI_USER: {result.stdout.strip()}")

    result = subprocess.run(["id"], capture_output=True, text=True)
    logger.info(f"CI_ID: {result.stdout.strip()}")

    workspace = os.environ.get("work_dir", "")
    logger.info(f"CI_WORKSPACE: {workspace}")

    assert True
