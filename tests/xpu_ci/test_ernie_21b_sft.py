import pytest
from conftest import run_command_and_validate


def test_ernie_21b_sft_training(project_root, base_value_dir, log_file):
    """Test ERNIE-21B SFT training loss values.
    
    This test runs the following shell command:
        paddleformers-cli train tests/xpu_ci/config/ernie_21b_sft.yaml
    
    Then validates that loss values match the baseline within tolerance of 1e-6.
    """
    # Define the exact shell command to execute
    cmd = "paddleformers-cli train tests/xpu_ci/config/ernie_21b_sft.yaml"
    
    # Execute command and validate results
    passed, error_msg = run_command_and_validate(
        cmd=cmd,
        baseline_path=base_value_dir / "ernie_21b_sft_loss.json",
        log_file=log_file,
        working_dir=project_root,
        tolerance=1e-4,
        timeout=3600
    )
    
    if not passed:
        pytest.fail(error_msg)