"""
Pytest configuration and fixtures for XPU CI tests.
"""
import pytest
import subprocess
from pathlib import Path
from utils.log_analyzer import parse_loss_values, compare_with_baseline

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


class TrainingTestRunner:
    """Helper class for training tests - handles log parsing and validation only."""
    
    def __init__(self, baseline_path, tolerance=1e-6):
        """
        Initialize the test runner.
        
        Args:
            baseline_path: Path to the baseline loss JSON file
            tolerance: Allowed absolute difference for loss comparison
        """
        self.baseline_path = Path(baseline_path)
        self.tolerance = tolerance
    
    def validate_losses(self, log_output, log_file):
        """
        Parse and validate loss values against baseline.
        
        Args:
            log_output: The log content string
            log_file: Path to the log file (for error messages)
            
        Returns:
            tuple: (passed, details, error_message)
        """
        # Parse loss values
        losses = parse_loss_values(log_output)
        
        if not losses:
            return False, {}, f"No loss values found in training output. Check log: {log_file}"
        
        # Check baseline file exists
        if not self.baseline_path.exists():
            return False, {}, (
                f"Baseline file not found: {self.baseline_path}\n"
                f"Found loss values: {losses}\n"
                f"You may need to create the baseline file with these values."
            )
        
        # Compare with baseline
        try:
            passed, details = compare_with_baseline(
                losses, 
                self.baseline_path, 
                tolerance=self.tolerance
            )
        except Exception as e:
            return False, {}, f"Error comparing with baseline: {e}\nCheck log: {log_file}"
        
        if not passed:
            failed_steps = [step for step, res in details.items() if not res["passed"]]
            msg = f"Loss values differ at steps: {failed_steps}\n"
            for step in failed_steps:
                res = details[step]
                msg += (
                    f"  Step {step}: current={res['current']:.8f}, "
                    f"baseline={res['baseline']:.8f}, "
                    f"diff={res['diff']:.2e}\n"
                )
            msg += f"\nFull log saved to: {log_file}"
            return False, details, msg
        
        return True, details, None
    
def run_command_and_validate(cmd, baseline_path, log_file, 
                            working_dir=None, tolerance=1e-6, timeout=3600):
    """
    Execute a shell command, capture output, and validate loss values.
    
    This is a standalone helper function that can be used for any training command.
    
    Args:
        cmd: Shell command to execute
        baseline_path: Path to baseline loss JSON file
        log_file: Path to save the log output
        working_dir: Working directory for command execution (default: project root)
        tolerance: Allowed absolute difference for loss comparison
        timeout: Command timeout in seconds
        
    Returns:
        tuple: (passed, error_message)
    """
    # Print command info for visibility
    print("\n" + "=" * 80)
    print("EXECUTING TEST COMMAND")
    print("=" * 80)
    print(f"Command: {cmd}")
    print(f"Working Directory: {working_dir or '(current)'}")
    print(f"Baseline File: {baseline_path}")
    print(f"Tolerance: {tolerance}")
    print(f"Timeout: {timeout}s")
    print(f"Log File: {log_file}")
    print("=" * 80 + "\n")
    
    # Execute command
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        error_msg = f"Command timed out after {timeout} seconds\nCommand: {cmd}"
        print(f"✗ {error_msg}")
        return False, error_msg
    
    # Combine stdout and stderr
    full_output = result.stdout + "\n" + result.stderr
    
    # Create detailed log with metadata
    log_header = f"""
{'=' * 80}
TEST EXECUTION LOG
{'=' * 80}
Executed Command: {cmd}
Working Directory: {working_dir or '(current)'}
Baseline File: {baseline_path}
Tolerance: {tolerance}
Timeout: {timeout}s
Return Code: {result.returncode}
{'=' * 80}

"""
    
    # Save log with header
    with open(log_file, "w") as f:
        f.write(log_header)
        f.write(full_output)
    
    # Print execution result
    print(f"Command finished with return code: {result.returncode}")
    if result.returncode != 0:
        print(f"⚠️  Command failed! Check log: {log_file}")
        return False, (
            f"Command failed with return code {result.returncode}.\n"
            f"Executed: {cmd}\n"
            f"Check log file: {log_file}"
        )
    else:
        print(f"✓ Command succeeded. Log saved to: {log_file}\n")
    
    # Validate losses
    runner = TrainingTestRunner(baseline_path, tolerance)
    passed, details, error_msg = runner.validate_losses(full_output, log_file)
    
    # Print validation result
    if passed:
        print("✓ Loss validation PASSED - all values within tolerance\n")
    else:
        print("✗ Loss validation FAILED - see details above\n")
    
    return passed, error_msg