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
from pathlib import Path

from ci_utils.log_analyzer import compare_with_baseline, parse_loss_values


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

        # Helper to extract loss-related lines from log
        def get_loss_lines(log_content, max_lines=30):
            """Extract lines containing loss information."""
            lines = log_content.split("\n")
            loss_lines = [l for l in lines if "loss:" in l.lower() or "global_step" in l.lower()]
            if len(loss_lines) > max_lines:
                return "\n".join(loss_lines[:max_lines]) + f"\n... ({len(loss_lines) - max_lines} more loss lines)"
            return "\n".join(loss_lines) if loss_lines else "(no loss lines found)"

        if not losses:
            msg = (
                f"\n{'=' * 80}\n"
                f"LOSS PARSING FAILED\n"
                f"{'=' * 80}\n"
                f"No loss values found in training output.\n"
                f"Expected format: 'loss: <value> learning_rate: ... global_step: <step>'\n"
                f"\n{'=' * 60}\n"
                f"LOG OUTPUT (last 80 lines):\n"
                f"{'=' * 60}\n"
                f"{_get_last_n_lines(log_output, 80)}\n"
                f"{'=' * 80}\n"
            )
            return False, {}, msg

        # Check baseline file exists
        if not self.baseline_path.exists():
            return (
                False,
                {},
                (
                    f"\n{'=' * 80}\n"
                    f"BASELINE FILE NOT FOUND\n"
                    f"{'=' * 80}\n"
                    f"Baseline file: {self.baseline_path}\n"
                    f"Found loss values: {losses}\n"
                    f"\nYou may need to create the baseline file with these values.\n"
                    f"{'=' * 80}\n"
                ),
            )

        # Compare with baseline
        try:
            passed, details = compare_with_baseline(losses, self.baseline_path, tolerance=self.tolerance)
        except Exception as e:
            return (
                False,
                {},
                (
                    f"\n{'=' * 80}\n"
                    f"BASELINE COMPARISON ERROR\n"
                    f"{'=' * 80}\n"
                    f"Error: {e}\n"
                    f"Baseline file: {self.baseline_path}\n"
                    f"Current losses: {losses}\n"
                    f"{'=' * 80}\n"
                ),
            )

        if not passed:
            failed_steps = [step for step, res in details.items() if not res["passed"]]
            msg = (
                f"\n{'=' * 80}\n"
                f"LOSS PRECISION COMPARISON FAILED\n"
                f"{'=' * 80}\n"
                f"Tolerance: {self.tolerance}\n"
                f"Baseline file: {self.baseline_path}\n"
                f"\nFailed steps: {failed_steps}\n"
                f"\n{'=' * 60}\n"
                f"DETAILED COMPARISON:\n"
                f"{'=' * 60}\n"
            )
            for step in sorted(details.keys()):
                res = details[step]
                status = "✓ PASS" if res["passed"] else "✗ FAIL"
                msg += (
                    f"  Step {step}: {status}\n"
                    f"    Current:  {res['current']:.8f}\n"
                    f"    Baseline: {res['baseline']:.8f}\n"
                    f"    Diff:     {res['diff']:.2e}\n"
                )
            msg += (
                f"\n{'=' * 60}\n"
                f"TRAINING LOG (loss lines):\n"
                f"{'=' * 60}\n"
                f"{get_loss_lines(log_output)}\n"
                f"{'=' * 80}\n"
            )
            return False, details, msg

        return True, details, None


def _get_last_n_lines(text, n=100):
    """Get the last n lines of text."""
    lines = text.strip().split("\n")
    if len(lines) <= n:
        return text
    return f"... (truncated, showing last {n} lines) ...\n" + "\n".join(lines[-n:])


def run_command_and_validate(cmd, baseline_path, log_file, working_dir=None, tolerance=1e-6, timeout=3600):
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
        result = subprocess.run(cmd, shell=True, cwd=working_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # Get partial output if available
        partial_stdout = e.stdout if e.stdout else ""
        partial_stderr = e.stderr if e.stderr else ""
        partial_output = partial_stdout + "\n" + partial_stderr
        error_msg = (
            f"Command timed out after {timeout} seconds\n"
            f"Command: {cmd}\n"
            f"\n{'=' * 60}\n"
            f"PARTIAL OUTPUT BEFORE TIMEOUT:\n"
            f"{'=' * 60}\n"
            f"{_get_last_n_lines(partial_output, 50)}\n"
        )
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
        # Include actual log content in error message for CI visibility
        error_msg = (
            f"\n{'=' * 80}\n"
            f"COMMAND EXECUTION FAILED\n"
            f"{'=' * 80}\n"
            f"Return Code: {result.returncode}\n"
            f"Command: {cmd}\n"
            f"Working Directory: {working_dir or '(current)'}\n"
            f"\n{'=' * 60}\n"
            f"STDERR OUTPUT:\n"
            f"{'=' * 60}\n"
            f"{_get_last_n_lines(result.stderr, 80) if result.stderr.strip() else '(empty)'}\n"
            f"\n{'=' * 60}\n"
            f"STDOUT OUTPUT (last 50 lines):\n"
            f"{'=' * 60}\n"
            f"{_get_last_n_lines(result.stdout, 50) if result.stdout.strip() else '(empty)'}\n"
            f"{'=' * 80}\n"
        )
        return False, error_msg
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
