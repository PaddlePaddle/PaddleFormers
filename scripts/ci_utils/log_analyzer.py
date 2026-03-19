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

import json
import re
from pathlib import Path  # noqa: F401


def parse_loss_values(log_content):
    """Parse loss values from training log

    The log format can be either:
    - loss: 7.11594725 learning_rate: 5e-07 global_step: 1 ...
    - loss: 12.24401569, learning_rate: 2.5e-06, global_step: 5, ...

    Args:
        log_content (str): The content of the log file

    Returns:
        dict: Dictionary containing step numbers as keys and loss values as values
    """
    # Support both formats: with and without commas after values
    loss_pattern = re.compile(r"loss:\s*([\d\.e+-]+),?\s+learning_rate:.*?global_step:\s*(\d+)")
    matches = loss_pattern.findall(log_content)
    return {int(step): float(loss) for loss, step in matches}


def compare_with_baseline(current_losses, baseline_path, tolerance=1e-6):
    """Compare current loss values with baseline

    Args:
        current_losses (dict): Current loss values {step: loss}
        baseline_path (str): Path to baseline JSON file
        tolerance (float): Allowed absolute difference

    Returns:
        tuple: (bool, dict) - (True if all within tolerance, detailed comparison results)
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    results = {}
    all_passed = True

    for step, current_loss in current_losses.items():
        baseline_loss = baseline.get(str(step))
        if baseline_loss is None:
            continue

        diff = abs(current_loss - baseline_loss)
        passed = diff <= tolerance
        if not passed:
            all_passed = False

        results[step] = {"current": current_loss, "baseline": baseline_loss, "diff": diff, "passed": passed}

    return all_passed, results
