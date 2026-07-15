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

import argparse
import re
import sys

import numpy as np


def parse_ground_truth(file_path):
    """
    Parses the ground truth file.
    Returns a dict: {step: loss}
    """
    gt_loss_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                step = int(parts[0])
                loss = float(parts[1])
                gt_loss_dict[step] = loss
    return gt_loss_dict


def parse_log_file(file_path):
    """
    Parses the log file to extract global_step and loss.
    Returns a dict: {step: loss}
    """
    loss_pattern = re.compile(r"(?:^|-\s*)loss:\s*([0-9\.]+)")
    step_pattern = re.compile(r"global_step:\s*(\d+)")

    loss_dict = {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "loss:" in line and "global_step:" in line:
                loss_match = loss_pattern.search(line)
                step_match = step_pattern.search(line)

                if loss_match and step_match:
                    loss_val = float(loss_match.group(1))
                    step_val = int(step_match.group(1))
                    loss_dict[step_val] = loss_val

    return loss_dict


def write_loss_file(loss_dict, file_path, steps=None):
    if steps is None:
        steps = sorted(loss_dict)
    with open(file_path, "w") as f:
        for step in steps:
            f.write(f"{step} {loss_dict[step]}\n")


def main():
    parser = argparse.ArgumentParser(description="Check loss values in log against ground truth.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the log file.")
    parser.add_argument(
        "--gt_file",
        type=str,
        default=None,
        help="Path to the ground truth file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Tolerance for loss comparison.",
    )

    parser.add_argument(
        "--compare_step",
        type=int,
        default=None,
        help="If set, only compare loss at this specific global step.",
    )
    parser.add_argument(
        "--log_loss_file",
        type=str,
        default=None,
        help="Record the loss values from the log file to this file.",
    )
    parser.add_argument(
        "--extract_loss_only",
        action="store_true",
        help="Only extract loss values from the log file and write them to --log_loss_file.",
    )
    args = parser.parse_args()

    print(f"Starting loss check with log file: {args.log_file}")
    print(f"Ground truth file: {args.gt_file}, Tolerance: {args.tolerance}")
    if args.compare_step is not None:
        print(f"Target Check Step: {args.compare_step}")

    log_dict = parse_log_file(args.log_file)
    if args.extract_loss_only:
        if args.log_loss_file is None:
            print("\033[91mError: --log_loss_file is required when --extract_loss_only is set.\033[0m")
            sys.exit(1)
        if not log_dict:
            print(f"\033[91mError: No loss values found in {args.log_file}.\033[0m")
            sys.exit(1)
        write_loss_file(log_dict, args.log_loss_file)
        print(f"\033[92mExtracted loss values to {args.log_loss_file}.\033[0m")
        return

    if args.gt_file is None:
        print("\033[91mError: --gt_file is required unless --extract_loss_only is set.\033[0m")
        sys.exit(1)

    gt_dict = parse_ground_truth(args.gt_file)

    if args.compare_step is not None:
        target_step = args.compare_step

        if target_step not in log_dict:
            print(f"\033[91mError: Step {target_step} not found in log file.\033[0m")
            sys.exit(1)
        if target_step not in gt_dict:
            print(f"\033[91mError: Step {target_step} not found in ground truth file.\033[0m")
            sys.exit(1)

        log_loss = log_dict[target_step]
        gt_loss = gt_dict[target_step]

        if args.log_loss_file is not None:
            write_loss_file(log_dict, args.log_loss_file, [target_step])

        print(f"\nChecking Step {target_step}:")
        print(f"  Log Loss: {log_loss}")
        print(f"  GT  Loss: {gt_loss}")

        actual_losses = [log_loss]
        target_losses = [gt_loss]

    else:
        common_steps = sorted(set(log_dict.keys()) & set(gt_dict.keys()))

        if not common_steps:
            print("\033[91mError: No common steps found between log and ground truth.\033[0m")
            sys.exit(1)

        print(f"\nExtracted {len(common_steps)} common steps for comparison.")

        actual_losses = [log_dict[s] for s in common_steps]
        target_losses = [gt_dict[s] for s in common_steps]

        if args.log_loss_file is not None:
            write_loss_file(log_dict, args.log_loss_file, common_steps)

        print("\nLog values (step loss):")
        print("\n".join([f"{s} {l:.8f}" for s, l in zip(common_steps, actual_losses)]))

    try:
        np.testing.assert_allclose(
            actual_losses,
            target_losses,
            rtol=args.tolerance,
            atol=args.tolerance,
        )
        print("\033[92m\nAll loss checks passed!\033[0m")
    except AssertionError as e:
        print(f"\033[91m\nCheck Failed!\n{e}\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
