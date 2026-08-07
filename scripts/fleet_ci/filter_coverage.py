#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep coverage XML compatible with fleet CI post-processing.")
    parser.add_argument("coverage_file")
    parser.add_argument("diff_file")
    args = parser.parse_args()

    coverage_file = Path(args.coverage_file)
    diff_file = Path(args.diff_file)
    if not coverage_file.exists():
        print(f"Skip missing coverage file: {coverage_file}")
        return 0
    if not diff_file.exists():
        print(f"Diff file not found, keep original coverage file: {diff_file}")
        return 0

    backup_file = coverage_file.with_name(f"{coverage_file.name}.backup")
    shutil.copy2(coverage_file, backup_file)
    print(f"Kept coverage file unchanged: {coverage_file}")
    print(f"Created backup: {backup_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
