#!/usr/bin/env python3

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_rate(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    lines_valid = int(float(root.attrib.get("lines-valid", 0)))
    lines_covered = int(float(root.attrib.get("lines-covered", 0)))
    if lines_valid or lines_covered:
        return lines_covered, lines_valid

    covered = 0
    valid = 0
    for line in root.findall(".//line"):
        valid += 1
        if int(line.attrib.get("hits", 0)) > 0:
            covered += 1
    return covered, valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and report coverage XML files.")
    parser.add_argument("coverage_files", nargs="*")
    parser.add_argument("--fail-under", type=float, default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    total_covered = 0
    total_valid = 0
    for filename in args.coverage_files:
        path = Path(filename)
        if not path.exists():
            print(f"Skip missing coverage file: {path}")
            continue
        covered, valid = parse_rate(path)
        total_covered += covered
        total_valid += valid
        rate = covered / valid * 100 if valid else 0.0
        print(f"{path}: {covered}/{valid} lines covered ({rate:.2f}%)")

    total_rate = total_covered / total_valid * 100 if total_valid else 0.0
    print(f"Total coverage: {total_covered}/{total_valid} lines covered ({total_rate:.2f}%)")

    if args.fail_under is not None and total_rate < args.fail_under:
        print(f"Coverage {total_rate:.2f}% is below fail-under {args.fail_under:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
