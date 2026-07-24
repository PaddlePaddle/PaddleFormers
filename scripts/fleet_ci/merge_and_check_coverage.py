#!/usr/bin/env python3

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

"""
Merge multiple coverage.xml files and check full coverage
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom


def parse_coverage_file(file_path):
    """Parse a coverage.xml file and return its structure."""
    if not os.path.exists(file_path):
        print(f"Warning: Coverage file not found: {file_path}")
        return None

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def merge_coverage_files(coverage_files):
    """
    Merge multiple coverage.xml files into one.
    The merged coverage combines line coverage data from all files.
    Returns: (merged_root, coverage_rate, filename_to_coverage_dict)
    """
    print("\n=== 合并覆盖率文件 ===")

    # Structure to store combined coverage data
    # Key: (filename, line_number) -> Value: hits
    line_coverage = defaultdict(int)

    # Files processed
    files_processed = 0
    files_skipped = 0

    # Exclude third-party or non-PaddleFleet code
    exclude_patterns = [
        "ops/deep_ep",
        "ops/deep_gemm",
        "ops/quack",
        "ops/flash_mask",
    ]

    for file_path in coverage_files:
        root = parse_coverage_file(file_path)
        if root is None:
            files_skipped += 1
            continue

        files_processed += 1

        # Iterate over all class elements
        for class_elem in root.findall(".//class"):
            filename = class_elem.get("filename", "")
            if not filename:
                continue

            # Only include PaddleFleet code, exclude third-party
            if not filename.startswith("paddleformers/fleet/") or any(
                pattern in filename for pattern in exclude_patterns
            ):
                continue

            # Iterate over lines
            lines_elem = class_elem.find("lines")
            if lines_elem is None:
                continue

            for line_elem in lines_elem.findall("line"):
                line_num = int(line_elem.get("number", 0))
                hits = int(line_elem.get("hits", 0))

                # Combine hits (max value)
                key = (filename, line_num)
                line_coverage[key] = max(line_coverage[key], hits)

    print(f"处理了 {files_processed} 个文件，跳过了 {files_skipped} 个文件")
    print(f"找到 {len(line_coverage)} 个唯一代码行")

    if not line_coverage:
        print("警告: 没有找到任何覆盖率数据")
        return None

    # Build merged coverage structure
    # Group by filename
    filename_to_lines = defaultdict(list)
    for (filename, line_num), hits in line_coverage.items():
        filename_to_lines[filename].append((line_num, hits))

    # Create merged XML
    merged_root = ET.Element("coverage")
    merged_root.set("version", "1.0")
    merged_root.set("timestamp", "0")

    # Add sources
    sources = ET.SubElement(merged_root, "sources")
    ET.SubElement(sources, "source").text = "paddleformers.fleet"

    # Add packages
    packages = ET.SubElement(merged_root, "packages")

    # Group files by package
    package = ET.SubElement(packages, "package")
    package.set("name", "merged")
    package.set("line-rate", "0")
    package.set("branch-rate", "0")
    package.set("complexity", "0")

    # Add classes
    classes = ET.SubElement(package, "classes")

    total_lines = 0
    covered_lines = 0

    # Store per-file coverage for reporting
    filename_coverage = {}

    # Sort by filename for consistent output
    sorted_filenames = sorted(filename_to_lines.keys())

    for filename in sorted_filenames:
        lines_data = filename_to_lines[filename]
        lines_data.sort()  # Sort by line number

        # Create class element
        class_elem = ET.SubElement(classes, "class")
        class_elem.set("name", filename.replace("/", ".").replace(".py", ""))
        class_elem.set("filename", filename)

        # Calculate per-file coverage
        file_total = len(lines_data)
        file_covered = sum(1 for _, hits in lines_data if hits > 0)

        class_elem.set(
            "line-rate", str(file_covered / file_total if file_total > 0 else 0)
        )
        class_elem.set(
            "branch-rate",
            str(file_covered / file_total if file_total > 0 else 0),
        )
        class_elem.set("complexity", "0")

        # Add methods (empty)
        methods = ET.SubElement(class_elem, "methods")

        # Add lines
        lines_elem = ET.SubElement(class_elem, "lines")
        for line_num, hits in lines_data:
            line_elem = ET.SubElement(lines_elem, "line")
            line_elem.set("number", str(line_num))
            line_elem.set("hits", str(hits))
            line_elem.set("branch", "0")

        total_lines += file_total
        covered_lines += file_covered

        # Store per-file coverage
        coverage_rate = file_covered / file_total if file_total > 0 else 0
        filename_coverage[filename] = {
            "total": file_total,
            "covered": file_covered,
            "rate": coverage_rate,
        }

    # Calculate overall coverage
    overall_coverage_rate = (
        covered_lines / total_lines if total_lines > 0 else 0
    )

    # Update coverage stats
    package.set("line-rate", str(overall_coverage_rate))
    package.set("branch-rate", str(overall_coverage_rate))

    merged_root.set("line-rate", str(overall_coverage_rate))
    merged_root.set("branch-rate", str(overall_coverage_rate))
    merged_root.set("lines-covered", str(covered_lines))
    merged_root.set("lines-valid", str(total_lines))

    print(f"\n{'=' * 60}")
    print(f"{'全量覆盖率报告'.center(60)}")
    print(f"{'=' * 60}")
    print(f"{'文件名':<50} {'覆盖率':>10}")
    print(f"{'-' * 60}")

    for filename in sorted_filenames:
        cov = filename_coverage[filename]
        short_filename = filename[-50:] if len(filename) > 50 else filename
        print(f"{short_filename:<50} {cov['rate'] * 100:>9.2f}%")

    print(f"{'-' * 60}")
    print(f"{'总计:':<50} {overall_coverage_rate * 100:>9.2f}%")
    print(f"{'总代码行数:':<50} {total_lines:>10}")
    print(f"{'已覆盖行数:':<50} {covered_lines:>10}")
    print(f"{'=' * 60}")

    return merged_root, overall_coverage_rate, filename_coverage


def print_full_coverage_report(
    filename_coverage, overall_coverage_rate, total_lines, covered_lines
):
    """Print a detailed full coverage report."""
    print(f"\n{'=' * 80}")
    print(f"{'全量覆盖率详细报告'.center(80)}")
    print(f"{'=' * 80}")

    # Group by directory
    dir_coverage = defaultdict(lambda: {"total": 0, "covered": 0, "rate": 0})

    for filename, cov in filename_coverage.items():
        dir_name = (
            "/".join(filename.split("/")[:-1]) if "/" in filename else "root"
        )
        dir_coverage[dir_name]["total"] += cov["total"]
        dir_coverage[dir_name]["covered"] += cov["covered"]

    # Calculate directory rates
    for dir_name, stats in dir_coverage.items():
        stats["rate"] = (
            stats["covered"] / stats["total"] if stats["total"] > 0 else 0
        )

    # Print summary
    print("\n【总体统计】")
    print(f"  总代码行数: {total_lines}")
    print(f"  已覆盖行数: {covered_lines}")
    print(f"  未覆盖行数: {total_lines - covered_lines}")
    print(f"  全量覆盖率: {overall_coverage_rate * 100:.2f}%")

    # Print by directory
    print("\n【按目录统计】")
    print(f"{'目录':<50} {'行数':>10} {'覆盖':>10} {'覆盖率':>10}")
    print(f"{'-' * 80}")

    sorted_dirs = sorted(
        dir_coverage.items(), key=lambda x: x[1]["rate"], reverse=True
    )
    for dir_name, stats in sorted_dirs:
        short_dir = dir_name[-50:] if len(dir_name) > 50 else dir_name
        print(
            f"{short_dir:<50} {stats['total']:>10} {stats['covered']:>10} {stats['rate'] * 100:>9.2f}%"
        )

    # Print per-file details
    print("\n【按文件详细统计】")
    print(f"{'文件':<55} {'行数':>10} {'覆盖':>10} {'覆盖率':>10}")
    print(f"{'-' * 80}")

    sorted_files = sorted(
        filename_coverage.items(), key=lambda x: x[1]["rate"], reverse=True
    )
    for filename, cov in sorted_files:
        short_filename = filename[-55:] if len(filename) > 55 else filename
        rate_str = f"{cov['rate'] * 100:.2f}%"
        status = (
            "✓" if cov["rate"] >= 0.8 else "✗" if cov["rate"] < 0.5 else "~"
        )
        print(
            f"{short_filename:<55} {cov['total']:>10} {cov['covered']:>10} {rate_str:>10} {status}"
        )

    print(f"{'=' * 80}\n")


def check_coverage(coverage_rate, fail_under, strict=False):
    """Check if coverage meets the threshold."""
    if not strict:
        # 仅报告模式，不进行阈值检查
        print("\n=== 覆盖率报告 ===")
        print(f"实际覆盖率: {coverage_rate * 100:.2f}%")
        return 0

    print("\n=== 覆盖率检查 ===")
    print(f"目标覆盖率: {fail_under}%")
    print(f"实际覆盖率: {coverage_rate * 100:.2f}%")

    if coverage_rate * 100 >= fail_under:
        print("✅ 覆盖率检查通过")
        return 0
    else:
        print(
            f"❌ 覆盖率检查失败: 需要 {fail_under}%，实际为 {coverage_rate * 100:.2f}%"
        )
        return 1


def save_merged_coverage(root, output_path):
    """Save merged coverage to file."""
    # Convert to string and pretty print
    rough_string = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(rough_string)
    pretty_string = dom.toprettyxml(indent="  ", encoding="utf-8")

    # Remove extra blank lines that minidom adds
    lines = pretty_string.decode("utf-8").split("\n")
    pretty_lines = [line for line in lines if line.strip()]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pretty_lines))
        f.write("\n")

    print(f"\n✅ 已保存合并后的覆盖率文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge coverage files and check coverage"
    )
    parser.add_argument(
        "coverage_files",
        nargs="+",
        help="Coverage XML files to merge",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=60.0,
        help="Minimum coverage percentage required (default: 60.0, ignored in report-only mode)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only print coverage report without threshold checking",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_coverage.xml",
        help="Output file for merged coverage (default: merged_coverage.xml)",
    )
    parser.add_argument(
        "--output-xml",
        action="store_true",
        help="Output merged coverage XML file (default: False)",
    )

    args = parser.parse_args()

    print("=== 全量覆盖率合并与检查工具 ===")

    # Merge coverage files
    result = merge_coverage_files(args.coverage_files)

    if result is None:
        print("❌ 无法合并覆盖率文件: 没有找到有效数据")
        sys.exit(1)

    merged_root, coverage_rate, filename_coverage = result

    # Calculate total lines for report
    total_lines = sum(cov["total"] for cov in filename_coverage.values())
    covered_lines = sum(cov["covered"] for cov in filename_coverage.values())

    # Print detailed coverage report
    print_full_coverage_report(
        filename_coverage, coverage_rate, total_lines, covered_lines
    )

    # Save merged coverage to XML file (only if --output-xml is set)
    if args.output_xml:
        save_merged_coverage(merged_root, args.output)

    # Check coverage (only if not in report-only mode)
    if args.report_only:
        print("=== 报告模式：不进行阈值检查 ===")
        exit_code = 0
    else:
        exit_code = check_coverage(coverage_rate, args.fail_under, strict=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
