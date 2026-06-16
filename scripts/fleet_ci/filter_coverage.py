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
filter diff.txt file to get changed files and lines
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_diff_file(diff_file):
    """
    --- a/src/file1.py
    +++ b/src/file1.py
    @@ -10,5 +10,7 @@

    return: dict
    """
    changed_files = defaultdict(set)
    current_file = None

    if not os.path.exists(diff_file):
        print(f"Error: Diff file not found: {diff_file}")
        return changed_files

    print(f"Parsing diff file: {diff_file}")

    with open(diff_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        #  (+++ b/path/to/file)
        if line.startswith("+++ b/"):
            current_file = line[6:]  # delete '+++ b/'
            print(f"  Found changed file: {current_file}")
            if "paddleformers/fleet" not in current_file:
                current_file = None
                print("    Skipping non-paddleformers.fleet file")
                continue
            elif line.endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".bmp",
                    ".pdf",
                    ".zip",
                    ".tar",
                    ".gz",
                    ".so",
                    ".dll",
                    ".exe",
                )
            ):
                current_file = None  # skip binary file
                print("    Skipping binary file")
                continue

        elif line.startswith("@@") and current_file:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start_line = int(match.group(1))
                line_count = int(match.group(2) or 1)

                for line_num in range(start_line, start_line + line_count):
                    changed_files[current_file].add(line_num)

    print("\nDiff analysis:")
    total_changed_lines = sum(len(lines) for lines in changed_files.values())
    print(f"  Changed files: {len(changed_files)}")
    print(f"  Total changed lines: {total_changed_lines}")

    for filename, lines in changed_files.items():
        if len(lines) <= 10:
            print(
                f"    {filename}: lines {sorted(lines)[:10]}{'...' if len(lines) > 10 else ''}"
            )
        else:
            print(f"    {filename}: {len(lines)} changed lines")
    return changed_files


def filter_coverage_by_diff(coverage_file, diff_data):
    """
    coverage.xml
    """
    if not os.path.exists(coverage_file):
        print(f"Error: Coverage file not found: {coverage_file}")
        return False

    print(f"\nFiltering coverage file: {coverage_file}")

    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()

        # backup
        import shutil

        backup_file = coverage_file + ".backup"
        shutil.copy2(coverage_file, backup_file)

        # stats
        stats = {
            "files_processed": 0,
            "files_kept": 0,
            "lines_kept": 0,
            "lines_removed": 0,
            "lines_covered": 0,
        }

        # filename -> class elements
        filename_to_classes = defaultdict(list)

        # collect all class elements
        for class_elem in root.findall(".//class"):
            filename = class_elem.get("filename", "")
            if filename:
                filename_to_classes[filename].append(class_elem)
                stats["files_processed"] += 1

        print(f"Found {stats['files_processed']} files in coverage report")

        # find files to keep
        files_to_keep = set()
        for coverage_filename in filename_to_classes.keys():
            for diff_filename in diff_data.keys():
                if filename_match(coverage_filename, diff_filename):
                    files_to_keep.add(coverage_filename)
                    break

        print(f"Files to keep after diff matching: {len(files_to_keep)}")

        # if no files to keep, create minimal coverage
        if not files_to_keep:
            print("No matching files found, creating minimal coverage")
            create_minimal_coverage(coverage_file)
            return True

        # create new coverage root
        new_root = ET.Element("coverage")
        new_root.set("version", "1.0")
        new_root.set("timestamp", root.get("timestamp", "0"))

        # add sources
        sources = ET.SubElement(new_root, "sources")
        ET.SubElement(sources, "source").text = "paddleformers.fleet"

        packages = ET.SubElement(new_root, "packages")

        # create a single package for changed files
        package_name = "changed_files"
        package_elem = ET.SubElement(packages, "package")
        package_elem.set("name", package_name)
        package_elem.set("line-rate", "0")
        package_elem.set("branch-rate", "0")
        package_elem.set("complexity", "0")

        # deal with files
        for filename in files_to_keep:
            # find matching diff file
            matched_diff_file = None
            changed_lines = set()

            for diff_file, lines in diff_data.items():
                if filename_match(filename, diff_file):
                    matched_diff_file = diff_file
                    changed_lines = lines
                    break

            if not matched_diff_file:
                continue

            # iterate over class elements
            for class_elem in filename_to_classes[filename]:
                stats["files_kept"] += 1

                # create new class element
                new_class = ET.Element("class")

                # copy attributes
                for attr_name, attr_value in class_elem.items():
                    new_class.set(attr_name, attr_value)

                # deal with lines
                lines_elem = class_elem.find("lines")
                if lines_elem is not None:
                    new_lines_elem = ET.SubElement(new_class, "lines")

                    # filter lines
                    for line_elem in lines_elem.findall("line"):
                        line_num = int(line_elem.get("number", 0))

                        if line_num in changed_lines:
                            # copy line element
                            new_line_elem = ET.SubElement(
                                new_lines_elem, "line"
                            )
                            for attr_name, attr_value in line_elem.items():
                                new_line_elem.set(attr_name, attr_value)

                            stats["lines_kept"] += 1

                            hits = int(line_elem.get("hits", 0))
                            if hits > 0:
                                stats["lines_covered"] += 1
                        else:
                            stats["lines_removed"] += 1

                # add new class to package
                package_elem.append(new_class)

        # compute coverage
        total_lines = stats["lines_kept"]
        covered_lines = stats["lines_covered"]

        if total_lines > 0:
            coverage_rate = covered_lines / total_lines
        else:
            coverage_rate = 0

        # update package
        package_elem.set("line-rate", str(coverage_rate))
        package_elem.set("branch-rate", str(coverage_rate))

        new_root.set("line-rate", str(coverage_rate))
        new_root.set("branch-rate", str(coverage_rate))
        new_root.set("lines-covered", str(covered_lines))
        new_root.set("lines-valid", str(total_lines))

        # write new coverage file
        tree = ET.ElementTree(new_root)
        tree.write(coverage_file, encoding="utf-8", xml_declaration=True)

        # print stats
        print("\n📊 Filtering Statistics:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files kept: {stats['files_kept']}")
        print(f"  Lines kept: {stats['lines_kept']}")
        print(f"  Lines removed: {stats['lines_removed']}")
        print(
            f"  Coverage: {covered_lines}/{total_lines} lines ({coverage_rate * 100:.1f}%)"
        )
        # validate XML
        try:
            ET.parse(coverage_file)
            print("\n✅ Valid XML generated")
            return True
        except ET.ParseError as e:
            print(f"❌ Generated invalid XML: {e}")
            # restore backup
            shutil.copy2(backup_file, coverage_file)
            return False

    except Exception as e:
        print(f"❌ Error filtering coverage: {e}")
        import traceback

        traceback.print_exc()
        return False


def filename_match(coverage_filename, diff_filename):
    """
    1. coverage: "module/file.py", diff: "src/module/file.py"
    2. coverage: "./module/file.py", diff: "module/file.py"
    3. coverage: "file.py", diff: "src/file.py"
    """
    import os

    # normalize path
    def normalize(path):
        # delteprefix
        path = path.removeprefix("./")
        # unixify
        path = path.replace("\\", "/")
        # delete suffix
        path = path.rstrip("/")
        return path.lower()

    cov_norm = normalize(coverage_filename)
    diff_norm = normalize(diff_filename)

    # compare rules
    if cov_norm == diff_norm:
        return True

    if diff_norm.endswith(cov_norm):
        return True

    if cov_norm.endswith(diff_norm):
        return True

    cov_basename = os.path.basename(cov_norm)
    diff_basename = os.path.basename(diff_norm)

    if cov_basename and diff_basename and cov_basename == diff_basename:
        cov_dir = os.path.dirname(cov_norm)
        diff_dir = os.path.dirname(diff_norm)

        if cov_dir in diff_dir or diff_dir in cov_dir:
            return True

    return False


def create_minimal_coverage(filename):
    """create a minimal coverage.xml file"""
    minimal_xml = """<?xml version="1.0"?>
<coverage version="1.0" timestamp="0">
  <sources>
    <source>.</source>
  </sources>
  <packages>
    <package name="changed_files" line-rate="0" branch-rate="0" complexity="0">
      <classes>
        <class name="ChangedCode" filename="changed.py" line-rate="0" branch-rate="0" complexity="0">
          <methods/>
          <lines/>
        </class>
      </classes>
    </package>
  </packages>
</coverage>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(minimal_xml)
    print(f"Created minimal coverage file: {filename}")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python filter_coverage_fixed.py <coverage.xml> <diff.txt>"
        )
        print("Example: python filter_coverage_fixed.py coverage.xml diff.txt")
        sys.exit(1)

    coverage_file = sys.argv[1]
    diff_file = sys.argv[2]

    # diff file
    diff_data = parse_diff_file(diff_file)

    # filter coverage file
    success = filter_coverage_by_diff(coverage_file, diff_data)

    if success:
        # show size
        if os.path.exists(coverage_file):
            size = os.path.getsize(coverage_file)
            print(f"\n✅ Filtered coverage saved: {size:,} bytes")

            if os.path.exists(coverage_file + ".backup"):
                orig_size = os.path.getsize(coverage_file + ".backup")
                reduction = (1 - size / orig_size) * 100
                print(f"   Reduced by: {reduction:.1f}%")
    else:
        print("\n❌ Failed to filter coverage")
        sys.exit(1)


if __name__ == "__main__":
    main()
