#!/usr/bin/env python3

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

"""
Compare config.yaml and config_ci.yaml, and copy new models to config_ci.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_yaml(filepath):
    """Load YAML file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(filepath, data):
    """Save YAML file"""
    import re

    class NoAliasDumper(yaml.SafeDumper):
        """Dumper that doesn't use anchors and aliases"""

        def ignore_aliases(self, data):
            return True

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if re.match(r"^(\s+)-\s+-\s+\S", line):

            indent_match = re.match(r"^(\s+)-\s+-\s+", line)
            indent = indent_match.group(1)
            array_items = []
            array_items.append(line[indent_match.end() :])

            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_indent_match = re.match(r"^" + re.escape(indent) + r"\s+-\s+", next_line)
                if next_indent_match:
                    array_items.append(next_line[next_indent_match.end() :])
                    j += 1
                else:
                    break

            result_lines.append(indent + "- [" + ", ".join(array_items) + "]")
            i = j
        else:
            result_lines.append(line)
            i += 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))


def merge_configs(config_path, config_ci_path, output_path=None):
    """
    Compare two YAML config files and copy new models from origin to update config.

    This script finds models that exist in the origin config but not in the update config,
    then adds those new models to the update config file.

    Args:
        config_path: Path to origin config file (source with new models)
        config_ci_path: Path to update config file (target that receives new models)
        output_path: Output file path. If None, overwrite the update config file
    """
    if output_path is None:
        output_path = config_ci_path

    config = load_yaml(config_path)
    config_ci = load_yaml(config_ci_path)

    config_name = Path(config_path).name
    config_ci_name = Path(config_ci_path).name

    print(f"Number of models in {config_name}: {len(config)}")
    print(f"Number of models in {config_ci_name}: {len(config_ci)}")
    print()

    new_models = {}
    updated_models = {}
    has_changes = False

    for model_name, model_config in config.items():
        if model_name not in config_ci:

            new_models[model_name] = model_config
            has_changes = True
            print(f"[NEW] Model: {model_name}")
        else:

            ci_config = config_ci[model_name]
            cli_args_updates = {}
            other_field_updates = {}

            if "cli_args" in model_config:
                config_cli_args = model_config.get("cli_args", {})
                ci_cli_args = ci_config.get("cli_args", {})

                for key, value in config_cli_args.items():
                    if key not in ci_cli_args:
                        cli_args_updates[key] = value

            for key in ["repo_id", "model_type"]:
                if key in model_config and key not in ci_config:
                    other_field_updates[key] = model_config[key]

            if cli_args_updates or other_field_updates:
                has_changes = True

                new_fields = {
                    "cli_args": cli_args_updates if cli_args_updates else None,
                    "other_fields": other_field_updates if other_field_updates else None,
                }

                print(f"[UPDATE] Model {model_name} has new fields:")
                for key, value in cli_args_updates.items():
                    print(f"  - cli_args.{key}: {value}")
                for key, value in other_field_updates.items():
                    print(f"  - {key}: {value}")

                updated_models[model_name] = new_fields

    if new_models:
        for model_name, model_config in new_models.items():
            new_model = model_config.copy()
            config_ci[model_name] = new_model

    if updated_models:
        print(f"\nUpdating {len(updated_models)} model(s)...")
        for model_name, new_fields in updated_models.items():

            if new_fields.get("cli_args"):
                if "cli_args" not in config_ci[model_name]:
                    config_ci[model_name]["cli_args"] = {}
                config_ci[model_name]["cli_args"].update(new_fields["cli_args"])

            if new_fields.get("other_fields"):
                for key, value in new_fields["other_fields"].items():
                    config_ci[model_name][key] = value

    if has_changes:
        save_yaml(output_path, config_ci)
    else:
        # No changes: preserve the original update_config (config.yaml) as output
        save_yaml(output_path, config_ci)
    return {
        "new_models": list(new_models.keys()),
        "updated_models": list(updated_models.keys()),
        "has_changes": has_changes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two YAML config files and copy new content to target file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    script_dir = Path(__file__).parent.absolute()

    parser.add_argument(
        "--origin_config", type=str, default="config.yaml", help="Source config file name (default: config.yaml)"
    )
    parser.add_argument(
        "--update_config",
        type=str,
        default="config_ci.yaml",
        help="Target config file name to be updated (default: config_ci.yaml)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path. If None, overwrite update_config (default: None)"
    )

    args = parser.parse_args()

    origin_config_path = Path(args.origin_config)
    update_config_path = Path(args.update_config)

    if not origin_config_path.exists():
        print(f"Error: {origin_config_path} does not exist")
        sys.exit(1)

    if not update_config_path.exists():
        print(f"Error: {update_config_path} does not exist")
        sys.exit(1)

    result = merge_configs(str(origin_config_path), str(update_config_path), args.output)

    if result["new_models"] or result["updated_models"]:
        if result["new_models"]:
            models_str = ",".join(result["new_models"])
            print("new_models=" + models_str)
        else:
            print("new_models=false")
