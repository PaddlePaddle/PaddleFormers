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
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_base_template():
    """Get base_loss and base_result template for new models"""
    return {
        "base_loss": {
            "sft_full_loss": 0.0,
            "sft_full_resume_loss": 0.0,
            "dpo_full_loss": 0.69314718,
            "dpo_full_resume_loss": 0.69314718,
            "pt_full_loss": 0.0,
            "pt_full_resume_loss": 0.0,
            "sft_lora_loss": 0.0,
            "sft_lora_resume_loss": 0.0,
            "dpo_lora_loss": 0.69314718,
            "dpo_lora_resume_loss": 0.69314718,
            "pt_lora_loss": 0.0,
            "pt_lora_resume_loss": 0.0,
            "sft_full_tp_pp_loss": 0.0,
            "sft_full_tp_pp_resume_loss": 0.0,
            "dpo_full_tp_pp_loss": 0.69314718,
            "dpo_full_tp_pp_resume_loss": 0.69314718,
            "pt_full_tp_pp_loss": 0.0,
            "pt_full_tp_pp_resume_loss": 0.0,
            "sft_lora_tp_pp_loss": 0.0,
            "sft_lora_tp_pp_resume_loss": 0.0,
            "dpo_lora_tp_pp_loss": 0.69314718,
            "dpo_lora_tp_pp_resume_loss": 0.69314718,
            "pt_lora_tp_pp_loss": 0.0,
            "pt_lora_tp_pp_resume_loss": 0.0,
            "sft_full_function_call_loss": 0.0,
            "sft_full_function_call_resume_loss": 0.0,
            "dpo_full_function_call_loss": 0.69314718,
            "dpo_full_function_call_resume_loss": 0.69314718,
        },
        "base_result": {
            "pt_full_excepted_result": [],
            "sft_full_excepted_result": [],
            "dpo_full_excepted_result": [],
            "pt_lora_excepted_result": [],
            "sft_lora_excepted_result": [],
            "dpo_lora_excepted_result": [],
            "pt_full_tp_pp_excepted_result": [],
            "sft_full_tp_pp_excepted_result": [],
            "dpo_full_tp_pp_excepted_result": [],
            "pt_lora_tp_pp_excepted_result": [],
            "sft_lora_tp_pp_excepted_result": [],
            "dpo_lora_tp_pp_excepted_result": [],
            "sft_full_function_call_excepted_result": [],
            "dpo_full_function_call_excepted_result": [],
        },
    }


def merge_configs(config_path, config_ci_path, output_path=None):
    """
    Compare config.yaml and config_ci.yaml, and copy new content to config_ci.yaml

    Args:
        config_path: Path to config.yaml
        config_ci_path: Path to config_ci.yaml
        output_path: Output file path. If None, overwrite config_ci.yaml
    """
    if output_path is None:
        output_path = config_ci_path

    # Load configuration files
    config = load_yaml(config_path)
    config_ci = load_yaml(config_ci_path)

    print(f"Number of models in config.yaml: {len(config)}")
    print(f"Number of models in config_ci.yaml: {len(config_ci)}")
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

            # Check other top-level fields (excluding base_loss and base_result)
            for key in ["repo_id", "model_type"]:
                if key in model_config and key not in ci_config:
                    other_field_updates[key] = model_config[key]

            # If there are any field updates, record them in updated_models
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

    # Add new models to config_ci
    if new_models:
        print(f"\nAdding {len(new_models)} new model(s) to config_ci.yaml...")
        base_template = get_base_template()
        for model_name, model_config in new_models.items():
            new_model = model_config.copy()
            new_model.update(base_template)
            config_ci[model_name] = new_model
            print(f"  ✓ Added model {model_name} with base_loss and base_result template")

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

            print(f"  ✓ Updated model {model_name}")

    # Save only if there are changes
    if has_changes:
        save_yaml(output_path, config_ci)
        print(f"\n Saved to {output_path}")
        print(f"Updated number of models in config_ci.yaml: {len(config_ci)}")
    else:
        print("No new content found, config files are already synchronized.")

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

    args = parser.parse_args()

    # Handle relative and absolute paths
    origin_config_path = Path(args.origin_config)
    update_config_path = Path(args.update_config)

    if not origin_config_path.exists():
        print(f"Error: {origin_config_path} does not exist")
        sys.exit(1)

    if not update_config_path.exists():
        print(f"Error: {update_config_path} does not exist")
        sys.exit(1)

    print("Starting to compare config files...")
    print(f"Source config file: {origin_config_path}")
    print(f"Target config file: {update_config_path}")
    print()

    result = merge_configs(str(origin_config_path), str(update_config_path))

    if result["new_models"] or result["updated_models"]:
        print("\nChange summary:")
        print(f"  New models: {len(result['new_models'])}")
        print(f"  Updated models: {len(result['updated_models'])}")
    else:
        print("\nNo new content found, config files are already synchronized.")
