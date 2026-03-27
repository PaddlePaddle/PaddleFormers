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

import os

import pytest
import yaml

CONFIG_PATH = "./scripts/regression/config.yaml"


def get_all_models_from_config() -> list:
    """
    return all models from config.yaml
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.keys())


def get_model_type(model_key: str) -> str:
    """Get model_type from config for a given model_key."""
    if not os.path.exists(CONFIG_PATH):
        return "text"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if model_key in data:
        return data[model_key].get("model_type", "text")
    return "text"


def pytest_addoption(parser):
    parser.addoption("--models", action="store", default="", help="eg: --models=llama,qwen3")
    parser.addoption(
        "--update-baseline",
        action="store",
        default="",
        help="Update baseline values eg: --update-baseline=all or glm_moe",
    )


@pytest.fixture
def selected_models(request):
    cli_value = request.config.getoption("--models")
    if cli_value == "all":
        return get_all_models_from_config()
    return [m.strip() for m in cli_value.split(",")]


def pytest_generate_tests(metafunc):
    if "model_key" in metafunc.fixturenames:
        cli_value = metafunc.config.getoption("--models")
        if cli_value == "all":
            models = get_all_models_from_config()
        else:
            models = [m.strip() for m in cli_value.split(",")]
        metafunc.parametrize("model_key", models)


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on model_type marker and model config.

    Tests marked with @pytest.mark.model_type("vl") will only run for VL models.
    Tests marked with @pytest.mark.model_type("text") will only run for text models.
    Default to "text".
    """
    for item in items:
        # Get model_key from test parameters
        model_key = None
        if hasattr(item, "callspec") and "model_key" in item.callspec.params:
            model_key = item.callspec.params["model_key"]

        if model_key is None:
            continue

        # Get model_type from config
        actual_model_type = get_model_type(model_key)

        # Get required model_type from marker (default to "text")
        marker = item.get_closest_marker("model_type")
        required_model_type = marker.args[0] if marker else "text"

        # Skip if model_type doesn't match
        if actual_model_type != required_model_type:
            item.add_marker(pytest.mark.skip())
