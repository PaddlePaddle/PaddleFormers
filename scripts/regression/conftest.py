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


def pytest_addoption(parser):
    parser.addoption("--models", action="store", default="glm", help="eg: --models=llama,qwen3")


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
