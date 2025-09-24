# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import copy
import gc
import inspect
import json
import os
import subprocess
import sys
import unittest
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
import yaml

from paddleformers.trainer.argparser import strtobool
from paddleformers.utils.download import DownloadSource
from paddleformers.utils.import_utils import is_package_available, is_paddle_available

__all__ = ["get_vocab_list", "stable_softmax", "cross_entropy"]


class PaddleFormersModelTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()


def get_vocab_list(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = [vocab.rstrip("\n").split("\t")[0] for vocab in f.readlines()]
        return vocab_list


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1 :]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def softmax_with_cross_entropy(logits, label, soft_label=False, axis=-1, ignore_index=-1):
    softmax = np.apply_along_axis(stable_softmax, -1, logits)
    return cross_entropy(softmax, label, soft_label, axis, ignore_index)


def assert_raises(Error=AssertionError):
    def assert_raises_error(func):
        def wrapper(self, *args, **kwargs):
            with self.assertRaises(Error):
                func(self, *args, **kwargs)

        return wrapper

    return assert_raises_error


def create_test_data(file=__file__):
    dir_path = os.path.dirname(os.path.realpath(file))
    test_data_file = os.path.join(dir_path, "dict.txt")
    with open(test_data_file, "w") as f:
        vocab_list = [
            "[UNK]",
            "AT&T",
            "B超",
            "c#",
            "C#",
            "c++",
            "C++",
            "T恤",
            "A座",
            "A股",
            "A型",
            "A轮",
            "AA制",
            "AB型",
            "B座",
            "B股",
            "B型",
            "B轮",
            "BB机",
            "BP机",
            "C盘",
            "C座",
            "C语言",
            "CD盒",
            "CD机",
            "CALL机",
            "D盘",
            "D座",
            "D版",
            "E盘",
            "E座",
            "E化",
            "E通",
            "F盘",
            "F座",
            "G盘",
            "H盘",
            "H股",
            "I盘",
            "IC卡",
            "IP卡",
            "IP电话",
            "IP地址",
            "K党",
            "K歌之王",
            "N年",
            "O型",
            "PC机",
            "PH值",
            "SIM卡",
            "U盘",
            "VISA卡",
            "Z盘",
            "Q版",
            "QQ号",
            "RSS订阅",
            "T盘",
            "X光",
            "X光线",
            "X射线",
            "γ射线",
            "T恤衫",
            "T型台",
            "T台",
            "4S店",
            "4s店",
            "江南style",
            "江南Style",
            "1号店",
            "小S",
            "大S",
            "阿Q",
            "一",
            "一一",
            "一一二",
            "一一例",
            "一一分",
            "一一列举",
            "一一对",
            "一一对应",
            "一一记",
            "一一道来",
            "一丁",
            "一丁不识",
            "一丁点",
            "一丁点儿",
            "一七",
            "一七八不",
            "一万",
            "一万一千",
            "一万一千五百二十颗",
            "一万一千八百八十斤",
            "一万一千多间",
            "一万一千零九十五册",
            "一万七千",
            "一万七千余",
            "一万七千多",
            "一万七千多户",
            "一万万",
        ]
        for vocab in vocab_list:
            f.write("{}\n".format(vocab))
    return test_data_file


def get_bool_from_env(key, default_value=False):
    if key not in os.environ:
        return default_value
    value = os.getenv(key)
    try:
        value = strtobool(value)
    except ValueError:
        raise ValueError(f"If set, {key} must be yes, no, true, false, 0 or 1 (case insensitive).")
    return value


_run_slow_test = get_bool_from_env("RUN_SLOW_TEST")


def slow(test):
    """
    Mark a test which spends too much time.
    Slow tests are skipped by default. Execute the command `export RUN_SLOW_TEST=True` to run them.
    """
    if not _run_slow_test:
        return unittest.skip("test spends too much time")(test)
    else:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            paddle.device.cuda.empty_cache()
        return test


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


def nested_simplify(obj, decimals=3):
    """
    Simplifies an object by rounding float numbers, and downcasting tensors/numpy arrays to get simple equality test
    within tests.
    """
    import numpy as np

    if isinstance(obj, list):
        return [nested_simplify(item, decimals) for item in obj]
    elif isinstance(obj, np.ndarray):
        return nested_simplify(obj.tolist())
    elif isinstance(obj, Mapping):
        return {nested_simplify(k, decimals): nested_simplify(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (str, int, np.int64)):
        return obj
    elif obj is None:
        return obj
    elif isinstance(obj, paddle.Tensor):
        return nested_simplify(obj.numpy().tolist(), decimals)
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, (np.int32, np.float32)):
        return nested_simplify(obj.item(), decimals)
    else:
        raise Exception(f"Not supported: {type(obj)}")


def require_package(*package_names):
    """decorator which can detect that it will require the specific package

    Args:
        package_name (str): the name of package
    """

    def decorator(func):
        for package_name in package_names:
            if not is_package_available(package_name):
                return unittest.skip(f"package<{package_name}> not found, so to skip this test")(func)
        return func

    return decorator


def skip_platform(*platform):
    """decorator which can detect that it will skip the specific platform

    Args:
        platform (str): the name of platform, including win32, cygwin, linux, and darwin
    """

    def decorator(func):
        for plat in platform:
            if sys.platform.startswith(plat):
                return unittest.skip(f"platform<{plat}> matched, so to skip this test")(func)
        return func

    return decorator


def is_slow_test() -> bool:
    """check whether is the slow test

    Returns:
        bool: whether is the slow test
    """
    return os.getenv("RUN_SLOW_TEST") is not None


def load_test_config(config_file: str, key: str, sub_key: str = None) -> dict | None:
    """parse config file to argv

    Args:
        config_dir (str, optional): the path of config file. Defaults to None.
        config_name (str, optional): the name key in config file. Defaults to None.
    """
    # 1. load the config with key and test env(default, test)
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert key in config, f"<{key}> should be the top key in configuration file"
    config = config[key]

    mode_key = "slow" if is_slow_test() else "default"

    if mode_key not in config:
        return None

    # 2. load base common config
    base_config = config.get("base", {})

    config = config.get(mode_key, {})
    config.update(base_config)

    # 3. load sub key config
    sub_config = config.get(sub_key, {})
    config.update(sub_config)

    # remove dict value
    for key in list(config.keys()):
        if isinstance(config[key], dict):
            config.pop(key)

    return config


def construct_argv(config: dict) -> list[str]:
    """construct argv by configs

    Args:
        config (dict): the config data

    Returns:
        list[str]: the argvs
    """
    # get current test
    # refer to: https://docs.pytest.org/en/latest/example/simple.html#pytest-current-test-environment-variable
    current_test = "tests/__init__.py"
    if "PYTEST_CURRENT_TEST" in os.environ:
        current_test = os.getenv("PYTEST_CURRENT_TEST").split("::")[0]

    argv = [current_test]
    for key, value in config.items():
        argv.append(f"--{key}")
        argv.append(str(value))

    return argv


@contextmanager
def argv_context_guard(config: dict):
    """construct argv by config

    Args:
        config (dict): the configuration to argv
    """
    old_argv = copy.deepcopy(sys.argv)
    argv = construct_argv(config)
    sys.argv = argv
    yield
    sys.argv = old_argv[:1]


def update_params(json_file: str, params: dict):
    """update params in json file

    Args:
        json_file (str): the path of json file
        params (dict): the parameters need to update
    """
    with open(json_file, "r") as f:
        data = json.load(f)
        data.update(params)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class SubprocessCallException(Exception):
    pass


def run_command(command: list[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


def skip_for_none_ce_case(test_case):
    """
    There are too many test case, we need skip for none CE envirmonet.
    """
    import os

    ce_env = strtobool(os.getenv("CE_TEST_ENV", "0"))
    if not ce_env:
        return unittest.skip("test skip for NONE CE case. If you want run this ci, please export CE_TEST_ENV=1 ")(
            test_case
        )

    return test_case


def require_paddle_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PaddlePaddle). These tests are skipped on a machine without
    multiple GPUs.

    To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    if not is_paddle_available():
        return unittest.skip("test requires PaddlePaddle")(test_case)

    import paddle

    return unittest.skipUnless(paddle.device.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)


def require_paddle_non_multi_gpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PaddlePaddle).
    """
    if not is_paddle_available():
        return unittest.skip("test requires PaddlePaddle")(test_case)

    import paddle

    return unittest.skipUnless(paddle.device.cuda.device_count() < 2, "test requires 0 or 1 GPU")(test_case)


def require_paddle_at_least_2_gpu(test_case):
    """
    Decorator marking a test that requires >= 2 GPU setup (in PaddlePaddle).
    """
    if not is_paddle_available():
        return unittest.skip("test requires PaddlePaddle")(test_case)

    import paddle

    return unittest.skipUnless(paddle.device.cuda.device_count() >= 2, "test requires at least 2 GPUs")(test_case)


def require_paddle_at_least_8_gpu(test_case):
    """
    Decorator marking a test that requires >= 8 GPU setup (in PaddlePaddle).
    """
    if not is_paddle_available():
        return unittest.skip("test requires PaddlePaddle")(test_case)

    import paddle

    return unittest.skipUnless(paddle.device.cuda.device_count() >= 8, "test requires at least 8 GPUs")(test_case)


def require_paddle_up_to_2_gpus(test_case):
    """
    Decorator marking a test that requires 0 or 1 or 2 GPU setup (in PaddlePaddle).
    """
    if not is_paddle_available():
        return unittest.skip("test requires PaddlePaddle")(test_case)

    import paddle

    return unittest.skipUnless(paddle.device.cuda.device_count() < 3, "test requires 0 or 1 or 2 GPUs")(test_case)


def require_gpu(min_gpus: int = 1):
    def actual_decorator(func):
        gpu_count = paddle.device.cuda.device_count()
        print("gpu count: ", gpu_count)
        if gpu_count < min_gpus:
            return unittest.skip(f"test requires {min_gpus} GPUs")(func)

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        return wrapper

    return actual_decorator


class GPUsTesting(unittest.TestCase):
    def init_dist_env(self, config: dict = {}):
        world_size = paddle.distributed.get_world_size()
        strategy = fleet.DistributedStrategy()
        hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        hybrid_configs.update(config)
        strategy.hybrid_configs = hybrid_configs

        fleet.init(is_collective=True, strategy=strategy)
        fleet.get_hybrid_communicate_group()


def set_proxy(download_hub: DownloadSource = None):
    """
    set network proxy for downloading model from aistudio/huggingface/modelscope
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if download_hub is None:
                return func(*args, **kwargs)
            elif download_hub == DownloadSource.HUGGINGFACE:
                if "HF_PROXY_PATH" not in os.environ:
                    print(
                        "`HF_PROXY_PATH` environment variable does not defined before using `set_proxy`, please define it first"
                    )
                proxy_path = os.path.abspath(os.environ["HF_PROXY_PATH"])
            elif download_hub == DownloadSource.AISTUDIO:
                if "AISTUDIO_PROXY_PATH" not in os.environ:
                    print(
                        "`AISTUDIO_PROXY_PATH` environment variable does not defined before using `set_proxy`, please define it first"
                    )
                proxy_path = os.path.abspath(os.environ["AISTUDIO_PROXY_PATH"])
            elif download_hub == DownloadSource.MODELSCOPE:
                if "AISTUDIO_PROXY_PATH" not in os.environ:
                    print(
                        "`AISTUDIO_PROXY_PATH` environment variable does not defined before using `set_proxy`, please define it first"
                    )
                proxy_path = os.path.abspath(
                    os.environ["AISTUDIO_PROXY_PATH"]
                )  # proxy_aistudio also suit for modelscope

            print(f"set proxy for {download_hub}, proxy path: {proxy_path}")
            command = f". {proxy_path} && env"

            proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            out, _ = proc.communicate()

            proxy_env = {}
            for line in out.decode().splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                proxy_env[key] = value

            ori_env = {}
            proxy_vars = ["http_proxy", "https_proxy", "no_proxy"]
            if download_hub == DownloadSource.AISTUDIO:
                proxy_vars.extend(["STUDIO_GIT_HOST", "STUDIO_CDN_HOST"])
            for key in proxy_vars:
                if key in proxy_env:
                    ori_env[key] = os.environ.get(key, "")
                    os.environ[key] = proxy_env[key]

            try:
                return func(*args, **kwargs)
            finally:
                for key, old_value in ori_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

        return wrapper

    return decorator
