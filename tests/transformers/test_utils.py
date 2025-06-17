# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import json
import os
import unittest

from paddlenlp.transformers import (
    AlbertForTokenClassification,
    AlbertModel,
    BertModel,
    utils,
)
from paddlenlp.transformers.bert.modeling import BertForTokenClassification


class TestUtils(unittest.TestCase):
    """Unittest for paddlenlp.transformers.utils.py module"""

    def test_find_transformer_model_type(self):
        """test for `find_transformer_model_type`"""
        self.assertEqual(utils.find_transformer_model_type(AlbertModel), "albert")
        self.assertEqual(utils.find_transformer_model_type(AlbertForTokenClassification), "albert")
        self.assertEqual(utils.find_transformer_model_type(BertModel), "bert")
        self.assertEqual(utils.find_transformer_model_type(BertForTokenClassification), "bert")


def check_json_file_has_correct_format(file_path):
    with open(file_path, "r") as f:
        try:
            json.load(f)
        except Exception as e:
            raise Exception(f"{e}: the json file should be a valid json")


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
