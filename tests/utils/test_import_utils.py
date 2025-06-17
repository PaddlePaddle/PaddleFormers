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

import unittest

from paddlenlp.utils import install_package, uninstall_package

from ..testing_utils import skip_for_none_ce_case


class ImportUntilsTest(unittest.TestCase):
    @skip_for_none_ce_case
    def test_install_specific_package(self):
        install_package("loguru", "0.6.0")
        from loguru import __version__

        assert __version__ == "0.6.0"

        install_package("loguru", "0.5.3")
        from loguru import __version__

        assert __version__ == "0.5.3"

    def test_uninstall_package(self):
        uninstall_package("paddlenlp")
        uninstall_package("empty-package")
