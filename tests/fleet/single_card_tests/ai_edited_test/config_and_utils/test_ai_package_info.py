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
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest


class TestPackageInfo(unittest.TestCase):
    """Tests for paddleformers.fleet.package_info module."""

    def test_package_name(self):
        from paddleformers.fleet.package_info import __package_name__

        self.assertEqual(__package_name__, "paddleformers.fleet")

    def test_contact_names(self):
        from paddleformers.fleet.package_info import __contact_names__

        self.assertEqual(__contact_names__, "PaddlePaddle")

    def test_contact_emails(self):
        from paddleformers.fleet.package_info import __contact_emails__

        self.assertEqual(__contact_emails__, "Paddle-better@baidu.com")

    def test_homepage(self):
        from paddleformers.fleet.package_info import __homepage__

        self.assertIn("paddlepaddle.org.cn", __homepage__)

    def test_repository_url(self):
        from paddleformers.fleet.package_info import __repository_url__

        self.assertIn("PaddleFleet", __repository_url__)
        self.assertIn("github.com", __repository_url__)

    def test_download_url(self):
        from paddleformers.fleet.package_info import __download_url__

        self.assertIn("PaddleFleet/releases", __download_url__)

    def test_description(self):
        from paddleformers.fleet.package_info import __description__

        self.assertIsInstance(__description__, str)
        self.assertIn("PaddleFleet", __description__)

    def test_license(self):
        from paddleformers.fleet.package_info import __license__

        self.assertEqual(__license__, ("Apache Software License",))

    def test_keywords(self):
        from paddleformers.fleet.package_info import __keywords__

        self.assertIsInstance(__keywords__, str)
        self.assertIn("paddlepaddle", __keywords__)
        self.assertIn("transformer", __keywords__)

    def test_version_exists(self):
        from paddleformers.fleet.package_info import __version__

        self.assertIsInstance(__version__, str)
        self.assertTrue(len(__version__) > 0)

    def test_commit_exists(self):
        from paddleformers.fleet.package_info import commit

        self.assertIsInstance(commit, str)
