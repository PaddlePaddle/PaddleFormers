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
"""Parsing exceptions"""


class DataSetNoFilePathNorRepoIDError(Exception):
    """Exception class for no file_path nor repo_id found when create_dataset."""

    def __init__(self, msg):
        """
        Init exception class for no file_path nor repo_id found when create_dataset
        Args:
            msg (str): exception message
        """
        super().__init__(msg)


class DataSetFileNotFoundError(Exception):
    """Exception class for no dataset file found."""

    def __init__(self, msg):
        """
        Init exception class for no dataset file found
        Args:
            msg (str): exception message
        """
        super().__init__(msg)


class DataSetFileCannotOpenError(Exception):
    """Exception class for cannot open the file."""

    def __init__(self, msg):
        """
        Init exception class for cannot open the file
        Args:
            msg (str): exception message
        """
        super().__init__(msg)


class DataSetParseError(Exception):
    """Exception class for parsing error."""

    def __init__(self, msg):
        """
        Init exception class for parsing error
        Args:
            msg (str): exception message
        """
        super().__init__(msg)


class DataSetFormattingNotSupportedError(Exception):
    """Exception class for formatting not supported."""

    def __init__(self, msg):
        """
        Init exception class for formatting not supported
        Args:
            msg (str): exception message
        """
        super().__init__(msg)
