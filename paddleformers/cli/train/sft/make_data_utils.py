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

"""SFT utils"""


class DataGenerator:
    """Generates an infinite stream of examples"""

    def __init__(self, data_source):
        """
            Initializes the iterator for a given data source.

        Args:
            data_source : IterableDataset

        Returns:
            None. - Initialization only. No return value.
        """
        self.data_source_iter = iter(data_source)
        self.data_source = data_source

    def __iter__(self):
        """
        Returns:
            Iterator: The iterator object itself.
        """
        return self

    def __next__(self):
        """
        Get the next item from the iterator. If there are no more items left, reset the iterator.

        Returns:
            Any: The next item from the iterator.
        """
        try:
            return next(self.data_source_iter)
        except StopIteration:
            self.data_source_iter = iter(self.data_source)
            return next(self.data_source_iter)
