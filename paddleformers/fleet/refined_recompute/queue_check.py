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

import queue
from collections import defaultdict

__all__ = ("global_rr_queue_log",)


class RefinedRcomputeQueue:
    """RefinedRcomputeQueue"""

    def __init__(self):
        """init"""
        self.rr_queue = defaultdict(queue.Queue)

    def update(self, queue: queue.Queue, queue_name="unknown"):
        """update"""
        queue_name = f"{queue_name}_{id(queue)}"
        if queue_name in self.rr_queue:
            raise ValueError(f"Queue name '{queue_name}' already exists.")
        self.rr_queue[queue_name] = queue

    def check(self):
        """check"""
        non_empty_queues = [name for name, queue in self.rr_queue.items() if queue.qsize() != 0]
        if non_empty_queues:
            raise ValueError(f"Queues {', '.join(non_empty_queues)} are not empty.")


global_rr_queue_log = RefinedRcomputeQueue()
