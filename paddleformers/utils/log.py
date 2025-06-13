# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import atexit
import contextlib
import functools
import json
import logging
import multiprocessing
import os
import signal
import threading
import time

import colorlog

loggers = {}

log_config = {
    "DEBUG": {"level": 10, "color": "purple"},
    "INFO": {"level": 20, "color": "green"},
    "TRAIN": {"level": 21, "color": "cyan"},
    "EVAL": {"level": 22, "color": "blue"},
    "WARNING": {"level": 30, "color": "yellow"},
    "ERROR": {"level": 40, "color": "red"},
    "CRITICAL": {"level": 50, "color": "bold_red"},
}


class Logger(object):
    """
    Default logger in PaddleFormers

    Args:
        name(str) : Logger name, default is 'PaddleFormers'
    """

    def __init__(self, name: str = None):
        name = "PaddleFormers" if not name else name
        self.logger = logging.getLogger(name)

        for key, conf in log_config.items():
            logging.addLevelName(conf["level"], key)
            self.__dict__[key] = functools.partial(self.__call__, conf["level"])
            self.__dict__[key.lower()] = functools.partial(self.__call__, conf["level"])

        self.format = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s",
            log_colors={key: conf["color"] for key, conf in log_config.items()},
        )

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = "DEBUG"
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._is_enable = True

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    def set_level(self, log_level: str):
        assert log_level in log_config, f"Invalid log level. Choose among {log_config.keys()}"
        self.logger.setLevel(log_level)

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    def __call__(self, log_level: str, msg: str):
        if not self.is_enable:
            return

        self.logger.log(log_level, msg, stacklevel=2)

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator

    @contextlib.contextmanager
    def processing(self, msg: str, interval: float = 0.1):
        """
        Continuously print a progress bar with rotating special effects.

        Args:
            msg(str): Message to be printed.
            interval(float): Rotation interval. Default to 0.1.
        """
        end = False

        def _printer():
            index = 0
            flags = ["\\", "|", "/", "-"]
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator("\r"):
                    self.info("{}: {}".format(msg, flag))
                time.sleep(interval)
                index += 1

        t = threading.Thread(target=_printer)
        t.start()
        yield
        end = True

    @functools.lru_cache(None)
    def warning_once(self, *args, **kwargs):
        """
        This method is identical to `logger.warning()`, but will emit the warning with the same message only once

        Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
        The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
        another type of cache that includes the caller frame information in the hashing function.
        """
        self.warning(*args, **kwargs)


class MetricsDumper(object):
    """
    Default JSONDumper in PaddleFormers

    Args:
        name(str) : Logger name, default is 'PaddleFormers'
    """

    def __init__(self, filename: str = None):
        self.filename = "./training_metrics" if not filename else filename
        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._write_json, args=(self.queue,))
        self.process.start()

        # process pid and the start time
        self.pid = os.getpid()

        # Ensure subprocess exits when main process is interrupted
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        atexit.register(self.close)

    def append(self, data):
        """
        Append a JSON object to the queue for background writing.

        :param data: The JSON object to append.
        """
        data.update({"metrics_dumper_pid": self.pid, "metrics_dumper_timestamp": int(time.time() * 1000)})
        self.queue.put(data)

    def _write_json(self, queue):
        """
        Write JSON objects from the queue to the file in the background.

        :param queue: The multiprocessing queue from which to read data.
        """
        while True:
            try:
                metrics = queue.get(timeout=10)  # Timeout to allow graceful shutdown
                if metrics is None:
                    break
                with open(self.filename, "a") as writer:
                    writer.write(json.dumps(metrics) + "\n")
            except:
                continue

    def _signal_handler(self, sig, frame):
        """
        Handle signals to ensure graceful shutdown.

        :param sig: Signal number.
        :param frame: Current stack frame.
        """
        print(f"Received signal {sig}. Shutting down...")
        self.close()

    def close(self):
        """
        Close the background process and ensure all data is written.
        """
        self.queue.put(None)  # Signal the process to exit
        self.process.join()


logger = Logger()
