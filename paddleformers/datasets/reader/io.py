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

import csv
import time

import orjson
import pyarrow.parquet as pq


def load_json(file_path):
    print(f"json file path: {file_path}")
    try:
        with open(file_path, "rb") as f:
            return orjson.loads(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"file {file_path} not exists")
    except orjson.JSONDecodeError:
        pass  # fallback to JSONL

    return _load_json_lazy(file_path)


def _load_json_lazy(file_path):
    """Lazy load JSONL file as generator to reduce memory usage."""
    t_start = time.perf_counter()
    count = 0
    try:
        with open(file_path, "rb") as file:
            for i, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    yield orjson.loads(line)
                    count += 1
                except orjson.JSONDecodeError as e:
                    raise ValueError(f"JSONL parse error at line {i}: {e}")
    finally:
        elapsed = time.perf_counter() - t_start
        print(
            f"[load json] done. total: {count} lines, elapsed: {elapsed:.2f}s, speed: {count/elapsed if elapsed > 0 else float('inf'):.0f} lines/s"
        )


def load_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"file {file_path} not exists")
    except IOError as e:
        raise ValueError(f"file {file_path} load failed: {e}")


def load_csv(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return list(csv.reader(f))
    except FileNotFoundError:
        raise FileNotFoundError(f"file {file_path} not exists")
    except (IOError, csv.Error) as e:
        raise ValueError(f"file {file_path} load failed: {e}")


def load_parquet(file_path):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"file {file_path} not exists")
    except Exception as e:
        raise ValueError(f"file {file_path} load failed: {e}")
