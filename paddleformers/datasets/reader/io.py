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
import json

import pyarrow.parquet as pq


def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"file {file_path} not exists")
    except json.JSONDecodeError:
        pass  # fallback to JSONL

    res = []
    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                res.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"JSONL parse error at line {i}: {e.msg}", e.doc, e.pos)
    return res


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
