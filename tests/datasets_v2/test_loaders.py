# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for datasets_v2 registry and loaders.

Run with: python -m pytest tests/datasets_v2/test_loaders.py -v
"""

import csv
import importlib
import json

# Workaround: broken torchcodec residual in env
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name == "torchcodec":
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec

import pytest
from datasets import Dataset  # noqa: F401

from paddleformers.datasets_v2.loaders import (
    _detect_file_format,
    _resolve_source,
    load_dataset,
    load_datasets,
)
from paddleformers.datasets_v2.registry import (
    _DATASET_REGISTRY,
    DatasetMeta,
    get_dataset_meta,
    list_datasets,
    parse_dataset_string,
    register_dataset,
    register_dataset_info,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the registry before and after each test."""
    _DATASET_REGISTRY.clear()
    yield
    _DATASET_REGISTRY.clear()


@pytest.fixture
def jsonl_file(tmp_path):
    """Create a temporary jsonl file with query/response data."""
    data = [
        {"query": "hello", "response": "hi there"},
        {"query": "what is 1+1", "response": "2"},
        {"query": "tell me a joke", "response": "why did the chicken cross the road?"},
    ]
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    return str(path)


@pytest.fixture
def csv_file(tmp_path):
    """Create a temporary CSV file with alpaca-style data."""
    path = tmp_path / "train.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instruction", "input", "output"])
        writer.writerow(["translate", "hello", "hola"])
        writer.writerow(["summarize", "long text", "short"])
    return str(path)


@pytest.fixture
def messages_jsonl(tmp_path):
    """Create a temporary jsonl file with messages format data."""
    data = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]},
    ]
    path = tmp_path / "messages.jsonl"
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    return str(path)


@pytest.fixture
def dataset_info_json(tmp_path, jsonl_file):
    """Create a temporary dataset_info.json for bulk registration."""
    info = [
        {"name": "ds_alpha", "path": jsonl_file, "preprocessor": "response", "tags": ["sft"]},
        {"name": "ds_beta", "path": jsonl_file, "preprocessor": "auto", "tags": ["sft", "zh"]},
    ]
    path = tmp_path / "dataset_info.json"
    with open(path, "w") as f:
        json.dump(info, f)
    return str(path)


# ============================================================
# Registry Tests
# ============================================================


class TestParseDatasetString:
    def test_simple_name(self):
        spec = parse_dataset_string("alpaca")
        assert spec.name == "alpaca"
        assert spec.sample is None

    def test_name_with_sample(self):
        spec = parse_dataset_string("alpaca#500")
        assert spec.name == "alpaca"
        assert spec.sample == 500

    def test_path_with_sample(self):
        spec = parse_dataset_string("/data/train.jsonl#1000")
        assert spec.name == "/data/train.jsonl"
        assert spec.sample == 1000

    def test_hub_id(self):
        spec = parse_dataset_string("tatsu-lab/alpaca")
        assert spec.name == "tatsu-lab/alpaca"
        assert spec.sample is None

    def test_invalid_sample_not_int(self):
        with pytest.raises(ValueError, match="Invalid sample count"):
            parse_dataset_string("alpaca#abc")

    def test_invalid_sample_zero(self):
        with pytest.raises(ValueError, match="positive"):
            parse_dataset_string("alpaca#0")

    def test_invalid_sample_negative(self):
        with pytest.raises(ValueError, match="positive"):
            parse_dataset_string("alpaca#-1")

    def test_whitespace_stripped(self):
        spec = parse_dataset_string("  alpaca#100  ")
        assert spec.name == "alpaca"
        assert spec.sample == 100


class TestRegisterDataset:
    def test_register_and_get(self):
        meta = DatasetMeta(name="test_ds", path="/tmp/test.jsonl")
        register_dataset(meta)
        result = get_dataset_meta("test_ds")
        assert result is meta
        assert result.path == "/tmp/test.jsonl"

    def test_register_duplicate_raises(self):
        register_dataset(DatasetMeta(name="dup", path="/a"))
        with pytest.raises(ValueError, match="already registered"):
            register_dataset(DatasetMeta(name="dup", path="/b"))

    def test_register_duplicate_exist_ok(self):
        register_dataset(DatasetMeta(name="dup", path="/a"))
        register_dataset(DatasetMeta(name="dup", path="/b"), exist_ok=True)
        assert get_dataset_meta("dup").path == "/b"

    def test_get_nonexistent(self):
        assert get_dataset_meta("no_such_ds") is None

    def test_list_all(self):
        register_dataset(DatasetMeta(name="bb"))
        register_dataset(DatasetMeta(name="aa"))
        assert list_datasets() == ["aa", "bb"]

    def test_list_by_tag(self):
        register_dataset(DatasetMeta(name="a", tags=["sft", "zh"]))
        register_dataset(DatasetMeta(name="b", tags=["dpo"]))
        register_dataset(DatasetMeta(name="c", tags=["sft"]))
        assert list_datasets(tag="sft") == ["a", "c"]
        assert list_datasets(tag="dpo") == ["b"]
        assert list_datasets(tag="nonexist") == []


class TestRegisterDatasetInfo:
    def test_bulk_register(self, dataset_info_json):
        metas = register_dataset_info(dataset_info_json)
        assert len(metas) == 2
        assert get_dataset_meta("ds_alpha") is not None
        assert get_dataset_meta("ds_beta") is not None
        assert "sft" in get_dataset_meta("ds_alpha").tags

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            register_dataset_info("/nonexistent/path.json")

    def test_missing_name_field(self, tmp_path):
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump([{"path": "/tmp/a.jsonl"}], f)
        with pytest.raises(ValueError, match="missing 'name'"):
            register_dataset_info(str(path))


# ============================================================
# Loader Tests
# ============================================================


class TestDetectFileFormat:
    def test_json(self):
        assert _detect_file_format("/a/b.json") == "json"

    def test_jsonl(self):
        assert _detect_file_format("/a/b.jsonl") == "json"

    def test_csv(self):
        assert _detect_file_format("/a/b.csv") == "csv"

    def test_tsv(self):
        assert _detect_file_format("/a/b.tsv") == "csv"

    def test_parquet(self):
        assert _detect_file_format("/a/b.parquet") == "parquet"

    def test_txt(self):
        assert _detect_file_format("/a/b.txt") == "text"

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            _detect_file_format("/a/b.xyz")


class TestResolveSource:
    def test_file(self, jsonl_file):
        assert _resolve_source(jsonl_file) == "file"

    def test_directory(self, tmp_path):
        assert _resolve_source(str(tmp_path)) == "directory"

    def test_hub(self):
        assert _resolve_source("tatsu-lab/alpaca") == "hub"


class TestLoadDataset:
    def test_local_jsonl_no_preprocess(self, jsonl_file):
        ds = load_dataset(jsonl_file, preprocess=False)
        assert len(ds) == 3
        assert "query" in ds.column_names
        assert "response" in ds.column_names

    def test_local_jsonl_auto_preprocess(self, jsonl_file):
        ds = load_dataset(jsonl_file)
        assert len(ds) == 3
        assert ds.column_names == ["messages"]
        row = ds[0]
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"

    def test_local_csv_alpaca_preprocess(self, csv_file):
        ds = load_dataset(csv_file)
        assert len(ds) == 2
        assert ds.column_names == ["messages"]
        msg = ds[0]["messages"]
        assert msg[0]["role"] == "user"
        assert "translate" in msg[0]["content"]
        assert msg[1]["content"] == "hola"

    def test_local_messages_preprocess(self, messages_jsonl):
        ds = load_dataset(messages_jsonl)
        assert len(ds) == 2
        assert ds.column_names == ["messages"]
        assert ds[0]["messages"][0]["content"] == "hi"

    def test_sampling_syntax(self, jsonl_file):
        ds = load_dataset(jsonl_file + "#2", preprocess=False)
        assert len(ds) == 2

    def test_oversampling_syntax(self, jsonl_file):
        ds = load_dataset(jsonl_file + "#10", preprocess=False)
        assert len(ds) == 10

    def test_registered_dataset(self, jsonl_file):
        register_dataset(
            DatasetMeta(
                name="my_registered",
                path=jsonl_file,
                preprocessor="response",
                tags=["test"],
            )
        )
        ds = load_dataset("my_registered")
        assert len(ds) == 3
        assert ds.column_names == ["messages"]

    def test_registered_with_none_preprocessor(self, jsonl_file):
        register_dataset(
            DatasetMeta(
                name="raw_ds",
                path=jsonl_file,
                preprocessor=None,
            )
        )
        ds = load_dataset("raw_ds")
        assert "query" in ds.column_names

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.jsonl", preprocess=False)

    def test_streaming_mode(self, jsonl_file):
        ds = load_dataset(jsonl_file, streaming=True, preprocess=False)
        from datasets import IterableDataset

        assert isinstance(ds, IterableDataset)
        rows = list(ds.take(2))
        assert len(rows) == 2

    def test_streaming_with_sample(self, jsonl_file):
        ds = load_dataset(jsonl_file + "#2", streaming=True, preprocess=False)
        rows = list(ds)
        assert len(rows) == 2


class TestLoadDatasets:
    def test_single_dataset(self, jsonl_file):
        ds = load_datasets(jsonl_file, preprocess=False)
        assert len(ds) == 3

    def test_multiple_concat(self, jsonl_file, csv_file):
        ds = load_datasets([jsonl_file, csv_file], preprocess=False)
        assert len(ds) == 5  # 3 jsonl + 2 csv

    def test_multiple_with_preprocess(self, jsonl_file, messages_jsonl):
        ds = load_datasets([jsonl_file, messages_jsonl])
        assert len(ds) == 5  # 3 + 2
        assert ds.column_names == ["messages"]


class TestLoadDirectory:
    def test_directory_with_jsonl_files(self, tmp_path):
        data = [{"query": "q1", "response": "r1"}, {"query": "q2", "response": "r2"}]
        for i in range(2):
            path = tmp_path / f"shard_{i}.jsonl"
            with open(path, "w") as f:
                for row in data:
                    f.write(json.dumps(row) + "\n")
        ds = load_dataset(str(tmp_path), preprocess=False)
        assert len(ds) == 4  # 2 files x 2 rows
