import os
import json
import pandas as pd
import pyarrow.parquet as pq

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(file_path):
    try:
        res = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                res.append(json.loads(line.strip()))
            return res
    except FileNotFoundError:
        raise FileNotFoundError(f"file {file_path} not exists")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON parse error: {e.msg}", e.doc, e.pos)

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return list(csv.reader(f))

def load_parquet(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df
