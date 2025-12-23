# 当前文件格式支持

当前支持json、jsonl、parquet三种格式，需保证文件名后缀和文件内容保持一致

# 新增文件格式支持

在paddleformers/datasets/reader/io.py里面实现各种类型文件的读取函数，例如parquet文件：
```python
def load_parquet(file_path):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        return df
    except Exception:
        raise ValueError(f"file {file_path} load failed")
```

然后在paddleformers/datasets/reader/file_reader.py中BaseReader的self.loader_map中进行注册：
```python
self.loader_map = {
    ".json": load_json,
    ".jsonl": load_json,
    ".txt": load_txt,
    ".csv": load_csv,
    ".parquet": load_parquet,
}
```

# 当前数据格式支持

当前支持erniekit和messages两种格式的数据

# 新增数据格式支持

在paddleformers/datasets/reader/convertor.py里面实现各种格式的转换函数，统一转换成messages格式，例如erniekit格式转messages格式：
```python
def erniekit_convertor(item):
    # erniekit dpo data
    if "src" in item and "tgt" in item and "response" in item:
        res = convert_dpo_txt_data(item)
    # erniekit sft data
    elif "src" in item and "tgt" in item:
        res = convert_txt_data(item)
    # erniekit pretraining data
    elif "text" in item:
        res = convert_pretraining_data(item)
    # erniekit multi modal data
    else:
        res = convert_mm_data(item)
    return res
```


然后在paddleformers/datasets/reader/file_reader.py中BaseReader的self.convertor_map中进行注册：
```python
self.convertor_map = {
    "erniekit": erniekit_convertor,
    "messages": messages_convertor,
}
```
