# 当前文件格式支持

当前支持 json、jsonl、parquet 三种格式，需保证文件名后缀和文件内容保持一致

# 新增文件格式支持

在 paddleformers/datasets/reader/io.py 里面实现各种类型文件的读取函数，例如 parquet 文件：
```python
def load_parquet(file_path):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        return df
    except Exception:
        raise ValueError(f"file {file_path} load failed")
```

然后在 paddleformers/datasets/reader/file_reader.py 中BaseReader 的self.loader_map 中进行注册：
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

当前支持 erniekit 和messages 两种格式的数据

# 新增数据格式支持

在 paddleformers/datasets/reader/convertor.py 里面实现各种格式的转换函数，统一转换成 messages 格式，例如 erniekit 格式转 messages 格式：
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


然后在 paddleformers/datasets/reader/file_reader.py 中BaseReader 的self.convertor_map 中进行注册：
```python
self.convertor_map = {
    "erniekit": erniekit_convertor,
    "messages": messages_convertor,
}
```
