# datasets_v2 架构设计文档

## 1. 概述

datasets_v2 是 PaddleFormers 的新一代数据处理模块，参考 ms-swift 的 dataset 架构，基于 HuggingFace datasets 库构建，目标是提供统一的、可扩展的数据加载与预处理管线。

**核心设计原则：**
- 显式 schema 作为管线契约（所有模块基于同一份字段定义交互）
- 基于 HF Dataset.map() 的批处理框架（兼容 Map 式和 Iterable 流式）
- 多格式归一化：无论输入是什么格式，输出统一为 messages 标准格式
- 面向多模态 + 纯文本双重场景

**对应 ms-swift 源码：** `/ms-swift/swift/dataset/`

---

## 2. 模块划分与开发顺序

```
datasets_v2/
├── schema.py              # Step 1: 数据契约（字段定义、类型、校验）
├── preprocessors/         # Step 2: 数据预处理（格式归一化）
│   ├── base.py            #   批处理框架
│   ├── response.py        #   query/response 格式
│   ├── messages.py        #   对话/ShareGPT 格式
│   ├── extra.py           #   Alpaca 等薄封装
│   ├── auto.py            #   自动格式检测与分发
│   └── __init__.py
├── ops.py                 # Step 3: 数据集操作（sample/split/concat/shuffle/shard）
├── registry.py            # Step 4: 数据集注册表
├── loaders.py             # Step 4: 数据加载器（本地/远程/多格式）
├── builder.py             # Step 5: load_dataset 统一入口
├── adapters.py            # Step 6: 训练框架适配（Paddle DataLoader 对接）
└── __init__.py            # 公共 API 导出
```

---

## 3. Step 1: schema.py — 数据契约

### 3.1 标准字段（29 个）

| 类别 | 字段 |
|------|------|
| 核心对话 | messages |
| 多模态 | images, videos, audios |
| 工具调用 | tools |
| Grounding | objects |
| 偏好学习 | rejected_/positive_/negative_ × 上述 6 个 |
| 标量 | rejected_response, label, channel, margin, teacher_prompt |

### 3.2 消息格式

```python
messages = [
    {"role": "system", "content": "...", "loss": False},   # loss 可选
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]
```

合法 role: `system`, `user`, `assistant`, `tool_call`, `tool_response`, `tool`

### 3.3 多模态数据格式

```python
images: List[{"bytes": Optional[bytes], "path": str}]
videos: List[str]   # 路径列表
audios: List[str]   # 路径列表
```

### 3.4 关键函数

- `check_messages(messages)` — 校验 messages 结构合法性
- `cast_images(images)` — 将 str/dict/list 归一化为 `List[ImageMedia]`
- `cast_media_list(media)` — 将 str 归一化为 `List[str]`
- `remove_non_standard_keys(row)` — 过滤非标准字段

---

## 4. Step 2: preprocessors/ — 数据预处理

### 4.1 BasePreprocessor（base.py）

核心机制：将 `HF Dataset.map(batched=True)` 与逐行处理逻辑桥接。

```
Dataset.map(batched=True, batch_size=1000)
    │
    ▼
_batched_preprocess(batched_row)
    │  _batched_to_rows: 列式 → 行式
    │  for row: preprocess(row) → dict / list[dict] / None
    │  _check_and_cast: 校验 + 类型归一化
    │  _rows_to_batched: 行式 → 列式
    ▼
返回给 HF 拼接
```

**关键设计：**
- `preprocess(row)` 由子类实现，返回 dict（一行）、list[dict]（展开）、None（跳过）
- 容错：`strict=False` 时单行出错只 warning + 跳过，不中断整体
- 列重命名：`DEFAULT_COLUMN_ALIASES` + 用户自定义 `columns` 参数
- `num_proc` 仅对 HfMapDataset 生效（IterableDataset 不支持多进程）

### 4.2 ResponsePreprocessor（response.py）

**输入格式：**
```python
{"query": "...", "response": "...", "system": "...", "history": [["q1","r1"], ...]}
```

**处理逻辑：**
1. pop response（若为 list 取第一个）
2. pop history（支持字符串格式自动 `ast.literal_eval`）
3. `history.append([query, response])`
4. `history_to_messages(history, system)` → 标准 messages

**列别名：** query ← prompt/input/instruction/question/problem; response ← answer/output/targets/text/completion/content

### 4.3 MessagesPreprocessor（messages.py）

**输入格式（三种）：**

标准格式：
```python
{"messages": [{"role": "user", "content": "..."}, ...]}
```

非标准 key：
```python
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

ShareGPT paired 格式：
```python
{"conversations": [{"human": "...", "gpt": "..."}, ...]}
```

**处理逻辑：**
1. 自动检测 role_key（role/from）、content_key（content/value）
2. 判断是否 ShareGPT 格式（第一条消息无 role/content 字段）
3. 统一角色名：human→user, gpt→assistant, function_call→tool_call 等
4. 插入 system 消息

### 4.4 AlpacaPreprocessor（extra.py）

继承 ResponsePreprocessor，仅做字段拼接：
```python
query = f"{instruction}\n{input}" if both else instruction or input
row['response'] = output
→ 委托给 ResponsePreprocessor.preprocess()
```

### 4.5 AutoPreprocessor（auto.py）

分发逻辑：
```python
if 'messages'/'conversation'/'conversations' in columns → MessagesPreprocessor
elif 'instruction' + 'input' in columns → AlpacaPreprocessor
else → ResponsePreprocessor
```

---

## 5. Step 3-7: 待实现模块

### 5.1 ops.py — 数据集操作

对应 ms-swift 的 `dataset/utils.py`，提供：
- `sample_dataset(dataset, n)` — 采样
- `split_dataset(dataset, ratio)` — 训练/验证拆分
- `concat_datasets([ds1, ds2, ...])` — 合并
- `interleave_datasets([ds1, ds2, ...], probs)` — 按比例交错
- `shuffle_dataset(dataset, seed)` — 打乱
- `shard_dataset(dataset, num_shards, index)` — 分片

### 5.2 registry.py — 数据集注册表

- 数据集名称 → 元信息（路径/URL、preprocessor 类型、列映射）的注册机制
- 支持内置数据集和用户自定义注册
- 对应 ms-swift 的 `dataset/register.py`

### 5.3 loaders.py — 数据加载

- 本地文件加载（json/jsonl/csv/parquet）
- HuggingFace Hub 加载
- 自定义 URL 下载
- 统一返回 HF Dataset/IterableDataset

### 5.4 builder.py — load_dataset 入口

用户侧唯一入口：
```python
from paddleformers.datasets_v2 import load_dataset
dataset = load_dataset("dataset_name", split="train", streaming=False)
```
内部串联 registry → loader → preprocessor → ops

### 5.5 adapters.py — 训练框架适配

- 将 HF Dataset 转为 Paddle DataLoader 可消费的格式
- 处理 collate_fn、padding、动态 batching 等

---

## 6. 与 ms-swift 的对应关系

| paddleformers datasets_v2 | ms-swift dataset | 说明 |
|---|---|---|
| schema.py | preprocessor/core.py (常量部分) | 独立成文件，更清晰 |
| preprocessors/base.py | preprocessor/core.py:RowPreprocessor | 去除采样逻辑，纯化批处理 |
| preprocessors/response.py | preprocessor/core.py:ResponsePreprocessor | 基本一致 |
| preprocessors/messages.py | preprocessor/core.py:MessagesPreprocessor | 基本一致 |
| preprocessors/extra.py | preprocessor/core.py:AlpacaPreprocessor | 基本一致 |
| preprocessors/auto.py | preprocessor/core.py:AutoPreprocessor | 基本一致 |
| ops.py | dataset/utils.py | 待实现 |
| registry.py | dataset/register.py | 待实现 |
| loaders.py | dataset/loader/ | 待实现 |
| builder.py | dataset/load_dataset.py | 待实现 |
| adapters.py | (trainer 层处理) | 待实现 |

---

## 7. 关键设计决策记录

1. **schema 独立成文件** — ms-swift 将字段定义散落在 preprocessor 中，我们集中到 schema.py 作为单一真相源
2. **HF Dataset 类型命名** — `HfMapDataset` / `HfIterableDataset`，体现来源（HF）和访问模式（Map/Iterable）
3. **DATASET_TYPE 放在 schema** — 跨模块使用的类型定义统一放在契约层
4. **preprocessors 分文件** — 基础类各自独立（response、messages），薄封装合并到 extra.py
5. **AlpacaPreprocessor 移除 instruction/input/output 的列映射** — 避免与 ResponsePreprocessor 的 QUERY_KEYS 冲突，这些字段由 preprocess() 手动处理
6. **_rows_to_batched 使用 set().union()** — 字段顺序不影响 HF Dataset（基于 key 访问），性能优于逐行收集

---

## 8. 测试

测试文件：`tests/datasets_v2/test_preprocessors.py`

调试配置：最外层 `.vscode/launch.json` → "Test datasets_v2 Preprocessors"

运行命令：
```bash
python -m pytest tests/datasets_v2/test_preprocessors.py -v
```
