# 1. 数据流基础参数说明

## 1.1. 参数说明

* 参数

|参数名|参数说明|
|-|-|
|train_dataset_path|训练数据集路径：允许指定多个路径，通过`,`分隔不同的数据集。|
|eval_dataset_path|验证数据集路径：允许指定多个路径，通过`,`分隔不同的数据集。|
|train_dataset_type|训练数据集格式，可选数据集格式为 `erniekit` / `messages`|
|eval_dataset_type|验证数据集格式，可选数据集格式为 `erniekit` / `messages`|
|dataset_type|数据集类型，离线预训练数据流为`pretrain`，在线数据流当前默认为`iterable`|

* 示例：

```yaml
# single-source
train_dataset_type: "erniekit"
train_dataset_path: "./examples/data/sft-train.jsonl"

# multi-source
train_dataset_type: "erniekit,erniekit"
train_dataset_path: "./examples/data/sft-train1.jsonl,./examples/data/sft-train2.jsonl"
```

# 2. 多源数据集混合策略

## 2.1. 参数说明

* 参数

|参数名|参数说明|
|-|-|
|train_dataset_prob|指定用于多源训练数据集混合概率|
|eval_dataset_prob|指定用于多源评估数据集混合概率|
|mix_strategy|指定多源数据集的混合策略|

* 示例

```yaml
# single-source
train_dataset_type: "erniekit"
train_dataset_path: "./examples/data/sft-train.jsonl"
train_dataset_prob: "1.0"

# multi-source
train_dataset_type: "erniekit,erniekit"
train_dataset_path: "./examples/data/sft-train1.jsonl,./examples/data/sft-train2.jsonl"
train_dataset_prob: "0.8,0.2"

mix_strategy: "concat"
```

## 2.2. 混合策略

在很多情况下，开发者可能有多个源的数据需要进行训练。在训练时，通过不同的策略进行混合使用。

PaddleFormers 目前支持四种数多源数据集拼接策略：`random`, `concat`, `interleave_under`, `interleave_over`

|多源数据集拼接策略|适用场景|限制|描述|
|-|-|-|-|
|`random`|数据集极大，需要严格的数据配比|最大步数 > 0|在`random`模式，基于输入的数据配比，构建一个固定大小（`num_samples_each_epoch`）的样本池，`data loader` 从该样本池中随机获取数据。|
|`concat`|需要训练数据集中的所有数据|无|在`concat`模式下，不使用输入的数据配比，而是多个数据集直接合并。数据集的大小等于输入多源数据集的总大小。当 max_steps = -1 时，设置`num_train_epochs`允许完整遍历输入数据集`num_train_epochs`回合。|
|`interleave_under`|当小数据集很重要但样本有限时|无|`interleave`表示根据数据比例对多个数据集进行交叉拼接。`interleave_under`表示欠采样，这意味着一旦其中一个数据集耗尽，采样就会停止。|
|`interleave_over`|当小数据集很重要但样本有限时|无|`interleave`表示根据数据比例对多个数据集进行交叉拼接。`interleave_over`表示过采样，意味着只有在所有数据集耗尽后才停止采样。|

* 注意：`num_samples_each_epoch`只适用于`random`数据采样策略。

# 3. 数据拼接策略

在大模型（LLM / 多模态模型）训练过程中，**序列长度不一致** 是一个天然且普遍存在的问题。不同样本在 token 数、模态数量（文本、图像、视频）上差异极大，如果不对数据进行合理组织，会导致 **严重的计算和显存浪费** ，成为训练效率和规模扩展的核心瓶颈。因此，我们引入 **packing、padding-free 等数据拼接策略** 解决该问题。

## 3.1. 参数说明

* 参数

|参数名|参数说明|默认值|
|-|-|-|
|packing|是否使用 packing 策略进行数据拼接|false|
|greedy_intokens|在 packing 为 true 的时候，是否使用贪心 packing 策略|true|
|truncate_packing|在 packing 为 true 的时候，是否使用 truncate packing 策略进行拼接（仅限于**在线预训练数据流**）|true|
|padding_free|将一个 batch 中的数据进行展平而避免数据 padding，packing=true/false 都可行|false|
|use_global_causal_attn|打开的时候为 Causal Attention，关闭的时候为 Causal Document Attention|false|

* 示例

```yaml
packing: true
greedy_intokens: true
truncate_packing: false
padding_free: false
use_global_causal_attn: false
```

## 3.2. 数据 packing 策略

`packing` 是一种优化批处理的技术，将多个短输入序列输入大语言模型（LLM）之前，先将它们合并成一个更长的序列，这能减少填充开销，并提高硬件利用率（例如，提升 GPU/TPU 的效率）。

`The greedy intokens strategy` 是一种`token`级别的优化方法，在批量处理过程中，以贪婪的方式优先填满可用的 `token budget`（例如，最大序列长度）。该策略确保模型在约束条件下生成尽可能多的`token`，最大程度减少容量浪费。

* 参数说明：

|packing|greedy_intokens|Packing Strategy|
|-|-|-|
|false|any|不开`packing`|
|true|false|开`packing`，但不使用贪心策略|
|true|true|开`packing`，同时使用贪心策略|

* 补充：在线预训练数据流中另外支持了`truncate_packing`的策略，支持将数据进行截断，有效降低 padding token，`truncate_packing`和`packing`设置为`True`即可使用，具体如下图所示：

<div align="center">
  <img width="671" height="371" alt="data_packing" src="https://github.com/user-attachments/assets/ccd7c0a7-5cbb-4ef6-b4f3-95ed7296266d" />
</div>

## 3.3. Padding Free

`padding_free` 将一个 batch 中的数据进行展平而避免数据 padding，从而降低显存占用并加快训练（同一 batch 的不同序列之间依旧是不可见的）。默认为 False。

相较于`packing`，`padding_free`不需要额外的预处理时间，但`packing`的训练速度更快且显存占用更稳定。

## 3.4. Attention Mask

数据流默认会传入一个因果的 Attention Mask，在 packing 情况下，

* 当`use_global_causal_attn`为 true 的时候，对应下图所示的`Causal Attention`，一个`Sequence`内的不同 sample 是可见的

* 当`use_global_causal_attn`为 false 的时候，对应下图所示的`Causal Document Attention`，一个`Sequence`内的不同 sample 是不可见的

<div align="center">
  <div style="display: flex; gap: 20px; align-items: center;">
    <img height="300" width="auto" alt="causal_attention_mask" src="https://github.com/user-attachments/assets/dcc2938c-fc06-42bc-96a2-35101b3e0ef8" />
    <img height="300" width="auto" alt="causal_doc_attention_mask" src="https://github.com/user-attachments/assets/bbcfa19c-a70c-4144-903d-cb63ea1b6145" />
  </div>
</div>

# 4. 离线数据流使用

相比在线数据流，离线数据流将分词、裁剪、packing 等复杂数据处理前移，在训练阶段仅做顺序读取与计算，显著降低 CPU 开销，提升训练稳定性和可复现性，更适合大规模分布式训练。

当前 PaddleFormers 仅支持**预训练数据**使用离线数据流

## 4.1. 离线数据流制作

离线数据流制作方法如下：

下载一个文本数据集，例如 [https://modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset](https://modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset)

格式需为 jsonl，每行格式例如 BazingaLyn/mini_pretrain_dataset/pretrain_hq_v7.jsonl：

```json
{"text": "番茄炒蛋\n材料：\n鸡蛋3个、番茄1个、油、盐、糖、水淀粉\n做法：..."}
{"text": "请描述一下如何正确规划个人理财。正确规划个人理财需要以下几个步骤..."}
{"text": "请输入一段描述有关海洋保护的情景对话。Person A: 哇，这个海滩真..."}
{"text": "鉴别两种不同类型的葡萄酒。鉴别葡萄酒的方法因其类型和品种而异，下..."}
```

运行`examples/tools/create_pretraining_data.py`，生成数据将会保存在当前目录下的`./pretrain_data.bin`和`./pretrain_data.idx`

```shell
python -u examples/tools/create_pretraining_data.py \
    --model_name_or_path "/path/to/your/Qwen3-0.6B-base" \
    --data_format "JSON" \
    --input_path "/path/to/your/BazingaLyn/mini_pretrain_dataset/pretrain_hq_v7.jsonl" \
    --append_eos \
    --output_prefix "./pretrain_data"  \
    --workers 1 \
    --log_interval 10000 \
    --data_impl "mmap"
```

* 参数说明

|参数名|类型|说明|
|-|-|-|
|`--model_name_or_path`|string|模型路径|
|`--data_format`|string|支持的文件格式，当前只支持 JSON|
|`--input_path`|string|输入的 json 文件的路径|
|`--append_eos`|store_true|是否在每条数据的结尾添加 eos token|
|`--output_prefix`|str|输出文件的前缀|
|`--workers`|int|运行的进程数|
|`--log_interval`|int|打印日志间隔|
|`--data_impl`|str|制作的数据集类型，默认为 mmap，也可以选择 lazy|

## 4.2. 参数说明

* 参数

|参数名|参数说明|
|-|-|
|input_dir|指定数据集的前缀，例如：数据集 `data-1-part0.bin` 需要设置为 `input_dir: "1.0 ./data-1-part0"`，`1.0` 为数据配比|
|split|`split` 字段为 `train/eval` 的分配比例，如：`split: "998,2"`, 其中 `train` 为训练集，`eval` 为评估集|
|dataset_type|`dataset_type` 指定为 `pretrain`，例如：`dataset_type: "pretrain"`|

* 示例

```yaml
dataset_type: "pretrain"
input_dir: "1.0 ./data/pre-training/demo_data/data-1-part0"
split: "998,2"
```
