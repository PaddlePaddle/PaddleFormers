# 1. 指定数据集路径、格式类型、配比

## 1.1. 预训练离线数据集

- **CLI**：修改 YAML 配置文件中的以下字段：
  - `input_dir` 指定数据集的前缀，例如：数据集 `data-1-part0.bin` 需要设置为 `input_dir: "1.0 ./data-1-part0"`，`1.0` 为数据配比；
  - `split` 字段为 `train/eval` 的分配比例，如：`split: "998,2"`, 其中 `train` 为训练集，`eval` 为评估集
  - `dataset_type` 指定为 `pretrain`，例如：`dataset_type: "pretrain"`

- 示例：
```yaml
dataset_type: "pretrain"
input_dir: "1.0 ./data/pre-training/demo_data/data-1-part0"
split: "998,2"
```

## 1.2. 预训练在线数据集 + 其他

- **CLI**：修改 YAML 配置文件中的以下字段：
  - `train_dataset_path` / `eval_dataset_path` 指定本地数据集文件的绝对或相对路径
  - `train_dataset_type` / `eval_dataset_type` 指定数据集格式 (`erniekit` / `chatml`)
  - `train_dataset_prob` / `eval_dataset_prob` 指定用于多源数据集混合概率

- 示例：
```yaml
# single-source
train_dataset_type: "erniekit"
train_dataset_path: "./examples/data/sft-train.jsonl"
train_dataset_prob: "1.0"

# multi-source
train_dataset_type: "erniekit,erniekit"
train_dataset_path: "./examples/data/sft-train1.jsonl,./examples/data/sft-train2.jsonl"
train_dataset_prob: "0.8,0.2"
```

# 2. 多源数据集拼接策略

目前支持四种数多源数据集拼接策略：`random`, `concat`, `interleave_under`, `interleave_over`

|多源数据集拼接策略|适用场景 |限制 |描述 |
|------------------|-----------------|------------------|------------------|
| `random`|数据集极大，需要严格的数据配比 |最大步数 > 0 |在`random`模式，基于输入的数据配比，构建一个固定大小（`num_samples_each_epoch`）的样本池，`data loader` 从该样本池中随机获取数据。 |
| `concat`|需要训练数据集中的所有数据 |无 |在`concat`模式下，不使用输入的数据配比，而是多个数据集直接合并。数据集的大小等于输入多源数据集的总大小。当 max_steps = -1 时，设置`num_train_epochs`允许完整遍历输入数据集`num_train_epochs`回合。 |
| `interleave_under`|当小数据集很重要但样本有限时 |无 |`interleave`表示根据数据比例对多个数据集进行交叉拼接。`interleave_under`表示欠采样，这意味着一旦其中一个数据集耗尽，采样就会停止。 |
| `interleave_over`|当小数据集很重要但样本有限时 |无 |`interleave`表示根据数据比例对多个数据集进行交叉拼接。`interleave_over`表示过采样，意味着只有在所有数据集耗尽后才停止采样。 |

- 注意：`num_samples_each_epoch`只适用于`random`数据采样策略。

# 3. 数据 packing 策略

`packing` 是一种优化批处理的技术，将多个短输入序列输入大语言模型（LLM）之前，先将它们合并成一个更长的序列，这能减少填充开销，并提高硬件利用率（例如，提升GPU/TPU的效率）。

`The greedy intokens strategy` 是一种`token`级别的优化方法，在批量处理过程中，以贪婪的方式优先填满可用的 `token budget`（例如，最大序列长度）。该策略确保模型在约束条件下生成尽可能多的`token`，最大程度减少容量浪费。

| packing      | greedy_intokens | Packing Strategy |
|--------------|-----------------|------------------|
| false | any   | 不开`packing`  |
| true  | false | 开`packing`，但不使用贪心策略|
| true  | true  | 开`packing`，同时使用贪心策略 |

- 补充：在线预训练数据流中另外支持了`truncate_packing`的策略，支持将数据进行截断，有效降低padding token，`truncate_packing`设置为`True`即可使用，具体如下图所示：

<div align="center">
<img src="https://github.com/user-attachments/assets/f7ec5b76-aee7-4f64-8331-ca00cac5339a">
</div>

# 4. Padding Free

`padding_free` 将一个batch中的数据进行展平而避免数据padding，从而降低显存占用并加快训练（同一batch的不同序列之间依旧是不可见的）。默认为False。

相较于`packing`，`padding_free`不需要额外的预处理时间，但`packing`的训练速度更快且显存占用更稳定。

# 5. Attention Mask

数据流默认会传入一个因果的Attention Mask，在packing情况下，当`use_global_causal_attn`为true的时候，对应下图所示的`Causal Attention`，一个`Sequence`内的不同sample是可见的，当`use_global_causal_attn`为false的时候，对应下图所示的`Causal Document Attention`，一个`Sequence`内的不同sample是不可见的

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <div>
    <img 
      src="https://github.com/user-attachments/assets/57c414e3-6783-4a40-a5bf-eb67c6129b06" 
      width="200px"
      alt="Causal Attention"
    >
    <br>
    <em>Causal Attention</em>
  </div>
  <div>
    <img 
      src="https://github.com/user-attachments/assets/ffd61730-32f0-4d25-8558-086d2d43aa1f" 
      width="200px"
      alt="Causal Document Attention"
    >
    <br>
    <em>Causal Document Attention</em>
  </div>
</div>
