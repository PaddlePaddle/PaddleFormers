# Data Format Specification

## Pre-training offline dataset

- **CLI**: Modify the following fields in the YAML configuration file:
  - `input_dir` specify the prefix of the dataset, for example: dataset `data-1-part0.bin` need to be set to `input_dir: "1.0 ./data-1-part0"`，`1.0` is the dataset prob
  - `split` specify `train/eval` distribution ratio, such as: `split: "998,2"`, `train` is the training set, `eval` for the evaluation set
  - `dataset_type` specify as`pretrain`, such as: `dataset_type: "pretrain"`

- Example:
```yaml
dataset_type: "pretrain"
input_dir: "1.0 ./data/pre-training/demo_data/data-1-part0"
split: "998,2"
```

## Pre-training online dataset + others

- **CLI**: Modify the following fields in the YAML config file:
  - Set `train_dataset_path` / `eval_dataset_path` to the absolute or relative path of your local dataset file
  - Set `train_dataset_type` / `eval_dataset_type` to the dataset format (erniekit/chatml)
  - Set `train_dataset_prob` / `eval_dataset_prob` for multi-source dataset mixing probabilities
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

# Data Packing Strategy

`Packing` is a technique used to optimize batch processing by combining multiple short input sequences into a single longer sequence before feeding them into the LLM. This reduces padding overhead and improves hardware utilization (e.g., GPU/TPU efficiency).

`The greedy intokens strategy` is a token-level optimization that prioritizes filling the available token budget (e.g., max sequence length) in a greedy manner during batch processing. It ensures that the model generates as many tokens as possible within the constraints, minimizing wasted capacity.

| packing      | greedy_intokens | Packing Strategy |
|--------------|-----------------|------------------|
| false | any   | No packing  |
| true  | false | packing is enabled without greedy intokens strategy |
| true  | true  | greedy intokens packing is enabled |

# Data Sampling Strategy

Currently, four data sampling strategies are supported: `random`, `concat`, `interleave_under`, `interleave_over`

| Data Sampling Strategy | Applicable Scenarios	| Limitations | Description |
|------------------|-----------------|------------------|------------------|
| `random`           | The dataset is extremely large and strict data proportioning is required | max_steps > 0 | In `random` mode, based on the input dataset probs, a fixed-size sample pool of `num_samples_each_epoch` is constructed, and the data loader randomly acquires data from this sample pool. |
| `concat`           | Need to train all data in the datasets | None | In `concat` mode, the input dataset probs are not used. Instead, multiple datasets are directly concatenated. The size of the dataset is equal to the total size of the input multi-source datasets. When max_steps = -1, setting `num_train_epochs` allows for a complete traversal of the input datasets for `num_train_epochs` rounds. |
| `interleave_under` | When small datasets are important but have limited samples | None | The `interleave` strategy involves cross-concatenating multiple datasets according to data proportioning. `interleave_under` indicates undersampling, meaning that sampling stops as soon as one of the datasets is exhausted. |
| `interleave_over`  | When small datasets are important but have limited samples | None | The `interleave` strategy involves cross-concatenating multiple datasets according to data proportioning. `interleave_over` indicates oversampling, meaning that sampling stops only after all datasets have been exhausted. |

- Note: `num_samples_each_epoch` only works in `random` data sampling strategy.
