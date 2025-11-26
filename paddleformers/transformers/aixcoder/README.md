# aiXcoder-7B Code Large Language Model

🏠 [Official website](https://aixcoder.com) | 🛠 [VS Code Plugin](https://marketplace.visualstudio.com/items?itemName=aixcoder.aixcoder) | 🛠 [Jetbrains Plugin](https://plugins.jetbrains.com/plugin/13574-aixcoder) | 🤗 [HuggingFace](https://huggingface.co/aiXcoder/aixcoder-7b-base)

Welcome to the PaddleFormers implementation of aiXcoder-7B Code Large Language Model. This model is designed to understand and generate code across multiple programming languages, offering state-of-the-art performance in code completion, comprehension, generation, and more tasks about programming languages.

## Table of Contents

1. [Model Introduction](#model-introduction)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Model Weights](#model-weights)
5. [Inference Example](#inference-example)
6. [Training](#training)
7. [License](#license)
8. [Citation](#citation)
9. [Acknowledgments](#acknowledgments)

## Model Introduction

As the capabilities of large code models are gradually being unearthed, aiXcoder has consistently pondered on how to make these models more beneficial in real development scenarios. To this end, we have open-sourced aiXcoder 7B Base, which has undergone extensive training on **1.2T Unique Tokens**, and the model's pre-training tasks as well as the contextual information have been uniquely designed for real-world code generation contexts.

aiXcoder 7B Base stands out as the most effective model in code completion scenarios among all models of similar parameter sizes, and it also surpasses mainstream models like CodeLlama 34B and StarCoder2 15B in the average performance on the multilingual nl2code benchmark.

### Key Features

- **Extensive Training**: Trained on 1.2T unique tokens from diverse code repositories
- **Multi-Language Support**: Proficient in multiple programming languages including Python, Java, JavaScript, C++, and more
- **Optimized for Real-World Scenarios**: Specifically designed for practical code completion and generation tasks
- **Superior Performance**: Outperforms models with larger parameter counts in code completion benchmarks
- **Context-Aware**: Uniquely designed to understand and utilize contextual information effectively

### Model Specifications

- **Parameters**: 7B
- **Architecture**: Based on LLaMA architecture with modifications for code understanding
- **Context Length**: Supports extended context for complex code understanding
- **Training Data**: 1.2T unique tokens from high-quality code repositories

## Architecture

The aiXcoder-7B model is built upon the LLaMA architecture with several key modifications optimized for code generation:

- **Enhanced Attention Mechanism**: Modified attention layers for better code structure understanding
- **Code-Specific Tokenization**: Optimized tokenizer for handling programming languages
- **Position Encoding**: Improved position encoding for capturing code syntax and structure
- **Layer Normalization**: Adjusted normalization for stable training on code data

## Quick Start

### Environment Requirements

#### Prerequisites

- Python 3.8 or higher
- PaddlePaddle 2.5.0 or higher
- CUDA 11.7 or higher (for GPU inference)

#### Installation

```bash
# Clone the PaddleFormers repository if you haven't already
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers

# Install required dependencies
pip install -r requirements.txt
```

### Model Configuration

The model configuration can be found in `configuration.py`. Key configuration parameters include:

- `hidden_size`: 4096
- `num_hidden_layers`: 32
- `num_attention_heads`: 32
- `intermediate_size`: 11008
- `max_position_embeddings`: 8192
- `vocab_size`: 49152

## Model Weights

### Pre-trained Weights

You can download the model weights from the following sources:

- **aiXcoder Base**: [HuggingFace Hub](https://huggingface.co/aiXcoder/aixcoder-7b-base)
- **aiXcoder Instruct**: Coming soon...

### PaddlePaddle Checkpoint

For PaddlePaddle users, converted checkpoints are available:

```bash
# Download converted PaddlePaddle weights
git clone https://95185b4db18fc391d6032c77f667f343ee79d592@git.aistudio.baidu.com/aiXcoder/aiXcoder-7B.git
```

## Inference Example

### Basic Code Completion

```python
import paddle
from paddleformers.transformers.aixcoder import AixcoderForCausalLM, AixcoderTokenizerFast

# 加载模型
model_path = "path/to/aixcoder-7b-base"
tokenizer = AixcoderTokenizerFast.from_pretrained(model_path)
model = AixcoderForCausalLM.from_pretrained(model_path)
model.eval()

# 输入代码
code = "def quick_sort(arr):"
inputs = tokenizer(code, return_tensors="pd")

# 生成
with paddle.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=100,
        temperature=0.7
    )

# 解码 - 关键是这里
generated_ids = outputs[0].numpy()  # 先转为 numpy
if generated_ids.ndim > 1:
    generated_ids = generated_ids.squeeze()  # 去除多余维度
generated_ids = generated_ids.tolist()  # 转为 list
generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_code)
```

### Expected Output

```python
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

## Training

### Distributed Training

For distributed training across multiple GPUs:

```bash
cd PaddleFormers/examples
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py aiXcoder-7B/full_tp_pp.yaml
```

### Fine-tuning on Custom Dataset

To fine-tune aiXcoder-7B on your custom dataset:

```bash
cd PaddleFormers/examples
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py aiXcoder-7B/full.yaml
```

## License

The model weights are licensed under the [aiXcoder Model License](./LICENSE):
- **Academic Research**: Free to use for academic research purposes
- **Commercial Use**: Please apply for commercial license by contacting support@aiXcoder.com

See the [LICENSE](./LICENSE) file for full license terms.

## Citation

If you use aiXcoder-7B in your research, please cite:

```bibtex
@misc{aixcoder2024,
  title={aiXcoder-7B: A State-of-the-Art Code Generation Model},
  author={aiXcoder Team},
  year={2024},
  publisher={aiXcoder},
  howpublished={\url{https://huggingface.co/aiXcoder/aixcoder-7b-base}}
}
```

## Acknowledgments

We would like to thank:
- All contributors to the open-source projects and datasets that made this work possible
- The PaddlePaddle team for their excellent deep learning framework
- The research community for continuous innovation in code generation models

## Contact

For questions, issues, or commercial licensing inquiries:
- Email: support@aiXcoder.com
- GitHub Issues: [PaddleFormers Issues](https://github.com/PaddlePaddle/PaddleFormers/issues)
- Official Website: [https://aixcoder.com](https://aixcoder.com)

---

*This implementation is part of the PaddleFormers project, bringing state-of-the-art language models to the PaddlePaddle ecosystem.*
