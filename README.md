<p align="center">
  <img src="https://github.com/user-attachments/assets/9d1c1937-7fac-48f8-9d61-f7ac67b61b18" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFormers/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleFormers?color=ccf"></a>
</p>

<h4 align="center">
    <a href=#news> News </a> |
    <a href=#highlights> Highlights </a> |
    <a href=#installation> Installation </a> |
    <a href=#quickstart> Quickstart </a> |
    <a href=#community> Community </a>
</h4>

**PaddleFormers** is a Transformer model library built on the [PaddlePaddle](https://www.paddlepaddle.org.cn) deep learning framework, delivering both **ease of use** and **high-performance capabilities**. It provides a unified model definition interface, modular training components, and comprehensive distributed training strategies specifically designed for large language model development pipelines. This enables developers to train large models efficiently with minimal complexity, making it suitable for diverse scenarios ranging from academic research to industrial applications.

## News

[2025/06/28] 🎉  **PaddleFormers 0.1** is officially released! This initial version supports SFT/DPO training paradigms, configurable distributed training via unified Trainer API, and integrates PEFT, MergeKit, and Quantization APIs for diverse LLM applications.

## Highlights

### ⚙️ Simplified Distributed Training
Implements 4D parallel strategies through unified Trainer API, lowering the barrier to distributed LLM training.

### 🛠 Efficient Post-Training
Integrates Packing dataflow and [FlashMask](https://arxiv.org/abs/2410.01359) operators for SFT/DPO training, eliminating padding waste and boosting throughput.

### 💾 Industrial Storage Solution
Features **Unified Checkpoint** storage tools for LLMs, enabling training resumption and dynamic resource scaling.  Additionally implements asynchronous storage (up to 95% faster) and Optimizer State Quantization (78% storage reduction), ensuring industrial training meets both efficiency and stability requirements.

## Installation

Requires Python 3.10+

```bash
# Install via source code
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers

# If you don’t need to train models, you can install only the lightweight basic version of paddleformers.
pip install -e .

# If you need to train models, you should install paddleformers with paddlefleet
# cuda12.6
pip install -e '.[paddlefleet]'  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/
# cuda12.9
pip install -e '.[paddlefleet]'  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/
# cuda13.0
pip install -e '.[paddlefleet]'  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/
```

## Quickstart

### Text Generation

This example shows how to load Qwen model for text generation with PaddleFormers `Auto API`:


```python
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype="bfloat16", convert_from_hf=True).eval()
input_features = tokenizer("Give me a short introduction to large language model.", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=128)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
```


### SFT Training

Getting started with supervised fine-tuning (SFT) using PaddleFormers:
```bash
paddleformers-cli train examples/config/sft/full.yaml
```

## Community

We welcome all contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This repository's source code is available under the [Apache 2.0 License](LICENSE).
