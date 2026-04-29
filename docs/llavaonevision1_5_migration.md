# LLaVA-OneVision-1.5 Migration Notes

This note tracks the PaddleFormers migration work for `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`.

## Source References

- PyTorch/HuggingFace reference repo: `/sda/data/Lichenyang/LLaVA-OneVision/LLaVA-OneVision-1.5`
- HF model id: `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`
- PaddleFormers target repo: `/sda/data/Lichenyang/PaddleFormers`
- Local requirement docs:
  - `PaddleFormers高热模型合入CheckList.pdf`
  - `新增模型添加CE流程.pdf`

## Acceptance Checklist

- Single-GPU forward precision:
  - Compare Paddle logits against Transformers logits.
  - Target diff: `1e-2` order.
- Generation:
  - Greedy output must be normal.
  - First 10 generated tokens should match Transformers for the selected case.
- Training loss:
  - Dataset: GSM8K.
  - Config: `examples/config/sft/full.yaml`.
  - Steps: 300.
  - Compare against ms-swift loss curve; error should fluctuate around 0.
- Unit tests:
  - Add PaddleFormers transformer tests for the new model.
- CI/CE:
  - Add reduced-layer model CI or CE coverage.
  - Add full-weight CE coverage where hardware permits.
- Compiler performance:
  - Compare training and inference speed with Paddle compiler on/off.
  - Target average improvement: over 10%; project close target: 20%.

## CE Flow Notes From PDF

- Tiny/random model should be uploaded to an AiStudio repo.
- Suggested model size: no more than 300 MB.
- CE environment:
  - A100 80G
  - Python 3.10
  - CUDA 12.6
- Required CE configurations should cover PT/SFT/DPO and LoRA variants where applicable.
- For non-MoE models, EP coverage is not required.
- Recommended deterministic environment variables:

```bash
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
```

## Implementation Phases

1. Add configuration and AutoConfig registration.
2. Port the text backbone and Rice vision tower.
3. Add conditional generation model and HF weight conversion mapping.
4. Add processor support or reuse compatible Qwen VL processor behavior.
5. Add forward-logits comparison script against Transformers.
6. Add greedy generation comparison script.
7. Add reduced-size unit tests.
8. Add CE configs and tiny/random model flow.
9. Run GSM8K SFT 300-step loss comparison.
10. Run compiler on/off performance comparison.

## Current Status

- Configuration scaffold has been added:
  - `paddleformers/transformers/llavaonevision1_5/configuration.py`
  - `paddleformers/transformers/llavaonevision1_5/__init__.py`
- Paddle-native Rice vision tower scaffold has been added:
  - `paddleformers/transformers/llavaonevision1_5/modeling.py`
- AutoConfig registration has been added:
  - `paddleformers/transformers/__init__.py`
  - `paddleformers/transformers/auto/configuration.py`
- Minimal configuration test scaffold has been added:
  - `tests/transformers/llavaonevision1_5/test_configuration.py`
- Minimal Rice vision modeling test scaffold has been added:
  - `tests/transformers/llavaonevision1_5/test_modeling.py`
- Syntax check passed for the new Python files.
