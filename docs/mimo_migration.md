# MiMo PaddleFormers Migration Notes

## Scope

This migration adds XiaomiMiMo/MiMo-7B-Base as a Qwen2-compatible decoder model with MiMo-specific configuration and MTP layer registration.

## Implemented

- `MiMoConfig` with `model_type = "mimo"` and MiMo-7B-Base defaults.
- `MiMoModel`, `MiMoForCausalLM`, and `MiMoForCausalLMPipe` Auto registration.
- MTP layers are represented in the model state tree so full HF checkpoints can map the additional weights.
- Tiny model unit tests under `tests/transformers/mimo`.
- Reduced-depth full-width asset and benchmark helpers under `scripts/mimo`.
- GSM8K 300-step SFT configs:
  - `examples/config/sft/mimo_gsm8k_300.yaml`
  - `examples/config/sft/mimo_gsm8k_reduced_depth_fullwidth_300.yaml`
  - `tests/config/benchmark/config/sft/MiMo-7B-Base.yaml`
  - `tests/config/benchmark/config/sft/MiMo-7B-Base-Reduced-Depth-FullWidth.yaml`

## Useful Commands

Create a tiny CI smoke checkpoint:

```bash
python scripts/mimo/create_tiny_random.py --output-dir ./.cache/mimo/tiny-random
```

Prepare both tiny and reduced CE assets, optionally copying a Qwen2/MiMo tokenizer:

```bash
TOKENIZER_DIR=/path/to/mimo-or-qwen2-tokenizer bash scripts/mimo/prepare_ce_assets.sh
```

Create a same-width reduced-depth checkpoint for acceptance fallback:

```bash
LAYERS=4 OUTPUT_DIR=./.cache/mimo/reduced-depth-4l-fullwidth-random bash scripts/mimo/prepare_reduced_assets.sh
```

Compare full HF and Paddle logits/generation:

```bash
python scripts/mimo/compare_forward.py --model XiaomiMiMo/MiMo-7B-Base --dtype bfloat16
```

Run 300-step GSM8K SFT on the reduced-depth checkpoint:

```bash
paddleformers-cli train examples/config/sft/mimo_gsm8k_reduced_depth_fullwidth_300.yaml
```

Compare compiler on/off inference and training:

```bash
bash scripts/mimo/compare_inference_compile_reduced.sh
bash scripts/mimo/compare_training_compile_reduced.sh
```

## Acceptance Items To Run With Full Assets

1. Single-card forward alignment against Transformers: target logits diff at `1e-2`.
2. Greedy generation alignment: first 10 generated tokens match Transformers.
3. GSM8K SFT for 300 steps with the hyperparameters in `examples/config/sft/mimo_gsm8k_300.yaml`.
4. CI/CE tiny model upload and CE config wiring after a PaddleFormers/tiny-random-mimo checkpoint is available.
5. Compiler on/off train and inference benchmark: target average speedup 20%.

## Notes

MiMo's HF remote code subclasses Qwen2 and does not call the MTP layers during normal causal LM forward. PaddleFormers follows the same behavior: base logits and generation use the Qwen2-compatible main decoder, while the MTP layers are present for checkpoint compatibility and future speculative decoding work.
