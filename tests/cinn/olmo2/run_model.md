# OLMo2 CINN training and inference smoke test

This test imports PaddleFormers from this checkout, not an external source snapshot.
It uses a random tiny model: 2 layers, hidden size 64, FFN size 128, 4 query heads,
2 KV heads, head dimension 16, original tokenizer/vocabulary, 12,919,296 parameters.
It is not a pretrained reduced-depth checkpoint or full-model quality benchmark.

## Environment and external assets

Reference environment: V100S 32GB, Python 3.10.20, Paddle 3.5.0.dev20260722
(commit 08de6be508389d5277b02e78fbc8742f1ed78160), CUDA 12.6, transformers 5.3.0,
tokenizers 0.22.2, numpy 1.26.4, plus this repository's requirements.
FP32, TF32 disabled, batch 1, sequence length 64, eager attention, no AMP or distributed execution.

Weights are not included. Obtain the test assets from the contributor:

- CHECKPOINT_DIR: initial.pdparams (shared random initial state), inference.pdparams
  (the dynamic model's historical 30000-step checkpoint).
- CINN_ASSET_DIR: config.json, tokenizer.json, tokenizer_config.json,
  special_tokens_map.json, vocab.json, merges.txt from the recorded OLMo2 asset bundle.

All assets must match asset_manifest.json. The test fails on missing/mismatched assets;
it does not silently regenerate weights. Public download hosting is pending.

## Run from the repository root

```bash
export CHECKPOINT_DIR=/path/to/weights
export CINN_ASSET_DIR=/path/to/olmo2/assets
export PYTHON_BIN=/path/to/python
bash tests/cinn/olmo2/run_model.sh
bash tests/cinn/olmo2/run_model_cinn.sh
"$PYTHON_BIN" tests/cinn/olmo2/compare_results.py tests/cinn/olmo2/results
# Longer training, same initial state for both modes:
MODE=dynamic STEPS=30000 RESULTS_DIR=/path/to/new/full bash tests/cinn/olmo2/train.sh
MODE=cinn STEPS=30000 RESULTS_DIR=/path/to/new/full bash tests/cinn/olmo2/train.sh
```

Quick entry points each perform 20 training steps and independent-process inference.
GPU_ID defaults to 0. Repeated runs require a new RESULTS_DIR; existing results are never overwritten.
Optional RESUME supplies a trusted full training-state checkpoint from the same mode/source.
Model, optimizer and RNG state are saved every 500 steps and at completion.

Data: the first 24 GSM8K training examples, included as data/gsm8k_24.jsonl
([source](https://github.com/openai/grade-school-math), MIT).
The first 20 examples train and the last 4 are held out. Tokenize question, newline and answer,
truncate to 65 tokens, form 64-token next-token labels and ignore padding labels.
This is prefix language modeling, not answer-only SFT. AdamW lr=2e-5, weight_decay=0.01,
seed 20261007, cyclic sample order.

## Compilation, outputs and acceptance scope

CINN compiles the full forward wrapper with explicit tensor arguments and tensor output.
Loss and optimizer execute dynamically. Configuration get remains Python-only;
there is no operator blacklist or model formula change. A CINN compile log marker is required.

Each output directory records source/framework commits, hashes, configuration, every-step
loss, held-out loss, first-step gradients and checkpoints. Inference reloads the same trained
checkpoint in both modes, warms up 10 times and times 30 synchronized fixed forward calls;
it does not benchmark generation. Compare logits.npy and inference.json across modes.
Training throughput excludes startup (first 100 steps per process for long runs).
Throughput improvement = (dynamic milliseconds / CINN milliseconds - 1) * 100%.

Smoke success requires finite losses/gradients, no missing trainable gradients, identical
starting weights/inputs and compiled execution. Twenty steps do not prove convergence.
The short-entry comparison additionally enforces absolute error limits: loss `1e-5`,
first-step gradients `1e-4`, and logits `1e-4` (all with zero relative tolerance).
These leave margin over the recorded FP32 smoke results below. Overrides are explicit
`--loss-atol`, `--grad-atol`, and `--logits-atol` arguments and are recorded in the output.
These short-entry thresholds are not acceptance criteria for pointwise agreement of
long training trajectories or for convergence.
Historical source snapshot 44c39fe19f7bc0b3c45c59562038c0b2706d4624 completed
30000 steps per mode with last-500 mean losses 0.003588/0.003526, demonstrating training-subset
fitting only. Those historical timings are not measurements of this checkout.
The original full 1B compilation failure is outside the scope of this tiny test.

Local entry regression on develop f7fa77f02ec637441e1a4fad1f9066b228774fb6
(2026-11-05): both modes completed 20 training steps and independent inference.
All 25 parameter gradients were retained and finite; first-step gradient max absolute
difference was 7.39e-6, loss max difference 9.54e-7, same-checkpoint inference logits
max difference 1.38e-5. This is short-entry validation, not a new convergence run.
