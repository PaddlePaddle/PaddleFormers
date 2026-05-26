# MiMo Acceptance Status

## Done In Code

- MiMo model/config registration in PaddleFormers.
- MTP layer state tree and HF-to-Paddle AOA mapping.
- Tiny unit test skeleton for `MiMoModel` and `MiMoForCausalLM`.
- Full and reduced-depth GSM8K SFT configs.
- CI single-card smoke config and integration script.
- CE/reduced asset generation helpers.
- HF/Paddle logits and generation comparison helpers.
- Split-env tiny reference comparison helpers:
  - `scripts/mimo/dump_tiny_qwen2_reference.py`
  - `scripts/mimo/compare_paddle_reference.py`
- Compiler on/off inference and training benchmark helpers.
- Model capability matrix entry.

## Verified Locally

Environment:

- Paddle side: `mimo-paddle`, Python 3.12, `paddlepaddle-gpu==3.4.0.post20260424+267502364e4`, `paddlefleet==0.3.0.dev20260425`.
- Torch side: `qwen3vl`, torch CUDA available.
- GPU used: `CUDA_VISIBLE_DEVICES=2`.

Results:

- `paddleformers-cli` starts successfully with `PADDLEFORMERS_DIST_LOG=/tmp/mimo/dist_log`.
- MiMo unit test passes:

```bash
CUDA_VISIBLE_DEVICES=2 /home/lichenyang/miniconda3/envs/mimo-paddle/bin/python -m unittest tests.transformers.mimo.test_modeling -v
```

Result: `Ran 22 tests`, `OK (skipped=3)`.

- 1-step SFT smoke passes using local tiny checkpoint:

```bash
PATH=/home/lichenyang/miniconda3/envs/mimo-paddle/bin:$PATH \
PADDLEFORMERS_DIST_LOG=/tmp/mimo/dist_log \
CUDA_VISIBLE_DEVICES=2 NNODES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=36888 \
/home/lichenyang/miniconda3/envs/mimo-paddle/bin/paddleformers-cli train /tmp/mimo_sft_smoke.yaml
```

Result: `train_loss=12.363482475280762`, checkpoint saved to `/tmp/mimo/checkpoints/smoke`.

- Tiny HF-Qwen2-compatible reference vs Paddle MiMo passes:

```bash
CUDA_VISIBLE_DEVICES=2 /home/lichenyang/miniconda3/envs/qwen3vl/bin/python \
  scripts/mimo/dump_tiny_qwen2_reference.py \
  --model /tmp/mimo/tiny-random-mimo-tokenizer \
  --output-dir /tmp/mimo/reference-tiny-qwen2 \
  --dtype float32 --device cuda:0 --max-new-tokens 10 --topk 10

CUDA_VISIBLE_DEVICES=2 /home/lichenyang/miniconda3/envs/mimo-paddle/bin/python \
  scripts/mimo/compare_paddle_reference.py \
  --model /tmp/mimo/tiny-random-mimo-tokenizer \
  --reference /tmp/mimo/reference-tiny-qwen2/reference.npz \
  --dtype float32 --device gpu --max-new-tokens 10 --topk 10 \
  --load-checkpoint-format flex_checkpoint
```

Result: `max_diff=3.3974647521972656e-06`, `mean_diff=3.872926015446865e-07`, first 10 greedy tokens match.

- Static checks pass:

```bash
PYTHONPYCACHEPREFIX=/tmp/mimo/pycache /home/lichenyang/miniconda3/envs/mimo-paddle/bin/python -m py_compile \
  scripts/mimo/compare_paddle_reference.py \
  scripts/mimo/dump_tiny_qwen2_reference.py \
  paddleformers/peft/lora/lora_layers.py \
  paddleformers/transformers/mimo/configuration.py \
  paddleformers/transformers/mimo/modeling.py

bash -n scripts/mimo/*.sh tests/integration_test/mimo_sft_single_card.sh
```

- Asset check passes for tiny/reference/GSM8K data; reduced full-width checkpoint is still optional-missing locally.

## Remaining Acceptance Work

1. Generate or upload reduced-depth full-width checkpoint assets:

```bash
TOKENIZER_DIR=/path/to/mimo-or-qwen2-tokenizer bash scripts/mimo/prepare_ce_assets.sh
python scripts/mimo/check_acceptance_assets.py
```

2. Full-model forward and generation alignment against Transformers:

```bash
python scripts/mimo/compare_forward.py --model XiaomiMiMo/MiMo-7B-Base --dtype bfloat16
```

Target: logits diff at `1e-2` level and first 10 greedy tokens identical.

3. If single-card full 7B cannot fit, use reduced-depth full-width fallback:

```bash
LAYERS=4 TOKENIZER_DIR=/path/to/mimo-or-qwen2-tokenizer bash scripts/mimo/prepare_reduced_assets.sh
bash scripts/mimo/run_gsm8k_sft_reduced_depth_fullwidth_300.sh
```

4. Run full 300-step GSM8K SFT where resources allow:

```bash
bash scripts/mimo/run_gsm8k_sft_300.sh
```

5. Run compiler performance comparisons:

```bash
bash scripts/mimo/compare_inference_compile_reduced.sh
bash scripts/mimo/compare_training_compile_reduced.sh
```

Target: average train/inference speedup at least 20%.

6. Upload CE tiny checkpoint to an approved repo, then update the CE baseline losses and generation tokens in `scripts/regression/config.yaml`.

## Current Local Blocker

Full official HF weights are not present locally yet. The local HF cache currently has official MiMo config/tokenizer assets, but not the 7B model shards. Full acceptance alignment still needs either the full HF weights or a reduced-depth full-width checkpoint generated from them.
