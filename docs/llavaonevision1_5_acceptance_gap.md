# LLaVA-OneVision-1.5 Acceptance Gap Summary

This note summarizes the current validation status and the remaining gap for
the PaddleFormers migration of `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`.

## Completed Items

- The PaddleFormers model implementation has been added:
  - configuration
  - Rice vision tower
  - text decoder
  - conditional generation model
  - AutoModel registration
  - HF-to-Paddle checkpoint mapping
- The full 8B converted Paddle checkpoint loads successfully.
- Generation validation is currently passing for the selected reference case:

```text
hf_first_10_tokens:
[[641, 279, 6802, 358, 646, 1490, 264, 5220, 11699, 389]]

paddle_first_10_tokens:
[[641, 279, 6802, 358, 646, 1490, 264, 5220, 11699, 389]]

first_10_tokens_match: True
```

- Unit tests have been added and pass:

```text
Ran 11 tests ... OK
```

- The unit coverage includes:
  - configuration serialization
  - Rice vision tower forward
  - text-only forward
  - image-token forward
  - AutoModel config path
  - AutoModel checkpoint reload path

## Current Blocking Gap

The remaining blocker is the strict single-card forward logits alignment target.

Target:

```text
Transformers vs Paddle logits diff should be around 1e-2.
```

Current best generation-preserving result:

```text
last token max_diff:  0.5
last token mean_diff: 0.05361148
```

Full-sequence logits diff remains larger because early multimodal tokens amplify
small BF16 differences across the sequence:

```text
full logits max_diff:  30.25
full logits mean_diff: 0.48766083
```

## Root-Cause Evidence

Several high-risk implementation issues have already been ruled out:

- HF-to-Paddle fused qkv weight layout matches exactly.
- HF-to-Paddle fused MLP gate/up layout matches exactly.
- Token embedding and early decoder-layer RMSNorm are aligned.
- Text layer-0 same-input attention/MLP probes are close when using the correct
  `rope_theta=1000000.0`.
- Rice vision block 0 starts close.
- Rice vision block 23 is also close when given the same HF block input.
- Final `lm_head` FP32 computation does not materially improve the gap.
- HF eager vs SDPA attention selection alone does not explain the difference.

The evidence points to accumulated BF16 numerical drift across:

- the 24-layer Rice vision tower, and
- the 36-layer text decoder after image features are inserted.

This is a cross-framework numerical-alignment problem rather than an obvious
checkpoint mapping or model-structure error.

## Tradeoff Observed

Disabling fused RMSNorm improves the last-token logits diff:

```text
last token mean_diff: ~0.04438
```

However, it breaks the already-passing first-10-token generation match. For this
reason, fused RMSNorm remains enabled by default and the non-fused path is kept
only as a diagnostic option.

## Suggested Acceptance Discussion

Given the current state, a practical acceptance compromise could be:

- Accept generation parity as the primary functional signal for this migration:
  - first 10 generated tokens match Transformers exactly on the reference case.
- Accept current last-token logits diff as a known BF16 cross-framework numerical
  gap:
  - `last_mean_diff ~= 5e-2`.
- Continue tracking the strict `1e-2` forward-logits target as a follow-up
  optimization task.

The remaining work required to force logits diff from `~5e-2` to `1e-2` is
expected to require deeper numerical work on Paddle/Torch BF16 operator parity,
especially in LayerNorm/RMSNorm, attention, and long multimodal residual paths.

## Remaining Non-Precision Tasks

These are still required for full project closure:

- GSM8K 300-step SFT loss-curve validation against ms-swift.
- CI/CE precision baseline upload for the new single-card script.
- Compiler on/off training performance comparison.
- Longer formal inference benchmark run with `scripts/llavaonevision1_5/compare_inference_compile.sh`.
- Final cleanup of migration scripts, docs, and validation artifacts.

The repository now contains initial CI/CE and benchmark entries:

- `tests/config/ci/llavaonevision1_5_sft_single.yaml`
- `tests/integration_test/llavaonevision1_5_sft_single_card.sh`
- `.github/workflows/fleet-model-test.yml`
- `tests/config/benchmark/config/sft/LLaVA-OneVision-1.5-8B-Instruct.yaml`
- `examples/config/sft/llavaonevision1_5_gsm8k_300.yaml`
- `scripts/llavaonevision1_5/benchmark_inference.py`
- `scripts/llavaonevision1_5/compare_inference_compile.sh`
- `scripts/llavaonevision1_5/check_acceptance_assets.py`
- `scripts/llavaonevision1_5/convert_gsm8k_to_erniekit.py`
- `scripts/llavaonevision1_5/prepare_ce_assets.sh`
- `scripts/llavaonevision1_5/run_gsm8k_sft_300.sh`

## Asset Gaps

The current tiny checkpoint contains model config and weights, but a
training-style CE run also needs tokenizer/processor assets in the tiny model
directory. Use:

```bash
python scripts/llavaonevision1_5/check_acceptance_assets.py \
  --tiny-dir ./tiny-random-llavaonevision1_5
```

Current local status:

- Tiny model weights/config: present.
- Tiny tokenizer assets: present after regenerating the tiny checkpoint with a
  local Qwen3-VL tokenizer.
- GSM8K converted erniekit data: not found locally.
- The reduced CE config is currently text-only SFT, so it does not require the
  VL dummy image directory.

## CE Smoke Attempt

The reduced single-card CE script is independent of `yq` and falls back to
plain `paddleformers-cli train` when `coverage` is unavailable. It also supports
`SKIP_PRECISION_CHECK=1` for local smoke runs before the official precision
baseline is uploaded.

Local run status:

```text
SKIP_PRECISION_CHECK=1 bash -x PaddleFormers/tests/integration_test/llavaonevision1_5_sft_single_card.sh single
```

Result on RTX 3090 GPU 1:

```text
Training completed.
***** train metrics *****
train_loss = 12.9211
Exit code 0
```

The official CE path still requires uploading
`llavaonevision1_5_sft_single_card_gt_loss.txt`; without it, the script reaches
the BOS precision download and receives 404, as expected for a new model
baseline.

## Performance Probe

A short text-only 8B inference probe has been run on RTX 3090 GPU 0.
The benchmark uses `--benchmark-mode manual-lm-head` and
`--attn-implementation eager` so the model path is compatible with
`paddle.jit.to_static` while still including the final vocabulary projection.

```text
seq_len: 64
steps: 5
dynamic tokens_per_sec:   1627.63
to_static tokens_per_sec: 2103.48
speedup: 29.24%
```

This satisfies the 20% inference speedup target on the short text-only probe.

The full default model path is not yet compatible with `to_static`:

- With the default SDPA path, Paddle static conversion fails in
  `sdpa_attention_forward` because `attn_mask_startend_row_indices` becomes an
  `UndefinedVar`.
- With `--attn-implementation eager`, conversion gets further but fails with a
  recursion error through `configuration_utils.__getattribute__` from
  `lm_head.forward`.

For this reason, the benchmark script provides the static-friendly
`manual-lm-head` mode.
