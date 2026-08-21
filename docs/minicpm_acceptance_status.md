# MiniCPM-1B Acceptance Status

This document tracks the current MiniCPM-1B migration status for PR review.

## Current Best Path

- Model: `openbmb/MiniCPM-1B-sft-bf16`
- Validation path: full-depth, full-width MiniCPM-1B.
- Checkpoint path: HF `pytorch_model.bin` converted to Paddle native `model_state.pdparams` with `scripts/minicpm/convert_hf_to_paddle_native.py`.
- Loader path: `AutoModelForCausalLM.from_pretrained(..., load_checkpoint_format="sharding_io", convert_from_hf=False)` for CLI training, or `"naive"` for direct script-level forward validation.

Direct Flex loading from the HF `.bin` checkpoint is not used for acceptance in this environment because PaddleFormers disables PyTorch inside the Paddle runtime, and Flex checkpoint metadata generation for this `.bin` checkpoint is incomplete. The PR therefore provides an explicit HF-to-Paddle native conversion script and validates the converted Paddle checkpoint.

## Validation

- Full MiniCPM-1B FP32 logits alignment:
  - `max_diff=4.57763671875e-05`
  - `mean_diff=8.224584234994836e-06`
  - `last_max_diff=3.0517578125e-05`
  - `last_mean_diff=7.700375135755166e-06`
- Generation alignment:
  - `first_10_tokens_match=True`
  - HF and Paddle new token ids: `[11225, 72, 5, 2219, 8107, 1379, 8360, 1410, 11225, 72]`
- Unit tests:
  - `python -m unittest tests.transformers.minicpm.test_modeling -v`
  - Result: `Ran 34 tests ... OK (skipped=5)`
- Tiny SFT smoke:
  - `scripts/minicpm/create_tiny_random.py` deterministically generates a tiny native checkpoint and tokenizer without external model assets.
  - The complete 10-step single-card BF16 SFT smoke passed with `global_step=10` and `train_loss=3.9330`; the trained checkpoint was saved successfully.
  - `tests/integration_test/preprocess.sh` prepares the checkpoint under `${CACHE_DIR}/minicpm/tiny-random-minicpm`, and the H20 single-card workflow runs `tests/integration_test/minicpm_sft_single_card.sh` for 10 steps.
- GSM8K 300-step SFT:
  - Full-depth, full-width MiniCPM-1B completed 300 steps with `scripts/minicpm/run_sft_gsm8k.sh`.
  - Training used `messages` JSONL GSM8K format, `max_seq_len=1024`, global batch size 4, BF16 O2, and `save_to_hf=false` to avoid unnecessary HF-format checkpoint conversion during acceptance runs.
  - Step losses: step 1 `0.3876`, step 50 `0.1475`, step 100 `0.1755`, step 150 `0.1921`, step 200 `0.1340`, step 250 `0.1957`, step 300 `0.1483`.
  - Eval losses: step 50 `0.5329`, step 100 `0.5421`, step 150 `0.5408`, step 200 `0.5424`, step 250 `0.5483`, step 300 `0.5435`.
  - Final train metrics: `train_loss=0.1506`, `train_runtime=0:33:06.83`, `train_steps_per_second=0.151`, `Total_Tokens_per_second_per_gpu=618.47`.
- ms-swift GSM8K 300-step baseline:
  - Full-depth, full-width MiniCPM-1B completed 300 steps with `scripts/minicpm/run_ms_swift_gsm8k.sh`.
  - Baseline environment used `ms-swift==2.5.2` and `transformers==4.36.0` to match the MiniCPM HF remote-code model. Newer ms-swift releases pull newer Transformers stacks and were not used for this baseline.
  - Step losses: step 50 `0.13974644`, step 100 `0.12065377`, step 150 `0.12042905`, step 200 `0.18350531`, step 250 `0.14136277`, step 300 `0.21005851`.
  - Eval losses: step 50 `0.58973086`, step 100 `0.56501245`, step 150 `0.55724013`, step 200 `0.55477822`, step 250 `0.55431318`, step 300 `0.55419004`.
  - Final train metrics: `train_loss=0.18120488`, `train_runtime=1063.4638s`, `train_steps_per_second=0.282`.
  - PaddleFormers and ms-swift eval curves converge into the same range. Eval loss deltas from step 50 to step 300 are `0.0568`, `0.0229`, `0.0164`, `0.0124`, `0.0060`, and `0.0107`, respectively.
- Inference compiler benchmark:
  - `scripts/minicpm/benchmark_inference_compile.py` compares dynamic forward with `paddle.jit.to_static(..., backend="CINN")` under the same checkpoint, dtype, input shape, and model path.
  - The current best GPU path uses `full_graph=True` and disables fused RMSNorm for both dynamic and static runs, allowing CINN to optimize the ordinary RMSNorm graph. All timed forward passes run on GPU.
  - Full-depth, full-width MiniCPM-1B BF16 architecture, batch size 1, seq len 128: dynamic latency `0.0737s`, static latency `0.0110s`, speedup `572.16%`, `max_diff=0.00220`.
  - Full-depth, full-width MiniCPM-1B BF16 architecture, batch size 1, seq len 512: dynamic latency `0.0765s`, static latency `0.0364s`, speedup `109.96%`, `max_diff=0.00250`.
  - These performance-only runs use a full-size random native checkpoint with the same `1536/3840/52` model structure and vocabulary as MiniCPM-1B. Quality and precision acceptance results above use the converted official checkpoint; random weights are not used for model-quality claims.

## Remaining Items

- Run the compiler-on/off training benchmark in the official acceptance environment; inference-side compiler speedup already exceeds 20% on the full MiniCPM-1B architecture.
- Fleet/PP is not exposed yet because the generic GPT path does not implement MiniCPM's embedding, residual, and LM-head scaling. It should be enabled only after a MiniCPM-specific provider and ordinary-vs-Fleet/PP forward alignment test are added.
