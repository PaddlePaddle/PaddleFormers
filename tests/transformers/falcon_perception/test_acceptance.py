# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Acceptance checks must reject short prefixes and differences outside the last prompt token."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "examples/falcon_perception/compare_reference.py"
spec = importlib.util.spec_from_file_location("falcon_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_forward_checks_every_prompt_position(tmp_path):
    a = np.zeros([1, 3, 7], dtype="float32")
    b = a.copy()
    b[0, 0, 4] = 0.011
    np.save(tmp_path / "reference.npy", a)
    np.save(tmp_path / "actual.npy", b)
    result = comparison.compare_forward_logits(tmp_path / "reference.npy", tmp_path / "actual.npy", 0.01)
    assert not result["passed"] and result["max_abs"] > 0.01
    b[0, 0, 4] = 0.009
    np.save(tmp_path / "actual.npy", b)
    assert comparison.compare_forward_logits(tmp_path / "reference.npy", tmp_path / "actual.npy", 0.01)["passed"]


def test_forward_rejects_missing_or_nonfinite_logits(tmp_path):
    assert not comparison.compare_forward_logits(tmp_path / "a.npy", tmp_path / "b.npy", 0.01)["passed"]
    np.save(tmp_path / "a.npy", np.full([1, 1, 2], np.nan, dtype="float32"))
    np.save(tmp_path / "b.npy", np.full([1, 1, 2], np.nan, dtype="float32"))
    assert not comparison.compare_forward_logits(tmp_path / "a.npy", tmp_path / "b.npy", 0.01)["passed"]


@pytest.mark.parametrize("length", [0, 8, 9])
def test_short_generation_does_not_pass_ten_token_requirement(length):
    result = comparison.compare_generation(list(range(length)), list(range(length)))
    assert not result["passed"] and not result["sufficient_tokens"]


def test_generation_checks_exactly_first_ten_real_tokens():
    assert comparison.compare_generation(list(range(12)), list(range(10)) + [99, 100])["passed"]
    result = comparison.compare_generation(list(range(12)), list(range(9)) + [99, 10, 11])
    assert not result["passed"] and result["first_mismatch_index"] == 9


def test_generated_tokens_are_taken_from_decode_inputs(tmp_path):
    np.savez(tmp_path / "step_000.npz", input_tokens=np.array([[12, 34]]), logits=np.array([[0.0, 1.0]]))
    np.savez(tmp_path / "step_001.npz", input_tokens=np.array([[1]]), logits=np.array([[1.0, 0.0]]))
    manifest = {"trace_files": ["step_000.npz", "step_001.npz"], "generated_tokens": [1]}
    assert comparison.read_generated_tokens(tmp_path, manifest) == [1]
    manifest["generated_tokens"] = [0]
    with pytest.raises(ValueError, match="Manifest tokens differ"):
        comparison.read_generated_tokens(tmp_path, manifest)


def test_additional_diagnostics_do_not_override_user_acceptance(tmp_path, monkeypatch):
    metadata = dict(
        dtype="float32",
        image="same.png",
        query="cat",
        min_dimension=256,
        max_dimension=512,
        max_new_tokens=64,
        predictions=[],
        generated_tokens=list(range(10)),
        trace_files=[f"step_{i:03d}.npz" for i in range(11)],
    )
    for name in ["reference", "actual"]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(metadata))
        np.save(folder / "prefill_logits.npy", np.zeros([1, 2, 3], dtype="float32"))
        for i, filename in enumerate(metadata["trace_files"]):
            np.savez(
                folder / filename,
                input_tokens=np.array([[20, 21]]) if i == 0 else np.array([[i - 1]]),
                logits=np.zeros([1, 3], dtype="float32"),
            )
    args = [
        "compare_reference.py",
        "--reference",
        str(tmp_path / "reference"),
        "--actual",
        str(tmp_path / "actual"),
        "--report",
        str(tmp_path / "report.json"),
    ]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as error:
        comparison.main()
    result = json.loads((tmp_path / "report.json").read_text())
    assert error.value.code == 0 and result["passed"]
    assert not result["diagnostics_passed"]  # Missing optional feature dumps.
    monkeypatch.setattr("sys.argv", args + ["--strict-diagnostics"])
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 1
