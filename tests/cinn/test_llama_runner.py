# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import paddle

from tests.cinn.llama_runner import parse_args, run_inference, run_training


class TinyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = paddle.nn.Embedding(128, 8)
        self.head = paddle.nn.Linear(8, 128)

    def forward(self, ids):
        return self.head(self.embedding(ids))


class LlamaRunnerTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(paddle.set_device, paddle.get_device())
        paddle.set_device("cpu")

    def test_entry_imports_do_not_start_runs(self):
        root = Path(__file__).parent
        with mock.patch("paddle.set_device") as device:
            for directory in ("llama4", "llama_guard4"):
                entry = runpy.run_path(str(root / directory / "run_model.py"))
                self.assertTrue(callable(entry["main"]))
            device.assert_not_called()

    def test_model_specific_defaults_and_cli_validation(self):
        for kind, warmup, iterations in (("text", 50, 200), ("guard", 10, 30)):
            options = parse_args(kind, ["--mode", "dynamic", "--output", "unused"])
            self.assertEqual((options.kind, options.warmup, options.iterations), (kind, warmup, iterations))
        for extra in (["--compile-vision"], ["--iterations", "0"], ["--warmup", "-1"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse_args("text", ["--mode", "dynamic", "--output", "unused", *extra])
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            parse_args("guard", ["--mode", "dynamic", "--output", "unused", "--infer"])

    def test_resume_matches_uninterrupted_training_and_emits_segment_probes(self):
        inputs = [paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")]
        paddle.seed(123)
        initial = {name: value.clone() for name, value in TinyNet().state_dict().items()}

        def train_segment(options):
            # Each CLI segment starts in a fresh process. Reset Paddle's names
            # too: AdamW accumulator names are part of its saved state.
            with paddle.utils.unique_name.guard():
                model = TinyNet()
                model.set_state_dict(initial)
                run_training(options, model, model, inputs, [], "test-commit", "initial-hash")
                return model

        with tempfile.TemporaryDirectory() as directory, mock.patch("paddle.device.synchronize"), mock.patch(
            "paddle.get_cuda_rng_state", return_value=[]
        ), mock.patch("paddle.set_cuda_rng_state"):
            root = Path(directory)
            options = parse_args("text", ["--mode", "dynamic", "--output", str(root / "full"), "--steps", "2"])
            options.output.mkdir()
            reference = train_segment(options)
            options.output = root / "first"
            options.output.mkdir()
            options.steps = 1
            train_segment(options)
            options.resume = options.output / "training_state.pdstate"
            options.output = root / "resumed"
            options.output.mkdir()
            options.steps = 2
            segmented = train_segment(options)
            self.assertEqual(json.loads((options.output / "resume.json").read_text())["start_step"], 1)
            self.assertTrue((options.output / "initial.npy").is_file())
            self.assertTrue((options.output / "grads.npz").is_file())
            for name, expected in reference.state_dict().items():
                np.testing.assert_array_equal(segmented.state_dict()[name].numpy(), expected.numpy(), err_msg=name)

    def test_inference_iteration_contract(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("paddle.device.synchronize") as synchronize:
            options = parse_args(
                "text", ["--mode", "dynamic", "--output", directory, "--warmup", "2", "--iterations", "3"]
            )
            net = mock.Mock()
            net.return_value.numpy.return_value = np.ones([1, 2], dtype="float32")
            run_inference(options, net, ["input"])
            net.eval.assert_called_once()
            self.assertEqual(net.call_count, 5)
            self.assertEqual(synchronize.call_count, 10)
            result = json.loads((Path(directory) / "inference.json").read_text())
            self.assertEqual((result["warmup"], result["iterations"], len(result["times_ms"])), (2, 3, 3))
