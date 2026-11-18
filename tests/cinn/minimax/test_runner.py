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

from tests.cinn.minimax.run_model import parse_args, run_inference, run_training


class TinyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = paddle.nn.Embedding(16, 8)
        self.head = paddle.nn.Linear(8, 16)

    def forward(self, ids):
        return self.head(self.embedding(ids))


class MiniMaxRunnerTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(paddle.set_device, paddle.get_device())
        paddle.set_device("cpu")

    def test_import_does_not_start_training(self):
        with mock.patch("paddle.set_device") as device:
            module = runpy.run_path(str(Path(__file__).with_name("run_model.py")))
            self.assertTrue(callable(module["main"]))
            device.assert_not_called()

    def test_invalid_inference_fails_before_gpu_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            with self.assertRaisesRegex(ValueError, "initial-checkpoint"):
                parse_args(["--mode", "dynamic", "--output", str(output), "--inference-only"])
            self.assertFalse(output.exists())

    def test_resume_matches_full_run_and_keeps_probes(self):
        tokens = [[1, 2, 3, 4]] * 20
        labels = [[2, 3, 4, 5]] * 20
        initial = {name: value.clone() for name, value in TinyNet().state_dict().items()}
        meta = {"mode": "dynamic", "source_commit": "test", "moe_implementation": "production_functional_accumulation"}

        def train_segment(options):
            with paddle.utils.unique_name.guard():
                model = TinyNet()
                model.set_state_dict(initial)
                run_training(options, model, model, tokens, labels, dict(meta))
                return model

        with tempfile.TemporaryDirectory() as directory, mock.patch("paddle.device.synchronize"), mock.patch(
            "paddle.get_cuda_rng_state", return_value=[]
        ), mock.patch("paddle.set_cuda_rng_state"):
            root = Path(directory)
            options = parse_args(
                ["--mode", "dynamic", "--output", str(root / "full"), "--steps", "2", "--probe-gradients"]
            )
            options.output.mkdir()
            expected = train_segment(options)
            options.output = root / "first"
            options.output.mkdir()
            options.steps = 1
            train_segment(options)
            options.resume = options.output / "training_state.pdstate"
            options.output = root / "resumed"
            options.output.mkdir()
            options.steps = 2
            actual = train_segment(options)
            self.assertEqual(json.loads((options.output / "run.json").read_text())["start_step"], 1)
            self.assertTrue((options.output / "initial_logits.npy").is_file())
            self.assertTrue((options.output / "first_gradients.pdparams").is_file())
            for name, tensor in expected.state_dict().items():
                np.testing.assert_array_equal(actual.state_dict()[name].numpy(), tensor.numpy(), err_msg=name)

    def test_independent_inference_does_not_train(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("paddle.device.synchronize"):
            options = parse_args(["--mode", "dynamic", "--output", directory])
            net = mock.Mock()
            net.return_value.numpy.return_value = np.ones([1, 2], dtype="float32")
            run_inference(options, net, [[1, 2]])
            net.eval.assert_called_once()
            net.train.assert_not_called()
            self.assertEqual(net.call_count, 40)
            result = json.loads((Path(directory) / "inference.json").read_text())
            self.assertEqual(len(result["times_ms"]), 30)
