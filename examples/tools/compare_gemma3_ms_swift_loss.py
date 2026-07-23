#!/usr/bin/env python3
"""Compare a Gemma3 causal-LM loss between ms-swift and Paddle.

The script has two modes because Paddle and PyTorch are normally installed in
separate environments.  Run the ``torch`` mode first; it writes the exact
encoded batch used by ms-swift to ``--batch``.  Run the ``paddle`` mode with
the same checkpoint and batch file afterwards.

Example::

    python compare_gemma3_ms_swift_loss.py torch \
        --model /path/to/gemma3-checkpoint --batch /tmp/gemma3-batch.npz
    python compare_gemma3_ms_swift_loss.py paddle \
        --model /path/to/gemma3-checkpoint --batch /tmp/gemma3-batch.npz

Both modes configure AdamW with the same hyperparameters and record every
step's loss.  The default zero learning rate exercises forward and backward
without allowing an initial framework delta to alter later weights.  Both
sides skip allocating optimizer moments in that no-op case so a 4B check fits
on a single GPU.  Set a nonzero learning rate to exercise update trajectories;
SGD is available for memory-constrained 4B checks.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="framework", required=True)

    torch_parser = subparsers.add_parser("torch", help="run one ms-swift Trainer step")
    torch_parser.add_argument("--model", required=True, help="Torch/Hugging Face Gemma3 checkpoint")
    torch_parser.add_argument("--batch", required=True, type=Path, help="output .npz file for the Paddle run")
    torch_parser.add_argument("--max-length", type=int, default=64)

    paddle_parser = subparsers.add_parser("paddle", help="evaluate the saved batch with Paddle")
    paddle_parser.add_argument("--model", required=True, help="Torch/Hugging Face Gemma3 checkpoint")
    paddle_parser.add_argument("--batch", required=True, type=Path, help=".npz written by torch mode")
    paddle_parser.add_argument("--tolerance", type=float, default=1e-2)

    for subparser in (torch_parser, paddle_parser):
        subparser.add_argument("--max-steps", type=int, default=1)
        subparser.add_argument("--learning-rate", type=float, default=0.0)
        subparser.add_argument("--weight-decay", type=float, default=0.0)
        subparser.add_argument("--adam-beta1", type=float, default=0.9)
        subparser.add_argument("--adam-beta2", type=float, default=0.999)
        subparser.add_argument("--adam-epsilon", type=float, default=1e-8)
        subparser.add_argument("--max-grad-norm", type=float, default=1.0)
        subparser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
        subparser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
        subparser.add_argument("--device", default="cuda:0")
        subparser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


class _CompatSwiftTrainer:  # pragma: no cover - exercised by the torch environment
    """Factory for a Trainer compatible with Transformers 5.x loss calls.

    ms-swift 4.4 wraps ``loss_function`` with a two-positional-argument
    function, while Transformers 5.x passes ``vocab_size`` positionally from
    Gemma3.  Accepting ``*args`` here preserves ms-swift's device fix and
    forwards the complete Transformers call.
    """

    @staticmethod
    def make(base_cls, optimizer_name):
        class CompatTrainer(base_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.step_losses = []

            @contextmanager
            def _patch_loss_function(self):
                model = self.model
                model_cls = model.__class__
                if not hasattr(model_cls, "loss_function"):
                    yield
                    return
                loss_function = model.loss_function
                old_loss_function = model_cls.loss_function

                @staticmethod
                @wraps(loss_function)
                def new_loss_function(logits, labels, *args, **kwargs):
                    labels = labels.to(logits.device)
                    return loss_function(logits, labels, *args, **kwargs)

                model_cls.loss_function = new_loss_function
                try:
                    yield
                finally:
                    model_cls.loss_function = old_loss_function

            def train(self, *args, **kwargs):
                with self._patch_loss_function():
                    return super().train(*args, **kwargs)

            def create_optimizer(self):
                if optimizer_name == "sgd" or self.args.learning_rate == 0:
                    import torch

                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=self.args.learning_rate,
                        weight_decay=self.args.weight_decay,
                    )
                    return self.optimizer
                return super().create_optimizer()

            def compute_loss(self, *args, **kwargs):
                result = super().compute_loss(*args, **kwargs)
                loss = result[0] if isinstance(result, tuple) else result
                self.step_losses.append(float(loss.detach().float().cpu()))
                return result

        return CompatTrainer


def _run_torch(args: argparse.Namespace) -> float:
    if args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        args.device = "cuda:0"

    import numpy as np
    import torch
    from datasets import Dataset
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.trainers import Trainer, TrainingArguments

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model, processor = get_model_processor(
        args.model,
        model_type="gemma3_vision",
        torch_dtype=torch_dtype,
        device_map=args.device,
    )
    template = get_template(processor, template_type="gemma3_vision", max_length=args.max_length)
    template.set_mode("train")
    rows = [
        {
            "messages": [
                {"role": "user", "content": "t4 t5"},
                {"role": "assistant", "content": "t6 t7"},
            ]
        }
        for _ in range(2)
    ]
    encoded = [template.encode(row, return_length=True) for row in rows]
    for item in encoded:
        item["token_type_ids"] = [0] * len(item["input_ids"])
    dataset = Dataset.from_list(encoded)
    collated = template.data_collator([encoded[0]])

    args_training = TrainingArguments(
        output_dir=str(args.batch.parent / "ms-swift-output"),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=args.device == "cpu",
        bf16=args.dtype == "bfloat16" and args.device != "cpu",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="constant",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
    )
    trainer_cls = _CompatSwiftTrainer.make(Trainer, args.optimizer)
    trainer = trainer_cls(model=model, args=args_training, template=template, train_dataset=dataset)
    result = trainer.train()
    losses = trainer.step_losses[: args.max_steps]
    loss = float(result.training_loss)

    args.batch.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.batch,
        input_ids=collated["input_ids"].detach().cpu().numpy(),
        attention_mask=collated["attention_mask"].detach().cpu().numpy(),
        labels=collated["labels"].detach().cpu().numpy(),
        token_type_ids=collated["token_type_ids"].detach().cpu().numpy(),
    )
    metadata = {
        "framework": "ms-swift",
        "model": str(Path(args.model).resolve()),
        "model_type": "gemma3_vision",
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "optimizer": "zero_lr_sgd" if args.learning_rate == 0 else args.optimizer,
        "losses": losses,
        "training_loss": loss,
        "torch": torch.__version__,
    }
    args.batch.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"framework": "ms-swift", "losses": losses, "batch": str(args.batch)}))
    return loss


def _run_paddle(args: argparse.Namespace) -> float:
    import numpy as np
    import paddle
    from paddleformers.transformers.gemma3.modeling import Gemma3ForConditionalGeneration

    paddle.seed(args.seed)
    paddle.set_device(args.device.replace("cuda", "gpu"))
    batch = np.load(args.batch)
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model, dtype=args.dtype)
    model.train()
    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm),
        "parameters": model.parameters(),
    }
    if args.optimizer == "adamw":
        optimizer = paddle.optimizer.AdamW(
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            **optimizer_kwargs,
        )
    else:
        optimizer = paddle.optimizer.SGD(**optimizer_kwargs)
    paddle_batch = {
        "input_ids": paddle.to_tensor(batch["input_ids"], dtype="int64"),
        "attention_mask": paddle.to_tensor(batch["attention_mask"]),
        "labels": paddle.to_tensor(batch["labels"], dtype="int64"),
        "token_type_ids": paddle.to_tensor(batch["token_type_ids"], dtype="int64"),
    }
    losses = []
    for _ in range(args.max_steps):
        outputs = model(**paddle_batch, use_cache=False, return_dict=True)
        loss = outputs.loss
        losses.append(float(loss.detach().astype("float32").cpu().item()))
        loss.backward()
        if args.learning_rate == 0:
            model.clear_gradients()
        else:
            optimizer.step()
            optimizer.clear_grad()

    result = {"framework": "paddle", "losses": losses, "batch": str(args.batch)}
    reference_path = args.batch.with_suffix(".json")
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        torch_losses = reference["losses"]
        if len(torch_losses) != len(losses):
            raise RuntimeError(f"Step count differs: Torch={len(torch_losses)}, Paddle={len(losses)}")
        deltas = [paddle_loss - torch_loss for paddle_loss, torch_loss in zip(losses, torch_losses)]
        result.update({"torch_losses": torch_losses, "loss_deltas": deltas, "tolerance": args.tolerance})
        if any(abs(delta) > args.tolerance for delta in deltas):
            raise RuntimeError(f"Paddle/ms-swift loss deltas {deltas} exceed tolerance {args.tolerance:.6g}")
    print(json.dumps(result))
    return losses[-1]


def main() -> None:
    args = _parse_args()
    if args.framework == "torch":
        _run_torch(args)
    else:
        _run_paddle(args)


if __name__ == "__main__":
    main()
