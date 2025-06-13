# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from datasets import Dataset
from paddle.distributed import fleet
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

from ..data import DataCollator, DataCollatorForSeq2Seq
from ..trainer import Trainer
from ..trainer.trainer_callback import TrainerCallback
from ..trainer.trainer_utils import EvalPrediction, has_length
from ..transformers import AutoModelForCausalLM, AutoTokenizer
from ..transformers.model_utils import PretrainedModel
from ..transformers.tokenizer_utils import PretrainedTokenizer
from ..utils.log import logger
from .extras.dataset_formatting import get_formatting_func_from_dataset
from .sft_config import SFTConfig

__all__ = ["SFTTrainer"]


class SFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: SFTConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
        do_generation: bool = False,
        gen_args=None,
        data_args=None,
        formatting_func: Optional[Callable] = None,
    ):

        self.do_generation = do_generation
        self.gen_args = gen_args
        self.data_args = data_args
        if self.do_generation:
            assert gen_args is not None
            assert data_args is not None

        if args is None:
            output_dir = "tmp_trainer"
            warnings.warn(f"No `SFTConfig` passed, using `output_dir={output_dir}`.")
            args = SFTConfig(output_dir=output_dir)
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = SFTConfig(**args_as_dict)

        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs to the SFTConfig, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            dtype = model_init_kwargs.get("dtype")
            if dtype is not None:
                # Convert to `paddle.dtype` if an str is passed
                if isinstance(dtype, str) and dtype != "auto":
                    dtype = getattr(paddle, dtype)
                if dtype != "auto" and not isinstance(dtype, paddle.dtype):
                    raise ValueError(
                        f"Invalid `dtype` passed to the SFTConfig. Expected a string with either `paddle.dtype` or 'auto', but got {dtype}."
                    )
                model_init_kwargs["dtype"] = dtype

        name_or_path = None
        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            name_or_path = model
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if tokenizer is None:
            if name_or_path is not None:
                tokenizer = AutoTokenizer.from_pretrained(name_or_path)
            else:
                raise ValueError("Please pass tokenizer")
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            args.max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {args.max_seq_length}"
            )

        self.dataset_num_proc = args.dataset_num_proc
        self.dataset_batch_size = args.dataset_batch_size

        if args.dataset_kwargs is None:
            args.dataset_kwargs = {}

        if formatting_func is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                args.dataset_kwargs["add_special_tokens"] = False

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with args.main_process_first():
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    tokenizer,
                    args.dataset_text_field,
                    args.max_seq_length,
                    formatting_func,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        tokenizer,
                        args.dataset_text_field,
                        args.max_seq_length,
                        formatting_func,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]
        if data_collator is None:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        super().__init__(
            model=model,
            criterion=criterion,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        dataset_text_field: str,
        max_seq_length,
        formatting_func: Optional[Callable],
        remove_unused_columns=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):

        if dataset is None:
            raise ValueError("The dataset should not be None")

        if skip_prepare_dataset:
            return dataset

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored."
                )

            def formatting_func(x):
                return x["input_ids"]

            return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(dataset, (paddle.io.IterableDataset, paddle.io.Dataset)) and not isinstance(
            dataset, datasets.IterableDataset
        ):
            return dataset

        return self._prepare_non_packed_dataloader(
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func,
            add_special_tokens,
            remove_unused_columns,
        )

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field: str,
        max_seq_length,
        formatting_func: Optional[Callable] = None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(formatting_func(element), list):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )
            labels = []
            if tokenizer.pad_token_id is not None:
                # raise ValueError(type(outputs["input_ids"]))
                if isinstance(outputs["input_ids"][0], list):
                    for x in outputs["input_ids"]:
                        sublabels = []
                        for y in x:
                            sublabels.append(-100 if y == tokenizer.pad_token_id else y)
                        sublabels.append(-100)
                        sublabels = sublabels[1:]
                        labels.append(sublabels)
                else:
                    for x in outputs["input_ids"]:
                        labels.append(-100 if x == tokenizer.pad_token_id else x)
                    labels.append(-100)
                    labels = labels[1:]

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "labels": labels}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:  # None for IterableDataset
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
            "batch_size": self.dataset_batch_size,
        }
        if isinstance(dataset, datasets.Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)

        print(tokenized_dataset[0])
        return tokenized_dataset

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if prediction_loss_only or self.args.pipeline_parallel_degree > 1:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            # all gather logits when enabling tensor_parallel_output
            if self.args.tensor_parallel_degree > 1 and getattr(self.args, "tensor_parallel_output", False):
                hcg = fleet.get_hybrid_communicate_group()
                model_parallel_group = hcg.get_model_parallel_group()
                gathered_logits = []
                dist.all_gather(gathered_logits, logits, group=model_parallel_group)
                logits = paddle.concat(gathered_logits, axis=-1)
            return (loss, logits.argmax(axis=-1, keepdim=True), labels)

        loss = None

        model.eval()
        with paddle.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                position_ids=inputs["position_ids"] if "position_ids" in inputs else None,
                max_length=max(self.data_args.max_length - inputs["input_ids"].shape[-1], 1),
                decode_strategy="sampling",
                top_k=self.gen_args.top_k,
                top_p=self.gen_args.top_p,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )[0]
            all_preds = []
            for pred_tokens in generated_tokens:
                pred_tokens = pred_tokens.numpy()
                pred_tokens = pred_tokens[pred_tokens != self.tokenizer.pad_token_id].tolist()
                all_preds.append(pred_tokens)
            max_pred_length = max([len(x) for x in all_preds])
            for index, preds in enumerate(all_preds):
                all_preds[index] = preds + [-100] * (max_pred_length - len(preds))
            all_preds = paddle.to_tensor(all_preds)

            if "labels" in inputs:
                all_labels = paddle.to_tensor(inputs["labels"])
            else:
                all_labels = None

        return (loss, all_preds, all_labels)

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        if "loss" in logs:
            logs["ppl"] = np.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = np.exp(logs["eval_loss"])

        super(SFTTrainer, self).log(logs, **kwargs)

    def get_ptq_dataloader(self, ptq_ds):
        if self.args.world_size <= 1:
            ptq_sampler = BatchSampler(
                dataset=ptq_ds,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            ptq_sampler = DistributedBatchSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                num_replicas=self.args.dataset_world_size,
                rank=self.args.dataset_rank,
                drop_last=self.args.dataloader_drop_last,
            )
        ptq_dataloader = DataLoader(
            ptq_ds,
            batch_sampler=ptq_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
        return ptq_dataloader

    def ptq_loop(
        self,
        dataloader: DataLoader,
        description: str,
        max_eval_iters: Optional[int] = -1,
    ):
        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total {description} steps = {max_eval_iters}")
            else:
                logger.info(f"  Total {description} steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total {description} steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")
        self.model.eval()
        with paddle.no_grad():
            for step, inputs in enumerate(dataloader):
                self.prediction_step(model=self.model, inputs=inputs, prediction_loss_only=True, ignore_keys=None)
                if max_eval_iters > 0 and step >= max_eval_iters - 1:
                    break
