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

import json
import logging

import paddle
from paddle.io import DataLoader, Dataset
from tqdm import tqdm

from ...data import DataCollatorForSeq2Seq
from ...transformers import AutoTokenizer
from .modeling_utils import ReftDataCollator

device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"


def make_data_collator(tokenizer, model, max_length):
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=max_length,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(
    dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool
) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


def do_predict(
    intervenable,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    batch_size: int = 4,
    data_collator=None,
    greedy_decoding=True,
    temperature=None,
    top_p=None,
    top_k=None,
    max_new_tokens=32,
    do_sample=False,
    predict_path=None,
    num_beams=4,
    max_length=2048,
):
    # switch the tokenizer mode first for generation tasks
    tokenizer.padding_side = "left"  # switch padding side for collator
    if greedy_decoding:
        num_beams = 1
    data_collator = make_data_collator(tokenizer, intervenable.model, max_length)
    eval_dataloader = make_dataloader(eval_dataset, batch_size, data_collator, shuffle=False)
    generations = []
    eval_iterator = tqdm(eval_dataloader, position=0, leave=True)
    with paddle.no_grad():
        for step, inputs in enumerate(eval_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, paddle.Tensor):
                    inputs[k] = v.to(device)

            # [layers, batch_size, positions]
            intervention_locations = paddle.transpose(inputs["intervention_locations"], perm=[1, 0, 2])
            # get left padding count, [batch_size], and add to locations
            left_padding = (inputs["input_ids"] == tokenizer.bos_token_id).nonzero(as_tuple=True)[1]

            if left_padding.numel() > 0:
                left_padding = left_padding.reshape([1, -1, 1]).to(device)  # [1, batch_size, 1]
                intervention_locations += left_padding
                # intervention_locations -= 1  # offset for the sink padding
            else:
                logging.info("Warning: No BOS token found, skipping left padding adjustment.")

            # repeat each batch by num_beams times in intervention locations
            # -> [layers, batch_size * num_beams, positions]
            intervention_locations = intervention_locations.repeat_interleave(num_beams, axis=1).tolist()

            # set generation args depending on task
            generation_args = {
                "base": {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                "unit_locations": intervention_locations,
                "intervene_on_prompt": True,
                "eos_token_id": tokenizer.eos_token_id,
                "early_stopping": True,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            # override generation args if necessary
            if temperature is not None:
                generation_args["temperature"] = temperature
            if top_p is not None:
                generation_args["top_p"] = top_p
            if top_k is not None:
                generation_args["top_k"] = top_k

            # generate with intervention on prompt
            _, steered_response = intervenable.generate(**generation_args)
            # detokenize in batch
            actual_preds = tokenizer.batch_decode(steered_response[0], skip_special_tokens=True)

            for inputs_id, label, pred in zip(inputs["input_ids"], inputs["labels"], actual_preds):
                filtered_labels = label[label != -100]
                generations += [
                    {
                        "src": tokenizer.decode(inputs_id, skip_special_tokens=True),
                        "trg": tokenizer.decode(filtered_labels, skip_special_tokens=True),
                        "pred": pred,
                    }
                ]

            if predict_path is not None:
                with open(predict_path, "w") as json_file:
                    json.dump(generations, json_file, indent=4, ensure_ascii=False)

    return generations
