# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import paddle

from ..transformers.tokenizer_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PretrainedTokenizerBase,
)

__all__ = [
    "DataCollatorWithPadding",
    "default_data_collator",
    "DataCollator",
    "DefaultDataCollator",
    "DataCollatorForTokenClassification",
    "DataCollatorForSeq2Seq",
    "DataCollatorForLanguageModeling",
    # "DataCollatorForWholeWordMask",
    "DataCollatorForEmbedding",
]

InputDataClass = NewType("InputDataClass", Any)
"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PaddlePaddle tensors or NumPy arrays.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "pd":
            return self.paddle_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


def default_data_collator(features: List[InputDataClass], return_tensors="pd") -> Dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    if return_tensors == "pd":
        return paddle_default_data_collator(features)
    elif return_tensors == "np":
        return numpy_default_data_collator(features)


def paddle_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], paddle.Tensor) else first["label"]
        dtype = "int64" if isinstance(label, int) else "float32"
        batch["labels"] = paddle.to_tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], paddle.Tensor):
            batch["labels"] = paddle.stack([f["label_ids"] for f in features])
        else:
            dtype = "int64" if type(first["label_ids"][0]) is int or np.int32 or np.int64 else "float32"
            batch["labels"] = paddle.to_tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, paddle.Tensor):
                batch[k] = paddle.stack([f[k] for f in features])
            else:
                batch[k] = paddle.to_tensor([f[k] for f in features])

    return batch


def numpy_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], np.ndarray) else first["label"]
        dtype = np.int64 if isinstance(label, int) else np.float32
        batch["labels"] = np.array([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], np.ndarray):
            batch["labels"] = np.stack([f["label_ids"] for f in features])
        else:
            dtype = np.int64 if type(first["label_ids"][0]) is int or np.int32 or np.int64 else np.float32
            batch["labels"] = np.array([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, np.ndarray):
                batch[k] = np.stack([f[k] for f in features])
            else:
                batch[k] = np.array([f[k] for f in features])

    return batch


@dataclass
class DefaultDataCollator(DataCollatorMixin):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    This is an object (like other data collators) rather than a pure function like default_data_collator. This can be
    helpful if you need to set a return_tensors value at initialization.
    Args:
        return_tensors (`bool`):
            Return Tensor or numpy array.
    """

    return_tensors: str = "pd"

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        return default_data_collator(features, return_tensors)


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        tokenizer (`paddleformers.transformers.PretrainedTokenizer`):
            The tokenizer used for encoding the data.
    """

    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pd"
    return_attention_mask: Optional[bool] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            return_attention_mask=self.return_attention_mask,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        # To fix windows bug for paddle inference dtype error
        # InvalidArgumentError: The type of data we are trying to retrieve does not match the type of data currently contained in the container
        if self.return_tensors == "np":
            batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        return batch


@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PretrainedTokenizer`] or [`PretrainedFasterTokenizer`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PaddlePaddle loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pd"

    def paddle_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pd" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = paddle.to_tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, paddle.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch = {k: paddle.to_tensor(v, dtype="int64") for k, v in batch.items()}
        return batch

    def numpy_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="np" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = np.array(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        return batch


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PretrainedTokenizer`] or [`PretrainedFasterTokenizer`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PaddlePaddle loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
        max_label_length (`int`, *optional*, Pad label to max_label_length. defaults to `None`):
    """

    tokenizer: PretrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pd"
    return_attention_mask: Optional[bool] = None
    max_label_length: Optional[int] = None

    def __call__(self, features, return_tensors=None):
        # Deep copy to avoid modifying features in-place
        batch = copy.deepcopy(features)
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in batch] if "labels" in batch[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            # Note(gongenlei): In pipeline, max_label_length = self.max_length
            if self.max_label_length is not None:
                max_label_length = self.max_label_length
            else:
                max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in batch:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=self.return_attention_mask,
        )
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        if "labels" in batch.keys():
            value = batch.pop("labels")
            batch["labels"] = value

        return batch


@dataclass
class DataCollatorForEmbedding:
    tokenizer: PretrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pd"
    return_attention_mask: Optional[bool] = None
    max_label_length: Optional[int] = None
    return_position_ids: Optional[bool] = True

    max_query_len: int = 512
    max_passage_len: int = 512

    def __call__(self, batch, return_tensors=None) -> Any:
        """Convert batch data into tensor."""
        input_keys = ["input_ids", "position_ids"]

        attn_key = "attention_mask"
        input_keys.append(attn_key)

        # Initialize query and passage lists
        queries = {key: [] for key in input_keys}
        passages = {key: [] for key in input_keys}

        batch_query_embedding_indices = []
        batch_passage_embedding_indices = []

        global_passage_idx = 0

        # Process each batch sequence
        for idx, batch_sequence in enumerate(batch):
            query_data = [pair.query for pair in batch_sequence]
            padded_query_token_ids, padded_query_position_ids, query_token_ids = self.process_data(
                query_data, self.tokenizer.pad_token_id, self.max_query_len
            )

            queries["input_ids"].append(padded_query_token_ids)
            queries["position_ids"].append(padded_query_position_ids)
            batch_query_embedding_indices.append([idx, len(query_token_ids[0]) - 1])

            queries[attn_key].append(self.gen_self_attn_mask(query_token_ids, self.max_query_len))

            for pair in batch_sequence:
                for passage in pair.passages:
                    passage_data = [passage]
                    padded_passage_token_ids, padded_passage_position_ids, passage_token_ids = self.process_data(
                        passage_data, self.tokenizer.pad_token_id, self.max_passage_len
                    )

                    passages["input_ids"].append(padded_passage_token_ids)
                    passages["position_ids"].append(padded_passage_position_ids)
                    batch_passage_embedding_indices.append([global_passage_idx, len(passage_token_ids[0]) - 1])

                    passages[attn_key].append(self.gen_self_attn_mask(passage_token_ids, self.max_passage_len))
                    global_passage_idx += 1

        for data in (queries, passages):
            for k, v in data.items():
                data[k] = paddle.to_tensor(np.concatenate(v))

        queries["embedding_indices"] = paddle.to_tensor(np.array(batch_query_embedding_indices, dtype="int32"))
        passages["embedding_indices"] = paddle.to_tensor(np.array(batch_passage_embedding_indices, dtype="int32"))

        if not self.return_position_ids:
            del queries["position_ids"]
            del passages["position_ids"]

        return {
            "query": queries,
            "passages": passages,
        }

    def process_data(self, data, pad_idx, max_len):
        """padding token_ids & position_ids."""
        token_ids = [sum((item.token_ids for item in data), [])]
        position_ids = [sum((item.position_ids for item in data), [])]
        padded_token_ids = self.pad_batch_data(token_ids, pad_id=pad_idx, max_seq_len=max_len)
        padded_position_ids = self.pad_batch_data(position_ids, pad_id=0, max_seq_len=max_len)
        return padded_token_ids, padded_position_ids, token_ids

    @staticmethod
    def pad_batch_data(insts, pad_id=0, max_seq_len=None, return_seq_len=False, pad_style="right"):
        """Pad sequences to the max sequence length in batch."""
        max_len = max_seq_len if max_seq_len is not None else max(map(len, insts))
        if pad_style == "left":
            inst_data = np.array([[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts])
        else:
            inst_data = np.array([list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])

        if return_seq_len:
            seq_len = np.array([len(inst) for inst in insts])
            return inst_data.astype("int64").reshape([-1, max_len]), seq_len
        else:
            return inst_data.astype("int64").reshape([-1, max_len])

    @staticmethod
    def gen_self_attn_mask(batch_token_ids: List[List[int]], max_seq_len: int):
        """Generate self attention mask for multiple sub-sequence."""
        input_mask_data = np.zeros((1, max_seq_len), dtype="float32")
        offset = 0
        for index, token_ids in enumerate(batch_token_ids):
            cur_len = len(token_ids)
            b = np.ones([cur_len])
            input_mask_data[0, offset : offset + cur_len] = b
            offset += cur_len
        return input_mask_data

    @staticmethod
    def gen_attn_mask_start_row_indices(batch_token_ids: List[List[int]], max_seq_len: int, sliding_window: int):
        """Generate attn_mask_start_row_indices for flash attention."""
        offset = 0
        attn_mask_start_row_indices = []
        for token_ids in batch_token_ids:
            cur_len = len(token_ids)
            if sliding_window > 0:
                for i in range(cur_len):
                    attn_mask_start_row_indices.append(offset + min(cur_len, i + sliding_window))
            else:
                attn_mask_start_row_indices.extend([offset + cur_len] * cur_len)
            offset += cur_len
        if offset < max_seq_len:
            attn_mask_start_row_indices.extend(list(range(offset + 1, max_seq_len + 1)))

        return np.array(attn_mask_start_row_indices, dtype=np.int32)[None, None]


def _paddle_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import paddle

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [paddle.to_tensor(e, dtype="int64") for e in examples]

    length_of_first = examples[0].shape[0]

    # Check if padding is necessary.

    are_tensors_same_length = all(x.shape[0] == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return paddle.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.shape[0] for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    # result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    result = paddle.full([len(examples), max_length], tokenizer.pad_token_id, dtype=examples[0].dtype)

    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def _numpy_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    import numpy as np

    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [np.array(e, dtype=np.int64) for e in examples]

    # Check if padding is necessary.
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return np.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = np.full(shape=(len(examples), max_length), fill_value=tokenizer.pad_token_id, dtype=examples[0].dtype)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.cpu().numpy()
    return x.tolist()


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>"""

    tokenizer: PretrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pd"

    def paddle_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pd", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _paddle_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.paddle_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def paddle_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import paddle

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = paddle.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]

            special_tokens_mask = paddle.to_tensor(special_tokens_mask, dtype="bool")
        else:
            special_tokens_mask = special_tokens_mask.cast("bool")

        def masked_fill(x, mask, value):
            y = paddle.full(x.shape, value, x.dtype)
            return paddle.where(mask.to("bool"), y, x)

        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix = masked_fill(probability_matrix, special_tokens_mask, value=0.0)
        masked_indices = paddle.bernoulli(probability_matrix).cast("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = paddle.bernoulli(paddle.full(labels.shape, 0.8)).cast("bool") & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            paddle.bernoulli(paddle.full(labels.shape, 0.5)).cast("bool") & masked_indices & ~indices_replaced
        )
        random_words = paddle.randint(len(self.tokenizer), shape=labels.shape, dtype="int64")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = paddle.bernoulli(paddle.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
