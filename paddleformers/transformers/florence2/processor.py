# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import re

from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import ProcessorMixin
from ..tokenizer_utils_base import PreTokenizedInput, TextInput

__all__ = ["Florence2Processor"]


class Florence2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls,
        sub_processor_type,
        pretrained_model_name_or_path,
        subfolder="",
        **kwargs,
    ):
        kwargs.setdefault("tokenizer_type", "bart")
        return super()._load_tokenizer_from_pretrained(
            sub_processor_type, pretrained_model_name_or_path, subfolder=subfolder, **kwargs
        )

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if image_processor is None or tokenizer is None:
            raise ValueError("Florence2Processor requires both an image processor and a tokenizer.")

        tokens = (
            ["<od>", "</od>", "<ocr>", "</ocr>"]
            + [f"<loc_{index}>" for index in range(1000)]
            + [
                "<cap>",
                "</cap>",
                "<ncap>",
                "</ncap>",
                "<dcap>",
                "</dcap>",
                "<grounding>",
                "</grounding>",
                "<seg>",
                "</seg>",
                "<sep>",
                "<region_cap>",
                "</region_cap>",
                "<region_to_desciption>",
                "</region_to_desciption>",
                "<proposal>",
                "</proposal>",
                "<poly>",
                "</poly>",
                "<and>",
            ]
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(getattr(tokenizer, "additional_special_tokens", [])) + tokens}
        )
        self.image_seq_length = getattr(image_processor, "image_seq_length", 577)
        self.task_prompts_without_inputs = {
            "<OCR>": "What is the text in the image?",
            "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
            "<CAPTION>": "What does the image describe?",
            "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
            "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
            "<OD>": "Locate the objects with category name in the image.",
            "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
            "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
        }
        self.task_prompts_with_input = {
            "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
            "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
            "<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
            "<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
            "<REGION_TO_CATEGORY>": "What is the region {input}?",
            "<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
            "<REGION_TO_OCR>": "What text is in the region {input}?",
        }
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _construct_prompts(self, texts):
        prompts = []
        for text in texts:
            for task, prompt in self.task_prompts_without_inputs.items():
                if task in text:
                    if text != task:
                        raise ValueError(f"Task token {task} must be the only token in the prompt.")
                    text = prompt
                    break
            for task, prompt in self.task_prompts_with_input.items():
                if task in text:
                    text = prompt.format(input=text.replace(task, ""))
                    break
            prompts.append(text)
        return prompts

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        return_tensors="pd",
        padding=False,
        truncation=None,
        max_length=None,
        **kwargs,
    ):
        if images is None:
            raise ValueError("`images` must be provided to Florence2Processor.")
        texts = text if isinstance(text, list) else [text or ""]
        if isinstance(images, list) and len(images) < len(texts):
            raise ValueError("Each Florence-2 prompt must have an associated image.")

        image_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "do_resize",
                "do_normalize",
                "image_mean",
                "image_std",
                "data_format",
                "input_data_format",
                "resample",
                "do_convert_rgb",
                "do_rescale",
            }
            and value is not None
        }
        image_inputs = self.image_processor(images=images, return_tensors=return_tensors, **image_kwargs)
        if max_length is not None:
            max_length -= self.image_seq_length
        text_inputs = self.tokenizer(
            self._construct_prompts(texts),
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_token_type_ids=False,
        )
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))

    @staticmethod
    def _dequantize(values, image_size):
        width, height = image_size
        return [(value + 0.5) * (width if index % 2 == 0 else height) / 1000 for index, value in enumerate(values)]

    def _parse_polygons(self, text, image_size):
        polygons, labels = [], []
        pattern = r"([^<]*)(?:<poly>)?((?:<loc_\d+>|<sep>)+)(?:</poly>)?"
        for phrase, encoded_polygons in re.findall(pattern, text):
            instance = []
            for encoded_polygon in encoded_polygons.split("<sep>"):
                values = [int(value) for value in re.findall(r"<loc_(\d+)>", encoded_polygon)]
                if len(values) >= 6 and len(values) % 2 == 0:
                    instance.append(self._dequantize(values, image_size))
            if instance:
                polygons.append(instance)
                labels.append(phrase.strip())
        return {"polygons": polygons, "labels": labels}

    def _parse_bboxes(self, text, image_size):
        phrase_pattern = r"([^<]+)((?:<loc_\d+>){4,})"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
        bboxes, labels = [], []
        for phrase, encoded_boxes in re.findall(phrase_pattern, text):
            for box in re.findall(box_pattern, encoded_boxes):
                bboxes.append(self._dequantize([int(value) for value in box], image_size))
                labels.append(phrase.strip())
        return {"bboxes": bboxes, "labels": labels}

    def post_process_generation(self, text, task, image_size):
        clean_text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        if task in {
            "<OCR>",
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<REGION_TO_CATEGORY>",
            "<REGION_TO_DESCRIPTION>",
            "<REGION_TO_OCR>",
        }:
            return {task: clean_text}
        if task == "<OCR_WITH_REGION>":
            pattern = r"(.+?)" + "".join([r"<loc_(\d+)>"] * 8)
            matches = re.findall(pattern, clean_text)
            return {
                task: {
                    "quad_boxes": [
                        self._dequantize([int(value) for value in match[1:]], image_size) for match in matches
                    ],
                    "labels": [match[0] for match in matches],
                }
            }
        if task in {"<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"}:
            return {task: self._parse_polygons(clean_text, image_size)}
        if task == "<REGION_PROPOSAL>":
            values = [int(value) for value in re.findall(r"<loc_(\d+)>", clean_text)]
            bboxes = [
                self._dequantize(values[index : index + 4], image_size) for index in range(0, len(values) - 3, 4)
            ]
            return {task: {"bboxes": bboxes, "labels": [""] * len(bboxes)}}
        if task == "<OPEN_VOCABULARY_DETECTION>" and "<poly>" in clean_text:
            return {task: self._parse_polygons(clean_text, image_size)}

        return {task: self._parse_bboxes(clean_text, image_size)}
