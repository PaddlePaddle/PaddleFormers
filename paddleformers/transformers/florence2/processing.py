# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

from transformers import BartTokenizer, CLIPImageProcessor

from ..image_processing_utils import BatchFeature
from ..processing_utils import ProcessorMixin


class Florence2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        return BartTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None or tokenizer is None:
            raise ValueError("Florence2Processor requires both image_processor and tokenizer.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an image_seq_length attribute.")
        self.image_seq_length = image_processor.image_seq_length
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": getattr(tokenizer, "additional_special_tokens", [])
                + ["<od>", "</od>", "<ocr>", "</ocr>"]
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
                ],
            }
        )
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
        super().__init__(image_processor, tokenizer)

    def _construct_prompts(self, texts):
        prompts = []
        for text in texts:
            for token, prompt in self.task_prompts_without_inputs.items():
                if token in text:
                    if text != token:
                        raise ValueError(f"Task token {token} must be the only token in the text.")
                    text = prompt
                    break
            for token, prompt in self.task_prompts_with_input.items():
                if token in text:
                    text = prompt.format(input=text.replace(token, ""))
                    break
            prompts.append(text)
        return prompts

    def __call__(
        self,
        text=None,
        images=None,
        tokenize_newline_separately=True,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors="pd",
        **kwargs,
    ):
        if images is None:
            raise ValueError("`images` are expected as arguments to a Florence2Processor instance.")
        if text is None:
            text = ""
        if isinstance(text, str):
            text = [text]
        if isinstance(images, (list, tuple)) and len(images) < len(text):
            raise ValueError("Each prompt must be associated with an image.")
        component_tensor_type = "np" if return_tensors == "pd" else return_tensors
        pixel_values = self.image_processor(images, return_tensors=component_tensor_type, **kwargs)["pixel_values"]
        if max_length is not None:
            max_length -= self.image_seq_length
        inputs = self.tokenizer(
            self._construct_prompts(text),
            return_tensors=component_tensor_type,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_token_type_ids=False,
        )
        return BatchFeature(data={**inputs, "pixel_values": pixel_values}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))


__all__ = ["Florence2Processor", "CLIPImageProcessor", "BartTokenizer"]
