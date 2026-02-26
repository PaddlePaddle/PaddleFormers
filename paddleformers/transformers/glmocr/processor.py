from typing import Optional, Union
import numpy as np

from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import PreTokenizedInput, TextInput
class GlmOcrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class GlmOcrProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]  # ✅ 关键：删掉 video_processor
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("GlmOcrTokenizer", "GlmOcrTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)  # ✅ 现在只需要 2 个

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[GlmOcrProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            GlmOcrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens.item(), 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])  # ✅ 删掉 video

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)  # ✅ 删 videos_inputs

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = GlmOcrProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["GlmOcrProcessor"]