# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import bisect
import re
from typing import Sequence, Union

import numpy as np
from typing_extensions import Unpack

from ...utils import auto_docstring
from ..audio_utils import AudioInput, load_audio
from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import (
    AllKwargsForChatTemplate,
    ProcessingKwargs,
    ProcessorMixin,
    VideosKwargs,
    render_jinja_template,
)
from ..tokenizer_utils_base import TextInput
from ..video_utils import VideoInput, make_batched_videos


class Qwen3OmniMoeVideosKwargs(VideosKwargs, total=False):
    """
    min_pixels (`int`, *optional*):
        Minimum number of pixels (height × width) for video frames after resizing. Frames smaller than this
        threshold will be upscaled to meet the minimum requirement.
    max_pixels (`int`, *optional*):
        Maximum number of pixels (height × width) for video frames after resizing. Frames larger than this
        threshold will be downscaled to fit within the limit.
    patch_size (`int`, *optional*):
        The spatial patch size used by the vision encoder. Video frames are divided into patches of this size
        in both height and width dimensions.
    temporal_patch_size (`int`, *optional*):
        The temporal patch size used by the vision encoder. This determines how many consecutive frames are
        grouped together as a single temporal patch.
    merge_size (`int`, *optional*):
        The merge size used for combining spatial patches. Multiple patches are merged together to reduce the
        sequence length while maintaining spatial information.
    min_frames (`int`, *optional*):
        Minimum number of frames to extract from the video. Videos with fewer frames will be padded or repeated
        to meet this requirement.
    max_frames (`int`, *optional*):
        Maximum number of frames to extract from the video. Longer videos will be truncated or sampled to fit
        within this limit.
    use_audio_in_video (`bool`, *optional*, defaults to `False`):
        Whether to incorporate audio information when processing videos. When enabled, audio tokens are
        interleaved with video tokens based on temporal alignment, creating a unified multimodal representation.
    seconds_per_chunk (`float`, *optional*, defaults to `2.0`):
        The duration (in seconds) of each video chunk when splitting long videos. This parameter controls how
        videos are divided into temporal segments for processing.
    position_id_per_seconds (`int` or `float`, *optional*, defaults to `25`):
        The number of position IDs allocated per second of video. This parameter controls the temporal resolution
        of position embeddings and is used to align video tokens with audio tokens when `use_audio_in_video=True`.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int
    use_audio_in_video: bool
    seconds_per_chunk: float
    position_id_per_seconds: int | float


class Qwen3OmniMoeProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen3OmniMoeVideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 13.0,
            "use_audio_in_video": False,
            "size": {
                "shortest_edge": 128 * 32 * 32,
                "longest_edge": 768 * 32 * 32,
            },
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
        },
    }


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3OmniMoeProcessor(ProcessorMixin):
    def __init__(
        self, image_processor=None, video_processor=None, feature_extractor=None, tokenizer=None, chat_template=None
    ):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(
        self,
        text: TextInput = None,
        images: ImageInput | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3OmniMoeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        fps = output_kwargs["videos_kwargs"].get("fps", 1.0)

        if audio is not None:
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos = make_batched_videos(videos)
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            fps = [fps] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                self.video_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = iter(videos_inputs["video_grid_thw"])
            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
            image_grid_thw,
            video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        # Extend mm token length
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths).item(), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length.item(), 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length_video
                        sample = sample.replace(self.video_token, "<|video_placeholder|>" * video_seq_length.item(), 1)
                    else:
                        audio_token_indices = np.arange(next(audio_lengths))
                        curr_video_grid_thw = next(video_grid_thw)
                        height = curr_video_grid_thw[1] // self.video_processor.merge_size
                        width = curr_video_grid_thw[2] // self.video_processor.merge_size
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = np.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        video_data_index, audio_data_index = 0, 0
                        placeholder_string = self.vision_bos_token + self.audio_bos_token
                        while video_data_index < len(video_token_indices) and audio_data_index < len(
                            audio_token_indices
                        ):
                            if video_token_indices[video_data_index] <= audio_token_indices[audio_data_index]:
                                placeholder_string += "<|video_placeholder|>"
                                video_data_index += 1
                            else:
                                placeholder_string += "<|audio_placeholder|>"
                                audio_data_index += 1
                        if video_data_index < len(video_token_indices):
                            placeholder_string += "<|video_placeholder|>" * (
                                len(video_token_indices) - video_data_index
                            )
                        if audio_data_index < len(audio_token_indices):
                            placeholder_string += "<|audio_placeholder|>" * (
                                len(audio_token_indices) - audio_data_index
                            )
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`np.ndarray`): A monotonically increasing list of token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).

        Returns:
            `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(generated_outputs[0], skip_special_tokens=skip_special_tokens, **kwargs)

    def post_process_multimodal_output(
        self, generated_outputs, skip_special_tokens=True, generation_mode=None, **kwargs
    ):
        """
        Post-process the output of a multimodal model to return the requested modality output.
        If the model cannot generated the requested modality, an error will be raised.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            generation_mode (`str`, *optional*):
                Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[Inion[str, np.ndarray]]`: The decoded text or generated audio.
        """
        if generation_mode is None or generation_mode == "text":
            return self.post_process_image_text_to_text(
                generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
            )

        elif generation_mode == "audio":
            # model supports only bs=1, so we will never get several audio outputs
            audio = generated_outputs[1].reshape(-1).detach().cpu().numpy()
            return [audio]

        else:
            raise ValueError(
                f"{self.__class__.__name__} got an unexpected generation_mode={generation_mode}. Supported options are only `text` and `audio"
            )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + video_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                # It's the name of a template, not a full template string
                chat_template = self.chat_template[chat_template]
            else:
                # It's a template string, render it directly
                pass

        # Check if tokenizer is fast - use backend attribute if available, otherwise fall back to class name
        is_tokenizers_fast = False
        if hasattr(self, "tokenizer"):
            if hasattr(self.tokenizer, "backend"):
                is_tokenizers_fast = self.tokenizer.backend == "tokenizers"
            else:
                # Fallback to class name check
                is_tokenizers_fast = self.tokenizer.__class__.__name__.endswith("Fast")

        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if kwargs.get("return_assistant_tokens_mask", False):
            if not is_tokenizers_fast:
                raise ValueError(
                    "`return_assistant_tokens_mask` is not possible with slow tokenizers. Make sure you have `tokenizers` installed. "
                    "If the error persists, open an issue to support a Fast tokenizer for your model."
                )
            else:
                kwargs["return_offsets_mapping"] = True  # force offset mapping so we can infer token boundaries

        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__:
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", True)
        mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for conversation in conversations:
                images, videos = [], []
                for message in conversation:
                    visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
                    audio_fnames = [
                        content[key]
                        for content in message["content"]
                        for key in ["audio", "url", "path"]
                        if key in content and content["type"] == "audio"
                    ]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                    # Audio models do not accept nested list of audios (yet!) so we construct a flat input audio list
                    if not mm_load_kwargs["load_audio_from_video"]:
                        for fname in audio_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))
                    else:
                        for fname in video_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))

                # Currently all processors can accept nested list of batches, but not flat list of visuals
                # So we'll make a batched list of images and let the processor handle it
                batch_images.append(images)
                batch_videos.append(videos)

        special_tokens_map = {}
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "special_tokens_map"):
            special_tokens = self.tokenizer.special_tokens_map
            # Filter out tokens that conflict with template kwargs
            special_tokens_map = {
                k: v for k, v in special_tokens.items() if k not in processed_kwargs["template_kwargs"]
            }

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **processed_kwargs["template_kwargs"],  # different flags such as `return_assistant_mask`
            **special_tokens_map,  # tokenizer special tokens are used by some templates
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
            # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
            # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
            # everything internally. The below line is to keep BC for that and be able to work with model that have
            # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
            # without actionable solution for users
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            # Always sample frames by default unless explicitly set to `False` by users. If users do not pass `num_frames`/`fps`
            # sampling should not done for BC.
            if "do_sample_frames" not in kwargs and (
                kwargs.get("fps") is not None or kwargs.get("num_frames") is not None
            ):
                kwargs["do_sample_frames"] = True

            images_exist = any((im is not None) for im_list in batch_images for im in im_list)
            videos_exist = any((vid is not None) for vid_list in batch_videos for vid in vid_list)
            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                audio=batch_audios if batch_audios else None,
                **kwargs,
            )

            if return_dict:
                if processed_kwargs["template_kwargs"].get("return_assistant_tokens_mask", False):
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        offsets = offset_mapping[i]
                        offset_starts = [start for start, end in offsets]
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_pos = bisect.bisect_left(offset_starts, assistant_start_char)
                            end_pos = bisect.bisect_left(offset_starts, assistant_end_char)

                            if not (
                                start_pos >= 0
                                and start_pos < len(offsets)
                                and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                            ):
                                # start_token is out of bounds maybe due to truncation.
                                continue
                            # Ensure end_pos is also within bounds
                            if end_pos > len(input_ids[i]):
                                end_pos = len(input_ids[i])
                            for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
                return out
            else:
                return out["input_ids"]
        return prompt


__all__ = ["Qwen3OmniMoeProcessor"]
