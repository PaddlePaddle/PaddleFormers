# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Phi-4 Multimodal audio."""

import numpy as np
import paddle
from transformers.audio_utils import mel_filter_bank

from ...utils.log import logger
from ..audio_processing_utils import SequenceFeatureExtractor
from ..feature_extraction_utils import BatchFeature


class Phi4MultimodalFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["audio_input_features", "audio_embed_sizes", "audio_attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        padding_value=0.0,
        audio_compression_rate=8,
        audio_downsample_rate=1,
        audio_feat_stride=1,
        mel_min_frequency=0,
        mel_max_frequency=7690,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.padding_value = padding_value
        self.audio_compression_rate = audio_compression_rate
        self.audio_downsample_rate = audio_downsample_rate
        self.audio_feat_stride = audio_feat_stride
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=self.feature_size,
            min_frequency=mel_min_frequency,
            max_frequency=mel_max_frequency,
            sampling_rate=self.sampling_rate,
            triangularize_in_mel_space=True,
            mel_scale="kaldi",
        ).astype(np.float32)

    def __call__(
        self,
        raw_speech,
        sampling_rate=None,
        pad_to_multiple_of=None,
        padding="longest",
        max_length=None,
        truncation=False,
        return_tensors=None,
        return_attention_mask=True,
        device="cpu",
        **kwargs,
    ):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor was trained using a sampling rate of "
                f"{self.sampling_rate}. Please provide audio sampled at {self.sampling_rate}, not {sampling_rate}."
            )
        if sampling_rate is None:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`."
            )

        speech_list = self._as_mono_float_list(raw_speech)
        audio_lengths = np.asarray([speech.shape[0] for speech in speech_list], dtype=np.int64)

        if truncation and max_length is not None:
            speech_list = [speech[:max_length] for speech in speech_list]
            audio_lengths = np.minimum(audio_lengths, max_length)

        padded_length = max(int(length) for length in audio_lengths)
        padded_length = max(padded_length, self.win_length)
        if padding not in (True, "longest", "max_length"):
            padded_length = max(int(audio_lengths[0]), self.win_length)
        if max_length is not None and padding == "max_length":
            padded_length = max(max_length, self.win_length)
        if pad_to_multiple_of is not None and padded_length % pad_to_multiple_of != 0:
            padded_length = ((padded_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        waveform = np.full((len(speech_list), padded_length), self.padding_value, dtype=np.float32)
        for idx, speech in enumerate(speech_list):
            length = min(speech.shape[0], padded_length)
            waveform[idx, :length] = speech[:length]

        input_features = self._np_extract_fbank_features(waveform, audio_lengths)
        feature_lengths = (audio_lengths - self.win_length) // self.hop_length + 1
        feature_lengths = np.maximum(feature_lengths, 1) * self.audio_feat_stride
        audio_embed_sizes = self._compute_audio_embed_size(feature_lengths)

        feature_attention_mask = None
        if return_attention_mask and len(feature_lengths) > 1:
            max_feature_length = int(feature_lengths.max())
            feature_attention_mask = np.arange(max_feature_length)[None, :] < feature_lengths[:, None]

        data = {
            "audio_input_features": input_features,
            "audio_embed_sizes": audio_embed_sizes.astype(np.int64),
        }
        if feature_attention_mask is not None:
            data["audio_attention_mask"] = feature_attention_mask
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _np_extract_fbank_features(self, waveform, audio_lengths):
        fft_window = np.hamming(self.win_length).astype(np.float64)
        batch_features = []
        max_frames = 0

        for speech, length in zip(waveform, audio_lengths):
            speech = speech[: max(int(length), self.win_length)]
            if speech.shape[0] < self.win_length:
                speech = np.pad(speech, (0, self.win_length - speech.shape[0]), constant_values=self.padding_value)
            num_frames = (speech.shape[0] - self.win_length) // self.hop_length + 1
            frames = np.stack(
                [speech[i * self.hop_length : i * self.hop_length + self.win_length] for i in range(num_frames)],
                axis=0,
            )
            frames_prev = np.roll(frames, 1, axis=-1)
            frames_prev[:, 0] = frames_prev[:, 1]
            frames = (frames - self.preemphasis * frames_prev) * 32768
            spectrum = np.fft.rfft(fft_window * frames, n=self.n_fft, axis=1).astype(np.complex64)
            spec_power = np.abs(spectrum) ** 2
            log_spec = np.log(np.clip(spec_power @ self.mel_filters, a_min=1.0, a_max=None)).astype(np.float32)
            batch_features.append(log_spec)
            max_frames = max(max_frames, log_spec.shape[0])

        padded = np.full((len(batch_features), max_frames, self.feature_size), self.padding_value, dtype=np.float32)
        for idx, features in enumerate(batch_features):
            padded[idx, : features.shape[0]] = features
        return padded

    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.audio_compression_rate
        remainder = audio_frames % self.audio_compression_rate
        result = integer + (remainder > 0).astype(integer.dtype)
        integer = result // self.audio_downsample_rate
        remainder = result % self.audio_downsample_rate
        return integer + (remainder > 0).astype(integer.dtype)

    @staticmethod
    def _as_mono_float_list(raw_speech):
        if isinstance(raw_speech, paddle.Tensor):
            raw_speech = raw_speech.detach().cpu().numpy()
        if isinstance(raw_speech, np.ndarray):
            if raw_speech.ndim == 1:
                raw_speech = [raw_speech]
            elif raw_speech.ndim == 2:
                raw_speech = list(raw_speech)
            else:
                raw_speech = [raw_speech.mean(axis=-1)]
        elif isinstance(raw_speech, (list, tuple)):
            raw_speech = [
                speech.detach().cpu().numpy() if isinstance(speech, paddle.Tensor) else speech for speech in raw_speech
            ]
        else:
            raise ValueError(f"Unsupported audio input type: {type(raw_speech)}")

        result = []
        for speech in raw_speech:
            speech = np.asarray(speech)
            if speech.ndim > 1:
                speech = speech.mean(axis=-1)
            result.append(speech.astype(np.float32))
        return result


__all__ = ["Phi4MultimodalFeatureExtractor"]
