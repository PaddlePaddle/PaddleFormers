# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from transformers import SiglipImageProcessor

from ..image_processing_utils import PaddleImageProcessingMixin


class PaliGemmaImageProcessor(PaddleImageProcessingMixin, SiglipImageProcessor):
    """SigLIP preprocessing with PaliGemma image-token metadata."""

    def __init__(self, image_seq_length=1024, **kwargs):
        self.image_seq_length = image_seq_length
        super().__init__(**kwargs)


__all__ = ["PaliGemmaImageProcessor"]
