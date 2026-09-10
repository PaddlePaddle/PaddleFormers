# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import transformers as hf

from ..image_processing_utils import warp_base_image_processor

Florence2ImageProcessor = warp_base_image_processor(hf.CLIPImageProcessor)

__all__ = ["Florence2ImageProcessor"]
