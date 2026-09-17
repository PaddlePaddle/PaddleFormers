# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from .image_processor import Llama4ImageProcessor


class Llama4ImageProcessorFast(Llama4ImageProcessor):
    """Compatibility alias for upstream checkpoints that name the fast processor."""


__all__ = ["Llama4ImageProcessorFast"]
