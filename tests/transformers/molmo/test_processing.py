# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import unittest
from unittest.mock import patch

from paddleformers.transformers.molmo.processing import (
    MolmoImageProcessor,
    MolmoProcessor,
)


class MolmoFallbackTest(unittest.TestCase):
    def test_failed_processor_load_logs_and_returns_default(self):
        tokenizer = object()

        class ProcessorProbe(MolmoProcessor):
            def __init__(self, image_processor=None, tokenizer=None, **kwargs):
                self.image_processor = image_processor
                self.tokenizer = tokenizer

        with (
            patch("paddleformers.transformers.molmo.processing.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch.object(MolmoImageProcessor, "from_pretrained", side_effect=ValueError("invalid config")),
        ):
            processor = ProcessorProbe.from_pretrained("local-model")
        self.assertIs(processor.tokenizer, tokenizer)
        self.assertIsInstance(processor.image_processor, MolmoImageProcessor)
