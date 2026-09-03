# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest


class TestTypeValidators(unittest.TestCase):
    """Tests for utils/type_validators.py"""

    def test_positive_any_number_none(self):
        from paddleformers.utils.type_validators import positive_any_number

        positive_any_number(None)  # Should not raise

    def test_positive_any_number_valid(self):
        from paddleformers.utils.type_validators import positive_any_number

        positive_any_number(5)
        positive_any_number(3.14)

    def test_positive_any_number_negative(self):
        from paddleformers.utils.type_validators import positive_any_number

        with self.assertRaises(ValueError):
            positive_any_number(-1)

    def test_positive_any_number_invalid_type(self):
        from paddleformers.utils.type_validators import positive_any_number

        with self.assertRaises(ValueError):
            positive_any_number("abc")

    def test_positive_int_none(self):
        from paddleformers.utils.type_validators import positive_int

        positive_int(None)  # Should not raise

    def test_positive_int_valid(self):
        from paddleformers.utils.type_validators import positive_int

        positive_int(5)

    def test_positive_int_float_raises(self):
        from paddleformers.utils.type_validators import positive_int

        with self.assertRaises(ValueError):
            positive_int(3.14)

    def test_padding_validator_none(self):
        from paddleformers.utils.type_validators import padding_validator

        padding_validator(None)  # Should not raise

    def test_padding_validator_bool(self):
        from paddleformers.utils.type_validators import padding_validator

        padding_validator(True)
        padding_validator(False)

    def test_padding_validator_valid_str(self):
        from paddleformers.utils.type_validators import padding_validator

        padding_validator("longest")
        padding_validator("max_length")
        padding_validator("do_not_pad")

    def test_padding_validator_invalid_str(self):
        from paddleformers.utils.type_validators import padding_validator

        with self.assertRaises(ValueError):
            padding_validator("invalid")

    def test_padding_validator_invalid_type(self):
        from paddleformers.utils.type_validators import padding_validator

        with self.assertRaises(ValueError):
            padding_validator(42)

    def test_truncation_validator_none(self):
        from paddleformers.utils.type_validators import truncation_validator

        truncation_validator(None)  # Should not raise

    def test_truncation_validator_valid_str(self):
        from paddleformers.utils.type_validators import truncation_validator

        truncation_validator("only_first")
        truncation_validator("only_second")
        truncation_validator("longest_first")
        truncation_validator("do_not_truncate")

    def test_truncation_validator_invalid_str(self):
        from paddleformers.utils.type_validators import truncation_validator

        with self.assertRaises(ValueError):
            truncation_validator("invalid")

    def test_image_size_validator_none(self):
        from paddleformers.utils.type_validators import image_size_validator

        image_size_validator(None)  # Should not raise

    def test_image_size_validator_int(self):
        from paddleformers.utils.type_validators import image_size_validator

        image_size_validator(224)

    def test_image_size_validator_valid_dict(self):
        from paddleformers.utils.type_validators import image_size_validator

        image_size_validator({"height": 224, "width": 224})

    def test_image_size_validator_invalid_dict(self):
        from paddleformers.utils.type_validators import image_size_validator

        with self.assertRaises(ValueError):
            image_size_validator({"invalid_key": 100})

    def test_device_validator_none(self):
        from paddleformers.utils.type_validators import device_validator

        device_validator(None)  # Should not raise

    def test_device_validator_str(self):
        from paddleformers.utils.type_validators import device_validator

        device_validator("cpu")
        device_validator("gpu")

    def test_device_validator_int(self):
        from paddleformers.utils.type_validators import device_validator

        device_validator(0)

    def test_device_validator_negative_int(self):
        from paddleformers.utils.type_validators import device_validator

        with self.assertRaises(ValueError):
            device_validator(-1)

    def test_device_validator_invalid_str(self):
        from paddleformers.utils.type_validators import device_validator

        with self.assertRaises(ValueError):
            device_validator("tpu")

    def test_tensor_type_validator_none(self):
        from paddleformers.utils.type_validators import tensor_type_validator

        tensor_type_validator(None)  # Should not raise

    def test_tensor_type_validator_valid(self):
        from paddleformers.utils.type_validators import tensor_type_validator

        tensor_type_validator("pd")
        tensor_type_validator("np")

    def test_tensor_type_validator_invalid(self):
        from paddleformers.utils.type_validators import tensor_type_validator

        with self.assertRaises(ValueError):
            tensor_type_validator("tf")

    def test_resampling_validator_none(self):
        from paddleformers.utils.type_validators import resampling_validator

        resampling_validator(None)  # Should not raise

    def test_resampling_validator_valid_int(self):
        from paddleformers.utils.type_validators import resampling_validator

        resampling_validator(0)
        resampling_validator(1)
        resampling_validator(5)

    def test_resampling_validator_invalid_int(self):
        from paddleformers.utils.type_validators import resampling_validator

        with self.assertRaises(ValueError):
            resampling_validator(10)
