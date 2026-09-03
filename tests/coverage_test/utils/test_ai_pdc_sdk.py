# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import patch


class TestPdcSdk(unittest.TestCase):
    """Tests for utils/pdc_sdk.py"""

    def test_pdc_flash_device_available_true(self):
        from paddleformers.utils.pdc_sdk import pdc_flash_device_available

        with patch("paddleformers.utils.pdc_sdk.os.path.exists", return_value=True):
            self.assertTrue(pdc_flash_device_available())

    def test_pdc_flash_device_available_false(self):
        from paddleformers.utils.pdc_sdk import pdc_flash_device_available

        with patch("paddleformers.utils.pdc_sdk.os.path.exists", return_value=False):
            self.assertFalse(pdc_flash_device_available())

    def test_PDCErrorCode_values(self):
        from paddleformers.utils.pdc_sdk import PDCErrorCode

        self.assertEqual(PDCErrorCode.Success.value, 0)
        self.assertEqual(PDCErrorCode.RemotePathNotExist.value, 1404)
        self.assertEqual(PDCErrorCode.LocalPathExist.value, 1405)
        self.assertEqual(PDCErrorCode.DownloadFail.value, 1406)
        self.assertEqual(PDCErrorCode.AgentConfigInvalid.value, 1407)
        self.assertEqual(PDCErrorCode.AFSToolsNotExist.value, 1408)
        self.assertEqual(PDCErrorCode.TrainConfigNotExist.value, 1409)
        self.assertEqual(PDCErrorCode.LocalPathNotExist.value, 1410)
        self.assertEqual(PDCErrorCode.CommandFail.value, 1501)
        self.assertEqual(PDCErrorCode.CalculateHashFail.value, 1502)
        self.assertEqual(PDCErrorCode.InvalidArgument.value, 1503)
        self.assertEqual(PDCErrorCode.CommandTimeout.value, 1504)
        self.assertEqual(PDCErrorCode.CheckSumCommandFail.value, 1505)
        self.assertEqual(PDCErrorCode.CopyTreeFailed.value, 1506)
        self.assertEqual(PDCErrorCode.UnknownError.value, 1999)

    def test_PDCTools_init(self):
        from paddleformers.utils.pdc_sdk import PDCTools

        tools = PDCTools()
        self.assertIsNotNone(tools._pdc_agent_bin)
        self.assertIsNotNone(tools._hash_sum_bin)
        self.assertIsNotNone(tools._train_config)
        self.assertIsNotNone(tools._tar_bin)

    def test_PDCErrorCode_enum(self):
        from paddleformers.utils.pdc_sdk import PDCErrorCode

        # Verify all error codes are integers
        for member in PDCErrorCode:
            self.assertIsInstance(member.value, int)

    def test_Flash_device_constant(self):
        from paddleformers.utils.pdc_sdk import FLASH_DEVICE

        self.assertIsInstance(FLASH_DEVICE, str)
        self.assertTrue(len(FLASH_DEVICE) > 0)

    def test_PDCErrorCode_is_enum(self):
        from enum import Enum

        from paddleformers.utils.pdc_sdk import PDCErrorCode

        self.assertTrue(issubclass(PDCErrorCode, Enum))
