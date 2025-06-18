# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import warnings

import numpy as np
import paddle

__all__ = ["CommonTest", "CpuCommonTest"]


# Assume all elements has same data type
def get_container_type(container):
    container_t = type(container)
    if container_t in [list, tuple]:
        if len(container) == 0:
            return container_t
        return get_container_type(container[0])
    return container_t


class CommonTest(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super(CommonTest, self).__init__(methodName=methodName)
        self.config = {}
        self.places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.places = ["gpu"]

    @classmethod
    def setUpClass(cls):
        """
        Set the decorators for all test function
        """
        for key, value in cls.__dict__.items():
            if key.startswith("test"):
                decorator_func_list = ["_test_places", "_catch_warnings"]
                for decorator_func in decorator_func_list:
                    decorator_func = getattr(CommonTest, decorator_func)
                    value = decorator_func(value)
                setattr(cls, key, value)

    def _catch_warnings(func):
        """
        Catch the warnings and treat them as errors for each test.
        """

        def wrapper(self, *args, **kwargs):
            with warnings.catch_warnings(record=True) as w:
                warnings.resetwarnings()
                # ignore specified warnings
                warning_white_list = [UserWarning]
                for warning in warning_white_list:
                    warnings.simplefilter("ignore", warning)
                func(self, *args, **kwargs)
                msg = None if len(w) == 0 else w[0].message
                self.assertFalse(len(w) > 0, msg)

        return wrapper

    def _test_places(func):
        """
        Setting the running place for each test.
        """

        def wrapper(self, *args, **kwargs):
            places = self.places
            for place in places:
                paddle.set_device(place)
                func(self, *args, **kwargs)

        return wrapper

    def _check_output_impl(self, result, expected_result, rtol, atol, equal=True):
        assertForNormalType = self.assertNotEqual
        assertForFloat = self.assertFalse
        if equal:
            assertForNormalType = self.assertEqual
            assertForFloat = self.assertTrue

        result_t = type(result)
        error_msg = "Output has diff at place:{}. \nExpect: {} \nBut Got: {} in class {}"
        if result_t in [list, tuple]:
            result_t = get_container_type(result)
        if result_t in [str, int, bool, set, bool, np.int32, np.int64]:
            assertForNormalType(
                result,
                expected_result,
                msg=error_msg.format(paddle.get_device(), expected_result, result, self.__class__.__name__),
            )
        elif result_t in [float, np.ndarray, np.float32, np.float64]:
            assertForFloat(
                np.allclose(result, expected_result, rtol=rtol, atol=atol),
                msg=error_msg.format(paddle.get_device(), expected_result, result, self.__class__.__name__),
            )
            if result_t == np.ndarray:
                assertForNormalType(
                    result.shape,
                    expected_result.shape,
                    msg=error_msg.format(
                        paddle.get_device(), expected_result.shape, result.shape, self.__class__.__name__
                    ),
                )
        else:
            raise ValueError(
                "result type must be str, int, bool, set, np.bool, np.int32, "
                "np.int64, np.str, float, np.ndarray, np.float32, np.float64"
            )

    def check_output_equal(self, result, expected_result, rtol=1.0e-5, atol=1.0e-8):
        """
            Check whether result and expected result are equal, including shape.
        Args:
            result: str, int, bool, set, np.ndarray.
                The result needs to be checked.
            expected_result: str, int, bool, set, np.ndarray. The type has to be same as result's.
                Use the expected result to check result.
            rtol: float
                relative tolerance, default 1.e-5.
            atol: float
                absolute tolerance, default 1.e-8
        """
        self._check_output_impl(result, expected_result, rtol, atol)

    def check_output_not_equal(self, result, expected_result, rtol=1.0e-5, atol=1.0e-8):
        """
            Check whether result and expected result are not equal, including shape.
        Args:
            result: str, int, bool, set, np.ndarray.
                The result needs to be checked.
            expected_result: str, int, bool, set, np.ndarray. The type has to be same as result's.
                Use the expected result to check result.
            rtol: float
                relative tolerance, default 1.e-5.
            atol: float
                absolute tolerance, default 1.e-8
        """
        self._check_output_impl(result, expected_result, rtol, atol, equal=False)


class CpuCommonTest(CommonTest):
    def __init__(self, methodName="runTest"):
        super(CpuCommonTest, self).__init__(methodName=methodName)
        self.places = ["cpu"]
