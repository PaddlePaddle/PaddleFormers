# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from paddleformers.mergekit import MergeConfig, MergeMethod


class TestMergeMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t1 = np.array(
            [
                [-0.834894061088562, 0.7672924399375916, -0.981352686882019, 0.8236614465713501, 0.19363074004650116],
                [
                    0.7413361668586731,
                    -0.44731196761131287,
                    0.9544159173965454,
                    0.07453861087560654,
                    0.5572543144226074,
                ],
                [
                    0.9128026962280273,
                    -0.23344580829143524,
                    0.8464474678039551,
                    0.14241063594818115,
                    -0.8475964069366455,
                ],
                [0.598555326461792, 0.9459332823753357, -0.35118913650512695, 0.5437421798706055, 0.6906668543815613],
            ],
            dtype="float32",
        )
        t2 = np.array(
            [
                [0.49925723671913147, -0.4865064024925232, 0.8579433560371399, -0.546754777431488, 0.6354734897613525],
                [
                    0.23720359802246094,
                    -0.9355064630508423,
                    0.5311998128890991,
                    0.05024348944425583,
                    -0.3130885660648346,
                ],
                [
                    0.9513155221939087,
                    -0.1657073199748993,
                    0.008428480476140976,
                    0.6909753680229187,
                    0.43041032552719116,
                ],
                [
                    0.5276866555213928,
                    -0.5949721932411194,
                    0.11636247485876083,
                    0.6154545545578003,
                    0.09229031205177307,
                ],
            ],
            dtype="float32",
        )
        cls.tensor_list = [t1, t2]

    @classmethod
    def to_paddle_tensor(cls, numpy_tensors):
        return [paddle.to_tensor(tensor, dtype="float32") for tensor in numpy_tensors]

    def test_linear(self):
        merge_config = MergeConfig(
            merge_type="linear",
            weight_list=[2, 8],
            normalize=True,
        )
        merge_method = MergeMethod(merge_config=merge_config)
        merged_tensor = merge_method.merge(self.tensor_list)
        self.assertEqual(merged_tensor.shape, (4, 5))
        expected_result = np.array(
            [
                [
                    0.2324269860982895,
                    -0.23574663698673248,
                    0.490084171295166,
                    -0.27267152070999146,
                    0.5471049547195435,
                ],
                [
                    0.3380300998687744,
                    -0.8378675580024719,
                    0.6158430576324463,
                    0.05510251224040985,
                    -0.13901998102664948,
                ],
                [
                    0.9436129927635193,
                    -0.17925502359867096,
                    0.17603228986263275,
                    0.581262469291687,
                    0.17480896413326263,
                ],
                [
                    0.5418604016304016,
                    -0.2867910861968994,
                    0.022852152585983276,
                    0.6011121273040771,
                    0.2119656205177307,
                ],
            ],
            dtype="float32",
        )
        self.assertTrue(np.array_equal(merged_tensor, expected_result))

    def test_slerp(self):
        merge_config = MergeConfig(
            merge_type="slerp",
            slerp_alpha=0.5,
        )
        merge_method = MergeMethod(merge_config=merge_config)
        merged_tensor = merge_method.merge(self.tensor_list)
        self.assertEqual(merged_tensor.shape, (4, 5))
        expected_result = np.array(
            [
                [
                    -0.241766095161438,
                    0.20225590467453003,
                    -0.08889424800872803,
                    0.19946154952049255,
                    0.5972206592559814,
                ],
                [0.704862117767334, -0.9960722923278809, 1.0701193809509277, 0.08988308906555176, 0.17587755620479584],
                [
                    1.3427623510360718,
                    -0.28751814365386963,
                    0.6157845854759216,
                    0.6003049612045288,
                    -0.30050763487815857,
                ],
                [0.8112550973892212, 0.2528044283390045, -0.1691504418849945, 0.8349930644035339, 0.5639800429344177],
            ],
            dtype="float32",
        )
        self.assertTrue(np.array_equal(merged_tensor, expected_result))
        with self.assertRaises(ValueError):
            merged_tensor = merge_method.merge(self.tensor_list + self.tensor_list)

    def test_ties(self):
        merge_config = MergeConfig(
            merge_type="ties",
            weight_list=[2, 8],
            normalize=True,
        )
        merge_method = MergeMethod(merge_config=merge_config)
        merged_tensor = merge_method.merge(self.tensor_list)
        self.assertEqual(merged_tensor.shape, (4, 5))
        expected_result = np.array(
            [
                [0.49925723671913147, -0.4865064024925232, 0.8579433560371399, -0.546754777431488, 0.5471049547195435],
                [
                    0.3380300998687744,
                    -0.8378675580024719,
                    0.6158429980278015,
                    0.05510251596570015,
                    -0.3130885660648346,
                ],
                [
                    0.9436129331588745,
                    -0.17925502359867096,
                    0.17603227496147156,
                    0.5812624096870422,
                    0.43041032552719116,
                ],
                [
                    0.5418604016304016,
                    -0.5949721932411194,
                    0.11636247485876083,
                    0.6011120676994324,
                    0.21196560561656952,
                ],
            ],
            dtype="float32",
        )
        self.assertTrue(np.array_equal(merged_tensor, expected_result))

    def test_linear_paddle(self):
        paddle_tensor_list = self.to_paddle_tensor(self.tensor_list)
        merge_config = MergeConfig(
            merge_type="linear",
            weight_list=[2, 8],
            normalize=True,
            tensor_type="pd",
        )
        merge_method = MergeMethod(merge_config=merge_config)
        merged_tensor = merge_method.merge(paddle_tensor_list)
        self.assertEqual(merged_tensor.shape, [4, 5])
        expected_result = paddle.to_tensor(
            [
                [
                    0.2324269860982895,
                    -0.23574663698673248,
                    0.490084171295166,
                    -0.27267152070999146,
                    0.5471049547195435,
                ],
                [
                    0.3380300998687744,
                    -0.8378675580024719,
                    0.6158430576324463,
                    0.05510251224040985,
                    -0.13901998102664948,
                ],
                [
                    0.9436129927635193,
                    -0.17925502359867096,
                    0.17603228986263275,
                    0.581262469291687,
                    0.17480896413326263,
                ],
                [
                    0.5418604016304016,
                    -0.2867910861968994,
                    0.022852152585983276,
                    0.6011121273040771,
                    0.2119656205177307,
                ],
            ],
            dtype="float32",
        )
        self.assertTrue(
            paddle.allclose(merged_tensor, expected_result, atol=1e-6),
            "Paddle linear merge result does not match expected result.",
        )

    def test_slerp_paddle(self):
        paddle_tensor_list = self.to_paddle_tensor(self.tensor_list)
        merge_config = MergeConfig(
            merge_type="slerp",
            slerp_alpha=0.5,
            tensor_type="pd",
        )
        merge_method = MergeMethod(merge_config=merge_config)
        merged_tensor = merge_method.merge(paddle_tensor_list)
        self.assertEqual(merged_tensor.shape, [4, 5])
        expected_result = paddle.to_tensor(
            [
                [
                    -0.241766095161438,
                    0.20225590467453003,
                    -0.08889424800872803,
                    0.19946154952049255,
                    0.5972206592559814,
                ],
                [0.704862117767334, -0.9960722923278809, 1.0701193809509277, 0.08988308906555176, 0.17587755620479584],
                [
                    1.3427623510360718,
                    -0.28751814365386963,
                    0.6157845854759216,
                    0.6003049612045288,
                    -0.30050763487815857,
                ],
                [0.8112550973892212, 0.2528044283390045, -0.1691504418849945, 0.8349930644035339, 0.5639800429344177],
            ],
            dtype="float32",
        )
        self.assertTrue(
            paddle.allclose(merged_tensor, expected_result, atol=1e-6),
            "Paddle slerp merge result does not match expected result.",
        )
        with self.assertRaises(ValueError):
            merge_method.merge(paddle_tensor_list + paddle_tensor_list)

    def test_ties_paddle(self):
        paddle_tensor_list = self.to_paddle_tensor(self.tensor_list)
        merge_config = MergeConfig(
            merge_type="ties",
            weight_list=[2, 8],
            normalize=True,
            tensor_type="pd",
        )
        merge_method = MergeMethod(merge_config=merge_config)
        merged_tensor = merge_method.merge(paddle_tensor_list)
        self.assertEqual(merged_tensor.shape, [4, 5])
        expected_result = paddle.to_tensor(
            [
                [0.49925723671913147, -0.4865064024925232, 0.8579433560371399, -0.546754777431488, 0.5471049547195435],
                [
                    0.3380300998687744,
                    -0.8378675580024719,
                    0.6158429980278015,
                    0.05510251596570015,
                    -0.3130885660648346,
                ],
                [
                    0.9436129331588745,
                    -0.17925502359867096,
                    0.17603227496147156,
                    0.5812624096870422,
                    0.43041032552719116,
                ],
                [
                    0.5418604016304016,
                    -0.5949721932411194,
                    0.11636247485876083,
                    0.6011120676994324,
                    0.21196560561656952,
                ],
            ],
            dtype="float32",
        )
        self.assertTrue(
            paddle.allclose(merged_tensor, expected_result, atol=1e-6),
            "Paddle ties merge result does not match expected result.",
        )
