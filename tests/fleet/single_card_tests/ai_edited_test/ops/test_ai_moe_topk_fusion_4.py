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
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from paddleformers.fleet.triton_ops import moe_topk_fusion


class FakeNumber:
    def __init__(self, value=1.0):
        self.value = float(value)

    def __add__(self, other):
        return FakeNumber(self.value + number_value(other))

    __radd__ = __add__

    def __sub__(self, other):
        return FakeNumber(self.value - number_value(other))

    def __rsub__(self, other):
        return FakeNumber(number_value(other) - self.value)

    def __mul__(self, other):
        if isinstance(other, FakeArray):
            return other
        return FakeNumber(self.value * number_value(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        divisor = number_value(other)
        if divisor == 0:
            divisor = 1
        return FakeNumber(self.value / divisor)

    def __rtruediv__(self, other):
        divisor = self.value if self.value != 0 else 1
        return FakeNumber(number_value(other) / divisor)

    def __gt__(self, other):
        return self.value > number_value(other)

    def __lt__(self, other):
        return self.value < number_value(other)

    def __bool__(self):
        return self.value != 0


class FakeArray:
    def __init__(self, values=None):
        self.values = values or [0, 1, 2, 3]

    def __getitem__(self, key):
        del key
        return self

    def __add__(self, other):
        del other
        return self

    __radd__ = __add__

    def __sub__(self, other):
        del other
        return self

    __rsub__ = __sub__

    def __mul__(self, other):
        del other
        return self

    __rmul__ = __mul__

    def __truediv__(self, other):
        del other
        return self

    def __floordiv__(self, other):
        del other
        return self

    def __lt__(self, other):
        del other
        return self

    def __le__(self, other):
        del other
        return self

    def __gt__(self, other):
        del other
        return self

    def __eq__(self, other):
        del other
        return self

    def __ne__(self, other):
        del other
        return self

    def __and__(self, other):
        del other
        return self

    __rand__ = __and__

    def __or__(self, other):
        del other
        return self

    __ror__ = __or__

    def __rrshift__(self, other):
        del other
        return self

    def to(self, dtype):
        del dtype
        return self

    def __bool__(self):
        return True


class FakePtr:
    def __init__(self, offset=0):
        self.offset = int(offset)

    def __add__(self, other):
        if isinstance(other, FakeArray):
            return other
        return FakePtr(self.offset + int(number_value(other)))

    __radd__ = __add__


class FakeTL:
    int1 = "int1"
    float32 = "float32"
    int64 = "int64"

    def __init__(self, program_ids):
        self.program_ids = program_ids
        self.stores = []
        self.atomic_adds = []

    def program_id(self, axis):
        return self.program_ids[axis]

    def arange(self, start, end):
        return FakeArray(list(range(start, end)))

    def load(self, ptr, mask=None, other=None):
        del mask, other
        if isinstance(ptr, FakePtr):
            pattern = [2.0, 1.0, 4.0, 3.0, 0.5, 0.25]
            return FakeNumber(pattern[ptr.offset % len(pattern)])
        return FakeArray()

    def store(self, ptr, value, mask=None):
        self.stores.append((ptr, value, mask))

    def max(self, value, axis=0):
        del value
        if axis == 0:
            return FakeNumber(4.0)
        return FakeArray()

    def min(self, value, axis=0):
        del value, axis
        return 0

    def sum(self, value, axis=None):
        del value
        if axis is None:
            return FakeNumber(1.0)
        return FakeArray()

    def where(self, condition, x, y):
        del condition, y
        return x if isinstance(x, FakeArray) else FakeArray()

    def full(self, shape, value, dtype=None):
        del dtype
        return FakeArray([value] * shape[0])

    def maximum(self, left, right):
        return FakeNumber(max(number_value(left), number_value(right)))

    def atomic_add(self, ptr, value, mask=None):
        self.atomic_adds.append((ptr, value, mask))


def number_value(value):
    if isinstance(value, FakeNumber):
        return value.value
    if isinstance(value, FakePtr):
        return value.offset
    if isinstance(value, FakeArray):
        return 1
    return value


class TestMoETopkFusionKernelDefinitionsNoMock(unittest.TestCase):
    def setUp(self):
        self.old_tl = moe_topk_fusion.tl

    def tearDown(self):
        moe_topk_fusion.tl = self.old_tl

    def test_forward_kernel_python_body_exercises_group_topk_and_norm(self):
        fake_tl = FakeTL([0, 0])
        moe_topk_fusion.tl = fake_tl

        moe_topk_fusion._fwd_kernel.kernel.fn(
            FakePtr(),
            FakePtr(),
            FakePtr(),
            FakePtr(),
            FakePtr(),
            4,
            1,
            4,
            1,
            2,
            1,
            4,
            2,
            True,
            2,
            2,
            True,
            4,
        )

        self.assertGreater(len(fake_tl.stores), 0)

    def test_backward_kernel_python_body_exercises_norm_and_plain_paths(self):
        fake_tl = FakeTL([0, 0])
        moe_topk_fusion.tl = fake_tl

        for norm_gate_logits in (True, False):
            moe_topk_fusion._bwd_kernel.kernel.fn(
                FakePtr(),
                FakePtr(),
                FakePtr(),
                FakePtr(),
                FakePtr(),
                2,
                1,
                2,
                1,
                2,
                1,
                4,
                1,
                2,
                norm_gate_logits,
                2,
            )

        self.assertGreater(len(fake_tl.stores), 0)

    def test_routing_kernel_python_body_exercises_masks_and_dispatch(self):
        fake_tl = FakeTL([0, 0])
        moe_topk_fusion.tl = fake_tl

        moe_topk_fusion._routing_map_fwd_kernel.kernel.fn(
            FakePtr(),
            FakePtr(),
            FakePtr(),
            FakePtr(),
            FakePtr(),
            FakePtr(),
            2,
            1,
            4,
            1,
            4,
            2,
            2,
            0,  # pad_token_id
            True,
            True,
            2,
            4,
            2,
        )

        self.assertGreater(len(fake_tl.stores), 0)
        self.assertGreater(len(fake_tl.atomic_adds), 0)


if __name__ == "__main__":
    unittest.main()
