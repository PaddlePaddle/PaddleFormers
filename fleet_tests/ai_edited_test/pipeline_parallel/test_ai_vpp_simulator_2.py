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
import builtins
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


class Axis:
    def __init__(self):
        self.patches = []
        self.texts = []
        self.lines = []

    def get_window_extent(self):
        return object()

    def add_patch(self, patch):
        self.patches.append(patch)

    def text(self, *args, **kwargs):
        self.texts.append((args, kwargs))

    def plot(self, *args, **kwargs):
        self.lines.append((args, kwargs))

    def set_xlim(self, *args, **kwargs):
        pass

    def set_ylim(self, *args, **kwargs):
        pass

    def set_yticks(self, *args, **kwargs):
        pass

    def set_yticklabels(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_title(self, *args, **kwargs):
        pass

    def grid(self, *args, **kwargs):
        pass

    def axis(self, *args, **kwargs):
        pass


class Figure:
    dpi = 100

    def get_size_inches(self):
        return (12, 6)


class ColorMap:
    def __call__(self, value):
        return (value, 0.0, 1.0 - value, 1.0)


class ColorManager:
    def get_cmap(self, name):
        del name
        return ColorMap()


class Pyplot:
    def __init__(self):
        self.cm = ColorManager()

    def subplots(self, *args, **kwargs):
        del args, kwargs
        return Figure(), Axis()

    def tight_layout(self):
        pass

    def savefig(self, path):
        with open(path, "wb") as output:
            output.write(b"plot")

    def Circle(self, *args, **kwargs):
        return Circle(*args, **kwargs)


class Patches:
    def Rectangle(self, *args, **kwargs):
        return Rectangle(*args, **kwargs)


class Rectangle:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Circle:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ScaledRange(list):
    def __mul__(self, value):
        return [item * value for item in self]

    def __rmul__(self, value):
        return self.__mul__(value)


def ensure_vpp_simulator_importable():
    try:
        from paddleformers.fleet.pipeline_parallel import vpp_simulator
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        patches = types.ModuleType("matplotlib.patches")
        pyplot.cm = ColorManager()
        pyplot.subplots = lambda *args, **kwargs: (Figure(), Axis())
        pyplot.tight_layout = lambda: None
        pyplot.savefig = lambda path: open(path, "wb").write(b"plot")
        pyplot.Circle = Circle
        patches.Rectangle = Rectangle
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot
        sys.modules["matplotlib.patches"] = patches
        from paddleformers.fleet.pipeline_parallel import vpp_simulator
    return vpp_simulator


class TestVPPSimulatorEdgeBranches(unittest.TestCase):
    def test_compute_bubble_rate_empty_schedule(self):
        VPPSimulator = ensure_vpp_simulator_importable().VPPSimulator

        simulator = VPPSimulator(pp_degree=2, vpp_degree=1, num_acc_steps=2)
        simulator._is_scheduled = True
        simulator.schedule_table = []

        self.assertEqual(simulator.compute_bubble_rate(), 0.0)

    def test_get_preorder_chunk_unknown_chunk_type(self):
        VPPSimulator = ensure_vpp_simulator_importable().VPPSimulator

        class UnknownChunk:
            chunk_type = "unknown"
            stage_id = 0
            layer_id = 0
            acc_step = 0

        simulator = VPPSimulator(pp_degree=2, vpp_degree=1, num_acc_steps=2)

        with self.assertRaises(NotImplementedError):
            simulator._get_preorder_chunk(UnknownChunk())

    def test_global_recorder_can_be_reset_to_none(self):
        vpp_simulator = ensure_vpp_simulator_importable()
        PPChunkRecorder = vpp_simulator.PPChunkRecorder
        get_global_pp_recorder = vpp_simulator.get_global_pp_recorder
        set_global_pp_chunk_recorder = (
            vpp_simulator.set_global_pp_chunk_recorder
        )

        recorder = PPChunkRecorder(2, 1, 2, 4, 0, 0)
        set_global_pp_chunk_recorder(recorder)
        self.assertIs(get_global_pp_recorder(), recorder)

        set_global_pp_chunk_recorder(None)
        self.assertIsNone(get_global_pp_recorder())


class TestVPPSimulatorDrawWithMatplotlibStubs(unittest.TestCase):
    def setUp(self):
        self.vpp_simulator = ensure_vpp_simulator_importable()
        self.original_plt = self.vpp_simulator.plt
        self.original_patches = self.vpp_simulator.patches
        self.original_range = getattr(self.vpp_simulator, "range", None)
        self.vpp_simulator.plt = Pyplot()
        self.vpp_simulator.patches = Patches()
        self.vpp_simulator.range = lambda *args: ScaledRange(
            builtins.range(*args)
        )

    def tearDown(self):
        self.vpp_simulator.plt = self.original_plt
        self.vpp_simulator.patches = self.original_patches
        if self.original_range is None:
            delattr(self.vpp_simulator, "range")
        else:
            self.vpp_simulator.range = self.original_range

    def test_draw_chunks_writes_schedule_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.vpp_simulator.VPPSimulator(
                    pp_degree=2, vpp_degree=2, num_acc_steps=4
                ).draw_chunks()
                self.assertTrue(os.path.exists("pipeline_schedule.png"))
            finally:
                os.chdir(old_cwd)

    def test_draw_balls_writes_balls_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.vpp_simulator.VPPSimulator(
                    pp_degree=2, vpp_degree=2, num_acc_steps=4
                ).draw_balls()
                self.assertTrue(os.path.exists("balls.png"))
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
