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

"""Scope checks for the flex-checkpoint load instrumentation.

The timing helpers and the local-resumability diagnosis are closures defined
inside ``Trainer._load_flex_checkpoint``. Nothing guarantees at import time that
a helper and its call sites stay in the same function body: moving one of them
into a neighbouring method still compiles and only fails at run time with
``NameError``, which needs a live distributed job to surface.

These tests parse trainer.py and assert the invariant statically, so no GPU and
no distributed init are required.
"""

import ast
import inspect
import os
import unittest

import paddleformers.trainer.trainer as trainer_module
import paddleformers.trainer.trainer_utils as trainer_utils_module

# closures and locals that the instrumentation introduces; every one of them must
# be created inside _load_flex_checkpoint and used nowhere else
INSTRUMENTATION_NAMES = {
    "flex_phase",
    "flex_phase_summary",
    "flex_fence",
    "flex_diagnose_component",
    "_flex_phase_stats",
    "_flex_nested_stats",
    "_flex_rank",
    "_flex_on_gpu",
    "_flex_storage_meta",
    "_flex_component_paths",
    "_t_recover",
    "_t_cast",
    "_cast_on_device",
    "_t_assign",
    "_n_written",
}

# what each timed phase is expected to wrap, as a substring of the first
# statement inside the with-block
EXPECTED_PHASE_TARGETS = {
    "metadata_load": "for metadata_file in metadata_paths:",
    "init_optimizer": "init_optimizer(self.optimizer",
    "optimizer_sharded_state_dict": "optimizer_sharded_state_dict = self.optimizer.sharded_state_dict(",
    "master_weight_load": "dist.load_state_dict(",
    "opt_state_load": "dist.load_state_dict(",
    "model_state_load": "dist.load_state_dict(",
    "optimizer_state_dict": "opt_state_dict = self.optimizer.state_dict()",
    "group_getter_and_split_opt_state": "group_getter = GroupGetter(self.model)",
}


def _load(module):
    path = inspect.getsourcefile(module)
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    return path, source, ast.parse(source)


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


class TestFlexLoadInstrumentationScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path, cls.source, cls.tree = _load(trainer_module)
        cls.lines = cls.source.split("\n")
        cls.func = _find_function(cls.tree, "_load_flex_checkpoint")
        assert cls.func is not None, f"_load_flex_checkpoint not found in {cls.path}"

    def _bound_names(self):
        bound = set()
        for node in ast.walk(self.func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
        return bound

    def test_every_helper_is_defined_inside_the_method(self):
        """A helper that drifts into another method still compiles; catch it here."""
        bound = self._bound_names()
        used = {n.id for n in ast.walk(self.func) if isinstance(n, ast.Name)}
        expected = INSTRUMENTATION_NAMES & used
        self.assertTrue(expected, "no instrumentation names found - did the naming change?")
        missing = sorted(expected - bound)
        self.assertEqual(missing, [], f"used but never bound inside _load_flex_checkpoint: {missing}")

    def test_no_instrumentation_name_leaks_outside_the_method(self):
        span = range(self.func.lineno, (self.func.end_lineno or self.func.lineno) + 1)
        outside = [
            (n.id, n.lineno)
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Name) and n.id in INSTRUMENTATION_NAMES and n.lineno not in span
        ]
        self.assertEqual(outside, [], f"instrumentation used outside _load_flex_checkpoint: {outside}")

    def test_each_phase_wraps_the_expected_statement(self):
        """Guards against a with-block landing on the wrong call of the same shape.

        dist.load_state_dict( appears several times in this file; a timer that
        wraps the EMA load instead of the model_state load would still be inside
        the method and would still run.
        """
        seen = {}
        for node in ast.walk(self.func):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                call = item.context_expr
                if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "flex_phase"):
                    continue
                if not (call.args and isinstance(call.args[0], ast.Constant)):
                    continue  # f-string name (recover_pass{N}), checked separately
                name = call.args[0].value
                first = self.lines[node.body[0].lineno - 1].strip()
                seen[name] = first
        for name, expected in EXPECTED_PHASE_TARGETS.items():
            self.assertIn(name, seen, f"phase {name!r} is gone")
            self.assertIn(expected, seen[name], f"phase {name!r} wraps {seen[name]!r}, expected {expected!r}")

    def test_diagnosis_precedes_the_matching_load(self):
        pairs = []
        for node in ast.walk(self.func):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "flex_diagnose_component"
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
            ):
                pairs.append((node.value.args[0].value, node.lineno))
        self.assertEqual(
            sorted(name for name, _ in pairs),
            ["master_weight", "model_state", "opt_state"],
            f"unexpected diagnosis call sites: {pairs}",
        )
        for name, lineno in pairs:
            following = " ".join(self.lines[lineno : lineno + 3])
            self.assertIn(f'flex_phase("{name}_load"', following, f"{name} diagnosis is not next to its load")

    def test_diagnosis_is_guarded_and_collective_free(self):
        """It touches paddle.distributed internals and must not add collectives."""
        diag = _find_function(self.func, "flex_diagnose_component")
        self.assertIsNotNone(diag, "flex_diagnose_component not found")
        self.assertTrue(
            any(isinstance(n, ast.Try) for n in ast.walk(diag)),
            "the paddle.distributed imports must stay inside try/except",
        )
        segment = "\n".join(self.lines[diag.lineno - 1 : (diag.end_lineno or diag.lineno)])
        for collective in ("all_reduce", "all_gather", "barrier", "broadcast"):
            self.assertNotIn(
                collective,
                segment,
                f"{collective} in the diagnosis would deadlock when the import guard "
                f"succeeds on some ranks only",
            )


class TestRestoreMasterWeightsInstrumentation(unittest.TestCase):
    def test_six_step_timing_is_present_and_fenced(self):
        path, source, tree = _load(trainer_utils_module)
        func = _find_function(tree, "_restore_master_weights_single")
        self.assertIsNotNone(func, f"_restore_master_weights_single not found in {path}")
        bound = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        for name in ("t0", "t_add", "t_pack", "t_merge", "t_restore", "t_unpack", "t_gather"):
            self.assertIn(name, bound, f"timing point {name} is missing")
        fence = _find_function(func, "_fence")
        self.assertIsNotNone(fence, "_fence helper is missing")
        segment = "\n".join(source.split("\n")[func.lineno - 1 : (func.end_lineno or func.lineno)])
        self.assertGreaterEqual(
            segment.count("_fence()"),
            7,
            "each of the six steps plus the entry needs a fence, otherwise a step's "
            "device work is charged to the next one",
        )
        self.assertIn("[mw-restore]", segment, "the summary log line is missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
