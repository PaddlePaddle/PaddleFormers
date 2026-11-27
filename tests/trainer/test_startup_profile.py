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

"""Unit tests for the opt-in startup span hooks.

The point of ``startup_profile`` is that it changes nothing until somebody opts in,
and that opting in cannot break the instrumented code. So what is pinned down here:

  1. With no provider registered, ``span`` is a no-op context manager that yields the
     same shared object every time -- no allocation, and no output.
  2. The no-op span still tolerates ``note(...)``, so instrumented code can annotate
     spans unconditionally.
  3. A registered provider receives the span name, the ``collective`` flag and any
     extra annotations, and its context manager brackets the body.
  4. Registration is reversible: ``set_span_provider`` returns the previous provider,
     and setting ``None`` restores no-op behaviour.
  5. ``span`` does not swallow exceptions -- neither the body's nor the provider's.
  6. ``trainer.py`` uses the indirection rather than importing a concrete profiler,
     which is what keeps PaddleFormers standalone.
"""

import contextlib
import inspect
import unittest

from paddleformers.trainer import startup_profile


class SpyProvider:
    """Records every span it is asked to open, and when each one entered/exited."""

    def __init__(self):
        self.calls = []
        self.events = []

    def __call__(self, name, collective=False, **extra):
        self.calls.append((name, collective, extra))
        return self._cm(name)

    @contextlib.contextmanager
    def _cm(self, name):
        self.events.append(("enter", name))
        try:
            yield name
        finally:
            self.events.append(("exit", name))


class TestStartupProfile(unittest.TestCase):
    def setUp(self):
        # Never leak a provider into other tests: the registry is module global.
        self._saved = startup_profile.set_span_provider(None)
        self.addCleanup(startup_profile.set_span_provider, self._saved)

    def test_noop_by_default(self):
        self.assertIsNone(startup_profile.get_span_provider())
        first = startup_profile.span("a")
        second = startup_profile.span("b", collective=True, n_params=3)
        # Same singleton => calling span() when profiling is off allocates nothing.
        self.assertIs(first, second)
        with startup_profile.span("a") as s:
            self.assertIs(s, first)
            # Annotating a disabled span must not raise, and must stay chainable.
            self.assertIs(s.note(k=1), s)

    def test_provider_receives_name_flags_and_extras(self):
        spy = SpyProvider()
        startup_profile.set_span_provider(spy)
        with startup_profile.span("read_metadata") as s:
            self.assertEqual(s, "read_metadata")
        with startup_profile.span("load_master_weight", collective=True, n=7):
            pass
        self.assertEqual(
            spy.calls,
            [("read_metadata", False, {}), ("load_master_weight", True, {"n": 7})],
        )
        # The body really runs inside the provider's context manager.
        self.assertEqual(
            spy.events,
            [
                ("enter", "read_metadata"),
                ("exit", "read_metadata"),
                ("enter", "load_master_weight"),
                ("exit", "load_master_weight"),
            ],
        )

    def test_registration_is_reversible(self):
        spy = SpyProvider()
        self.assertIsNone(startup_profile.set_span_provider(spy))
        self.assertIs(startup_profile.get_span_provider(), spy)
        self.assertIs(startup_profile.set_span_provider(None), spy)
        self.assertIsNone(startup_profile.get_span_provider())
        with startup_profile.span("after_uninstall"):
            pass
        self.assertEqual(spy.calls, [])

    def test_body_exception_propagates_and_span_is_closed(self):
        spy = SpyProvider()
        startup_profile.set_span_provider(spy)
        with self.assertRaises(ValueError):
            with startup_profile.span("boom"):
                raise ValueError("from body")
        self.assertEqual(spy.events, [("enter", "boom"), ("exit", "boom")])

    def test_provider_exception_is_not_hidden(self):
        def broken(name, collective=False, **extra):
            raise RuntimeError("bad provider")

        startup_profile.set_span_provider(broken)
        # Profiling code is expected to swallow its own errors; hiding them here would
        # also hide bugs in the instrumented body.
        with self.assertRaises(RuntimeError):
            with startup_profile.span("x"):
                pass

    def test_trainer_does_not_depend_on_a_concrete_profiler(self):
        from paddleformers.trainer import trainer as trainer_mod

        self.assertIs(trainer_mod._sprof_span, startup_profile.span)
        source = inspect.getsource(trainer_mod)
        self.assertNotIn("startup_profiler", source)
        # Spans are only reachable through the indirection.
        self.assertIn("from .startup_profile import span as _sprof_span", source)


if __name__ == "__main__":
    unittest.main()
