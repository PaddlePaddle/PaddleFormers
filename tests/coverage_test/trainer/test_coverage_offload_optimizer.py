# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/offload_optimizer.py"""

import unittest
from unittest.mock import patch

import paddle
from paddle import _C_ops
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.optimizer import Optimizer

import paddleformers.trainer.utils.offload_optimizer as offload_optimizer_module
from paddleformers.trainer.utils.offload_optimizer import (
    hack_offload_optimizer,
    hack_offload_optimizer_eb5,
    offload,
    reload,
)

try:
    from paddle.optimizer.muon import Muon

    _MUON_AVAILABLE = True
except ImportError:  # pragma: no cover - Muon absent on very old Paddle
    Muon = None
    _MUON_AVAILABLE = False


class TestOffload(unittest.TestCase):
    """Tests for offload function."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_offload_cuda(self):
        """Test offload moves tensor to pinned memory on CUDA."""
        x = paddle.randn([4, 4])
        offload(x)
        # After offload, tensor should be on pinned place or cpu
        self.assertTrue(x.place.is_cuda_pinned_place() or x.place.is_cpu_place())

    def test_offload_cpu(self):
        """Test offload on CPU-only environment."""
        x = paddle.randn([4, 4])
        # On CPU, should use CPUPlace
        with patch("paddleformers.trainer.utils.offload_optimizer.paddle.is_compiled_with_cuda", return_value=False):
            with patch(
                "paddleformers.trainer.utils.offload_optimizer.paddle.is_compiled_with_xpu", return_value=False
            ):
                offload(x)


class TestReload(unittest.TestCase):
    """Tests for reload function."""

    def test_reload_basic(self):
        """Test reload moves tensor back to default device."""
        x = paddle.randn([4, 4])
        reload(x)
        # After reload, tensor should be on the default device
        self.assertIsNotNone(x)


class TestHackOffloadOptimizer(unittest.TestCase):
    """Tests for hack_offload_optimizer function."""

    def test_hack_offload_optimizer_patches_optimizer(self):
        """Test that hack_offload_optimizer patches _add_accumulator."""
        original_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
        try:
            hack_offload_optimizer()
            # After hacking, _add_accumulator should be replaced
            new_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
            self.assertNotEqual(original_add_accumulator, new_add_accumulator)
        finally:
            # Restore original method
            setattr(paddle.optimizer.Optimizer, "_add_accumulator", original_add_accumulator)

    def test_hack_offload_optimizer_eb5_mode(self):
        """Test that hack_offload_optimizer with eb5 mode calls eb5 variant."""
        original_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
        try:
            hack_offload_optimizer(mode="eb5")
            # After hacking, _add_accumulator should be replaced
            new_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
            self.assertNotEqual(original_add_accumulator, new_add_accumulator)
        finally:
            setattr(paddle.optimizer.Optimizer, "_add_accumulator", original_add_accumulator)


class TestHackOffloadOptimizerEb5(unittest.TestCase):
    """Tests for hack_offload_optimizer_eb5 function."""

    def test_hack_offload_optimizer_eb5_patches_optimizer(self):
        """Test that hack_offload_optimizer_eb5 patches _add_accumulator."""
        original_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
        try:
            hack_offload_optimizer_eb5()
            new_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
            self.assertNotEqual(original_add_accumulator, new_add_accumulator)
        finally:
            setattr(paddle.optimizer.Optimizer, "_add_accumulator", original_add_accumulator)


class _StubParam:
    """Minimal stand-in for a Muon parameter exposing name and offload flag."""

    def __init__(self, name, is_offload_opt):
        """Store the param name and its is_offload_opt gating flag."""
        self.name = name
        self.is_offload_opt = is_offload_opt


class _StubMuon:
    """Minimal Muon-like ``self`` for driving the patched update entry points."""

    _moment_acc_str = "moment1"

    def __init__(self, master_weights, momentums):
        """Hold master-weight and momentum lookup tables keyed by param name."""
        self._master_weights = master_weights
        self._momentums = momentums

    def _get_accumulator(self, acc_str, param):
        """Return the momentum buffer registered for ``param``."""
        return self._momentums[param.name]


@unittest.skipUnless(_MUON_AVAILABLE, "Muon optimizer not available")
class TestMuonMasterWeightLifecycle(unittest.TestCase):
    """Regression tests: master weight must be reloaded before the original Muon
    update and offloaded afterwards only when is_offload_opt is True. The lifecycle
    now lives inside the _muon_update_group / _muon_update entry points, so these
    tests execute those wrappers and assert the observable device-movement order."""

    def setUp(self):
        """Snapshot every attribute hack_offload_optimizer mutates."""
        self._snapshot = {
            "add_acc": Optimizer._add_accumulator,
            "adam": _C_ops.adam_,
            "adamw": _C_ops.adamw_,
            "insert_sync": HybridParallelOptimizer._insert_sync,
            "apply": Muon._apply_optimize,
            "group": getattr(Muon, "_muon_update_group", None),
            "update": getattr(Muon, "_muon_update", None),
        }

    def tearDown(self):
        """Restore all snapshotted attributes to keep the process hermetic."""
        Optimizer._add_accumulator = self._snapshot["add_acc"]
        _C_ops.adam_ = self._snapshot["adam"]
        _C_ops.adamw_ = self._snapshot["adamw"]
        HybridParallelOptimizer._insert_sync = self._snapshot["insert_sync"]
        Muon._apply_optimize = self._snapshot["apply"]
        self._restore_optional(Muon, "_muon_update_group", self._snapshot["group"])
        self._restore_optional(Muon, "_muon_update", self._snapshot["update"])

    @staticmethod
    def _restore_optional(cls, name, original):
        """Restore ``cls.name`` to ``original`` or delete it when it was absent."""
        if original is not None:
            setattr(cls, name, original)
        elif hasattr(cls, name):
            delattr(cls, name)

    def _run_group_wrapper(self, stub, group_params_grads):
        """Install a recording original _muon_update_group, hack it, and run the
        wrapper. Returns the ordered event log of device movements and update."""
        events = []
        Muon._muon_update_group = lambda self, gpg, *a, **k: events.append(("update",))
        hack_offload_optimizer()
        wrapper = Muon._muon_update_group
        with (
            patch.object(offload_optimizer_module, "reload", lambda t: events.append(("reload", id(t)))),
            patch.object(offload_optimizer_module, "offload", lambda t: events.append(("offload", id(t)))),
        ):
            wrapper(stub, group_params_grads)
        return events

    def _run_update_wrapper(self, stub, param, momentum_buffer):
        """Install a recording original _muon_update (forcing the fallback branch
        by removing _muon_update_group), hack it, and run the wrapper."""
        events = []
        if hasattr(Muon, "_muon_update_group"):
            delattr(Muon, "_muon_update_group")
        Muon._muon_update = lambda self, *a, **k: events.append(("update",))
        hack_offload_optimizer()
        wrapper = Muon._muon_update
        with (
            patch.object(offload_optimizer_module, "reload", lambda t: events.append(("reload", id(t)))),
            patch.object(offload_optimizer_module, "offload", lambda t: events.append(("offload", id(t)))),
        ):
            wrapper(stub, param, paddle.zeros([2, 2]), 0.1, momentum_buffer, 0.95, 5, True, 1e-9, 0.0, 3)
        return events

    def test_update_group_reloads_master_before_update_and_offloads_when_enabled(self):
        """_muon_update_group reloads master weight before the update and offloads it after when is_offload_opt=True."""
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _StubParam("w0", is_offload_opt=True)
        stub = _StubMuon({"w0": mw}, {"w0": mom})

        events = self._run_group_wrapper(stub, [(param, paddle.zeros([2, 2]))])

        idx = events.index(("update",))
        before, after = events[:idx], events[idx + 1 :]
        self.assertIn(("reload", id(mw)), before)
        self.assertIn(("reload", id(mom)), before)
        self.assertIn(("offload", id(mw)), after)
        self.assertIn(("offload", id(mom)), after)

    def test_update_group_skips_master_offload_when_disabled(self):
        """_muon_update_group still reloads master weight but never offloads it when is_offload_opt=False."""
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _StubParam("w0", is_offload_opt=False)
        stub = _StubMuon({"w0": mw}, {"w0": mom})

        events = self._run_group_wrapper(stub, [(param, paddle.zeros([2, 2]))])

        idx = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:idx])
        self.assertNotIn(("offload", id(mw)), events)
        self.assertNotIn(("offload", id(mom)), events)

    def test_update_group_offloads_only_enabled_params_in_mixed_group(self):
        """In a mixed group only the is_offload_opt=True param's master weight and momentum are offloaded."""
        mw_on, mw_off = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        mom_on, mom_off = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        p_on = _StubParam("on", is_offload_opt=True)
        p_off = _StubParam("off", is_offload_opt=False)
        stub = _StubMuon({"on": mw_on, "off": mw_off}, {"on": mom_on, "off": mom_off})

        events = self._run_group_wrapper(stub, [(p_on, paddle.zeros([2, 2])), (p_off, paddle.zeros([2, 2]))])

        idx = events.index(("update",))
        before = events[:idx]
        self.assertIn(("reload", id(mw_on)), before)
        self.assertIn(("reload", id(mw_off)), before)
        self.assertIn(("offload", id(mw_on)), events[idx + 1 :])
        self.assertNotIn(("offload", id(mw_off)), events)
        self.assertNotIn(("offload", id(mom_off)), events)

    def test_muon_update_reloads_master_before_update_and_offloads_when_enabled(self):
        """The _muon_update fallback reloads master weight before the update and offloads it after when is_offload_opt=True."""
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _StubParam("w0", is_offload_opt=True)
        stub = _StubMuon({"w0": mw}, {"w0": mom})

        events = self._run_update_wrapper(stub, param, mom)

        idx = events.index(("update",))
        before, after = events[:idx], events[idx + 1 :]
        self.assertIn(("reload", id(mw)), before)
        self.assertIn(("reload", id(mom)), before)
        self.assertIn(("offload", id(mw)), after)
        self.assertIn(("offload", id(mom)), after)

    def test_muon_update_skips_master_offload_when_disabled(self):
        """The _muon_update fallback still reloads master weight but never offloads it when is_offload_opt=False."""
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _StubParam("w0", is_offload_opt=False)
        stub = _StubMuon({"w0": mw}, {"w0": mom})

        events = self._run_update_wrapper(stub, param, mom)

        idx = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:idx])
        self.assertNotIn(("offload", id(mw)), events)
        self.assertNotIn(("offload", id(mom)), events)


if __name__ == "__main__":
    unittest.main()
