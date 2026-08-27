# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/zero_cost_checkpoint.py"""

import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.trainer.utils.zero_cost_checkpoint import (
    ZCCTaskType,
    ZCCWorkerStatus,
    ZeroCostCheckpointCallback,
    ZeroCostCheckpointManager,
    _unwrap_opt_for_fused_states,
    md5,
    sharded_state_dict_compatibility,
)


class _FakeValue:
    """Stand-in for a `multiprocessing.Value`; only `.value` is accessed."""

    def __init__(self, value):
        self.value = value


class _FakeWorker:
    """Minimal ZCC worker double exposing the fields the manager barrier reads."""

    def __init__(self, worker_id=0, status=ZCCWorkerStatus.IDLE.value, global_step=0):
        self.worker_id = worker_id
        self.status = _FakeValue(status)
        self.global_step = _FakeValue(global_step)
        self.task_queue = MagicMock()  # get_idle_worker_for_saving puts PREPARE tasks here


def _bare_manager(**overrides):
    """Build a ZeroCostCheckpointManager without spawning worker processes.

    The offload barrier only touches a handful of plain attributes, so we bypass
    __init__ (which would spawn processes / require fleet) and set them directly.
    """
    m = ZeroCostCheckpointManager.__new__(ZeroCostCheckpointManager)
    m.workers = overrides.get("workers", [])
    m.current_worker = overrides.get("current_worker", None)
    m.global_step = overrides.get("global_step", 0)
    m.pipeline_hooks_steps = overrides.get("pipeline_hooks_steps", 1)
    m.current_pipeline_hook_step = overrides.get("current_pipeline_hook_step", 1)
    m.ready_to_save = overrides.get("ready_to_save", True)
    return m


class TestZCCTaskType(unittest.TestCase):
    """Tests for ZCCTaskType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        self.assertEqual(ZCCTaskType.UPDATE.value, 0)
        self.assertEqual(ZCCTaskType.PREPARE.value, 1)
        self.assertEqual(ZCCTaskType.OFFLOAD.value, 2)
        self.assertEqual(ZCCTaskType.FINISH.value, 3)
        self.assertEqual(ZCCTaskType.SET_EMA_STATE_DICT.value, 5)


class TestZCCWorkerStatus(unittest.TestCase):
    """Tests for ZCCWorkerStatus enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        self.assertEqual(ZCCWorkerStatus.IDLE.value, 0)
        self.assertEqual(ZCCWorkerStatus.OFFLOADING.value, 1)
        self.assertEqual(ZCCWorkerStatus.DUMPING.value, 2)
        self.assertEqual(ZCCWorkerStatus.ERROR.value, 3)


class TestMd5(unittest.TestCase):
    """Tests for md5 function."""

    def test_basic_md5(self):
        """Test that md5 returns correct hash for a tensor."""
        tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        result = md5(tensor)
        expected = hashlib.md5(tensor.numpy().tobytes()).hexdigest()
        self.assertEqual(result, expected)

    def test_consistent_md5(self):
        """Test that md5 is consistent for same tensor."""
        tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        result1 = md5(tensor)
        result2 = md5(tensor)
        self.assertEqual(result1, result2)


class TestUnwrapOptForFusedStates(unittest.TestCase):
    """Tests for _unwrap_opt_for_fused_states function."""

    def test_no_inner_opt(self):
        """Test with optimizer that has no _inner_opt."""
        optimizer = MagicMock(spec=[])
        result = _unwrap_opt_for_fused_states(optimizer)
        self.assertEqual(result, optimizer)

    def test_single_inner_opt(self):
        """Test unwrapping one level of _inner_opt."""
        inner_opt = MagicMock(spec=[])
        optimizer = MagicMock(spec=["_inner_opt"])
        optimizer._inner_opt = inner_opt
        result = _unwrap_opt_for_fused_states(optimizer)
        self.assertEqual(result, inner_opt)

    def test_nested_inner_opt_stops_at_sharding(self):
        """Test unwrapping stops at DygraphShardingOptimizer."""
        from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
            DygraphShardingOptimizer,
        )

        sharding_opt = MagicMock(spec=DygraphShardingOptimizer)
        sharding_opt.__class__ = DygraphShardingOptimizer
        outer_opt = MagicMock(spec=["_inner_opt"])
        outer_opt._inner_opt = sharding_opt
        result = _unwrap_opt_for_fused_states(outer_opt)
        self.assertEqual(result, sharding_opt)


class TestShardedStateDictCompatibility(unittest.TestCase):
    """Tests for sharded_state_dict_compatibility decorator."""

    def test_normal_dict_passthrough(self):
        """Test that normal dict is passed through unchanged."""

        @sharded_state_dict_compatibility
        def test_func(state_dict):
            return state_dict

        state_dict = {"key1": paddle.randn([2, 3])}
        result = test_func(state_dict)
        self.assertEqual(set(result.keys()), {"key1"})

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves the original function name."""

        @sharded_state_dict_compatibility
        def my_function(state_dict):
            return state_dict

        self.assertEqual(my_function.__name__, "my_function")


class TestSyncOffloadStatus(unittest.TestCase):
    """Barrier semantics of `ZeroCostCheckpointManager.sync_offload_status`.

    The worker only publishes `global_step.value` after the *whole* D2H of that step
    finishes (zero_cost_checkpoint.py: `offloaded_numels == all_numel`). The barrier
    must therefore keep polling until the worker's step catches up to the manager's
    current step -- it must not accept the worker's stale echo of a previous step.
    """

    def test_stale_worker_step_keeps_polling(self):
        # manager is offloading step 5; worker still reports step 4 (previous offload
        # not finished). The barrier must wait rather than pass on the stale value.
        worker = _FakeWorker(global_step=4)
        m = _bare_manager(workers=[worker], current_worker=worker, global_step=5)

        def _advance(_seconds):
            # Simulate the worker finishing the in-flight D2H after two poll cycles.
            _advance.calls += 1
            if _advance.calls >= 2:
                worker.global_step.value = 5

        _advance.calls = 0
        with patch("paddleformers.trainer.utils.zero_cost_checkpoint.time.sleep", side_effect=_advance) as slept:
            m.sync_offload_status()

        self.assertGreaterEqual(slept.call_count, 2)  # actually waited
        self.assertIsNone(m.current_worker)  # released only after the step matched
        self.assertEqual(m.current_pipeline_hook_step, 0)

    def test_matching_step_returns_without_waiting(self):
        worker = _FakeWorker(global_step=7)
        m = _bare_manager(workers=[worker], current_worker=worker, global_step=7)
        with patch("paddleformers.trainer.utils.zero_cost_checkpoint.time.sleep") as slept:
            m.sync_offload_status()
        slept.assert_not_called()
        self.assertIsNone(m.current_worker)
        self.assertEqual(m.current_pipeline_hook_step, 0)


class TestMaybeSyncOffloadStatus(unittest.TestCase):
    """Step-begin variant added by the fix: sync only when it is safe to block."""

    def test_no_current_worker_is_noop(self):
        m = _bare_manager(current_worker=None)
        m.sync_offload_status = MagicMock()
        m.maybe_sync_offload_status()
        m.sync_offload_status.assert_not_called()

    def test_single_chunk_synced_at_step_begin(self):
        # Whole offload already dispatched in one chunk -> safe to wait at step begin.
        worker = _FakeWorker(global_step=5)
        m = _bare_manager(
            workers=[worker],
            current_worker=worker,
            global_step=5,
            pipeline_hooks_steps=1,
            current_pipeline_hook_step=1,
        )
        m.maybe_sync_offload_status()
        self.assertIsNone(m.current_worker)  # sync_offload_status ran
        self.assertEqual(m.current_pipeline_hook_step, 0)

    def test_multi_chunk_not_fully_dispatched_does_not_wait(self):
        # PP>1 / grad-accum: remaining chunks are only sent during this step's fwd/bwd,
        # so blocking here would deadlock. Must decline and leave state untouched.
        worker = _FakeWorker(global_step=4)
        m = _bare_manager(
            workers=[worker],
            current_worker=worker,
            global_step=5,
            pipeline_hooks_steps=4,
            current_pipeline_hook_step=1,
        )
        m.sync_offload_status = MagicMock()
        m.maybe_sync_offload_status()
        m.sync_offload_status.assert_not_called()
        self.assertIs(m.current_worker, worker)  # unchanged
        self.assertEqual(m.current_pipeline_hook_step, 1)


class TestOnStepEndRefreshesGlobalStep(unittest.TestCase):
    """`on_step_end` must refresh `manager.global_step` on every offloading step.

    Without this the value stays frozen at the first save's step, and the barrier
    ends up comparing a value against itself (worker echo) and never waits.
    """

    def _make_callback(self, manager):
        cb = ZeroCostCheckpointCallback.__new__(ZeroCostCheckpointCallback)
        cb.manager = manager
        cb.runtime_timer = MagicMock()
        cb.zcc_ema_interval = 2
        # Stub the heavy collaborators; only the global_step bookkeeping is under test.
        cb.maybe_update_zcc_worker = MagicMock()
        cb._get_save_infos_based_on_steps = MagicMock(return_value=(None, None))
        cb.get_rng_states = MagicMock(return_value=None)
        return cb

    def test_save_branch_writes_current_step(self):
        manager = MagicMock()
        manager.global_step = 3  # stale value from a previous save
        cb = self._make_callback(manager)
        args = SimpleNamespace(zcc_save_ema_coef=None, pipeline_model_parallel_size=1)
        state = SimpleNamespace(global_step=7)
        control = SimpleNamespace(should_save=True)
        cb.on_step_end(args, state, control, model=MagicMock(), lr_scheduler=MagicMock(), optimizer=MagicMock())
        self.assertEqual(manager.global_step, 7)

    def test_ema_branch_writes_current_step(self):
        manager = MagicMock()
        manager.global_step = 3
        cb = self._make_callback(manager)
        args = SimpleNamespace(zcc_save_ema_coef=0.999, pipeline_model_parallel_size=1)
        state = SimpleNamespace(global_step=8)  # 8 % zcc_ema_interval(2) == 0
        control = SimpleNamespace(should_save=False)
        cb.on_step_end(args, state, control, model=MagicMock(), lr_scheduler=MagicMock(), optimizer=MagicMock())
        self.assertEqual(manager.global_step, 8)

    def test_ema_branch_skipped_off_interval_does_not_touch_step(self):
        manager = MagicMock()
        sentinel = object()
        manager.global_step = sentinel
        cb = self._make_callback(manager)
        args = SimpleNamespace(zcc_save_ema_coef=0.999, pipeline_model_parallel_size=1)
        state = SimpleNamespace(global_step=7)  # 7 % 2 != 0 -> no offload this step
        control = SimpleNamespace(should_save=False)
        cb.on_step_end(args, state, control, model=MagicMock(), lr_scheduler=MagicMock(), optimizer=MagicMock())
        self.assertIs(manager.global_step, sentinel)


_SLEEP = "paddleformers.trainer.utils.zero_cost_checkpoint.time.sleep"


class TestMultiWorkerPoolBarrier(unittest.TestCase):
    """Concurrency: with a real worker pool, idle-selection + reuse must never let a
    *historical* global_step echo (from any worker) release the current barrier.

    Drives the real `get_idle_worker_for_saving` (idle pick) and `sync_offload_status`
    (barrier) across 3 consecutive saves: fresh pick, pick-around-a-busy-worker, and
    reuse of a worker still carrying a stale step.
    """

    def _drive_barrier(self, m, target, decoys):
        """Poll the barrier: on the 1st tick the decoys echo the *current* step (must be
        ignored); only on a later tick does `target` actually finish. A regression that
        watches any worker instead of `current_worker` would exit after the 1st tick,
        leaving `target` still stale and the sleep count at 1 -- the asserts below catch it.
        """
        ticks = {"n": 0}

        def _advance(_seconds):
            ticks["n"] += 1
            if ticks["n"] == 1:
                for d in decoys:
                    d.global_step.value = m.global_step  # adversarial / historical echo
                self.assertIsNotNone(m.current_worker)  # still engaged, not released early
            else:
                target.global_step.value = m.global_step  # the real completion

        with patch(_SLEEP, side_effect=_advance) as slept:
            m.sync_offload_status()

        self.assertGreaterEqual(slept.call_count, 2)  # ignored the decoy echo, kept polling
        self.assertEqual(target.global_step.value, m.global_step)
        self.assertIsNone(m.current_worker)  # released only after `target` reached the step

    def test_pool_selection_and_reuse_ignore_stale_echo(self):
        wA, wB, wC = (_FakeWorker(worker_id=i) for i in range(3))
        m = _bare_manager(workers=[wA, wB, wC], global_step=0)
        save = (("flash", "persistent"), ("lr", "state", "rng"))  # dummy PREPARE payload

        # save #1 @10: all idle -> first idle (A) picked; A completes.
        m.global_step = 10
        m.get_idle_worker_for_saving(save)
        self.assertIs(m.current_worker, wA)
        wA.status.value = ZCCWorkerStatus.OFFLOADING.value
        self._drive_barrier(m, target=wA, decoys=[])

        # save #2 @20: A still DUMPING -> idle pick skips to B; A's echo must not release B.
        m.global_step = 20
        wA.status.value = ZCCWorkerStatus.DUMPING.value
        m.get_idle_worker_for_saving(save)
        self.assertIs(m.current_worker, wB)  # selection skipped the busy worker A
        wB.status.value = ZCCWorkerStatus.OFFLOADING.value
        self._drive_barrier(m, target=wB, decoys=[wA])

        # save #3 @30: A idle again but carries a stale echo -> reused; only A finishing releases.
        m.global_step = 30
        wA.status.value = ZCCWorkerStatus.IDLE.value
        wB.status.value = ZCCWorkerStatus.DUMPING.value
        self.assertLess(wA.global_step.value, 30)  # historical echo from an earlier save
        m.get_idle_worker_for_saving(save)
        self.assertIs(m.current_worker, wA)  # reused
        wA.status.value = ZCCWorkerStatus.OFFLOADING.value
        self._drive_barrier(m, target=wA, decoys=[wB, wC])


if __name__ == "__main__":
    unittest.main()
