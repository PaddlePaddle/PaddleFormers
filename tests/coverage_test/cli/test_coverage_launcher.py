# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import patch

from paddleformers.cli import launcher as launcher_mod


class TestLaunch(unittest.TestCase):
    def test_train_command_runs_numa_binding(self):
        with patch.object(launcher_mod, "_maybe_bind_trainer_numa", side_effect=RuntimeError("stop")) as mock_bind:
            with patch.object(sys, "argv", ["launcher", "train"]):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    launcher_mod.launch()

        mock_bind.assert_called_once_with()

    def test_numa_binding_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(launcher_mod, "_reexec_with_numactl") as mock_reexec:
                launcher_mod._maybe_bind_trainer_numa()

        mock_reexec.assert_not_called()

    def test_numa_binding_maps_local_rank_to_numa_node(self):
        cases = [
            ("0", 0, 0),
            ("1", 1, 0),
            ("2", 2, 1),
            ("3", 3, 1),
        ]

        for rank_env, expected_rank, expected_numa_node in cases:
            with self.subTest(rank_env=rank_env):
                with patch.dict(os.environ, {"BIND_TRAINER_NUMA": "1", "PADDLE_LOCAL_RANK": rank_env}, clear=True):
                    with patch.object(launcher_mod, "_reexec_with_numactl") as mock_reexec:
                        with patch("builtins.print"):
                            launcher_mod._maybe_bind_trainer_numa()

                mock_reexec.assert_called_once_with(expected_rank, expected_numa_node)

    def test_numa_binding_uses_flags_selected_gpus_when_local_rank_missing(self):
        with patch.dict(os.environ, {"BIND_TRAINER_NUMA": "1", "FLAGS_selected_gpus": "2,3"}, clear=True):
            with patch.object(launcher_mod, "_reexec_with_numactl") as mock_reexec:
                with patch("builtins.print"):
                    launcher_mod._maybe_bind_trainer_numa()

        mock_reexec.assert_called_once_with(2, 1)

    def test_numa_binding_requires_rank_when_enabled(self):
        with patch.dict(os.environ, {"BIND_TRAINER_NUMA": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PADDLE_LOCAL_RANK or FLAGS_selected_gpus"):
                launcher_mod._maybe_bind_trainer_numa()

    def test_numa_binding_rejects_unsupported_rank(self):
        with patch.dict(os.environ, {"BIND_TRAINER_NUMA": "1", "PADDLE_LOCAL_RANK": "4"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "only supports local rank 0-3"):
                launcher_mod._maybe_bind_trainer_numa()

    def test_reexec_with_numactl_builds_command_and_env(self):
        with patch.dict(os.environ, {"EXISTING_ENV": "1"}, clear=True):
            with patch.object(launcher_mod.shutil, "which", return_value="/usr/bin/numactl") as mock_which:
                with patch.object(launcher_mod.os, "execvpe") as mock_execvpe:
                    with patch.object(launcher_mod.sys, "executable", "/usr/bin/python"):
                        with patch.object(launcher_mod.sys, "argv", ["launcher", "train", "--config", "cfg"]):
                            with patch("builtins.print"):
                                launcher_mod._reexec_with_numactl(2, 1)

        mock_which.assert_called_once_with("numactl")
        mock_execvpe.assert_called_once()
        executable, args, env = mock_execvpe.call_args.args
        self.assertEqual(executable, "/usr/bin/numactl")
        self.assertEqual(
            args,
            [
                "/usr/bin/numactl",
                "--cpunodebind=1",
                "--membind=1",
                "/usr/bin/python",
                "-u",
                "launcher",
                "train",
                "--config",
                "cfg",
            ],
        )
        self.assertEqual(env["EXISTING_ENV"], "1")
        self.assertEqual(env[launcher_mod.BIND_TRAINER_NUMA_EXECED], "1")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    def test_reexec_with_numactl_is_idempotent(self):
        with patch.dict(os.environ, {launcher_mod.BIND_TRAINER_NUMA_EXECED: "1"}, clear=True):
            with patch.object(launcher_mod.shutil, "which") as mock_which:
                with patch.object(launcher_mod.os, "execvpe") as mock_execvpe:
                    launcher_mod._reexec_with_numactl(0, 0)

        mock_which.assert_not_called()
        mock_execvpe.assert_not_called()

    def test_reexec_with_numactl_requires_numactl(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(launcher_mod.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "requires numactl"):
                    launcher_mod._reexec_with_numactl(0, 0)


if __name__ == "__main__":
    unittest.main()
