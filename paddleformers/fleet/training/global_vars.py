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

"""PaddleFleet global variables."""

from paddleformers.fleet.timers import Timers

_GLOBAL_ARGS = None
_GLOBAL_TIMERS = None
_GLOBAL_PROFILE_TIMERS = None
_GLOBAL_TRAINING_LOGS = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def get_timers() -> Timers:
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def get_profile_timers():
    """Return the active profile timers object if available."""
    return _GLOBAL_PROFILE_TIMERS


def get_global_training_logs():
    """Return the active training logs object if one has been registered."""
    return _GLOBAL_TRAINING_LOGS


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_global_training_logs(logs):
    """Set the active training logs object."""
    global _GLOBAL_TRAINING_LOGS
    _GLOBAL_TRAINING_LOGS = logs


def set_profile_timers(timers):
    """Set the active timers object used by transformer profile scopes."""
    global _GLOBAL_PROFILE_TIMERS
    _GLOBAL_PROFILE_TIMERS = timers


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, f"{name} is not initialized."


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, f"{name} is already initialized."


def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = None

    global _GLOBAL_PROFILE_TIMERS
    _GLOBAL_PROFILE_TIMERS = None

    global _GLOBAL_TRAINING_LOGS
    _GLOBAL_TRAINING_LOGS = None


def set_global_variables(args):
    """Set args, timers etc."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
    set_args(args)
    _set_timers()


def unset_global_variables():
    global _GLOBAL_ARGS
    global _GLOBAL_TIMERS
    global _GLOBAL_PROFILE_TIMERS
    global _GLOBAL_TRAINING_LOGS
    _GLOBAL_ARGS = None
    _GLOBAL_TIMERS = None
    _GLOBAL_PROFILE_TIMERS = None
    _GLOBAL_TRAINING_LOGS = None
