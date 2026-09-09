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

"""Opt-in span hooks for profiling the startup path (launch -> first loss).

Why this exists
---------------
The cold-start path of a large pretraining job (flex-checkpoint load, optimizer
build, first pipeline step) can take minutes, and the interesting sub-phases live
*inside* ``Trainer``. Downstream trainers therefore need timing anchors here, but
PaddleFormers must not grow a dependency on any particular profiler.

So this module only declares *where* the anchors are. It records nothing on its
own and has no default output. A downstream framework opts in by registering one
callable::

    from paddleformers.trainer.startup_profile import set_span_provider
    set_span_provider(my_profiler.span)   # span(name, collective=..., **extra)

The provider must return a context manager. Until one is registered -- i.e. for
every standalone PaddleFormers user -- :func:`span` returns a shared no-op object,
so the cost is one module-global load plus one ``is None`` test per call and no
allocation. That matters because a couple of the anchors sit in the per-step
pipeline path, not just in one-shot startup code.

Anything the provider raises is *not* caught here. Profiling code is expected to
swallow its own errors; hiding exceptions at this boundary would also hide bugs in
the instrumented body.
"""

__all__ = ["set_span_provider", "get_span_provider", "span"]


class _NullSpan:
    """Zero-cost stand-in used when no provider is registered."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def note(self, **kwargs):
        """Accept (and drop) the annotations a real span would attach."""
        return self


_NULL_SPAN = _NullSpan()

_span_provider = None


def set_span_provider(provider):
    """Register the callable that turns a span name into a context manager.

    Args:
        provider: ``callable(name: str, collective: bool = False, **extra)`` that
            returns a context manager, or ``None`` to uninstall.

    Returns:
        The previously registered provider, so callers can restore it (useful in
        tests).
    """
    global _span_provider
    previous, _span_provider = _span_provider, provider
    return previous


def get_span_provider():
    """Return the registered provider, or ``None`` if profiling is off."""
    return _span_provider


def span(name, collective=False, **extra):
    """Time the wrapped block, if and only if a provider is registered.

    Args:
        name: Span name. Providers are expected to nest it under the enclosing
            span, so keep it local (``"read_metadata"``, not a full path).
        collective: Whether the block contains collective communication. Lets a
            provider add device syncs and separate real communication time from
            waiting for the slowest rank.
        **extra: Free-form annotations forwarded to the provider.
    """
    if _span_provider is None:
        return _NULL_SPAN
    return _span_provider(name, collective=collective, **extra)
