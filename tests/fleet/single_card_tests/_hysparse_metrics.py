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

"""Reusable, test-only precision metrics for HySparse validation.

Compares a candidate tensor ``got`` against a reference ``ref`` (any mix of
paddle tensors / numpy arrays, any float dtype). Everything is up-cast to
``float32`` and flattened before comparison so that a bf16 kernel output can be
scored against an fp32 dense reference.

Non-finite handling is explicit and happens *before* any ``allclose``:
positions where both sides are the *same* infinity (+inf/+inf or -inf/-inf)
are treated as matching; every other non-finite position (nan, or an inf that
the other side does not match) is counted as a hard mismatch that fails the
comparison regardless of the numeric tolerance. Numeric metrics (max abs, max
relative, RMSE, cosine) are computed only over the entries finite on both
sides, with a small ``eps`` guarding the relative denominator and zero refs.
"""

from dataclasses import dataclass

import numpy as np

# (rtol, atol) keyed by the *lowest*-precision dtype seen across got/ref.
_DTYPE_TOL = {
    "bfloat16": (1.0e-2, 2.0e-2),
    "float16": (1.0e-3, 1.0e-3),
    "float32": (1.0e-5, 1.0e-6),
    "float64": (1.0e-7, 1.0e-8),
}
# Ordered low -> high precision; the coarser dtype wins the tolerance.
_PRECISION_ORDER = ("bfloat16", "float16", "float32", "float64")
_DEFAULT_TOL = _DTYPE_TOL["float32"]


def _dtype_key(x):
    """Best-effort dtype name ('bfloat16'/'float16'/'float32'/...) for x."""
    dt = getattr(x, "dtype", None)
    s = str(dt).lower() if dt is not None else ""
    for name in _PRECISION_ORDER:
        if name in s:
            return name
    return None


def _tol_for(got, ref):
    keys = [k for k in (_dtype_key(got), _dtype_key(ref)) if k is not None]
    if not keys:
        return _DEFAULT_TOL
    # coarsest (earliest in precision order) dtype governs the tolerance.
    key = min(keys, key=_PRECISION_ORDER.index)
    return _DTYPE_TOL[key]


def _to_f32_flat(x):
    """Flatten x to a contiguous 1-D float32 numpy array."""
    if x is None:
        raise ValueError("tensor is None (missing gradient?)")
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "numpy"):
        try:
            arr = x.astype("float32").numpy()  # handles bf16 paddle tensors
        except Exception:
            arr = x.numpy()
    else:
        arr = np.asarray(x)
    return np.ascontiguousarray(arr).astype(np.float32).reshape(-1)


@dataclass
class Metrics:
    n: int
    got_all_finite: bool
    ref_all_finite: bool
    n_nonfinite_mismatch: int
    max_abs: float
    max_rel: float
    rmse: float
    cosine: float
    rel_l2: float
    allclose: bool
    rtol: float
    atol: float


def compute_metrics(got, ref, *, eps=1.0e-12, rtol=None, atol=None):
    """Return :class:`Metrics` comparing ``got`` against ``ref``."""
    g = _to_f32_flat(got)
    r = _to_f32_flat(ref)
    if g.shape != r.shape:
        raise ValueError(f"shape mismatch: got {g.shape} vs ref {r.shape}")

    d_rtol, d_atol = _tol_for(got, ref)
    rtol = d_rtol if rtol is None else rtol
    atol = d_atol if atol is None else atol

    g_fin = np.isfinite(g)
    r_fin = np.isfinite(r)
    both_fin = g_fin & r_fin

    matched_inf = (np.isposinf(g) & np.isposinf(r)) | (
        np.isneginf(g) & np.isneginf(r)
    )
    nonfinite_any = (~g_fin) | (~r_fin)
    # A non-finite position is a mismatch unless it is a matching infinity.
    n_nonfinite_mismatch = int((nonfinite_any & ~matched_inf).sum())

    gf = g[both_fin]
    rf = r[both_fin]
    if gf.size:
        abs_err = np.abs(gf - rf)
        max_abs = float(abs_err.max())
        max_rel = float((abs_err / (np.abs(rf) + eps)).max())
        rmse = float(np.sqrt(np.mean(abs_err * abs_err)))
        denom = float(np.linalg.norm(gf) * np.linalg.norm(rf)) + eps
        cosine = float(np.dot(gf, rf) / denom)
        # Relative Frobenius error ||got-ref|| / ||ref||. Unlike cosine this is
        # scale-SENSITIVE: a gradient off by a constant factor or offset (which
        # cosine ignores) shows up here, so it is the magnitude-aware guard.
        rel_l2 = float(np.linalg.norm(gf - rf) / (np.linalg.norm(rf) + eps))
        close_finite = bool(np.allclose(gf, rf, rtol=rtol, atol=atol))
    else:
        max_abs = max_rel = rmse = 0.0
        cosine = 1.0
        rel_l2 = 0.0
        close_finite = True

    allclose = bool(close_finite and n_nonfinite_mismatch == 0)
    return Metrics(
        n=int(g.size),
        got_all_finite=bool(g_fin.all()),
        ref_all_finite=bool(r_fin.all()),
        n_nonfinite_mismatch=n_nonfinite_mismatch,
        max_abs=max_abs,
        max_rel=max_rel,
        rmse=rmse,
        cosine=cosine,
        rel_l2=rel_l2,
        allclose=allclose,
        rtol=rtol,
        atol=atol,
    )


def format_metrics(name, m):
    """One-line human-readable summary of a :class:`Metrics`."""
    return (
        f"[{name}] n={m.n} finite(got={m.got_all_finite},ref={m.ref_all_finite}) "
        f"nf_mismatch={m.n_nonfinite_mismatch} max_abs={m.max_abs:.3e} "
        f"max_rel={m.max_rel:.3e} rmse={m.rmse:.3e} rel_l2={m.rel_l2:.3e} "
        f"cos={m.cosine:.6f} allclose(rtol={m.rtol:g},atol={m.atol:g})={m.allclose}"
    )


def assert_close(
    testcase,
    name,
    got,
    ref,
    *,
    min_cos=None,
    max_rel_l2=None,
    require_allclose=False,
    require_finite=True,
    rtol=None,
    atol=None,
    eps=1.0e-12,
    verbose=True,
):
    """Compute metrics, print them, and assert the requested guarantees.

    ``require_finite`` rejects any non-finite mismatch (the default). Set
    ``min_cos`` to enforce a cosine floor, ``max_rel_l2`` to enforce a
    magnitude-sensitive relative-Frobenius ceiling (catches constant-factor /
    offset errors that cosine is blind to), and/or ``require_allclose`` for a
    dtype-aware ``allclose``. Returns the :class:`Metrics` for further checks.
    """
    m = compute_metrics(got, ref, eps=eps, rtol=rtol, atol=atol)
    if verbose:
        print(format_metrics(name, m))
    if require_finite:
        testcase.assertEqual(
            m.n_nonfinite_mismatch,
            0,
            f"{name}: {m.n_nonfinite_mismatch} non-finite mismatch(es)",
        )
    if min_cos is not None:
        testcase.assertGreater(
            m.cosine, min_cos, f"{name}: cosine {m.cosine:.6f} <= {min_cos}"
        )
    if max_rel_l2 is not None:
        testcase.assertLess(
            m.rel_l2,
            max_rel_l2,
            f"{name}: rel_l2 {m.rel_l2:.3e} >= {max_rel_l2:g}",
        )
    if require_allclose:
        testcase.assertTrue(
            m.allclose,
            f"{name}: not allclose (max_abs={m.max_abs:.3e}, "
            f"max_rel={m.max_rel:.3e}, rtol={m.rtol:g}, atol={m.atol:g})",
        )
    return m
