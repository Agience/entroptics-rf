"""Compiled (numba) drop-in kernels for the entroptics-rf steady-state receive path.

rf-local, so the published entroptics core package is untouched (the entropy reduction is consumed
from PyPI as ``entroptics.entropy.shannon_bits``; see the recommendation in research/realtime).  The
one kernel here is the constant-modulus row read :func:`entroptics_rf.waveform._cm_rows`, whose cost
is a transcendental reduction ``-sum p log2 p`` over the mode energies of each row -- the entropy
deficit against an even spread, floored at the flat-Dirichlet cap.  It fuses the ``shannon_bits``
reduction, the deficit normalization and the cap into one pass with no per-row Python.

``fastmath=False`` (fastmath reorders float ops).  The kernel is used only when numba imports AND
it reproduces the Python reference (``waveform._cm_rows_py``) within rtol 1e-9 / atol 1e-12 on a
fixed corpus (``waveform._gate_cm_rows``); ``ENTROPTICS_FAST=0`` forces the Python path.

Dense linear algebra (``_scaled_scan``/``_scaled_coords``/``_joint_coords``) is deliberately NOT
ported here: those are numpy BLAS ``cgemm`` projections, and a numba matmul sums in a different
order than BLAS, so it is not bit-identical and the scan's argmax over candidate starts could flip.
They stay on the (single-BLAS-thread) numpy path.  See research/realtime/port_results.md.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def _cm_rows_kernel(e2, m, log2m, null, cap):
    """``e2``: (N, m) float64 row energies (NaNs still present).  Returns (N,) constant-modulus read.

    Per row: NaN anywhere -> NaN (the reference re-masks NaN rows); else the Shannon entropy in bits
    of the normalized energies (``-sum p log2 p``, p>0 terms only, 0 when the row has no energy),
    then ``(log2 m - H) / null`` capped at ``cap``.  Same branches and same cap as the reference;
    the only difference from numpy is the reduction order, ~1e-15 relative on m = 16 or 64 values."""
    N = e2.shape[0]
    out = np.empty(N, np.float64)
    for i in range(N):
        has_nan = False
        s = 0.0
        for j in range(m):
            v = e2[i, j]
            if np.isnan(v):
                has_nan = True
            else:
                s += v
        if has_nan:
            out[i] = np.nan
            continue
        if s <= 0.0:
            H = 0.0
        else:
            acc = 0.0
            for j in range(m):
                p = e2[i, j] / s
                if p > 0.0:
                    acc += p * np.log2(p)
            H = -acc
        val = (log2m - H) / null
        if val > cap:
            val = cap
        out[i] = val
    return out


def cm_rows(e, null: float, cap: float) -> np.ndarray:
    """Fused constant-modulus read of ``e`` (..., m); ``null`` = _cm_null(m), ``cap`` = 1+3*spread(m).

    Same signature-shape contract as ``waveform._cm_rows``: reduces the last axis, keeps the leading
    shape."""
    e = np.asarray(e, dtype=np.float64)
    m = int(e.shape[-1])
    e2 = np.ascontiguousarray(e.reshape(-1, m))
    out = _cm_rows_kernel(e2, m, float(np.log2(m)), float(null), float(cap))
    return out.reshape(e.shape[:-1])
