"""
The analog mode: the payload is a direction, so channel loss is drift rather than bit errors.

The property that distinguishes it from the digital path is the absence of a cliff -- fidelity
decays smoothly and keeps meaning well past where bits would have collapsed.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf

D = 256
SEED = 99


def _vec(seed=0):
    r = np.random.default_rng(seed)
    return r.standard_normal(D) + 1j * r.standard_normal(D)


def _cos(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _noisy(iq, snr_db, seed=0):
    r = np.random.default_rng(seed)
    p = np.mean(np.abs(iq) ** 2)
    n = (r.standard_normal(len(iq)) + 1j * r.standard_normal(len(iq))) \
        * np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return (iq + n).astype(np.complex64)


def test_spread_operator_has_orthonormal_columns():
    """The adjoint is only the exact inverse if the columns are orthonormal."""
    M = wf.spread_operator(512, D, SEED)
    assert M.shape == (512, D)
    assert np.allclose(M.conj().T @ M, np.eye(D), atol=1e-10)


def test_clean_recovery_is_exact():
    v = _vec()
    got = wf.read_embedding(wf.compose_embedding(v, SEED, spread=1.0), D, SEED)
    assert _cos(got, v) > 0.999999


@pytest.mark.parametrize("spread", [1.0, 2.0, 4.0])
def test_spread_sets_the_sample_cost(spread):
    iq = wf.compose_embedding(_vec(), SEED, spread=spread)
    assert len(iq) == max(D, int(round(D * spread)))


def test_degradation_is_graceful_and_monotone():
    """No cliff: meaning survives far below where a bit stream would have collapsed."""
    v = _vec()
    iq = wf.compose_embedding(v, SEED, spread=1.0)
    cos = [_cos(wf.read_embedding(_noisy(iq, s, seed=s), D, SEED), v) for s in (18, 12, 6, 0)]
    assert cos == sorted(cos, reverse=True), "fidelity should fall monotonically with SNR"
    assert cos[0] > 0.98
    assert cos[-1] > 0.5, "at 0 dB the direction should still carry meaning, not collapse"


def test_spreading_buys_reach():
    """N/D is the processing gain: more samples per dimension, better recovery at the same SNR."""
    v = _vec()
    narrow = _cos(wf.read_embedding(_noisy(wf.compose_embedding(v, SEED, spread=1.0), 0, 1), D, SEED), v)
    wide = _cos(wf.read_embedding(_noisy(wf.compose_embedding(v, SEED, spread=4.0), 0, 1), D, SEED), v)
    assert wide > narrow


def test_wrong_key_recovers_nothing():
    """A wrong seed is a different operator, so the direction is gone -- not merely degraded."""
    v = _vec()
    got = wf.read_embedding(wf.compose_embedding(v, SEED), D, SEED + 1)
    assert _cos(got, v) < 0.2


def test_scale_is_not_carried_but_direction_is():
    """Direction is the payload; the transmitter normalizes, so amplitude cannot survive."""
    v = _vec()
    a = wf.read_embedding(wf.compose_embedding(v, SEED), D, SEED)
    b = wf.read_embedding(wf.compose_embedding(1000.0 * v, SEED), D, SEED)
    assert _cos(a, b) > 0.999999


def test_zero_vector_is_refused_rather_than_silently_meaningless():
    with pytest.raises(ValueError):
        wf.compose_embedding(np.zeros(D), SEED)
    with pytest.raises(ValueError):
        wf.compose_embedding(np.zeros(0), SEED)


def test_the_dimension_is_measured_when_it_is_not_given():
    """`read_embedding` needed D until the spread operator was nested; now it measures it.

    Untested until coverage said so: the function existed, its behaviour was in the paper, and
    nothing in the suite ever called it.  It was also still listed as a documented exception in
    the entry-point audit, which is the audit asserting a shared parameter that had been removed.

    The read works because the operator is nested in D, so the energy outside the first D columns
    falls monotonically and reaches the arithmetic's floor exactly at the dimension sent."""
    for D, spread in ((256, 1.0), (256, 2.0), (100, 2.0), (64, 4.0)):
        v = np.sin(np.linspace(0, 9, D)) + 0j
        iq = wf.compose_embedding(v, SEED, spread=spread)
        assert wf.embedded_dimension(iq, SEED) == D, f"D={D} spread={spread}"
        got = wf.read_embedding(iq, seed=SEED)          # no D supplied
        assert len(got) == D
        assert _cos(got, v) > 0.999999


def test_measuring_the_dimension_agrees_with_being_told_it():
    """The blind path must be the told path, not a second implementation of it."""
    D = 256
    v = np.sin(np.linspace(0, 9, D)) + 0j
    iq = wf.compose_embedding(v, SEED, spread=2.0)
    assert np.allclose(wf.read_embedding(iq, seed=SEED), wf.read_embedding(iq, D, SEED))


def test_under_noise_the_dimension_read_reports_the_whole_space():
    """Noise fills every dimension, so no residual reaches the floor and the read returns N.

    That is safe rather than wrong -- reading at D = N still recovers the direction, because the
    surplus dimensions carry only what the channel put there.  What is lost is the knowledge of D,
    not the message.  Pinned so the failure mode stays the benign one."""
    D = 256
    v = np.sin(np.linspace(0, 9, D)) + 0j
    iq = wf.compose_embedding(v, SEED, spread=2.0)
    noisy = _noisy(iq, 12, seed=3)
    assert wf.embedded_dimension(noisy, SEED) == len(np.asarray(noisy).ravel())
    got = wf.read_embedding(noisy, seed=SEED)
    assert _cos(got[:D], v) > 0.9
