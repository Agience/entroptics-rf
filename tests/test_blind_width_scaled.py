"""
The width, read blind at a rate ratio that is not one.

A receiver at another rate sees a sender's rows as a period of L / rho of its own samples.  The
keyed aperture is a separate draw per width, so the period is read per width: one constant-modulus
map per band of scales, folded at every period the band allows (``_period_read``).  These pin that
the width is then a search bound and not a parameter -- ``_discover_scaled(iq, key, None)`` reads it
with the scale -- and the pieces that make the read sound: another width does not read
constant-modulus at the true period, the scan's sub-sample phases wrap to the right sample, and a
noise-lifted entropy read does not widen the scale bracket.  Every mode is lit (K = L).
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf
from _ratechannel import at_rate, row_start

SEED = 1234


def _capture(L, K, rho, rows, *, snr=20, delay=-5.75, carrier=2e-4, seed=1):
    """``rows`` rows of a cyclic buffer, the capture starting inside its last row so the stream
    does not restart mid-capture (a restart breaks the differential chain the alphabet reads)."""
    bits = np.random.default_rng(seed).integers(0, 2, 2 * rows * K)
    tx = wf.compose(bits, K, L, SEED, M=4)
    return bits, at_rate(tx, rho, int(rows * L / rho), delay=delay, carrier=carrier, snr_db=snr,
                         seed=seed)


def test_the_width_is_read_blind_at_a_coprime_ratio():
    """No width given: every width the capture can evidence is searched, and the sender's width,
    scale and alphabet come back, with the payload on what was measured."""
    L, K, rho = 32, 32, 0.514403
    bits, y = _capture(L, K, rho, 160)
    d = wf._discover_scaled(y, SEED, None, carrier_span=0.3)
    assert (d.L, d.K, d.M) == (L, K, 4)
    assert abs(d.scale / rho - 1) < 2e-5, f"scale {d.scale!r} against {rho}"
    got = wf.read(y, d.L, d.K, SEED, M=d.M, offset=d.offset, phase=d.phase, carrier=d.carrier,
                  row0=d.row0, scale=d.scale)
    ref = bits[d.row0 * 2 * K:][:got.size]
    assert np.mean(got[:ref.size] != ref) < 5e-3


def test_equal_rates_are_read_blind_too():
    """rho = 1 is a scale like any other: the bracket reaches it and the read locks there."""
    L, K = 32, 32
    _, y = _capture(L, K, 1.0, 160, delay=-5.0)
    d = wf._discover_scaled(y, SEED, None, carrier_span=0.3)
    assert (d.L, d.K, d.M) == (L, K, 4)
    assert abs(d.scale - 1.0) < 2e-5


def test_the_period_is_read_with_the_width():
    """Given the width, the period read returns the sender's scale to its fold resolution, and the
    lock reads constant-modulus where the best chance fold of the width does not (20 dB, measured
    over five captures: lock 0.15-0.20, best chance fold 0.73-0.82)."""
    L, K, rho = 64, 64, 0.8333333
    _, y = _capture(L, K, rho, 160)
    lo, hi = wf._coarse_scale(y)
    got = wf._period_read(y, L, SEED, None, lo, hi, wf._coarse_center(y) or 0.0,
                          np.arange(-0.3, 0.31, wf.PERIOD_CARRIER_STEP))
    got.sort(key=lambda q: q[0])
    best = got[0]
    assert best[1] == L
    assert abs(best[3] / rho - 1) < 4 * best[5] / rho
    chance = [q[0] for q in got if abs(q[3] / rho - 1) > 1e-3 or abs(q[2] - best[2]) > 2]
    assert best[0] < 0.3 and (not chance or min(chance) > 2.5 * best[0])


def test_another_width_does_not_read_constant_modulus_at_the_true_period():
    """Why the period cannot be read without the width: the aperture of L +- 1, laid on the true
    rows (the same received period, so the scale (L +- 1) / P), reads like an isotropic capture
    (measured 1.00 and 0.99; the true width reads 0.033, the row read's neighbouring-rows floor)."""
    L, K, rho = 64, 64, 0.514403
    _, y = _capture(L, K, rho, 120, snr=None, carrier=0.0, delay=-200.0)
    P = L / rho
    s0 = row_start(2, L, rho, -200.0)
    assert wf._scaled_cm(y, L, SEED, None, s0, rho, 0.0, 32) < 0.06
    for W in (L - 1, L + 1):
        assert wf._scaled_cm(y, W, SEED, None, s0, W / P, 0.0, 32) > 0.7


def test_the_scan_wraps_a_whole_sample_phase_to_the_sample_before():
    """A row starting just past a received sample (delta = 1 - epsilon before the next one) is read
    from the sample BEFORE at phase zero.  The period here is 125.0005 samples and the first row
    starts at 100.0002, so every later row lands there.  Wrapped the wrong way, the scan placed the
    rows two samples early, at 98.0."""
    L, K = 64, 64
    rho = L / 125.0005
    _, y = _capture(L, K, rho, 40, snr=None, carrier=0.0, delay=-100.0002 * rho)
    v, s0, c = wf._scaled_scan(y, L, SEED, None, rho, 8, [0.0])[0]
    assert abs(s0 - 100.0002) < 0.25, f"scan placed the rows at {s0}"


def test_a_noise_lifted_entropy_read_does_not_widen_the_bracket():
    """At 10 dB the entropy read sits ~19% above a sender at rho 0.514 while the rank edge does not
    move, so the bracket is the rank edge's: it holds the truth and stops a few percent above it."""
    L, K, rho = 64, 64, 0.514403
    _, y = _capture(L, K, rho, 160, snr=10)
    lo, hi = wf._coarse_scale(y)
    assert lo < rho < hi
    assert hi < 1.04 * rho


def test_the_scan_keeps_its_best_starts_as_a_shortlist():
    """``keep`` returns each carrier's lowest local minima over the starts, best first -- distinct
    candidates for a caller that ranks them on more evidence -- and keep=1 is the scan as it was."""
    L, K = 64, 64
    rho = L / 125.0005
    _, y = _capture(L, K, rho, 40, snr=6, carrier=0.0, delay=-100.0002 * rho)
    cs = [-0.1, 0.0, 0.1]
    one = wf._scaled_scan(y, L, SEED, None, rho, 8, cs)
    many = wf._scaled_scan(y, L, SEED, None, rho, 8, cs, keep=3)
    for c in cs:
        mine = [q for q in many if q[2] == c]
        assert 1 <= len(mine) <= 3
        assert [q[0] for q in mine] == sorted(q[0] for q in mine)
        assert len({q[1] for q in mine}) == len(mine)
        assert mine[0] == next(q for q in one if q[2] == c)


def test_a_settle_score_can_weigh_the_rows_its_own_way():
    """``reduce`` receives the per-row reads of the rows the capture holds; None is their mean."""
    L, K, rho = 64, 64, 0.514403
    _, y = _capture(L, K, rho, 60, snr=20, carrier=0.0, delay=-200.0)
    s0 = row_start(2, L, rho, -200.0)
    base = wf._scaled_cm(y, L, SEED, None, s0, rho, 0.0, 16)
    assert wf._scaled_cm(y, L, SEED, None, s0, rho, 0.0, 16, reduce=np.mean) == base
    assert wf._scaled_cm(y, L, SEED, None, s0, rho, 0.0, 16, reduce=lambda r: float(r.size)) == 16
    assert wf._scaled_cm(y, L, SEED, None, s0, rho, 0.0, 16, reduce=np.max) >= base
