"""
The scale is a continuous coordinate.

A sender chooses its own rate, so the receiver sees a row of L transmitted samples as L / rho of
its own, rho real.  These pin the read that evaluates the aperture at the receiver's sampling
positions: that it is the existing read at equal rates, that it closes at coprime and harmonic
ratios, and that it refuses a sender wider than the receiver rather than inventing a fit.  Every mode
is lit (K = L), so the geometry is read by the keyed constant-modulus read.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf
from _ratechannel import at_rate, row_start

K, L, SEED = 64, 64, 1234


def _tx(rows=200, bit_seed=1):
    bits = np.random.default_rng(bit_seed).integers(0, 2, 2 * rows * K)
    return wf.compose(bits, K, L, SEED, M=4)


def test_at_equal_rates_it_is_the_existing_read():
    """Whole positions, rho = 1: the least-squares read is the adjoint read, to round-off."""
    iq = _tx()
    S = wf.screen(iq, L, offset=0, max_rows=40)
    want = S @ wf._aperture(L, SEED, None).conj().T
    got = wf._scaled_coords(iq, L, SEED, None, np.arange(40) * float(L), 1.0)
    assert np.max(np.abs(got - want)) < 1e-10 * np.max(np.abs(want))


@pytest.mark.parametrize("rho", [0.5, 1 / 3, 0.514403, 0.602816, 0.8333333])
def test_the_constant_modulus_read_closes_at_the_sender_scale(rho):
    """Harmonic and coprime ratios alike: on the sender's scale every row reads constant-modulus
    (row read, measured 0.029-0.034 -- the neighbouring rows' floor); one part in a thousand off
    it, it reads like a Gaussian capture (measured 0.87)."""
    tx = _tx()
    delay = 17.3
    y = at_rate(tx, rho, int(60 * L / rho) + 400, delay=delay)
    s0 = row_start(1, L, rho, delay)
    on = wf._scaled_cm(y, L, SEED, None, s0, rho, 0.0, 48)
    off = wf._scaled_cm(y, L, SEED, None, s0, rho * (1 + 1e-3), 0.0, 48)
    assert on < 0.06, f"the constant-modulus read is {on:.4f} on the true scale"
    assert off > 10 * on, f"a 1e-3 scale error reads {off:.4f} against {on:.4f}"


def test_a_sender_wider_than_the_receiver_is_refused():
    """Fewer samples than modes in a row is an underdetermined fit that can empty any mode, so it
    is no evidence: the read reports rows of nothing, worse than any real read."""
    y = at_rate(_tx(), 1.0, 8000)
    assert wf._scaled_cm(y, L, SEED, None, 3.0, 1.01, 0.0, 16) == wf._cm_missing(L)


def _read_at(rho, rows=120):
    bits = np.random.default_rng(3).integers(0, 2, 2 * rows * K)
    tx = wf.compose(bits, K, L, SEED, M=4)
    delay = -5.75                     # row 0 begins inside the capture
    y = at_rate(tx, rho, int((rows + 1) * L / rho) + 50, delay=delay)
    s0 = row_start(0, L, rho, delay)
    off = int(np.ceil(s0))
    got = wf.read(y, L, K, SEED, M=4, offset=off, phase=off - s0, scale=rho)
    n = min(got.size, bits.size)
    assert n >= bits.size - 2 * K * 2
    return float(np.mean(got[:n] != bits[:n]))


@pytest.mark.parametrize("rho", [0.5, 0.514403, 0.8333333])
def test_the_payload_reads_at_the_sender_scale(rho):
    """The keyed differential read on coordinates taken at the sender's scale, on a noiseless
    channel.  The bound is the floor of reading each row on its own (BER 1e-4..3e-3 at every
    ratio, measured when the read was row by row), which the joint read is under by 40 dB; the
    test below pins every bit."""
    assert _read_at(rho) < 3e-3


@pytest.mark.parametrize("rho", [0.5, 0.8333333])
def test_every_bit_returns_at_any_ratio(rho):
    """Noiseless, every bit: the read models the neighbouring rows' tails (the joint read,
    tests/test_joint_read.py), so the boundaries cost nothing."""
    assert _read_at(rho) == 0.0


@pytest.mark.parametrize("rho", [0.514403, 0.5])
def test_discovery_measures_the_sender_scale(rho):
    """Blind but for the key and a width range: the scale, width and alphabet are all
    read off a capture taken at a coprime (or harmonic) ratio, through noise and a carrier offset,
    and the payload decodes on what was measured."""
    rows = 160
    bits = np.random.default_rng(4).integers(0, 2, 2 * rows * K)
    tx = wf.compose(bits, K, L, SEED, M=4)
    y = at_rate(tx, rho, int(rows * L / rho), delay=-1000.4, carrier=2e-4, snr_db=20, seed=1)
    d = wf._discover_scaled(y, SEED, (63, 64, 65), carrier_span=0.3)
    assert (d.L, d.K, d.M) == (L, K, 4)
    assert abs(d.scale / rho - 1) < 2e-5, f"scale {d.scale!r} against {rho}"
    got = wf.read(y, d.L, d.K, SEED, M=d.M, offset=d.offset, phase=d.phase, carrier=d.carrier,
                  row0=d.row0, scale=d.scale)
    start = d.row0 * 2 * K                       # bits carried from the row the capture joined on
    ref = bits[start:start + got.size]
    assert np.mean(got[:ref.size] != ref) < 5e-3


@pytest.mark.parametrize("f_off", [0.2, -0.15])
def test_a_sender_off_the_receivers_centre_is_found(f_off):
    """A sender may choose its centre anywhere in the receiver's band: at half the receiver's rate
    and 0.2 cycles/sample off centre it reads as ~26 cycles per row of carrier, far beyond any
    oscillator span.  The band's centre is measured from the occupancy and the read probes there."""
    rho, rows = 0.5, 160
    bits = np.random.default_rng(6).integers(0, 2, 2 * rows * K)
    tx = wf.compose(bits, K, L, SEED, M=4)
    y = at_rate(tx, rho, int(rows * L / rho), delay=-800.3, carrier=f_off, snr_db=20, seed=2)
    assert wf._coarse_center(y) == pytest.approx(f_off, abs=0.01)
    d = wf._discover_scaled(y, SEED, (64,), carrier_span=0.3)
    assert (d.L, d.K, d.M) == (L, K, 4)
    assert d.carrier == pytest.approx(f_off * L / rho, abs=0.05)
    assert abs(d.scale / rho - 1) < 2e-5
