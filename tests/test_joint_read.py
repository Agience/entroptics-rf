"""
The joint read: a receiver at another rate reads each row together with what surrounds it.

Between the transmitted instants the signal is the band-limited continuation of the whole stream,
so the neighbouring rows' content reaches into a row through the converter's 1/d tails.  The row
read models a row alone and meets that as a floor (measured at L=64: coordinate error -19.7 to
-20.2 dB against the transmitted coordinates).  ``_joint_coords`` models it, as a window around the
row plus a skeleton of far samples, and nothing on the air changes (research/joint_read.md).  These
pin:

- the floor is gone at coprime and harmonic scales, with and without a carrier;
- the dense coherent orders the floor broke now read bit-exact, noiseless;
- the read passes no more noise than the row read, to within a few percent;
- where it cannot run it is the row read exactly (a sender wider than the receiver, a row at the
  capture's edge), or near a scale of 1 the band read.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf
from _ratechannel import at_rate

L, K, SEED = 64, 64, 1234
DELAY = -2.0 * L - 0.37                 # the capture starts two rows before the burst, mid-sample


def _capture(rho, *, rows=40, M=4, coherent=False, carrier=0.0, snr_db=None, seed=0):
    b = int(np.log2(M))
    bits = np.random.default_rng(seed).integers(0, 2, b * rows * K)
    tx = (wf.compose_qam if coherent else wf.compose)(bits, K, L, SEED, M=M)
    T = tx.size // L
    y = at_rate(tx, rho, int((T * L + 4 * L) / rho), delay=DELAY, carrier=carrier,
                snr_db=snr_db, seed=seed + 1)
    return bits, y, T, tx


def _starts(rho, T):
    return (np.arange(T) * L - DELAY) / rho


def _truth(tx):
    """The transmitted coordinates, row by row (row radius and all): the transmitted samples read
    on the aperture at the transmitted rate."""
    return wf.screen(tx, L) @ np.conj(wf._aperture(L, SEED, None)).T


def _error_db(A, truth):
    """The read's coordinate error against the transmitted coordinates, in dB of their energy, on
    the rows the read returned."""
    ok = ~np.isnan(A).any(axis=1)
    return 10 * np.log10(np.sum(np.abs(A[ok] - truth[ok]) ** 2) / np.sum(np.abs(truth[ok]) ** 2))


@pytest.mark.parametrize("rho", [0.5, 0.514403, 0.8333333])
def test_the_neighbouring_rows_floor_is_gone(rho):
    """Noiseless, true geometry: the row read sits at its floor (measured -19.7 to -20.2 dB of
    coordinate error) and the joint read is over 45 dB under it (-66 to -73 dB), near the channel
    model's own (it quantizes positions to 1/1024 sample)."""
    _, y, T, tx = _capture(rho)
    s = _starts(rho, T)
    truth = _truth(tx)
    row = _error_db(wf._scaled_coords(y, L, SEED, None, s, rho), truth)
    joint = _error_db(wf._joint_coords(y, L, SEED, None, s, rho), truth)
    assert row > -23.0, f"the row read reads {row:.1f} dB: the floor this pins has moved"
    assert joint < -55.0, f"joint read {joint:.1f} dB against the row read's {row:.1f}"


def test_the_floor_is_gone_under_a_carrier():
    """The carrier is taken off across the whole window, relative to the row's own start, which
    is the row read's convention, so the two agree on what a coordinate is: the transmitted
    coordinates turned by the carrier's phase at the row's start (joint read measured -68 dB
    against them, row read -20 dB)."""
    rho, f = 0.6028, 3e-3
    _, y, T, tx = _capture(rho, carrier=f)
    c = f * L / rho                                 # cycles per row, the module's unit
    s = _starts(rho, T)
    A = wf._joint_coords(y, L, SEED, None, s, rho, c)
    B = wf._scaled_coords(y, L, SEED, None, s, rho, c)
    assert _error_db(A, _truth(tx) * np.exp(2j * np.pi * f * s)[:, None]) < -55.0
    ok = ~np.isnan(A).any(axis=1)
    # same coordinates as the row read, less its floor: the two differ by about the floor (0.10)
    rel = np.linalg.norm(A[ok] - B[ok]) / np.linalg.norm(B[ok])
    assert 1e-3 < rel < 0.3


@pytest.mark.parametrize("M", [16, 64])
def test_dense_coherent_orders_read_bit_exact_at_another_rate(M):
    """The informed coherent read at a continuous scale: the row read's floor breaks 64-QAM
    noiseless (16-QAM clears it); the joint read returns every bit of both."""
    rho = 0.8333333
    bits, y, _, _ = _capture(rho, M=M, coherent=True)
    s0 = -DELAY / rho
    off = int(np.ceil(s0))
    A = wf._coherent_project(y, L, K, SEED, off, off - s0, 0.0, scale=rho)
    got = wf._qam_bits(wf._dekeyed(A, 0, K, SEED, 0.0, L=L), M)
    n = min(got.size, bits.size)
    assert n >= 30 * K * int(np.log2(M))
    assert np.array_equal(got[:n], bits[:n])

    row = wf._scaled_coords(y, L, SEED, None, s0 + np.arange(A.shape[0]) * L / rho, rho)[:, :K]
    gr = wf._qam_bits(wf._dekeyed(row, 0, K, SEED, 0.0, L=L), M)
    if M >= 64:
        assert np.mean(gr[:n] != bits[:n]) > 1e-3, "the row read no longer errs: re-measure"


def test_the_payload_reads_through_the_public_reader():
    """``read`` at a scale goes through the joint read: DQPSK at 14 dB, where the row read's
    floor and the noise add."""
    rho = 0.514403
    bits, y, _, _ = _capture(rho, rows=60, snr_db=14.0)
    s0 = -DELAY / rho
    off = int(np.ceil(s0))
    got = wf.read(y, L, K, SEED, M=4, offset=off, phase=off - s0, scale=rho)
    assert got.size >= bits.size                     # and some wrapped rows past the burst
    assert np.mean(got[:bits.size] != bits) < 2e-3


@pytest.mark.parametrize("rho", [0.514403, 0.8333333, 0.9])
def test_it_passes_no_more_noise_than_the_row_read(rho):
    """On noise alone, the energy each read returns is what it lets through.  The skeleton's
    columns cost a few percent of it (measured 1.01-1.05)."""
    g = np.random.default_rng(5).standard_normal((2, 40 * L * 2))
    x = g[0] + 1j * g[1]
    s = 1.5 * L / rho + np.arange(30) * L / rho
    a = np.sum(np.abs(wf._joint_coords(x, L, SEED, None, s, rho)) ** 2)
    b = np.sum(np.abs(wf._scaled_coords(x, L, SEED, None, s, rho)) ** 2)
    assert 0.9 < a / b < 1.10


def test_where_it_cannot_run_it_is_the_row_read_or_the_band_read():
    """A sender wider than the receiver has fewer samples per row than modes, and no window
    fixes that: no row reads, as on the row read.  A scale within a hair of 1 would need a margin
    past ``JOINT_REACH``: it is the band read (``_band_coords``), which needs no spare samples and
    has no floor where the row read has one (measured at 0.999: -38.7 dB of coordinate error
    against the row read's -17.8; tests/test_blind_equal_rates.py).  A row at the
    capture's edge has no margin: it is the row read, value for value."""
    rho = 1.37
    _, y, T, _ = _capture(rho, rows=12)
    s = _starts(rho, T)
    assert np.isnan(wf._joint_coords(y, L, SEED, None, s, rho)).all()
    assert np.isnan(wf._scaled_coords(y, L, SEED, None, s, rho)).all()
    rho = 0.999
    _, y, T, tx = _capture(rho, rows=12)
    s = _starts(rho, T)
    A = wf._joint_coords(y, L, SEED, None, s, rho)
    assert np.array_equal(A, wf._band_coords(y, L, SEED, None, s, rho), equal_nan=True)
    B = wf._scaled_coords(y, L, SEED, None, s, rho)
    assert np.array_equal(np.isnan(A), np.isnan(B))
    assert _error_db(A, _truth(tx)) < _error_db(B, _truth(tx)) - 10.0
    assert wf._joint_window(L, 1.37) is None and wf._joint_window(L, 0.999) is None

    rho = 0.8333333
    _, y, T, _ = _capture(rho)
    edge = np.array([0.2, y.size - L / rho - 1.2])  # whole rows, with no margin either side
    A = wf._joint_coords(y, L, SEED, None, edge, rho)
    B = wf._scaled_coords(y, L, SEED, None, edge, rho)
    assert not np.isnan(A).any()
    assert np.allclose(A, B, rtol=0, atol=1e-12)
