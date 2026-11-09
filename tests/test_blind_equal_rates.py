"""
Equal nominal rates on two clocks: the receiver's samples fall between the sender's.

Two radios at one nominal rate sample at a position that drifts through every fraction of a sample
(0.75 ppm at 2.4 Msps: once every 0.6 s), so a matched-rate capture is almost never on the sender's
instants.  These pin what the blind read does there (the simulated bench, research/rank_leak.md
"Blind decode on the current waveform"):

  * the band read (``waveform._band_coords``) has no floor where the row read has one of -12 to
    -22 dB, which sank blind 16- and 64-QAM;
  * the coherent read takes one gain for every mode (``waveform._dekeyed``);
  * ``decode_scaled`` reads dense QAM blind at equal rates off the sender's grid,
    family and order included;
  * the joint settle reads a sender a hair wider than the receiver rather than holding it at 1.

Short captures, one width, so each runs in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf
from _ratechannel import at_rate, row_start

L, SEED = 64, 1234


def _qam_stream(M, bursts=2, rows=60, seed=0):
    rng = np.random.default_rng(seed)
    b = int(np.log2(M))
    bits = [rng.integers(0, 2, rows * L * b) for _ in range(bursts)]
    return bits, np.concatenate([wf.compose_qam(x, L, L, SEED, M=M) for x in bits])


def _evm_db(A, M):
    """Payload error of rows read from a reference row at A[0], after the one-gain de-key."""
    Y = wf._dekeyed(A, 0, L, SEED, 0.0, L=L)
    return 10 * np.log10(np.mean(np.abs(Y - wf._qam(M)[wf._slice_qam(Y, M)]) ** 2))


@pytest.mark.parametrize("frac", [0.25, 0.5])
def test_the_band_read_has_no_floor_between_the_senders_samples(frac):
    """Noiseless 16-QAM at equal rates, true geometry, the receiver ``frac`` of a sample off the
    sender.  Measured (one gain): row read -16.5 / -14.0 dB at a quarter / half a sample, band read
    -35.0 / -31.3 dB; on the sender's instants both are exact."""
    _, tx = _qam_stream(16)
    y = at_rate(tx, 1.0, tx.size - 300, delay=frac)
    starts = row_start(61, L, 1.0, frac) + L * np.arange(40)        # the second burst's reference on
    row = _evm_db(wf._scaled_coords(y, L, SEED, None, starts, 1.0), 16)
    band = _evm_db(wf._joint_coords(y, L, SEED, None, starts, 1.0), 16)
    assert row > -18.0
    assert band < -30.0
    assert band < row - 12.0


def test_the_band_read_is_exact_on_the_senders_instants():
    _, tx = _qam_stream(16)
    y = at_rate(tx, 1.0, tx.size - 300, delay=0.0)
    starts = 61.0 * L + L * np.arange(40)
    want = wf._scaled_coords(y, L, SEED, None, starts, 1.0)
    got = wf._band_coords(y, L, SEED, None, starts, 1.0)
    assert np.max(np.abs(got - want)) < 1e-9 * np.max(np.abs(want))


def test_the_band_read_holds_no_row_the_capture_does_not():
    """The same rows as the row read are NaN: one the capture does not hold whole, and a sender
    wider than the receiver by more than SQUARE_REACH."""
    _, tx = _qam_stream(4, bursts=1, rows=20)
    starts = np.array([-3.0, 0.0, tx.size - 10.0])
    got = wf._band_coords(tx, L, SEED, None, starts, 1.0)
    assert np.isnan(got[0]).all() and not np.isnan(got[1]).any() and np.isnan(got[2]).all()
    assert np.isnan(wf._band_coords(tx, L, SEED, None, [0.0], 1.01)).all()


def test_the_coherent_read_takes_one_gain_for_every_mode():
    """Every mode is divided by the mean of the reference row's per-mode reads."""
    rng = np.random.default_rng(5)
    ref = wf._reference_row(L, SEED)
    h = 0.8 * np.exp(0.3j) * (1 + 0.2 * (rng.standard_normal(L) + 1j * rng.standard_normal(L)))
    pay = rng.standard_normal((3, L)) + 1j * rng.standard_normal((3, L))
    A = np.vstack([ref * h, pay])
    rad = wf.row_radius(L, SEED, None, rows=4)
    want = pay / np.mean(h) * (rad[0] / rad[1:])[:, None]
    assert np.allclose(wf._dekeyed(A, 0, L, SEED, 0.0, L=L), want)


@pytest.mark.parametrize("M,snr_db", [(16, 24), (64, 28), (256, 32)])
def test_scaled_decode_reads_dense_qam_blind_off_the_senders_grid(M, snr_db):
    """Equal nominal rates, the sender 0.75 ppm fast, the receiver 0.43 of a sample off it, a
    carrier: the family, the order and the payload read blind.  Before the band read, the one gain
    and the equal-rates hypothesis, such captures read as differential or as 4-QAM at BER ~0.5
    (simulated bench, 22 dB)."""
    bits, tx = _qam_stream(M, bursts=1, rows=60, seed=M)
    tx = np.concatenate([tx, np.zeros(400, complex)])
    y = at_rate(tx, 1.0 + 0.75e-6, tx.size - 64, delay=-37.43, carrier=5e-4)
    rng = np.random.default_rng(M)
    p = np.mean(np.abs(y[100:2000]) ** 2)
    y = y + (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) \
        * np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    got, d = wf.decode_scaled(y, SEED, (64,), carrier_span=0.3)
    assert d.payload == "coherent" and d.M == M
    n = min(got.size, bits[0].size)
    assert n >= bits[0].size - L * int(np.log2(M))            # every payload row but a short last
    assert float(np.mean(got[:n] != bits[0][:n])) < 5e-3


def test_the_joint_settle_reads_a_sender_a_hair_wider_than_the_receiver():
    """A sender 4 ppm fast at one nominal rate, 300 rows: settled on every row, from scale 1 and a
    start 0.02 of a sample off, it reads above 1 (measured +4.02 ppm, the start within 1e-4)."""
    bits = np.random.default_rng(2).integers(0, 2, 2 * 300 * L)
    tx = wf.compose(bits, L, L, SEED, M=4)
    rho = 1.0 + 4e-6
    y = at_rate(tx, rho, tx.size - 200, delay=0.31)
    s0 = row_start(0, L, rho, 0.31) + L / rho                   # the first row held whole
    T = int((y.size - s0 - 2) // (L / rho)) - 1
    s, r, c = wf._settle_scaled(y, L, SEED, None, s0 + 0.02, 1.0, 0.0, T, 2.0 / (T * L) * 1e6, 0.05,
                                joint=True)
    assert r > 1.0
    assert abs(r / rho - 1.0) < 0.5e-6
    assert abs(s - s0) < 0.01
