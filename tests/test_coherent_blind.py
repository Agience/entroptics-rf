"""
The coherent payload, read blind: a capture and a key in, QAM bits out.

The receiver is told neither the order M nor the carrier.  The geometry comes from discovery as for
the differential payload; the reference row is located, the row-rate carrier is read off the
payload over every row, and M is the square lattice the de-keyed symbols close on.  See
research/coherent_blind.md for the BER against SNR these pin.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf

K, L, SEED = 64, 64, 1234
F = 3e-4                      # carrier offset, cycles per sample: nu = F * L = 0.0192 per row


def _channel(x, *, lead=23, snr_db=None, seed=0):
    """``lead`` samples before the burst (a non-zero row offset), a carrier offset, AWGN."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(lead, complex), np.asarray(x, complex), np.zeros(41, complex)])
    x = x * np.exp(2j * np.pi * F * np.arange(x.size))
    if snr_db is not None:
        p = np.mean(np.abs(x[lead:lead + 1000]) ** 2)
        x = x + (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) \
            * np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return x.astype(np.complex64)


def _bits(M, rows=300, seed=0):
    return np.random.default_rng(seed).integers(0, 2, int(np.log2(M)) * rows * K)


@pytest.mark.parametrize("M,snr_db", [(16, None), (64, None), (256, None), (1024, None),
                                      (16, 24), (64, 30), (256, 36), (1024, 42)])
def test_blind_decode_reads_the_order_and_the_payload(M, snr_db):
    """Measured with 600 rows at K = L = 64: BER 0 at every order noiseless and for 16- and 64-QAM
    at 24 and 30 dB, 1.3e-5 for 256-QAM at 36 dB and 6.8e-5 for 1024-QAM at 42 dB.  The lock reads
    0.44-0.52 of the constant-modulus null at the true geometry (QAM is not unit modulus) with the
    runner-up 1.42-1.71x the winner.  The bound leaves room for a 300-row capture's own draw."""
    bits = _bits(M, seed=M)
    iq = _channel(wf.compose_qam(bits, K, L, SEED, M=M), snr_db=snr_db, seed=M)
    got, d = wf.decode(iq, SEED, range(56, 72))
    assert d.payload == "coherent"
    assert (d.L, d.K, d.offset, d.M) == (L, K, 23, M)
    n = min(bits.size, got.size)
    assert n == bits.size
    ber = float(np.mean(bits[:n] != got[:n]))
    if snr_db is None:
        assert ber == 0.0
    else:
        assert ber < 1e-3, f"M={M} at {snr_db} dB: BER {ber:.2e}"


def test_the_row_rate_carrier_is_read_off_the_payload():
    """Discovery reads the carrier coarsely with the geometry; the coherent read refines it over
    every row, which is what 1024-QAM needs.  Measured: within 4e-6 of the truth at every order."""
    bits = _bits(256, seed=3)
    iq = _channel(wf.compose_qam(bits, K, L, SEED, M=256), snr_db=36, seed=3)
    _, d = wf.decode(iq, SEED, range(56, 72))
    assert abs(d.carrier - F * L) < 1e-5, f"carrier read {d.carrier!r}, true {F * L}"


def test_the_reference_row_is_found_when_the_capture_starts_before_the_burst():
    """A capture that opens three rows and change before the burst: the reference is located."""
    bits = _bits(64, seed=5)
    iq = _channel(wf.compose_qam(bits, K, L, SEED, M=64), lead=3 * L + 23, snr_db=35, seed=5)
    got, d = wf.decode(iq, SEED, range(56, 72))
    assert (d.payload, d.M, d.reference) == ("coherent", 64, 3)
    assert np.array_equal(got[:bits.size], bits)


def test_coherent_qpsk_is_read_as_coherent():
    bits = _bits(4, seed=6)
    got, d = wf.decode(_channel(wf.compose_qam(bits, K, L, SEED, M=4)), SEED, range(56, 72))
    assert (d.payload, d.M) == ("coherent", 4)
    assert np.array_equal(got[:bits.size], bits)


@pytest.mark.parametrize("snr_db", [None, 12])
def test_a_differential_capture_still_decodes_as_differential(snr_db):
    bits = np.random.default_rng(7).integers(0, 2, 2 * 300 * K)
    iq = _channel(wf.compose(bits, K, L, SEED, M=4), snr_db=snr_db, seed=7)
    got, d = wf.decode(iq, SEED, range(56, 72))
    assert (d.payload, d.M) == ("differential", 4)
    n = min(bits.size, got.size)
    assert np.mean(bits[:n] != got[:n]) < (1e-12 if snr_db is None else 1e-2)


def test_the_lattice_closure_is_signed():
    """A 16-QAM point sits half a step off the 64-QAM lattice, so 64 does not close on 16-QAM --
    which is what lets the order be the argmax rather than a rise weighed between rungs."""
    Y = wf._qam(16)[np.random.default_rng(8).integers(0, 16, 4000)]
    c = [wf._lattice_closure(Y, m) for m in wf.QAM_LADDER]
    assert c[1] == pytest.approx(1.0)
    assert c[2] < 0.05 and c[3] < 0.05
    assert wf._qam_order(Y)[0] == 16


def test_read_qam_removes_the_row_rate_carrier_itself():
    """The informed reader takes the carrier and removes both halves; no caller pre-processing."""
    bits = _bits(1024, seed=9)
    iq = _channel(wf.compose_qam(bits, K, L, SEED, M=1024), lead=0)
    got = wf.read_qam(iq, L, K, SEED, M=1024, carrier=F * L)
    assert np.array_equal(got[:bits.size], bits)
