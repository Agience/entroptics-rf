"""
The coherent QAM mode (rate) and the interference read (situational awareness).

These tests pin the measurements the docs quote for both, including the ones that bound where each
stops working.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import Link, ber, loopback, waveform as wf

K, L, SEED = 64, 64, 1234


def _awgn(x, snr_db, rng):
    p = np.mean(np.abs(x) ** 2)
    n = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) \
        * np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return (x + n).astype(np.complex64)


def _cfo(x, f):
    return (x * np.exp(2j * np.pi * f * np.arange(len(x)))).astype(np.complex64)


def _bits(M, rows=500, seed=0):
    return np.random.default_rng(seed).integers(0, 2, int(round(np.log2(M))) * rows * K)


# ── the QAM alphabet ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [4, 16, 64, 256, 1024])
def test_qam_is_unit_power_and_gray_coded(M):
    """Gray coding is what makes a nearest-neighbour slip cost one bit, which is most of why the
    coherent mode delivers the rate it does."""
    c = wf._qam(M)
    assert c.size == M
    assert np.isclose(np.mean(np.abs(c) ** 2), 1.0)
    s = int(round(np.sqrt(M)))
    half = int(round(np.log2(s)))
    for i in range(s - 1):                       # adjacent points on each axis differ in one bit
        for q in range(s):
            a = (wf._gray(i) << half) | wf._gray(q)
            b = (wf._gray(i + 1) << half) | wf._gray(q)
            assert bin(a ^ b).count("1") == 1


def test_qam_rejects_non_square_orders():
    for M in (2, 8, 32, 128):
        with pytest.raises(ValueError):
            wf._qam(M)


# ── the coherent write/read pair ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [4, 16, 64])
def test_coherent_round_trip_is_bit_exact_on_a_clean_channel(M):
    bits = _bits(M)
    got = wf.read_qam(wf.compose_qam(bits, K, L, SEED, M=M), L, K, SEED, M=M)
    assert np.array_equal(got[:bits.size], bits)


def test_reference_row_is_keyed_not_known_plaintext():
    """An all-ones reference row would put the plain sum of the aperture rows on the air -- one
    linear equation in B, free to anyone listening.  The keyed row must not be constant."""
    r = wf._reference_row(K, SEED)
    assert r.shape == (K,)
    assert np.allclose(np.abs(r), 1.0)
    assert np.std(np.angle(r)) > 0.5
    assert not np.allclose(r, wf._reference_row(K, SEED + 1))


@pytest.mark.parametrize("compose,M", [(wf.compose, 4), (wf.compose_qam, 16)])
def test_neither_mode_puts_a_known_plaintext_row_on_the_air(compose, M):
    """Both modes open on the keyed reference row.

    An opening row of ``A[0] = 1`` would put the plain sum of the aperture rows on the air -- the
    leak :func:`_reference_row` exists to prevent -- and it would be a constant for a given key,
    so an observer holding no key could align two captures to each other by it.
    """
    bits = _bits(M)
    iq = compose(bits, K, L, SEED, M=M)
    first = np.asarray(iq[:L])
    plain = np.ones(K) @ wf.mode_basis(K, L, SEED)          # what an all-ones row would send
    plain = plain / np.sqrt(np.mean(np.abs(plain) ** 2)) * np.sqrt(np.mean(np.abs(first) ** 2))
    cos = abs(np.vdot(plain, first)) / (np.linalg.norm(plain) * np.linalg.norm(first))
    assert cos < 0.5, f"the first row still carries the plain aperture sum (cos={cos:.3f})"

    # and it moves with the key.  It does not move per transmission: freshness would need a
    # carried nonce, and the carrier mode that would hold one is not built (see waveform.py).
    assert not np.allclose(np.asarray(compose(bits, K, L, SEED + 1, M=M)[:L]), first)


def test_coherent_wrong_key_reads_as_noise():
    bits = _bits(16)
    iq = wf.compose_qam(bits, K, L, SEED, M=16)
    got = wf.read_qam(iq, L, K, SEED + 1, M=16)
    assert abs(np.mean(bits != got[:bits.size]) - 0.5) < 0.05


def test_coherent_delivers_more_rate_than_differential():
    """The whole reason the mode exists: more delivered rate at the same SNR.  Measured at K = L = 64
    and 18 dB: 3.78 bits/sample coherent 16-QAM against 2.95 differential 8-PSK, 28% more."""
    def delivered(link, snr):
        sent, rx = loopback(link, _bits(link.M, rows=500), snr_db=snr, seed=3)
        p = min(max(ber(sent, rx.bits), 1e-15), 1 - 1e-15)
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return (K / L) * np.log2(link.M) * (1 - h)

    coh = delivered(Link(K=K, L=L, seed=SEED, M=16, coherent=True), 18)
    dif = delivered(Link(K=K, L=L, seed=SEED, M=8), 18)
    assert coh > dif * 1.2, f"coherent {coh:.2f} vs differential {dif:.2f} bits/sample"

def test_the_lock_still_works_on_a_coherent_waveform():
    """The screen is unchanged, so discovery must not care which alphabet rode on it.

    QAM coordinates are not unit modulus, so the constant-modulus read does not fall to 0 at the
    true geometry as it does for a PSK payload: noiseless 16-QAM reads 0.44 of the null there,
    against ~1.01 one sample out, and the runner-up scores 1.86x the winner.  That is enough for
    the width and the row.  Which payload rides -- and so whether three modes went to a beacon --
    is decode's to settle (it gives the modes back on a coherent capture), so this pins only that
    the lock accounts for all L modes."""
    d = wf.discover(wf.compose_qam(_bits(16, rows=600), K, L, SEED, M=16), SEED, range(40, 140))
    assert d.locked and d.L == L and d.offset == 0
    assert d.K + (wf.BEACON_MODES if d.beacon else 0) == L
    assert d.score < 0.6, f"the true geometry read {d.score:.3f} of the null"
    assert d.margin > 1.3, f"runner-up only {d.margin:.3f}x the winner"


# ── interference ──────────────────────────────────────────────────────────────────────────────

def _with_tone(iq, dbc, f=0.031):
    return (iq + (10 ** (dbc / 20)) * np.exp(2j * np.pi * f * np.arange(len(iq)))).astype(np.complex64)


def test_clean_waveform_shows_no_dominant_mode():
    """The signal is flat by construction -- that is the low-detectability property -- so nothing dominates.
    Measured at K = L = 64: strehl 0.028, dominance 0.013."""
    intf = wf.interference(wf.compose(_bits(4), K, L, SEED), L)
    assert intf.strehl < 0.1 and intf.dominance < 0.1


def test_interference_rises_monotonically_with_a_cw_interferer():
    """No spectrum analyser, no reference capture, no template: any concentration is foreign.

    Measured at K = L = 64, dominance against tone level (dBc): -99 0.013, -13 0.052, -10 0.095,
    -6 0.205, 0 0.503 (strehl 0.028, 0.067, 0.109, 0.217, 0.512)."""
    iq = wf.compose(_bits(4), K, L, SEED)
    d = [wf.interference(_with_tone(iq, x), L).dominance for x in (-99, -13, -10, -6, 0)]
    assert d == sorted(d), f"not monotone in interferer power: {d}"
    assert d[-1] > 10 * d[0], "a 0 dBc interferer should be unmistakable"


def test_interference_is_detected_well_before_the_link_breaks():
    """Worth having only if it warns in time to act.  Measured at -13 dBc, K = L = 64: BER 0.0032
    with dominance 0.052, four times the clean waveform's 0.013."""
    bits = _bits(4)
    iq = wf.compose(bits, K, L, SEED)
    j = _with_tone(iq, -13)
    got = wf.read(j, L, K, SEED)
    assert np.mean(bits != got[:bits.size]) < 0.01              # link still fine
    assert wf.interference(j, L).dominance > 3 * wf.interference(iq, L).dominance   # already visible


def test_discovery_holds_as_far_as_the_read_is_worth_having():
    """The width survives a tone at every level tried; what gives way is the payload, not the lock.

    Measured at K = L = 64 over L +/- 8: the width and row read correctly from -13 dBc to 0 dBc,
    with constant-modulus score 0.11 / 0.21 / 0.43 / 0.63 / 0.80 at -13 / -10 / -6 / -3 / 0 dBc
    and blind-decode BER 0.003 / 0.023 / 0.10 / 0.20 / 0.30 -- the read holds well past the point
    where the payload is worth having."""
    iq = wf.compose(_bits(4), K, L, SEED)
    assert wf.discover(_with_tone(iq, -3), SEED, range(L - 8, L + 9)).L == L
    assert wf.discover(_with_tone(iq, -13), SEED, range(L - 8, L + 9)).L == L


# ── the carrier-recovery ladder (this is where the rate lives) ─────────────────────────────────

@pytest.mark.parametrize("M", [16, 64, 256, 1024])
def test_dense_qam_decodes_error_free_on_a_clean_carrier(M):
    """1024-QAM is the WiFi 6 MCS11 constellation, and it is what carries 9.98 bits/s/Hz here
    (every mode lit, 600 rows plus the reference).  Measured at 45 dB: BER 0 for every order."""
    bits = _bits(M, rows=600)
    iq = wf.compose_qam(bits, K, L, SEED, M=M)
    got = wf.read_qam(_awgn(iq, 45, np.random.default_rng(2)), L, K, SEED, M=M)
    assert np.mean(bits != got[:bits.size]) < 1e-3

def test_spectral_efficiency_beats_the_ofdm_reference_point():
    """802.11ax MCS11 (1024-QAM, rate 5/6, 20 MHz, 1 SS) delivers 7.17 bits/s/Hz.  Uncoded, this
    waveform clears that (measured 9.98 at 45 dB, BER 0); the margin is the overhead OFDM spends on
    cyclic prefix, pilots and guard tones and this does not."""
    T, M = 600, 1024
    bits = _bits(M, rows=T)
    iq = wf.compose_qam(bits, K, L, SEED, M=M)
    got = wf.read_qam(_awgn(iq, 45, np.random.default_rng(4)), L, K, SEED, M=M)
    p = float(np.mean(bits != got[:bits.size]))
    h = 0.0 if p <= 0 else -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    delivered = (K / L) * np.log2(M) * (T / (T + 1)) * (1 - h)
    assert delivered > 7.17, f"{delivered:.2f} bits/s/Hz does not clear 802.11ax MCS11"
