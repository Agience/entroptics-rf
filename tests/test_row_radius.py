"""
The per-row power: every row goes out at its own radius, ``r_t^2 ~ Gamma(L, 1/L)``.

Every mode is lit (K = L), so without it every row carries energy L exactly -- unit-modulus
coordinates on an orthonormal aperture -- and a key-free observer finds the row period from the
power envelope alone (research/row_energy_leak.md).  With it a row's energy is exactly
``L r_t^2``, so the radius is all that keeps the envelope white.  These tests pin three things:
the rows' energies are distributed as white noise's, the energy detector that found the period
reads noise level on a keyed burst, and both reads -- differential and coherent, keyed and open --
still return every bit.  The receiver of an open broadcast never learns the sender's radius.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import StreamTransmitter, Link, waveform as wf

K, L, SEED = 64, 64, 1234                                    # every mode lit


def _burst(seed, rows=3000, rng=1):
    bits = np.random.default_rng(0).integers(0, 2, 2 * K * rows)
    return np.asarray(wf.compose(bits, K, L, seed, M=4, rng=rng), dtype=complex)


def _window_dispersion(x, q, offsets):
    """``q * var(e) / mean(e)^2`` of the q-sample window energies at each offset.  White noise
    reads 1 at every offset (its window energy is Gamma(q)); equal-energy rows read 0 at the row
    boundary."""
    p = np.abs(x) ** 2
    c = np.concatenate([[0.0], np.cumsum(p)])
    out = []
    for o in offsets:
        T = (p.size - o) // q
        e = c[o + q * np.arange(1, T + 1)] - c[o + q * np.arange(T)]
        out.append(q * e.var() / e.mean() ** 2)
    return np.array(out)


def _contrast(x, q):
    """The offset contrast of the window-energy variance, ``min_o var / median_o var`` over every
    integer offset: the lab's D1 contrast (research/row_energy_leak.md), the read that ranked the
    true width first of 585 at 0 dB.  A white envelope has the same variance at every offset."""
    v = _window_dispersion(x, q, range(q))
    return float(v.min() / np.median(v))


def test_the_radius_is_gamma_L_seekable_and_keyed():
    r = wf.row_radius(L, SEED, rows=4096)
    r2 = r ** 2
    assert abs(r2.mean() - 1) < 4 / np.sqrt(L * r2.size)
    assert abs(r2.var() * L - 1) < 0.1                            # shape L, not K: Var r^2 = 1/L
    assert np.array_equal(wf.row_radius(L, SEED, first=100, rows=77), r[100:177])
    assert np.array_equal(wf.row_radius(L, SEED, first=4000, rows=96), r[4000:4096])
    assert not np.allclose(wf.row_radius(L, SEED + 1, rows=64), r[:64])
    # open: the sender's own draw, from its rng, never a public stream
    a = wf.row_radius(L, None, rows=64, rng=1)
    assert np.array_equal(a, wf.row_radius(L, None, rows=64, rng=1))
    assert not np.allclose(a, wf.row_radius(L, None, rows=64, rng=2))


@pytest.mark.parametrize("seed", [SEED, None])
def test_row_energy_is_distributed_like_white_noise(seed):
    """The note's statistic, cheap: L-sample window energies at offsets across the row.  Plain
    rows read 0 at the boundary; with the radius every offset reads white noise's 1."""
    x = _burst(seed)
    offs = (0, 1, L // 8, L // 4, L // 2)
    got = _window_dispersion(x, L, offs)
    noise = np.array([1, 1j]) @ np.random.default_rng(5).standard_normal((2, x.size))
    ref = _window_dispersion(noise, L, offs)
    assert np.all(np.abs(got - 1) < 0.12), got
    assert np.all(np.abs(ref - 1) < 0.12), ref
    # the leak this closes: the same burst with the radius divided back out
    plain = (x.reshape(-1, L) / wf.row_radius(L, seed, rows=x.size // L, rng=1)[:, None]).ravel()
    assert _window_dispersion(plain, L, (0,))[0] < 1e-6


@pytest.mark.parametrize("seed", [SEED, None])
def test_the_energy_detector_reads_noise_level(seed):
    """D1 contrast at the true width, scored against noise-only captures of the same length: the
    keyed (and open) burst sits in the noise range, while the plain one stands far out."""
    x = _burst(seed, rows=3125)                                  # 200 000 samples, noise-free
    rng = np.random.default_rng(7)
    null = np.array([_contrast(rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size), L)
                     for _ in range(8)])
    z = lambda s: (null.mean() - s) / null.std()                 # a deficit counts as structure
    plain = (x.reshape(-1, L) / wf.row_radius(L, seed, rows=x.size // L, rng=1)[:, None]).ravel()
    assert z(_contrast(plain, L)) > 50
    assert abs(z(_contrast(x, L))) < 4


@pytest.mark.parametrize("seed", [SEED, None])
def test_the_differential_read_needs_no_radius(seed):
    """Blind, keyed and open: the differential is phase-only, so the radius never enters."""
    bits = np.random.default_rng(3).integers(0, 2, 2 * K * 200)
    got, d = wf.decode(wf.compose(bits, K, L, seed, M=4), seed, range(56, 72))
    assert (d.L, d.K, d.M) == (L, K, 4)
    assert np.array_equal(got[:bits.size], bits)


@pytest.mark.parametrize("M", [64, 1024])
def test_the_keyed_coherent_read_divides_the_radius_out(M):
    """Keyed, the reader knows r_t / r_0 from the key and takes it off exactly: noiseless 1024-QAM
    is bit-exact.  Left on, the radius is a ~6% gain error per row, past half of 1024-QAM's step."""
    bits = np.random.default_rng(4).integers(0, 2, int(np.log2(M)) * K * 100)
    iq = wf.compose_qam(bits, K, L, SEED, M=M)
    assert np.array_equal(wf.read_qam(iq, L, K, SEED, M=M)[:bits.size], bits)
    A = wf._coherent_project(np.asarray(iq), L, K, SEED, 0, 0.0, 0.0)
    r = wf.row_radius(L, SEED, rows=A.shape[0])
    left_on = wf._dekeyed(A, 0, K, SEED, 0.0, L=L) * (r[1:] / r[0])[:, None]
    if M == 1024:
        assert np.mean(wf._qam_bits(left_on, M)[:bits.size] != bits) > 1e-3


@pytest.mark.parametrize("M", [16, 64, 256])
def test_the_open_coherent_read_measures_each_rows_level(M):
    """Open, the radius is the sender's own draw: the reader levels each row by its own power and
    fits it to the lattice.  Noiseless, informed and blind, every bit returns.

    Blind open 16-QAM is left out: :func:`decode` names it differential (closure 0.50 against
    0.36) with or without the radius, because the open mode's identity mode keys let a QAM
    differential close on 16-PSK.  That predates the radius and is not this test's subject."""
    bits = np.random.default_rng(5).integers(0, 2, int(np.log2(M)) * K * 120)
    iq = wf.compose_qam(bits, K, L, None, M=M)
    assert np.array_equal(wf.read_qam(iq, L, K, None, M=M)[:bits.size], bits)
    if M == 16:
        return
    got, d = wf.decode(iq, None, range(56, 72))
    assert d.payload == "coherent" and d.M == M
    assert np.array_equal(got[:bits.size], bits)


def test_a_stream_carries_the_radii_a_compose_would():
    """StreamTransmitter indexes the radius by the absolute screen row across buffers, the
    reference being row 0, so a stream and a one-shot compose put the same powers on the air:
    with every mode lit, row t carries exactly ``L r_t^2``."""
    link = Link(K=K, L=L, seed=SEED, M=4)
    tx = StreamTransmitter(link)
    rng = np.random.default_rng(6)
    iq = np.concatenate([tx.send(rng.integers(0, 2, n)) for n in (2 * K * 70, 2 * K * 3 + 5,
                                                                    2 * K * 90)])
    e = np.sum(np.abs(iq.reshape(-1, L)) ** 2, axis=1)
    r = wf.row_radius(L, SEED, rows=e.size)
    assert np.allclose(e / L, r ** 2, rtol=1e-4)
