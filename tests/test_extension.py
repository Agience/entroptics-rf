"""
The per-row cyclic extension: this sender writes none, and the receiver looks for none.

A cyclic extension wraps each row r in ``[r[-P:], r, r[:P]]``.  It bought a receiver at another
sample rate ~2 dB of floor, but a row repeated on the wire is a lag-correlation line a blind
observer finds below the noise (research/extension.md), so ``compose`` does not write one.  The
joint read recovers that floor in the receiver (research/joint_read.md).  So the receiver searches
rows L samples apart only: searching extensions as well cost up to 5x the folds of the scaled
search, and every false lead in its blind-width misses was an extension P > 0.

The price is that a foreign sender that does wrap its rows is not read.  What these tests pin is
that it is not MISread either: at a wrong row stride every row but the first is cut across a
boundary, so no geometry reads constant modulus, and the capture reads the way noise does -- the
constant-modulus read near 1, no margin, an alphabet closure at the noise null -- rather than as a
confident wrong payload.  Measured with the constant-modulus read (K = L = 64): P = 1, 2, 4,
clean and 12 dB, DQPSK and 8-PSK, score 0.970-0.987, margin 0.69-0.91, closure 1.3-3.4 times the
null; pure noise (300 rows) reads 0.979-0.983, 0.83-0.89 and 1.2-3.1.  The same payload sent
plainly reads 3e-15, a margin of 2e14 and 138 times the null.
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from entroptics_rf import waveform as wf

K, L, SEED = 64, 64, 1234


def _bits(M=4, rows=300, seed=0, K=K):
    return np.random.default_rng(seed).integers(0, 2, int(np.log2(M)) * rows * K)


def _foreign(iq, P, L=L):
    """What a sender that does wrap its rows puts on the air: ``[r[-P:], r, r[:P]]`` per row."""
    S = np.asarray(iq).reshape(-1, L)
    return wf.to_iq(S[:, np.arange(-P, L + P) % L])


def _channel(iq, *, cut=37, f=3e-4, snr_db=None, seed=0):
    """A capture that starts ``cut`` samples in, with a carrier offset of ``f`` cycles per sample
    and AWGN."""
    x = np.asarray(iq, dtype=complex)[cut:]
    x = x * np.exp(2j * np.pi * f * np.arange(x.size))
    if snr_db is not None:
        r = np.random.default_rng(seed)
        p = float(np.mean(np.abs(x) ** 2))
        x = x + (r.standard_normal(x.size) + 1j * r.standard_normal(x.size)) \
            * np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return x.astype(np.complex64)


def _reads_as_nothing(d, n):
    """The evidence a reader acts on says no transmission: no constant-modulus geometry, no margin
    over the runner-up, and an alphabet closure within a few times the unkeyed null (a real lock
    reads tens to hundreds of times it; noise averages 2.5)."""
    assert d.score > 0.5, f"a foreign capture read constant modulus: {d.score:.3g}"
    assert d.margin < 1.5, f"a foreign capture claimed a margin of {d.margin:.3g}"
    assert wf.closure_significance(d, n) < 10.0


# -- the wire format ------------------------------------------------------------------------------

def test_this_sender_writes_no_extension():
    """Low detectability over 2 dB: no compose path takes an extension, and a link cannot ask for one."""
    from entroptics_rf.runner import Link
    bits = _bits(rows=4)
    for f in (wf.compose, wf.compose_qam):
        with pytest.raises(TypeError):
            f(bits, K, L, SEED, extension=2)
    with pytest.raises(TypeError):
        Link(K=K, L=L, seed=SEED, extension=2)
    assert wf.compose(bits, K, L, SEED).size % L == 0


def test_the_receiver_searches_no_extension():
    """Nothing in the read path takes, searches or reports a row stride other than L."""
    assert not hasattr(wf, "MAX_EXTENSION")
    assert "extension" not in {f.name for f in dataclasses.fields(wf.Discovery)}
    for f in (wf.screen, wf.read, wf.read_soft, wf.read_qam, wf.coherent_closure,
              wf.sender_signature, wf.constant_modulus_read, wf.capacity, wf.bits_per_sample,
              wf.discover, wf.decode):
        assert "extension" not in inspect.signature(f).parameters, f.__name__


def test_a_plain_capture_decodes_blind():
    """The control for everything below: the plain screen through the same channel locks at
    round-off and the payload is bit-exact."""
    bits = _bits(seed=7)
    y = _channel(wf.compose(bits, K, L, SEED))
    got, d = wf.decode(y, SEED)
    assert (d.L, d.K, d.M, d.offset) == (L, K, 4, L - 37)
    assert d.score < 1e-12 and wf.closure_significance(d, y.size) > 100
    want = bits[2 * K:]
    n = min(want.size, got.size)
    assert n == want.size and np.array_equal(got[:n], want[:n])


# -- a foreign extension is declined, not misread -------------------------------------------------

@pytest.mark.parametrize("P, M, snr_db", [(1, 4, None), (4, 8, None), (2, 4, 12)])
def test_a_foreign_extension_reads_as_no_transmission(P, M, snr_db):
    """A differential sender that wraps its rows, at equal rates through a cut, a carrier and
    (once) noise: no geometry reads constant modulus, so discovery reports what it reports on
    noise."""
    bits = _bits(M=M, seed=10 * P + M)
    y = _channel(_foreign(wf.compose(bits, K, L, SEED, M=M), P), snr_db=snr_db, seed=P)
    _reads_as_nothing(wf.discover(y, SEED), y.size)


def test_a_foreign_extension_on_a_coherent_payload_reads_as_no_transmission():
    """The coherent payload is named by its closure after discovery, so it inherits the decline:
    a 64-QAM sender with P=2 does not come back as a coherent lock."""
    bits = _bits(M=64, seed=5)
    iq = np.concatenate([np.zeros(23, complex),
                         _foreign(wf.compose_qam(bits, K, L, SEED, M=64), 2)])
    y = _channel(iq, cut=0)
    _, d = wf.decode(y, SEED)
    _reads_as_nothing(d, y.size)
    assert d.lattice < 0.5


def test_a_stream_does_not_acquire_a_foreign_extension():
    """The streaming receiver tests the closure against the null before it locks, so a stream of
    wrapped rows is not acquired; the same stream unwrapped is."""
    from entroptics_rf.runner import Link, StreamReceiver, StreamTransmitter
    link = Link(K=K, L=L, seed=SEED, M=4)
    bits = _bits(rows=400, seed=11)
    tx = StreamTransmitter(link)
    iq = np.concatenate([tx.send(bits[i:i + 3000]) for i in range(0, bits.size, 3000)])
    assert StreamReceiver(Link(K=K, L=L, seed=SEED, M=4)).acquire(iq[17:])
    assert not StreamReceiver(Link(K=K, L=L, seed=SEED, M=4)).acquire(_foreign(iq, 2)[17:])
