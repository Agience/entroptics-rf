"""
Streaming: one unbroken screen, the lock paid once, arbitrary buffer boundaries.

A continuous link is not a sequence of captures, and the failures that matter are the ones at the
seams -- a buffer that ends mid-row, a capture that starts mid-row, a receiver that re-acquires
when it should be holding.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import Link, StreamReceiver, StreamTransmitter, waveform as wf

LINK = Link(K=64, L=64, seed=1234, M=4)


def _stream(link, chunk_sizes, bit_seed=0):
    rng = np.random.default_rng(bit_seed)
    chunks = [rng.integers(0, 2, n) for n in chunk_sizes]
    tx = StreamTransmitter(link)
    iq = np.concatenate([tx.send(c) for c in chunks])
    return np.concatenate(chunks), iq


def _drain(rx, iq, prime, step):
    got = [rx.feed()]
    for i in range(prime, len(iq), step):
        got.append(rx.feed(iq[i:i + step]))
    return np.concatenate(got)


def test_continuous_stream_is_bit_exact_across_irregular_buffers():
    """The headline: push arbitrary chunk sizes in, pull the same bits out."""
    sent, iq = _stream(LINK, (5000, 137, 90000, 4321, 60000))
    rx = StreamReceiver(LINK)
    assert rx.acquire(iq[:40000])
    got = _drain(rx, iq, 40000, 17000)
    n = min(sent.size, got.size)
    assert n > 100_000
    assert np.mean(sent[:n] != got[:n]) == 0.0


@pytest.mark.parametrize("step", [1, 63, 64, 65, 1000, 4096])
def test_buffer_boundaries_never_lose_a_row(step):
    """Rows straddle buffers whenever the step is not a multiple of L, which is the normal case.
    The receiver carries the partial row and the previous row's coordinates; if either were
    dropped, the seam rows would decode as noise."""
    sent, iq = _stream(LINK, (64 * 2 * 800,))
    rx = StreamReceiver(LINK)
    assert rx.acquire(iq[:40000])
    got = _drain(rx, iq, 40000, step)
    n = min(sent.size, got.size)
    assert np.mean(sent[:n] != got[:n]) == 0.0


def test_the_stream_spends_one_reference_row_in_total_not_one_per_buffer():
    """That is the whole reason streaming is a mode of its own: at 800 rows the overhead is
    0.125%, and it does not grow with the number of buffers the caller happens to use."""
    n_bits = 64 * 2 * 800
    sent, iq = _stream(LINK, (n_bits,))
    _, iq_many = _stream(LINK, tuple([n_bits // 8] * 8))
    assert len(iq) == len(iq_many), "buffering changed the airtime; the seam is not free"
    assert len(iq) == (800 + 1) * 64
    assert n_bits / len(iq) > 0.99 * LINK.bits_per_sample


def test_acquire_is_paid_once_and_the_lock_is_held():
    """After acquisition a buffer costs a matrix product, not a sweep.  If ``feed`` re-locked,
    this would be far slower and the held state would be pointless."""
    import time
    sent, iq = _stream(LINK, (64 * 2 * 4000,))
    rx = StreamReceiver(LINK)
    t0 = time.perf_counter(); rx.acquire(iq[:60000]); t_acq = time.perf_counter() - t0
    rx.feed()
    body = iq[60000:]
    t0 = time.perf_counter(); rx.feed(body); t_feed = time.perf_counter() - t0
    assert t_feed < t_acq * 20, f"feed {t_feed:.3f}s vs acquire {t_acq:.3f}s"


def test_stream_recovers_the_row_offset_of_a_capture_that_starts_mid_row():
    """A real capture never starts on a row boundary, and at the wrong offset the read is noise
    (measured BER 0.50).  Acquisition must measure the offset, not assume it."""
    sent, iq = _stream(LINK, (64 * 2 * 900,))
    for cut in (1, 17, 33, 63):
        rx = StreamReceiver(LINK)
        assert rx.acquire(iq[cut:cut + 50000])
        assert rx.offset == (64 - cut) % 64, f"cut {cut}: offset {rx.offset}"


def test_offset_capture_still_decodes_exactly_after_the_lost_first_row():
    """Cutting into the stream costs the partial row it lands in -- and nothing else.  The rest
    decodes bit-exact, one row later."""
    sent, iq = _stream(LINK, (64 * 2 * 900,))
    cut = 17
    rx = StreamReceiver(LINK)
    assert rx.acquire(iq[cut:cut + 50000])
    got = _drain(rx, iq[cut:], 50000, 20000)
    per_row = LINK.K * 2
    n = min(sent.size - per_row, got.size)
    assert np.mean(sent[per_row:per_row + n] != got[:n]) == 0.0


def test_feed_before_acquire_raises_rather_than_returning_garbage():
    with pytest.raises(RuntimeError):
        StreamReceiver(LINK).feed(np.zeros(1000, dtype=np.complex64))


def test_failed_acquisition_reports_itself():
    """Noise must not acquire.

    The assertion compares `acquire`'s answer against False, rather than against `rx.locked`:
    comparing it to the state that produced it is a tautology that passes whatever `acquire`
    decides.  What this pins is that pure noise is refused -- with the significance test disabled
    it acquires as L=61, M=2 (measured on this capture), a geometry nothing rides."""
    noise = (np.random.default_rng(0).standard_normal(30_000)
             + 1j * np.random.default_rng(1).standard_normal(30_000)).astype(np.complex64)
    rx = StreamReceiver(LINK)
    assert rx.acquire(noise, candidates=range(60, 69)) is False
    assert not rx.locked


def test_streaming_refuses_the_coherent_mode_explicitly():
    """The coherent mode needs a reference row per screen, so it is a captured-frame mode.  The
    constructor says so, rather than producing a stream that will not decode."""
    with pytest.raises(ValueError):
        StreamTransmitter(Link(K=64, L=64, seed=1, M=16, coherent=True))


def test_mode_basis_is_cached_and_read_only():
    """The aperture is a pure function of (K, L, seed, nonce), so it is derived once and shared
    rather than re-factorized per buffer, and sharing is only safe read-only.  A QR per buffer
    dominates the decode."""
    a = wf.mode_basis(64, 64, 1234)
    b = wf.mode_basis(64, 64, 1234)
    assert a is b
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0, 0] = 0


def test_streaming_matches_the_one_shot_read():
    """Streaming must not be a different modem -- same bits, same aperture, same answer."""
    sent, iq = _stream(LINK, (64 * 2 * 500,))
    rx = StreamReceiver(LINK)
    assert rx.acquire(iq[:40000])
    streamed = _drain(rx, iq, 40000, 9000)
    one_shot = wf.read(iq, LINK.L, LINK.K, LINK.seed, M=LINK.M)
    n = min(streamed.size, one_shot.size)
    assert np.array_equal(streamed[:n], one_shot[:n])


def test_acquire_sizes_itself_and_needs_no_window_from_the_caller():
    """The last pre-declared parameter, removed.

    A caller handing `acquire` a slice was choosing a sample count.  Six attempts were needed
    because two separate things had to be understood: the candidate range must be derived from the
    whole buffer (a short prefix truncates discover's isqrt width search and the read returns
    whatever sits at the top of what is left), and a row0 of 0 must be distinguishable from "the
    alignment has not resolved yet" (it needs ~64 rows where the geometry needs 4, so a
    stop-when-it-repeats rule otherwise stops on the run of defaults).

    This hands over the whole buffer and checks the alignment is right at depth -- which is what
    fails if either understanding is lost."""
    sent, iq = _stream(LINK, (64 * 2 * 2000,))
    for join in (0, 37, 250, 1500):
        rx = StreamReceiver(LINK)
        assert rx.acquire(iq[join * 64:]), f"no lock joining at row {join}"
        assert rx.lock.L == 64 and rx.lock.K == 64
        assert rx.lock.aligned, f"claimed a lock without an alignment at row {join}"
        assert rx.lock.row0 == join, f"joined at {join}, aligned to {rx.lock.row0}"


def test_a_row_zero_that_is_a_default_is_not_reported_as_an_alignment():
    """`aligned` exists because row0 = 0 is two different answers.

    Below the rows the alignment needs it returns 0, which is also what a capture beginning on row
    0 returns.  Conflating them cost a stream that joined at row 1500 its whole payload: correct
    geometry, BER 0.50."""
    sent, iq = _stream(LINK, (64 * 2 * 800,))
    body = np.asarray(iq[250 * 64:])
    short = wf._alphabet(body, 64, 64, LINK.seed, None, 0, 16, 0.0, 0.0,
                         depth=int(wf.JOIN_SEARCH_ROWS))
    long = wf._alphabet(body, 64, 64, LINK.seed, None, 0, 128, 0.0, 0.0,
                        depth=int(wf.JOIN_SEARCH_ROWS))
    assert short[3] == 0 and short[4] is False, "too few rows should not claim an alignment"
    assert long[3] == 250 and long[4] is True, "enough rows should find the true row and say so"


def test_acquire_returns_false_when_the_stream_has_not_carried_enough():
    """The documented streaming shape, never tested.

    `acquire` promises to return False rather than raise when a receiver has joined a live stream
    and not yet heard enough of it, so the caller feeds it again.  A receiver that raised instead
    would have to be wrapped in a try by every caller, and one that returned True would decode
    noise."""
    rx = StreamReceiver(LINK)
    noise = (np.random.default_rng(0).standard_normal(9000)
             + 1j * np.random.default_rng(1).standard_normal(9000)).astype(np.complex64)
    assert rx.acquire(noise, candidates=range(60, 69)) is False
    assert not rx.locked


def test_feeding_before_a_lock_is_an_error_not_an_empty_answer():
    """Silence would let a caller stream for hours and wonder where the bits went."""
    rx = StreamReceiver(LINK)
    with pytest.raises(RuntimeError, match="acquire"):
        rx.feed(np.zeros(64, dtype=np.complex64))
