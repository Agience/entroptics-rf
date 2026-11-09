"""The row beacon: what a receiver joining a stream reads instead of guessing.

The payload's rotation is indexed by the absolute row, so a receiver that joins late has the rows
but not their numbers, and must search for the one it landed on -- a search whose cost grows with
how long the stream has been running.  The beacon carries the number.
"""
import numpy as np
import pytest

from entroptics_rf import waveform as wf

L, SEED = 64, 1234
K = L - wf.BEACON_MODES       # every mode is lit: the payload rides on what the beacon leaves


def _payload(rows, rng=1):
    return np.random.default_rng(rng).integers(0, 2, 2 * rows * K)


def test_the_beacon_names_the_row_it_rides_on():
    """A beacon frame names its own absolute row, at any depth and with no search.

    The index is carried in full within one frame, so nothing accumulates across frames and a
    fragment holding one whole frame is enough."""
    for t0 in (0, 1, 37, 5000, 1234567):
        d = wf._beacon_symbols(t0, 3 * wf.BEACON_FRAME, SEED)
        got = wf.read_beacon(d, SEED)
        assert got is not None, f"a capture of three frames must read at t0={t0}"
        assert got[0] == t0, f"beacon named row {got[0]} riding row {t0}"
        assert got[1] == pytest.approx(1.0), "an unchannelled beacon must agree exactly"


def test_a_capture_shorter_than_a_frame_reads_nothing():
    """Under one whole frame there is no index to read, and that is reported as no read.

    Returning a row anyway would be inventing one, which is the failure the search already
    avoids."""
    d = wf._beacon_symbols(100, wf.BEACON_FRAME - 1, SEED)
    assert wf.read_beacon(d, SEED) is None


def test_the_beacon_is_keyed():
    """Without the key the beacon does not read, and says so through its agreement.

    The repetitions of the index agree only when the rotation comes off correctly, so a wrong key
    scatters them.  That agreement is what separates a beacon from no beacon, with no level: it
    is compared against the other candidates rather than against a constant."""
    d = wf._beacon_symbols(5000, 3 * wf.BEACON_FRAME, SEED)
    right = wf.read_beacon(d, SEED)
    wrong = wf.read_beacon(d, SEED + 1)
    assert right[0] == 5000 and right[1] == pytest.approx(1.0)
    assert wrong[0] != 5000, "a wrong key must not recover the row"
    assert wrong[1] < right[1], (
        f"a wrong key must agree less than the right one, {wrong[1]:.3f} against {right[1]:.3f}")


def test_a_beaconed_stream_decodes_blind_from_past_the_search_reach():
    """The receiver is told nothing: it measures that a beacon rides and reads the row from it.

    The cut is deeper than `JOIN_SEARCH_ROWS`, so the alignment search cannot reach it.  What the
    beacon buys is exactly this -- a join whose cost is the frame rather than the age of the
    stream."""
    deep = wf.JOIN_SEARCH_ROWS + 900
    bits = _payload(deep + 300)
    iq = np.asarray(wf.compose(bits, K, L, SEED, M=4, beacon=True))

    got, d = wf.decode(iq[deep * L:], SEED)
    assert d.beacon, "a beaconed capture must measure as beaconed"
    assert d.K == K, f"the payload rides above the beacon, expected K={K} got {d.K}"
    assert d.row0 == deep, f"the beacon names row {deep}, read {d.row0}"
    want = bits[deep * K * 2:]
    n = min(len(want), len(got))
    assert n and np.array_equal(got[:n], want[:n])


def test_an_unbeaconed_stream_still_reads_as_unbeaconed():
    """A capture with no beacon measures as having none, and loses no modes to one.

    The two hypotheses are compared by closure, so this pins that the comparison does not invent
    a beacon on a capture that carries none."""
    bits = np.random.default_rng(2).integers(0, 2, 2 * 400 * L)     # every mode is payload: K = L
    iq = np.asarray(wf.compose(bits, L, L, SEED, M=4))
    got, d = wf.decode(iq, SEED)
    assert not d.beacon, "a capture with no beacon must not measure as beaconed"
    assert d.K == L, f"every mode is payload here, expected {L} got {d.K}"
    n = min(len(bits), len(got))
    assert np.array_equal(got[:n], bits[:n])


def test_the_beacon_outlives_the_payload_it_serves():
    """The row survives noise the payload does not, which is the order it has to be.

    A beacon read wrong costs the whole payload, since the row is the keystream, so it carries a
    32-bit index three times over BPSK where the payload carries QPSK once.  Measured, the row is
    still exact at 3 dB, where the payload (K = L - 3 = 61, informed read) reads BER 0.23 and is
    not usable uncoded."""
    bits = _payload(1600)
    iq = np.asarray(wf.compose(bits, K, L, SEED, M=4, beacon=True))
    join = 1000
    frag = iq[join * L:(join + 4 * wf.BEACON_FRAME) * L]
    power = float(np.mean(np.abs(frag) ** 2))

    for snr_db in (12, 6, 3):
        sigma = np.sqrt(power / (2 * 10 ** (snr_db / 10)))
        rng = np.random.default_rng(5)
        noisy = frag + (rng.standard_normal(len(frag))
                        + 1j * rng.standard_normal(len(frag))) * sigma
        S = wf.screen(noisy, L)
        A = wf.project(S, wf.mode_basis(K + wf.BEACON_MODES, L, SEED, None))
        d = A[1:] * np.conj(A[:-1])
        got = wf.read_beacon(d[:, :wf.BEACON_MODES], SEED)
        assert got is not None and got[0] == join, (
            f"at {snr_db} dB the beacon read row {got[0] if got else None}, riding {join}")


def test_a_beaconed_screen_needs_room_for_the_beacon():
    """The beacon takes modes from the payload, and running out of them is an error, not a shrink.

    Every mode is lit, so a beaconed screen carries exactly ``L - BEACON_MODES`` payload modes.
    More would quietly drop payload modes the caller asked for; fewer would leave a mode dark, a
    noise-only direction an observer without the key reads the width from.  Both are refused."""
    for k in (L - 1, L, L - wf.BEACON_MODES - 1):
        with pytest.raises(ValueError, match="every mode is lit"):
            wf.compose(_payload(8), k, L, SEED, M=4, beacon=True)
    iq = wf.compose(_payload(8), L - wf.BEACON_MODES, L, SEED, M=4, beacon=True)
    assert len(iq) % L == 0


def test_a_beaconed_transmission_carries_its_sender_mark():
    """A beacon and a sender mark together: the mark rotates every mode the screen uses, the
    beacon's included, so the burst names its sender and still decodes blind."""
    bits = _payload(200, rng=3)
    iq = np.asarray(wf.compose(bits, K, L, SEED, M=4, beacon=True, sender=b"A" * 8))
    got, d = wf.decode(iq, SEED)
    assert d.beacon and d.K == K
    n = min(len(bits), len(got))
    assert n and np.array_equal(got[:n], bits[:n])
    sig = wf.sender_signature(iq, L, K + wf.BEACON_MODES, SEED)
    assert wf.sender_match(sig, b"A" * 8) > 0.99
    assert wf.sender_match(sig, b"B" * 8) < 0.2
