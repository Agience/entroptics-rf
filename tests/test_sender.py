"""The sender mark: a per-mode rotation, read back and identified against a null.

Row 0 carries the keyed reference row as well as the sender rotation, so the two are entangled at
the point of reading and the tests here pin that the reference is divided back out.  Read without
that step the true sender scores below strangers -- 0.043 against 0.203 at K = L = 64, noiseless
-- which is the failure these tests make visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf

K, L, SEED = 64, 64, 1234
SENDERS = ["alice", "bob", "carol", "dave", "erin"]


def _sent(sender, snr=None, seed=SEED):
    bits = np.random.default_rng(1).integers(0, 2, 2 * 400 * K)
    iq = wf.compose(bits, K, L, seed, M=4, sender=sender)
    if snr is None:
        return iq
    from entroptics_rf.radio import LoopbackRadio
    r = LoopbackRadio(snr_db=snr, seed=7)
    r.transmit(iq)
    return r.receive(len(iq)).iq


@pytest.mark.parametrize("snr", [None, 12.0, 6.0])
def test_the_true_sender_is_the_one_that_matches(snr):
    """Row 0 carries the keyed reference row as well as the sender rotation, so the read divides
    the reference out; without that it identifies strangers."""
    sig = wf.sender_signature(_sent("alice", snr), L, K, SEED)
    scores = {s: wf.sender_match(sig, s) for s in SENDERS}
    assert max(scores, key=scores.get) == "alice", f"read {scores}"
    assert scores["alice"] > 0.9, f"the true sender must match strongly, got {scores['alice']:.3f}"
    others = [v for s, v in scores.items() if s != "alice"]
    assert max(others) < 0.5, f"a stranger must not, got {max(others):.3f}"


@pytest.mark.parametrize("snr", [None, 12.0, 6.0])
def test_the_sender_is_resolved_against_a_null_not_a_hand_calibrated_cosine(snr):
    """``sender_resolve`` carries the decision: the exact permutation null of re-pairing the two
    sides, so ``resolved`` is against a cutoff the level implies rather than a number chosen."""
    sig = wf.sender_signature(_sent("alice", snr), L, K, SEED)
    who, c = wf.sender_resolve(sig, SENDERS)
    assert who == "alice", f"resolved {who} with z={c.z:.2f}"
    assert c.resolved and c.z > c.cutoff


def test_a_sender_not_on_the_list_resolves_to_nobody():
    """The right answer for a stranger is None, not the nearest name on the list."""
    sig = wf.sender_signature(_sent("frank"), L, K, SEED)
    who, c = wf.sender_resolve(sig, SENDERS)
    assert who is None, f"claimed {who} at z={c.z:.2f}"


def test_an_unmarked_transmission_carries_no_sender():
    sig = wf.sender_signature(_sent(None), L, K, SEED)
    who, _ = wf.sender_resolve(sig, SENDERS)
    assert who is None


def test_the_wrong_key_reads_no_sender():
    """The mark is readable by anyone holding the aperture, which is what makes it an address
    rather than a secret -- and the aperture is still the key's."""
    sig = wf.sender_signature(_sent("alice"), L, K, SEED + 1)
    who, _ = wf.sender_resolve(sig, SENDERS)
    assert who is None


@pytest.mark.parametrize("snr,worst_ber", [(0.0, 0.25), (-9.0, 0.40)])
def test_the_mark_outlives_the_payload(snr, worst_ber):
    """The address travels about 15 dB further than the message.

    A constant per-mode phase accumulates over every row of the capture; a payload symbol gets one
    row.  So a holder of the aperture can identify a transmitter it cannot read -- which is a
    property worth pinning, because it is the cost of the mark being free, and a transmitter that
    needs to be unidentifiable has to send unmarked rather than rely on range."""
    from entroptics_rf.runner import ber
    bits = np.random.default_rng(1).integers(0, 2, 2 * 400 * K)
    iq = wf.compose(bits, K, L, SEED, M=4, sender="alice")
    from entroptics_rf.radio import LoopbackRadio
    r = LoopbackRadio(snr_db=snr, seed=7)
    r.transmit(iq)
    y = r.receive(len(iq)).iq

    got = wf.read(y, L, K, SEED, M=4)
    n = min(len(bits), len(got))
    assert ber(bits[:n], got[:n]) > worst_ber, "premise: the payload should be failing here"

    who, coupling = wf.sender_resolve(wf.sender_signature(y, L, K, SEED), SENDERS)
    assert who == "alice", f"the mark should still resolve at {snr} dB"
    assert coupling.z > coupling.cutoff
