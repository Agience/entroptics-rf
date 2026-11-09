"""
The audit: the receiver is given the key, and nothing else.

This is the property the codec stands on.  A transmission is contiguous data that anyone who knows
the mathematics can read; if it is secured, the one thing agreed in advance is the key, and for an
open broadcast not even that.  Every other quantity -- how wide the screen is, where a row starts,
which alphabet rides it -- is measured from the signal.  How many modes carry payload is not a
quantity at all: every mode is lit (K = L, a property of the wire format rather than of a link),
so there is nothing about it to agree or to measure.

A parameter the receiver must be told is a parameter the codec failed to measure.  These tests
fail if one appears, including by signature: they inspect what reception requires, so a new
argument cannot be added quietly.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from entroptics_rf import waveform as wf

GEOMETRIES = [(64, 64, 4), (64, 64, 2), (96, 96, 4), (48, 48, 8), (128, 128, 4)]


def _sent(L, K, M, seed, cut=0, rows=300, bit_seed=0):
    bits = np.random.default_rng(bit_seed).integers(0, 2, int(np.log2(M)) * rows * K)
    return bits, wf.compose(bits, K, L, seed, M=M)[cut:]


def _payload_after_first_row(bits, K, M):
    """A capture that starts mid-row loses the row it landed in, and nothing else."""
    return bits[K * int(np.log2(M)):]


# ── the signature audit ───────────────────────────────────────────────────────────────────────

def test_reception_takes_only_a_capture_and_a_key():
    """Mechanical, so it cannot rot: the receiver's required arguments are the capture and the
    key.  Everything else is optional, and optional here means a search bound or a cost knob,
    never a parameter of the transmission."""
    sig = inspect.signature(wf.decode)
    required = [n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    assert required == ["iq"], f"reception requires more than a capture: {required}"
    assert sig.parameters["seed"].default is None, "the key must default to the open broadcast"


def test_discovery_takes_only_a_capture_and_a_key():
    sig = inspect.signature(wf.discover)
    required = [n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    assert required == ["iq"]


def test_no_receive_path_carries_an_undeclared_knob():
    """The two audits above check what reception *requires*, and an optional argument is
    invisible to both -- yet an optional argument on a receive path is still a knob a caller can
    turn, and one that nothing reads is a phantom setting no other guard in this file would catch.

    So the optional arguments are pinned as a set.  Each is the key or a search bound with its
    provenance in PAPER §RF-9.1, and adding another is then a deliberate act rather than a
    drift."""
    declared = {"seed", "candidates", "carrier_span", "join_rows"}
    for name in ("decode", "decode_scaled", "discover"):
        sig = inspect.signature(getattr(wf, name))
        optional = {n for n, p in sig.parameters.items()
                    if p.default is not inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        assert optional == declared, (
            f"{name} takes {sorted(optional)}, declared set is {sorted(declared)}: a new "
            "optional argument on a receive path needs a provenance entry in PAPER §RF-9.1")


def test_every_declared_bound_is_actually_read():
    """A bound that nothing reads is worse than a tuned one: it reports a setting the read does
    not have.  Each declared bound is therefore shown to *change something* when it changes."""
    _, iq = _sent(64, 64, 4, 1234)
    base = wf.discover(iq, 1234)
    assert base.L == 64

    # candidates: a range excluding the true width cannot return it
    assert wf.discover(iq, 1234, range(70, 90)).L != 64

    # carrier_span: a carrier outside the span is not found, inside it is.  Read over widths
    # around the truth: the default sweep gives the same answers (61 and 64) at twice the cost.
    off = (iq * np.exp(2j * np.pi * (3.0 / 64) * np.arange(len(iq)))).astype(np.complex64)
    assert wf.discover(off, 1234, range(56, 72), carrier_span=1.0).L != 64
    assert wf.discover(off, 1234, range(56, 72), carrier_span=4.0).L == 64

    # join_rows: a join past the reach is not aligned, one inside it is
    _, deep = _sent(64, 64, 4, 1234, rows=900, bit_seed=1)
    cut = deep[600 * 64:]
    assert wf.discover(cut, 1234, join_rows=64).row0 != 600
    assert wf.discover(cut, 1234, join_rows=1024).row0 == 600


# ── every parameter is measured, not agreed ───────────────────────────────────────────────────

@pytest.mark.parametrize("L,K,M", GEOMETRIES)
@pytest.mark.parametrize("cut", [0, 17])
def test_every_parameter_is_measured(L, K, M, cut):
    """The receiver is handed a capture and a key.  It must work out the rest."""
    bits, iq = _sent(L, K, M, 1234, cut=cut)
    d = wf.discover(iq, 1234, range(40, 140))
    assert d.locked
    assert d.L == L, f"width: read {d.L}, sent {L}"
    assert d.offset == (L - cut) % L, f"offset: read {d.offset}"
    assert d.K == K == L, f"mode count: read {d.K}, sent {K}; every mode is lit"
    assert d.M == M, f"alphabet: read {d.M}, sent {M}"


@pytest.mark.parametrize("L,K,M", GEOMETRIES)
def test_decode_end_to_end_with_nothing_but_the_key(L, K, M):
    bits, iq = _sent(L, K, M, 1234, cut=17)
    got, d = wf.decode(iq, 1234, range(40, 140))
    want = _payload_after_first_row(bits, K, M)
    n = min(want.size, got.size)
    assert n > 0 and np.array_equal(got[:n], want[:n])


# ── the open broadcast needs nothing at all ───────────────────────────────────────────────────

@pytest.mark.parametrize("L,K,M", GEOMETRIES)
def test_public_broadcast_needs_no_prior_knowledge_whatsoever(L, K, M):
    """``seed=None`` is the zero state -- the aperture anyone can derive from the mathematics.
    With it the receiver is handed a capture and nothing else."""
    bits, iq = _sent(L, K, M, None, cut=17)
    got, d = wf.decode(iq)                      # no key, no geometry, no alphabet
    assert (d.L, d.K, d.M, d.offset) == (L, K, M, (L - 17) % L)
    want = _payload_after_first_row(bits, K, M)
    n = min(want.size, got.size)
    assert np.array_equal(got[:n], want[:n])


def test_the_zero_state_is_derivable_not_published():
    """The public aperture is the keyed construction with an empty seed field: a receiver derives
    it from the spec; it is not a table anyone had to be given.  Pin that it is not some ordinary
    seed value wearing a disguise -- every real seed encodes to at least one tagged byte."""
    a = wf.mode_basis(58, 64, None)
    assert np.array_equal(a, wf.mode_basis(58, 64, None))
    for imposter in (0, 1, -1, 2, b"", b"\x00", b"public"):
        assert not np.allclose(a, wf.mode_basis(58, 64, imposter))


# ── the key is the only thing that can be agreed ──────────────────────────────────────────────

def test_the_key_is_the_only_secret_and_it_is_sufficient()  :
    """A holder of the key reads the payload; a non-holder measures the same geometry but reads
    noise.  So the key protects the content and nothing else has to be arranged."""
    bits, iq = _sent(64, 64, 4, 1234)
    right, _ = wf.decode(iq, 1234, range(56, 72))
    wrong, dw = wf.decode(iq, 9999, range(56, 72))
    n = min(bits.size, right.size, wrong.size)
    assert np.array_equal(right[:n], bits[:n])
    assert abs(np.mean(bits[:n] != wrong[:n]) - 0.5) < 0.05


def test_a_wrong_key_cannot_even_find_the_screen():
    """Discovery reads the aperture, so without the key there is no geometry to find either --
    the transmission does not announce its own shape to a stranger."""
    _, iq = _sent(64, 64, 4, 1234)
    good = wf.discover(iq, 1234, range(56, 72))
    bad = wf.discover(iq, 9999, range(56, 72))
    assert good.score < 1e-9 < bad.score
    assert good.margin > 1e6 > bad.margin


# ── the reads that make it possible ───────────────────────────────────────────────────────────

def test_the_aperture_is_nested_which_is_what_lets_the_final_row_be_short():
    """Every row lights all L modes except a message's last, which carries what is left on the
    first few.  A per-K draw would make that short row an unrelated basis the receiver would have
    to be told about.  Nesting is a wire-format property: the K-mode aperture is the first K rows
    of the L-mode one, so the short row is read on the aperture the receiver already has."""
    full = wf.mode_basis(64, 64, 1234)
    for K in (1, 7, 40, 58, 63):
        assert np.array_equal(wf.mode_basis(K, 64, 1234), full[:K])


@pytest.mark.parametrize("snr_db", [None, 12, 6, 3])
def test_the_mode_count_is_the_width_not_a_read(snr_db):
    """There is no mode count to read: every mode is lit, so discovery reports K = L at every SNR
    and no edge in the mode energies is looked for.  A count read from the data would be a
    decision the receiver could get wrong; a count fixed by the wire format is neither agreed per
    link nor measured, so it leaves the key as the only shared thing.  On a flat profile the old
    energy-gap read invented an edge (L - 1, margin ~2, rank_leak.md), which is why it is gone
    rather than kept as a check."""
    _, iq = _sent(64, 64, 4, 1234)
    if snr_db is not None:
        r = np.random.default_rng(snr_db)
        p = np.mean(np.abs(iq) ** 2)
        iq = (iq + (r.standard_normal(len(iq)) + 1j * r.standard_normal(len(iq)))
              * np.sqrt(p / (2 * 10 ** (snr_db / 10)))).astype(np.complex64)
    d = wf.discover(iq, 1234, range(56, 72))
    assert (d.L, d.offset, d.K) == (64, 0, 64), f"read L={d.L} offset={d.offset} K={d.K}"


@pytest.mark.parametrize("M", [2, 4, 8])
def test_the_alphabet_is_read_by_closure(M):
    """Raising a differential symbol to the power M lands every point of an M-PSK alphabet on one
    place.  It closes at the true M and at every multiple, so the answer is the smallest closure."""
    _, iq = _sent(64, 64, M, 1234)
    got, closure, redundancy, _row, _al, _k, _b = wf._alphabet(iq, 64, 64, 1234, None, 0, 16)
    assert got == M and closure > 0.5
    assert redundancy < 0.05, "random payload should exercise the alphabet evenly"


def test_a_coherent_payload_closes_at_no_alphabet():
    """Which is how the payload mode announces itself rather than being agreed: a coherent QAM
    transmission has no differential alphabet to close, and says so."""
    bits = np.random.default_rng(0).integers(0, 2, 4 * 300 * 64)
    iq = wf.compose_qam(bits, 64, 64, 1234, M=16)
    _, closure, *_ = wf._alphabet(iq, 64, 64, 1234, None, 0, 16)
    _, diff_closure, *_ = wf._alphabet(
        wf.compose(bits, 64, 64, 1234, M=4), 64, 64, 1234, None, 0, 16)
    assert closure < diff_closure


def test_a_skewed_source_is_reported_not_hidden():
    """A source that only ever sends two of four symbols is genuinely indistinguishable from a
    two-symbol source when its points sit on the smaller constellation.  So the read reports the
    alphabet that was used, and the redundancy -- first-order entropy against log2(M), from
    ``entroptics.sequence.block_entropies`` -- says how skewed the usage was.  The ambiguity is
    real; what matters is that the receiver can see it."""
    n = 2 * 400 * 64
    even, _ = _sent(64, 64, 4, 1234, bit_seed=0)
    sparse = (np.arange(n) % 64 == 0).astype(np.int64)
    d_even = wf.discover(wf.compose(even, 64, 64, 1234, M=4), 1234, range(56, 72))
    d_skew = wf.discover(wf.compose(sparse, 64, 64, 1234, M=4), 1234, range(56, 72))
    assert d_even.M == 4 and d_even.redundancy < 0.05
    assert d_skew.redundancy > 0.2, "a skewed source must be visible in the read"


# ── the alphabet is decided by the constellation, not by how much of it was used ───────────────

@pytest.mark.parametrize("name,make,expect_M", [
    ("random", lambda n: np.random.default_rng(0).integers(0, 2, n), 4),
    ("half the points", lambda n: np.tile([0, 0, 1, 1], n // 4), 4),
    # One point lies on every constellation, so a constant source carries no geometry to read an
    # alphabet from and the read reports the smallest one.  A transmission is a function of the
    # message alone, so nothing supplies the symbol diversity the payload did not: the ambiguity
    # is visible where it is real.  ``redundancy`` carries it, and the decode stays exact.
    ("constant", lambda n: np.zeros(n, np.int64), 2),
])
def test_a_partly_used_alphabet_is_still_that_alphabet(name, make, expect_M):
    """A source that uses two of four points sits at 0 and 270 degrees, which is not the two-point
    constellation, so M=4 stays distinguishable.  Deciding M by how many symbols appeared instead
    of by the geometry costs BER 0.49 on exactly that payload.

    The claim is about geometry, and the two-point case is what tests it.  A single point is not
    partial use of an alphabet; it is no evidence of one."""
    n = 2 * 400 * 64
    bits = make(n)
    iq = wf.compose(bits, 64, 64, 1234, M=4)
    got, d = wf.decode(iq, 1234, range(58, 72))
    m = min(bits.size, got.size)
    assert d.M == expect_M, f"{name}: read M={d.M}"
    assert np.mean(bits[:m] != got[:m]) == 0.0, f"{name}: decode must stay exact"
    if expect_M != 4:
        assert d.redundancy > 0.95, (
            f"{name}: an alphabet that cannot be read must SAY so through redundancy, "
            f"got {d.redundancy:.3f}")


def test_redundancy_measures_the_alphabet_left_idle():
    """0 when traffic exercises the alphabet, 0.5 when half of it is idle, 1 for a constant
    source.  Measured against the constellation rather than against the symbols that happened to
    appear -- the library's own ``redundancy_rate`` answers the second question and reads 0 for a
    constant source, which is right for its question and wrong for this one."""
    n = 2 * 400 * 64
    def red(bits):
        return wf.discover(wf.compose(bits, 64, 64, 1234, M=4), 1234, range(58, 72)).redundancy
    assert red(np.random.default_rng(0).integers(0, 2, n)) < 0.05
    assert 0.4 < red(np.tile([0, 0, 1, 1], n // 4)) < 0.6
    assert red(np.zeros(n, np.int64)) > 0.95


def test_the_aperture_has_three_coordinates_not_four():
    """A scale and one translation per axis.  ``offset`` and ``phase`` are the whole and
    fractional parts of one real position; the split exists because slicing an array needs an
    integer, and it is not a second coordinate."""
    bits, iq = _sent(64, 64, 4, 1234, cut=17)
    d = wf.discover(iq, 1234, range(56, 72))
    assert d.position == pytest.approx(d.offset + d.phase)
    # The settle brackets the phase within half a sample of the integer offset, so the offset is
    # the position's nearest integer.  Here the true phase is 0 and the settle lands on it
    # (measured 0.0); a settle landing a hair below 0 is equally correct, and would flip a floor.
    assert -0.5 <= d.phase <= 0.5
    assert d.offset == round(d.position)


def test_a_sender_that_leaves_a_mode_dark_is_still_read_as_lighting_all_of_them():
    """The decision the mode count made no longer exists, and a foreign sender cannot bring it
    back.

    The sender here is built test-side, not by ``compose`` (which lights every mode): a real
    capture is projected onto the keyed aperture and its last mode emptied, the one-spare-mode
    shape the codec used to emit.  The receiver is still handed only the key.  It locates the
    screen and reads it as L modes, because L is what the wire format says -- it does not go
    looking for the dark mode, so there is no count it could be told or get wrong.  The lit modes
    read exactly; the dark one reads as noise, which is that sender's problem.  Measured: K = 63
    and 58 lock (64, 0) with K read as 64 and every lit mode exact (CM score 0.038 and 0.24
    against ~1e-15 for a full screen); at K = 40 the rows are far from constant modulus (score
    0.998) and nothing reads (closure 0.022, BER 0.5)."""
    L, K = 64, 63
    bits, iq = _sent(L, L, 4, 1234)
    B = wf.mode_basis(L, L, 1234)
    A = wf.project(wf.screen(np.asarray(iq), L), B)
    A[:, K:] = 0.0
    foreign = (A @ B).ravel().astype(np.complex64)
    got, d = wf.decode(foreign, 1234)
    assert (d.L, d.offset, d.K) == (L, 0, L), f"read L={d.L} offset={d.offset} K={d.K}"
    n = (min(bits.size, got.size) // (2 * L)) * (2 * L)
    lit = lambda b: b[:n].reshape(-1, 2 * L)[:, :2 * K]
    assert n > 0 and np.array_equal(lit(got), lit(bits)), "the lit modes must read exactly"


@pytest.mark.parametrize("used", [1, 17, 40, 63])
def test_the_final_rows_occupancy_is_free_and_no_convention_replaces_it(used):
    """How many modes the last row uses is a fact about the message length, not a setting.

    Every other row lights all L modes (K = L is the wire format, rank_leak.md cure (e)), so the
    only occupancy left free is the final row's, and it must stay free: padding it to a fixed
    count would be a convention the receiver has to know.  So the read must hold whatever the
    message leaves in the last row -- the geometry is located on the full rows, read as L modes,
    and the payload decodes exactly up to the message's end."""
    L = 64
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 2 * (299 * L + used))
    iq = wf.compose(bits, L, L, 1234, M=4)
    got, d = wf.decode(iq, 1234)
    assert (d.L, d.K) == (L, L), f"blind read gave L={d.L}, K={d.K}"
    n = min(len(bits), len(got))
    assert n == len(bits) and np.mean(bits[:n] != got[:n]) == 0.0


def test_which_payload_rides_is_measured_not_agreed():
    """The wire format offers two payloads, and the receiver must not be told which one it got.

    Before this, `decode` took the differential path unconditionally: a coherent capture came back
    at BER 0.4965 -- a confident wrong answer, which is worse than no answer, and it made the
    payload format a parameter the two ends had to agree on out of band.  The two closures name it
    instead, 1.0000 against 0.0060 on a differential capture and 0.0158 against 0.3592 on a
    coherent one (measured at K = L = 64), and the larger wins."""
    K, L, SEED = 64, 64, 1234
    pb = np.random.default_rng(1).integers(0, 2, 2 * 400 * K)
    psk = wf.compose(pb, K, L, SEED, M=4)
    qb = np.random.default_rng(2).integers(0, 2, 4 * 400 * K)
    qam = wf.compose_qam(qb, K, L, SEED, M=16)

    got, d = wf.decode(psk, SEED)
    assert d.payload == "differential", f"a differential capture read as {d.payload}"
    n = min(len(pb), len(got))
    assert n > 0 and np.array_equal(got[:n], pb[:n])

    got, d = wf.decode(qam, SEED)
    assert d.payload == "coherent", f"a coherent capture read as {d.payload}"
    assert d.M == 16, f"the coherent order is measured too; read {d.M}"
    n = min(len(qb), len(got))
    assert n > 0 and np.array_equal(got[:n], qb[:n])


def test_the_discovery_record_is_built_positionally_and_stays_that_way():
    """Discovery is constructed by position, so a field inserted in the middle silently reassigns
    every field after it.  Adding `payload` ahead of `row0` did exactly that -- row0 took its
    default of 0, which is the wrong keystream index, and decode returned confident nonsense while
    every geometry read stayed correct.  This pins the two that carry meaning by position."""
    K, L, SEED = 64, 64, 1234
    bits = np.random.default_rng(1).integers(0, 2, 2 * 300 * K)
    iq = wf.compose(bits, K, L, SEED, M=4)
    d = wf.discover(iq[37 * L:], SEED)
    assert d.row0 == 37, f"row0 came back {d.row0}; a field was inserted ahead of it"
    assert isinstance(d.payload, str)


# ── the entry-point audit ─────────────────────────────────────────────────────────────────────
#
# The signature audit above checks the front door.  It cannot see how many doors there are, and
# that is exactly how the payload format became a shared parameter: `decode`'s signature stayed
# clean while the wire format forked, and choosing between `read` and `read_qam` fell to the
# caller.  This audits the readers as a set, so that adding one is a deliberate act.

#: readers that legitimately need more than a capture and a key, and why.  Adding an entry here
#: is the deliberate act -- it is a rule exception, and it belongs in the paper as one.
DOCUMENTED_EXCEPTIONS = {
    "read_qam": "the informed coherent reader: takes M and the carrier as given; decode measures "
                "both and reads a coherent payload itself",
    "read": "the geometry it takes is what decode measures and passes",
    "read_soft": "same geometry as read; returns the values before the slice",
    "read_beacon": "a component of the blind read, not a receive path: it takes the differential "
                   "decode has already formed, and decode reaches it by measurement",
}


def test_every_reader_is_either_blind_or_a_declared_exception():
    readers = sorted(n for n in wf.__all__ if n.startswith("read"))
    assert readers, "no readers exported; this audit is looking at the wrong names"
    for name in readers:
        fn = getattr(wf, name)
        sig = inspect.signature(fn)
        required = [n for n, p in sig.parameters.items()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        blind = required == ["iq"]
        assert blind or name in DOCUMENTED_EXCEPTIONS, (
            f"{name} requires {required} and is not a declared exception -- either it should be "
            "reachable from decode by measurement, or the exception belongs in the paper")


def test_every_wire_format_is_named_by_the_blind_reader():
    """A transmitter can emit three formats.  The receiver must not be told which it got.

    `decode` identifies the two that ride the screen.  The analog embedding rides none -- it has
    no aperture geometry -- so the sweep returns whatever width fits best, and the evidence for it
    is nil: closure 0.070 at 2.7x the null, against 1.0000 at 139x for a screen.  `decode`
    reports that evidence rather than deciding on the caller's behalf, and every reader in this
    package declines it (see `waveform.closure_significance`, paper RF-8).

    So this pins the evidence, which is what a caller reads."""
    K, L, SEED = 64, 64, 1234
    pb = np.random.default_rng(1).integers(0, 2, 2 * 300 * K)
    _, d = wf.decode(wf.compose(pb, K, L, SEED, M=4), SEED)
    assert d.payload == "differential"

    qb = np.random.default_rng(2).integers(0, 2, 4 * 300 * K)
    _, d = wf.decode(wf.compose_qam(qb, K, L, SEED, M=16), SEED)
    assert d.payload == "coherent"

    # The embedding rides no screen, so the sweep returns a width that fits nothing.  What it
    # must not do is claim good evidence for it.  Asserting the geometry it happens to return
    # would pin an accident: nesting the spread operator moved that from K=1 to K=14 without
    # changing anything a caller acts on.  So this asserts the evidence instead.
    emb = wf.compose_embedding(np.sin(np.linspace(0, 9, 768)), SEED)
    got, d = wf.decode(emb, SEED)
    assert d.closure < 0.2, (
        f"the embedding closed at {d.closure:.3f} on an alphabet it does not carry")
    assert d.margin < 3.0, (
        f"the embedding claimed a margin of {d.margin:.2f}; a real screen reads orders above this")



def test_the_public_aperture_is_as_white_as_a_keyed_one():
    """An open broadcast reads as full width, like a keyed one, so a receiver does not mistake it
    for a narrower sender and an observer cannot tell the two apart by their spectrum (a low-discrepancy
    zero state such as R2 reads 0.45-0.52 occupied at full width).

    300 rows, because on 70 the coarse read itself scatters past the bracket now and then, white
    aperture or not: measured over 60 draws, the keyed bracket missed 1.0 on 2 (1 with the per-row
    radius), and on none at 300 rows.  ``rng`` fixes the open mode's sender-random radius."""
    bits = np.random.default_rng(0).integers(0, 2, 2 * 64 * 300)
    for seed in (None, 1234):
        lo, hi = wf._coarse_scale(np.asarray(wf.compose(bits, 64, 64, seed, rng=0)))
        assert lo < 1.0 <= hi * 1.05
