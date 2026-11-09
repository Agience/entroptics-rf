"""
Finding the screen: one read, one answer.

``discover`` is the only way a receiver locates a transmission.  There is no second route and no
fallback, so these tests pin that single read -- what it costs, where it stops, and that it needs
no threshold to be believed.  What it must not require is audited separately, in
``test_no_preshared_parameters.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf

SEED = 1234
L = 64
K = L
WIDTHS = range(40, 140)


def _iq(k=K, rows=300, width=L, seed=SEED, cut=0, M=4):
    bits = np.random.default_rng(0).integers(0, 2, int(np.log2(M)) * rows * k)
    return wf.compose(bits, k, width, seed, M=M)[cut:]


def _cfo(x, f):
    return (x * np.exp(2j * np.pi * f * np.arange(len(x)))).astype(np.complex64)


def _awgn(x, snr_db, seed=0):
    r = np.random.default_rng(seed)
    p = np.mean(np.abs(x) ** 2)
    return (x + (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x)))
            * np.sqrt(p / (2 * 10 ** (snr_db / 10)))).astype(np.complex64)


# ── the read ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [0, 1, 17, 33, 63])
def test_finds_width_and_offset(cut):
    """A real capture starts mid-row, and one sample out is fatal -- so the offset is half the
    answer, not a refinement."""
    d = wf.discover(_iq(cut=cut), SEED, WIDTHS)
    assert d.locked and d.L == L and d.offset == (L - cut) % L


@pytest.mark.parametrize("width,k", [(48, 48), (64, 64), (96, 96), (128, 128)])
def test_finds_a_range_of_geometries(width, k):
    d = wf.discover(_iq(k=k, width=width, cut=17), SEED, WIDTHS)
    assert d.L == width and d.offset == (width - 17) % width and d.K == k


def test_the_score_is_a_cliff_not_a_threshold():
    """Why there is no tuned constant in this decision: the answer is not close.  At the true
    geometry every row's coordinates on the keyed modes are unit modulus, so its energy is spread
    exactly evenly; every other cut mixes them toward complex Gaussian.  Measured: the
    constant-modulus read gives 1.7e-15 at the truth and a runner-up margin of 5e14."""
    iq = _iq()
    d = wf.discover(iq, SEED, WIDTHS)
    assert d.score < 1e-12
    assert d.margin > 1e9


@pytest.mark.parametrize("rows", [2, 4, 16])
def test_needs_almost_no_data(rows):
    """Pinned on the fixed-row-count read underneath ``discover``, which finds its own rows.  The
    sensitivity is the property worth naming: two rows are enough on a clean capture, which is
    what makes the widening cheap when the widening is not needed."""
    d = wf._discover_at(_iq(cut=17), SEED, WIDTHS, None, rows, 1.0)
    assert (d.L, d.offset, d.K) == (L, 47, K)


@pytest.mark.parametrize("snr_db", [18, 12, 6])
def test_the_read_finds_its_own_rows(snr_db):
    """The row count is not a caller parameter: the read starts at the fewest rows its own reads
    can work with and widens until the answer stops changing -- exact equality of the width and
    the row, not a band on a measured quantity.

    A clean capture agrees immediately and pays almost nothing; only a struggling one pays for the
    deep reads.  Measured over twenty captures at 6 dB (K = L = 64): a fixed sixteen-row read
    places the width and row 20/20 but reads the payload at BER 0.50 every time, against 20/20 at
    BER 0.129 for a sixty-four-row read -- the geometry is cheap, and what the deeper rows buy is
    everything read after it."""
    d = wf.discover(_awgn(_iq(cut=17), snr_db, seed=snr_db), SEED, WIDTHS)
    assert (d.L, d.offset) == (L, 47)


def test_widening_costs_nothing_when_it_is_not_needed():
    """The widening must not tax the easy case: a clean capture agrees at the first doubling, so
    it reads little more of the capture than a fixed sixteen-row read does."""
    import time
    iq = _iq(cut=17)
    t0 = time.perf_counter(); wf.discover(iq, SEED, WIDTHS); adaptive = time.perf_counter() - t0
    t0 = time.perf_counter(); wf._discover_at(iq, SEED, WIDTHS, None, 16, 1.0)
    fixed = time.perf_counter() - t0
    assert adaptive < fixed * 4, f"clean capture cost {adaptive:.2f}s against {fixed:.2f}s fixed"


def test_a_full_rank_screen_is_what_is_found():
    """Every mode is lit, and that is what the read finds: nothing marks the geometry but the
    constant modulus of each row's keyed coordinates, so no mode is left dark to be found by.
    The mode count is not read -- it is L -- and the score is at the noiseless floor (measured
    1.4e-15)."""
    d = wf.discover(_iq(k=L), SEED, [L])
    assert d.K == L and d.score < 1e-12


# ── adaptive carrier ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("f", [0.0, 3e-4, 1e-3, 3e-3])
def test_adaptive_carrier_reads_carrier_zero_first(f):
    """The carrier is one oscillator offset for the whole capture, read closed-form once the
    geometry settles, so a capture whose offset sits within the geometry read's tolerance is found
    from a single carrier-zero probe -- the three-to-four-times-faster common case.  The winner's
    margin over its nearest rival is the evidence the cheap read is a true lock (a cliff, ~1e14
    clean), so ``discover`` trusts it without widening the carrier search."""
    iq = _cfo(_iq(cut=17), f)
    d = wf.discover(iq, SEED, WIDTHS)
    assert (d.L, d.offset) == (L, 47)
    d0 = wf._discover_span(iq, SEED, WIDTHS, 0.0, wf.JOIN_SEARCH_ROWS)
    assert d0.margin > wf.CARRIER_SKIP_MARGIN          # the fast path is taken


def test_adaptive_carrier_widens_past_the_zero_probe_envelope():
    """Beyond the range the carrier-zero read tolerates (~0.5 cycles per row), the cheap read is
    not a clear lock -- its winner no longer beats its rivals by the margin -- so ``discover``
    widens to the requested span and finds the geometry there.  Same answer, paid for only when
    the oscillator needs it."""
    iq = _cfo(_iq(cut=17), 1.5e-2)                     # ~0.96 cycles/row, past the carrier-zero envelope
    d0 = wf._discover_span(iq, SEED, WIDTHS, 0.0, wf.JOIN_SEARCH_ROWS)
    assert d0.margin <= wf.CARRIER_SKIP_MARGIN         # the cheap read declines
    d = wf.discover(iq, SEED, WIDTHS)                  # the full span still finds it
    assert (d.L, d.offset) == (L, 47)


def test_the_parallel_sweep_is_bit_identical_to_serial():
    """The acquisition sweep runs its independent (carrier, width) reads across cores, each matmul
    still single-threaded, so the lock is the serial sweep's exactly -- every field of the read,
    geometry, score and margin.  The worker count is a speed knob, never part of the decision."""
    from dataclasses import astuple
    iq = _cfo(_iq(cut=17), 1e-3)
    saved = wf._SWEEP_WORKERS
    try:
        wf._SWEEP_WORKERS = 1
        serial = wf.discover(iq, SEED, WIDTHS)
        wf._SWEEP_WORKERS = 4
        parallel = wf.discover(iq, SEED, WIDTHS)
    finally:
        wf._SWEEP_WORKERS = saved
    assert astuple(serial) == astuple(parallel)


def test_adaptive_carrier_is_cheaper_than_the_full_span():
    """The whole point: the common case pays one carrier probe, not the full sweep.  Measured on a
    clean small-offset capture, the adaptive read costs about a quarter of the forced full-span
    read."""
    import time
    iq = _cfo(_iq(cut=17), 3e-4)
    wf.discover(iq, SEED, WIDTHS)                                    # warm
    wf._discover_span(iq, SEED, WIDTHS, 1.0, wf.JOIN_SEARCH_ROWS)    # warm
    t0 = time.perf_counter(); wf.discover(iq, SEED, WIDTHS); adaptive = time.perf_counter() - t0
    t0 = time.perf_counter()
    wf._discover_span(iq, SEED, WIDTHS, 1.0, wf.JOIN_SEARCH_ROWS)
    full = time.perf_counter() - t0
    assert adaptive < full * 0.6, f"adaptive {adaptive:.2f}s vs full span {full:.2f}s"


# ── harmonics ─────────────────────────────────────────────────────────────────────────────────

def test_harmonics_are_rejected():
    """A cut at 2L holds two whole rows, so it is genuinely low-rank and dips on any fill read --
    on a mis-started capture it out-dips the fundamental.  It cannot put its rows inside the
    aperture, which is why the read asks about the aperture."""
    iq = _iq(rows=600, cut=17)
    d = wf.discover(iq, SEED, [64, 128, 192])
    assert d.L == L and d.offset == 47


# ── carrier offset: where this read stops ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cfo", [0.0, 1e-5, 1e-4, 1e-3])
def test_holds_across_the_carrier_offsets_the_read_survives(cfo):
    """The per-column phase ramp rotates rows partly out of the mode space, so the score degrades
    with offset: this read is carrier-tolerant rather than carrier-invariant.  It holds to ~3e-3,
    past the ~1e-3 the differential read itself tolerates, so it is not the binding constraint.
    Gross offset is a front-end job."""
    d = wf.discover(_cfo(_iq(), cfo), SEED, WIDTHS)
    assert d.L == L and d.offset == 0


def test_carrier_offset_is_measured_not_suffered():
    """The carrier is a coordinate, not a degradation.  The read finds nu = f*L and the score
    stays at the floor, which is what makes a two-clock link a measurement problem rather than a
    tracking one."""
    iq = _iq()
    for f in (0.0, 1e-4, 1e-3, 1e-2):
        d = wf.discover(_cfo(iq, f), SEED, [L], carrier_span=1.0)
        assert d.L == L
        assert abs(d.carrier - f * L) < 0.02, f"cfo {f}: read nu {d.carrier:.4f}"


def test_widening_keeps_the_best_evidenced_candidate_not_the_newest():
    """Convergence says when to stop widening.  It does not say what to return.

    A doubling that returns a different geometry has not converged, and it has not necessarily
    improved either, so the candidates are re-read on every row the capture has -- identical
    evidence -- and compared there.  Keeping whichever came last instead throws a lock away for
    an answer that merely arrived later.  The span is deliberately wider than the shipped
    default, because that is where the widening disagrees with itself: measured at this span it
    proposes (139, 36), (107, 52) and (70, 49) at 0.91-0.95 on 8, 16 and 32 rows before (64, 0)
    at 3e-15 on 64 and 128.  There the truth also arrives last, so the end-to-end read does not
    by itself separate "best" from "newest"; the comparison is therefore also pinned directly,
    with the wrong proposal offered as the newest.
    """
    x = _cfo(_iq(), 0.64 / L)
    d = wf.discover(x, SEED, WIDTHS, carrier_span=8.0)
    assert d.L == L and d.offset == 0, f"kept ({d.L}, {d.offset}) at score {d.score:.3e}"
    assert d.score < 1e-12, f"returned a geometry its own read scores at {d.score:.3e}"
    assert d.closure > 0.9

    import dataclasses
    late = dataclasses.replace(d, L=139, offset=36, score=0.91)   # the 8-row proposal, as newest
    kept = wf._best_of(x, SEED, [d, late], late)
    assert (kept.L, kept.offset) == (L, 0), f"kept the newest ({kept.L}, {kept.offset})"


def test_a_shallower_read_is_not_preferred_on_a_luckier_score():
    """The other half of the same rule.  Scores from different row counts are not comparable --
    a shallow read is a noisier one, so a wrong geometry can score low at eight rows on luck --
    so the comparison is made after re-reading every candidate on the whole capture.  Comparing
    the proposing scores instead prefers the shallowest read and costs this lock outright."""
    d = wf.discover(_awgn(_iq(), 6, seed=6), SEED, WIDTHS)
    assert d.L == L and d.offset == 0


# ── noise ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("snr_db", [18, 12, 9, 6])
def test_survives_noise(snr_db):
    d = wf.discover(_awgn(_iq(), snr_db, seed=snr_db), SEED, WIDTHS)
    assert d.L == L and d.offset == 0


def test_pure_noise_is_not_mistaken_for_a_transmission():
    noise = (np.random.default_rng(0).standard_normal(20_000)
             + 1j * np.random.default_rng(1).standard_normal(20_000)).astype(np.complex64)
    d = wf.discover(noise, SEED, WIDTHS)
    assert d.score > 1e-3 and d.margin < 1e3


# ── the evidence read, at one geometry ────────────────────────────────────────────────────────

def test_low_snr_shows_up_in_the_margin_rather_than_being_hidden():
    """At 3 dB the read is marginal, and it says so: the margin collapses toward 1 (measured 1.24)
    instead of the ~5e14 a clean lock gives, so a receiver can see it is guessing."""
    iq = _awgn(_iq(), 3, seed=3)
    assert wf.discover(iq, SEED, WIDTHS).margin < 10.0


def test_the_public_read_agrees_with_the_discovery_score_at_the_locked_geometry():
    """``constant_modulus_read`` is the same evidence at a geometry a caller already has -- a
    diagnostic, not a second locator.  It must not disagree with the read that did the locating.
    Measured: 1.5e-15 at the lock, 1.00-1.03 one or two samples either side."""
    iq = _iq()
    d = wf.discover(iq, SEED, WIDTHS)
    assert wf.constant_modulus_read(iq, d.L, SEED, offset=d.offset) < 1e-9
    assert wf.constant_modulus_read(iq, d.L, SEED, offset=d.offset + 1) > 0.5


# ── cost ──────────────────────────────────────────────────────────────────────────────────────

def test_discovery_is_light():
    """Acquisition is the one expensive step, so it has a budget -- counted in work, not seconds.

    A wall-clock budget measures the machine as much as the code: this same read takes 1.24 s on a
    quiet box and blew a 2.0 s ceiling on a loaded one, while doing byte-for-byte identical work.
    So the budget is the work itself.  Discovery's cost is the sweep, and the sweep is one
    constant-modulus score (``_cm_score``) per (candidate width, carrier probe) per doubling of
    the row count -- measured here at
    exactly 100 x 8 x 2, the carrier probe converging in eight and a clean capture agreeing at the
    first doubling.  The bound below allows a third sweep and no more, so a capture that needs one
    extra doubling passes and a per-offset Python loop creeping back into the read does not."""
    iq = _iq()
    widths = len(list(WIDTHS))
    calls, real = [0], wf._cm_score

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    wf._cm_score = counted
    try:
        wf.discover(iq, SEED, WIDTHS)
    finally:
        wf._cm_score = real
    assert calls[0] <= widths * 8 * 3, (
        f"{calls[0]} constant-modulus scores for {widths} widths -- "
        f"{calls[0] / widths:.1f} probe-sweeps, expected 16")


@pytest.mark.parametrize("snr_db", [12, 6, 3])
def test_the_read_costs_almost_nothing_against_an_oracle(snr_db):
    """The reads are not what limits the link.

    A receiver handed the true width, row, mode count, alphabet and continuous coordinates is the
    ceiling this codec can reach, and the distance to it is what discovery costs.  Settling the
    continuous pair on the whole capture puts that distance within 0.5% of the oracle BER at
    12, 9, 6 and 3 dB (measured -0.2%, +0.4%, +0.3% and -0.1% over these four trials), locking
    every time.  The bound below is headroom over those measured ratios, not a target."""
    from entroptics_rf.runner import ber
    sent = np.random.default_rng(0).integers(0, 2, 2 * 400 * K)
    reads, oracles = [], []
    for trial in range(4):        # a ratio of two small BERs is counting noise on one capture
        iq = _awgn(wf.compose(sent, K, L, SEED, M=4), snr_db, seed=100 * snr_db + trial)
        got, d = wf.decode(iq, SEED, WIDTHS)
        assert (d.L, d.K, d.M) == (L, K, 4), f"lost the lock on trial {trial}: {d}"
        n = min(sent.size, got.size)
        reads.append(ber(sent[:n], got[:n]))
        ob = wf.read(iq, L, K, SEED, M=4)
        m = min(sent.size, ob.size)
        oracles.append(ber(sent[:m], ob[:m]))
    read, oracle = float(np.mean(reads)), float(np.mean(oracles))
    assert read <= oracle * 1.5 + 1e-4, (
        f"the read costs {read/max(oracle,1e-9):.2f}x the oracle ({read:.5f} vs {oracle:.5f})")


@pytest.mark.parametrize("name", ["dc", "tone", "repeat"])
def test_the_geometry_read_alone_does_not_identify_this_waveform(name):
    """A degenerate signal locks a geometry as convincingly as a weak real link does.

    The constant-modulus read asks whether some width and offset spread each row's energy evenly
    over the keyed modes, and a DC level, a single tone and a period-L repeat all find a width
    where they do so better than chance.  Measured: DC locks (76, 0) at 0.72, a tone (24, 7) and
    a repeat (24, 0) at 0.416 -- against 0.43 for a real screen at 6 dB, which decodes.  So the
    score cannot tell an impostor from a link worth reading.

    Only the closure separates them, and it separates them totally: 1.0000 against 0.016-0.027.
    This pins that the second read is load-bearing, so that nobody later decides the geometry
    evidence is sufficient on its own -- a receiver scanning a band with a carrier in it takes a
    confident geometry lock on the carrier."""
    n = np.arange(8192)
    x = {"dc": np.ones(8192),
         "tone": np.exp(2j * np.pi * 0.017 * n),
         "repeat": np.tile(np.exp(2j * np.pi * np.arange(64) / 64), 128)}[name]
    d = wf.discover(x.astype(np.complex64), SEED)
    assert d.locked, f"{name} did not lock at all; this test assumes it does"
    assert d.score < 0.9, f"{name} scored {d.score:.3g}: the premise of this test has changed"
    assert d.closure < 0.2, (
        f"{name} closed at {d.closure:.4f} -- an impostor should carry no alphabet")

    bits = np.random.default_rng(1).integers(0, 2, 2 * 300 * K)
    real = wf.discover(wf.compose(bits, K, L, SEED, M=4), SEED, WIDTHS)
    assert real.closure > 0.9, "the real waveform should close where the impostors do not"


@pytest.mark.parametrize("nu", [0.064, 0.192, 0.64])
def test_the_carrier_is_removed_in_the_right_direction(nu):
    """The sign of the carrier de-rotation is load-bearing above a small offset, and was untested.

    Found by mutation: flipping `exp(-2j.pi.carrier)` to `exp(+2j.pi.carrier)` in `read` passed
    the entire suite.  Every test that applied a carrier used 3e-4 cycles/sample -- nu = 0.0192 at
    L = 64 -- where the differential absorbs the error and both signs decode.  At nu >= 0.064 they
    do not: measured BER 0.0000 against 0.3971, 0.5972 and 0.5020 at nu = 0.064, 0.192 and 0.64.

    So this asserts both halves: the correct sign decodes exactly, and the wrong one does not.
    Asserting only the first would leave the mutation alive at any offset the fixture happens to
    use."""
    bits = np.random.default_rng(1).integers(0, 2, 2 * 400 * K)
    base = wf.compose(bits, K, L, SEED, M=4)
    n = np.arange(len(base))
    iq = (np.asarray(base) * np.exp(2j * np.pi * nu * n / L)).astype(np.complex64)

    d = wf.discover(iq, SEED, WIDTHS)
    assert d.locked and abs(d.carrier - nu) < 0.01, f"carrier read {d.carrier:+.4f} for nu={nu}"

    right = wf.read(iq, d.L, d.K, SEED, M=d.M, offset=d.offset,
                    phase=d.phase, carrier=d.carrier, row0=d.row0)
    m = min(len(bits), len(right))
    assert np.mean(bits[:m] != right[:m]) == 0.0, "the measured carrier should decode exactly"

    wrong = wf.read(iq, d.L, d.K, SEED, M=d.M, offset=d.offset,
                    phase=d.phase, carrier=-d.carrier, row0=d.row0)
    m = min(len(bits), len(wrong))
    assert np.mean(bits[:m] != wrong[:m]) > 0.3, (
        "the opposite sign should destroy the payload; if it no longer does, the carrier has "
        "stopped being load-bearing here and this test is measuring nothing")


@pytest.mark.parametrize("snr_db,least", [(18, 50.0), (12, 40.0), (6, 12.0)])
def test_a_real_transmission_stands_well_above_the_null(snr_db, least):
    """The significance read must accept everything the link can actually decode.

    A fixed closure level could not: the closure falls with SNR (measured 1.000 clean, 0.590 at
    12 dB, 0.123 at 6), so a level high enough to reject a tone rejects a 6 dB link decoding at
    BER 0.11.  The ratio to the null is what is stable, because the null's own scale is
    1/sqrt(TK) and moves with the capture: measured 141x at 18 dB, 94x at 12, 20x at 6."""
    bits = np.random.default_rng(1).integers(0, 2, 2 * 400 * K)
    iq = _awgn(wf.compose(bits, K, L, SEED, M=4), snr_db, seed=7)
    d = wf.discover(iq, SEED, WIDTHS)
    sig = wf.closure_significance(d, len(np.asarray(iq).ravel()))
    got = wf.read(iq, d.L, d.K, SEED, M=d.M, offset=d.offset,
                  phase=d.phase, carrier=d.carrier, row0=d.row0)
    n = min(len(bits), len(got))
    assert np.mean(bits[:n] != got[:n]) < 0.15, "premise: this SNR should still decode"
    assert sig > least, f"{snr_db} dB read {sig:.1f}x the null; a reader would decline it"


@pytest.mark.parametrize("name", ["noise", "dc", "tone", "repeat"])
def test_nothing_without_an_alphabet_reaches_the_readers_operating_point(name):
    """And it must decline everything the link cannot decode, whatever geometry the sweep invents.

    Measured on these captures: noise 3.49x the null, DC 1.90x, a tone 1.90x, a period-L repeat
    1.71x.  The readers sit at 8x."""
    n = np.arange(20000)
    x = {"noise": (np.random.default_rng(5).standard_normal(20000)
                   + 1j * np.random.default_rng(6).standard_normal(20000)),
         "dc": np.ones(20000),
         "tone": np.exp(2j * np.pi * 0.031 * n),
         "repeat": np.tile(np.exp(2j * np.pi * np.arange(64) / 64), 312)}[name]
    d = wf.discover(x.astype(np.complex64), SEED)
    sig = wf.closure_significance(d, 20000)
    assert sig < 5.0, f"{name} read {sig:.2f}x the null -- too close to a transmission"


@pytest.mark.parametrize("nu", [3.0, 8.0])
def test_a_carrier_past_the_search_span_fails_loudly(nu):
    """A search bound that mislocks silently is worse than one that is too small.

    Past the span the read cannot find the carrier, and what matters is what it does then: it must
    not return a confident geometry.  Measured, it drops to 1.8x and 2.7x the null where a real
    capture reads over a hundred, so a reader declines it (paper RF-3.2)."""
    bits = np.random.default_rng(1).integers(0, 2, 2 * 400 * K)
    base = np.asarray(wf.compose(bits, K, L, SEED, M=4))
    iq = (base * np.exp(2j * np.pi * nu * np.arange(len(base)) / L)).astype(np.complex64)
    d = wf.discover(iq, SEED, WIDTHS)
    assert wf.closure_significance(d, len(iq)) < 8.0, (
        "an unreachable carrier must not read as a transmission")


def test_the_deepest_recoverable_join_is_one_less_than_the_count():
    """`JOIN_SEARCH_ROWS` is a count of candidate rows, so the last reachable row is count - 1.

    Pinned because the name reads as a depth: joining at 4095 is exact and at 4096 it is not, and
    a reader who took the constant as a maximum join depth would be wrong by one row -- which is
    the whole payload, since a wrong alignment is the wrong keystream."""
    deep = wf.JOIN_SEARCH_ROWS - 1
    bits = np.random.default_rng(1).integers(0, 2, 2 * (deep + 400) * K)
    big = np.asarray(wf.compose(bits, K, L, SEED, M=4))
    got, d = wf.decode(big[deep * L:], SEED)
    want = bits[deep * K * 2:]
    n = min(len(want), len(got))
    assert d.row0 == deep and np.array_equal(got[:n], want[:n])


def test_the_join_reach_is_the_callers_to_set():
    """A join past the default reach is recoverable by asking for a deeper one.

    Nothing in a capture says how late a receiver tuned in, so how far to hunt is a deployment's
    answer rather than the codec's (paper RF-9.1).  What the codec owes is that raising the reach
    works, and that not raising it declines rather than inventing an alignment: outside the reach
    the closure collapses and every reader in the package refuses the capture."""
    deep = wf.JOIN_SEARCH_ROWS + 904
    bits = np.random.default_rng(4).integers(0, 2, 2 * (deep + 400) * K)
    big = np.asarray(wf.compose(bits, K, L, SEED, M=4))
    cut = big[deep * L:]

    got, d = wf.decode(cut, SEED, join_rows=2 * wf.JOIN_SEARCH_ROWS)
    want = bits[deep * K * 2:]
    n = min(len(want), len(got))
    assert d.row0 == deep and np.array_equal(got[:n], want[:n]), (
        f"a reach containing the join must recover it exactly, read row0 {d.row0} for {deep}")

    _, shallow = wf.decode(cut, SEED)
    assert shallow.row0 != deep, "the default reach cannot contain this join"
    assert wf.closure_significance(shallow, len(cut)) < 8.0, (
        "a join outside the reach must read as no transmission, not a confident wrong row")


def test_an_open_broadcast_is_joinable_at_any_point():
    """With no key there is no row rotation, so an open fragment reads from wherever it starts.

    The per-row rotation exists to keep a blindly separated mode unreadable (paper RF-5), and it
    is what obliges a keyed receiver to find the row it joined on.  An open broadcast has nothing
    to keep unreadable, its aperture and reference row being public, so it carries no rotation and
    offers no alignment to search for.  A fragment cut from past the join reach therefore reads
    exactly, where the same cut of a keyed stream cannot."""
    deep = wf.JOIN_SEARCH_ROWS + 900
    bits = np.random.default_rng(7).integers(0, 2, 2 * (deep + 200) * K)
    full = np.asarray(wf.compose(bits, K, L, None, M=4))
    frag = full[deep * L:(deep + 128) * L]

    got, d = wf.decode(frag, None)
    want = bits[deep * K * 2:]
    n = min(len(want), len(got))
    assert n and np.array_equal(got[:n], want[:n]), (
        "an open fragment past the join reach must read exactly, position being irrelevant")
    assert d.row0 == 0, f"an unrotated read has no row to find, got row0 {d.row0}"
