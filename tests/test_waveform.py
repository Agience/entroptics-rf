"""The write/read path: keying, the keyed aperture, and the digital payload."""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf


# ── the key ───────────────────────────────────────────────────────────────────────────────────

def test_keystream_is_deterministic_and_domain_separated():
    """One seed, many independent streams -- the property the reference and the aperture rest on.

    If the domains were not separated, the reference (which is transmitted) would leak the data
    aperture (which must not be derivable)."""
    a = wf._keyed_normal(wf._D_BASIS, 1234, (8, 4))
    assert np.array_equal(a, wf._keyed_normal(wf._D_BASIS, 1234, (8, 4)))       # deterministic
    assert not np.allclose(a, wf._keyed_normal(wf._D_SPREAD, 1234, (8, 4)))     # domain-separated
    assert not np.allclose(a, wf._keyed_normal(wf._D_BASIS, 1235, (8, 4)))      # seed-separated
    assert not np.allclose(a, wf._keyed_normal(wf._D_BASIS, 1234, (8, 4), b"n"))  # nonce-separated


def test_seed_encoding_is_injective_across_types():
    """int 0x41 and b'A' must not alias onto one key."""
    assert wf._seed_bytes(0x41) != wf._seed_bytes(b"A")
    assert wf._seed_bytes(-1) != wf._seed_bytes(1)
    assert wf._seed_bytes(0) != wf._seed_bytes(b"")


def test_keyed_uniforms_are_open_interval():
    """Strictly inside (0, 1): a 0 would make the Box-Muller log infinite."""
    u = wf._keyed_uniform(wf._D_BASIS, 7, 20000)
    assert u.min() > 0.0 and u.max() < 1.0


def test_keyed_normals_are_standard_complex():
    """x + iy with x, y ~ N(0,1) independent, so the variance is 2."""
    z = wf._keyed_normal(wf._D_BASIS, 7, (200, 200))
    assert abs(z.mean()) < 0.02
    assert abs(z.var() - 2.0) < 0.05


# ── the aperture ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K,L", [(4, 8), (58, 64), (64, 64), (1, 16)])
def test_mode_basis_is_orthonormal(K, L):
    B = wf.mode_basis(K, L, 1234)
    assert B.shape == (K, L)
    assert np.allclose(B @ B.conj().T, np.eye(K), atol=1e-11)


def test_adjoint_is_the_pseudoinverse():
    """The read uses B^H rather than pinv(B); they must be the same matrix, or the read is wrong."""
    B = wf.mode_basis(58, 64, 1234)
    assert np.allclose(np.linalg.pinv(B), B.conj().T, atol=1e-10)


def test_qr_phase_is_canonical():
    """Q is pinned so two machines derive the same aperture: R's diagonal is positive real.

    numpy leaves Q free up to a per-column unit-modulus factor, so without this the link works in
    one process and fails across a real one."""
    G = wf._keyed_normal(wf._D_BASIS, 99, (32, 8))
    Q = wf._canonical_qr(G)
    R = Q.conj().T @ G
    d = np.diagonal(R)
    assert np.allclose(d.imag, 0, atol=1e-10)
    assert (d.real > 0).all()


def test_mode_basis_rejects_impossible_geometry():
    with pytest.raises(ValueError):
        wf.mode_basis(65, 64, 1)          # more modes than the screen has dimensions
    with pytest.raises(ValueError):
        wf.mode_basis(0, 64, 1)


# ── the ravel ─────────────────────────────────────────────────────────────────────────────────

def test_ravel_round_trips():
    """screen() and to_iq() are inverse, and neither transforms anything."""
    iq = (np.arange(64 * 5) + 1j * np.arange(64 * 5)).astype(np.complex64)
    S = wf.screen(iq, 64)
    assert S.shape == (5, 64)
    assert np.allclose(wf.to_iq(S, unit_rms=False), iq)


def test_partial_row_is_dropped_not_padded():
    """A partial row is not an observation of the screen."""
    assert wf.screen(np.zeros(64 * 3 + 17, dtype=complex), 64).shape == (3, 64)


def test_screen_rejects_a_capture_too_short():
    with pytest.raises(ValueError):
        wf.screen(np.zeros(10, dtype=complex), 64)


# ── the digital payload ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [2, 4, 8, 16])
def test_round_trip_is_bit_exact_on_a_clean_channel(M):
    K, L, seed = 32, 32, 1234
    bits = np.random.default_rng(0).integers(0, 2, 400 * K)
    iq = wf.compose(bits, K, L, seed, M=M)
    got = wf.read(iq, L, K, seed, M=M)
    assert got.size >= bits.size
    assert np.array_equal(got[:bits.size], bits)


def test_composed_iq_is_unit_rms_complex64():
    iq = wf.compose(np.random.default_rng(0).integers(0, 2, 4000), 32, 32, 1)
    assert iq.dtype == np.complex64
    assert abs(np.sqrt(np.mean(np.abs(iq) ** 2)) - 1.0) < 1e-5


@pytest.mark.parametrize("compose,M", [(wf.compose, 4), (wf.compose_qam, 16)])
def test_a_short_payload_is_not_padded_out_to_a_grid(compose, M):
    """Nothing is sent that the message did not ask to send.

    A short final row is carried on fewer modes rather than padded out to a grid, which the nested
    aperture makes free (``mode_basis(K2, L, seed)`` is the first K2 rows of
    ``mode_basis(K, L, seed)``).  So the transmission is a function of the message length and the
    geometry and of nothing else, and the only rounding is inside the final symbol, under log2(M)
    bits.

    Padding a five-bit payload out to T*K symbols would put over a hundred bits of traffic on the
    air that the message never asked for."""
    K, L, seed = 24, 24, 5
    bits = np.array([1, 0, 1, 1, 0], dtype=np.int64)      # far shorter than one grid
    iq = compose(bits, K, L, seed, M=M)

    b = int(round(np.log2(M)))
    n_sym = int(np.ceil(bits.size / b))
    rows = max(1, int(np.ceil(n_sym / K)))
    assert len(iq) == (rows + 1) * L, "the composed length must follow the message, not a grid"

    # the final row must light exactly the modes the message needed, and no more
    A = wf.project(wf.screen(iq, L), wf.mode_basis(K, L, seed))
    lit = np.abs(A[-1]) ** 2
    K_last = n_sym - (rows - 1) * K
    assert np.all(lit[:K_last] > 1e-12), "the modes the message used must carry energy"
    assert np.allclose(lit[K_last:], 0.0, atol=1e-12), (
        f"modes past the message must carry NOTHING; got {lit[K_last:].max():.3e}")


def test_wrong_key_reads_as_noise():
    """The seed is the payload's only lock, so a wrong one must give a coin flip, not a hint."""
    K, L = 32, 32
    bits = np.random.default_rng(0).integers(0, 2, 400 * K)
    iq = wf.compose(bits, K, L, 1234)
    got = wf.read(iq, L, K, 1235)
    assert abs(np.mean(bits != got[:bits.size]) - 0.5) < 0.05


def test_no_receive_path_takes_a_nonce():
    """Only the key may be shared.

    A nonce a caller has to supply is a second agreed parameter.  The audit in
    ``test_no_preshared_parameters.py`` checks required arguments, so an optional nonce would pass
    it; this test pins the stronger property, that no function on the receive path accepts a nonce
    at all, by signature.

    What carrying a nonce would take is written up in the design note in ``waveform.py``.  One
    seed keys one waveform, and freshness is key management rather than a hidden agreement."""
    import inspect
    for fn in (wf.compose, wf.read, wf.compose_qam, wf.read_qam, wf.discover, wf.decode):
        assert "nonce" not in inspect.signature(fn).parameters, (
            f"{fn.__name__} takes a nonce: that is a second shared parameter")


def test_read_of_an_unusable_capture_is_empty_not_an_exception():
    """One row carries no differential, so there is nothing to report -- and nothing to crash on."""
    assert wf.read(np.zeros(32, dtype=complex), 32, 16, 1).size == 0


def test_capacity_matches_what_compose_actually_carries():
    K, L, M = 32, 32, 4
    bits = np.random.default_rng(0).integers(0, 2, 400 * K)
    iq = wf.compose(bits, K, L, 1, M=M)
    assert wf.capacity(K, L, len(iq), M) == wf.read(iq, L, K, 1, M=M).size
    assert wf.bits_per_sample(K, L, M) == pytest.approx(2 * K / L)


def test_papr_is_in_the_ofdm_class():
    """~9-11 dB, on a par with OFDM.  That is the comparison to draw: neither a win nor a
    loss."""
    iq = wf.compose(np.random.default_rng(0).integers(0, 2, 2 * 400 * 64), 64, 64, 1)
    assert 7.0 < wf.optics(iq, 64).papr_db < 13.0


# ── the optics record ─────────────────────────────────────────────────────────────────────────

def test_optics_reports_the_named_library_reads():
    from entroptics import Aperture
    iq = wf.compose(np.random.default_rng(0).integers(0, 2, 2 * 200 * 32), 32, 32, 1)
    o = wf.optics(iq, 32)
    ap = Aperture(wf.screen(iq, 32), window=None)
    assert o.fill == pytest.approx(float(ap.phi_F))
    assert o.etendue == pytest.approx(float(ap.etendue))
    assert o.strehl == pytest.approx(float(ap.strehl))
    assert o.occupied == pytest.approx(float(ap.etendue) * float(ap.space_bandwidth))


def test_batch_aperture_already_defaults_to_the_full_frame():
    """This package passes ``window=None`` explicitly, and for batch input that is a no-op.

    entroptics defaults ``window`` to None when constructed from an array, and to MIN_WINDOW only
    on the streaming path.  Pinned so the two defaults stay distinguishable: read against the
    streaming path, an explicit ``window=None`` looks like it changes the fill, and it does
    not."""
    from entroptics import Aperture
    iq = wf.compose(np.random.default_rng(0).integers(0, 2, 2 * 400 * 64), 64, 64, 1)
    S = wf.screen(iq, 64)
    assert S.shape[0] > 128                                        # long enough for it to matter
    assert float(Aperture(S).phi_F) == pytest.approx(float(Aperture(S, window=None).phi_F))
    assert float(Aperture(S, window=128).phi_F) != pytest.approx(float(Aperture(S).phi_F))
    assert wf.optics(iq, 64).fill == pytest.approx(float(Aperture(S, window=None).phi_F))


# ── soft output ───────────────────────────────────────────────────────────────────────────────

def test_read_soft_is_exactly_what_read_rounds():
    """read() must be read_soft() plus a slice, with nothing else in between.

    The point of exposing the soft values is that they are the *same* values -- if the two paths
    could drift apart, an outer code would be decoding something the hard path never saw."""
    K, L, SEED = 64, 64, 1234
    bits = np.random.default_rng(1).integers(0, 2, 2 * 300 * K)
    iq = wf.compose(bits, K, L, SEED, M=4)
    hard = wf.read(iq, L, K, SEED, M=4)
    d = wf.read_soft(iq, L, K, SEED)
    assert d.shape == (len(iq) // L - 1, K)
    idx = wf._slice_psk(d, 4).ravel()
    again = ((idx[:, None] >> np.arange(1, -1, -1)) & 1).astype(np.int64).ravel()
    assert np.array_equal(again, hard)


def test_the_soft_values_carry_usable_confidence():
    """Erasing the least-confident symbols must actually remove errors.

    Model-free: rank by angular distance to a decision boundary, drop the worst fifth, and the
    error rate has to fall.  If it did not, the slice would already be extracting everything and
    read_soft would be offering the caller nothing.  Measured at 9 dB (K = L = 64), erasing a
    fifth takes the symbol error rate from 0.037 to 0.005; the bound below is loose because the
    point is the direction, not the number."""
    K, L, SEED, M = 64, 64, 1234, 4
    rng = np.random.default_rng(0)
    bits = np.random.default_rng(1).integers(0, 2, 2 * 400 * K)
    iq = wf.compose(bits, K, L, SEED, M=M)
    p = float(np.mean(np.abs(iq) ** 2))
    y = iq + (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))) \
        * np.sqrt(p / (2 * 10 ** (9 / 10)))
    d = wf.read_soft(y.astype(np.complex64), L, K, SEED)
    idx = wf._slice_psk(d, M)
    got = ((idx.ravel()[:, None] >> np.arange(1, -1, -1)) & 1).astype(np.int64).ravel()
    n = min(len(bits), len(got))
    err = (bits[:n].reshape(-1, 2) != got[:n].reshape(-1, 2)).any(1)
    conf = np.pi - np.abs(np.angle(d.ravel()[:len(err)] ** M))
    keep = np.argsort(conf)[int(0.2 * len(conf)):]
    assert err[keep].mean() < err.mean() / 3, (
        f"erasing the least-confident fifth took {err.mean():.4f} to {err[keep].mean():.4f}; "
        "the soft values are not carrying confidence")


def test_a_short_final_row_leaves_its_unused_modes_dark():
    """No padding means the modes the message does not reach carry nothing at all.

    Every row but the last lights all L modes; only the final row may use fewer, and those it
    leaves dark (measured at L = 64: under 1e-15 against a full row's per-mode energy of ~1.3).

    Not merely that no filler bits are generated -- the last row still occupies L samples on the
    wire either way -- but that it radiates in proportion to what it carries.  Under a grid filler
    those modes carried keyed noise at full amplitude, which is power spent on bits the message
    never asked to send."""
    K, L, SEED = 64, 64, 1234
    for used in (1, 17, 63):
        nsym = 40 * K + used
        bits = np.random.default_rng(1).integers(0, 2, nsym * 2)
        iq = wf.compose(bits, K, L, SEED, M=4)
        S = wf.screen(np.asarray(iq).ravel(), L, offset=0)
        A = wf.project(S, wf.mode_basis(K, L, SEED, None))
        last, prev = np.abs(A[-1]) ** 2, np.abs(A[-2]) ** 2
        assert last[used:].max() < 1e-12 * prev.mean(), (
            f"{K - used} unused modes carry {last[used:].max():.3g}, not dark")
        # each row goes out at its own keyed power (row_radius): compare per unit radius
        r = wf.row_radius(L, SEED, rows=S.shape[0])
        share = (last.sum() / r[-1] ** 2) / (prev.sum() / r[-2] ** 2)
        assert abs(share - used / K) < 0.02, (
            f"last row draws {share:.3f} of a full row's power for {used}/{K} modes")


@pytest.mark.parametrize("bad", [complex(np.nan, 0), complex(np.inf, 0), complex(1.0, np.nan)])
def test_a_corrupted_capture_says_so_instead_of_reading_as_a_dead_band(bad):
    """One non-finite sample poisons every read that touches it.

    Before this guard the failure was illegible: the geometry score came back non-finite, nothing
    locked, and the caller was told only that there was no signal -- so a dead band and a
    corrupted recording looked identical.  On a bench that is the difference between retuning and
    rebuilding the front end.

    Deliberately not repaired.  Dropping the sample shifts every sample after it and destroys the
    timing the construction rests on; zeroing it is this package guessing what a caller's missing
    data meant."""
    K, L, SEED = 64, 64, 1234
    bits = np.random.default_rng(1).integers(0, 2, 2 * 200 * K)
    iq = np.asarray(wf.compose(bits, K, L, SEED, M=4)).copy()
    iq[500] = bad
    with pytest.raises(ValueError, match="non-finite"):
        wf.decode(iq, SEED)
    with pytest.raises(ValueError, match="non-finite"):
        wf.read(iq, L, K, SEED, M=4)


def test_the_guard_does_not_touch_a_clean_capture():
    K, L, SEED = 64, 64, 1234
    bits = np.random.default_rng(1).integers(0, 2, 2 * 200 * K)
    iq = wf.compose(bits, K, L, SEED, M=4)
    got, d = wf.decode(iq, SEED)
    n = min(len(bits), len(got))
    assert (d.L, d.K) == (L, K) and np.array_equal(got[:n], bits[:n])


@pytest.mark.parametrize("call,match", [
    (lambda: wf.screen(np.zeros(64, dtype=np.complex64), 0), "width must be positive"),
    (lambda: wf.screen(np.zeros(64, dtype=np.complex64), -8), "width must be positive"),
    (lambda: wf.compose(np.array([0, 2, 1]), 8, 8, 1234, M=4), "bits must be 0 or 1"),
    (lambda: wf.compose(np.array([0, 1, -1]), 8, 8, 1234, M=4), "bits must be 0 or 1"),
    (lambda: wf.mode_basis(9, 8, 1234), "0 < K <= L"),
    (lambda: wf.spread_operator(8, 9, 1234), "0 < D <= N"),
])
def test_a_contract_violation_says_which_one(call, match):
    """The guards are the 'fails loudly' half of the design, and coverage showed most were never
    exercised.  A silent wrong answer from a bad argument is the worst outcome this package has,
    so each guard is pinned to the message it owes the caller."""
    with pytest.raises(ValueError, match=match):
        call()


def test_an_empty_read_returns_empty_rather_than_guessing():
    """Degenerate inputs return nothing, not a fabricated answer."""
    K, L, SEED = 64, 64, 1234
    one_row = wf.compose(np.random.default_rng(1).integers(0, 2, 2 * K), K, L, SEED, M=4)[:L]
    assert wf.read(one_row, L, K, SEED, M=4).size == 0
    assert wf.read_soft(one_row, L, K, SEED).shape == (0, K)
    assert wf.embedded_dimension(np.zeros(0, dtype=np.complex64), SEED) == 0
