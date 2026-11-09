"""The security claim, pinned against the actual attacks rather than asserted.

The waveform is $X = AB$ with $A$ unit-modulus and $B$ orthonormal, which is exactly the input the
Analytical Constant Modulus Algorithm was written for.  ACMA does not attack the key: it solves
for the aperture from two facts that are properties of the design rather than of the seed, and in
an earlier form of this waveform it recovered every mode stream and an attacker holding those
coefficients read the payload at BER 0.0000.

So these tests do not check that the attack fails.  It does not fail -- given a capture longer
than K^2 rows ACMA recovers the coefficient matrix *exactly*, at every geometry measured, and
there is no way to prevent that, because constant modulus is what a differential phase alphabet
is.  What they check is that recovering it buys nothing: the keyed per-mode rotation of
``mode_keys`` leaves the attacker at chance on the payload even when recovery is 1.000.

That distinction is the whole security argument, and before these tests nothing in the suite
pinned it.  The attack code lives outside the shipped package on purpose -- it is the adversary's
toolbox, and must not become a dependency of the transmitter.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from entroptics_rf import waveform as wf
from entroptics_rf.runner import ber

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import attack_suite as atk                                             # noqa: E402

SEED = 1234
ROWS = 400


def _capture(K, L, rows=ROWS, seed=SEED):
    bits = np.random.default_rng(5).integers(0, 2, 2 * rows * K)
    return bits, wf.compose(bits, K, L, seed, M=4)


def _coefficients(iq, K, L, T, seed=SEED):
    """What the key holder actually sent, in coefficient form -- the thing an attack aims at."""
    S = wf.screen(np.asarray(iq).ravel(), L, offset=0, max_rows=T)
    return S @ wf.mode_basis(K, L, seed, None).conj().T


def _best_payload(A_hat, bits):
    """The best the attacker can do with recovered coefficients, over every constant de-rotation.

    ACMA resolves the aperture only up to a per-mode constant phase, so the attacker is entitled
    to try them all.  A constant phase is exactly what a differential cancels, so the keyed
    rotation changes every row and this search has nothing constant to cancel."""
    d = A_hat[1:] * np.conj(A_hat[:-1])
    best = 1.0
    for rot in np.exp(2j * np.pi * np.arange(32) / 32):
        idx = wf._slice_psk(d * rot, 4).ravel()
        got = ((idx[:, None] >> np.arange(1, -1, -1)) & 1).astype(np.int64).ravel()
        n = min(len(bits), len(got))
        best = min(best, ber(bits[:n], got[:n]))
    return best


def test_acma_recovers_the_coefficients_and_it_does_not_help():
    """The concentration was never the thing being hidden; the message is.

    At K = L = 16 the capture is 400 rows against K^2 = 256, so ACMA is above its sample-size
    threshold and recovers exactly.  Measured 1.000 against a random-rotation floor of ~0.45.
    The attacker holding that exact answer still reads the payload at chance."""
    K = L = 16
    bits, iq = _capture(K, L)
    Y = atk.subspace(iq, L, K)
    A_hat = atk.acma(Y, K)
    A_true = _coefficients(iq, K, L, len(Y))

    recovered = float(np.median(atk.recovery(A_hat, A_true)))
    floor = atk.random_rotation_floor(K, len(Y))
    assert recovered > floor + 0.3, (
        f"ACMA no longer recovers ({recovered:.3f} vs floor {floor:.3f}) -- if the wire format "
        "changed this is worth knowing, but the security claim below is what matters")

    payload = _best_payload(A_hat, bits)
    assert payload > 0.45, (
        f"BROKEN: an attacker with the exact coefficients read the payload at BER {payload:.4f}")


def test_ilsp_is_no_better_than_acma():
    """ILSP needs only a finite alphabet, so it is the attack a keyed constellation must survive."""
    K = L = 16
    bits, iq = _capture(K, L)
    Y = atk.subspace(iq, L, K)
    A_hat = atk.ilsp(Y, K, atk.alphabet_of(4, 1))
    payload = _best_payload(A_hat, bits)
    assert payload > 0.45, f"BROKEN: ILSP read the payload at BER {payload:.4f}"


@pytest.mark.parametrize("K,L", [(16, 16), (64, 64)])
def test_the_key_holder_is_unaffected(K, L):
    """The defence costs the receiver nothing: told the geometry, it reads the payload exactly."""
    bits, iq = _capture(K, L)
    got = wf.read(iq, L, K, SEED, M=4)
    n = min(len(bits), len(got))
    assert ber(bits[:n], got[:n]) == 0.0


def test_without_the_key_there_is_no_constellation_to_slice():
    """An eavesdropper cannot even measure M.

    The keyed rotation scatters the differential uniformly, so the closure that identifies the
    alphabet reads ~0 without the key and ~1 with it.  This is the same read the receiver uses to
    pick M, which is why an attacker cannot shortcut to it."""
    K = L = 64
    _, iq = _capture(K, L)
    T = len(np.asarray(iq).ravel()) // L
    A = _coefficients(iq, K, L, T)
    d = A[1:] * np.conj(A[:-1])

    def closure(z):
        u = z / np.maximum(np.abs(z), 1e-30)
        return float(abs(np.mean(u ** 4)))

    without = closure(d)
    with_key = closure(d * np.conj(wf.mode_keys(K, SEED, None, first=0, rows=d.shape[0])))
    assert with_key > 0.9, f"the key holder should see a constellation, saw {with_key:.3f}"
    assert without < 0.1, f"LEAK: M is readable without the key (closure {without:.3f})"


def test_recovery_needs_a_capture_longer_than_k_squared():
    """The factorization has a sample-size threshold, and it is not the defence.

    ACMA's design matrix is K^2-dimensional, so recovery turns on at T ~ K^2 -- measured, K = 32
    flips between T/K^2 = 0.78 and 1.47.  This test pins the threshold so that nobody mistakes a
    short capture for security: collect more and the recovery arrives.  The payload does not."""
    K = L = 32
    short_bits, short_iq = _capture(K, L, rows=400)          # T/K^2 = 0.39
    Ys = atk.subspace(short_iq, L, K)
    short = float(np.median(atk.recovery(atk.acma(Ys, K), _coefficients(short_iq, K, L, len(Ys)))))
    floor = atk.random_rotation_floor(K, len(Ys))
    assert short < floor + 0.15, f"expected no recovery below K^2 rows, got {short:.3f}"

    long_bits, long_iq = _capture(K, L, rows=1500)           # T/K^2 = 1.47
    Yl = atk.subspace(long_iq, L, K)
    A_hat = atk.acma(Yl, K)
    long = float(np.median(atk.recovery(A_hat, _coefficients(long_iq, K, L, len(Yl)))))
    assert long > 0.9, f"expected exact recovery above K^2 rows, got {long:.3f}"

    # and it still buys nothing
    assert _best_payload(A_hat, long_bits) > 0.45, "BROKEN: recovery above K^2 rows read the payload"


def _cond(x, W):
    """lambda_max / lambda_min of the covariance over windows of W samples."""
    x = np.asarray(x).ravel()
    T = len(x) // W
    X = x[:T * W].reshape(T, W)
    ev = np.maximum(np.linalg.eigvalsh(X.conj().T @ X / T), 0.0)
    pos = ev[ev > 0]
    return float(ev.max() / pos.min()) if pos.size else float("inf")


def _spare_mode_sender(K, L, rows, seed=SEED, rng=1):
    """What a sender leaving L - K modes dark would put on the air -- built here, because the codec
    refuses to (``compose`` raises unless K = L).

    Unit-modulus coordinates on the first K rows of the keyed aperture, each row at its keyed
    radius: the second-order structure of the old K < L waveform exactly (the differential walk
    and the keyed rotation are unit modulus too, so they change nothing a covariance sees).  It is
    the reference for what a spare mode leaks (research/rank_leak.md)."""
    w = np.exp(2j * np.pi * np.random.default_rng(rng).random((rows + 1, K)))
    S = (w @ wf.mode_basis(K, L, seed)) * wf.row_radius(L, seed, rows=rows + 1)[:, None]
    return wf.to_iq(S)


def test_the_rank_leak_is_a_comb_and_K_equals_L_closes_it():
    """The geometry leak of RF-5.3, and the pin that the shipped waveform has closed it.

    A screen occupying K of every L dimensions also occupies 2K of every 2L, so the covariance is
    rank-deficient at *every multiple* of L rather than at L alone -- an observer sweeping the
    window sees a repeating tell whose spacing is L itself.  Off the comb it is indistinguishable
    from noise.  A sender with a spare mode (built test-side; the codec refuses K < L) shows the
    comb; the codec's own compose lights every mode and must show nothing at L, 2L or off the comb
    (research/rank_leak.md, cure (e)).  If this fails on the codec half, a dark mode is back."""
    K, L = 60, 64
    rows = 1500
    iq = _spare_mode_sender(K, L, rows)

    on_comb = [_cond(iq, w) for w in (L, 2 * L)]
    off_comb = [_cond(iq, w) for w in (L // 2, L // 4, 3 * L // 4)]
    assert min(on_comb) > 1e4, f"the comb should be a decisive tell, got {on_comb}"
    assert max(off_comb) < 10, f"off the comb should look like noise, got {off_comb}"

    # the codec will not write a spare mode at all
    with pytest.raises(ValueError):
        wf.compose(np.zeros(2 * rows * K, dtype=np.int64), K, L, SEED, M=4)

    # K = L removes the emptiness, and with it the leak: the shipped waveform
    full = np.random.default_rng(5).integers(0, 2, 2 * rows * L)
    iq_full = wf.compose(full, L, L, SEED, M=4)
    codec = [_cond(iq_full, w) for w in (L, 2 * L, L // 2, L // 4, 3 * L // 4)]
    assert max(codec) < 10, f"K = L should close the leak, got {codec}"


def _deficiency(x, W, off):
    """How many eigenvalues sit at the numerical floor: the measured rank deficiency.

    The spectrum separates by about five orders of magnitude -- null directions land near 1e-8
    relative (the capture is complex64) and live ones no lower than 5e-3 -- so any cutoff between
    1e-7 and 1e-3 returns the same integer.  This is reading a gap, not choosing a threshold."""
    x = np.asarray(x).ravel()[off:]
    T = len(x) // W
    ev = np.maximum(np.linalg.eigvalsh(
        (lambda X: X.conj().T @ X / T)(x[:T * W].reshape(T, W))), 0.0)
    return int(np.sum(ev < 1e-5 * ev.max()))


@pytest.mark.parametrize("K,L", [(56, 64), (60, 64), (62, 64), (63, 64)])
def test_the_rank_deficiency_is_exactly_the_spare_dimension_less_the_offset(K, L):
    """def(o) = max(0, L - K - o), and nothing about it is approximate.

    A window of L samples at offset o spans the tail of one row and the head of the next, so it
    lives in a space of dimension min(L, K + o) and is deficient only while K + o < L.  This fixes
    the leak's entire extent -- it is a comb in the window and a band of exactly 2(L-K)-1 offsets
    wide -- and shows that the emptiness the dark-mode read needs is the same emptiness that
    leaks.  Pinned because it is the sharpest statement available about what an unkeyed observer
    can recover from a spare mode: even K = L-1 leaks, which is why the codec lights every mode and
    finds the geometry by the constant-modulus read instead (research/rank_leak.md).  The spare-mode
    screens are built test-side (the codec refuses K < L); the codec's own K = L screen must be
    full rank at every offset."""
    iq = _spare_mode_sender(K, L, 2000)
    spare = L - K
    got = [_deficiency(iq, L, o) for o in range(spare + 2)]
    assert got == [max(0, spare - o) for o in range(spare + 2)], (
        f"K={K} L={L}: deficiency by offset {got}, expected "
        f"{[max(0, spare - o) for o in range(spare + 2)]}")

    leaking = sum(1 for o in range(L) if _deficiency(iq, L, o) > 0)
    assert leaking == 2 * spare - 1, f"{leaking} leaking offsets, expected {2 * spare - 1}"

    bits = np.random.default_rng(1).integers(0, 2, 2 * 2000 * L)
    shipped = wf.compose(bits, L, L, SEED, M=4)
    codec = [_deficiency(shipped, L, o) for o in range(spare + 2)]
    assert codec == [0] * (spare + 2), f"the K = L screen is rank-deficient by offset: {codec}"


def test_seed_reuse_is_a_two_time_pad_on_the_payload():
    """Reusing a seed leaks the difference of the two payloads, not merely the geometry.

    The keyed per-row rotation is the whole payload defence, and it is a function of the seed, so
    two messages under one seed carry the same rotation.  An observer holding both forms
    (d1.rho)/(d2.rho) = d1/d2 and it divides out -- no key, no search.

    A fresh key per transmission is a precondition of the construction, supplied by key management
    above this layer -- this test does not report a defect.  It pins the precondition, so that the
    cost of violating it stays measured rather than assumed, and so that anyone weakening
    ``mode_keys`` later finds out here what they have actually weakened."""
    K = L = 64
    b1 = np.random.default_rng(11).integers(0, 2, 2 * 300 * K)
    b2 = np.random.default_rng(22).integers(0, 2, 2 * 300 * K)
    same_a = wf.compose(b1, K, L, SEED, M=4)
    same_b = wf.compose(b2, K, L, SEED, M=4)
    T = len(np.asarray(same_a).ravel()) // L

    def differential(iq, key):
        S = wf.screen(np.asarray(iq).ravel(), L, offset=0, max_rows=T)
        A = wf.project(S, wf.mode_basis(K, L, key, None))
        return A[1:] * np.conj(A[:-1])

    def closes(x, y):
        r = x * np.conj(y) / np.maximum(np.abs(y) ** 2, 1e-30)
        u = r / np.maximum(np.abs(r), 1e-30)
        return float(abs(np.mean(u ** 4)))

    d1, d2 = differential(same_a, SEED), differential(same_b, SEED)
    assert closes(d1, d2) > 0.9, (
        "seed reuse should cancel the rotation -- if this now fails the defence changed, and "
        "RF-3.1 needs re-reading rather than this test relaxing")

    # a fresh seed leaves nothing to divide out
    other = wf.compose(b2, K, L, SEED + 1, M=4)
    d3 = differential(other, SEED)
    assert closes(d1, d3) < 0.2, "two distinct seeds must not cancel"
