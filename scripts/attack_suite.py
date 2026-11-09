"""Keyless attacks on the shaped prototype, using only published methods.

This is the adversary side, so classic DSP and linear algebra are fair game here: the file is
what the waveform has to survive rather than part of it, and nothing in it is importable by
``entroptics_rf``.

Every attack is handed the subspace for free -- it knows K and L (the codec lights every mode, so
K = L and the subspace is the whole window; a sender with a spare mode would give it away through
the covariance rank) -- so what is tested is only whether the rotation inside that subspace can be
recovered.  That rotation is the key material.

  ACMA     van der Veen & Paulraj 1996.  Exact, non-iterative.  Needs the sources to be constant
           modulus.  Breaks the shaped prototype completely.
  FastICA  Hyvarinen.  Needs the sources to be non-Gaussian, since it maximizes kurtosis.  Fails
           against a Gaussian source by the ICA impossibility theorem rather than by difficulty.
  ILSP     Talwar, Viberg & Paulraj 1994.  Needs only a finite alphabet, so it is the attack a
           discrete Gaussian-shaped constellation still has to answer: it alternates
           nearest-alphabet projection with a Procrustes fit and needs no modulus assumption.

A recovery score near 1.0 means the stream was recovered.  The floor is not 0: an attack that
learns nothing still returns some rotation of the subspace, whose best accidental alignment with
a true stream is ``random_rotation_floor(K)``.  Every table prints that floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_root = Path(__file__).resolve()
while not (_root / "src" / "entroptics_rf").is_dir():        # the repo, found not assumed
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))



# ----------------------------------------------------------------- shared front end

def subspace(iq, L, K, offset=0):
    """Y = A R with R unitary: the rank leak, which no attack below has to work for."""
    x = np.asarray(iq)[offset:]
    T = len(x) // L
    X = x[:T * L].reshape(T, L)
    C = X.conj().T @ X / T
    _, EV = np.linalg.eigh(C)
    W = EV[:, ::-1][:, :K].conj().T
    Y = X @ W.conj().T
    return Y / np.sqrt(np.mean(np.abs(Y) ** 2))


def recovery(A_hat, A_true):
    """Phase- and permutation-invariant recovery, per recovered stream."""
    n = min(len(A_hat), len(A_true))
    P, Q = A_hat[:n].copy(), A_true[:n].copy()
    P -= P.mean(0); Q -= Q.mean(0)
    P /= np.linalg.norm(P, axis=0) + 1e-30
    Q /= np.linalg.norm(Q, axis=0) + 1e-30
    return np.abs(P.conj().T @ Q).max(1)


def random_rotation_floor(K, T, trials=8, rng=None):
    """What an attack that learned NOTHING scores, by construction."""
    rng = rng or np.random.default_rng(0)
    out = []
    for _ in range(trials):
        A = (rng.standard_normal((T, K)) + 1j * rng.standard_normal((T, K)))
        Q, _ = np.linalg.qr(rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K)))
        out.append(np.median(recovery(A @ Q, A)))
    return float(np.mean(out))


# ----------------------------------------------------------------- the attacks

def acma(Y, K):
    """Constant-modulus factorization. Exact when |a_tk| is constant."""
    T = len(Y)
    Q = np.einsum("ti,tj->tij", Y.conj(), Y)
    D = (Q[1:] - Q[0]).reshape(T - 1, K * K).conj()
    G = D.conj().T @ D
    _, EG = np.linalg.eigh(G)
    ns = EG[:, :K].T
    P = (ns.T @ np.random.default_rng(0).standard_normal(K)).reshape(K, K)
    P = P + P.conj().T
    _, V = np.linalg.eigh(P)
    return Y @ V


def fastica(Y, K, iters=200, rng=None):
    """Complex FastICA, kurtosis contrast. Recovers non-Gaussian sources; Gaussian ones never."""
    rng = rng or np.random.default_rng(0)
    C = Y.conj().T @ Y / len(Y)
    d, E = np.linalg.eigh(C)
    Z = Y @ (E / np.sqrt(np.maximum(d, 1e-12))).conj()          # white: E[z z^H] = I
    W = []
    for _ in range(K):
        w = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        w /= np.linalg.norm(w)
        for _ in range(iters):
            y = Z @ w
            g = np.abs(y) ** 2
            wn = (Z.conj() * (y * g)[:, None]).mean(0) - 2 * w   # kurtosis fixed point
            for u in W:                                          # deflate
                wn -= (u.conj() @ wn) * u
            nn = np.linalg.norm(wn)
            if nn < 1e-12:
                break
            wn /= nn
            if np.abs(np.abs(wn.conj() @ w) - 1) < 1e-10:
                w = wn
                break
            w = wn
        W.append(w)
    return Z @ np.array(W).conj().T


def ilsp(Y, K, alphabet, restarts=4, iters=60, rng=None):
    """Iterative least squares with projection: needs only that the alphabet be finite."""
    rng = rng or np.random.default_rng(0)
    alph = np.asarray(alphabet)
    alph = alph / np.sqrt(np.mean(np.abs(alph) ** 2))
    best, best_err = None, np.inf
    for _ in range(restarts):
        M = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        R, _ = np.linalg.qr(M)
        for _ in range(iters):
            A = Y @ R
            A = A / np.sqrt(np.mean(np.abs(A) ** 2))
            P = alph[np.abs(A[..., None] - alph).argmin(-1)]      # nearest alphabet point
            U, _, Vh = np.linalg.svd(Y.conj().T @ P)
            Rn = U @ Vh
            if np.allclose(Rn, R, atol=1e-10):
                R = Rn
                break
            R = Rn
        A = Y @ R
        A = A / np.sqrt(np.mean(np.abs(A) ** 2))
        err = np.abs(A - alph[np.abs(A[..., None] - alph).argmin(-1)]).mean()
        if err < best_err:
            best, best_err = A, err
    return best


def _amp_levels(n: int) -> np.ndarray:
    """n equiprobable Rayleigh quantiles at unit mean power -- the amplitude ladder a shaped
    constellation would use.  Inlined rather than imported: the adversary's toolbox must not
    depend on any particular transmitter design, least of all a shelved one."""
    if n == 1:
        return np.ones(1)
    q = (np.arange(n) + 0.5) / n
    r = np.sqrt(-np.log1p(-q))
    return r / np.sqrt(np.mean(r ** 2))


def alphabet_of(M_phase, N_amp):
    """The finite set the coefficients actually live on: a polar grid."""
    r = _amp_levels(N_amp)
    th = np.exp(2j * np.pi * np.arange(M_phase) / M_phase)
    return (r[:, None] * th[None, :]).ravel()


def gaussianity(A):
    """How far the coefficients are from circular complex Gaussian. All three -> 0 if Gaussian."""
    a = A.ravel()
    a = a / np.sqrt(np.mean(np.abs(a) ** 2))
    return (float(np.abs(np.mean(a ** 2))),                       # E[a^2]   (circularity)
            float(np.abs(np.mean(a ** 4))),                       # E[a^4]   (phase symmetry)
            float(np.mean(np.abs(a) ** 4) - 2.0))                 # kurtosis excess (0 if Gaussian)
