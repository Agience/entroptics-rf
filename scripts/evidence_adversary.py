"""Claim 2 -- to a receiver using any conventional method, the waveform is noise.

This file is the adversary rather than the system, so it is full of FFTs, matched filters and
pulse shaping.  It stands outside the package for that reason -- ``test_no_fft`` scans
``src/entroptics_rf/*.py`` -- and nothing here is imported by the waveform.

A battery of null results proves nothing unless the same battery fires on something, so every
detector is run against positive controls -- QPSK, 16-QAM and DSSS -- built the conventional way.
A detector that cannot see QPSK is broken, and its silence on entroptics would mean nothing.

Each statistic is z-scored against a null distribution built from independent Gaussian noise
traces, so "looks like noise" is a measured distance in units of the noise's own spread rather
than an eyeball judgement.  |z| < 3 is indistinguishable from noise at that sample size.

    python scripts/evidence_adversary.py
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve()
while not (_root / "src" / "entroptics_rf").is_dir():        # the repo, found not assumed
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))

from entroptics_rf import waveform as wf                             # noqa: E402
from entroptics_rf.runner import Link, random_bits                   # noqa: E402


# ----------------------------------------------------------------- the signals under test

def rrc(beta: float, sps: int, span: int = 8) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
        den = np.pi * t * (1 - (4 * beta * t) ** 2)
        h = num / den
    h[t == 0] = 1 - beta + 4 * beta / np.pi
    edge = np.abs(np.abs(4 * beta * t) - 1) < 1e-8
    h[edge] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return h / np.sqrt(np.sum(h ** 2))


def linear_mod(n: int, rng, order: int, sps: int = 4, beta: float = 0.35) -> np.ndarray:
    """Textbook pulse-shaped linear modulation -- what every blind analyser is built to find."""
    if order == 4:
        sym = np.exp(1j * (np.pi / 4 + rng.integers(0, 4, n // sps + 32) * np.pi / 2))
    else:
        m = int(np.sqrt(order))
        lv = np.arange(-(m - 1), m, 2)
        sym = rng.choice(lv, n // sps + 32) + 1j * rng.choice(lv, n // sps + 32)
    up = np.zeros(len(sym) * sps, dtype=complex)
    up[::sps] = sym
    x = np.convolve(up, rrc(beta, sps))[:n]
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)


def dsss(n: int, rng, chips: int = 31, sps: int = 2) -> np.ndarray:
    """Direct-sequence spread spectrum: already noise-like to the eye, still cyclostationary."""
    code = rng.choice([-1.0, 1.0], chips)
    bits = rng.choice([-1.0, 1.0], n // (chips * sps) + 4)
    c = np.repeat(np.concatenate([b * code for b in bits]), sps)
    c = c + 1j * np.roll(c, chips * sps // 2)
    x = np.convolve(c, rrc(0.35, sps))[:n]             # chip filtering, as any real DSSS link has
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)


def noise(n: int, rng) -> np.ndarray:
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n))
            / np.sqrt(2)).astype(np.complex64)


def entroptics(n: int, rng, K=64, L=64, M=4, seed="fresh") -> np.ndarray:
    """A fresh key and fresh payload every trace, so nothing below can key off one realization."""
    seed = int(rng.integers(1, 2 ** 31)) if isinstance(seed, str) else seed
    link = Link(K=K, L=L, seed=seed, M=M)
    bits = random_bits(int(round(np.log2(M))) * (n // L + 8) * K,
                       seed=int(rng.integers(1, 2 ** 31)))
    x = wf.compose(bits, link.K, link.L, link.seed, M=link.M)[:n]
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)


# ----------------------------------------------------------------- the adversary's toolbox

def _pkmean(v: np.ndarray) -> float:
    v = np.abs(v)
    v = v[1:len(v) // 2]                               # drop DC, keep one side
    m = v.mean()
    return float(v.max() / m) if m > 0 and np.isfinite(m) else 0.0


def d_flatness(x):
    """Spectral flatness. 1.0 == white. The first thing anyone looks at."""
    n = 1024
    f = x[:(len(x) // n) * n].reshape(-1, n)
    psd = (np.abs(np.fft.fft(f, axis=1)) ** 2).mean(0)
    psd /= psd.mean()
    return float(np.exp(np.log(psd + 1e-30).mean()))


def d_psd_peak(x):
    """Carrier / tone detection: peak-to-mean of the PSD."""
    n = 1024
    f = x[:(len(x) // n) * n].reshape(-1, n)
    return _pkmean((np.abs(np.fft.fft(f, axis=1)) ** 2).mean(0))


def d_symbol_line(x):
    """Spectral line in |x|^2 at the symbol rate -- the classic symbol-rate estimator."""
    p = np.abs(x) ** 2
    return _pkmean(np.fft.fft(p - p.mean()))


def d_x2(x):
    """Spectrum of x^2: exposes BPSK and residual-carrier structure."""
    return _pkmean(np.fft.fft(x ** 2))


def d_x4(x):
    """Spectrum of x^4: the standard QPSK carrier and cyclostationary grab."""
    return _pkmean(np.fft.fft(x ** 4))


def d_autocorr(x):
    """Largest off-zero autocorrelation: repetition, preambles, spreading codes."""
    n = 1 << int(np.ceil(np.log2(len(x) * 2)))
    f = np.fft.fft(x, n)
    r = np.fft.ifft(np.abs(f) ** 2)[:len(x) // 4]
    r = np.abs(r / r[0])
    return float(r[1:].max())


def d_kurtosis(x):
    """Amplitude kurtosis. 3.0 == Gaussian. Modulations sit away from 3."""
    a = np.abs(x)
    sd = a.std()
    return float((((a - a.mean()) / sd) ** 4).mean()) if sd > 0 else 0.0


def d_c40(x):
    """Normalized |C40| cumulant -- a core automatic-modulation-classification feature."""
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    m40, m20 = np.mean(x ** 4), np.mean(x ** 2)
    return float(np.abs(m40 - 3 * m20 ** 2))


def d_mme(x, L=16):
    """Max/min eigenvalue of the covariance -- blind spectrum sensing with no signal knowledge."""
    n = (len(x) // L) * L
    X = x[:n].reshape(-1, L)
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    return float(ev.max() / max(ev.min(), 1e-12))


def d_gzip(x):
    """Can a general-purpose compressor find any structure at all? 1.0 == none."""
    q = np.stack([x.real, x.imag], -1).ravel()
    q = np.clip(np.round(q / (4 * q.std()) * 127), -127, 127).astype(np.int8).tobytes()
    return len(gzip.compress(q, 9)) / len(q)


DETECTORS = [
    ("spectral flatness", d_flatness), ("PSD peak/mean", d_psd_peak),
    ("symbol-rate line", d_symbol_line), ("x^2 spectrum", d_x2), ("x^4 spectrum", d_x4),
    ("autocorr peak", d_autocorr), ("amp kurtosis", d_kurtosis), ("|C40| cumulant", d_c40),
    ("eigenvalue MME", d_mme), ("gzip ratio", d_gzip),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1 << 14, help="samples per trace")
    p.add_argument("--reps", type=int, default=24, help="independent traces per signal type")
    a = p.parse_args(argv)

    kinds = [("noise (null)", noise), ("entroptics", entroptics),
             ("ent 1 key", lambda n, r: entroptics(n, r, seed=1234)),
             ("ent public", lambda n, r: entroptics(n, r, seed=None)),
             ("QPSK", lambda n, r: linear_mod(n, r, 4)),
             ("16-QAM", lambda n, r: linear_mod(n, r, 16)),
             ("DSSS", dsss)]

    vals = {}
    for name, gen in kinds:
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
        traces = [gen(a.n, rng) for _ in range(a.reps)]
        vals[name] = {d: np.array([f(t) for t in traces]) for d, f in DETECTORS}

    print(f"{a.reps} independent traces of {a.n} samples per signal type.")
    print("z = distance from the noise null, in units of the null's own standard deviation.")
    print("|z| < 3 means the detector cannot tell it from noise.\n")

    hdr = f"{'detector':>18} | {'noise mean':>10} | " + " | ".join(f"{k:>9}" for k, _ in kinds[1:])
    print(hdr)
    print("-" * len(hdr))
    for dname, _ in DETECTORS:
        null = vals["noise (null)"][dname]
        mu, sd = null.mean(), (null.std() or 1e-12)
        cells = []
        for k, _ in kinds[1:]:
            z = (vals[k][dname].mean() - mu) / sd
            cells.append(f"{z:>+9.1f}" if abs(z) < 1e4 else f"{z:>+9.1e}")
        print(f"{dname:>18} | {mu:>10.4g} | " + " | ".join(cells))
    print("\n(the noise column is a raw mean; every other column is a z-score against it)")

    print("\nbest detector against each signal, over all ten:")
    for k, _ in kinds[1:]:
        zs = [abs((vals[k][d].mean() - vals["noise (null)"][d].mean())
                  / (vals["noise (null)"][d].std() or 1e-12)) for d, _ in DETECTORS]
        best = DETECTORS[int(np.argmax(zs))][0]
        flag = "   <== INDISTINGUISHABLE FROM NOISE" if max(zs) < 3 else "   <== detected"
        print(f"  {k:>10}: max |z| = {max(zs):>12.1f}  (via {best}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
