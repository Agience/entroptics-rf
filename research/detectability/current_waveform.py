"""The ten-detector blind battery and the fourth-order observer, on the current waveform, vs SNR.

What a key-free observer sees of the waveform ``compose`` writes today (K = L = 64, DQPSK, keyed
row radius), against conventional QPSK, 16-QAM and DSSS run through the same battery.

Battery (``scripts/evidence_adversary.py``, imported, not copied): spectral flatness, PSD
peak/mean (PAR), symbol-rate line, x^2 and x^4 spectra, autocorrelation peak, amplitude kurtosis,
|C40| cumulant, max/min eigenvalue (window 16), gzip ratio.  Every one is scale-invariant, so the
observer does not need the noise floor; a radiometer, which does, is a separate question
(README.md).

Per cell (signal, SNR, capture length): ``reps`` traces of signal + complex Gaussian noise at that
SNR, each scored against ``nulls`` noise-only traces of the same length.  A trace is a hit when its
statistic falls outside the null's central 99% (two-sided, P_fa = 1%).  A detector *detects* a
cell when its hits exceed what P_fa alone gives with probability < 1e-3 (binomial).  The
ensemble z of the earlier battery (mean over traces against the null's per-trace spread) is
reported as well.

Fourth-order observer (``--cum4``, ``research/rank_leak.md`` §3's statistic): the cumulant
matrix of whitened L-sample windows, scored -(sum of its L most negative eigenvalues), on 417 ms
captures (1 000 800 samples at 2.4 MS/s).  Side information is varied: width and row offset given
(``aligned``), width given and offset wrong by L/2 (``half``), width wrong by one (``w63``).  No
side information at all is the |C40| detector of the battery.

    OMP_NUM_THREADS=4 python research/detectability/current_waveform.py            # battery, ~5 min
    OMP_NUM_THREADS=4 python research/detectability/current_waveform.py --cum4     # adds ~15 min

Writes ``current_waveform.json`` beside this file and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from entroptics_rf import waveform as wf                       # noqa: E402
import evidence_adversary as ea                                # noqa: E402

SNRS = (-10.0, 0.0, 10.0, 20.0, None)                         # None: no receiver noise
PFA = 0.01
FAMILY = 1e-3


def keyed(n, rng):
    return ea.entroptics(n, rng)                               # fresh key, fresh payload


def public(n, rng):
    return ea.entroptics(n, rng, seed=None)


KINDS = [("entroptics keyed", keyed), ("entroptics open", public),
         ("QPSK", lambda n, r: ea.linear_mod(n, r, 4)),
         ("16-QAM", lambda n, r: ea.linear_mod(n, r, 16)),
         ("DSSS", ea.dsss)]


def at_snr(x, snr, rng):
    if snr is None:
        return x
    s = 10.0 ** (-snr / 20.0)
    y = x + s * ea.noise(x.size, rng)
    return (y / np.sqrt(np.mean(np.abs(y) ** 2))).astype(np.complex64)


def k_crit(reps: int, p: float = PFA, alpha: float = FAMILY) -> int:
    """Fewest hits out of ``reps`` that P_fa alone reaches with probability < alpha."""
    tail = 1.0
    for k in range(reps + 1):
        if tail < alpha:
            return k
        tail -= math.comb(reps, k) * p ** k * (1 - p) ** (reps - k)
    return reps + 1


def battery(n: int, reps: int, nulls: int, seed: int = 20261030) -> dict:
    rng = np.random.default_rng(seed)
    null_traces = [ea.noise(n, rng) for _ in range(nulls)]
    null = {d: np.array([f(t) for t in null_traces]) for d, f in ea.DETECTORS}
    lo = {d: np.quantile(v, PFA / 2) for d, v in null.items()}
    hi = {d: np.quantile(v, 1 - PFA / 2) for d, v in null.items()}
    kc = k_crit(reps)
    cells = []
    for kind, gen in KINDS:
        base = [gen(n, rng) for _ in range(reps)]
        for snr in SNRS:
            traces = [at_snr(x, snr, rng) for x in base]
            for d, f in ea.DETECTORS:
                v = np.array([f(t) for t in traces])
                hits = int(np.sum((v < lo[d]) | (v > hi[d])))
                z = float((v.mean() - null[d].mean()) / (null[d].std() or 1e-12))
                cells.append(dict(n=n, kind=kind, snr=snr, detector=d, hits=hits, reps=reps,
                                  detected=hits >= kc, z=z))
    return dict(n=n, reps=reps, nulls=nulls, k_crit=kc, cells=cells)


# ------------------------------------------------------------------- fourth order, given geometry

def c4_stat(x, q: int, offset: int) -> float:
    """-(sum of the q most negative eigenvalues) of the whitened fourth-order cumulant matrix of
    q-sample windows starting at ``offset`` (rank_leak.md §3; lab ``cum4.py``)."""
    T = (x.size - offset) // q
    X = x[offset: offset + T * q].reshape(T, q).astype(np.complex64)
    R = (X.T @ X.conj()).astype(complex) / T
    w, U = np.linalg.eigh(R)
    Wh = (U / np.sqrt(w)[None, :]).conj().astype(np.complex64)
    M = np.zeros((q * q, q * q), dtype=np.complex64)
    for a in range(0, T, 2048):
        Z = X[a:a + 2048] @ Wh
        V = (Z[:, :, None] * Z[:, None, :].conj()).reshape(-1, q * q)
        M += V.T @ V.conj()
    M = M.astype(complex) / T
    eye = np.eye(q).ravel()
    C = M - np.outer(eye, eye) - np.eye(q * q)
    ev = np.linalg.eigvalsh((C + C.conj().T) / 2)
    return float(-ev[:q].sum())


def cum4(L: int = 64, n: int = 1_000_800, snrs=(0.0, 10.0, 20.0), sigs: int = 2,
         nulls: int = 3, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    rows = n // L + 8
    out = []

    def rec(kind, snr, stat, v, secs):
        r = dict(kind=kind, snr=snr, stat=stat, value=v, secs=round(secs, 1))
        out.append(r)
        print(json.dumps(r), flush=True)

    for i in range(nulls):
        y = ea.noise(n, rng)
        for stat, q, off in (("aligned", L, 0), ("half", L, L // 2), ("w63", L - 1, 0)):
            t0 = time.time()
            rec("null", None, stat, c4_stat(y, q, off), time.time() - t0)
    for snr in snrs:
        for i in range(sigs):
            k = int(rng.integers(1, 2 ** 31))
            x = wf.compose(rng.integers(0, 2, 2 * L * rows), L, L, k, M=4)[:n]
            y = at_snr((x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64), snr, rng)
            stats = (("aligned", L, 0),) if snr != max(snrs) else (
                ("aligned", L, 0), ("half", L, L // 2), ("w63", L - 1, 0))
            for stat, q, off in stats:
                t0 = time.time()
                rec("entroptics keyed", snr, stat, c4_stat(y, q, off), time.time() - t0)
            t0 = time.time()
            rec("entroptics keyed", snr, "C40 (no side info)", ea.d_c40(y), time.time() - t0)
    # the per-sample |C40| null at this length, for the no-side-information row
    for i in range(max(nulls, 8)):
        y = ea.noise(n, rng)
        rec("null", None, "C40 (no side info)", ea.d_c40(y), 0.0)
    return dict(L=L, n=n, records=out)


def summarise_cum4(c: dict) -> list[str]:
    recs = c["records"]
    lines = ["| statistic | SNR dB | z per capture, against noise-only captures |", "|---|---|---|"]
    stats = sorted({r["stat"] for r in recs}, key=lambda s: ["aligned", "half", "w63"].index(s)
                   if s in ("aligned", "half", "w63") else 9)
    for s in stats:
        nv = np.array([r["value"] for r in recs if r["kind"] == "null" and r["stat"] == s])
        mu, sd = nv.mean(), nv.std(ddof=1) or 1e-12
        for snr in sorted({r["snr"] for r in recs if r["stat"] == s and r["kind"] != "null"}):
            zs = [(r["value"] - mu) / sd for r in recs
                  if r["stat"] == s and r["kind"] != "null" and r["snr"] == snr]
            lines.append(f"| {s} | {snr:+.0f} | " + ", ".join(f"{z:.1f}" for z in zs)
                         + f" (nulls {len(nv)}) |")
    return lines


def table(b: dict) -> list[str]:
    snr_lbl = ["clean" if s is None else f"{s:+.0f}" for s in SNRS]
    lines = [f"n = {b['n']} samples, {b['reps']} traces per cell, {b['nulls']} null traces, "
             f"P_fa {PFA:.0%} per trace, detected = hits >= {b['k_crit']}", ""]
    for kind, _ in KINDS:
        lines += [f"**{kind}** -- hits/{b['reps']} (ensemble z) per SNR dB", "",
                  "| detector | " + " | ".join(snr_lbl) + " |",
                  "|---|" + "---|" * len(SNRS)]
        for d, _ in ea.DETECTORS:
            row = []
            for s in SNRS:
                c = next(c for c in b["cells"] if c["kind"] == kind and c["detector"] == d
                         and c["snr"] == s)
                z = f"{c['z']:+.1f}" if abs(c["z"]) < 1e4 else f"{c['z']:+.1e}"
                row.append(f"{'**' if c['detected'] else ''}{c['hits']}{'**' if c['detected'] else ''} ({z})")
            lines.append(f"| {d} | " + " | ".join(row) + " |")
        cnt = [sum(c["detected"] for c in b["cells"] if c["kind"] == kind and c["snr"] == s)
               for s in SNRS]
        lines += ["| **detectors that detect** | " + " | ".join(f"{k}/10" for k in cnt) + " |", ""]
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=32)
    p.add_argument("--cum4", action="store_true", help="also run the fourth-order observer")
    p.add_argument("--out", default=str(HERE / "current_waveform.json"))
    a = p.parse_args(argv)

    res = {"waveform": "compose K=L=64 M=4 (DQPSK), keyed row radius", "battery": []}
    prev = Path(a.out)
    if prev.exists():
        res = json.loads(prev.read_text())
    res["battery"] = []
    for n, nulls in ((16384, 400), (262144, 200)):
        t0 = time.time()
        b = battery(n, a.reps, nulls)
        res["battery"].append(b)
        print("\n".join(table(b)), f"\n({time.time() - t0:.0f} s)\n", flush=True)
    if a.cum4:
        res["cum4"] = cum4()
    if "cum4" in res:
        print("\n".join(summarise_cum4(res["cum4"])))
    prev.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
