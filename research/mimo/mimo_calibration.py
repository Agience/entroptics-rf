"""Does the blind keyed-MVDR array need per-boot PHASE CALIBRATION, or only a shared clock?

Extends the GO result in `spatial_mimo.py` / `spatial_mimo.md` (an N-antenna receiver
blindly separates min(N,K) co-channel keyed streams, steering each beam by maximizing that
key's constant-modulus lock -- matching the genie zero-forcing ceiling with NO pilots, NO
angles). That receiver steers by the KEY, not by an absolute angle, so -- unlike a
direction-finder such as a KrakenSDR -- it may not need the per-boot phase-calibration rig
(noise source + RF switch) that an angle-based array requires. This study measures that.

A real cheap coherent RTL-SDR array has two separate imperfections:

  (1) STATIC per-element miscalibration. Each RX chain has an unknown but STABLE complex gain
      g_k * exp(j*phi_k) (cable lengths, mixer phase, per-dongle group delay). A direction
      finder must calibrate these out per boot, because it maps measured phase -> absolute
      angle. The actual array manifold is  A_actual = diag(g_k e^{j phi_k}) @ A_nominal.
      Experiment 1 asks: does the BLIND keyed-MVDR still separate under unknown static
      per-element errors? Contrast with a genie zero-forcer that uses the NOMINAL (stale /
      uncalibrated) steering vectors -- the angle-based receiver's situation -- which should
      degrade. A genie_actual (knows the true per-element manifold) is the immune upper bound,
      showing the capacity is still there and the issue is purely one of calibration knowledge.

  (2) COHERENCE loss (shared-clock stability). If the elements do NOT share a perfect clock,
      the per-element phase DRIFTS within a single capture (random walk). This smears the
      sample covariance the MVDR is built from. Experiment 2 sweeps the end-to-end RMS phase
      drift over one capture and measures where separation fails -- the real shared-clock spec.

Why the hypothesis for (1): the blind beam is chosen by the key's lock, not by absolute angle,
so a wrong absolute angle is fine as long as the chosen WEIGHT separates the streams. For N = 2
a static per-element phase error is exactly a re-labelling of the search angle (the actual
signature [1, (g1/g0) e^{j(dphi + pi sin th)}] is still reachable by the nominal 2-element
manifold when amplitudes match), so blind should be fully immune to phase-only error. For N > 2
a non-linear per-element phase profile is NOT reachable by any nominal angle, so immunity is not
guaranteed and must be measured -- the finding of this script.

Model recap (from spatial_mimo): ULA, d = lambda/2, element k phase exp(j*pi*k*sin theta);
K equal-power co-channel single-antenna keyed transmitters at distinct angles; per-element AWGN.
Calibration error is injected as an N-element multiplicative term on the received columns:
    Y = (A @ S) * (G[:,None] * exp(1j * Phi))  +  AWGN
with Phi static per element (Exp 1) or a per-sample random walk per element (Exp 2).

Public library only: compose, decode, read, ber, random_bits, constant_modulus_read.
Array/front-end helpers (steering, MVDR, blind_separate) are reused verbatim from spatial_mimo.

SIM ONLY. No hardware, no transmit. Re-runnable:
    OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src \
        .venv/Scripts/python.exe research/mimo/mimo_calibration.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Reuse the proven array-channel + blind-MVDR helpers from the sibling study.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_mimo import (  # noqa: E402
    steering, source_angles, blind_separate, bits_for,
    L_DEFAULT, T_ROWS, CANDIDATES, FEC_THR,
)
from entroptics_rf import compose, decode, read, ber, random_bits  # noqa: E402

TRIALS = 3                # independent (key, bit, error) seed sets per cell

# Two proven operating points from spatial_mimo (the robust ones):
#   N=2, K=2, dtheta=60  -- the N=2 sweet spot (steering vectors orthogonal near +-30 deg)
#   N=4, K=2, dtheta=30  -- oversized array (2 spare elements), robust to close angle / low SNR
CONFIGS = [
    {"N": 2, "K": 2, "dtheta": 60.0, "tag": "N=2 K=2 dth=60 (sweet spot)"},
    {"N": 4, "K": 2, "dtheta": 30.0, "tag": "N=4 K=2 dth=30 (oversized)"},
]
SNRS = [None, 12]                         # clean and 12 dB per-source per-element

# Experiment 1 sweeps
PHASE_RMS = [0.0, 5.0, 10.0, 20.0, 45.0, 90.0]     # static per-element phase RMS (deg)
AMP_RMS_DB = 1.0                                    # amplitude error for the phase+amp variant

# Experiment 2 sweep: end-to-end RMS phase DRIFT accumulated over one capture (deg)
DRIFT_RMS = [0.0, 2.0, 5.0, 10.0, 20.0, 45.0]


# ---------------------------------------------------------------------------
# Channel with calibration error
# ---------------------------------------------------------------------------
def make_sources(N, K, dtheta, snr, trial):
    """Compose K equal-power QPSK keyed sources and their array manifold A (N x K)."""
    angles = source_angles(K, dtheta)
    assert angles is not None, "unplaceable geometry"
    keys = [1000 + 100 * trial + 7 * i for i in range(K)]
    bseeds = [1 + 100 * trial + 7 * i for i in range(K)]
    S = np.stack([compose(bits_for(T_ROWS, L_DEFAULT, 4, s), K=L_DEFAULT, L=L_DEFAULT,
                          seed=k, M=4) for k, s in zip(keys, bseeds)])
    A = np.stack([steering(N, a) for a in angles], axis=1)          # N x K
    return S, A, keys, bseeds


def static_error(N, phase_rms_deg, amp_rms_db, rng):
    """Fixed per-element complex gain g_k e^{j phi_k}, UNKNOWN to the receiver.
    Element 0 is the reference (no error) so only the DIFFERENTIAL error matters."""
    phi = np.deg2rad(phase_rms_deg) * rng.standard_normal(N)
    if amp_rms_db > 0:
        g = 10.0 ** (amp_rms_db / 20.0 * rng.standard_normal(N))
    else:
        g = np.ones(N)
    phi[0] = 0.0
    g[0] = 1.0
    return g * np.exp(1j * phi)                                     # (N,)


def drift_phase(N, n, drift_rms_deg, rng):
    """Per-element phase random walk over n samples with end-to-end RMS = drift_rms_deg.
    A walk of n steps with per-step std s has final std s*sqrt(n), so s = drift/sqrt(n).
    Element 0 is the reference (common-mode drift is harmless); only differential drift hurts."""
    if drift_rms_deg <= 0:
        return np.zeros((N, n))
    step = np.deg2rad(drift_rms_deg) / np.sqrt(n)
    w = np.cumsum(step * rng.standard_normal((N, n)), axis=1)
    w[0, :] = 0.0
    return w


def build_Y(S, A, snr_db, N, cal_vec=None, drift=None, rng=None):
    """Y = (A @ S) * (cal_vec[:,None] * exp(j*drift)) + AWGN. cal_vec (N,) static; drift (N,n)."""
    X = A @ S                                                       # N x n
    if cal_vec is not None:
        X = X * cal_vec[:, None]
    if drift is not None:
        X = X * np.exp(1j * drift)
    if snr_db is not None:
        npow = 1.0 / (10.0 ** (snr_db / 10.0))
        s = np.sqrt(npow / 2.0)
        X = X + s * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
    return X


def genie_zf(Y, A):
    return np.linalg.pinv(A) @ Y


def read_streams(Shat, keys, bseeds):
    """Blind-phase read of each separated stream; return list of per-stream BER."""
    out = []
    for i, (k, s) in enumerate(zip(keys, bseeds)):
        bits = bits_for(T_ROWS, L_DEFAULT, 4, s)
        g = read(Shat[i], L_DEFAULT, L_DEFAULT, k, M=4)
        out.append(ber(bits, g))
    return out


def decode_streams(Y, keys, bseeds, L, N):
    """Blind keyed-MVDR separation + fully blind decode of each stream; return per-stream BER."""
    out = []
    for k, s in zip(keys, bseeds):
        bits = bits_for(T_ROWS, L_DEFAULT, 4, s)
        comb, _th = blind_separate(Y, k, L, N)
        g, _ = decode(comb, k, candidates=CANDIDATES)
        out.append(ber(bits, g))
    return out


def summarise(bers):
    bers = np.asarray(bers, float)
    return float(bers.mean()), float((bers < FEC_THR).mean())   # mean BER, fraction good


# ===========================================================================
# Experiment 1: static per-element miscalibration
# ===========================================================================
def run_static(cfg, snr, phase_rms, amp_rms_db):
    """Return mean BER + good-fraction for genie_actual, genie_nominal, blind."""
    N, K, dtheta = cfg["N"], cfg["K"], cfg["dtheta"]
    ga, gn, bl = [], [], []
    for t in range(TRIALS):
        S, A, keys, bseeds = make_sources(N, K, dtheta, snr, t)
        erng = np.random.default_rng(900_000 + 1000 * t + N * 31 + int(phase_rms) * 7
                                     + int(amp_rms_db * 10))
        nrng = np.random.default_rng(50_000 + 1000 * t + N * 31 + int(dtheta))
        cal = static_error(N, phase_rms, amp_rms_db, erng)
        Y = build_Y(S, A, snr, N, cal_vec=cal, rng=nrng)
        A_actual = cal[:, None] * A
        ga += read_streams(genie_zf(Y, A_actual), keys, bseeds)   # knows true manifold (immune)
        gn += read_streams(genie_zf(Y, A), keys, bseeds)          # nominal/stale angles only
        bl += decode_streams(Y, keys, bseeds, L_DEFAULT, N)       # blind key-steered
    return summarise(ga), summarise(gn), summarise(bl)


# ===========================================================================
# Experiment 2: coherence loss (per-element phase drift within a capture)
# ===========================================================================
def run_drift(cfg, snr, drift_rms):
    """Return mean BER + good-fraction for genie_nominal (mean manifold) and blind."""
    N, K, dtheta = cfg["N"], cfg["K"], cfg["dtheta"]
    gn, bl = [], []
    for t in range(TRIALS):
        S, A, keys, bseeds = make_sources(N, K, dtheta, snr, t)
        n = S.shape[1]
        drng = np.random.default_rng(700_000 + 1000 * t + N * 31 + int(drift_rms) * 7)
        nrng = np.random.default_rng(50_000 + 1000 * t + N * 31 + int(dtheta))
        dft = drift_phase(N, n, drift_rms, drng)
        Y = build_Y(S, A, snr, N, drift=dft, rng=nrng)
        gn += read_streams(genie_zf(Y, A), keys, bseeds)          # ZF at the mean (nominal) manifold
        bl += decode_streams(Y, keys, bseeds, L_DEFAULT, N)       # blind key-steered
    return summarise(gn), summarise(bl)


# ===========================================================================
# Reporting
# ===========================================================================
def fmt_snr(s):
    return "clean" if s is None else f"{s}dB"


def main():
    print("Blind keyed-MVDR array: does it need per-boot PHASE CAL, or only a shared clock?")
    print(f"QPSK, L={L_DEFAULT}, {T_ROWS} rows, {TRIALS} seeds/cell; good = per-stream BER < "
          f"{FEC_THR:g}. SIM ONLY, no transmit.")
    print("genie_actual = ZF knowing the TRUE per-element manifold (immune upper bound).")
    print("genie_nominal = ZF using NOMINAL/stale steering vectors (the angle-based receiver).")
    print("blind = keyed-MVDR steered by each key's constant-modulus lock (no angles, no cal).\n")

    static_rows = []
    drift_rows = []

    # ---- Experiment 1: static per-element miscalibration (phase only) -------
    print("=" * 78)
    print("EXPERIMENT 1a -- STATIC per-element PHASE miscalibration (amplitude exact)")
    print("Sweep: per-element phase RMS (deg), fixed within capture, UNKNOWN to receiver.")
    print("=" * 78)
    for cfg in CONFIGS:
        for snr in SNRS:
            print(f"\n{cfg['tag']}  |  SNR {fmt_snr(snr)}")
            print(f"{'phiRMS':>7} | {'BER_act':>9} {'good':>5} | {'BER_nom':>9} {'good':>5} "
                  f"| {'BER_blind':>9} {'good':>5}  {'blind vs nominal':>18}")
            for pr in PHASE_RMS:
                (ba, gaok), (bn, gnok), (bb, bbok) = run_static(cfg, snr, pr, 0.0)
                static_rows.append({"cfg": cfg["tag"], "snr": snr, "phi": pr, "amp": 0.0,
                                    "ba": ba, "bn": bn, "bb": bb,
                                    "gaok": gaok, "gnok": gnok, "bbok": bbok})
                note = "blind immune" if (bbok >= gaok - 1e-9 and bbok > gnok + 1e-9) \
                    else ("both hold" if bbok >= 1 - 1e-9 and gnok >= 1 - 1e-9 else "")
                print(f"{pr:>7.0f} | {ba:>9.4f} {gaok:>5.2f} | {bn:>9.4f} {gnok:>5.2f} "
                      f"| {bb:>9.4f} {bbok:>5.2f}  {note:>18}")

    # ---- Experiment 1b: static phase + amplitude error ----------------------
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 1b -- STATIC phase + AMPLITUDE error (amp RMS {AMP_RMS_DB:.1f} dB)")
    print("=" * 78)
    for cfg in CONFIGS:
        for snr in SNRS:
            print(f"\n{cfg['tag']}  |  SNR {fmt_snr(snr)}  |  amp RMS {AMP_RMS_DB:.1f} dB")
            print(f"{'phiRMS':>7} | {'BER_act':>9} {'good':>5} | {'BER_nom':>9} {'good':>5} "
                  f"| {'BER_blind':>9} {'good':>5}")
            for pr in PHASE_RMS:
                (ba, gaok), (bn, gnok), (bb, bbok) = run_static(cfg, snr, pr, AMP_RMS_DB)
                static_rows.append({"cfg": cfg["tag"], "snr": snr, "phi": pr, "amp": AMP_RMS_DB,
                                    "ba": ba, "bn": bn, "bb": bb,
                                    "gaok": gaok, "gnok": gnok, "bbok": bbok})
                print(f"{pr:>7.0f} | {ba:>9.4f} {gaok:>5.2f} | {bn:>9.4f} {gnok:>5.2f} "
                      f"| {bb:>9.4f} {bbok:>5.2f}")

    # ---- Experiment 2: coherence loss (within-capture drift) ----------------
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 -- COHERENCE loss: per-element phase DRIFT within one capture")
    print("Sweep: end-to-end RMS phase drift (deg) accumulated across the capture (random walk).")
    print("=" * 78)
    for cfg in CONFIGS:
        for snr in SNRS:
            print(f"\n{cfg['tag']}  |  SNR {fmt_snr(snr)}")
            print(f"{'driftRMS':>8} | {'BER_nom':>9} {'good':>5} | {'BER_blind':>9} {'good':>5}"
                  f"  {'verdict':>14}")
            for dr in DRIFT_RMS:
                (bn, gnok), (bb, bbok) = run_drift(cfg, snr, dr)
                drift_rows.append({"cfg": cfg["tag"], "snr": snr, "drift": dr,
                                   "bn": bn, "bb": bb, "gnok": gnok, "bbok": bbok})
                v = "ok" if bbok >= 2 - 1e-9 else ("degrading" if bbok >= 1 else "FAIL")
                print(f"{dr:>8.0f} | {bn:>9.4f} {gnok:>5.2f} | {bb:>9.4f} {bbok:>5.2f}  {v:>14}")

    verdict(static_rows, drift_rows)


def verdict(static_rows, drift_rows):
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # good-fraction == 1.0 means BOTH streams of min(N,K)=2 delivered. (good is a fraction.)
    ALL = 1.0 - 1e-9

    # (a) static phase immunity, measured on the N=2 array (the minimal DIY / Pluto case)
    phase_only = [r for r in static_rows if r["amp"] == 0.0]
    n2 = [r for r in phase_only if r["cfg"].startswith("N=2")]
    n4 = [r for r in phase_only if r["cfg"].startswith("N=4")]
    blind_n2_imm = all(r["bbok"] >= ALL for r in n2)                 # blind both streams at every phi
    nom_n2_coll = any(r["gnok"] < ALL for r in n2 if r["phi"] >= 20)  # nominal ZF drops a stream
    print("(a) STATIC per-element miscalibration (phase, UNKNOWN to RX):")
    print(f"    N=2 (minimal DIY / Pluto): blind delivers BOTH streams at EVERY phi (<=90 deg "
          f"RMS): {blind_n2_imm}")
    print(f"    N=2: nominal angle-based ZF has dropped a stream by phi>=20 deg: {nom_n2_coll}")
    # largest phi where N=4 blind still delivers both streams, clean vs 12 dB
    for snr in SNRS:
        okphi = [r["phi"] for r in n4 if r["snr"] == snr and r["bbok"] >= ALL]
        failphi = [r["phi"] for r in n4 if r["snr"] == snr and r["bbok"] < ALL]
        mx = max(okphi) if okphi else 0.0
        ff = min(failphi) if failphi else None
        print(f"    N=4 blind, SNR {fmt_snr(snr):>6}: both streams up to ~{mx:.0f} deg phi RMS"
              f"{'' if ff is None else f'; first drop at {ff:.0f} deg'}")
    if blind_n2_imm and nom_n2_coll:
        print("    => On N=2 the blind key-steered receiver is IMMUNE to static per-element phase")
        print("       error (to 90 deg RMS) while the angle-based ZF collapses. No per-boot")
        print("       phase-cal rig is needed for N=2. N=4 is immune when clean but noise exposes")
        print("       residual off-manifold error at large phi (not fully immune).")

    # amplitude sensitivity (clean, N=2)
    amp2 = [r for r in static_rows if r["amp"] > 0.0 and r["snr"] is None and r["cfg"].startswith("N=2")]
    amp_ok = all(r["bbok"] >= ALL for r in amp2)
    print(f"    phase+amplitude ({AMP_RMS_DB:.1f} dB) clean on N=2: blind still delivers both "
          f"streams everywhere: {amp_ok}")

    # (b) coherence: max drift where each method still delivers both streams
    print("\n(b) COHERENCE (within-capture differential phase DRIFT, end-to-end RMS):")
    for cfg in CONFIGS:
        for snr in SNRS:
            rows = sorted([r for r in drift_rows if r["cfg"] == cfg["tag"] and r["snr"] == snr],
                          key=lambda r: r["drift"])
            def span(key):
                ok = [r["drift"] for r in rows if r[key] >= ALL]
                fail = [r["drift"] for r in rows if r[key] < ALL]
                return (max(ok) if ok else 0.0), (min(fail) if fail else None)
            bmx, bff = span("bbok")
            gmx, gff = span("gnok")
            bffs = ">45" if bff is None else f"{bff:.0f}"
            gffs = ">45" if gff is None else f"{gff:.0f}"
            print(f"    {cfg['tag']:<32} SNR {fmt_snr(snr):>6}: "
                  f"blind both to ~{bmx:.0f} deg (fail {bffs}); "
                  f"genie both to ~{gmx:.0f} deg (fail {gffs})")

    print("\nBUILD GUIDANCE (cheapest coherent RTL-SDR array):")
    print("  - Shared clock: YES, required. All dongles off one reference oscillator / clock")
    print("    fan-out so the elements stay phase-coherent within a capture.")
    print("  - Per-boot phase-cal rig (noise source + RF switch, KrakenSDR-style): NOT required.")
    print("    Static per-element phase is absorbed by the key-locked beam search (it steers by")
    print("    the key, not absolute angle) -- clean on N=2 to 90 deg RMS, where the angle-based")
    print("    ZF has long collapsed. Drop the cal rig; keep only the shared clock.")
    print("  - Clock stability spec: keep DIFFERENTIAL (element-to-element) phase drift within a")
    print("    capture small. Blind both-stream recovery held to ~20 deg/capture on N=2 and")
    print("    ~5 deg/capture on N=4 here; the de-interfered beam (genie) tolerated ~20 deg")
    print("    before failing at 45. Target << 5-10 deg/capture differential for margin.")
    print("  - Shorten captures or raise sample rate to bound accumulated drift if the clock")
    print("    fan-out is imperfect; a common-mode drift (identical on all elements) is harmless.")
    print("  - Caveats: amplitude mismatch and N>2 non-linear phase are only PARTIALLY absorbed")
    print("    (see 1b and the N=4 noisy cells); a one-time (NOT per-boot) element gain/phase trim")
    print("    still helps the oversized array. N=2 is the clean, fully-immune case.")


if __name__ == "__main__":
    main()
