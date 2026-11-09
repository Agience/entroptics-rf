"""Spatial co-channel multiplexing for entroptics: does a MULTI-ANTENNA receiver
recover the co-channel capacity that SINGLE-antenna key-stacking could not?

Background (research/multiplex/overlaid_keys.md).  Two independently-keyed screens
superposed on ONE antenna do NOT multiplex: a wrong-key read of a screen is noise-like,
but it is *full-power* noise with no processing gain, so per-stream SINR is pinned to the
power split and at equal power both streams read at chance.  Both screens are full rank in
the same L keyed modes -- there is no free dimension on a single antenna.

This study adds the one thing that single-antenna stacking lacks: a SPATIAL degree of
freedom.  Co-channel keyed transmitters arriving from different directions present
different array signatures to an N-element receiver, so an N-element array has N spatial
dimensions and should separate up to ~N co-channel sources by linear combining -- the way
MIMO does -- WITHOUT any pilots or training.  A quadcopter relay can carry up to 4 antennas
(one per arm), so N = 2 and N = 4 are both tested.

MODEL (uniform linear array, narrowband steering-vector mixing -- standard array processing
in the RX front-end; it is NOT the codec).  RX array of N elements at spacing d = lambda/2,
so element k sees a source at angle theta with phase exp(j*pi*k*sin(theta)).  K co-channel
single-antenna keyed transmitters (independent keys + bits), each at a distinct angle,
superposed at EQUAL per-source power:
    Y[k_elem, n] = sum_src steer_k(theta_src) * s_src[n] + per-element complex AWGN
Each s_src is composed by the public library (compose, K=L=64, distinct seeds, >=90 rows so
the blind width lock settles).  SNR is defined per source, per element.

SEPARATION -- two receivers, both entroptics-pure (NO pilots, NO training, NO FFT or
resampling in the codec; the spatial combine is a weighted sum of coherent antenna captures,
and the weights are found either by a genie or BLIND):
  1. GENIE zero-forcing (capacity CEILING): W = pinv(A), A = [steer(theta_1) ... steer(theta_K)]
     (N x K, true angles known).  Separated Shat = W @ Y; each stream read at its true geometry
     by the informed single-key reader.  This establishes whether the capacity is even there.
  2. BLIND (the real entroptics-pure question): separate WITHOUT knowing the angles and
     WITHOUT pilots.  For each held key, grid-search the array steering angle and form the
     MVDR (minimum-variance distortionless-response) weight from the sample covariance alone,
     choosing the angle that MAXIMIZES that key's constant-modulus lock on the combined
     stream; then read the combined stream with the fully blind `decode`.  The keys + the
     array replace training.  We report how close blind gets to the genie ceiling.

Public library only: compose, decode, compose_qam, read_qam, read, ber, random_bits,
constant_modulus_read.  The array math (steering vectors, pinv, sample-covariance MVDR) is
plain narrowband array processing applied to the antenna captures BEFORE the codec; it reads
no payload and holds no screen geometry.

SIM ONLY.  No hardware, no transmit.  Re-runnable:
    OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src \
        .venv/Scripts/python.exe research/mimo/spatial_mimo.py
"""
from __future__ import annotations

import numpy as np

from entroptics_rf import (
    compose, decode, compose_qam, read_qam, read, ber, random_bits,
    constant_modulus_read,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
L_DEFAULT = 64            # screen width (modes); K_modes == L, every mode lit
T_ROWS = 96               # payload rows -- >= ~90 so the blind width lock settles
CANDIDATES = range(60, 69)
TRIALS = 3                # independent (key, bit) seed sets per cell
FEC_THR = 1e-2            # a stream counts as "delivered" below this raw BER (lenient)
MAX_ABS_ANGLE = 75.0      # a ULA half-space cutoff: sources beyond this are not placeable

# sweeps
DTHETAS = [15, 30, 60, 90]            # angular separation between adjacent sources (deg)
SNRS = [None, 20, 12, 9]              # per-source per-element SNR (dB); None = clean
ARRAYS = [2, 4]                        # N receive elements

# blind angle search grid: coarse pass then local refine
COARSE = np.arange(-MAX_ABS_ANGLE, MAX_ABS_ANGLE + 1e-6, 3.0)
LOAD = 1e-2               # MVDR diagonal-loading fraction (of trace/N) for a stable inverse


def bits_for(rows, K_modes, M, seed):
    """A payload filling exactly `rows` rows so every source has identical sample length."""
    b = int(np.log2(M))
    return random_bits(rows * K_modes * b, seed=seed)


def steering(N, theta_deg):
    """Narrowband ULA steering vector, d = lambda/2: element k phase = exp(j*pi*k*sin theta)."""
    k = np.arange(N)
    return np.exp(1j * np.pi * k * np.sin(np.deg2rad(theta_deg)))


def source_angles(K, dtheta):
    """K source angles, centred on broadside, adjacent spacing `dtheta` (deg).
    Returns None if any source would fall outside the placeable half-space."""
    idx = np.arange(K) - (K - 1) / 2.0
    ang = idx * dtheta
    if np.max(np.abs(ang)) > MAX_ABS_ANGLE + 1e-9:
        return None
    return ang


def build_Y(S, angles, snr_db, N, rng):
    """Mix K sources (rows of S) through the array and add per-element complex AWGN.
    Each source arrives at unit power per element (|steer|=1, compose is unit-rms), so
    `snr_db` is the per-source, per-element SNR.  Returns (Y [N x n], A [N x K])."""
    A = np.stack([steering(N, a) for a in angles], axis=1)   # N x K
    Y = A @ S                                                # N x n
    if snr_db is not None:
        npow = 1.0 / (10.0 ** (snr_db / 10.0))               # per-source power is 1
        s = np.sqrt(npow / 2.0)
        Y = Y + s * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    return Y, A


def genie_zf(Y, A):
    """Zero-forcing with the true array manifold: Shat = pinv(A) @ Y (K x n)."""
    return np.linalg.pinv(A) @ Y


def mvdr_weight(Rinv, a):
    """Distortionless MVDR weight for steering `a` from an inverse covariance."""
    num = Rinv @ a
    return num / (a.conj() @ num)


def blind_separate(Y, key, L, N):
    """Find the array angle for `key` with NO pilots: grid-search the steering angle, form the
    MVDR weight from the sample covariance alone, and keep the angle whose combined stream reads
    the lowest constant-modulus score under `key` (best keyed lock).  Returns (combined, theta)."""
    n = Y.shape[1]
    R = (Y @ Y.conj().T) / n
    R = R + LOAD * (np.trace(R).real / N) * np.eye(N)
    Rinv = np.linalg.inv(R)

    def best_over(grid):
        bcm, bth, bcomb = np.inf, None, None
        for th in grid:
            w = mvdr_weight(Rinv, steering(N, th))
            comb = w.conj() @ Y
            cm = constant_modulus_read(comb, L, key)
            if cm < bcm:
                bcm, bth, bcomb = cm, th, comb
        return bcm, bth, bcomb

    _, th0, _ = best_over(COARSE)
    fine = np.arange(th0 - 3.0, th0 + 3.0 + 1e-6, 0.5)
    _, th1, comb1 = best_over(fine)
    return comb1, th1


# ===========================================================================
# Main sweep
# ===========================================================================
def run_cell(N, K, dtheta, snr, M=4, qam=False):
    """One (N, K, dtheta, SNR) cell averaged over TRIALS seeds.
    Returns a dict with mean per-stream BER and mean good-stream count for genie and blind,
    or None if the geometry is not placeable."""
    angles = source_angles(K, dtheta)
    if angles is None:
        return None
    K_modes = L_DEFAULT
    bersg, bersb = [], []
    goodg = goodb = 0.0
    for t in range(TRIALS):
        keys = [1000 + 100 * t + 7 * i for i in range(K)]
        bseeds = [1 + 100 * t + 7 * i for i in range(K)]
        if qam:
            S = np.stack([compose_qam(bits_for(T_ROWS, K_modes, M, s), K=K_modes,
                                      L=L_DEFAULT, seed=k, M=M)
                          for k, s in zip(keys, bseeds)])
        else:
            S = np.stack([compose(bits_for(T_ROWS, K_modes, M, s), K=K_modes,
                                  L=L_DEFAULT, seed=k, M=M)
                          for k, s in zip(keys, bseeds)])
        # pad to common length (QAM reference row can make lengths differ by a row)
        n = min(x.shape[0] for x in S) if S.ndim == 1 else S.shape[1]
        rng = np.random.default_rng(50_000 + 1000 * t + N * 31 + K * 7 + int(dtheta))
        Y, A = build_Y(S, angles, snr, N, rng)

        # --- GENIE zero-forcing, read at the true geometry (capacity ceiling) ---
        Shat = genie_zf(Y, A)
        for i, (k, s) in enumerate(zip(keys, bseeds)):
            bits = bits_for(T_ROWS, K_modes, M, s)
            if qam:
                g = read_qam(Shat[i], L_DEFAULT, K_modes, k, M=M)
            else:
                g = read(Shat[i], L_DEFAULT, K_modes, k, M=M)
            e = ber(bits, g)
            bersg.append(e)
            goodg += (e < FEC_THR)

        # --- BLIND: MVDR angle search per key, then the fully blind decode ---
        for i, (k, s) in enumerate(zip(keys, bseeds)):
            bits = bits_for(T_ROWS, K_modes, M, s)
            comb, _th = blind_separate(Y, k, L_DEFAULT, N)
            if qam:
                g, _ = decode(comb, k, candidates=CANDIDATES)
            else:
                g, _ = decode(comb, k, candidates=CANDIDATES)
            e = ber(bits, g)
            bersb.append(e)
            goodb += (e < FEC_THR)

    return {
        "N": N, "K": K, "dtheta": dtheta, "snr": snr,
        "ber_genie": float(np.mean(bersg)),
        "ber_blind": float(np.mean(bersb)),
        "good_genie": goodg / TRIALS,   # mean good streams per trial (of K)
        "good_blind": goodb / TRIALS,
    }


def fmt_snr(s):
    return "clean" if s is None else f"{s}"


def main():
    print("Spatial co-channel MIMO for entroptics -- SIM ONLY, no transmit.")
    print(f"ULA d=lambda/2; QPSK (M=4); L={L_DEFAULT}, {T_ROWS} rows, {TRIALS} seeds/cell; "
          f"good = BER < {FEC_THR:g}.")
    print("GENIE = zero-forcing at true angles, informed read (capacity ceiling).")
    print("BLIND = MVDR angle search by keyed constant-modulus lock, then blind decode.\n")

    all_rows = []
    for N in ARRAYS:
        Ks = list(range(2, N + 1))
        for K in Ks:
            print(f"--- N = {N} elements, K = {K} co-channel sources "
                  f"(min(N,K) = {min(N, K)} streams possible) ---")
            print(f"{'dtheta':>6} {'SNR':>6} | {'BER_genie':>10} {'good_g':>7} "
                  f"| {'BER_blind':>10} {'good_b':>7}  {'note':>18}")
            for dtheta in DTHETAS:
                for snr in SNRS:
                    r = run_cell(N, K, dtheta, snr)
                    if r is None:
                        print(f"{dtheta:>6} {fmt_snr(snr):>6} | "
                              f"{'--':>10} {'--':>7} | {'--':>10} {'--':>7}  "
                              f"{'n/a: geometry':>18}")
                        continue
                    all_rows.append(r)
                    note = ""
                    if r["good_blind"] >= min(N, K) - 1e-9:
                        note = "blind == full"
                    elif r["good_genie"] >= min(N, K) - 1e-9 and r["good_blind"] < min(N, K):
                        note = "blind < ceiling"
                    print(f"{dtheta:>6} {fmt_snr(snr):>6} | "
                          f"{r['ber_genie']:>10.4f} {r['good_genie']:>7.2f} | "
                          f"{r['ber_blind']:>10.4f} {r['good_blind']:>7.2f}  {note:>18}")
            print()

    # 16-QAM spot check at a promising point
    print("--- 16-QAM spot check (N=4, K=2, dtheta=30) ---")
    print(f"{'SNR':>6} | {'BER_genie':>10} {'good_g':>7} | {'BER_blind':>10} {'good_b':>7}")
    for snr in [None, 12]:
        r = run_cell(4, 2, 30, snr, M=16, qam=True)
        print(f"{fmt_snr(snr):>6} | {r['ber_genie']:>10.4f} {r['good_genie']:>7.2f} | "
              f"{r['ber_blind']:>10.4f} {r['good_blind']:>7.2f}")

    verdict(all_rows)


def verdict(rows):
    print("\n=== VERDICT ===")
    # Best aggregate good-stream counts anywhere
    clean = [r for r in rows if r["snr"] is None]
    best_g = max((r["good_genie"] for r in rows), default=0.0)
    best_b = max((r["good_blind"] for r in rows), default=0.0)
    print(f"- Best genie good-stream count anywhere: {best_g:.2f}; best blind: {best_b:.2f}.")

    # Does the array deliver ~min(N,K) streams (vs single-antenna's < 1)?
    two_plus_g = [r for r in rows if r["good_genie"] >= 2 - 1e-9]
    two_plus_b = [r for r in rows if r["good_blind"] >= 2 - 1e-9]
    print(f"- Cells delivering >= 2 co-channel streams: genie {len(two_plus_g)}, "
          f"blind {len(two_plus_b)} (of {len(rows)} placeable cells).")

    # N=4 full-rank K=4
    k4 = [r for r in rows if r["N"] == 4 and r["K"] == 4]
    if k4:
        bg = max(r["good_genie"] for r in k4)
        bb = max(r["good_blind"] for r in k4)
        print(f"- N=4, K=4 best: genie {bg:.2f} streams, blind {bb:.2f} streams.")

    if best_b >= 2 - 1e-9:
        print("- GO: a multi-antenna receiver DOES recover co-channel multi-stream capacity "
              "that\n  single-antenna key-stacking could not. The spatial dimension is the "
              "missing lever.")
    else:
        print("- NO-GO: the array did not deliver >= 2 usable co-channel streams.")
    print("  See spatial_mimo.md for where blind falls short of the genie ceiling, the "
          "angular-\n  separation / SNR break points, and the hardware implied "
          "(N coherent RX chains at ~lambda/2).")


if __name__ == "__main__":
    main()
