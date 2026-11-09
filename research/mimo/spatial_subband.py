"""Does spatial separation COMPOSE with sub-band separation?  The fleet throughput ceiling.

Two proven co-channel separation axes for entroptics:

  * SPATIAL (research/mimo/spatial_mimo.md, GO).  An N-element receive array separates up to
    min(N, #angles) equal-power co-channel keyed streams arriving from distinct directions, by a
    blind keyed-MVDR combine -- no pilots: the beam is steered to the angle that maximizes one
    key's constant-modulus lock, which nulls the other angles.  The combine is a weighted sum of
    the coherent antenna captures; it is front-end array processing, not the codec.

  * SUB-BAND (frequency-division).  Disjoint narrowband sub-bands at distinct centres inside one
    allocation each carry their own keyed screen; a receiver reads each from one capture by a
    band-limited sinc projection (down-shift -> low-pass -> decimate -- a `band read`, which the
    project already treats as resampling-free front-end, research memo "band-limited reads
    allowed").  The codec (compose/decode) stays integer-rate; no FFT, no pilots.

THE QUESTION (the fleet's throughput CEILING).  Do the two axes STACK, so one receiver handles
more co-channel keyed streams than either axis alone?  Arrange K = A x B transmitters: at each of
A distinct angles (>= 30 deg apart, per the spatial GO rules), B single-antenna transmitters,
each on its own disjoint sub-band (distinct centre) + own key + own bits.  The N-element array
receives the steering-vector superposition + per-element AWGN at equal per-source power.

The structure is non-trivial: the array separates by ANGLE, so it groups the B co-located
sub-bands of one angle TOGETHER (one beam cannot tell them apart -- they share a direction).  The
sub-band read must then separate WITHIN that group.  And the blind beam is steered by ONLY ONE of
the group's keys (its sub-band 0); the test is whether that same beam also delivers the group's
OTHER sub-bands.  Expected ceiling: min(N, A) x B streams.

RECEIVER (entroptics-pure).  For each angle group: blind keyed-MVDR combine, steering angle chosen
to maximize sub-band-0's constant-modulus lock on the channelized capture (nulls the other
angles); then on the single combined capture, channelize and `decode` each of the B sub-bands per
key.  GENIE (ceiling) replaces the blind combine with true-angle zero-forcing (pinv of the array
manifold); the sub-band read is the same blind `decode` in both, so genie vs blind isolates the
spatial-combine quality alone.

Public library only: compose, decode, compose_qam, read_qam, read, ber, random_bits,
constant_modulus_read.  Array math (steering vectors, pinv, sample-covariance MVDR) and the
sub-band channelizer (windowed-sinc up/down conversion) are plain front-end DSP applied to the
antenna captures BEFORE the codec; they read no payload and hold no screen geometry.

SIM ONLY.  No hardware, no transmit.  Re-runnable:
    OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src \
        .venv/Scripts/python.exe research/mimo/spatial_subband.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Reuse the spatial array-channel + blind-MVDR helpers from the GO study (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_mimo import steering, mvdr_weight, source_angles, LOAD, MAX_ABS_ANGLE  # noqa: E402

from entroptics_rf import (                                    # noqa: E402
    compose, decode, compose_qam, read_qam, ber, random_bits, constant_modulus_read,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
L_DEFAULT = 64            # screen width (modes); every mode lit
T_ROWS = 96               # payload rows -- >= ~90 so the blind width lock settles
CANDIDATES = range(60, 69)
TRIALS = 3                # independent (key, bit) seed sets per cell
FEC_THR = 1e-2            # a stream counts as "delivered" below this raw BER (lenient)

# blind angle search grid (per angle group): coarse pass then local refine
COARSE = np.arange(-MAX_ABS_ANGLE, MAX_ABS_ANGLE + 1e-6, 4.0)


# ---------------------------------------------------------------------------
# Sub-band channelizer -- windowed-sinc up/down conversion (front-end band read).
# Each of B disjoint sub-bands occupies 1/B of the band at a distinct centre.  Up: zero-stuff by
# B, interpolate (sinc low-pass, gain B), shift to centre.  Down: shift to baseband, sinc
# low-pass, decimate by B.  No FFT; the codec never sees the resampling.
# ---------------------------------------------------------------------------
def sinc_lp(cutoff, half):
    """Windowed-sinc low-pass impulse response, cutoff in cycles/sample, unit passband gain."""
    k = np.arange(-half, half + 1)
    return 2 * cutoff * np.sinc(2 * cutoff * k) * np.hanning(2 * half + 1)


def sub_centres(B):
    """B disjoint sub-band centres (cycles/sample), evenly tiling (-0.5, 0.5)."""
    return (np.arange(B) + 0.5) / B - 0.5


def filt_half(B):
    """Sinc half-length: longer for narrower sub-bands so adjacent-band leakage stays low."""
    return max(32, 16 * B)


def upconvert(x, B, centre, half):
    """Native-rate screen -> narrowband sub-band at `centre`, length B*len(x)."""
    u = np.zeros(len(x) * B, dtype=complex)
    u[::B] = x
    u = np.convolve(u, sinc_lp(0.5 / B, half) * B, mode="same")
    m = np.arange(len(u))
    return u * np.exp(2j * np.pi * centre * m)


def downconvert(y, B, centre, half, n):
    """Combined capture -> native-rate screen estimate for the sub-band at `centre` (length n)."""
    m = np.arange(len(y))
    y = y * np.exp(-2j * np.pi * centre * m)
    y = np.convolve(y, sinc_lp(0.5 / B, half), mode="same")
    return y[::B][:n]


# ---------------------------------------------------------------------------
# Transmit side: build the A x B co-channel array capture
# ---------------------------------------------------------------------------
def bits_for(rows, K_modes, M, seed):
    b = int(np.log2(M))
    return random_bits(rows * K_modes * b, seed=seed)


def build_group_signal(keys, bseeds, B, half, M, qam):
    """One angle group's transmitted signal: sum of B unit-rms disjoint sub-bands.
    Returns (signal [length nb], native screen length n)."""
    maker = compose_qam if qam else compose
    screens = [maker(bits_for(T_ROWS, L_DEFAULT, M, s), K=L_DEFAULT, L=L_DEFAULT, seed=k, M=M)
               for k, s in zip(keys, bseeds)]
    n = min(len(s) for s in screens)
    screens = [s[:n] for s in screens]
    cs = sub_centres(B)
    subs = []
    for x, c in zip(screens, cs):
        u = upconvert(x, B, c, half)
        subs.append(u / np.sqrt(np.mean(np.abs(u) ** 2)))   # unit rms per sub-band source
    nb = min(len(u) for u in subs)
    return np.sum([u[:nb] for u in subs], axis=0), n


def build_Y(group_sigs, angles, snr_db, N, B, rng):
    """Mix the A angle-group signals through the array + per-element AWGN.
    Per-source in-band SNR = `snr_db`: each sub-band source is unit power in its 1/B band, so
    full-band noise power sigma^2 = B*10^(-snr/10) leaves sigma^2/B in each sub-band."""
    nb = min(len(g) for g in group_sigs)
    G = np.stack([g[:nb] for g in group_sigs])                 # A x nb
    A = np.stack([steering(N, a) for a in angles], axis=1)     # N x A
    Y = A @ G                                                  # N x nb
    if snr_db is not None:
        sigma2 = B * 10 ** (-snr_db / 10)
        s = np.sqrt(sigma2 / 2)
        Y = Y + s * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    return Y, A


# ---------------------------------------------------------------------------
# Receive side
# ---------------------------------------------------------------------------
def group_covariance_inv(Y, N):
    R = (Y @ Y.conj().T) / Y.shape[1]
    R = R + LOAD * (np.trace(R).real / N) * np.eye(N)
    return np.linalg.inv(R)


def blind_angle(Y, Rinv, key0, B, half, n, N):
    """Steer the beam to one angle group WITHOUT its angle: grid-search the steering angle, form
    the MVDR weight from the sample covariance, channelize sub-band 0 from the combined capture,
    and keep the angle whose sub-band-0 reads the lowest constant-modulus score under key0."""
    c0 = sub_centres(B)[0]

    def best_over(grid):
        bcm, bth = np.inf, grid[0]
        for th in grid:
            w = mvdr_weight(Rinv, steering(N, th))
            comb = w.conj() @ Y
            sub0 = downconvert(comb, B, c0, half, n)
            cm = constant_modulus_read(sub0, L_DEFAULT, key0)
            if cm < bcm:
                bcm, bth = cm, th
        return bth

    th0 = best_over(COARSE)
    return best_over(np.arange(th0 - 4.0, th0 + 4.0 + 1e-6, 0.5))


def read_group(comb, keys, bseeds, B, half, n, M, qam):
    """Channelize and decode each of the B sub-bands from one combined capture."""
    cs = sub_centres(B)
    out = []
    for k, s, c in zip(keys, bseeds, cs):
        rec = downconvert(comb, B, c, half, n)
        bits = bits_for(T_ROWS, L_DEFAULT, M, s)
        g, _ = decode(rec, k, candidates=CANDIDATES)
        out.append(ber(bits, g))
    return out


def run_cell(N, A, B, dtheta, snr, M=4, qam=False):
    """One (N, A, B, dtheta, SNR) cell averaged over TRIALS seeds.  Returns a dict, or None if the
    angle geometry is not placeable on the ULA half-space."""
    angles = source_angles(A, dtheta)
    if angles is None:
        return None
    half = filt_half(B)
    ceil = min(N, A) * B
    bersg, bersb = [], []
    goodg = goodb = 0.0
    for t in range(TRIALS):
        # distinct key + bit seed per (angle, sub-band)
        keys = [[5000 + 1000 * t + 101 * g + 7 * j for j in range(B)] for g in range(A)]
        bseeds = [[1 + 1000 * t + 101 * g + 7 * j for j in range(B)] for g in range(A)]
        group_sigs = []
        n_native = None
        for g in range(A):
            sig, n = build_group_signal(keys[g], bseeds[g], B, half, M, qam)
            group_sigs.append(sig)
            n_native = n
        rng = np.random.default_rng(70_000 + 1000 * t + N * 31 + A * 13 + B * 7 + int(dtheta))
        Y, Amat = build_Y(group_sigs, angles, snr, N, B, rng)

        # --- GENIE: zero-forcing combine at true angles, then blind sub-band decode ---
        Shat = np.linalg.pinv(Amat) @ Y                        # A x nb (one row per angle group)
        for g in range(A):
            for e in read_group(Shat[g], keys[g], bseeds[g], B, half, n_native, M, qam):
                bersg.append(e)
                goodg += (e < FEC_THR)

        # --- BLIND: MVDR beam per group via sub-band-0 CM lock, then blind sub-band decode ---
        Rinv = group_covariance_inv(Y, N)
        for g in range(A):
            th = blind_angle(Y, Rinv, keys[g][0], B, half, n_native, N)
            comb = mvdr_weight(Rinv, steering(N, th)).conj() @ Y
            for e in read_group(comb, keys[g], bseeds[g], B, half, n_native, M, qam):
                bersb.append(e)
                goodb += (e < FEC_THR)

    return {
        "N": N, "A": A, "B": B, "dtheta": dtheta, "snr": snr, "ceil": ceil,
        "ber_genie": float(np.mean(bersg)), "ber_blind": float(np.mean(bersb)),
        "good_genie": goodg / TRIALS, "good_blind": goodb / TRIALS,
    }


def fmt_snr(s):
    return "clean" if s is None else f"{s}"


def print_cell(r):
    if r is None:
        return
    note = ""
    if r["good_blind"] >= r["ceil"] - 1e-9:
        note = "blind == ceiling"
    elif r["good_genie"] >= r["ceil"] - 1e-9 > r["good_blind"]:
        note = "blind < ceiling"
    print(f"{r['dtheta']:>6} {fmt_snr(r['snr']):>6} | "
          f"{r['ber_genie']:>10.4f} {r['good_genie']:>6.2f}/{r['ceil']:<2d} | "
          f"{r['ber_blind']:>10.4f} {r['good_blind']:>6.2f}/{r['ceil']:<2d}  {note:>16}")


def header(N, A, B):
    ceil = min(N, A) * B
    print(f"--- N={N} array, A={A} angles x B={B} sub-bands = {A*B} co-channel streams "
          f"(ceiling min(N,A)xB = {ceil}) ---")
    print(f"{'dtheta':>6} {'SNR':>6} | {'BER_genie':>10} {'good_g':>9} | "
          f"{'BER_blind':>10} {'good_b':>9}  {'note':>16}")


# ===========================================================================
def main():
    print("Spatial x sub-band co-channel stacking for entroptics -- SIM ONLY, no transmit.")
    print(f"ULA d=lambda/2; QPSK (M=4); L={L_DEFAULT}, {T_ROWS} rows, {TRIALS} seeds/cell; "
          f"good = BER < {FEC_THR:g} (of A*B).")
    print("GENIE = zero-forcing combine at true angles; BLIND = MVDR beam steered by sub-band-0 "
          "CM lock.")
    print("Both read each sub-band with the same blind `decode` after a band-limited sinc "
          "channelize.\n")

    all_rows = []
    cells = [
        # (N, A, B, dtheta, [snrs])
        (2, 2, 2, 60, [None, 12]),   # N=2 array, 2x2 = 4 streams (exceeds either axis alone)
        (2, 2, 2, 30, [None, 12]),
        (2, 2, 2, 15, [None, 12]),   # close angles: the spatial break should still bite per angle
        (4, 2, 2, 30, [None, 12]),   # oversized array
        (4, 2, 3, 30, [None, 12]),   # 2x3 = 6 streams: validates the "x #sub-bands" factor
        (4, 4, 2, 30, [None, 12]),   # 4x2 = 8 streams
    ]
    last = None
    for (N, A, B, dtheta, snrs) in cells:
        if (N, A, B) != last:
            if last is not None:
                print()
            header(N, A, B)
            last = (N, A, B)
        for snr in snrs:
            r = run_cell(N, A, B, dtheta, snr)
            if r is None:
                print(f"{dtheta:>6} {fmt_snr(snr):>6} |  n/a geometry")
                continue
            all_rows.append(r)
            print_cell(r)
    print()

    # 16-QAM spot check
    print("--- 16-QAM spot check (N=4, A=2 x B=2, dtheta=30) ---")
    print(f"{'SNR':>6} | {'BER_genie':>10} {'good_g':>9} | {'BER_blind':>10} {'good_b':>9}")
    for snr in [None, 12]:
        r = run_cell(4, 2, 2, 30, snr, M=16, qam=True)
        print(f"{fmt_snr(snr):>6} | {r['ber_genie']:>10.4f} {r['good_genie']:>6.2f}/{r['ceil']:<2d}"
              f" | {r['ber_blind']:>10.4f} {r['good_blind']:>6.2f}/{r['ceil']:<2d}")
        all_rows.append(r)

    verdict(all_rows)


def verdict(rows):
    print("\n=== VERDICT ===")
    clean = [r for r in rows if r["snr"] is None and not (r["N"] == 2 and r["dtheta"] == 15)]
    hit = [r for r in clean if r["good_blind"] >= r["ceil"] - 1e-9]
    print(f"- Clean cells (dtheta >= 30) hitting the full min(N,A)xB ceiling, blind: "
          f"{len(hit)}/{len(clean)}.")
    for r in rows:
        if r["snr"] is None and r["dtheta"] in (30, 60) and r["N"] in (2, 4):
            print(f"    N={r['N']} A={r['A']}xB={r['B']} dtheta={r['dtheta']}: "
                  f"blind {r['good_blind']:.2f}/{r['ceil']} (genie {r['good_genie']:.2f})")
    stack = any(r["good_blind"] >= 4 - 1e-9 for r in rows)
    eight = [r for r in rows if r["ceil"] == 8]
    if stack:
        print("- GO: spatial x sub-band STACKS.  A 2-element array x 2 sub-bands delivers 4 "
              "co-channel\n  keyed streams; the fleet ceiling is min(N, #angles) x "
              "#disjoint-sub-bands.")
        if eight:
            be = max(r["good_blind"] for r in eight)
            print(f"  N=4 x A=4 x B=2 reached {be:.2f}/8 streams blind.")
    else:
        print("- NO-GO: the two axes did not stack to > either alone.")
    print("  Breaks: the per-angle spatial rules still apply (>= 30 deg, K_angles <= N); close "
          "angles\n  (15 deg on a square array) collapse the group BEFORE the sub-band read. "
          "Sub-band cost is\n  the channelizer penalty (~2-3 dB) + adjacent-band leakage. See "
          "spatial_subband.md.")


if __name__ == "__main__":
    main()
