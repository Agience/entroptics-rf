"""Co-channel keyed multiplexing: can two independently-keyed entroptics screens,
linearly superposed in the SAME band at the SAME time, each be recovered by the plain
single-key blind receiver (decode / read_qam) that holds both keys?

This is the entroptics-pure analogue of spatial multiplexing.  The hypothesis under test:
two different keys produce quasi-orthogonal keyed apertures, so a read keyed to screen A
sees screen B as noise-like interference (and vice versa), letting both decode at a NET
aggregate > 1 stream for the SAME total transmit power.

HARD CONSTRAINTS (entroptics-pure).  Nothing is pre-shared but the keys.  No pilots, no
preamble, no equalizer, no channel estimation, no FFT, no resampling, no interference
cancellation / successive-IC, no joint ML over both keys.  Each stream is read by the plain
single-key receiver as if the other were noise.  We use ONLY the public library:
    compose, decode, compose_qam, read_qam, read, ber, random_bits, constant_modulus_read.
`read` (informed, true-geometry, still single-key, still no cancellation) is used ONLY as a
diagnostic to separate "the blind lock failed" from "the SINR was too low".

SIM ONLY.  No hardware, no transmit.  Re-runnable:
    OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src \
        .venv/Scripts/python.exe research/multiplex/overlaid_keys.py
"""
from __future__ import annotations

import numpy as np

from entroptics_rf import (
    compose, decode, compose_qam, read_qam, read, ber, random_bits,
    constant_modulus_read,
)

# ---------------------------------------------------------------------------
# Geometry and sweep configuration
# ---------------------------------------------------------------------------
L_DEFAULT = 64            # screen width (modes); K == L, every mode lit
T_ROWS = 100              # payload rows -- enough that the blind width lock settles
CANDIDATES = range(60, 69)
TRIALS = 3                # independent (key, bit) seed sets per cell; BER is the mean
FEC_THR = 1e-2            # a stream counts as "delivered" below this raw BER (lenient)

# (KEY_A, KEY_B) and (bitsA_seed, bitsB_seed) for each independent trial
TRIAL_SEEDS = [
    ((111, 222), (1, 2)),
    ((333, 444), (3, 4)),
    ((555, 666), (5, 6)),
]

SPLITS = [0.5, 0.7, 0.9]          # pA (pB = 1 - pA); pA + pB == 1 == baseline total power
SNRS = [None, 20, 15, 12, 9]      # total SNR (dB) against the combined superposed power


def bits_for(rows, K, M, seed):
    """A payload that fills exactly `rows` rows (no short final row, so both screens
    have identical sample length)."""
    b = int(np.log2(M))
    return random_bits(rows * K * b, seed=seed)


def add_awgn(iq, snr_db, rng):
    """Complex AWGN at `snr_db` against the signal's own power -- the library convention
    (LoopbackRadio): noise power = mean|iq|^2 / 10**(snr/10)."""
    if snr_db is None:
        return iq
    p = float(np.mean(np.abs(iq) ** 2))
    s = np.sqrt(p / (2.0 * 10.0 ** (snr_db / 10.0)))
    return iq + s * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))


# ===========================================================================
# Sweep 1 -- the make-or-break sweep: blind decode of both co-channel streams
# ===========================================================================
def sweep_main(L=L_DEFAULT, M=4):
    K = L
    print(f"\n=== SWEEP 1  two keyed screens superposed, both read by blind decode "
          f"(L={L}, K={K}, M={M} DQPSK, {T_ROWS} rows, {TRIALS} trials) ===")
    print("per-stream BER (mean over trials); 'both?' = both streams below "
          f"{FEC_THR:g} raw BER; baseline = one screen at full power")
    print(f"{'pA/pB':>8} {'SINR_A':>7} {'SNR':>5} {'BER_A':>8} {'BER_B':>8} "
          f"{'good':>5} {'base1':>8}")
    rows = []
    for pA in SPLITS:
        pB = 1.0 - pA
        sinr_a = 10 * np.log10(pA / pB)
        for snr in SNRS:
            berA = berB = 0.0
            base = 0.0
            for (KA, KB), (sA, sB) in TRIAL_SEEDS:
                bitsA = bits_for(T_ROWS, K, M, sA)
                bitsB = bits_for(T_ROWS, K, M, sB)
                iqA = compose(bitsA, K=K, L=L, seed=KA, M=M)
                iqB = compose(bitsB, K=K, L=L, seed=KB, M=M)
                rng = np.random.default_rng(10_000 + KA)
                iq = add_awgn(np.sqrt(pA) * iqA + np.sqrt(pB) * iqB, snr, rng)
                gA, _ = decode(iq, KA, candidates=CANDIDATES)
                gB, _ = decode(iq, KB, candidates=CANDIDATES)
                berA += ber(bitsA, gA)
                berB += ber(bitsB, gB)
                # baseline: one screen alone at the full total power (== unit), same SNR
                rngb = np.random.default_rng(20_000 + KA)
                iqbase = add_awgn(iqA, snr, rngb)   # unit power already
                gbase, _ = decode(iqbase, KA, candidates=CANDIDATES)
                base += ber(bitsA, gbase)
            berA /= TRIALS; berB /= TRIALS; base /= TRIALS
            good = int(berA < FEC_THR) + int(berB < FEC_THR)
            rows.append((pA, pB, sinr_a, snr, berA, berB, good, base))
            snr_s = "clean" if snr is None else f"{snr}"
            print(f"{pA:.1f}/{pB:.1f} {sinr_a:>6.1f}  {snr_s:>5} "
                  f"{berA:>8.4f} {berB:>8.4f} {good:>5d} {base:>8.4f}")
    return rows


# ===========================================================================
# Sweep 2 -- the mechanism: are the two keyed apertures near-orthogonal?
# ===========================================================================
def sweep_orthogonality(L=L_DEFAULT, M=4):
    K = L
    print(f"\n=== SWEEP 2  aperture orthogonality / cross-read sanity (L={L}, M={M}) ===")
    print("cm = constant-modulus read (0 = locked structure, ~1 = noise-like).  A pure B "
          "screen\nread with KEY_A should look like noise (cm~1, BER~0.5) -- the claimed "
          "mechanism.")
    print(f"{'trial':>5} {'cm(B,KB)':>9} {'cm(B,KA)':>9} {'xBER(B|KA)':>11} "
          f"{'xBER(A|KB)':>11}")
    out = []
    for (KA, KB), (sA, sB) in TRIAL_SEEDS:
        bitsA = bits_for(T_ROWS, K, M, sA)
        bitsB = bits_for(T_ROWS, K, M, sB)
        iqA = compose(bitsA, K=K, L=L, seed=KA, M=M)
        iqB = compose(bitsB, K=K, L=L, seed=KB, M=M)
        cm_self = constant_modulus_read(iqB, L, KB)
        cm_cross = constant_modulus_read(iqB, L, KA)
        gx, _ = decode(iqB, KA, candidates=CANDIDATES)   # pure B, wrong key A
        gy, _ = decode(iqA, KB, candidates=CANDIDATES)    # pure A, wrong key B
        xb, xa = ber(bitsB, gx), ber(bitsA, gy)
        out.append((cm_self, cm_cross, xb, xa))
        print(f"{KA:>5} {cm_self:>9.4f} {cm_cross:>9.4f} {xb:>11.4f} {xa:>11.4f}")
    return out


# ===========================================================================
# Sweep 3/4 -- is the other screen just AWGN?  and does the blind lock survive?
# ===========================================================================
def sweep_interference_vs_noise(L=L_DEFAULT, M=4):
    K = L
    print(f"\n=== SWEEP 3/4  co-channel screen vs equal-power AWGN; blind vs informed "
          f"(L={L}, M={M}, clean channel otherwise) ===")
    print("informed = plain single-key read at the TRUE geometry (no cancellation, "
          "diagnostic only).\nIf informed(vs screen) ~ informed(vs AWGN), the other screen "
          "is simply full-power noise;\nif blind >> informed, the blind acquisition itself "
          "failed.")
    print(f"{'pA':>4} {'SINR':>6} {'inf|screenB':>12} {'inf|AWGN':>10} "
          f"{'blind|screenB':>14}")
    out = []
    for pA in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pB = 1.0 - pA
        sinr = 10 * np.log10(pA / pB)
        inf_scr = inf_awgn = bld_scr = 0.0
        for (KA, KB), (sA, sB) in TRIAL_SEEDS:
            bitsA = bits_for(T_ROWS, K, M, sA)
            bitsB = bits_for(T_ROWS, K, M, sB)
            iqA = compose(bitsA, K=K, L=L, seed=KA, M=M)
            iqB = compose(bitsB, K=K, L=L, seed=KB, M=M)
            iq_int = np.sqrt(pA) * iqA + np.sqrt(pB) * iqB
            rng = np.random.default_rng(30_000 + KA)
            s = np.sqrt(pB / 2.0)
            iq_awgn = np.sqrt(pA) * iqA + s * (
                rng.standard_normal(len(iqA)) + 1j * rng.standard_normal(len(iqA)))
            inf_scr += ber(bitsA, read(iq_int, L, K, KA, M=M))
            inf_awgn += ber(bitsA, read(iq_awgn, L, K, KA, M=M))
            gb, _ = decode(iq_int, KA, candidates=CANDIDATES)
            bld_scr += ber(bitsA, gb)
        inf_scr /= TRIALS; inf_awgn /= TRIALS; bld_scr /= TRIALS
        out.append((pA, sinr, inf_scr, inf_awgn, bld_scr))
        print(f"{pA:>4.1f} {sinr:>6.1f} {inf_scr:>12.4f} {inf_awgn:>10.4f} "
              f"{bld_scr:>14.4f}")
    return out


# ===========================================================================
# Sweep 5 -- does a wider screen separate better?
# ===========================================================================
def sweep_width(M=4):
    print(f"\n=== SWEEP 5  width: does a wider screen separate co-channel streams? "
          f"(equal split 0.5/0.5, clean) ===")
    print(f"{'L':>4} {'blind BER_A':>12} {'blind BER_B':>12} {'informed BER_A':>15}")
    out = []
    for L in [64, 128]:
        K = L
        cand = range(L - 4, L + 5)
        berA = berB = inf = 0.0
        for (KA, KB), (sA, sB) in TRIAL_SEEDS:
            bitsA = bits_for(T_ROWS, K, M, sA)
            bitsB = bits_for(T_ROWS, K, M, sB)
            iqA = compose(bitsA, K=K, L=L, seed=KA, M=M)
            iqB = compose(bitsB, K=K, L=L, seed=KB, M=M)
            iq = np.sqrt(0.5) * iqA + np.sqrt(0.5) * iqB
            gA, _ = decode(iq, KA, candidates=cand)
            gB, _ = decode(iq, KB, candidates=cand)
            berA += ber(bitsA, gA); berB += ber(bitsB, gB)
            inf += ber(bitsA, read(iq, L, K, KA, M=M))
        berA /= TRIALS; berB /= TRIALS; inf /= TRIALS
        out.append((L, berA, berB, inf))
        print(f"{L:>4} {berA:>12.4f} {berB:>12.4f} {inf:>15.4f}")
    return out


# ===========================================================================
# Sweep 6 -- coherent QAM and a higher differential order
# ===========================================================================
def sweep_alphabets(L=L_DEFAULT):
    K = L
    print(f"\n=== SWEEP 6  coherent 16-QAM (read_qam, informed single key) and 8-DPSK "
          f"(blind), equal split 0.5/0.5, clean ===")
    out = {}

    # Coherent 16-QAM, informed single-key reader (offset 0, carrier 0, M known).
    M = 16
    berA = berB = 0.0
    for (KA, KB), (sA, sB) in TRIAL_SEEDS:
        bitsA = bits_for(T_ROWS, K, M, sA)
        bitsB = bits_for(T_ROWS, K, M, sB)
        iqA = compose_qam(bitsA, K=K, L=L, seed=KA, M=M)
        iqB = compose_qam(bitsB, K=K, L=L, seed=KB, M=M)
        n = min(len(iqA), len(iqB))
        iq = np.sqrt(0.5) * iqA[:n] + np.sqrt(0.5) * iqB[:n]
        berA += ber(bitsA, read_qam(iq, L, K, KA, M=M))
        berB += ber(bitsB, read_qam(iq, L, K, KB, M=M))
    out["qam16_informed"] = (berA / TRIALS, berB / TRIALS)
    print(f"  16-QAM informed : BER_A={berA/TRIALS:.4f}  BER_B={berB/TRIALS:.4f}")

    # 8-DPSK, blind.
    M = 8
    berA = berB = 0.0
    for (KA, KB), (sA, sB) in TRIAL_SEEDS:
        bitsA = bits_for(T_ROWS, K, M, sA)
        bitsB = bits_for(T_ROWS, K, M, sB)
        iqA = compose(bitsA, K=K, L=L, seed=KA, M=M)
        iqB = compose(bitsB, K=K, L=L, seed=KB, M=M)
        iq = np.sqrt(0.5) * iqA + np.sqrt(0.5) * iqB
        gA, _ = decode(iq, KA, candidates=CANDIDATES)
        gB, _ = decode(iq, KB, candidates=CANDIDATES)
        berA += ber(bitsA, gA); berB += ber(bitsB, gB)
    out["dpsk8_blind"] = (berA / TRIALS, berB / TRIALS)
    print(f"  8-DPSK  blind   : BER_A={berA/TRIALS:.4f}  BER_B={berB/TRIALS:.4f}")
    return out


def verdict(main_rows):
    print("\n=== VERDICT ===")
    # best aggregate across the whole blind sweep
    best_good = max(r[6] for r in main_rows)
    # clean baseline (one screen, full power) decodes at BER ~0
    print(f"- Best 'both streams delivered' count anywhere in the blind sweep: {best_good} "
          f"(of 2).")
    equal = [r for r in main_rows if abs(r[0] - 0.5) < 1e-9]
    be = equal[0]
    print(f"- Equal split, clean: BER_A={be[4]:.3f}, BER_B={be[5]:.3f} "
          f"(both ~0.5 = no separation).")
    if best_good >= 2:
        print("- GO: at least one operating point delivers two co-channel streams.")
    else:
        print("- NO-GO: no power split / SNR delivers two usable co-channel streams at "
              "equal total power.")
        print("  The plain blind read does NOT separate overlaid keyed screens: the other "
              "screen acts as\n  full-power noise (no processing gain), and below ~9 dB SINR "
              "the blind acquisition\n  itself fails to lock.  Keyed-aperture division is a "
              "dead end for the pure read.")


def main():
    print("Co-channel keyed multiplexing of entroptics screens -- SIM ONLY, blind read.")
    main_rows = sweep_main()
    sweep_orthogonality()
    sweep_interference_vs_noise()
    sweep_width()
    sweep_alphabets()
    verdict(main_rows)


if __name__ == "__main__":
    main()
