"""Relative bearing (AOA) from the keyed array signal -- the GPS-free positioning primitive.

A fleet member already recovers a neighbour's *array angle* for free: ``blind_separate`` grid-searches
the steering angle and keeps the one whose keyed constant-modulus lock is best (spatial_mimo). That
angle IS the neighbour's bearing in the array frame -- no pilots, no GPS, measured from the neighbour's
own transmission (the entroptics "measure, don't pre-share" principle applied to position).

This measures how good that bearing is vs SNR and array size, and the one caveat the calibration study
flagged: data SEPARATION needs no array calibration, but ANGLE estimation does -- a static per-element
phase error re-labels the angle (bias). So a calibrated array gives an absolute relative bearing; an
uncalibrated one gives a consistent-but-biased bearing (fine for differential/relative geometry).

Run (from this dir):  PYTHONPATH=../../src <venv>/python relative_bearing.py
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entroptics_rf.waveform import compose
from mimo_calibration import L_DEFAULT, T_ROWS, bits_for, build_Y, static_error, blind_separate, steering

rng = np.random.default_rng(0)
KEY, BSEED = 1234, 1

def source():
    s = compose(bits_for(T_ROWS, L_DEFAULT, 4, BSEED), K=L_DEFAULT, L=L_DEFAULT, seed=KEY, M=4)
    return s[None, :]                                   # 1 x n

def aoa_error(N, theta_true, snr_db, cal_phase_deg, trials=8):
    S = source(); A = steering(N, theta_true).reshape(N, 1)
    errs = []
    for t in range(trials):
        r = np.random.default_rng(100 * t + 1)
        cal = static_error(N, cal_phase_deg, 0.0, r) if cal_phase_deg > 0 else None
        Y = build_Y(S, A, snr_db, N, cal_vec=cal, drift=None, rng=r)
        _, theta_hat = blind_separate(Y, KEY, L_DEFAULT, N)
        errs.append(theta_hat - theta_true)
    e = np.array(errs)
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(e))   # rms, bias (deg)

print("== bearing (AOA) accuracy from the keyed array, CALIBRATED (no per-element error) ==")
print("  N  | SNR dB | true   | RMS err (deg) over angles {-45..45}")
for N in (2, 4):
    for snr in (20, 10, 0):
        rmss = [aoa_error(N, th, snr, 0.0)[0] for th in (-45, -30, -15, 0, 15, 30, 45)]
        print(f"  {N}  |  {snr:>4d}  | sweep  | mean RMS {np.mean(rmss):5.1f}  (worst {max(rmss):4.1f})")

print("\n== calibration caveat: UNCALIBRATED array (static per-element phase error), N=4, SNR 20 dB ==")
print("  per-elem phase RMS | AOA RMS err | AOA bias   (separation is immune; angle is not)")
for cp in (0, 5, 10, 20):
    rms, bias = aoa_error(4, 20.0, 20.0, float(cp), trials=12)
    print(f"  {cp:>6d} deg          | {rms:7.1f}     | {bias:+6.1f} deg")
print("\n=> a CALIBRATED keyed array gives a usable GPS-free bearing at modest SNR; an uncalibrated one")
print("   adds a consistent bias (re-labelled angle) -> fine for relative/differential geometry, biased")
print("   for absolute. Combine bearings across neighbours (+ array orientation) for relative position.")
