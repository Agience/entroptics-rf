"""Relative range / range-rate — the harder half of GPS-free nav (bearing is the easy half, see
../mimo/relative_bearing.py). The picture:
  * RANGE-RATE (radial velocity) is a CLEAN primitive: the receiver already recovers the carrier
    offset per burst, which is the Doppler shift = radial velocity. Measured here vs SNR.
  * ABSOLUTE RANGE is weak one-way: RSS ranging is coarse (and needs the sender's TX power), precise
    range needs two-way TOF (a reply -> breaks reply-free) or a synced TDOA receiver network.
  * Integration: bearing (strong) + bearing-rate + range-rate (Doppler) + own IMU motion -> bearings-
    only relative tracking recovers relative POSITION over time WITHOUT a direct range measurement.
Run (from this dir):  PYTHONPATH=../../src <venv>/python relative_range.py
"""
import numpy as np
from entroptics_rf import waveform as wf

FC, RATE, L, K, M = 915e6, 2.4e6, 64, 64, 4
C = 3e8
rows_per_s = RATE / L
rng = np.random.default_rng(0)
bits = rng.integers(0, 2, 2000 * 2 * K)
iq0 = wf.compose(bits, K, L, 1234, M=M)
n = np.arange(iq0.size)

def carrier_rms(snr_db, f0_cyc_row, trials=10):
    sig = float(np.mean(np.abs(iq0) ** 2)); N0 = sig / 10 ** (snr_db / 10)
    errs = []
    for t in range(trials):
        r = np.random.default_rng(10 * t + 1)
        x = iq0 * np.exp(2j * np.pi * (f0_cyc_row / L) * n)                 # inject carrier (cyc/row)
        x = x + (r.standard_normal(iq0.size) + 1j * r.standard_normal(iq0.size)) * np.sqrt(N0 / 2)
        d = wf.discover(x, 1234, candidates=(63, 64, 65), carrier_span=0.3)
        errs.append(getattr(d, "carrier", np.nan) - f0_cyc_row)
    return float(np.sqrt(np.nanmean(np.array(errs) ** 2)))

print("== RANGE-RATE from the recovered carrier (Doppler), single burst (~%d rows, %.1f ms) ==" % (iq0.size // L, iq0.size / RATE * 1e3))
print("  SNR dB | carrier RMS (cyc/row) | Doppler RMS (Hz) | radial-velocity RMS (m/s)")
for snr in (20, 10, 0):
    e = carrier_rms(snr, 0.01)
    f_hz = e * rows_per_s                                                   # cyc/row -> Hz
    v = f_hz * C / FC                                                       # Doppler -> m/s
    print(f"   {snr:>4d}  |   {e:.2e}        |   {f_hz:7.1f}       |   {v:6.2f}")

print("\n== ABSOLUTE RANGE from RSS (coarse; also needs the sender's TX power) ==")
print("  rel. range error = ln(10)*sigma_dB/(10*n)  [n = path-loss exponent]")
print("  sigma_dB | n=2    n=3    n=4")
for s in (2, 3, 6):
    print(f"   {s:>4d}    | " + "  ".join(f"{np.log(10)*s/(10*nn)*100:4.0f}%" for nn in (2, 3, 4)))

print("""
=> Range-rate (Doppler) is clean and improves with SNR / burst length -> good radial-velocity.
   Absolute range from one-way RSS is coarse (tens of %); precise range needs two-way TOF (a reply,
   which breaks the reply-free contract) or a TDOA receiver network (synced). So for navigation without GPS:
   lean on BEARING (array, <1deg) + RANGE-RATE (Doppler) + own IMU motion -> bearings-only tracking
   gives relative position over time; add a ranging exchange only where an absolute scale is required.""")
