# Relative range / range-rate — the harder half of GPS-free nav

Bearing is the easy half (`../mimo/relative_bearing.md`: <1° from the array, free). Range is harder;
here's the split (`relative_range.py`).

- **Range-rate (radial velocity) — CLEAN.** The receiver already recovers the carrier offset per burst
  = the Doppler shift = radial velocity. Measured over a single 53 ms burst: **±1.95 m/s @ 20 dB SNR,
  ±7.3 m/s @ 10 dB** (carrier RMS 1.6e-4 / 6.0e-4 cyc/row). Blind acquire loses it at 0 dB (the known
  blind-SNR wall; a tracked read goes lower). Improves with SNR / burst length / tracking across bursts.
  **Confirmed on hardware** (cabled two-clock bench, `doppler_hw.md`): the recovered carrier
  tracks an imposed Doppler-equivalent offset at ~1.1 m/s short-term precision / 2.74 m/s RMS across ±50 m/s,
  single burst, read blind across independent clocks — matching the sim.
- **Absolute range from RSS — COARSE** (and needs the sender's TX power): rel. range error =
  ln(10)·σ_dB/(10n). At σ_dB 2–3 dB that's **±23–35 % (n=2), ±12–17 % (n=4)**. A rough range, not a fix.
- **Precise absolute range needs** either **two-way TOF** (a reply — breaks the reply-free contract) or
  a time-synchronized network of cooperating nodes. Neither is free in a reply-free one-way mesh.

**So, for navigation without GPS:** lean on **bearing (array, <1°) + range-rate (Doppler) + own IMU motion →
bearings-only relative tracking**, which recovers relative *position over time* without a direct range
measurement; add an explicit ranging exchange only where an absolute scale is actually required. The
strong, free primitives are bearing and range-rate; absolute range is the deliberate-cost piece.

Completes the GPS-free positioning picture with `../mimo/relative_bearing.md`.
