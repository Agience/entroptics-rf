# Range-rate / Doppler — hardware validation

The relative-range note (`relative_range.md`) established **range-rate from the recovered carrier** as the
clean GPS-free-nav primitive, in simulation: **±1.95 m/s @ 20 dB** over a single ~53 ms burst. This note
records the **hardware measurement** of that primitive, on the cabled two-clock bench.

## Bench

Two-clock path: **Pluto TX1 → 30 dB → NESDR**, two *independent*
crystals. Current waveform (K = L = 64, keyed row radius, DQPSK), **TX 2.0 MS/s → RX 2.4 MS/s** (occupancy
0.83), 915 MHz, cabled, low gain (tx_gain −26 dB). Blind receiver: `waveform._discover_scaled(x, key, [64],
carrier_span=0.3, join_rows=9001)` on the 60k–176k sample window — told only the key; it recovers the clock
ratio (scale), start, width, alphabet, and carrier.

**Context — a 2-clock decode on the same bench:** blind decode found L = 64, M = 4,
scale = 0.83333 (the exact 2.0/2.4 ratio, read blind), closure-significance 96.6 (accepted), **BER = 0.0001**
over 1509 rows. The current waveform decodes blind across independent clocks essentially error-free on real
hardware.

## Method

A frequency offset at 915 MHz is Doppler-equivalent: **v = Δf · c / f_c = 0.328 m/s per Hz**. We impose a
**known** offset by tuning the Pluto TX LO to `FC + Δf` while the NESDR stays at `FC`, and read back the
carrier the blind receiver recovers: `f_Hz = d.carrier [cycles/row] · R_tx / L` (R_tx/L = 31 250 Hz/cycle
here). The Pluto↔NESDR LO difference is a drifting baseline (~1.3 ppm), so Δf = 0 baselines are interleaved
and subtracted by interpolation. Every capture is a single burst; TX is stopped and verified off after each.

## Result

All 13 captures locked cleanly (closure-significance ~265). Baseline LO offset ≈ **1180 Hz (1.29 ppm)**,
drifting a few Hz over the run.

| set Δf (Hz) | set v (m/s) | recovered Δf (Hz) | error (Hz) | error (m/s) |
|---:|---:|---:|---:|---:|
| +6   | +2.0  | −8.5   | −14.5 | −4.77 |
| +15  | +4.9  | +13.3  | −1.7  | −0.54 |
| +60  | +19.7 | +65.2  | +5.2  | +1.71 |
| +150 | +49.2 | +135.3 | −14.7 | −4.81 |
| −60  | −19.7 | −68.7  | −8.7  | −2.84 |
| −150 | −49.2 | −144.8 | +5.2  | +1.71 |

- **Short-term precision** (baseline scatter about the drift fit): **3.3 Hz ≈ 1.09 m/s**.
- **Recovered-Δf RMS error across ±50 m/s**: **8.3 Hz ≈ 2.74 m/s**.
- Recovered offsets track the imposed ones from **±5 m/s up**; the ±2 m/s (6 Hz) point is at the noise floor.

## Interpretation

The range-rate / Doppler primitive, previously sim-only, is **confirmed on real hardware at ~1–3 m/s** —
matching the simulation's ±1.95 m/s @ 20 dB — from a **single burst**, read **blind across independent
clocks**. This is the clean GPS-free-nav primitive (bearing + range-rate + IMU → bearings-only relative
tracking; see `relative_range.md` and `../mimo/relative_bearing.md`).

**Caveats / how to tighten.** Single burst at ~19–20 dB SNR; crystal drift between captures limits the
absolute baseline; the ±2 m/s point needs more than one burst. Back-to-back captures, higher SNR, or short
integration would lower the floor. The absolute (not differential) baseline is dominated by the untracked
crystal drift — range-rate (the *change*) is what this measures and what matters.

**Reproduce:** on a cabled Pluto+/NESDR bench. Decode gotchas that cost time and are worth knowing: across clocks use `decode_scaled`/
`_discover_scaled` (not `decode`); set **TX rate < RX rate** so the full-occupancy signal fits the RX window;
set **`join_rows` = buffer rows (here 9001)**, not the default 4096, or a capture starting deep in the cyclic
buffer never finds the reference row; and score BER with a cycle-aware differential reader, not a naive bit compare.
