# Unsynchronized two-way ranging — blind clock-rate correction (sim)

The hard part of two-way ranging between independent radios is the **clock-rate** error. A clock
*offset* cancels for free in a reciprocal exchange (each node times the other), but a clock *rate*
difference — untrimmed crystals differ by tens of ppm — biases the measured reply interval and, with
it, the range. The standard fix is a third timing exchange or a disciplined common clock. Entroptics
removes it for free: **the receiver already recovers the clock-rate ratio blind**, as the scale (and
carrier) of the keyed read. Correcting the TWR with that recovered rate makes the range limited by the
read's arrival-time precision, not by clock discipline.

## What was measured (`unsync_ranging.py`, no hardware)

An independent-clock receiver is modelled by resampling a keyed buffer (L=64, K=64, M=4, ~192k
samples) to a ppm-offset rate (scipy FFT resample — a band-limited **channel** model, not a codec op).
The blind `_discover_scaled` then recovers `d.scale = Rtx/Rrx` **without being told it** — that scale
*is* the clock-rate ratio. This read does no resampling (band-limited projection), so it stays
entroptics-pure.

| imposed offset | lock (closure-sig) | recovered-rate error |
|---|---|---|
| +5 ppm (scale 0.999995) | 437 (strong) | **2.1×10⁻⁷ = 0.21 ppm** |
| +20 ppm (scale 0.99998) | 437 (strong) | **8.3×10⁻⁷ = 0.83 ppm** |
| +50 ppm (scale 0.99995) | 437 (strong) | 2.1×10⁻⁶ = 2.1 ppm |

Across 5 noisy repeats at +20 ppm / 10 dB SNR the read is highly repeatable — **jitter 8.6×10⁻⁹**
(≈0.009 ppm); the residual is a small systematic ~4%-of-offset term, not noise, and tightens with
capture length and SNR.

**Regime note (and why reciprocity saves it).** `scale = Rtx/Rrx`. A positive offset (receiver crystal
fast) gives scale < 1, which the blind discovery reads strongly. A negative offset (receiver slow)
gives scale > 1, where `_discover_scaled` clamps its search bracket at unity and locks weakly (the −50
ppm case: closure-sig 17.6, barely corrected). This does **not** limit ranging: in a reciprocal
exchange node A sees `scale_AB` and node B sees `scale_BA ≈ 1/scale_AB`, so **one node always sits in
the scale ≤ 1 good regime** — the ranging uses that node's read. The achievable precision is the
strong-lock number above.

## Ranging impact

TWR range error from a relative clock-rate offset `e` (offset already cancelled by reciprocity):
`d_err = ½·c·Treply·e`.

| reply delay | uncorrected (20 ppm) | corrected (0.83 ppm) | corrected (0.21 ppm floor) |
|---|---|---|---|
| 1 ms | 3.00 m | **0.124 m** (24×) | **0.031 m** (97×) |
| 10 ms | 29.98 m | 1.24 m (24×) | 0.31 m (97×) |
| 100 ms | 299.8 m | 12.4 m (24×) | 3.1 m (97×) |

So the blind-rate correction turns a metres-scale unsynchronized-TWR error into **decimetre-to-
centimetre**, a 24–100× reduction, with ranging then limited by the read's arrival-time precision.
This is grounded in hardware: the same blind scale/carrier read locked on the cabled two-clock bench
at the exact 0.8333 ratio (closure-sig 96.6) and tracked Doppler to ~1–3 m/s (`doppler_hw.md`).

## Gaps

- Sim + the hardware blind-rate read, **not a full end-to-end ranging demo**. The arrival-time floor
  that this correction leaves is characterized separately (`toa_floor.md`): the blind read pins ToA
  to σ ≈ 0.02–0.17 m one-way in sim (2 MHz, 5–20 dB) and **0.10 m measured on the real cabled capture**
  (2.4 MHz, ideal-buffer floor 0.000), so both residuals are decimetre-scale. An *absolute* ToA vs a
  known path-length delta still wants a split-cable bench that imposes a known delay.
- The scale>1 clamp is a discovery-side artifact; reciprocity sidesteps it but a symmetric read would
  be a nice refinement.
- Relative positioning between cooperating nodes is not yet flown, and fusing range with bearing is
  sim-level. Absolute earth location still needs an external reference; this gives the relative
  geometry of the cooperating nodes only.

Completes the relative-positioning picture with `relative_range.md` (range-rate), `relative_bearing.md`
(AOA) and `doppler_hw.md` (hardware Doppler).
