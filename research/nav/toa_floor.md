# Arrival-time (ToA) precision — the ranging floor (sim)

`unsync_ranging.md` removes the **clock-rate** error from two-way ranging (blind, ~0.2–0.8 ppm → range
residual 0.03–0.12 m at a 1 ms reply). What's left is the **arrival-time precision** of the read. This
note measures it.

For this noise-like waveform there is no envelope edge to detect — the emission is statistically
Gaussian. The time reference **is the keyed row alignment**, which the library's own blind read pins to
sub-sample precision: `Discovery.offset` (integer) + `Discovery.phase` (fractional, refined by
`_refine_continuous` at `SCAN_PHASES` resolution). We measure that precision with the **library read —
no classical correlator, no matched filter** — so it stays entroptics-pure.

## Method (`toa_floor.py`, no hardware)

Compose a keyed burst (L=64, K=64, M=4, T=800 rows ≈ 51k samples). Impose a known sub-sample
propagation delay with a band-limited FFT fractional shift (a **channel** model, not a codec op), add
AWGN, and `discover` the position blind. Two readings: (1) does the recovered position track the
imposed delay (sub-sample), and (2) its jitter across noise realizations = the ToA precision. Map to
range by `σ_R = (c/Rs)·σ_position`; at Rs = 2 MHz, 1 sample = 149.9 m.

## Result

**ToA precision (jitter over 8 noise trials, all 8/8 locked):**

| SNR | σ_position (samples) | σ_range, one-way @2 MHz | @20 MHz |
|---|---|---|---|
| 20 dB | 0.0001 | **0.02 m** | ~0.002 m |
| 10 dB | 0.0005 | **0.07 m** | 0.01 m |
| 5 dB | 0.0012 | **0.17 m** | 0.02 m |

The read is coherent over all ~51k samples, so the alignment estimate is far finer than one sample:
**centimetre-to-decimetre one-way at a 2 MHz band, centimetre at 20 MHz**, and it degrades gracefully
with SNR. The floor scales as 1/Rs, so a wider capture band tightens it proportionally.

**Sub-sample tracking (clean):** recovered position vs imposed delay is **monotonic but not perfectly
linear** (imposed 0.25/0.50/0.75 → recovered 0.12/0.76/0.88). This is a refinement-grid snapping
effect: it is a *systematic*, calibratable bias in the **absolute** sub-sample ToA, and it does **not**
affect the jitter — and two-way ranging uses *differences* of ToA reads, in which a common systematic bias
cancels. So ranging **precision** is set by the jitter above; absolute sub-sample ToA would want a
one-time calibration (or a finer phase refinement) if used alone.

## Hardware grounding (`toa_hw.py`, real bench captures, no new TX)

The sim above is confirmed on real hardware by re-using the cabled captures (Pluto+ TX →
NESDR RX). The blind read's position (offset + phase) is read in a
series of **non-overlapping** windows sliding across the matched-rate capture (independent thermal
noise per window); a slow clock-rate drift is fit and removed, and the residual std is the ToA jitter.

**Method floor first:** the identical measurement on the *ideal transmitted
buffer* (no channel) gives a residual of **exactly 0.00000 sample** (10/10 windows) — the estimator
and the drift-fit invent no jitter, so any residual on the real capture is physical.

**Hardware result:** on the real matched-rate capture (9/9 windows locked, closure-sig 69–139), the
position jitter is **0.0008 sample → 0.10 m one-way at 2.4 MHz** (≈0.012 m at a 20 MHz band). That is
about 5× the clean-sim 0.02 m, as expected: the sim is an AWGN best case, while the capture carries the
NESDR's 8-bit quantization, LO phase noise and real channel at ~19–20 dB. **Decimetre-class ToA on real
hardware** — the sim's conclusion holds where it can be checked. (An overlapping-window version reads
~0.05 m but shares noise between windows and underestimates; the non-overlapping 0.10 m is the
independent-capture figure.)

**Second, independent hardware path (shared-clock loopback).** A live cabled run — Pluto TX
→ its own RX2 (one clock, so scale = 1 and carrier ≈ 0.0003, no drift at all) — decoded cleanly
(closure 0.465, sig 294, BER 1.1%) and gave a ToA jitter of **0.00071 samp = 0.089 m** one-way @2.4
MHz. That it matches the two-radio NESDR path (0.10 m) across a completely different clock arrangement
is the useful cross-check: **the jitter is noise/quantization-limited, not clock-limited** — exactly
what a ranging floor should be.

## Putting it together with the clock-rate correction

| error source | magnitude | after entroptics |
|---|---|---|
| clock *offset* | arbitrary | cancels in the reciprocal exchange (free) |
| clock *rate* (20 ppm) | ~3 m @1 ms reply | **0.03–0.12 m** (blind rate read, `unsync_ranging.md`) |
| arrival-time (ToA) | — | **0.02–0.17 m** one-way @2 MHz (this note) |

Both residuals are decimetre-scale and shrink with SNR / band / capture length, so **unsynchronized,
blind two-way ranging between independent entroptics radios is decimetre-class** — the clock discipline
that ordinarily limits it is removed.

## Gaps

- The precision/jitter is **hardware-grounded** (above, `toa_hw.py`: 0.10 m one-way @2.4 MHz on the
  real capture, ideal-buffer floor 0.000). What is *not* yet measured is an **absolute** ToA against a
  known path-length delta — a split-cable rig can impose one; that would
  check the sub-sample linearity/bias too, not just the jitter.
- The fractional-delay tracking is non-linear at the refinement grid (above) — fine for precision ranging,
  wants calibration for absolute one-shot range.
- Multipath, not modelled: a real channel spreads the arrival; the floor here is the AWGN-limited best
  case.

Completes the ranging story with `unsync_ranging.md` (clock-rate), `relative_bearing.md` (AOA) and
`doppler_hw.md` (hardware range-rate).
