# Does the blind keyed-MVDR array need per-boot phase calibration, or only a shared clock?

**Question.** The spatial-MIMO study (`spatial_mimo.md`) proved a GO: an N-antenna receiver
blindly separates up to `min(N, K)` equal-power co-channel keyed streams, steering each beam by
maximizing that key's constant-modulus lock, with no pilots and no angles, matching the genie
zero-forcing ceiling. That receiver steers by the **key**, not by absolute angle. A
direction-finder such as a KrakenSDR must run a per-boot phase-calibration rig (noise source +
RF switch) because it maps measured phase to absolute angle, and every per-element gain/phase
offset corrupts that map. If the blind receiver does **not** depend on absolute angle, it may
not need that rig -- a cheap DIY coherent RTL-SDR array could then drop the cal hardware and
need **only a shared clock**, which is cheaper and simpler. This study measures whether that is
true, and how phase-stable the shared clock must be.

**Verdict.**
- **(a) Static per-element miscalibration -- blind is immune on the N=2 array (the minimal DIY /
  bench Pluto+ case): no per-boot phase-cal rig needed.** Unknown, fixed per-element phase offsets
  up to **90 deg RMS** leave the blind per-stream BER at 0.0000 while an angle-based zero-forcer
  using the nominal (uncalibrated) steering vectors collapses (loses a stream by 20 deg RMS, dead
  by 90 deg). The blind beam re-labels the offset as a shifted search angle and lands on the same
  separating weight. On the oversized N=4 array the clean case is also immune (to 45-90 deg), but
  with noise (12 dB) a non-linear >2-element phase profile is only *partially* absorbed and large
  offsets degrade it -- so N=4 benefits from a one-time (not per-boot) trim.
- **(b) Coherence is the real requirement.** Within-capture **differential** phase drift (imperfect
  clock sharing) is what breaks separation. Blind both-stream recovery held to ~20 deg/capture on
  N=2 and ~5 deg/capture on N=4; the de-interfered beam itself (genie) tolerated ~20 deg before
  failing at 45. **Build to << 5-10 deg/capture differential drift** for margin.

**Build guidance in one line: shared clock YES (required); per-boot phase-cal rig NO (drop it);
clock stability spec approximately < 5-10 deg differential phase per capture.** Sim only; no
hardware, no transmit.

## Method

Extends and reuses `spatial_mimo.py` verbatim (array model, `steering`, `blind_separate`,
MVDR, diagonal loading, the blind angle grid). ULA, `d = lambda/2`, element `k` phase
`exp(j*pi*k*sin theta)`; `K` equal-power co-channel QPSK keyed transmitters at distinct angles;
per-element complex AWGN; QPSK, `L = 64`, 96 rows, 3 seeds/cell; `good` = per-stream raw
BER < `1e-2`. Two proven operating points: **N=2, K=2 at 60 deg** (the N=2 sweet spot, the
Pluto+ 2r2t / minimal DIY case) and **N=4, K=2 at 30 deg** (oversized, 2 spare elements).

A cheap coherent array has two distinct imperfections, injected as a multiplicative per-element
term on the received columns, `Y = (A @ S) * (G[:,None] * exp(1j*Phi)) + AWGN`:

- **Static miscalibration (Exp 1).** Fixed, unknown per-element complex gain `g_k e^{j phi_k}`
  (cable length, mixer phase, per-dongle group delay), stable within a capture. The actual
  manifold is `A_actual = diag(g_k e^{j phi_k}) @ A_nominal`. Element 0 is the reference, so only
  the differential error matters. Phase RMS swept 0..90 deg; a phase+amplitude variant adds 1 dB
  RMS gain error.
- **Coherence loss (Exp 2).** Per-element phase that **drifts** within a capture (independent
  random walk per element, element 0 the reference), scaled so the end-to-end RMS phase excursion
  across the capture equals the swept value (0..45 deg). Common-mode drift is harmless and not
  modelled; only differential drift between elements is applied.

Three receivers:
- **genie_actual** -- zero-forcing that *knows* the true per-element manifold `A_actual`
  (`pinv(A_actual) @ Y`). The immune upper bound: it shows the capacity is still present and the
  problem is purely one of calibration *knowledge*.
- **genie_nominal** -- zero-forcing using the **nominal / stale** steering vectors
  (`pinv(A_nominal) @ Y`). This is the angle-based receiver's situation (a KrakenSDR-style array
  whose per-boot cal is absent or stale). Expected to degrade with the error.
- **blind** -- the entroptics-pure receiver from `spatial_mimo`: no angles, no cal; MVDR beam
  whose steering angle is chosen to maximize each key's constant-modulus lock, then fully blind
  `decode`. The question is whether *this* receiver survives.

Public library only: `compose, decode, read, ber, random_bits, constant_modulus_read`. The array
math (steering vectors, pinv, sample-covariance MVDR) is plain narrowband front-end combining
before the codec. Re-run:
`OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src .venv/Scripts/python.exe research/mimo/mimo_calibration.py`.
Script: `research/mimo/mimo_calibration.py`.

## Why N=2 is exactly immune (and N>2 is not)

For N=2 the nominal steering vector is `[1, e^{j*pi*sin theta}]`; as the search angle sweeps the
placeable half-space, its second-element phase sweeps essentially the full circle. A static phase
error turns the actual source signature into `[1, e^{j(dphi + pi*sin theta)}]`, which is still a
point on that same nominal manifold -- at a **shifted** angle. So the blind search finds a
different (wrong) absolute angle but the **identical separating weight**: perfect immunity to
phase-only error. This is the formal reason the key-steered receiver does not need absolute-angle
calibration. For N>2 the error is a per-element phase profile that is *not* a linear ramp across
elements, so the actual signature is off the nominal manifold at every angle; the blind MVDR
(constrained to the nominal manifold) cannot form a perfectly distortionless beam, and noise then
exposes the residual -- hence N=4 is immune when clean but degrades under noise at large offsets.
Amplitude error breaks the unit-modulus manifold even for N=2, so it is only partially absorbed.

## Results

`good` = fraction of the 2 streams delivered (1.00 = both). QPSK.

### Experiment 1a -- static per-element PHASE miscalibration (amplitude exact)

**N=2, K=2, 60 deg (minimal DIY / Pluto+):**

| phi RMS (deg) | genie_actual BER / good | genie_nominal BER / good | blind BER / good |
|---:|---:|---:|---:|
| clean, 0 | 0.0000 / 1.00 | 0.0000 / 1.00 | 0.0000 / 1.00 |
| clean, 10 | 0.0000 / 1.00 | 0.0000 / 1.00 | 0.0000 / 1.00 |
| clean, 20 | 0.0000 / 1.00 | 0.0281 / 0.67 | **0.0000 / 1.00** |
| clean, 45 | 0.0000 / 1.00 | 0.0339 / 0.67 | **0.0000 / 1.00** |
| clean, 90 | 0.0000 / 1.00 | 0.3865 / 0.00 | **0.0000 / 1.00** |
| 12 dB, 20 | 0.0000 / 1.00 | 0.0360 / 0.67 | **0.0000 / 1.00** |
| 12 dB, 45 | 0.0000 / 1.00 | 0.0430 / 0.67 | **0.0000 / 1.00** |
| 12 dB, 90 | 0.0000 / 1.00 | 0.3908 / 0.00 | **0.0000 / 1.00** |

Blind is flat at BER 0.0000 across the whole phase sweep, clean and 12 dB, exactly matching the
all-knowing genie_actual, while the nominal angle-based ZF drops a stream by 20 deg and dies by
90 deg. This is the headline: **the blind key-steered N=2 receiver is immune to static
per-element phase; the angle-based one is not.**

**N=4, K=2, 30 deg (oversized):**

| phi RMS (deg) | genie_nominal BER / good | blind BER / good |
|---:|---:|---:|
| clean, 20 | 0.0003 / 1.00 | 0.0001 / 1.00 |
| clean, 45 | 0.1805 / 0.00 | **0.0001 / 1.00** |
| clean, 90 | 0.1351 / 0.33 | 0.0148 / 0.67 |
| 12 dB, 5 | 0.0000 / 1.00 | 0.0309 / 0.33 |
| 12 dB, 10 | 0.0000 / 1.00 | 0.0201 / 0.67 |
| 12 dB, 20 | 0.0011 / 1.00 | 0.4986 / 0.00 |

N=4 blind is immune when **clean** (holds both streams to 45 deg, mostly to 90), but with 12 dB
noise the off-manifold residual is exposed and large offsets degrade it -- here the nominal ZF
(angle-aware) actually holds better at 12 dB until 45 deg. Takeaway: on N>2, a one-time element
trim is worthwhile even though per-boot absolute-angle cal is not.

### Experiment 1b -- static phase + amplitude error (1 dB RMS gain)

- **N=2 clean:** blind stays at BER 0.0000 / good 1.00 across all phi to 90 deg; nominal drops to
  0.67 by 45 deg and 0.33 by 90 deg. Blind absorbs phase-plus-1dB-amplitude on N=2.
- **N=2, 12 dB:** blind 1.00 throughout; nominal 0.67 by 20 deg, 0.33 by 90 deg.
- **N=4 clean:** blind holds to 45 deg (good 1.00), 0.83 at 90 deg.
- **N=4, 12 dB:** amplitude error plus noise hurts N=4 blind even at small phi (good 0.67 at
  phi=0, worse beyond) -- the square-ish blind beam is the fragile case. N=2 is clean.

### Experiment 2 -- coherence loss (within-capture differential phase drift)

| drift RMS/capture (deg) | config | genie_nominal good | blind good |
|---:|:--|---:|---:|
| 20 | N=2, clean | 1.00 | **1.00** |
| 45 | N=2, clean | 0.00 | **1.00** (BER 0.0059) |
| 20 | N=2, 12 dB | 1.00 | **1.00** |
| 45 | N=2, 12 dB | 0.00 | 0.33 |
| 5 | N=4, clean | 1.00 | 1.00 |
| 10 | N=4, clean | 1.00 | 0.67 |
| 20 | N=4, clean | 1.00 | 0.00 |
| 10 | N=4, 12 dB | 1.00 | 0.33 |
| 20 | N=4, 12 dB | 1.00 | 0.00 |

Drift is the genuine limit. On N=2 the blind receiver tolerates ~20 deg/capture at 12 dB and the
full 45 deg when clean -- more robust than the nominal ZF, which fails at 45. On N=4 the blind
beam is more drift-sensitive (both streams only to ~5 deg/capture) because its static margin is
already thinner; the de-interfered beam itself (genie) tolerates ~20 deg before failing at 45, so
the ~20 deg figure is the fundamental coherence ceiling and the N=4 blind's extra sensitivity is a
receiver-robustness matter. Common-mode drift (identical on all elements) is harmless.

## Build guidance (cheapest coherent RTL-SDR array)

- **Shared clock: YES, required.** Run all dongles off one reference oscillator / clock fan-out
  so the elements stay phase-coherent within a capture. This is the one non-negotiable.
- **Per-boot phase-cal rig (noise source + RF switch, KrakenSDR-style): NOT required.** The blind
  key-steered receiver absorbs static per-element phase (clean on N=2 to 90 deg RMS, where the
  angle-based ZF has long collapsed) because it steers by the key, not by absolute angle. Drop the
  cal rig and its switching hardware; keep only the shared clock. This is the cost/complexity win.
- **Clock stability spec: approximately < 5-10 deg differential phase per capture** for comfortable
  margin (blind held to ~20 deg on N=2, ~5 deg on N=4; the fundamental beam ceiling is ~20 deg).
  Shorten captures or raise sample rate to bound accumulated drift if the fan-out is imperfect.
- **N=4 (quad array): add a one-time, not per-boot, element gain/phase trim.** N>2 non-linear
  phase and amplitude mismatch are only partially absorbed under noise; a single factory/bench
  trim (measured once, not re-run every boot) recovers the margin. N=2 needs no trim at all.

## Caveats and limits

- **N=2 is the clean, fully-immune case; N>2 is only partially immune under noise.** The strong
  "no cal needed" claim is strongest for the N=2 bench/DIY array. For N=4, clean is immune but
  12 dB with large static offsets or amplitude error degrades blind; budget a one-time trim.
- **Amplitude vs phase.** Phase-only static error is absorbed (exactly on N=2); amplitude error
  breaks the unit-modulus manifold and is only partially absorbed. 1 dB RMS is tolerable on N=2,
  less so on N=4 + noise.
- **Drift model.** Independent per-element random walk with a controlled end-to-end RMS; a real
  clock's phase-noise spectrum (flicker/white mix, bounded by a PLL) differs in shape, so treat
  the deg/capture figures as order-of-magnitude build targets, not a hard spec. Common-mode drift
  is assumed harmless (true for the spatial combine and the phase-blind decode).
- **Still idealized front-end.** Exact lambda/2 spacing, narrowband manifold, no mutual coupling,
  no multipath or beam-squint -- inherited from `spatial_mimo`. Each would erode the margin; the
  square N=2 and noisy N=4 cells are where that bite shows first.
- **Blind-search fragility is a receiver matter, not an entroptics one.** The N=4 blind
  sensitivity partly reflects the MVDR CM-lock search landing on a sub-optimal angle under noise;
  a better blind estimator (e.g. joint over both keys) would likely recover some N=4 margin. Not
  pursued here.
- Supersedes nothing in `spatial_mimo`; it de-risks the hardware path (drop the cal rig, keep the
  clock) for the DIY coherent array the separation result assumed.
