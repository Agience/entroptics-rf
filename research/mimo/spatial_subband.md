# Spatial x sub-band: the two separation axes stack to set the fleet throughput ceiling

**Question.** entroptics has two proven ways to carry co-channel keyed streams for one receiver.
**Spatial** (`research/mimo/spatial_mimo.md`, GO): an N-element array separates up to
`min(N, #angles)` equal-power co-channel streams arriving from distinct directions, by a blind
keyed-MVDR combine -- the beam is steered to the angle that maximizes one key's constant-modulus
lock, which nulls the other angles. **Sub-band** (frequency-division): disjoint narrowband
sub-bands at distinct centres inside one allocation each carry a keyed screen, read from one
capture by a band-limited sinc projection. Do the two axes **compose** -- does one receiver then
handle more co-channel keyed streams than either axis alone? This is the fleet's throughput
ceiling.

**Verdict: GO -- they stack, cleanly.** The validated fleet-capacity formula is

> **delivered co-channel keyed streams = min(N, #angles) x #disjoint-sub-bands**

measured end to end with a blind receiver: a **2-element** array x **2** sub-bands delivers **4**
co-channel streams (`2x2`); a **4-element** array reaches **4** (`2 angles x 2`), **6**
(`2 x 3`) and a full **8** (`4 angles x 2`). Blind matches the genie (true-angle zero-forcing)
ceiling in every feasible cell, and 16-QAM rides the same separation. The structure is
non-trivial and it is what makes the result worth measuring rather than assuming: the array
separates by **angle**, so it groups the B co-located sub-bands of one direction *together* --
one beam cannot tell them apart. The beam is steered using only **one** of the group's keys (its
sub-band 0), yet it delivers the group's **other** sub-bands just as well, because the narrowband
array steering is frequency-flat across the group. The sub-band read then separates within the
group. Where it breaks is the ordinary spatial wall inherited unchanged -- close angles on a
square array -- which now takes out a whole B-sub-band group at once. Sim only; no hardware, no
transmit.

## Method

- **Transmitters.** `K = A x B`: at each of A distinct angles, B single-antenna transmitters,
  each with its own key, its own bits, and its own disjoint sub-band. Angles are centred on
  broadside with adjacent spacing `dtheta >= 30 deg` (the spatial GO rule); a geometry past the
  ULA +-75 deg half-space is skipped.
- **Sub-band channelizer** (front-end band read, not the codec). Each of B sub-bands occupies
  `1/B` of the band at a distinct centre tiling `(-0.5, 0.5)`. **Up**: zero-stuff the native-rate
  screen by B, interpolate with a windowed-sinc low-pass (cutoff `0.5/B`, half-length `16B`, Hann
  window), shift to centre; normalize to unit rms. **Down**: shift to baseband, same sinc
  low-pass, decimate by B -- recovering the native-rate screen estimate for one sub-band and
  band-limiting the others out. No FFT; the codec (`compose`/`decode`) only ever sees an
  integer-rate screen. This is the `band read` the project already treats as resampling-free
  front-end DSP.
- **Array channel** (reused from `spatial_mimo.py`): ULA, `d = lambda/2`, narrowband steering
  `exp(j*pi*k*sin theta)`. `Y[elem,n] = sum_angle steer(theta)[elem] * group_signal(angle)[n] +
  per-element complex AWGN`. Each group's signal is the sum of its B unit-rms sub-bands.
- **SNR is per-source, in-band.** Each sub-band source is unit power inside its `1/B` band, so
  full-band noise power `sigma^2 = B*10^(-SNR/10)` leaves `sigma^2/B` in each sub-band -- i.e.
  `SNR` is what a sub-band stream sees in its own band, directly comparable to the per-source SNR
  of `spatial_mimo.md`. `None` = clean.
- **Receiver, blind (entroptics-pure).** Per angle group: grid-search the array steering angle
  (coarse 4 deg then 0.5 deg refine over the half-space); at each angle form the **MVDR** weight
  from the sample covariance alone (light diagonal loading), channelize **sub-band 0** from the
  combined capture, and keep the angle whose sub-band 0 reads the lowest `constant_modulus_read`
  under its key (best keyed lock -- this nulls the other angles). With that one beam, channelize
  and blind-`decode` **all B** sub-bands of the group.
- **Receiver, genie (ceiling).** Replace the blind combine with true-angle zero-forcing
  `W = pinv(A)`; the sub-band read is the **same** blind `decode`. Genie vs blind therefore
  isolates the spatial-combine quality alone.
- No pilots, preamble, equalizer, channel estimation, FFT, resampling, or joint ML in the codec.
  Public library only: `compose, decode, compose_qam, read_qam, read, ber, random_bits,
  constant_modulus_read`. QPSK (M=4), L=64, 96 rows, 3 seeds/cell, `good` = raw BER < 1e-2.

Re-run: `OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src .venv/Scripts/python.exe
research/mimo/spatial_subband.py`. Script: `research/mimo/spatial_subband.py`.

## Results

`good_g` / `good_b` = mean delivered streams per trial, out of the `A*B` transmitted and the
`min(N,A)*B` ceiling, for genie and blind. QPSK unless noted.

### 1. N = 2 array (the bench Pluto+ 2r2t): 2 angles x 2 sub-bands = 4 streams

| dtheta | SNR | BER_genie | good_g | BER_blind | good_b | ceiling |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | clean | 0.0000 | 4.00 | 0.0000 | 4.00 | 4 |
| 60 | 12 | 0.0008 | 4.00 | 0.0008 | 4.00 | 4 |
| 30 | clean | 0.0000 | 4.00 | 0.0000 | 4.00 | 4 |
| 30 | 12 | 0.0065 | 4.00 | 0.0060 | 4.00 | 4 |
| 15 | clean | 0.0000 | 4.00 | 0.0000 | 4.00 | 4 |
| 15 | 12 | 0.0871 | 0.00 | 0.0690 | 0.00 | 4 |

- **Four co-channel keyed streams on two antennas**, down to 12 dB per-source SNR at `>= 30 deg`
  angular separation -- `2x` the spatial axis alone (which gave 2 streams on N=2) and `2x` the
  sub-band axis alone. The axes multiply.
- **The spatial break is inherited unchanged, and now takes a whole group.** At 15 deg / 12 dB
  the square `N = A = 2` manifold is ill-conditioned (no spare element to average noise), so the
  beam cannot separate the two angles -- and because each angle is a *group*, losing the angle
  loses **both** its sub-bands at once (4 -> 0). Clean still resolves 15 deg. This is exactly the
  `spatial_mimo` square-array wall; the sub-band axis neither helps nor hurts it.
- **Blind == genie** everywhere, edging ahead at low SNR (the loaded MVDR amplifies noise less
  than pure zero-forcing). The sub-band-0 CM lock is a sharp enough beam-pointing objective to
  carry the whole group.

### 2. N = 4 array: oversizing, a third sub-band, and the full 8

| cell | dtheta | SNR | BER_genie | good_g | BER_blind | good_b | ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| A=2 x B=2 | 30 | clean | 0.0000 | 4.00 | 0.0000 | 4.00 | 4 |
| A=2 x B=2 | 30 | 12 | 0.0002 | 4.00 | 0.0002 | 4.00 | 4 |
| A=2 x B=3 | 30 | clean | 0.0000 | 6.00 | 0.0000 | 6.00 | 6 |
| A=2 x B=3 | 30 | 12 | 0.0001 | 6.00 | 0.0001 | 6.00 | 6 |
| A=4 x B=2 | 30 | clean | 0.0000 | 8.00 | 0.0000 | 8.00 | 8 |
| A=4 x B=2 | 30 | 12 | 0.0002 | 8.00 | 0.0002 | 8.00 | 8 |

- **The formula holds as each factor is varied.** Holding `min(N,A)` at 2 and raising B from 2 to
  3 gives 4 -> 6 (the `x #sub-bands` factor). Raising A from 2 to 4 on the N=4 array gives 4 -> 8
  (the `min(N, #angles)` factor). Both at BER ~1e-4 at 12 dB -- the four-element array has spare
  degrees of freedom (`N - A`) for the 2-angle cases, so these sit well clear of any break.
- **A full 8 co-channel keyed streams from one 4-element capture**, blind, at 12 dB. This is the
  quadcopter-relay number: 4 arms (N=4) x 2 sub-bands.

### 3. 16-QAM spot check (coherent) -- N = 4, A = 2 x B = 2, dtheta = 30

| SNR | BER_genie | good_g | BER_blind | good_b | ceiling |
|---:|---:|---:|---:|---:|---:|
| clean | 0.0016 | 4.00 | 0.0017 | 4.00 | 4 |
| 12 | 0.0077 | 4.00 | 0.0078 | 4.00 | 4 |

Coherent 16-QAM rides the stacked separation as well as QPSK: four clean co-channel streams, blind
== genie. The denser alphabet changes neither the spatial combine nor the sub-band split; each
path hands the coherent reader a de-interfered, band-limited capture and it does the rest.

## The fleet-capacity formula

> **streams(one receiver) = min(N, #resolvable angles) x #disjoint-sub-bands**

The two axes are **orthogonal degrees of freedom** and multiply. The array contributes
`min(N, #angles)` spatial channels (the `spatial_mimo` result, unchanged); each spatial channel,
once combined, is an ordinary single-antenna capture, so it carries whatever the band supports in
disjoint sub-bands -- `#disjoint-sub-bands` of them. The product is the ceiling, and the blind
receiver reaches it. For the bench and the relay:

| platform | N | typical #angles | sub-bands B | co-channel streams |
|---|---:|---:|---:|---:|
| Pluto+ 2r2t (bench today) | 2 | 2 | 2 | **4** |
| Pluto+ 2r2t | 2 | 2 | 3 | 6 |
| quadcopter relay (1 ant/arm) | 4 | 4 | 2 | **8** |
| quadcopter relay | 4 | 4 | 3 | 12 |

Sub-bands cost nothing on the array side and angles cost nothing on the frequency side, so the
fleet planner can trade one for the other: short of angular separation (crowded sky, few
elements) buy capacity with sub-bands; short of clean band buy it with angles and antennas.

## Where it breaks (and what it costs)

- **The per-angle spatial rules still apply, per group.** Everything from `spatial_mimo.md`
  carries over to the *angle* axis unchanged: keep `#angles <= N` (ideally `<= N - 1` for
  robustness), sources `>= ~30 deg` apart, and mind the square-array low-SNR wall. The new point
  is that an angle is now a **group of B sub-bands**, so when the array fails to resolve an angle
  it loses all B of that group's streams at once (Sweep 1, 15 deg / 12 dB: 4 -> 0). The sub-band
  axis does not rescue a collapsed beam.
- **Sub-band cost is a channelizer penalty, not a separation failure.** The windowed-sinc up/down
  conversion adds ~2-3 dB versus an ideal full-band stream (the 12 dB cells read ~6e-3 where a
  lone stream reads ~3e-3), plus residual leakage between adjacent sub-bands that share a band
  edge. A longer filter or a small roll-off margin trades compute / spectral efficiency for lower
  leakage; the 16B-tap Hann sinc used here already clears the clean channel to BER 0 and 12 dB to
  well under 1e-2.
- **Narrowband flatness.** The array steering is modelled frequency-flat across the whole
  allocation -- that is *why* one beam serves all B sub-bands of a group. Over a wide allocation
  beam-squint tilts the beam across sub-bands; keeping each sub-band inside the array's flat
  bandwidth (narrow fractional bandwidth, or per-sub-band steering) preserves the stack. Not
  modelled here.
- **Co-located-group assumption.** The B sub-bands of a group are taken to arrive from one angle
  (co-located transmitters, e.g. one multi-band node, or a cluster inside one beamwidth). Spreading
  a group's sub-bands across angles just relabels them as more angles -- still covered by the
  formula, with `#angles` counting the distinct directions.

## Hardware implied

- **N coherent receive chains at ~lambda/2** (the `spatial_mimo` requirement, unchanged) plus a
  band-limited channelizer in the RX front-end -- a windowed-sinc down-conversion per sub-band,
  cheap and FFT-free. No new agreement on the air: sub-band centres are sender-chosen and the
  receiver measures width and carrier blindly (`decode`).
- **N = 2 is the bench today.** The Pluto+ 2r2t runs the entire N = 2 column: **4** co-channel
  keyed streams (2 angles x 2 sub-bands) with no new hardware beyond the channelizer in software.
- **N = 4 is the relay.** Four phase-coherent receivers + a 4-element array deliver **8** streams
  (4 x 2). A 2-D arm layout removes the ULA half-space limit and improves angular coverage, so the
  real platform should do at least as well as this ULA sim.

## Caveats and follow-ons

- **Idealized front-end**, as in `spatial_mimo.md`: perfect element coherence, exact `lambda/2`,
  flat narrowband manifold, static channel. Element gain/phase mismatch, mutual coupling,
  beam-squint and multipath each erode the manifold conditioning and so the close-angle / low-SNR
  margin -- the square-array group (N = A = 2) is where that bite shows first. The blind MVDR uses
  the sample covariance, the realistic estimator, so that part is not idealized.
- **Equal power, distinct angles, disjoint sub-bands** -- the favourable case, chosen so any
  failure is the array's or the channelizer's, not a power imbalance.
- **Supersedes the "untested, natural next study" note** in `spatial_mimo.md` ("sub-band
  multiplexing should stack on top ... untested here"). It stacks, and the ceiling is
  `min(N, #angles) x #disjoint-sub-bands`.
- **Still receive-side separation of single-antenna transmitters.** TX-side spatial multiplexing
  (a transmitter with its own N antennas carrying N streams over one channel use) is the separate
  follow-on from `spatial_mimo.md`; it would multiply this ceiling again on the spatial factor.
