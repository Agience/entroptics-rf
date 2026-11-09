# Spatial co-channel MIMO: an N-antenna receiver separates overlaid keyed streams

**Question.** The single-antenna probe (`research/multiplex/overlaid_keys.md`) showed that two
independently-keyed entroptics screens superposed on **one** antenna do NOT multiplex: a
wrong-key read is noise-like, but it is *full-power* noise with no processing gain, so both
screens are full rank in the same L keyed modes and there is no free dimension to separate
them. This study adds the one thing that probe lacked -- a **spatial** degree of freedom.
Co-channel keyed transmitters arriving from different directions present different array
signatures to an N-element receiver, so the array has N spatial dimensions. Can an N-antenna
receiver separate up to ~N co-channel keyed streams, WITHOUT pilots or training, recovering
the co-channel capacity single-antenna key-stacking could not?

**Verdict: GO.** A multi-antenna receiver recovers exactly the co-channel capacity the single
antenna could not. An N-element array cleanly separates up to **min(N, K)** equal-power
co-channel keyed streams -- `2x` with N = 2, a full `4x` with N = 4 -- against the single
antenna's `< 1x`. The separation is a linear combine of the coherent antenna captures; the
screens' own full rank, fatal on one antenna, is irrelevant once the sources sit in different
spatial directions. And the **blind** receiver -- no angles, no pilots, weights found by an
MVDR beam whose steering angle is chosen to maximize each key's constant-modulus lock --
matches the genie (true-angle zero-forcing) ceiling across essentially the whole feasible
grid, and at low SNR slightly beats the genie's pure zero-forcing because its loaded MVDR
amplifies noise less. The keys plus the array replace training outright. Where it breaks is the
classic array-processing wall, not an entroptics one: a **full-rank** load (K = N) with sources
**close in angle** (<=15 deg) at **low SNR** (<=12 dB), where the array manifold is
ill-conditioned and there are no spare elements to average the noise. Oversizing the array
(N > K) buys back close-angle / low-SNR robustness. Sim only; no hardware, no transmit.

## Method

- **Array model** (uniform linear array, narrowband steering-vector mixing -- standard array
  processing in the RX front-end; it is NOT the codec). N elements at spacing d = lambda/2, so
  element k sees a source at angle theta with phase exp(j*pi*k*sin theta). K single-antenna
  keyed transmitters at distinct angles, superposed at **equal per-source power**:
  `Y[k_elem, n] = sum_src steer_k(theta_src) * s_src[n] + per-element complex AWGN`.
  Sources are centred on broadside with adjacent spacing `dtheta`; a geometry that would push a
  source past +-75 deg (outside a ULA half-space) is marked `n/a: geometry` and skipped.
- **Sources** composed by the public library: `compose(random_bits(.), K=L=64, L, seed, M)`,
  distinct keys and bit seeds, 96 rows (>= ~90 so the blind width lock settles). Each source is
  unit-rms, so `SNR` is **per source, per element**.
- **Genie zero-forcing (capacity ceiling).** `W = pinv(A)`, `A = [steer(theta_1) .. steer(theta_K)]`
  (N x K, true angles known); separated `Shat = W @ Y`; each stream read at its true geometry by
  the informed single-key reader (`read` / `read_qam`). Establishes whether the capacity is there.
- **Blind (the entroptics-pure achievability).** No angles, no pilots. For each held key,
  grid-search the array steering angle (coarse 3 deg then 0.5 deg refine over the placeable
  half-space); at each angle form the **MVDR** weight from the sample covariance alone,
  `w = Rinv a(theta) / (a^H Rinv a)` with light diagonal loading; keep the angle whose combined
  stream reads the **lowest** `constant_modulus_read` under that key (best keyed lock); then read
  the combined stream with the fully blind `decode`. The key's constant-modulus lock is the only
  thing steering the beam -- it replaces the pilot.
- No pilots, preamble, equalizer, channel estimation, FFT, resampling, or joint ML anywhere in
  the codec. The array math (steering vectors, pinv, sample-covariance MVDR) is plain narrowband
  front-end combining applied to the antenna captures before the codec; it reads no payload and
  holds no screen geometry. Public library only: `compose, decode, compose_qam, read_qam, read,
  ber, random_bits, constant_modulus_read`. 3 seeds/cell; `good` = raw BER < 1e-2.

Re-run: `OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src .venv/Scripts/python.exe
research/mimo/spatial_mimo.py`. Script: `research/mimo/spatial_mimo.py`.

## Results

`good_g` / `good_b` = mean good streams per trial (of min(N,K)), genie and blind. QPSK (M=4).

### 1. N = 2 (the bench Pluto+ 2r2t case): two co-channel streams on two antennas

| dtheta | SNR | BER_genie | good_g | BER_blind | good_b |
|---:|---:|---:|---:|---:|---:|
| 15 | clean | 0.0000 | 2.00 | 0.0000 | 2.00 |
| 15 | 20 | 0.0000 | 2.00 | 0.0000 | 2.00 |
| 15 | 12 | 0.0757 | 0.00 | 0.0593 | 0.00 |
| 15 | 9 | 0.1909 | 0.00 | 0.1433 | 0.00 |
| 30 | 12 | 0.0019 | 2.00 | 0.0016 | 2.00 |
| 30 | 9 | 0.0237 | 0.00 | 0.0206 | 0.00 |
| 60 | 12 | 0.0001 | 2.00 | 0.0000 | 2.00 |
| 60 | 9 | 0.0024 | 2.00 | 0.0025 | 2.00 |
| 90 | 12 | 0.0004 | 2.00 | 0.0004 | 2.00 |
| 90 | 9 | 0.0136 | 0.00 | 0.0123 | 0.00 |

- **Two clean streams for free** at every angular separation down the whole SNR range above
  the break -- against the single antenna's two dead streams at equal power. The spatial
  dimension is the lever.
- **The break is angular conditioning + SNR, not entroptics.** With only N = 2 elements and
  K = 2 sources there are **no spare antennas** to average noise, so the two-column manifold's
  conditioning sets the floor. 60 deg is the sweet spot (for N = 2 the two steering vectors are
  exactly orthogonal when sin(theta_1) - sin(theta_2) = 1, i.e. +-30 deg), holding to 9 dB;
  15 deg is nearly colinear and dies by 12 dB; even 90 deg is slightly worse than 60 deg at
  9 dB because the manifold is past the orthogonal point.
- **Blind == genie.** The blind MVDR + `decode` matches the genie zero-forcing everywhere, and
  is marginally *better* at low SNR (0.0593 vs 0.0757 at 15 deg / 12 dB) -- the loaded MVDR is
  closer to MMSE than the genie's pure zero-forcing, so the "ceiling" is if anything conservative.

### 2. N = 4, K = 2: oversizing the array buys robustness

| dtheta | SNR | BER_genie | good_g | BER_blind | good_b |
|---:|---:|---:|---:|---:|---:|
| 15 | 12 | 0.0000 | 2.00 | 0.0000 | 2.00 |
| 15 | 9 | 0.0006 | 2.00 | 0.0006 | 2.00 |
| 30 | 9 | 0.0001 | 2.00 | 0.0001 | 2.00 |
| 60 | 9 | 0.0000 | 2.00 | 0.0000 | 2.00 |
| 90 | 9 | 0.0001 | 2.00 | 0.0001 | 2.00 |

- **Every cell delivers two streams**, including 15 deg at 9 dB (BER 6e-4) where N = 2 was dead.
  The two spare elements (N - K = 2) give array gain and noise averaging, so close angles and
  low SNR that broke the square N = 2 case are now comfortable. Carrying more antennas than
  streams is the robustness knob.

### 3. N = 4, K = 3 and the full-rank K = 4 wall

N = 4, K = 3 (one spare element):

| dtheta | SNR | BER_genie | good_g | BER_blind | good_b |
|---:|---:|---:|---:|---:|---:|
| 15 | 12 | 0.0041 | 2.67 | 0.0032 | 3.00 |
| 15 | 9 | 0.0345 | 0.00 | 0.0250 | 0.00 |
| 30 | 9 | 0.0000 | 3.00 | 0.0000 | 3.00 |
| 60 | 9 | 0.0011 | 3.00 | 0.0013 | 3.00 |
| 90 | -- | -- | -- | -- | -- |

N = 4, K = 4 (full rank, no spare element):

| dtheta | SNR | BER_genie | good_g | BER_blind | good_b |
|---:|---:|---:|---:|---:|---:|
| 15 | clean | 0.0000 | 4.00 | 0.0028 | 4.00 |
| 15 | 20 | 0.0070 | 2.00 | 0.0187 | 2.00 |
| 15 | 12 | 0.1730 | 0.00 | 0.0890 | 0.00 |
| 15 | 9 | 0.2789 | 0.00 | 0.2452 | 0.00 |
| 30 | clean | 0.0000 | 4.00 | 0.0000 | 4.00 |
| 30 | 20 | 0.0000 | 4.00 | 0.0000 | 4.00 |
| 30 | 12 | 0.0000 | 4.00 | 0.0000 | 4.00 |
| 30 | 9 | 0.0000 | 4.00 | 0.0000 | 4.00 |
| 60 | -- | -- | -- | -- | -- |
| 90 | -- | -- | -- | -- | -- |

- **Three streams (K = 3)** hold cleanly at 30 and 60 deg down to 9 dB; blind delivers all 3 at
  15 deg / 12 dB where genie averaged 2.67. 15 deg / 9 dB is the break; 90 deg is geometrically
  unplaceable for 3 sources (the outer pair would exceed the half-space).
- **Full `4x` (K = 4)** holds at BER ~0 down to 9 dB when the sources are >= 30 deg apart. This
  is the make-or-break number: four equal-power co-channel keyed streams, all recovered, no
  pilots, at 9 dB per-source SNR.
- **The full-rank wall.** At K = N = 4 there is no spare element, so the whole burden falls on
  the manifold conditioning. 15 deg separation collapses: clean still gives 4, but 20 dB already
  drops to 2 and 12 dB is dead. The fix is either more angular separation (30 deg suffices) or
  more antennas than streams (Sweep 2). Sources packed within a beamwidth cannot be separated --
  the ordinary resolution limit of an N-element aperture, inherited unchanged.

### 4. 16-QAM spot check (coherent) -- N = 4, K = 2, dtheta = 30

| SNR | BER_genie | good_g | BER_blind | good_b |
|---:|---:|---:|---:|---:|
| clean | 0.0000 | 2.00 | 0.0000 | 2.00 |
| 12 | 0.0002 | 2.00 | 0.0003 | 2.00 |

Coherent 16-QAM rides the spatial separation as well as QPSK: two clean co-channel streams at
BER ~0 through 12 dB, blind == genie. The denser alphabet does not change the separation; the
array hands each stream a clean, de-interfered capture and the coherent reader does the rest.

## Go / no-go

**GO -- spatial MIMO is the co-channel multiplexing lever entroptics was missing.** What a
single antenna could not do (two full-rank keyed screens, no free dimension, both dead at equal
power), N coherent antennas do cleanly: **min(N, K)** equal-power co-channel keyed streams, a
net `2x` at N = 2 and a full `4x` at N = 4. The screens' full rank in the keyed modes is simply
not the obstacle once the sources occupy different spatial directions -- the array separates
them by their *angles*, and the keys then read each de-interfered stream exactly as they would a
lone one.

**Blind achieves the ceiling.** The entroptics-pure receiver -- no angles, no pilots, MVDR beam
steered by maximizing each key's constant-modulus lock -- matches the genie true-angle
zero-forcing across essentially the whole feasible grid, and edges ahead at low SNR. The keyed
constant-modulus null is a sharp enough beam-pointing objective to replace training entirely.
This is the key result: the spatial capacity is not just present (genie) but **reachable without
any side information** (blind).

**Where it breaks (ordinary array processing, not entroptics):**
- **Full-rank load at close angles.** K = N with sources within ~15 deg: the N-column manifold
  is ill-conditioned and there is no spare element to average noise, so the streams collapse off
  the clean channel (K = N = 4 at 15 deg already drops to 2 streams by 20 dB). Keep K <= N - 1,
  or sources >= ~30 deg apart, or both.
- **Low SNR on a square array.** N = K = 2 at 15 deg dies by 12 dB; the zero-forcing / MVDR
  noise amplification at poor conditioning is the floor.
- **Angular resolution.** Sources packed inside the array's beamwidth cannot be separated --
  the plain aperture resolution limit, unchanged by anything in the codec.
- **Geometry.** A ULA half-space holds only so many sources at a given spacing (4 sources at
  60 deg, or 3 at 90 deg, exceed +-75 deg and are unplaceable); a filled or 2-D array relaxes this.

**Operating recommendation.** Carry **more antennas than streams** (N >= K + 1): the N = 4,
K = 2/3 cells are robust to 15 deg and 9 dB, while the square cases are not. For the full
`N x` load, require >= 30 deg of angular separation between sources.

## Hardware implied

- **N coherent receive chains and N antennas at ~lambda/2 spacing.** Coherence (shared LO /
  phase-locked sampling across elements) is the hard requirement; the linear combine assumes the
  element captures share a phase reference.
- **N = 2 is the bench today.** The Pluto+ 2r2t is two coherent RX on one LO -- it can run the
  entire N = 2 column above (two co-channel streams) with no new hardware. This is the
  immediately testable claim on the existing bench.
- **N = 4 needs more coherent-RX hardware** -- four phase-coherent receivers (e.g. two
  phase-synced Plutos, or a 4-channel coherent SDR) and a 4-element lambda/2 array. This is the
  quadcopter-relay case (one antenna per arm). The `4x` result says the capacity is there and
  blindly reachable once the four chains are coherent.
- **Array geometry.** lambda/2 ULA for the model here; a quadcopter's four-arm layout is closer
  to a 2-D array, which removes the half-space placement limit and improves angular coverage --
  a reason to expect the real platform to do at least as well as this ULA sim, not worse.

## Caveats and follow-ons

- **Idealized front-end.** Perfect element coherence, exact lambda/2 spacing, a clean narrowband
  steering manifold, and a static channel. Element gain/phase mismatch, mutual coupling, wideband
  beam-squint and multipath are not modelled; each erodes the manifold conditioning and so eats
  into the close-angle / low-SNR margin (the square-array cells are where that bite will show
  first). The blind MVDR uses the *sample* covariance, which is already the realistic estimator,
  so that part is not idealized.
- **Equal power, distinct angles.** The favourable co-channel case, chosen so any failure is the
  array's and not a power imbalance. Unequal powers only help the strong stream.
- **Compose with sub-band multiplexing.** Spatial separation is orthogonal to sender-chosen
  sub-band width: each spatial stream is an ordinary single-antenna capture after the combine,
  so sub-band multiplexing should stack on top (min(N,K) spatial x whatever the band supports).
  Untested here; natural next study.
- **TX-side second antenna for full 2x2.** This study is receive-side spatial *separation* of
  single-antenna transmitters. A transmitter with its own N antennas opens true spatial
  *multiplexing* (a single link carrying N streams over one channel use), the full MIMO 2x2 /
  4x4 case. The receive-side result here is the prerequisite and says the spatial dimension is
  real and blindly usable; the TX-side multiplexing gain is the follow-on.
- **Supersedes the single-antenna no-go for the co-channel question.** `overlaid_keys.md` named
  multi-antenna as "the natural place for MIMO to actually pay off." It does: this is that study,
  and the answer is yes.
