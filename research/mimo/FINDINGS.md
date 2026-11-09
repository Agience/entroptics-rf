# entroptics MIMO / multiplexing — findings

Consolidated result of four sim studies. All sim, entroptics-pure (no pilots/guards/equalizers/FFT/resampling in the codec), library-only
(`compose`/`decode`/`read`/`constant_modulus_read`/…). Source studies, in order:
`research/multiplex/overlaid_keys.md`, `research/mimo/spatial_mimo.md`, `.../mimo_calibration.md`,
`.../spatial_subband.md` (each has a re-runnable `.py`).

## One line

Multiple co-channel keyed streams are separated **blind** (no pilots) by an antenna array steered on
each key's constant-modulus lock, and this **composes with disjoint sub-bands**, giving a fleet capacity

> **simultaneous co-channel streams = min(N antennas, #angles) × #disjoint sub-bands**

with the blind receiver reaching the genie (perfect-knowledge) ceiling.

## The arc (why each step)

1. **Co-channel key-stacking on ONE antenna — NO-GO** (`overlaid_keys.md`). Two keyed screens
   superposed on a single antenna deliver *fewer* good bits than one at the same total power. The
   apertures are quasi-orthogonal, but both are full-rank over all L modes → **no free dimension**: a
   wrong-key screen is exactly equal-power AWGN, so per-stream SINR is pinned to the power ratio.
   Keys alone are not a capacity lever. (Bonus: a foreign-key screen reads as structureless noise →
   coexistence.)

2. **Spatial (multi-antenna) separation — GO** (`spatial_mimo.md`). An N-element array adds the free
   dimension the single antenna lacked. It cleanly separates **min(N, K)** equal-power co-channel
   keyed streams (net 2× at N=2, 4× at N=4). The **blind** receiver — an MVDR beam steered by
   **maximizing each key's constant-modulus lock** (no angles, no pilots), then blind decode — **matches
   the genie true-angle zero-forcing ceiling**, and is marginally better at low SNR (loaded-MVDR ≈ MMSE).
   The keyed CM null is a sharp enough beam-pointing objective to replace training entirely.
   Rules: keep #streams ≤ N−1 and sources ≥ ~30° apart (closer ill-conditions at low SNR — ordinary
   array physics, not entroptics).

3. **Calibration — the DIY array needs NONE of the hard part** (`mimo_calibration.md`). Because the
   beam is steered by the KEY, not by absolute angle, it is **immune to static per-element
   miscalibration** (N=2: unknown fixed phase to 90° RMS → BER 0, = genie; a direction-finder like a
   KrakenSDR needs calibration because DF needs absolute angle — we only need to *separate*). So the
   per-boot cal rig (noise source + RF switch) is **not required**. The one hard requirement is a
   **shared clock** for within-capture coherence (differential drift < ~5–10°/capture for margin;
   common-mode drift is harmless). N=4 adds a **one-time** (not per-boot) element trim; N=2 needs zero.

4. **Composition — spatial × sub-band STACK** (`spatial_subband.md`). The array separates by *angle*
   (grouping an angle's co-located sub-bands); a disjoint-sub-band read then splits within the group
   (one beam serves the whole group because narrowband steering is frequency-flat). Validated
   multiplicative: N=2×2 bands → 4; N=4, 4 angles ×2 bands → 8. Blind = genie. A failed beam loses that
   angle's whole sub-band group; sub-banding costs ~2–3 dB channelizer overhead.

## Fleet capacity

| Receiver | N (coherent RX) | sub-bands | co-channel streams |
|---|---|---|---|
| Pluto+ 2r2t | 2 | 2 | **4** |
| 4-arm quad relay (DIY RTL array) | 4 | 2 | **8** |
| " | 4 | 3 | 12 |

The capacity knobs are **(antennas × sub-bands)**; the other ceiling is the receiver's **decode
compute**. Spatial separation is RX-side of
single-antenna transmitters (the drones stay single-antenna); TX-side antennas would open true
2×2/4×4 spatial multiplexing — a later study.

## Hardware (cheap, DIY, off-the-shelf)

- **N=2: free** on a Pluto+ 2r2t (2 coherent RX), **zero calibration**.
- **N=4 (quad relay): DIY coherent RTL-SDR array ~$80–130** — 4× RTL-SDR + one 28.8 MHz osc fanned out
  to all dongles' clock inputs (the whole coherence rig). **No noise source / RF switch** (calibration
  not needed, step 3). One-time element trim. Array = the quad's 4 arms at ~λ/2 (16 cm @915) adjacent,
  i.e. a stock 5-inch frame; the 2-D square resolves az+el (better than the sim's ULA).

## Novelty

Classical MIMO separates spatial streams with **pilots / training** to estimate the channel. Here the
separation is **blind** — the per-stream keyed **constant-modulus lock** is the beam-pointing objective,
so no pilots, no training, nothing pre-shared but the key. That (a) keeps the parameter-free
core intact, (b) removes the per-boot array calibration a coherent array normally needs
(the beam doesn't need absolute angle), and (c) composes cleanly with keyed sub-band multiplexing. The
same keyed structure that makes a single transmission noise-like to a non-holder is what separates many
of them spatially for a holder.

## Caveats & what the bench must confirm

Idealized coherent front-end throughout (exact λ/2 ULA, narrowband-flat, no mutual coupling, no
multipath; drift modeled as a controlled random walk — a real clock's phase-noise spectrum differs).
Amplitude mismatch is only partially absorbed (worse N≥4 under noise). So the **on-metal check** remains:
the N=2 case runs on a Pluto+ 2r2t with zero new hardware — two keyed transmitters,
2 coherent RX, confirm blind separation on real coherent samples. That validates the whole chain before
the 4-arm array is built.
