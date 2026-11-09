# Relative bearing (AOA) from the keyed array — relative positioning between cooperating nodes

**Question.** GPS may be unavailable or degraded, so the mesh can't rely on absolute position. Can a node
recover the bearing of a cooperating neighbour (one that shares its key) from that neighbour's own keyed
transmission — no pilots, no GPS — so the "I'm here" is the *signal itself*? This is relative positioning
between nodes of one mesh, not the location of an unknown transmitter. (`relative_bearing.py`.)

**Answer: yes, and it's already a byproduct of the MIMO receiver.** `blind_separate` (spatial_mimo)
grid-searches the steering angle and keeps the one whose keyed constant-modulus lock is best; that angle
is the neighbour's bearing in the array frame. Measured:

- **Calibrated array — accurate to < 1° down to 0 dB SNR.** AOA RMS error over angles −45..+45°:
  **N=2** 0.0°/0.0°/0.6° at 20/10/0 dB; **N=4** 0.0°/0.0°/0.0° (worst 0.2°). So the bearing is
  essentially exact at usable SNR, even with a 2-element array.
- **Calibration caveat (the one from `mimo_calibration`): data *separation* needs no array cal, but
  *angle* does.** A static per-element phase error re-labels the angle. N=4, 20 dB: per-element phase
  RMS 5° → **0.7°** AOA error; 10° → **1.4°**; 20° → **breaks (~48° RMS, −25° bias).** So a *reasonably*
  calibrated array (per-element phase ≲ 10°) gives a usable absolute bearing; beyond that it degrades.
  Differential/relative use (bearing *between* neighbours) is more forgiving of a common offset.

**So.** The relative-positioning primitive is real and cheap: each node measures its cooperating
neighbours' bearings to ~1° from their keyed transmissions, reusing the same λ/2 array that does MIMO.
Combine bearings across neighbours (+ the array's own orientation, from
the IMU) → the fleet's **relative geometry**, no GPS. Limits: needs a one-time array calibration for
absolute bearing; gives internal geometry, not earth location (absolute position still needs GPS/INS/visual);
one-way *range* needs a separate ranging scheme (two-way ranging between the nodes). Foundations in `spatial_mimo` (blind
keyed MVDR) and `mimo_calibration` (cal immunity for separation).

## Hardware validation (cabled coherent 2-RX array)

The AOA mechanism above is grounded on real hardware. Rig: a
Pluto+ 2r2t with its two RX on one LO/clock (so RX1/RX2 are **phase-coherent** — the point of a 2r2t
part for an array); one mirrored TX source → 30 dB pad → splitter → RX1 (cable P) and RX2 (cable Q),
the emulated angle set by the cable-length difference ΔL. (The Pluto mirrors one stream onto both TX
jacks, so this is a **1-source AOA array, not 2×2 MIMO** — true spatial-mux needs a second TX radio.)

**Estimator** (entroptics-pure measurement, N=2): discover the geometry on RX1, project **both** RX onto
the same keyed aperture (the keyed read denoises and cancels the keying identically on both channels),
inter-element phase Δφ = angle(Σ A2·conj(A1)); θ = arcsin((Δφ − Δφ_cal)/π). Self-tested on synthetic
coherent data: recovered a known 37° inter-element phase **exactly** at 20/10/5 dB.

**Measured.** All captures came back with RX1↔RX2 **coherence 0.995–0.996** — a genuinely phase-coherent
array. A **swap calibration** (capture, then swap the two RX cables → ΔL sign flips) separates the
geometry from the fixed RX-channel offset:

- **ΔL ≈ 5 cm pair:** the inter-element phase **flipped cleanly on the cable swap** (−115.5° → +130.9°, a
  **246° swing**). A fixed channel artifact would *not* flip — so this is unambiguously the geometric
  array phase. Decomposed (physical small-offset solution): **geometric 56.8°, channel offset +7.7° →
  bearing ≈ 18°.**
- **Channel offset is small and independently confirmed:** the second swap pair gave offset +3.7°, so
  both pairs agree the RX chains are matched to within ~4–8° — a near-calibrated array, matching the
  sim's finding that angle estimation needs calibration but a small one suffices.

**Limits (what the available cables allowed):**
- **One clean baseline.** The ΔL ≈ 10 cm pair landed at the **λ_cable/2 endfire ambiguity** (≈11 cm): its
  swap swing collapsed to 12.5° (vs 246°), giving a self-inconsistent 2° "bearing" (smaller than the
  5 cm pair's 18°, impossible for a larger ΔL). So it is not a usable second point, and **phase-vs-angle
  linearity is not confirmed on hardware** — that needs intermediate lengths (ΔL ≈ 2–3 and 6–8 cm) which
  were not available.
- Narrowband (2.4 MHz at 915 MHz): ΔL is resolved only via **carrier phase** (wraps every λ_cable ≈
  21 cm), not group delay — an intrinsic property of the band, not a setup flaw.

**=> The AOA mechanism is validated on real hardware** — a phase-coherent keyed 2-element array recovers a
geometric bearing, blind, with a small (~8°) calibration offset. The quantitative multi-angle accuracy
(the <1° sim figure) is **not yet swept** on hardware (endfire-limited by the available cables).
