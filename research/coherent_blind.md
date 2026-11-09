# The coherent payload, read blind

`decode(iq, key)` now reads a `compose_qam` capture end to end. The receiver is given the capture and the
key. The width, the row offset, the mode count, the carrier, the reference row and the QAM order are
all measured. Before this, `decode` named a coherent capture and returned no bits.

## The read

1. **Geometry.** Discovery runs unchanged and measures `L`, `offset`+`phase`, `K` and a first carrier,
   all from the dark mode. A coherent capture closes on no differential hypothesis, so the beacon
   hypothesis can win a contest between two near-zero closures and take three modes. The coherent read
   gives those modes back (`K + BEACON_MODES`).
2. **Reference row.** `_reference_at` returns the row most coherent with the keyed reference:
   `argmax_t |A_t · conj(ref)| / (‖A_t‖ √K)`. That is ~1 on the reference and ~1/√K on a payload or noise
   row. A capture that starts rows before the burst still finds it (tested at 3 rows + 23 samples).
3. **Row-rate carrier.** Row `t` after the reference carries `exp(2πiνt)`, where ν is in cycles per row. This is
   the same convention as `_aperture`'s column ramp `exp(2πiν l/L)`, and `read_soft` removes the same
   phase once per differential with `exp(−2πiν)`. Discovery's ν comes from the dark mode, which sees only
   the column ramp. Measured, it is 1e-4 off at 40 dB and up to 3e-3 off at 20 dB. Over 600 rows, 1e-4 is
   0.38 rad, and a 1024-QAM corner tolerates about 0.03 rad. `_row_carrier` reads ν off the payload in
   three steps:
   - It forms `z_t = Σ_k Y_tk⁴`. The fourth power strips the data, because a square lattice is
     symmetric under a quarter turn.
   - It takes lag-k phase reads `arg Σ z_{t+k} conj z_t / 8πk` for k = 1, 2, 4, … up to T/2. Each read
     knows ν modulo 1/(4k) and is placed on the branch nearest the read before it. Lag 1 is placed
     nearest the discovery's ν.
   - It maximizes the full coherent sum `|Σ z_t e^{−8πiνt}|` inside the last half-branch, using bounded
     Brent.

   This uses closed forms and one bounded settle. There is no grid, no loop gain and no threshold. The
   aperture is then re-evaluated at the refined ν, which corrects the column half, and row `t` is
   de-rotated by `exp(−2πiνt)`. Measured |ν̂ − ν| ≤ 8.2e-6 cycles/row at every point in the table below.
4. **Payload test.** The payload is coherent when the fourth-power closure `|mean u⁴|` beats the
   differential closure, the same comparison `coherent_closure` makes. That closure stays in
   `Discovery.closure` because it is a mean of unit phasors and `closure_significance` can read it.
5. **Order M.** `_lattice_closure(Y, M) = max(0, −mean cos(2πx/step_M))` over both axes, with the
   step read off `_qam(M)`. Every M-QAM point is an odd multiple of half a step, so this reads exactly 1
   on the lattice. It is **signed**: a set offset by half a step reads −1. Steps of successive
   unit-power orders are in a ratio of about 2. So a 16-QAM point sits half a step off the 64-QAM
   lattice, and 64 does not close on 16-QAM: the signed read gives 0 where the unsigned one gives 0.94.
   Only the true order closes, so the read is the argmax over (4, 16, 64, 256, 1024), with ties going to
   the smallest. That is the differential rule, "the smallest hypothesis that closes", in the form this
   read allows. There is no level. The lattice closure is reported as `Discovery.lattice`, and
   `redundancy` comes from `entroptics.sequence.block_entropies` on the sliced symbols, as for the
   differential alphabet.

`read_qam` is still the informed reader, taking M and the carrier as given. It now removes the row-rate
half of the carrier itself, so callers no longer pre-process. That is safe for existing callers, which
all pass `carrier=0`. A coherent `decode` returns these fields:

- `payload="coherent"`, `M`, `K`, the refined `carrier`, and `closure` as the fourth-power closure.
- `lattice`, the closure on the M-QAM lattice.
- `reference`, the capture row the reference was found on.
- `row0=0` and `aligned=False`, because there is no keystream to align.

## Order-read separation

Measured at L=64, K=60, 600 rows, with ν = 0.0192 cycles/row and the read handed a carrier 5e-4 off.
The columns are the closures at each rung:

| sent | SNR | 4 | 16 | 64 | 256 | 1024 | read | BER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | clean | **1.000** | 0.000 | 0.257 | 0.771 | 0.000 | 4 | 0 |
| 4 | 10 dB | **0.383** | 0.000 | 0.000 | 0.000 | 0.000 | 4 | 4.4e-3 |
| 16 | clean | 0.157 | **1.000** | 0.000 | 0.000 | 0.038 | 16 | 0 |
| 16 | 15 dB | 0.120 | **0.257** | 0.000 | 0.003 | 0.000 | 16 | 2.1e-2 |
| 64 | clean | 0.139 | 0.038 | **1.000** | 0.000 | 0.000 | 64 | 0 |
| 64 | 21 dB | 0.134 | 0.025 | **0.252** | 0.000 | 0.000 | 64 | 1.8e-2 |
| 256 | clean | 0.142 | 0.032 | 0.009 | **0.988** | 0.000 | 256 | 0 |
| 256 | 27 dB | 0.135 | 0.031 | 0.000 | **0.265** | 0.000 | 256 | 1.5e-2 |
| 1024 | clean | 0.139 | 0.030 | 0.008 | 0.002 | **0.982** | 1024 | 0 |
| 1024 | 40 dB | 0.140 | 0.029 | 0.008 | 0.003 | **0.669** | 1024 | 2.2e-4 |
| 1024 | 33 dB | 0.137 | 0.032 | 0.012 | 0.006 | **0.260** | 1024 | 1.3e-2 |
| 1024 | 30 dB | **0.135** | 0.034 | 0.011 | 0.004 | 0.061 | **4 (wrong)** | 0.50 |

**Where it fails.** The 4-QAM rung holds about 0.14 on any denser QAM. This comes from the finite
constellation's own envelope, not from a lattice. When the true lattice closes below that, the read
returns 4 and the decode is BER 0.50. In the BER table below this happens only where the informed reader
is already at 4% BER or worse:

| M | lowest SNR where the order read fails | informed BER there |
|---|---|---|
| 64 | 18 dB | 5.8% |
| 256 | 24 dB | 5.0% |
| 1024 | 30 dB | 4.0% |

Every point where the informed BER is ≤ 2.2% reads M correctly.

## BER against SNR, uncoded

The setup:

- K=63, L=64, T=600 payload rows plus one reference row. A burst is 38,464 samples.
- The burst is preceded by 23 samples, so `offset`=23, and followed by 41.
- The carrier offset is 3e-4 cycles/sample (ν = 0.0192 cycles/row). The channel is AWGN at the stated
  SNR against the unit-RMS signal.
- There are 3 independent captures per point, so each point covers 3 × 600 × 63 × log2(M) bits. A
  BER of 0 means no errors in 136k bits (M=16) to 1.13M bits (M=1024).
- **Blind** is `decode(iq, key)` with the default candidate range: the whole receiver. It took 4–7 s
  per capture on one core.
- **Informed** is `read_qam(iq, L, K, key, M=M, offset=23, carrier=ν_true)`.

Every blind run named the payload coherent. M was read correctly at every point not marked ✗, where ✗
means it was read as 4.

| SNR dB | 16 blind | 16 informed | 64 blind | 64 informed | 256 blind | 256 informed | 1024 blind | 1024 informed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 2.29e-2 | 2.29e-2 | 0.50 ✗ | 1.06e-1 | 0.50 ✗ | 1.88e-1 | 0.50 ✗ | 2.51e-1 |
| 18 | 3.44e-3 | 3.45e-3 | 0.50 ✗ | 5.82e-2 | 0.50 ✗ | 1.39e-1 | 0.50 ✗ | 2.05e-1 |
| 21 | 1.85e-4 | 1.79e-4 | 2.18e-2 | 2.18e-2 | 0.50 ✗ | 8.89e-2 | 0.50 ✗ | 1.57e-1 |
| 24 | 0 | 0 | 3.85e-3 | 3.76e-3 | 0.50 ✗ | 4.98e-2 | 0.50 ✗ | 1.12e-1 |
| 27 | 0 | 0 | 2.26e-4 | 2.51e-4 | 2.00e-2 | 2.01e-2 | 0.50 ✗ | 7.54e-2 |
| 30 | 0 | 0 | 1.5e-6 | 2.9e-6 | 4.14e-3 | 4.18e-3 | 0.50 ✗ | 4.03e-2 |
| 33 | 0 | 0 | 0 | 0 | 2.41e-4 | 2.78e-4 | 1.62e-2 | 1.44e-2 |
| 36 | 0 | 0 | 0 | 0 | 5.5e-6 | 6.6e-6 | 3.91e-3 | 3.41e-3 |
| 40 | 0 | 0 | 0 | 0 | 0 | 0 | 9.35e-5 | 7.05e-5 |
| 45 | 0 | 0 | 0 | 0 | 0 | 0 | 7.9e-6 | 0 |

Where M is read correctly, blind tracks informed to within the run-to-run scatter at M ≤ 256. At
1024-QAM blind is 1.1–1.3× informed from 33 to 40 dB. At 45 dB blind made 9 errors in 1.13M bits and
informed made 0. That is the residual of the measured carrier: about 5e-6 cycles/row is up to 0.02 rad
at the far end of 600 rows.

## Spectral efficiency

Raw bits/s/Hz = log2(M) · K/L · T/(T+1), with K/L = 63/64 and one reference row in 601:

| M | raw b/s/Hz |
|---:|---:|
| 16 | 3.931 |
| 64 | 5.896 |
| 256 | 7.862 |
| 1024 | 9.827 |

The table below gives the best M per SNR through the blind path. "Delivered" is raw × (1 − h(BER)), the
binary-symmetric-channel bound used in the README. It is not a coded result.

| SNR dB | best M | raw | blind BER | delivered |
|---:|---:|---:|---:|---:|
| 15 | 16 | 3.93 | 2.29e-2 | 3.31 |
| 18 | 16 | 3.93 | 3.44e-3 | 3.80 |
| 21 | 64 | 5.90 | 2.18e-2 | 5.00 |
| 24 | 64 | 5.90 | 3.85e-3 | 5.68 |
| 27 | 256 | 7.86 | 2.00e-2 | 6.75 |
| 30 | 256 | 7.86 | 4.14e-3 | 7.56 |
| 33 | 1024 | 9.83 | 1.62e-2 | 8.65 |
| 36 | 1024 | 9.83 | 3.91e-3 | 9.46 |
| 40 | 1024 | 9.83 | 9.35e-5 | 9.81 |
| 45 | 1024 | 9.83 | 7.9e-6 | 9.83 |

## What does not work

- **Order read at low SNR.** The read fails where the true lattice closure falls below the 4-rung's
  structural ~0.14 (see the second table above). The result is a wrong M and BER 0.50, not a degraded
  decode. The transmitter picks M, so in practice this sits below the SNR where that M is usable
  uncoded.
- **Open broadcast (`seed=None`) of coherent 16-QAM or denser is still read as differential**, at BER
  0.50. With no keyed rotation, the differential closure of a QAM capture is 0.499, above the
  fourth-power 0.36, so the payload test picks differential. This predates this change. Two
  alternatives were measured and rejected:
  - Comparing the lattice closure fixes the open case but breaks keyed differential captures at low
    SNR. A scatter of unit phasors reads about 0.34 on the 4-QAM lattice, against a differential closure
    of 0.03 at 3 dB.
  - Normalizing the fourth-power closure by the alphabet's own |E u⁴| has the same problem.

  Keyed coherent captures of every order, and open coherent 4-QAM, are read correctly.
- **Sparse or constant sources.** A constant source is one point. It closes on whichever lattice
  holds that point. Measured, an all-zero 64-QAM source reads 64 at redundancy 1.00 and decodes
  bit-exact. A source on the inner sixteen points of 64-QAM still reads 64, because those points are not
  on the 16-QAM lattice, at redundancy 0.33. A source confined to a set that really is a coarser
  lattice cannot be told from that lattice. The same is true of the differential alphabet, and
  `redundancy` is what shows it.
- **Continuous scale.** The coherent read goes through `_scaled_rows` when `Discovery.scale ≠ 1`.
  `discover` never sets that (only `_discover_scaled` does), so `decode` reaches this path only through
  a scaled discovery. The read inherits that path's row-by-row floor, where the dark mode reads about
  −19 dB because neighbouring rows' tails are not modelled. Measured on noiseless captures at
  ρ = 0.5, 0.833 and 0.514:
  - 4-QAM is bit-exact.
  - 16-QAM reads at BER 4.9e-3 to 8.3e-3.
  - 64-QAM fails, with M read as 4 and BER 0.50.

  Denser coherent orders at a non-unit scale need the joint read across row boundaries that the scale
  section names.
- **Gain from one row.** Per-mode gains still come from the reference row alone, so its noise enters
  every symbol of that mode. This applies equally to the informed path, and it is part of why the
  1024-QAM row needs about 40 dB.
