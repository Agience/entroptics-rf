# The per-row cyclic extension

> **Neither end uses one.** The sender writes rows back to back, L samples
> apart, and the receiver reads rows L samples apart only: it searches no extension. A foreign
> sender that wraps its rows is declined, not read and not misread (measured below). The reason is
> detectability; the floor the extension bought a receiver at another rate is recovered in the
> receiver by the joint read (`research/joint_read.md`), which puts nothing on the air.

An extension wraps row `r`, which is `L` samples long, as `[r[-P:], r, r[:P]]`, so rows sit `L + 2P`
apart on the wire. It was built and measured as a sender option with `P` measured by the receiver
(a third discrete coordinate searched with the width, `P = 0..4`). The measurements that decided
against it are kept here.

## Detectability: why the sender does not write it

A blind observer knows nothing about the waveform. It scans every lag from 8 to 600 for
`|sum y[n] y*[n+lag]|`, and every period from 16 to 600 for a line in the power envelope `|y|^2`.
Each statistic is scored as a z against noise-only captures of the same length. The setup is
200 000 samples (83 ms at 2.4 Msps), at an equal rate, in AWGN. The noise-only maximum z is 3.0–3.6.

| on the air | lag-correlation detector | power-envelope detector |
|---|---|---|
| plain screen, P=0 (L=64 and 128) | no line at any SNR from -20 to 0 dB (max z 4–5.7, at random lags) | no line (max z 4–6, at random periods) |
| **P=2, L=128** | **lag 128: z 8.4 at -5 dB, 15.7 at 0 dB** | no line |
| **tapered row edges** (8 samples, the "option 3" stand-in) | not measured | **row rate: z 8.4 at -10 dB, 55 at 0 dB** |

The lag-correlation z grows as the square root of the capture length. A 10 s capture therefore
finds P=2 at about -15 dB. The line sits at the row length, so it also tells the observer L.
Tapering is worse still. The plain screen has no feature either detector finds, and that is what
the wire keeps. (The plain screen does carry one key-free trace these two detectors do not look
for, the equal energy of every differential row: `research/row_energy_leak.md`.)

## What it bought

Each row carries different data, so the sample stream jumps at every row boundary. A receiver at
the sender's rate samples exactly the transmitted instants, and the jump costs it nothing. A
receiver at another rate samples the band-limited signal between them, where the neighbouring
rows' converter tails (falling as `1/d`) reach into the row it reads. The extension moves the
nearest jump `P` samples away from the row's core.

Lab prototype, noiseless, true geometry, row
read, L = 128, K = 124. Each cell is the dark-mode floor at ρ = 0.514 / 0.833.

| P | overhead | dark floor (dB) |
|---:|---:|---|
| 0 | 0.0% | −22.8 / −22.7 |
| 1 | 1.5% | −25.0 / −24.5 |
| **2** | **3.0%** | **−24.9 / −24.8** |
| 4 | 5.9% | −24.5 / −24.5 |
| 8 | 11.1% | −24.4 / −24.4 |
| 16 | 20.0% | −24.3 / −24.3 |

About 2 dB, all of it taken by `P = 1–2`; the tails reach past any extension. Blind through
`_discover_scaled` at L = 128 the 2 dB survived intact: DQPSK went from 2.6e-5 / 1.05e-4 BER to 0 at
ρ = 0.514 / 0.833, coherent 16-QAM from ~2e-3 to 0.8–1.8e-4, and 64-QAM from 5–7% to about 1%,
all noiseless, all with the row read. At equal rates it bought nothing and cost `2P/(L+2P)` of the
rate. The joint read (`research/joint_read.md`) takes the same floor from −22.8 to −66..−70 dB with
no extension, so the extension has nothing left to buy.

## Why the receiver searches none

The sender writes none, and searching for one cost the receiver in two ways:

- **Chance.** Every extension hypothesis is another set of folds that can read dark by chance. In
  the blind-width matrix (`research/blind_width_scaled.md`) every false lead was an extension
  `P > 0`, and the one cell that missed (L = 128, equal rates, 10 dB) was lost to an extension-4
  chance fold. Rerun with rows L apart only, that cell locks, given the width and blind, and every
  other rerun cell still locks with the same true-fold read.
- **Cost.** Measured single-threaded, the same captures before (searching `P = 0..4`) and after,
  interleaved on one machine (medians):

| read | searching P = 0..4 | rows L apart only |
|---|---:|---:|
| `decode`, integer, L=64, clean (3 captures x 2) | 1.55 s (dark score 1.22 s) | **1.14 s** (0.89 s) |
| `decode`, integer, L=64, 6 dB (3 x 2) | 5.00 s (4.09 s) | **3.14 s** (2.60 s) |
| `_discover_scaled`, L=64, ρ 0.514, 20 dB, width given (3 x 2) | 21.2 s (period read 0.17 s) | 21.1 s (0.11 s) |
| `_discover_scaled`, same, blind, widths 2..101 (3 x 1) | 54.8 s (period read 24.0 s) | **49.7 s** (15.7 s) |

On the integer path every dark score regrouped one sliding product five times over a longer span,
and at 6 dB the extra hypotheses sometimes kept the widening going one doubling longer (3836 or
3288 dark scores where the plain search needs 3288 or 2192). The answers and the BER are identical
in every capture. On the scaled path the folds are 5.3–5.6× fewer (the sum of `(L + 2P)²` over
`P = 0..4` against `L²`); the per-width maps, which every fold shares, are unchanged, so the period
read is about 1.5–2× faster. The settles, identical work in both, dominate the total and varied
by ±30% with the machine's load (another job shared it), so the blind total moves by the period
read's saving and no more. With the width given the period read was already 0.1–0.2 s, so the
total does not move.

Removing it also removed the plumbing: `Discovery.extension`, `MAX_EXTENSION`, the `extension=`
argument of every read, and the `(L + 2P)/L` per-row carrier factor. A row is `L` samples of the
capture everywhere in the codec.

## A foreign sender that uses one

A capture of `[r[-P:], r, r[:P]]` rows, read by a receiver that cuts rows `L` apart, has every row
but the first cut across a boundary, so no geometry is dark. Measured
(L = 64, K = 60, a 37-sample cut,
3e-4 cycles/sample carrier):

| capture | score | margin | closure / null | payload |
|---|---:|---:|---:|---|
| plain screen, clean | 1.2e-15 | 4.7e13 | 134 | bit-exact |
| plain screen, 12 dB | 0.054 | 3.5 | 82 | BER 1.8e-3 |
| P = 1, 2, 4; DQPSK and 8-PSK; clean and 12 dB (12 captures) | 0.70–0.91 | 1.03–1.17 | 1.3–4.3 | ~0.5 (noise) |
| P = 2, coherent 64-QAM | 0.83 | 1.21 | 2.8 | lattice 0.0, ~0.5 |
| pure noise (3 captures) | 0.73–0.74 | 1.02–1.11 | 1.9–3.4 | — |
| P = 2 at ρ 0.514, L = 128, `_discover_scaled` width given | 0.79 | 1.08 | 4.1 | — |

The foreign capture reads like noise on every piece of evidence a reader acts on, so it is
declined, not taken as a confident wrong lock. The streaming receiver, which tests the closure
against the null before it locks, does not acquire it. What is lost is the foreign sender's
payload; nothing reads it. `tests/test_extension.py` pins all of this. A blind search over a
foreign capture is slower than over a real one (7–14 s against 1.5–2 s at equal rates, 234 s
against 69 s scaled), because nothing converges and the widening runs to the capture's end.
