# The row period, read per width: blind width at a non-unit rate ratio

At equal rates, `discover` reads the width L blind: it scans every width for the dark-mode cliff. At
another rate the receiver sees a row of L transmitted samples as a period of P = L / rho of its own
samples, with rho real. Without a period read, `_discover_scaled` must be given the candidate widths,
and for each one sweep a grid of scales fine enough that eight rows never drift a quarter sample. That grid has
about 6.4·L points across a 5% bracket, and an 8-row scan at every point. With the width given, the
grid alone cost 10-280 s at L=64 and an estimated 4 min to 5 hours at L=256 (measured below). A blind search over every
width would be out of reach.

Scope: these measurements were taken on a screen with dark modes (K = L − 4) and equal-energy
rows. The codec lights every mode and keys a per-row power (`rank_leak.md`, `row_energy_leak.md`),
which closes the row-energy trace described below.

The question is whether P could be read directly, instead of sweeping (L, rho) pairs.

## What can be read without the width

**Nothing keyed.** The aperture is a separate keyed draw for each width. The keystream is shared,
but it is reshaped and QR-factorized per L, so the dark mode of width L' says nothing about rows of
width L. Measured (`test_another_width_is_not_dark_at_the_true_period`): lay the apertures of
L ± 1 on the true rows, at the same received period and so at scale (L ± 1) / P. They read above 0.5,
like an isotropic capture. The true width reads under 0.05. The dark mode tests a period only
together with the width that keyed it.

**Almost nothing without the key.** That is the low-detectability property: the energy is visible to a radiometer, but the structure is not classifiable and the content not readable. One key-free trace exists. Every
row of the differential payload has exactly the same energy on the wire: the screen at the true
offset has a row-energy spread of 8e-9, against 2-8% at any other offset. A key-free observer can
therefore score every width by the spread of its row energies. Measured at equal rates on 200k
samples in AWGN, for widths 16-600:

| sender | -10 dB | -5 dB | 0 dB | +5 dB | +10 dB |
|---|---|---|---|---|---|
| L=64: rank of the true width | 396 | 197 | 124 | 104 | 13 |
| L=128: rank of the true width | 150 | 497 | 355 | 8 | **1** |

The true width came first of 585 in one case only: L=128 at +10 dB. This is too weak to build a
receiver on: it is noise-limited below +10 dB and the receiver already
has a sharper keyed read. It is still a leak, and it grows as the square root of the capture length.
It is reported here for the detectability audit and is not used (open issue below).

The coarse occupancy (`_coarse_scale`) gives rho to a few percent, and the band centre to about
0.01 cycles per sample. Neither gives P to the quarter sample over eight rows that the keyed read
needs.

**Conclusion.** No read of L / rho exists that does not go through the width. The width is part of
the key's identity, so a blind receiver pays per width it tries. What can be made direct is the
read of the period *within* each width. That is what replaces the scale grid.

## The read: one map per scale band, folded at every period

`_period_read(x, power, L, key, ..., lo, hi, centre, offsets)`, in `waveform.py`, section "The row
period, read per width":

1. **The map.** For one width and one band of scales, the dark coefficient of a single row is formed
   at every start position over eight rows' span of the capture. Positions are spaced
   `1/PERIOD_POSITIONS` of a transmitted sample. There is one sliding product per sub-sample phase,
   and every phase and carrier probe is computed in one matrix product. The filter is the scaled
   read's least squares (`_scaled_coords`), written as a filter on the samples. It reads every k-th
   received sample when those samples still over-sample the sender (`PERIOD_SPACING`). Its Gram
   `E^H E` is a Dirichlet kernel of the index difference, so it is written in closed form
   (`_dirichlet_gram`).
2. **One map serves a band of scales.** Inside a row, the scale only stretches it. Read centred on
   the map's row, a row stretched by a fraction of a sample stays dark. A map therefore covers a
   band 1/(2L) wide (`PERIOD_BANDS = 2`).
3. **The fold.** Every period in the band is then a sum of map values at s0, s0+P, …, s0+7P. That is
   eight look-ups per hypothesis, where the grid computed eight dot products. Every period the band
   allows, every start and every carrier probe is folded. Rows are folded L apart only: the
   receiver searches no cyclic extension (`research/extension.md`).
4. **Suppression and refinement.** Each map keeps the best fold at each start, and the best
   `PERIOD_PER_MAP` starts that are at least a transmitted sample apart. The width's best
   `max(8, L/4)` folds are re-read on 32 rows, from the map's own filters, on a period grid four
   times finer. A lock reads 0.04-0.10 there, and the best chance fold of a width reads 0.27-0.5.
5. Everything after is unchanged. The refined candidates of all widths are sorted. The best
   `PERIOD_RANK = 16` are re-read on 64 rows and merged into four leads, and those are settled,
   compared on 256 rows and jointly settled on every row.

`_discover_scaled(iq, key, None)` now searches every width the capture can evidence:
`2 .. min(256, isqrt(samples × hi))`, the integer read's range expressed in the sender's samples.
Given a width range, it searches that range.

Nothing is added on the air. The read uses no transform, no correlation against a template, no
resampling and no filter fitting: the map is the dark-mode read evaluated at every start.
`test_no_fft` and `test_no_preshared_parameters` pass, and the `discover`/`decode` signatures are
unchanged. The public front door stays the integer read. See "Should this be the front door" below.

## Fixes found on the way (all in `waveform.py`)

- **`_scaled_scan` phase wrap (pre-existing bug).** Take a row whose start lies within 1/128 of a
  sample after a received sample, so that delta rounds to a whole sample. That row was read from
  `first + 1` instead of `first - 1`: two samples late. It hit about 1 row in 128. Pinned by
  `test_the_scan_wraps_a_whole_sample_phase_to_the_sample_before`: the old code returns start 98.0,
  and the fixed code returns 100.0 for the true 100.0002. The mesh's `receive_known` uses this scan.
- **Coarse bracket under noise (`_bracket_of`).** Noise lifts the entropy read (2^H_F) and cannot
  move the rank edge. A soft band edge lifts the rank edge and lowers the entropy read. So when h > k,
  the excess is noise. The bracket becomes the rank edge's own, from k/(1 + `COARSE_ROLLOFF` = 0.06,
  the hardware maximum) up to k. In simulation at 10 dB, h read +5% to +34% high, which made the
  bracket up to 35% wide. k stayed within ±1.2%. The bracket now still holds the truth, and its top
  sits within 4% of it.
- **The band centre at full band.** A band that fills the receiver's has no edge, so the centroid of
  its "occupied" bins is noise: −0.038 cycles/sample at rho 1, or −1.2 cycles/row at L=32. The
  centre is now held to (1 − rho)/2, the room the band leaves. An unshifted carrier set is also
  probed whenever the centre read is not ~0, because the centre is good only to ~1 cycle per row at
  L/rho ~ 100. A hardware capture at 0.9375 read 0.99 cycles/row for a sender on centre, and every
  probe missed it.
- **Settles stop at equal rates.** `_settle_scaled` no longer moves the scale above 1. A row a hair
  short of L samples still reads when its samples happen to fall L to a row. On the hardware capture
  at rho = 1, a settle wandered to +59 ppm that way, into a fit of a sender wider than the receiver
  (BER 0.46). The band touching rho = 1 is also read on a second map at exactly 1.
- **The refined period is held inside the bracket.** At rho = 1, refinement could step to a period
  just under L (rho > 1). The exact read refuses that (fewer samples than modes), which made the true
  lock lose its 64-row re-read.
- **`_scaled_gram` uses the closed-form Gram.** It agrees with `E0^H E0` to 1e-14 and skips an
  n·L² product on every scale step of every settle.

## Measurements

The channel is `tests/_ratechannel.py` `at_rate` with a 2e-4 cycles/sample carrier offset. Captures
are 160 rows (280 at L=256), K = L − 4, DQPSK. Each capture starts inside the buffer's last row, so
the cyclic stream does not restart mid-capture, which breaks the differential chain the alphabet
reads. Runs were single-threaded (`OPENBLAS_NUM_THREADS=1`) on a desktop CPU, 6-8 runs in
parallel, so absolute times are inflated by perhaps 10-30%.

The matrix and the full acquisitions below were measured while the receiver also folded a cyclic
extension P = 0..4 with every width. The receiver now folds P = 0 only; a reduced rerun of the lock
stage on that receiver follows the matrix ("Rows L apart only").

"Old grid est." is the replaced scale grid with the width given. It is the grid size under the old
bracket, multiplied by one timed `_scaled_scan` at 7 carriers × 5 extensions.

### The lock: is the lead after the 64-row re-read the sender?

The lock stage is everything before the settles, and it is the only part this change replaces.
Each cell was run three ways: with the width given (`(L,)`), blind (`None`, every width from 2 to
`isqrt(samples x hi)`), and against the replaced scale grid, whose time is estimated. The scan
times are seconds for the period reads plus the 64-row re-read. "v32" is the 32-row dark read of
the true fold against the best chance fold of the blind search.

| L | rho | SNR | given: lock / scan s | old grid est. s | blind: lock / widths / scan s | v32 true / best chance (blind) |
|---|---|---|---|---|---|---|
| 32 | 0.3 | 20 dB | yes / 0.69 | 26 (281 scales) | yes / 71 / 17 | 0.068 / 0.327 |
| 32 | 0.3 | 10 dB | yes / 1.13 | 160 (868 scales) | yes / 71 / 18 | 0.099 / 0.324 |
| 32 | 0.3 | clean | yes / 0.68 | 18 (176 scales) | yes / 71 / 17 | 0.062 / 0.342 |
| 32 | 0.514403 | 20 dB | yes / 0.58 | 11 (251 scales) | yes / 72 / 18 | 0.073 / 0.314 |
| 32 | 0.514403 | 10 dB | yes / 1.00 | 63 (604 scales) | yes / 72 / 16 | 0.108 / 0.346 |
| 32 | 0.514403 | clean | yes / 0.68 | 8 (188 scales) | yes / 72 / 17 | 0.068 / 0.363 |
| 32 | 0.833333 | 20 dB | yes / 0.76 | 6 (202 scales) | yes / 71 / 16 | 0.098 / 0.352 |
| 32 | 0.833333 | 10 dB | yes / 1.06 | 9 (300 scales) | yes / 71 / 13 | 0.131 / 0.346 |
| 32 | 0.833333 | clean | yes / 0.52 | 5 (186 scales) | yes / 71 / 13 | 0.104 / 0.333 |
| 32 | 0.95 | 20 dB | yes / 0.27 | 2 (219 scales) | yes / 72 / 6 | 0.154 / 0.294 |
| 32 | 0.95 | 10 dB | yes / 0.23 | 2 (196 scales) | yes / 72 / 5 | 0.252 / 0.287 |
| 32 | 0.95 | clean | yes / 0.29 | 3 (223 scales) | yes / 72 / 7 | 0.138 / 0.302 |
| 32 | 1 | 20 dB | yes / 0.65 | 3 (126 scales) | yes / 70 / 10 | 0.056 / 0.328 |
| 32 | 1 | 10 dB | yes / 0.52 | 2 (118 scales) | yes / 70 / 5 | 0.129 / 0.340 |
| 32 | 1 | clean | yes / 0.61 | 2 (126 scales) | yes / 70 / 12 | 0.046 / 0.323 |
| 64 | 0.3 | 20 dB | yes / 0.61 | 132 (446 scales) | yes / 101 / 22 | 0.044 / 0.317 |
| 64 | 0.3 | 10 dB | yes / 0.66 | 278 (1518 scales) | yes / 101 / 21 | 0.100 / 0.272 |
| 64 | 0.3 | clean | yes / 0.58 | 51 (279 scales) | yes / 101 / 13 | 0.034 / 0.350 |
| 64 | 0.514403 | 20 dB | yes / 0.49 | 31 (367 scales) | yes / 101 / 21 | 0.054 / 0.331 |
| 64 | 0.514403 | 10 dB | yes / 0.56 | 90 (1036 scales) | yes / 101 / 22 | 0.073 / 0.271 |
| 64 | 0.514403 | clean | yes / 0.39 | 25 (288 scales) | yes / 101 / 12 | 0.051 / 0.338 |
| 64 | 0.833333 | 20 dB | yes / 0.41 | 10 (281 scales) | yes / 102 / 17 | 0.055 / 0.339 |
| 64 | 0.833333 | 10 dB | yes / 0.41 | 14 (461 scales) | yes / 102 / 17 | 0.097 / 0.341 |
| 64 | 0.833333 | clean | yes / 0.33 | 9 (297 scales) | yes / 102 / 11 | 0.048 / 0.348 |
| 64 | 0.95 | 20 dB | yes / 0.38 | 10 (397 scales) | yes / 102 / 14 | 0.067 / 0.351 |
| 64 | 0.95 | 10 dB | yes / 0.48 | 13 (380 scales) | yes / 102 / 23 | 0.131 / 0.312 |
| 64 | 0.95 | clean | yes / 0.51 | 15 (403 scales) | yes / 102 / 14 | 0.062 / 0.331 |
| 64 | 1 | 20 dB | yes / 0.69 | 7 (195 scales) | yes / 100 / 12 | 0.032 / 0.328 |
| 64 | 1 | 10 dB | yes / 0.67 | 7 (195 scales) | yes / 100 / 12 | 0.121 / 0.310 |
| 64 | 1 | clean | yes / 0.67 | 7 (196 scales) | yes / 100 / 12 | 0.024 / 0.329 |
| 128 | 0.3 | 20 dB | yes / 1.46 | 744 (923 scales) | yes / 144 / 71 | 0.043 / 0.311 |
| 128 | 0.3 | 10 dB | yes / 1.69 | 2499 (2966 scales) | yes / 144 / 70 | 0.082 / 0.278 |
| 128 | 0.3 | clean | yes / 1.53 | 474 (548 scales) | yes / 144 / 69 | 0.036 / 0.305 |
| 128 | 0.514403 | 20 dB | yes / 1.82 | 218 (691 scales) | yes / 144 / 60 | 0.065 / 0.293 |
| 128 | 0.514403 | 10 dB | yes / 1.86 | 576 (1947 scales) | yes / 144 / 58 | 0.079 / 0.315 |
| 128 | 0.514403 | clean | yes / 1.43 | 137 (486 scales) | yes / 144 / 39 | 0.048 / 0.286 |
| 128 | 0.833333 | 20 dB | yes / 0.98 | 44 (504 scales) | yes / 144 / 38 | 0.046 / 0.314 |
| 128 | 0.833333 | 10 dB | yes / 0.74 | 70 (867 scales) | yes / 144 / 38 | 0.136 / 0.285 |
| 128 | 0.833333 | clean | yes / 0.82 | 52 (518 scales) | yes / 144 / 28 | 0.058 / 0.313 |
| 128 | 0.95 | 20 dB | yes / 0.82 | 49 (746 scales) | yes / 145 / 35 | 0.054 / 0.306 |
| 128 | 0.95 | 10 dB | yes / 1.32 | 64 (682 scales) | yes / 145 / 57 | 0.155 / 0.320 |
| 128 | 0.95 | clean | yes / 0.79 | 53 (761 scales) | yes / 145 / 35 | 0.054 / 0.341 |
| 128 | 1 | 20 dB | yes / 1.67 | 41 (346 scales) | yes / 142 / 32 | 0.070 / 0.352 |
| 128 | 1 | 10 dB | NO [128, 0.987387, 4] / 1.57 | 40 (343 scales) | NO [78, 0.9961089, 1] / 142 / 32 | 0.685 / 0.358 |
| 128 | 1 | clean | yes / 1.39 | 37 (347 scales) | yes / 142 / 30 | 0.043 / 0.288 |
| 256 | 0.3 | 20 dB | yes / 7.34 | 5853 (1866 scales) | yes / 255 / 435 | 0.033 / 0.273 |
| 256 | 0.3 | 10 dB | yes / 7.56 | 17996 (5837 scales) | yes / 255 / 434 | 0.086 / 0.267 |
| 256 | 0.3 | clean | yes / 7.34 | 3549 (1129 scales) | yes / 255 / 440 | 0.025 / 0.273 |
| 256 | 0.514403 | 20 dB | yes / 8.20 | 1661 (1466 scales) | yes / 255 / 460 | 0.030 / 0.285 |
| 256 | 0.514403 | 10 dB | yes / 7.97 | 4340 (3902 scales) | yes / 255 / 464 | 0.071 / 0.257 |
| 256 | 0.514403 | clean | yes / 8.60 | 1206 (998 scales) | yes / 255 / 461 | 0.025 / 0.300 |
| 256 | 0.833333 | 20 dB | yes / 6.29 | 508 (1002 scales) | yes / 255 / 303 | 0.049 / 0.285 |
| 256 | 0.833333 | 10 dB | yes / 5.77 | 819 (1716 scales) | yes / 255 / 302 | 0.080 / 0.281 |
| 256 | 0.833333 | clean | yes / 4.27 | 458 (906 scales) | yes / 255 / 150 | 0.045 / 0.292 |
| 256 | 0.95 | 20 dB | yes / 8.95 | 1039 (1358 scales) | yes / 255 / 230 | 0.039 / 0.303 |
| 256 | 0.95 | 10 dB | yes / 12.46 | 1256 (1320 scales) | yes / 255 / 345 | 0.124 / 0.303 |
| 256 | 0.95 | clean | yes / 7.61 | 1315 (1393 scales) | yes / 255 / 232 | 0.057 / 0.244 |
| 256 | 1 | 20 dB | yes / 2.57 | 272 (557 scales) | yes / 255 / 156 | 0.040 / 0.306 |
| 256 | 1 | 10 dB | yes / 2.90 | 247 (556 scales) | yes / 255 / 155 | 0.108 / 0.254 |
| 256 | 1 | clean | yes / 2.65 | 232 (557 scales) | yes / 255 / 157 | 0.028 / 0.297 |

The sender leads in **59 of 60 cells given the width and 59 of 60 blind**. The one miss was L=128
at exactly equal rates, 10 dB, with a fractional sample offset. There an extension-4 chance fold
read below the true fold, and the refinement slots the extension folds took left the true fold's
best start unrefined (its first refined fold read 0.69 on 32 rows, ranked 27th given and ~2050th
blind). With rows folded L apart only, that cell locks (next section).

With the width given, the scan stage costs **0.2-1.9 s at L <= 128 and 2.6-15 s at L=256**. The
replaced grid cost 2-2500 s and 230-18000 s. Blind over the whole evidenced range it costs
**5-23 s (captures of L=32/64), 26-71 s (L=128) and 120-460 s (L=256)**. The widest widths
dominate.

### Rows L apart only (the current receiver)

The same captures (seed 1), lock stage with the extension coordinate gone. "Before" is the
last matrix run (or the earlier run where the cell was not rerun). Both
ran single-threaded with 6-8 other jobs on the machine, so the scan times compare to perhaps 20%.

| L | rho | SNR | given before -> now | blind before -> now | v32 true / best chance, now | blind scan s before -> now |
|---|---|---|---|---|---|---|
| 128 | 1 | 10 dB | **NO** (P=4 lead; true 27th) -> yes, 1st | **NO** (L=78) -> yes, 1st | 0.132 / 0.359 | 31.7 -> 15.6 |
| 128 | 1 | 20 dB | yes -> yes | yes -> yes | 0.070 / 0.292 | 31.4 -> 18.0 |
| 128 | 0.95 | 10 dB | yes -> yes | yes -> yes | 0.155 / 0.311 | 57.1 -> 30.4 |
| 128 | 0.514 | 10 dB | yes -> yes | yes -> yes | 0.079 / 0.317 | 57.8 -> 35.9 |
| 64 | 1 | 10 dB | yes -> yes | yes -> yes | 0.121 / 0.326 | 12.0 -> 5.7 |
| 64 | 0.514 | 10 dB | yes -> yes | yes -> yes | 0.073 / 0.325 | 21.5 -> 12.6 |
| 32 | 1 | 10 dB | yes -> yes | yes -> yes | 0.129 / 0.362 | 4.8 -> 2.8 |
| 32 | 0.514 | 10 dB | yes -> yes | yes -> yes | 0.108 / 0.310 | 16.3 -> 6.6 |
| 256 | 1 | 10 dB | yes -> yes | not rerun | 0.108 / - | - |

Every cell leads with the sender, the former miss included, and the true fold's 32-row read is the
same number as before wherever it was refined before: removing the extension changed no true fold,
only the chance folds it competes with. The blind scan is 1.6-2.5x faster; the folds themselves
are 5.3-5.6x fewer (the sum of (L + 2P)^2 over P = 0..4 against L^2), and the per-width maps, which
are shared, are unchanged. The full acquisition of the former miss, width given, now reads
L=128, K=124, M=4, scale 1 - 5e-8, BER 0.030 at 10 dB, in 18 s.

### Full acquisitions (the real `_discover_scaled`, settles included)

| L | mode | cells | L, scale right | K, M right | total s (median, range) | scan s (median) | settles s | joint settle s |
|---|---|---|---|---|---|---|---|---|
| 32 | given | 15 | 15 | 15 | 19 (3-56) | 0.1 | 7 | 8 |
| 32 | blind | 5 | 5 | 5 | 17 (15-38) | 8.5 | 6 | 4 |
| 64 | given | 15 | 15 | 15 | 40 (19-78) | 0.4 | 27 | 15 |
| 64 | blind | 5 | 5 | 5 | 44 (23-61) | 18.9 | 14 | 11 |
| 128 | given | 15 | 14 | 14 | 101 (39-507) | 1.5 | 68 | 23 |
| 128 | blind | 5 | 5 | 5 | 108 (80-195) | 52.9 | 46 | 27 |
| 256 | given | 2 | 2 | 2 | 537 (410-665) | 4.6 | 178 | 349 |

"Given" covers rho in {0.3, 0.514, 0.833, 0.95, 1} x {clean, 20, 10 dB}. "Blind" covers the same
ratios at 20 dB. Every successful cell reads K and M right, with BER 0 clean and at 20 dB, and
2e-4 to 1.3e-2 at 10 dB. The single miss is the equal-rate 10 dB cell above. With the width given,
the settles are now the cost, not the lock.

Timed on a quiet machine, single thread, at L=64, rho 0.514, 20 dB:

| | old (scale grid) | new |
|---|---|---|
| width given | 28.2 s | 14.5 s |
| mesh's range L+-4 (9 widths) | 181.5 s | 16.3 s |
| blind, widths 2..101 | (est. ~40 min) | 27.6 s |

The "new" figures include the closed-form Gram in `_scaled_gram`, which also speeds the settles.

### Hardware (the 8 bench captures, `caps_rates`, L=64, 116k samples)

With widths (63, 64, 65), all 8 read L=64, K=60, M=4, within 0.5-1.5 ppm of the ratio. BER is 0
at 0.33, 0.50, 0.514, 0.603 and 0.833, 5e-4 at 0.9375 (previously 3.4e-3), and 0.024 / 0.0096 at
equal rates (unchanged). That BER is cycle-aware, read through the joint read. Blind (every width,
2..~240) on 0.333, 0.514 and 0.833: all three read L=64, K=60, M=4, BER 0, in 255-789 s. Those times
were taken with 7 other jobs running.

## Should this be the front door

No. The measurements say the blind search is affordable and right, but not cheap. It costs a minute
or two for widths up to 128 and several minutes when the capture can evidence 256, against ~1.5 s
for the integer read at equal rates. Equal rates also remain the integer read's case: it is right
on the one cell the scaled path misses. So `discover` is unchanged.

`_discover_scaled(iq, key, None)` is the blind non-unit-ratio read, for a caller that has decided
to pay once per sender. The mesh's `receive_burst` already routes a narrower sender there, and it
can pass `candidates=None`. A per-sender cache (blind once, then `receive_known` at the measured
(L, rho) per burst) is the natural use.

## What did not work

- **A width-free period read.** It does not exist: see above. The keyed read needs the width, and
  the key-free trace is the detectability leak.
- **One map per 1/L of scale.** The stretch tolerance is real, about 0.03 noiseless at a quarter
  sample. At 10 dB, though, the true fold of L=256, rho 0.3 read 0.13 and ranked about 300th of its
  width, and the width locked wrong. Two maps per 1/L (1.7× the cost) read it at 0.057, ranked first.
- **Finer positions, a finer period step, or finer carrier steps instead** (8 positions per sample,
  a 0.25-sample period step, 0.1-cycle carriers). None of them moved the true fold out of the noise
  at L=256, 10 dB. The map spacing was the tolerance that mattered.
- **Keeping the top-k folds of a width without suppression.** At L=256, 10 dB the width's top 64
  were copies of a handful of chance folds (neighbouring periods and starts), and they crowded the
  true fold out.
- **Reading every third sample at rho 0.3** (spacing 0.9). Noiseless it is exact. At 10 dB the fit
  is nearly square and lifts the noise, and L=32 was not found at all. A spacing under 0.7 (k=2)
  fixed it.
- **8 rows' fold alone, without a 32-row refinement**, does not separate a lock from chance once
  hundreds of widths are folded. The best chance folds of a blind search read 0.05-0.08 on 8 rows,
  the same as a 10 dB lock.

## Open issues

- **Cost of the blind search.** It is dominated by the widest widths (∝ L³ summed over widths),
  ~7 min at rho 0.3 over 2..256. A deployment can bound the width range, or read once per sender
  and keep (L, rho) as that neighbour's measured state. That choice is the deployment's.
- **The settles now dominate an acquisition with the width given.** This is the existing engine:
  four leads settled on 64 and 256 rows, then the joint read settled on every row. At L=256, rho 0.3
  the settles took ~6 min and the joint settle ~6.5 min, because `_joint_gram` takes a complete QR
  per scale step.
- **The row-energy leak.** Equal-energy rows mark L / rho for a key-free observer. The codec's keyed
  per-row power (`waveform.row_radius`, `row_energy_leak.md`) closes it, and nothing here depends
  on the leak.
- **A foreign sender that wraps its rows in a cyclic extension is not read.** The receiver folds
  rows L apart only. Such a capture reads like noise (no dark fold, no margin), not as a wrong
  lock; see `research/extension.md`.
- **Equal rates remain the integer read's case.** At rho = 1 a row has no spare samples, so the map
  is marginal there; the cell that missed now locks, but `discover` is still the read for equal
  rates (1.5 s against 15-18 s for this path at L=128).
- **Single-thread Python, contention.** The matrix ran 6-8 jobs at once on one desktop, so its
  absolute times are high by perhaps 10-30% (the quiet-machine row above is clean).
