# The joint read: the neighbouring rows, modelled in the receiver

A receiver at another sample rate reads each row between the transmitted instants. There the
signal is the band-limited continuation of the whole stream, not of one row. The row read
(`_scaled_coords`) models a row as the continuation of that row alone, so the neighbouring rows'
converter tails, which fall as 1/d, are error. That error was the floor: the dark mode read
about −22.8 dB at L=128, noiseless. Coherent 16-QAM read at ~2e-3 BER and 64-QAM at 2–3%, both
noiseless.

The joint read models those tails. It changes nothing on the air: no extension, no repetition, no
pilot, no guard band. Each row is still read by least squares on the received samples, against
the same exponentials evaluated at real positions. Nothing is resampled and no transform is used.
`test_no_fft` and `test_no_preshared_parameters` pass unchanged, and `decode`/`discover` take the
same arguments.

Noiseless, at the true geometry, the floor falls from −22 dB to −66…−69 dB. That is the test
channel's own floor, since `tests/_ratechannel.py` quantizes each position to 1/1024 of a sample.
With noise, every case is now noise-limited. The noise the read passes is 1.01–1.05× the row
read's. 16-QAM and 64-QAM read bit-exact noiseless at ρ = 0.514 and 0.833, blind as well as
informed.

## The read

`_joint_coords(iq, L, seed, nonce, starts, scale, carrier)` returns the same coordinates as
`_scaled_coords`, without the neighbouring rows' tails in them. It has two parts, solved together
as one least-squares fit per row.

1. **The window.** The received samples from `margin` transmitted samples before the row to
   `margin` after it are modelled as one band-limited signal, periodic over the window
   (`L + 2·margin` exponentials). These are the row read's exponentials on a longer period, so the
   same factorization holds. The design matrix is `E0 · Λ`, where `E0` depends only on (scale,
   sample count) and `Λ` is a per-exponential phase set by where the row's first sample fell. The
   Gram, its QR, and the evaluation at the row's L transmitted positions are cached per (scale,
   count) (`_joint_gram`). Every row of the capture is read in a few batched matrix products.
2. **The far field.** What lies beyond the window reaches into it as `Σ_k s_k sinc(u − k)`, k
   outside. The window's own periodic wrap is an error of the same form. Over the window that is
   `sin(πu)` times a smooth function whose singularities lie beyond the edges. A skeleton of
   transmitted samples at geometric distances beyond each edge spans it. Those distances are
   1, 2, 4, … L/2, so q = 2·log2(L) columns, which is 14 at L=128. The skeleton is fitted with the
   window and discarded. Writing it as `sinc(u − k)` at whole k (`sin(πu)(−1)^k/(π(u − k))`)
   keeps every column finite, because the skeleton never meets a sample.

   The skeleton's columns depend on the row's sub-sample start, so they are solved per row, in
   the complement of the window's span: `q × q` normal equations per row, batched. The projection
   onto the complement goes through whichever cached basis is smaller, the span or its
   complement.

The window is the row's own read made longer. The skeleton removes what a longer window alone
cannot remove, because a window's floor falls only as 1/D.

**Margin.** The skeleton needs spare samples. A window of span W transmitted samples holds about
W/ρ received ones, and the model has W + q unknowns. So the margin is L/4, or wider near ρ = 1:
`margin = max(L/4, ⌈((q + 2)/(1/ρ − 1) − L)/2⌉)`. It is bounded by `JOINT_REACH = 4` rows, a
declared cost bound with its measurements in the provenance block. Past that bound, which means
ρ > 0.986 at L=128, the row read is used, as before. A row whose window the capture does not hold
(the first or last) is also read by the row read. Every row is read either way.

**Where it is used.** `_scaled_rows`, which carries every read at a scale: `read`, `read_soft`,
`read_qam`'s coherent path through `_coherent_project`/`_coherent_read`, `_mode_energy` (K) and
`_alphabet` (M, row0). `_discover_scaled` also uses it for its final settle on every row and for
the final score and runner-up (`_scaled_dark(..., joint=True)`). The scan, the 64- and 256-row
ranking and the earlier settles stay on the row read, because they only shortlist and the row read
costs a tenth as much.

## Floor and BER, true geometry

Through the codec's own readers: `read(..., scale=ρ)` for DQPSK, and `_coherent_project` +
`_dekeyed` (the informed coherent read at a scale) for QAM. `tests/_ratechannel.py`, L=128, K=124,
150 payload rows, capture from 100.25 samples before the burst. "row" is the row read (HEAD) and
"joint" is this read, on the same capture. The dark is `_scaled_dark` over every row. Errors are
in parentheses.

| ρ | payload | SNR | dark row | dark joint | BER row | BER joint | bits |
|---|---|---|---:|---:|---:|---:|---:|
| 0.514 | DQPSK | clean | −22.3 | **−69.6** | 0 (0) | 0 (0) | 37 200 |
| 0.514 | DQPSK | 30 | −21.7 | −32.8 | 0 (0) | 0 (0) | 37 200 |
| 0.514 | DQPSK | 25 | −20.9 | −27.8 | 0 (0) | 0 (0) | 37 200 |
| 0.514 | DQPSK | 20 | −19.2 | −22.8 | 0 (0) | 0 (0) | 37 200 |
| 0.514 | 16-QAM | clean | −23.3 | **−69.0** | 1.41e-3 (105) | **0** | 74 400 |
| 0.514 | 16-QAM | 30 | −22.7 | −32.8 | 1.40e-3 (104) | **0** | 74 400 |
| 0.514 | 16-QAM | 25 | −21.8 | −27.8 | 1.67e-3 (124) | **0** | 74 400 |
| 0.514 | 16-QAM | 20 | −19.9 | −22.8 | 2.58e-3 (192) | **0** | 74 400 |
| 0.514 | 64-QAM | clean | −22.7 | **−68.6** | 2.37e-2 (2644) | **0** | 111 600 |
| 0.514 | 64-QAM | 30 | −22.4 | −32.8 | 2.53e-2 (2824) | **0** | 111 600 |
| 0.514 | 64-QAM | 25 | −21.7 | −27.8 | 2.96e-2 (3301) | **9.0e-6 (1)** | 111 600 |
| 0.514 | 64-QAM | 20 | −20.0 | −22.8 | 4.44e-2 (4955) | **6.8e-3 (756)** | 111 600 |
| 0.833 | DQPSK | clean | −22.1 | **−66.2** | 0 (0) | 0 (0) | 37 200 |
| 0.833 | DQPSK | 30 | −21.5 | −30.8 | 0 (0) | 0 (0) | 37 200 |
| 0.833 | DQPSK | 25 | −20.6 | −25.8 | 0 (0) | 0 (0) | 37 200 |
| 0.833 | DQPSK | 20 | −18.5 | −20.8 | 5.4e-5 (2) | **0** | 37 200 |
| 0.833 | 16-QAM | clean | −22.9 | **−66.7** | 2.67e-3 (199) | **0** | 74 400 |
| 0.833 | 16-QAM | 30 | −22.3 | −30.8 | 3.04e-3 (226) | **0** | 74 400 |
| 0.833 | 16-QAM | 25 | −21.2 | −25.8 | 3.78e-3 (281) | **0** | 74 400 |
| 0.833 | 16-QAM | 20 | −18.9 | −20.8 | 7.63e-3 (568) | **1.75e-4 (13)** | 74 400 |
| 0.833 | 64-QAM | clean | −22.5 | **−66.8** | 3.11e-2 (3467) | **0** | 111 600 |
| 0.833 | 64-QAM | 30 | −21.8 | −30.8 | 3.50e-2 (3911) | **0** | 111 600 |
| 0.833 | 64-QAM | 25 | −20.7 | −25.8 | 4.30e-2 (4800) | **1.1e-3 (122)** | 111 600 |
| 0.833 | 64-QAM | 20 | −18.5 | −20.8 | 6.63e-2 (7400) | **2.5e-2 (2740)** | 111 600 |

The floor gain is **44–47 dB noiseless**, and that is bounded by the channel model, not the read.
With noise the joint dark reads the noise: −SNR − 10·log10(1/ρ) less the read's ~0.1 dB penalty,
so −32.8 at 30 dB and ρ = 0.514. The row read's dark sits 10, 6 and 3 dB above that at 30, 25 and
20 dB, which is its floor adding in. The remaining errors are noise errors. Coherent 64-QAM at
equal rates reads 2.2e-2 at 21 dB and 3.8e-3 at 24 dB (`research/coherent_blind.md`). The receiver
at ρ has 10·log10(1/ρ) more samples per symbol (+2.9 dB at 0.514, +0.8 dB at 0.833), and the
joint numbers sit where that predicts.

For comparison, the cyclic extension (`research/extension.md`), which is detectable on the air,
bought 2 dB (−22.8 → −24.9). The joint read buys 44 dB with nothing on the air.

Other widths and scales (noiseless, true geometry, dark mode, row → joint, dB):

| L | ρ = 0.3 | 0.514 | 0.833 | 0.9 | 0.95 |
|---:|---|---|---|---|---|
| 64 | −18.7 → −70.0 | −18.6 → −68.8 | −18.6 → −66.5 | −18.3 → −65.1 | −18.5 → −66.3 |
| 128 | −23.2 → −70.4 | −23.4 → −68.2 | −23.1 → −68.0 | −22.7 → −65.6 | −22.1 → −65.6 |
| 256 | −22.9 → −71.7 | −22.3 → −68.4 | −22.1 → −67.0 | −21.8 → −65.4 | −21.5 → −67.0 |

The margin widens near 1. At L=64 it is 31 samples at 0.9 and 102 at 0.95. At L=128 it is 89
at 0.95 and 328 at 0.98, where the read still gives −65.6 dB. The floor does not degrade with the
margin; the one-off build does (see Cost).

## Blind

`_discover_scaled(y, key, (128,), carrier_span=0.3)` through `tests/_ratechannel.py`, with carrier
2e-4 cycles per received sample. DQPSK captures start 300.4 transmitted samples in (mid-row). QAM
captures start 40 samples before the burst. Every capture ends before the cyclic buffer wraps.
Everything is measured: width, mode count, scale, P, row, carrier, and for QAM the reference row
and the order M (`_coherent_read`, `_qam_order`, with `decode`'s `K + BEACON_MODES` give-back).
"HEAD" is the whole pipeline with the joint read swapped back to the row read, which is HEAD's
behaviour exactly. "joint" is the codec as it now is. Both run on the same capture.

| ρ | payload | SNR | L/K/M read, HEAD → joint | scale error, HEAD → joint | score (dB), HEAD → joint | BER HEAD | BER joint | bits |
|---|---|---|---|---|---|---:|---:|---:|
| 0.514 | DQPSK | clean | 128/124/4 → 128/124/4 | -2.1e-07 → -1.9e-10 | -22.9 → -68.9 | 8.4e-05 (3) | 0 | 35712 |
| 0.514 | DQPSK | 30 | 128/124/4 → 128/124/4 | -1.2e-07 → +9.3e-08 | -22.3 → -32.7 | 2.8e-05 (1) | 0 | 35712 |
| 0.514 | DQPSK | 25 | 128/124/4 → 128/124/4 | -1.2e-07 → +1.6e-07 | -21.3 → -27.7 | 2.8e-05 (1) | 0 | 35712 |
| 0.514 | DQPSK | 20 | 128/124/4 → 128/124/4 | -5.7e-08 → +2.7e-07 | -19.4 → -22.7 | 0 | 0 | 35712 |
| 0.514 | 16-QAM | clean | 128/124/16 → 128/124/16 | -1.7e-07 → -3.1e-09 | -22.3 → -68.9 | 2.8e-03 (219) | 0 | 77872 |
| 0.514 | 16-QAM | 30 | 128/124/16 → 128/124/16 | -2.1e-08 → +1.2e-07 | -22.1 → -32.3 | 2.8e-03 (221) | 0 | 77872 |
| 0.514 | 16-QAM | 25 | 128/124/16 → 128/124/16 | +9.6e-08 → +1.4e-07 | -21.4 → -27.3 | 3.2e-03 (250) | 0 | 77872 |
| 0.514 | 16-QAM | 20 | 128/124/16 → 128/124/16 | +3.0e-07 → +4.0e-07 | -19.7 → -22.3 | 4.9e-03 (381) | 0 | 77872 |
| 0.514 | 64-QAM | clean | 128/124/64 → 128/124/64 | -1.6e-07 → -1.9e-09 | -23.0 → -69.7 | 2.5e-02 (2705) | 0 | 109368 |
| 0.514 | 64-QAM | 30 | 128/124/64 → 128/124/64 | -5.4e-07 → -3.9e-07 | -22.7 → -32.7 | 2.6e-02 (2888) | 0 | 109368 |
| 0.514 | 64-QAM | 25 | 128/124/64 → 128/124/64 | -8.5e-07 → -6.7e-07 | -22.0 → -27.7 | 3.1e-02 (3395) | 1.8e-04 (20) | 109368 |
| 0.514 | 64-QAM | 20 | 128/124/4 → 128/124/64 | -1.4e-06 → -1.2e-06 | -20.2 → -22.7 | 5.0e-01 (18207) | 1.0e-02 (1126) | 109368 |
| 0.833 | DQPSK | clean | 128/124/4 → 128/124/4 | -1.3e-07 → -8.7e-10 | -22.6 → -65.9 | 0 | 0 | 35712 |
| 0.833 | DQPSK | 30 | 128/124/4 → 128/124/4 | +6.1e-08 → +2.6e-07 | -22.1 → -30.8 | 5.6e-05 (2) | 0 | 35712 |
| 0.833 | DQPSK | 25 | 128/124/4 → 128/124/4 | +2.2e-07 → +4.7e-07 | -21.0 → -25.8 | 5.6e-05 (2) | 0 | 35712 |
| 0.833 | DQPSK | 20 | 128/124/4 → 128/124/4 | +5.1e-07 → +8.6e-07 | -18.7 → -20.9 | 8.4e-05 (3) | 0 | 35712 |
| 0.833 | 16-QAM | clean | 128/124/16 → 128/124/16 | +3.8e-08 → -9.1e-10 | -22.4 → -66.4 | 2.3e-03 (181) | 0 | 77872 |
| 0.833 | 16-QAM | 30 | 128/124/16 → 128/124/16 | -1.7e-07 → -1.0e-08 | -21.7 → -30.7 | 2.5e-03 (193) | 0 | 77872 |
| 0.833 | 16-QAM | 25 | 128/124/16 → 128/124/16 | -4.6e-07 → -1.2e-08 | -20.5 → -25.8 | 3.7e-03 (285) | 0 | 77872 |
| 0.833 | 16-QAM | 20 | 128/124/16 → 128/124/16 | -1.1e-06 → +8.3e-09 | -18.3 → -20.8 | 8.4e-03 (656) | 4.5e-04 (35) | 77872 |
| 0.833 | 64-QAM | clean | 128/124/64 → 128/124/64 | -8.7e-07 → +2.7e-09 | -23.0 → -67.2 | 2.5e-02 (2728) | 0 | 109368 |
| 0.833 | 64-QAM | 30 | 128/124/64 → 128/124/64 | +5.4e-07 → -1.1e-07 | -22.4 → -31.0 | 3.0e-02 (3227) | 0 | 109368 |
| 0.833 | 64-QAM | 25 | 128/124/64 → 128/124/64 | +1.9e-06 → +1.1e-06 | -21.0 → -25.7 | 4.3e-02 (4710) | 1.6e-03 (176) | 109368 |
| 0.833 | 64-QAM | 20 | 128/124/4 → 128/124/64 | +1.1e-07 → -2.3e-07 | -19.0 → -21.0 | 5.0e-01 (18198) | 2.4e-02 (2650) | 109368 |

The score is the dark mode over every row at the measured geometry. HEAD reads it on the row read
and the joint pipeline on the joint read. "L/K/M" is what discovery and the order read returned.
For QAM, K includes `decode`'s beacon give-back.

- **Noiseless**, every payload reads bit-exact through the joint pipeline. The score falls from
  about −22.5 dB to −66…−70 dB, which is the true-geometry floor. The scale error falls from
  ~1e-7 to ~1e-9, because the final settle now sees a minimum the row read's floor hid. HEAD's
  blind read made 3 DQPSK errors at ρ = 0.514, 181–219 errors in 16-QAM (2.3–2.8e-3) and 2 700
  in 64-QAM (2.5e-2).
- **The coherent order read gains margin.** The lattice closure on the true order rises from
  0.28 to 0.999 noiseless (64-QAM), and from 0.13 to 0.55–0.71 at 25 dB. At 20 dB, HEAD's 64-QAM
  closure (0.12) fell under the 4-QAM rung's structural ~0.14 and the order was read as 4, giving
  BER 0.50 (`research/coherent_blind.md`). The joint pipeline reads 64 there (closure 0.21–0.35)
  and decodes at 1.0–2.4e-2. Those are noise errors, in line with the equal-rate 64-QAM figures.
- **With noise**, the joint pipeline's BER is of the same order as the informed joint read's
  above. Those are different captures, so this is a comparison of orders, not of error counts.
  For 64-QAM it is 1.6e-3 against 1.1e-3 at 25 dB and ρ = 0.833, and 2.4e-2 against 2.5e-2 at
  20 dB. The exception is 64-QAM at 25 dB and ρ = 0.514, at 1.8e-4 against 9e-6. There the
  measured scale is 6.7e-7 off (HEAD's was 8.5e-7), and the gap is most likely that geometry
  error, measured through noise. This was not separated further.

Discovery wall-clock for these runs was 0.7–1.7× HEAD's. They ran 8 at a time on 8 cores, so take
the single-run figures in the cost table as the comparison.

## Cost

One core (`OPENBLAS_NUM_THREADS=1`), this machine, nothing else running. The read of 150 rows
(`_scaled_coords` against `_joint_coords`) is the median of 7, after the first call. "Build" is
that first call, which factorizes the window once per (scale, sample count) and caches it.

| L | ρ | margin | row read | joint read | ratio | first call (build) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.514 | 16 | 1.2 ms | 28.9 ms | 23× | 44 ms |
| 64 | 0.833 | 16 | 1.3 ms | 11.4 ms | 9× | 21 ms |
| 64 | 0.900 | 31 | 0.9 ms | 15.7 ms | 17× | 21 ms |
| 64 | 0.950 | 102 | 1.3 ms | 25.1 ms | 19× | 127 ms |
| 128 | 0.514 | 32 | 4.5 ms | 57.6 ms | 13× | 181 ms |
| 128 | 0.833 | 32 | 2.1 ms | 21.8 ms | 10× | 46 ms |
| 128 | 0.900 | 32 | 2.0 ms | 19.0 ms | 10× | 48 ms |
| 128 | 0.950 | 89 | 2.0 ms | 27.9 ms | 14× | 105 ms |

So the read costs 9–23× the row read. In absolute terms that is 10–60 ms for 150 rows, or 37 000
DQPSK bits. Per row, the cost is `n · min(W, n − W) · q` for the skeleton's projection plus
`n · W` for the window. Here n is the window's received samples, W = L + 2·margin, and
q = 2·log2 L. At ρ = 0.514 the projection dominates (n − W ≈ W). At 0.833 the complement is small
(n − W ≈ 38) and the read costs less than half as much.

Near ρ = 1, the build (a complete QR of the n × n window) grows as the cube of the margin. At
L=128 it is 34 ms at ρ = 0.9, 86 ms at 0.95, 0.85 s at 0.98 and 2.5 s at 0.986, where the margin
reaches `JOINT_REACH` = 4 rows. Past that bound the row read is used.

Discovery (`_discover_scaled`, L=128, DQPSK, noiseless, run alone):

| ρ | HEAD | joint final settle |
|---:|---:|---:|
| 0.514 | 150 s | 120 s |
| 0.833 | 57 s | 79 s |

Only the final settle on every row and the final score are joint, so discovery moves by −20% to
+40%. Most of its time is still the scan. The four L=64 scaled-discovery tests in
`tests/test_scale.py` pass in 40–104 s each.

## What did not work

Each of these was measured at L=128, noiseless, true geometry, ρ = 0.514 and 0.833. The scripts
are in the session scratchpad, and the method is reproducible from this description.

- **A longer window alone** (periodic over row ± margin, no skeleton). The floor falls only as
  1/D with the distance D to the window's edge: margin 32 reads −27.2, 64 reads −28.5, 128 reads
  −30.1 and 256 reads −32.0 dB. The cost grows as the window squared. This is the
  "block-tridiagonal / sliding window" read on its own: it is limited by the same 1/d tails, moved
  further out.
- **Fourier extension**: the window modelled on a longer period, so a gap of unobserved
  exponentials absorbs the edges. At ρ = 0.514 it reaches −56 to −69 dB, with noise gain 1.05,
  under SVD truncation. At ρ = 0.833 the gap eats the received redundancy and the fit is nearly
  square. A truncation that keeps the noise gain ≤ 1.13 reaches only −37 to −42 dB. Looser
  truncation gives −51 to −58 dB at noise gain 10²–10⁵. It also needs a truncation level, which is
  a tuned constant. Rejected.
- **No margin**, with either a gap or a skeleton. The row's own edge samples are then
  unconstrained. With a gap the dark reads −5 dB. With the skeleton it reads −9 dB at 0.833, with
  noise gain 10² to 10¹⁵.
- **Non-integer skeleton positions** (poles at half samples). A received sample can land on a
  pole, and the SVD failed to converge. Whole positions are far transmitted samples, and those
  never meet a received one.
- **Iterative decision-free cancellation**: read every row, subtract the neighbours' modelled
  tails (the sinc of the estimated transmitted samples, R rows each side, less the row read's own
  periodic images), re-read. Three passes at R = 1, 2, 4 do not move the floor (−23.0 → −23.3 →
  −22.4 at 0.514, −22.6 → −21.1 at 0.833). **Even with the true transmitted samples as the
  oracle, the floor stops at −26 dB.** The row read's own model is the limit: it carries the
  Nyquist exponential as e^{−iπu}, where the converter's band-limited continuation has cos(πu).
  That mismatch is `sin(πu)` times a smooth function, so the joint read's skeleton spans it. It is
  also, very likely, the −33 to −37 dB that a repeated row read (`research/extension.md`).
- **A non-periodic (sinc) window.** This is the same LS with the window's unknowns as sinc kernels
  at its transmitted samples. With the skeleton it reaches the same −65 to −68 dB, and without it
  −35 to −38. Its matrix depends on each row's sub-sample start, so no Gram is shared between rows
  and it does not batch. The periodic window gets the same floor with a cached factorization.
- **ρ > 1** (a sender wider than the receiver; measured at ρ = 1.37, K = 124). A row holds
  L/ρ = 93 received samples against 124 symbols. The whole capture holds fewer samples than the
  symbols it carries, so no joint model can recover them. The joint read declines exactly as the
  row read does: `_joint_window` is None, every row is NaN, `read` returns nothing, and the dark
  reads L. This is unchanged and pinned in `tests/test_joint_read.py`.

## Limits and what is not done

- **The skeleton reaches L/2.** Measured octave by octave: 3 per side read −46 to −50 dB, 5 read
  −54 to −67, and 7 (that is, up to L/2 at L=128) read −67 to −69. Past that the channel model's
  floor is reached, so nothing more is visible on this channel. A hardware channel has its own
  floor (roll-off, IQ imbalance), and that floor, not this one, will set the gain there. This has
  not been measured on the bench yet.
- **The model is the brick-wall band-limited continuation**, the same assumption the row read
  makes. A converter whose roll-off cuts into the band is a different continuation. Whatever the
  window and skeleton cannot represent stays as error. This is untested on hardware.
- **Near ρ = 1** the margin grows as 1/(1/ρ − 1), and the build as its cube. Past `JOINT_REACH`
  (ρ > 0.986 at L=128, ρ > 0.976 at L=64) the row read is used, along with its −22 dB floor.
- **The scan and the early settles stay on the row read.** They shortlist, and the final settle on
  the joint read is what makes the geometry exact. Moving the 64/256-row settles to the joint read
  would cost about 10× per evaluation and was not measured.
