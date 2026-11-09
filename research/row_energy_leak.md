# The row-energy leak and the keyed per-row radius

A row of the screen is `w_t B`, with `B` the keyed orthonormal aperture and `w_t` the unit-modulus
differential coordinates, so sent as is every row would carry the same energy on the air:
`||w_t B||^2 = K` exactly, every row. A key-free observer can find the row period `L / rho` from
that alone. The codec therefore sends each row at a keyed per-row power, `r_t^2 ~ Gamma(L, 1/L)`
(`waveform.row_radius`), drawn so that the envelope is second-order white, and the readers take it
off. This note measures the leak with detectors built for the job, against noise-only captures of
the same length; derives and checks the law; measures what it costs the key holder; and measures
it on the codec path ([The shipped codec](#the-shipped-codec)).

The study sections run on a screen with four dark modes (K = L − 4), with the radius applied by a
wrapper that scales each composed row. The shipped codec lights every mode (K = L), so a row's
energy is exactly `L r_t^2` and the radius is what keeps the envelope white; the rank, fold and
ripple features a spare mode adds, which this note lists, do not exist at K = L
(`rank_leak.md`).

Low detectability, as used here, means: the energy is visible to a radiometer; the structure is
not classifiable by the blind detector battery; the content is unreadable without the key. The
row-energy leak is a classifiability leak: it marks the transmission as a structured signal and
hands over its period.

**Summary.**

- Equal-energy rows leak the period strongly. Scored as a z against the noise-only spread (a
  200k-sample statistic normalized by the noise mean only understates it), the offset-contrast
  detector (D1) ranks the true width first or second among 585 at 0 dB from 83 ms of capture,
  and first from +5 dB.
  At 417 ms it gives z 38 at 0 dB and z 6 at −5 dB (L=64, rho = 1). An alignment-free presence
  detector (D2, the low-frequency deficit of `|y|^2`) gives z 23 at 0 dB. At rho = 125/243, after
  cutting to the occupied band, D1 at the true period gives z 17 at −5 dB and 54 at 0 dB. Extrapolated
  to 10 s (24 M samples), D2 reaches about −10 dB at L=64, rho = 1.
- A keyed per-row power `r_t^2 ~ Gamma(L, 1/L)` (shape L, not K) closes it. Every windowed
  energy then has white noise's variance at every offset, to ±0.5% on 3.8 M samples. The envelope
  autocorrelation's in-row sum is −0.50 for equal-energy rows and −0.0015 with the radius. The
  aligned window energy is Gamma(L)-distributed, which is exactly the white-noise law. With the
  radius, D1, D2 and the offset contrast stay inside the noise-only range at every SNR, length, L,
  rho and mode measured.
- It costs the key holder little. DQPSK BER rises by 0–15% relative (about 0.1 dB). The dark floor
  and blind discovery are unchanged. PAPR at the 99.99th percentile is 9.64–9.67 dB, the value for
  a Gaussian, against 9.4–9.5 dB for equal-energy rows. Coherent 16-QAM needs `r_t` divided out;
  the key holder knows it from the key, so the coherent reader does that.
- At K < L the radius alone does not make the structure unclassifiable. Three key-free features
  of the spare dimension remain, and the radius does not touch them. The rank deficiency (the
  minimum eigenvalue of the L-sample window covariance; PAPER §RF-5.3) is by far the strongest. It
  gives z 207 at 0 dB and 10 at −10 dB from 417 ms. The fold (`E|x_l|^2 = diag(B^H B)_l` is a
  fixed keyed profile every row repeats) has z growing as N. The spectral ripple shows as
  lag-correlation lines. The radius removes the only alignment-free, period-revealing feature; it
  does not move the detection threshold of an observer who runs the eigenvalue detector at
  rho = 1. The shipped codec lights every mode, which closes all three (`rank_leak.md`).

## The observer, the captures, the score

**Captures.** Random DQPSK payload, `K = L − 4`, L ∈ {64, 128}, keyed (seed 1234) and open
(`seed=None`). AWGN at SNR −10…+20 dB. At rho = 1 the received samples are the transmitted ones from
a random start. At rho = 125/243 (= 0.514403, the tests' ratio) the capture goes through
`tests/_ratechannel.at_rate`. That channel is the exact band-limited interpolation of the cyclic
buffer, with positions quantized to 1/1024 sample. It costs one inverse FFT per distinct sub-sample
position, and there are 243 of them at this ratio, so 4 M samples take about 40 s. A polyphase
resampler is not used for the long captures: its image leakage puts cyclostationary lines into a
white input, and the harmonic detector reads stage-2 z 20–97 on pure noise through one, against
4.9 through `at_rate`. SNR is `at_rate`'s: signal power over noise power
across the whole received band. At rho = 0.514 the in-band SNR is therefore 2.9 dB higher than the
figure quoted.

**Lengths.** 200 000 samples (83 ms at 2.4 Msps), 1 000 000 (417 ms) and 4 000 000 (1.67 s,
keyed only, a subset of SNRs). 10 s (24 M samples) is extrapolated from the measured scaling, as
stated where it is used.

**The null and the score.** Every statistic is computed on M noise-only captures of the same length:
M = 16 at 200k, 10–12 at 1M, 6–8 at 4M. At rho = 1 the null is white noise. At rho ≠ 1 it is a
white Gaussian "transmission" through the same converter at the same SNR. The null then has the
capture's occupancy, and only the row structure separates the two. That matters because at
rho < 1 presence is trivially visible from the spectrum. The question is whether the signal is
distinguishable from a Gaussian signal in the same band. The score is
`z = (stat − mean_null) / sd_null` per candidate period or lag, with deficits negated so that
larger always means more structure. Along long grids the null sd is smoothed by a running median
over ±15 grid points. "Noise-only max z" is the leave-one-out maximum over the grid on each null
capture. It is 2–6 for the scanned detectors here, so z above
about 7 is a detection and z under 4 is not. Values above ~30 carry roughly ±25% from the null-sd
estimate.

## Detectors

`p = |y|^2 / mean|y|^2` is the power envelope.

| | statistic | what it finds |
|---|---|---|
| D1 aligned variance | windows of the candidate period q from the cumulative sum of p (linear interpolation, offsets every 1/4 sample at real q). `−min_o var(e)/mean(e)^2`. At rho = 1 every integer q 16–600 | presence and period |
| D1 contrast | `−min_o var / median_o var` at the same q | the period. A white envelope has the same window variance at every offset; equal-energy rows have it at one offset |
| D2 low-frequency deficit | `−mean I(f), 0 < f ≤ 1/q` over `mean I(f)` (periodogram of p), max over cut-offs q = 16…8192 | presence, alignment-free. The aligned rows' envelope fluctuation sums to zero, so `|y|^2` has a spectral hole below about 1/period. Equivalent to the variance-time curve flattening past the period |
| D5 fold / harmonic sum | `mean_{k≤H} I(k/q) / level(k/q)`, `H` = harmonics below 0.5, `level` = running median / ln 2, so the null is analytic: `z = (S−1)·sqrt(H)`. At rho ≠ 1 the scan is blind: a Welch stage (2^15-sample segments) over every real q 16–600 on a grid fine enough to keep the top harmonic within half a bin, then the full-resolution sum refined around the top 10 peaks and around the truth | a fixed periodic mean profile, i.e. the synchronous average |
| first harmonic | `I(1/q)/level` | the cyclic-feature detector's statistic |
| lag correlation | `|Σ y[n] y*[n+ℓ]| / Σ|y|^2`, ℓ = 8…600 | the cyclic-prefix detector's statistic |
| rank | `−min_o λ_min / mean λ` of the covariance of q-sample windows at offset o, at q = L and at q = L+1 (off the comb), rho = 1 only | PAPER §RF-5.3's leak, scored the same way |

At rho ≠ 1, D1 and its contrast are evaluated at the true real period L/rho, and at four reference
periods (×1.01, ×0.99, ×1.37, ×0.73). This is the read of an observer who already has the period
from D5 or D2's knee. A blind D1 scan over real periods needs a step of about `q/N` and is not
run here. D2, D1 and D5 are also run after band-limiting the capture to the occupied band
`|f| < rho/2`, which an observer sees. That is labelled "band-limited".

## The leak, with and without the radius

Cells read `z equal-energy rows / z keyed row power`; the radius is derived in the next section,
and both arms are given here so each row reads once. Ranks in brackets count candidates ahead of
the true period's family (L and its multiples, 0 = first), among 585 integer periods at rho = 1.
For the blind D5 at rho ≠ 1 they count distinct stage-1 peaks; "-" means the true period is not
among the top 200. Keyed mode, N = 1 000 000 (417 ms).

**rho = 1, L = 64**

| detector | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | +20 dB |
|---|---|---|---|---|---|---|---|
| D1 contrast at L [rank] | −1.6 [110] / −0.8 | 6.1 [0] / −1.5 | 38 [0] / 1.3 | 115 [0] / −1.5 | 218 [0] / −1.6 | 297 [0] / 0.5 | 331 [0] / −1.3 |
| D1 aligned variance at L | 2.9 / 0.1 | 5.6 / −0.3 | 28 / 1.1 | 63 / −0.8 | 90 / −1.0 | 102 / −1.0 | 107 / −0.2 |
| D2 low-frequency deficit | 5.0 / 3.3 | 5.7 / 2.4 | 23 / 2.4 | 52 / 0.8 | 73 / 1.5 | 84 / 1.3 | 88 / 1.5 |
| D5 fold at L [rank] | 2.3 [6] / 2.5 [0] | 2.3 [0] / 3.9 [1] | 19 [0] / 23 [0] | 53 [0] / 56 [0] | 92 [0] / 81 [0] | 99 [0] / 98 [0] | 95 [0] / 90 [0] |
| first harmonic at L | −0.7 / 0.1 | 0.7 / 0.6 | −0.5 / −0.5 | 6.4 / 4.8 | 12 / 12 | 12 / 9.3 | 8.8 / 14 |
| rank: λ_min at window L | 10 / 10 | 78 / 75 | 207 / 202 | 338 / 338 | 415 / 416 | 446 / 446 | 457 / 457 |
| lag correlation, max over lags | 4.3 / 4.6 | 5.5 / 5.4 | 8.8 / 6.7 | 8.3 / 9.0 | 9.4 / 9.5 | 9.4 / 9.9 | 11 / 10 |

**rho = 1, L = 128**

| detector | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | +20 dB |
|---|---|---|---|---|---|---|---|
| D1 contrast at L [rank] | 2.7 [17] / 0.4 | 1.6 [56] / 0.3 | 22 [0] / 0.5 | 65 [0] / 1.9 | 121 [0] / 0.7 | 164 [0] / −0.8 | 183 [0] / 0.0 |
| D1 aligned variance at L | 2.6 / 1.9 | 5.1 / 0.6 | 22 / 0.6 | 49 / −1.1 | 70 / −0.6 | 80 / −1.5 | 84 / −1.3 |
| D2 low-frequency deficit | 2.6 / 2.9 | 6.6 / 2.4 | 19 / 1.6 | 44 / 1.2 | 64 / 2.2 | 74 / 0.3 | 77 / 0.4 |
| D5 fold at L [rank] | 0.4 [59] / −1.0 | −0.4 [100] / −0.7 | 4.2 [0] / 5.5 [0] | 11 [0] / 9.5 [0] | 15 [0] / 16 [0] | 15 [0] / 16 [0] | 14 [0] / 20 [0] |
| rank: λ_min at window L | −0.8 / −0.4 | 18 / 15 | 74 / 73 | 135 / 135 | 171 / 171 | 185 / 185 | 190 / 190 |
| lag correlation, max over lags | 5.5 / 5.5 | 5.1 / 4.2 | 5.5 / 4.1 | 7.4 / 5.3 | 5.9 / 6.3 | 9.2 / 5.1 | 6.1 / 6.4 |

**rho = 125/243, L = 64** (period 124.42 received samples)

| detector | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | +20 dB |
|---|---|---|---|---|---|---|---|
| D2, band-limited | 1.8 / 1.6 | 10 / 1.0 | 46 / 2.2 | 38 / 0.8 | 74 / 0.6 | 91 / 1.2 | 45 / 1.0 |
| D2, full band | 2.5 / 1.8 | 6.3 / 1.0 | 32 / 1.6 | 36 / 0.8 | 70 / 0.6 | 90 / 1.2 | 45 / 0.9 |
| D1 contrast at L/rho, band-limited | 0.0 / −0.9 | 17 / −1.4 | 54 / −0.4 | 96 / −0.2 | 199 / 0.3 | 231 / −1.8 | 181 / 0.2 |
| D1 aligned variance at L/rho | 1.9 / 0.2 | 7.9 / −0.5 | 36 / 0.3 | 47 / 0.1 | 72 / −1.4 | 80 / −0.8 | 52 / 0.3 |
| D5 fold, band-limited [blind rank] | 3.4 [48] / 2.5 [84] | 13 [-] / 7.8 [15] | 24 [135] / 26 [0] | 57 [0] / 50 [0] | 62 [0] / 61 [0] | 60 [0] / 60 [0] | 71 [0] / 61 [0] |
| lag correlation, max over lags | 6.2 / 6.2 | 4.8 / 4.4 | 5.3 / 13 | 8.3 / 5.0 | 7.5 / 8.6 | 9.0 / 12 | 6.2 / 6.6 |

**rho = 125/243, L = 128** (period 248.83)

| detector | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | +20 dB |
|---|---|---|---|---|---|---|---|
| D2, band-limited | 0.5 / 1.4 | 7.4 / 2.0 | 26 / −0.3 | 30 / 1.5 | 39 / 1.5 | 46 / 1.9 | 39 / 0.0 |
| D1 contrast at L/rho, band-limited | −0.6 / 4.3 | 5.3 / 0.4 | 26 / −0.5 | 44 / −0.4 | 84 / −0.6 | 143 / 2.5 | 59 / −0.6 |
| D1 aligned variance at L/rho | −2.1 / −1.6 | 4.8 / −0.5 | 19 / −3.1 | 30 / 0.1 | 36 / −0.5 | 39 / 0.2 | 33 / −1.2 |
| D5 fold, band-limited [blind rank] | 3.0 [97] / 2.4 | 2.7 [20] / 3.9 | 5.6 [-] / 6.0 [-] | 9.0 [49] / 10 [14] | 11 [-] / 8.5 [40] | 12 [23] / 12 [-] | 9.8 [57] / 11 [110] |
| lag correlation, max over lags | 8.0 / 4.6 | 5.2 / 11 | 5.8 / 8.3 | 6.1 / 7.7 | 7.2 / 7.6 | 5.3 / 6.1 | 5.1 / 7.1 |

What the tables say about the leak itself:

- **Presence, alignment-free (D2).** It needs no period and no offset. With equal-energy rows,
  z ≥ 19 at 0 dB everywhere and 5–10 at −5 dB, from 417 ms. Band-limiting first adds the 2.9 dB
  the occupancy hands the observer: z 46 against 32 at 0 dB, L=64.
- **The period (D1 contrast).** At rho = 1 the true width ranks first of 585 from 0 dB up at both
  widths, and from −5 dB at L=64. At 83 ms it ranks first from +5 dB. A mean-only normalization
  ranks the true width only 13th at +10 dB: the LF deficit lowers every q ≥ L, and a mean-only
  normalization ranks those ahead. The offset contrast cancels the deficit.
- **rho ≠ 1 is not harder for the observer.** The period is real, but D1 at it is as strong as at
  rho = 1, and band-limiting makes it stronger.
- **The classical detectors miss it** (`extension.md`). The first harmonic alone is
  noise-level up to 0 dB. The lag scan has no line at the row length.
- **Open and keyed modes read alike** on every row-energy statistic: the per-row energy is K in
  both. The one systematic difference is a lag-correlation line
  at lag 18 in the open L=64 aperture (z 13–17 at 1M, rho = 1; 31–37 at rho ≠ 1, +10 dB). It is
  the public aperture's spectral ripple (below), present with or without the radius.

### Length

Keyed, rho = 1, equal-energy rows. The cell is z at 83 ms / 417 ms / 1.67 s (200k / 1M / 4M
samples).

| detector | L | −10 dB | −5 dB | 0 dB |
|---|---|---|---|---|
| D2 presence | 64 | 2.8 / 5.0 / 2.4 | 1.5 / 5.7 / 15 | 7.7 / 23 / 55 |
| D2 presence | 128 | 1.2 / 2.6 / 2.9 | 1.6 / 6.6 / 7.6 | 7.3 / 19 / 37 |
| D1 aligned variance | 64 | 0.5 / 2.9 / 0.4 | 0.9 / 5.6 / 13 | 7.7 / 28 / 54 |
| D1 contrast [rank at 4M] | 64 | 0.9 / −1.6 / 0.7 | −0.9 / 6.1 / 9.0 [0] | 5.5 / 38 / 43 [0] |
| D1 contrast [rank at 4M] | 128 | −0.2 / 2.7 / 1.4 | 5.5 / 1.6 / 4.3 [1] | 12 / 22 / 39 [0] |
| D5 fold (unchanged by the radius) | 64 | 1.3 / 2.3 / 3.4 | 1.7 / 2.3 / 20 | 2.4 / 19 / 103 |
| D5 fold (unchanged by the radius) | 128 | 1.1 / 0.4 / 1.4 | 0.8 / −0.4 / 3.9 | 2.4 / 4.2 / 16 |
| rank λ_min (unchanged by the radius) | 64 | 2.7 / 10 / 47 | 9.2 / 78 / 183 | 42 / 207 / 428 |
| rank λ_min (unchanged by the radius) | 128 | −0.4 / −0.8 / 9.0 | 0.7 / 18 / 73 | 19 / 74 / 206 |

The variance statistics (D1, D2) grow like √N. Each 4× in N multiplies z by 1.9–2.6 wherever z
clears the noise, which is √4 within the null-sd uncertainty. The fold grows like N: 19 → 103 at
0 dB and 2.3 → 20 at −5 dB, L=64. That is the deflection of a fixed periodic mean, as
`z ≈ N·(L−K)/L^2·Δ/√(2L)` predicts (85·Δ per million samples at L=64, Δ = (s/(1+s))^2). The rank
statistic grows faster than √N near its threshold (×4.5 per 4× at −10 dB) and about as √N above it.

Keyed, rho = 125/243, 1.67 s. Cells read equal-energy rows / keyed row power.

| detector | L | −10 dB | −5 dB | 0 dB | +5 dB |
|---|---|---|---|---|---|
| D2 presence, band-limited | 64 | 3.4 / 1.5 | 28 / 4.6 | 39 / 1.0 | 127 / 0.7 |
| D2 presence, band-limited | 128 | 2.7 / 0.9 | 25 / 4.3 | 36 / 1.4 | 66 / 1.2 |
| D1 contrast at L/rho, band-limited | 64 | −0.5 / 0.0 | 20 / 1.3 | 84 / −0.1 | 216 / 0.3 |
| D1 contrast at L/rho, band-limited | 128 | −0.3 / −1.6 | 33 / 2.6 | 85 / 0.3 | 111 / 0.2 |
| D5 fold, band-limited [blind rank] | 64 | 8.6 [0] / 8.4 [28] | 37 [49] / 35 [0] | 112 [0] / 111 [0] | 195 [0] / 192 [0] |
| D5 fold, band-limited [blind rank] | 128 | 2.7 [7] / 3.5 [-] | 7.3 [20] / 6.0 [-] | 15 [25] / 15 [170] | 29 [19] / 31 [29] |

The −5 dB keyed-row-power D2 cells (4.3–5.2) share one 6-capture null whose own leave-one-out
maximum is 3.8 (band-limited) and 8.4 (full band). The cells do not grow with SNR: they read
0.7–1.6 at 0 and +5 dB. They are null scatter, not a residual. The blind D5 scan at rho ≠ 1 finds
L=64's period from −10 dB in 1.67 s, with or without the radius.

**10 s (24 M samples), extrapolated** from the 1.67 s points with these laws. z scales with SNR
through Δ, which is 0.0577 at −5 dB, 0.0083 at −10 dB and 0.00094 at −15 dB. The lowest SNR at
which each feature reaches z ≈ 5 in 10 s at rho = 1:

| feature | L=64, equal-energy rows | L=128, equal-energy rows | keyed row power |
|---|---|---|---|
| D2 presence (√N) | about −10 dB (z ~37 at −5) | about −8.5 dB (z ~19 at −5) | none (null) |
| D1 contrast / aligned variance, period (√N) | about −9 to −10 dB (z 22–32 at −5) | about −7 to −8.5 dB (z 11–19 at −5) | none (null) |
| D5 fold, period (∝N) | about −13 dB (z ~120 at −5, ~17 at −10) | about −9 dB (z ~25 at −5) | unchanged |
| rank λ_min, width (≥√N) | below −10 dB (47 at −10 already in 1.67 s) | about −10 dB (9.0 at −10 in 1.67 s) | unchanged |

At rho = 125/243 the observer gains another 2.9 dB by band-limiting, so the equal-energy figures
there are about 3 dB lower still.

## The keyed per-row radius

Row t is sent as `r_t · w_t B`, with `E r_t^2 = 1`.

**Why Gamma(L, 1/L).** If `x = r u` with `u` uniform on the unit sphere of C^L and
`r^2 ~ Gamma(L, 1)`, then x is exactly i.i.d. CN(0, 1). The codec's rows are not uniform on the
sphere. They live in a K-dimensional keyed subspace, with unit-modulus coordinates. So the claim to
check is the second-order one, which is what D1, D2 and every envelope-spectrum detector see.
Write `P = B^H B` (the projector onto the used modes), `a_l = P_ll` and
`c_lm = Σ_k |B_kl|^2 |B_km|^2`. For phase-uniform coordinates,

    E[|x_l|^2 |x_m|^2] = E r^4 · (a_l a_m + |P_lm|^2 − c_lm).

Summed over a whole row, `Σ_lm (|P_lm|^2 − c_lm) = tr P^2 − K = 0`, so `Var(e_t) = Var(r^2) K^2`.
White noise of the same mean power has `Var = L (K/L)^2 = K^2/L` over the row. Equality needs
`Var(r^2) = 1/L`: shape L, not K. Shape K gives L/K = 1.067 at L=64, measured. For a window
straddling two rows, the scale term `Var(r^2)·(mean)^2` replaces the fixed-sum deficit of each
partial row to O(1/L), so it holds at every offset. That is the hypothesis, and it is measured
below. The aligned window energy is then `r_t^2 K`, which is Gamma(L)-distributed: exactly white
noise's law in every moment, not only the variance.

**Drawing it.** `r_t^2 = −(1/L) Σ_{i<L} ln u_{t,i}`: the mean of L unit exponentials from keyed
uniforms in the radius's own domain (`entroptics-rf/row-radius/v1`), seekable per 64-row block
like `mode_keys`. No special function and no transform are involved. An equivalent form is
`r_t^2 = ||g_t||^2 / (2L)` with `g_t` L keyed complex normals from `_keyed_normal`.

**Noise-free check** (3.84 M samples, keyed; open mode agrees). "Window" is
`L·var/mean^2` of L-sample windows at offsets 0, 1, 2, 4, L/8, L/4 and L/2 into a row. White is 1
at every offset. The ACF sums are of the envelope autocorrelation over lags 1…L−1 and L…3L. White
is 0, ±0.004–0.008. Fold is the period-L synchronous average's variance in white units (1). Line is
the first harmonic at the row rate (1).

| L | law | window, min–max over offsets | ACF 1…L−1 | ACF L…3L | fold | line |
|---|---|---|---|---|---|---|
| 64 | none (equal-energy rows) | 0.000 – 0.487 | −0.499 | −0.001 | 73 | 24 |
| 64 | Gamma(L) (shipped, keyed) | 0.994 – 1.003 | −0.002 | 0.000 | 69 | 23 |
| 64 | Gamma(K) | 1.023 – 1.067 | +0.032 | 0.002 | 66 | 33 |
| 64 | lognormal, Var 1/L | 0.997 – 1.005 | −0.002 | 0.005 | 67 | 29 |
| 64 | uniform, Var 1/L | 0.990 – 1.004 | −0.003 | −0.001 | 69 | 21 |
| 64 | per-mode Rayleigh | 1.031 – 1.067 | +0.035 | 0.002 | 74 | 29 |
| 64 | Gamma(L), sender-random (shipped, open mode) | 0.997 – 1.009 | +0.001 | 0.008 | 68 | 36 |
| 128 | none (equal-energy rows) | 0.000 – 0.490 | −0.500 | −0.003 | 8.6 | 16 |
| 128 | Gamma(L) (shipped, keyed) | 0.996 – 1.013 | +0.007 | −0.014 | 8.1 | 21 |
| 128 | Gamma(K) | 1.017 – 1.036 | +0.018 | −0.005 | 9.5 | 19 |
| 128 | per-mode Rayleigh | 1.022 – 1.038 | +0.018 | −0.004 | 8.1 | 19 |

- Shape L is right and shape K is over-dispersed by L/K, as derived. A K-dim subspace with torus
  coordinates is close enough to uniform for second-order whiteness. The residual is inside the
  measurement.
- Any law with `Var r^2 = 1/L` passes this second-order check. Lognormal and uniform match only the
  variance, though: their aligned-window skew is 3/√L and 0 against Gamma's 2/√L. A long-capture
  third-moment detector at the aligned offset would separate them. Gamma(L) matches the white law
  exactly.
- Per-mode Rayleigh magnitudes make each row CN(0, P), whose energy is Gamma(K). That is
  over-dispersed like Gamma(K), and it turns every mode into a fading channel. It is not used.
- **Fold and line do not move.** They are not the energy leak. They are the next section.

The radius adds no periodic feature: the row-rate line and fold are unchanged (table above), and so
are the lag scan and the first harmonic in every detection table. With the radius, D1, its
contrast and D2 sit inside the noise-only range in every cell measured: both widths, both rates,
keyed and open, every SNR, 83 ms to 1.67 s.

## What the radius does not close at K < L

All three remaining features come from the spare dimension: at K = L − 4, four of L modes are
dark. The shipped codec has no spare dimension (K = L), so none of the three exists there
(`rank_leak.md`); they are set out here because they bound what the radius alone does.

1. **The rank deficiency.** The window covariance at the true width has L − K eigenvalues at the
   noise floor. Scored against the noise-only spread, it is the strongest key-free feature
   measured: z 42 at 0 dB from 83 ms, and from 417 ms z 207 at 0 dB, 78 at −5 dB and 10 at −10 dB
   (L=64). At L=128 the figures are 74 / 18 / −0.8. At L+1 it reads noise, so it identifies the
   width exactly. It grows faster than √N near threshold: at −5 dB, L=64, z goes from 9 to 78 for
   5× the samples, and from 1.67 s it reads z 47 at −10 dB (L=64) and 9.0 (L=128). Read as λmax/λmin it looks SNR-bounded; with λ_min scored against its
   own null spread it is not SNR-bounded at these lengths. The radius leaves it exactly as it is
   (the two arms agree to the noise). At rho ≠ 1 an observer that resamples to the transmitted
   rate reads it as strongly as at rho = 1 (`rank_leak.md` §1).
2. **The fold.** `E|x_l|^2` over rows is `a_l = diag(P)_l`, with P the projector onto the K used
   modes. With L − K = 4 dark modes it varies by about `2/L` along the row: rel. sd 0.034 at L=64
   and 0.016 at L=128. Every row repeats the same keyed profile. It is a cyclostationary mean, so
   D5's z grows as N: `z ≈ N·(L−K)/L^2·Δ/√(2L)`, `Δ = (s/(1+s))^2`, about 85·Δ per million
   samples at L=64 and 15–20·Δ at L=128, as measured. From 417 ms, L=64 gives z 19–23 at 0 dB and
   53 at +5 dB, and ranks the true width first. At rho ≠ 1 the blind two-stage scan finds the
   period from +5 dB at 417 ms and from −10 dB at 1.67 s (L=64). It is unchanged by the radius.
   A flat complement closes it at K < L. Span the dark complement by constant-modulus rows: a
   keyed diagonal times L − K keyed DFT rows. Then take the K data modes as an orthonormal basis
   of its complement. That makes `diag(P) = K/L` exactly. Noise-free, the fold statistic falls
   from 69 to 1.1 at L=64 and from 8.9 to 0.86 at L=128 (white: 1), with orthonormality at 1e-15
   and the dark modes still dark (`rank_leak.md`, cure (b)). At K = L, P is the identity and the
   profile is flat without it.
3. **The spectral ripple.** The off-diagonal of P is −Q (Q the dark projector), so the signal's
   autocorrelation is not zero at short lags. Its PSD carries a fixed keyed ripple. Lag-correlation
   lines appear at lags below L: z 10–17 at 1M, +10 dB, L=64, with the open aperture at lag 18. It
   is presence only and does not mark L. The radius does not change it, and the flat complement
   does not remove it either; at K = L, Q is zero.

So at K < L, against an observer at rho = 1 who runs the eigenvalue detector, the radius changes
nothing measurable. What it removes is the cheapest leak: the only one that needs neither the
period, the offset nor a rate hypothesis (D2), and the one that turns a rate-unknown capture into
a period (D1 at rho ≠ 1, z 17 at −5 dB). The full screen removes the rest of the second-order
features (`rank_leak.md`).

## Cost to the key holder

**DQPSK, informed read** (true geometry; rho = 1 through the integer read, rho = 125/243 through
`read(..., scale=rho)`, the joint read). 4 captures per cell, 288–298 kbit. Dark is the K = L − 4
screen's dark-mode level relative to the mean mode, averaged over the captures.

| rho | L | SNR | BER, equal-energy rows | BER, keyed row power | dark, equal-energy rows | dark, keyed row power |
|---|---|---|---|---|---|---|
| 1 | 64 | ∞ | 0 | 0 | −152 dB | −152 dB |
| 1 | 64 | 6 dB | 0.0970 | 0.0996 | −7.0 | −7.0 |
| 1 | 64 | 9 dB | 0.0217 | 0.0231 | −9.5 | −9.5 |
| 1 | 64 | 12 dB | 1.42e-3 | 1.57e-3 | −12.3 | −12.2 |
| 1 | 128 | 6 / 9 / 12 dB | 0.1019 / 0.0243 / 1.47e-3 | 0.1028 / 0.0241 / 1.75e-3 | −6.9 / −9.5 / −12.4 | −7.0 / −9.5 / −12.3 |
| 0.514 | 64 | ∞ | 0 | 0 | −50.0 | −49.5 |
| 0.514 | 64 | 6 / 9 / 12 dB | 0.0269 / 2.15e-3 / 5.6e-5 | 0.0280 / 2.43e-3 / 4.9e-5 | −9.1 / −12.0 / −14.9 | −9.3 / −12.0 / −14.9 |
| 0.514 | 128 | ∞ | 0 | 0 | −49.8 | −55.1 |
| 0.514 | 128 | 6 / 9 / 12 dB | 0.0279 / 2.41e-3 / 1.3e-5 | 0.0278 / 2.26e-3 / 1.7e-5 | −9.4 / −12.0 / −14.9 | −9.2 / −12.1 / −14.9 |

The BER rises by 0–15% relative, about 0.1 dB, which is the per-row SNR spread. Per-row power has
sd 0.54 dB at L=64 and 0.39 dB at L=128. DQPSK is phase-only, so the receiver never needs `r_t`.
The dark level is a ratio and does not move.

**Blind discovery** (K = L − 4, dark-mode discovery). 160-row bursts. Locked means
(L, K, M) right and the scale within 1e-4.

| | equal-energy rows | keyed row power |
|---|---|---|
| rho 1, `decode(y, key, (L,))`, L 64 and 128, keyed and open, 20 dB | 12/12, BER 0 | 12/12, BER 0 |
| rho 1, same, 6 dB | 12/12, BER 0.095–0.111 | 12/12, BER 0.093–0.141 |
| rho 0.514, `_discover_scaled(y, key, (L,))`, L 64 and 128, keyed and open, 20 dB | 12/12, BER 0 | 12/12, BER 0 |
| rho 0.514, L=64, 6 dB, keyed | 9/13, BER ~0.03 | 10/13 |
| rho 0.514, L=64, 6 dB, open | 4/13 | 8/13 |
| rho 0.514, L=128, 6 dB, keyed | 12/19, BER ~0.03 | 14/19 |
| rho 0.514, L=128, 6 dB, open | 3/3 | 3/3 |

At 20 dB and at equal rates discovery is unaffected. At rho 0.514 and 6 dB the scaled discovery is
at its margin in both cases, and the radius shows no loss beyond the scatter of a dozen captures.
The full screen's constant-modulus read normalizes each row by its own mean energy, so the radius
drops out of it exactly (`rank_leak.md`).

**Coherent 16-QAM** (24 captures × 100 rows, L=64, the informed coherent read,
gains from the keyed reference row). Mean BER.

| rho | SNR | equal-energy rows | keyed row power, read as is | keyed row power, `r_t/r_0` divided out (keyed reader) | keyed row power, `r_t` from the row's own energy |
|---|---|---|---|---|---|
| 1 | 18 dB | 3.1e-3 | 5.9e-3 | 2.3e-3 | 4.7e-3 |
| 1 | 21 dB | 1.0e-4 | 9.7e-4 | 5.7e-5 | 4.1e-4 |
| 0.514 | 18 dB | 7.5e-3 | 7.0e-3 | 3.7e-3 | 5.9e-3 |
| 0.514 | 21 dB | 1.7e-3 | 5.0e-3 | 2.6e-3 | 4.2e-3 |

- A reader that ignores the radius loses a factor 2–10. The radius is a gain error of sd ≈ 6% on
  every symbol.
- Dividing out the keyed `r_t/r_0` restores it. Here it does better than equal-energy rows,
  because this key's reference row draws `r_0^2 = 1.15`. The single reference row's SNR then sets
  the whole capture's channel estimate, and that is a property of the key rather than of the
  radius. The capture-to-capture spread at rho 0.514 is dominated by the reference row's noise in
  both cases (median and mean differ by up to 9×).
- Estimating `r_t` from the row's own energy alone (K symbols) recovers about half the loss.
  16-QAM's energy spread makes the per-row estimate itself ~3.6% in amplitude. The open reader
  does more than this (below).
- `decode` on 16-QAM captures of the K = L − 4 screen (8 per cell, 160 rows) named the payload
  coherent 2/8 and 5/8 (equal-energy rows, 24 / 18 dB) against 5/8 and 3/8 (keyed row power).
  The coherent/differential decision there is unreliable on 160-row 16-QAM captures with or
  without the radius; it is not a radius effect.

**PAPR** (per sample, 4 M transmitted samples; 1.94 M received at rho 0.514, noiseless).

| | 99.99th pct, tx | max, tx | 99.99th pct, rx 0.514 | max, rx 0.514 |
|---|---|---|---|---|
| equal-energy rows, L=64 | 9.43 dB | 12.0 | 9.43 | 10.9 |
| Gamma(L), L=64 | 9.64 | 11.8 | 9.68 | 11.9 |
| equal-energy rows, L=128 | 9.53 | 12.4 | 9.51 | 11.4 |
| Gamma(L), L=128 | 9.67 | 12.2 | 9.62 | 11.8 |
| complex Gaussian | 9.67 | 11.6 | | |

The radius moves the tail to the Gaussian's: +0.21 dB at L=64 and +0.14 dB at L=128 at the 99.99th
percentile. The maximum is within the scatter of a few-sample statistic.

## Design

1. **The wire carries a keyed per-row power `r_t^2 ~ Gamma(L, 1/L)`, on every row including the
   reference row.** It is the mean of L keyed unit exponentials (equivalently `||g||^2/(2L)` for L
   keyed complex normals), under its own domain string and seekable per 64-row block. That keeps
   it entroptics-pure and independent of the mode keys. The shape is L, not K. Lognormal and
   uniform are not used: they match only the variance. Per-mode magnitudes are not used: they are
   over-dispersed by L/K and fade every mode.
2. **Keyed, not sender-random, in keyed mode**, so a coherent reader can divide out `r_t/r_0`
   exactly. DQPSK needs nothing either way. The keyed draw also keeps the transmission a function
   of (message, key), which the codec relies on elsewhere.
3. **Open mode: sender-random Gamma(L).** A public keyed stream indexed by absolute row would put
   back the join problem the open mode does without. A spec-aware observer can read an open
   broadcast anyway, so a public radius buys nothing there. The sender-random draw passes the same
   checks (table above). Open-mode DQPSK is unaffected. Open-mode coherent QAM measures `r_t`
   from each row.
4. **Cost:** about 0.1 dB on DQPSK; no change to blind discovery; +0.14–0.21 dB PAPR at the
   99.99th percentile, to the Gaussian value; and one reader step for coherent QAM.
5. **The radius is one part of low detectability, not all of it.** It closes the equal-energy
   leak completely, and with it the only alignment-free, rate-free route to the period. At K < L
   a key-free observer would still classify the transmission through the rank deficiency, the
   fold and the spectral ripple, all of which come from K < L; the rank deficiency has no cure
   short of filling the spare dimension, which is what the shipped codec does (`rank_leak.md`).
   The energy stays visible to a radiometer in any case; what the radius and the full screen
   remove is any structure the blind battery can classify.

## The shipped codec

**The wire.** `compose`, `compose_qam` and `StreamTransmitter` send row t of the screen as
`r_t · w_t B`. `row_radius(L, seed, first=, rows=)` returns `r_t`, with `r_t^2` the mean of L unit
exponentials `-ln u` over keyed uniforms. The draw has its own domain, `entroptics-rf/row-radius/v1`,
and is seekable per 64-row block like `mode_keys`. It is indexed by the screen row, with the
reference row as row 0. A stream indexes it by the absolute screen row across buffers, so a stream
and a one-shot compose of the same rows carry the same radii. The differential walk stays unit
modulus: the radius scales what goes on the air, not the state. With `seed=None` the radius is the
sender's own draw (`rng=`, fresh entropy by default), so nothing about it is public and an open
receiver can still join anywhere.

**The reference row takes a radius.** Exempt, it would be one row of energy exactly K at the head
of every burst. An observer finds the head from the burst's onset, and over a few bursts the head
window's energy has no spread at the true width and only there, which hands over L. The mesh sends
short bursts (7-34 rows), so that would be 3-14% of its rows. The cost lands on the coherent read.
Its gains are measured on the reference row, whose SNR is `r_0^2` times the mean, so the reference
row's share of the error moves with the key: by −0.95 to +0.84 dB between the 5th and 95th
percentiles of `r_0^2` at L=64 (−0.66 to +0.61 at L=128), and by `10 log10(L/(L-1))` = 0.07 dB averaged over keys. The codec
puts no added structure ahead of gain, so it takes that trade. One caveat applies either way. Under a reused key every
burst starts at row 0, so every burst carries the same radius sequence. The reference row and the
mode keys already repeat the same way, and far more visibly; freshness is key management
(`waveform`'s nonce note).

**The readers.**
- Differential: nothing to take off. The differential is phase-only, and its magnitude carries
  `r_t r_{t+1}`. That is the symbol's own amplitude, so the mesh's magnitude-weighted soft bits
  (`fec.soft_bits_from_differential(..., weight_by_magnitude=True)`) weight a symbol by its own
  SNR, which is what they should do.
- Coherent, keyed: `_dekeyed(A, r, K, seed, carrier, L=)` multiplies row t after the reference by
  `r_0 / r_t`, exactly, from the key. `L` is a required keyword (the radius's shape), so a
  caller that omits it fails loudly instead of reading 6% gain errors. `read_qam`,
  `_coherent_read` and `decode` pass it, and so do the mesh's `_coherent_rows`, `test_fec_qam` and
  `research/mcs/mcs_coherent.py`.
- Coherent, open: `_row_levels(Y, M, n, L)` measures each row's level, in four steps. It reads
  the power and weighs it against the law (Gamma(L) about the burst's common level). It searches
  ±5 sd for the least-squares scale. It fits the row by least squares and then by soft-decision
  EM. Finally it weighs the row's fit, by its observed information, against the law again.
  `read_qam` and `decode` apply it with `seed=None`. The mesh applies it per (family, M)
  hypothesis in `RxBurst.soft_bits` (field `RxBurst.keyed`), on the frame's own rows only. It
  levels only rows already known to be payload. `RxBurst.qam` keeps the raw rows, so the
  burst-length read (the rank edge of the per-row level) is unaffected. `hypotheses()` orders the
  open lattice closures on power-levelled rows.
- Discovery is unaffected. The constant-modulus read divides each row by its own mean energy, and
  the alignment, closure, reference-row and beacon reads all normalize per symbol or per row.

### Measured on the codec path

Run on a 22-core server.

Leak detectors, equal-energy rows / keyed row power. The two arms share the same bits and noise;
the equal-energy arm is the codec with `row_radius` patched to ones. Each capture is 1 M samples
(417 ms), and z is against 10 noise-only captures. The brackets give the rank of the true width's
family among 585 integer periods.

| rho, L, mode | SNR | D1 contrast | D1 aligned variance | D2 presence |
|---|---|---|---|---|
| 1, 64, keyed | −5 dB | 5.0 [2] / 0.5 [61] | 3.0 / −1.1 | 2.5 / 1.7 |
| 1, 64, keyed | 0 dB | 24.5 [0] / −1.1 [40] | 17.5 / 0.1 | 13.0 / 2.7 |
| 1, 64, keyed | +10 dB | 163 [0] / 0.9 [19] | 61.2 / 0.6 | 46.6 / 3.0 |
| 1, 64, open | 0 / +10 dB | 26.3 / −1.4; 162 / 2.3 | 16.8 / −1.8; 61.1 / 1.3 | 12.6 / −0.4; 46.7 / 2.2 |
| 1, 128, keyed | 0 / +10 dB | 12.3 / −1.1; 96.5 / −1.2 | 10.1 / −1.6; 38.5 / −0.5 | 8.4 / 2.0; 29.1 / 0.7 |
| 1, 128, open | 0 / +10 dB | 12.3 / 0.3; 96.7 / 1.0 | 10.8 / 0.0; 38.6 / 0.5 | 8.9 / 1.1; 29.1 / 1.7 |
| 125/243, 64, keyed, band-limited | 0 / +10 dB | 32.6 / −0.7; 183 / 0.1 | 43.5 / −2.1; 86.4 / 1.0 | 38.8 / 1.9; 80.1 / 2.4 |

The noise-only leave-one-out maxima over the grid (mean / worst) are: D2 2.1 / 5.3, D1 aligned
variance 1.6 / 3.7, and D1 contrast 7.3 / 9.8 over 585 periods (1.6-1.9 at the single true period
at rho ≠ 1). With the radius every cell sits in that range, and the true width does not rank
near the top. At K = L the same detectors read noise on the codec's own transmissions at every
length and rate measured (`rank_leak.md`, "Detectability on the codec path").

PAPR, per sample over 4 M samples (99.99th percentile, and max where given):

| | equal-energy rows | keyed row power |
|---|---|---|
| L=64, keyed | 9.42 / 12.1 dB | 9.64 / 12.5 dB |
| L=64, open | 9.40 dB | 9.67 dB |
| L=128, keyed | 9.56 dB | 9.64 dB |
| L=128, open | 9.54 dB | 9.67 dB |

A complex Gaussian reads 9.65 / 12.5 dB. The peak over a 32064-sample capture (`Optics.papr_db`,
K=60, L=64, eight keys) is 9.8 dB (9.0-10.2) for equal-energy rows and 10.0 dB (9.4-10.5) with
the radius.

Coded PER. The chain is the mesh's FEC (P = 500 bytes, K=7 convolutional code, CRC-32) at L=64,
K=60, matched rate, informed read, 2000 packets a point. The keyed arms rotate over 16 keys, so a
threshold is the average over `r_0`. Thresholds are at 1% PER, interpolated in log PER.

| mode | keyed, equal-energy rows | keyed, keyed row power | open, equal-energy rows | open, sender-random row power (levels measured) |
|---|---|---|---|---|
| DQPSK 1/2 (Gray, magnitude-weighted) | < 6.5 (0.95% at 6.5) | 6.58 | | |
| 16-QAM 1/2 | 12.35 | 12.52 | 12.24 | 12.37 |
| 16-QAM 3/4 | 16.12 | 16.27 | 16.04 | 16.25 |
| 64-QAM 1/2 | 17.90 | 18.00 | 18.01 | 18.32 |
| 64-QAM 3/4 | 21.70 | 21.88 | 21.62 | 21.91 |

So the radius costs 0.1-0.2 dB keyed and 0.1-0.3 dB open. The radius-free coherent thresholds
agree with `entroptics-mesh/research/mcs/coherent.md` (m64, per-mode N0: 12.28, 16.0, 17.69 and
21.87) to within 0.2 dB. The DQPSK cell is not that table's chain: it uses Gray labels on a 1 dB
grid, where the table used natural labels on a 2 dB grid.

256-QAM, open, against a reader handed the true radius (600 packets): 2/3 at
27 dB fails 7 against 3, and 3/4 at 28.5 dB fails 2 against 0. In the mesh's own open-mode bursts
at threshold + 5 dB (eight seeds, 2.4 and 1.0 Msps), 256-QAM decodes 31 of 32 both with and
without the radius.

Simpler open readers fall short. Power alone, least squares, magnitude fixed points and grid
searches all leave 5-40× the genie's failures at 64- and 256-QAM. Two causes dominate, and the
four steps of `_row_levels` answer them:
- A frame's final row can carry a single symbol: 64-QAM 3/4 is 901 symbols on 60 modes. Levelled on
  that symbol alone, the row is confidently wrong at the frame's tail, and 72 of 600 packets fail
  against 1 for the genie. The law prior, at the start and at the finish, cures it.
- Hard decisions bias the fit where a row's symbol errors are many. The EM pass cures it.

**Tests.**
- entroptics-rf `tests/test_row_radius.py` (13 tests):
  - the radius is Gamma(L), seekable and keyed; in open mode it is the sender's own draw;
  - windowed row energy at five offsets matches white noise's: `L var/mean^2` is within 0.12 of
    1, against below 1e-6 at the row boundary without the radius;
  - the D1 contrast at the true width reads |z| < 4 against noise, where the radius-free burst
    reads z > 50;
  - blind differential round trips, keyed and open;
  - keyed coherent 64- and 1024-QAM are bit-exact, and 1024-QAM errs with the radius left on;
  - open coherent 16/64/256-QAM read informed, and 64/256 blind;
  - a stream carries the same radii as a compose.
- entroptics-rf, elsewhere:
  - `test_waveform`: the short final row's power share is compared per unit radius;
  - `test_joint_read`: passes `L=` to `_dekeyed`;
  - `test_no_preshared_parameters`: the coordinates test asserts the nearest-integer offset (a
    settle can land −1.2e-10 below a whole sample), and the white-aperture test reads 300 rows (on
    70 the coarse read misses its bracket on 1-2 of 60 draws, with or without the radius).
- entroptics-mesh: `test_phy_mcs` has `test_open_mode_levels_each_row_of_a_dense_coherent_burst`
  (64-QAM 2/3 and 3/4, 256-QAM 2/3 and 3/4, open, at matched rate and 1.0 → 2.4 Msps).
  `phy.transmit_burst(..., rng=)` makes the open radius reproducible in tests.

## Limits

- **The coherent reference row.** A single keyed reference row sets the whole capture's channel
  estimate, and with the radius its SNR is `r_0^2` times the mean, so it varies by key. The row is
  not exempt ([The shipped codec](#the-shipped-codec)). A gain read averaged over the payload rows
  would remove both this and the 2-4 dB the single reference row costs; the codec has no such
  read.
- **10 s figures are extrapolated** from 0.083–1.67 s using the measured scaling laws. The rank
  detector's growth near threshold is faster than √N and is extrapolated least.
- **Hardware.** All of this is AWGN through an ideal converter. A real front end's I/Q imbalance,
  DC offset and AGC add envelope structure of their own. Neither the observer's null nor the
  radius's whiteness is measured on bench captures (Pluto+/NESDR).
