# The rank leak, the fold, and the full screen

The codec lights every mode (K = L, K + 3 = L under a row beacon; only a message's final row may
use fewer), sends each row at a keyed power `r_t^2 ~ Gamma(L, 1/L)` (`row_radius`,
`row_energy_leak.md`), and the receiver places the geometry by a keyed constant-modulus read. This
note gives the measurements behind that design: what a spare mode (K < L) hands a key-free
observer, five ways to close it measured against each other, and what the shipped codec measures
on its own path ([The shipped codec](#the-shipped-codec)).

Low detectability, as used here, means three things. The energy is visible: a radiometer sees it,
and out-detects the decoder by about 20–28 dB (`detectability/README.md`). The structure is not
classifiable: the blind detector battery cannot tell the transmission from noise or name its
geometry. The content is unreadable without the key. The leaks measured here are of the second
kind: key-free features that classify the transmission as a structured signal and hand over its
geometry.

Every row of a screen lives in the same K-dimensional keyed subspace of C^L. When K < L, the
covariance of L-sample windows at the row offset has L − K directions holding noise only, and a
key-free observer finds them by eigen-decomposition (§RF-5.3). The same spare dimension gives two
more key-free features. The fold is the fixed per-sample power profile `diag(B^H B)` that every
row repeats. The spectral ripple is the projector's off-diagonal, which shows up as
lag-correlation lines. `row_energy_leak.md` measures all three at rho = 1 with K = L − 4, and the
rank leak is the strongest key-free tell there. This note:

1. measures the rank leak and the fold at rho ≠ 1, across occupancy (K = L−1 … L−8), SNR
   (−15…+10 dB) and capture length (83 ms … 10 s), keyed and open;
2. measures five cures, each for its detectability against every detector and its cost to the
   key holder;
3. measures the shipped one, (e), on the codec's own path.

Sections 1–3 run on a lab transmitter bit-exact with the codec's differential walk, with each cure
applied at the coordinate level. Every transmission carries the keyed per-row radius, so the
equal-energy leak is closed throughout, and what is measured is what remains.

## Summary

- **A spare mode shows the width, at any occupancy and any rate.** One spare mode (K = L − 1) is
  as visible as eight. In 10 s at L = 64 the true width stands at z 19–27 at −15 dB, and at z
  109–124 at −10 dB, for every K from L−8 to L−1. The statistic sees that some direction carries
  noise only while the average carries signal, and one such direction is enough. At rho = 0.514
  and 0.833, an observer that resamples to the sender's rate reads the leak as strongly as at
  rho = 1: z 27–42 at −10 dB from 1.67 s, against 30–41 at rho = 1. The ratio comes key-free from
  the occupied band's edge, to about 1e-4. The leak survives about 3e-5 of ratio error when it is
  read in 512-row blocks, so the rate costs the observer a search of about ten hypotheses. The
  leak is not SNR-bounded at any rate.
- **The fold scales with the empty fraction and with N.** In 10 s at L = 64 it reaches z 586 at
  0 dB for K = L − 4, and z 124 for K = L − 1. The flat complement closes it exactly, at every
  length.
- **The shipped codec lights every mode, and every second-order detector reads inside the
  noise-only range.** Measured on the codec's own transmissions, rank, fold, lag, row-energy and
  kurtosis detectors read noise at every length, rate and SNR measured. What a full screen still
  carries is fourth-order (§3).
- **The cures, ranked** (detectability first, then cost):

  | cure | rank leak | fold | ripple | 4th order | key-holder cost |
  |---|---|---|---|---|---|
  | (e) K = L, constant-modulus discovery read (shipped) | closed | closed | closed | same as K < L | none measurable: blind lock equal or better (rho 0.514, 3 dB: 11/12 vs 10/12); DQPSK ~0.3 dB per symbol for +6.7% capacity; a receiver-side read |
  | (d) keyed aperture per absolute row | closed | closed | closed | closed | the row index joins the geometry hunt: ×(T + reach) per hypothesis (0.4 s → 8–11 s per width and carrier probe at reach 4096); joining a long-running stream is unbounded |
  | (d′) the same, periodic in the row (R = 8) | not closed (per-phase z 90 at 0 dB, 1M) | weakened | — | — | ×R |
  | (b) flat complement | unchanged | closed | unchanged | same | none at rho 1; marginal at rho 0.514, 3 dB (4/12 vs 10/12, 12 captures) |
  | (c) K = L − 1 | unchanged (z 109 at −10 dB, 10 s) | ×1/4 | ×1/4 | same | none |
  | (a) filler in the dark modes | z falls only as the filler nears full power; −1 dB still z 113 at +10 dB, 1.67 s | weakened | weakened | same | breaks blind discovery from −6 dB filler (rho 0.514: 2/12 at 20 dB) |

- **The codec uses (e).** The screen is full (K = L), and the geometry is found by the per-row
  dispersion of the mode energies on the keyed aperture instead of by a dark mode's energy. Every
  second-order key-free feature measured (rank, fold, ripple, row energy, presence) then reads as
  noise at 10 s, at both rates. Blind discovery keeps its lock rate and its time at every rate.
  The aperture and the keystream are those of a K < L screen; the spec says "every mode is lit",
  and the receiver has no mode-count read. (d) is the only cure that also removes the fourth-order
  residue, and it costs the late join; the codec does not use it.

## Method

**Captures.** The lab transmitter is the codec's differential walk, bit-exact with
`wf.compose` (max deviation 1e-7, the complex64 cast). Each cure is then applied at the coordinate
level. DQPSK random payload, keyed seed 1234 unless "open" (`seed=None`, the whitened public
aperture). rho = 1 is the transmitted samples; rho = 125/243 (0.514) and 5/6 (0.833) go through
`tests/_ratechannel.at_rate`. AWGN at the stated SNR over the received band. Lengths are in
received samples at 2.4 Msps: 200k (83 ms), 1M (417 ms), 4M (1.67 s), 24M (10 s, rho = 1 only).

**The observer.** At rho ≠ 1 the observer resamples the capture to the sender's rate by Fourier
truncation, with the true ratio and sub-sample phase. This genie read
is an upper bound. How much of it a blind observer recovers is measured separately (below). The
codec does not resample; the observer may.

**Detectors** (each larger = more structure):

| | statistic |
|---|---|
| rank | `−min_o λ_min/mean λ` of the L-sample window covariance over offsets o (PAPER §RF-5.3) |
| rank1 | the same at window L+1 (off the comb, reference) |
| rankR | period-R observer: covariance per row phase (row index mod R), averaged |
| fold | D5 harmonic sum of `|x|^2` at period L, z analytic `(S−1)√H` |
| lag | max over lags 8…600 of the normalized lag correlation (the ripple) |
| d2, d1c | low-frequency deficit and D1 offset contrast (the row-energy leak; closed by the radius) |
| kurt | per-sample fourth moment `E|x|^4/(E|x|^2)^2 − 2`, alignment-free presence |
| c4 | fourth-order cumulant matrix of whitened aligned windows: the constant-modulus structure ACMA uses |

Each statistic except fold is scored as `z = (stat − mean_null)/sd_null` against noise-only
captures through the same pipeline: 16/10/6/4 nulls at 200k/1M/4M/24M at rho = 1, 10/8/5 at
rho ≠ 1. The leave-one-out noise-only max |z| is 1.5–3.6 for rank, lag and kurt in every cell (one
0.833 kurt cell 5.6). So z ≥ 7 is a detection, and z < 4 is noise.

## 1. The rank leak and the fold, measured

**Rank leak, rho = 1, keyed.** z by SNR (−15 / −10 / −5 / 0 / +5 / +10 dB).

| L | K | 83 ms | 417 ms | 1.67 s | 10 s |
|---|---|---|---|---|---|
| 64 | 56 | 0.7 / 1.2 / 17 / 53 / 97 / 121 | −0.9 / 10 / 62 / 160 / 260 / 318 | 4.5 / 47 / 172 / 394 / 619 / 748 | 27 / 124 / 367 / 799 / 1230 / 1478 |
| 64 | 60 | 0.2 / 0.0 / 12 / 51 / 95 / 121 | 0.9 / 6.7 / 58 / 156 / 259 / 317 | 2.6 / 41 / 163 / 393 / 618 / 746 | 21 / 121 / 363 / 799 / 1229 / 1477 |
| 64 | 62 | 0.9 / 1.2 / 8.6 / 46 / 93 / 120 | 0.4 / 6.7 / 56 / 157 / 259 / 317 | 0.8 / 40 / 162 / 387 / 616 / 746 | |
| 64 | 63 | −0.7 / 1.5 / 2.8 / 49 / 93 / 120 | −0.4 / 3.5 / 54 / 155 / 258 / 317 | −0.9 / 30 / 160 / 388 / 614 / 746 | 19 / 109 / 358 / 793 / 1227 / 1477 |
| 64 | 64 | | noise (\|z\| ≤ 2.3) | noise (≤ 3.1) | noise (≤ 2.2) |
| 128 | 120 | | −1.4 / 2.2 / 30 / 100 / 179 / 226 | 0.7 / 26 / 149 / 394 / 648 / 792 | 7.5 / 62 / 228 / 523 / 819 / 989 |
| 128 | 124 | 0.1 / 0.9 / 1.1 / 32 / 73 / 105 | 0.2 / 2.3 / 23 / 97 / 177 / 225 | −0.2 / 13 / 141 / 390 / 644 / 792 | 2.5 / 61 / 225 / 521 / 817 / 988 |
| 128 | 126 | | −0.6 / −0.4 / 23 / 96 / 178 / 224 | 1.9 / 15 / 131 / 386 / 641 / 792 | |
| 128 | 127 | −0.8 / 0.9 / −0.4 / 17 / 71 / 105 | 0.2 / 0.9 / 19 / 95 / 176 / 224 | −3.7 / 6.7 / 133 / 388 / 640 / 790 | 0.6 / 51 / 219 / 516 / 816 / 988 |
| 128 | 128 | | noise (≤ 1.5) | | noise (≤ 0.8) |

- **Occupancy barely matters.** From −5 dB up, every K < L reads the same z to within the noise.
  Below that, more empty modes help the observer by a factor 1.1–2. The detection limit moves by
  about 1 dB from K = L−8 to L−1. The reason is that `λ_min/mean λ` compares the quietest
  direction with the average, and one direction holding noise only is enough to make it
  `N/(N + S·K/L)`. The threshold is set by the sample covariance's edge spread, about
  `2√(L/T)` in relative power, not by L − K. At 10 s (T = 375k windows at L = 64) that edge is
  0.026, so an SNR near −16 dB, which matches z ~ 20 at −15 dB.
- **It grows faster than √N.** At L = 64, K = 60, −10 dB, z goes 0 / 6.7 / 41 / 121 over
  83 ms / 417 ms / 1.67 s / 10 s. The detection limit falls about 3 dB per 4× of length. It is
  not SNR-bounded.
- **Open = keyed** to within the noise (rank is basis-invariant): at 1M, K = 60, open reads
  −0.7 / 9.5 / 56 / 155 / 258 / 318 against keyed 0.9 / 6.7 / 58 / 156 / 259 / 317.
- **K = L is noise at every length**, including 10 s at +10 dB.

A clean capture's eigenvalue ratio, which K = L − 1 shrinks by six orders of magnitude, is not a
detectability measure: scored against its own null, one spare mode is as visible as eight, and
noise does not bound the leak at any capture length measured.

**The fold** (D5 z at period L, keyed, rho = 1; −10 / −5 / 0 / +5 / +10 dB):

| L | K | 417 ms | 10 s |
|---|---|---|---|
| 64 | 56 | 2.3 / 12 / 46 / 98 / 136 | 41 / 218 / 986 / 2185 / 3113 |
| 64 | 60 | 3.0 / 2.5 / 23 / 61 / 80 | 21 / 128 / 585 / 1368 / 1934 |
| 64 | 62 | 0.5 / 2.1 / 12 / 33 / 43 | |
| 64 | 63 | 0.9 / 1.8 / 6.6 / 13 / 18 | 5.6 / 28 / 124 / 296 / 389 |
| 128 | 124 | −0.5 / 0.5 / 5.4 / 14 / 20 | 3.0 / 25 / 99 / 223 / 316 |
| 128 | 127 | −0.6 / 1.0 / 2.7 / 1.5 / 2.2 | 0.1 / 4.9 / 19 / 50 / 65 |
| 64, 128 | L | noise | noise |

The fold goes as `N·(L−K)` and falls with L, as `row_energy_leak.md` derives. K = L − 1 cuts it
4–5× against K = L − 4 but does not close it. With the radius in, it is the best key-free handle
on the period of a K < L screen. The true width ranks first among 585 integer periods from 0 dB
at 417 ms (K = L − 4, L = 64).

**The ripple** (lag-correlation max, 10 s, L = 64, −10 / −5 / 0 / +5 / +10 dB): K = 56: 22 / 75 /
179 / 274 / 335; K = 60: 6.0 / 32 / 76 / 139 / 169; K = 63: −0.4 / 8.6 / 49 / 88 / 106. It is
presence only (lags below L), and it scales with L − K like the fold.

**rho ≠ 1 (genie observer, keyed).** Rank z, −15 / −10 / −5 / 0 / +5 / +10 dB:

| L | K | rho | 417 ms | 1.67 s |
|---|---|---|---|---|
| 64 | 60 | 0.514 | 1.1 / 14 / 78 / 163 / 226 / 256 | 6.0 / 42 / 137 / 256 / 345 / 384 |
| 64 | 63 | 0.514 | −1.0 / 8.5 / 75 / 160 / 225 / 254 | 5.4 / 39 / 130 / 256 / 343 / 384 |
| 64 | 60 | 0.833 | 1.5 / 13 / 80 / 214 / 337 / 402 | 2.9 / 31 / 123 / 273 / 415 / 489 |
| 64 | 63 | 0.833 | −1.0 / 3.9 / 75 / 208 / 334 / 400 | 2.1 / 27 / 117 / 271 / 412 / 485 |
| 128 | 124 | 0.514 | | 1.7 / 25 / 128 / 280 / 385 / 434 |
| 128 | 127 | 0.514 | | 0.7 / 19 / 124 / 275 / 381 / 433 |
| 128 | 124 | 0.833 | | −1.0 / 12 / 121 / 311 / 493 / 591 |
| 128 | 127 | 0.833 | | −2.6 / 4.0 / 115 / 305 / 487 / 585 |
| 64 | 64 | 0.514 | noise (≤ 1.8) | noise (≤ 1.4) |

The fold at 1.67 s, L = 64, rho 0.514: K = 60: 4.3 / 28 / 89 / 152 / 179 (−10 … +10 dB); K = 63:
−0.5 / 6.7 / 15 / 30 / 36. At low SNR rho < 1 helps the observer: band-limiting to the sender's
band gains 10 log(1/rho) of SNR (2.9 dB at 0.514). At high SNR it holds fewer transmitted samples
(N·rho), so z is lower. A non-unit rate does not remove the subspace.

**The blind observer at rho ≠ 1** (L = 64, K = 60, rho 0.514, 1.67 s, 6 captures
and 6 nulls per cell; median z):

- *Ratio from the band edge* (Welch PSD, half-level crossings): error 5e-5 / 8e-5 (sd 1e-5) at
  0 / +5 dB. At −5 dB the crossing read fails on some captures (sd 0.15). A better edge read or a
  coarse scan replaces it there.
- *Tolerance.* Read on the whole capture, the leak needs the ratio to about 3e-6: z 279 / 81 / 52 /
  10 at errors 0 / 3e-6 / 1e-5 / 3e-5 (0 dB). Read in 512-row blocks, each with its own best offset
  and one of four sub-sample phases, it tolerates 1e-5 without loss (z 199 → 143) and 3e-5 with
  half (99). It is gone at 1e-4 (z 10). At −5 dB the blocks read z 26 at the truth and 25 out to
  1e-5.
- So an observer that reads the edge to 1e-4 and steps 1e-5 needs about 20 hypotheses, each one
  resample plus one covariance per block. That is seconds of compute. At −5 dB, without the edge,
  a ±2% scan at 1.5e-5 is about 2600 hypotheses, which is hours of compute and a look-elsewhere of
  about √(2 ln 2600) = 4. rho ≠ 1 costs the observer compute, not detectability.

At 417 ms the picture is the same. The edge reads to 4e-5 / 9e-5 at 0 / +5 dB and fails at −5 dB
(and throws on 2 of 6 captures there). The 512-row blocks hold z 257 out to 1e-5 and 44 at 3e-5
(0 dB), and they are gone at 1e-4 (z 3).

## 2. The cures

All at rho = 1, L = 64, keyed, with the radius, unless stated. z by SNR −15 / −10 / −5 / 0 / +5 /
+10 dB. Cost columns are blind discovery with the width given, 160-row bursts,
12 captures per cell (8 at L = 128). "Lock" means width, offset, K and alphabet right, and at
rho ≠ 1 the scale within 1e-4. The informed reads are DQPSK (4 × 300–600 rows) and coherent 16-QAM
(12–60 × 100 rows, keyed radius divided out).

### (a) Low-level filler in the dark modes

Sender-random constant-modulus coordinates in the L − K spare modes, at `fill` dB per mode
relative to a data mode.

| fill | rank, 417 ms | rank, 1.67 s | fold, 1.67 s (0 / +10 dB) | dark read, noiseless | blind lock rho 1 (20 / 6 / 3 dB) | blind lock rho 0.514 (20 / 6 / 3 dB) |
|---|---|---|---|---|---|---|
| none | 0.9 / 6.7 / 58 / 156 / 259 / 317 | 2.6 / 41 / 163 / 393 / 618 / 746 | 103 / 323 | −281 dB | 12 / 12 / 12 | 12 / 12 / 10 |
| −20 dB | 0.7 / 5.2 / 56 / 154 / 256 / 314 | 0.4 / 39 / 168 / 387 / 611 / 738 | 96 / 320 | | | |
| −10 dB | −0.9 / 6.0 / 48 / 134 / 229 / 279 | 2.8 / 32 / 146 / 345 / 548 / 662 | 82 / 264 | −9.7 dB | 12 / 12 / 11 | 12 / 10 / 5 |
| −6 dB | 0.2 / 2.6 / 35 / 108 / 181 / 225 | 1.9 / 22 / 114 / 278 / 443 / 538 | | −5.8 dB | 12 / 12 / 12 | 2 / 4 / 2 |
| −3 dB | 0.5 / 1.1 / 19 / 63 / 106 / 135 | 1.2 / 7.5 / 62 / 167 / 273 / 338 | 27 / 79 | −2.9 dB | 8 / 8 / 7 | 0 / 1 / 0 |
| −1 dB | −0.7 / 0.2 / 1.5 / 8.6 / 24 / 37 | 0.3 / 1.4 / 14 / 44 / 87 / 113 | 6.7 / 14 | | | |

(L = 128 at −1 dB, 417 ms: 2.2 / 2.6 / 2.7 / 6.3 / 11.5.) The deficit the observer sees and the
one the dark read sees are the same number, `1 − 10^{fill/10}`. The key holder's advantage is that
it projects onto a known direction rather than searching the spectrum's edge, which is worth
roughly √L in that deficit. But it must lock on a burst (here 160 rows), while the observer
integrates every second the link is up. At 10 s the observer needs a deficit of about
`2√(L/T)` ≈ 0.03 at L = 64. A 160-row dark read needs about 0.3–0.4 at 6 dB. No filler level
serves both, and the measurements agree: at −3 dB the leak is still z 167 at 0 dB in 1.67 s while
the scaled blind read already fails. The informed reads do not care (BER within scatter at every
level), but discovery is what the dark mode is for. A filler the receiver could subtract would
need the filler's content, which is a pilot indexed by absolute row: the entroptics-pure
constraint rules it out, and it has the row problem of (d). Not used.

### (b) The flat complement (fold only)

The dark rows are constant-modulus (keyed diagonal × keyed DFT rows), and the data rows span their
complement (nested in order). In the receiver it is installed by replacing
`_coeffs`, so every codec read runs on it unchanged.

| | rank | fold | lag |
|---|---|---|---|
| 1.67 s | 6.5 / 40 / 167 / 393 / 618 / 747 | 1.3 / 1.2 / −1.1 / 1.2 / −0.3 / 0.3 | −1.3 / −1.4 / 0.7 / 20 / 43 / 49 |
| 10 s | 26 / 118 / 365 / 797 / 1229 / 1478 | 0.3 / −1.2 / 1.2 / −0.4 / −1.3 / −1.1 | 5.5 / 2.8 / 32 / 86 / 141 / 170 |
| 1.67 s, rho 0.514 | 3.3 / 42 / 137 / 258 / 345 / 384 | −0.6 / 0.7 / 2.4 / 1.8 / 5.4 / 5.4 | 1.4 / 6.3 / 5.6 / 24 / 27 / 34 |

The fold is closed at every length (10 s at +10 dB: −1.1, against 1934). The rank and the ripple
are untouched. Cost: DQPSK and 16-QAM unchanged (rho 1, 12 dB: 1.55e-3 vs 1.54e-3; rho 0.514,
12 dB: 1.39e-5 both). Blind lock is unchanged at rho = 1 (12 / 12 / 12). At rho 0.514 it locks
12 / 10 / 4 of 12 at 20 / 6 / 3 dB, against 12 / 12 / 10 for the codec aperture. The difference
at 3 dB (margin 1.1 against 3.5) is larger than twelve captures' scatter would make likely, and it
is not explained here: (b) is dominated by (e), which closes the fold as a side effect.

### (c) K = L − 1

The minimum the dark read needs. It does nothing to the rank leak (the table in §1), and cuts the
fold and the ripple about 4× (K = 63 against 60 at 10 s, 0 dB: fold 124 against 585, lag 49
against 76). Cost: none to discovery (rho 0.514: 12 / 11 / 9 of 12, against 12 / 12 / 10). DQPSK
is about 0.2 dB denser per symbol (rho 1, 12 dB: 2.2e-3 vs 1.5e-3). Not a cure.

### (d) Keyed variation of the dark subspace by row

**Aperiodic.** The aperture of absolute row t is a keyed Haar unitary `A_t` (lab: PCG64 from
(key, chunk); a wire version would draw it seekably like `mode_keys`). The average covariance is
then exactly white, `diag` is flat, and no fixed basis exists for a fourth-order observer.
Measured: rank, fold, lag, d2, d1c and kurt all inside the noise at 417 ms, 1.67 s and 10 s (10 s,
+10 dB: rank −0.5, fold −0.7, lag −2.5), and at rho 0.514. The fourth-order statistic below is
noise as well.

**What it costs: the receiver does not know where the dark mode is until it knows which row it
is reading.** Discovery becomes a joint hunt over (offset, absolute row). The lab runs it at
rho = 1 with the width given, as the codec's join hunt does: one matrix product per offset against
the dark modes of every candidate row, then a shifted-diagonal sum, with hypotheses compared by
significance (`(1 − score)·√rows`), not by score. Measured on 160-row bursts, 12 per cell:

| SNR | reach (rows) | lock | z true / best wrong | time per width per carrier probe | BER |
|---|---|---|---|---|---|
| 6 dB | burst only / 1024 / 4096 | 12 / 12 / 12 | 10.1 / 3.3, 3.9, 4.0 | 0.5 / 3.2 / 10.9 s | 0.0987 (base 0.098) |
| 3 dB | same | 12 / 12 / 12 | 8.5 / 3.3, 3.8, 4.0 | 0.4 / 2.6 / 8.4 s | 0.221 |
| 0 dB | same | 12 / 12 / 12 | 6.6 / 3.2, 3.7, 4.1 | 0.4 / 2.3 / 8.1 s | 0.333 |

It locks, and it finds the absolute row as part of the geometry. At 3 dB that is better than a
K = L − 4 screen's separate row read on the same kind of burst, where the full read (row and
alphabet) comes out right 6 of 12 times. The
problems are elsewhere. Cost grows as T × reach per offset, per carrier probe, per width, and per
scale at rho ≠ 1. The period map cannot fold eight rows by look-ups,
because each row has a different filter for each row hypothesis. A stream already running for a
minute (1.9 M rows at L = 64) cannot be joined blind. The best wrong hypothesis rises as
√(2 ln(hypotheses)), from 3.3 to 4.1 here, so the lock threshold rises with the reach. For
bursts that start their keystream at row 0 (the mesh's case), the burst-only reach costs about
the same as the integer discovery at rho = 1.

**Periodic (R = 8).** It is no cure. A period-R observer computes the covariance per row phase:
z_R 1.7 / 18 / 90 / 176 / 228 at 417 ms (−10 … +10 dB), 12 / 107 / 329 / 557 / 689 at 1.67 s. Even
the plain rank statistic still reads 15–67 at 0 dB. Any row-dependence the receiver can enumerate
cheaply, the observer can enumerate too.

### (e) K = L, found by a keyed constant-modulus read (shipped)

With every mode lit, the screen at the true geometry is a unitary image of unit-modulus
coordinates. The covariance is σ²I, the diagonal is flat, and the projector's off-diagonal is
zero. All three second-order features vanish by construction. What still separates the true
geometry is that its coordinates on the keyed aperture are unit modulus. At any other width,
offset, scale or carrier they are mixtures of many coordinates, and so close to complex Gaussian.
The lab read is the dispersion of each row's mode energies around that row's mean, scaled by
its Gaussian value `c_m`:

    D = mean_t mean_k (|z_tk|^2 / mean_k |z_tk|^2 − 1)^2 / c_m

It reads ~1 off the geometry and `1 − (s/(1+s))^2` on it. Normalizing by the row's own mean removes
the radius exactly. It is an energy statistic on keyed projections, like the dark read: no pilot,
no reference waveform, no transform. The lab substitutes it for every dark-mode read of a
K = L − 4 codec: the integer scan (m = 16 modes per offset), the settles and comparisons (all L
modes), and the period map (the one dark filter replaced by 16 mode filters and the per-row dark
energy by the per-row dispersion, folded by the same look-ups). With those in place that codec's
`decode` and `_discover_scaled` run unchanged. The mode count is taken as L, because the full
screen is the convention. The shipped codec reads the entropy form of the same deficit
([The codec](#the-codec)).

**Detectability** (K = L, keyed, with the radius): noise in every second-order detector, at every length
and rate measured. 10 s at rho = 1: rank |z| ≤ 2.2, fold ≤ 1.1, lag ≤ 2.1, d1c and d2 noise, kurt
≤ 3.5, at both L = 64 and L = 128. At rho 0.514, 1.67 s: rank ≤ 1.4, fold ≤ 1.6, lag ≤ 1.6.

**Cost.**

| | spare-mode screen (K = L − 4, dark read) | full screen (K = L, CM read, shipped) |
|---|---|---|
| blind lock, rho 1, L 64, width given (20 / 6 / 3 dB) | 12 / 12 / 12 geometry; full read 12 / 12 / 6 | 12 / 12 / 12; full read 12 / 12 / 6 |
| blind lock, rho 0.514, L 64 (20 / 6 / 3 dB) | 12 / 12 / 10 | 12 / 12 / 11 |
| blind lock, L 128, 6 dB, rho 1 / 0.514 | 8 / 8, 6 / 8 | 8 / 8, 8 / 8 |
| blind width (2…256), rho 1, 10 dB | 4 / 4, 1.8 s | 4 / 4, 3.2 s |
| discovery time, rho 0.514, width given | 37–51 s | 27–28 s (machine shared; not a speed claim) |
| margin (runner-up / score), 20 dB | 50 (rho 1), 135 (rho 0.514) | 38, 81 |
| DQPSK BER, rho 1, 6 / 9 / 12 dB | 0.098 / 0.023 / 1.5e-3 | 0.108 / 0.028 / 2.3e-3 |
| DQPSK BER, rho 0.514, 6 / 9 / 12 dB | 0.029 / 2.3e-3 / 1.4e-5 | 0.033 / 3.3e-3 / 4.2e-5 |
| 16-QAM, rho 0.514, 18 / 21 dB (60 captures each) | 5.2e-3 / 2.8e-3 | 5.3e-3 / 3.5e-3 |
| bits per row | 2(L−4) | 2L (+6.7% at L = 64, +3.2% at 128) |

The DQPSK difference is the per-symbol SNR. The same power is spread over L symbols instead of
L − 4, which is 0.28 dB at L = 64. 16-QAM at K = L and +0.28 dB reads 5.2e-3, the same as K = L − 4
at 18 dB. The CM read is a ratio, so it gets no worse with the radius. Its value at the true
geometry, noiseless, is 0.000 (the dark read's analogue is −281 dB). With noise it is the
`1 − (s/(1+s))^2` predicted: 0.37 / 0.22 / 0.12 at 6 / 9 / 12 dB, rho 1. At low SNR the CM read's
evidence is second-order in s/(1+s) per mode, where the dark read's is first-order in one mode.
Using 16 modes in the scan and all L in the comparisons makes up the difference down to 3 dB.
Below that, where both fail on 160 rows, the CM read would need more rows sooner.

**How the receiver finds the geometry blind.** The search is the dark-mode receiver's, with one
substitution. Widths are scanned and offsets slid. The scale is read per width by the period map,
the leads are settled by Powell and compared on all rows, and the alphabet and row are read by the
keyed closure. The criterion "least energy in the last mode" becomes "least dispersion of the
row's mode energies". Every step of `discover`, `_discover_at`, `_best_of`, `_refine_continuous`,
`_discover_scaled`, `_period_read`, `_settle_scaled` and the joint settle runs unchanged on it.
Per hypothesis the scan costs m mode filters instead of one: 16 in the lab, about 1.8× the blind
integer width scan in wall time.

## 3. What remains: the fourth-order structure

Every variant here except (d) is a constant-modulus source through a fixed unitary mixture
(PAPER §RF-5.4), whatever K is. Given the width and the offset, the fourth-order cumulant matrix of
whitened windows has L eigenvalues near −(s/(1+s))^2 where noise has none. This is the structure
ACMA inverts, and it is the only key-free handle (e) leaves.

Measured with the width and the row offset given (genie), at the aligned offset ("aligned") and at
offset L/2 ("half"), z against 5 noise-only captures (3 at L = 64), median of 5 (3) captures:

| L | K | cure | 417 ms: −5 / 0 / +10 dB | 1.67 s: −5 / 0 / +10 dB |
|---|---|---|---|---|
| 32 | 32 | full | 1.1 / 154 / 702 (half: −0.1 / 3.2 / 59) | 3.9 / 508 / 1305 (half: 0.5 / 14 / 402) |
| 32 | 28 | base | 0.5 / 108 / 981 | 9.2 / 301 / 2685 |
| 32 | 28 | rowkey | −1.0 / 0.1 / −0.7 | −0.3 / 0.3 / 0.9 |
| 64 | 64 | full | — / 1.3 / 558 (half: — / 0.8 / 3.7) | |

Against the rank leak in the same capture (L = 64, 417 ms: z 58 at −5 dB and 6.7 at −10 dB), the
fourth-order read at L = 64 is noise at 0 dB and detects only at +10 dB. At L = 32 it reaches 0 dB.
The width doubles the fourth-order observer's threshold by about 10 dB, as the L² window count
predicts. The half-offset read is much weaker but not zero at high SNR, so the offset scan can be
coarse. Aperiodic (d) leaves nothing (|z| ≤ 2.5 in every cell).

So (e) moves the key-free observer from a second-order read to a fourth-order one, where:
- the evidence scales as (s/(1+s))^4 instead of (s/(1+s))^2;
- the statistic needs about L² samples' worth of windows per hypothesis instead of L;
- each (width, offset) hypothesis costs `T·L^4` instead of `T·L^2`, plus an L²×L² eigenproblem.

At L = 64 and 10 s that is about 6e12 multiply-adds per hypothesis, and 37 000 hypotheses for
585 widths × 64 offsets. It is not out of reach of a determined observer with long captures at
positive SNR, and a K < L screen carries the same residue. Only (d) removes it.

## Design decisions

1. **The codec uses (e): the full screen, K = L, with discovery by the constant-modulus read.** It
   closes the rank leak, the fold and the ripple at once and exactly, at every rate. It needs
   nothing of the aperture, the keystream, the radius or the reference row that a K < L screen
   does not already have. Blind discovery keeps its lock rate and its time. It buys 6.7% (L = 64)
   or 3.2% (L = 128) more bits per row for about 0.3 dB per symbol.
2. **What (e) means for the wire and the spec.**
   - The sender always uses all L modes. Only a message's final row may use fewer (nested, first
     k modes).
   - No mode is dark. What makes a screen findable is that its coordinates on the keyed aperture
     are unit modulus, not that a mode is empty.
   - The receiver has no mode-count read. On a flat profile an energy-gap read returns L − 1 with
     margin ~2, where a real edge has ≥ 14.6, so it must not run. K = L is the convention, and no
     sender uses fewer modes.
   - The receiver's scan, settles, comparisons and period map read the constant-modulus deficit.
     The mesh's carrier re-read and its open-mode reads have no unused modes to read either, and
     use the same substitution ([The mesh on it](#the-mesh-on-it)).
   - Open mode: the same. The public aperture is whitened, so the CM read is the same read with
     `seed=None`. (Open broadcasts are readable by anyone holding the spec in any case.)
   - Coherent 16-QAM is unaffected. The reference row and the keyed `r_t/r_0` division are those
     of a K < L screen.
3. **The per-row radius stays.** At K = L the row energy is exactly `L·r_t^2`, so without the
   radius the equal-energy leak (D1/D2) would return.
4. **(a), (b), (c) and periodic (d) are not used.** (a) cannot satisfy observer and receiver at
   once. (b) and (c) leave the rank leak untouched. Periodic (d) is visible to a per-phase
   observer.
5. **Aperiodic (d) is not used.** It is the only cure that also removes the fourth-order structure
   (and with it ACMA's coefficient recovery, PAPER §RF-5.4). It costs the blind late join, and it
   multiplies discovery by the row reach. For bursts whose keystream starts at row 0 the cost is
   small at rho = 1. It composes with (e): a keyed per-row unitary at K = L with a CM read over
   (offset, row).

## Limits of these measurements

- **Hardware.** All of this is AWGN through an ideal converter. On hardware, matched-rate filter
  edges mix modes (dark −13.5 dB). They make the CM read at the true geometry
  weaker, the same kind of weakening the dark read suffers. The observer's null would also carry
  front-end structure.
- **The fourth-order observer** is measured with the width and the offset given. A practical
  scan over (width, offset), and better fourth-order statistics than the cumulant spectrum
  (e.g. ACMA's own residual), are not measured.
- **10 s at rho ≠ 1** is not run (the codec-path detectors run 10 s at rho = 1 and 1.67 s at
  0.514 and 0.833; every trend matches rho = 1's).

## The shipped codec

What the codec does under (e), and what it measures on its own path rather than on the lab's
coordinate-level rebuild. Comparisons are against a spare-mode screen: K = L − 4 with the dark
read.

### The codec

- **Writer.** `compose` and `compose_qam` refuse any K but L (with a row beacon, K + 3 = L);
  `runner.Link` the same; the mesh's `TxPlan.K` defaults to L. Only a message's final row uses
  fewer modes. The aperture, keystream, row radius, reference row, beacon and stream are a
  spare-mode screen's.
- **The read.** Each row's Shannon entropy over its mode energies against log2 m, over the flat
  Dirichlet's mean deficit — the library's own `entropy.shannon_bits` and
  `entropy.dirichlet_entropy_moments` (exact), not the lab's dispersion (the second-order form of
  the same deficit). On 160 rows of unit-modulus against Gaussian coordinates the entropy separates
  as well or better: z 20.0 / 11.5 / 5.6 / 2.2 at 6 / 3 / 0 / −3 dB on 16 modes against 16.6 /
  10.3 / 5.1 / 2.0 for the dispersion; 42.8 / 24.2 / 11.7 / 4.3 against 32.0 / 19.8 / 10.7 / 4.6
  on 64. It is the read throughout: the integer scan (`_cm_score`, 16 modes, single
  precision), the settles and comparisons (`_cm_at`, `_scaled_cm`, all L), the scaled scan, and
  the period map and its refinement (16 mode filters per phase; the fold is a mean). The
  codec has no mode-count read and no mode-leakage read;
  `constant_modulus_read` is the public evidence read at a given geometry. `Discovery.K` stays,
  always L (L − 3 with a beacon), because readers need the payload's width.
- **Two rules the codec's read carries beyond the lab's.**
  1. *Widths compare by significance, not score.* The read's null is 1 at every width but its
     spread is not: 0.97 per row at 2 modes, 0.32 at 16, 0.16 at 64 (the library's exact sd).
     Compared by score, the fully blind integer read (widths 2 to 102) locks a 6 dB and a 3 dB
     L = 64 capture on width 2 at 0.98. Hypotheses are therefore ranked by
     `(1 − score) √rows / spread(m)` — in the integer scan, in `_best_of`, and in the scaled
     shortlist and final choice — and both lock on 64.
  2. *A row reads at most 1 + 3 spread(m).* Evidence is a row falling below the null; a row far
     above it — energy on a few modes — is no more evidence against a geometry than a Gaussian
     row. A message's own final row is one: on 7 lit modes of 64 it reads 5.3, which uncapped on
     a 14-row burst weighs as much as five Gaussian rows and costs matched 256-QAM bursts their
     true start in the mesh. The cap moves 0.1% of the null's rows.
- **Dense QAM is weaker evidence than PSK.** Noiseless, a row of 16 / 64 / 256-QAM reads 0.44 /
  0.50 / 0.52 (PSK 0). The codec's 16-mode scans are validated on DQPSK; `_scaled_scan`
  takes a `modes` argument for a caller that wants more. The mesh's known-geometry scan reads up
  to 16 rows, where a dark-mode scan reads 8 (32 bursts, with the cap above): matched 256-QAM 3/4 at 28 dB delivers 27 frames on
  8 rows, 29 on 16, and 29 on 16 rows of all 64 modes at 2.6x the CPU; 2/3 at 30 dB delivers 31
  on 16 rows and 32 on all modes. ~100 ms read CPU a burst.
- **Runner.** `receive` and `StreamReceiver.feed` read the payload at the lock's K (a capture of
  noise locks a width of its own, narrower than the link's L). `StreamReceiver.acquire` narrows to
  a width only when both agreeing reads held `MIN_ROWS` rows of it: on the CM read the first 64 and
  128 samples of a stream cut 63 samples in both read L = 16, and narrowing on that would pin it.

### Detectability on the codec path

The codec's `compose` (DQPSK, keyed row radius) through the same channel, observer and
statistics as above, z against the same noise-only nulls. Every
second-order detector reads noise at every cell: L = 64 and 128 at rho = 1 for 417 ms, 1.67 s
and 10 s; L = 64 at rho 0.514 and 0.833 for 417 ms and 1.67 s; open at 417 ms; SNR −15 … +10 dB.
Rank |z| ≤ 4.4, fold ≤ 2.4, lag ≤ 5.6, D1c ≤ 3.3, D2 ≤ 6.1, kurt ≤ 1.8. The two cells above the
nulls' leave-one-out range (lag 5.6 at L = 64, 10 s, +10 dB; D2 6.1 at L = 128, 10 s, +5 dB) are
single cells that do not grow with SNR or length; at the next SNR they read 1.0 and −0.7. A
spare-mode screen (K = L − 4) on the same path reads rank z 157 / 391 at 0 dB (417 ms / 1.67 s,
rho 1), 138 at rho 0.514; fold 22 / 82; lag 5.4 / 31 — the lab's figures.

### Lock and cost

DQPSK, 12 captures a cell (geometry / full read; full = width, offset or scale, K and
M right). "Spare-mode" is K = L − 4 with the dark read; "full screen" is K = L with the CM read,
shipped.

| channel | design | rows | +6 dB | +3 dB | 0 dB | −3 dB |
|---|---|---|---|---|---|---|
| rho 1, width given | spare-mode | 160 | 12 / 12 | 11 / 5 | 9 / 1 | 4 / 0 |
| rho 1, width given | full screen | 160 | 12 / 12 | 11 / 7 | 12 / 4 | 4 / 2 |
| rho 1 | spare-mode | 640 | 12 / 12 | 11 / 11 | 10 / 2 | 8 / 0 |
| rho 1 | full screen | 640 | 12 / 12 | 11 / 10 | 8 / 2 | 7 / 3 |
| rho 0.514, width given | spare-mode | 160 | 11 / 11 | 7 / 7 | 2 / 1 | 0 / 0 |
| rho 0.514 | full screen | 160 | 12 / 12 | 12 / 12 | 2 / 1 | 0 / 0 |
| rho 0.514 | spare-mode | 640 | 12 / 12 | 3 / 3 | 3 / 2 | 0 / 0 |
| rho 0.514 | full screen | 640 | 12 / 12 | 12 / 12 | 7 / 5 | 1 / 0 |

Below 3 dB the CM read holds the geometry where the dark read does not at rho ≠ 1 (0 dB, 640
rows: 7/12 against 3/12) and as often at rho = 1; −3 dB is out of reach for both. Where the
geometry locks, the payload BER is the modulation's (0.11 at 6 dB, 0.23 at 3 dB at rho = 1).

**Blind width at rho ≠ 1** (`_discover_scaled(y, key, None)`, 160 rows, 4 captures a cell): L = 32
and 64 at rho 0.514 and 0.833, 20 and 10 dB — 4/4 in every cell, both designs, BER equal within
scatter (≤ 0.012 at 10 dB, 0 at 20). It costs 1.4–3.3× the dark map: 19–81 s against 9–41 s
(median per capture on a 22-core server, 18 jobs sharing it). The 16 mode filters per phase are the cost.
**Blind integer** (`decode(y, key)`, widths 2–102, 300 rows): 6/6 geometry at 12 / 6 / 3 dB in
both designs, 3.7 / 4.3 / 8.2 s against 2.1 / 3.3 / 7.5 s. With the width given at rho = 1 the
read costs the same (0.5–3.3 s against 0.5–2.3 s).

### What a full screen costs the link

Every symbol carries L/K less power: 0.28 dB at L = 64 against K = 60. 500-byte
frames through the informed read (the codec's read at the true geometry, fec soft bits, Viterbi,
CRC), matched (m64) and rho 0.514 (r64), PER:

| | m64, spare-mode (K = 60) | m64, full screen (shipped) | r64, spare-mode (K = 60) | r64, full screen (shipped) |
|---|---|---|---|---|
| DQPSK 1/2, 6 dB | 0.067 | 0.163 | 0.14 | 0.29 |
| DQPSK 1/2, 7 dB | 0.003 | 0.003 | 0.003 | 0.007 |
| 16-QAM 1/2, 11 dB | 0.32 | 0.50 | 0.47 | 0.46 |
| 16-QAM 1/2, 12 dB | 0.043 | 0.043 | 0.067 | 0.083 |
| 64-QAM 3/4, 21 dB | 0.050 | 0.133 | 0.097 | 0.147 |
| 64-QAM 3/4, 22 dB | 0.003 | 0.017 | 0.020 | 0.027 |

The 1%-PER thresholds sit about 0.3–0.5 dB higher on the full screen; above them goodput is 4–7%
higher (DQPSK 1/2 at 8 dB 0.906 against 0.962 b/sample, 64-QAM 3/4 at 22 dB 3.66 against 3.84,
m64). The mesh planner's thresholds (`adapt`) are measured at K = L through the mesh fast path
(entroptics-mesh `research/mcs/kL.md`).

### The mesh on it

No mesh read uses a spare mode. The known-geometry scan and settle use the CM read. The
start/carrier re-read reads the CM spread of the burst's keyed reference row — the first row of
either family, unit modulus on every mode — which tracks the coherent residual (on a 256-QAM
burst: 0.0022 at the residual's minimum, −28.3 dB, against 0.0142 / −18.2 dB with the carrier
0.05 off); the carrier and the start are picked alternately on it, since one row reads both. A
coherent burst reads that one row and no more: adding the payload rows that follow it takes
matched 256-QAM 3/4 at 28 dB from 29 to 13 (2 rows) and 17 (4 rows) of 32 bursts, their own QAM
levels outweighing the geometry. A differential burst is the opposite case -- every row unit
modulus, and its carrier branches only 1/M apart -- and on the reference row alone D16PSK 2/3 at
30 dB (1.0 → 2.4 Msps) takes the branch 1/16 off on 4 of 16 bursts and DQPSK 1/2 at 20 dB fails 1
of 32. So the re-read follows the family the closures favour: a differential burst reads its
first 16 rows and chooses among its own carrier branches, a coherent burst its reference row
among the fourth power's (0/16, 0/16 and 0/32 failures with this rule). The open mode has no K
estimate (K = L). The per-row residual turn is a non-decision-directed read, `_row_turn`: per
row, the fourth moment weighted toward the outer points, `−Σ|y|⁸(y/|y|)⁴`, a slope through the
reference. A decision-directed turn is not used. Frames delivered of 32 (the shipped 16-row
scan), 256-QAM 3/4 at 26 / 27 / 28 / 29 / 30 dB:

| | matched | 1.0 → 2.4 Msps |
|---|---|---|
| no residual-turn read | 10 / 20 / 25 / 29 / 30 | 6 / 15 / 23 / 26 / 27 |
| decision-directed (not used) | 15 / 25 / 31 / 32 / 32 | 13 / 23 / 32 / 32 / 32 |
| `_row_turn` (shipped) | 18 / 27 / 29 / 32 / 32 | 12 / 23 / 31 / 32 / 32 |

256-QAM 2/3: 29–32 of 32 with no read; with either read 31/32 matched (one burst lost to the scan) and 32/32 at 1.0 → 2.4 Msps from 27 dB. The
planner's 256-QAM 3/4 threshold, measured at K = L and the fast-path read (entroptics-mesh
`research/mcs/kL.md`), sits at 26.3 dB (`adapt.MODES`).

### Bench captures from a spare-mode sender

The bench's DQPSK rate-pair and coherent-QAM captures come from a K = 60 sender (60 of 64 modes,
no row radius). Read offline by the shipped receiver, they test how it meets a sender that leaves
modes dark; they are not captures of the shipped waveform, and nothing is transmitted.

- **DQPSK, eight rate pairs** (`_discover_scaled`, widths 63–65, samples 60k–176k): the width is
  64 every time and the scale lands within 0.5–1.6 ppm of the rate ratio (the known +0.75 ppm
  clock offset), as the dark read does. The CM read still finds a geometry with four dark modes
  (score 0.16–0.30, margin 3.1–5.8). Decoded as a full screen the payload is misframed — 60
  symbols a row read as 64 — and the receiver says so: closure 0.005–0.012, 1.7–4× its 1/√(TK) null and under the
  reader's 8× (`runner.ACQUIRE_SIGNIFICANCE`), so it is declined (two captures even read as beaconed, K = 61, on nothing).
  Read at K = 60 at the geometry the shipped receiver measures, the payload reads as the
  dark-mode receiver reads it: BER 0 on 5 pairs, 2e-4 (1.92 → 2.048), 0.011 and 0.009 at matched
  2.048 and 2.4.
- **Coherent 4 / 16 / 64-QAM at matched 2.4 Msps**: width 64 found, score 0.58–0.70, margin
  1.1–1.5; the payload is not named (read as differential M=2 twice, 4-QAM once), closure
  0.004–0.007, and the decode is BER 0.50 — declined. An informed K = 60 read at that geometry
  reads 0.08 / 0.11 / 0.19 (the dark-mode receiver reads 0.08–0.11 for 4-QAM on this bench; these
  captures also lack the row radius the keyed coherent read divides out).


### Blind decode on the current waveform

Simulated bench captures (the bench's
own transmit buffers through `tests/_ratechannel.py`, +0.75 ppm, +1.24 kHz, 22 dB, 8-bit, DC
removed; nothing transmitted) put the full screen's blind decode through three conditions, each of
which the receiver handles below. Without that handling 4 of 8 DQPSK rate pairs read the wrong
alphabet or a false beacon, and blind 16- and 64-QAM read as differential or as 4-QAM (BER ~0.5).
None of the three is the alphabet, beacon or order read itself, and none is the row radius:
every closure reads unit phasors, so the radius drops out exactly.

1. **A join past the reach.** The bench sends a 9001-row cyclic buffer, and a capture starts
   anywhere in it. The reach is `JOIN_SEARCH_ROWS` = 4096. Four of the pairs begin at
   keystream rows 6624–8330, so at that reach the alignment settles on a chance row and the
   closure sits near its null at 1.7–5.7x. The reader's 8x cutoff declines that, as designed. The
   "alphabet" and the beacon are then reads of that chance row. 0.8 → 2.4 Msps decodes at BER 0
   there only because the bench's own BER read is cycle-aware. With the reach set to the buffer's
   cycle, every pair reads M = 4 at 132–309x its null. The reach is the deployment's to set, and
   the bench's transmitter runs one cycle, so the bench analysis passes the cycle.
   The beacon hypothesis is weighed against its own null (`_alphabet` multiplies the closure by
   sqrt(K)). With three fewer modes, a chance closure reads 2.4% higher at L = 64. The weighting
   keeps the two hypotheses fair, and it only decides captures that neither hypothesis explains.
2. **At equal rates, the row read has a floor wherever the receiver's samples fall between the
   sender's.** The row read models a row as periodic over its own L samples. At rho = 1 the
   neighbouring rows' converter tails reach every sample unless the two clocks sample the same
   instants, and the joint read cannot take over because there are no spare samples. Noiseless
   16-QAM at the true geometry, with the per-mode gain, reads −148 dB on the sender's instants and
   −21.7 / −14.8 / −12.4 dB a tenth / quarter / half a sample off. Two clocks drift through every
   fraction (at 0.75 ppm and 2.4 Msps, once every 0.6 s), so on the air this is the common case.
   It is what defeats dense QAM on the row read: a 16-QAM row reads ~0.45 of the constant-modulus
   null on the sender's instants and 0.84–0.94 on the simulated window, where a Gaussian row reads
   1. The mesh's emulator puts matched-rate bursts on whole samples (entroptics-mesh
   `emu/air.py`), so its matched tables do not include this floor.

   **The band read** (`_band_coords`). Each row is read on a window of JOINT_REACH rows
   either side, against every transmitted sample's band-limited continuation,
   `s_l = rho sum_n x[n] sinc(rho (n - s) - l)`, and then projected onto its modes. At rho <= 1
   those sincs are orthogonal on the receiver's samples, so this is the least-squares read with the
   other rows in the model, truncated to the window. The joint read uses it wherever its own
   window has no room: rho within ~2.5% of 1 at L = 64, and a hair above 1. Noiseless 16-QAM with
   one gain reads −35.0 / −31.3 dB a quarter / half a sample off (row read −16.5 / −14.0), −37.3
   at rho 0.98 (row read −17.9), and exactly on the sender's instants. Like every read, it is a
   projection onto the modes' waveforms on the air: no filter shapes the capture and nothing
   resamples it.

   The integer screen (`decode`) assumes coincident samples. Its aperture is the one-sided
   one, which is the band-limited continuation only on whole samples. So `decode_scaled` is
   the blind receiver for two clocks at one nominal rate: the geometry comes from
   `_discover_scaled`, then the same payload read follows. `_discover_scaled` never returns a scale
   of exactly 1, because that would switch every reader to the integer screen.
3. **Dense QAM at the scaled search's eight-row fold.** On 16 modes, a 16-QAM row reads
   ~0.5 of the null and scatters nearly as widely as a Gaussian row. In one window the true fold
   reads 0.68 and ranks 35th of 56 pooled (best chance fold 0.585), so it would not reach the
   32-row refinement, where it reads 0.62 against 0.78. So equal rates are read as their own
   hypothesis (`_equal_rates`). At scale 1 the period is fixed, so every quarter-sample start and
   carrier probe is read at the refinement's depth (`_scaled_scan`, 32 rows) and joins the
   candidates.

Two further reads go with the band read:

- **One gain** (`_dekeyed`). Every mode is divided by the mean of the reference row's per-mode
  reads, as the mesh's fast path does (`phy._common_gain` is this read). On the band
  read at equal rates, per-mode gains read −27.0 dB half a sample off and −39.2 dB a tenth off;
  one gain reads −31.3 and −42.5 dB.
- **The scale may pass 1** (`_settle_scaled`). The joint settle on every row may move the
  scale past 1. Held at 1, a sender +0.75 ppm fast bows the DQPSK closure from 0.945 to 0.906
  toward the ends of an 1810-row capture. Settled free, the matched pairs read +0.75 / +0.76 ppm.
  The row-read settles keep the cap at 1, which guards against a +59 ppm wander.

**Simulated bench** (widths 63–65; ratio error, alphabet, closure x null where 8 is the reader's
cutoff, BER):

| TX → RX | without the handling above | shipped receiver |
|---|---|---|
| 0.8 → 2.4 | +0.74 ppm, M=4, 5.7 declined, BER 0 | +0.74 ppm, M=4, 334, BER 0 |
| 1.2 → 2.4 | beacon K=61 M=8, 1.8 declined, misread | +0.76 ppm, M=4, 14.6, BER 0 |
| 1.23457 → 2.4 | +0.75 ppm, M=4, 330, BER 0 | same |
| 1.23457 → 2.048 | +0.75 ppm, M=4, 329, BER 0 | same |
| 2 → 2.4 | M=8, 1.7 declined, BER 0.50 | +0.76 ppm, M=4, 145, BER 0 |
| 1.92 → 2.048 | M=8, 3.2 declined, BER 0.50 | +0.85 ppm, M=4, 322, BER 0 (worst 15k segment 1e-4) |
| 2.4 → 2.4 | 0 ppm (capped), M=4, 303, BER 0 | +0.76 ppm, M=4, 322, BER 0 |
| 2.048 → 2.048 | 0 ppm (capped), M=2, 3.1 declined, BER 0.50 | +0.75 ppm, M=4, 321, BER 0 |

At 1.2 → 2.4 and 2 → 2.4 the window spans the buffer's wrap, and the alignment can hold only one
side of it (14.6x and 145x). That is a bench artefact.

Coherent QAM at matched 2.4 Msps, three one-burst windows each, through the integer `decode` and
through `decode_scaled`:

- 4-QAM: `decode` coherent, M=4, BER 0.0043 / 1e-4 / 0; `decode_scaled` 0 / 0 / 0.
- 16-QAM: `decode` differential (M 2 / 8 / 8), BER 0.49; `decode_scaled` coherent, M=16, BER 0 / 0 / 0.
- 64-QAM: `decode` differential or 4-QAM, BER 0.49; `decode_scaled` coherent, M=64, BER 0.0037 /
  0.0036 / 0.0031. By theory, uncoded 64-QAM at 22 dB with 8-bit samples gives ~0.002.

The family and order reads are the same in both. On a clean read the lattice closures separate:
16-QAM reads 0.81–0.83 against 0.14–0.15 on the 4-QAM rung, and 64-QAM reads 0.45–0.48 against
0.13–0.14. None of this touches detectability: the detectors read the air, not the receiver.
The analysis runs in 97 s wall on a 22-core server with 8 jobs.

Tests: `tests/test_blind_equal_rates.py` covers the band read's floor and exactness, the one
gain, `decode_scaled` on 16 / 64 / 256-QAM off the sender's grid at 24 / 28 / 32 dB, and the
joint settle above 1. `test_joint_read.py` pins the band read where the joint read has no
room. On those short (60-row) dense bursts, a scaled read without the band read gives 16-QAM at
BER 0.039 and reads 64- and 256-QAM as 4-QAM, and the integer `decode` reads all three as
differential or the wrong order.

Limits:

- The integer `decode` assumes coincident samples, by construction. `decode_scaled` is the
  receiver for two clocks, at a cost of seconds per width.
- A 60-row dense burst at equal rates has a noiseless floor of ~−26 dB even at the true geometry
  (−31 to −35 dB mid-stream), so 256-QAM at 32 dB decodes there at BER 0.0027.
- The order read gives way where a dense lattice's closure falls under the coarsest rung's ~0.14:
  256-QAM below ~24 dB, where its uncoded BER is already several percent.
- The mesh's matched-rate tables are measured on whole-sample timing, so for two real clocks they
  are optimistic by the floor above. Its fast path reads matched rows with the band read too.
