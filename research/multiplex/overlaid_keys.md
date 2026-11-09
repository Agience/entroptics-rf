# Co-channel keyed multiplexing: two overlaid screens, one plain receiver

**Question.** Can two independently-keyed entroptics screens, linearly *superposed* in the
same band at the same time (co-channel), each be recovered by a receiver that holds both keys
using ONLY the existing blind read (`decode` / `read_qam`)? This is the entroptics-pure
analogue of spatial multiplexing: the hypothesis is that two different keys give
quasi-orthogonal keyed apertures, so a read keyed to A sees B as noise-like interference. If
both decode at a **net aggregate > 1 stream for the same total transmit power**, keyed-aperture
division is a real capacity lever.

**Verdict: NO-GO.** Keyed-aperture division does not multiplex. Two co-channel screens at
equal total power deliver **fewer** good bits than one screen, not more. The mechanism half of
the hypothesis is true -- a wrong-key read of a screen *is* noise-like -- but that is exactly
why it fails: the other screen lands on the keyed modes as interference at its **full power**,
with **no processing gain**, so per-stream SINR is just the power ratio p_other/p_self. A fixed
total power cannot make both ratios large. On top of that, below ~9 dB SINR the **blind
acquisition itself cannot lock** through the co-channel screen, so even a lopsided split buys
nothing useful. Sim only; no hardware, no transmit.

## Method

- `compose(random_bits(n, seed), K=L, L, seed=KEY, M)` builds each screen; K == L (every mode
  lit). Payloads fill exactly 100 rows so both screens have identical sample length.
- Superpose `iq = sqrt(pA)*iqA + sqrt(pB)*iqB` with `pA + pB = 1` (= the unit-power 1-stream
  baseline). Optional complex AWGN at a *total* SNR against the combined power, using the
  library's own convention (`LoopbackRadio`: noise power = mean|iq|^2 / 10^(SNR/10)).
- Read each stream with the plain single-key blind receiver:
  `decode(iq, KEY, candidates=range(60,69))`. Nothing pre-shared but the keys. No pilots,
  preamble, equalizer, channel estimation, FFT, resampling, cancellation, or joint ML.
- `read(...)` (informed, true-geometry, still single-key, still no cancellation) and
  `constant_modulus_read(...)` are used **only as diagnostics** to split "the blind lock
  failed" from "the SINR was too low", and to show the mechanism.
- 3 independent (key, bit-seed) trials per cell; reported BER is the mean. L = 64, M = 4
  (DQPSK) unless stated. Baseline = one screen alone at the full total power, same SNR.

Re-run: `OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=src .venv/Scripts/python.exe
research/multiplex/overlaid_keys.py`. Script: `research/multiplex/overlaid_keys.py`.

## Results

### 1. Blind decode of both co-channel streams (L=64, M=4 DQPSK, 100 rows)

`SINR_A` = 10·log10(pA/pB) is stream A's signal-to-interference ratio from the power split
alone. `base1` is one screen at full power at the same SNR. "good" = streams below 1e-2 BER.

| pA/pB | SINR_A dB | SNR | BER_A | BER_B | good | base1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.5/0.5 | 0.0 | clean | 0.504 | 0.495 | 0 | 0.000 |
| 0.5/0.5 | 0.0 | 20 | 0.499 | 0.496 | 0 | 0.000 |
| 0.5/0.5 | 0.0 | 15 | 0.501 | 0.500 | 0 | 0.000 |
| 0.5/0.5 | 0.0 | 12 | 0.499 | 0.501 | 0 | 0.003 |
| 0.5/0.5 | 0.0 | 9 | 0.503 | 0.499 | 0 | 0.027 |
| 0.7/0.3 | 3.7 | clean | 0.303 | 0.496 | 0 | 0.000 |
| 0.7/0.3 | 3.7 | 20 | 0.304 | 0.504 | 0 | 0.000 |
| 0.7/0.3 | 3.7 | 15 | 0.500 | 0.503 | 0 | 0.000 |
| 0.7/0.3 | 3.7 | 12 | 0.502 | 0.501 | 0 | 0.003 |
| 0.7/0.3 | 3.7 | 9 | 0.499 | 0.503 | 0 | 0.027 |
| 0.9/0.1 | 9.5 | clean | 0.021 | 0.505 | 0 | 0.000 |
| 0.9/0.1 | 9.5 | 20 | 0.027 | 0.499 | 0 | 0.000 |
| 0.9/0.1 | 9.5 | 15 | 0.042 | 0.504 | 0 | 0.000 |
| 0.9/0.1 | 9.5 | 12 | 0.065 | 0.500 | 0 | 0.003 |
| 0.9/0.1 | 9.5 | 9 | 0.113 | 0.501 | 0 | 0.027 |

- **Equal split (0.5/0.5): total collapse.** Both streams read ~0.5 (a coin flip) at every
  SNR, clean included. No separation whatsoever.
- **No split anywhere delivers two streams.** The weaker stream is dead (~0.5) at every split.
  The strong stream only begins to read when it has ~9.5 dB over the other (0.9/0.1), and even
  then at BER 0.021 clean -- degraded, and its partner carries nothing.
- **Aggregate is < 1 stream, always.** The best cell is 0.9/0.1 clean: one degraded stream
  (0.021) + one dead stream, versus the baseline's single clean stream at BER 0.000. Two
  screens at equal total power deliver *fewer* good bits than one. `good` = 0 in every cell.

### 2. The mechanism: the apertures ARE near-orthogonal (and that is the problem)

A pure screen read with the *wrong* key. `cm` is the constant-modulus read (0 = locked
keyed structure, ~1 = noise-like). `xBER` is the wrong-key blind decode.

| trial | cm(B, KB) self | cm(B, KA) cross | xBER(B\|KA) | xBER(A\|KB) |
|---|---:|---:|---:|---:|
| 1 | 0.0000 | 1.027 | 0.500 | 0.495 |
| 2 | 0.0000 | 0.994 | 0.504 | 0.485 |
| 3 | 0.0000 | 0.984 | 0.502 | 0.500 |

The self-read locks perfectly (cm = 0); the cross-read reads pure noise (cm ~ 1, BER ~ 0.5).
So a different key genuinely makes a screen look like structureless noise -- the claimed
mechanism holds. But "looks like noise" means **full-power noise**: B deposits all of its
energy onto A's keyed modes as a complex-Gaussian-like interference. There is no subspace the
key projects B out of, because both screens are *full rank* across all L modes. Two full-rank
screens cannot be linearly separated without cancellation, and cancellation is out of scope.

### 3. The other screen is exactly full-power AWGN; and the blind lock dies first

Clean channel; informed = plain single-key read at the true geometry (diagnostic).

| pA | SINR dB | informed, vs screen B | informed, vs equal-power AWGN | blind, vs screen B |
|---|---:|---:|---:|---:|
| 0.5 | 0.0 | 0.341 | 0.345 | 0.504 |
| 0.6 | 1.8 | 0.280 | 0.283 | 0.500 |
| 0.7 | 3.7 | 0.200 | 0.205 | 0.303 |
| 0.8 | 6.0 | 0.109 | 0.109 | 0.240 |
| 0.9 | 9.5 | 0.021 | 0.019 | 0.021 |

- **Interference = AWGN.** The informed read against screen B tracks the informed read against
  equal-power Gaussian noise to within noise (e.g. 0.109 vs 0.109 at 6 dB). The co-channel
  screen is, to the keyed read, nothing but added noise at power p_other. No processing gain,
  no spreading gain -- unlike CDMA, where N chips buy an N× gain, here both signals fill the
  same L modes, so no dimension is free.
- **The blind lock is the weaker link.** Below ~9 dB SINR the blind `decode` (0.50, 0.50,
  0.303, 0.240) is far worse than the informed read (0.341...0.109): the blind
  width/offset/carrier acquisition cannot settle through a co-channel screen that is within
  ~9 dB. The two meet only at 9.5 dB, where acquisition finally survives. So even the
  "interference is just noise, ride it with FEC" fallback is blocked for the pure receiver by
  acquisition, not only by SINR.

### 4. A wider screen does not help

Equal split, clean.

| L | blind BER_A | blind BER_B | informed BER_A |
|---|---:|---:|---:|
| 64 | 0.504 | 0.495 | 0.341 |
| 128 | 0.497 | 0.499 | 0.339 |

Doubling L adds modes to *both* screens equally, so the per-mode SINR is unchanged. Width is
not a separation lever.

### 5. Coherent QAM and higher differential order -- same wall

Equal split, clean.

| waveform | reader | BER_A | BER_B |
|---|---|---:|---:|
| 16-QAM | `read_qam`, informed single key | 0.287 | 0.287 |
| 8-DPSK | `decode`, blind | 0.500 | 0.496 |

16-QAM at 0 dB SINR reads ~0.29 even with the geometry handed to it (a denser constellation
drowns sooner in equal-power interference). 8-DPSK blind collapses like DQPSK. Changing the
alphabet does not change the SINR arithmetic.

## Go / no-go

**Dead end, for the entroptics-pure co-channel read.** Keyed apertures are quasi-orthogonal in
the sense that a wrong-key read is noise-like, but they are not *separable*: both screens are
full-rank in the same L-dimensional keyed space, so the second screen lands as full-power,
spreading-gain-free interference. Per-stream SINR is pinned to the power split, a fixed total
power cannot make both large, and at equal split both streams read at chance. The blind
receiver fails even earlier, losing acquisition below ~9 dB SINR. The aggregate at equal total
power is strictly **worse** than one stream. No power balance, SNR, width, or alphabet in the
sweep moves this.

**The SINR floor where it breaks.** The strong stream needs roughly **+9 dB** over the other
just for the blind lock to survive (0.9/0.1 clean, BER 0.021); informed, the floor relaxes to
the ordinary DQPSK SINR curve (~0.11 at 6 dB, ~0.02 at 9.5 dB). Either way the *weak* stream,
sitting at the negative of that ratio, is always dead.

## Caveats and follow-ons

- **Linear superposition, perfect co-channel alignment, same L and timing** -- the most
  favourable case for the hypothesis. It still fails, so a messier channel will not rescue it.
- **This is the single-antenna, pure-read question only.** Three things were deliberately left
  out of scope and are the only places a win could lie:
  1. **Interference cancellation / successive-IC.** Because the interference is *exactly*
     full-power AWGN of a decodable screen (Sweep 3), a receiver that decoded the strong stream,
     re-composed it, and subtracted it could in principle free the weak one. That is explicitly
     out of scope here (it needs a re-compose + subtract step, i.e. cancellation). It is the
     single most promising follow-on and deserves its own study -- but note it is no longer
     "the plain read separates them as-is".
  2. **Spatial / multi-antenna.** Two receive antennas give a second independent look and a
     genuine extra dimension; co-channel keyed streams might then separate by linear combining
     the way MIMO does. That is a real degree of freedom this study does not have and is the
     natural place for "MIMO or similar" to actually pay off.
  3. **>2 screens** will not help the pure read -- it only piles on more full-power
     interference and lowers every SINR further. Rule it out unless paired with (1) or (2).
- **The mechanism result (Sweep 2) is itself useful elsewhere:** a screen under a foreign key
  is indistinguishable from noise to that key's constant-modulus read (cm ~ 1), which is a
  coexistence statement (a foreign-keyed screen reads as noise, not as structure), not a multiplexing one.
