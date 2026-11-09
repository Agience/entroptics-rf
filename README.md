# entroptics-rf

[![Licence: AGPL-3.0-only or commercial](https://img.shields.io/badge/licence-AGPL--3.0--only%20or%20commercial-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-0F9D58)](tests/)
[![No FFT](https://img.shields.io/badge/DSP-none%20in%20the%20path-6E56CF)](tests/test_no_fft.py)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

**The RF signal *is* an entropy-matched screen.**

A seed-keyed optical aperture carries the payload in its coherent modes. The receiver is told
the key, and nothing else. For an open broadcast it is told nothing at all. No FFT and no DSP
anywhere in the path.

Measured, uncoded: 9.98 bits/s/Hz delivered, against 802.11ax MCS11's 7.17.

```python
from entroptics_rf import decode, decode_scaled

bits, d = decode(iq, key)       # a capture and a key.  That is the whole receiver.
bits, d = decode(iq)            # ...or an open broadcast, with no key at all.
bits, d = decode_scaled(iq, key, range(60, 69))  # a sender on another clock or rate: the
                                # scale and start read continuously (seconds a width)

d.L, d.offset, d.phase, d.carrier   # the aperture coordinates, all measured
d.K, d.M, d.redundancy              # modes (always L: every mode is lit), alphabet, how evenly used
d.row0, d.payload, d.aligned        # which row it joined on, which payload rides, did it align
d.score, d.margin, d.closure        # the evidence: ~1e-16, ~1e14 and 1.000 when locked
```

A parameter the receiver must be told is a parameter the codec failed to measure, so there are
none. The only thing that can be agreed in advance is the key, and only if the transmission is
meant to be private. Otherwise it is contiguous data that anyone who knows the mathematics can
read. `tests/test_no_preshared_parameters.py` enforces this mechanically — it inspects what
reception requires, so an argument cannot be added quietly.

Nothing is added to the waveform to make acquisition possible: no preamble, no pilot tone, no
cyclic prefix, no training sequence, no template. That absence is also where the efficiency
comes from.

---

## The aperture is continuous

A mode shape is a function of a real argument, not a list of samples. So the receiver can
evaluate it wherever its sampling instants actually landed, and wherever its oscillator actually
sits. Timing and carrier stop being impairments to correct and become coordinates to measure.

**Three coordinates** — a scale, and one translation per axis:

| coordinate | what it is |
|---|---|
| `L` | the scale — how many samples a row spans |
| `offset` + `phase` | translation on the sample axis: which sample a row starts on, and where between two samples it really starts |
| `carrier` | translation on the mode axis, ν = f·L cycles per row |

`offset` is an integer and `phase` is a real number of samples; together they are the one
continuous sample-axis position, split because the search over the integer part is an enumeration
and the fractional part is a settle.

The family is forced, not chosen. Wanting the rows to stay orthonormal at *every* real offset is
wanting shift-covariance, and the eigenfunctions of translation are the complex exponentials — so
a continuously shiftable aperture is necessarily a keyed mixture of them. The key lives in the
mixture: each mode is a dense keyed combination of every exponential. One coefficient per mode
would be one exponential per mode, which is OFDM; the mixture is what makes it not that.

Measured `||B Bᴴ − I|| < 1e-13` at every phase and carrier tried, worst 9.2e-14. Nothing resamples the
capture — the signal only ever meets a matrix.

## How the receiver discovers everything

**Every mode is lit.** One draw per `(L, seed)` gives the full L-mode aperture, and a
transmission excites all of it; only a message's final row uses fewer (the first k — the aperture
is nested, so a short row is the same aperture used less far). Nothing is left dark, because a
dark mode is a direction an observer *without* the key can find: the windowed covariance at the
true width then has a noise-only eigenvalue, and at L = 64 that gives the width away from −15 dB
in 10 s, whatever the number of dark modes and at every rate
([`research/rank_leak.md`](research/rank_leak.md)). A full screen's covariance is white.

What marks the true geometry for a key holder is the coordinates themselves: unit modulus (every
differential symbol, the keyed reference row) or a square lattice's few levels. At the true
geometry each row's energy spreads evenly over the keyed modes; at any other cut the rows are
mixtures of many coordinates, close to complex Gaussian, and their energies scatter like
exponentials. That evenness is the measurement — each row's Shannon entropy over its mode
energies (`entroptics.entropy.shannon_bits`) against log₂ m, over the flat Dirichlet's mean
deficit (`entroptics.entropy.dirichlet_entropy_moments`, exact): the *constant-modulus read*.

| quantity | how it is read | evidence |
|---|---|---|
| `L`, `offset`+`phase`, `carrier` | the coordinates whose rows spread evenly over the keyed modes | 1.5e-15 clean vs ~0.6–1 off the geometry — a 10¹⁴ cliff; 0.43 / 0.64 at 6 / 3 dB |
| `K` | not read: it is L (L − 3 under a row beacon, which is measured) | — |
| `M` | the smallest alphabet that *closes* under its own power | 0.003 below the true M, 1.000 at it |
| `redundancy` | how much of the alphabet went idle, from `entroptics.sequence.block_entropies` | 0.00 even, 0.50 half-used, 1.00 constant |
| payload mode | coherent QAM closes at *no* differential alphabet | — |

Every one is a cliff or a comparison between measured candidates, so none carries a threshold.
Widths are compared by significance against the read's own null spread, which is wider for a
narrow width. All three aperture coordinates come from one read, descended coarse to fine.

**Cost** (measured on a busy 8-core machine): 0.9 s for a narrow width range, 4.8 s for 100
widths, 5.9 s fully blind, the scan forming 16 coordinates per hypothesis. For a stream this is paid once; thereafter
a buffer costs one matrix product.

## Public broadcast is the zero state

`seed=None` is an open transmission, and its aperture is not a magic table anyone had to be given.
It is the keyed construction with the seed field empty: the same domain-separated SHAKE-256
stream every keyed aperture is drawn from, which anyone can compute from the spec alone, and which
no key can alias (every real seed encodes to at least one tagged byte). A receiver *derives* it.

It is as white as a keyed aperture, so open and keyed transmissions read — and look — alike. A
low-discrepancy sequence such as R2 would not do: it is not spectrally white, so a full-width open
broadcast would read as half occupied and an observer could tell an open transmission from a keyed
one by its spectrum.

It is the same path: one compose, one discovery, one read, whether the traffic is addressed or
open. What changes is only who can derive the aperture, and so who can read the payload.

| | receiver given | result |
|---|---|---|
| public | `decode(iq)` — nothing | L=64 K=64 M=4 off=47, bit-exact |
| keyed | `decode(iq, key)` | bit-exact |
| stranger, no key | | score 0.99, margin 0.88, closure 0.004 — cannot even find the geometry |
| wrong key | | BER 0.502 |

## Who sent it — carried for free

Rotating each mode of the aperture by its own phase leaves the payload untouched, because the read
is differential across rows and a constant per-mode phase cancels in `A[t+1]·conj(A[t])` exactly.
So the rotation costs nothing: no power, no rate, no change in how the transmission looks.

That same cancellation is why this waveform keys a per-row rotation. Blind separation recovers
the aperture up to a constant per-mode phase — exactly what the differential divides out — so
without a per-row rotation the property that carries a sender mark for free would hand an
attacker the payload at BER 0.0000. `mode_keys` changes the rotation every row. The sender mark
needs no such defence: it is an address rather than a secret.

| sender | payload BER | strehl | alice | bob | carol |
|---|---|---|---|---|---|
| alice | 0.0000 | 0.0317 | 1.000 | 0.040 | 0.184 |
| bob | 0.0000 | 0.0318 | 0.040 | 1.000 | 0.110 |

It survives 6 dB SNR with carrier offset (0.951, still correct). It is an address, not a
secret — anyone holding the aperture can read it, so it says who sent a transmission, not who
may read it.

## Spectral efficiency

For complex baseband, bits per sample are bits/s/Hz. Coherent QAM, uncoded (measured at
K/L = 0.984, a screen with a spare mode; at K = L the 1024-QAM ceiling is 9.98, measured BER 0 at
45 dB):

| SNR dB | best M | raw b/s/Hz | BER | **delivered** |
|---:|---:|---:|---:|---:|
| 18 | 64 | 5.90 | 0.0580 | 4.01 |
| 24 | 64 | 5.90 | 0.0041 | 5.67 |
| 30 | 256 | 7.86 | 0.0031 | 7.62 |
| 36 | 1024 | 9.83 | 0.0034 | 9.51 |
| 42 | 1024 | 9.83 | 0.0000 | 9.83 |

### Against 802.11ax (WiFi 6), 20 MHz, 1 spatial stream

| MCS | modulation | code rate | Mbit/s | b/s/Hz |
|---:|---|---:|---:|---:|
| 9 | 256-QAM | 3/4 | 103.2 | 5.16 |
| 11 | 1024-QAM | 5/6 | 143.4 | 7.17 |

**The margin is overhead, not cleverness.** Same constellations; the difference is what each
spends on things that are not payload:

| | cyclic prefix | pilots | guard tones | mode headroom | reference row | **total** |
|---|---:|---:|---:|---:|---:|---:|
| 802.11ax | 5.9% | 3.3% | 5.5% | — | — | 14.0% |
| entroptics-rf | none | none | none | none | 0.17% | 0.17% |

No cyclic prefix because there is no OFDM symbol to protect, no pilots because the read is
differential or keyed-reference, no guard tones because the feature axis is *time*, and no mode
headroom because every mode is lit. At 1024-QAM with the same rate-5/6 code WiFi 6 uses, this
would deliver 8.32 against 7.17 — +16%, purely structural.

> **Scope.** WiFi 6 reaches its rates over real channels with FEC, equalization and MIMO. These
> are single-stream AWGN simulation with no outer code. The overhead comparison is structural and
> holds; the SNR comparison does not account for coding gain, which is why the matched-code row is
> the one to read.

## Throughput

Single core, pure numpy, steady-state streaming (acquisition paid once; measured at K = 60, a screen with
spare modes -- the per-buffer cost is one product, so K = 64 costs ~7% more per row and
carries 7% more):

| L | K | M | b/s/Hz | RX Msps | RX Mbit/s |
|---:|---:|---:|---:|---:|---:|
| 64 | 60 | 2 | 0.94 | 19.9 | 18.7 |
| 64 | 60 | 4 | 1.87 | 13.2 | 24.8 |
| 64 | 60 | 16 | 3.75 | 11.4 | 42.8 |

Efficiency is independent of L while cost grows with K ≈ L — for a fast link keep L small and
raise M.

## Streaming

The stream is one screen that never ends, so the reference row is spent once for the whole
transmission and the lock acquired once:

```python
tx = StreamTransmitter(link)
iq = tx.send(bits)                # push arbitrary chunk sizes; whole rows come out

rx = StreamReceiver(link)
rx.acquire(iq[:40_000])           # the discovery, paid once
bits = rx.feed(next_buffer)       # thereafter: one matrix product per buffer
```

Rows straddling buffer boundaries are carried, so any chunking returns the same bits — tested at
step sizes 1, 63, 64, 65, 1000 and 4096.

## Its structure reads like noise

The energy is there to see: a radiometer (energy detector) out-detects the decoder by about
20–28 dB ([`research/detectability/README.md`](research/detectability/README.md)). What reads
like noise is the structure, and without the key the content cannot be read. Measured on a clean
capture: `resolved_modes` = 0 and `Projection.coherence` ≈ 0.5 — *the same
readings pure noise gives*. `Aperture.extract()` has nothing to pull out and `Aperture.sweep()`
finds no coherent band. Those reads are working: the same `extract()` recovers `K_signal = 4`
from a planted rank-4 screen, and `sweep()` finds a planted band in a wide field. The waveform's
singular spectrum is flat by construction, which *is* the i.i.d. bulk those reads test against.
Keyed and public both measure strehl ≈ 0.031.

**Security consequence, plainly:** discovery reads the aperture, so a stranger cannot find the
geometry either — but the sender rotation is an address, readable by anyone holding the aperture.
The key protects content.

## Payloads are carried as given

The transmitter writes at native resolution and conditions nothing. The payload is already
keyed by the waveform — the aperture is the cipher — so a second scrambling layer buys no
confidentiality and only multiplies errors.

The consequence only shows on real payloads, never on random bits: a source that does not exercise
the whole alphabet is genuinely explained by a smaller one, and the receiver reads the alphabet
that was *actually used*. That is a correct reading, and `redundancy` makes it visible — measured,
a 1-in-64 sparse pattern reads M=2 at redundancy 0.80, while English text and source code both read
M=4 at 0.00 and are bit-exact.

## No tuned constants in the decision path

Every read that decides something is threshold-free — it separates by orders of magnitude, or it
is a data-derived split:

| decision | how it is settled |
|---|---|
| `L`, `offset`+`phase`, `carrier` | the constant-modulus read: ~1e-15 at the truth against ~1 elsewhere, so the argmin needs no cut; widths compared by significance against the read's own null spread |
| `K` | not a decision: every mode is lit |
| `M`, row alignment | the smallest exponent at which the differential closes, searched jointly |
| which sender | `entroptics.reads.coupling`, scored against the exact permutation null |

The one declared input is `far = 0.05`, the false-alarm level of `sender_resolve`, at the
library's own default; it is corrected by the number of candidate senders. Everything else in
`waveform.py` is a forced universal (φ, for the carrier probes), a handful of search bounds
(`MIN_ROWS`, `CM_MODES`, `JOIN_SEARCH_ROWS`, the width cap, the carrier span), and numerical
guards. Provenance
is tabulated at the top of the module.

## Measured limits

- **The structure is solvable without the key; the payload under it is not.** `compose` makes
  `X = A·B` where every entry of A is unit modulus and B has orthonormal rows — the exact input to
  ACMA (van der Veen & Paulraj 1996) and ILSP (Talwar/Viberg/Paulraj 1994), whatever K is. Measured at K=60,
  L=64, ACMA recovers all 60 mode streams at agreement 1.000 from 118 ms of clean capture. What it
  recovers is unreadable: `mode_keys` applies a keyed unit-modulus rotation to
  the *differential*, varying row to row, so no constant per-mode ambiguity cancels it. Measured
  under it, the same attacker reads BER 0.497 where the key holder reads 0.000000. The
  rotation costs no power, no rate, and no change to how the transmission looks. (Each row also
  carries its own power, below; an aligned observer divides it out by the row's energy, so it is
  no defence here.)
- **Rows do not carry equal energy.** With A unit modulus and B orthonormal every row would carry
  energy K exactly, and the power envelope alone would give a key-free observer the row period
  (from 417 ms at L=64, the true width ranks first of 585 at z 24 from 0 dB). So each row is sent
  at its own power, `row_radius`: r_t² ~ Gamma(L, 1/L), keyed per row in its own domain, and in
  the open mode the sender's own draw. The rows' energies are then distributed as white noise's,
  and every row-energy statistic reads inside the noise-only range. The differential read never needs it;
  the coherent read divides it out from the key, or in the open mode measures it per row. It
  costs 0.1-0.2 dB on the coded thresholds and moves PAPR at 1e-4 from 9.42 to 9.64 dB, the
  Gaussian's value (`research/row_energy_leak.md`).
- **No second-order read of the air gives the geometry away; a fourth-order one could.** With a
  spare mode, the covariance over windows of exactly L samples is rank deficient, and that is not
  bounded by SNR: at L = 64 a 10 s capture shows the true width at z 19–27 from −15 dB, one
  spare mode as visible as eight, and resampling to another rate does not remove it
  ([`research/rank_leak.md`](research/rank_leak.md)). So every mode is lit and the receiver
  finds the geometry by the constant-modulus read instead. Measured on the codec's own
  transmissions, the rank, the fold (the per-sample power profile), the lag correlation and the
  row-energy detectors all read inside the noise-only range at every length and rate measured
  (rank_leak.md, "The shipped codec"). What a full screen still carries is fourth-order: given the width and
  the row offset, the fourth-order cumulant of whitened windows shows the constant-modulus
  structure (at L = 64 from about +10 dB in 417 ms). A keyed aperture per absolute row would
  close that too, at the cost of the blind late join.
- **No freshness. One seed keys one waveform.** Two transmissions under the same seed share an
  aperture and a reference row. That is the two-time-pad case, and it is deliberate: a nonce a
  caller supplies is a second shared parameter and only the key may be shared. Carrying one was
  built and rejected — a capture that starts late *truncates* a head block rather than shifting
  it, and a one-sample cut proved as fatal as a five-hundred-sample one. Freshness is key
  management, above this layer. See the design note in `waveform.py`.
- **Discovery enumerates two coordinates.** The width and the row offset are discrete and are
  found by descending the same read over candidates. The carrier is not gridded: it is read in
  closed form off the reference row and then settled by one bounded refinement. A coarse carrier
  probe set is still needed — golden-ratio ordered, extended until the geometry it returns stops changing
  — because the direct read must be told where the reference row is, and locating that row with
  the carrier assumed zero holds only to about 0.06 cycles/row, roughly 2 ppm where commodity
  parts are 10.
- **Discovery costs what the capture makes it cost.** There is no row count to pass: the read
  starts at the fewest rows it can work with and widens until the discrete answer stops changing,
  which is exact integer equality, not a threshold. Measured 1.6 s at 18 dB rising to 10.6 s at
  3 dB — cheap when easy, expensive only when it must be.
- **The reads are not what limits the link.** Against a receiver handed the true parameters,
  discovery costs 7% at 12 dB, 5% at 9, 2.5% at 6 and 2% at 3, locking every time. What remains is
  the modulation.
- **The alphabet is genuinely ambiguous when the used points lie on a smaller constellation.** A
  source using two of four points at 0° and 270° reads M=4 and is bit-exact; one using 0° and 180°
  is indistinguishable from M=2. A *constant* source carries no evidence of an alphabet at all, so
  the read returns the smallest on the ladder and `redundancy` reads ~1.
- **Multipath** degrades gracefully: discovery locks through a −6 dB echo; the payload survives to
  about −10 dB. No equalizer.
- **CW interference degrades the payload, not the lock.** With a mid-band tone the width and row
  lock at every level tried, and with no mode count to misread the blind decode degrades smoothly:
  BER 0.003 / 0.023 / 0.10 / 0.20 / 0.30 at −13 / −10 / −6 / −3 / 0 dBc. The interference read rises monotonically across
  the same levels — strehl 0.067 / 0.109 / 0.217 / 0.349 / 0.512 — so a receiver can see that the
  band holds interference.
- **The coherent mode is read blind; its order read gives way only where the payload already has.**
  `decode` locates the reference row, reads the row-rate carrier off the payload over every row
  (the geometry read's carrier is ~1e-4 off at 40 dB, which is 0.38 rad over 600 rows), and reads M
  as the square lattice the symbols close on. Blind BER tracks a receiver handed M and the true
  carrier at every SNR measured; the order read fails where the true lattice closes below ~0.14,
  e.g. 1024-QAM at 30 dB, where the payload is already at a few percent. An open broadcast
  (`seed=None`) of coherent 16-QAM or denser still reads as differential. See
  [`research/coherent_blind.md`](research/coherent_blind.md).
- **No FEC, hard decisions** — which is most of the remaining gap to Shannon at low SNR.
- **Hardware (cabled Pluto+ → NESDR bench, full-screen K = L waveform).** All
  eight TX/RX rate pairs (harmonic, coprime and matched, 0.33–1.0 occupancy) lock blind and
  decode: BER 0 on five, ≤0.0008 on the 94%-occupancy pair, 0.007–0.014 on the matched pairs,
  all at ~19–20 dB SNR. Coherent QAM decodes with no codec change when sent in the flat band
  centre (≤0.5 occupancy): 4- and 16-QAM BER 0, 64-QAM 0.007; at full occupancy the analog
  front end's band-edge roll-off (7–12 dB) pulls 16/64-QAM down (they read as 4-QAM) — a flatness
  limit, not a codec one. Emissions: the signal stays in its own channel (99% occupied width ≈ the
  intended width), and the broadband out-of-band emission was measured ≥22 dB below the in-band
  peak at every occupancy incl. full occupancy, clearing the RSS-247 §6.6 20 dB line on a cabled
  bench; not a certified measurement. The only sub-20 dB
  readings are discrete SDR image spurs (linear with TX backoff, worst on commensurate TX/RX ratios,
  absent on coprime where worst-case is 24 dB): a hardware-cal item, not a waveform property
  (`research/emission.md`). Not yet on the air: real-time decode (the Python
  prototype is ~45–300× slower than real time) and a two-radio link at distance.

## Layout

| module | role |
|---|---|
| [`waveform.py`](src/entroptics_rf/waveform.py) | the entroptics adapter — writes and reads the IQ, the only place the library is touched |
| [`radio.py`](src/entroptics_rf/radio.py) | radio controls — loopback, ADALM-Pluto, RTL-SDR |
| [`runner.py`](src/entroptics_rf/runner.py) | `Link`, `transmit`, `receive`, `loopback`, `ber`, streaming |
| [`__main__.py`](src/entroptics_rf/__main__.py) | the CLI |

Nothing outside `src/` is imported by the codec:

| folder | role |
|---|---|
| [`scripts/`](scripts) | the attack suite (ten blind detectors with positive controls, and ACMA / FastICA / ILSP), an evidence receiver, two trace utilities (`sim_trace.py`, `to_cf32.py`), a paper consistency check (`check_paper.py`) and a mutation audit of the test suite (`mutation_audit.py`) |
| [`research/`](research) | [`PAPER.md`](research/PAPER.md) — the construction written up with its definitions and measurements |

**entroptics does the optics.** Check the library before writing any primitive:
[`waveform.py`](src/entroptics_rf/waveform.py) is the only module that touches it.

## Install & use

```bash
pip install entroptics-rf                # the codec
pip install "entroptics-rf[sdr]"         # + ADALM-Pluto, via libiio
pip install "entroptics-rf[rtlsdr]"      # + RTL-SDR
```

From a checkout:

```bash
pip install -e ".[test]"      # core + pytest;  ".[sdr]" for Pluto, ".[rtlsdr]" for RTL-SDR
pytest                        # the suite
python examples/quickstart.py # blind keyed decode, open broadcast, and the key gating reading
```

```bash
python -m entroptics_rf selftest              # the whole waveform on a simulated link
python -m entroptics_rf focus --cut 17        # what discovery sees, offset by offset
python -m entroptics_rf ber --snr-list 18,12,9,6,3
python -m entroptics_rf tx --radio pluto --freq 915e6 --cyclic
```

```python
import numpy as np
from entroptics_rf import compose, decode, discover, optics, interference

iq = compose(np.random.randint(0, 2, 40_000), K=64, L=64, seed=1234, sender="alice")   # K = L
bits, d = decode(iq, 1234)          # the whole receiver
optics(iq, d.L), interference(iq, d.L)
```

Requires Python ≥ 3.10 and [entroptics](https://github.com/Agience/entroptics) ≥ 0.2.10.

Security issues: email **connect@agience.ai** rather than opening a public issue.

AGPL-3.0-only, or a commercial licence — see [`LICENSE`](LICENSE) and
[`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md). Patent pending — see [`PATENTS.md`](PATENTS.md).
Licensing inquiries: **connect@agience.ai**.

## Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8, 5, and 5.5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.
