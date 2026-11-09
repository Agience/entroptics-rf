# Detectability — energy versus signal

The waveform is detectable as energy and not as a structured signal. A radiometer (energy detector)
sees its energy; the blind structure detectors cannot classify it as a signal; without the key it cannot
be read. This page measures both sides. "Low detectability" elsewhere in this repository means exactly
this: energy visible, structure not classifiable, content unreadable. Scripts in this folder reproduce it.

## Three curves vs SNR (signal power relative to the receiver noise floor)

| | threshold | source |
|---|---|---|
| **Decoder** (authorized RX, FEC-usable) | **≈ +5–6 dB** | `detectability3b.py` (ideal buffer + digital AWGN, tracked read, M=4): raw PHY BER 0.002/0.027/0.108/0.232/0.341 at 12/9/6/3/0 dB → chance by −8 dB; FEC corrects ~10–15% raw. Blind cold-acquire needs > 3 dB. |
| **Radiometer** (energy detector), 8 ms burst | **detects to ≈ −14 dB** | `detectability2.py` (cabled bench, coprime, interleaved nulls): z = (P_sig−P_noise)/(P_noise/√N) ≈ SNR_lin·√N, **validated** (SNR 19.4 dB → z_full 87104 ≈ 87×1000; −3.2 dB → 473). z≥5 at N≈19.5k. |
| **Radiometer**, 0.4 s integration | **detects to ≈ −23 dB** | same, N=1e6. |
| **Structure detectors** (is-it-a-signal?) | **not classified at any power measured** | rank/fold/kurt/lag z ≈ 0 at every power, vs the K=60 (spare-mode) control at **112–224σ** through the identical pipeline (bench). |

### Key finding
**The radiometer out-detects the decoder by ~20–28 dB.** At the power needed to close a link (≈ +5 dB
SNR at the RX), an energy detector with a comparable look at the TX sees the energy with a wide margin:
there is no power at which the link decodes and a radiometer sees nothing. What the waveform withholds
is classification and content: to the structure detectors it is statistically Gaussian, and without the
key it is unreadable. (The keyed screen is *dense* — every mode carries payload — so it
is NOT spread-spectrum; payload BER-vs-SNR is ordinary DQPSK, no process gain on the bits.)

## Reproduce
- `detectability3b.py` — decode-vs-SNR (offline, ideal buffer + digital noise). `PYTHONPATH=../../src <venv>/python`.
- `detectability2.py` — the radiometer curve; **transmits on the cabled bench** (coprime, −26 dB and lower,
  interleaved nulls). Optional: point `ENTROPTICS_BENCH_KL` at external bench helpers to rerun this hardware arm;
  `detectability3b.py` and `current_waveform.py` run without it. Verify TX off after.

## Method notes
- The **NESDR is 8-bit** → at low TX gain the signal underfills the ADC and quantization kills the decode;
  measure the decode threshold with **digital noise on a strong/ideal buffer**, not low on-air power.
- **Interleave nulls** per level (the noise floor drifts) and **force the known M=4** (`discover` mis-reads
  M=2 at low SNR, giving BER≈0.5).
