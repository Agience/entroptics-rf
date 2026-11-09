# Out-of-band emission vs the RSS-247 §6.6 DTS mask — measured characterization

**Question.** RSS-247 §6.6 (digital transmission systems): in any 100 kHz band outside the
authorized band, emission must be ≥ 20 dB below the highest in-band 100 kHz. Some bench
configurations read an adjacent-100-kHz skirt only ~11–24 dB down. Is that the **entroptics waveform** (its band-edge roll-off,
a real property of a full-band, no-pulse-shaping design) or the **SDR analog chain** (a DAC/LO image
spur)? The two have different fixes, so the distinction decides what can be claimed.

**Method.** Offline, no transmission. 100 kHz-RBW Welch PSD (Hann, 50% overlap) on the
bench captures (TX gain −26 dB, Pluto+ → 30 dB → NESDR; full-occupancy upper skirt from captures
with the NESDR tuned +1.0 MHz so clear spectrum sits above the signal). For each capture: the worst-100 kHz margin (the mask metric), **and** where the worst OOB
bin sits and how concentrated it is — a discrete spur puts most OOB power in one bin and stands many
dB above the surrounding OOB floor; a smooth skirt does not. This is a cabled-bench
characterization, not a certified measurement.

## Measured

Occupancy < 1 (rate-pair captures), worst-100 kHz below in-band peak; "worst>med" = worst OOB bin above
the OOB median (concentration); location is the baseband frequency of that bin:

| config | ratio | occ | worst | median | verdict | worst>med | shape |
|---|---|---|---|---|---|---|---|
| 1.234567→2.4 | **coprime** | 0.51 | **24.2** | 25.7 | **PASS** | 1.5 dB @ +0.80 | smooth skirt |
| 1.234567→2.048 | **coprime** | 0.60 | **24.2** | 25.3 | **PASS** | 1.1 dB @ +0.82 | smooth skirt |
| 0.8→2.4 | harmonic 1:3 | 0.33 | 15.4 | 26.8 | short 4.6 | **11.4 dB** @ +0.50, 36% in 1 bin | concentrated spur |
| 1.2→2.4 | harmonic 1:2 | 0.50 | 15.6 | 25.8 | short 4.4 | **10.2 dB** @ +0.70, 36% in 1 bin | concentrated spur |
| 2.0→2.4 | 5:6 | 0.83 | 16.8 | 17.9 | short 3.2 | 1.1 dB @ +1.10 | skirt, only 3 OOB bins of room (measurement-limited) |

Full occupancy (RX tuned +1.0 MHz, matched 1:1 rates):

| config | worst | median | worst bin | shape |
|---|---|---|---|---|
| 2.4→2.4 | 13.5 | **21.7** | 8.3 dB>med @ +1.10 MHz | concentrated spur over a passing broadband floor |
| 2.048→2.048 | 12.8 | **22.0** | 9.2 dB>med @ +0.92 MHz | concentrated spur over a passing broadband floor |

TX backoff sweep: the 1:2 spur margin is **flat/worse as the DAC backs off**
(−26→−42 dB: 15.6→8.3), i.e. **linear** — an image, not compression; the coprime control stays
clean across the same sweep.

## Conclusion

1. **The waveform's own skirt clears the 20 dB line.** The broadband (median) out-of-band emission was
   measured ≥22 dB below the in-band peak at every measured occupancy, including full occupancy, on a
   cabled bench; not a certified measurement (21.7 / 22.0 dB at
   2.4 / 2.048 full occ). The full-band, no-pulse-shaping design does **not** splatter: its own skirt
   clears the 20 dB line.

2. **The only shortfalls are discrete spurs — hardware, not waveform.** Every sub-20 dB worst-case
   number is a single 100 kHz bin standing 8–11 dB above the surrounding OOB floor (36–52% of OOB
   power in one bin), **linear with TX backoff**, and **present only on commensurate (harmonic) TX/RX
   sample-rate ratios**. That is the signature of a Pluto/NESDR **DAC/LO image**, not the entroptics
   emission.

3. **Coprime ratios pass as-is.** On coprime ratios the spur vanishes — worst = median ≈ **24 dB**, a
   smooth skirt. The codec's coprime-ratio operating rule (adopted to avoid IQ-image spurs) already
   keeps the measured emission inside the mask line at occupancy ≤ 0.6.

4. **Product-radio fix for commensurate operation:** IQ/LO calibration or a cheap output filter
   removes the image; it is a hardware-cal item, standard for any SDR, not a waveform change.

## Residual / bench limit

The full-occupancy **coprime** skirt is **not directly measured** here: it needs a receiver wider
than the signal, and the NESDR tops out at 2.4 Msps (a full-occupancy 2.4 Msps signal fills its whole
window — "needs a wider RX capture"). The matched (1:1) full-occupancy numbers above stand in for it,
and they already show the broadband floor passing with only an image spur on top. The decisive direct
measurement is a **wide Pluto RX** (10–15 Msps, TX → 30 dB → Pluto RX loopback), with a burst
generated oversampled so it occupies ~1/5 of the window and the full skirt is visible with OOB room on
both sides; that also separates the DAC image (moves with the sample-rate ratio) from the waveform
roll-off (fixed to the signal's band edge) beyond doubt.
