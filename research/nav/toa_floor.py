"""Ranging FLOOR: how precisely the blind keyed read pins arrival time (ToA).

Complements unsync_ranging.py (which removes the CLOCK-RATE error). Once the rate is corrected, the
range is limited by the arrival-time precision of the read. For this noise-like waveform there is no
envelope edge to detect; the time reference IS the keyed row alignment, which the library's own blind
read pins to sub-sample precision (Discovery.offset + Discovery.phase, refined by _refine_continuous
at SCAN_PHASES resolution). We measure that precision with the library read -- NO classical
correlator, NO matched filter -- so it stays entroptics-pure.

Method (no hardware): compose a keyed burst, impose a known sub-sample propagation delay by a
band-limited FFT fractional shift (a CHANNEL model, not a codec op), add AWGN, blind-read the
position, and (1) confirm the recovered position tracks the imposed delay (slope ~1 = true sub-sample
ToA), (2) measure the position jitter across noise = the ToA precision. Map to range by
sigma_R = (c / Rs) * sigma_position.
"""
import os, sys, time, numpy as np
RF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, RF)
from entroptics_rf import waveform as wf

C = 299_792_458.0
L, K, M, KEY = 64, 64, 4, 1234
T = 800
RS = 2_000_000                               # bench sample rate; 1 sample = c/Rs = 150 m
rng = np.random.default_rng(11)
bits = rng.integers(0, 2, T * K * 2).astype(np.int8)
tx = wf.compose(bits, K, L, KEY, M=M).astype(np.complex64)
N = tx.size
print(f"tx {N} samples (T={T}), Rs={RS/1e6:g} MHz -> {C/RS:.1f} m/sample", flush=True)

def fshift(x, d):                            # band-limited fractional delay by d samples (channel)
    Xf = np.fft.fft(x); k = np.fft.fftfreq(x.size)
    return np.fft.ifft(Xf * np.exp(-2j * np.pi * k * d)).astype(np.complex64)

def pos(y, snr_db):
    p = np.mean(np.abs(y) ** 2)
    n = np.sqrt(p / (2 * 10 ** (snr_db / 10))) * (np.random.randn(y.size) + 1j * np.random.randn(y.size))
    z = (y + n).astype(np.complex64); z = z - z.mean()
    d = wf.discover(z, KEY, [L], carrier_span=0.1)
    if d.L != L or not getattr(d, "locked", d.score < 0.8):
        return None
    return d.offset + d.phase

# clean baseline position (truth reference) at d=0
base = wf.discover(tx - tx.mean(), KEY, [L], carrier_span=0.1)
p0 = base.offset + base.phase
print(f"clean baseline position = {p0:.4f} samples (offset {base.offset}, phase {base.phase:.4f})\n", flush=True)

# (1) sub-sample tracking: recovered position vs imposed delay (clean)
print("--- sub-sample tracking (clean) ---", flush=True)
print(" imposed_d   recovered_dpos", flush=True)
for d in (0.0, 0.25, 0.5, 0.75):
    pr = wf.discover(fshift(tx, d) - fshift(tx, d).mean(), KEY, [L], carrier_span=0.1)
    print(f"  {d:.2f}        {(pr.offset + pr.phase) - p0:+.4f}", flush=True)

# (2) jitter across noise at several SNR = ToA precision
print("\n--- ToA precision (jitter across 8 noise trials) ---", flush=True)
print(" SNR   lock    sigma_pos(samp)   sigma_range(1-way)   @20MHz", flush=True)
for snr in (20, 10, 5):
    t0 = time.time(); got = []
    for i in range(8):
        np.random.seed(200 + i); r = pos(tx, snr)
        if r is not None: got.append(r)
    if len(got) >= 2:
        s = float(np.std(got)); sr = C / RS * s; sr20 = C / 20e6 * s
        print(f" {snr:>3} dB  {len(got)}/8   {s:.4f}          {sr:8.2f} m          {sr20:6.2f} m   ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f" {snr:>3} dB  {len(got)}/8   (too few locks)", flush=True)
print("\nNote: sigma_range is one-way (distance = c*ToF). TWR/TDOA combine >=2 such reads; the floor"
      "\nscales as 1/Rs, so a wider capture band tightens it proportionally. This is the arrival-time"
      "\nfloor that the blind clock-rate correction (unsync_ranging) leaves as the ranging limit.", flush=True)
