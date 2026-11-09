"""HARDWARE grounding of the ToA ranging floor (toa_floor.py is the simulation). No new TX -- it reads
cabled-bench captures (Pluto+ TX -> NESDR RX). Optional external capture path: point
ENTROPTICS_BENCH_KL at a directory holding caps_rates/sent_iq.npy and caps_rates/tx2400000_rx2400000.npy
to rerun it; nothing else in the repository needs those captures.

The arrival-time observable for this noise-like waveform is the keyed ROW ALIGNMENT, which the blind
read pins as Discovery.offset + Discovery.phase (sub-sample). We measure its JITTER on the real
capture by reading the position in a series of windows sliding across the record at a stride that is a
multiple of L (so the nominal row phase repeats); what remains is a slow clock-rate drift (fit and
removed) plus the noise-induced ToA scatter -- that residual std, in samples, times c/Rs is the
one-way range precision on real hardware.

DISCIPLINE (tighten): the method has its own floor, so we run the identical measurement first on the
ideal transmitted buffer (sent_iq, no channel) -- its residual is the numerical/method floor -- and
only the HW residual ABOVE that floor is the hardware ToA jitter. No classical correlator / matched
filter: the library's own blind read is the estimator (entroptics-pure).
"""
import os, sys, numpy as np
RF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, RF)
from entroptics_rf import waveform as wf
BK = os.environ.get("ENTROPTICS_BENCH_KL", "")  # optional external capture directory (see the docstring)
KEY, L, RS = 1234, 64, 2_400_000
C = 299_792_458.0
JOIN = 9001                                   # TX buffer was ~9000 rows; cyclic row0 can be anywhere

def positions(x, W=90000, stride=64 * 1500, nmax=10):   # stride (96000) > W: NON-overlapping windows
    x = np.asarray(x).ravel()                            # -> independent thermal noise per read
    starts = list(range(60000, min(len(x) - W, 60000 + stride * nmax), stride))
    P, S = [], []
    for s in starts:
        w = x[s:s + W].astype(np.complex64); w = w - w.mean()
        d = wf.discover(w, KEY, [L], carrier_span=0.3, join_rows=JOIN)
        sig = wf.closure_significance(d, w.size)
        if d.L == L and sig >= 20.0:
            P.append(d.offset + d.phase); S.append(sig)
    return np.array(P), np.array(S), len(starts)

def jitter(P):
    """Unwrap position mod L, fit a line vs index (slow clock drift), return residual std (samples)."""
    if len(P) < 4:
        return float("nan"), P
    u = np.unwrap(P / L * 2 * np.pi) / (2 * np.pi) * L   # unwrap the mod-L row phase
    k = np.arange(len(u))
    fit = np.polyfit(k, u, 1)
    res = u - np.polyval(fit, k)
    return float(np.std(res)), res

print("=== FLOOR: ideal transmitted buffer (sent_iq, no channel) ===", flush=True)
tx = np.load(BK + r"\caps_rates\sent_iq.npy")
# sent_iq is the TX buffer; tile if shorter than needed so windows slide
if tx.size < 700000:
    tx = np.tile(tx, int(np.ceil(800000 / tx.size)))
Pf, Sf, nf = positions(tx)
sf, _ = jitter(Pf)
print(f"  {len(Pf)}/{nf} windows locked (sig {Sf.min():.0f}-{Sf.max():.0f}); "
      f"position floor std = {sf:.5f} samp = {C/RS*sf:.3f} m\n", flush=True)

print("=== HARDWARE: real matched-rate capture (Pluto TX 2.4 -> NESDR RX 2.4) ===", flush=True)
hw = np.load(BK + r"\caps_rates\tx2400000_rx2400000.npy")
Ph, Sh, nh = positions(hw)
sh, res = jitter(Ph)
print(f"  {len(Ph)}/{nh} windows locked (sig {Sh.min():.0f}-{Sh.max():.0f})", flush=True)
print(f"  position jitter std = {sh:.5f} samp", flush=True)
print(f"  -> one-way range precision = {C/RS*sh:.3f} m @ {RS/1e6:g} MHz "
      f"(= {C/20e6*sh:.3f} m if the band were 20 MHz)", flush=True)
hw_only = np.sqrt(max(sh**2 - sf**2, 0.0))
print(f"  method floor subtracted: hardware-only jitter ~ {hw_only:.5f} samp = {C/RS*hw_only:.3f} m", flush=True)
print(f"\n  sim prediction (toa_floor.py) @ ~20 dB: 0.0001 samp / 0.02 m -- compare.", flush=True)
