"""Pure decode-vs-SNR via the IDEAL transmit buffer + digital complex-Gaussian noise (no RF, no ADC
quantization, perfect geometry). TRACKED read (known geometry, M forced to the true 4) = the deployed
authorized-receiver threshold; BLIND re-discover = the cold-acquisition threshold."""
import os, sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
BK = os.environ.get("ENTROPTICS_BENCH_KL", "")  # bench captures live outside this repo; set to your bench_kl dir
sys.path.insert(0, BK)
import bench_common as bc
import analyze
from entroptics_rf import waveform as wf

iq = np.load(f"{BK}/caps_rates/sent_iq.npy").astype(complex)
bits = np.load(f"{BK}/caps_rates/sent_bits.npy")
P = float(np.mean(np.abs(iq) ** 2))
rng = np.random.default_rng(0)
d = wf._discover_scaled(iq, bc.KEY, (63, 64, 65), carrier_span=0.3)       # perfect geometry (scale 1)
print(f"ideal geometry: L={d.L} M={d.M} scale={d.scale:.4f} score={10*np.log10(max(d.score,1e-12)):.0f} dB", flush=True)

def noisy(snr_db):
    Padd = P / 10 ** (snr_db / 10)
    n = (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)) * np.sqrt(Padd / 2)
    return (iq + n).astype(complex)

print("\nTRACKED read (known geometry, M=4) vs SNR:", flush=True)
for snr in [12, 9, 6, 3, 0, -2, -4, -6, -8, -10, -12]:
    ber, *_ = analyze.cyclic_dqpsk_ber(noisy(snr), d, bits, 4)
    print(json.dumps(dict(snr_db=snr, tracked_ber=round(float(ber), 4))), flush=True)

print("\nBLIND re-acquire (cold) vs SNR:", flush=True)
for snr in [3, -1, -5, -9]:
    db = wf._discover_scaled(noisy(snr), bc.KEY, (63, 64, 65), carrier_span=0.3)
    ok = db.L in (63, 64, 65) and 10 * np.log10(max(db.score, 1e-12)) < -3
    ber = "-"
    if ok:
        try: ber = round(float(analyze.cyclic_dqpsk_ber(noisy(snr), db, bits, db.M if db.M in (2,4) else 4)[0]), 4)
        except Exception: ber = "err"
    print(json.dumps(dict(snr_db=snr, blind_lock=bool(ok), score_db=round(10*np.log10(max(db.score,1e-12)),1), blind_ber=ber)), flush=True)
