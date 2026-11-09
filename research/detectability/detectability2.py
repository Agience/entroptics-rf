"""Detectability v2: interleaved nulls, robust decode, saved per-level. TRANSMITS (cabled, <=-26 dB).
Per TX gain: capture a TX-OFF null immediately before (handles floor drift), then the signal.
  SNR = (P_sig - P_null)/P_null  [dB]
  radiometer z at N:  (P_sig - P_null)/(P_null/sqrt(N))   (observer integrating N samples, knows P_null)
  decode: blind _discover_scaled; report lock + BER, or 'no-lock' (never raises)."""
import os, sys, time, json, warnings, numpy as np
warnings.filterwarnings("ignore")
BK = os.environ.get("ENTROPTICS_BENCH_KL", "")  # bench captures live outside this repo; set to your bench_kl dir
sys.path.insert(0, BK)
import bench_common as bc
import analyze
from entroptics_rf.radio import PlutoRadio, RtlSdrRadio
from entroptics_rf import waveform as wf

TX, RX = 1_234_567, 2_400_000
GAINS = [-26, -32, -38, -44, -50]
WIN = bc.WIN_DISCOVER
bits, iq = bc.rates_buffer()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detectability_rows.json")

rx = RtlSdrRadio(sample_rate=RX, center_freq=bc.FC, gain=bc.NESDR_GAIN); rx.receive(1 << 16)
Nb = 65 * 300                                    # ~a short burst-sized window

def zrad(P_s, P_n, N):
    return (P_s - P_n) / (P_n / np.sqrt(N))

def decode(cap):
    try:
        x = cap[WIN[0]:WIN[1]]
        d = wf._discover_scaled(x, bc.KEY, (63, 64, 65), carrier_span=0.3)
        locked = d.L in (63, 64, 65) and 10 * np.log10(max(d.score, 1e-12)) < -3   # low score = concentrated
        if not locked:
            return "no-lock", None, round(10 * np.log10(max(d.score, 1e-12)), 1)
        ber, worst, r0, rows = analyze.cyclic_dqpsk_ber(x, d, bits, d.M if d.M in (2, 4, 8, 16) else bc.M_RATES)
        return round(ber, 4), int(d.L), round(10 * np.log10(max(d.score, 1e-12)), 1)
    except Exception as e:
        return f"err:{type(e).__name__}", None, None

rows = []
for g in GAINS:
    tx = PlutoRadio(bc.PLUTO_URI, sample_rate=TX, center_freq=bc.FC, tx_gain_db=float(g), bandwidth=max(TX, 1_000_000))
    try:
        tx.stop_transmit(); time.sleep(0.4)
        null = rx.receive(bc.N_CAP).iq                      # local null (TX buffer destroyed)
        P_n = float(np.mean(np.abs(null) ** 2))
        tx.transmit(iq, cyclic=True); time.sleep(0.4)
        cap = rx.receive(bc.N_CAP).iq
    finally:
        tx.close()
    P_s = float(np.mean(np.abs(cap) ** 2))
    snr_lin = (P_s - P_n) / P_n
    snr_db = 10 * np.log10(snr_lin) if snr_lin > 0 else -99.0
    ber, L, score = decode(cap)
    row = dict(gain=g, snr_db=round(snr_db, 1),
               z_burst=round(zrad(P_s, P_n, Nb), 1), z_0p1s=round(zrad(P_s, P_n, 240_000), 1),
               z_full=round(zrad(P_s, P_n, len(cap)), 1), decode_ber=ber, lockL=L, score_db=score)
    rows.append(row); print(json.dumps(row), flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)                # save after EACH level

rx.close()
print("done -> " + OUT, flush=True)
