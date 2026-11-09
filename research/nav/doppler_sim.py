"""Doppler on the simulated channel: does the link decode for moving nodes, and how well does the
receiver read range-rate off it?  The sweeps behind ``tests/test_doppler.py``.

Channel: ``tests/_ratechannel.moving`` -- a sender at range R(t) = R0 + v t + a t^2 / 2, narrowband
propagation computed exactly (code Doppler as a time-varying sample position, carrier Doppler
as the phase -2 pi fc (R(t) - R0) / c), plus AWGN.  Waveform: ``compose``, K = L = 64, DQPSK,
2.4 MS/s.  Receiver: ``waveform.decode`` (burst; integer screen) and ``runner.StreamReceiver``.

Range-rate is read off the carrier the receiver measures: v = -nu * (rate / L) * c / fc, nu in
cycles per row.  One burst cannot tell Doppler from the oscillators' difference; part 3 removes
the latter with a static calibration burst, as the bench did (``doppler_hw.md``).

    OMP_NUM_THREADS=4 PYTHONPATH=../../src python doppler_sim.py        # ~5 min
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from entroptics_rf import waveform as wf                           # noqa: E402
from entroptics_rf.runner import Link, StreamReceiver, random_bits  # noqa: E402
from _ratechannel import C, moving                                  # noqa: E402

L, RATE, KEY = 64, 2.4e6, 4321
WIDTHS = (63, 64, 65)


def burst(rows, seed=1):
    bits = random_bits(2 * L * rows, seed=seed)
    return bits, wf.compose(bits, L, L, KEY, M=4)


def v_of(nu, fc):
    return -float(nu) * RATE / L * C / fc


def decode_case(bits, tx, fc, v, a=0.0, snr=15.0, ppm=0.0, seed=3, delay=-6.0):
    y = moving(tx, tx.size - 8, fc=fc, rate=RATE, range_rate=v, accel=a, osc_ppm=ppm,
               delay=delay, snr_db=snr, seed=seed)
    got, d = wf.decode(y, KEY, WIDTHS)
    ref = bits[d.row0 * 2 * L:][:got.size]
    ber = float(np.mean(got[:ref.size] != ref)) if ref.size else 1.0
    return ber, d


def part1():
    print("== 1. burst decode of a moving sender, 400 rows (10.7 ms), 15 dB ==")
    print(" fc MHz |    v m/s |  a m/s2 |    BER   | nu read  | v read m/s | error m/s")
    bits, tx = burst(400)
    for fc in (915e6, 2.4e9):
        for v in (0.0, 30.0, -100.0, 300.0, -1000.0):
            for a in (0.0, 1000.0):
                ber, d = decode_case(bits, tx, fc, v, a)
                vh = v_of(d.carrier, fc)
                print(f" {fc / 1e6:6.0f} | {v:8.0f} | {a:7.0f} | {ber:.5f}  | {d.carrier:+.5f} |"
                      f" {vh:+10.2f} | {vh - v:+7.2f}", flush=True)


def part2(trials=12):
    print("\n== 2. carrier (range-rate) precision per burst, over noise draws ==")
    print(" rows | SNR dB | nu RMS cyc/row | nu mean | RMS m/s @915 MHz | @2.4 GHz | BER max")
    for rows in (400, 2000):
        bits, tx = burst(rows)
        for snr in (10.0, 15.0, 20.0):
            errs, bers = [], []
            for t in range(trials):
                v = (-1) ** t * 37.0 * (t + 1)
                ber, d = decode_case(bits, tx, 2.4e9, v, snr=snr, seed=100 + t)
                errs.append(d.carrier - (-2.4e9 * v / C / RATE * L))
                bers.append(ber)
            e = np.array(errs)
            rms = float(np.sqrt(np.mean(e ** 2)))
            print(f" {rows:4d} | {snr:6.0f} | {rms:.2e}       | {e.mean():+.1e} | "
                  f"{rms * RATE / L * C / 915e6:8.2f}         | {rms * RATE / L * C / 2.4e9:6.2f}  |"
                  f" {max(bers):.4f}", flush=True)


def part3():
    print("\n== 3. the stream receiver under Doppler (1200 rows, acquire on 400, 15 dB) ==")
    print(" fc MHz |  v m/s | a m/s2 | nu at lock | BER by quarter of the stream")
    bits, tx = burst(1200)
    for fc, v, a in ((915e6, 0, 0), (915e6, 300, 0), (2.4e9, 100, 0), (2.4e9, 300, 0),
                     (2.4e9, -1000, 0), (915e6, 0, 5000)):
        y = moving(tx, tx.size - 8, fc=fc, rate=RATE, range_rate=v, accel=a, snr_db=15, seed=3)
        rx = StreamReceiver(Link(K=L, L=L, seed=KEY))
        ok = rx.acquire(y[:400 * L], candidates=range(63, 66))
        got = np.concatenate([rx.feed(y[i:i + 8192]) for i in range(400 * L, y.size, 8192)])
        sent = bits[rx.lock.row0 * 2 * L:]
        n = min(sent.size, got.size)
        q = [round(float(x.mean()), 4) for x in np.array_split(sent[:n] != got[:n], 4)]
        print(f" {fc / 1e6:6.0f} | {v:6.0f} | {a:6.0f} | {rx.lock.carrier:+.4f}    | {q}"
              f"{'' if ok else '  (no lock)'}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    part1()
    part2()
    part3()
    print(f"\n({time.time() - t0:.0f} s)")
