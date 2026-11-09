"""A moving sender: the burst receiver decodes through Doppler and reads range-rate off the link.

The channel is ``_ratechannel.moving``: range R(t) = R0 + v t + a t^2 / 2, with the code Doppler
(a time-varying sample position) and the carrier Doppler (-fc v / c, plus the chirp from a)
computed exactly.  Range-rate is read off the carrier the receiver already measures,
v = -nu (rate / L) c / fc with nu in cycles per row; nothing is added to the receiver for it.

One burst cannot tell Doppler from the two oscillators' difference, so the navigation test
removes that with a static calibration burst, as the bench did (research/nav/doppler_hw.md), and
then integrates range-rate into range change: relative range over time, one-way, no GPS.  Absolute
range is not measured one-way (research/nav/relative_range.md).

The stream receiver (``runner.StreamReceiver``) does not apply the carrier it measured: its
``feed`` projects at carrier zero, so its BER climbs with Doppler (research/nav/doppler_sim.py,
part 3).  That is pinned below as an expected failure, so a fix shows up as an XPASS.

Sweeps and the precision table: research/nav/doppler_sim.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import waveform as wf
from entroptics_rf.runner import Link, StreamReceiver, random_bits

from _ratechannel import C, moving

L, RATE, KEY = 64, 2.4e6, 4321
WIDTHS = (63, 64, 65)


def _burst(rows: int, seed: int = 1):
    bits = random_bits(2 * L * rows, seed=seed)
    return bits, wf.compose(bits, L, L, KEY, M=4)


def _rx(bits, tx, *, fc, v, a=0.0, ppm=0.0, snr=15.0, seed=3):
    """Decode one burst from a sender moving at ``v`` m/s; the capture starts inside the last row
    of the cyclic buffer, so no row boundary is given."""
    y = moving(tx, tx.size - 8, fc=fc, rate=RATE, range_rate=v, accel=a, osc_ppm=ppm,
               delay=-6.0, snr_db=snr, seed=seed)
    got, d = wf.decode(y, KEY, WIDTHS)
    ref = bits[d.row0 * 2 * L:][:got.size]
    ber = float(np.mean(got[:ref.size] != ref)) if ref.size else 1.0
    return ber, d


def _v(nu: float, fc: float) -> float:
    return -float(nu) * RATE / L * C / fc


BITS, TX = _burst(400)                        # 400 rows: 10.7 ms at 2.4 MS/s


@pytest.mark.parametrize("fc,v,a", [
    (915e6, 30.0, 0.0),                       # a car
    (915e6, -300.0, 0.0),                     # an aircraft, closing
    (2.4e9, 300.0, 1000.0),                   # an aircraft at 2.4 GHz, pulling 100 g
    (2.4e9, -1000.0, 0.0),                    # nu = +0.21 cycles/row
])
def test_a_moving_sender_decodes_and_its_range_rate_is_read(fc, v, a):
    ber, d = _rx(BITS, TX, fc=fc, v=v, a=a)
    assert d.L == L
    assert ber < 1e-3, f"BER {ber} at fc {fc:.3g} Hz, v {v} m/s"
    # carrier precision at 400 rows, 15 dB: RMS 5.3e-4 cycles/row over 12 draws (doppler_sim.py
    # part 2); the bound, 2.5e-3 cycles/row, is 31 m/s at 915 MHz and 12 m/s at 2.4 GHz.
    assert abs(d.carrier - (-fc * v / C / RATE * L)) < 2.5e-3, (d.carrier, _v(d.carrier, fc), v)


def test_relative_range_is_tracked_from_range_rate_without_gps():
    """GPS-free relative navigation at its basic level.

    Node B flies past node A at 2.4 GHz: range-rate runs from -120 m/s to +60 m/s over 4 s
    (constant 45 m/s^2), and the two oscillators differ by 1.3 ppm, unknown to A.  A hears one
    53 ms burst every second at 20 dB.  A burst while B is static first fixes the oscillator
    offset (assumed constant over the run); every later burst's carrier, less that offset, is B's
    range-rate, and integrating it gives B's range change.  Asserted: each range-rate within
    4 m/s, and the 4 s range change (-120 m) within 8 m.  The calibration's own error enters every
    burst alike, so it integrates: 1 m/s of it is 4 m over 4 s."""
    fc, ppm, acc, v0 = 2.4e9, 1.3, 45.0, -120.0
    bits, tx = _burst(2000, seed=3)
    _, cal = _rx(bits, tx, fc=fc, v=0.0, ppm=ppm, snr=20.0, seed=50)
    times = np.arange(0.0, 4.01, 1.0)
    v_true = v0 + acc * times
    v_read = []
    for k, (t, v) in enumerate(zip(times, v_true)):
        ber, d = _rx(bits, tx, fc=fc, v=v, a=acc, ppm=ppm, snr=20.0, seed=60 + k)
        assert ber < 5e-3
        v_read.append(_v(d.carrier - cal.carrier, fc))
    v_read = np.array(v_read)
    assert np.max(np.abs(v_read - v_true)) < 4.0, np.round(v_read - v_true, 2)
    dr_true = v0 * times[-1] + 0.5 * acc * times[-1] ** 2            # -480 + 360 = -120 m
    print("range-rate error m/s", np.round(v_read - v_true, 2))
    dr_read = float(np.sum((v_read[1:] + v_read[:-1]) / 2 * np.diff(times)))
    print(f"range change read {dr_read:.2f} m, true {dr_true:.2f} m")
    assert abs(dr_read - dr_true) < 8.0, (dr_read, dr_true)


def test_the_stream_receiver_decodes_through_doppler():
    """feed() reads at the carrier and sampling phase acquire() measured, as read_soft does, so a
    300 m/s range rate at 2.4 GHz decodes through the stream as it does through a burst."""
    bits, tx = _burst(800)
    y = moving(tx, tx.size - 8, fc=2.4e9, rate=RATE, range_rate=300.0, snr_db=15, seed=3)
    rx = StreamReceiver(Link(K=L, L=L, seed=KEY))
    assert rx.acquire(y[:400 * L], candidates=range(63, 66))
    got = np.concatenate([rx.feed(y[i:i + 8192]) for i in range(400 * L, y.size, 8192)])
    sent = bits[rx.lock.row0 * 2 * L:]
    n = min(sent.size, got.size)
    assert np.mean(sent[:n] != got[:n]) < 1e-3
