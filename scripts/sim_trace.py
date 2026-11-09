"""Create a simulated trace: compose, push through the loopback channel, write .cf32.

``tx`` and ``rx`` build separate radios, so the loopback buffer one fills is not the buffer the
other reads, and there is no CLI path to a simulated capture file.  This is that path: one radio,
transmit then receive, exactly as LoopbackRadio documents the loop.

Everything printed is measured from the trace after the channel rather than from the bits that
went in, so the numbers say what a receiver holding only the key would see.

    python scripts/sim_trace.py trace.cf32 --snr 12 --cfo 3e-4
    inspectrum trace.cf32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_root = Path(__file__).resolve()
while not (_root / "src" / "entroptics_rf").is_dir():        # the repo, found not assumed
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))

from to_cf32 import write_cf32                                       # noqa: E402

from entroptics_rf import waveform as wf                             # noqa: E402
from entroptics_rf.radio import LoopbackRadio                        # noqa: E402
from entroptics_rf.runner import Link, ber, random_bits              # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="trace to write, e.g. trace.cf32")
    p.add_argument("--K", type=int, default=None,
                   help="modes carried; every mode is lit, so K = L (the default)")
    p.add_argument("--L", type=int, default=64, help="screen width")
    p.add_argument("--seed", type=int, default=1234, help="THE KEY")
    p.add_argument("--M", type=int, default=4, help="alphabet order")
    p.add_argument("--coherent", action="store_true", help="coherent square QAM")
    p.add_argument("--sender", default=None, help="mark the transmission with this identifier")
    p.add_argument("--rows", type=int, default=500, help="screen rows to compose")
    p.add_argument("--samples", type=int, default=1 << 18, help="trace length")
    p.add_argument("--rate", type=float, default=2e6, help="sample rate, Hz")
    p.add_argument("--freq", type=float, default=915e6, help="centre frequency, Hz")
    p.add_argument("--snr", type=float, default=None, help="channel SNR in dB (default clean)")
    p.add_argument("--cfo", type=float, default=0.0, help="carrier offset, cycles/sample")
    p.add_argument("--delay", type=float, default=0.0, help="fractional sample delay")
    p.add_argument("--image-dbc", type=float, default=None, dest="image_dbc",
                   help="IQ image level in dBc")
    p.add_argument("--noise", action="store_true",
                   help="write bare channel noise instead -- the null trace to compare against")
    a = p.parse_args(argv)
    a.K = a.L if a.K is None else a.K

    link = Link(K=a.K, L=a.L, seed=a.seed, M=a.M, sender=a.sender, coherent=a.coherent)
    bits = random_bits(int(round(np.log2(link.M))) * a.rows * link.K, seed=1)
    iq = (wf.compose_qam(bits, link.K, link.L, link.seed, M=link.M) if a.coherent
          else wf.compose(bits, link.K, link.L, link.seed, M=link.M, sender=a.sender))

    if a.noise:                       # same power, same length, no screen -- for A/B in the viewer
        rng = np.random.default_rng(0)
        iq = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))) / np.sqrt(2)
        iq = iq.astype(np.complex64)

    radio = LoopbackRadio(sample_rate=a.rate, center_freq=a.freq, snr_db=a.snr,
                          cfo=a.cfo, delay=a.delay, image_dbc=a.image_dbc, seed=7)
    radio.transmit(iq)
    cap = radio.receive(a.samples)

    write_cf32(cap.iq, a.out, a.rate, a.freq)
    print(f"{a.out}  {len(cap)} samples  {a.rate/1e6:.4f} Msps @ {a.freq/1e6:.3f} MHz"
          f"  {'NOISE (null trace)' if a.noise else 'K=%d L=%d M=%d' % (link.K, link.L, link.M)}")
    print(f"channel: snr={'clean' if a.snr is None else '%g dB' % a.snr} cfo={a.cfo} "
          f"delay={a.delay} image={a.image_dbc}")

    o = wf.optics(cap.iq, link.L)
    print(f"optics:  PAPR={o.papr_db:.1f} dB  fill={o.fill:.4f}  etendue={o.etendue:.4f}  "
          f"occupied={o.occupied:.1f}/{o.capacity:.0f} DOF  resolved={o.resolved}")
    intf = wf.interference(cap.iq, link.L)
    print(f"interference: strehl={intf.strehl:.4f} dominance={intf.dominance:.4f}")

    d = wf.discover(cap.iq, link.seed)
    print(f"blind read: L={d.L} offset={d.offset} K={d.K} M={d.M} "
          f"margin={d.margin:.1e} closure={d.closure:.3f}")
    if not a.noise and d.L:
        got = (wf.read_qam if a.coherent else wf.read)(
            cap.iq, d.L, d.K, link.seed, M=d.M, offset=d.offset)
        n = min(len(bits), len(got))
        print(f"BER={ber(bits[:n], got[:n]):.4f} over {n} bits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
