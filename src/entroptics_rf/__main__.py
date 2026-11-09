"""
__main__.py -- the command line.  ``python -m entroptics_rf <command>``

    selftest              the whole waveform on a simulated link, printed as a table
    focus                 what discovery reads off a capture, and the constant-modulus score
                          behind it
    ber                   BER against SNR, blind lock in the loop
    embed                 the analog embedding mode: cosine fidelity against SNR
    tx                    transmit on a real radio
    rx                    capture from a real radio and read it

Every command takes ``--L --seed --M`` for the aperture agreement (``--K`` defaults to L: every
mode is lit, and any other count is refused); the radio commands take
``--radio`` (loopback / pluto[:uri] / rtlsdr), ``--freq``, ``--rate`` and ``--gain``.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__, waveform as wf
from .radio import open_radio
from .runner import Link, ber, loopback, random_bits, receive, transmit

ROWS = 500        # rows of payload the demo commands compose, enough for the blind lock to
                  # place the row offset with margin to spare


#: This reader's operating point, as a multiple of the null rather than an absolute closure: the
#: closure degrades with SNR, so a fixed level rejects the weak links a receiver exists for.  See
#: waveform.closure_significance for the measurements.  Declared per paper RF-3.2.
#: measured distributions, which is what places it: over twelve noise realizations the worst
#: impostor reads 3.7x the null and every other tried reads less; paper RF-3.2 has the table
#: 2.30x; a transmission reads 137.7x at 18 dB, 96.1x at 12, 57.5x at 9 and 22.5x at 6, where it
#: decodes at BER 0.095.  Eight sits near the geometric middle of that gap -- 2.2x clear of the
#: worst impostor and 2.8x below the weakest usable signal.  It rejects 3 dB (5.4x), where the
#: payload reads BER 0.217 and is not usable uncoded anyway.
SIGNIFICANCE_CALL = 8.0


def _key(text: str):
    """The key as the waveform takes it: an int if it is written as one, otherwise the text.

    ``int``, ``str`` and ``bytes`` are three different keys -- the derivation is domain-separated
    over the encoded value, so 1234, "1234" and b"1234" produce three unrelated apertures.  Digits
    are read as an integer so that ``--seed 1234`` keeps meaning what it has always meant, and
    anything else is passed through as text, which is what a human-chosen key looks like."""
    try:
        return int(text)
    except (TypeError, ValueError):
        return text


def _link(a) -> Link:
    return Link(K=a.L if a.K is None else a.K, L=a.L, seed=a.seed, M=a.M, sender=a.sender,
                coherent=getattr(a, 'coherent', False))


def cmd_selftest(a) -> int:
    link = _link(a)
    print(f"entroptics-rf {__version__}  (entroptics >= {wf.REQUIRED_ENTROPTICS})")
    print(f"aperture: K={link.K} L={link.L} K/L={link.mode_ratio:.3f} M={link.M} "
          f"-> {link.bits_per_sample:.2f} bits/sample"
          f"{'  sender=%s' % link.sender if link.sender else ''}")

    bits = random_bits(int(round(np.log2(link.M))) * ROWS * link.K, seed=1)
    iq = (wf.compose_qam(bits, link.K, link.L, link.seed, M=link.M) if link.coherent
          else wf.compose(bits, link.K, link.L, link.seed, M=link.M, sender=link.sender))
    o = wf.optics(iq, link.L)
    print(f"composed {len(iq)} samples, no IFFT   PAPR={o.papr_db:.1f} dB   "
          f"fill={o.fill:.4f}  occupied={o.occupied:.1f} of {o.capacity:.0f} DOF")

    sent, rx = loopback(link, bits, snr_db=None, cfo=0.0)
    print(f"clean loopback: lock L={rx.L} (true {link.L}) margin={rx.lock.margin:.1e}  "
          f"BER={ber(sent, rx.bits):.4f}")

    _rd = wf.read_qam if link.coherent else wf.read
    wrong = _rd(iq, link.L, link.K, a.seed + 1, M=link.M)
    print(f"wrong key:      BER={ber(bits, wrong):.4f}  (0.5 = the key is the payload's only lock)")

    print()
    print(f"{'SNR dB':>7} {'lock L':>7} {'margin':>7} {'BER':>9}")
    for snr in (18, 12, 9, 6, 3):
        s, r = loopback(link, bits, snr_db=snr, cfo=a.cfo, seed=snr)
        print(f"{snr:>7} {r.L:>7} {r.lock.margin:>7.0e} {ber(s, r.bits):>9.4f}")
    return 0


def cmd_focus(a) -> int:
    """What discovery sees: the constant-modulus score at every row offset, and what it read
    out."""
    link = _link(a)
    bits = random_bits(int(round(np.log2(link.M))) * ROWS * link.K, seed=1)
    iq = (wf.compose_qam(bits, link.K, link.L, link.seed, M=link.M) if link.coherent
          else wf.compose(bits, link.K, link.L, link.seed, M=link.M, sender=link.sender))
    if a.snr is not None:
        from .radio import LoopbackRadio
        rr = LoopbackRadio(snr_db=a.snr, cfo=a.cfo, seed=0)
        rr.transmit(iq)
        iq = rr.receive(len(iq)).iq
    iq = iq[a.cut:]

    d = wf.discover(iq, link.seed, a.widths and range(*a.widths))
    print(f"discovery  sent K={link.K} L={link.L} M={link.M}"
          f"{'  snr=%g dB' % a.snr if a.snr is not None else '  (clean)'}"
          f"{'  capture cut %d samples in' % a.cut if a.cut else ''}")
    print(f"  read: L={d.L}  offset={d.offset}  K={d.K}  M={d.M}"
          f"   score={d.score:.2e}  margin={d.margin:.1e}  closure={d.closure:.3f}")

    curve = wf._cm_score(iq, d.L or link.L, link.seed, None, 16)
    if curve is not None:
        order = np.argsort(curve)
        print(f"  {'offset':>7} {'modulus score':>17}")
        for o in list(order[:3]) + [int(order[-1])]:
            print(f"  {int(o):>7} {curve[int(o)]:>17.3e}"
                  + ("   <== LOCK" if int(o) == d.offset else ""))

    want = (link.L - a.cut) % link.L
    ok = (d.L, d.offset, d.K, d.M) == (link.L, want, link.K, link.M)
    print(f"{'HIT' if ok else 'MISS'}: everything above was measured from the capture"
          f"{'' if ok else '  (wanted L=%d offset=%d K=%d M=%d)' % (link.L, want, link.K, link.M)}")
    return 0 if ok else 1


def cmd_ber(a) -> int:
    link = _link(a)
    bits = random_bits(int(round(np.log2(link.M))) * ROWS * link.K, seed=1)
    print(f"K={link.K} L={link.L} K/L={link.mode_ratio:.3f} M={link.M} "
          f"({link.bits_per_sample:.2f} bits/sample), cfo={a.cfo}")
    print(f"{'SNR dB':>7} {'lock L':>7} {'margin':>7} {'BER':>9}")
    for snr in [float(s) for s in a.snr_list.split(",")]:
        s, r = loopback(link, bits, snr_db=snr, cfo=a.cfo, seed=int(snr) & 0xFF)
        print(f"{snr:>7g} {r.L:>7} {r.lock.margin:>7.0e} {ber(s, r.bits):>9.4f}")
    return 0


def cmd_embed(a) -> int:
    from .radio import LoopbackRadio
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(a.D) + 1j * rng.standard_normal(a.D)
    iq = wf.compose_embedding(vec, a.seed, spread=a.spread)
    print(f"analog embedding: D={a.D} -> {len(iq)} samples (spread={a.spread})")
    print(f"{'SNR dB':>7} {'cosine':>9}")
    for snr in [float(s) for s in a.snr_list.split(",")]:
        rr = LoopbackRadio(snr_db=snr, seed=int(snr) & 0xFF)
        rr.transmit(iq)
        got = wf.read_embedding(rr.receive(len(iq)).iq, a.D, a.seed)
        cos = abs(np.vdot(got, vec)) / (np.linalg.norm(got) * np.linalg.norm(vec))
        print(f"{snr:>7g} {cos:>9.4f}")
    return 0


def cmd_tx(a) -> int:
    link = _link(a)
    bits = random_bits(a.bits, seed=1)
    radio = open_radio(a.radio, sample_rate=a.rate, center_freq=a.freq)
    try:
        iq = transmit(radio, bits, link, cyclic=a.cyclic)
        print(f"transmitting {len(iq)} samples on {a.radio} at {a.freq/1e6:.3f} MHz, "
              f"{a.rate/1e6:.3f} Msps, PAPR={wf.optics(iq, link.L).papr_db:.1f} dB")
        if a.cyclic:
            input("cyclic transmit running -- press Enter to stop... ")
    finally:
        radio.close()
    return 0


def cmd_rx(a) -> int:
    link = _link(a)
    radio = open_radio(a.radio, sample_rate=a.rate, center_freq=a.freq)
    try:
        rx = receive(radio, link, a.samples, blind=not a.informed)
    finally:
        radio.close()
    print(f"captured {len(rx.capture)} samples at {rx.capture.sample_rate/1e6:.4f} Msps")
    # Whether a screen is there is the closure's answer, not `locked`'s.  `Discovery.locked` only
    # says the sweep returned a width, which it almost always does: pure noise came back as
    # L=138, K=136, M=8 and 8008 bits of nothing, printed under the word "lock" and exiting 0.
    # The geometry score alone is not the evidence a transmission is there: noise always offers
    # some best width (and the dark-mode read this replaced scored a DC level or a bare tone BETTER
    # than a real screen).  The alphabet closure is what separates: 1.000 for a transmission
    # against 0.011-0.023 for noise, DC and tones alike.  The level below is this reader's
    # operating point, declared here and shown with
    # the evidence, in the sense of paper RF-9.1 -- the codec still decides nothing.
    sig = 0.0 if rx.lock is None else wf.closure_significance(rx.lock, len(rx.capture))
    reads_as_signal = rx.lock is None or sig >= SIGNIFICANCE_CALL
    print(f"lock: L={rx.L}"
          + (f" offset={rx.lock.offset} K={rx.lock.K} M={rx.lock.M}"
             f"  score={rx.lock.score:.2e} margin={rx.lock.margin:.1e}"
             f" closure={rx.lock.closure:.3f} ({sig:.0f}x null)"
             if rx.lock else " (told, not measured)"))
    if not reads_as_signal:
        print(f"NO SCREEN -- the geometry above is the sweep's best guess, but the constellation "
              f"closes at {rx.lock.closure:.3f}, only {sig:.1f}x what an unkeyed capture this "
              f"size gives, against {SIGNIFICANCE_CALL}x for a transmission. Nothing is riding "
              f"it; any bits read from this are noise.")
        return 1
    intf = wf.interference(rx.capture.iq, rx.L)
    print(f"optics: etendue={rx.optics.etendue:.4f} PAPR={rx.optics.papr_db:.1f} dB")
    print(f"interference: strehl={intf.strehl:.4f} dominance={intf.dominance:.4f} "
          f"(a flat waveform has no dominant mode, so concentration here is foreign)")
    print(f"read {rx.bits.size} bits")
    if a.out:
        np.save(a.out, rx.capture.iq)
        print(f"capture saved -> {a.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="entroptics_rf", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"entroptics-rf {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--K", type=int, default=None,
                        help="modes carried: L (every mode is lit; the default)")
        sp.add_argument("--L", type=int, default=64, help="screen width")
        sp.add_argument("--seed", type=_key, default=1234,
                        help="THE KEY -- any text; digits are read as an integer")
        sp.add_argument("--M", type=int, default=4,
                        help="alphabet order: differential PSK 2/4/8/16, or square QAM 4/16/64/256/1024")
        sp.add_argument("--coherent", action="store_true",
                        help="coherent square QAM against the keyed reference row (higher rate)")
        sp.add_argument("--sender", default=None,
                        help="mark the transmission with this sender identifier (free)")
        return sp

    def radioargs(sp):
        sp.add_argument("--radio", default="loopback", help="loopback | pluto[:uri] | rtlsdr")
        sp.add_argument("--freq", type=float, default=915e6, help="centre frequency, Hz")
        sp.add_argument("--rate", type=float, default=2e6, help="sample rate, Hz")
        return sp

    s = common(sub.add_parser("selftest", help="the whole waveform on a simulated link"))
    s.add_argument("--cfo", type=float, default=3e-4)
    s.set_defaults(fn=cmd_selftest)

    s = common(sub.add_parser("focus", help="the focus curve the lock actually sees"))
    s.add_argument("--cut", type=int, default=0, help="start the capture this many samples in")
    s.add_argument("--widths", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="bound the width search (default: fully blind)")
    s.add_argument("--snr", type=float, default=None)
    s.add_argument("--cfo", type=float, default=0.0)
    s.set_defaults(fn=cmd_focus)

    s = common(sub.add_parser("ber", help="BER against SNR, blind lock in the loop"))
    s.add_argument("--snr-list", default="18,12,9,6,3", dest="snr_list")
    s.add_argument("--cfo", type=float, default=3e-4)
    s.set_defaults(fn=cmd_ber)

    s = common(sub.add_parser("embed", help="analog embedding fidelity against SNR"))
    s.add_argument("--D", type=int, default=768)
    s.add_argument("--spread", type=float, default=1.0)
    s.add_argument("--snr-list", default="18,12,6,0", dest="snr_list")
    s.set_defaults(fn=cmd_embed)

    s = radioargs(common(sub.add_parser("tx", help="transmit on a radio")))
    s.add_argument("--bits", type=int, default=40_000)
    s.add_argument("--cyclic", action="store_true", help="transmit repeatedly until stopped")
    s.set_defaults(fn=cmd_tx)

    s = radioargs(common(sub.add_parser("rx", help="capture from a radio and read it")))
    s.add_argument("--samples", type=int, default=1 << 18)
    s.add_argument("--informed", action="store_true", help="skip the lock; use the agreed L")
    s.add_argument("--out", default=None, help="save the raw capture to this .npy")
    s.set_defaults(fn=cmd_rx)

    a = p.parse_args(argv)
    try:
        return int(a.fn(a))
    except (RuntimeError, ValueError, ImportError, OSError, FileNotFoundError) as e:
        # The operational failures: no signal, a geometry that cannot exist, a driver that is not
        # installed, a device that is not there.  These are answers, not defects, and a traceback
        # presents them as defects.  Anything else still raises, because a real bug should look
        # like one.
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
