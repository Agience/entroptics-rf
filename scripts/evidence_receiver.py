"""Claim 1 -- an entroptics receiver sees and decodes the waveform; without the key it cannot.

There are two modes:

  public   composed with ``seed=None``.  The aperture is the zero state: the same
           domain-separated SHAKE-256 stream a keyed aperture is drawn from, with the seed field
           empty, which anyone can derive from the spec -- no key, no table, no agreement.
           Anyone running entroptics reads it, and nobody else does.
  keyed    composed with a seed.  Only the seed holder reads it; a public reader gets 0.5.

In both modes the receiver is handed the key (or nothing, for public) and nothing else.  L,
offset, K and M are all measured from the capture by ``discover``, which is rule 3.

    python scripts/evidence_receiver.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve()
while not (_root / "src" / "entroptics_rf").is_dir():        # the repo, found not assumed
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))

from entroptics_rf import waveform as wf                             # noqa: E402
from entroptics_rf.radio import LoopbackRadio                        # noqa: E402
from entroptics_rf.runner import ber, random_bits                    # noqa: E402


def trace(bits, K, L, seed, M, snr, cfo, samples):
    iq = wf.compose(bits, K, L, seed, M=M)
    r = LoopbackRadio(sample_rate=2e6, snr_db=snr, cfo=cfo, seed=7)
    r.transmit(iq)
    return r.receive(samples).iq


def decode(iq, bits, seed, K, L, M):
    """Blind: discover the geometry with this key alone, then read.  Returns (discovery, BER)."""
    d = wf.discover(iq, seed)
    if not d.L:
        return d, float("nan")
    g = wf.read(iq, d.L, d.K, seed, M=d.M, offset=d.offset)
    n = min(len(bits), len(g))
    return d, ber(bits[:n], g[:n])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, default=None, help="modes carried; every mode is lit: L")
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--rows", type=int, default=400)
    p.add_argument("--samples", type=int, default=1 << 14)
    p.add_argument("--cfo", type=float, default=3e-4)
    p.add_argument("--wrong", type=int, default=4, help="wrong keys tried per point")
    p.add_argument("--snr-list", default="clean,18,12,9,6,3", dest="snr_list")
    a = p.parse_args(argv)

    K, L, M = (a.L if a.K is None else a.K), a.L, a.M
    bits = random_bits(int(round(np.log2(M))) * a.rows * K, seed=1)
    snrs = [None if t == "clean" else float(t) for t in a.snr_list.split(",")]
    tags = a.snr_list.split(",")

    print(f"sent: K={K} L={L} M={M}  cfo={a.cfo}  {a.samples} samples/trace")
    print("the receiver gets the key and NOTHING else; L, offset, K, M are all measured\n")

    print("=== KEYED (seed=%d): the holder reads it, nobody else does ===" % a.seed)
    print(f"{'SNR':>6} | {'HOLDER  L':>9} {'off':>4} {'K':>4} {'M':>3} {'closure':>8} {'BER':>7}"
          f" | {'WRONG KEY closure':>17} {'BER':>7} {'found geom':>11}")
    print("-" * 100)
    for tag, snr in zip(tags, snrs):
        iq = trace(bits, K, L, a.seed, M, snr, a.cfo, a.samples)
        d, b = decode(iq, bits, a.seed, K, L, M)
        clo, bw, hit = [], [], 0
        for j in range(a.wrong):
            wrong = a.seed + 1 + j
            dw = wf.discover(iq, wrong)
            clo.append(dw.closure)
            hit += int(dw.L == L and dw.offset == d.offset)
            # hand the wrong key the TRUE geometry for free; withhold only the key itself
            gw = wf.read(iq, L, K, wrong, M=M, offset=d.offset)
            n = min(len(bits), len(gw))
            bw.append(ber(bits[:n], gw[:n]))
        print(f"{tag:>6} | {d.L:>9} {d.offset:>4} {d.K:>4} {d.M:>3} {d.closure:>8.3f} {b:>7.4f}"
              f" | {np.mean(clo):>17.3f} {np.mean(bw):>7.4f} {hit:>4}/{a.wrong:<6}")

    print("\n=== PUBLIC (seed=None): anyone running entroptics reads it, with no key ===")
    print(f"{'SNR':>6} | {'ANYONE  L':>9} {'off':>4} {'K':>4} {'M':>3} {'closure':>8} {'BER':>7}"
          f" | {'key-holder closure':>18} {'public-on-keyed BER':>19}")
    print("-" * 100)
    for tag, snr in zip(tags, snrs):
        iq = trace(bits, K, L, None, M, snr, a.cfo, a.samples)
        d, b = decode(iq, bits, None, K, L, M)
        dk = wf.discover(iq, a.seed)                    # public capture, someone using a key
        iqk = trace(bits, K, L, a.seed, M, snr, a.cfo, a.samples)
        gk = wf.read(iqk, L, K, None, M=M, offset=0)    # keyed capture, public reader, true geom
        n = min(len(bits), len(gk))
        print(f"{tag:>6} | {d.L:>9} {d.offset:>4} {d.K:>4} {d.M:>3} {d.closure:>8.3f} {b:>7.4f}"
              f" | {dk.closure:>18.3f} {ber(bits[:n], gk[:n]):>19.4f}")

    print("\nWRONG KEY and public-on-keyed BER are measured with the TRUE L, offset, K and M")
    print("handed over free.  0.5 means the decode carries no information about the payload.")
    print("closure is the receiver's own confidence that a screen is present at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
