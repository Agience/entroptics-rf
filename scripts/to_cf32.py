"""Convert a saved capture to the raw interleaved float32 file inspectrum reads.

``python -m entroptics_rf rx --out cap.npy`` writes a numpy array; inspectrum wants headerless
interleaved I,Q float32.  This strips the 128-byte header.  The .cf32 extension is what makes
inspectrum pick complex<float> by itself.

inspectrum 0.2.3 (Ubuntu jammy) does not read SigMF, so type the sample rate into its Sample rate
box.  ``--rate`` still writes the sidecar, because .npy carries no metadata and the sidecar is the
only place the rate and centre frequency of a capture survive.

    python scripts/to_cf32.py cap.npy --rate 2e6 --freq 915e6
    inspectrum cap.cf32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_cf32(iq, dst: Path, rate: float | None = None,
               freq: float | None = None) -> Path:
    """The one place a trace is written, so every tool lays out bytes identically."""
    dst = Path(dst)
    np.asarray(iq).ravel().astype(np.complex64).tofile(dst)

    if rate is not None:
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": float(rate),
                "core:version": "1.0.0",
            },
            "captures": [{"core:sample_start": 0,
                          **({"core:frequency": float(freq)} if freq is not None else {})}],
            "annotations": [],
        }
        dst.with_suffix(".sigmf-meta").write_text(json.dumps(meta, indent=2))
    return dst


def to_cf32(src: Path, dst: Path | None = None,
            rate: float | None = None, freq: float | None = None) -> Path:
    iq = np.load(src) if src.suffix == ".npy" else np.fromfile(src, dtype=np.complex64)
    return write_cf32(iq, dst or src.with_suffix(".cf32"), rate, freq)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="capture saved by `rx --out`")
    p.add_argument("-o", "--out", type=Path, default=None, help="default: <src>.cf32")
    p.add_argument("--rate", type=float, default=None, help="sample rate in Hz, e.g. 2e6; also writes the SigMF sidecar")
    p.add_argument("--freq", type=float, default=None, help="centre frequency in Hz, for the SigMF sidecar")
    a = p.parse_args(argv)

    dst = to_cf32(a.src, a.out, a.rate, a.freq)
    n = dst.stat().st_size // 8
    print(f"{dst}  {n} samples, complex float32"
          + (f", {a.rate/1e6:.4f} Msps" if a.rate else "  (type the sample rate into inspectrum)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
