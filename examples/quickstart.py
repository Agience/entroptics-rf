"""
entroptics-rf quickstart -- a runnable "hello world" for the physical layer.

Run it:   PYTHONPATH=src python examples/quickstart.py
          (or just `python examples/quickstart.py` once the package is pip-installed)

The whole claim of entroptics-rf, shown three ways, with no radio hardware (a simulated loopback
channel stands in for the air):

  1. PRIVATE, BLIND.  The transmitter picks a width, an alphabet and a carrier.  The receiver is
     told ONLY the key -- not the width, not the carrier, not the alphabet, not the code rate -- and
     MEASURES every one of them from the capture, confirmed by the constant-modulus lock.  A
     parameter the receiver must be told is a parameter the codec failed to measure, so there are
     none.  (No FFT, no pilot, no preamble, no cyclic prefix anywhere in the path.)

  2. OPEN BROADCAST.  With no key at all (seed=None, the public zero-state aperture) the same
     capture is contiguous data anyone who knows the mathematics can read.

  3. THE KEY GATES READING.  The one thing that can be agreed in advance is the key, and only if the
     transmission is meant to be private.  Decode a keyed capture with the WRONG key and the lock
     collapses -- the payload does not come out.

For a sender on a different clock or sample rate, the same receiver is `decode_scaled` (it reads the
scale and start continuously); see the README and research/ for the cross-clock measurements.
"""
from __future__ import annotations

from entroptics_rf import Link, ber, decode, loopback, random_bits, transmit
from entroptics_rf.radio import LoopbackRadio

KEY = 0xA17C0DE                 # the shared key (int or bytes); the ONLY thing agreed in advance
WIDTHS = range(60, 69)          # a width-search hint so the demo stays quick (L is still measured)


def run_quickstart(verbose: bool = True) -> tuple[bool, dict]:
    facts: dict = {}

    # 1. PRIVATE, BLIND -- over a noisy channel with a carrier offset the receiver has not been told.
    # A fully blind width search (told nothing, not even the width range) needs enough rows to pin
    # the screen width from the capture alone -- ~80+; 12000 QPSK bits is ~90 rows.
    link = Link(K=64, L=64, seed=KEY, M=4)                  # the transmitter's choice: 64-wide, QPSK
    sent, rx = loopback(link, n_bits=12000, snr_db=12.0, cfo=3e-4, blind=True)
    private_ber = ber(sent, rx.bits)
    facts["private_ber"] = private_ber
    facts["width_measured"] = rx.L                          # measured from the capture, never told
    facts["lock_margin"] = None if rx.lock is None else rx.lock.margin

    # 2. OPEN BROADCAST -- no key at all; the same scheme, publicly readable.
    open_link = Link(K=64, L=64, seed=None, M=4)
    sent_o, rx_o = loopback(open_link, n_bits=12000, snr_db=12.0, blind=True)
    facts["open_ber"] = ber(sent_o, rx_o.bits)

    # 3. THE KEY GATES READING -- one keyed capture, read with the right key then a wrong one.
    bits = random_bits(4000, seed=1)
    iq = transmit(LoopbackRadio(), bits, Link(K=64, L=64, seed=KEY, M=4))
    right_bits, d_right = decode(iq, KEY, WIDTHS)
    wrong_bits, d_wrong = decode(iq, KEY ^ 0xFFFF, WIDTHS)
    facts["right_key_ber"] = ber(bits, right_bits)
    facts["wrong_key_ber"] = ber(bits, wrong_bits)
    facts["right_key_closure"] = float(d_right.closure)
    facts["wrong_key_closure"] = float(d_wrong.closure)

    ok = (private_ber < 1e-2 and facts["open_ber"] < 1e-2
          and facts["right_key_ber"] < 1e-2 and facts["wrong_key_ber"] > 0.3)

    if verbose:
        print("1. PRIVATE, BLIND (told only the key):")
        print(f"     width measured     : {facts['width_measured']}  (never told -- the transmitter chose it)")
        print(f"     lock margin        : {facts['lock_margin']:.3g}  (the constant-modulus evidence)")
        print(f"     BER                : {private_ber:.4f}")
        print("2. OPEN BROADCAST (no key at all):")
        print(f"     BER                : {facts['open_ber']:.4f}  (readable by anyone)")
        print("3. THE KEY GATES READING (one keyed capture, two keys):")
        print(f"     right key -> BER {facts['right_key_ber']:.4f}, lock closure {facts['right_key_closure']:.3f}  (locks, payload out)")
        print(f"     wrong key -> BER {facts['wrong_key_ber']:.4f}, lock closure {facts['wrong_key_closure']:.3f}  (no lock, no payload)")
        print(f"\noverall: {'OK' if ok else 'FAILED'}")
    return ok, facts


if __name__ == "__main__":
    ok, _ = run_quickstart()
    raise SystemExit(0 if ok else 1)
