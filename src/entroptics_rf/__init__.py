"""
entroptics-rf -- the RF signal is an entropy-matched screen.

A seed-keyed optical aperture carries the payload in its coherent modes, every one of them lit.
The receiver finds the screen again by sweeping candidate reshapes and reading how evenly each
row's energy spreads over the keyed modes -- unit-modulus coordinates do so only at the true
geometry.  No FFT and no DSP anywhere in the path -- not in the write, not in the acquisition, not
in the read.

    import numpy as np
    from entroptics_rf import Link, loopback, ber

    link = Link(K=64, L=64, seed=1234)          # the aperture agreement: all L modes, keyed
    sent, rx = loopback(link, n_bits=40_000, snr_db=12, cfo=3e-4)
    rx.L, rx.lock.margin, ber(sent, rx.bits)    # the width was measured, not told

Three layers, and they do not leak into each other:

    waveform    the entroptics adapter -- writes and reads the IQ.  The only place the optics
                library is touched, and the only place the RF-to-screen ravel happens.
    radio       radio controls -- simulated loopback, ADALM-Pluto, RTL-SDR.  Raw IQ in and out.
    runner      the wiring and the measurement loop.

entroptics is the domain-agnostic optics engine and stays that way: no IQ and no RF term ever
crosses into it.
"""
from .waveform import (                                        # noqa: F401
    Aperture, REQUIRED_ENTROPTICS, MIN_ROWS,
    screen, to_iq, Optics, optics,
    mode_basis, spread_operator, project,
    sender_rotation, sender_signature, sender_match,
    compose, read, compose_qam, read_qam,
    Interference, interference,
    constant_modulus_read, Discovery, discover, decode, decode_scaled,
    compose_embedding, read_embedding,
    capacity, bits_per_sample,
)
from .radio import (                                           # noqa: F401
    Radio, LoopbackRadio, PlutoRadio, RtlSdrRadio, open_radio, Capture,
)
from .runner import (                                          # noqa: F401
    Link, Reception, transmit, receive, loopback, ber, random_bits,
    StreamTransmitter, StreamReceiver,
)

__version__ = "0.1.0"

__all__ = [
    # the entroptics adapter -- write and read the IQ waveform
    "Aperture", "REQUIRED_ENTROPTICS", "MIN_ROWS",
    "screen", "to_iq", "Optics", "optics",
    "mode_basis", "spread_operator", "project",
    "sender_rotation", "sender_signature", "sender_match",
    "compose", "read", "compose_qam", "read_qam",
    "Interference", "interference",
    "constant_modulus_read", "Discovery", "discover", "decode", "decode_scaled",
    "compose_embedding", "read_embedding",
    "capacity", "bits_per_sample",
    # radio controls
    "Radio", "LoopbackRadio", "PlutoRadio", "RtlSdrRadio", "open_radio", "Capture",
    # operations
    "Link", "Reception", "transmit", "receive", "loopback", "ber", "random_bits",
    "StreamTransmitter", "StreamReceiver",
    "__version__",
]
