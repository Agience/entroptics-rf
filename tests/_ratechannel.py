"""
A transmitter and a receiver on different clocks, for the tests.

The transmitted samples are a cyclic buffer, and a converter turns them into the band-limited
signal they are samples of.  The receiver samples that signal at its own rate: sample n lands at
transmitted position ``n * rho + delay``, rho = tx_rate / rx_rate.  For a cyclic buffer the
band-limited continuation is exact -- the trigonometric interpolation of one period -- so this
channel adds nothing the radios do not: no filter, no window, no approximation beyond quantizing
each position to 1/Q of a transmitted sample.

This is test infrastructure modelling a converter.  It lives under ``tests/`` because the package
holds no transform (``test_no_fft.py``); a model of the physical channel is not the codec.
"""
from __future__ import annotations

import numpy as np

Q = 1024          # positions quantized to 1/Q of a transmitted sample


def at_rate(tx, rho: float, n: int, *, delay: float = 0.0, carrier: float = 0.0,
            snr_db: float | None = None, seed: int = 0) -> np.ndarray:
    """``n`` receiver samples of the cyclic transmission ``tx`` taken at ``rho`` transmitted samples
    per received one, starting at transmitted position ``delay``.  ``carrier`` is the oscillator
    difference in cycles per received sample."""
    tau = np.arange(int(n)) * float(rho) + float(delay)
    y = at_positions(tx, tau)
    if carrier:
        y = y * np.exp(2j * np.pi * float(carrier) * np.arange(y.size))
    return _awgn(y, snr_db, seed)


def at_positions(tx, tau) -> np.ndarray:
    """The band-limited cyclic continuation of ``tx`` sampled at transmitted positions ``tau``
    (any real sequence, not only a uniform grid)."""
    tx = np.asarray(tx, dtype=complex)
    tau = np.asarray(tau, dtype=float)
    N = tx.size
    X = np.fft.fft(tx)
    k = np.fft.fftfreq(N) * N                                   # signed bins
    q = np.rint(tau * Q).astype(np.int64)
    whole, frac = np.divmod(q, Q)
    y = np.empty(tau.size, dtype=complex)
    for f in np.unique(frac):
        ph = np.exp(2j * np.pi * k * (f / Q) / N)
        if N % 2 == 0:
            ph[N // 2] = np.cos(np.pi * f / Q)                  # the Nyquist bin of a real converter
        shifted = np.fft.ifft(X * ph)
        sel = frac == f
        y[sel] = shifted[whole[sel] % N]
    return y


def _awgn(y, snr_db, seed):
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p = float(np.mean(np.abs(y) ** 2))
        s = np.sqrt(p / (2.0 * 10.0 ** (float(snr_db) / 10.0)))
        y = y + s * (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size))
    return y


C = 299_792_458.0     # m/s


def moving(tx, n: int, *, fc: float, rate: float, range_rate: float, accel: float = 0.0,
           osc_ppm: float = 0.0, delay: float = 0.0, rho: float = 1.0,
           snr_db: float | None = None, seed: int = 0) -> np.ndarray:
    """``n`` receiver samples of ``tx`` from a sender whose range is ``R(t) = R0 + range_rate*t
    + accel*t^2/2`` (m, s; positive range_rate = opening).

    Narrowband propagation, nothing linearized: receiver sample ``i`` at ``t = i/rate`` sees the
    transmission emitted at ``t - (R(t) - R0)/c`` -- the code Doppler, a time-varying sample-rate
    scale -- and carries the carrier phase ``-2.pi.fc.(R(t) - R0)/c``, which is the Doppler shift
    ``-fc.range_rate/c`` plus a chirp ``-fc.accel.t/c``.  ``osc_ppm`` adds the two oscillators'
    difference at ``fc``, a constant offset the receiver cannot tell from Doppler on one burst.
    ``rate`` is the receiver's rate; ``rho`` the sender's over the receiver's, as for :func:`at_rate`."""
    t = np.arange(int(n)) / float(rate)
    dR = float(range_rate) * t + 0.5 * float(accel) * t * t
    tau = (t - dR / C) * float(rate) * float(rho) + float(delay)
    y = at_positions(tx, tau)
    phase = -2.0 * np.pi * float(fc) * dR / C + 2.0 * np.pi * float(fc) * osc_ppm * 1e-6 * t
    return _awgn(y * np.exp(1j * phase), snr_db, seed)


def row_start(t: int, L: int, rho: float, delay: float) -> float:
    """Receiver position where transmitted row ``t`` begins."""
    return (t * L - float(delay)) / float(rho)
