"""
runner.py -- the operations: put a waveform on a radio, take one off, and measure the result.

:mod:`entroptics_rf.waveform` knows the optics and nothing about hardware; :mod:`entroptics_rf.radio`
knows the hardware and nothing about the optics.  This module is the short piece of wiring between
them, plus the measurement loop that makes a claim checkable.

    Link          the transmitter's own choices (L, M) and the one shared thing (seed).  L is
                  in it for the transmitter's benefit; a blind receiver measures L for itself.
                  K = L: every mode is lit.
    transmit      compose the payload and hand the IQ to a radio.
    receive       capture, discover the geometry, read.  Blind by default: the width comes from
                  the signal.
    loopback      the whole link over a simulated channel, in one call.
    ber           bit error rate between what went in and what came out.

Blind is the default, and it costs something.  ``receive`` measures the width from the capture
rather than being told it, which is the point of the waveform, and the measurement is a sweep -- a
caller who genuinely knows the width passes ``blind=False`` and skips it.  The two paths stay
visibly distinct, so a measurement never credits the blind path with a number the informed one
produced.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
from dataclasses import replace

from . import waveform as wf
from .radio import Capture, LoopbackRadio, Radio

__all__ = ["Link", "Reception", "transmit", "receive", "loopback", "ber", "random_bits",
           "StreamTransmitter", "StreamReceiver"]


@dataclass(frozen=True)
class Link:
    """The aperture agreement: what both ends know before anything is transmitted.

    ``K``          modes carried (the capacity): L, every mode lit.  A spare mode would be a
                   noise-only direction an observer without the key reads the width from
                   (research/rank_leak.md); the pilot-free lock reads the coordinates' constant
                   modulus instead, which a full screen keeps.
    ``L``          screen width.  The transmitter sets it; a blind receiver measures it.
    ``seed``       the key.  int or bytes; the full value is used.
    ``M``          differential PSK order (2, 4, 8, 16).
    ``sender``     an optional sender identifier, marked into the waveform as a rotation -- free,
                   and invisible to the payload (see :func:`waveform.sender_rotation`).
    ``coherent``   False (default) = differential PSK, carrier-blind, M in (2, 4, 8, 16).
                   True = square QAM against the keyed reference row, M in (4, 16, 64, 256,
                   1024): ~40% more delivered rate, and the caller removes the row-rate phase of
                   any carrier offset.  See the coherent section of
                   :mod:`entroptics_rf.waveform`.
    """
    K: int
    L: int
    seed: object
    M: int = 4
    sender: object = None
    coherent: bool = False

    def __post_init__(self) -> None:
        if int(self.L) < 1 or int(self.K) != int(self.L):
            raise ValueError(f"every mode is lit: need K == L >= 1, got K={self.K}, L={self.L}")

    @property
    def bits_per_sample(self) -> float:
        """Spectral efficiency of this geometry, bits per complex sample."""
        return wf.bits_per_sample(self.K, self.L, self.M)

    @property
    def mode_ratio(self) -> float:
        """K/L -- 1: every mode is lit."""
        return int(self.K) / int(self.L)

    def capacity(self, n_samples: int) -> int:
        """Payload bits a capture of ``n_samples`` carries at this geometry."""
        return wf.capacity(self.K, self.L, n_samples, self.M)


@dataclass(frozen=True)
class Reception:
    """What came off the radio: the bits, and the evidence for how they were recovered.

    ``lock`` is None when the receiver was told the width instead of measuring it -- so a caller
    can always tell a blind result from an informed one, rather than inferring it."""
    bits: np.ndarray
    L: int
    lock: "wf.Discovery | None"
    optics: "wf.Optics"
    capture: Capture

    @property
    def blind(self) -> bool:
        return self.lock is not None


def random_bits(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible payload for measurements.  Not keyed -- this is test traffic, not a key."""
    return np.random.default_rng(seed).integers(0, 2, int(n)).astype(np.int64)


def transmit(radio: Radio, bits, link: Link, *, cyclic: bool = False) -> np.ndarray:
    """Compose ``bits`` onto the link's aperture and hand the IQ to ``radio``.  Returns the IQ
    sent, so a caller can measure PAPR or keep it as the reference for a loopback."""
    if link.coherent:
        iq = wf.compose_qam(bits, link.K, link.L, link.seed, M=link.M)
    else:
        iq = wf.compose(bits, link.K, link.L, link.seed, M=link.M,
                        sender=link.sender)
    radio.transmit(iq, cyclic=cyclic)
    return iq


def receive(radio: Radio, link: Link, n: int, *, blind: bool = True,
            candidates=None) -> Reception:
    """Capture ``n`` samples, measure the geometry, and read the payload.

    With ``blind=True`` (default) the width and row offset come from :func:`waveform.discover` and
    the link's own ``L`` is used only for the optics report.  With ``blind=False`` the link's
    ``L`` is used directly and no discovery runs."""
    cap = radio.receive(int(n))
    lock = None
    if blind:
        # The receiver has the key, so discovery uses it: the constant-modulus read settles width
        # and offset structurally.  candidates=None takes the fully blind default range; pass a range
        # only to bound the sweep cost.
        lock = wf.discover(cap.iq, link.seed, candidates)
        if not lock.locked:
            return Reception(np.zeros(0, dtype=np.int64), 0, lock,
                             wf.optics(cap.iq, link.L), cap)
        # The payload's width is the lock's (every mode of the width it found), not the link's: a
        # capture of noise locks some width of its own, and reading it at the link's K would ask
        # for more modes than that width has.
        L, off, K = int(lock.L), int(lock.offset), int(lock.K)
    else:
        L, off, K = int(link.L), 0, int(link.K)
    reader = wf.read_qam if link.coherent else wf.read
    bits = reader(cap.iq, L, K, link.seed, M=link.M, offset=off)
    return Reception(bits, L, lock, wf.optics(cap.iq, L), cap)


def loopback(link: Link, bits=None, *, n_bits: int | None = None, snr_db: float | None = None,
             cfo: float = 0.0, delay: float = 0.0, image_dbc: float | None = None,
             seed: int = 0, blind: bool = True, samples: int | None = None) -> tuple:
    """The whole link over a simulated channel.  Returns ``(sent_bits, Reception)``.

    ``n_bits`` sizes a random payload when ``bits`` is not given.  The capture length defaults to
    exactly what was transmitted; pass ``samples`` to capture more or less than one buffer and
    exercise the partial-capture path."""
    if bits is None:
        bits = random_bits(n_bits if n_bits is not None else 2 * 400 * link.K, seed=seed)
    bits = np.asarray(bits)
    radio = LoopbackRadio(snr_db=snr_db, cfo=cfo, delay=delay, image_dbc=image_dbc, seed=seed)
    iq = transmit(radio, bits, link)
    rx = receive(radio, link, samples if samples is not None else len(iq), blind=blind)
    return bits, rx


# ---------------------------------------------------------------------------
# Streaming -- one unbroken screen, the lock paid once
# ---------------------------------------------------------------------------
# A continuous link is not a sequence of captures.  Framing each buffer as its own screen would
# spend a reference row per buffer and re-acquire the lock every time; instead the stream is one
# screen that never ends, and both ends carry the row state across buffer boundaries.  The
# differential mode makes that natural -- row t is row t-1 rotated, so the only state is the last
# row -- and the acquisition cost is paid once for the whole transmission.
#
# What the receiver holds: the sample carry (a buffer rarely ends on a row boundary), the locked
# (L, offset), and the previous row's coordinates.  It does not re-sweep: after ``acquire`` the
# per-buffer cost is one matrix product, which is what makes the link fast.

class StreamTransmitter:
    """Continuous bits -> IQ, as one screen that never ends.

    ``send`` emits IQ for as many whole rows as the bits fill and keeps the remainder, so a caller
    can push arbitrary-sized chunks and the row grid stays intact across all of them."""

    def __init__(self, link: Link, *, rng=None) -> None:
        if link.coherent:
            raise ValueError("streaming carries the differential mode; the coherent mode needs a "
                             "per-buffer reference row and is a captured-frame mode")
        self.link = link
        self._bits = np.zeros(0, dtype=np.int64)
        self._a = None                     # the running differential row (the whole state)
        self._nonce = None
        self._drow = 0                     # absolute differential row, for the keyed rotation
        # the open mode's per-row power is the sender's own draw (wf.row_radius); keyed, unused
        self._rng = np.random.default_rng(rng)
        self.rows_sent = 0

    def send(self, bits) -> np.ndarray:
        """Push bits; get back the IQ for every whole row they complete (possibly empty)."""
        link = self.link
        b = wf._bits_per_symbol(link.M)
        const = wf._psk(link.M)
        self._bits = np.concatenate([self._bits, np.asarray(bits).ravel().astype(np.int64)])
        per_row = link.K * b
        rows = self._bits.size // per_row
        if rows == 0:
            return np.zeros(0, dtype=np.complex64)
        use, self._bits = self._bits[:rows * per_row], self._bits[rows * per_row:]
        w = (1 << np.arange(b - 1, -1, -1)).astype(np.int64)
        d = const[(use.reshape(-1, b) * w).sum(1)].reshape(rows, link.K)

        first = self._a is None
        A = np.empty((rows + 1, link.K), dtype=complex)
        # Keyed, never all-ones: an all-ones opening row would put the plain sum of the aperture
        # rows on the air, and it would be the same row on every stream this key ever opens.
        A[0] = wf._reference_row(link.K, link.seed, self._nonce) if first else self._a
        # The keyed per-mode rotation is indexed by the absolute differential row, so a stream
        # carries its own index across buffers -- the same keystream a one-shot compose uses.
        d = d * wf.mode_keys(link.K, link.seed, self._nonce,
                             first=self._drow, rows=int(d.shape[0]))
        self._drow += int(d.shape[0])
        np.cumprod(d, axis=0, out=A[1:])
        A[1:] *= A[0][None, :]
        self._a = A[-1]
        out = A if first else A[1:]         # after the first call the seam row was already sent
        S = out @ wf.mode_basis(link.K, link.L, link.seed, self._nonce)
        # Each row at its own power, indexed by the absolute screen row (the reference is row 0),
        # so a stream carries the same radii a one-shot compose of the same rows would.  The walk
        # above stays unit modulus: the radius scales what goes on the air, not the state.
        S = wf._radiate(S, link.L, link.seed, self._nonce, first=self.rows_sent, rng=self._rng)
        self.rows_sent += out.shape[0]
        return wf.to_iq(S, unit_rms=False).astype(np.complex64)


#: A reader's operating point for "is a transmission present at all", as a multiple of the null
#: rather than an absolute closure.  `Discovery.locked` only says the sweep returned a width, and
#: it almost always does: pure noise acquires a width (measured on the dark read: L=66, K=65, and
#: then 54240 bits of nothing).  The closure separates them, though not at a fixed level: it
#: degrades with SNR (1.000
#: clean, 0.611 at 12 dB, 0.143 at 6), so a level high enough to reject a tone also rejects a
#: 6 dB transmission decoding at BER 0.093.  Against the null's own scale the ratio is stable
#: across SNR and across capture size (see wf.closure_significance).
#:
#: Eight sits near the geometric middle of the measured gap -- 2.2x clear of the worst impostor
#: and 2.8x below the weakest usable signal, which decodes at BER 0.095.  It rejects 3 dB (5.4x),
#: where the payload reads BER 0.217 and is not usable uncoded anyway.  Paper RF-3.2 carries the
#: full distribution.  Declared here and overridable by the caller.
ACQUIRE_SIGNIFICANCE = 8.0


class StreamReceiver:
    """Acquire the lock once, then decode buffer after buffer at matrix-product cost.

    ``acquire`` runs discovery on a priming capture and holds ``(L, offset)``; ``feed`` then
    costs one projection per buffer.  That split is the point: the sweep is the expensive part of
    this waveform, and a streaming link pays it once rather than per buffer."""

    def __init__(self, link: Link) -> None:
        self.link = link
        self.L = None
        self.offset = 0
        self.lock = None
        self._carry = np.zeros(0, dtype=np.complex64)
        self._nonce = None                 # no nonce is carried; see waveform's note
        self._drow = 0                     # absolute differential row, for the keyed rotation
        self._prev = None                  # previous row's coordinates -- the differential seam
        self.rows_read = 0

    @property
    def locked(self) -> bool:
        return self.L is not None

    def acquire(self, iq, candidates=None, *,
                min_significance: float = ACQUIRE_SIGNIFICANCE,
                join_rows: int = wf.JOIN_SEARCH_ROWS) -> bool:
        """Lock the width and the row offset from a priming capture.  Paid once.

        Keyed, so the offset is settled structurally rather than statistically -- which matters
        for a stream, where being one sample out costs every bit that follows.

        How much of the capture to use is not the caller's to decide.  Hand it the buffer; it
        reads the prefix it needs and stops.  Two things had to be understood before that worked,
        and five earlier attempts failed on them:

        *The candidate range must come from the whole buffer.*  ``discover`` draws its widths from
        ``isqrt(len(iq))``, so handing it a short prefix truncates the search and the read returns
        whatever sits at the top of what is left -- the first 64 samples of a real transmission
        returned a convinced L=8, and 128 and 256 samples both returned L=11 with offset 7,
        agreeing with each other on nothing.  The answers were never the problem; the range was.
        Derive the range once from everything available and widen only the prefix under it, and
        the width and the row offset settle at 256 samples on cuts of 1, 17 and 63.

        *A row0 of zero is two different answers.*  The alignment needs about sixty-four rows to
        resolve -- measured, at every join depth from 0 to 1500 and every SNR down to 6 dB, while
        the geometry resolves at four -- and until it does it reports row 0, which is also what a
        capture that genuinely began on row 0 reports.  A rule that stops when the answer repeats
        therefore stopped on the run of defaults, took row 0 for a stream that joined at 1500, and
        decoded correct geometry at BER 0.50.  :func:`~entroptics_rf.waveform._align_and_close`
        now says whether it settled, so the two can be told apart.

        Widening therefore continues until the read both settles its alignment and says the same
        thing twice.  A clean capture pays two or three reads of a few hundred rows; a marginal one
        keeps widening until the buffer runs out, which is a fact about the data rather than a
        bound anyone set.

        Returns True once locked.  A stream that has not yet carried enough returns False and is
        fed again -- the natural streaming shape, since a receiver joining a live stream has only
        what has arrived so far."""
        link = self.link
        body = np.asarray(iq)
        if candidates is None:
            hi = max(2, min(256, math.isqrt(len(body))))
            candidates = range(2, hi + 1)          # from the whole buffer, never from the prefix
        take, prev, lock, seen = int(wf.MIN_ROWS) ** 2, None, None, len(body)
        geom, prev_geo = None, None
        while True:
            trial = body if take >= len(body) else body[:take]
            # Once the width has settled, stop re-sweeping for it.  The geometry converges at four
            # rows and the alignment at about sixty-four, so every prefix in between pays a full
            # 255-width sweep to re-derive a width it already knows -- measured, 9.5 s of a 17.9 s
            # acquisition spent on prefixes too short to align at all.  Narrowing the candidates
            # to the settled width is the same move discover makes internally when it widens only
            # the sweep, applied one layer out.
            lock = wf.discover(trial, link.seed, geom or candidates,
                               join_rows=join_rows)
            seen = len(trial)          # the closure came from THIS much, not from the whole buffer
            # Narrow only once the width has said the same thing TWICE.  Narrowing on the first
            # lock pins whatever a 64-sample prefix invented -- measured, L=25 -- and every later
            # read is then confined to it.  The geometry has its own agreement rule, one doubling
            # ahead of the alignment's.
            # And only when both reads that agree held MIN_ROWS rows of that width: a width is
            # evidenced by its rows, and on the constant-modulus read the first 64 and 128 samples
            # of a real transmission both read L=16 (one and two rows of the true 64, re-cut), so
            # narrowing on their agreement pinned a width the 256-sample read had already corrected.
            wide = (int(lock.L), int(lock.offset)) if lock.locked else None
            if (geom is None and wide is not None and wide == prev_geo
                    and len(trial) >= 2 * int(wf.MIN_ROWS) * wide[0]):
                geom = range(wide[0], wide[0] + 1)
            prev_geo = wide
            here = ((lock.L, lock.offset, lock.K, lock.M, lock.row0)
                    if lock.locked and lock.aligned else None)
            if here is not None and here == prev:
                break
            if take >= len(body):
                break
            prev, take = here, take * 2
        self.lock = lock
        if not lock.locked:
            return False
        # Measure the closure on EVERYTHING, at the geometry just found.  Not on the prefix the
        # widening stopped at: that prefix is minimal by construction -- it stops as soon as the
        # geometry and the alignment settle -- so its closure is poor even when the link is fine,
        # and testing there rejected a 6 dB stream that decodes at BER 0.094.  The sweep is done
        # by now, so this costs one projection.
        full = wf.read_soft(body, lock.L, lock.K, link.seed, offset=lock.offset,
                            phase=lock.phase, carrier=lock.carrier, row0=lock.row0)
        if full.size:
            u = full / np.maximum(np.abs(full), 1e-30)
            lock = replace(lock, closure=float(abs(np.mean(u ** int(lock.M)))))
        if wf.closure_significance(lock, len(body)) < float(min_significance):
            # A width came back, but nothing is riding it.  Returning True here is how a stream of
            # noise became 54240 bits of noise: the sweep always finds *a* geometry, and only the
            # closure says whether a transmission is on it -- measured against the null, because
            # an absolute closure falls with SNR and would reject the weak links this exists for.
            return False
        self.L, self.offset = int(lock.L), int(lock.offset)
        # A stream acquired mid-transmission has lost whole rows, and the keyed rotation is
        # indexed by the absolute one.  ``discover`` measured which row this capture began on
        # (Discovery.row0), so the stream starts its keystream there rather than at zero.
        self._drow = int(lock.row0)
        # The carry starts at the measured offset, so every row the decoder sees is a whole row
        # of the transmitter's grid.
        self._carry = np.asarray(iq, dtype=np.complex64)[self.offset:]
        self._prev = None
        return True

    def feed(self, iq=None) -> np.ndarray:
        """Decode every whole row now available; carry the rest.  Returns bits (possibly empty).

        The first row of a stream is the differential reference and yields no bits; every row
        after it does, including the ones that straddle a buffer boundary -- which is the whole
        reason the previous row's coordinates are held."""
        if not self.locked:
            raise RuntimeError("acquire() first: nothing is locked")
        link = self.link
        if iq is not None:
            self._carry = np.concatenate([self._carry, np.asarray(iq, dtype=np.complex64)])
        L = self.L
        rows = self._carry.size // L
        if rows < 1:
            return np.zeros(0, dtype=np.int64)
        S = self._carry[:rows * L].reshape(rows, L)
        self._carry = self._carry[rows * L:]
        K = int(self.lock.K)               # every mode of the locked width (= link.K on a real lock)
        # The aperture at the sampling phase and the carrier acquire() measured, as read_soft reads
        # it: the carrier's per-column half is a ramp on the mode index, taken here, and its
        # per-row half rotates the differential, taken below.  Projecting at carrier zero instead
        # leaves the per-column ramp as crosstalk between modes, which costs BER from about 0.03
        # cycles per row.
        phase, carrier = float(self.lock.phase), float(self.lock.carrier)
        A = wf.project(S, wf.mode_basis(K, L, link.seed, self._nonce, phase, carrier))
        self.rows_read += rows
        if self._prev is not None:
            A = np.vstack([self._prev[None, :], A])
        self._prev = A[-1]
        if A.shape[0] < 2:
            return np.zeros(0, dtype=np.int64)
        d = A[1:] * np.conj(A[:-1])
        if d.shape[0]:                     # take the keyed per-mode rotation back off
            d = d * np.conj(wf.mode_keys(K, link.seed, self._nonce,
                                         first=self._drow, rows=int(d.shape[0])))
            self._drow += int(d.shape[0])
            if carrier:                    # the row-rate half: every row advances by 2.pi.carrier
                d = d * np.exp(-2j * np.pi * carrier)
        b = wf._bits_per_symbol(link.M)
        idx = wf._slice_psk(d, link.M).ravel()
        return ((idx[:, None] >> np.arange(b - 1, -1, -1)) & 1).astype(np.int64).ravel()


def ber(sent, got) -> float:
    """Bit error rate over the overlap.  Returns 1.0 when nothing came back, so a failed lock
    scores as a total loss rather than as a suspiciously good 0.0 over zero bits."""
    sent = np.asarray(sent).ravel()
    got = np.asarray(got).ravel()
    n = min(sent.size, got.size)
    if n == 0:
        return 1.0
    return float(np.mean(sent[:n] != got[:n]))
