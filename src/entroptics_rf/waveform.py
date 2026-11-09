"""
waveform.py -- the writer and the reader: a strongly-typed RF-domain adapter over entroptics.

The RF signal is an entropy-matched screen.  The library does the optics; this wrapper owns the
one reduction between the two domains -- the ravel.  entroptics is strictly 2-D and IQ is 1-D, so
the adapter hands every read the ``(T, L)`` reshape that is the screen.

Nothing is preprocessed.  No filtering, no resampling, no equalization, no windowing, no
transform: the capture reaches the library exactly as the radio delivered it.  There is no FFT
anywhere in this package -- not in the write, not in the acquisition, not in the read -- because
the feature axis is time, a role in the optics rather than a physical quantity, so composing a
screen costs a matrix product and reading one costs its adjoint.  ``tests/test_no_fft.py``
enforces that on the source, since a single convenience import would end it.

    WRITE   compose(bits, K, L, seed)       bits -> keyed mode coordinates A -> S = diag(r) A B,
                                            r the per-row radius -> S.ravel() is the IQ.  No IFFT.
            compose_qam(bits, K, L, seed)   square QAM on the keyed modes, for rate
            compose_embedding(vec, seed)    a semantic vector, carried analog
            compose(..., beacon=True)       carries the row it rides on, so a receiver joining
                                            late reads the row instead of searching for it

    READ    discover(iq, seed)              -> the geometry and the alphabet, measured
            decode(iq, seed)                -> discover, then read: a capture and a key in,
                                               bits out
            read(iq, L, K, seed)            -> S B^H -> differential decode -> bits
            read_qam(iq, L, K, seed)        -> the coherent decode, against the reference row
            read_embedding(iq, seed)        -> the keyed adjoint; D measured, or given
            optics(iq, L)                   -> the named optical reads of a capture

Optics-to-RF dictionary -- each wrapper read maps to a named library read [E] and the RF quantity
it carries:

    Optics.fill        <- Aperture.phi_F            [E Sec 3.3]  feature-axis fill
    Optics.reach       <- Aperture.magnification    [E Sec 3.1]  1/fill: oversampling of the modes
    Optics.etendue     <- Aperture.etendue          [E Sec 3.3]  occupied phase-space area
    Optics.capacity    <- Aperture.space_bandwidth  [E Sec 3.3]  degrees of freedom the screen can
                                                                 carry
    Optics.occupied    <- etendue * space_bandwidth [E Sec 3.3]  the spots actually filled (~K)
    Optics.strehl      <- Aperture.strehl           [E Lem 3.4]  power in the dominant coherent mode
    Optics.dominance   <- Aperture.dominance        [E Sec 6]    leading mode's excess over the mean
    Optics.resolved    <- Aperture.resolved_modes   [E Def 8.2]  modes above the derived noise floor
    Optics.papr_db     <- (none)                                 a transmitter constraint, not an
                                                                 optical read

These are diagnostics on a capture, read at a width the caller supplies.  ``fill`` is 2^H
normalized by L, so it compares captures at one width rather than widths against each other.

How the geometry is found.  Every mode is lit: a transmission uses all L modes of the keyed
aperture, and only a message's final row may use fewer (the first k, the aperture being nested).
The coordinates a row carries are unit modulus (or a square lattice's few levels), so at the true
width, row offset, scale and carrier each row's energy is spread evenly over the keyed modes; at
any other cut the rows are re-cut across mode boundaries, the coordinates are mixtures close to
complex Gaussian, and their energies scatter like exponentials.  :func:`discover` reads that
difference at each candidate width -- the constant-modulus read, each row's entropy deficit over
its mode energies against the flat Dirichlet's (see the section above :func:`_cm_score`).  It is
~0 at the true geometry and ~1 at any other: measured 1.4e-15 against ~0.6-1 across wrong widths
and offsets, a cliff, so the argmin carries no threshold.  Nothing is added to the signal to make
this work -- no preamble, no pilot, no training sequence -- and nothing is matched against a
template, so there is no reference waveform to correlate against.

Why no mode is left dark.  A screen with a spare mode is findable the same way by a key holder --
the spare mode is dark at the true geometry -- and by anyone else: the windowed covariance at the
true width then has a direction holding noise only, which an observer without the key finds by
eigen-decomposition.  Measured (research/rank_leak.md), at L=64 in 10 s the true width stands out
at z 19-27 from -15 dB, as visible with one spare mode as with eight, and resampling to another
rate hides nothing.  A full screen's covariance is white, its power profile flat and its lag
correlation zero: every second-order read of it is noise at every length and rate measured.

Rows sit back to back, L samples apart.  The sender writes no cyclic extension or prefix, because
a repeated stretch of a row is a lag-correlation line a blind observer finds below the noise, and
the receiver looks for none (research/extension.md).

The read is keyed.  Only a holder of the seed can build the aperture, so only a holder can place
the geometry; with ``seed=None`` the aperture is the public zero state and anyone can.  The key
protects content and alignment, not the fact of transmission.

The open mode is joinable at any point.  The per-row rotation defends the payload, and with no key
there is nothing to defend, so the open mode carries none (see :func:`mode_keys`) and offers no
alignment to search for.  An open broadcast therefore reads from any point of any fragment, while
a keyed one has to locate the row it joined on -- which is what ``JOIN_SEARCH_ROWS`` bounds.

A keyed stream can carry the row instead of being searched for it.  ``compose(..., beacon=True)``
puts a keyed row beacon on the bottom ``BEACON_MODES`` modes, and a receiver reads the row out of
it: a join then costs one frame rather than the age of the stream, and a fragment cut from beyond
any search reach still decodes.  The beacon is measured, never declared -- :func:`_alphabet` reads
both hypotheses and the larger closure names which rides -- so a beaconed capture and a plain one
both decode from a capture and a key.  It costs those modes: a beaconed link carries K modes of
payload where a plain one carries K + BEACON_MODES.


A blind lock needs about L^2 samples, because a width is only scorable when the capture supplies
rows at it.  The default candidate range runs to ``isqrt(len(iq))`` for exactly that reason.

The lock is carrier-blind to first order, and where it is not the carrier is a coordinate rather
than an impairment.  A carrier offset multiplies sample n by exp(2.pi.i.f.n); with n = t*L+l that
factors into a per-row phase and a per-column phase, and the per-column half is a phase ramp on
the aperture's mode index (nu = f*L).  So the aperture is evaluated at that carrier and the offset
is measured on the same axis as everything else -- read in closed form off the keyed reference row
by :func:`_read_carrier`, then settled.  Measured clean, the read is exact to five decimals at
every carrier tried from 0 to 1.92 -- 1.92 cycles/row being some 30 000 ppm of a 915 MHz part
at 2 Msps, far past anything an oscillator does.

The differential read is carrier-tolerant rather than carrier-blind: the differential across rows
cancels the row-rate term exactly, while the per-column ramp causes mode crosstalk growing with
f*L.  Gross carrier offset is a front-end job; what reaches here is the residual.

The waveform's structure reads like noise to every structure read but one (its energy is plain to a
radiometer; what is not visible is a structured signal).  Its singular spectrum is flat by
construction (measured max/min = 2.1 across the live modes), which is precisely the i.i.d. bulk
the Tracy-Widom floor tests against.  Measured on a clean capture: ``resolved_modes`` = 0 and
``Projection.coherence`` ~ 0.5 -- the readings pure noise gives.  ``Aperture.extract()`` has
nothing to pull out and ``Aperture.sweep()`` finds no coherent band, while the same ``extract()``
recovers K_signal = 4 from a planted rank-4 screen.  That is the property, measured: visible as
energy, not classifiable as a structured signal, and unreadable without the key.

Every row goes out at its own power.  The coordinates are unit modulus and the aperture
orthonormal, so without that every row would carry energy K exactly, and the power envelope alone
would hand a key-free observer the row period: from 417 ms at L=64 the windowed-energy variance
ranked the true width first of 585 at z 38 from 0 dB (research/row_energy_leak.md).  Row t is sent
scaled by r_t with r_t^2 ~ Gamma(L, 1/L) (:func:`row_radius`), keyed from the seed and the screen
row in its own domain, or the sender's own draw in the open mode, and its windowed energy is then
distributed as white noise's at every offset.  The differential read is phase-only and never
needs it.  The coherent read divides r_t / r_0 out from the key, or in the open mode measures each
row's level.  With every mode lit the energy of row t is exactly L r_t^2, so the radius is what
keeps the envelope white; the full screen closes the rest (above).

Where every constant comes from.  Nothing in the decision path is tuned.  The reads that decide
something are threshold-free -- they separate by orders of magnitude, or they are scored against a
closed-form null at the reader's declared level -- and everything else is a search bound, a spec,
or a numerical guard.

    decides something
      constant-modulus read   no threshold: the true geometry and any other differ by ~1e15
                              noiseless, so the argmin stands on its own.  Its null is the flat
                              Dirichlet's mean entropy deficit, a closed form.
      alphabet M              the smallest exponent at which the differential closes, searched
                              jointly with the row alignment.  Closure is ~1 at the alphabet and
                              ~0 below it.
      sender resolution       scored against the exact permutation null of
                              ``entroptics.reads.coupling``, at the level below.

    declared inputs (the caller's, not the library's)
      far = 0.05              the false-alarm level of :func:`sender_resolve`, entroptics' one
                              external input, at the library's own default.

    search bounds and cost knobs -- nothing is thresholded against them
      MIN_ROWS = 8            where :func:`discover` starts widening; 2 is the forced floor.
      CM_MODES = 16           how many modes the scans read per hypothesis.
      JOIN_SEARCH_ROWS = 4096 how many candidate rows the join hunt walks: how late a join
                              this receiver can recover, which the deployment sets.
      width cap 256           the widest candidate width; a wider screen needs an explicit
                              candidate range.
      carrier_span = 1.0      how far the two oscillators may differ, in cycles per row.

Discovery sweeps.  The cost is one constant-modulus scan per (carrier, width) candidate, and the
sweep is what makes the read blind: with an unknown carrier the width cannot be shortlisted first.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize as _minimize
from scipy.optimize import minimize_scalar as _minimize_scalar

# The installed entroptics package -- the optics engine.  No IQ and no RF term ever crosses into
# it; this module is the entire boundary.  The version floor matches the packaging requirement
# (pyproject: entroptics>=0.2.10) and is re-checked here, at import and with a reason, so a too-old
# release fails cleanly at import rather than mid-read with a bare AttributeError.
_lib = importlib.import_module("entroptics")
REQUIRED_ENTROPTICS = "0.2.10"

if tuple(int(x) for x in _lib.__version__.split(".")[:3]) < \
        tuple(int(x) for x in REQUIRED_ENTROPTICS.split(".")[:3]):
    raise ImportError(
        f"entroptics {_lib.__version__} installed; entroptics-rf needs >= {REQUIRED_ENTROPTICS}")

Aperture = _lib.Aperture        # the single front door: every optical read goes through it
_sequence = importlib.import_module("entroptics.sequence")   # the ordered-axis reads on a
                                # symbol stream: alphabet size, redundancy, entropy rate

__all__ = [
    "Aperture", "REQUIRED_ENTROPTICS", "MIN_ROWS",
    "screen", "to_iq", "Optics", "optics",
    "mode_basis", "mode_keys", "row_radius", "spread_operator", "project",
    "BEACON_MODES", "BEACON_FRAME", "beacon_rotation", "read_beacon",
    "sender_rotation", "sender_signature", "sender_match", "sender_resolve",
    "compose", "read", "read_soft", "compose_qam", "read_qam",
    "Interference", "interference",
    "constant_modulus_read", "Discovery", "discover", "decode", "decode_scaled", "coherent_closure",
    "closure_significance",
    "compose_embedding", "read_embedding", "embedded_dimension",
    "capacity", "bits_per_sample",
]

BEACON_MODES = 3        # modes the row beacon occupies, taken from the bottom of the aperture
                        # so the payload follows it and a short final row still drops payload
                        # modes off the end.  Three because the index has to survive the channel:
                        # a frame carries BEACON_FRAME * BEACON_MODES bits, and at three modes
                        # that is 96, enough for a 32-bit row index repeated three times and
                        # soft-combined.  They cost payload: a beaconed link carries K modes where
                        # an unbeaconed one carries K + 3.

BEACON_FRAME = 32       # rows per beacon frame.  It bounds the join: a receiver tries the frame's
                        # BEACON_FRAME possible phases and reads the index, so joining costs the
                        # frame rather than the age of the stream.  Thirty-two keeps the smallest
                        # readable fragment at the 64 rows the alignment already needs, since two
                        # frames fit there.

_BEACON_INDEX_BITS = 32  # frames a beacon can name: 2^32 frames of 32 rows is 1.4e11 rows, some
                         # 1.6 years of a 2 Msps link at L=64.

MIN_ROWS = 8            # where the row widening starts.  Starting at 2, 4 or 8 returned the same
                        # geometry and the same BER -- only the cost of the first attempt moves.
                        # 2 is the forced floor, the fewest rows a differential exists on.  A
                        # search bound: no read is thresholded against it.

SQUARE_REACH = 1 / 16   # transmitted samples: how far L received samples may overreach a row of a
                        # sender a hair wider than the receiver and still be read as a square fit
                        # (:func:`_row_window`)
JOINT_REACH = 4         # the widest margin the joint read (:func:`_joint_coords`) takes either side
                        # of a row, in rows.  The margin has to hold enough spare received samples
                        # to pay for the far-field skeleton, and that grows as the scale nears 1:
                        # at L=128 it is L/4 up to rho 0.89, 89 samples at 0.95, 328 at 0.98 and
                        # 512 (the bound) at 0.986.  The floor holds all the way (-65 to -66 dB
                        # noiseless at 0.9, 0.95 and 0.98), but the per-scale build, a complete QR
                        # of the window, grows as the cube of it: measured 34 ms at 0.9, 86 ms at
                        # 0.95, 0.85 s at 0.98 and 2.5 s at 0.986.  So past four rows a scale that
                        # close to 1 is read by the band read (:func:`_band_coords`), which needs no
                        # spare samples and takes the same four rows of margin.  It bounds a cost,
                        # not a conclusion: every row is still read, and nothing is compared
                        # against it.

# Keystream domains.  One per keyed use, versioned: how a stream is consumed is wire format.
_D_BASIS = b"entroptics-rf/mode-basis/v1"
_D_SPREAD = b"entroptics-rf/spread-operator/v1"
_D_SENDER = b"entroptics-rf/sender/v1"
_D_REF = b"entroptics-rf/reference-row/v1"
_D_FILL = b"entroptics-rf/filler/v1"
_D_MODEKEY = b"entroptics-rf/mode-key/v1"
_D_BEACON = b"entroptics-rf/row-beacon/v1"
_D_RADIUS = b"entroptics-rf/row-radius/v1"


# ---------------------------------------------------------------------------
# the ravel -- the one reduction this adapter owns.  IQ (1-D) -> screen (2-D).
# ---------------------------------------------------------------------------

def _require_finite(iq, who: str) -> np.ndarray:
    """Reject a capture carrying NaN or Inf, and say so.

    One non-finite sample poisons every read that touches it: the geometry read comes back NaN
    and nothing locks, leaving a dead band and a corrupted recording indistinguishable -- on a
    bench, the difference between retuning and rebuilding the front end.  So the count and the
    first bad index are reported.

    The capture is reported, not repaired.  Dropping the sample would shift every sample after it
    and destroy the timing the construction rests on; zeroing it would decide on the caller's
    behalf what their missing data means.

    Checked at the public boundary, which leaves :func:`screen` free for the width sweep that
    calls it hundreds of times per acquisition."""
    a = np.asarray(iq)
    ok = np.isfinite(a)            # complex-aware: False if either part is non-finite
    if a.size and not ok.all():
        bad = int(np.count_nonzero(~ok))
        first = int(np.argmax(~ok))
        raise ValueError(
            f"{who}: capture carries {bad} non-finite sample(s), first at index {first}. "
            "Nothing is repaired here -- dropping one would shift the timing every read depends "
            "on, and zeroing it would be guessing what your front end meant.")
    return a


def screen(iq, L: int, *, offset: int = 0, max_rows: int | None = None) -> np.ndarray:
    """Raw IQ -> the ``(T, L)`` screen entroptics reads.  The adapter's whole job.

    ``offset`` is where the first row starts.  It matters absolutely: a capture that begins
    mid-row produces rows straddling two true rows, whose mode structure is gone -- measured BER
    0.50 at any non-zero offset.  A real capture never starts on a row boundary, so the offset is
    something a receiver must measure, and :func:`discover` measures it.

    Trailing samples that do not fill a row are dropped: a partial row is not an observation of
    the screen.  ``max_rows`` caps the height to bound cost on a long capture.  Nothing else is
    done to the samples -- this is a reshape, not a transform."""
    iq = np.asarray(iq)
    L = int(L)
    if L <= 0:
        raise ValueError(f"screen width must be positive, got {L}")
    offset = int(offset) % L
    if offset:
        iq = iq[offset:]
    T = len(iq) // L
    if max_rows is not None:
        T = min(T, int(max_rows))
    if T < 1:
        raise ValueError(f"capture of {len(iq)} samples cannot fill one row of width {L}")
    return iq[:T * L].reshape(T, L)


def _held(n: int, offset: int, L: int) -> int:
    """How many whole rows of width ``L`` a capture of ``n`` samples holds from ``offset``."""
    return (int(n) - int(offset)) // int(L)


def to_iq(S, *, unit_rms: bool = True) -> np.ndarray:
    """A composed screen -> transmittable IQ.  The inverse ravel, also not a transform.

    complex64 because that is what SDR drivers take; unit RMS because the radio's own gain stage
    sets the operating point and a waveform carrying its own scale would fight it."""
    iq = np.asarray(S).ravel()
    if unit_rms:
        p = float(np.mean(np.abs(iq) ** 2))
        if p > 0:
            iq = iq / np.sqrt(p)
    return iq.astype(np.complex64)


@dataclass(frozen=True)
class Optics:
    """The named optical reads of one capture at one width.  See the dictionary in the module
    docstring for the library read behind each field."""
    L: int
    T: int
    fill: float             # Aperture.phi_F           -- feature-axis fill
    reach: float            # Aperture.magnification   -- 1/fill
    etendue: float          # Aperture.etendue         -- occupied phase-space area
    capacity: float         # Aperture.space_bandwidth -- DOF the screen can carry
    occupied: float         # etendue * capacity       -- the spots actually filled
    strehl: float           # Aperture.strehl          -- power in the dominant coherent mode
    dominance: float        # Aperture.dominance       -- leading mode's excess over the mean
    resolved: int           # Aperture.resolved_modes  -- modes above the derived floor
    papr_db: float          # not an optical read: a transmitter constraint


def optics(iq, L: int, *, max_rows: int | None = None) -> Optics:
    """Read the optics of a capture at width ``L`` -- every field a named library read.

    ``resolved`` counts modes against the derived Tracy-Widom floor (Def 8.2), and this waveform's
    spectrum is flat by construction -- the i.i.d. bulk that floor tests against.  So it reads 0
    on a healthy capture and is not a way to recover K; that is the noise-like structure showing up in a
    read.  K is recovered instead from the keyed mode-energy profile, by :func:`discover`."""
    S = screen(iq, L, max_rows=max_rows)
    ap = Aperture(S, window=None)      # the full frame, not the default trailing window
    p = np.abs(np.asarray(S).ravel()) ** 2
    mean = float(p.mean())
    return Optics(
        L=int(S.shape[1]), T=int(S.shape[0]),
        fill=float(ap.phi_F), reach=float(ap.magnification),
        etendue=float(ap.etendue), capacity=float(ap.space_bandwidth),
        occupied=float(ap.etendue) * float(ap.space_bandwidth),
        strehl=float(ap.strehl), dominance=float(ap.dominance),
        resolved=int(ap.resolved_modes),
        papr_db=float("inf") if mean <= 0 else 10.0 * float(np.log10(float(p.max()) / mean)),
    )


# ---------------------------------------------------------------------------
# the seed is the key -- SHAKE256 keystream, domain-separated
# ---------------------------------------------------------------------------
# SHAKE256, the NIST FIPS-202 XOF from stdlib hashlib, rather than np.random.default_rng: PCG64's
# state is recoverable from a handful of outputs, so a known plaintext would hand an attacker
# every other stream drawn from the same seed.  Without the seed a SHAKE256 stream is
# computationally indistinguishable from random, and the domain tags make the aperture and the
# reference cryptographically independent rather than merely statistically so -- which matters
# because they are drawn from one seed and meet in one screen.
#
# Freshness: `nonce` is folded in, so one seed can key many transmissions.  It is not signalled in
# band -- both ends must agree on it.  Seed reuse with a repeated nonce is a two-time-pad, as for
# any stream cipher.  The symmetric key authenticates; it does not sign.

# The zero state.  ``seed=None`` is public broadcast, and its aperture is derivable rather than
# published: the same domain-separated SHAKE-256 stream every keyed aperture is drawn from, with
# the seed field empty.  Every real seed encodes to at least one tagged byte (:func:`_seed_bytes`),
# so no key -- not 0, not b"" -- can land on it, and anyone can compute it from the spec alone.
# A low-discrepancy sequence such as R2 would not do: it is not spectrally white, so a full-width
# open broadcast reads as 0.45-0.52 occupied (keyed: 0.95-1.02), a receiver takes it for a narrower
# sender, the known-geometry read miscounts its modes, and an open transmission is told from a
# keyed one by its spectrum.  A hash stream is as
# white as a keyed one, so open and keyed transmissions read -- and look -- alike.
_PHI = 0.6180339887498949            # 1/phi, the worst-approximable number: the additive
                                     # recurrence on it is the canonical 1-D low-discrepancy
                                     # sequence (see _carrier_probes)


def _zero_state_uniform(domain: bytes, nonce_bytes: bytes, n: int) -> np.ndarray:
    """The public aperture's uniform stream: the keyed construction with an empty seed field.

    Derivable by anyone -- no key, no table, no agreement beyond the domain tag that is part of the
    public spec -- and as white as any keyed aperture."""
    pre = b"".join([len(domain).to_bytes(4, "big"), domain,
                    len(nonce_bytes).to_bytes(4, "big"), nonce_bytes,
                    (0).to_bytes(4, "big")])
    raw = np.frombuffer(hashlib.shake_256(pre).digest(int(n) * 8), dtype=">u8")
    return (raw.astype(np.float64) + 0.5) / 2.0 ** 64


def _seed_bytes(seed) -> bytes:
    """Canonical, injective seed encoding.  The type tag keeps int 0x41 and b'A' distinct; the
    full seed is used, never truncated to 64 bits.  ``seed=None`` never reaches here -- the public
    aperture takes the zero-state stream instead (see :func:`_keyed_uniform`)."""
    if isinstance(seed, (bytes, bytearray, memoryview)):
        return b"\x01" + bytes(seed)
    if isinstance(seed, str):                    # a name is a perfectly good identifier, and it
        return b"\x03" + seed.encode("utf-8")    # gets its own tag so "1" cannot alias int 1
    n = int(seed)
    sign = b"-" if n < 0 else b"+"
    n = abs(n)
    return b"\x00" + sign + (n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b"\x00")


def _keyed_uniform(domain: bytes, seed, n: int, nonce=None) -> np.ndarray:
    """n keyed uniforms in the open interval (0, 1) -- open because these feed a Box-Muller log.
    Fields are length-prefixed, so no two (domain, nonce, seed) triples can collide."""
    n = int(n)
    if n <= 0:
        return np.zeros(0)
    nb = b"" if nonce is None else (bytes(nonce) if isinstance(nonce, (bytes, bytearray, memoryview))
                                    else _seed_bytes(nonce))
    if seed is None:
        return _zero_state_uniform(domain, nb, n)
    sb = _seed_bytes(seed)
    pre = b"".join([len(domain).to_bytes(4, "big"), domain,
                    len(nb).to_bytes(4, "big"), nb,
                    len(sb).to_bytes(4, "big"), sb])
    raw = np.frombuffer(hashlib.shake_256(pre).digest(n * 8), dtype=">u8")
    return (raw.astype(np.float64) + 0.5) / 2.0 ** 64


def _keyed_normal(domain: bytes, seed, shape, nonce=None) -> np.ndarray:
    """Keyed standard complex normals via Box-Muller.  Deterministic and platform-independent, so
    two ends sharing only a seed build the same aperture."""
    shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
    n = int(np.prod(shape)) if shape else 0
    if n == 0:
        return np.zeros(shape, dtype=complex)
    u = _keyed_uniform(domain, seed, 2 * n, nonce)
    r = np.sqrt(-2.0 * np.log(u[:n]))
    th = 2.0 * np.pi * u[n:]
    return (r * np.cos(th) + 1j * (r * np.sin(th))).reshape(shape)


def _canonical_qr(G: np.ndarray) -> np.ndarray:
    """Reduced QR with Q pinned by R's diagonal being positive real.

    numpy fixes Q only up to a unit-modulus factor per column -- a LAPACK detail.  Two ends on
    different machines must derive the same aperture from the same seed, so the phase is pinned to
    the unique canonical choice.  Without this the protocol works in one process and fails across
    a link for a reason no capture could reveal."""
    Q, R = np.linalg.qr(G)
    d = np.diagonal(R)
    mag = np.abs(d)
    s = np.where(mag > 1e-300, d / np.where(mag > 1e-300, mag, 1.0), 1.0)
    return Q * s[None, :]


# ---------------------------------------------------------------------------
# the keyed aperture -- the seed defines the optics, so the optics are the key
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=2048)
def _mode_basis_cached(K: int, L: int, seed, nonce, phase: float, carrier: float) -> np.ndarray:
    B = _aperture(L, seed, nonce, phase, carrier)[:K]   # nested: the K-mode aperture is the first K
    B.flags.writeable = False
    return B


def mode_basis(K: int, L: int, seed, nonce=None, phase: float = 0.0,
               carrier: float = 0.0) -> np.ndarray:
    """The digital aperture: K orthonormal rows in C^L, from the seed alone.  (K, L).

    K is the screen's rank = its space-bandwidth product = its capacity, so K <= L: asking for
    more modes than the screen has dimensions is asking for capacity that does not exist.

    ``phase`` evaluates the aperture at real position ``l + phase`` and ``carrier`` at a mode-index
    shift of ``carrier`` -- a fractional sampling offset and a carrier offset each cost nothing but
    naming them, because the mode shapes are continuous in both.

    Nested in K: this is the first K rows of the one L-mode aperture for (L, seed), not a fresh
    draw per K.  A transmission lights all L modes, so the nesting is what lets a message's final
    row use fewer -- the same aperture used less far -- and what lets a row beacon take the bottom
    modes with the payload above it.  A per-K draw would make every short row an unrelated basis,
    so nesting is wire format, not an implementation detail.

    Cached, and the result is read-only.  A streaming receiver calls this once per buffer, and
    re-running a QR per buffer was measured to dominate the decode -- the aperture is a pure
    function of (K, L, seed, nonce), so it is derived once and shared.  Copy it if you need to
    modify one."""
    K, L = int(K), int(L)
    if not (0 < K <= L):
        raise ValueError(f"need 0 < K <= L, got K={K}, L={L}")
    return _mode_basis_cached(K, L, seed, nonce, float(phase), float(carrier))



_KEY_BLOCK = 64          # rows per keystream block; a seek costs one block, never the whole run


def _block_nonce(nonce, block: int) -> bytes:
    """The nonce for one keystream block: the caller's nonce, then the block index.

    Length-prefixed through the same encoding everything else uses, so no two (nonce, block)
    pairs collide and a block index cannot be confused with a nonce that happens to end in
    digits."""
    base = (b"" if nonce is None else
            bytes(nonce) if isinstance(nonce, (bytes, bytearray, memoryview))
            else _seed_bytes(nonce))
    return len(base).to_bytes(4, "big") + base + b"blk" + int(block).to_bytes(8, "big")


def mode_keys(K: int, seed, nonce=None, *, first: int = 0, rows: int = 1) -> np.ndarray:
    """A keyed unit-modulus rotation per (row, mode): ``(rows, K)``, rows ``first .. first+rows``.

    This is what makes a recovered mode unreadable.  The transmitted field is ``X = A B`` with
    every entry of ``A`` unit modulus and ``B`` orthonormal -- a constant-modulus source through a
    unitary mixture, which is exactly the input to ACMA (van der Veen & Paulraj 1996) and ILSP
    (Talwar, Viberg & Paulraj 1994).  Neither attacks the key.  They solve for the aperture from
    the signal, using two facts that are properties of the design rather than of the seed: that
    ``|A_tk| = 1`` and that ``B B^H = I``.  The row covariance hands over the K-dimensional row
    space; projecting onto it leaves ``Y = A R`` with R unitary; and ``|y_t v| = const`` then has
    exactly K orthonormal solutions, the columns of ``R^H``.  That is an algebraic identity, not a
    search: measured, ACMA recovers all 60 mode streams from 118 ms of clean capture.

    What such a solution leaves is a per-mode constant phase -- and a constant per-mode phase is
    exactly what the differential ``A[t+1] conj(A[t])`` cancels.  So the rotation is applied to
    the differential and varies from row to row.  What an attacker recovers is then the keyed
    difference rather than the symbol, and no constant ambiguity cancels it.  Measured under it:
    ACMA still recovers the coefficients at 1.000 -- the concentration stays visible, and it is
    not what the key hides -- while the same attacker reads BER 0.497 and the key holder reads
    0.000000 at any length.

    It costs nothing.  Unit modulus is no power and no rate, and every coordinate keeps its
    modulus, so the width, the row offset and the rest of the geometry read exactly as they would
    without it.  It is not a second layer of encryption: it is the same seed keying the same
    waveform, as the aperture already is.
    """
    # Seekable.  Drawn per block of rows rather than as one run from the origin, so the cost of
    # row t does not grow with t.  That is what makes a deep alignment search affordable: over R
    # candidates it costs O(R) per block against O(R^2) from the origin, and a receiver joining a
    # stream late is the ordinary case.
    K, first, rows = int(K), int(first), int(rows)
    if seed is None:
        # The open mode has no rotation.  This exists to keep a blindly separated mode
        # unreadable -- separation recovers the aperture up to a constant per-mode phase, a
        # differential cancels a constant phase, and changing the rotation every row leaves
        # nothing constant to cancel.  Without a key the aperture and the reference row are
        # public anyway, so there is nothing here to keep unreadable.
        #
        # What it costs is the whole of the join problem: a rotation indexed by the absolute row
        # is why a receiver must find the row it joined on.  Dropping it where it defends nothing
        # makes an open broadcast readable from any point of any fragment, which is what a
        # receiver tuning into a broadcast needs.  Measured on 128-row fragments: with the
        # rotation a cut at row 4500 reads row0 65 at BER 0.6316, without it row0 0 at BER
        # 0.0000, and the optical reads do not move (resolved 2 either way, strehl 0.0254 against
        # 0.0262) -- the rotation was never what made this look like noise.
        return np.ones((rows, K), dtype=complex)
    lo, hi = first // _KEY_BLOCK, (first + rows - 1) // _KEY_BLOCK
    out = np.concatenate([
        _keyed_uniform(_D_MODEKEY, seed, _KEY_BLOCK * K,
                       _block_nonce(nonce, b)).reshape(_KEY_BLOCK, K)
        for b in range(lo, hi + 1)])
    start = first - lo * _KEY_BLOCK
    return np.exp(2j * np.pi * out[start:start + rows])


def row_radius(L: int, seed, nonce=None, *, first: int = 0, rows: int = 1,
               rng=None) -> np.ndarray:
    """The amplitude ``r_t`` of screen rows ``first .. first+rows``: ``(rows,)``, with
    ``r_t^2 ~ Gamma(L, 1/L)``, mean 1.  Row t goes on the air as ``r_t * w_t B``.

    Without it every row would carry the same energy: the coordinates are unit modulus and the
    aperture orthonormal, so ``||w_t B||^2 = K`` exactly.  A key-free observer then finds the row
    period from the power envelope alone.  The windowed energy has no variance at the row length
    and the row offset, and ``|y|^2`` has a spectral hole below the row rate.  Measured from 417 ms
    at L=64 (research/row_energy_leak.md), the period ranks first of 585 widths at z 38 from 0 dB
    and z 6 from -5 dB, and presence reads z 23 at 0 dB with neither a period nor an offset.

    The shape is L, not K.  The energy of L samples of white noise is Gamma(L) about its mean.  A
    row scaled by ``r_t`` with ``Var r_t^2 = 1/L`` has exactly that law in a window aligned on the
    row, and the same variance to O(1/L) in a window straddling two rows.  Shape K is
    over-dispersed by L/K.  Measured noise-free on 3.8 M samples, the windowed-energy variance is
    within 0.5% of white noise's at every offset, where without the radius it is 0 at the row
    boundary.  After it the energy detectors sit inside the noise-only range at every SNR, length,
    width and rate measured.  It closes the row-energy leak and nothing else; the rank deficiency,
    the fold and the spectral ripple a spare mode would add are closed by lighting every mode
    (research/rank_leak.md).

    Keyed, it is the mean of L unit exponentials ``-ln u`` over keyed uniforms in its own domain,
    drawn per block of rows like :func:`mode_keys`, so a seek costs one block.  It is indexed by
    the screen row, the reference row being row 0, so the coherent reader divides ``r_t / r_0``
    out exactly (:func:`_dekeyed`).  The differential read is phase-only and never needs it.

    Open (``seed=None``), it is the sender's own draw from ``rng``, never a public stream, and
    ``first`` is unused.  A public stream indexed by the absolute row would put back the join
    problem the open mode drops (see :func:`mode_keys`), and it would hide nothing, since anyone
    can read an open broadcast.  No receiver needs it: the differential read is phase-only and the
    coherent read measures each row's level (:func:`_row_levels`).

    The reference row takes a radius like every other row.  Exempt, it would be one row of fixed
    energy at the head of every burst, a place an observer finds from the burst's onset; over a
    few bursts the head window's energy then has no spread at the true width and only there,
    which hands over L.  The cost falls on the coherent read: the gains are measured on the
    reference row, whose SNR is ``r_0^2`` times the mean, so a key's ``r_0`` shifts its coherent
    threshold (measured in research/row_energy_leak.md)."""
    L, first, rows = int(L), int(first), int(rows)
    if rows <= 0:
        return np.zeros(0)
    if seed is None:
        return np.sqrt(np.random.default_rng(rng).gamma(L, 1.0 / L, rows))
    lo, hi = first // _KEY_BLOCK, (first + rows - 1) // _KEY_BLOCK
    u = np.concatenate([
        _keyed_uniform(_D_RADIUS, seed, _KEY_BLOCK * L,
                       _block_nonce(nonce, b)).reshape(_KEY_BLOCK, L)
        for b in range(lo, hi + 1)])
    start = first - lo * _KEY_BLOCK
    return np.sqrt(-np.mean(np.log(u[start:start + rows]), axis=1))


def _radiate(S, L: int, seed, nonce=None, *, first: int = 0, rng=None) -> np.ndarray:
    """Screen rows ``S`` (rows from screen row ``first``) scaled by their :func:`row_radius`."""
    S = np.asarray(S)
    return S * row_radius(L, seed, nonce, first=first, rows=S.shape[0], rng=rng)[:, None]


def beacon_rotation(seed, nonce=None) -> np.ndarray:
    """``(BEACON_FRAME, BEACON_MODES)`` keyed rotation, indexed by the row within the frame.

    Indexed by the row within the frame and not by the absolute row, which is the whole point: a
    receiver that does not know the absolute row can still de-rotate this by trying the frame's
    phases.  It repeats every frame, and that is safe here where it would not be for the payload:
    dividing two beacon rows a frame apart yields the difference of two frame indices, which is
    not secret.  The payload's rotation (:func:`mode_keys`) stays indexed by the absolute row.
    """
    u = _keyed_uniform(_D_BEACON, seed, BEACON_FRAME * BEACON_MODES, nonce)
    return np.exp(2j * np.pi * u.reshape(BEACON_FRAME, BEACON_MODES))


def _beacon_codeword(frame_index: int) -> np.ndarray:
    """The bits one frame carries: its own index, repeated to fill the frame.

    Repetition rather than a code, because the receiver needs the repetitions for a second job --
    their agreement is what identifies the frame phase, with no threshold and nothing added to
    carry a sync pattern."""
    idx = int(frame_index) & ((1 << _BEACON_INDEX_BITS) - 1)
    bits = (idx >> np.arange(_BEACON_INDEX_BITS - 1, -1, -1)) & 1
    slots = BEACON_FRAME * BEACON_MODES
    return np.tile(bits, slots // _BEACON_INDEX_BITS + 1)[:slots].astype(np.int64)


def _beacon_symbols(t_first: int, rows: int, seed, nonce=None) -> np.ndarray:
    """``(rows, BEACON_MODES)`` beacon differentials for absolute rows ``t_first ...``."""
    rot = beacon_rotation(seed, nonce)
    out = np.empty((int(rows), BEACON_MODES), dtype=complex)
    for i in range(int(rows)):
        t = int(t_first) + i
        p = t % BEACON_FRAME
        word = _beacon_codeword(t // BEACON_FRAME)
        bit = word[p * BEACON_MODES:(p + 1) * BEACON_MODES]
        out[i] = (1.0 - 2.0 * bit) * rot[p]          # BPSK on the keyed rotation
    return out


def read_beacon(d_beacon, seed, nonce=None):
    """Read the absolute row from beacon differentials, or ``None`` if none is riding.

    ``d_beacon`` is ``(rows, BEACON_MODES)``, the de-referenced differential of the beacon modes,
    row ``i`` being some unknown absolute row ``t0 + i``.

    The frame phase is unknown, so all ``BEACON_FRAME`` of them are tried and scored by how far
    the repetitions of the index agree once de-rotated.  At the true phase they agree; at any
    other the de-rotation is wrong and they scatter.  The phases are compared against each other
    and the best taken -- a comparison between measured candidates, not a level.

    Returns ``(t0, agreement)``: the absolute row of ``d_beacon[0]``, and how firmly the
    repetitions agreed (1 = exactly).  ``None`` when the capture holds no complete frame."""
    d = np.asarray(d_beacon)
    if d.ndim != 2 or d.shape[1] != BEACON_MODES or d.shape[0] < BEACON_FRAME:
        return None
    rot = beacon_rotation(seed, nonce)
    reps = (BEACON_FRAME * BEACON_MODES) // _BEACON_INDEX_BITS
    best = None
    for p0 in range(BEACON_FRAME):
        i0 = (-p0) % BEACON_FRAME               # first row of the first whole frame in view
        if i0 + BEACON_FRAME > d.shape[0]:
            continue
        block = d[i0:i0 + BEACON_FRAME]         # exactly one frame, phases 0 .. BEACON_FRAME-1
        soft = np.real(block * np.conj(rot)).ravel()     # (BEACON_FRAME * BEACON_MODES,)
        soft = soft.reshape(reps, _BEACON_INDEX_BITS)    # each row one copy of the index
        agree = float(np.mean(np.abs(np.sum(np.sign(soft), axis=0))) / reps)
        combined = soft.sum(axis=0)
        idx = int(np.sum((combined < 0).astype(np.int64)
                         * (1 << np.arange(_BEACON_INDEX_BITS - 1, -1, -1))))
        t0 = idx * BEACON_FRAME - i0
        if t0 >= 0 and (best is None or agree > best[1]):
            best = (t0, agree)
    return best


def spread_operator(N: int, D: int, seed, nonce=None) -> np.ndarray:
    """The analog aperture: (N, D) with orthonormal columns, spreading D dims over N samples.
    N/D is the processing gain -- the analog redundancy dial.

    Nested in D, for the reason :func:`mode_basis` is nested in K: these are the first D columns
    of the one N-dimensional operator for (N, seed), rather than a fresh draw per D.  A per-D draw
    makes every D an unrelated subspace, which a receiver cannot search cheaply: it costs a fresh
    QR per candidate, 1.27 s each and some 1958 s to sweep D over a 1536-sample capture.  Nested,
    the same sweep is one QR and then a projection per candidate.

    The residual energy outside the first D columns is a cliff either way -- 0.000000 at the true
    D against 0.36 to 0.73 elsewhere, including 0.499 at D-1 and 0.483 at D+1, the sharpest read
    in this construction -- so nesting costs the sweep no sharpness and buys
    :func:`embedded_dimension` its affordability.

    The nesting is therefore wire format, exactly as the aperture's is."""
    N, D = int(N), int(D)
    if not (0 < D <= N):
        raise ValueError(f"need 0 < D <= N, got N={N}, D={D}")
    return _spread_full(N, seed, nonce)[:, :D]


@functools.lru_cache(maxsize=8)
def _spread_full(N: int, seed, nonce) -> np.ndarray:
    """The one N-dimensional analog aperture every D is a prefix of.  Cached: it is a pure
    function of (N, seed, nonce), and a receiver sweeping D would otherwise redraw it per
    candidate, which is the cost the nesting exists to remove."""
    Q = _canonical_qr(_keyed_normal(_D_SPREAD, seed, (int(N), int(N)), nonce))
    Q.flags.writeable = False
    return Q


def project(S, B) -> np.ndarray:
    """Read the mode coordinates out of a screen: ``A = S B^H``.

    B has orthonormal rows, so B^H IS the Moore-Penrose pseudoinverse.  ``pinv`` would run an SVD
    to rediscover that at ~L^3 cost and with rounding the adjoint does not have."""
    return np.asarray(S) @ np.asarray(B).conj().T


# ---------------------------------------------------------------------------
# Who sent it -- a rotation of the aperture, carried for free
# ---------------------------------------------------------------------------
# Rotating each mode of the aperture by its own phase leaves the payload untouched, because the
# read is differential across rows and a constant per-mode phase cancels in ``A[t+1] conj(A[t])``
# exactly.  So the rotation is a channel that costs nothing: no power, no rate, no change to how
# the transmission looks (measured strehl 0.0317 rotated against 0.0317 not, so it still reads as
# noise).
#
# And it is recoverable.  The receiver projects with the unrotated aperture and the rotation is
# what is left, up to the one global phase the channel also imposes -- which divides out.
# Measured: the true sender scores 1.000 against 0.155-0.226 for others, and 0.951 at 6 dB SNR
# with a carrier offset still identifies correctly.
#
# This is an address, not a secret.  Anyone holding the aperture can read it, so it says who sent
# a transmission rather than who may read it; the key remains the only thing protecting content.

def sender_rotation(K: int, sender, nonce=None) -> np.ndarray:
    """The per-mode phase pattern that marks a sender: (K,), unit modulus.

    Derived from the sender's own identifier through the same keystream as everything else, so a
    sender needs no allocation from anyone -- two distinct identifiers give patterns that are
    near-orthogonal by construction."""
    return np.exp(1j * 2.0 * np.pi * _keyed_uniform(_D_SENDER, sender, int(K), nonce))


def sender_signature(iq, L: int, K: int, seed, *, nonce=None, offset: int = 0,
                     phase: float = 0.0, carrier: float = 0.0) -> np.ndarray:
    """Read the sender's rotation back out of a capture: (K,), unit modulus.

    The global phase is divided out, because the channel imposes one of its own and the two are
    not separable -- so what identifies a sender is the pattern across modes, not its absolute
    phase.  Compare with :func:`sender_match`."""
    S = screen(iq, L, offset=offset)
    A = project(S, mode_basis(K, L, seed, nonce, phase, carrier))
    if A.shape[0] < 1:
        return np.zeros(K, dtype=complex)
    # The reference row is keyed, so row 0 carries its phases as well as the sender rotation.
    # A key holder divides the known one out; without that step the pattern identifying a sender
    # sits under the pattern hiding the aperture and the read finds strangers.
    th = np.angle(A[0] * np.conj(_reference_row(K, seed, nonce)))
    return np.exp(1j * (th - th[0]))


def sender_match(signature, sender, nonce=None) -> float:
    """How well a signature matches a candidate sender: 1 is exact, ~1/sqrt(K) is a stranger."""
    sig = np.asarray(signature)
    want = sender_rotation(sig.size, sender, nonce)
    want = want * np.conj(want[0])
    return float(abs(np.vdot(sig, want)) / max(sig.size, 1))


def sender_resolve(signature, senders, *, nonce=None, far: float = 0.05):
    """Which sender sent this, with the evidence -- ``(name | None, Coupling)``.

    :func:`sender_match` returns a bare cosine, which a reader would have to calibrate against a
    geometry and an SNR.  This read carries its own null instead: ``entroptics.reads.coupling``
    standardizes the alignment of two sides against the exact permutation null of re-pairing them,
    so the decision is ``resolved`` against a cutoff the declared level implies rather than a
    threshold anyone chose.

    The K modes are the ordered axis, which is the right null for this question: it asks whether
    this pattern across modes aligns better than a random relabelling of the modes would.

    Measured at K=60, L=64, sender among five: the true sender reads z = 10.9 / 10.7 / 10.3 at
    clean / 12 dB / 6 dB and resolves; the nearest stranger reads 1.8 / 1.6 / 1.4 and does not,
    against a cutoff of 2.58.  ``far`` is the reader's level and the one declared input; nothing
    else here is thresholded.

    Returns ``(None, best_coupling)`` when nothing clears the cutoff, which is the answer for a
    transmission from a sender not on the list."""
    sig = np.asarray(signature).reshape(-1, 1)
    names = list(senders)
    # The level is corrected for how many times the question is asked.  One candidate at far=0.05
    # fires on a stranger 5% of the time; five candidates fire about 23% of the time, and measured
    # they do -- an unlisted sender resolves to the nearest name at z=2.27 against a 1.96 cutoff.
    # entroptics' own sweep corrects the same way, deriving its gate from far/n_patches so that
    # the field-wide rate stays near far however wide the field is.
    level = float(far) / max(len(names), 1)
    best_name, best = None, None
    for name in names:
        want = sender_rotation(sig.size, name, nonce)
        want = (want * np.conj(want[0])).reshape(-1, 1)
        c = _lib.reads.coupling(sig, want, far=level)
        if best is None or c.z > best.z:
            best_name, best = name, c
    if best is None:
        return None, None
    return (best_name if best.resolved else None), best


# ---------------------------------------------------------------------------
# Digital payload -- differential M-PSK on the keyed modes
# ---------------------------------------------------------------------------
# Differential across rows, per mode: row t is row t-1 rotated by the symbol.  The receiver
# therefore needs no phase reference at all, which is what lets the waveform carry no pilot.
# M-PSK rather than QAM for the same reason: a differential amplitude would need one.  The
# coherent mode below spends a reference row and gets QAM in exchange.
#
# Payload entropy.  The transmitter writes at native resolution: it places what it was given on
# the screen and conditions nothing.  There is no scrambler, because the payload is already keyed
# by the waveform -- the aperture is the cipher, so a second layer buys no confidentiality and
# adds error multiplication -- and because a modem that rewrites its input misreports what it
# carried.
#
# The consequence shows up on real payloads and never on random bits.  A payload that does not
# exercise the whole alphabet is genuinely explained by a smaller one, and the receiver reads the
# alphabet actually used: measured, a 1-in-64 sparse bit pattern reads M=2 where M=4 was
# configured, a correct reading of a transmission that only ever sent two symbols.  A payload with
# a long constant run makes the screen low-rank, so the transmission stops looking like noise
# (strehl 0.84 against 0.03).  Both are properties of the source, and both disappear once the
# payload carries the entropy it claims to: measured on English text and on source code, every
# parameter reads correctly and the payload is bit-exact.
#
# Sparsity is a reconstruction question rather than a modulation one, and entroptics answers it on
# the read side -- ``Dynamics.predict`` / ``rollout`` extrapolate a trajectory from the operator,
# so a sparse or gapped payload is filled in where it is read rather than padded before it is
# sent.

def _psk(M: int) -> np.ndarray:
    """The M-PSK alphabet (unit power)."""
    M = int(M)
    if M not in (2, 4, 8, 16):
        raise ValueError(f"M must be one of 2, 4, 8, 16 (differential PSK); got {M}")
    return np.exp(2j * np.pi * np.arange(M) / M)


def _bits_per_symbol(M: int) -> int:
    return int(round(np.log2(int(M))))


def _slice_psk(d, M: int) -> np.ndarray:
    """Nearest M-PSK index, by angle rather than by distance to every point.

    For a unit-modulus alphabet the nearest point is the quantized angle, which is O(T*K) and
    allocation-free.  ``argmin`` over the constellation gives the identical answer but
    materializes a (T, K, M) temporary -- 61 MB at M=16 on a 4000-row buffer -- and that
    allocation, rather than the arithmetic, is what dominates a streaming decode."""
    M = int(M)
    q = np.angle(d) * (M / (2.0 * np.pi))
    return np.mod(np.rint(q).astype(np.int64), M)


def _slice_qam(z, M: int) -> np.ndarray:
    """Nearest square-QAM index, by independent I and Q decisions plus the Gray map.

    A square constellation is a product of two PAM axes, so the nearest point is found one axis
    at a time -- O(T*K) instead of the (T, K, M) distance array, which at M=256 is the difference
    between a buffer that streams and one that swaps."""
    M = int(M)
    s = int(round(np.sqrt(M)))
    half = int(round(np.log2(s)))
    # the alphabet is normalized by its own RMS, so the lattice step is read off the alphabet
    step = float(np.min(np.abs(np.diff(np.unique(np.real(_qam(M)))))))
    lo = -(s - 1) * step / 2.0
    i = np.clip(np.rint((np.real(z) - lo) / step), 0, s - 1).astype(np.int64)
    q = np.clip(np.rint((np.imag(z) - lo) / step), 0, s - 1).astype(np.int64)
    return (_gray_arr(i) << half) | _gray_arr(q)


def _gray_arr(n: np.ndarray) -> np.ndarray:
    return n ^ (n >> 1)


def capacity(K: int, L: int, n_samples: int, M: int = 4) -> int:
    """Payload bits a capture of ``n_samples`` carries at this geometry.  One row of L samples
    carries K symbols; the first row is the differential reference and carries none."""
    rows = int(n_samples) // int(L)
    return max(0, rows - 1) * int(K) * _bits_per_symbol(M)


def bits_per_sample(K: int, L: int, M: int = 4) -> float:
    """Asymptotic spectral efficiency of the geometry, in bits per complex sample."""
    return _bits_per_symbol(M) * int(K) / int(L)


# ---------------------------------------------------------------------------
# No nonce is carried
# ---------------------------------------------------------------------------
# A nonce a caller must supply is a second shared parameter, and only the key may be shared, so no
# path here passes one.  The keystream still accepts one, because domain separation and freshness
# are properties a keystream should have.
#
# One seed keys one waveform.  Two transmissions under the same seed share an aperture and a
# reference row -- the two-time-pad case.  Freshness is key management, which lives above this
# layer: a system that wants it gives each transmission its own key.
#
# Why a nonce cannot be measured like everything else.  The other parameters are low-entropy and
# continuous -- a width, a row, a carrier -- each with a smooth read and a cliff at the truth, so
# redundancy substitutes for agreement.  A 128-bit nonce has no concentration to maximize over
# 2^128 and no gradient to follow, so it would have to be carried.  A block at the head does not
# carry it, because a capture starts where the receiver switched on and a late start truncates the
# block rather than shifting it: measured, a one-sample cut is as fatal as a five-hundred-sample
# one, and it breaks the property this codec is built on -- that any window reads.
#
# The shape that would carry one is a carrier mode: a direction keyed by the seed alone, holding
# the nonce on every row, so any window of enough rows recovers it with gain that grows with the
# window.  That is a wire-format change.


def compose(bits, K: int, L: int, seed, *, M: int = 4,
            sender=None, unit_rms: bool = True, rng=None,
            beacon: bool = False) -> np.ndarray:
    """Write the waveform: bits -> keyed mode coordinates -> screen -> IQ.  No IFFT.

    ``bits`` is packed MSB-first into log2(M)-bit symbols laid out (T, K): one differential stream
    per mode.  ``sender`` marks the transmission with that sender rotation -- free, and invisible
    to the payload (see :func:`sender_rotation`).

    Every mode is lit: ``K`` must be L, or L - ``BEACON_MODES`` with a beacon (which takes the rest).
    A spare mode is a direction holding noise only in the windowed covariance at the true width,
    and an observer without the key finds the width from it (research/rank_leak.md); the receiver
    places the geometry by the constant-modulus read instead, and reads no mode count.

    Nothing is sent that the message did not ask to send, so a transmission is a function of the
    message length and the geometry alone.  A message that does not fill its last row is carried
    on fewer modes in that row rather than padded out to a full grid.  The nested aperture makes
    that free: ``mode_basis(K2, L, seed)`` is the first K2 rows of ``mode_basis(K, L, seed)`` (see
    :func:`mode_basis`), so a short row is the same aperture used less far -- no second
    factorization, nothing to agree, and no filler on the wire.

    The one rounding is inside the final symbol: a message ending mid-symbol leaves under log2(M)
    bits of room, filled with keyed bits rather than zeros, because a constant tail is structure
    an observer could see and a keyed one is not.

    There is no cyclic extension or prefix: a row repeated on the wire is a lag-correlation line a
    blind observer finds below the noise (measured: P=2 at L=128 shows lag 128 at z 8.4 from -5 dB
    SNR in 83 ms; the plain screen shows no line at any lag).  The ~2 dB it bought a receiver at
    another rate is the receiver's to recover, not the wire's (research/extension.md).

    Each row goes out at its own power, :func:`row_radius` (keyed; with ``seed=None`` drawn from
    ``rng``), so the rows' energies are distributed as white noise's rather than all equal.

    Returns unit-RMS complex64 IQ of length (T+1)*L."""
    bits = np.asarray(bits).ravel().astype(np.int64)
    if bits.size and (bits.min() < 0 or bits.max() > 1):
        raise ValueError("bits must be 0 or 1")
    K, L = int(K), int(L)
    b = _bits_per_symbol(M)
    const = _psk(M)

    nonce = None                           # no nonce is carried; see the note above
    # The beacon takes the bottom modes and the payload rides above it, so a short final row still
    # drops payload modes off the end and the aperture stays nested.  It costs those modes: at
    # L=64 a beaconed link carries K where an unbeaconed one carries K + BEACON_MODES.
    skip = BEACON_MODES if beacon else 0
    _require_full(K + skip, L, f"K={K}" + (f" + {skip} beacon modes" if skip else ""))
    B = mode_basis(K + skip, L, seed, nonce)
    if sender is not None:
        B = B * sender_rotation(K + skip, sender, nonce)[:, None]   # every mode the screen uses

    ref = _reference_row(K + skip, seed, nonce)      # keyed, never all-ones: see _reference_row
    n_sym = int(np.ceil(bits.size / b)) if bits.size else 0
    if n_sym == 0:                                  # nothing to carry: the reference row alone
        return to_iq(_radiate(ref[None, :] @ B, L, seed, nonce, rng=rng), unit_rms=unit_rms)

    need = n_sym * b                                # to the symbol boundary, not to a full grid
    if bits.size < need:
        pad = need - bits.size                      # under log2(M) bits, keyed so it reads as noise
        bits = np.concatenate([bits, (_keyed_uniform(_D_FILL, seed, pad, nonce) < 0.5).astype(np.int64)])

    w = (1 << np.arange(b - 1, -1, -1)).astype(np.int64)
    d = const[(bits[:need].reshape(-1, b) * w).sum(1)]        # (n_sym,) in row-major (t, k) order

    T = max(1, int(np.ceil(n_sym / K)))
    K_last = n_sym - (T - 1) * K                    # modes the final row uses, 1..K
    rows = [ref[None, :] @ B]                       # the differential reference row
    walk = ref.astype(complex).copy()
    mk = mode_keys(K, seed, nonce, rows=T)          # what keeps a recovered mode unreadable
    bs = _beacon_symbols(0, T, seed, nonce) if skip else None
    at = 0
    for t in range(T):
        k = K if t < T - 1 else K_last
        if skip:
            walk[:skip] = walk[:skip] * bs[t]       # the row this is, carried rather than guessed
        walk[skip:skip + k] = (walk[skip:skip + k] * d[at:at + k]
                               * mk[t, :k])         # each mode carries its own differential
        rows.append(walk[:skip + k][None, :] @ B[:skip + k])   # nested, used skip+k modes far
        at += k
    # every row, the reference included, at its own power: see row_radius
    return to_iq(_radiate(np.concatenate(rows, axis=0), L, seed, nonce, rng=rng),
                 unit_rms=unit_rms)


def _require_full(lit: int, L: int, what: str) -> None:
    """Every mode is lit (research/rank_leak.md): a writer that leaves one dark is refused."""
    if int(lit) != int(L):
        raise ValueError(
            f"every mode is lit: the screen carries all L={L} modes, got {what}.  A dark mode is a "
            "noise-only direction an observer without the key reads the width from "
            "(research/rank_leak.md); a short message uses fewer modes on its final row only.")


def read(iq, L: int, K: int, seed, *, M: int = 4, offset: int = 0,
         phase: float = 0.0, carrier: float = 0.0, row0: int = 0,
         skip: int = 0, scale: float = 1.0) -> np.ndarray:
    """Read the waveform: screen -> keyed mode coordinates -> differential decode -> bits.

    No phase reference and no pilot: the differential across rows cancels each mode's own phase
    and the row-rate part of any carrier offset.  Returns the recovered bits, including whatever
    keyed filler :func:`compose` padded with -- a caller needing the exact payload length carries
    it in its own framing."""
    b = _bits_per_symbol(M)
    d = read_soft(iq, L, K, seed, offset=offset, phase=phase, carrier=carrier, row0=row0,
                  skip=skip, scale=scale)
    if not d.size:
        return np.zeros(0, dtype=np.int64)
    idx = _slice_psk(d, M).ravel()
    return ((idx[:, None] >> np.arange(b - 1, -1, -1)) & 1).astype(np.int64).ravel()


def read_soft(iq, L: int, K: int, seed, *, offset: int = 0,
              phase: float = 0.0, carrier: float = 0.0, row0: int = 0,
              skip: int = 0, scale: float = 1.0) -> np.ndarray:
    """The de-rotated differential, ``(rows-1, K)``, before any slice: the soft values.

    :func:`read` computes exactly this and then rounds it to the nearest constellation point.
    What that rounding costs depends on where the link sits.  Ranking symbols by angular distance
    to a decision boundary and erasing the least confident -- a model-free probe, nothing assumed
    about the noise and nothing fitted -- gives:

        SNR      as read   erase 10%   erase 20%
        12 dB     0.0020      0.0000      0.0000
         9 dB     0.0292      0.0069      0.0028
         6 dB     0.1316      0.0996      0.0767
         3 dB     0.2979      0.2762      0.2571

    At 12 dB every error is a marginal call and a tenth of the symbols carry all of them.  At
    9 dB the confidence still buys a factor of ten.  By 3 dB it buys 1.2: the constellation is
    genuinely smeared, and the least-confident quintile errs at 0.474 against 0.188 for the most
    confident, a spread of 2.5 where at 6 dB it is 13.

    So on this link the recoverable margin is largest in the middle, and an outer code fed these
    values has the most to gain around 9-12 dB.

    Values are raw rather than log-likelihood ratios, because an LLR needs a noise model and this
    package fits none.  A caller holding an SNR estimate can form its own; a caller without one
    can still rank, erase, or weight by ``abs(angle(d ** M))``, which needs no model at all.
    """
    K, L = int(K), int(L)
    skip = int(skip)                               # modes below the payload -- a beacon, or none
    nonce = None                                   # no nonce is carried; see the note above
    if scale != 1.0:                               # the sender's scale, read continuously
        A = _scaled_rows(_require_finite(iq, "read"), L, seed, nonce, offset, phase, scale,
                         carrier)[:, :K + skip]
        if A.shape[0] < 2:
            return np.zeros((0, K), dtype=complex)
    else:
        S = screen(_require_finite(iq, "read"), L, offset=offset)
        if S.shape[0] < 2:
            return np.zeros((0, K), dtype=complex)
        A = project(S, mode_basis(K + skip, L, seed, nonce, phase, carrier))
    A = A[:, skip:]                                # the payload sits above whatever is below it
    d = A[1:] * np.conj(A[:-1])                     # differential: phase-only, magnitude drops out
    if d.shape[0]:                                  # the keyed per-mode rotation comes back off,
        d = d * np.conj(mode_keys(K, seed, nonce,   # indexed from the row the capture began on
                                  first=int(row0), rows=d.shape[0]))
    if carrier:
        # the measured carrier, used a second time: an offset advances every row by 2.pi.carrier,
        # which rotates the whole differential constellation.  Same nu, no loop.
        d = d * np.exp(-2j * np.pi * float(carrier))
    return d


# ---------------------------------------------------------------------------
# Coherent payload -- square QAM on the keyed modes, for rate
# ---------------------------------------------------------------------------
# The differential mode above buys carrier-blindness by spending the row-to-row phase, which costs
# the ~3 dB every differential link costs and confines it to PSK (a differential amplitude would
# need a phase reference anyway).  PSK saturates at log2(M) and is far from Gaussian at high SNR,
# which is what puts that link at ~47% of Shannon.
#
# The coherent mode spends one row instead: the screen's first row is a keyed reference that
# measures each mode's complex gain, and square QAM rides the rest.  The overhead is one row in
# T+1 -- 1.7% at T=600, where OFDM spends around 14% on cyclic prefix, pilots and guard tones --
# and that margin is where the rate comes from.  Measured at K=63, L=64, T=600, uncoded: 1024-QAM
# decodes at BER < 1e-3 at 45 dB and delivers 9.83 bits/s/Hz, against 802.11ax MCS11's 7.17
# (1024-QAM, rate 5/6, 20 MHz, 1 SS).
#
# Carrier offset splits into two halves here, and the coherent read handles one of them.  With
# n = t*L + l, exp(2.pi.i.f.n) factors into a per-column ramp exp(2.pi.i.nu.l/L) and a per-row
# phase exp(2.pi.i.nu.t), nu = f*L:
#
#   the per-column half is a coordinate of the aperture (see :func:`_aperture`), so passing the
#   measured ``carrier`` to :func:`read_qam` removes it exactly -- no tracking loop, no tuned
#   constant;
#
#   the per-row half is the phase the differential mode cancels for free and this mode does not.
#   The gain is measured from the reference row only, so a row-rate phase accumulates against
#   that one reference; the read removes it, de-rotating row t (counted from the reference) by
#   exp(-2.pi.i.nu.t) -- the same sign read_soft applies once per differential.
#
# Measured at L=64, T=600, clean, with ``carrier=nu`` passed: with the rows de-rotated, BER is
# 0.0000 at every M from 16 to 1024 and every offset up to 1e-3 cycles/sample; without that step
# the same frames read ~0.42.  Gross offset is a front-end job.
#
# The blind read (:func:`decode`) needs the carrier far more precisely than the geometry does.
# Discovery reads nu off the geometry read, which sees only the per-column ramp: measured (on the
# dark mode it replaced), it lands 1e-4 off at 40 dB and up to 3e-3 at 20 dB, and over 600 rows
# 1e-4 is 0.38 rad -- far past
# the 0.03 rad a 1024-QAM corner tolerates.  The per-row phase has T rows of lever, so it is read
# off the payload itself (:func:`_row_carrier`) and the aperture re-evaluated there.  The order M
# is then read the way the differential alphabet is: the smallest square lattice the de-keyed
# symbols close on (:func:`_qam_order`).

def _gray(n):
    """Binary reflected Gray code."""
    return n ^ (n >> 1)


def _qam(M: int) -> np.ndarray:
    """Square M-QAM at unit average power, Gray-coded independently on each axis.

    Gray mapping matters for the delivered rate rather than for the symbol errors: a
    nearest-neighbour slip costs one bit instead of an average of log2(M)/2 of them."""
    M = int(M)
    s = int(round(np.sqrt(M)))
    if s * s != M or (s & (s - 1)) != 0:
        raise ValueError(f"M must be a square power of two (4, 16, 64, 256, 1024); got {M}")
    half = int(round(np.log2(s)))
    pts = np.empty(M, dtype=complex)
    for i in range(s):
        for q in range(s):
            pts[(_gray(i) << half) | _gray(q)] = (2 * i - (s - 1)) + 1j * (2 * q - (s - 1))
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _reference_row(K: int, seed, nonce=None) -> np.ndarray:
    """The keyed unit-modulus reference row.

    Keyed rather than all-ones so the screen carries no known-plaintext row: an all-ones row puts
    the plain sum of the aperture's rows on the air, which is one linear equation in B given away
    for free.  Unit modulus so it measures gain without spending extra power."""
    return np.exp(1j * 2.0 * np.pi * _keyed_uniform(_D_REF, seed, int(K), nonce))


def compose_qam(bits, K: int, L: int, seed, *, M: int = 16,
                unit_rms: bool = True, rng=None) -> np.ndarray:
    """Write the coherent waveform: bits -> Gray-coded square QAM on the keyed modes.

    Row 0 is the keyed reference row the receiver measures the channel with; rows 1.. carry the
    payload.  Same screen, same ravel, same absence of any transform -- only the alphabet and the
    reference differ from :func:`compose`.  Like :func:`compose`, it sends nothing the message did
    not: a short final row uses fewer modes rather than a grid of filler.  Like :func:`compose`,
    every row goes out at its own power (:func:`row_radius`), the reference row included, and every
    mode is lit: ``K`` must be L."""
    bits = np.asarray(bits).ravel().astype(np.int64)
    if bits.size and (bits.min() < 0 or bits.max() > 1):
        raise ValueError("bits must be 0 or 1")
    K, L = int(K), int(L)
    _require_full(K, L, f"K={K}")
    const = _qam(M)
    b = _bits_per_symbol(M)
    nonce = None
    B = mode_basis(K, L, seed, nonce)
    n_sym = int(np.ceil(bits.size / b)) if bits.size else 0
    if n_sym == 0:
        return to_iq(_radiate(_reference_row(K, seed, nonce)[None, :] @ B, L, seed, nonce, rng=rng),
                     unit_rms=unit_rms)
    need = n_sym * b                                # to the symbol boundary, not to a full grid
    if bits.size < need:                            # under log2(M) bits, keyed so it reads as noise
        pad = need - bits.size
        bits = np.concatenate([bits, (_keyed_uniform(_D_FILL, seed, pad, nonce) < 0.5).astype(np.int64)])
    w = (1 << np.arange(b - 1, -1, -1)).astype(np.int64)
    sym = const[(bits[:need].reshape(-1, b) * w).sum(1)]              # (n_sym,) row-major (t, k)

    # No padding, for the reason in :func:`compose`: a short final row is carried on fewer modes,
    # which the nested aperture makes free, rather than on filler the message never asked to send.
    T = max(1, int(np.ceil(n_sym / K)))
    K_last = n_sym - (T - 1) * K
    rows = [_reference_row(K, seed, nonce)[None, :] @ B]
    at = 0
    for t in range(T):
        k = K if t < T - 1 else K_last
        rows.append(sym[at:at + k][None, :] @ B[:k])
        at += k
    return to_iq(_radiate(np.concatenate(rows, axis=0), L, seed, nonce, rng=rng),
                 unit_rms=unit_rms)


def read_qam(iq, L: int, K: int, seed, *, M: int = 16,
             offset: int = 0, phase: float = 0.0, carrier: float = 0.0):
    """Read the coherent waveform: measure the channel off the reference row, then slice QAM.

    There is no carrier tracking loop and no tuned constant.  ``carrier`` is used for both of its
    halves: the mode-axis half is a coordinate of the aperture (see :func:`_aperture`), and the
    row-rate half exp(2.pi.i.carrier.t) is divided out of row t, counted from the reference row.
    No caller pre-processing is needed.

    This is the informed reader: it takes M and the carrier as given and the reference on row 0
    at ``offset``.  :func:`decode` measures all three and reads the payload itself.

    Each row's power (:func:`row_radius`) comes off with the gains: keyed, exactly; open, as each
    row's own level reads it (:func:`_row_levels`)."""
    K, L = int(K), int(L)
    _qam(M)                                            # validates M
    b = _bits_per_symbol(M)
    A = _coherent_project(iq, L, K, seed, offset, phase, carrier)
    if A.shape[0] < 2:
        return np.zeros(0, dtype=np.int64)
    Y = _dekeyed(A, 0, K, seed, carrier, L=L)
    if seed is None:
        Y = _row_levels(Y, M, L=L)
    return _qam_bits(Y, M)


def _qam_bits(Y, M: int) -> np.ndarray:
    """Slice de-keyed symbols to M-QAM and unpack MSB-first."""
    b = _bits_per_symbol(M)
    flat = _slice_qam(Y, M).ravel()
    return ((flat[:, None] >> np.arange(b - 1, -1, -1)) & 1).astype(np.int64).ravel()


def _coherent_project(iq, L: int, K: int, seed, offset: int, phase: float, carrier: float,
                      scale: float = 1.0) -> np.ndarray:
    """Mode coordinates ``(rows, K)`` of every row the capture holds, at the given geometry --
    read continuously through :func:`_scaled_rows` when the scale is not 1."""
    if scale != 1.0:
        return _scaled_rows(iq, L, seed, None, offset, phase, scale, carrier)[:, :int(K)]
    return project(screen(iq, L, offset=offset), mode_basis(K, L, seed, None, phase, carrier))


def _dekeyed(A, r: int, K: int, seed, carrier: float, *, L: int) -> np.ndarray:
    """The symbols after the reference row ``r``: every mode divided by the one gain the keyed
    reference measures, and row t after it de-rotated by exp(-2.pi.i.carrier.t).

    One gain for every mode -- the mean of the reference row's per-mode reads -- not each mode's
    own.  Every keyed mode spans the whole band, so a channel reaches each mode's own coordinate
    with nearly the same gain, the band's mean response, and everything else it does is mixing
    between modes, which no per-mode gain undoes.  What one reference row reads per mode is that
    gain plus the row's noise and its own share of the mixing, and dividing it out of every payload
    row spends both again on each symbol; the mean measures the one gain there is with the noise of
    K reads.  Measured in the mesh's fast path at the true geometry (entroptics-mesh
    ``phy._common_gain``, 40 frames a cell, decoded per-mode / one gain): 4-QAM 1/2 at 8.6 dB
    34 / 40 against 40 / 40, 16-QAM 1/2 at 12 dB 30 / 40, 64-QAM 2/3 at 19.6 dB 29 / 40; and here,
    at equal rates with the receiver's samples between the sender's (tests/_ratechannel.py, 16-QAM,
    noiseless): on the row read -12.4 -> -14.0 dB half a sample off and -21.7 -> -23.5 dB a tenth
    off, on the band read (:func:`_band_coords`) -27.0 -> -31.3 and -39.2 -> -42.5 dB.

    Keyed, row t after the reference also has its power taken off: the gain read on the reference
    carries ``r_0`` and row t carries ``r_t`` (:func:`row_radius`, indexed from the reference as
    screen row 0), so row t is multiplied by ``r_0 / r_t`` -- exact, from the key, which is why
    ``L`` (the radius's shape) is required.  Open, the radius is the sender's own and the rows come
    back at ``r_t / r_0``; :func:`_row_levels` measures it where the payload rows are known."""
    h = A[int(r), :int(K)] * np.conj(_reference_row(K, seed, None))   # the reference: unit modulus
    g = complex(np.mean(h)) if h.size else 1.0
    Y = A[int(r) + 1:, :int(K)] / (g if abs(g) > 1e-300 else 1e-300)
    if seed is not None and Y.shape[0]:
        rad = row_radius(int(L), seed, None, rows=Y.shape[0] + 1)
        Y = Y * (rad[0] / rad[1:])[:, None]
    if carrier and Y.shape[0]:
        Y = Y * np.exp(-2j * np.pi * float(carrier) * np.arange(1, Y.shape[0] + 1))[:, None]
    return Y


def _row_levels(Y, M=None, n=None, L=None) -> np.ndarray:
    """Open-mode coherent rows brought to unit level, each by its own measurement.

    With ``seed=None`` the per-row power is the sender's own draw (:func:`row_radius`), so the
    reader measures it.  ``n`` counts the payload symbols in C order, so a short final row is
    levelled on the symbols it carries.  With the order ``M`` unknown only the first step is
    taken.  The steps:

      1. power: each row divided by its mean power over its symbols (unit for any square QAM).
         Its relative error is ``sqrt(Var|x|^2 / n) / 2``, 4% for 256-QAM on 60 modes, and far
         worse for a short row: one inner 64-QAM point reads the level 4x low.  With the width
         ``L`` given, it is weighed against the law before anything else: the radius is
         Gamma(L), so every row's level sits about the burst's common level with relative
         variance 1/(4L), and the two reads are combined by their precisions.
      2. search: 101 scales over +-5 sd of that start, scored by the hard-decided least-squares
         cost ``|Z - a Q(Z/a)|^2``.  A fit from the start alone can settle on a wrong scale:
         noiseless, 256-QAM on 60 modes left 3 of 120 rows in error that way, none with the
         search.  Scored after the division instead, every residual shrinks as 1/a^2, and a row
         with residual phase chose a level 15% high.
      3. least squares on the sliced symbols until the decisions stop changing, then the same
         fit on soft decisions (EM): each symbol's posterior mean under complex noise of variance
         N0, pooled over the burst and re-read every pass, axis by axis.  Hard decisions bias the
         fit where a row's symbol errors are many.
      4. with ``L``, the law again: each row's level is combined with the burst's common level,
         the row weighted by the observed information of its soft fit, so a row of few or
         ambiguous symbols takes the common level and a full clean row keeps its own.

    Coded, against a reader handed the true radius (lab row_energy/adopted/levels.py, 600
    packets a cell, L=64, K=60): 16-QAM 1/2 at 12.5 dB 10 failures against 9, 64-QAM 3/4 at
    22.5 dB 3 against 3, 256-QAM 2/3 at 27 dB 7 against 3, 256-QAM 3/4 at 28.5 dB 2 against 0.
    Power alone failed 12, 37, 135 and 168 of them, and power, least squares and EM without the
    law 9, 65, 19 and 28 (a 901-symbol 64-QAM 3/4 frame ends in a row of one symbol).

    Every row is lifted to unit level, a row of noise included, so this is for rows that are the
    payload."""
    Y = np.asarray(Y)
    if Y.ndim != 2 or not Y.size:
        return Y
    w = (np.ones(Y.shape, dtype=bool) if n is None
         else (np.arange(Y.size) < int(n)).reshape(Y.shape))
    p = np.sum(np.abs(Y) ** 2 * w, axis=1) / np.maximum(w.sum(axis=1), 1)
    g = np.sqrt(np.where(p > 0, p, 1.0))                # each row's level; Z = Y / g
    Z = Y / g[:, None]
    if M is None:
        return Z
    pts, prev = _qam(M), None
    # The power read's own precision (relative, on the level): 4 n / Var|x|^2.
    sd = 0.5 * np.sqrt(np.var(np.abs(pts) ** 2) / np.maximum(w.sum(axis=1), 1))
    # A constant-modulus alphabet (4-QAM) reads its level exactly from the power (Var|x|^2 = 0);
    # the floor keeps that meaning without the 0/0 that turned the level into NaN.
    sd = np.maximum(sd, 1e-6)
    if L is not None:
        # The start, too, weighs the row's power read against the law: a row of few symbols reads
        # its power badly (one inner 64-QAM point reads the level 4x low), so it starts near the
        # burst's common level instead, and the search is centred there.
        pe, pp = 1.0 / sd ** 2, 4.0 * int(L)
        m = float(np.sum(pe * g) / np.sum(pe))
        g = (pe * g + pp * m) / (pe + pp)
        sd = 1.0 / np.sqrt(pe + pp)
        Z = Y / g[:, None]
    # The search: 101 scales over +-5 sd of the start -- a resolution of 0.1 sd, which the
    # least-squares fit below then finishes.  A search grid, not a threshold.
    a = 1.0 + sd[:, None] * np.linspace(-5.0, 5.0, 101)[None, :]          # (rows, grid)
    X = Z[:, None, :] / a[:, :, None]                                       # (rows, grid, K)
    # Scored in the row's own units, |Z - a Q(Z/a)|^2, which is the least-squares (ML) cost of the
    # hard-decided row.  Scored after the division instead, every residual shrinks as 1/a^2, and
    # a row with residual phase picked a level 15% high (a 256-QAM burst that decoded without the
    # radius then failed).
    res = np.sum(np.abs(Z[:, None, :] - a[:, :, None] * pts[_slice_qam(X, M)]) ** 2
                 * w[:, None, :], axis=2)
    g = g * a[np.arange(Z.shape[0]), np.argmin(res, axis=1)]
    Z = Y / g[:, None]
    for _ in range(8):                      # a bound on the iterations, not a decision
        idx = _slice_qam(Z, M)
        if prev is not None and np.array_equal(idx, prev):
            break
        s = pts[idx] * w
        num = np.real(np.sum(np.conj(s) * Z, axis=1))
        den = np.sum(np.abs(s) ** 2, axis=1)
        ok = (den > 0) & (num > 0)
        g = g * np.where(ok, num / np.where(ok, den, 1.0), 1.0)
        Z = Y / g[:, None]
        prev = idx
    # Hard decisions bias the fit where the row's symbol errors are many.  So the fit is finished
    # on soft ones: each symbol's posterior mean under complex noise of variance N0 (pooled over
    # the burst, re-read every pass), axis by axis since a square QAM is two PAM axes -- the EM
    # iteration for each row's scale.
    lv = np.unique(np.round(np.real(pts), 12))
    s = pts[_slice_qam(Z, M)] * w
    n0 = max(float(np.sum(np.abs(Z - s) ** 2 * w) / max(int(w.sum()), 1)), 1e-12)
    info = np.zeros(Y.shape[0])
    for _ in range(8):                      # a bound on the iterations, not a decision
        num, e2, info, res = (np.zeros(Y.shape[0]), np.zeros(Y.shape[0]), np.zeros(Y.shape[0]),
                              0.0)
        for x in (np.real(Z), np.imag(Z)):
            d = (x[..., None] - lv) ** 2
            lw = -d / n0
            pw = np.exp(lw - lw.max(axis=-1, keepdims=True))
            pw /= pw.sum(axis=-1, keepdims=True)
            m2 = pw @ lv ** 2
            num += np.sum(x * (pw @ lv) * w, axis=1)
            e2 += np.sum(m2 * w, axis=1)
            res += float(np.sum((pw * d).sum(axis=-1) * w))
            # the observed information on the row's scale (Louis): the complete-data information
            # less the posterior variance of the score, which is large where the symbol's
            # identity is uncertain -- a row of one ambiguous symbol says little about its level
            t = x[..., None] * lv - lv ** 2
            vt = np.sum(pw * t ** 2, axis=-1) - np.sum(pw * t, axis=-1) ** 2
            info += np.sum((m2 / (n0 / 2) - vt / (n0 / 2) ** 2) * w, axis=1)
        a = np.where(e2 > 0, num / np.where(e2 > 0, e2, 1.0), 1.0)
        g = g * np.where(a > 0, a, 1.0)
        Z = Y / g[:, None]
        n0 = max(res / max(int(w.sum()), 1), 1e-12)
    if L is None:
        return Z
    # the law's prior on each row's level, weighed against the row's own read
    own = np.maximum(info, 0.0) / g ** 2                                   # 1 / var of each read
    m = float(np.sum(own * g) / np.sum(own)) if np.sum(own) > 0 else float(np.median(g))
    prior = 4.0 * int(L) / m ** 2                                          # 1 / (m^2 / 4L)
    g = (prior * m + own * g) / (prior + own)
    return Y / g[:, None]


def _reference_at(A, K: int, seed) -> int:
    """Which row is the keyed reference: the row most coherent with it.

    The reference row is a vector the key alone predicts, so dividing it out leaves the channel's
    one complex gain on every mode, while any payload row or pre-burst noise leaves a scatter --
    ~1 against ~1/sqrt(K).  An argmax, so a capture that starts before the burst still finds the
    row the gains are measured on."""
    ref = _reference_row(int(K), seed, None)
    num = np.abs(A @ np.conj(ref))
    den = np.sqrt(np.sum(np.abs(A) ** 2, axis=1) * int(K))
    return int(np.argmax(num / np.maximum(den, 1e-300)))


def _row_carrier(Y, nu0: float) -> float:
    """The row-rate carrier, in cycles per row, read off the coherent payload itself.

    ``Y`` is the de-keyed payload before any row de-rotation, so row t carries exp(2.pi.i.nu.t).
    The fourth power takes the data off a square lattice (it is quarter-turn symmetric) and leaves
    exp(8.pi.i.nu.t) on the row sum z_t -- the same propagation-constant read :func:`_read_carrier`
    makes on the sample axis, made here on the row axis, where the lever is T rows rather than L
    samples.

    A lag-k read arg(sum_t z_{t+k} conj z_t) / (8.pi.k) knows nu modulo 1/(4k).  Lag 1 is
    placed on the branch nearest ``nu0`` (the discovery's read, well inside the 1/8 that needs),
    and each doubling of the lag is placed on the branch nearest the read before it, to half the
    capture.  Then the full coherent sum |sum_t z_t exp(-8.pi.i.nu.t)| is maximized inside the
    last lag's half-branch.  No grid, no loop gain, no threshold: closed forms and one bounded
    settle, kept only if it does not make the read worse."""
    z = np.sum(np.asarray(Y) ** 4, axis=1)
    T = int(z.size)
    nu, last, k = float(nu0), 1, 1
    if T < 2:
        return nu
    while k <= T // 2:
        s = complex(np.sum(z[k:] * np.conj(z[:-k])))
        if s == 0:
            break
        a, per = float(np.angle(s)) / (8.0 * np.pi * k), 1.0 / (4.0 * k)
        nu = a + round((nu - a) / per) * per
        last, k = k, 2 * k
    t = np.arange(1, T + 1)
    f = lambda v: -abs(complex(np.sum(z * np.exp(-8j * np.pi * float(v) * t))))
    w = 1.0 / (8.0 * last)
    res = _minimize_scalar(f, bounds=(nu - w, nu + w), method="bounded",
                           options={"xatol": 1e-10})
    return float(res.x) if f(res.x) <= f(nu) else nu


def _coherent_read(iq, L: int, K: int, seed, offset: int, phase: float, carrier: float,
                   scale: float = 1.0):
    """The blind coherent read at a measured geometry: ``(Y, nu, r)`` -- the de-keyed payload with
    both halves of the carrier off, the carrier it measured, and the reference row's index in the
    capture.  ``carrier`` is the discovery's read and only places the first branch."""
    K = int(K)
    A = _coherent_project(iq, L, K, seed, offset, phase, carrier, scale)
    if A.shape[0] < 2:
        return np.zeros((0, K), dtype=complex), float(carrier), 0
    r = _reference_at(A, K, seed)
    nu = _row_carrier(_dekeyed(A, r, K, seed, 0.0, L=L), carrier)
    A = _coherent_project(iq, L, K, seed, offset, phase, nu, scale)   # the column half, at nu
    return _dekeyed(A, r, K, seed, nu, L=L), nu, r


QAM_LADDER = (4, 16, 64, 256, 1024)   # the square orders compose_qam writes: the hypotheses


def _lattice_closure(Y, M: int) -> float:
    """How firmly the symbols sit on the M-QAM lattice: -mean cos(2.pi.x/step) over both axes,
    floored at 0.

    Every point of a square M-QAM lies on an odd multiple of half the lattice step, so each term
    is exactly -1 on the lattice and the closure is 1.  It is signed, so position counts as well as
    period: a set on the even multiples (the lattice shifted by half a step) reads -1 and is
    refused.  That matters because unit-power lattices of successive orders have steps in a ratio
    of ~2 (2.05, 2.01, 2.00 from 16 up): a 16-QAM point sits on an even multiple of the 64-QAM
    half-step, so the unsigned read closes 16-QAM at 64 (0.94) where this one reads 0.  The step is
    read off the alphabet, as :func:`_slice_qam` does."""
    step = float(np.min(np.abs(np.diff(np.unique(np.real(_qam(M)))))))
    x = np.concatenate([np.real(Y).ravel(), np.imag(Y).ravel()])
    return float(max(0.0, -np.mean(np.cos(2.0 * np.pi * x / step)))) if x.size else 0.0


def _qam_order(Y, ladder=QAM_LADDER):
    """Which square lattice rides, and how much of it the source used: ``(M, closure,
    redundancy, closures)``.

    Each rung of the ladder is a hypothesis, and the order is the one whose lattice the symbols
    close on: the argmax of :func:`_lattice_closure`, ties going to the smallest.  That is the
    differential alphabet's rule -- the smallest hypothesis that closes -- in the form the signed
    read allows: a lattice closes on its own constellation and not on a coarser one (a 16-QAM
    point sits half a step off the 64-QAM lattice), so the rung that closes is unique and no rise
    between rungs has to be weighed.  No level is involved.

    Separation, measured at L=64, K=60, 600 rows, carrier 0.0192 cycles/row, discovery's carrier
    handed in 5e-4 off (closures at 4/16/64/256/1024):
        4-QAM     clean 1.00/0.00/0.26/0.77/0.00   10 dB 0.38/0.00/0.00/0.00/0.00
        16-QAM    clean 0.16/1.00/0.00/0.00/0.04   15 dB 0.12/0.26/0.00/0.00/0.00
        64-QAM    clean 0.14/0.04/1.00/0.00/0.00   21 dB 0.13/0.03/0.25/0.00/0.00
        256-QAM   clean 0.14/0.03/0.01/0.99/0.00   27 dB 0.14/0.03/0.00/0.27/0.00
        1024-QAM  clean 0.14/0.03/0.01/0.00/0.98   33 dB 0.14/0.03/0.01/0.01/0.26
    The coarsest rung holds ~0.14 on any denser QAM -- the finite constellation's own envelope,
    not a lattice -- so the read gives way where the true closure falls below that: measured,
    1024-QAM at 30 dB closes at 0.06 and reads 4.  That is where the payload is already in error
    at a few percent (see research/coherent_blind.md).

    A constant source is one point, which closes on whichever lattice it happens to sit nearest;
    a source confined to the inner points of 64-QAM still reads 64, since those points are not on
    the 16-QAM lattice.  ``redundancy`` -- first-order entropy of the sliced symbols against
    log2(M), from ``entroptics.sequence.block_entropies`` -- makes a skewed source visible, as it
    does for the differential alphabet (1.00 constant, 0.33 for the inner sixteen of 64)."""
    c = np.array([_lattice_closure(Y, m) for m in ladder])
    best = int(np.argmax(c))
    M = int(ladder[best])
    sym = _slice_qam(Y, M).ravel()
    H1 = float(_sequence.block_entropies(sym, 1)[0]) if sym.size else 0.0
    redundancy = float(min(1.0, max(0.0, 1.0 - H1 / np.log2(M))))
    return M, float(c[best]), redundancy, c


# ---------------------------------------------------------------------------
# Interference -- a flat waveform makes any concentration foreign
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Interference:
    """What stands above this waveform's own flat spectrum.

    ``strehl``     power in the dominant coherent mode (``Aperture.strehl``).  Small when the
                   screen carries only our own flat modes; it climbs as anything concentrated
                   arrives.
    ``dominance``  the leading mode's excess over the mean eigenvalue (``Aperture.dominance``),
                   already normalized to [0, 1] by the library -- the read to threshold on.

    Both are entroptics reads, reported raw.  There is no derived "clean = 1.0" ratio, because
    the clean baseline depends on the geometry and the capture length and a normalization would
    hide that dependence rather than remove it.  Calibrate against a clean capture at the geometry
    in use -- the numbers in :func:`interference` are one such ladder.
    """
    strehl: float
    dominance: float


def interference(iq, L: int, *, max_rows: int | None = None) -> Interference:
    """Is something in the band that is not ours?

    This waveform has a flat spectrum by construction -- that is what makes it look like noise --
    so it has no dominant mode of its own.  A carrier, a tone, a narrowband interferer or another
    transmitter is concentrated by nature and lands as exactly the dominant mode our own signal
    does not have.  So the interference read is entroptics' ``strehl``, referenced to the 1/K a
    clean screen shows: no spectrum analyser, no reference capture, no template.

    Measured against a CW interferer at L=64, K=58, 400 rows -- strehl / dominance:
    clean 0.032 / 0.016, -13 dBc 0.064 / 0.050, -10 dBc 0.105 / 0.092, -6 dBc 0.210 / 0.199,
    0 dBc 0.504 / 0.496.  Monotone, and it tracks the error rate the interferer causes.

    The RFI labels are not reported.  entroptics' footprint dichotomy -- persistent narrowband
    (phi_F <= phi_T) against broadband transient -- assumes the feature axis is frequency; here it
    is time, so those labels do not transfer.  The fills themselves are generic and stay available
    through ``Aperture.footprints``."""
    ap = Aperture(screen(iq, int(L), max_rows=max_rows), window=None)
    return Interference(float(ap.strehl), float(ap.dominance))


# ---------------------------------------------------------------------------
# Discovery -- the receiver is given the key, and nothing else
# ---------------------------------------------------------------------------
# Everything about the transmission is read out of the transmission: the screen width, where a row
# starts, its scale and carrier, whether a beacon rides, and which alphabet the payload uses.  The
# only thing agreed in advance is the key, and for an open broadcast not even that --
# ``seed=None`` is the zero-state aperture anyone can derive.  A parameter the receiver must be
# told is a parameter the codec has failed to measure, so there are none.  The mode count is not
# among them: every mode is lit, which is the spec rather than an agreement (research/rank_leak.md).

# The aperture is continuous.  A mode shape is a function of a real argument rather than a list of
# samples, so the receiver evaluates it wherever its sampling instants actually landed.  That
# makes a fractional timing offset -- and a clock-rate difference, which is the same thing
# accumulating -- a parameter to be measured on the same axis as everything else, rather than an
# impairment to be corrected.  Nothing resamples the capture; the signal only ever meets a matrix.
#
# The family is forced.  Rows that stay orthonormal at every real offset are rows with
# shift-covariance, and the eigenfunctions of translation are the complex exponentials, so a
# continuously shiftable aperture is necessarily a keyed mixture of them.  The key lives in the
# mixture: each mode is a dense keyed combination of every exponential, so a mode is spread across
# the whole band and the aperture stays unknowable.  One keyed coefficient per mode instead of a
# dense mixture would be one exponential per mode, which is OFDM; the mixture is what makes it
# something else.
#
# The shift is then a per-mode phase on the coefficients, which is why orthonormality survives it
# exactly: B(d) = C . diag(e^{2.pi.i.m.d/L}) . E, and unitary times diagonal-unitary times unitary
# is unitary.  Measured: ||B B^H - I|| < 1e-13 at every offset tried, worst 9.2e-14 over
# phase and carrier together.

@functools.lru_cache(maxsize=256)
def _coeffs(L: int, seed, nonce) -> np.ndarray:
    """The aperture's identity: keyed spectral coefficients, (L, L), orthonormal rows.

    This is what the seed names.  The aperture at any offset is these coefficients read out at
    that offset, so the identity is offset-free and the offset is not part of the key."""
    C = _canonical_qr(_keyed_normal(_D_BASIS, seed, (L, L), nonce)).conj().T
    C.flags.writeable = False
    return C


@functools.lru_cache(maxsize=256)
def _scaled_coeffs(L: int, seed, nonce) -> np.ndarray:
    """``_coeffs`` conjugate-transposed and scaled by ``sqrt(L)``: the map a scaled row read applies
    last, from the row's least-squares mode weights to the screen's modes.  A pure function of
    (L, seed, nonce), and the settle's row read rebuilt it on every evaluation -- cached read-only."""
    Cc = _coeffs(L, seed, nonce).conj().T * np.sqrt(int(L))
    Cc.flags.writeable = False
    return Cc


@functools.lru_cache(maxsize=256)
def _scan_coeffs(L: int, seed, nonce, m: int) -> np.ndarray:
    """The first ``m`` rows of ``_coeffs`` conjugated and scaled by ``sqrt(L)``: the filter the scan
    reads its CM modes through.  Cached read-only (the scan rebuilt it on every call)."""
    cm = np.conj(_coeffs(L, seed, nonce)[:int(m)]) * np.sqrt(int(L))
    cm.flags.writeable = False
    return cm


@functools.lru_cache(maxsize=256)
def _aperture_dual(L: int, seed, nonce) -> np.ndarray:
    """``_aperture`` (at zero phase and carrier) conjugate-transposed: the map the joint read applies
    last, from the row's own samples to the screen's modes.  Cached read-only."""
    AH = _aperture(L, seed, nonce).conj().T
    AH.flags.writeable = False
    return AH


@functools.lru_cache(maxsize=2048)
def _aperture(L: int, seed, nonce, phase: float = 0.0, carrier: float = 0.0) -> np.ndarray:
    """The full L-mode aperture at real position ``l + phase``, on carrier ``carrier``: (L, L).

    Two continuous axes, and they are the same idea twice.  ``phase`` is a real number of samples
    -- where the receiver's sampling instants fell.  ``carrier`` is a real number of cycles per
    row -- how far its oscillator sits from the transmitter's, since a carrier offset multiplies
    sample n by e^{2.pi.i.f.n} and within a row that shifts the aperture's mode index by f*L.  So
    timing is a phase on the sample axis and carrier is a phase on the mode axis, and both are
    parameters to be measured rather than impairments to be corrected.

    Orthonormality survives both exactly: each is a unitary diagonal, and unitary times
    diagonal-unitary is unitary.  Measured ||B B^H - I|| < 1e-13 at every value tried, worst
    9.2e-14 over phase and carrier together."""
    C = _coeffs(L, seed, nonce)
    m = np.arange(L)
    l = np.arange(L) + float(phase)
    E = np.exp(2j * np.pi * np.outer(m, l) / L) / np.sqrt(L)
    A = C @ E
    if carrier:
        A = A * np.exp(2j * np.pi * float(carrier) * l / L)[None, :]
    A.flags.writeable = False
    return A



# ---------------------------------------------------------------------------
# The scale as a continuous coordinate
# ---------------------------------------------------------------------------
# The sender chooses its own sample rate, so a row of L transmitted samples spans L / rho of the
# receiver's, rho = tx_rate / rx_rate, and that is a real number: coprime, harmonic and matched
# rates are the same case.  Receiver sample n of a row that starts at real receiver position s
# sits at transmitted position u = (n - s) * rho, and the aperture's mode shapes are functions of
# a real argument, so the receiver evaluates them there.  Nothing resamples the capture.
#
# The exponentials are indexed symmetrically about zero here.  At whole transmitted positions the
# two indexings are the same matrix; between them, the symmetric one is the band-limited
# continuation a transmitter's converter actually produces, and the one-sided one aliases the
# upper half of the band.
#
# A row then holds more samples than modes (rho < 1), so its coordinates are the least-squares
# solution against the aperture at those positions.  It costs two small inverses per capture: a
# row whose first sample lies delta into it has exponential matrix E0 . Lambda_delta, where E0
# depends only on (rho, the row's sample count) and Lambda_delta is a per-exponential phase, so its
# Gram is Lambda* G0 Lambda and its dual is Lambda* G0^-1 E0^H.  A row holds one of two counts.
#
# A row with FEWER samples than modes (rho > 1, a sender wider than the receiver) is
# underdetermined: its least-squares fit can place any mode's energy anywhere, so it carries no
# evidence about the geometry and reads NaN.  Measured (on the dark mode the constant-modulus read
# replaced): admitting such rows let the ratio search settle at 1.0015, on a fit that was not a lock.

def _signed(L: int) -> np.ndarray:
    m = np.arange(int(L))
    return np.where(m < L // 2, m, m - L)


@functools.lru_cache(maxsize=16)
def _scaled_gram(L: int, rho: float, n: int):
    """The exponentials at a row's ``n`` sample positions on scale ``rho``, their conjugate, and the
    inverse of their Gram: ``(E0, G0^-1, E0*)``.  A pure function of the three, so it is cached.  A
    settle moves the start and the carrier at a fixed scale far more often than it moves the scale,
    and rebuilding the inverse for every one of those steps was most of what a settle cost.  The
    conjugate is cached with it: both row reads consume ``E0.conj()`` (or its transpose), which was
    re-allocated on every one of the settle's hundreds of evaluations."""
    j = np.arange(int(n))
    E0 = np.exp(2j * np.pi * np.outer(j * float(rho), _signed(L)) / int(L))
    # E0^H E0 in closed form (:func:`_dirichlet_gram`): the product cost n L^2 per scale, and a
    # settle moves the scale a few hundred times.
    G = _dirichlet_gram(L, rho, n) if float(rho) < 1.0 else E0.conj().T @ E0
    G0inv = np.linalg.inv(G)
    E0c = E0.conj()
    for a in (E0, G0inv, E0c):
        a.flags.writeable = False
    return E0, G0inv, E0c


def _row_window(s, L: int, rho: float):
    """The received samples a row read takes: ``(first, count)`` for rows starting at real
    positions ``s``.  They are the samples inside the row's own transmitted cells, transmitted
    positions [-1/2, L - 1/2) -- each transmitted sample's half-sample either side -- so the window
    is centred on the row and moves by one sample where a start crosses a half-sample cell edge,
    not where it crosses a whole sample.

    Why centred: the read models a row as periodic over its L positions, so a sample at transmitted
    position L - e is read as the row's position -e.  A window [0, L) -- the first sample at or
    after the start -- puts that sample in the read as soon as the start passes a whole sample:
    at matched rates, the next row's first sample read as this row's, a jump of one sample's error
    exactly at the true start.  Centred, the window's edges sit half a sample from the row's own
    samples at every start within half a sample of the true one.

    A scale a hair above 1 (a sender clocked a few ppm fast) makes rows a fraction of a sample
    shorter than L; read by cells, a row then holds L - 1 samples wherever a cell edge falls in
    that fraction, and reads NaN (underdetermined).  Where L received samples overreach the row by
    at most ``SQUARE_REACH`` of a transmitted sample -- L (scale - 1); 1/16 is ~1000 ppm at L=64 --
    the row is read on L of them, and the read stays square.  A sender wider than that is still
    underdetermined (:func:`_scaled_coords`)."""
    L, rho = int(L), float(rho)
    lo = np.asarray(s, dtype=float) - 0.5 / rho
    first = np.ceil(lo - 1e-12).astype(np.int64)
    count = np.ceil(lo + L / rho - 1e-12).astype(np.int64) - first
    if 1.0 < rho and L * (rho - 1.0) <= SQUARE_REACH:
        count = np.maximum(count, L)
    return first, count


def _scaled_coords(iq, L: int, seed, nonce, starts, scale: float, carrier: float = 0.0) -> np.ndarray:
    """Mode coordinates ``(len(starts), L)`` of rows starting at real receiver positions ``starts``
    when one transmitted sample spans ``1 / scale`` received ones, read on the samples
    :func:`_row_window` gives.  NaN for a row the capture does not hold whole, or one with fewer
    samples than modes."""
    x = np.asarray(iq).ravel()
    L, rho = int(L), float(scale)
    s = np.asarray(starts, dtype=float).ravel()
    first, count = _row_window(s, L, rho)
    delta = first - s
    out = np.full((s.size, L), np.nan, dtype=complex)
    ms = _signed(L)
    Cc = _scaled_coeffs(L, seed, nonce)
    for n in np.unique(count):
        rows = np.flatnonzero((count == n) & (first >= 0) & (first + n <= x.size))
        if n < L or rows.size == 0:
            continue
        j = np.arange(n)
        E0, G0inv, E0c = _scaled_gram(L, rho, int(n))
        X = x[first[rows, None] + j[None, :]]
        if carrier:
            X = X * np.exp(-2j * np.pi * float(carrier) * (j[None, :] + delta[rows, None]) * rho / L)
        W = (X @ E0c) @ G0inv.T
        W = W * np.exp(-2j * np.pi * np.outer(delta[rows], ms) * rho / L)
        out[rows] = W @ Cc
    return out


# ---------------------------------------------------------------------------
# The joint read: the neighbouring rows' tails, modelled
# ---------------------------------------------------------------------------
# :func:`_scaled_coords` reads a row as the band-limited continuation of that row alone.  The
# signal between the transmitted instants is the band-limited continuation of the whole stream,
# so the neighbouring rows' content reaches into the row being read through the converter's
# tails, which fall only as 1/d.  That is the floor a receiver at another rate meets: about -23 to
# -24 dB on a dark mode at L=128, noiseless.  It is the other rows' content, not noise, so it is
# modelled rather than averaged.
#
# Two parts, both a least-squares read of the received samples, as the row read is:
#
#   * The window.  The received samples from ``margin`` transmitted samples before the row to
#     ``margin`` after it are modelled as one band-limited signal, periodic over the window.  The
#     exponentials are those of the row read on a longer period, so the Gram is again E0 . Lambda
#     with Lambda a per-exponential phase, factorized and cached per (scale, sample count).
#   * The far field.  What lies beyond the window reaches into it as sum_k s_k sinc(u - k), k
#     outside, and the window's own periodic wrap is an error of the same form.  Over the window
#     that is sin(pi u) times a smooth function whose singularities sit beyond the edges, and a few
#     transmitted samples at geometric distances beyond each edge -- 1, 2, 4, ... L/2 -- span it.
#     They are fitted with the window and discarded.  Their columns depend on where the row's
#     first sample fell, so they are solved per row, in the complement of the window's span: a
#     q x q system per row, q = 2 log2(L) (14 at L=128).
#
# Measured on tests/_ratechannel.py, noiseless, true geometry, L=128, K=124 (four spare modes as a
# probe of the floor, from before every mode was lit): the dark modes read
# -68.9 dB at rho 0.514 and -66.5 dB at 0.833, against -24.0 / -23.1 dB for the row read.  That is
# the channel model's own floor, which quantizes each position to 1/1024 of a sample.  The noise
# the read passes is 1.01-1.05 times the row read's.  Each octave of the skeleton counts: 3 per
# side read -46 to -50 dB, 5 read -54 to -67, 7 read -67 to -69.  research/joint_read.md has the
# rest, including what did not work.
#
# The skeleton needs spare samples, q at least: a window of span W transmitted samples holds
# about W / rho received ones, and the model has W + q unknowns.  So the margin is L/4, or wider
# when the scale is near 1, up to ``JOINT_REACH`` rows.  A row whose window the capture does not
# hold -- the first or the last -- is read by the row read.

def _joint_window(L: int, rho: float):
    """``(margin, skeleton)`` for the joint read at scale ``rho``: the margin in transmitted samples
    either side of the row, and the skeleton's positions relative to the row's first sample.  None
    when no margin within ``JOINT_REACH`` rows holds the spare samples (a scale at or near 1, or
    above it)."""
    L, rho = int(L), float(rho)
    if not 0.0 < rho < 1.0:
        return None
    d = 2.0 ** np.arange(int(np.log2(max(L // 2, 1))) + 1)       # 1, 2, 4, ... L/2
    q = 2 * d.size
    span = (q + 2) / (1.0 / rho - 1.0)        # q spare samples, and one either side for rounding
    m = max(L // 4, int(np.ceil((span - L) / 2.0)))
    if m > JOINT_REACH * L:
        return None
    return m, np.concatenate([-m - d, L + m - 1 + d])


@functools.lru_cache(maxsize=16)
def _joint_gram(L: int, rho: float, n: int, margin: int):
    """The window's exponentials at ``n`` received positions, period L + 2*margin: orthonormal
    bases of their span and of its complement, the least-squares dual, and the exponentials at the
    row's L transmitted positions.  A pure function of the four, so it is cached."""
    Lp = int(L) + 2 * int(margin)
    ms = _signed(Lp)
    E0 = np.exp(2j * np.pi * np.outer(np.arange(int(n)) * float(rho), ms) / Lp)
    Qf, _ = np.linalg.qr(E0, mode="complete")
    H = np.linalg.solve(E0.conj().T @ E0, E0.conj().T)
    F = np.exp(2j * np.pi * np.outer(np.arange(int(L)), ms) / Lp)
    out = (Qf[:, :Lp].copy(), Qf[:, Lp:].copy(), H, F)
    for a in out:
        a.flags.writeable = False
    return out


def _joint_coords(iq, L: int, seed, nonce, starts, scale: float, carrier: float = 0.0) -> np.ndarray:
    """Mode coordinates ``(len(starts), L)`` of rows starting at real receiver positions ``starts``,
    read jointly with what surrounds them (see the section note): the coordinates
    :func:`_scaled_coords` returns, without the neighbouring rows' tails in them.  A row whose
    window the capture does not hold is read by :func:`_scaled_coords`.  Near equal rates, where no
    margin within JOINT_REACH rows holds the spare samples, every row is read by the band read
    (:func:`_band_coords`)."""
    x = np.asarray(iq).ravel()
    L, rho = int(L), float(scale)
    s = np.asarray(starts, dtype=float).ravel()
    win = _joint_window(L, rho)
    if win is None:
        return _band_coords(x, L, seed, nonce, s, rho, carrier)
    m, ks = win
    Lp, q = L + 2 * m, ks.size
    first = np.ceil(s - m / rho - 1e-12).astype(np.int64)
    count = np.ceil(s + (L + m) / rho - 1e-12).astype(np.int64) - first
    u0 = (first - s) * rho                  # the window's first sample, in transmitted samples
    out = np.full((s.size, L), np.nan, dtype=complex)
    done = np.zeros(s.size, dtype=bool)
    c64 = _use_c64()                     # single-precision reads, opt-in (see :func:`_use_c64`)
    AH = _aperture_dual(L, seed, nonce)
    if c64:
        AH = AH.astype(np.complex64)
    sg = np.where(ks % 2 == 0, 1.0, -1.0) / np.pi
    for n in np.unique(count):
        rows = np.flatnonzero((count == n) & (first >= 0) & (first + n <= x.size))
        if rows.size == 0 or n < Lp + q:
            continue
        Q, Qp, H, F = _joint_gram(L, rho, int(n), m)
        j = np.arange(n)
        U = j[None, :] * rho + u0[rows, None]       # each sample's transmitted position in its row
        X = x[first[rows, None] + j[None, :]]
        if carrier:
            X = X * np.exp(-2j * np.pi * float(carrier) * U / L)
        # the skeleton: sinc(U - k) at whole k, written as sin(pi U) (-1)^k / (pi (U - k))
        N = np.sin(np.pi * U)[:, :, None] * (sg / (U[:, :, None] - ks))
        if c64:                          # cast the dense products (zgemm -> cgemm); the far-field
            X = X.astype(np.complex64)   # q x q solve below then runs in c64 on c64 inputs
            N = N.astype(np.complex64)
            Q = Q.astype(np.complex64); Qp = Qp.astype(np.complex64)
            H = H.astype(np.complex64); F = F.astype(np.complex64)
        r = rows.size
        Nt = N.transpose(1, 0, 2).reshape(n, r * q)
        # its fit in the complement of the window's span, through whichever basis is smaller
        if Qp.shape[1] < Q.shape[1]:
            Bh = Qp.conj().T
            BN = (Bh.real @ Nt + 1j * (Bh.imag @ Nt)).reshape(-1, r, q).transpose(1, 0, 2)
            BNh = BN.conj().transpose(0, 2, 1)
            G = BNh @ BN
            h = (BNh @ (X @ Qp.conj())[..., None])[..., 0]
        else:
            Bh = Q.conj().T
            QN = (Bh.real @ Nt + 1j * (Bh.imag @ Nt)).reshape(-1, r, q).transpose(1, 0, 2)
            QNh = QN.conj().transpose(0, 2, 1)
            NT = N.transpose(0, 2, 1)
            G = NT @ N - QNh @ QN
            h = (NT @ X[..., None])[..., 0] - (QNh @ (X @ Q.conj())[..., None])[..., 0]
        far = np.linalg.solve(G, h[..., None])[..., 0]
        X = X - (N @ far[..., None])[..., 0]        # the window with the far field taken off
        ph = np.exp(-2j * np.pi * np.outer(u0[rows], _signed(Lp)) / Lp)
        C = (X @ H.T) * (ph.astype(np.complex64) if c64 else ph)
        out[rows] = (C @ F.T) @ AH                  # the row's own samples, then its modes
        done[rows] = True
    if not done.all():
        rest = np.flatnonzero(~done)
        out[rest] = _scaled_coords(x, L, seed, nonce, s[rest], rho, carrier)
    return out


# ---------------------------------------------------------------------------
# The band read: the joint read's counterpart near equal rates
# ---------------------------------------------------------------------------
# Near scale 1 the joint read has no spare samples to fit its skeleton with (a window holds about as
# many received samples as transmitted ones), and past JOINT_REACH rows of margin it gives way.  The
# row read that used to stand in for it there has a floor set by where the receiver's samples fall
# between the sender's: it models a row as periodic over its own L samples, and at equal rates the
# neighbouring rows' converter tails reach every sample of a row unless the two clocks happen to
# sample the same instants.  Measured (tests/_ratechannel.py, matched rates, 16-QAM, noiseless,
# true geometry, payload error after the per-mode gain): -148 dB with the samples on the sender's
# instants, -21.7 dB a tenth of a sample off, -14.8 dB a quarter off, -12.4 dB half a sample off.
# Two radios on separate clocks sample at a position that drifts through every fraction (0.75 ppm
# at 2.4 Msps: once every 0.6 s), so on the air that floor is the common case, and it is what sank
# blind 16- and 64-QAM at matched rates (the constant-modulus evidence of a 16-QAM row rose from
# 0.45 to ~0.85 of the null, where a Gaussian row reads 1).
#
# The band read takes each row on the samples of a window JOINT_REACH rows either side, against
# each transmitted sample's own band-limited continuation -- the converter's sinc, centred on that
# sample -- rather than against the row's periodic one:
#
#     s_l = rho * sum_n x[n] sinc(rho (n - s) - l),        l = 0 .. L-1,
#
# then its modes, as every read takes them.  At rho <= 1 the sincs sampled at the receiver's
# instants are orthogonal with norm 1/rho, so this is the least-squares read of the row with every
# other row's samples in the model, truncated to the window; the truncation is the read's only
# error, and it falls as the margin grows.  Measured as above (per-mode / common gain): a tenth of
# a sample off -39.2 / -42.5 dB, a quarter -30.2 / -35.0, half a sample -27.0 / -31.3, at a margin
# of four rows (L=64); one row of margin reads -23.5 / -27.1 at half a sample, sixteen rows -35.4.
# The band is the sender's whole band -- no guard, so the tails fall only as 1/d, and the margin is
# what that costs.  It is a projection of the received samples onto the modes' waveforms on the air,
# as every read here is: nothing resamples the capture and no filter shapes it.
#
# The kernel depends on where the window's first sample falls inside a transmitted sample, so it is
# built per position, on a grid of BAND_POSITIONS per sample: a position within 1/2048 of a sample
# is a mean phase error over the band of (pi/2048)^2/3, -61 dB, under anything the read resolves.
# Samples outside the capture count as zero: a burst read near the capture's edge has fewer of its
# neighbours' tails in view, which is the capture's limit, not the read's.

BAND_POSITIONS = 1024    # positions per transmitted sample the band read's kernels are built at (a
                         # precision bound: see the note above); not in any decision


def _band_coords(iq, L: int, seed, nonce, starts, scale: float, carrier: float = 0.0) -> np.ndarray:
    """Mode coordinates ``(len(starts), L)`` of rows starting at real receiver positions ``starts``,
    read on the band-limited continuation of every transmitted sample over a margin of JOINT_REACH
    rows either side (see the section note): what :func:`_joint_coords` reads near equal rates.  NaN
    where :func:`_scaled_coords` is NaN -- a row the capture does not hold whole, or a sender wider
    than the receiver by more than SQUARE_REACH."""
    x = np.asarray(iq).ravel()
    L, rho = int(L), float(scale)
    s = np.asarray(starts, dtype=float).ravel()
    out = np.full((s.size, L), np.nan, dtype=complex)
    first, count = _row_window(s, L, rho)
    rows = np.flatnonzero((count >= L) & (first >= 0) & (first + count <= x.size))
    if rows.size == 0:
        return out
    m = JOINT_REACH * L
    W = int(np.ceil((L + 2 * m) / rho)) + 2
    lo = np.ceil(s[rows] - (m + 0.5) / rho - 1e-12).astype(np.int64)   # each window's first sample
    u0 = (lo - s[rows]) * rho                     # ... in transmitted samples from the row's start
    xp = np.concatenate([np.zeros(W, dtype=complex), x.astype(complex), np.zeros(W, dtype=complex)])
    j = np.arange(W)
    X = xp[(lo + W)[:, None] + j[None, :]]
    if carrier:                                   # the ramp over each window, as two factors
        X = (X * np.exp(-2j * np.pi * float(carrier) * j * rho / L)[None, :]
             * np.exp(-2j * np.pi * float(carrier) * u0 / L)[:, None])
    Xr, Xi = np.ascontiguousarray(X.real), np.ascontiguousarray(X.imag)
    ell = np.arange(L)
    sgn = np.where(ell % 2 == 0, 1.0, -1.0) * (rho / np.pi)
    S = np.empty((rows.size, L), dtype=complex)
    q = np.rint(u0 * BAND_POSITIONS).astype(np.int64)
    for qq in np.unique(q):
        sel = np.flatnonzero(q == qq)
        U = qq / BAND_POSITIONS + j * rho                                   # (W,)
        # rho sinc(U - l) = rho (-1)^l sin(pi U) / (pi (U - l)): one sine per sample, and the
        # samples that land on a transmitted instant exactly take its value
        D = U[:, None] - ell[None, :]
        hit = np.abs(D) < 0.5 / BAND_POSITIONS
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.where(hit, rho, (np.sin(np.pi * U)[:, None] * sgn[None, :]) / D)
        S[sel] = Xr[sel] @ K + 1j * (Xi[sel] @ K)       # a real kernel: two real products
    out[rows] = S @ _aperture(L, seed, nonce).conj().T
    return out


def _scaled_held(n: int, start: float, L: int, scale: float) -> int:
    """Rows a capture of ``n`` samples holds whole from real position ``start``, when a row spans
    L / scale received samples."""
    return int(np.floor((int(n) - float(start) - 1) / (int(L) / float(scale))))


def _scaled_cm(iq, L: int, seed, nonce, start: float, scale: float, carrier: float,
               rows: int, joint: bool = False, reduce=None) -> float:
    """The constant-modulus read (:func:`_cm_rows`) at a continuous scale, on all L modes of
    ``rows`` rows.  ~1 off the geometry, ~0 on it.  A geometry whose rows the capture does not
    hold, or that is underdetermined, reads :func:`_cm_missing` -- rows of nothing, worse than any
    real read, and finite so a bounded settle can step away from it.  ``joint`` reads the rows with
    :func:`_joint_coords`, which costs ten to twenty times as much and has no neighbouring rows'
    floor.

    ``reduce`` turns the rows' reads (a 1-D array, one per row) into the score, lower better; the
    mean when None.  A caller that knows more about where its signal sits among the rows -- a
    burst that fills only some of them -- says so here."""
    stride = L / float(scale)
    # The rows the capture holds, up to the number asked for: the ceiling is the capture itself.
    # Asking for rows past its end would read them as missing and score every geometry alike.
    held = _scaled_held(np.asarray(iq).size, start, L, scale)
    rows = min(int(rows), held)
    if rows < 2:
        return _cm_missing(L)
    A = (_joint_coords if joint else _scaled_coords)(
        iq, L, seed, nonce, float(start) + np.arange(rows) * stride, scale, carrier)
    if A.size == 0 or np.isnan(A).any():
        return _cm_missing(L)
    R = _cm_rows(A.real ** 2 + A.imag ** 2)
    return float(np.mean(R) if reduce is None else reduce(R))


def _scaled_rows(iq, L: int, seed, nonce, offset: int, phase: float, scale: float,
                 carrier: float, max_rows=None) -> np.ndarray:
    """All-mode coordinates of the rows a capture holds whole, from the row the geometry places at
    ``offset - phase`` (the same position the integer read uses), at continuous ``scale``.  The
    rows are read jointly (:func:`_joint_coords`): this is the payload's read, where the floor is
    the BER."""
    stride = int(L) / float(scale)
    s0 = float(offset) - float(phase)
    T = _scaled_held(len(iq), s0, L, scale)
    if max_rows is not None:
        T = min(T, int(max_rows))
    if T < 1:
        return np.zeros((0, int(L)), dtype=complex)
    A = _joint_coords(iq, L, seed, nonce, s0 + np.arange(T) * stride, scale, carrier)
    return A[~np.isnan(A).any(axis=1)]



JOIN_SEARCH_ROWS = 4096  # the DEFAULT reach of the join hunt, in candidate rows.  It is how
                         # late a join this receiver can recover, and nothing in a capture derives
                         # it: a receiver cannot know how long it was not listening.  So it is the
                         # deployment's to set, and ``discover``, ``decode`` and
                         # ``StreamReceiver.acquire`` all take ``join_rows`` to say so.
                         #
                         # What it costs, measured at L=64 where a 2 Msps stream makes 31,250
                         # rows a second:
                         #
                         #     reach      acquire      stream covered
                         #      1024        1.23 s            0.03 s
                         #      4096        1.81 s            0.13 s
                         #     16384        4.56 s            0.52 s
                         #     65536       15.35 s            2.10 s
                         #
                         # Linear past about 4096, near 0.22 ms per candidate row, and correct at
                         # every reach tried.  So this default covers a tenth of a second of that
                         # stream, and joining a stream that has run for a minute means 1,875,000
                         # candidates and some seven minutes of hunting.  A deployment that needs
                         # late joins raises it and pays that; one that starts its receivers with
                         # its transmitters does not.
                         #
                         # A count of candidates, so the deepest join a given reach recovers is
                         # reach - 1: at the default, 4095 is exact and 4096 reads row 13 at BER
                         # 0.4995.
                         #
                         # Nothing is thresholded against it, and the hunt reaches only as far as
                         # the capture can evidence (see :func:`_align_and_close`), so a short or
                         # noisy capture never pays the margin a deep search costs.  Past the
                         # reach the read declines rather than misleads: such a join scores 1.1 to
                         # 2.4 times the null where a real one scores hundreds, so a reader
                         # refuses it instead of taking an invented alignment.


def _alignment_score(dp, span, K: int, ladder, depth: int, probe: int) -> np.ndarray:
    """How firmly each candidate alignment closes, for every candidate at once.

    Candidate r scores |mean_{t,k} (dp[t,k] span[r+t,k])^m|, and the keystream is unit modulus, so
    normalizing dp once makes that (dp^m)(span^m) exactly -- a correlation in r rather than a
    search over r.  With P = dp^m and S = span^m, the product G = S P^T has G[j,t] = sum_k
    S[j,k] P[t,k], and the score is the shifted diagonal sum sum_t G[r+t,t]: a stride view of G,
    costing nothing.  So one matmul per rung does the work of a four-thousand-iteration Python
    loop, which costs 1.40 s of a 1.49 s discovery.  Materializing the candidates instead -- a
    126 MB temporary per rung -- measures slower than the loop, because the cost is never the
    arithmetic.

    The depth is walked in blocks only to bound that temporary, which a wide probe would otherwise
    grow as probe squared.  The block size changes nothing the function returns; it is not in the
    decision path."""
    P = dp / np.maximum(np.abs(dp), 1e-30)      # |span| = 1, so |P span| = 1 already and the
    score = np.zeros(int(depth))                # power needs no renormalizing afterwards
    blk = max(1, int(4_000_000 // max(int(probe), 1)))
    for m in ladder:
        Pm = (P ** int(m)).T
        for lo in range(0, int(depth), blk):
            hi = min(int(depth), lo + blk)
            G = (span[lo:hi + probe] ** int(m)) @ Pm
            s0, s1 = G.strides
            diag = np.lib.stride_tricks.as_strided(
                G, shape=(hi - lo, int(probe)), strides=(s0, s0 + s1))
            np.maximum(score[lo:hi], np.abs(diag.sum(axis=1)) / float(probe * K),
                       out=score[lo:hi])
    return score


def _settled_rows(d, K: int, seed, nonce, ladder, depth: int, T: int) -> dict:
    """For each reach, the row it settles on -- or nothing, if this capture cannot evidence it.

    The probe -- how many rows of the differential are weighed -- widens until the alignment
    repeats.  Widening the reach would be unsound, since a wider range contains rows the narrower
    one did not and a repeat could mean the true row is still outside; widening the probe is the
    sound kind, being the same candidate range judged on more evidence.  The best wrong closure
    falls like 1/sqrt(rows) while the true one is fixed by SNR, so rows are the only currency, and
    a fixed number of them was a constant standing in the decision path.

    One scan serves every reach.  A candidate's score does not depend on how many other
    candidates are weighed against it, so the answer at reach R is the argmax of the first R
    entries of the same full-depth score.  Sharing the scan costs depth where scanning each reach
    separately costs sum(R) = 2 x depth, and holds discovery inside a 2.0 s
    budget.  Reaches settle at their own probes regardless, exactly as if each had widened alone.

    Measured over a 4096-row reach: at 12 and 6 dB the alignment is already right on 32 rows
    (0.630 against a best wrong 0.081, and 0.150 against 0.071); at 3 dB it reads 2327, 3564 and
    1231 on 32, 64 and 128 rows and the true row on 256 and 512.  Running out of capture before a
    reach repeats is reported by omission rather than papered over -- that is a read this capture
    could not make, and the caller uses the reaches it could."""
    reaches, r = [], min(int(depth), 32)
    while True:
        reaches.append(r)
        if r >= int(depth):
            break
        r = min(int(depth), r * 2)
    settled, probe, prev = {}, min(T, 32), None
    while True:
        # The whole range is drawn once and every candidate is a slice of it, so a scan costs one
        # hash call rather than one per row hunted.  Drawn per candidate, the hashing rather than
        # the arithmetic would be the search.
        span = np.conj(mode_keys(K, seed, nonce, first=0, rows=int(depth) + probe))
        score = _alignment_score(d[:probe], span, int(K), ladder, int(depth), probe)
        if prev is not None:
            for R in reaches:
                if R not in settled:
                    a = int(np.argmax(score[:R]))
                    if a == int(np.argmax(prev[:R])):
                        settled[R] = a
        prev = score
        if probe >= T or len(settled) == len(reaches):
            return settled
        probe = min(T, probe * 2)


def _align_and_close(d, K: int, seed, nonce, ladder,
                     depth: int) -> tuple[int, np.ndarray, bool]:
    """The row alignment and the closure ladder, searched together.

    Each candidate alignment is scored at each candidate alphabet, and the true pair is the one
    that closes -- at its own exponent, which for the alphabets in use is small and
    noise-tolerant.  The alphabet is the smallest that closes.

    Searching the alignment at one exponent that every alphabet divides -- the ladder's lowest
    common multiple -- would look alphabet-agnostic and is not: for a ladder of powers of two that
    exponent is the largest alphabet, and raising a differential to the 16th power amplifies phase
    noise sixteenfold, destroying the cliff the search depends on.  Measured that way the
    alignment reads 11 at 12 dB and 26 at 6 dB against a truth of 0, and a wrong alignment is the
    wrong keystream, so the payload comes out at BER 0.50 where an oracle reads 0.0009.  Searched
    jointly there is no common multiple to choose.

    The read is keyed, and that carries twice: only a holder can align, so an eavesdropper who has
    recovered the coefficients still cannot index the keystream, and a receiver that can align has
    thereby confirmed it holds the right key."""
    T = int(d.shape[0])
    if T < 2:
        return 0, np.zeros(len(ladder))
    best_r = 0
    # A full scan to ``depth``, rather than a widening search.  Widening until the answer repeats
    # is sound where more evidence refines one estimate; here a wider range does not improve the
    # answer, it contains a row the narrower range did not, so a repeat can mean the true row is
    # still outside.  Measured, an early stop returned rows 73, 90 and 221 for true joins at 500,
    # 2000 and 2900, and a wrong alignment is the wrong keystream.
    #
    # The scan is affordable because the keystream is seekable (see :func:`mode_keys`): a
    # candidate costs the same at row 60000 as at row 0.  Measured at 12 dB over 32 rows, the true
    # row was found at joins of 500, 2000 and 3500 with the closure ten times the best wrong
    # candidate, in about a second over four thousand candidates.
    #
    # What bounds the depth is the margin rather than the cost.  The best wrong closure grows like
    # sqrt(log R) while the true one is fixed by SNR, so a deep enough search is eventually
    # swamped: at 6 dB over 32 rows the true closure is 0.150 against a best wrong 0.055, and more
    # rows are what buys the margin back.
    # The probe widens until the alignment repeats.  Widening the DEPTH would be unsound -- a
    # wider range contains rows the narrower one did not, so a repeat there can mean the true row
    # is still outside it -- but widening the probe is the sound kind: the same candidate range,
    # judged on more evidence.  That is what buys back the margin the depth spends.  The best
    # wrong closure falls like 1/sqrt(rows) while the true one is fixed by SNR, so rows are the
    # only currency here, and a fixed number of them was a constant standing in the decision path.
    # Measured over a 4096-row range: at 12 and 6 dB the alignment is already right on 32 rows
    # (0.630 against a best wrong 0.081, and 0.150 against 0.071), so a clean capture pays one
    # extra scan and stops; at 3 dB it reads 2327, 3564 and 1231 on 32, 64 and 128 rows and the
    # true row on 256 and 512.  At 1 dB it does not settle on any probe the capture can offer,
    # which is the true edge of the read rather than a number to raise.
    # The reach is evidenced, not declared.  A wider range is strictly more places for noise
    # to win: the best wrong closure grows like sqrt(log R) while the true one is fixed by SNR,
    # so reaching for a deep join costs margin that a short or noisy capture may not have.
    # Searching the full range regardless buys the deep join at the common case's expense --
    # measured, a 400-row capture at 3 dB starting at row 0 reads its alignment as row 2811 and
    # the payload comes out at BER 0.50.
    #
    # So the answer is the one from the BEST-CLOSING reach that settles.  Settling is not a threshold:
    # it is the probe widening until the alignment repeats (:func:`_resolve_alignment`), which is
    # exact integer agreement.  A reach the capture cannot evidence does not settle and is not
    # taken, so a short capture at 3 dB keeps a shallow answer while a long one takes the full
    # range.  Settling alone is not enough to be believed, though: a stable noise peak repeats too
    # -- measured, a 400-row capture at 3 dB settled on row 2811 at a reach of 4096 -- so the rows
    # the reaches propose are compared by how well each closes over the whole capture.
    #
    # Every reach is tried, and stopping at the first that fails to settle would be wrong: for a
    # capture that truly joined at row 2000, every reach below 2000 excludes the true row, so its
    # failure to settle is expected and says nothing about a deeper one.  Measured, stopping early
    # returned row 0 for true joins at 32, 2000 and 2900, at BER 0.50.  A shallow reach is cheap,
    # so trying all of them costs about twice the deepest scan alone.
    best_col, best_fit = np.zeros(len(ladder)), -1.0
    settled = _settled_rows(d, K, seed, nonce, ladder, int(depth), T)
    for r in sorted(set(settled.values())):     # one entry per distinct settled row
        # Each settled reach is a candidate row, and the aperture says which is right --
        # the same move the carrier makes in :func:`_discover_at`, where a closed-form read and a
        # probe are settled by whichever reads better on the aperture.  Here the candidates
        # are scored by how well the row they name closes, measured on every row the capture has
        # rather than on the probe that proposed it.  That is far more evidence than the scan
        # carries, and it separates cleanly: a true row closes at 0.15 to 0.63 where a stable
        # noise peak closes at 0.008.  A comparison between measured candidates, not a cutoff.
        uu = d * np.conj(mode_keys(K, seed, nonce, first=r, rows=T))
        uu = uu / np.maximum(np.abs(uu), 1e-30)
        col = np.array([float(abs(np.mean(uu ** int(m)))) for m in ladder])
        if float(col.max()) > best_fit:
            best_r, best_col, best_fit = r, col, float(col.max())
    if best_fit < 0.0:                  # nothing settled anywhere: closure of the unshifted read
        uu = d * np.conj(mode_keys(K, seed, nonce, first=0, rows=T))
        uu = uu / np.maximum(np.abs(uu), 1e-30)
        best_col = np.array([float(abs(np.mean(uu ** int(m)))) for m in ladder])
    # Whether it settled is reported separately from what it settled on.  Returning 0 for "no
    # alignment found" would make that answer indistinguishable from a capture that genuinely
    # began on row 0, and the two need telling apart: the alignment needs 64 rows to resolve at
    # any join depth and any SNR down to 6 dB, while the geometry resolves at 4.  A caller
    # widening its prefix therefore sees row0 read 0, 0, 0, 0 and then the truth, so a rule that
    # stops when the answer repeats would stop on the run of defaults.  The separate flag is what
    # lets runner.StreamReceiver.acquire size itself.
    return best_r, best_col, bool(best_fit >= 0.0)

def _payload_at(d, K: int, seed, nonce, ladder, depth: int, row0=None):
    """Alphabet, closure and redundancy for one payload hypothesis.

    ``row0=None`` searches for the row alignment; an integer takes it as given, which is what a
    beacon supplies.  Everything else is the same read, so the two hypotheses are scored on one
    scale and can be compared."""
    if row0 is None:
        row0, c, aligned = _align_and_close(d, K, seed, nonce, ladder, depth)
    else:
        aligned = True
        uu = d
        if uu.shape[0]:
            uu = uu * np.conj(mode_keys(K, seed, nonce, first=int(row0), rows=uu.shape[0]))
            uu = uu / np.maximum(np.abs(uu), 1e-30)
        c = np.array([float(abs(np.mean(uu ** int(m)))) if uu.size else 0.0 for m in ladder])
    dd = d
    if dd.shape[0]:
        dd = dd * np.conj(mode_keys(K, seed, nonce, first=int(row0), rows=dd.shape[0]))
    jump = int(np.argmax(c[1:] / np.maximum(c[:-1], 1e-12))) + 1 if c.size > 1 else 0
    best = jump if c[jump] >= c[0] else 0
    M = int(ladder[best])
    sym = _slice_psk(dd, M).ravel()
    # Redundancy against the alphabet the constellation supports, not the one the source happened
    # to use.  ``sequence.redundancy_rate`` measures against the observed alphabet, which is right
    # for its own question and wrong for this one: a constant payload is maximally skewed and
    # would read 0, because a one-symbol alphabet has no capacity to be redundant against.  H_1
    # over log2(M) asks the question wanted here -- how much of the available alphabet went
    # unused -- and reads 1 for a constant source, 0.5 for two symbols of four, 0 for even use.
    H1 = float(_sequence.block_entropies(sym, 1)[0]) if sym.size else 0.0
    redundancy = float(1.0 - H1 / np.log2(M)) if M > 1 else 0.0
    redundancy = float(min(1.0, max(0.0, redundancy)))
    # The closure decides the constellation; how many symbols happened to appear only describes
    # it, and the redundancy above carries that (it reads ~1 for a constant source).  M is not
    # overridden by the usage count, because fewer symbols appearing is not the same as a smaller
    # alphabet: a source using two of four points sits at 0 and 270 degrees, which is not the
    # two-point constellation, so M=4 stays distinguishable and reading it as 2 costs BER 0.49.
    # Genuine ambiguity needs the used points to lie on the smaller constellation, and where that
    # happens the two readings are the same transmission.
    return M, float(c[best]), redundancy, int(row0), bool(aligned)


def _alphabet(iq, L: int, K: int, seed, nonce, offset: int, max_rows: int,
              phase: float = 0.0, carrier: float = 0.0, ladder=(2, 4, 8, 16),
              depth: int = 64, scale: float = 1.0):
    """Which differential alphabet is riding, and how redundantly the source used it.

    The geometry picks the candidate: raising a differential symbol to the power M lands every
    point of an M-PSK alphabet on the same place, so the mean of the M-th powers has unit modulus
    at the alphabet and ~0 below it.  It closes at every multiple of the true M too, so the answer
    is the smallest that closes -- the same "smallest hypothesis that explains the data" the mode
    count uses.

    The redundancy comes from the sequence itself, through
    ``entroptics.sequence.block_entropies``: how much of the alphabet the source left idle.  That
    is the number that matters on real traffic, because it makes an ambiguity visible.  A source
    that only ever sends two of four symbols is indistinguishable from a two-symbol source when
    its points sit on the smaller constellation, so the read reports the alphabet used and says
    how skewed the usage was.  Measured on the decoded stream: random data and English text both
    give alphabet 4 at redundancy 0.00, while a 1-in-64 sparse pattern gives alphabet 2 at
    redundancy 0.80.

    Returns ``(M, closure, redundancy, row0)``.  If nothing closes, the payload is not
    differential PSK -- which is how a coherent-QAM transmission announces itself."""
    if scale != 1.0:
        A = _scaled_rows(iq, L, seed, nonce, offset, phase, scale, carrier,
                         max(int(max_rows), 8))[:, :int(K)]
        T = A.shape[0]
    else:
        T = _held(len(iq), offset, L)
        T = min(T, max(int(max_rows), 8))
    if T < 3:
        return int(ladder[0]), 0.0, 0.0, 0, False, int(K), False
    if scale == 1.0:
        S = screen(iq, L, offset=offset, max_rows=T)
        A = S @ mode_basis(K, L, seed, nonce, phase, carrier).conj().T
    d = A[1:] * np.conj(A[:-1]) * np.exp(-2j * np.pi * float(carrier))
    # De-rotate before measuring, and find which rotation.  The keyed rotation of
    # :func:`mode_keys` scatters the constellation uniformly, so an alphabet read taken before it
    # is removed closes on nothing -- measured 0.008 against 1.000 after.  That is what it is for,
    # and it is why an eavesdropper cannot read M either; a key holder reads it second.
    #
    # Whether a row beacon rides is measured, not assumed.  Both hypotheses are read off the same
    # differential -- the payload on every occupied mode with the row hunted for, and the payload
    # above a beacon with the row read out of it -- and the larger closure names which is riding.
    # A beaconed capture closes at ~1 on the second and on near nothing on the first, a payload
    # read across the beacon's modes being two unrelated things read as one.
    plain = _payload_at(d, K, seed, nonce, ladder, depth)
    best_read, K_pay, beacon = plain, int(K), False
    # Only under a key.  The open mode carries no rotation (see :func:`mode_keys`), so its payload
    # closes at ~1 at every alignment and on every subset of modes: both hypotheses tie, and the
    # beaconed one can win on nothing and take three payload modes with it.  An open broadcast has
    # no use for a beacon in any case, being joinable at any point already.
    if seed is not None and int(K) > BEACON_MODES and d.shape[0] >= BEACON_FRAME:
        mark = read_beacon(d[:, :BEACON_MODES], seed, nonce)
        if mark is not None:
            cand = _payload_at(d[:, BEACON_MODES:], int(K) - BEACON_MODES, seed, nonce,
                               ladder, depth, row0=int(mark[0]))
            # Each closure against its own null, the modulus of a mean of T x K unit phasors
            # (:func:`closure_significance`): the beaconed hypothesis reads three modes fewer, so
            # compared raw its chance closure sits sqrt(K / (K - 3)) higher (2.4% at L = 64).  That
            # tilt only decides a capture neither hypothesis explains, which the reader's
            # significance cutoff declines anyway -- measured, a simulated bench capture whose join
            # lay past the reach read a beacon at K=61, M=8, 1.8x its null -- but it is a tilt.
            if cand[1] * np.sqrt(int(K) - BEACON_MODES) > best_read[1] * np.sqrt(int(K)):
                best_read, K_pay, beacon = cand, int(K) - BEACON_MODES, True

    return (*best_read, K_pay, bool(beacon))


# ---------------------------------------------------------------------------
# The constant-modulus read -- what places the geometry
# ---------------------------------------------------------------------------
# Every mode is lit (research/rank_leak.md).  A screen whose last mode stays dark is findable by a
# key holder, and by anyone else as well: the windowed covariance at the true width then has a
# direction holding noise only, and an observer without the key finds it by eigen-decomposition --
# measured at L=64 in 10 s from -15 dB, whatever the number of dark modes, and at every rate.  A
# full screen has no such direction.  Its covariance is white, its per-sample power profile flat,
# its lag correlation zero, so every second-order read an observer can take of it reads as noise.
#
# What still marks the true geometry for a key holder is the coordinates themselves.  They are unit
# modulus (every differential symbol; the keyed reference row), or a square lattice's few levels
# (coherent QAM), and the per-row radius scales a whole row at once.  So at the true width, row
# start, scale and carrier each row's energy is spread EVENLY over the keyed modes.  At any other
# cut a row's coordinates are mixtures of many coordinates of two rows, close to complex Gaussian,
# and their energies are those of exponentials: normalized over the row, a draw from the flat
# Dirichlet.
#
# The read is that difference, in the library's own terms: each row's Shannon entropy over its
# mode energies (``entroptics.entropy.shannon_bits``) against log2 m, the even spread, scaled by
# the deficit the flat Dirichlet gives on average (``entroptics.entropy.dirichlet_entropy_moments``,
# the exact Wolpert-Wolf moments).  It is ~1 off the geometry and 0 on it noiseless -- the row's
# fill is complete -- and rises toward 1 as the noise does: on 16 modes of unit-modulus
# coordinates, 0.43 / 0.64 / 0.82 / 0.93 at a per-mode SNR of 6 / 3 / 0 / -3 dB.  Dividing by the
# row's own total takes the radius off exactly.  An energy statistic on keyed projections,
# as the dark mode was: no pilot, no template, no transform.
#
# Measured against the dispersion of the energies (rank_leak.md's lab read, the second-order form
# of the same deficit), on 160 rows of unit-modulus coordinates against Gaussian ones, z of the
# separation at 6 / 3 / 0 / -3 dB: 16 modes 20.0 / 11.5 / 5.6 / 2.2 against 16.6 / 10.3 / 5.1 /
# 2.0; 64 modes 42.8 / 24.2 / 11.7 / 4.3 against 32.0 / 19.8 / 10.7 / 4.6.  The entropy is the
# better-tailed of the two under the null, and the null's mean is a closed form.

CM_MODES = 16            # modes the scans read per hypothesis (the integer scan, the scaled scan, the
                         # period map); settles and comparisons read all L.  A cost bound: the scan
                         # forms m coordinates where the dark read formed one.  At 16 the scan's
                         # evidence at 3 dB matches the dark read's (rank_leak.md, cure (e)).


@functools.lru_cache(maxsize=None)
def _cm_null(m: int) -> float:
    """The flat Dirichlet's mean entropy deficit over ``m`` modes, in bits: what a row of complex
    Gaussian coordinates reads on average (``entroptics.entropy.dirichlet_entropy_moments``)."""
    mean, _ = _lib.entropy.dirichlet_entropy_moments(int(m), 1.0)
    return float(np.log2(int(m)) - mean)


def _cm_rows_py(e) -> np.ndarray:
    """The constant-modulus read of each row of mode energies ``e`` (..., m): the row's entropy
    deficit against an even spread, over the flat Dirichlet's mean deficit.  0 where a row's
    energy is spread evenly over the modes, ~1 where its coordinates are Gaussian.  NaN rows stay
    NaN.

    Each row is held to at most three of the null's own sds above its mean (1 + 3 spread(m),
    :func:`_cm_spread`).  The read is evidence by how far a row falls BELOW the null; a row far
    above it -- energy on a few modes -- is no more evidence against a geometry than a Gaussian
    row is, and a message's own final row is such a row: lit on the few modes the message
    reaches, it read 5.3 at L=64 on 7 modes, which on a 14-row burst weighed as much as five
    Gaussian rows and lost the true start to a chance one (measured on matched 256-QAM bursts in
    the mesh).  Three sds leaves the null itself as it was (it moves 0.1% of its rows).

    Pure-Python/numpy reference (the oracle for the compiled :func:`_cm_rows`)."""
    e = np.asarray(e)
    m = int(e.shape[-1])
    H = np.asarray(_lib.entropy.shannon_bits(np.where(np.isnan(e), 0.0, e), axis=-1), dtype=float)
    out = np.minimum((np.log2(m) - H) / _cm_null(m), 1.0 + 3.0 * _cm_spread(m))
    return np.where(np.isnan(e).any(axis=-1), np.nan, out)


# ---------------------------------------------------------------------------
# Compiled constant-modulus read (numba), oracle-gated against _cm_rows_py.
# The per-row transcendental reduction (-sum p log2 p over the mode energies) is fused with the
# deficit normalization and the cap in one compiled pass; entroptics_rf._fast holds the kernel.
# Used only when numba imports AND the kernel reproduces _cm_rows_py within rtol 1e-9 / atol 1e-12
# on a fixed corpus; ENTROPTICS_FAST=0 forces the Python path.
# ---------------------------------------------------------------------------
_fast_cm_ok: bool | None = None


def _fast_disabled() -> bool:
    return os.environ.get("ENTROPTICS_FAST", "").strip().lower() in ("0", "off", "false", "no")


def _use_c64() -> bool:
    """Opt-in single precision for the dense-LA JOINT read (:func:`_joint_coords`), behind
    ``ENTROPTICS_C64``.  Default OFF: the read runs in complex128, byte-for-byte as before, and
    complex128 is the oracle/fallback.

    When ON, :func:`_joint_coords` casts its dense BLAS products (zgemm -> cgemm) to complex64 --
    lower precision on the SAME projection (no FFT/pilot/resampling; it does not change what is
    computed).  It is admissible only where a sub-ulp perturbation cannot flip a decision:

      * the PAYLOAD read -- the joint coordinates go through slicing + FEC + CRC, which the oracle
        gate holds byte-identical to the complex128 decode on the corpus;
      * NOT the full/blind scan (:func:`_scaled_scan` / :func:`_cm_score`), whose argmin over
        candidate starts has far-apart near-ties a c64 perturbation could flip -- that stays
        complex128.  The §3a LOCAL scan has no far competitor (structurally flip-safe), but on the
        ~24-start local grid the cgemm win is swamped by the cast (measured x86 ~0.7-1.0x), so the
        scan stays double; the win is in the joint read.
      * NOT :func:`_scaled_coords` either -- its matmuls are only ~1.1x in c64 on x86 (below the
        ~1.3x bar) and it is called many times per burst (the settle), so the per-call cast is a
        net cost; it stays double.

    Measured x86 (OPENBLAS_NUM_THREADS=1, fastmath=False): ~1.7-1.9x on :func:`_joint_coords` for a
    sender narrower than the receiver (the joint window fires); at matched rates the read is the
    band read and is untouched.  aarch64 NEON (4x f32 vs 2x f64) was expected to pay off more; it
    does not.  Re-measured on the Pi-4B (Cortex-A72, scipy-openblas, 1 thread): c64 runs 0.84-0.94x
    at coprime rho and ~1.0-1.27x at rho 5/6 -- a net slowdown at the rates that matter, never the
    1.3x bar, as that build's zgemm already matches its cgemm.  And it is not exact there: at
    rho=0.514, 14 dB the single-precision read put 5-41 bit errors per 7680 into a burst c128 reads
    bit-exact (11 of 12 seeds).  c64 stays default-OFF, complex128 is the read, and this lever is
    closed."""
    return os.environ.get("ENTROPTICS_C64", "").strip().lower() in ("1", "on", "true", "yes")


def _gate_cm_rows(_fast) -> bool:
    """True iff the compiled read reproduces _cm_rows_py within rtol 1e-9/atol 1e-12 on a fixed
    corpus: Gaussian energies, near-even rows, all-zero rows, NaN rows, at m = 16 and 64."""
    rng = np.random.default_rng(0xCAFE)
    for m in (2, 16, 64):
        null, cap = _cm_null(m), 1.0 + 3.0 * _cm_spread(m)
        for shape in ((37, m), (120, m), (5, 3, m)):
            e = rng.standard_normal(shape) ** 2 + rng.standard_normal(shape) ** 2
            flat = e.reshape(-1, m)
            if flat.shape[0] >= 4:
                flat[0] = 0.0                      # all-zero row
                flat[1] = 1.0                      # perfectly even row
                flat[2, 0] = np.nan                # NaN row -> NaN
                e = flat.reshape(shape)
            ref = _cm_rows_py(e)
            got = _fast.cm_rows(e, null, cap)
            rm, gm = np.isnan(ref), np.isnan(got)
            if not np.array_equal(rm, gm):
                return False
            if not np.allclose(ref[~rm], got[~gm], rtol=1e-9, atol=1e-12):
                return False
    return True


def _use_fast_cm() -> bool:
    global _fast_cm_ok
    if _fast_disabled():
        return False
    if _fast_cm_ok is None:
        try:
            from . import _fast
            _fast_cm_ok = bool(_gate_cm_rows(_fast))
        except Exception:
            _fast_cm_ok = False
    return _fast_cm_ok


def _cm_rows(e) -> np.ndarray:
    """Constant-modulus row read (see :func:`_cm_rows_py`).  Compiled kernel when gated-available,
    else the pure-Python reference; outputs agree within rtol 1e-9 / atol 1e-12."""
    if _use_fast_cm():
        from . import _fast
        return _fast.cm_rows(e, _cm_null(int(np.asarray(e).shape[-1])),
                             1.0 + 3.0 * _cm_spread(int(np.asarray(e).shape[-1])))
    return _cm_rows_py(e)


@functools.lru_cache(maxsize=None)
def _cm_spread(m: int) -> float:
    """One row's sd of the read under the flat Dirichlet, on the read's scale: the library's exact
    entropy sd over the mean deficit.  0.97 at 2 modes, 0.32 at 16, 0.16 at 64."""
    _, sd = _lib.entropy.dirichlet_entropy_moments(int(m), 1.0)
    return float(sd / _cm_null(int(m)))


def _cm_significance(score, rows, m: int):
    """How far a mean read over ``rows`` rows of ``m`` modes stands below the flat Dirichlet's, in
    its own null sd: ``(1 - score) sqrt(rows) / spread(m)``.

    Hypotheses of different widths are compared on this, not on the score.  The score's null is
    1 at every width, but its spread is not: a row of 2 modes scatters about it three times as far
    as a row of 16, so over the many offsets and carrier probes of a scan the best chance read of a
    narrow width sits far lower than a wide one's.  Compared by score, a 6 dB capture at L=64
    (truth 0.44) was measured locking on width 2 at 0.98; compared by significance, the truth
    stands out by its evidence, which is what the rows are."""
    return (1.0 - np.asarray(score, dtype=float)) * np.sqrt(np.maximum(rows, 0)) / _cm_spread(m)


def _cm_missing(L: int) -> float:
    """What a geometry whose rows the capture does not hold reads: an empty row's unheld deficit,
    ``log2 m`` over the null's mean, plus three of its sds -- above anything :func:`_cm_rows` returns, which holds a row
    to 1 + 3 spread(m).  Finite, so a bounded settle can step away from it."""
    m = int(L)
    return float(np.log2(m) / _cm_null(m) + 3.0 * _cm_spread(m))


@functools.lru_cache(maxsize=1024)
def _scan_modes(L: int, seed, nonce) -> np.ndarray:
    """The first ``min(L, CM_MODES)`` modes of the aperture at carrier zero, conjugated: (m, L).

    Split out because the carrier is only a per-column phase ramp on the aperture, so a search
    over carriers needs these rows and a ramp rather than a new L x L aperture per point.
    Measured: a 9-point carrier grid over 100 widths costs 22 s this way and 131 s building an
    aperture per point."""
    C = np.conj(_aperture(L, seed, nonce, 0.0, 0.0)[:min(int(L), CM_MODES)])
    C.flags.writeable = False
    return C


def _carrier_probes(span: float, n: int) -> np.ndarray:
    """``n`` carrier offsets spanning ``[-span, span]``, in golden-ratio order.

    Not a grid.  A fixed count of evenly spaced points is two arbitrary choices -- how many, and
    where.  The additive recurrence on 1/phi is the canonical low-discrepancy sequence in one
    dimension, phi being the worst-approximable number, so it is the forced answer to "sample this
    interval with no parameters".

    It is extensible, which is what it is for here: every prefix is already well spread, so a
    caller takes as many probes as the capture needs and stops when the answer stops
    changing rather than committing to a count in advance.  An evenly spaced grid has no such
    prefix -- refining it means discarding it and starting again."""
    if span <= 0:
        return np.zeros(1)
    i = np.arange(int(n))
    return float(span) * (2.0 * np.mod(0.5 + i * _PHI, 1.0) - 1.0)


def _cm_score(iq, W: int, seed, nonce, max_rows: int, carrier: float = 0.0):
    """The constant-modulus read (see the section note) at every row offset of one width, on the
    aperture's first ``CM_MODES`` modes: (W,).

    Every offset's rows are projected onto those modes at once -- the windows of the capture times
    the conjugate modes, in blocks so no more than a block of windows is ever stacked -- and each
    offset reads the mean of its rows' deficits.  ~0 at the true geometry and ~1 everywhere else,
    comparably across widths, so the argmin carries no threshold.  It is a projection onto
    aperture modes rather than a correlation against a transmitted template -- there is no
    reference waveform anywhere in it."""
    W = int(W)
    if W < 2:
        return None
    x = np.asarray(iq)
    T = min(int(max_rows), (len(x) - (W - 1)) // W)
    if T < 1:
        return None
    span = T * W
    C = _scan_modes(W, seed, nonce)
    if carrier:                          # the carrier is a column ramp, not a new aperture
        C = C * np.exp(-2j * np.pi * float(carrier) * np.arange(W) / W)[None, :]
    # In single precision, as the radio delivers it: the product is the scan's whole cost and runs
    # four times faster there (measured 0.21 against 0.91 ms per 800 x 100 x 16 block).  A lock then
    # reads ~1e-7 noiseless rather than ~1e-15 -- the scan only shortlists; every score a decision
    # rests on is re-read in double (:func:`_cm_at`).
    C = C.astype(np.complex64)
    V = np.lib.stride_tricks.sliding_window_view(x[:span + W - 1], W)
    Z = np.empty((span, C.shape[0]), dtype=np.complex64)
    blk = max(W, (1 << 20) // W)
    for a in range(0, span, blk):        # stacked a block at a time: the cost is the product
        Z[a:a + blk] = np.array(V[a:a + blk], dtype=np.complex64) @ C.T
    e = (Z.real ** 2 + Z.imag ** 2).reshape(T, W, -1)
    return _cm_rows(e).mean(axis=0)


def _cm_at(iq, L: int, seed, nonce, offset: int, max_rows: int,
           phase: float, carrier: float) -> float:
    """The constant-modulus read at one geometry and one point on the two continuous axes, on all
    L modes."""
    T = _held(len(iq), offset, L)
    T = min(T, int(max_rows))
    if T < 1:
        return float("inf")
    S = screen(iq, L, offset=offset, max_rows=T)
    A = S @ np.conj(_aperture(L, seed, nonce, phase, carrier)).T
    return float(np.mean(_cm_rows(A.real ** 2 + A.imag ** 2)))


@dataclass
class Discovery:
    """Everything the receiver measured, and the evidence for it.

    Three coordinates, not four.  The aperture has a scale and one translation per axis:

        ``L``         the scale -- how many samples a row spans.
        ``position``  translation on the sample axis, a real number of samples.  ``offset`` and
                      ``phase`` are its whole and fractional parts, separate only because slicing
                      an array needs an integer; they are one coordinate.
        ``carrier``   translation on the mode axis, in cycles per row (nu = f*L).

    ``K``        modes carrying payload: L, since every mode is lit, or L - ``BEACON_MODES`` when
                 a row beacon rides the bottom modes.  Not measured -- the full screen is the
                 spec (research/rank_leak.md) -- but kept, because a reader needs the payload's
                 width and a beacon takes three modes of it.  ``M``   the alphabet: M-PSK for a
                 differential payload, square M-QAM for a coherent one (see ``payload``).
    ``score``    the constant-modulus read that settled (L, offset) -- 0 locked noiseless, rising
                 toward ~1 with noise; ~1 not locked.
    ``margin``   the runner-up geometry's score over the winner's.  The runner-up is the best
                 competing (width, offset) at any carrier -- the winner's own worse carrier probes
                 are the same hypothesis read less well, not rivals.
    ``closure``  how firmly the alphabet closed (1 = exactly).
    ``redundancy`` how much of the alphabet went unused: 0 when traffic exercises it evenly, 0.5
                 when half of it is idle, 1 for a constant source.  Reported because a skewed
                 source makes the alphabet genuinely ambiguous, and a receiver should see that
                 alongside the bare answer.
    """
    L: int | None
    offset: int
    K: int
    M: int
    score: float
    margin: float
    closure: float
    phase: float = 0.0
    carrier: float = 0.0
    redundancy: float = 0.0
    row0: int = 0            # the absolute row this capture began on, measured
    # Appended, and it must stay last: Discovery is built positionally, and
    # inserting a field ahead of row0 silently fed row0 the default 0 -- the wrong
    # keystream index, and a decode that returned confident nonsense.
    payload: str = "differential"   # which payload the capture measures as
    aligned: bool = False           # did the row alignment settle, or is row0 its default?
    beacon: bool = False            # does a row beacon ride the bottom modes?  When it does,
                                    # ``K`` is the payload above it and ``row0`` was read rather
                                    # than searched for.
    scale: float = 1.0              # transmitted samples per received one (the sender's rate over
                                    # the receiver's).  1 is the integer screen; anything else is
                                    # read continuously -- see the scale section.
    reference: int = 0              # coherent payload only: the capture row (from ``offset``) the
                                    # keyed reference row was found on.  For a coherent payload
                                    # ``M`` is the square-QAM order, ``closure`` the fourth-power
                                    # closure that named the payload, ``carrier`` the row-rate
                                    # read, and row0/aligned keep their defaults -- that payload
                                    # has no keystream to align.
    lattice: float = 0.0            # coherent payload only: how firmly the symbols closed on the
                                    # M-QAM lattice (1 = exactly) -- the evidence for ``M``.

    @property
    def position(self) -> float:
        """Where the first row starts, in samples -- the single real coordinate that ``offset``
        and ``phase`` are the whole and fractional parts of."""
        return float(self.offset) + float(self.phase)

    @property
    def locked(self) -> bool:
        return self.L is not None


def _read_carrier(iq, L: int, K: int, seed, nonce, offset: int) -> float:
    """Read the oscillator difference off one row.  Closed form: no grid, no loop, no span.

    ``K`` is the modes the reference row spans: L, every mode being lit.

    The differential reference row is ``_reference_row(K, seed, nonce)``, so the samples it puts
    on the wire are a vector the key alone predicts, before any payload exists -- and one an
    observer without the key cannot predict, because the row is keyed rather than all-ones.  A
    carrier multiplies sample ``l`` by ``e^{2.pi.i.nu.l/L}``, so the received reference row
    divided by the predicted one is that ramp and nothing else, and the slope of the ramp is the
    carrier.  A slope is a mean per-step phase advance, ``arg(sum_l r_l conj(r_{l-1}))`` -- the
    propagation constant of PAPER Definition 6.1, read on the sample axis instead of the mode
    axis.

    Measured clean, the read is exact to five decimals at every carrier tried -- nu = 0, 0.0192,
    0.064, 0.192, 0.64, 1.2 and 1.92 all read back with zero error.  1.92 cycles/row is some
    30 000 ppm of a 915 MHz part at 2 Msps, far past the range the geometry read alone
    survives and far past anything an oscillator does.  A wrong key
    predicts the wrong row and reads +14, which no oscillator spec admits, so a bad key fails
    loudly."""
    L, K = int(L), int(K)
    row = np.asarray(iq).ravel()[int(offset):int(offset) + L]
    if row.size < L:
        return 0.0
    predicted = _reference_row(K, seed, nonce) @ _aperture(L, seed, nonce, 0.0, 0.0)[:K]
    r = row / predicted
    step = complex(np.sum(r[1:] * np.conj(r[:-1])))
    return float(np.angle(step) * L / (2.0 * np.pi)) if step != 0 else 0.0


def _settle(fn, start, span) -> tuple[float, float]:
    """Settle the two continuous coordinates on an exact read, bounded around where they were
    measured.

    The pair is settled in one call rather than alternately because one call is simpler; the two
    coordinates are independent, and measured they give the same answer either way (BER 0.00144
    jointly against 0.00143 alternately).  Accuracy comes from the number of rows the read is
    given, not from the optimizer -- see :func:`discover`.

    Powell rather than a gradient method, because the read has no derivative to offer.  The
    bracket comes from the values the coarse read measured, never from a grid.  The result is kept
    only if it beats where it started, so a refinement that wandered cannot make the answer worse
    than the read it refined."""
    p0, c0 = float(start[0]), float(start[1])
    sp, sc = float(span[0]), float(span[1])
    res = _minimize(fn, [p0, c0], method="Powell",
                    bounds=[(p0 - sp, p0 + sp), (c0 - sc, c0 + sc)],
                    options={"xtol": 1e-6, "ftol": 1e-9})
    got = (float(res.x[0]), float(res.x[1]))
    return got if fn(got) <= fn((p0, c0)) else (p0, c0)


# The acquisition sweep reads every (carrier, width) hypothesis, and they are independent: each
# one's constant-modulus read is a single-threaded BLAS matmul (the thread pin is for bit-identity
# within a matmul, not across them) and a nogil deficit kernel, so running them on separate threads
# gives each core its own matmul and leaves the scores bit-for-bit what the serial sweep returns.
# A blind lock otherwise runs one core while the rest idle; env ENTROPTICS_SWEEP_WORKERS overrides.
try:
    _SWEEP_WORKERS = max(1, int(os.environ.get("ENTROPTICS_SWEEP_WORKERS", "") or
                                min(8, os.cpu_count() or 1)))
except ValueError:
    _SWEEP_WORKERS = min(8, os.cpu_count() or 1)


def _sweep(fn, items):
    """Map ``fn`` over ``items``, across threads when there is more than one worker and more than
    one item.  The returned list is in the same order as ``items`` (``ThreadPoolExecutor.map``
    preserves it), so a caller that reduces the results serially gets the same answer -- including
    which of two equal scores wins -- at any worker count."""
    items = list(items)
    if _SWEEP_WORKERS <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=_SWEEP_WORKERS) as pool:
        return list(pool.map(fn, items))


def _discover_at(iq, seed, candidates, nonce, max_rows: int,
                 carrier_span: float) -> Discovery:
    """Read the transmission's own parameters at a fixed row count: the geometry and the
    alphabet, measured from a capture and a key -- and for a broadcast, from a capture alone.

    Nothing else is supplied and nothing else is assumed: not the width, not where a row begins,
    not the alphabet.  Each is a read, and each read is a comparison between measured candidates,
    so none of them carries a threshold.  The mode count is not read: every mode is lit, so it is
    L (L - ``BEACON_MODES`` under a beacon, which :func:`_alphabet` measures).

    ``candidates``   widths to consider.  The default runs from 2, the narrowest screen, to
                     ``isqrt(len(iq))`` capped at 256 -- the widest width the capture can
                     evidence.  Search bounds, not decisions.
    ``carrier_span`` how far the oscillators may differ, in cycles per row -- a search bound on a
                     continuous axis, like the width range.  A front end that can be further off
                     says so; nothing is thresholded against it.
    ``seed=None``    the public zero state -- an open broadcast anyone can read.

    :func:`discover` is the front door; this is the fixed-row-count read it widens.

    The carrier coarse level is the one probe set here.  :func:`_read_carrier` reads the
    oscillator difference in closed form off the reference row, but it must be told where that row
    is, and locating the row with the carrier assumed zero holds only to about 0.06 cycles per row
    -- roughly 2 ppm at 915 MHz and 2 Msps, where commodity parts are 10.  The probes cover the
    gap between the range the geometry read tolerates and the range a real oscillator occupies.
    """
    iq = _require_finite(np.asarray(iq).ravel(), "discover")
    if candidates is None:
        hi = max(2, min(256, int(math.isqrt(len(iq)))))
        candidates = range(2, hi + 1)
    # The carrier is probed in golden-ratio order, and the probing extends until the geometry it
    # returns stops changing -- the same stopping rule the row widening uses, and for the same
    # reason: agreement of two integers is exact, so no count is chosen and nothing is
    # thresholded.
    span = float(carrier_span)
    best = (None, 0, float("inf"), -float("inf"))
    best_c = 0.0
    # The best read each geometry reaches, over every carrier probe.  Keyed by (width, offset),
    # because that pair is the hypothesis: the same geometry scored at a worse carrier is the same
    # hypothesis read less well, not a competing one, and counting it as a rival collapses the
    # margin onto the carrier's sharpness -- measured 7.6 where the nearest real rival was 40.
    # Geometries are compared by significance (:func:`_cm_significance`), which is the score
    # within a width and fair across widths; each keeps its (significance, score).
    geom: dict[tuple[int, int], tuple[float, float]] = {}
    seen, prev = 0, None
    cand = list(candidates)
    while True:
        probes = _carrier_probes(span, seen + 4)[seen:]
        # Each batch's (carrier, width) reads are independent, so they run across cores; the argmax
        # bookkeeping below stays serial and in the original (carrier, width) order, so the result
        # does not depend on the worker count (see :func:`_sweep`).
        grid = [(float(c0), int(W)) for c0 in probes for W in cand]
        scored = _sweep(lambda cw: _cm_score(iq, cw[1], seed, nonce, max_rows, cw[0]), grid)
        for (c0, W), sc in zip(grid, scored):
            if sc is None:
                continue
            T = min(int(max_rows), (len(iq) - (W - 1)) // W)
            z = _cm_significance(sc, T, min(W, CM_MODES))
            o = int(np.argmax(z))
            cands = [(W, o)]
            if sc.size > 1:                     # the next offset at this width is a different
                j = int(np.argmax(np.delete(z, o)))          # geometry, so it competes too
                cands.append((W, j + (1 if j >= o else 0)))
            for w2, off2 in cands:
                if z[off2] > geom.get((w2, off2), (-float("inf"), 0.0))[0]:
                    geom[(w2, off2)] = (float(z[off2]), float(sc[off2]))
            if z[o] > best[3]:
                best = (W, o, float(sc[o]), float(z[o]))
                best_c = float(c0)
        seen += int(len(probes))
        here = (best[0], best[1])
        if span <= 0 or (prev == here and seen >= 8) or seen >= 64:
            break
        prev = here
    L, off, score, _ = best
    if L is None:
        return Discovery(None, 0, 0, 0, float("inf"), 1.0, 0.0, 0.0, 0.0, 0.0)
    # The runner-up is the most significant competing geometry; the margin is its score over the
    # winner's, the cliff a lock shows (1e14 clean) and that collapses toward 1 as the read grows
    # marginal.
    rival = max((v for g, v in geom.items() if g != (L, off)), default=None)
    runner = rival[1] if rival is not None else float("inf")

    # The fine levels, on the two continuous coordinates.  The carrier is read off the reference
    # row in closed form and then settled; the sub-sample position is settled the same way.  Each
    # is a bounded refinement of an exact read around a value that was measured.
    #
    # The closed-form read and the coarse probe are two candidates, and the aperture says which
    # is right: take whichever spreads the rows' energy more evenly over the modes.  Accepting the
    # read only when it lands within some window of the probe would introduce a threshold; this is
    # instead the same read that settles everything else.
    read_c = _read_carrier(iq, L, L, seed, nonce, off)
    carrier = min((read_c, best_c),
                  key=lambda c: _cm_at(iq, L, seed, nonce, off, max_rows, 0.0, c))
    phase, carrier = _settle(
        lambda v: _cm_at(iq, L, seed, nonce, off, max_rows, float(v[0]), float(v[1])),
        (0.0, carrier), (0.5, 0.25))
    score = _cm_at(iq, L, seed, nonce, off, max_rows, phase, carrier)

    M, closure, redundancy, row0, aligned, K, beacon = _alphabet(
        iq, L, L, seed, nonce, off, max_rows, phase, carrier)
    return Discovery(int(L), int(off), int(K), int(M), float(score),
                     float(runner / score) if score > 0 else float("inf"), closure,
                     float(phase), float(carrier), float(redundancy), int(row0),
                     aligned=bool(aligned), beacon=bool(beacon))


def _refine_continuous(iq, seed, nonce, d: Discovery,
                       carrier_span: float, join_rows: int = JOIN_SEARCH_ROWS) -> Discovery:
    """Re-settle the continuous coordinates on every row the capture has, at the geometry the
    widening already agreed on.

    The widening stops when the discrete answer stops changing, and four integers settle long
    before a real number does -- so without this step the continuous pair would be settled on
    however few rows the integers happened to need.  Measured at 12 dB: settled on the whole
    capture the receiver reads BER 0.0014, against 0.0047 settled on the integers' rows and 0.0013
    for a receiver handed the true values.

    It is cheap, because the expensive part of discovery is the sweep over candidate widths and
    this does not repeat it: the width and the row are already known, so this is one bounded
    refinement and the reads that follow it."""
    body = np.asarray(iq).ravel()
    L, off = int(d.L), int(d.offset)
    rows = max(2, _held(len(body), off, L))        # the capture, not a chosen number
    phase, carrier = _settle(
        lambda v: _cm_at(body, L, seed, nonce, off, rows, float(v[0]), float(v[1])),
        (d.phase, d.carrier), (0.5, 0.25))
    score = _cm_at(body, L, seed, nonce, off, rows, phase, carrier)
    # With no key there is no row-dependent rotation (see :func:`mode_keys`), so every
    # alignment reads alike and there is no row to hunt for.  Searching one costs 1.8 s at the
    # default reach and can only return 0.
    M, closure, redundancy, row0, aligned, K, beacon = _alphabet(
        body, L, L, seed, nonce, off, rows, phase, carrier,
        depth=1 if seed is None else int(join_rows))
    return Discovery(L, off, int(K), int(M), float(score), d.margin, closure,
                     float(phase), float(carrier), float(redundancy), int(row0),
                     aligned=bool(aligned), beacon=bool(beacon))


def _best_of(iq, seed, seen, proposed: Discovery) -> Discovery:
    """Choose between the geometries the widening proposed, by re-reading each on the same
    evidence.

    The widening reads at 8 rows, then 16, then 32, and each read proposes a geometry.  The
    scores that proposed them are NOT comparable: a shallow read is a noisier one, so a wrong
    geometry can score low at 8 rows on luck where the true one scores higher, and preferring the
    lower of the two systematically prefers the shallowest read -- the opposite of what the
    widening exists to do.  Measured, comparing them as they stand costs the 6 dB lock outright.

    So the candidates are re-read on every row the capture has, which is the move
    :func:`_settled_rows` already makes on the join axis: the reaches propose rows cheaply, and
    the rows they propose are then compared by how well each closes over the whole capture.
    Identical evidence, an exact read, a comparison between measured candidates and no cutoff.
    The score is the constant-modulus read on all L modes, so it is ~1e-15 at a true geometry
    noiseless and ~1 at any other, at any width.

    It confirms far more often than it overturns.  Measured over twenty captures at 3 dB -- where
    the widening does propose more than one geometry every time -- the comparison returned the
    same answer in all twenty, bit-identical at BER 0.21694 either way.  So this costs the
    shipped operating point nothing, and it is a guard rather than an improvement.
    Where the proposals disagree the old rule kept whichever came last: traced at ``carrier_span=8``, a
    read at 8 rows returned (64, 0) at 1.25e-15 and the widening then replaced it with (130, 80)
    at 2.21e-01, (121, 120), and (87, 16), and kept the last -- fifteen orders worse, and a wrong
    width also shortens the ceiling in :func:`discover`, so the bad answer starved the search
    that would have corrected it."""
    cands: dict[tuple, Discovery] = {}
    for d in seen:
        if d.locked:
            cands.setdefault((d.L, d.offset), d)
    if len(cands) < 2:
        return proposed
    best, best_z = proposed, -float("inf")
    for d in cands.values():
        rows = _held(len(iq), d.offset, d.L)       # every row this geometry has
        s = _cm_at(iq, int(d.L), seed, None, int(d.offset), rows, d.phase, d.carrier)
        z = float(_cm_significance(s, rows, int(d.L)))     # fair across widths
        if z > best_z:
            best, best_z = d, z
    return best


# ---------------------------------------------------------------------------
# Discovery at a continuous scale
# ---------------------------------------------------------------------------
# Search bounds, each with its provenance.  Nothing is thresholded against them.
COARSE_SPREAD = 0.025    # margin added either side of the coarse bracket (see _coarse_scale).  The
                         # two reads usually err in opposite directions and bracket the truth, but
                         # not always: on one hardware capture (0.8 -> 2.4 Msps) both read high,
                         # by 1.3% and 4.3%.  The closer read's worst measured error is 2.5%
                         # (2^H_F at 20 dB in simulation), so the bracket is widened by that.
CARRIER_STEP = 0.1       # carrier probe spacing for the scaled scan, cycles per row: the 8-row dark
                         # scan this replaced locked with the carrier 0.03-0.1 from the probe on
                         # every capture.
SCAN_POSITIONS = 4       # row starts tried per received sample in the scan (quarter samples).
SCAN_PHASES = 64         # sub-sample positions the scan's mode filters are built at.
SCAN_LOCAL_SAMPLES = 3.0  # default half-window (received samples) of the local scan on a TRACKED
                          # sender: once a sender's start phase is known from a previous burst's
                          # lock, the quarter-sample grid is restricted to this many samples either
                          # side of the prior instead of searching the whole row (:func:`_scaled_scan`
                          # ``start0``).  The grid VALUES kept are an exact subset of the full grid's,
                          # so each surviving candidate's score is bit-for-bit the full scan's; the
                          # only question is whether the full scan's winning start survived the window
                          # (``phy.receive_known`` falls back to the full scan when it did not, or
                          # when the local read is not confidently clear of its runner-up).
COARSE_ROLLOFF = 0.06    # how far a soft band edge can lift the rank-edge read above the truth: up to
                         # +5.9% on the hardware captures (0.33-0.94 ratios), never below -0.2% off
                         # the matched rate.  Used when noise has lifted the entropy read past the
                         # rank edge (see _bracket_of).

# The period read (see the section note above :func:`_period_read`).  Each is a resolution or a
# cost bound, and each was measured on tests/_ratechannel.py at L 32-256, rho 0.3-0.95, 10 dB to
# noiseless; nothing is compared against any of them.
PERIOD_ROWS = 8          # rows one fold sums -- the scan's eight, whose cliff shortlists
PERIOD_REFINE_ROWS = 32  # rows the best folds of a width are re-read on: a lock read 0.03-0.10
                         # there at 10 dB on the dark read, the best chance fold of a width 0.3-0.5
PERIOD_POSITIONS = 4     # map positions per TRANSMITTED sample: an eighth of a sample from any row
                         # start, which cost a lock 0.01-0.02 of dark share on the dark read this
                         # map was built for (16 positions: 0.002)
PERIOD_STRIDE_STEP = 0.5  # transmitted samples the fold's last row moves between neighbouring
                         # periods; the refinement reads four times finer
PERIOD_CARRIER_STEP = 0.2  # carrier probe spacing of the map, cycles per row: 0.1 from any carrier,
                         # about 3% of dark share on the dark read
PERIOD_SPACING = 0.7     # the map reads every k-th received sample, k the largest whose spacing
                         # (k * rho transmitted samples) stays under this.  Noiseless the read is
                         # unchanged at any spacing under one sample, but near one the fit is
                         # almost square and lifts the noise: at 0.9 (k=3, rho 0.3) a 10 dB L=32
                         # sender was not found at all; at 0.6 (k=2) it read 0.11 and locked
PERIOD_RANK = 16         # refined candidates, over every width, re-read on 64 rows
PERIOD_PER_MAP = 4       # distinct starts kept per map for the width's refinement
PERIOD_BANDS = 2         # maps per 1/L of scale.  A row stretched by a quarter sample end to end
                         # (one map per 1/L, read centred) still reads 0.03 noiseless, but at 10 dB
                         # the true fold read 0.13 and ranked ~300th of its width at L=256, rho 0.3;
                         # two maps per 1/L read it 0.057, ranked 1st, for 1.7x the cost


def _coarse_scale(iq):
    """The sender's rate over the receiver's as a bracket ``(lo, hi)``, from the capture alone and
    without a key.  Rows of the capture are projected onto the exponentials (a matrix product) and
    the occupied share is read twice, by two library reads that answer it differently:

        ``entroptics.entropy.geometry``   2^H_F / W, the matched scale.  Noise in the empty part of
                                          the band lifts it (+2.5% at 20 dB, +19% at 10 dB, in
                                          simulation); a receiver's roll-off lowers it (-8% at
                                          full band on hardware).
        ``entroptics.occupied_modes``     the rank edge of the ordered profile.  Unmoved by noise
                                          (+-0.2% from clean to 10 dB) but softened by real band
                                          edges (+1.3 to +5.9% on hardware).

    They err in opposite directions, so they bracket the truth: measured, it lay between them on
    every hardware capture and at every simulated SNR.  A rank edge more than a factor of two from
    the entropy read found no edge at all (a sender filling the band leaves none; measured 0.002
    against 0.92) and is not used.  None when the capture is too short for 32 rows of the
    projection."""
    got = _coarse_projection(iq)
    if got is None:
        return None
    P, W = got
    return _bracket_of(P, W)


def _coarse_projection(iq):
    """Rows of the capture on the exponentials, (T, W), and W -- the projection both coarse reads
    take.  None when the capture is too short for 32 rows of it."""
    x = np.asarray(iq).ravel()
    W = 1024
    while W >= 64 and x.size // W < 32:
        W //= 2
    if W < 64:
        return None
    T = x.size // W
    m = np.arange(W) - W // 2
    E = np.exp(-2j * np.pi * np.outer(np.arange(W), m) / W) / np.sqrt(W)
    return x[:T * W].reshape(T, W) @ E, W


def _coarse_center(iq):
    """Where the sender's band sits in the receiver's, in cycles per received sample: the centroid of
    the occupied exponentials, the occupied set being the high group of ``entroptics.level_edge`` on
    the occupancy profile.  A sender that chose an offset centre shows as a carrier offset of
    ``center * L / scale`` cycles per row -- far outside any oscillator's span, which is why it is
    measured here rather than searched.  None when the capture is too short."""
    got = _coarse_projection(iq)
    if got is None:
        return None
    P, W = got
    prof = np.mean(np.abs(P) ** 2, axis=0)
    k = int(_lib.level_edge(prof).k)
    if k < 1 or k >= W:
        return 0.0
    level = np.sort(prof)[::-1][k - 1]
    on = prof >= level
    m = (np.arange(W) - W // 2) / W
    return float(np.sum(m[on] * prof[on]) / np.sum(prof[on]))


def _bracket_of(P, W):
    h = float(2.0 ** _lib.entropy.geometry(P)["H_F"] / W)
    k = float(_lib.occupied_modes(np.mean(np.abs(P) ** 2, axis=0)).k / W)
    edge = 0.5 * h <= k <= 2.0 * h
    if edge and h > k:
        # Noise lifts the entropy read and cannot move the rank edge; a soft band edge lifts the
        # rank edge and lowers the entropy read.  So an entropy read above the rank edge is noise,
        # and the truth sits at the rank edge or up to COARSE_ROLLOFF below it.  Measured in
        # simulation at 10 dB: h reads +5% to +34% high and k within +-1.2%, so bracketing on h
        # searched up to a third more scale than the truth needed.
        return (k * (1 - COARSE_SPREAD) / (1 + COARSE_ROLLOFF) - 1.0 / W,
                k * (1 + COARSE_SPREAD) + 1.0 / W)
    reads = [h] + ([k] if edge else [])
    # One bin of the projection is the read's own resolution, on top of the measured spread.
    return min(reads) * (1 - COARSE_SPREAD) - 1.0 / W, max(reads) * (1 + COARSE_SPREAD) + 1.0 / W


def _scaled_scan(iq, L: int, seed, nonce, scale: float, rows: int, carriers, modes=None,
                 keep: int = 1, start0=None, start_win: float = None, _debug: bool = False,
                 _c64_debug: bool = False):
    """The constant-modulus read at every row start on a quarter-sample grid over one row,
    ``rows`` rows each, for every carrier in ``carriers``.

    ``start0`` (received samples, a phase in ``[0, L/scale)``) turns this into a LOCAL scan for a
    tracked sender: the quarter-sample grid is restricted to ``start_win`` samples either side of
    ``start0`` (default ``SCAN_LOCAL_SAMPLES``) rather than the whole row.  The kept grid positions
    are an exact subset of the full grid's, so every surviving candidate's score is bit-for-bit what
    the full scan computed there -- the local read is the full read with the far-away starts dropped.
    ``None`` (the default) scans the whole row, unchanged.  First-contact blind acquisition always
    passes ``None``; only a sender whose start phase a previous burst already located passes a prior.

    Only the first ``modes`` coordinates are formed (``CM_MODES`` by default; more modes read a
    dense coherent QAM row more firmly -- it reads ~0.5 of the null at the true start, and on 16
    modes scatters about that nearly as widely as a Gaussian row does): for a row whose first sample lies delta
    into it that is one product with a bank of filters fixed by delta, built at ``SCAN_PHASES``
    sub-sample positions.  The filters depend on the scale alone -- a carrier is a ramp across
    them -- so they are built once per scale and every carrier is read against them.  Building them
    per carrier was most of the scan's cost (measured at L=128: 33 of 87 ms per call, repeated for
    each of seven carriers).

    Returns ``[(score, start, carrier), ...]``, the best start of each carrier that has a readable
    one -- or its ``keep`` best: the lowest local minima of the read over the starts, so a caller
    that ranks them on more evidence gets distinct candidates rather than one minimum's slope.  A
    scan on a few modes is a shortlist: a dense QAM row reads ~0.5 of the null on 16 modes with a
    spread near a Gaussian row's, so over a capture's ~3000 starts and carriers the true one can
    rank a few places down (measured on 256-QAM bursts of 12-14 rows in 16-row scans)."""
    x = np.asarray(iq).ravel()
    L, rho = int(L), float(scale)
    ell = L / rho
    cs = np.atleast_1d(np.asarray(carriers, dtype=float))
    ms = _signed(L)
    m = min(L, CM_MODES if modes is None else int(modes))
    cm = _scan_coeffs(L, seed, nonce, m)                                   # (m, L)
    dq = np.arange(SCAN_PHASES) / SCAN_PHASES
    st = np.arange(0.0, ell, 1.0 / SCAN_POSITIONS)
    if start0 is not None:
        # Local scan on a tracked sender: keep only the full grid's positions within start_win of
        # the prior phase.  Subsetting the SAME array keeps the surviving positions (and so their
        # reads) bit-identical to the full scan's; the window wraps the row so a prior near a row
        # edge still keeps both sides of it (the phase is periodic over the row).
        w = float(SCAN_LOCAL_SAMPLES if start_win is None else start_win)
        d = np.abs((st - float(start0) + 0.5 * ell) % ell - 0.5 * ell)
        st = st[d <= w + 1e-9]
    S = (st[:, None] + np.arange(int(rows))[None, :] * ell).ravel()
    first = np.ceil(S - 1e-12).astype(np.int64)
    delta = first - S
    k = np.rint(delta * SCAN_PHASES).astype(np.int64)
    first, k = first - (k == SCAN_PHASES), k % SCAN_PHASES   # delta ~1: the sample before
    count = np.ceil(S + ell - 1e-12).astype(np.int64) - first
    num = np.full((cs.size, S.size), np.nan)
    for n in np.unique(count):
        sel = np.flatnonzero((count == n) & (first + n <= x.size))
        if n < L or sel.size == 0:
            continue
        j = np.arange(n)
        E0, G0inv, E0c = _scaled_gram(L, rho, int(n))
        ks = k[sel]
        # Only the sub-sample phases that actually occur in this count group carry a product, so
        # the filter bank is built for those alone -- typically a quarter of SCAN_PHASES on a scan
        # grid, and the unused ones were pure cost.  The filters for a used phase are the identical
        # entries the full-bank build produced (``dq[used]`` are the same sub-sample positions), so
        # the read is bit-for-bit what building all SCAN_PHASES gave.
        used = np.unique(ks)
        dqu = dq[used]
        Q = cm[None, :, :] * np.exp(-2j * np.pi * np.outer(dqu, ms) * rho / L)[:, None, :]
        H = (Q @ G0inv) @ E0c.T                                            # (phase, mode, sample)
        ramp = np.exp(-2j * np.pi * cs[:, None, None]
                      * (j[None, None, :] + dqu[None, :, None]) * rho / L)
        Hc = H[None, :, :, :] * ramp[:, :, None, :]       # (carrier, phase, mode, sample)
        X = x[first[sel, None] + j[None, :]]
        order = np.argsort(ks, kind="stable")
        lo = np.searchsorted(ks[order], used, side="left")
        hi = np.searchsorted(ks[order], used, side="right")
        for p in range(used.size):                  # one product per occupied sub-sample phase
            ii = order[lo[p]:hi[p]]
            Hp = Hc[:, p].reshape(cs.size * m, n).T
            if _c64_debug:          # research-only: the complex64 (cgemm) read, to measure its
                Xp = X[ii].astype(np.complex64)      # num[] perturbation against the local gap
                Z = (Xp @ Hp.astype(np.complex64)).astype(np.complex128)
            else:
                Z = X[ii] @ Hp                                              # (rows, carrier x mode)
            e = (Z.real ** 2 + Z.imag ** 2).reshape(ii.size, cs.size, m)
            num[:, sel[ii]] = _cm_rows(e).T
    num = num.reshape(cs.size, st.size, -1)
    v = np.mean(num, axis=2)
    v = np.where(np.isnan(v), np.inf, v)
    found = []
    for ci in range(cs.size):
        if not np.isfinite(v[ci]).any():
            continue
        if int(keep) <= 1:
            i = int(np.argmin(v[ci]))
            found.append((float(v[ci, i]), float(st[i]), float(cs[ci])))
            continue
        w = v[ci]
        lo = np.flatnonzero((w <= np.r_[np.inf, w[:-1]]) & (w <= np.r_[w[1:], np.inf]) & np.isfinite(w))
        for i in lo[np.argsort(w[lo], kind="stable")][:int(keep)]:
            found.append((float(w[i]), float(st[i]), float(cs[ci])))
    if _debug:                       # margin analysis (research/realtime); not on any production path
        return found, st, v, cs
    return found


def _settle_scaled(iq, L: int, seed, nonce, start: float, scale: float, carrier: float,
                   rows: int, scale_span_ppm: float, carrier_span: float = 0.1,
                   joint: bool = False):
    """Settle (start, scale, carrier) on ``rows`` rows inside a bracket the caller measured.  The
    scale moves in parts per million so one tolerance serves all three; kept only if it improves on
    where it started (:func:`_settle`'s rule).  ``joint`` settles on the joint read's coordinates."""
    s0, r0, c0 = float(start), float(scale), float(carrier)
    f = lambda z: _scaled_cm(iq, L, seed, nonce, z[0], r0 * (1 + z[1] * 1e-6), z[2], rows,
                             joint)
    z0 = (s0, 0.0, c0)
    # Not past equal rates on the row read: a row a hair short of L samples still reads when its
    # samples happen to fall L to a row, and a settle on a hardware capture at rho = 1 wandered to
    # +59 ppm that way, into a fit of a sender wider than the receiver (BER 0.46).
    #
    # The joint settle is the exception.  It reads every row the capture holds, where a few ppm of
    # scale is a visible drift of the rows against the sender's (0.75 ppm over 1800 rows is 0.09 of a
    # sample), and two radios at one nominal rate differ by their clocks, so half of all senders are
    # a hair wider than the receiver.  Held at 1 the read bows: measured on a simulated bench capture
    # (2.4 -> 2.4 Msps, sender +0.75 ppm, 1810 rows, band read), the DQPSK closure per 200 rows fell
    # 0.945 -> 0.906 toward both ends, where at the sender's scale it read 0.945 flat; settled free,
    # both matched simulated pairs read the scale at +0.75 / +0.76 ppm.
    if joint or r0 > 1.0:
        top = float(scale_span_ppm)
    else:
        top = min(float(scale_span_ppm), (1.0 / r0 - 1.0) * 1e6)
    res = _minimize(f, list(z0), method="Powell",
                    bounds=[(s0 - 0.5, s0 + 0.5), (-scale_span_ppm, max(top, 0.0)),
                            (c0 - carrier_span, c0 + carrier_span)],
                    options={"xtol": 1e-4, "ftol": 1e-9})
    z = res.x if f(res.x) <= f(z0) else z0
    return float(z[0]), r0 * (1 + float(z[1]) * 1e-6), float(z[2])


# ---------------------------------------------------------------------------
# The row period, read per width
# ---------------------------------------------------------------------------
# What a receiver at another rate sees of a sender's rows is their period in its own samples,
# L / rho.  Without the key the air does not mark that period -- the screen repeats nothing, every
# mode is lit, and each row goes out at its own keyed power -- and the keyed aperture is an
# independent draw per width, so the constant-modulus read tests a period only together with the
# width that keyed it.  Measured: at the true period the widths either side read like an isotropic
# capture.  So the period is read per width, and read directly rather than swept as a grid of
# scales (research/blind_width_scaled.md):
#
#   * The map.  For one width and one band of scales, the first ``CM_MODES`` coordinates of a
#     single row are formed at every start position over eight rows of the capture: one sliding
#     product per sub-sample phase, every phase, carrier probe and mode in one matrix product, and
#     each row's constant-modulus read taken there.  Inside a row the scale only stretches it, and
#     a row stretched by a fraction of a sample end to end still reads; read centred on the map's
#     row it is stretched half as far, so one map serves a band of scales 1/(2L) wide
#     (PERIOD_BANDS).
#   * The fold.  Every period in that band is then a mean over the map at s0, s0 + P, ... -- eight
#     look-ups per hypothesis where the scale grid computed eight reads.  Every period the band
#     allows, every start and every carrier probe is folded.
#   * The refinement.  The best folds of the width are re-read on 32 rows from the map's own
#     filters, over a period grid four times finer.  That separates a lock (even rows) from the best
#     of the millions of chance folds (Gaussian ones).
#
# A row's filter is the scaled read's least squares (:func:`_scaled_coords`), on every k-th received
# sample when k of them still over-sample the sender (PERIOD_SPACING).  Its Gram is a Dirichlet
# kernel of the index difference, so it is written in closed form.  Measured against the scale grid
# it replaces, with the width given, on tests/_ratechannel.py: 0.3-0.7 s against 10-280 s at L=64,
# 3-15 s against an estimated 4 min to 5 h at L=256, the same lock in every cell (on the dark read
# this map was built for; the constant-modulus map's costs are in research/rank_leak.md).

class _PeriodMap:
    """One width's map at one scale: ``num[c, i]`` the constant-modulus read of a row starting at
    ``i / Q`` received samples on carrier probe ``c``, ``den[i]`` 1 where the map holds a row there
    (NaN where it does not), and the per-phase filters that formed them (``filt[q] = (taps,
    (carriers, taps, modes))``)."""
    __slots__ = ("L", "rho", "Q", "k", "carriers", "num", "den", "filt")


def _dirichlet_gram(L: int, spacing: float, n: int) -> np.ndarray:
    """``E^H E`` for ``n`` samples ``spacing`` transmitted samples apart on the L signed
    exponentials, in closed form: entry (m, m') is the geometric sum of e^{2.pi.i.spacing.(m'-m)/L}.
    A spacing under one sample keeps every ratio off 1 for m' != m."""
    ms = _signed(L)
    d = ms[None, :] - ms[:, None]
    a = 2.0 * np.pi * float(spacing) * d / int(L)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = (1.0 - np.exp(1j * a * int(n))) / (1.0 - np.exp(1j * a))
    G[d == 0] = int(n)
    return G


def _period_map(x, L: int, cm, rho: float, carriers, positions: int) -> _PeriodMap:
    """The constant-modulus map of one width at scale ``rho`` over ``positions`` map positions, on
    the coefficient rows ``cm`` (m, L) (see the section note)."""
    L, rho = int(L), float(rho)
    m = int(cm.shape[0])
    Q = int(np.ceil(PERIOD_POSITIONS * rho))
    k = max(1, int(np.floor(PERIOD_SPACING / rho)))
    ms = _signed(L)
    cs = np.asarray(carriers, dtype=float)
    dq = np.arange(Q) / Q                                   # where the row starts inside a sample
    nd = (np.ceil(L / rho - dq - 1e-12).astype(np.int64) - 1) // k + 1   # samples read per phase
    C = cm[:, :, None] * np.exp(-2j * np.pi * np.outer(ms, dq) * rho / L)[None]   # (m, L, Q)
    filt = [None] * Q
    for n in np.unique(nd):
        qs = np.flatnonzero(nd == n)
        j = np.arange(int(n)) * k
        # the modes' coefficients of the least-squares read, as filters on the samples: E^* G^-T c
        rhs = C[:, :, qs].transpose(1, 0, 2).reshape(L, m * qs.size)
        h = (np.exp(-2j * np.pi * np.outer(j * rho, ms) / L) @ np.linalg.solve(
            np.conj(_dirichlet_gram(L, k * rho, int(n))), rhs)).reshape(j.size, m, qs.size)
        for col, q in enumerate(qs):
            ramp = np.exp(-2j * np.pi * cs[:, None] * (j[None, :] + dq[q]) * rho / L)
            filt[q] = (j, h[None, :, :, col] * ramp[:, :, None])         # (carrier, tap, mode)
    taps = int(nd.max())
    span = (taps - 1) * k + 1
    F = max(min(int(positions) // Q + 2, x.size - span + 1), 0)
    width = cs.size * m
    H = np.zeros((taps, Q * width), dtype=np.complex64)
    for q in range(Q):
        H[:filt[q][0].size, q * width:(q + 1) * width] = \
            filt[q][1].transpose(1, 0, 2).reshape(filt[q][0].size, width)
    V = np.lib.stride_tricks.sliding_window_view(x, span)[:F, ::k]
    D = np.empty((F, Q, cs.size), dtype=np.float32)         # (start sample, phase, carrier)
    blk = max(1, (1 << 22) // max(Q * width, 1))
    for a in range(0, F, blk):                              # a block of starts at a time: the
        Z = np.ascontiguousarray(V[a:a + blk]).astype(np.complex64) @ H     # product is the cost
        e = (Z.real ** 2 + Z.imag ** 2).reshape(-1, Q, cs.size, m)
        D[a:a + blk] = _cm_rows(e)
    pm = _PeriodMap()
    pm.L, pm.rho, pm.Q, pm.k, pm.carriers, pm.filt = L, rho, Q, k, cs, filt
    pm.num = np.full((cs.size, F * Q), np.nan, dtype=np.float32)
    pm.den = np.full(F * Q, np.nan, dtype=np.float32)
    f = np.arange(F)
    for q in range(Q):
        i = f * Q - q                                       # start f - q/Q, on the map's grid
        sel = i >= 0
        pm.num[:, i[sel]] = D[sel, q, :].T
        pm.den[i[sel]] = 1.0
    return pm


def _period_rows(x, pm: _PeriodMap, fine, ci: int) -> np.ndarray:
    """The constant-modulus read of hypotheses whose rows start at map positions ``fine`` (h,
    rows), on carrier probe ``ci``, from the map's filters: one per hypothesis, ``inf`` where a row
    leaves the capture."""
    fine = np.asarray(fine, dtype=np.int64)
    f = -((-fine) // pm.Q)
    q = f * pm.Q - fine
    num = np.zeros(fine.shape)
    bad = np.zeros(fine.shape, dtype=bool)
    for qq in np.unique(q):
        j, H = pm.filt[int(qq)]
        sel = q == qq
        ff = f[sel]
        bad[sel] = (ff < 0) | (ff + j[-1] >= x.size)
        W = x[np.clip(ff, 0, x.size - j[-1] - 1)[:, None] + j[None, :]]
        Z = W @ H[ci]                                       # (rows, modes)
        num[sel] = _cm_rows(Z.real ** 2 + Z.imag ** 2)
    v = num.mean(axis=1)
    v[bad.any(axis=1)] = np.inf
    return v


def _period_read(x, L: int, seed, nonce, lo: float, hi: float, centre: float, offsets):
    """The row period of width ``L`` read directly, for every scale in ``[lo, hi]`` (see the section
    note).  Returns the width's best refined hypotheses as ``(score, L, start, scale, carrier,
    step)``: ``score`` the 32-row constant-modulus read, ``step`` the scale resolution the fold
    had."""
    L = int(L)
    R, R2 = PERIOD_ROWS, PERIOD_REFINE_ROWS
    # How many of the width's folds are refined: a cost bound that grows with the hypotheses the
    # width folds (~L^2).  The true fold ranked 1st-2nd of its width in every measured cell; before
    # the map spacing was halved it ranked as low as ~300th at L=256, 10 dB.
    keep = max(8, L // 4)
    cm = np.conj(_coeffs(L, seed, nonce)[:min(L, CM_MODES)]) * np.sqrt(L)
    bands = max(1, int(np.ceil(np.log(hi / lo) * L * PERIOD_BANDS)))
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), bands + 1))
    pool = []
    # The band that reaches equal rates is also read on a map at 1 exactly.  At rho = 1 a row has
    # no spare samples to absorb a stretch, and which map read the true fold better changed from
    # capture to capture (measured at 10-20 dB: L=32 locked only on the map at 1, L=256 only on
    # the map at the band's centre); reading both costs one map per width.
    maps = [(a, b, float(np.sqrt(a * b))) for a, b in zip(edges[:-1], edges[1:])]
    if hi >= 1.0:
        maps.append((edges[-2], edges[-1], 1.0))
    for a, b, rc in maps:
        rc = min(rc, 1.0)
        ellc = L / rc
        ds = PERIOD_STRIDE_STEP / (rc * R)                  # period step, received samples
        st = np.arange(L / b, L / a + ds, ds)              # every period the band allows
        smax = st.max()
        if R * smax + ellc + 2 > x.size:
            continue
        Q = int(np.ceil(PERIOD_POSITIONS * rc))
        # A band of width rc fits the receiver's only within (1 - rc) / 2 of its centre.  A band
        # that fills the receiver's has no edge, and the centroid of its "occupied" bins is noise:
        # measured -0.038 cycles/sample at rho 1, which is -1.2 cycles/row at L=32 and put every
        # carrier probe off the sender.  The measured centre is held to the room the band leaves.
        f0 = float(np.clip(centre, -(1.0 - rc) / 2, (1.0 - rc) / 2))
        # The centre read is good to about 0.01 cycles/sample, which is a cycle per row at L/rho
        # ~100: on a hardware capture at 0.9375 it read 0.0145 (0.99 cycles/row) for a sender on
        # the receiver's centre, and every probe missed.  So an unshifted band is probed as well
        # (measured at 10 dB, rho 0.95: a centre read 0.40 cycles/row off lost the sender too).
        probes = f0 * L / rc + np.asarray(offsets)
        if abs(f0 * L / rc) > PERIOD_CARRIER_STEP / 4:
            probes = np.concatenate([probes, np.asarray(offsets)])
        pm = _period_map(x, L, cm, rc, probes,
                         int(np.ceil((R * smax + ellc) * Q)) + 2 * Q + 4)
        n = pm.den.size
        shift = (st - ellc) / 2 * Q                         # centre each period's row on the map's
        M = int(np.ceil(smax * Q))                          # starts over one period
        p0 = int(np.ceil(max(0.0, -float(shift.min())))) + 1
        o = p0 + np.rint(np.arange(R)[None, :] * st[:, None] * Q + shift[:, None]).astype(np.int64)
        good = o[:, -1] + M <= n
        if not good.any() or M > n:
            continue
        st, shift, o = st[good], shift[good], o[good]
        Wd = np.lib.stride_tricks.sliding_window_view(pm.den, M)
        Wn = [np.lib.stride_tricks.sliding_window_view(pm.num[c], M) for c in range(pm.num.shape[0])]
        B = Wd[o[:, 0]].copy()
        A = np.stack([w[o[:, 0]] for w in Wn])
        for t in range(1, R):
            B += Wd[o[:, t]]
            for c, w in enumerate(Wn):
                A[c] += w[o[:, t]]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = A / B[None]
        v = np.where(np.arange(M)[None, None, :] < np.ceil(st * Q)[None, :, None], v, np.inf)
        v = np.where(np.isnan(v), np.inf, v)
        # The best fold at each start, and the best few starts at least a transmitted sample
        # apart: neighbouring periods and starts of one chance fold are one hypothesis, and
        # keeping them all crowded the true fold out of the refinement (measured at L=256,
        # 10 dB: the width's top 64 were copies of a handful of chance folds).
        flat = v.transpose(2, 0, 1).reshape(M, -1)
        arg = np.argmin(flat, axis=1)
        vi = flat[np.arange(M), arg]
        taken = []
        for i0 in np.argsort(vi, kind="stable"):
            if not np.isfinite(vi[i0]) or len(taken) == PERIOD_PER_MAP:
                break
            if all(min(abs(int(i0) - t), M - abs(int(i0) - t)) >= Q for t in taken):
                taken.append(int(i0))
                ci, si = divmod(int(arg[i0]), v.shape[1])
                pool.append((float(vi[i0]), float(st[si]), int(i0) + p0, int(ci), pm, ds,
                             float(shift[si])))
    pool.sort(key=lambda z: z[0])
    found = []
    dss = np.linspace(-1.0, 1.0, 9)
    di = np.array([-1, 0, 1])
    for _, s, i0, ci, pm, ds, shift in pool[:keep]:
        # within the bracket: a period refined past L / hi is a sender wider than the bracket, at
        # rho = 1 wider than the receiver, which the exact read refuses (fewer samples than modes)
        S = np.maximum(s + ds * dss, L / hi)
        fine = ((i0 + di)[None, :, None]
                + np.rint(np.arange(R2)[None, None, :] * S[:, None, None] * pm.Q + shift).astype(np.int64))
        v = _period_rows(x, pm, fine.reshape(-1, R2), ci)
        best = int(np.argmin(v))
        si, ii = divmod(best, di.size)
        rho = L / S[si]
        found.append((float(v[best]), L, float((i0 + di[ii]) / pm.Q), float(rho),
                      float(pm.carriers[ci]), float(rho * ds / S[si])))
    return found


def _equal_rates(x, L: int, seed, nonce, offsets):
    """Equal rates read as the hypothesis it is: every start on the scan's quarter-sample grid and
    every carrier probe at scale exactly 1, each on ``PERIOD_REFINE_ROWS`` rows -- the depth the
    period read refines its folds at, so the two kinds of candidate are scored alike.  Returns the
    best ``PERIOD_PER_MAP`` starts in :func:`_period_read`'s form.

    The period read shortlists on an eight-row fold, and on a dense QAM capture that fold is chance:
    a 16-QAM row reads ~0.5 of the null on 16 modes noiseless and scatters nearly as widely as a
    Gaussian row, so over the band's thousands of folds the true one can rank far down.  Measured on
    a simulated bench window (16-QAM, 2.4 -> 2.4 Msps, 22 dB, the receiver's samples ~0.4 of a
    sample off the sender's): the true fold read 0.68 and ranked 35th of 56 pooled (the best chance
    fold 0.585), so it never reached the refinement, where on 32 rows it reads 0.62 against a best
    chance 0.78.  At scale 1 the period is fixed, so the starts and probes are few -- 4L x the probes
    -- and every one of them can be read at the refinement's depth (0.1-0.4 s at L = 64).  The scale
    the settles start from is 1; ``step`` is the 1-map's band, 1/(PERIOD_BANDS L), as a fold there
    would carry."""
    found = _scaled_scan(x, int(L), seed, nonce, 1.0, PERIOD_REFINE_ROWS, np.asarray(offsets),
                         keep=PERIOD_PER_MAP)
    step = 1.0 - math.exp(-1.0 / (PERIOD_BANDS * int(L)))
    return [(float(v), int(L), float(s), 1.0, float(c), step) for v, s, c in found if np.isfinite(v)]


def _discover_scaled(iq, seed, candidates, nonce=None, carrier_span: float = 1.0,
                     join_rows: int = JOIN_SEARCH_ROWS) -> Discovery:
    """Discovery when the sender's rate is not the receiver's: width, scale, start and carrier.

    ``candidates`` is the width range to search; ``None`` searches every width the capture can
    evidence, from 2 to ``isqrt(samples x scale)`` capped at 256 -- the integer read's range, in the
    sender's samples.  The width is never told: it is a search bound.

    The coarse scale comes from the capture's own occupancy (:func:`_coarse_scale`).  Then each
    candidate width's row period is read directly (:func:`_period_read`): one map per band of
    scales 1/L wide, folded at every period, start and carrier probe the band allows, the width's
    best folds re-read on 32 rows.  Those reads rank every width against every other; the best
    ``PERIOD_RANK`` are re-read on 64 rows before neighbours are merged, since merging first keeps
    whichever scored best on fewer rows, and at a shallow lock that is chance (measured: the true
    scale was merged into a neighbour that settles 7 dB worse).  The best four are settled inside
    the resolution the fold had, compared on 256 rows, and the winner settled on every row the
    capture has -- a scale settled on 128 rows leaves ppm-level error that grows along the capture
    (measured BER 0.005 -> 0.33 across 60k samples).

    Not the front door.  The width is a separate key per value, so a blind search pays for every
    width it tries, and the widest dominate: measured (research/blind_width_scaled.md) the whole
    range costs minutes where one width costs about a second before the settles.  How that is paid
    -- once per sender and kept, or bounded by the deployment -- is the caller's choice, so
    :func:`discover` stays the integer read, and this is reached explicitly."""
    body = _require_finite(np.asarray(iq).ravel(), "discover")
    bracket = _coarse_scale(body)
    if bracket is None:
        return Discovery(None, 0, 0, 0, float("inf"), 1.0, 0.0)
    lo, hi = bracket
    # A receiver's own roll-off reads a full-band sender up to 8% low (measured), so a bracket
    # within that of 1 reaches 1: equal rates are a hypothesis in their own right.
    if hi * 1.08 >= 1.0:
        hi = 1.0
    f0 = _coarse_center(body) or 0.0          # the band's measured centre, cycles per received sample
    if candidates is None:
        candidates = range(2, max(2, min(256, math.isqrt(int(body.size * hi)))) + 1)
    offsets = np.arange(-float(carrier_span), float(carrier_span) + 1e-9, PERIOD_CARRIER_STEP)
    cands = []
    for L in candidates:
        L = int(L)
        if L < 2:
            continue
        cands += _period_read(body, L, seed, nonce, max(lo, 1e-6), hi, f0, offsets)
        if hi >= 1.0:
            cands += _equal_rates(body, L, seed, nonce, offsets)
    if not cands:
        return Discovery(None, 0, 0, 0, float("inf"), 1.0, 0.0)
    # Compared by significance (:func:`_cm_significance`): within a width it orders as the score,
    # across widths it is fair to the narrow ones' wider chance spread.
    cands.sort(key=lambda q: -_cm_significance(q[0], PERIOD_REFINE_ROWS, min(q[1], CM_MODES)))
    ranked = sorted(cands[:PERIOD_RANK],
                    key=lambda q: -_cm_significance(
                        _scaled_cm(body, q[1], seed, nonce, q[2], q[3], q[4], 64), 64, q[1]))
    lead = []
    for q in ranked:
        if all(q[1] != p[1] or abs(q[3] - p[3]) > 2 * q[5] or abs(q[2] - p[2]) > 1.0
               for p in lead):
            lead.append(q)
        if len(lead) == 4:
            break
    settled = []
    for _, L, s0, rho, c, step in lead:
        z = _settle_scaled(body, L, seed, nonce, s0, rho, c, 64, step / rho * 1e6)
        z = _settle_scaled(body, L, seed, nonce, *z, 256, 0.5 / (256 * L) * 1e6)
        settled.append((L,) + z)
    L, s0, rho, c = max(settled, key=lambda z: _cm_significance(
        _scaled_cm(body, z[0], seed, nonce, z[1], z[2], z[3], 256),
        min(256, _scaled_held(body.size, z[1], z[0], z[2])), z[0]))
    T = int((body.size - s0 - 2) // (L / rho)) - 1
    # The last settle is on every row, read jointly (:func:`_joint_coords`).  On the row read the
    # coordinates carry the neighbouring rows' floor, about -23 dB, and the settle stops where
    # that floor hides the geometry: measured at L=128, noiseless (on the dark read this replaced),
    # a joint read at that geometry reads -35 to -54 dB, and at the jointly settled one -66 to
    # -70 dB, the scale then within 3e-9.  The score and the runner-up are read the same way, so
    # the margin compares like with like.
    s0, rho, c = _settle_scaled(body, L, seed, nonce, s0, rho, c, T, 2.0 / (T * L) * 1e6, 0.05,
                                joint=True)
    score = _scaled_cm(body, L, seed, nonce, s0, rho, c, T, joint=True)
    runner = min((_scaled_cm(body, z[0], seed, nonce, z[1], z[2], z[3], 256, joint=True)
                  for z in settled if (z[0], round(z[1])) != (L, round(s0))),
                 default=float("inf"))

    off = int(np.ceil(s0))
    phase = off - s0                          # the integer read's convention: rows start at offset - phase
    # A scale read continuously stays on the continuous read.  Every reader switches to the
    # integer screen at a scale of exactly 1, and that screen's aperture is the band-limited
    # continuation only on whole samples: a capture 0.43 of a sample off the sender, read there at
    # the phase this read reports, came out at a payload error of +11 dB (64-QAM, noiseless; -26.5
    # dB on the continuous read).  The next float above 1 reads identically to 1 on the continuous
    # read.
    if rho == 1.0:
        rho = float(np.nextafter(1.0, 2.0))
    M, closure, redundancy, row0, aligned, K, beacon = _alphabet(
        body, L, L, seed, nonce, off, T, phase, c, depth=1 if seed is None else int(join_rows),
        scale=rho)
    return Discovery(int(L), off, int(K), int(M), float(score),
                     float(runner / score) if score > 0 else float("inf"), closure,
                     float(phase), float(c), float(redundancy), int(row0),
                     aligned=bool(aligned), beacon=bool(beacon), scale=float(rho))


# The adaptive carrier reads at carrier zero first and widens to the requested span only when
# that cheap read is not a clear lock.  The trust test is the winner's margin over its nearest
# rival: a true lock beats every rival by a cliff (1e4 clean, ~1.8 at 6 dB), while a chance
# geometry barely wins -- measured, a 3 dB capture locked width 14 at margin 1.03 where the truth
# was width 64.  1.2 sits above that noise with buffer and below every true lock in the measured
# sweep; a false negative only widens the carrier search (the correct, slower path), never a wrong
# answer, so the bar is set for robustness rather than to maximize the fast path.
CARRIER_SKIP_MARGIN = 1.2


def _discover_span(iq, seed, candidates, carrier_span: float, join_rows: int) -> Discovery:
    """The row-widening read at one carrier span: start at ``MIN_ROWS`` and double until the
    geometry agrees with itself, then settle.  :func:`discover` wraps this with the cheap-first
    carrier escalation; this is the body that each span runs."""
    body = np.asarray(iq).ravel()
    rows = int(MIN_ROWS)
    best = _discover_at(iq, seed, candidates, None, rows, carrier_span)
    seen = [best]
    prev = (best.L, best.offset)
    while True:
        rows *= 2
        if rows * 2 > len(body) // max(best.L or 2, 2):     # the capture, not a chosen ceiling
            best = _best_of(iq, seed, seen, best)
            return _refine_continuous(iq, seed, None, best, carrier_span,
                                     join_rows) if best.locked else best
        d = _discover_at(iq, seed, candidates, None, rows, carrier_span)
        here = (d.L, d.offset)
        seen.append(d)
        if here == prev:                # said the same thing on twice the evidence: converged
            return _refine_continuous(iq, seed, None, _best_of(iq, seed, seen, d),
                                      carrier_span, join_rows) if d.locked else d
        best, prev = d, here


def discover(iq, seed=None, candidates=None, *,
             carrier_span: float = 1.0,
             join_rows: int = JOIN_SEARCH_ROWS) -> Discovery:
    """Read the transmission's own parameters, using as much of the capture as it takes.

    The read is the same at every width; what changes is how many rows it is given, and the rows
    are the accuracy.  Measured over five captures at 6 dB: sixteen rows locked three times out of
    five at BER 0.274, sixty-four locked five out of five at 0.106, against 0.096 for a receiver
    handed the true geometry.  So there is no row count to pass -- the read starts at the fewest
    any of its own reads can work with and widens until it agrees with itself.

    The stopping rule is not a threshold.  It does not ask whether the margin is good enough,
    which would be a number someone had to choose; it asks whether the answer stopped changing.
    The width, the row, the mode count and the alphabet are integers, so agreement between two
    successive doublings is exact equality rather than a band.  A read that says the same thing
    twice on twice the evidence has converged.

    What it converges ON is a separate question, answered by :func:`_best_of`: a doubling that
    returns a different geometry has not converged, but it has not necessarily improved either,
    so the candidates the widening proposed are re-read on every row the capture has and
    compared there.  Keeping whichever came last is what this did until it was measured doing
    it, and it discarded a lock at 1.25e-15 for an answer at 0.859 that merely arrived later.

    Doubling from ``MIN_ROWS`` costs less than it looks: a clean capture agrees immediately and
    pays about one extra read, and only a capture that is genuinely struggling pays for the deep
    ones.  The ceiling is the capture itself -- rows it does not contain -- which is a fact about
    the data rather than a bound anyone set.

    ``_discover_at`` is the fixed-row-count read underneath, kept for pinning that sensitivity in
    tests.  This is the front door.

    The carrier is read cheapest-first (see :data:`CARRIER_SKIP_MARGIN`): one probe at carrier
    zero, widening to ``carrier_span`` only when that read is not a clear lock.  A capture whose
    oscillator sits within the geometry read's tolerance is found three to four times faster, and
    one that is not falls back to the full span, so the answer is unchanged.
    """
    if carrier_span > 0.0:
        d0 = _discover_span(iq, seed, candidates, 0.0, join_rows)
        if d0.locked and d0.margin > CARRIER_SKIP_MARGIN:
            return d0
    return _discover_span(iq, seed, candidates, carrier_span, join_rows)


def coherent_closure(iq, L: int, K: int, seed, *, offset: int = 0,
                     phase: float = 0.0, carrier: float = 0.0) -> float:
    """How well the capture closes as a coherent payload, the companion to the differential
    closure of :func:`_alphabet`.

    That read answers "is an M-PSK alphabet riding across rows", and is near zero on a coherent
    capture.  This one asks the same question of the other payload: divide out the keyed reference
    row, take each mode's gain from it, and see whether what remains lands on a square-QAM
    lattice.  The two reads separate cleanly, and the larger closure names the payload without a
    threshold -- the way the carrier is chosen by whichever reads better on the aperture:

        capture             differential    coherent
        PSK,  right key           1.0000      0.0090
        QAM,  right key           0.0201      0.3598
        PSK,  wrong key           0.0098      0.0023
        QAM,  wrong key           0.0617      0.0016

    With the wrong key nothing closes on either read.

    ``carrier`` is the discovery's read.  The reference row is located (:func:`_reference_at`) and
    the row-rate carrier read off the payload (:func:`_row_carrier`) before the closure is taken,
    so a carrier offset does not smear the fourth power across rows."""
    Y, _, _ = _coherent_read(iq, int(L), int(K), seed, offset, phase, carrier)
    return _fourth_closure(Y)


def _fourth_closure(Y) -> float:
    """|mean u^4| of the unit phasors of de-keyed symbols: ~0.36 on 16-QAM, ~0 on a scatter."""
    if not np.asarray(Y).size:
        return 0.0
    u = Y / np.maximum(np.abs(Y), 1e-30)
    return float(abs(np.mean(u ** 4)))


def closure_significance(d: "Discovery", n_samples: int) -> float:
    """How far the alphabet closure stands above what an unkeyed capture of this size would
    give.

    The closure identifies a transmission where the geometry read alone may not (see
    :func:`discover`),
    while not being comparable across captures on its own: it degrades with SNR -- measured on
    one transmission, 1.000 clean, 0.888 at 18 dB, 0.611 at 12, 0.143 at 6 and 0.030 at 3.  A
    fixed level set high enough to reject a tone also rejects a 6 dB transmission that decodes at
    BER 0.093.

    What is stable is the ratio to the null.  A closure computed on unkeyed data is the modulus
    of a mean of T*K unit phasors, so it falls as 1/sqrt(T*K) -- a scale the capture states.
    Against it a transmission reads 155, 138, 95, 22 and 4.7 times the null from clean down to
    3 dB, while an impostor reads single digits: over twelve realizations Gaussian noise averages
    2.5 and reaches 3.7 at worst.  Paper RF-3.2 carries the full impostor distribution.

    The ratio holds across capture size, so one level means the same thing on any capture: noise
    averages 2.4 to 2.9 and reaches 3.4 at worst over captures from 4000 to 64000 samples.

    Returns the ratio.  What to do with it is the caller's: the codec decides nothing, and a
    reader's operating point belongs to the reader (paper RF-3.2)."""
    if not d.locked or not d.L:
        return 0.0
    T = max(int(n_samples) // int(d.L) - 1, 1)
    null = 1.0 / math.sqrt(max(T * max(int(d.K), 1), 1))
    return float(d.closure / null) if null > 0 else 0.0


def decode(iq, seed=None, candidates=None, *, carrier_span: float = 1.0,
           join_rows: int = JOIN_SEARCH_ROWS):
    """The receiver, whole: a capture and a key in, bits out.

    Nothing else is passed because nothing else is known -- the width, the row offset, the mode
    count and the alphabet are all measured by :func:`discover`.  With ``seed=None`` not even the
    key is needed and this reads an open broadcast.

    The geometry is read on the integer screen, which assumes the receiver samples on the sender's
    instants.  :func:`decode_scaled` is the same receiver at a continuous scale and a real start.

    Returns ``(bits, Discovery)``; the discovery carries the measured parameters and the evidence
    behind them, so a caller can see what was read rather than trust a bare answer."""
    return _decode_at(iq, seed, discover(iq, seed, candidates, carrier_span=carrier_span,
                                         join_rows=join_rows))


def decode_scaled(iq, seed=None, candidates=None, *, carrier_span: float = 1.0,
                  join_rows: int = JOIN_SEARCH_ROWS):
    """:func:`decode` with the geometry read at a continuous scale and a real start
    (:func:`_discover_scaled`): a sender at any rate, and at the receiver's own rate on another
    clock.

    Two radios at one nominal rate sample at a position that drifts through every fraction of a
    sample (0.75 ppm at 2.4 Msps: once every 0.6 s), and the integer screen assumes the two
    coincide.  Off the sender's instants a dense QAM row no longer reads as one there: measured on
    a simulated bench window (16-QAM, 2.4 -> 2.4 Msps, 22 dB, ~0.4 of a sample off) the integer
    scan read 0.84-0.94 of the constant-modulus null at the true offset, where a Gaussian row reads
    1, and blind 16- and 64-QAM came out as differential or 4-QAM at BER ~0.5; read here, every
    window decoded at the right family and order (research/rank_leak.md, "Blind decode on the
    current waveform").  It costs the scaled search's time: seconds a width on a burst, where the
    integer read takes about one.  ``candidates`` is the width range, as for :func:`decode` (a
    search bound; ``None`` searches every width the capture can evidence, which costs minutes)."""
    return _decode_at(iq, seed, _discover_scaled(iq, seed, candidates, carrier_span=carrier_span,
                                                 join_rows=join_rows))


def _decode_at(iq, seed, d: "Discovery"):
    """The payload read at a discovery's geometry: which payload rides, its order, its bits."""
    if not d.locked:
        return np.zeros(0, dtype=np.int64), d
    # Which payload rides is measured, not assumed.  The two closures are compared and the
    # larger names the payload, which keeps the payload format out of the parameters the two ends
    # agree on; see :func:`coherent_closure` for the four cases and their separation.  Taking the
    # differential path unconditionally returns BER 0.4965 on a coherent capture.
    #
    # The coherent read sees every occupied mode.  A coherent capture closes on no differential
    # hypothesis, so the beacon one can win a contest between two near-zero closures and hand
    # three modes to a beacon that is not there; those modes are given back here.
    K_occ = int(d.K) + (BEACON_MODES if d.beacon else 0)
    Y, nu, ref = _coherent_read(iq, d.L, K_occ, seed, d.offset, d.phase, d.carrier, d.scale)
    coherent = _fourth_closure(Y)
    if coherent > d.closure:
        # The order is measured like everything else: the square lattice the de-keyed symbols
        # close on (:func:`_qam_order`).  The rows before the reference carry nothing.
        #
        # ``closure`` stays the evidence that named the payload -- the fourth-power closure, a
        # mean of unit phasors like the differential one, so :func:`closure_significance` reads it
        # against the same 1/sqrt(T*K) null.  The lattice closure is not that kind of number: a
        # scatter of unit phasors reads ~0.3 on the 4-QAM lattice, so reported as ``closure`` it
        # made a fragment carrying nothing read 45x the null.  It is reported as ``lattice``.
        #
        # Open, each row comes back at the sender's own radius r_t / r_0, so the rows are
        # levelled by their own power before the lattice is read, and fitted to it after.
        M, lattice, redundancy, _ = _qam_order(_row_levels(Y) if seed is None else Y)
        if seed is None:
            Y = _row_levels(Y, M, L=d.L)
        return _qam_bits(Y, M), replace(
            d, K=K_occ, M=M, closure=coherent, redundancy=redundancy, carrier=nu, row0=0,
            aligned=False, beacon=False, payload="coherent", reference=int(ref),
            lattice=float(lattice))
    return read(iq, d.L, d.K, seed, M=d.M, offset=d.offset,
                phase=d.phase, carrier=d.carrier, row0=d.row0,
                skip=BEACON_MODES if d.beacon else 0,
                scale=d.scale), replace(d, payload="differential")


# ---------------------------------------------------------------------------
# The evidence read, at one geometry
# ---------------------------------------------------------------------------
# :func:`discover` is the front door and the only way to locate a screen.  What follows is the
# same evidence at a geometry a caller already has -- a diagnostic, not a second locator.

def constant_modulus_read(iq, L: int, seed, *, offset: int = 0, nonce=None,
                          max_rows: int | None = None, phase: float = 0.0,
                          carrier: float = 0.0) -> float:
    """How evenly each row spreads its energy over the keyed modes, at a geometry the caller has:
    the constant-modulus read :func:`discover` locates with (see the section above
    :func:`_cm_score`), on all L modes.

    At the true width, row offset and carrier every row's coordinates are unit modulus (or a
    square lattice's few levels) and the read is 0 noiseless -- measured 1.4e-15 -- rising toward
    1 with noise.  At any other cut the rows are mixtures of many coordinates, close to complex
    Gaussian, and it reads ~1: measured 0.6-1 one sample out and anywhere else.  A cliff, so the
    argmin needs no threshold.

    The read is keyed.  Only a holder of the aperture can place the offset, so an eavesdropper may
    detect that something is transmitting without being able to align to it.  With ``seed=None``
    the aperture is the public zero state and anyone can align, which is what broadcast means.

    It needs almost no data: one row's coordinates are already unit modulus.  Each row is read
    against its own mean energy, so the per-row radius (:func:`row_radius`) drops out exactly."""
    L = int(L)
    T = _held(len(iq), offset, L)
    if max_rows is not None:
        T = min(T, int(max_rows))
    if T < 1:
        return float("inf")
    return _cm_at(iq, L, seed, nonce, int(offset), T, float(phase), float(carrier))


# ---------------------------------------------------------------------------
# Analog payload -- the embedding rides as a direction, not as bits
# ---------------------------------------------------------------------------
# Channel loss becomes meaning-preserving drift rather than bit errors: no cliff, no quantizer
# floor.  D dims spread over N samples; N/D is the processing gain, trading fidelity and reach
# against packing.  Direction is the payload; scale is not.

def compose_embedding(vec, seed, *, spread: float = 2.0, nonce=None,
                      unit_rms: bool = True) -> np.ndarray:
    """Write an embedding: spread a D-dim vector over N = round(D*spread) samples through the
    keyed operator.  ``spread`` >= 1 trades samples for reach."""
    vec = np.asarray(vec, dtype=complex).ravel()
    D = vec.size
    if D == 0:
        raise ValueError("embedding is empty")
    nrm = np.linalg.norm(vec)
    if nrm <= 0:
        raise ValueError("embedding has zero norm -- direction is the payload, so it is undefined")
    N = max(D, int(round(D * float(spread))))
    return to_iq(spread_operator(N, D, seed, nonce) @ (vec / nrm), unit_rms=unit_rms)


def embedded_dimension(iq, seed, nonce=None) -> int:
    """How many dimensions the embedding carries, measured from the capture and the key.

    The spread operator is nested in D (see :func:`spread_operator`), so the energy the capture
    leaves outside the first D columns falls as D grows and reaches the floor exactly at the
    dimension that was sent -- a step, the same shape the mode count is read from.  Monotone, so
    it bisects: about eleven projections rather than a sweep.

    Measured, recovering D with no prior knowledge of it: 768 at spread 2 in 69.5 ms, 768 at
    spread 1 in 31.8 ms, 300 in 15.0 ms, 64 in 2.4 ms -- each exact, each reconstructing the
    original direction at cosine 1.000000.

    Under noise it returns N, which is safe.  Noise fills every dimension, so no residual reaches
    the floor and the read reports the whole space.  Reading the embedding at D = N still recovers
    the direction -- measured cosine 0.9962, 0.9849, 0.9418 and 0.8202 at 18, 12, 6 and 0 dB
    against a true D of 768 -- because the surplus dimensions carry only what the channel put
    there.  What is lost is the knowledge of D, not the message."""
    iq = np.asarray(iq).ravel()
    N = len(iq)
    if N == 0:
        return 0
    Q = _spread_full(N, seed, nonce)
    energy = float(np.sum(np.abs(iq) ** 2))
    if energy <= 0.0:
        return 0

    def outside(D: int) -> float:
        S = Q[:, :D]
        return float(np.sum(np.abs(iq - S @ (S.conj().T @ iq)) ** 2)) / energy

    # The floor is read, not picked.  ``entropy.resolution_floor`` is the library's answer to
    # exactly this question -- the scale below which a quantity cannot be told from the round-off
    # in computing it -- and it derives that from the working dtype's epsilon, so it follows the
    # backend and the compute precision instead of asserting them.  The residual here is already
    # relative (divided by the capture's own energy), so the pooled scale it is measured against
    # is 1.
    #
    # Nothing about the answer turns on the floor's exact value: the residual is 3.1e-16 at the
    # true D and 4.3e-4 one dimension below, twelve orders apart.
    #
    # The edge read that finds the mode count does not apply here, for a structural reason.
    # That read wants two flat levels -- used modes at one energy, empty ones at the floor.  An
    # embedding's coefficients are the vector's own components, which are whatever the vector is,
    # so a dimension it barely uses is indistinguishable from one it never had.  Measured, the
    # split lands at 1533 of 1536 against a true 768 even where the signal band sits 127x above
    # the noise band.
    # ``typical`` is N rather than 1: the residual is a sum of N per-sample contributions, each
    # resolvable only to the arithmetic's epsilon, so the scale below which the sum cannot be
    # told from round-off is N times that.  Measured, the residual at the true D is 3.1e-16
    # against an epsilon of 2.2e-16 -- above a one-sample floor, which is why passing 1.0
    # here overshoots and returns a D too large.
    floor = float(N) * _lib.entropy.macheps(np, Q)
    lo, hi = 1, N
    while lo < hi:                       # monotone in D, so about log2(N) projections
        mid = (lo + hi) // 2
        if outside(mid) <= floor:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def read_embedding(iq, D: int | None = None, seed=None, *, nonce=None) -> np.ndarray:
    """Read an embedding: recover the direction by the keyed adjoint, exact for orthonormal
    columns.  Compare against the original by cosine; the scale is not carried.

    ``D=None`` measures the dimension (:func:`embedded_dimension`) instead of taking it from the
    caller, so a receiver needs only the capture and the key."""
    iq = np.asarray(iq)
    if D is None:
        D = embedded_dimension(iq, seed, nonce)
    return spread_operator(len(iq), int(D), seed, nonce).conj().T @ iq
