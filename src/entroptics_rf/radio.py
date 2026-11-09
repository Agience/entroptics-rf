"""
radio.py -- radio controls.  The hardware side of the link, and the only place it lives.

One small interface, three implementations.  Everything above this module deals in raw complex
IQ and nothing else, so the waveform never learns which radio it is riding on and the radio never
learns what it is carrying.

    LoopbackRadio   a simulated link -- AWGN, carrier offset, fractional delay, IQ imbalance.
                    No hardware, deterministic under a seed.  This is what the tests and the
                    self-test run on, and what a new waveform idea should be tried on first.
    PlutoRadio      ADALM-Pluto / AD936x via pyadi-iio.  Full duplex, so it can also loop itself
                    back over a cable -- which proves the reshape lock at one clock, and says
                    nothing about two.
    RtlSdrRadio     RTL-SDR via pyrtlsdr.  Receive only, 8-bit -- the cheap independent-clock
                    receiver, and the one to report two-clock numbers from.

The drivers are optional.  ``pyadi-iio`` and ``pyrtlsdr`` are imported inside the constructors, so
this module imports cleanly on a machine with no SDR attached and the simulated path always works.
A missing driver raises at construction with the pip name to install, rather than at import with a
traceback from a library the user never asked for.

Scaling is the radio's job.  The waveform hands over unit-RMS complex64; each radio applies its
own transmit scale (a Pluto wants roughly full-scale 12-bit integers, a simulator wants none) and
each receiver hands back complex samples with its DC offset removed.  That removal is the one
thing done to a capture here, because a DC offset is the receiver's own additive artefact rather
than part of the signal.  Nothing else is filtered, resampled or transformed on the way in or
out.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

__all__ = ["Radio", "LoopbackRadio", "PlutoRadio", "RtlSdrRadio", "open_radio", "Capture"]


@dataclass
class Capture:
    """What a receiver hands back: the samples and the settings they were taken at.

    The sample rate travels with the capture, because a receiver that does not report it makes
    its own captures unreadable later -- and because the point of this waveform is that the reader
    needs its own rate, not the transmitter's."""
    iq: np.ndarray
    sample_rate: float
    center_freq: float
    gain: float | None = None

    def __len__(self) -> int:
        return int(len(self.iq))


class Radio:
    """The interface.  Subclasses implement whichever half they physically have.

    ``transmit`` sends one buffer; ``receive`` returns a :class:`Capture` of ``n`` samples.  A
    receive-only radio raises on ``transmit`` rather than pretending to send, because a silent
    no-op transmit is the single most expensive failure mode on a bench."""

    name = "radio"

    def transmit(self, iq, *, cyclic: bool = False) -> None:
        raise NotImplementedError(f"{self.name} cannot transmit")

    def receive(self, n: int) -> Capture:
        raise NotImplementedError(f"{self.name} cannot receive")

    def stop_transmit(self) -> None:
        """Stop a cyclic transmission.  A no-op where transmission is not persistent."""

    def close(self) -> None:
        """Release the device."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class LoopbackRadio(Radio):
    """A simulated link: what was transmitted, plus the impairments a real one adds.

    ``snr_db``      additive white Gaussian noise, set against the transmitted power.
    ``cfo``         carrier offset in cycles per sample -- a coordinate discovery measures and
                    the differential read is only tolerant of, so it is the interesting knob.
    ``delay``       fractional sample delay, applied by linear interpolation.  Present because a
                    real capture never starts on a row boundary, and a waveform that only works
                    when it does is not a waveform.
    ``image_dbc``   IQ-imbalance image at this level, as a conjugate leak.  A real direct-conversion
                    front end has one; -25 dBc is a fair cheap-radio figure.
    ``seed``        makes the whole channel deterministic, so a failure can be replayed exactly.

    The buffer is held until it is read, so ``transmit`` then ``receive`` is the whole loop and
    there is no timing to get wrong."""

    name = "loopback"

    def __init__(self, *, sample_rate: float = 2_000_000, center_freq: float = 915_000_000,
                 snr_db: float | None = None, cfo: float = 0.0, delay: float = 0.0,
                 image_dbc: float | None = None, seed: int = 0) -> None:
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.snr_db = snr_db
        self.cfo = float(cfo)
        self.delay = float(delay)
        self.image_dbc = image_dbc
        self._rng = np.random.default_rng(seed)
        self._buf: np.ndarray | None = None

    def transmit(self, iq, *, cyclic: bool = False) -> None:
        self._buf = np.asarray(iq, dtype=np.complex64).copy()

    def stop_transmit(self) -> None:
        self._buf = None

    def _channel(self, x: np.ndarray, n: int) -> np.ndarray:
        if self.delay:                                   # fractional delay by linear interpolation
            d = float(self.delay)
            i = int(np.floor(d))
            f = d - i
            x = (1 - f) * np.roll(x, i) + f * np.roll(x, i + 1)
        reps = int(np.ceil(n / len(x)))
        y = np.tile(x, reps)[:n].astype(np.complex128)
        if self.cfo:
            y = y * np.exp(2j * np.pi * self.cfo * np.arange(n))
        if self.image_dbc is not None:
            y = y + (10.0 ** (float(self.image_dbc) / 20.0)) * np.conj(y)
        if self.snr_db is not None:
            p = float(np.mean(np.abs(y) ** 2))
            s = np.sqrt(p / (2.0 * 10.0 ** (float(self.snr_db) / 10.0)))
            y = y + s * (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n))
        return y.astype(np.complex64)

    def receive(self, n: int) -> Capture:
        if self._buf is None:
            raise RuntimeError(
                "nothing is being transmitted on this loopback -- it carries only what this\n"
                "process has sent, so transmit first (or use 'selftest' / 'ber', which do\n"
                "both ends together); a real radio needs --radio pluto or --radio rtlsdr")
        y = self._channel(self._buf, int(n))
        return Capture(y - y.mean(), self.sample_rate, self.center_freq)


_PLUTO_FULL_SCALE = 2 ** 14   # what the AD936x expects, per ADI's own examples: the DAC is
                              # 12-bit and the part is driven within +/-2**14, not int16 range


class PlutoRadio(Radio):
    """ADALM-Pluto / AD936x over pyadi-iio.  Transmit and receive.

    A cabled Pluto-to-Pluto or TX-to-RX loop runs both ends off one clock.  That proves the
    waveform survives a real front end -- quantization, gain, LO leakage -- and it says nothing
    about independent clocks, which is the harder claim.  Report the two cases apart.

    On a two-transmitter part (2r2t) the same buffer goes out of both TX jacks at one gain.  Which
    jack an FPGA data lane reaches is drawn at random once per iio connection: measured on a
    Pluto+ (``v0.38_plutoplus``), lane 0 alone reached jack TX1 in 6 of 12 fresh connections and
    held its assignment across 10 buffers within one.  Mirroring the samples onto both lanes put
    them on TX1 in 10 of 10.

    The receive lanes were measured the same way: RX2 lands on lane 1 deterministically,
    and the analog RX path is always live (RSSI on a TX2->RX2 cable reads the same on every
    connection).  What is drawn per connection is whether the RX DATA path comes up live at all, and
    that turns on HOW the lanes are enabled: pyadi's ``rx_enabled_channels=[0, 1]`` subset brings it
    up dead almost every time, while a raw-libiio read with all four converter channels enabled is
    live in 7-8 of 8 connections.  :meth:`receive_both` takes that reliable path (with a
    reconnect-until-live retry) and returns both lanes coherently; the single-lane :meth:`receive`
    reads lane 0 as before.

    Buffers are released before the connection, always.  Left to interpreter shutdown, the iio
    context can be freed first and the buffer's destructor then writes through it -- an access
    violation in ``iio.Buffer.__del__``, reproducible by exiting with an RX buffer held."""

    name = "pluto"

    def __init__(self, uri: str = "ip:pluto.local", *, sample_rate: float = 2_000_000,
                 center_freq: float = 915_000_000, tx_gain_db: float = -10.0,
                 rx_gain_db: float = 40.0, bandwidth: float | None = None,
                 tx_scale: float | None = None) -> None:
        try:
            import adi
        except ImportError as e:
            raise ImportError("PlutoRadio needs pyadi-iio: pip install 'entroptics-rf[sdr]'") from e
        self._uri = uri
        self._sdr = adi.ad9361(uri=uri)
        self._release = weakref.finalize(self, _pluto_release, self._sdr)
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.tx_scale = None if tx_scale is None else float(tx_scale)
        # A 2r2t part lists I and Q for each transmitter; pyadi's own attribute, read not assumed.
        self._tx_lanes = max(1, len(getattr(self._sdr, "_tx_channel_names", ())) // 2)
        self._bw = int(bandwidth if bandwidth is not None else sample_rate)
        self._tx_gain_db = float(tx_gain_db)
        self.rx_gain_db = float(rx_gain_db)
        self._apply_config()

    def _apply_config(self) -> None:
        """Push this radio's settings onto the (current) pyadi session.  Also run after a
        reconnect, so a rebuilt session comes up in exactly the state the first one had."""
        self._sdr.sample_rate = int(self.sample_rate)
        self._sdr.tx_lo = self._sdr.rx_lo = int(self.center_freq)
        self._sdr.tx_rf_bandwidth = self._sdr.rx_rf_bandwidth = self._bw
        self._sdr.gain_control_mode_chan0 = "manual"
        self._sdr.rx_hardwaregain_chan0 = float(self.rx_gain_db)
        for lane in range(self._tx_lanes):
            setattr(self._sdr, f"tx_hardwaregain_chan{lane}", float(self._tx_gain_db))
        self._sdr.rx_enabled_channels = [0]
        self._sdr.tx_enabled_channels = list(range(self._tx_lanes))
        self._sdr.tx_cyclic_buffer = True

    def _reconnect(self) -> None:
        """Tear down and rebuild the pyadi session with the same settings.  The one cure for a RX
        data path that comes up dead on a fresh connection (see :meth:`receive_both`); a new iio
        connection re-rolls the lane alignment."""
        import adi
        self._release()                                   # release the old session's buffers
        self._sdr = adi.ad9361(uri=self._uri)
        self._release = weakref.finalize(self, _pluto_release, self._sdr)
        self._apply_config()

    def transmit(self, iq, *, cyclic: bool = True) -> None:
        """Send one buffer, scaled to the DAC rather than to a chosen number.

        ``tx_scale=None`` fills the converter's documented range from this buffer's own peak.
        A fixed scale has to be picked against the waveform's PAPR -- 10.5 dB at K=60, L=64 -- and
        the old default of 2**12 was picked that way: safe, but leaving 6 dB of the converter
        unused, which is 6 dB of receive SNR given away before the signal reaches the air.
        Peak-filling needs no such guess.

        The target is 2**14, not int16 full scale.  The AD936x DAC is 12 bits and ADI's own
        examples state the part expects samples within +/-2**14; driving to 32767 would overshoot
        the documented range by 6 dB.  That number is a fact about the hardware, not a choice
        about this waveform.

        For a cyclic buffer this is the whole transmission, so the scale is constant.  A caller
        sending successive non-cyclic buffers that must share one gain passes ``tx_scale``
        explicitly rather than letting each buffer normalize itself.

        The buffer is destroyed first.  pyadi-iio will not change cyclic mode under a live buffer,
        so without this a second ``transmit`` on one session either raises or silently keeps
        sending the first payload -- and a bench cannot tell the difference from a dead link."""
        x = np.asarray(iq, dtype=np.complex64)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        scale = self.tx_scale if self.tx_scale is not None else (
            _PLUTO_FULL_SCALE / peak if peak > 0 else 1.0)
        self.stop_transmit()                              # pyadi: no mode change under a buffer
        self._sdr.tx_cyclic_buffer = bool(cyclic)
        y = (x * scale).astype(np.complex64)
        self._sdr.tx(y if self._tx_lanes == 1 else [y] * self._tx_lanes)   # see the class note

    def stop_transmit(self) -> None:
        try:
            self._sdr.tx_destroy_buffer()
        except Exception:                                 # already torn down: nothing to release
            pass

    def receive(self, n: int) -> Capture:
        if int(n) != self._sdr.rx_buffer_size:
            # pyadi reads the size only when it builds a buffer, so a live one keeps the old size
            # and every later capture silently comes back at the first one's length.
            self._sdr.rx_destroy_buffer()
            self._sdr.rx_buffer_size = int(n)
        self._sdr.rx()                                    # drain one stale buffer before measuring
        y = np.asarray(self._sdr.rx(), dtype=np.complex64)
        return Capture(y - y.mean(), self.sample_rate, self.center_freq, self.rx_gain_db)

    def receive_both(self, n: int, *, retries: int = 6, live_std: float = 2.0,
                     rx2_gain_db: float | None = None):
        """Capture both receive lanes of a 2r2t part from one read -- RX1 on lane 0 and RX2 on lane
        1, off the one sampling clock, so the pair is phase-coherent for spatial work (e.g. the
        keyed blind MVDR combine).  Returns ``(rx1, rx2)`` as two :class:`Capture`.

        Read through raw libiio with ALL FOUR converter channels enabled, not pyadi's
        ``rx_enabled_channels=[0, 1]`` subset.  Measured on a Pluto+: the subset path
        brings the RX data path up DEAD (both lanes flatlined) almost every connection, while the
        all-channels raw read delivers live data in 7-8 of 8 fresh reads, with RX2 landing on lane 1
        deterministically; the analog RX path itself is always fine (RSSI unchanged on the dead
        inits).  On the ~1-in-8 dead init this reconnects the session and retries up to ``retries``
        times -- note a reconnect tears down any live TX buffer, so for a cabled TX->RX2 self-test
        start the transmission after a live read is obtained (a held data path stays live).

        ``live_std`` is the per-lane sample standard deviation (in converter counts) below which a
        lane is taken to be a dead init rather than real signal or noise; the default clears the
        observed dead floor (~1 count) with margin while passing ordinary receiver noise."""
        import iio
        rxdev = self._sdr._rxadc                           # cf-ad9361-lpc
        if rx2_gain_db is None:
            rx2_gain_db = self.rx_gain_db
        # RX2 wants its own manual gain set; harmless to re-set, guarded for a 1r1t part.
        try:
            self._sdr.gain_control_mode_chan1 = "manual"
            self._sdr.rx_hardwaregain_chan1 = float(rx2_gain_db)
        except Exception:                                  # not a 2-RX part: nothing to capture on lane 1
            pass
        last = None
        for _ in range(max(1, int(retries))):
            try:
                self._sdr.rx_destroy_buffer()              # drop pyadi's single-lane buffer
            except Exception:
                pass
            for i in range(4):
                rxdev.find_channel(f"voltage{i}").enabled = True
            buf = iio.Buffer(rxdev, int(n), False)
            try:
                for _ in range(3):                         # warm-up: the first refills can be stale
                    buf.refill()
                raw = np.frombuffer(buf.read(), dtype=np.int16).reshape(-1, 4).astype(np.float32)
            finally:
                del buf
            rx1 = raw[:, 0] + 1j * raw[:, 1]
            rx2 = raw[:, 2] + 1j * raw[:, 3]
            last = (float(np.std(rx1)), float(np.std(rx2)))
            if max(last) > float(live_std):                # data path is up
                return (Capture(rx1 - rx1.mean(), self.sample_rate, self.center_freq, self.rx_gain_db),
                        Capture(rx2 - rx2.mean(), self.sample_rate, self.center_freq, float(rx2_gain_db)))
            self._reconnect()
        raise RuntimeError(f"RX data path did not init live after {retries} reconnects "
                           f"(both lanes flat, last std {last} counts < {live_std})")

    def close(self) -> None:
        self._release()                                   # idempotent: a finalizer runs once


def _pluto_release(sdr) -> None:
    """Release a Pluto session's buffers ahead of its context, and clear its TX channel enables.

    pyadi's libiio-0.x path enables the channels a new TX buffer needs but never disables the ones
    an earlier buffer enabled, so a session that moves between TX channels builds a buffer wider
    than the data it is given.  Clearing them here leaves the next session the state it expects."""
    for release in (sdr.tx_destroy_buffer, sdr.rx_destroy_buffer):
        try:
            release()
        except Exception:                                 # already torn down: nothing to release
            pass
    try:
        for name in sdr._tx_channel_names:
            sdr._txdac.find_channel(name, True).enabled = False
    except Exception:
        pass


class RtlSdrRadio(Radio):
    """RTL-SDR over pyrtlsdr.  Receive only.

    The independent-clock receiver: cheap, 8-bit, its own oscillator.  It reports the sample rate
    the hardware tuned to rather than the one requested, because the two differ and the difference
    is exactly the clock independence the waveform is meant to survive."""

    name = "rtlsdr"

    def __init__(self, *, sample_rate: float = 2_400_000, center_freq: float = 915_000_000,
                 gain: float | str = 32.8, device_index: int = 0) -> None:
        try:
            from rtlsdr import RtlSdr
        except ImportError as e:
            raise ImportError("RtlSdrRadio needs pyrtlsdr: pip install 'entroptics-rf[rtlsdr]'") from e
        self._sdr = RtlSdr(device_index=device_index)
        self._sdr.sample_rate = float(sample_rate)
        self._sdr.center_freq = float(center_freq)
        if gain == "auto":
            self._sdr.set_agc_mode(True)
        else:
            self._sdr.set_agc_mode(False)
            self._sdr.set_manual_gain_enabled(True)
            self._sdr.gain = float(gain)
        # Both settings are read back rather than echoed.  A tuner lands on the nearest step
        # its PLL can reach, and the difference between asked and got is exactly the clock
        # independence this receiver exists to demonstrate -- echoing the request would hide
        # the measurement.  The rate was already read back; the frequency was not.
        self.sample_rate = float(self._sdr.get_sample_rate())
        self.center_freq = float(self._sdr.get_center_freq())
        self.gain = gain

    def receive(self, n: int) -> Capture:
        self._sdr.read_samples(64 * 1024)                 # drain stale samples before measuring
        y = np.asarray(self._sdr.read_samples(int(n)), dtype=np.complex64)
        return Capture(y - y.mean(), self.sample_rate, self.center_freq,
                       None if self.gain == "auto" else float(self.gain))

    def close(self) -> None:
        try:
            self._sdr.close()
        except Exception:
            pass


def open_radio(spec: str = "loopback", **kw) -> Radio:
    """Open a radio from a short spec string -- what the CLI takes.

        loopback                the simulated link (default)
        pluto                   ip:pluto.local
        pluto:ip:192.168.2.1    an explicit pyadi URI
        rtlsdr                  the first RTL-SDR

    Keyword arguments are passed through to the constructor, so sample rate, frequency and gain
    are set the same way whichever radio is named."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    if kind in ("loopback", "sim", ""):
        return LoopbackRadio(**kw)
    if kind == "pluto":
        return PlutoRadio(rest or "ip:pluto.local", **kw)
    if kind in ("rtlsdr", "rtl"):
        return RtlSdrRadio(**kw)
    raise ValueError(f"unknown radio {spec!r}; use loopback, pluto[:uri] or rtlsdr")
