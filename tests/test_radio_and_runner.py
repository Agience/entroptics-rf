"""Radio controls and the operations that ride on them."""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf import (Link, LoopbackRadio, PlutoRadio, Reception, RtlSdrRadio, ber,
                           loopback, open_radio, random_bits, receive, transmit)
from entroptics_rf import waveform as wf


# ── radio controls ────────────────────────────────────────────────────────────────────────────

def test_loopback_returns_what_was_transmitted():
    r = LoopbackRadio()
    iq = np.exp(2j * np.pi * 0.01 * np.arange(1000)).astype(np.complex64)
    r.transmit(iq)
    cap = r.receive(1000)
    assert len(cap) == 1000
    assert np.allclose(cap.iq, iq - iq.mean(), atol=1e-5)


def test_capture_carries_its_own_settings():
    """A capture that does not report the rate it was taken at is unreadable later -- and the
    point here is that the reader needs its own rate, not the transmitter's."""
    r = LoopbackRadio(sample_rate=2.4e6, center_freq=915e6)
    r.transmit(np.ones(100, dtype=np.complex64))
    cap = r.receive(100)
    assert cap.sample_rate == 2.4e6 and cap.center_freq == 915e6


def test_loopback_impairments_are_applied_and_deterministic():
    iq = np.ones(4096, dtype=np.complex64)
    a = LoopbackRadio(snr_db=10, cfo=1e-3, image_dbc=-25, seed=7)
    b = LoopbackRadio(snr_db=10, cfo=1e-3, image_dbc=-25, seed=7)
    a.transmit(iq); b.transmit(iq)
    x, y = a.receive(4096).iq, b.receive(4096).iq
    assert np.array_equal(x, y), "a seeded channel must replay exactly"
    assert not np.allclose(x, iq - iq.mean()), "impairments were not applied"


def test_loopback_repeats_the_buffer_when_asked_for_more():
    """A real receiver captures whatever is on the air, including several passes of a cyclic
    transmission -- so asking for more than one buffer must not fail."""
    r = LoopbackRadio()
    r.transmit(np.arange(100).astype(np.complex64))
    assert len(r.receive(350)) == 350


def test_receive_only_radio_refuses_to_transmit():
    """A silent no-op transmit is the most expensive failure mode on a bench."""
    class RxOnly(RtlSdrRadio):
        def __init__(self):
            pass
    with pytest.raises(NotImplementedError):
        RxOnly().transmit(np.ones(4))


def test_loopback_with_nothing_transmitted_raises():
    with pytest.raises(RuntimeError):
        LoopbackRadio().receive(10)


def test_radio_is_a_context_manager():
    with LoopbackRadio() as r:
        r.transmit(np.ones(64, dtype=np.complex64))
        assert len(r.receive(64)) == 64


def test_open_radio_dispatch():
    assert isinstance(open_radio("loopback"), LoopbackRadio)
    assert isinstance(open_radio(""), LoopbackRadio)
    with pytest.raises(ValueError):
        open_radio("no-such-radio")


@pytest.mark.parametrize("cls,pkg", [(PlutoRadio, "pyadi-iio"), (RtlSdrRadio, "pyrtlsdr")])
def test_missing_driver_names_the_package_to_install(cls, pkg):
    """Hardware drivers are optional, so the failure has to be actionable rather than a traceback
    from a library the user never asked for.  Skipped when the driver is present."""
    try:
        cls()
    except ImportError as e:
        assert pkg in str(e)
    except Exception:
        pytest.skip(f"{cls.__name__} driver present; no import failure to check")


# ── the link agreement ────────────────────────────────────────────────────────────────────────

def test_link_rejects_impossible_geometry():
    with pytest.raises(ValueError):
        Link(K=65, L=64, seed=1)
    with pytest.raises(ValueError):          # every mode is lit: a dark mode leaks the width
        Link(K=60, L=64, seed=1)


def test_link_reports_its_own_rate():
    link = Link(K=64, L=64, seed=1, M=4)
    assert link.mode_ratio == pytest.approx(1.0)
    assert link.bits_per_sample == pytest.approx(2.0)
    assert link.capacity(64 * 11) == 10 * 64 * 2


# ── the operations ────────────────────────────────────────────────────────────────────────────

def test_blind_loopback_recovers_the_payload_and_measured_the_width():
    link = Link(K=64, L=64, seed=1234)
    sent, rx = loopback(link, n_bits=2 * 300 * 64)
    assert isinstance(rx, Reception) and rx.blind
    assert rx.L == link.L, "the width was measured, and it is the right one"
    assert ber(sent, rx.bits) == 0.0


def test_blind_and_informed_agree_on_a_clean_link():
    """The lock must not be doing anything the informed path would disagree with."""
    link = Link(K=64, L=64, seed=1234)
    bits = random_bits(2 * 300 * 64, seed=2)
    _, blind = loopback(link, bits, blind=True)
    _, told = loopback(link, bits, blind=False)
    assert told.lock is None and blind.lock is not None
    assert np.array_equal(blind.bits, told.bits)


def test_loopback_survives_noise_carrier_offset_and_an_iq_image():
    link = Link(K=64, L=64, seed=1234)
    sent, rx = loopback(link, n_bits=2 * 400 * 64, snr_db=18, cfo=3e-4, image_dbc=-25, seed=3)
    assert rx.L == link.L
    assert ber(sent, rx.bits) < 0.01


@pytest.mark.parametrize("delay", [0.0, 0.1, 0.2])
def test_loopback_survives_a_fractional_sample_delay(delay):
    """A real capture never starts on a sample boundary either.  A fractional delay smears each
    row across two, so it rotates the screen partly out of the aperture.  Measured clean at
    K = L = 64 with the constant-modulus read: the width still reads right at 0.5 samples, but the
    payload degrades well before (BER 0.06 / 0.21 / 0.34 at 0.3 / 0.4 / 0.5 samples), so a front
    end owes it timing to well under a third of a sample.  Tested with margin."""
    link = Link(K=64, L=64, seed=1234)
    sent, rx = loopback(link, n_bits=2 * 400 * 64, delay=delay, snr_db=20, seed=4)
    assert rx.L == link.L


def test_transmit_returns_the_iq_it_sent():
    link = Link(K=32, L=32, seed=7)
    r = LoopbackRadio()
    iq = transmit(r, random_bits(2000, seed=0), link)
    assert iq.dtype == np.complex64 and len(iq) % link.L == 0


def test_receive_reports_optics_alongside_the_bits():
    link = Link(K=32, L=32, seed=7)
    r = LoopbackRadio()
    iq = transmit(r, random_bits(2 * 200 * 32, seed=0), link)
    rx = receive(r, link, len(iq))
    assert rx.optics.L == link.L and rx.optics.T > 0
    assert np.isfinite(rx.optics.papr_db) and np.isfinite(rx.optics.fill)


def test_ber_scores_a_failed_lock_as_total_loss():
    """Zero bits recovered must not score as a suspiciously perfect 0.0."""
    assert ber(np.array([1, 0, 1]), np.zeros(0)) == 1.0
    assert ber(np.array([1, 0, 1]), np.array([1, 0, 1])) == 0.0
    assert ber(np.array([1, 0, 1, 1]), np.array([0, 0, 1, 1])) == pytest.approx(0.25)


def test_no_lock_returns_an_empty_reception_rather_than_raising():
    """A receiver pointed at noise should report that it found nothing, not crash the caller."""
    link = Link(K=64, L=64, seed=1234)
    r = LoopbackRadio()
    noise = (np.random.default_rng(0).standard_normal(4000)
             + 1j * np.random.default_rng(1).standard_normal(4000)).astype(np.complex64)
    r.transmit(noise)
    rx = receive(r, link, 4000, candidates=range(60, 69))
    assert isinstance(rx, Reception)          # a Reception either way; the lock tells the story
    assert rx.bits.size == 0 or rx.lock is not None


# ── the SDR drivers, exercised against stub devices ───────────────────────────────────────────
#
# No hardware has ever run this waveform, and the two driver paths were reached by nothing but an
# ImportError test.  A stub device cannot prove the waveform survives a real front end, but it can
# prove the driver drives the API the way the vendor library requires -- which is where a
# never-executed path actually breaks, and where the failure looks like a dead link rather than a
# bug.  These tests record what the driver does to the device and assert on the sequence.

import sys
import types


class _FakeAd9361:
    def __init__(self, uri=None):
        self.uri = uri
        self.log = []
        self.sent = None
        self._cyclic = False
        self.rx_buffer_size = 0

    def __setattr__(self, k, v):
        if k == "tx_cyclic_buffer" and hasattr(self, "log"):
            self.log.append("set_cyclic")
        object.__setattr__(self, k, v)

    def tx(self, x):
        self.log.append("tx")
        self.sent = np.asarray(x)

    def rx(self):
        self.log.append("rx")
        return np.zeros(self.rx_buffer_size or 8, dtype=np.complex64)

    def tx_destroy_buffer(self):
        self.log.append("destroy")

    def rx_destroy_buffer(self):
        self.log.append("rx_destroy")


def _with_fake_adi(monkeypatch):
    dev = _FakeAd9361()

    def build(uri=None):
        dev.uri = uri          # record it: the URI reaching the device is a thing worth testing
        return dev

    mod = types.ModuleType("adi")
    mod.ad9361 = build
    monkeypatch.setitem(sys.modules, "adi", mod)
    return dev


def test_pluto_fills_the_converter_without_clipping(monkeypatch):
    """A fixed transmit scale has to be guessed against the waveform's PAPR; a measured one does
    not.  The peak must land on int16 full scale -- using less is receive SNR given away before
    the signal reaches the air, and using more is clipping."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio()
    iq = wf.compose(np.random.default_rng(0).integers(0, 2, 2 * 200 * 64), 64, 64, 1234, M=4)
    r.transmit(iq)
    from entroptics_rf.radio import _PLUTO_FULL_SCALE as FS
    peak = float(np.max(np.abs(dev.sent)))
    assert peak <= FS * (1 + 1e-4), f"overshoots the documented range: peak {peak} vs {FS}"
    assert peak > 0.9 * FS, f"only {20*np.log10(FS/peak):.1f} dB of the converter used"
    # and it must survive the int16 cast pyadi-iio performs, without wrapping
    assert np.max(np.abs(dev.sent.real.astype(np.int16))) > 0, "wrapped on the int16 cast"


def test_pluto_releases_the_buffer_before_sending_again(monkeypatch):
    """pyadi-iio will not change cyclic mode under a live buffer.  Without a teardown a second
    transmit either raises or silently keeps sending the first payload, and a bench cannot tell
    that from a dead link."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio()
    x = np.ones(256, dtype=np.complex64)
    r.transmit(x)
    r.transmit(x)
    # every tx must be preceded by a destroy, and the cyclic flag set only after that destroy
    for i, ev in enumerate(dev.log):
        if ev == "tx":
            assert "destroy" in dev.log[:i], f"tx at {i} with no preceding destroy: {dev.log}"
            assert dev.log[i - 1] == "set_cyclic" and dev.log[i - 2] == "destroy", dev.log


def test_pluto_honours_an_explicit_scale(monkeypatch):
    """A caller sending successive buffers that must share one gain sets the scale itself."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio(tx_scale=1000.0)
    r.transmit(np.full(64, 0.5, dtype=np.complex64))
    assert np.allclose(np.abs(dev.sent), 500.0)


class _FakeRtl:
    def __init__(self, device_index=0):
        self.sample_rate = 0.0
        self.center_freq = 0.0
        self.gain = 0.0
        self.closed = False

    def set_agc_mode(self, on):
        pass

    def set_manual_gain_enabled(self, on):
        pass

    # the hardware lands on the nearest PLL step, not the request
    def get_sample_rate(self):
        return self.sample_rate * 1.000037

    def get_center_freq(self):
        return self.center_freq + 213.0

    def read_samples(self, n):
        return np.zeros(int(n), dtype=np.complex128)

    def close(self):
        self.closed = True


def test_rtlsdr_reports_what_it_tuned_to_not_what_was_asked(monkeypatch):
    """The gap between requested and achieved is exactly the clock independence this receiver
    exists to demonstrate, so echoing the request back would hide the measurement.  The rate was
    already read back; the centre frequency was not."""
    mod = types.ModuleType("rtlsdr")
    mod.RtlSdr = _FakeRtl
    monkeypatch.setitem(sys.modules, "rtlsdr", mod)
    from entroptics_rf.radio import RtlSdrRadio
    r = RtlSdrRadio(sample_rate=2_400_000, center_freq=915_000_000)
    assert r.sample_rate != 2_400_000, "sample rate was echoed, not read back"
    assert r.center_freq != 915_000_000, "centre frequency was echoed, not read back"
    assert abs(r.center_freq - 915_000_213) < 1.0
    with pytest.raises(NotImplementedError):
        r.transmit(np.zeros(4, dtype=np.complex64))


def test_pluto_drains_a_stale_buffer_before_measuring(monkeypatch):
    """The first buffer back from a device is whatever was sitting in it.

    On a bench that reads as the *previous* capture -- the right answer to the last question --
    which is the most confusing failure a receiver can have.  Both drivers drain one read before
    keeping one, and neither was exercised by a test."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio()
    cap = r.receive(4096)
    assert dev.log.count("rx") == 2, f"expected drain + read, got {dev.log}"
    assert dev.rx_buffer_size == 4096
    assert len(cap) == 4096
    assert cap.sample_rate == r.sample_rate and cap.center_freq == r.center_freq
    assert cap.gain == r.rx_gain_db


def test_rtlsdr_drains_before_measuring_and_carries_its_settings(monkeypatch):
    import sys as _sys
    import types as _types
    mod = _types.ModuleType("rtlsdr")
    reads = []

    class _Rec(_FakeRtl):
        def read_samples(self, n):
            reads.append(int(n))
            return super().read_samples(n)

    mod.RtlSdr = _Rec
    monkeypatch.setitem(_sys.modules, "rtlsdr", mod)
    from entroptics_rf.radio import RtlSdrRadio
    r = RtlSdrRadio(sample_rate=2_400_000, center_freq=915_000_000, gain=32.8)
    cap = r.receive(8192)
    assert len(reads) == 2 and reads[1] == 8192, f"expected drain then read, got {reads}"
    assert len(cap) == 8192
    assert cap.sample_rate == r.sample_rate and cap.gain == 32.8


def test_a_receive_removes_dc(monkeypatch):
    """LO leakage puts a DC term on every capture; both drivers subtract the mean."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio()
    monkeypatch.setattr(dev, "rx", lambda: np.full(1024, 3 + 4j, dtype=np.complex64))
    cap = r.receive(1024)
    assert abs(complex(np.mean(cap.iq))) < 1e-6, "a constant capture should come back centred"


def test_auto_gain_is_agc_and_reports_no_gain(monkeypatch):
    """`gain="auto"` hands the decision to the device, so the capture carries no gain figure."""
    import sys as _sys
    import types as _types
    mod = _types.ModuleType("rtlsdr")
    mod.RtlSdr = _FakeRtl
    monkeypatch.setitem(_sys.modules, "rtlsdr", mod)
    from entroptics_rf.radio import RtlSdrRadio
    r = RtlSdrRadio(gain="auto")
    assert r.receive(256).gain is None


def test_close_is_safe_to_call_twice(monkeypatch):
    """A teardown that raises on a device already released turns tidy-up into a crash."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio()

    def boom():
        raise RuntimeError("already destroyed")

    monkeypatch.setattr(dev, "tx_destroy_buffer", boom)
    r.close()
    r.close()


def test_close_releases_the_rx_buffer_before_the_context(monkeypatch):
    """An RX buffer left to interpreter shutdown can outlive its iio context, and its destructor
    then writes through freed memory -- an access violation, measured on a Pluto+.  close() must
    release it, and must do so once however often it is called."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio()
    r.receive(64)
    r.close()
    r.close()
    assert dev.log.count("rx_destroy") == 2, f"resize + close, once each; got {dev.log}"


def test_a_new_capture_size_rebuilds_the_rx_buffer(monkeypatch):
    """pyadi reads the buffer size only when it builds a buffer; a live one keeps the old size."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    r = PlutoRadio()
    r.receive(64)
    before = dev.log.count("rx_destroy")
    r.receive(64)
    assert dev.log.count("rx_destroy") == before, "same size: the live buffer is reused"
    assert len(r.receive(128)) == 128 and dev.log.count("rx_destroy") == before + 1


def test_a_two_transmitter_part_mirrors_the_buffer_at_one_gain(monkeypatch):
    """A 2r2t Pluto assigns FPGA lanes to TX jacks at random once per connection (measured: lane 0
    alone reached TX1 in 6 of 12 connections).  Mirroring the samples onto both lanes, at one
    gain, puts them on TX1 every time; a single-transmitter part is driven as before."""
    from entroptics_rf.radio import PlutoRadio
    dev = _with_fake_adi(monkeypatch)
    dev._tx_channel_names = ["voltage0", "voltage1", "voltage2", "voltage3"]
    r = PlutoRadio(tx_gain_db=-26.0)
    assert dev.tx_enabled_channels == [0, 1]
    assert dev.tx_hardwaregain_chan0 == dev.tx_hardwaregain_chan1 == -26.0
    r.transmit(np.exp(2j * np.pi * np.arange(64) / 8))
    assert isinstance(dev.sent, np.ndarray) and dev.sent.shape == (2, 64)
    assert np.array_equal(dev.sent[0], dev.sent[1])
    dev = _with_fake_adi(monkeypatch)
    PlutoRadio(tx_gain_db=-26.0)
    assert dev.tx_enabled_channels == [0] and not hasattr(dev, "tx_hardwaregain_chan1")


def test_open_radio_builds_the_named_driver(monkeypatch):
    import sys as _sys
    import types as _types
    from entroptics_rf.radio import open_radio, PlutoRadio, RtlSdrRadio
    dev = _with_fake_adi(monkeypatch)
    assert isinstance(open_radio("pluto"), PlutoRadio)
    assert isinstance(open_radio("pluto:ip:192.168.2.1"), PlutoRadio)
    assert dev.uri == "ip:192.168.2.1", "an explicit URI must reach the device"
    mod = _types.ModuleType("rtlsdr")
    mod.RtlSdr = _FakeRtl
    monkeypatch.setitem(_sys.modules, "rtlsdr", mod)
    assert isinstance(open_radio("rtlsdr"), RtlSdrRadio)
    assert isinstance(open_radio("rtl"), RtlSdrRadio)


def test_a_transmit_only_radio_refuses_to_receive():
    """The mirror of the receive-only case: a radio must not invent a capture it cannot take."""
    from entroptics_rf.radio import Radio

    class TxOnly(Radio):
        name = "txonly"

        def transmit(self, iq, *, cyclic=False):
            pass

    with pytest.raises(NotImplementedError, match="cannot receive"):
        TxOnly().receive(64)


def test_stopping_a_loopback_transmit_clears_what_it_was_holding():
    """A loopback carries only what it was handed, so stopping must actually stop it -- otherwise
    a later receive returns a transmission the caller believes it cancelled."""
    from entroptics_rf.radio import LoopbackRadio
    r = LoopbackRadio(snr_db=None, seed=0)
    r.transmit(np.ones(256, dtype=np.complex64))
    assert len(r.receive(256)) == 256
    r.stop_transmit()
    with pytest.raises(RuntimeError, match="nothing is being transmitted"):
        r.receive(256)
