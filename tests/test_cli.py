"""The CLI: the first thing anyone types, and until now the only surface no test touched.

These are smoke tests in the useful sense -- each subcommand is run end to end on the loopback
radio and has to exit 0 -- plus the two behaviours that are easy to get wrong and that nothing
else pins: an expected failure must read as an answer rather than a traceback, and the key must be
expressible as something other than an integer.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_rf.__main__ import main

SMALL = ["--K", "32", "--L", "32"]


@pytest.mark.parametrize("argv", [
    ["selftest"],
    ["focus", "--widths", "28", "36"],
    ["ber", "--snr-list", "12"],
    ["embed", "--D", "128", "--snr-list", "12"],
    ["tx", "--bits", "2000"],
])
def test_every_subcommand_runs_on_the_loopback(argv, capsys):
    assert main(argv + SMALL) == 0
    out = capsys.readouterr().out
    assert out.strip(), f"{argv[0]} printed nothing"


def test_an_expected_failure_is_an_answer_not_a_traceback(capsys):
    """`rx` on a loopback nothing has transmitted into is a legitimate thing to type.

    It reports no signal and exits cleanly, rather than raising a RuntimeError traceback that
    reads as a crash.  The distinction matters because the same command against absent hardware
    fails the same way, and a user who cannot tell 'no signal' from 'this program is broken' will
    debug the wrong thing."""
    code = main(["rx", "--samples", "4096"] + SMALL)
    err = capsys.readouterr().err
    assert code == 1
    assert "Traceback" not in err
    assert err.startswith("error: ")
    assert "transmit first" in err, "the message should say what to do about it"


def test_impossible_geometry_is_reported_the_same_way(capsys):
    code = main(["ber", "--K", "70", "--L", "64", "--snr-list", "12"])
    err = capsys.readouterr().err
    assert code == 1 and "Traceback" not in err and err.startswith("error: ")


def test_the_key_need_not_be_a_number():
    """A key is a key.  The waveform takes bytes, str or int as three different keys, and the
    CLI can express all three."""
    from entroptics_rf.__main__ import _key
    assert _key("1234") == 1234, "digits must stay an integer, or every existing key changes"
    assert _key("hunter2") == "hunter2"
    assert _key("-7") == -7


def test_a_text_key_actually_keys_the_waveform():
    """And it must reach the aperture, not merely be accepted by the parser."""
    from entroptics_rf import waveform as wf
    a = wf.mode_basis(8, 16, 1234, None)
    b = wf.mode_basis(8, 16, "hunter2", None)
    assert not np.allclose(a, b)


def test_rx_reports_a_lock_when_there_is_something_to_receive(monkeypatch, capsys, tmp_path):
    """`rx` succeeding was never exercised -- only its failure was.

    It cannot succeed on a fresh loopback, because the CLI builds a new radio per command and a
    loopback carries only what that same object was handed.  So the radio is injected already
    holding a transmission, which is what a real one does.  Coverage found this: the whole
    reporting path -- the lock line, the optics, the interference read, `--out` -- ran in no test."""
    import entroptics_rf.__main__ as cli
    from entroptics_rf.radio import LoopbackRadio
    from entroptics_rf import waveform as wf

    K, L, SEED = 32, 32, 1234
    bits = np.random.default_rng(0).integers(0, 2, 2 * 400 * K)
    radio = LoopbackRadio(snr_db=None, seed=0)
    radio.transmit(wf.compose(bits, K, L, SEED, M=4))
    monkeypatch.setattr(cli, "open_radio", lambda *a, **k: radio)

    out_file = tmp_path / "capture.npy"
    code = cli.main(["rx", "--K", str(K), "--L", str(L), "--seed", str(SEED),
                     "--samples", str(400 * L), "--out", str(out_file)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "lock: L=32" in out, out
    assert "read " in out and "bits" in out
    assert "interference:" in out and "optics:" in out
    assert out_file.exists(), "--out should have written the capture"


def test_rx_says_so_when_there_is_no_screen(monkeypatch, capsys):
    """Noise must be called noise, not printed under the word "lock" and exited 0.

    `Discovery.locked` only says the sweep returned a width, which it almost always does: pure
    noise locks some width, and bits are read from it.  The geometry score cannot fix that either
    -- a DC level or a bare tone locks some geometry with a constant-modulus score near 0, as good
    as a real screen's.  The alphabet closure is what separates, 1.000 against 0.011-0.023, so
    that is what the command judges on and prints."""
    import entroptics_rf.__main__ as cli
    from entroptics_rf.radio import LoopbackRadio

    rng = np.random.default_rng(0)
    radio = LoopbackRadio(snr_db=None, seed=0)
    radio.transmit((rng.standard_normal(20000) + 1j * rng.standard_normal(20000)
                    ).astype(np.complex64))
    monkeypatch.setattr(cli, "open_radio", lambda *a, **k: radio)
    code = cli.main(["rx", "--K", "32", "--L", "32", "--samples", "20000"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "NO SCREEN" in out, out
    assert "closure=" in out and "null)" in out, "show the evidence and its scale"
    assert "noise" in out.lower(), "it should say what the bits are worth"


def test_focus_applies_an_snr_when_asked():
    """The --snr branch of `focus` was never taken by a test."""
    assert main(["focus", "--K", "32", "--L", "32", "--widths", "28", "36", "--snr", "12"]) == 0
