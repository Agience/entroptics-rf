"""Keep the PHY quickstart (examples/quickstart.py) runnable and accurate.

examples/quickstart.py is a user's first contact with the physical layer; if it stops delivering,
the "hello world" is broken.  This runs it and checks the three claims it prints: a receiver told
only the key measures the width and decodes; an open (key-free) broadcast decodes; and the key
actually gates reading (the wrong key does not lock, so no payload comes out).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_QS = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


@pytest.fixture(scope="module")
def quickstart():
    spec = importlib.util.spec_from_file_location("entroptics_rf_quickstart", _QS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quickstart_demonstrates_blind_decode_and_key_gating(quickstart):
    ok, facts = quickstart.run_quickstart(verbose=False)
    assert ok
    assert facts["width_measured"] == 64              # measured blind, never told
    assert facts["private_ber"] < 1e-2                # keyed, blind, over noise + CFO
    assert facts["open_ber"] < 1e-2                   # open broadcast, no key
    assert facts["right_key_ber"] < 1e-2              # the right key locks and reads
    assert facts["wrong_key_ber"] > 0.3               # the wrong key does not -- no payload
    assert facts["right_key_closure"] > 0.5 > facts["wrong_key_closure"]
