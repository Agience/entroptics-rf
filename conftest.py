"""Auto-mark the long-running tests ``slow`` so ``pytest -m "not slow"`` is a fast lane.

The substrings name the long-running tests (the blind/scale/coherent sweeps dominate the
rf suite). Plain ``pytest`` with no ``-m`` still runs everything; the marker only takes effect when
you filter, so this changes nothing about a default run.
"""
import pytest

_SLOW = ("scale", "joint_read", "blind_width", "coherent")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if any(p in item.nodeid for p in _SLOW):
            item.add_marker(pytest.mark.slow)
