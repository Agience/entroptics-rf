"""The ten-detector blind battery on the current waveform, with conventional positive controls.

A small, fixed-seed version of ``research/detectability/current_waveform.py`` (which holds the
SNR sweep and the results file).  A key-free observer runs every detector of
``scripts/evidence_adversary.py`` against a noise-only null; a trace is a hit outside the null's
central 98%.  The current waveform (compose, K = L = 64, DQPSK, keyed row radius) must read as
noise on all ten, and QPSK must not, or the battery's silence would mean nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evidence_adversary as ea                                # noqa: E402

N, REPS, NULLS = 16384, 8, 80


@pytest.fixture(scope="module")
def bands():
    rng = np.random.default_rng(11)
    null = [ea.noise(N, rng) for _ in range(NULLS)]
    out = {}
    for name, f in ea.DETECTORS:
        v = np.array([f(t) for t in null])
        out[name] = (np.quantile(v, 0.01), np.quantile(v, 0.99))
    return out


def _hits(bands, traces):
    return {name: int(sum(not (bands[name][0] <= f(t) <= bands[name][1]) for t in traces))
            for name, f in ea.DETECTORS}


@pytest.mark.parametrize("snr", [None, 20.0])
def test_the_current_waveform_reads_as_noise_on_all_ten_detectors(bands, snr):
    rng = np.random.default_rng(12)
    traces = []
    for _ in range(REPS):
        x = ea.entroptics(N, rng)
        if snr is not None:
            x = x + 10 ** (-snr / 20) * ea.noise(N, rng)
            x = (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)
        traces.append(x)
    hits = _hits(bands, traces)
    assert max(hits.values()) <= 2, hits          # P_fa 2%: 8 traces rarely give 3 by chance


def test_the_battery_sees_conventional_qpsk(bands):
    rng = np.random.default_rng(13)
    hits = _hits(bands, [ea.linear_mod(N, rng, 4) for _ in range(REPS)])
    assert min(hits.values()) == REPS, hits
