"""
The invariant: no FFT and no DSP anywhere in this package.

This is the claim the design rests on -- the feature axis is time, so a screen is composed by a
matrix product and read by its adjoint, and no transform is needed at either end.  It is enforced
on the source rather than asserted in a docstring, because a single convenience import would end
it.

The scan is blunt by design.  It reads the shipped source and fails on any spectral or filtering
call, whatever its provenance -- numpy, scipy, or a hand-rolled convolution.  A contributor who
genuinely needs one comes here and changes this file, which is the conversation the invariant
exists to force.
"""
from __future__ import annotations

import pathlib
import re

import pytest

import entroptics_rf

PKG = pathlib.Path(entroptics_rf.__file__).parent
SOURCES = sorted(PKG.glob("*.py"))

# Spectral transforms, filtering, and the resampling that usually implies one.  `correlate` and
# `convolve` are here because a matched filter is the acquisition idea this waveform replaces:
# the lock is a property of the capture rather than a comparison against a template.
FORBIDDEN = re.compile(
    r"""\b(
        (np|numpy|scipy|sp|xp)\s*\.\s*fft\b     # numpy.fft / scipy.fft
      | \bfft\s*\(                              # a bare fft( call
      | \b(i?fft2?|i?rfft|fftshift|fftfreq|hfft|ihfft)\s*\(
      | scipy\s*\.\s*signal
      | \b(lfilter|filtfilt|firwin|butter|cheby[12]|resample_poly|decimate|welch|periodogram
        |spectrogram|stft|istft|hilbert|convolve|correlate|convolve2d)\s*\(
    )""",
    re.VERBOSE,
)


def _code_lines(path: pathlib.Path):
    """Source lines with comments and docstring bodies stripped.

    The prose in this package discusses FFTs at length -- it has to, since not using one is the
    point -- so a scan that reads comments would fail on its own explanation.  Only executable
    code is checked."""
    text = path.read_text(encoding="utf-8")
    # drop triple-quoted blocks (docstrings and block comments), then trailing # comments
    text = re.sub(r'"""(?:.|\n)*?"""', '""', text)
    text = re.sub(r"'''(?:.|\n)*?'''", "''", text)
    for i, line in enumerate(text.splitlines(), start=1):
        code = line.split("#", 1)[0]
        if code.strip():
            yield i, code


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_transform_in_source(path):
    """No spectral transform, filter, or correlation in any shipped module."""
    hits = [(i, line.strip()) for i, line in _code_lines(path) if FORBIDDEN.search(line)]
    assert not hits, (
        f"{path.name} reaches for a transform/filter, which breaks the package's central claim:\n"
        + "\n".join(f"  line {i}: {s}" for i, s in hits))


def test_the_scan_can_actually_fail(tmp_path):
    """The guard above is only worth having if it detects a violation, so plant one.

    A regex that stopped matching would otherwise read as a passing suite forever."""
    bad = tmp_path / "violation.py"
    bad.write_text("import numpy as np\ndef f(x):\n    return np.fft.fft(x)\n", encoding="utf-8")
    hits = [i for i, line in _code_lines(bad) if FORBIDDEN.search(line)]
    assert hits, "the FFT scan failed to flag a plain np.fft.fft call"


def test_entroptics_reads_used_are_fft_free():
    """The library call this package leans on -- ``Aperture.phi_F`` -- is a correlation
    eigenvalue read, not a spectral estimate.

    entroptics does contain one FFT (an autocorrelation read this package never calls), so the
    no-FFT claim covers the path taken, and this test pins which path that is."""
    import numpy as np
    from entroptics import Aperture, reads

    W = np.random.default_rng(0).standard_normal((64, 16)) \
        + 1j * np.random.default_rng(1).standard_normal((64, 16))
    # phi_F is defined from axis_spectrum, the per-axis correlation eigenspectrum
    assert np.isclose(Aperture(W, window=None).phi_F, reads.phi_F(W))
    assert reads.axis_spectrum(W, 1).shape[0] == 16
