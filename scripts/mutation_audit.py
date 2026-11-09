"""Does the suite catch deliberate defects, or does it only pass?

Kept in the repository because a test count is not a measure of anything.  Seventeen mutations
have been run this way; sixteen were caught, and the one that survived -- a flipped carrier sign
-- went unnoticed because every test that applied a carrier used an offset small enough that both
signs decoded.  See paper RF-9.

Add a mutation whenever a behaviour matters and you are not sure a test constrains it.

Each mutation is a plausible mistake -- a sign, an index, an off-by-one -- applied to the shipped
source, with the suite run against it.  A mutation that survives is a gap: the tests do not
constrain that behaviour, and someone could make that mistake without being told.

The source is restored in a finally, always.
"""
import shutil
import subprocess
import sys
from pathlib import Path

# This file is `<repo>/scripts/mutation_audit.py`, so the repo root is one level up:
# parents[0] is `scripts/`, parents[1] is the repo root.  Derived, not hardcoded, so the
# audit runs from any checkout.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "entroptics_rf" / "waveform.py"
BAK = SRC.with_suffix(".py.mutation-backup")

MUTATIONS = [
    ("differential conjugation flipped",
     "    d = A[1:] * np.conj(A[:-1])                     # differential: phase-only, magnitude drops out",
     "    d = np.conj(A[1:]) * A[:-1]                     # MUTANT"),
    ("carrier sign flipped in the read",
     "        d = d * np.exp(-2j * np.pi * float(carrier))",
     "        d = d * np.exp(2j * np.pi * float(carrier))   # MUTANT"),
    ("row alignment always reports zero",
     "    best_col, best_fit = np.zeros(len(ladder)), -1.0\n    settled = _settled_rows(",
     "    best_col, best_fit = np.zeros(len(ladder)), -1.0\n    settled = {} if True else _settled_rows("),
    ("scan rows and offsets transposed",
     "    e = (Z.real ** 2 + Z.imag ** 2).reshape(T, W, -1)",
     "    e = (Z.real ** 2 + Z.imag ** 2).reshape(W, T, -1).transpose(1, 0, 2)   # MUTANT"),
    ("keyed rotation neutralized",
     "    return np.exp(2j * np.pi * out[start:start + rows])",
     "    return np.ones_like(out[start:start + rows], dtype=complex)   # MUTANT"),
    # The widening's candidates go uncompared, so it keeps whichever came last -- the rule that
    # returned (87, 16) at 0.859 for a capture it had already read as (64, 0) at 1.25e-15.
    ("widening keeps its newest candidate, uncompared",
     "    if len(cands) < 2:\n        return proposed",
     "    if True:\n        return proposed   # MUTANT"),
]


def run_suite():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-x", "-q", "--tb=no",
                        "-p", "no:randomly"],
                       cwd=ROOT, capture_output=True, text=True, timeout=2000)
    return r.returncode, r.stdout


# A foreground timeout SIGKILLs this and the finally never runs, which once left a mutation live
# in the source with the suite still green -- the batch had died before reaching the test that
# would have caught it.  So restore any leftover backup before doing anything else.
if BAK.exists():
    print(f"a previous run left {BAK.name}; restoring first")
    shutil.copy2(BAK, SRC)
    BAK.unlink()
assert "MUTANT" not in SRC.read_text(encoding="utf-8"), "source already mutated; fix before running"

shutil.copy2(SRC, BAK)
try:
    original = SRC.read_text(encoding="utf-8")
    print(f"{'mutation':<38} {'verdict':<10} first failure")
    print("-" * 92)
    for name, old, new in MUTATIONS:
        if old not in original:
            print(f"{name:<38} {'SKIPPED':<10} anchor not found")
            continue
        SRC.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            code, out = run_suite()
        except subprocess.TimeoutExpired:
            code, out = -1, "(timeout)"
        if code == 0:
            print(f"{name:<38} {'SURVIVED':<10} <- nothing caught it")
        else:
            fails = [l for l in out.split("\n") if l.startswith("FAILED")]
            first = fails[0].replace("FAILED ", "")[:44] if fails else out.strip()[-44:]
            print(f"{name:<38} {'caught':<10} {first}")
finally:
    shutil.copy2(BAK, SRC)
    BAK.unlink()
    print("\nsource restored")
