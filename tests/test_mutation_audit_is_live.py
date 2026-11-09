"""
The invariant: every mutation the audit claims to apply still applies.

``scripts/mutation_audit.py`` is what the paper's test-quality claim rests on (PAPER §RF-9): the
suite is checked against deliberate defects rather than only against the code as written.  The
script locates each defect by an exact source string, which makes it silently perishable -- rename
a variable, rewrap a line, and the target stops matching.  A mutation that does not apply mutates
nothing, the suite passes, and the audit reports ``caught`` for a test that was never challenged.
The report is then worse than no report, because it certifies coverage that was not measured.

Measured, this is not hypothetical: renaming ``_best_of``'s ``fallback`` parameter to ``proposed``
defanged the entry written for it in the same sitting.

So the targets are checked here, on every run.  This does not re-run the audit -- that costs a full
suite per mutation -- it checks only that each mutation still has something to bite.
"""
from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "mutation_audit.py"
SOURCE = ROOT / "src" / "entroptics_rf" / "waveform.py"


def _mutations():
    """The MUTATIONS table, read without importing the script -- importing it would run the audit."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "MUTATIONS" for t in node.targets):
            return [tuple(ast.literal_eval(e) for e in entry.elts)
                    for entry in node.value.elts]
    raise AssertionError("MUTATIONS table not found in mutation_audit.py")


MUTATIONS = _mutations()


def test_the_audit_has_mutations_at_all():
    assert len(MUTATIONS) >= 5, f"only {len(MUTATIONS)} mutations; the audit is not saying much"


@pytest.mark.parametrize("name,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_still_has_something_to_bite(name, old, new):
    """The target must occur exactly once.  Zero means the defect can no longer be introduced and
    the audit's verdict for it is vacuous; more than one means the mutation lands in places it was
    never reasoned about, so what the audit measured is not what it says."""
    src = io.open(SOURCE, encoding="utf-8").read()      # newlines translated, as the script reads it
    n = src.count(old)
    assert n == 1, (
        f"mutation {name!r} matches {n} places in waveform.py, not 1 -- its audit verdict is "
        f"not evidence of anything until the target is realigned")


@pytest.mark.parametrize("name,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_actually_changes_the_source(name, old, new):
    """A replacement equal to its target is a no-op wearing a mutation's name."""
    assert old != new, f"mutation {name!r} replaces its target with itself"


def test_the_shipped_source_is_not_mutated():
    """The audit copies the source aside and restores it in a ``finally``.  A foreground timeout
    kills that, which once left a mutation live with the suite still green."""
    assert "MUTANT" not in SOURCE.read_text(encoding="utf-8"), (
        "waveform.py carries a MUTANT marker: an audit run was interrupted, and "
        "src/entroptics_rf/waveform.py.mutation-backup holds the original")
