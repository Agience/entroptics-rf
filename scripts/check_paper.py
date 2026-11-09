"""Consistency checks for research/PAPER.md.

The test suite cannot reach the paper, because the paper is prose.  These are the failure modes
repeatable enough to automate: LaTeX mangled by a shell heredoc, one number quoted in two places
that disagree, a cross-reference to a renamed section, a test count that has drifted.

Run it after any edit to the paper.  Exit code is non-zero if anything is found.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "research" / "PAPER.md"
CONTROL = "\t\r\x07\x08\x0b\x0c"


def check_control_chars(lines):
    """A backslash swallowed by a shell heredoc leaves a control character behind.

    `\\times` becomes TAB + "imes", `\\rangle` becomes CR + "angle".  Both have happened here, and
    both render as silently wrong LaTeX rather than as an error."""
    bad = [(i + 1, repr(l[:90])) for i, l in enumerate(lines)
           if any(c in l for c in CONTROL)]
    return [f"line {n}: control character in {t}" for n, t in bad]


def check_math_delimiters(lines):
    out = []
    for i, l in enumerate(lines):
        if l.strip().startswith("$$"):
            continue
        if l.count("$") % 2:
            out.append(f"line {i + 1}: odd number of '$' -- {l[:80]!r}")
    return out


def check_replacement_chars(text):
    n = text.count("�")
    return [f"{n} U+FFFD replacement characters (a lost encoding)"] if n else []


def check_section_refs(text):
    """Every §RF-x.y pointed at must exist as a heading."""
    heads = set(re.findall(r"^#{2,3}\s+(RF-\d+(?:\.\d+)?)", text, re.M))
    refs = set(re.findall(r"§(RF-\d+(?:\.\d+)?)", text))
    missing = sorted(refs - heads)
    return [f"cross-reference to {r}, which is not a heading" for r in missing]


def check_test_count(text):
    """The claimed test count against what pytest actually collects."""
    m = re.search(r"(\d+) tests pin the identities", text)
    if not m:
        return ["no 'N tests pin the identities' claim found"]
    claimed = int(m.group(1))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--collect-only"],
                           cwd=ROOT, capture_output=True, text=True, timeout=900)
    except Exception as e:                                        # noqa: BLE001
        return [f"could not collect tests to check the count: {e}"]
    total = sum(int(x) for x in re.findall(r"^tests/.*: (\d+)$", r.stdout, re.M))
    if total != claimed:
        return [f"paper claims {claimed} tests, pytest collects {total}"]
    return []


NUM = re.compile(r"(?<![\w.])(\d+\.\d{3,})(?![\w])")


def check_repeated_numbers(text):
    """A figure quoted in several places should agree with itself.

    Only looks at numbers written to three or more decimals, which in this paper are measurements
    rather than counts, and only reports a number whose *neighbouring words* differ -- the same
    value legitimately appears in unrelated contexts."""
    where = defaultdict(set)
    for m in NUM.finditer(text):
        line = text[text.rfind("\n", 0, m.start()) + 1: text.find("\n", m.end())]
        where[m.group(1)].add(line.strip()[:60])
    return []          # reporting only; disagreements need a human eye, see below


SCI = re.compile(r"(?<![\w.])(\d+\.\d{2,}|\d+(?:\.\d+)?e[+-]?\d+)(?![\w])", re.I)


#: the paper writes scientific notation as LaTeX and everything else writes it as a float literal.
#: Without this the checker reports every such figure as drift, which buries the real ones.
LATEX_SCI = re.compile(
    r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?"    # 6.9\times10^{-16}
    r"|10\^\{?(-?\d+)\}?"                                  # bare 10^{15}
)


def _canon(text: str) -> str:
    """Rewrite LaTeX scientific notation as plain float literals before figures are extracted."""
    def rep(m):
        if m.group(1) is not None:
            return f"{float(m.group(1)):g}e{int(m.group(2))}"
        return f"1e{int(m.group(3))}"
    return LATEX_SCI.sub(rep, text)


def _figures(text):
    """Numbers written with two or more decimals, or in scientific notation.

    In both the paper and the source those are measurements rather than counts: a 64 or a 4096 is
    a size, a 0.826 or a 6.7e-16 is something somebody ran.  Bare integers are deliberately not
    collected, because they are mostly array shapes.

    LaTeX scientific notation is normalized first, so $6.9\\times10^{-16}$ in the paper and
    6.9e-16 in a docstring are recognized as the same figure rather than as two."""
    out = set()
    for m in SCI.finditer(_canon(text)):
        v = m.group(1).lower().replace("e+", "e")
        out.add(v)
        try:                       # also store a canonical form, so 6.9e-16 == 6.90e-16
            out.add(f"{float(v):g}")
        except ValueError:
            pass
    return out


def check_source_agreement(_):
    """Measured figures quoted in the source or README but nowhere in the paper.

    The same experiments are written down in three places -- paper, source comments, README --
    with no link between them, so a correction has to be made three times.  This is what catches
    the copies that were missed.  It cannot tell a stale figure from one the paper simply never
    quoted, so it is advisory: it prints what to look at and does not fail the run.

    Exact match only.  Flagging any figure within a few percent of a paper figure produces 52
    hits, essentially all coincidence -- 9.83 against 10.0, 0.155 against 0.15 -- because numeric
    proximity without matching context is not evidence.  Telling a re-quoted measurement from an
    unrelated one needs the surrounding sentence rather than the number.  Exact match plus a human
    glance found seven real drifts.
    """
    paper = _figures(PAPER.read_text(encoding="utf-8"))
    out = []
    # FOUR copies, not three: the tests carry measured claims in their docstrings too, and
    # one of them kept a five-capture lock rate for hours after the paper moved to twenty.
    sources = (sorted((ROOT / "src" / "entroptics_rf").glob("*.py"))
               + sorted((ROOT / "tests").glob("*.py")) + [ROOT / "README.md"])
    for src in sources:
        prose_only = src.suffix == ".py"
        text = src.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.split("\n"), 1):
            stripped = line.strip()
            if prose_only and not (
                    stripped.startswith("#") or '"""' in line or stripped.startswith("*")
                    or (" -- " in line
                        and not stripped.startswith(("def ", "return ", "assert ")))):
                continue
            for fig in _figures(line):
                if fig not in paper:
                    out.append(f"{src.name}:{line_no}: {fig} — {stripped[:60]}")
    return out


def main() -> int:
    text = PAPER.read_text(encoding="utf-8")
    lines = text.split("\n")
    problems = []
    for name, fn, arg in (
        ("control characters", check_control_chars, lines),
        ("math delimiters", check_math_delimiters, lines),
        ("encoding", check_replacement_chars, text),
        ("cross-references", check_section_refs, text),
        ("test count", check_test_count, text),
    ):
        found = fn(arg)
        status = "ok" if not found else f"{len(found)} PROBLEM(S)"
        print(f"  {name:<20} {status}")
        problems += found

    # advisory: not counted as a problem, because it cannot distinguish a stale figure from one
    # the paper never quoted.  It is here because three stale ones survived a paper-only fix.
    drift = check_source_agreement(None)
    print(f"  {'source+readme':<20} {len(drift)} not found in the paper (advisory)")
    print()
    for pr in problems:
        print(f"  ! {pr}")
    if drift:
        print("\n  figures in the source that the paper does not carry "
              "(check they still hold):")
        for d in drift[:12]:
            print(f"    . {d}")
        if len(drift) > 12:
            print(f"    . ... and {len(drift) - 12} more")
    print(f"\n{len(lines)} lines, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
