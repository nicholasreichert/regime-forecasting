"""Wrap generated tables so they never overflow the column.

The result tables are produced programmatically and their widths depend on the
numbers in them, so a table that fits today can overflow after a rerun with
different data. Rather than hand-tune ``\\tabcolsep`` per table -- which would
not survive regeneration -- every generated ``tabular`` is wrapped in
``\\fitwidth`` (defined in the preamble), which scales down only when the
natural width exceeds the column.

Run after the table generators; idempotent.
"""

from __future__ import annotations

import re
from pathlib import Path

TABLES = Path("paper") / "tables"

# Full-width table* environments already span both columns and must not be
# scaled against \columnwidth.
_TWOCOL = re.compile(r"\\begin\{table\*\}")


def fit_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if r"\fitwidth" in text or _TWOCOL.search(text):
        return False
    if r"\begin{tabular}" not in text:
        return False

    start = text.index(r"\begin{tabular}")
    end_marker = r"\end{tabular}"
    end = text.index(end_marker) + len(end_marker)

    wrapped = (
        text[:start]
        + "\\fitwidth{%\n"
        + text[start:end]
        + "%\n}"
        + text[end:]
    )
    path.write_text(wrapped, encoding="utf-8")
    return True


def main() -> None:
    if not TABLES.exists():
        print(f"[skip] {TABLES} does not exist")
        return
    changed = 0
    for fp in sorted(TABLES.glob("*.tex")):
        if fit_file(fp):
            print(f"  wrapped {fp.name}")
            changed += 1
    print(f"fit {changed} table(s) to column width")


if __name__ == "__main__":
    main()
