#!/usr/bin/env python3
"""Lint the LaTeX in every Markdown file against GitHub's MathJax subset.

GitHub renders `$…$` and `$$…$$` with MathJax configured to refuse any macro
that is itself defined with `\\newcommand` (an XSS guard) and to refuse
cross-reference and HTML-hook macros. When one appears, GitHub replaces the whole
expression with

    The following macros are not allowed: operatorname

which is how a perfectly good derivation turns into a red box on the rendered
page. `\\operatorname` is the one that bites in practice: it is internally a
`\\newcommand`, so it is blocked, while `\\mathrm` — the primitive it expands to
— is allowed and looks identical.

Usage:
    uv run python tools/mdmath.py check
    uv run python tools/mdmath.py fix      # rewrites the mechanical cases

The scan skips fenced blocks and inline code, so a document may discuss a
forbidden macro as long as it quotes it in backticks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# macro -> (replacement hint, why)
DISALLOWED: dict[str, tuple[str, str]] = {
    # defined with \newcommand inside MathJax: blocked by the safe extension
    "operatorname": (r"\mathrm{name}", "blocked; use \\mathrm{name}"),
    "DeclareMathOperator": ("define the operator inline", "definition macro"),
    "newcommand": ("inline the definition", "definition macro"),
    "renewcommand": ("inline the definition", "definition macro"),
    "providecommand": ("inline the definition", "definition macro"),
    "def": ("inline the definition", "definition macro"),
    "require": ("GitHub enables its own extension set", "blocked"),
    # HTML / CSS / unicode hooks: blocked as injection vectors
    "href": ("plain text", "blocked HTML hook"),
    "url": ("plain text", "blocked HTML hook"),
    "class": ("plain text", "blocked HTML hook"),
    "style": ("plain text", "blocked HTML hook"),
    "cssId": ("plain text", "blocked HTML hook"),
    "htmlId": ("plain text", "blocked HTML hook"),
    "htmlClass": ("plain text", "blocked HTML hook"),
    "htmlStyle": ("plain text", "blocked HTML hook"),
    "unicode": ("the literal character", "blocked"),
    "enclose": ("plain text", "blocked"),
    # cross-references: not implemented by GitHub's renderer
    "label": ("a hard-coded number", "unsupported"),
    "tag": ("a hard-coded number", "unsupported"),
    "ref": ("a hard-coded number", "unsupported"),
    "eqref": ("a hard-coded number", "unsupported"),
    # not part of the loaded extension set: renders as an undefined control seq.
    "bm": (r"\mathbf{…} or \boldsymbol{…}", "undefined"),
    "cancel": (r"\text{cancelled}", "undefined"),
    "abs": (r"\lvert x \rvert", "undefined"),
    "norm": (r"\lVert x \rVert", "undefined"),
    "ceil": (r"\lceil x \rceil", "undefined"),
    "floor": (r"\lfloor x \rfloor", "undefined"),
    "argmax": (r"\arg\max", "undefined"),
    "argmin": (r"\arg\min", "undefined"),
    "softmax": (r"\mathrm{softmax}", "undefined"),
    "sigmoid": (r"\sigma(x)", "undefined"),
    "relu": (r"\mathrm{relu}", "undefined"),
    "Var": (r"\mathrm{Var}", "undefined"),
    "Cov": (r"\mathrm{Cov}", "undefined"),
    "R": (r"\mathbb{R}", "undefined"),
    "N": (r"\mathbb{N}", "undefined"),
    "Z": (r"\mathbb{Z}", "undefined"),
    "Q": (r"\mathbb{Q}", "undefined"),
}

# environments MathJax only accepts inside a display block of its own
BAD_ENVS = ("align", "align*", "alignat", "eqnarray", "equation", "equation*")

FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1", re.S | re.M)
INLINE = re.compile(r"`[^`\n]*`")
MACRO = re.compile(r"\\([a-zA-Z]+)")


def mask(text: str) -> str:
    """Blank out code so quoted macros are not reported, keeping line numbers."""
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    return INLINE.sub(blank, FENCE.sub(blank, text))


def scan(text: str) -> list[tuple[int, str]]:
    """Every `(line, macro)` the GitHub renderer would reject."""
    found: list[tuple[int, str]] = []
    for lineno, line in enumerate(mask(text).splitlines(), start=1):
        for name in MACRO.findall(line):
            if name in DISALLOWED:
                found.append((lineno, name))
        for env in BAD_ENVS:
            if re.search(r"\\begin\{" + re.escape(env) + r"\}", line):
                found.append((lineno, f"begin{{{env}}}"))
    return found


def fix(text: str) -> str:
    """Rewrite the mechanical cases; leave anything subtle to a human."""
    return re.sub(r"\\operatorname\*?\{([a-zA-Z]+)\}", r"\\mathrm{\1}", text)


def markdown_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.md")
        if not {".git", ".venv", "node_modules"} & set(p.parts)
    )


def check() -> int:
    problems = 0
    for path in markdown_files():
        for lineno, name in scan(path.read_text()):
            hint = DISALLOWED.get(name, ("", "unsupported environment"))[0]
            print(f"{path.relative_to(ROOT)}:{lineno}: \\{name} — use {hint}")
            problems += 1
    if problems:
        print(f"{problems} unsupported math macro(s)")
        return 1
    print(f"ok · math renders on GitHub across {len(markdown_files())} files")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "check":
        return check()
    if cmd == "fix":
        touched = 0
        for path in markdown_files():
            text = path.read_text()
            new = fix(text)
            if new != text:
                path.write_text(new)
                touched += 1
        print(f"rewrote {touched} file(s)")
        return check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
