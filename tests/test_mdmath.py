"""Markdown math must render on GitHub, which allows only a MathJax subset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import mdmath  # noqa: E402


def test_scanner_flags_blocked_macros():
    assert mdmath.scan("$\\operatorname{Var}(x)$") == [(1, "operatorname")]
    assert mdmath.scan("$$\\begin{align} x \\end{align}$$") == [(1, "begin{align}")]


def test_scanner_ignores_code_and_prose():
    text = "```\n\\operatorname{x}\n```\n\nand `\\operatorname{y}` in a span\n"
    assert mdmath.scan(text) == []
    assert mdmath.scan("plain prose about $\\mathrm{Var}$") == []


def test_every_document_renders_on_github():
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: \\{name}"
        for path in mdmath.markdown_files()
        for line, name in mdmath.scan(path.read_text())
    ]
    assert not offenders, "\n".join(offenders)
