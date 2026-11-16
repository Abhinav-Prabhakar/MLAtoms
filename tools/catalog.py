#!/usr/bin/env python3
"""Catalog tooling for MLAtoms.

`catalog.json` is the single source of truth for what the repository contains.
This script renders the generated index block inside every README, validates the
tree against it, and reports coverage.

Usage:
    uv run python tools/catalog.py render
    uv run python tools/catalog.py check
    uv run python tools/catalog.py stats
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog.json"

BEGIN = "<!-- catalog:begin -->"
END = "<!-- catalog:end -->"

TIER_LABEL = {1: "1", 2: "2", 3: "3"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load() -> dict:
    return json.loads(CATALOG.read_text())


def chapters(data: dict) -> list[dict]:
    return data["chapters"]


def all_entries(data: dict) -> list[tuple[dict, dict]]:
    return [(ch, e) for ch in chapters(data) for e in ch["entries"]]


def chapter_dir(ch: dict) -> Path:
    return ROOT / ch["dir"]


def is_implemented(ch: dict, entry: dict) -> bool:
    """An atom exists once both its implementation and its companion doc exist."""
    d = chapter_dir(ch)
    stem = entry["file"][:-3]  # strip ".py"
    return (d / entry["file"]).is_file() and (d / f"{stem}.md").is_file()


def count_variants(ch: dict) -> int:
    return sum(len(e["variants"]) for e in ch["entries"])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def anchor(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9 \-_]", "", slug)
    return slug.strip().replace(" ", "-")


def chapter_block(ch: dict) -> str:
    atoms = ch["entries"]
    variants = count_variants(ch)
    done = sum(1 for e in atoms if is_implemented(ch, e))

    out = [
        f"**{len(atoms)} atoms · {variants} variants · "
        f"{done} implemented**",
        "",
        "| Atom | File | Tier | Variants | What it does |",
        "|:--|:--|:--:|--:|:--|",
    ]
    for e in atoms:
        mark = "✓" if is_implemented(ch, e) else "·"
        out.append(
            f"| {mark} {e['name']} | [`{e['file']}`]({e['file']}) "
            f"| {TIER_LABEL[e['tier']]} | {len(e['variants'])} | {e['blurb']} |"
        )

    out += ["", "**Variant map** — every variant below is implemented in its atom's file "
            "and derived in its companion `.md`.", ""]
    for e in atoms:
        doc = e["file"][:-3] + ".md"
        vs = " · ".join(f"`{v}`" for v in e["variants"])
        out.append(f"- **{e['name']}** ([`{e['file']}`]({e['file']}) · [{doc}]({doc})) — {vs}")
    return "\n".join(out)


def root_block(data: dict) -> str:
    rows = [
        "| # | Chapter | Atoms | Variants | Tier 1 | Focus |",
        "|:--|:--|--:|--:|--:|:--|",
    ]
    for ch in chapters(data):
        t1 = sum(1 for e in ch["entries"] if e["tier"] == 1)
        rows.append(
            f"| {ch['id']} | [{ch['title']}]({ch['dir']}/README.md) "
            f"| {len(ch['entries'])} | {count_variants(ch)} | {t1} "
            f"| {ch['tagline']} |"
        )
    atoms, variants = totals(data)
    rows += [
        "",
        f"**{len(chapters(data))} chapters · {atoms} atoms · {variants} variants**",
    ]
    return "\n".join(rows)


def totals(data: dict) -> tuple[int, int]:
    atoms = sum(len(ch["entries"]) for ch in chapters(data))
    variants = sum(count_variants(ch) for ch in chapters(data))
    return atoms, variants


def replace_block(text: str, block: str, label: str) -> str:
    i, j = text.find(BEGIN), text.find(END)
    if i == -1 or j == -1:
        raise SystemExit(f"{label}: missing {BEGIN} / {END} markers")
    return f"{text[:i]}{BEGIN}\n{block.strip()}\n{END}{text[j + len(END):]}"


def chapter_template(ch: dict) -> str:
    return f"""# {ch['id']} · {ch['title']}

> {ch['tagline']}

{ch['purpose']}

{BEGIN}
{END}

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
"""


def render(verbose: bool = True) -> None:
    data = load()
    for ch in chapters(data):
        d = chapter_dir(ch)
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.md"
        block = chapter_block(ch)
        if readme.is_file():
            readme.write_text(replace_block(readme.read_text(), block, ch["dir"]))
        else:
            readme.write_text(replace_block(chapter_template(ch), block, ch["dir"]))
        if verbose:
            print(f"  {ch['dir']}/README.md")

    root = ROOT / "README.md"
    if root.is_file():
        root.write_text(replace_block(root.read_text(), root_block(data), "README.md"))
        if verbose:
            print("  README.md")

    if verbose:
        atoms, variants = totals(data)
        print(f"\n{len(chapters(data))} chapters · {atoms} atoms · {variants} variants")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def math_problems() -> list[str]:
    r"""Every Markdown expression must survive GitHub's MathJax subset.

    GitHub replaces the whole formula with "The following macros are not
    allowed: …" when it meets a macro it refuses, so a single `\operatorname`
    silently destroys a derivation on the rendered page.
    """
    tools = str(Path(__file__).resolve().parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from mdmath import markdown_files, scan

    out: list[str] = []
    for path in markdown_files():
        for lineno, name in scan(path.read_text()):
            out.append(
                f"{path.relative_to(ROOT)}:{lineno}: \\{name} does not render on "
                f"GitHub (run `python tools/mdmath.py fix`)"
            )
    return out


def check() -> int:
    data = load()
    problems: list[str] = []
    seen_dirs: set[str] = set()
    seen_ids: set[str] = set()

    required = {"name", "file", "tier", "blurb", "variants"}

    for ch in chapters(data):
        if ch["dir"] in seen_dirs:
            problems.append(f"duplicate chapter dir: {ch['dir']}")
        seen_dirs.add(ch["dir"])
        if ch["id"] in seen_ids:
            problems.append(f"duplicate chapter id: {ch['id']}")
        seen_ids.add(ch["id"])

        if not chapter_dir(ch).is_dir():
            problems.append(f"{ch['dir']}: directory missing (run `render`)")

        names: set[str] = set()
        files: set[str] = set()
        for e in ch["entries"]:
            missing = required - e.keys()
            if missing:
                problems.append(f"{ch['dir']}/{e.get('file', '?')}: missing keys {sorted(missing)}")
                continue
            if not e["file"].endswith(".py"):
                problems.append(f"{ch['dir']}/{e['file']}: not a .py file")
            if "/" in e["file"]:
                problems.append(f"{ch['dir']}/{e['file']}: nested path")
            if e["file"] in files:
                problems.append(f"{ch['dir']}: duplicate file {e['file']}")
            if e["name"] in names:
                problems.append(f"{ch['dir']}: duplicate atom name {e['name']}")
            if e["tier"] not in (1, 2, 3):
                problems.append(f"{ch['dir']}/{e['file']}: bad tier {e['tier']}")
            if not e["variants"]:
                problems.append(f"{ch['dir']}/{e['file']}: no variants")
            if len(set(e["variants"])) != len(e["variants"]):
                problems.append(f"{ch['dir']}/{e['file']}: duplicate variants")
            for v in e["variants"]:
                if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", v):
                    problems.append(f"{ch['dir']}/{e['file']}: variant not snake_case: {v!r}")
            files.add(e["file"])
            names.add(e["name"])

            py = chapter_dir(ch) / e["file"]
            md = chapter_dir(ch) / (e["file"][:-3] + ".md")
            if py.is_file() != md.is_file():
                problems.append(
                    f"{ch['dir']}/{e['file']}: atom needs both {e['file']} and "
                    f"{md.name} (found py={py.is_file()}, md={md.is_file()})"
                )

    problems.extend(math_problems())

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1

    atoms, variants = totals(data)
    print(f"ok · {len(chapters(data))} chapters · {atoms} atoms · {variants} variants")
    return 0


def stats() -> int:
    data = load()
    atoms, variants = totals(data)
    print(f"chapters {len(chapters(data))}\natoms    {atoms}\nvariants {variants}")
    for tier in (1, 2, 3):
        n = sum(1 for _, e in all_entries(data) if e["tier"] == tier)
        v = sum(len(e["variants"]) for _, e in all_entries(data) if e["tier"] == tier)
        print(f"tier {tier}   {n:>4} atoms  {v:>5} variants")
    done = sum(1 for ch, e in all_entries(data) if is_implemented(ch, e))
    print(f"done     {done:>4} atoms")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "render"
    if cmd == "render":
        render()
        return 0
    if cmd == "check":
        return check()
    if cmd == "stats":
        return stats()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
