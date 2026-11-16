#!/usr/bin/env python3
"""Create the two files for one atom: `name.py` and `name.md`.

Usage:
    uv run python tools/scaffold.py 04-trees-and-ensembles cart
    uv run python tools/scaffold.py cart --dry-run

The atom is looked up in catalog.json by file stem or by name; the templates are
filled in and written, then the chapter README is re-rendered. Existing files are
never overwritten.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import catalog  # noqa: E402


def find(query: str) -> tuple[dict, dict]:
    data = catalog.load()
    q = query.lower().removesuffix(".py")
    hits = [
        (ch, e)
        for ch, e in catalog.all_entries(data)
        if e["file"][:-3] == q or e["name"].lower() == q or q in ch["dir"]
    ]
    if len(hits) != 1:
        names = "\n".join(f"  {ch['dir']}/{e['file']}" for ch, e in hits) or "  (none)"
        raise SystemExit(f"{query!r} matched {len(hits)} atoms:\n{names}")
    return hits[0]


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    dry = "--dry-run" in argv
    if not args:
        print(__doc__)
        return 2

    ch, entry = find(args[-1])
    d = catalog.chapter_dir(ch)
    py, md = d / entry["file"], d / (entry["file"][:-3] + ".md")

    for path in (py, md):
        if path.exists():
            print(f"exists, leaving alone: {path.relative_to(ROOT)}")

    variants = ", ".join(entry["variants"])
    header = (
        f'"""{entry["name"]} — {entry["blurb"]}\n\n'
        f"Chapter:  {ch['dir']}\n"
        f"Companion: {md.name}\n"
        f"Tier {entry['tier']} · {len(entry['variants'])} variants: "
        f"{', '.join(entry['variants'])}\n"
        f'"""'
    )
    if not dry and not py.exists():
        body = (ROOT / "templates" / "algorithm.py").read_text()
        body = body.split('"""', 2)[2].lstrip("\n")
        py.write_text(header + "\n\n" + body)
        print(f"wrote {py.relative_to(ROOT)}")

    if not dry and not md.exists():
        body = (ROOT / "templates" / "atom.md").read_text()
        body = body.replace(
            "# Atom name",
            f"# {entry['name']}",
        ).replace(
            "`file.py` · chapter · tier · variants",
            f"`{entry['file']}` · {ch['title']} · Tier {entry['tier']} · {len(entry['variants'])} variants",
        )
        rows = "\n".join(f"| `{v}` | | |" for v in entry["variants"])
        body = body.replace("| `variant` | | |", rows)
        md.write_text(body)
        print(f"wrote {md.relative_to(ROOT)}")

    if not dry:
        subprocess.run([sys.executable, str(ROOT / "tools" / "catalog.py"), "render"], check=True)
    print(f"\nvariants: {variants}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
