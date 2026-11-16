"""Invariants of the catalog. These run in CI and before every release."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "catalog.json").read_text())
CHAPTERS = CATALOG["chapters"]


def test_chapter_ids_and_dirs_are_unique():
    ids = [ch["id"] for ch in CHAPTERS]
    dirs = [ch["dir"] for ch in CHAPTERS]
    assert len(ids) == len(set(ids))
    assert len(dirs) == len(set(dirs))


def test_chapter_dirs_exist_and_are_numbered():
    for ch in CHAPTERS:
        assert ch["dir"].startswith(ch["id"] + "-")
        assert (ROOT / ch["dir"] / "README.md").is_file()


def test_entries_are_well_formed():
    for ch in CHAPTERS:
        files, names = set(), set()
        for e in ch["entries"]:
            assert set(e) == {"name", "file", "tier", "blurb", "variants"}
            assert e["file"].endswith(".py") and "/" not in e["file"]
            assert e["tier"] in (1, 2, 3)
            assert e["variants"], f"{e['file']} has no variants"
            assert len(set(e["variants"])) == len(e["variants"])
            for v in e["variants"]:
                assert re.fullmatch(r"[a-z0-9][a-z0-9_]*", v), v
            assert e["file"] not in files, f"duplicate {e['file']} in {ch['dir']}"
            assert e["name"] not in names, f"duplicate {e['name']} in {ch['dir']}"
            files.add(e["file"])
            names.add(e["name"])


def test_atoms_have_both_files_or_neither():
    for ch in CHAPTERS:
        for e in ch["entries"]:
            py = ROOT / ch["dir"] / e["file"]
            md = ROOT / ch["dir"] / (e["file"][:-3] + ".md")
            assert py.is_file() == md.is_file(), f"{py.name}: py={py.is_file()} md={md.is_file()}"


def test_catalog_is_not_empty():
    atoms = sum(len(ch["entries"]) for ch in CHAPTERS)
    variants = sum(len(e["variants"]) for ch in CHAPTERS for e in ch["entries"])
    assert atoms > 200
    assert variants > 1000
