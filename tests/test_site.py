"""Tests for the static site generator.

`tools/site.py check` already proves links and metadata on the real build; these
tests pin the contract a page has to satisfy, so a change in the generator fails
here before it reaches Pages.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.I)


def load_site() -> ModuleType:
    """Load `tools/site.py` under a private name; `site` is a stdlib module."""
    spec = importlib.util.spec_from_file_location("mlatoms_site", ROOT / "tools" / "site.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their own module
    spec.loader.exec_module(module)
    return module


site = load_site()


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site") / "_site"
    site.build(out)
    return out


def published(out: Path) -> set[str]:
    """Pages listed in the sitemap: everything except listings, 404 and the home."""
    return {
        p.relative_to(out).as_posix()
        for p in out.rglob("*.html")
        if not p.name.endswith(".py.html") and p.name != "404.html"
    }


def test_every_page_has_metadata(built: Path) -> None:
    for page in sorted(built.rglob("*.html")):
        text = page.read_text()
        assert "<title>" in text, page
        assert 'name="description" content="' in text, page
        assert 'rel="canonical"' in text, page
        assert 'property="og:image"' in text, page
        assert 'name="twitter:card"' in text, page
        assert 'property="og:title"' in text, page


def test_no_markdown_leaks_into_the_output(built: Path) -> None:
    assert not list(built.rglob("*.md"))
    for page in built.rglob("*.html"):
        for url in re.findall(r'(?:href|src)="([^"]+)"', page.read_text()):
            if not EXTERNAL.match(url):
                assert not url.partition("#")[0].endswith(".md"), f"{page}: {url}"


def test_internal_links_resolve(built: Path) -> None:
    for page in built.rglob("*.html"):
        for url in re.findall(r'(?:href|src)="([^"]+)"', page.read_text()):
            if EXTERNAL.match(url):
                continue
            target = url.partition("#")[0].split("?")[0]
            if target:
                assert (page.parent / target).exists(), f"{page}: {url}"


def test_search_index_covers_every_page(built: Path) -> None:
    index = json.loads((built / "search-index.json").read_text())
    assert {entry["u"] for entry in index} == published(built)
    assert all(entry["t"] and entry["b"] for entry in index)


def test_sitemap_lists_every_page(built: Path) -> None:
    sitemap = (built / "sitemap.xml").read_text()
    locs = {
        url.replace(site.SITE["url"], "") for url in re.findall(r"<loc>([^<]+)</loc>", sitemap)
    }
    assert locs == published(built)


def test_implemented_atoms_get_a_companion_and_a_listing(built: Path) -> None:
    cat = json.loads((ROOT / "catalog.json").read_text())
    for chapter in cat["chapters"]:
        for entry in chapter["entries"]:
            py = ROOT / chapter["dir"] / entry["file"]
            if not py.is_file():
                continue
            stem = Path(entry["file"]).stem
            companion = built / chapter["dir"] / f"{stem}.html"
            listing = built / chapter["dir"] / f"{entry['file']}.html"
            assert companion.is_file(), companion
            assert listing.is_file(), listing
            assert f"<h1>{entry['file']}</h1>" in listing.read_text()
            assert (built / chapter["dir"] / entry["file"]).is_file()


def test_server_assets_exist(built: Path) -> None:
    for name in (
        "assets/style.css",
        "assets/site.js",
        "assets/favicon.svg",
        "assets/favicon.ico",
        "assets/apple-touch-icon.png",
        "assets/og.png",
        "site.webmanifest",
        "sitemap.xml",
        "robots.txt",
        ".nojekyll",
    ):
        assert (built / name).is_file(), name


def test_check_passes_on_a_fresh_build(tmp_path: Path) -> None:
    assert site.check(tmp_path / "_site") == 0
