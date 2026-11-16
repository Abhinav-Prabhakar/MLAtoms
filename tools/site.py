#!/usr/bin/env python3
"""Static site generator for MLAtoms.

Every Markdown file in the repository becomes a page: the README as the home
page, each chapter index, each atom companion and each document under `docs/`.
Every implemented atom also gets a highlighted source listing.

Nothing here is a framework. `build` writes a self-contained `_site/` with
per-page metadata, Open Graph cards, a favicon, a search index, a sitemap and a
404 page; `check` builds and then proves every internal link resolves.

Usage:
    uv run python tools/site.py build
    uv run python tools/site.py check
    uv run python tools/site.py assets
"""

from __future__ import annotations

import html
import json
import posixpath
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from pygments import format as pygments_format
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.util import ClassNotFound

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tools" / "site_assets"
OUT = ROOT / "_site"
CATALOG = ROOT / "catalog.json"

SITE = {
    "name": "MLAtoms",
    "url": "https://abhinav-prabhakar.github.io/MLAtoms/",
    "repo": "https://github.com/Abhinav-Prabhakar/MLAtoms",
    "tagline": "The periodic table of machine learning.",
    "description": (
        "One algorithm per file, every variant of it, and a derivation that fits "
        "on a page. A reference implementation library of machine learning atoms."
    ),
    "version": "0.1.0",
    "license": "MIT",
}

MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _math(src: str, opts: dict) -> str:
    return f"\\[{src}\\]" if opts["display_mode"] else f"\\({src}\\)"


def _highlight(code: str, lang: str, _attrs: str) -> str:
    if lang == "mermaid":
        return (
            '<pre class="mermaid-source"><code class="language-mermaid">'
            f"{html.escape(code)}</code></pre>"
        )
    try:
        lexer = get_lexer_by_name(lang, stripnl=False) if lang else None
    except ClassNotFound:
        lexer = None
    if lexer is None:
        return f'<pre class="highlight"><code>{html.escape(code)}</code></pre>'
    body = pygments_format(lexer.get_tokens(code), HtmlFormatter(nowrap=True))
    return f'<pre class="highlight"><code>{body}</code></pre>'


def make_markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True, "highlight": _highlight})
    md.enable(["table", "strikethrough"])
    md.use(dollarmath_plugin, renderer=_math, allow_space=False)
    md.use(footnote_plugin)
    md.use(tasklists_plugin, enabled=True)
    md.use(anchors_plugin, min_level=2, max_level=3, permalink=True, permalinkSymbol="#")
    return md


MD = make_markdown()


def plain(text: str) -> str:
    """Inline Markdown to text, good enough for titles, metadata and search."""
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__|\*|_)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_tags(body: str) -> str:
    body = re.sub(r"<(script|style|pre|code)[^>]*>.*?</\1>", " ", body, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", html.unescape(body)).strip()


HREF = re.compile(r'(href|src)="([^"]+)"')
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:|^//|^#", re.I)


def map_link(url: str) -> str:
    """Source-tree path to published path: `.md` becomes HTML, `.py` a listing."""
    if SCHEME.match(url) or url.startswith("/"):
        return url
    path, _, frag = url.partition("#")
    if not path:
        return url
    if path.endswith(".md"):
        head, _, name = path.rpartition("/")
        name = "index.html" if name == "README.md" else name[:-3] + ".html"
        path = f"{head}/{name}" if head else name
    elif path.endswith(".py"):
        path += ".html"
    return f"{path}#{frag}" if frag else path


def rewrite_links(body: str) -> str:
    def repl(match: re.Match[str]) -> str:
        attr, url = match.group(1), html.unescape(match.group(2))
        return f'{attr}="{html.escape(map_link(url), quote=True)}"'

    return HREF.sub(repl, body)


def render(text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Render Markdown and collect h2/h3 headings for the table of contents."""
    env: dict = {}
    tokens = MD.parse(text, env)
    headings: list[tuple[int, str, str]] = []
    level = 0
    anchor: str | None = None
    for token in tokens:
        if token.type == "heading_open":
            level, anchor = int(token.tag[1:]), token.attrGet("id")
        elif token.type == "inline" and anchor is not None:
            if level in (2, 3):
                headings.append((level, plain(token.content), anchor))
            anchor = None
    return rewrite_links(MD.renderer.render(tokens, MD.options, env)), headings


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class Page:
    kind: str  # home | chapter | atom | doc | docs-index | source | 404
    out: str
    title: str
    description: str
    src: Path | None = None
    body: str = ""
    headings: list[tuple[int, str, str]] = field(default_factory=list)
    chapter: dict | None = None
    entry: dict | None = None
    series: str = ""
    search: str = ""
    summary: str = ""
    base: str | None = None

    @property
    def url(self) -> str:
        return SITE["url"] + self.out

    @property
    def root(self) -> str:
        return self.base if self.base is not None else "../" * self.out.count("/")

    def href(self, other: Page) -> str:
        return posixpath.relpath(other.out, posixpath.dirname(self.out) or ".")


def link_from(page: Page, out: str) -> str:
    """Relative href from `page` to the output path `out`."""
    return html.escape(posixpath.relpath(out, posixpath.dirname(page.out) or "."), quote=True)


def current_attr(target: Page, page: Page) -> str:
    return ' aria-current="page"' if target.out == page.out else ""


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text())


def implemented(chapter: dict, entry: dict) -> bool:
    d = ROOT / chapter["dir"]
    return (d / entry["file"]).is_file() and (d / f"{Path(entry['file']).stem}.md").is_file()


def first_paragraph(text: str) -> str:
    for block in text.split("\n\n"):
        candidate = plain(block.strip())
        if len(candidate) < 50 or candidate.startswith("#"):
            continue
        if re.match(r"^\S+\.py\b", candidate) or "Tier " in candidate[:70]:
            continue
        if candidate.startswith("|") or candidate.startswith(">"):
            continue
        return candidate
    return SITE["description"]


def clip(text: str, limit: int = 158) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    cut = cut[: cut.rfind(" ")] if " " in cut else cut
    return cut.rstrip(" ,.;:") + "…"


def collect() -> tuple[list[Page], list[Page]]:
    cat = load_catalog()
    pages: list[Page] = []
    sources: list[Page] = []

    home = Page(
        kind="home",
        out="index.html",
        title="MLAtoms — the periodic table of machine learning",
        description=SITE["description"],
        src=ROOT / "README.md",
    )
    body, home.headings = render((ROOT / "README.md").read_text())
    body = re.sub(r'\s*<div align="center">.*?</div>\s*', "", body, count=1, flags=re.S)
    home.body = re.sub(r"^\s*<hr\s*/?>\s*", "", body, count=1)
    pages.append(home)

    for chapter in cat["chapters"]:
        cdir = ROOT / chapter["dir"]
        readme = cdir / "README.md"
        if not readme.is_file():
            continue
        body, heads = render(readme.read_text())
        pages.append(
            Page(
                kind="chapter",
                out=f"{chapter['dir']}/index.html",
                title=f"{chapter['id']} · {chapter['title']}",
                description=clip(chapter["tagline"]),
                src=readme,
                body=body,
                headings=heads,
                chapter=chapter,
                series="main",
            )
        )
        for entry in chapter["entries"]:
            if not implemented(chapter, entry):
                continue
            stem = Path(entry["file"]).stem
            doc = cdir / f"{stem}.md"
            text = doc.read_text()
            body, heads = render(text)
            # the doc's own subtitle line duplicates the page meta bar
            body = re.sub(
                rf"<p><code>{re.escape(stem)}\.py</code>.*?</p>\s*", "", body, count=1, flags=re.S
            )
            summary = f"{entry['blurb']} {first_paragraph(text)}"
            pages.append(
                Page(
                    kind="atom",
                    out=f"{chapter['dir']}/{stem}.html",
                    title=entry["name"],
                    description=clip(summary),
                    summary=summary,
                    src=doc,
                    body=body,
                    headings=heads,
                    chapter=chapter,
                    entry=entry,
                    series="main",
                    search=" ".join(entry["variants"]),
                )
            )
            sources.append(
                Page(
                    kind="source",
                    out=f"{chapter['dir']}/{entry['file']}.html",
                    title=f"{entry['name']} — {entry['file']}",
                    description=clip(f"Source of {entry['file']}: {entry['blurb']}"),
                    src=cdir / entry["file"],
                    chapter=chapter,
                    entry=entry,
                )
            )

    for doc in sorted((ROOT / "docs").glob("*.md")):
        if doc.name == "README.md":
            continue
        text = doc.read_text()
        body, heads = render(text)
        title = plain(re.search(r"^#\s+(.+)$", text, re.M).group(1))
        pages.append(
            Page(
                kind="doc",
                out=f"docs/{doc.stem}.html",
                title=title,
                description=clip(first_paragraph(text)),
                summary=first_paragraph(text),
                src=doc,
                body=body,
                headings=heads,
                series="docs",
            )
        )

    pages.append(
        Page(
            kind="docs-index",
            out="docs/index.html",
            title="Documents",
            description="Conventions, tiers, recipes, notation and the roadmap behind MLAtoms.",
            series="docs",
        )
    )
    return pages, sources


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    return html.escape(text, quote=True)


ICON_SUN = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/>'
    '<path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.2 5.2l1.6 1.6'
    'M17.2 17.2l1.6 1.6M18.8 5.2l-1.6 1.6M6.8 17.2l-1.6 1.6"/></svg>'
)
ICON_SEARCH = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.4"/>'
    '<path d="m16 16 4.2 4.2"/></svg>'
)
ICON_GITHUB = (
    '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">'
    '<path d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38v-1.34c-2.23.48-2.7-1.07-2.7-1.07'
    '-.36-.93-.89-1.18-.89-1.18-.73-.5.06-.49.06-.49.8.06 1.23.83 1.23.83.72 1.22 1.87.87 2.33.66'
    '.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12'
    ' 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12'
    '.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38'
    'A8 8 0 0 0 8 0Z"/></svg>'
)


def head(page: Page, og_type: str, extra: str = "") -> str:
    root = page.root
    desc = esc(page.description)
    canonical = esc(page.url)
    title = esc(page.title if page.kind == "home" else f"{page.title} · MLAtoms")
    og_image = esc(SITE["url"] + "assets/og.png")
    jsonld = {
        "home": {"@type": "WebSite", "name": SITE["name"], "url": SITE["url"]},
        "chapter": {"@type": "CollectionPage", "name": page.title},
        "atom": {"@type": "TechArticle", "headline": page.title},
        "doc": {"@type": "TechArticle", "headline": page.title},
        "docs-index": {"@type": "CollectionPage", "name": page.title},
        "source": {"@type": "SoftwareSourceCode", "name": page.title, "programmingLanguage": "Python"},
        "404": {"@type": "WebPage", "name": page.title},
    }.get(page.kind, {"@type": "WebPage", "name": page.title})
    jsonld.update({"@context": "https://schema.org", "url": page.url, "description": page.description})
    if page.src is not None:
        jsonld["license"] = f"{SITE['repo']}/blob/main/LICENSE"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="author" content="Abhinav Prabhakar">
<meta name="generator" content="tools/site.py">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#fbfaf7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0d0e10" media="(prefers-color-scheme: dark)">
<meta name="robots" content="index, follow, max-image-preview:large">
<link rel="canonical" href="{canonical}">
<link rel="icon" href="{root}assets/favicon.svg" type="image/svg+xml">
<link rel="icon" href="{root}assets/favicon.ico" sizes="48x48">
<link rel="apple-touch-icon" href="{root}assets/apple-touch-icon.png">
<link rel="manifest" href="{root}site.webmanifest">
<link rel="sitemap" href="{root}sitemap.xml" type="application/xml">
<meta property="og:type" content="{og_type}">
<meta property="og:site_name" content="MLAtoms">
<meta property="og:title" content="{esc(page.title)}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="{og_image}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="MLAtoms — the periodic table of machine learning">
<meta property="og:locale" content="en_US">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{esc(page.title)}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{og_image}">
<meta name="twitter:image:alt" content="The MLAtoms wordmark and tagline">
<script type="application/ld+json">{json.dumps(jsonld)}</script>
<link rel="stylesheet" href="{root}assets/style.css">
<script>
  try {{
    var t = localStorage.getItem("mlatoms-theme");
    if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  }} catch (e) {{}}
</script>
{extra}</head>"""


def header(page: Page) -> str:
    root = page.root
    at_chapters = ' aria-current="page"' if page.kind in ("chapter", "atom", "source", "home") else ""
    at_docs = ' aria-current="page"' if page.kind in ("doc", "docs-index") else ""
    nav = "".join(
        [
            f'<a href="{root}index.html#contents"{at_chapters}>Chapters</a>',
            f'<a href="{root}docs/index.html"{at_docs}>Docs</a>',
            f'<a href="{SITE["repo"]}">GitHub</a>',
        ]
    )
    return f"""<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
  <div class="site-header__inner">
    <a class="brand" href="{root}index.html">
      <img src="{root}assets/favicon.svg" alt="" width="22" height="22">
      <span>MLAtoms</span>
    </a>
    <nav class="site-nav" aria-label="Primary">{nav}</nav>
    <button class="icon-btn" type="button" data-search-open aria-label="Search">
      <span aria-hidden="true">{ICON_SEARCH}</span>
      <span class="icon-btn__label">Search</span>
      <kbd class="search-key">/</kbd>
    </button>
    <button class="icon-btn" type="button" data-theme-toggle aria-label="Switch theme">
      <span aria-hidden="true">{ICON_SUN}</span>
    </button>
  </div>
</header>
<dialog class="search" aria-label="Search MLAtoms">
  <form class="search__form" method="dialog" role="search">
    {ICON_SEARCH}
    <input type="search" name="q" placeholder="Search atoms, variants, chapters…" autocomplete="off" spellcheck="false" aria-label="Search query">
    <kbd class="search-key">esc</kbd>
  </form>
  <ul class="search__results"></ul>
  <p class="search__status search__hint">Type to search atoms, variants, chapters and docs.</p>
</dialog>"""


def rail(page: Page, pages: list[Page]) -> str:
    cat = load_catalog()
    parts: list[str] = []

    if page.chapter is not None:
        chapter = page.chapter
        items = []
        for entry in chapter["entries"]:
            here = page.entry is not None and entry["file"] == page.entry["file"]
            current = ' aria-current="page"' if here else ""
            label = f'<span>{esc(entry["name"])}</span>'
            if implemented(chapter, entry):
                stem = Path(entry["file"]).stem
                href = link_from(page, f"{chapter['dir']}/{stem}.html")
                items.append(f'<li><a href="{href}"{current}><span class="tick">✓</span>{label}</a></li>')
            else:
                items.append(
                    f'<li><span class="muted" style="display:flex;gap:.4rem;padding:.22rem 0;'
                    f'font-size:.875rem;opacity:.72"><span class="dot">·</span>{label}</span></li>'
                )
        parts.append(
            f'<p class="rail-title">{esc(chapter["id"])} · {esc(chapter["title"])}</p>'
            f'<ul class="rail-list">{"".join(items)}</ul>'
        )
        idx = next(i for i, c in enumerate(cat["chapters"]) if c["dir"] == chapter["dir"])
        jumps = []
        if idx > 0:
            prev = cat["chapters"][idx - 1]
            jumps.append(
                f'<li><a href="{link_from(page, prev["dir"] + "/index.html")}">'
                f'<span class="jump">← {esc(prev["id"])} · {esc(prev["title"])}</span></a></li>'
            )
        if idx + 1 < len(cat["chapters"]):
            nxt = cat["chapters"][idx + 1]
            jumps.append(
                f'<li><a href="{link_from(page, nxt["dir"] + "/index.html")}">'
                f'<span class="jump">{esc(nxt["id"])} · {esc(nxt["title"])}&nbsp;→</span></a></li>'
            )
        if jumps:
            parts.append(
                f'<div class="rail-sub"><p class="rail-title">Chapters</p>'
                f'<ul class="rail-list">{"".join(jumps)}</ul></div>'
            )
    else:
        chapters = "".join(
            f'<li><a href="{link_from(page, c["dir"] + "/index.html")}">'
            f'<span class="num">{esc(c["id"])}</span><span>{esc(c["title"])}</span></a></li>'
            for c in cat["chapters"]
        )
        parts.append(f'<p class="rail-title">Chapters</p><ul class="rail-list">{chapters}</ul>')

        doc_pages = [p for p in pages if p.kind in ("doc", "docs-index")]
        doc_pages.sort(key=lambda p: (p.kind == "docs-index", p.title))
        docs = "".join(
            f'<li><a href="{link_from(page, p.out)}"{current_attr(p, page)}>'
            f'<span class="dot">·</span><span>{esc(p.title)}</span></a></li>'
            for p in doc_pages
        )
        parts.append(
            f'<div class="rail-sub"><p class="rail-title">Documents</p>'
            f'<ul class="rail-list">{docs}</ul></div>'
        )

    if page.headings:
        toc = "".join(
            f'<li class="lvl{lvl}"><a href="#{esc(slug)}">{esc(text)}</a></li>'
            for lvl, text, slug in page.headings[:24]
        )
        parts.append(
            f'<div class="rail-sub rail-toc"><p class="rail-title">On this page</p>'
            f'<ul class="rail-list">{toc}</ul></div>'
        )
    return (
        '<aside class="layout__rail" aria-label="Section">'
        f'<div class="rail-inner">{"".join(parts)}</div></aside>'
    )


def page_meta(page: Page) -> str:
    if page.kind == "atom" and page.entry and page.chapter:
        ch, entry = page.chapter, page.entry
        return (
            '<p class="page-meta">'
            f'<a href="index.html">{esc(ch["id"])} · {esc(ch["title"])}</a>'
            '<span class="page-meta__sep">/</span>'
            f'<span class="pill pill--t{entry["tier"]}">tier {entry["tier"]}</span>'
            '<span class="page-meta__sep">/</span>'
            f'<span>{len(entry["variants"])} variants</span>'
            '<span class="page-meta__sep">/</span>'
            f'<a href="{esc(entry["file"])}.html">source</a>'
            "</p>"
        )
    if page.kind == "chapter" and page.chapter:
        ch = page.chapter
        done = sum(1 for e in ch["entries"] if implemented(ch, e))
        return (
            '<p class="page-meta"><a href="../index.html">MLAtoms</a>'
            f'<span class="page-meta__sep">/</span><span>chapter {esc(ch["id"])}</span>'
            f'<span class="page-meta__sep">/</span><span>{done} of '
            f'{len(ch["entries"])} implemented</span></p>'
        )
    if page.kind == "source" and page.entry:
        return (
            f'<p class="page-meta"><a href="./index.html">{esc(page.chapter["id"])} · '
            f'{esc(page.chapter["title"])}</a><span class="page-meta__sep">/</span>'
            f'<a href="{esc(Path(page.entry["file"]).stem)}.html">{esc(page.entry["name"])}</a>'
            '<span class="page-meta__sep">/</span><span>source</span></p>'
        )
    return ""


def hero() -> str:
    cat = load_catalog()
    atoms = sum(len(c["entries"]) for c in cat["chapters"])
    variants = sum(len(e["variants"]) for c in cat["chapters"] for e in c["entries"])
    return f"""<div class="hero">
  <p class="page-meta">v{SITE["version"]} <span class="page-meta__sep">/</span> {SITE["license"]}
    <span class="page-meta__sep">/</span> numpy only</p>
  <h1>{SITE["tagline"]}</h1>
  <p class="lede">One algorithm per file. Every variant of it. A derivation that fits on a page —
  written for the moment you have to implement it yourself.</p>
  <div class="actions">
    <a class="btn btn--primary" href="#contents">Browse the chapters</a>
    <a class="btn" href="docs/recipes.html">Task → atom</a>
    <a class="btn" href="{SITE["repo"]}"><span aria-hidden="true">{ICON_GITHUB}</span> Source</a>
  </div>
  <ul class="stats">
    <li><b>{len(cat["chapters"])}</b><span>chapters</span></li>
    <li><b>{atoms}</b><span>atoms</span></li>
    <li><b>{variants}</b><span>variants</span></li>
  </ul>
</div>"""


def docs_index(page: Page, pages: list[Page]) -> str:
    items = "".join(
        f'<li><a href="{link_from(page, p.out)}"><b>{esc(p.title)}</b>'
        f"<span>{esc(p.summary or p.description)}</span></a></li>"
        for p in pages
        if p.kind == "doc"
    )
    return (
        "<h1>Documents</h1>"
        '<p class="lede">How the repository is built, what it promises, and where it is going.</p>'
        f'<ul class="doc-list">{items}</ul>'
    )


def footer(page: Page) -> str:
    if page.src is not None:
        rel = page.src.relative_to(ROOT).as_posix()
        edit = f'<a href="{SITE["repo"]}/edit/main/{rel}">Edit this page</a>'
    else:
        edit = f'<a href="{SITE["repo"]}">Source</a>'
    return f"""<footer class="site-footer">
  <div class="site-footer__inner">
    <span>MLAtoms · {SITE["license"]} · built from Markdown</span>
    <nav aria-label="Footer">{edit}<a href="{page.root}docs/index.html">Docs</a><a href="{page.root}index.html">Home</a></nav>
  </div>
</footer>"""


def pager(page: Page, pages: list[Page]) -> str:
    if not page.series:
        return ""
    seq = [p for p in pages if p.series == page.series and p.kind != "docs-index"]
    try:
        i = next(k for k, p in enumerate(seq) if p.out == page.out)
    except StopIteration:
        return ""
    prev = seq[i - 1] if i > 0 else None
    nxt = seq[i + 1] if i + 1 < len(seq) else None
    if not prev and not nxt:
        return ""

    def link(target: Page | None, css: str, label: str) -> str:
        if target is None:
            return ""
        return (
            f'<a class="{css}" href="{esc(page.href(target))}"><small>{label}</small>'
            f"<span>{esc(target.title)}</span></a>"
        )

    return (
        f'<nav class="pager" aria-label="Pagination">{link(prev, "prev", "Previous")}'
        f'{link(nxt, "next", "Next")}</nav>'
    )


def render_page(page: Page, pages: list[Page]) -> str:
    root = page.root
    body = page.body
    if page.kind == "home":
        body = hero() + body
    elif page.kind == "docs-index":
        body = docs_index(page, pages)

    extra = ""
    if 'class="math' in body:
        extra = (
            '<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],'
            'displayMath:[["\\\\[","\\\\]"]] ,tags:"none",processEscapes:true},'
            'options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]},'
            'svg:{fontCache:"global"}};</script>\n'
            f'<script defer src="{MATHJAX}"></script>\n'
        )

    article = "prose code-page" if page.kind == "source" else "prose"
    layout = "layout layout--wide" if page.kind == "source" else "layout"
    content = (
        f'<div class="{layout}">'
        f'<div><div class="{article}">{page_meta(page)}{body}</div>{pager(page, pages)}</div>'
        f"{rail(page, pages)}</div>"
    )
    og_type = "website" if page.kind in ("home", "chapter") else "article"

    return f"""{head(page, og_type, extra)}
<body data-root="{root}">
{header(page)}
<main id="main" class="shell">
{content}
</main>
{footer(page)}
<script src="{root}assets/site.js" defer></script>
</body>
</html>
"""


def source_page(page: Page) -> str:
    code = page.src.read_text()
    lexed = pygments_format(PythonLexer(stripnl=False).get_tokens(code), HtmlFormatter(nowrap=True))
    stem = Path(page.entry["file"]).stem
    bar = (
        f'<p class="code-page__bar"><span>{len(code.splitlines())} lines</span>'
        f"<span>·</span><span>{len(page.entry['variants'])} variants</span>"
        f'<a class="btn" href="{esc(page.entry["file"])}">Raw file</a>'
        f'<a class="btn" href="{esc(stem)}.html">Companion</a></p>'
    )
    page.body = (
        f'<h1>{esc(page.entry["file"])}</h1>{bar}'
        f'<pre class="highlight"><code>{lexed}</code></pre>'
    )
    page.headings = []
    return render_page(page, [page])


def render_404(pages: list[Page]) -> str:
    page = Page(
        kind="404",
        out="404.html",
        title="Page not found",
        description="That page is not in the catalog. Try the chapter index, or search everything.",
        base=SITE["url"],
    )
    page.body = f"""<h1>404</h1>
<p class="lede">That page is not in the catalog. It may be an atom that has not landed yet —
the chapter index shows which ones are implemented.</p>
<div class="actions">
  <a class="btn btn--primary" href="{SITE["url"]}index.html">All chapters</a>
  <a class="btn" href="{SITE["url"]}docs/index.html">Documents</a>
</div>
<p class="page-meta" style="margin-top:2rem">Search everything with <kbd class="search-key">/</kbd>
or <kbd class="search-key">⌘K</kbd>.</p>"""
    return render_page(page, pages)


# --------------------------------------------------------------------------- #
# Dead-link stubs
# --------------------------------------------------------------------------- #
def _source_for(target: str) -> Path | None:
    """The repository file a published path was generated from, if any."""
    if target.endswith(".py.html"):
        return ROOT / target[:-5]
    if target.endswith("/"):
        target += "index.html"
    if target.endswith(".html"):
        stem = target[:-5]
        if stem.endswith("/index") or stem == "index":
            return ROOT / (stem[: -len("index")] + "README.md")
        return ROOT / f"{stem}.md"
    return ROOT / target


def stub_dead_links(
    text: str, page_out: str, out: Path, planned: set[str]
) -> tuple[str, int]:
    """Turn links to atoms that have not landed yet into plain, muted text."""
    base = posixpath.dirname(page_out) or "."
    stubbed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal stubbed
        href, inner = match.group(1), match.group(2)
        if SCHEME.match(href) or href.startswith("/") or href.startswith("#"):
            return match.group(0)
        target = posixpath.normpath(posixpath.join(base, href.partition("#")[0]))
        if target in planned or (out / target).exists():
            return match.group(0)
        source = _source_for(href.partition("#")[0])
        if source is not None and source.exists():
            return match.group(0)  # a real generation bug: let `check` report it
        stubbed += 1
        return f'<span class="stub" title="Not implemented yet">{inner}</span>'

    return ANCHOR.sub(repl, text), stubbed


ANCHOR = re.compile(r'<a\b[^>]*?href="([^"]+)"[^>]*>(.*?)</a>', re.S)


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
def _mark(size: int, bg: str = "#0d0e10", fg: str = "#ff8f63", dot: str = "#f6f4ef"):
    from PIL import Image, ImageDraw

    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s * 0.22, fill=bg)
    pad = s * 0.20
    d.ellipse([pad, pad, s - pad, s - pad], outline=fg, width=max(2, int(s * 0.069)))
    radius = s * 0.097
    c = s / 2
    d.ellipse([c - radius, c - radius, c + radius, c + radius], fill=fg)
    ring = c - pad
    ex, ey = c + ring * 0.7071, c - ring * 0.7071
    sr = s * 0.056
    d.ellipse([ex - sr, ey - sr, ex + sr, ey + sr], fill=dot)
    return img.resize((size, size), Image.LANCZOS)


SERIF_FONTS = (
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
)
SANS_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _font(size: int, *, serif: bool = False):
    """A real typeface when the machine has one; Pillow's default otherwise.

    The social card is built in CI, so it prefers fonts every runner has rather
    than whatever the author's laptop happens to ship.
    """
    from PIL import ImageFont

    for candidate in SERIF_FONTS if serif else SANS_FONTS:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def write_assets(out: Path) -> list[str]:
    """Copy the stylesheet and script, draw the raster icons and the OG card."""
    from PIL import Image, ImageDraw

    dest = out / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    for name in ("style.css", "site.js", "favicon.svg"):
        shutil.copy2(ASSETS / name, dest / name)
        made.append(f"assets/{name}")

    _mark(64).save(dest / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    made.append("assets/favicon.ico")
    for size, name in ((180, "apple-touch-icon.png"), (512, "icon-512.png")):
        _mark(size).save(dest / name)
        made.append(f"assets/{name}")

    W, H = 1200, 630
    card = Image.new("RGB", (W, H), "#0d0e10")
    d = ImageDraw.Draw(card)
    for y in range(H):
        t = y / (H - 1)
        d.line([(0, y), (W, y)], fill=(13 + int(6 * t), 14 + int(6 * t), 16 + int(8 * t)))
    mark = _mark(96)
    card.paste(mark, (72, 64), mark)
    big = _font(88, serif=True)
    mid = _font(33)
    small = _font(22)
    d.text((72, 198), "MLAtoms", font=big, fill="#ffffff")
    d.line([(72, 322), (W - 72, 322)], fill="#2a2e34", width=2)
    d.text((72, 356), "One algorithm per file. Every variant of it.", font=mid, fill="#b9bcc4")
    d.text((72, 404), "A derivation that fits on a page.", font=mid, fill="#b9bcc4")
    cat = load_catalog()
    atoms = sum(len(c["entries"]) for c in cat["chapters"])
    variants = sum(len(e["variants"]) for c in cat["chapters"] for e in c["entries"])
    d.text(
        (72, 494),
        f"{len(cat['chapters'])} CHAPTERS  ·  {atoms} ATOMS  ·  {variants} VARIANTS",
        font=small,
        fill="#ff8f63",
    )
    d.text((72, 566), "github.com/Abhinav-Prabhakar/MLAtoms", font=small, fill="#7b7f88")
    card.save(dest / "og.png", optimize=True)
    made.append("assets/og.png")
    return made


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(out: Path = OUT) -> tuple[list[Page], list[Page]]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pages, sources = collect()

    write_assets(out)
    (out / ".nojekyll").write_text("")

    manifest = {
        "name": "MLAtoms",
        "short_name": "MLAtoms",
        "description": SITE["description"],
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "background_color": "#0d0e10",
        "theme_color": "#0d0e10",
        "icons": [
            {"src": "assets/favicon.svg", "sizes": "any", "type": "image/svg+xml"},
            {"src": "assets/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {"src": "assets/apple-touch-icon.png", "sizes": "180x180", "type": "image/png"},
        ],
    }
    (out / "site.webmanifest").write_text(json.dumps(manifest, indent=2) + "\n")

    planned = {p.out for p in pages} | {p.out for p in sources} | {"404.html"}

    index = []
    for page in pages:
        target = out / page.out
        target.parent.mkdir(parents=True, exist_ok=True)
        markup, _ = stub_dead_links(render_page(page, pages), page.out, out, planned)
        target.write_text(markup)
        if page.kind == "atom":
            src_file = ROOT / page.chapter["dir"] / page.entry["file"]
            shutil.copy2(src_file, target.with_name(page.entry["file"]))
        text = strip_tags(page.body) or page.description
        index.append(
            {
                "u": page.out,
                "t": page.title,
                "k": (
                    f"chapter {page.chapter['id']}"
                    if page.kind == "chapter"
                    else f"{page.chapter['id']} · {page.chapter['title']}"
                    if page.chapter
                    else "Documents"
                    if page.kind in ("doc", "docs-index")
                    else "MLAtoms"
                ),
                "d": page.description,
                "b": text[:1400],
                "v": page.search,
                "w": {
                    "atom": 1.0,
                    "chapter": 0.85,
                    "doc": 0.8,
                    "home": 0.7,
                    "docs-index": 0.6,
                }.get(page.kind, 0.6),
            }
        )

    for page in sources:
        target = out / page.out
        target.parent.mkdir(parents=True, exist_ok=True)
        markup, _ = stub_dead_links(source_page(page), page.out, out, planned)
        target.write_text(markup)

    (out / "search-index.json").write_text(json.dumps(index, separators=(",", ":")))
    (out / "404.html").write_text(render_404(pages))

    sitemap = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    priority = {"home": "1.0", "chapter": "0.9", "atom": "0.8", "docs-index": "0.6", "doc": "0.6"}
    for page in pages:
        sitemap.append(
            f"  <url><loc>{esc(page.url)}</loc>"
            f"<priority>{priority.get(page.kind, '0.5')}</priority></url>"
        )
    sitemap.append("</urlset>")
    (out / "sitemap.xml").write_text("\n".join(sitemap) + "\n")

    (out / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {SITE['url']}sitemap.xml\n"
    )
    return pages, sources


# --------------------------------------------------------------------------- #
# Check
# --------------------------------------------------------------------------- #
LINK = re.compile(r'(?:href|src)="([^"]+)"')
DEAD_MD = re.compile(r'href="(?!https?:|//|/|mailto:)([^"]*?\.md)(#[^"]*)?"')
REQUIRED = (
    ("title", "<title>"),
    ("description", 'name="description" content="'),
    ("canonical", 'rel="canonical"'),
    ("og:title", 'property="og:title"'),
    ("og:image", 'property="og:image"'),
    ("twitter card", 'name="twitter:card"'),
    ("structured data", "application/ld+json"),
    ("stylesheet", "assets/style.css"),
)


def where(path: Path) -> str:
    """`_site/` when it is inside the repo, an absolute path otherwise."""
    try:
        return path.relative_to(ROOT).as_posix() + "/"
    except ValueError:
        return str(path)


def check(out: Path = OUT) -> int:
    pages, sources = build(out)
    problems: list[str] = []
    html_files = sorted(out.rglob("*.html"))

    for path in html_files:
        rel = path.relative_to(out).as_posix()
        text = path.read_text()
        for tag, needle in REQUIRED:
            if needle not in text:
                problems.append(f"{rel}: missing {tag}")
        if 'content=""' in text:
            problems.append(f"{rel}: empty metadata")
        if DEAD_MD.search(text):
            problems.append(f"{rel}: unresolved .md link")

        for url in LINK.findall(text):
            url = html.unescape(url)
            if SCHEME.match(url) or url.startswith("/") or url.startswith("data:"):
                continue
            target = url.partition("#")[0].split("?")[0]
            if target and not (path.parent / target).exists():
                problems.append(f"{rel}: broken link → {url}")

    for stray in out.rglob("*.md"):
        problems.append(f"{stray.relative_to(out)}: raw Markdown in the output")

    sitemap = (out / "sitemap.xml").read_text()
    if sitemap.count("<loc>") != len(pages):
        problems.append("sitemap.xml: entry count does not match the page count")
    index = json.loads((out / "search-index.json").read_text())
    if len(index) != len(pages):
        problems.append("search-index.json: entry count does not match the page count")
    if not (out / "assets/og.png").is_file():
        problems.append("assets/og.png: missing social card")

    if problems:
        for problem in problems[:40]:
            print(f"  {problem}", file=sys.stderr)
        print(f"fail · {len(problems)} site problems", file=sys.stderr)
        return 1

    atoms = sum(1 for p in pages if p.kind == "atom")
    print(f"ok · {len(pages)} pages · {atoms} atoms · {atoms} source listings · 0 broken links")
    print(f"     {where(out)} · {len(list(out.rglob('*')))} files")
    return 0


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "build"
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else OUT
    if command == "build":
        pages, sources = build(out)
        print(f"built {len(pages) + len(sources)} pages → {where(out)}")
        return 0
    if command == "check":
        return check(out)
    if command == "assets":
        out.mkdir(parents=True, exist_ok=True)
        made = write_assets(out)
        print("wrote " + ", ".join(made))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
