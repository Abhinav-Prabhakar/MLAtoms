# The website

The repository is the source of truth; the site is a projection of it.

`tools/site.py` reads every `README.md`, every atom companion, every file under
`docs/` and every implemented `.py`, and writes a static site into `docs/` with
no framework, no client-side routing and no database. The Markdown sources stay
where they are, so the site and the repository share one tree. One Markdown file
in, one HTML file out, at the same path relative to the site root:

```
05-probabilistic/mcmc.md       →  05-probabilistic/mcmc.html
05-probabilistic/mcmc.py       →  05-probabilistic/mcmc.py.html   (highlighted listing)
05-probabilistic/README.md     →  05-probabilistic/index.html
README.md                      →  index.html
```

The atom's own `.py` is also copied beside its listing, so the "Raw file" link
serves the exact thing that is in the repository.

## Why not a docs framework

MkDocs, Docusaurus and VitePress all solve a different problem: a hand-written
site with a navigation tree. Here the navigation *is* `catalog.json`, the pages
*are* the Markdown, and the only templating needed is a page shell. A generator
under 700 lines has no plugin churn, no Node toolchain, and `check` can prove
the whole site in one pass.

## Commands

```bash
uv run python tools/site.py build    # write docs/
uv run python tools/site.py check    # build, then fail on any broken link
uv run python tools/site.py assets   # re-render only the icons and the OG card
python3 -m http.server 8765 -d docs  # preview at localhost:8765
```

`build` empties the output tree first, keeping the top-level `docs/*.md`
sources, which are the input as well as part of the published tree.

`check` is the contract, and CI runs it on every push:

- every page has a title, description, canonical URL, Open Graph and Twitter
  card, structured data and the stylesheet;
- no link points at a `.md` file, and no Markdown reaches the output;
- every local `href` and `src` resolves on disk;
- the search index and the sitemap list exactly the pages that exist;
- every implemented atom has both a companion page and a source listing.

An atom that is in the catalog but not yet implemented is *not* a broken link:
its row is written as a muted `<span>` labelled "not implemented yet". That is
why the chapter indexes can list the whole plan while the site stays green.

## What the pages carry

| Piece | Source | Notes |
|:--|:--|:--|
| Home | `README.md` | the centred title block is replaced by the hero |
| Chapter | `<chapter>/README.md` | the generated atom table links only what exists |
| Atom | `<stem>.md` | its own subtitle line is folded into the page meta bar |
| Source | `<stem>.py` | Pygments listing, line count, raw copy button |
| Documents | `docs/*.md` | index generated from the files |
| Search | all of the above | `search-index.json`, titles weighted over body |

Math is rendered by MathJax from the same `$…$` and `$$…$$` the repository uses,
so a formula that renders on GitHub renders here. The `mdmath` check in
`tools/catalog.py` runs first and refuses `\operatorname`, which GitHub's
sanitizer will not render.

## Assets

Everything in `docs/assets/` comes from `tools/site_assets/` (stylesheet,
script, favicon) or is drawn at build time by Pillow: the ICO, the Apple touch
icon and the 1200×630 social card, whose figures are read from the catalog so it
can never disagree with the site.

## Design

Warm paper in light mode, near-black in dark, one copper accent
(`#a8481f` / `#ff8f63`), serif headings against a system sans body, and no
gradients. Dark is the default when the operating system asks for it. The theme
toggle writes `localStorage`, and a tiny inline script applies it before first
paint so the page never flashes.

## Deployment

GitHub Pages serves `main` / `docs` directly: no workflow, no deploy branch and
no build step outside this repository. `uv run python tools/site.py check` must
pass before the `docs/` tree is committed; `.nojekyll` keeps Pages from running
the output through Jekyll. The site is served from
<https://abhinav-prabhakar.github.io/MLAtoms/>.
