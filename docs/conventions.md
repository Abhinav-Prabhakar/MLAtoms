# Conventions

The rules below exist so that 250 atoms written over months still read like one
person wrote them. They are enforced by review, and partly by
`uv run python tools/catalog.py check`.

## Atoms and variants

An **atom** is an algorithm family. It is one Python file and one Markdown file
with the same stem:

```
04-trees-and-ensembles/
├── cart.py          # implementation
├── cart.md          # intuition, derivation, variants, complexity, failures
├── xgboost.py
└── xgboost.md
```

A **variant** is a named specialisation of the atom — a different loss, a
different regulariser, a different approximation. Every variant listed in
`catalog.json` is:

1. implemented in the atom's `.py`, selected by a `variant=` argument,
2. documented in a row of the atom's `.md` variant table,
3. covered by at least one assertion in the atom's demo.

Variants share a mathematical core. If two variants share nothing, they are two
atoms. This is the difference between `attention:mqa` and a separate file, and
between `svm` and `kernel_ridge`.

```
atom            variant                 how it is addressed
attention   →   multi_head, mqa, gqa →  Attention(variant="gqa")
```

That keeps one file per algorithm while still covering every published
specialisation, and keeps the variant space greppable: `grep -r 'variant="mqa"'`.

## The Python file

```python
"""Atom name — one line on what it does.

Chapter:   04-trees-and-ensembles
Companion: cart.md
Tier 1 · 7 variants: gini, entropy, ...
"""
```

- **Dependencies:** `numpy` and the standard library. `scipy` is allowed only
  where re-implementing it would obscure the algorithm (special functions,
  optimisation). Never `sklearn`, `torch`, `pandas`.
- **Size:** aim for 80–300 lines. Over 400 means the atom is a framework; split
  it or cut the demo.
- **Determinism:** every stochastic component takes `seed` or `rng`. A demo must
  produce identical output twice in a row.
- **Shape contracts:** every public function states input and output shapes in
  its docstring. Shape bugs are the most common bug in from-scratch code.
- **Class or function?** A class with `fit`/`predict` when the atom learns
  parameters; plain functions when it is a transform (`dtw`, `fft_features`) or
  a one-shot solver (`ols`). Don't wrap stateless maths in a class.
- **Numerics:** use `np.linalg` and `np.logaddexp` rather than hand-rolled
  inverses and `log(sum(exp(x)))`. Say so in the `.md` when the naive form is
  the educational point.
- **Failures:** raise on the input you know breaks the method (singular matrix,
  `n < d` without regularisation, empty cluster) with a message that says what
  to do instead.

## The Markdown file

Sections, in order, each as long as it needs to be and no longer:

| Section | Must answer |
|:--|:--|
| Intuition | What is the one idea? Why does it work? |
| Variants | What changes per variant, and when to pick it. |
| Derivation | The objective and the update, enough to re-derive the code. |
| Code walkthrough | How the file is actually built: the real snippets, in reading order, and the one non-obvious line in each. |
| Complexity | Time/memory in `n, d, k, T`; what dominates. |
| Failure modes | Three ways it breaks and the diagnostic for each. |
| Applications | Where it is the right tool; where it is a trap. |
| References | Primary source, plus the one survey worth reading. |

The **Code walkthrough** is not a summary of the derivation — it is the map from
mathematics to the lines on disk. Quote the actual code (trimmed with `...` where
a block is long), name the entry point, and say which line is the one that would
be wrong in a naive implementation. A reader should be able to open the `.py` and
find each snippet verbatim.

Rules of style:

- **Lead with the result.** First sentence of a section is the claim, not the
  preamble.
- **One equation beats three paragraphs.** Number the equations you reference
  later.
- **No filler.** Delete "it is important to note", "in today's world",
  "basically". If a sentence survives deletion, delete it.
- **Concrete numbers.** "O(n²d)" not "fast"; "needs n > 10·d" not "needs enough
  data".
- **Say the failure out loud.** The value of this repository is the honest
  section, not the derivation everyone already has.

## Math

GitHub renders `$…$` and `$$…$$` with a sandboxed MathJax: any macro that it
defines internally with `\newcommand` is refused, and the whole formula is
replaced by "The following macros are not allowed: …". `\operatorname` is the
trap that bites — it is a `\newcommand` in disguise, so an entire derivation
disappears behind a red box.

Rules:

- **`\mathrm{name}` for multi-letter operators**, never `\operatorname`:
  `\mathrm{Var}`, `\mathrm{diag}`, `\mathrm{softmax}`, `\mathrm{rank}`.
- **No definition or reference macros**: `\newcommand`, `\renewcommand`, `\def`,
  `\DeclareMathOperator`, `\label`, `\tag`, `\ref`, `\eqref`. Number equations
  by hand when the text refers back to them.
- **No HTML hooks**: `\href`, `\class`, `\style`, `\unicode`, `\enclose`.
- **Blackboard bold is spelled out**: `\mathbb{R}`, not `\R`.
- **Environments**: `cases`, `aligned`, `array`, `matrix` and friends inside a
  `$$` block. Never `align`/`equation`, which GitHub does not put in display
  mode for you.
- A forbidden macro may be *discussed* as long as it is quoted in backticks or a
  fenced block — the linter skips code.

`uv run python tools/mdmath.py check` lints every `.md`; `… fix` rewrites the
mechanical cases. `tools/catalog.py check` runs the same scan, so CI fails on a
macro that would not render.

## The demo

Every `.py` ends with `if __name__ == "__main__": main()`. The demo:

1. generates small synthetic data with a known answer,
2. fits the atom,
3. prints something that would look wrong if the implementation were broken,
4. asserts the variants agree where theory says they must.

```
uv run python 04-trees-and-ensembles/cart.py
```

No plotting in the demo. Plots belong in `figures/` and are generated by
`uv run python figures/make.py`.

## Naming

- Chapter folders: `NN-kebab-case`, zero-padded, unchanged once published.
- Atom files: `snake_case.py`. The stem is the import name and the doc stem.
- Public classes: `PascalCase`. Public functions: `snake_case`, verb-first.
- Variant strings: `snake_case`, stable forever — they are the public API of
  the atom.

## Adding an atom

```bash
# 1. add the entry to catalog.json (name, file, tier, blurb, variants)
# 2. scaffold the two files from the templates
uv run python tools/scaffold.py 04-trees-and-ensembles hist_gbdt
# 3. write, run the demo, then
uv run python tools/catalog.py check
uv run pytest
```

`scaffold` fills the header, the variant table, and re-renders the chapter
README. It never overwrites an existing file, so it is safe to re-run.
