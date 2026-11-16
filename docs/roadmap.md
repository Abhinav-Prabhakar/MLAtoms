# Roadmap

The catalog is fixed; the code arrives in phases. Each phase is independently
useful, so the repository never depends on a future phase to make sense.

## Phase 0 — skeleton (done)

- catalog: chapters, atoms, variants, tiers
- tooling: render, check, scaffold
- conventions, tiers, recipes, notation
- templates for the two files of an atom

## Phase 1 — the spine (~60 atoms)

Everything a practitioner needs to ship a tabular model end to end.

| Chapter | Atoms |
|:--|:--|
| 00 primitives | activations, losses, metrics, init, normalization, distances, linalg |
| 01 optimization | sgd, adam, schedules, em |
| 02 linear | ols, ridge, lasso, logistic, glm |
| 04 trees | cart, random_forest, gradient_boosting, xgboost, lightgbm, adaboost |
| 07 clustering | kmeans, gmm, dbscan, hdbscan, spectral, agglomerative |
| 20 evaluation | resampling, bootstrap, calibration, interpretability |

Exit criterion: `docs/recipes.md` rows 1–4 are fully served by implemented atoms.

## Phase 2 — classical breadth (~80 atoms)

Kernels, probabilistic models, latent structure, manifolds, density, sequences,
time series. This is where the repository becomes a reference rather than a
starter kit.

## Phase 3 — neural (~70 atoms)

Neural blocks, training procedures, generative models, graphs, representation,
recommenders, RL.

## Phase 4 — frontier (~50 atoms)

Tabular frontier, causality, signal/vision classics, tier 3 everywhere.
Causality and the tabular frontier are last because they are the atoms most
likely to change under you; writing them early means rewriting them.

## Phase 5 — polish

- `figures/make.py` regenerating every diagram from the implementations
- a benchmark harness that runs every implemented atom on a shared synthetic suite
- cross-links: every atom links its prerequisites and its successors
- a printable one-page map of the whole table

## What is deliberately out of scope

- **No framework.** No `BaseEstimator`, no pipelines, no `fit_transform` chaining.
  Composition is the reader's job and should stay visible.
- **No GPU kernels.** Triton and CUDA belong in the projects that need them; the
  atoms here are readable, not fast.
- **No vendored datasets.** Every demo generates its data.
- **No LLM wrappers.** Prompting is not an algorithm; the adjacent atoms
  (`attention`, `bpe`, `dpo`) are here, the applications are not.
