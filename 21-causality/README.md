# 21 · Causality

> From correlation to intervention: identification, estimation and discovery.

Prediction is not explanation. This folder is the toolkit for answering 'what happens if we act?' — potential outcomes and estimators first, then the discovery algorithms that recover structure from observational data.

<!-- catalog:begin -->
**16 atoms · 75 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · Potential outcomes | [`potential_outcomes.py`](potential_outcomes.py) | 1 | 6 | ATE/ATT/CATE and the identification assumptions behind them. |
| · Propensity scores | [`propensity.py`](propensity.py) | 1 | 5 | IPW, matching, trimming and covariate balance. |
| · do-calculus | [`do_calculus.py`](do_calculus.py) | 1 | 5 | Backdoor and frontdoor adjustment; finding valid adjustment sets. |
| · Doubly robust | [`doubly_robust.py`](doubly_robust.py) | 2 | 5 | AIPW, TMLE and cross-fitted double machine learning. |
| · Meta-learners | [`meta_learners.py`](meta_learners.py) | 2 | 6 | S-, T-, X- and R-learners for heterogeneous effects. |
| · Instrumental variables | [`instrumental_variables.py`](instrumental_variables.py) | 2 | 5 | Two-stage least squares and weak-instrument diagnostics. |
| · Difference-in-differences | [`diff_in_diff.py`](diff_in_diff.py) | 2 | 5 | Parallel trends, two-way fixed effects and event studies. |
| · PC algorithm | [`pc_algorithm.py`](pc_algorithm.py) | 2 | 5 | Constraint-based discovery with conditional independence tests. |
| · Causal forests | [`causal_forest.py`](causal_forest.py) | 3 | 4 | Honest splitting for conditional effect estimation. |
| · Synthetic control | [`synthetic_control.py`](synthetic_control.py) | 3 | 4 | Donor-pool weighting with placebo inference. |
| · Regression discontinuity | [`regression_discontinuity.py`](regression_discontinuity.py) | 3 | 5 | Local polynomial estimation and bandwidth choice. |
| · Mediation analysis | [`mediation.py`](mediation.py) | 3 | 4 | Natural direct and indirect effects. |
| · Score-based discovery | [`score_based_discovery.py`](score_based_discovery.py) | 3 | 4 | GES and hill-climbing over DAGs with BIC scores. |
| · LiNGAM | [`lingam.py`](lingam.py) | 3 | 4 | Non-Gaussian linear identifiability via ICA. |
| · NOTEARS | [`notears.py`](notears.py) | 3 | 4 | Continuous acyclicity constraint for gradient-based discovery. |
| · Sensitivity analysis | [`sensitivity.py`](sensitivity.py) | 3 | 4 | E-values and bounds for unmeasured confounding. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Potential outcomes** ([`potential_outcomes.py`](potential_outcomes.py) · [potential_outcomes.md](potential_outcomes.md)) — `ate` · `att` · `cate` · `sate` · `randomized` · `observational`
- **Propensity scores** ([`propensity.py`](propensity.py) · [propensity.md](propensity.md)) — `ipw` · `matching` · `stratification` · `trimming` · `balance_check`
- **do-calculus** ([`do_calculus.py`](do_calculus.py) · [do_calculus.md](do_calculus.md)) — `backdoor` · `frontdoor` · `adjustment_set` · `d_separation` · `counterfactual`
- **Doubly robust** ([`doubly_robust.py`](doubly_robust.py) · [doubly_robust.md](doubly_robust.md)) — `aipw` · `tmle` · `dml` · `cross_fitting` · `orthogonal_scores`
- **Meta-learners** ([`meta_learners.py`](meta_learners.py) · [meta_learners.md](meta_learners.md)) — `s_learner` · `t_learner` · `x_learner` · `r_learner` · `dr_learner` · `causal_ml`
- **Instrumental variables** ([`instrumental_variables.py`](instrumental_variables.py) · [instrumental_variables.md](instrumental_variables.md)) — `2sls` · `control_function` · `weak_instruments` · `late` · `gmm_iv`
- **Difference-in-differences** ([`diff_in_diff.py`](diff_in_diff.py) · [diff_in_diff.md](diff_in_diff.md)) — `twfe` · `event_study` · `staggered` · `synthetic_did` · `parallel_trends_test`
- **PC algorithm** ([`pc_algorithm.py`](pc_algorithm.py) · [pc_algorithm.md](pc_algorithm.md)) — `pc` · `fci` · `rfci` · `cpdag` · `conditional_independence`
- **Causal forests** ([`causal_forest.py`](causal_forest.py) · [causal_forest.md](causal_forest.md)) — `grf` · `honest_splitting` · `causal_tree` · `policy_tree`
- **Synthetic control** ([`synthetic_control.py`](synthetic_control.py) · [synthetic_control.md](synthetic_control.md)) — `synthetic_control` · `synthetic_did` · `placebo` · `conformal_inference`
- **Regression discontinuity** ([`regression_discontinuity.py`](regression_discontinuity.py) · [regression_discontinuity.md](regression_discontinuity.md)) — `sharp` · `fuzzy` · `local_linear` · `bandwidth_selection` · `bias_corrected`
- **Mediation analysis** ([`mediation.py`](mediation.py) · [mediation.md](mediation.md)) — `natural_effects` · `mediation_formula` · `interventional_effects` · `sensitivity`
- **Score-based discovery** ([`score_based_discovery.py`](score_based_discovery.py) · [score_based_discovery.md](score_based_discovery.md)) — `ges` · `hill_climbing` · `order_search` · `bic_score`
- **LiNGAM** ([`lingam.py`](lingam.py) · [lingam.md](lingam.md)) — `direct_lingam` · `ica_lingam` · `parsimonious` · `var_lingam`
- **NOTEARS** ([`notears.py`](notears.py) · [notears.md](notears.md)) — `notears` · `notears_linear` · `notears_mlp` · `dag_constraint`
- **Sensitivity analysis** ([`sensitivity.py`](sensitivity.py) · [sensitivity.md](sensitivity.md)) — `e_value` · `rosenbaum` · `omitted_variable_bounds` · `negative_control`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
