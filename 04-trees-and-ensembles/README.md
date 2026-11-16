# 04 · Trees & ensembles

> Recursive partitioning and the boosting/bagging machines that still win tabular benchmarks.

Gradient-boosted trees remain the default answer for tabular data. This folder builds the tree from scratch, then the four boosters worth knowing, then the ensembling meta-algorithms that generalise all of it.

<!-- catalog:begin -->
**14 atoms · 81 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · CART | [`cart.py`](cart.py) | 1 | 7 | Split search, impurity criteria and cost-complexity pruning. |
| · Random forest | [`random_forest.py`](random_forest.py) | 1 | 6 | Bagging plus feature subsampling, with out-of-bag error. |
| · Gradient boosting | [`gradient_boosting.py`](gradient_boosting.py) | 1 | 7 | Functional gradient descent with shrinkage and subsampling. |
| · XGBoost | [`xgboost.py`](xgboost.py) | 1 | 9 | Second-order gain, regularised leaves and sparsity-aware splits. |
| · LightGBM | [`lightgbm.py`](lightgbm.py) | 1 | 7 | Histogram binning, GOSS, EFB and leaf-wise growth. |
| · AdaBoost | [`adaboost.py`](adaboost.py) | 1 | 6 | Exponential loss, reweighting and the margin interpretation. |
| · C4.5 | [`c45.py`](c45.py) | 2 | 5 | Entropy, information gain, gain ratio and missing-value handling. |
| · Extremely randomised trees | [`extra_trees.py`](extra_trees.py) | 2 | 4 | Randomised split thresholds for extra variance reduction. |
| · CatBoost | [`catboost.py`](catboost.py) | 2 | 5 | Ordered boosting and target statistics for categorical features. |
| · Stacking & blending | [`stacking.py`](stacking.py) | 2 | 6 | Out-of-fold meta-learning, voting and blending. |
| · BART | [`bart.py`](bart.py) | 3 | 5 | Bayesian additive regression trees by MCMC, with uncertainty. |
| · Oblique trees | [`oblique_trees.py`](oblique_trees.py) | 3 | 4 | Splits that are not axis-aligned. |
| · Rule lists | [`rule_lists.py`](rule_lists.py) | 3 | 5 | If-then models you can read out loud. |
| · Tsetlin machine | [`tsetlin_machine.py`](tsetlin_machine.py) | 3 | 5 | Propositional clauses learned by Tsetlin automata; bit-parallel and interpretable. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **CART** ([`cart.py`](cart.py) · [cart.md](cart.md)) — `gini` · `entropy` · `variance` · `friedman_mse` · `cost_complexity_pruning` · `surrogate_splits` · `missing_values`
- **Random forest** ([`random_forest.py`](random_forest.py) · [random_forest.md](random_forest.md)) — `classification` · `regression` · `extra_trees` · `oob` · `permutation_importance` · `balanced_subsample`
- **Gradient boosting** ([`gradient_boosting.py`](gradient_boosting.py) · [gradient_boosting.md](gradient_boosting.md)) — `regression` · `binary` · `multiclass` · `shrinkage` · `subsample` · `quantile` · `deviance`
- **XGBoost** ([`xgboost.py`](xgboost.py) · [xgboost.md](xgboost.md)) — `exact` · `approx` · `hist` · `dart` · `linear_booster` · `reg_alpha` · `reg_lambda` · `colsample` · `sparsity_aware`
- **LightGBM** ([`lightgbm.py`](lightgbm.py) · [lightgbm.md](lightgbm.md)) — `goss` · `efb` · `leaf_wise` · `histogram` · `dart` · `categorical` · `monotone`
- **AdaBoost** ([`adaboost.py`](adaboost.py) · [adaboost.md](adaboost.md)) — `discrete` · `real` · `gentle` · `samme` · `samme_r` · `stagewise`
- **C4.5** ([`c45.py`](c45.py) · [c45.md](c45.md)) — `entropy` · `gain_ratio` · `missing_values` · `pruning` · `continuous_splits`
- **Extremely randomised trees** ([`extra_trees.py`](extra_trees.py) · [extra_trees.md](extra_trees.md)) — `classification` · `regression` · `max_features` · `random_threshold`
- **CatBoost** ([`catboost.py`](catboost.py) · [catboost.md](catboost.md)) — `ordered_target_stats` · `plain` · `ctr_variants` · `langevin` · `monotone`
- **Stacking & blending** ([`stacking.py`](stacking.py) · [stacking.md](stacking.md)) — `voting` · `averaging` · `stacking` · `blending` · `oof` · `rank_average`
- **BART** ([`bart.py`](bart.py) · [bart.md](bart.md)) — `bart` · `pbart` · `lbart` · `xbart` · `sparse_bart`
- **Oblique trees** ([`oblique_trees.py`](oblique_trees.py) · [oblique_trees.md](oblique_trees.md)) — `random_projection` · `linear_split` · `svm_split` · `pca_split`
- **Rule lists** ([`rule_lists.py`](rule_lists.py) · [rule_lists.md](rule_lists.md)) — `oner` · `ripper` · `decision_list` · `cn2` · `apriori`
- **Tsetlin machine** ([`tsetlin_machine.py`](tsetlin_machine.py) · [tsetlin_machine.md](tsetlin_machine.md)) — `classifier` · `regressor` · `coalesced` · `drop_clause` · `weighted`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
