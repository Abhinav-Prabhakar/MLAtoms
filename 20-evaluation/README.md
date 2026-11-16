# 20 · Evaluation & uncertainty

> Knowing whether you are right — and how much you do not know.

A model is only as good as the evidence behind it. This folder covers honest resampling, calibrated probabilities, conformal guarantees, drift monitoring and the statistical tests that stop you shipping noise.

<!-- catalog:begin -->
**11 atoms · 68 variants · 4 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Resampling | [`resampling.py`](resampling.py) | 1 | 8 | Holdout, k-fold, stratified/group/time-series and nested CV. |
| ✓ Bootstrap | [`bootstrap.py`](bootstrap.py) | 1 | 5 | Percentile, BCa and paired bootstrap intervals. |
| ✓ Calibration | [`calibration.py`](calibration.py) | 1 | 6 | Platt, isotonic and temperature scaling; reliability diagrams. |
| ✓ Interpretability | [`interpretability.py`](interpretability.py) | 1 | 9 | Permutation importance, SHAP, PDP/ICE and integrated gradients. |
| · Conformal prediction | [`conformal.py`](conformal.py) | 2 | 7 | Split, Jackknife+ and Mondrian sets with coverage guarantees. |
| · Uncertainty estimation | [`uncertainty.py`](uncertainty.py) | 2 | 6 | MC dropout, deep ensembles and epistemic/aleatoric decomposition. |
| · Drift detection | [`drift_detection.py`](drift_detection.py) | 2 | 7 | ADWIN, DDM, KS and PSI monitors for production. |
| · A/B testing | [`ab_testing.py`](ab_testing.py) | 2 | 5 | Power, sequential tests and CUPED variance reduction. |
| · Hypothesis tests | [`hypothesis_tests.py`](hypothesis_tests.py) | 2 | 6 | McNemar, Wilcoxon, permutation tests and multiplicity control. |
| · Selective prediction | [`selective_prediction.py`](selective_prediction.py) | 3 | 4 | Abstention policies and risk-coverage curves. |
| · Fairness metrics | [`fairness.py`](fairness.py) | 3 | 5 | Demographic parity, equalised odds and group calibration. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Resampling** ([`resampling.py`](resampling.py) · [resampling.md](resampling.md)) — `holdout` · `kfold` · `stratified` · `group` · `time_series_split` · `nested_cv` · `repeated_cv` · `leave_one_out`
- **Bootstrap** ([`bootstrap.py`](bootstrap.py) · [bootstrap.md](bootstrap.md)) — `percentile` · `bca` · `paired` · `block_bootstrap` · `bayesian_bootstrap`
- **Calibration** ([`calibration.py`](calibration.py) · [calibration.md](calibration.md)) — `platt` · `isotonic` · `temperature` · `beta` · `dirichlet` · `reliability_diagram`
- **Interpretability** ([`interpretability.py`](interpretability.py) · [interpretability.md](interpretability.md)) — `permutation` · `shap` · `kernel_shap` · `tree_shap` · `pdp` · `ice` · `ale` · `integrated_gradients` · `lime`
- **Conformal prediction** ([`conformal.py`](conformal.py) · [conformal.md](conformal.md)) — `split` · `full` · `jackknife_plus` · `cv_plus` · `mondrian` · `adaptive` · `conformalized_quantile`
- **Uncertainty estimation** ([`uncertainty.py`](uncertainty.py) · [uncertainty.md](uncertainty.md)) — `mc_dropout` · `deep_ensembles` · `quantile_regression` · `evidential` · `gaussian_process` · `swag`
- **Drift detection** ([`drift_detection.py`](drift_detection.py) · [drift_detection.md](drift_detection.md)) — `adwin` · `ddm` · `eddm` · `ks` · `psi` · `mmd` · `page_hinkley`
- **A/B testing** ([`ab_testing.py`](ab_testing.py) · [ab_testing.md](ab_testing.md)) — `frequentist` · `sequential` · `cuped` · `bayesian` · `delta_method`
- **Hypothesis tests** ([`hypothesis_tests.py`](hypothesis_tests.py) · [hypothesis_tests.md](hypothesis_tests.md)) — `mcnemar` · `wilcoxon` · `permutation` · `t_test` · `friedman` · `holm_bonferroni`
- **Selective prediction** ([`selective_prediction.py`](selective_prediction.py) · [selective_prediction.md](selective_prediction.md)) — `threshold` · `risk_coverage` · `conformal_abstain` · `learning_to_defer`
- **Fairness metrics** ([`fairness.py`](fairness.py) · [fairness.md](fairness.md)) — `demographic_parity` · `equalized_odds` · `equal_opportunity` · `calibration_by_group` · `disparate_impact`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
