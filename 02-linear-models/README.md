# 02 · Linear models

> Closed-form, penalised and likelihood-based regression — still the best value per line of code.

Linear models are the control group of machine learning and, on small tabular problems, frequently the winner. This folder covers the algebra (QR, SVD, coordinate descent), the statistics (standard errors, effective degrees of freedom) and the extensions that make them flexible (splines, GLMs, quantiles).

<!-- catalog:begin -->
**15 atoms · 90 variants · 6 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Ordinary least squares | [`ols.py`](ols.py) | 1 | 7 | Four ways to solve the same normal equations, with standard errors. |
| ✓ Ridge | [`ridge.py`](ridge.py) | 1 | 6 | L2 shrinkage solved by SVD: effective degrees of freedom and generalised cross-validation. |
| ✓ Lasso | [`lasso.py`](lasso.py) | 1 | 6 | L1 sparsity via coordinate descent and soft-thresholding, plus the full path. |
| ✓ Elastic net | [`elastic_net.py`](elastic_net.py) | 1 | 4 | L1 + L2: sparse selection that keeps correlated features together. |
| ✓ Logistic regression | [`logistic_regression.py`](logistic_regression.py) | 1 | 8 | IRLS/Newton fitting, multinomial extension, separability handling. |
| ✓ Generalised linear models | [`glm.py`](glm.py) | 1 | 9 | Exponential families, link functions and IRLS. |
| · Robust regression | [`robust_regression.py`](robust_regression.py) | 2 | 6 | When a handful of points would otherwise own the fit. |
| · Quantile regression | [`quantile_regression.py`](quantile_regression.py) | 2 | 6 | Predict conditional quantiles, not just the mean. |
| · Splines | [`splines.py`](splines.py) | 2 | 7 | Piecewise polynomials that turn a linear model into a flexible one. |
| · Additive models | [`gam.py`](gam.py) | 2 | 6 | GAMs fitted by backfitting, with smooth terms and interactions. |
| · Isotonic regression | [`isotonic.py`](isotonic.py) | 2 | 5 | Pool-adjacent-violators for monotone fits and calibration maps. |
| · Bayesian linear regression | [`bayesian_linear.py`](bayesian_linear.py) | 2 | 7 | A posterior instead of a point estimate, in closed form. |
| · Generalised least squares | [`gls.py`](gls.py) | 2 | 5 | Weighted least squares under heteroscedastic, correlated noise. |
| · Partial least squares | [`pls.py`](pls.py) | 3 | 5 | PLS and NIPALS for wide, strongly collinear predictors. |
| · Tobit regression | [`tobit.py`](tobit.py) | 3 | 3 | Censored regression by maximum likelihood. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Ordinary least squares** ([`ols.py`](ols.py) · [ols.md](ols.md)) — `normal_equations` · `qr` · `svd_pinv` · `cholesky` · `recursive` · `with_intercept` · `standard_errors`
- **Ridge** ([`ridge.py`](ridge.py) · [ridge.md](ridge.md)) — `closed_form` · `svd` · `generalized_ridge` · `loo_cv` · `group_ridge` · `nuclear_norm`
- **Lasso** ([`lasso.py`](lasso.py) · [lasso.md](lasso.md)) — `coordinate_descent` · `lars` · `homotopy` · `path` · `positive_lasso` · `group_lasso`
- **Elastic net** ([`elastic_net.py`](elastic_net.py) · [elastic_net.md](elastic_net.md)) — `naive` · `coordinate_descent` · `path` · `group_elastic_net`
- **Logistic regression** ([`logistic_regression.py`](logistic_regression.py) · [logistic_regression.md](logistic_regression.md)) — `binomial` · `multinomial` · `ordinal` · `irls` · `lbfgs` · `elastic_net` · `class_weight` · `separable_fix`
- **Generalised linear models** ([`glm.py`](glm.py) · [glm.md](glm.md)) — `poisson` · `gamma` · `binomial` · `tweedie` · `negative_binomial` · `log_link` · `identity_link` · `offsets` · `dispersion`
- **Robust regression** ([`robust_regression.py`](robust_regression.py) · [robust_regression.md](robust_regression.md)) — `huber` · `irls_huber` · `ransac` · `theil_sen` · `lad` · `tukey_biweight`
- **Quantile regression** ([`quantile_regression.py`](quantile_regression.py) · [quantile_regression.md](quantile_regression.md)) — `pinball` · `linear_programming` · `irwls` · `multiple_quantiles` · `noncrossing` · `penalized`
- **Splines** ([`splines.py`](splines.py) · [splines.md](splines.md)) — `natural_cubic` · `b_spline` · `p_spline` · `truncated_power` · `smoothing_spline` · `cyclic` · `tensor_product`
- **Additive models** ([`gam.py`](gam.py) · [gam.md](gam.md)) — `backfitting` · `penalized` · `interactions` · `by_variable` · `logistic_gam` · `tgam`
- **Isotonic regression** ([`isotonic.py`](isotonic.py) · [isotonic.md](isotonic.md)) — `pava` · `isotonic_l2` · `decreasing` · `calibration_map` · `isotonic_boost`
- **Bayesian linear regression** ([`bayesian_linear.py`](bayesian_linear.py) · [bayesian_linear.md](bayesian_linear.md)) — `conjugate` · `bayesian_ridge` · `ards` · `empirical_bayes` · `predictive` · `spike_slab` · `horseshoe`
- **Generalised least squares** ([`gls.py`](gls.py) · [gls.md](gls.md)) — `wls` · `fgls` · `gls_cholesky` · `ar1_correlation` · `robust_sandwich`
- **Partial least squares** ([`pls.py`](pls.py) · [pls.md](pls.md)) — `pls1` · `pls2` · `nipals` · `simpls` · `kernel_pls`
- **Tobit regression** ([`tobit.py`](tobit.py) · [tobit.md](tobit.md)) — `tobit` · `two_limit` · `iv_tobit`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
