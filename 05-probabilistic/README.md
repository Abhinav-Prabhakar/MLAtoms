# 05 · Probabilistic models

> Probability as the model: priors, posteriors, filters and approximate inference.

A point prediction is a lossy summary. This folder builds the machinery for distributions: conjugate posteriors, Gaussian processes, state-space filters and the two workhorse approximations (variational inference, MCMC).

<!-- catalog:begin -->
**15 atoms · 87 variants · 9 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Naive Bayes | [`naive_bayes.py`](naive_bayes.py) | 1 | 7 | Conditional independence as a feature, not a bug. |
| ✓ Discriminant analysis | [`discriminant_analysis.py`](discriminant_analysis.py) | 1 | 6 | LDA, QDA, shrinkage and regularised discriminant analysis. |
| ✓ Gaussian process regression | [`gaussian_process.py`](gaussian_process.py) | 1 | 9 | Kernels, posterior mean/variance and marginal likelihood by Cholesky. |
| ✓ Bayesian inference | [`bayesian_inference.py`](bayesian_inference.py) | 1 | 7 | MLE, MAP, conjugate priors and credible intervals. |
| ✓ MCMC | [`mcmc.py`](mcmc.py) | 2 | 7 | Sampling the posterior when algebra fails. |
| ✓ Variational inference | [`variational_inference.py`](variational_inference.py) | 2 | 5 | Turn integration into optimisation. |
| ✓ Kalman filter | [`kalman.py`](kalman.py) | 2 | 6 | Optimal filtering for linear-Gaussian state space models. |
| ✓ Particle filter | [`particle_filter.py`](particle_filter.py) | 2 | 6 | Sequential Monte Carlo for nonlinear, non-Gaussian state. |
| ✓ Hierarchical Bayes | [`hierarchical_bayes.py`](hierarchical_bayes.py) | 2 | 5 | Partial pooling, shrinkage and empirical Bayes. |
| · Graphical models | [`pgm.py`](pgm.py) | 2 | 6 | Factor graphs, variable elimination and belief propagation. |
| · GP classification | [`gp_classification.py`](gp_classification.py) | 3 | 5 | Gaussian processes with non-Gaussian likelihoods. |
| · Dirichlet process | [`dirichlet_process.py`](dirichlet_process.py) | 3 | 5 | Nonparametric priors that grow the model with the data. |
| · Laplace approximation | [`laplace_approximation.py`](laplace_approximation.py) | 3 | 4 | Gaussian around the mode: predictive uncertainty from the Hessian. |
| · Expectation propagation | [`expectation_propagation.py`](expectation_propagation.py) | 3 | 3 | Moment matching to approximate intractable posteriors. |
| · Copulas | [`copula.py`](copula.py) | 3 | 6 | Dependence beyond correlation, with correct margins. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Naive Bayes** ([`naive_bayes.py`](naive_bayes.py) · [naive_bayes.md](naive_bayes.md)) — `gaussian` · `multinomial` · `bernoulli` · `categorical` · `complement` · `semi_supervised` · `tfidf`
- **Discriminant analysis** ([`discriminant_analysis.py`](discriminant_analysis.py) · [discriminant_analysis.md](discriminant_analysis.md)) — `lda` · `qda` · `shrinkage_lda` · `rda` · `fisher_lda` · `multiclass`
- **Gaussian process regression** ([`gaussian_process.py`](gaussian_process.py) · [gaussian_process.md](gaussian_process.md)) — `rbf` · `matern` · `rational_quadratic` · `periodic` · `linear` · `sum_kernel` · `product_kernel` · `sparse_gp` · `warped`
- **Bayesian inference** ([`bayesian_inference.py`](bayesian_inference.py) · [bayesian_inference.md](bayesian_inference.md)) — `mle` · `map` · `conjugate_normal` · `beta_binomial` · `dirichlet_multinomial` · `credible_intervals` · `posterior_predictive`
- **MCMC** ([`mcmc.py`](mcmc.py) · [mcmc.md](mcmc.md)) — `metropolis_hastings` · `gibbs` · `hmc` · `nuts` · `slice_sampling` · `adaptive_metropolis` · `diagnostics`
- **Variational inference** ([`variational_inference.py`](variational_inference.py) · [variational_inference.md](variational_inference.md)) — `mean_field` · `cavi` · `reparameterization` · `structured_vi` · `black_box_vi`
- **Kalman filter** ([`kalman.py`](kalman.py) · [kalman.md](kalman.md)) — `kf` · `ekf` · `ukf` · `information_filter` · `rts_smoother` · `missing_obs`
- **Particle filter** ([`particle_filter.py`](particle_filter.py) · [particle_filter.md](particle_filter.md)) — `sir` · `sis` · `auxiliary_pf` · `systematic_resample` · `stratified_resample` · `regularized_pf`
- **Hierarchical Bayes** ([`hierarchical_bayes.py`](hierarchical_bayes.py) · [hierarchical_bayes.md](hierarchical_bayes.md)) — `partial_pooling` · `full_pooling` · `no_pooling` · `empirical_bayes` · `noncentered`
- **Graphical models** ([`pgm.py`](pgm.py) · [pgm.md](pgm.md)) — `factor_graph` · `variable_elimination` · `belief_propagation` · `junction_tree` · `sum_product` · `max_product`
- **GP classification** ([`gp_classification.py`](gp_classification.py) · [gp_classification.md](gp_classification.md)) — `laplace` · `expectation_propagation` · `probit` · `logistic` · `one_vs_rest`
- **Dirichlet process** ([`dirichlet_process.py`](dirichlet_process.py) · [dirichlet_process.md](dirichlet_process.md)) — `crp` · `stick_breaking` · `dpmm` · `truncated_dp` · `dp_means`
- **Laplace approximation** ([`laplace_approximation.py`](laplace_approximation.py) · [laplace_approximation.md](laplace_approximation.md)) — `diag_hessian` · `full_hessian` · `kfac_laplace` · `predictive`
- **Expectation propagation** ([`expectation_propagation.py`](expectation_propagation.py) · [expectation_propagation.md](expectation_propagation.md)) — `ep` · `damped_ep` · `ep_for_gp`
- **Copulas** ([`copula.py`](copula.py) · [copula.md](copula.md)) — `gaussian` · `t` · `clayton` · `gumbel` · `frank` · `vine`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
