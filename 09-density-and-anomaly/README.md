# 09 · Density & anomaly

> How likely is this point? Density estimation and the detectors built on it.

Anomaly detection is density estimation with a threshold. This folder gives you the estimators (KDE, copulas, extreme-value tails) and the practical scorers (LOF, isolation forest, Mahalanobis, reconstruction error) that industry actually deploys.

<!-- catalog:begin -->
**11 atoms · 49 variants · 7 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Kernel density estimation | [`kde.py`](kde.py) | 1 | 8 | Kernels, bandwidth rules and tree acceleration. |
| ✓ Isolation forest | [`isolation_forest.py`](isolation_forest.py) | 1 | 4 | Anomalies are isolated in fewer random splits. |
| ✓ Local outlier factor | [`lof.py`](lof.py) | 1 | 4 | Local reachability density ratios: density-relative outliers. |
| ✓ Histogram density | [`histogram_density.py`](histogram_density.py) | 2 | 4 | Binning rules and the bias-variance of histogram estimates. |
| ✓ Mahalanobis / MCD | [`mahalanobis.py`](mahalanobis.py) | 2 | 4 | Hotelling T-squared and robust covariance for multivariate outliers. |
| ✓ HBOS | [`hbos.py`](hbos.py) | 2 | 3 | Histogram-based outlier scores with dynamic binning. |
| ✓ Autoencoder anomaly | [`autoencoder_anomaly.py`](autoencoder_anomaly.py) | 2 | 5 | Reconstruction error as a novelty score, with thresholding. |
| · Change-point detection | [`change_point.py`](change_point.py) | 2 | 5 | Finding where the process changed, not just that it is unusual. |
| · Elliptic envelope | [`elliptic_envelope.py`](elliptic_envelope.py) | 2 | 4 | Robust covariance ellipsoids as a novelty boundary. |
| · COPOD | [`copod.py`](copod.py) | 3 | 3 | Empirical copula tail probabilities: parameter-free outlier scoring. |
| · Extreme value theory | [`extreme_value.py`](extreme_value.py) | 3 | 5 | Peaks-over-threshold and generalised Pareto tail modelling. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Kernel density estimation** ([`kde.py`](kde.py) · [kde.md](kde.md)) — `gaussian` · `epanechnikov` · `tophat` · `silverman` · `scott` · `cv_bandwidth` · `tree_kde` · `product_kernel`
- **Isolation forest** ([`isolation_forest.py`](isolation_forest.py) · [isolation_forest.md](isolation_forest.md)) — `iforest` · `extended` · `anomaly_score` · `max_samples`
- **Local outlier factor** ([`lof.py`](lof.py) · [lof.md](lof.md)) — `lof` · `simplified_lof` · `cluster_lof` · `lof_knn`
- **Histogram density** ([`histogram_density.py`](histogram_density.py) · [histogram_density.md](histogram_density.md)) — `fd_rule` · `sturges` · `scott` · `bayesian_blocks`
- **Mahalanobis / MCD** ([`mahalanobis.py`](mahalanobis.py) · [mahalanobis.md](mahalanobis.md)) — `hotelling_t2` · `mcd` · `ogk` · `shrinkage`
- **HBOS** ([`hbos.py`](hbos.py) · [hbos.md](hbos.md)) — `static_bins` · `dynamic_bins` · `adaptive`
- **Autoencoder anomaly** ([`autoencoder_anomaly.py`](autoencoder_anomaly.py) · [autoencoder_anomaly.md](autoencoder_anomaly.md)) — `ae` · `vae` · `lstm_ae` · `denoising_ae` · `thresholding`
- **Change-point detection** ([`change_point.py`](change_point.py) · [change_point.md](change_point.md)) — `bocpd` · `pelt` · `cusum` · `binary_segmentation` · `window_based`
- **Elliptic envelope** ([`elliptic_envelope.py`](elliptic_envelope.py) · [elliptic_envelope.md](elliptic_envelope.md)) — `mcd_ellipse` · `min_cov_det` · `shrinkage` · `chi2_threshold`
- **COPOD** ([`copod.py`](copod.py) · [copod.md](copod.md)) — `copod` · `copod_skew` · `ecod`
- **Extreme value theory** ([`extreme_value.py`](extreme_value.py) · [extreme_value.md](extreme_value.md)) — `pot` · `gpd` · `block_maxima` · `gev` · `return_level`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
