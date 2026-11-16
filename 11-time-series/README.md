# 11 · Time series

> Forecasting atoms: from exponential smoothing to patch transformers.

Forecasting is a separate discipline because the data breaks i.i.d. assumptions. This folder spans the classical state-space methods, the decomposition toolbox, and the modern neural forecasters — each with the baseline it must beat.

<!-- catalog:begin -->
**16 atoms · 77 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · Naive baselines | [`baselines.py`](baselines.py) | 1 | 6 | The forecasts your model has to beat. |
| · ETS / Holt-Winters | [`ets.py`](ets.py) | 1 | 7 | Error-trend-seasonal state-space exponential smoothing. |
| · ARIMA | [`arima.py`](arima.py) | 1 | 7 | AR, MA, differencing, SARIMA and ACF/PACF identification. |
| · Vector autoregression | [`var.py`](var.py) | 2 | 6 | Multivariate autoregression and Granger causality. |
| · GARCH | [`garch.py`](garch.py) | 2 | 6 | Volatility clustering and variance forecasts. |
| · STL decomposition | [`stl.py`](stl.py) | 2 | 4 | LOESS seasonal-trend decomposition and robust variants. |
| · Spectral analysis | [`spectral_analysis.py`](spectral_analysis.py) | 2 | 5 | Periodograms, Welch averaging and spectral features. |
| · DLinear | [`dlinear.py`](dlinear.py) | 2 | 4 | Decomposition plus a linear layer: the transformer sanity check. |
| · Theta method | [`theta.py`](theta.py) | 3 | 4 | Theta lines: a simple method that beat far fancier ones. |
| · Croston's method | [`croston.py`](croston.py) | 3 | 4 | Intermittent demand by separating occurrence and size. |
| · N-BEATS | [`nbeats.py`](nbeats.py) | 3 | 4 | Basis-expansion blocks with doubly residual stacking. |
| · N-HiTS | [`nhits.py`](nhits.py) | 3 | 3 | Multi-rate sampling and hierarchical interpolation. |
| · PatchTST | [`patchtst.py`](patchtst.py) | 3 | 4 | Patch tokens with channel independence. |
| · TSMixer | [`tsmixer.py`](tsmixer.py) | 3 | 4 | All-MLP token and channel mixing. |
| · DeepAR | [`deepar.py`](deepar.py) | 3 | 5 | Probabilistic autoregressive RNN forecasting with likelihoods. |
| · Temporal fusion transformer | [`tft.py`](tft.py) | 3 | 4 | Variable selection networks and interpretable attention. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Naive baselines** ([`baselines.py`](baselines.py) · [baselines.md](baselines.md)) — `naive` · `seasonal_naive` · `drift` · `mean` · `random_walk` · `snaive`
- **ETS / Holt-Winters** ([`ets.py`](ets.py) · [ets.md](ets.md)) — `simple_es` · `holt` · `damped_trend` · `holt_winters_add` · `holt_winters_mul` · `ets_state_space` · `auto_ets`
- **ARIMA** ([`arima.py`](arima.py) · [arima.md](arima.md)) — `ar` · `ma` · `arma` · `arima` · `sarima` · `arimax` · `auto_arima`
- **Vector autoregression** ([`var.py`](var.py) · [var.md](var.md)) — `var` · `varx` · `granger` · `irf` · `fevd` · `var_ridge`
- **GARCH** ([`garch.py`](garch.py) · [garch.md](garch.md)) — `arch` · `garch` · `egarch` · `gjr_garch` · `igarch` · `garch_t`
- **STL decomposition** ([`stl.py`](stl.py) · [stl.md](stl.md)) — `stl` · `robust` · `mstl` · `seasonal_windows`
- **Spectral analysis** ([`spectral_analysis.py`](spectral_analysis.py) · [spectral_analysis.md](spectral_analysis.md)) — `periodogram` · `welch` · `multitaper` · `lomb_scargle` · `stft_features`
- **DLinear** ([`dlinear.py`](dlinear.py) · [dlinear.md](dlinear.md)) — `dlinear` · `nlinear` · `moving_avg_decomp` · `channel_independent`
- **Theta method** ([`theta.py`](theta.py) · [theta.md](theta.md)) — `theta` · `optimised_theta` · `dstm` · `dynamic_theta`
- **Croston's method** ([`croston.py`](croston.py) · [croston.md](croston.md)) — `croston` · `sba` · `tsb` · `optimized_croston`
- **N-BEATS** ([`nbeats.py`](nbeats.py) · [nbeats.md](nbeats.md)) — `generic` · `interpretable` · `nbeatsx` · `ensemble`
- **N-HiTS** ([`nhits.py`](nhits.py) · [nhits.md](nhits.md)) — `nhits` · `multi_rate` · `hierarchical_interp`
- **PatchTST** ([`patchtst.py`](patchtst.py) · [patchtst.md](patchtst.md)) — `patchtst` · `channel_independent` · `supervised` · `self_supervised`
- **TSMixer** ([`tsmixer.py`](tsmixer.py) · [tsmixer.md](tsmixer.md)) — `tsmixer` · `tsmixer_ext` · `time_mix` · `feature_mix`
- **DeepAR** ([`deepar.py`](deepar.py) · [deepar.md](deepar.md)) — `deepar` · `gaussian_likelihood` · `negbin_likelihood` · `student_t` · `quantile_output`
- **Temporal fusion transformer** ([`tft.py`](tft.py) · [tft.md](tft.md)) — `tft` · `variable_selection` · `static_covariates` · `interpretable_attention`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
