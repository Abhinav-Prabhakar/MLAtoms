# Bootstrap

`bootstrap.py` · Evaluation · Tier 1 · 5 variants

## Intuition

The bootstrap answers the only question that matters after you compute a number:
*how much would it move if I had drawn a different sample?* You cannot draw a new
sample, so you redraw the one you have — with replacement — and watch the statistic
move.

The mechanics are almost embarrassingly simple. Resample `n` rows with replacement,
recompute the statistic, repeat `B` times, and look at the spread:

```text
observed data  x₁ x₂ … xₙ                     statistic  θ̂ = 1.005
                     │
     ┌───────────────┼───────────────┐         B resamples of size n,
     ▼               ▼               ▼         each row drawn with replacement
  x₃ x₃ x₇ … x₁   x₂ x₅ x₅ … x₉   x₁ x₁ x₄ … x₂
     │               │               │
     ▼               ▼               ▼
    θ*₁             θ*₂             θ*_B          → percentile interval
```

Why this works: the empirical distribution `F̂` puts mass `1/n` on each observed point,
and sampling from `F̂` mimics sampling from the true `F`. That substitution — `F → F̂` —
is the *bootstrap principle*, and it is exact for statistics that are smooth functions of
sample moments. The demo's percentile interval for a mean is `[0.863, 1.152]` while the
textbook CLT interval is `0.282` wide against the bootstrap's `0.288`: they agree to 2%,
which is what theory predicts for `n = 200`.

The variants exist because the substitution is not always the same substitution:

- **`percentile`** reads quantiles off the bootstrap distribution. Simple, transformation
  respecting, first-order accurate.
- **`bca`** adjusts *which* quantiles it reads, correcting for median bias and for
  skewness. Second-order accurate; it needs a jackknife pass.
- **`paired`** changes what a row is: two measurements of the same unit are resampled
  together, so the interval is on their *difference*, not on two independent estimates.
- **`block_bootstrap`** changes what a row is the other way: for dependent data, rows are
  not independent draws, so blocks of consecutive rows are. The demo's AR(1) series with
  `ρ = 0.85` gets an interval `3.5×` wider than the iid bootstrap's — exactly the
  `√((1+ρ)/(1−ρ)) = 3.5` that the autocorrelation implies.
- **`bayesian_bootstrap`** replaces the multinomial draw with Dirichlet weights. It is the
  posterior of a nonparametric model rather than a resampling scheme, and it never assigns
  zero weight to an observed point.

## Variants

| Variant | Resamples | Interval | Assumption it fixes |
|:--|:--|:--|:--|
| `percentile` | rows iid | quantiles `α/2`, `1−α/2` | nothing beyond iid rows |
| `bca` | rows iid | adjusted quantiles from `z₀`, `a` | skewness and median bias |
| `paired` | row *pairs* jointly | quantiles of the difference | two vectors measured on the same units |
| `block_bootstrap` | contiguous blocks | quantiles | serial dependence |
| `bayesian_bootstrap` | Dirichlet weights | quantiles of the weighted statistic | no resampling, no zero mass |

`percentile` is the default and the one to reach for first. `bca` costs one extra pass of
`n` statistic evaluations and is worth it when the statistic is skewed — a variance, a
ratio, a quantile, or the mean of an exponential. In the demo, BCa moves the lower bound
up by `0.010` on the same data, and over 200 replications its coverage is `0.950` against
the percentile's `0.940`: at `n = 100` both are close to nominal, and the gap widens as `n`
falls.

`paired` is the variant people forget. If you evaluate two models on the same test set,
their errors are correlated — a hard row is hard for both — and treating the two error
vectors as independent samples throws that correlation away. The demo makes the point
numerically: the same comparison gives a width of `0.097` when paired and `0.487` when
resampled independently, a factor of five. Pairing is not a refinement; it is the
difference between detecting a `0.054` MSE gap and not.

`block_bootstrap` is the only correct choice for ordered data. The iid bootstrap destroys
the dependence structure: it draws `x_t` and `x_s` independently even though
`corr(x_t, x_s) = ρ^|t−s|`. The block version keeps runs of `block_size` rows intact, and
circular wrapping (`(s + j) % n`) means every row has the same chance of being inside a
block. The cost is a bias-variance trade in the block length: too short and dependence
survives, too long and there are too few distinct blocks.

`bayesian_bootstrap` uses `w ~ Dirichlet(1, …, 1)` and reports the weighted statistic. It
is a genuinely different justification — the Dirichlet is the posterior over the
probability vector of a nonparametric categorical model — but the result is numerically
close to the percentile interval (`0.286` against `0.288` width here). Its practical
advantage is smoothness: no row ever gets exactly zero weight, so a statistic that is
sensitive to duplicates or to ties behaves better.

## Derivation

**The plug-in principle.** Let `θ = T(F)` be the functional you care about — the mean is
`T(F) = ∫x dF(x)`. You observe `F̂`, the empirical distribution, so you estimate
`θ̂ = T(F̂)`. The bootstrap asks: what is the distribution of `T(F̂*) − T(F̂)` when `F̂*` is
drawn from `F̂`? The claim of the bootstrap is that this mimics the distribution of
`T(F̂) − T(F)` when `F̂` is drawn from `F`. For the mean, both are asymptotically
`N(0, σ²/n)`, which is why the percentile interval reproduces the CLT interval to `O(n^{-1/2})`.

**Percentile interval.** With `θ*_1…θ*_B` the bootstrap replicates,

$$
CI_{\text{pct}} = \bigl[\hat q_{\alpha/2},\; \hat q_{1-\alpha/2}\bigr],
\qquad
\hat q_p = \text{empirical } p\text{-quantile of } \{\theta^*_b\}
$$

It is *transformation respecting*: if you bootstrap `log θ` and exponentiate the interval,
you get the same answer as bootstrapping `θ` directly. That property is why it is the
default for ratios and variances, where a symmetric interval would be nonsense.

**Why percentile can be wrong, and what BCa does.** Write the bootstrap distribution's
quantiles in terms of a monotone transformation `g` that makes the statistic normal:

$$
g(\hat\theta) - g(\theta) \sim N(-z_0,\; 1), \qquad
\text{Var}\bigl(g(\hat\theta)\bigr) \text{ depends on } \theta \text{ through } a
$$

`z_0` is the *median bias* — where `θ̂` sits in its own bootstrap distribution — and `a` is
the *acceleration*, the skewness of the statistic as seen by the jackknife. Both are
estimable without knowing `g`:

$$
\hat z_0 = \Phi^{-1}\!\left(\frac{\#\{\theta^*_b < \hat\theta\} + \tfrac12\#\{= \hat\theta\}}{B}\right),
\qquad
\hat a = \frac{\sum_i (\bar\theta_{(\cdot)} - \theta_{(i)})^3}
{6\Bigl[\sum_i (\bar\theta_{(\cdot)} - \theta_{(i)})^2\Bigr]^{3/2}}
$$

where `θ_(i)` is the statistic with row `i` deleted (the jackknife) and `θ̄_(·)` its mean.
The adjusted quantiles are

$$
\alpha' = \Phi\!\left(z_0 + \frac{z_0 + z_\alpha}{1 - a(z_0 + z_\alpha)}\right),
\qquad
CI_{\text{BCa}} = \bigl[\hat q_{\alpha'},\; \hat q_{1-\alpha''}\bigr]
$$

with the same formula at `1 − α`. The correction is a monotone remapping of the quantile
levels: it shifts the interval away from the side where the bootstrap distribution is
compressed, which is exactly the side where the percentile interval undercovers. Its
coverage error is `O(n^{-1})` against the percentile's `O(n^{-1/2})` — one order better,
which is why the demo's two numbers separate as `n` shrinks.

The sign of `z₀` matters and is easy to get wrong: use strict `<` with a half-count for
ties, not `≤`. Counting ties as "above" biases `z₀` downward by half the tie mass, which
on a discrete statistic can move the interval by more than the correction it is meant to
apply.

**Paired resampling.** For two vectors measured on the same `n` units, the statistic is a
function of the pairs `(a_i, b_i)`. Resampling a common index set preserves the pairing:

$$
\theta^* = T\bigl(\{(a_{i_1}, b_{i_1}), …, (a_{i_n}, b_{i_n})\}\bigr), \qquad i_k \sim U\{1..n\}
$$

The variance of a difference is `Var(a − b) = Var(a) + Var(b) − 2Cov(a,b)`, so pairing
removes the covariance term. When the two measurements share most of their noise — the
same rows, the same difficulty — that term dominates and the paired interval is far
narrower. When they are genuinely independent samples, `Cov ≈ 0` and pairing changes
nothing; it never hurts, which is why "resample the test rows, not the two error vectors"
is the right default for model comparison.

**Block bootstrap.** For a stationary series, dependence means the effective sample size is
not `n`. For AR(1) with autocorrelation `ρ`,

$$
\text{Var}(\bar x) = \frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}
$$

so the honest standard error is `√((1+ρ)/(1−ρ))` times the iid one — `3.5×` at `ρ = 0.85`,
which is precisely the ratio the demo measures. The block bootstrap reproduces that factor
by keeping `b` consecutive rows together: resampling `⌈n/b⌉` blocks with uniformly random
circular starts and truncating to `n` preserves the within-block autocovariance. The usual
choice is `b ≈ n^{1/3}`, which balances the bias from dependence left inside a block
against the variance from having only `n/b` effectively independent blocks.

**Bayesian bootstrap.** Model the data as `x_i ~ F`, `F ~ DP(α, F₀)`; the posterior over the
probability vector at the observed points is `Dirichlet(1, …, 1)` when `α → 0` (Rubin's
choice). A draw is a weight vector `w`, and the induced estimate is `Σ w_i x_i`. Its
variance, computed from the Dirichlet's covariance, is

$$
\text{Var}\Bigl(\sum_i w_i x_i\Bigr)
= \frac{1}{n}\cdot\frac{1}{n}\sum_i (x_i - \bar x)^2 \,\bigl(1 + O(1/n)\bigr)
$$

which is the bootstrap variance to first order. The two intervals therefore agree
asymptotically — the demo's `0.286` versus `0.288` — while the Bayesian one has a cleaner
justification and no resampling artefacts.

**Does the interval earn its nominal coverage?** The demo runs 200 independent samples of
size 100 from `Exp(1)`, builds a 95% interval for the mean each time, and counts how often
it contains `1.0`: `0.940` for percentile and `0.950` for BCa. At `n = 50` the same
experiment gives `0.915` for percentile — the undercoverage the acceleration term is
designed to fix. Coverage is the property that distinguishes a confidence interval from a
plausible-looking range, and it is cheap to check by simulation for any statistic you
actually report.

## Code walkthrough

Every variant produces an index array, so a single loop can evaluate them all:

```python
    def _indices(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n_
        if self.variant == "block_bootstrap":
            b = min(self.block_size, n)
            n_blocks = -(-n // b)
            starts = rng.integers(0, n, size=n_blocks)
            return np.concatenate([(s + np.arange(b)) % n for s in starts])[:n]
        return rng.integers(0, n, size=n)
```

`-(-n // b)` is ceiling division without `math.ceil`: block count times block size must
cover `n`, and the final `[:n]` truncates the overshoot. `(s + np.arange(b)) % n` is the
*circular* block: a block starting near the end wraps to the beginning, so the first rows
are not systematically under-sampled, which the linear version would do. `rng.integers(0,
n, size=n)` is the iid bootstrap — note it draws `n` indices with replacement from
`range(n)`, not a permutation, which is the whole difference between the bootstrap and
cross-validation.

Paired data is indexed uniformly because a row is a *tuple* of observations:

```python
    def _take(self, idx: np.ndarray) -> float:
        if self.variant == "paired":
            return float(self._statistic(tuple(a[idx] for a in self._arrays)))
        return float(self._statistic(self._data[idx]))
```

`tuple(a[idx] for a in self._arrays)` applies the same index array to every vector, so a
resample can never pair `y_true[3]` with `pred_a[7]`. That single line is what makes the
paired interval `0.097` wide instead of `0.487`: the covariance between the two error
vectors survives the resampling.

BCa needs two extra statistics, both cheap:

```python
        below = np.sum(dist < self.theta_)
        ties = np.sum(dist == self.theta_)
        z0 = _norm_ppf((below + 0.5 * ties) / dist.size)
        jack = self._jackknife()
        dev = jack.mean() - jack
        scale = 6.0 * float(np.sum(dev ** 2)) ** 1.5
        accel = float(np.sum(dev ** 3)) / max(scale, EPS)
```

`below + 0.5 * ties` is the mid-rank position of the observed statistic inside its own
bootstrap distribution; `z0 > 0` means the replicates sit mostly below `θ̂`, i.e. the
statistic is biased upward. `dev` is the jackknife's deviation from its own mean, and
`accel` is its standardised third moment — the skewness of the statistic under
leave-one-out. `max(scale, EPS)` guards the constant statistic, where `dev` is all zeros
and `0/0` would make the interval `nan`.

The quantile remapping is three lines, and the guard is not cosmetic:

```python
        for j, p in enumerate((alpha, 1.0 - alpha)):
            z = _norm_ppf(p)
            denom = 1.0 - accel * (z0 + z)
            out[j] = 0.5 if abs(denom) < EPS else _norm_cdf(z0 + (z0 + z) / denom)
```

`1 − a(z₀ + z)` can cross zero when the acceleration is large (a very skewed statistic), and
the remapped level then flips past the middle of the distribution. Falling back to `0.5`
keeps the interval finite and, at that point, as uninformative as the data warrants.
`_norm_ppf` is Acklam's rational approximation with a Newton step, so the file needs no
SciPy; the demo's BCa interval matches `scipy.stats.bootstrap(method="BCa")` to six
decimals.

The main loop is one branch for the Bayesian variant and one for everything else:

```python
        for b in range(self.n_resamples):
            if self.variant == "bayesian_bootstrap":
                dist[b] = float(statistic(self._data, rng.dirichlet(np.ones(self.n_))))
            else:
                dist[b] = self._take(self._indices(rng))
```

`rng.dirichlet(np.ones(n))` is the posterior draw and the statistic receives the weights
directly, so the Bayesian variant never materialises a resample — that is why its interface
takes `statistic(data, weights)` while the others take `statistic(resample)`. Keeping the
same `dist` array for both means the quantile code and `se_` are shared.

The observed statistic is computed once in `_setup`, and the Bayesian case needs a
different call:

```python
        if self.variant == "bayesian_bootstrap":
            self.theta_ = float(statistic(self._data, np.full(n, 1.0 / n)))
        else:
            self.theta_ = float(statistic(self._data))
```

`np.full(n, 1/n)` is the uniform weight vector, which is the Bayesian bootstrap's point
estimate. Without this branch `z₀` would be computed against a statistic that is `nan` or
an error, and every BCa-style correction downstream would silently degenerate.

The interval itself is a quantile read, which is the entire difference between the
variants:

```python
        if self.variant == "bca":
            self.ci_ = self._bca_quantiles()
        else:
            alpha = (1.0 - self.level) / 2.0
            self.ci_ = np.quantile(dist, [alpha, 1.0 - alpha])
```

`np.quantile` with `[α/2, 1−α/2]` is the percentile interval; the BCa branch passes
*adjusted* levels into the same call. `self.width_` is stored because the demo compares
widths constantly and recomputing `ci_[1] - ci_[0]` at every call site invites a sign error.

## Complexity

| Step | Time | Memory |
|:--|:--|:--|
| One replicate | `O(n + T)` for resampling plus statistic cost `T` | `O(n)` |
| `percentile` / `paired` | `B·(n + T)` | `O(B)` |
| `block_bootstrap` | `B·(n + T)` + `O(n log n)` block assembly | `O(B)` |
| `bayesian_bootstrap` | `B·(n + T)` for the Dirichlet draw | `O(B + n)` |
| `bca` | `(B + n)·(n + T)` — the jackknife adds `n` fits | `O(B + n)` |
| Indexing (paired) | `O(n·m)` for `m` paired vectors | `O(n·m)` |

`B = 2000` is the usual default: the Monte Carlo error of a quantile estimate falls as
`B^{-1/2}`, so 2000 replicates give interval endpoints stable to about 2% of their width,
which is finer than the statistical error. BCa costs one extra pass of `n` statistic
evaluations — negligible for a mean, decisive for a model refit, where each `T` is minutes.
Memory is `O(B)` for the distribution, which is why the class stores it: plotting the
bootstrap distribution is often more informative than the interval.

## Failure modes

1. **Non-smooth statistics.** The bootstrap works for smooth functionals. The maximum, the
   mode, the number of distinct values and the argmax of a likelihood are not smooth: the
   resampled statistic can be identical across many replicates, the distribution is
   lumpy, and the percentile interval is wrong. Use a smoothed bootstrap or a different
   tool.
2. **Heavy tails.** For a distribution with infinite variance (Pareto with `α < 2`), the
   bootstrap distribution of the mean has no stable spread and the interval is
   meaningless. Check the tail before trusting the number; no resampling scheme invents
   moments that do not exist.
3. **Dependent rows.** The iid bootstrap on autocorrelated or clustered data is
   overconfident by `√((1+ρ)/(1−ρ))` — the demo's `3.5×`. Blocks or a cluster bootstrap
   are the fix, and the block length is a real tuning decision.
4. **`B` too small.** With `B = 200` the 2.5% quantile is estimated from the 5th smallest
   replicate; it moves by percent-level amounts between seeds. Use `B ≥ 1000` for
   intervals, more for tail quantiles.
5. **Bootstrapping the wrong thing.** Resampling the two error vectors independently when
   the models were evaluated on the same rows throws away the pairing and can turn a
   significant difference into "no effect". The demo's 5× width ratio is the size of the
   mistake.
6. **Percentile interval on a biased estimator.** The percentile interval is centred on
   `θ̂`, so if `θ̂` is biased the interval is too — it bounds the wrong quantity
   confidently. The bootstrap can estimate the bias (`θ̄* − θ̂`) and correct it, but that is
   a separate step, not something the interval does.
7. **Small `n`.** Undercoverage grows as `n` falls: `0.915` at `n = 50` for a skewed
   statistic. BCa helps but does not eliminate it. Below `n ≈ 30`, prefer exact or
   likelihood-based intervals where they exist.
8. **Statistic near a boundary.** A variance estimated as near-zero, or a proportion near
   0 or 1, gives a bootstrap distribution piled against the boundary and a one-sided
   interval. Transform first (`log`, logit) and back-transform the endpoints.
9. **Silent dependence between the statistic and `n`.** Bootstrapping a statistic that
   depends on the sample size (an AIC, a penalised likelihood) resamples at the wrong size
   for part of its definition. Compute the statistic on the resample, not a precomputed
   table of per-row values, unless you have verified the two agree.
10. **Weights ignored.** For the Bayesian variant the statistic must actually use the
    weights. Passing `np.mean` there throws them away and silently reports the point
    estimate `B` times, giving a zero-width interval.

## Applications

- **Confidence intervals for any metric.** Accuracy, F1, AUC, median error, the 95th
  percentile latency — no distributional assumption, one implementation. This is the
  default way to put error bars on a reported number.
- **Model comparison on a test set.** Paired bootstrap over the test rows gives an interval
  on the *difference* in error, which is the quantity a decision actually depends on.
- **A/B tests with awkward metrics.** Revenue per user, session length, conversion —
  heavily skewed and zero-inflated, where a t-test's assumptions fail and the bootstrap
  merely needs a resample.
- **Time-series and sequential data.** Block or stationary bootstrap for forecast error,
  autocorrelated residuals, and any metric computed over an ordered stream.
- **Small-sample inference.** Where asymptotic intervals are unreliable and the data are
  too few for a nonparametric test, the bootstrap often still works.
- **Estimator bias and stability.** `θ̄* − θ̂` estimates the bias, and the spread of the
  replicates says how much the estimate depends on which rows were sampled.
- **Simulation calibration.** Bootstrap a metric on synthetic data with a known truth to
  check that your reported interval covers it — the demo's 200-replication check, applied
  to whatever you are about to ship.

## References

- Efron, *Bootstrap Methods: Another Look at the Jackknife*, 1979, Annals of Statistics —
  the bootstrap, and the jackknife connection that BCa's acceleration term still uses.
- Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993 — the standard reference for
  the percentile and BCa intervals and for the transformation-respecting argument.
- DiCiccio & Efron, *Bootstrap Confidence Intervals*, 1996, Statistical Science — the
  taxonomy of intervals and the `O(n^{-1})` accuracy of BCa.
- Davison & Hinkley, *Bootstrap Methods and Their Application*, 1997 — worked derivations,
  including the paired and block schemes.
- Rubin, *The Bayesian Bootstrap*, 1981, Annals of Statistics — Dirichlet weights and the
  posterior interpretation.
- Künsch, *The Jackknife and the Bootstrap for General Stationary Observations*, 1989,
  Annals of Statistics — the block bootstrap and its consistency for stationary series.
- Politis & Romano, *The Stationary Bootstrap*, 1994, JASA — random block lengths and
  guidance on choosing the block size.
- Hall, *Theoretical Comparison of Bootstrap Confidence Intervals*, 1988, Annals of
  Statistics — why BCa is second-order accurate and the percentile interval is not.
