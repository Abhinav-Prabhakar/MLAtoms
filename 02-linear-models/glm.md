# Generalised linear models

`glm.py` · Linear models · Tier 1 · 9 variants

## Intuition

A linear model makes two assumptions that are usually wrong together: the mean is
linear in the features, and the noise is Gaussian with constant variance. A GLM
keeps the first and generalises the second:

$$
\mathbb{E}[y] = \mu = g^{-1}(x^\top\beta + \text{offset}),
\qquad
\operatorname{Var}[y] = \phi\,V(\mu)
$$

The **link** `g` says how the mean depends on the linear predictor; the **variance
function** `V(μ)` says how the noise scales with the mean. Counts have `V(μ)=μ`
(Poisson), positive skewed data has `V(μ)=μ²` (Gamma), proportions have
`V(μ)=μ(1−μ)` (Binomial). Once those two functions are fixed, one algorithm —
iteratively reweighted least squares — fits the model, and one formula gives the
standard errors.

That is the reason this is one atom and not six: Poisson, Gamma, Binomial, Tweedie
and negative binomial differ only in `V`, and log/identity/logit differ only in `g`.
The demo's Tweedie variant reproduces Poisson and Gamma *exactly* at `p=1,2`, which
is the API proving the point.

```
        ┌─────────────────────────────────────────────┐
        │  η = Xβ + offset                            │
        │      │                                      │
        │      g⁻¹                                    │
        │      ▼                                      │
        │  μ ──► V(μ), g'(μ) ──► w, z ──► WLS ──► β'  │
        │      ▲                                      │
        │      └──────────────────────────────────────┘
        └─────────────────────────────────────────────┘
```

## Variants

| Variant | Family | Link | `φ` | The data it is for |
|:--|:--|:--|:--|:--|
| `poisson` | Poisson | `log` (canonical) | fixed 1 | counts, rates, event totals |
| `gamma` | Gamma | `log` | estimated | positive continuous, right-skewed (cost, rainfall) |
| `binomial` | Binomial | `logit` (canonical) | fixed 1 | binary outcomes, proportions |
| `tweedie` | Tweedie `V=μ^p` | `log` | estimated | zero-inflated positive data (insurance claims) |
| `negative_binomial` | NB2 `V=μ+μ²/θ` | `log` | fixed 1 | overdispersed counts |
| `log_link` | Gaussian | `log` | estimated | positive continuous with multiplicative effects |
| `identity_link` | Gaussian | `identity` | estimated | exactly OLS, as a sanity check |
| `offsets` | Poisson | `log` | fixed 1 | rates with a known exposure |
| `dispersion` | quasi-Poisson | `log` | estimated | counts whose SEs must survive overdispersion |

The naming is deliberately operational, in the same style as `logistic_regression`:
`poisson` and `gamma` are families, `offsets` and `dispersion` are *modelling
situations*, and `log_link`/`identity_link` are link choices on a Gaussian family.
The table is the map; `identity_link` and `ols` agree to machine precision, and the
demo asserts it.

## Derivation

**Exponential family.** Write the density as

$$
f(y;\theta,\phi) = \exp\!\left(\frac{y\theta - b(\theta)}{\phi} + c(y,\phi)\right)
$$

Then `μ = b'(θ)` and `Var[y] = φ b''(θ) = φ V(μ)`. Every family above is this form
with a particular `b`: Poisson `b(θ)=e^θ`, Gamma `b(θ)=−log(−θ)`, Binomial
`b(θ)=log(1+e^θ)`, Gaussian `b(θ)=θ²/2`, Tweedie `b(θ)` given by the
Jørgensen parameterisation, and NB2 as a Poisson–Gamma mixture.

**Score.** With `η_i = x_iᵀβ + o_i`, `μ_i = g^{-1}(η_i)` and `dμ/dη = 1/g'(μ)`,

$$
\frac{\partial \ell}{\partial \beta}
= \sum_i \frac{y_i-\mu_i}{\phi}\,\frac{1}{g'(\mu_i)}\,x_i
$$

For the **canonical** link (`θ = η`), `dθ/dη = 1`, so the score simplifies to

$$
\nabla\ell = \frac{1}{\phi}X^\top(y-\mu)
$$

which is zero when `Xᵀ(y−μ)=0` — the residual is orthogonal to every column, exactly
as in OLS. The demo checks this identity for Poisson and binomial, and the analogous
`Xᵀ(μ(μ−y))=0` for a Gaussian log link.

**Fisher information.** The expected negative Hessian is

$$
\mathcal{I}(\beta) = \frac{1}{\phi}X^\top W X,
\qquad
W = \operatorname{diag}\!\left(\frac{(d\mu_i/d\eta_i)^2}{V(\mu_i)}\right)
= \operatorname{diag}\!\left(\frac{1}{V(\mu_i)\,g'(\mu_i)^2}\right)
$$

**IRLS.** Fisher scoring is `β ← β + I⁻¹∇ℓ`, which rearranges into a weighted least
squares problem:

$$
\beta_{\text{new}}
= \arg\min_\beta \sum_i w_i\big(z_i - x_i^\top\beta - o_i\big)^2,
\qquad
z_i = \eta_i + (y_i-\mu_i)\,g'(\mu_i)
$$

`z` is the *working response*: a first-order Taylor expansion of `g(y)` around `μ`.
For the identity link `z=y` and the loop converges in one step, which is why
`identity_link` is OLS. For a canonical link IRLS is Newton's method and converges
quadratically; for any other link it is Fisher scoring and converges linearly, which
is why the code backtracks.

**Deviance.** The fit is compared to the saturated model `μ=y`:

$$
D = 2\phi\big[\ell(y) - \ell(\mu)\big]
$$

which gives the four formulas in the code:

| Family | `D/2` |
|:--|:--|
| Poisson | `Σ [y log(y/μ) − (y−μ)]` |
| Gamma | `Σ [−log(y/μ) + (y−μ)/μ]` |
| Binomial | `Σ [y log(y/μ) + (1−y) log((1−y)/(1−μ))]` |
| NB2 | `Σ [y log(y/μ) − (y+θ) log((y+θ)/(μ+θ))]` |
| Gaussian | `Σ (y−μ)² / 2` |
| Tweedie `p∉{1,2}` | `Σ [y^{2−p}/((1−p)(2−p)) − yμ^{1−p}/(1−p) + μ^{2−p}/(2−p)]` |

**Dispersion.** When `φ` is unknown, the Pearson estimator is

$$
\hat\phi = \frac{1}{n-\operatorname{rank}(X)}\sum_i \frac{(y_i-\mu_i)^2}{V(\mu_i)}
$$

and `Cov(β̂) = φ̂ (XᵀWX)⁻¹`. Poisson and binomial fix `φ=1`; the `dispersion`
variant estimates it, which leaves the coefficients unchanged and multiplies every
standard error by `√φ̂`.

**Tweedie and NB.** Tweedie interpolates: `p=1` is Poisson, `p=2` is Gamma, and
`1<p<2` is a compound Poisson–Gamma with a point mass at zero — the shape of an
insurance portfolio (many zero claims, positive skewed payouts). NB2 is a Poisson
whose rate is Gamma-distributed, giving `V(μ)=μ+μ²/θ`; `θ→∞` recovers Poisson,
which the demo checks to `7.5e-9`.

**Why Gamma uses a log link.** The canonical link is `g(μ)=1/μ`, which is unstable:
`η` near zero sends `μ` to infinity and the working weights blow up. The log link is
non-canonical but numerically tame and multiplicative, which is how cost and rate
models are actually specified. That choice is why `gamma` is fitted by Fisher
scoring with a line search rather than by pure Newton.

## Code walkthrough

The file is one model plus a handful of family kernels. The variant table is the
only place where names are resolved:

```python
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    family, link = {
        "poisson": ("poisson", "log"),
        "gamma": ("gamma", "log"),
        ...
        "dispersion": ("poisson", "log"),
    }[variant]
    phi_known = family in ("poisson", "binomial", "negative_binomial")
    if variant == "dispersion":
        phi_known = False
```

Two things are decided here and nowhere else: which `(family, link)` pair the
variant means, and whether `φ` is known. `dispersion` deliberately starts from
Poisson and then *unfixes* `φ`, so its coefficients are identical to `poisson` and
only the covariance changes — the demo asserts both halves of that statement.

The stable sigmoid is the same branchwise form used in `logistic_regression.py`:

```python
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
```

`1/(1+e^{-z})` is safe for `z ≥ 0` and `e^z/(1+e^z)` for `z < 0`; evaluating either
formula on the whole array produces `inf/inf = nan` in one tail, and a `nan`
probability makes every weight and the deviance `nan` from the first iteration.
This is not cosmetic: it is what lets the binomial demo use logits of magnitude 5
without a guard.

The initial value is a smoothed response pushed through the link:

```python
    mu0 = _clip_mu(0.5 * (y + float(np.mean(y))), family)
    beta = np.linalg.lstsq(Xa, _link(mu0, link) - off, rcond=None)[0]
```

`(y+ȳ)/2` is strictly interior for binary `y` (hence a finite logit) and strictly
positive for counts; using `y` directly makes the logit infinite for every 0/1 row.
The regression is on `η₀ − offset`, because the offset is a known additive term and
must not be absorbed into `β`. For the identity link this start is already a scaled
OLS solution, and the first IRLS step lands exactly on OLS.

The loop body is IRLS in six lines:

```python
        eta = Xa @ beta + off
        mu = _clip_mu(_inv_link(eta, link), family)
        gp = _link_grad(mu, link)
        w = sw / (_variance(mu, family, power, theta) * gp * gp)
        w = np.maximum(w, 0.0)
        z = eta - off + (y - mu) * gp          # working response for beta only
        sqrt_w = np.sqrt(w)
        beta_new = np.linalg.lstsq(Xa * sqrt_w[:, None], z * sqrt_w, rcond=None)[0]
```

`w = 1/(V·g'²)` is Fisher information per observation, and `z` is the linearised
response. The subtraction of `off` is the offset's whole implementation: `eta`
contains it, but the regression being solved is for `Xβ`, so the target must not.
Scaling rows by `√w` turns weighted least squares into ordinary least squares on
`(√w X, √w z)`; using `lstsq` instead of forming `XᵀWX` avoids squaring the
condition number, and it returns the minimum-norm solution when the design is
rank-deficient.

Non-canonical links can overshoot, so every step is checked against the deviance:

```python
        step = 1.0
        for _ in range(40):
            cand = beta + step * (beta_new - beta)
            mu_c = _clip_mu(_inv_link(Xa @ cand + off, link), family)
            dev_c = _deviance(y, mu_c, family, power, theta)
            if np.isfinite(dev_c) and dev_c <= dev + 1e-12 * abs(dev):
                break
            step *= 0.5
        else:
            cand, dev_c = beta, dev
```

The `1e-12 * abs(dev)` slack is not a fudge: an accepted full step can raise the
deviance by one ulp, and without the slack the loop halves the step forever and
never converges. The `for ... else` runs only when no step was accepted, in which
case the old `β` is kept — Python's `else` on a loop is the cleanest way to express
"no break happened" and avoids a separate `accepted` flag.

The deviance dispatch handles Tweedie's boundaries before touching the generic
formula:

```python
    if power == 1.0:
        return _deviance(y, mu, "poisson", power, theta)
    if power == 2.0:
        return _deviance(y, mu, "gamma", power, theta)
```

The generic Tweedie expression divides by `(1−p)(2−p)`, so at exactly `p=1` or
`p=2` it is `0/0`. Delegating at the boundary is what makes `tweedie(power=1)`
produce *bit-identical* coefficients to `poisson` — the demo prints `0.0e+00`, not
"close enough". A tolerance-based comparison would hide the fact that the variance
and deviance functions must agree exactly at the limit.

NB2's deviance is written with `log1p` rather than a ratio of logs:

```python
        t = (np.where(y > 0.0, y * np.log(np.maximum(y, EPS) / mu), 0.0)
             - (y + theta) * np.log1p((y - mu) / (mu + theta)))
```

For large `θ`, `(y+θ)/(μ+θ) → 1` and `log` of that ratio loses most of its
significant digits; `log1p((y−μ)/(μ+θ))` is exact to first order in the small
quantity. Without it, the `θ=1e7` limit does not reproduce Poisson to `1e-8`, and
that is precisely the limit the demo uses to test the family.

Inference is separated from fitting and uses the rank, not the column count:

```python
    rank = int(np.linalg.matrix_rank(Xa))
    pearson = float(np.sum(sw * (y - mu) ** 2 / var))
    dispersion = 1.0 if phi_known else pearson / max(n - rank, 1)
    cov = dispersion * np.linalg.pinv(info)
```

`n − rank` is the correct residual degrees of freedom: a duplicated column consumes
a parameter but carries no information, and dividing by `n − d` would make `φ̂` too
large. `pinv` keeps the covariance finite for a rank-deficient design, where some
directions genuinely have no estimable variance. For the `dispersion` variant the
demo's overdispersed counts give `φ̂ = 1.659`, and every standard error is exactly
`√1.659 = 1.288` times the Poisson one — the assertion compares the ratio to
`√φ̂` to `1e-8`, which is the entire meaning of quasi-Poisson.

## Complexity

| Step | Cost |
|:--|:--|
| One IRLS iteration | `O(nd²)` for the weighted least squares, `O(nd)` for `μ, w, z` |
| Covariance | `O(nd² + d³)` once, at the end |
| Iterations | 4–10 canonical links; 5–30 non-canonical |

Memory is `O(nd)` for the design plus `O(d²)` for the covariance. The loop never
forms `XᵀWX`; `lstsq` works on the `n×d` weighted design, which is the same
asymptotic cost with better conditioning. For `d` in the thousands and `n` in the
millions, the design is the memory bottleneck, not the iterations.

## Failure modes

1. **The response is outside the family's support.** Negative counts, `y=0` for
   Gamma, or a binomial proportion above 1 produce `nan` in the deviance on the
   first iteration. Validate `y` before fitting; the fit itself cannot infer the
   support from an array.
2. **Overdispersion.** Poisson assumes `Var=μ`. Real counts often have
   `Var>μ`, so the Poisson SEs are too small and every `p`-value is optimistic.
   The `dispersion` variant estimates `φ` and fixes the inference; NB2 fixes both
   the mean and the variance *model*, which is stronger and usually better.
3. **Separation in the binomial model.** If a hyperplane separates successes from
   failures, `μ` tends to 0/1, the weights `μ(1−μ)` vanish, and `β` diverges. The
   likelihood has no finite maximiser; add an `L2` penalty (as
   `logistic_regression.separable_fix` does) or use Firth's method.
4. **A non-canonical link with a bad start.** Fisher scoring can step to a
   negative `μ` for the log link. The code's backtracking rejects such a step, but
   a custom loop without it will produce `log(negative) = nan`. This is the reason
   step-halving is in the loop rather than in a comment.
5. **Offsets on the wrong scale.** `offset` must be `log(exposure)`, not
   `exposure`. Passing exposure directly fits `η = Xβ + E`, which is a different
   model with a bizarre intercept. The demo's unoffset fit absorbs `0.6·x₀` into
   `β₀` and moves the slope by `0.607`; that is the size of the mistake.
6. **Reading `φ` as a goodness-of-fit statistic.** For Poisson and binomial the
   code fixes `φ=1` by design; a `deviance/(n−d)` ratio near 1 is a coincidence,
   not evidence that the model fits. Use Pearson residuals or a formal
   overdispersion test.
7. **Tweedie power outside `(1,2)`.** `p=0` is Gaussian, `p=1` Poisson, `p=2`
   Gamma, but the generic deviance formula is only valid for `p∉{0,1,2}` and
   requires `y>0` for `p>2`. Do not extrapolate the parameter without checking the
   support.
8. **Ignoring uncertainty in `θ`.** NB2 treats the size as known; estimating it
   (profile likelihood or a grid) and then fixing it understates the standard
   errors. The same applies to the Tweedie power.
9. **Interpreting the intercept under an offset.** With `η = β₀ + Xβ + log E`, the
   intercept is a log *rate* at `E=1`, not a log count. Report rates, or state the
   exposure explicitly.
10. **Weights versus IRLS weights.** The `weights` argument is a *prior* weight
    (e.g. inverse variance, trials). It enters `w` multiplicatively; it is not the
    same object as `V(μ)`, and passing one where the other is meant changes the
    estimand, not just the precision.

## Applications

- **Epidemiology and public health.** Poisson regression with `log(person-years)`
  as an offset is the standard incidence-rate model; the `offsets` variant is
  exactly that. Binomial with logit gives odds ratios for risk factors.
- **Insurance pricing.** Tweedie with `1<p<2` models aggregate claim cost: a point
  mass at zero plus a Gamma severity, one GLM instead of a two-part model. Claim
  counts use Poisson or NB2.
- **Ecology and genomics.** Counts of organisms, reads, or mutations are almost
  always overdispersed; NB2 or quasi-Poisson is the default, and the dispersion
  estimate is reported alongside the coefficients.
- **Customer spend and demand.** Gamma with a log link for positive continuous
  amounts (revenue per user, rainfall, time-to-event proxies); coefficients are
  multiplicative effects on the mean.
- **Credit and conversion.** Binomial logit for default and conversion; the GLM
  framing makes the linear predictor, the link, and the calibration explicit, and
  standard errors are available without a separate bootstrap.
- **A/B testing.** Rate metrics are Poisson/binomial GLMs with an offset for
  exposure; the coefficient on the treatment indicator is the log rate ratio, and
  its SE is the one the dispersion variant corrects.
- **As an OLS sanity check.** `identity_link` is OLS; if it does not match
  `ols.py`, the IRLS loop is wrong, not the statistics.

## References

- Nelder & Wedderburn, *Generalized Linear Models*, 1972. JRSS-A — the original
  IRLS formulation.
- McCullagh & Nelder, *Generalized Linear Models*, 2nd ed., 1989 — the canonical
  reference for deviance, links and dispersion.
- Green, *Iteratively Reweighted Least Squares for Maximum Likelihood Estimation*,
  1984. JRSS-B — convergence and the non-canonical link case.
- Jørgensen, *Exponential Dispersion Models*, 1987. JRSS-B — Tweedie families and
  the variance-power parameterisation.
- Hilbe, *Negative Binomial Regression*, 2nd ed., 2011 — NB1/NB2 and the
  Poisson–Gamma mixture.
- Dunn & Smyth, *Generalized Linear Models With Examples in R*, 2018 — offsets,
  dispersion and practical diagnostics.
- Agresti, *Categorical Data Analysis*, 3rd ed., 2013 — binomial and ordinal GLMs
  in the applied setting.
