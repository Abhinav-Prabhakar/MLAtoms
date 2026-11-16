# Bayesian inference

`bayesian_inference.py` · Probabilistic models · Tier 1 · 7 variants

## Intuition

Bayes' theorem is a division. The interesting part is that for a few
prior–likelihood pairs the division closes: the posterior comes back in the same
family as the prior, and "doing Bayesian inference" becomes addition.

$$\underbrace{p(\theta \mid y)}_{\text{posterior}} \;\propto\;
\underbrace{p(\theta)}_{\text{prior}} \times
\underbrace{p(y \mid \theta)}_{\text{likelihood}} \qquad (1)$$

Conjugacy is the property that makes this exact and cheap. A Gaussian prior with
a Gaussian likelihood returns a Gaussian; a Beta prior with Bernoulli data
returns a Beta; a Dirichlet with categorical data returns a Dirichlet. The
posterior's parameters are the prior's parameters plus a sufficient statistic —
literally `a + successes`, `α + counts`. No sampler, no optimiser, no
approximation, and the answer is the whole distribution rather than a point.

That changes what you can ask. A point estimate answers "what value?"; a
posterior answers "what values, and how sure?", which is what a decision rule
needs. A credible interval is read directly off the distribution, and the
predictive distribution of the next observation is a different (wider) object
than the posterior of the parameter.

The MLE and the MAP sit at the edges of this family: the MLE is the posterior
mode under a flat prior, and the MAP is a penalised likelihood with the
negative log-prior as the penalty. Writing both as methods of the same object
makes the relationship visible rather than rhetorical.

## Variants

| Variant | Family | Posterior | What it computes |
|:--|:--|:--|:--|
| `mle` | normal | — | `\bar y`; the likelihood with no prior |
| `map` | normal | `N(μ_n, σ_n^2)` | the mode: prior and data weighted by precision |
| `conjugate_normal` | normal | `N(μ_n, σ_n^2)` | the full Gaussian posterior |
| `beta_binomial` | beta | `\mathrm{Beta}(a+k, b+n-k)` | Bernoulli data, Beta prior, predictive pmf |
| `dirichlet_multinomial` | dirichlet | `\mathrm{Dir}(α + c)` | categorical data, one update by addition |
| `credible_intervals` | any | the above | equal-tailed and HPD intervals |
| `posterior_predictive` | any | the above | the distribution of the *next* draw |

The two interval/predictive variants are not new models — they are what you do
with a posterior once you have one. Pass `family="beta"` to run them on the
discrete posterior; the HPD machinery is where the two families differ
substantially.

## Derivation

### Normal–normal: precisions add

Take `μ \sim N(μ_0, σ_0^2)` and `y_i \mid μ \sim N(μ, σ^2)` with `σ^2` known.
The log posterior is quadratic in `μ`, so the posterior is Gaussian; completing
the square gives

$$\frac{1}{σ_n^2} = \frac{1}{σ_0^2} + \frac{n}{σ^2}, \qquad
μ_n = σ_n^2\left(\frac{μ_0}{σ_0^2} + \frac{n\bar y}{σ^2}\right). \qquad (2)$$

**Precisions add, and the mean is the precision-weighted average.** Written as a
shrinkage toward the prior,

$$μ_n = (1-w)\,μ_0 + w\,\bar y, \qquad w = \frac{nσ_0^2}{nσ_0^2 + σ^2}, \qquad (3)$$

the weight on the data is the fraction of total precision the data contributes.
`n = 0` gives `w = 0` (the prior is the posterior); `σ_0 → ∞` gives `w → 1` (the
MLE). Nothing about this depends on the shape of the prior beyond its first two
moments, which is why Gaussian priors are the default even when they are not
literally believed.

### MAP is a penalised likelihood

Maximising (1) instead of integrating it gives the mode. For the Gaussian case
the log-prior is `-(μ-μ_0)^2/2σ_0^2`, so

$$\hat μ_{\text{MAP}} = \arg\min_μ\;
\frac{1}{2σ^2}\sum_i (y_i-μ)^2 + \frac{1}{2σ_0^2}(μ-μ_0)^2, \qquad (4)$$

which is ridge regression with `λ = σ^2/σ_0^2`. Every "regularised" estimator is
a MAP estimate under some prior; the regulariser *is* the prior, and its strength
is the prior's precision. For a symmetric posterior the MAP and the posterior
mean coincide, which is why (2) and (4) give the same number — and why the
distinction between them only becomes visible in skewed families like Beta.

### Beta–Bernoulli: counting

With `p \sim \mathrm{Beta}(a,b)` and `k` successes in `n` Bernoulli trials,

$$p \mid y \sim \mathrm{Beta}(a+k,\; b+n-k), \qquad
\mathbb{E}[p \mid y] = \frac{a+k}{a+b+n}. \qquad (5)$$

`a` and `b` are **pseudo-counts**: a `\mathrm{Beta}(2,2)` prior is worth two
successes and two failures you did not have to collect. That is also the honest
way to see what a prior costs — with `n = 10` it is a fifth of the data; with
`n = 10^6` it is invisible.

The *posterior predictive* for the next observation is not `p` evaluated at its
mean; it is the posterior averaged over `p`:

$$P(y_{n+1}=1 \mid y) = \mathbb{E}[p \mid y] = \frac{a+k}{a+b+n}, \qquad (6)$$

and for `m` future trials the count follows a beta-binomial,

$$P(K=k) = \binom{m}{k}\frac{B(a'+k,\; b'+m-k)}{B(a',b')}, \qquad (7)$$

with `a' = a+k`, `b' = b+n-k`. Its variance is the binomial variance **inflated**
by the posterior spread:

$$\mathrm{Var}[K] = \frac{m\,a'b'\,(a'+b'+m)}{(a'+b')^2(a'+b'+1)}, \qquad (8)$$

which exceeds `m\,p̂(1-p̂)` whenever `a'+b'` is finite. Using a binomial
predictive after a Beta fit understates the uncertainty, and with small `n` it
understates it badly — that is the beta-binomial's entire reason to exist.

### Dirichlet–categorical: counting in a vector

The same update, componentwise: `α \leftarrow α + c`, where `c` is the vector of
counts. The marginal of each coordinate is `\mathrm{Beta}(α_i, α_0-α_i)`, so
every Beta result above applies per category, and the predictive for one new
draw is

$$P(z_{n+1}=i \mid c) = \frac{α_i + c_i}{α_0 + n},
\qquad \mathrm{Var} = \frac{p_i(1-p_i)}{α_0+n+1}, \qquad (9)$$

which is the multinomial variance divided by `α_0+n+1` — again, the posterior
spread entering the predictive.

### Credible intervals, and what they are not

A `95%` equal-tailed interval puts `2.5%` of the posterior mass in each tail; the
highest-posterior-density (HPD) interval is the *shortest* interval with `95%`
mass, defined by `f(lo) = f(hi)`. They coincide for any symmetric unimodal
posterior, which is why the normal case needs no search and the Beta case does.

A `95%` credible interval is not a `95%` confidence interval, and the difference
is not academic. The credible interval says "given the prior, `θ` is in here with
probability `0.95`" — a statement about `θ`. The confidence interval says "this
procedure covers the truth `95%` of the time" — a statement about the procedure.
They agree exactly in one important case: a **flat prior** on a location
parameter, where the posterior is proportional to the likelihood. Then the
Bayesian interval is the frequentist one, and the demo checks the coverage
empirically at `94%` over 500 replications.

With an informative prior they diverge, and the direction is predictable: the
credible interval is centred near the prior, so it under-covers when the prior is
wrong and over-covers when it is right. That is not a bug, it is the cost of
using information you did not collect.

### Why the predictive is wider than the posterior

The posterior describes `θ`; the predictive describes a future `y`. For the
normal case the two differ by exactly the observation variance:

$$y_{n+1} \mid y \;\sim\; N\!\left(μ_n,\; σ^2 + σ_n^2\right). \qquad (10)$$

Plugging `\hat θ` into the likelihood instead — the "plug-in predictive" — gives
`N(μ_n, σ^2)` and is over-confident by an amount that vanishes as `n → ∞` but
dominates for small `n`. The gap `σ_n^2` is the uncertainty you paid data to
remove, and it belongs in the forecast.

## Code walkthrough

The only piece that is not textbook algebra is the Beta CDF, and without scipy
it has to be written. `betainc` uses the standard split — a power series near
zero and a continued fraction near one — because the continued fraction
converges slowly where the series is fast, and vice versa. The Lentz iteration
is what makes the continued fraction numerically stable:

```python
    for m in range(1, 300):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
```

`betainc` chooses the direction by the standard criterion `x < (a+1)/(a+b+2)`:
evaluate the continued fraction for `I_x` when `x` is small, and for `1-I_{1-x}`
when it is large, so the fraction always runs on the side where it converges:

```python
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b
```

The quantile is a bisection on a monotone function. 200 iterations is far more
than needed for `1e-14`, and bisection cannot fail the way Newton can when the
density vanishes at an endpoint — which it does for every `a < 1`:

```python
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)
```

The normal fit is equation (2) written once. Note that `post_var_` is computed
first and then used for the mean: computing the mean first and the variance
after invites the two to disagree when one is edited:

```python
            prec = 1.0 / sigma0 ** 2 + self.n_ / self.sigma ** 2
            self.post_var_ = 1.0 / prec
            self.post_mean_ = self.post_var_ * (
                mu0 / sigma0 ** 2 + self.n_ * self.mean_ / self.sigma ** 2)
            self.post_sd_ = math.sqrt(self.post_var_)
```

The Beta MAP needs two guards that the formula hides. The mode `(a-1)/(a+b-2)`
exists only for `a, b > 1`; when either is at or below one the density is
unbounded at an endpoint and the mode is `0` or `1`, not the ratio:

```python
            if self.family == "beta":
                a, b = self.post_a_, self.post_b_
                if a <= 1.0:
                    return 0.0
                if b <= 1.0:
                    return 1.0
                return (a - 1.0) / (a + b - 2.0)
```

The HPD interval is a one-dimensional root-find. `lo` moves right, `hi` is
recovered from the mass constraint, and the objective is the density gap. The
search range matters: `lo` can run up to `F^{-1}(1-\text{level})`, not to
`1-\text{level}`, because a concentrated posterior's shortest interval may start
far from zero:

```python
        def gap(lo: float) -> float:
            hi = beta_ppf(betainc(a, b, lo) + level, a, b)
            return beta_pdf(hi, a, b) - beta_pdf(lo, a, b)

        left, right = 1e-12, beta_ppf(1.0 - level, a, b)
        for _ in range(200):
            mid = 0.5 * (left + right)
            if gap(mid) > 0.0:          # gap decreases in `lo`: root is right
                left = mid
            else:
                right = mid
```

The beta-binomial variance is the one formula people get wrong, because the
naive answer is the binomial one. The extra `(a'+b'+m)/(a'+b'+1)` factor is the
whole point: it is the inflation due to not knowing `p`:

```python
        if self.family == "beta":
            a, b = self.post_a_, self.post_b_
            tot = a + b
            mean = n * a / tot
            # beta-binomial: the binomial variance inflated by the Beta spread
            var = n * a * b * (tot + n) / (tot ** 2 * (tot + 1.0))
            return mean, var
```

The discrete pmf is computed entirely in logs and exponentiated once, because
`\binom{m}{k}` overflows immediately for `m = 40` and the Beta functions
underflow at the same time:

```python
            logpmf = (np.array([math.lgamma(n + 1) - math.lgamma(i + 1)
                                - math.lgamma(n - i + 1) for i in k])
                      + np.array([log_beta(a + i, b + n - i) for i in k])
                      - log_beta(a, b))
            return np.exp(logpmf)
```

## Complexity

`n` observations, `K` categories, `m` future trials.

| Operation | Normal | Beta | Dirichlet |
|:--|:--|:--|:--|
| `fit` | `O(n)` | `O(n)` | `O(n + K)` |
| `estimate` | `O(1)` | `O(1)` | `O(K)` |
| `credible_interval` | `O(1)` | `O(200 · cf)` | `O(200 · cf)` |
| `predictive_pmf` | `O(1)` per point | `O(m · cf)` | `O(K)` |

Everything is linear in the data because the posterior depends on the data only
through sufficient statistics — that is what conjugacy buys. The interval costs
are bisection iterations times a continued fraction, which is why they are
constant-bounded but not `O(1)` in the same sense as the estimate. A sampler
would be `O(T)` with `T` in the thousands and an `O(1/T)` error; for these three
families it would be strictly worse.

## Failure modes

**The prior does the work when `n` is small.** With `n = 3`, a `\mathrm{Beta}(20,20)`
prior outvotes the data entirely, and the posterior sits near `0.5` no matter
what was observed. The diagnostic is the effective sample size `a+b` against
`n`; the fix is a weaker prior, not a different estimator.

**A credible interval is not a confidence interval.** Under an informative prior
the 95% credible interval can cover the truth 70% of the time if the prior is
wrong, and 99% if it is right. The diagnostic is a simulation with a known truth
(the demo does exactly this); the fix is either a flat prior or an explicit
sensitivity analysis across priors — not a relabelling of the interval.

**MAP is not invariant to reparameterisation.** A uniform prior on `σ` and a
uniform prior on `σ^2` are different priors, and they give different MAPs for the
same data because the Jacobian differs. The posterior *distribution* is invariant
when you transform it; a mode is not, when the prior was declared in the other
coordinates. The diagnostic is a transform of the estimate that does not match
the estimate of the transform. Posteriors are invariant, modes are not — which is
another reason to report the interval.

**An improper flat prior can give an improper posterior.** `σ_0 = 10^6` is a
numerical stand-in for "no prior", and it works here because the normal
likelihood is proper. In a hierarchical model the same trick produces a
posterior that is not integrable, and the sampler wanders off. The diagnostic is
a posterior that does not stabilise as the chain lengthens.

**Binomial instead of beta-binomial.** Fitting `p` with a Beta and then
forecasting with `\mathrm{Binomial}(m, \hat p)` drops the `(a'+b'+m)/(a'+b'+1)`
factor and understates the variance of every count forecast. The diagnostic is
predictive coverage — the nominal 95% band contains the outcome far less than
95% of the time. This is the most common error in A/B-test forecasting.

## Applications

A/B testing is the canonical use: a Beta posterior on the conversion rate, a
credible interval on the lift, and a beta-binomial predictive for the rest of the
month. The `beta_binomial` variant is also the standard model for over-dispersed
counts — anything counted with clustering (defects per batch, goals per match)
where the binomial variance is too small.

The conjugate normal is the basis of Kalman filtering (`kalman.py` is this
update, applied recursively), of empirical Bayes in hierarchical models, and of
shrinkage estimators like the James–Stein estimator. The Dirichlet update is the
M-step of latent Dirichlet allocation and the foundation of `gmm.py`'s Bayesian
cousin.

Where it is a trap: any model where the posterior is not conjugate and a sampler
is needed (the conjugacy here is a convenience, not a law); any decision that
depends on the tails of the posterior, where the prior's shape matters more than
its moments; and any setting where the parameter is high-dimensional — the
Dirichlet is the only member here that scales to more than a handful of
parameters.

## References

- A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari and D. B. Rubin,
  *Bayesian Data Analysis*, 3rd ed., CRC 2013 — chapters 2 and 3 are the
  conjugate models here, and chapter 5 is the prior-sensitivity argument.
- K. P. Murphy, *Machine Learning: A Probabilistic Perspective*, MIT 2012 —
  §3.4 and §3.5 cover the exponential-family view and the Dirichlet in the
  notation most code uses.
- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer 2006 —
  §2.3 for the Gaussian–Gaussian case and the predictive distribution.
- C. P. Robert, *The Bayesian Choice*, 2nd ed., Springer 2007 — decision-theoretic
  treatment, including why MAP is not invariant and what to use instead.
- P. Diaconis and D. Ylvisaker, *Conjugate priors for exponential families*,
  Annals of Statistics 7 (1979) — why the conjugate families are exactly the ones
  that are linear in the sufficient statistics.
- E. T. Jaynes, *Probability Theory: The Logic of Science*, Cambridge 2003 —
  chapter 12 on the relation between credible and confidence intervals under a
  flat prior, which is the demo's 94% check.
