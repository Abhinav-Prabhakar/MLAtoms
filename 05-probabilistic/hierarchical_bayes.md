# Hierarchical Bayes

`hierarchical_bayes.py` · Probabilistic models · Tier 2 · 5 variants

## Intuition

You have measurements from a dozen groups — schools, stores, patients, A/B test
cells — and you want each group's mean. Two obvious estimators, both wrong in an
instructive way.

Pool everything: one grand mean, applied to every group. Unbiased if the groups
really are identical, hopeless if they are not. Use each group's own data: the
mean of four observations, with a standard error of $\sigma/2$. Unbiased, and so
noisy that the ranking of twelve groups is mostly noise.

The hierarchical model says: the group means are themselves drawn from a
distribution. That single assumption produces a third estimator which is neither,
and which is better than both.

$$y_{ij} = \mu + a_j + \varepsilon_{ij}, \qquad a_j \sim N(0, \tau^2), \qquad \varepsilon_{ij} \sim N(0, \sigma^2) \qquad (1)$$

$\tau^2$ is how different the groups really are; $\sigma^2$ is how noisy one
observation is. Given those two numbers, the posterior mean of group $j$'s mean
is a weighted average of the grand mean and the group's own average:

$$\hat\theta_j = \lambda_j\,\mu + (1-\lambda_j)\,\bar y_j, \qquad \lambda_j = \frac{\sigma^2/n_j}{\tau^2 + \sigma^2/n_j} \qquad (2)$$

Read the weight. When the group has lots of data ($n_j$ large) or the groups
really differ ($\tau^2$ large), $\lambda_j \to 0$ and you trust the group. When
the data are noisy or the groups are nearly identical, $\lambda_j \to 1$ and you
trust the pool. The estimate is *shrunk* toward the centre by exactly the amount
the model says is justified.

This is the whole atom. The variants are the three special cases and the two ways
to fill in $\tau^2$ and $\sigma^2$: closed-form shrinkage with the variances
known (`partial_pooling`), the two limits (`no_pooling`, `full_pooling`),
empirical Bayes by EM (`empirical_bayes`), and a sampler that puts a prior on the
variances too (`noncentered`).

The demo's headline is the risk comparison. With twelve groups of four, and a
true $\tau^2 = 0.36$ against $\sigma^2 = 1$:

| Estimator | MSE against the true group means |
|:--|--:|
| no pooling | 0.2609 |
| full pooling | 0.1868 |
| partial pooling ($\lambda = 0.410$) | **0.1263** |

The shrinkage estimator beats both extremes, and it is not a lucky draw: it is
the Bayes estimator under (1), so no other function of the data has lower
expected squared error when the model is true.

## Variants

| Variant | $\tau^2, \sigma^2$ | Estimate of $\theta_j$ | Use when |
|:--|:--|:--|:--|
| `partial_pooling` | given | $\lambda_j\mu + (1-\lambda_j)\bar y_j$ | you know the variance components, or are testing the shrinkage itself |
| `no_pooling` | — | $\bar y_j$ | groups are genuinely unrelated; the baseline the other two must beat |
| `full_pooling` | $\tau^2 = 0$ | $\bar y$ for every group | the groups are exchangeable and identical |
| `empirical_bayes` | estimated by EM | plug-in shrinkage | the standard default: two unknowns, fitted by maximum marginal likelihood |
| `noncentered` | sampled | posterior mean over draws | you want the uncertainty in $\tau^2$ and $\sigma^2$, not a point plug-in |

The middle three are one line each and exist to be compared against. The
interesting pair is `empirical_bayes` — which treats $(\tau^2, \sigma^2)$ as
fixed at their MLEs and is therefore overconfident, because it ignores its own
uncertainty in them — and `noncentered`, which integrates over them.

Then there is the parameterisation. `noncentered` is named for a numerical trick,
not a model: write $a_j = \tau z_j$ with $z_j \sim N(0, 1)$ and the sampler's
geometry stops depending on $\tau$. Same posterior, but a sampler that can
traverse it when $\tau$ is small. The demo measures the difference in effective
sample size: 2839 against 96 for $\log\tau$ over 3000 draws, at identical cost.

## Derivation

### 1. The shrinkage weight

Condition on $a_j$. The group mean is Gaussian around the group effect,

$$\bar y_j \mid a_j \;\sim\; N\!\left(\mu + a_j,\; \sigma^2/n_j\right) \qquad (3)$$

and the prior is $a_j \sim N(0, \tau^2)$. Two Gaussians make a Gaussian, with
precision the sum of precisions:

$$P_j = \frac{n_j}{\sigma^2} + \frac{1}{\tau^2}, \qquad \mathbb{E}[a_j \mid y] = \frac{n_j(\bar y_j - \mu)/\sigma^2}{P_j}, \qquad \mathrm{Var}(a_j \mid y) = \frac{1}{P_j}$$

Adding $\mu$ back and rearranging gives (2). Two identities are worth keeping:

$$1 - \lambda_j = \frac{n_j/\sigma^2}{P_j}, \qquad \mathrm{Var}(\theta_j \mid y) = \frac{1}{P_j} = (1-\lambda_j)\,\frac{\sigma^2}{n_j}$$

The second is the one that matters downstream: the posterior variance is the
sampling variance discounted by the same factor that shrinks the mean. Borrowing
strength buys precision, not just a better point estimate.

### 2. Why it beats both extremes

Write the risk of an estimator $\lambda\mu + (1-\lambda)\bar y_j$, averaging over
the prior on $\theta_j$ and the sampling noise. The two error terms are
independent, so

$$R(\lambda) = \lambda^2\tau^2 + (1-\lambda)^2\frac{\sigma^2}{n_j} \qquad (4)$$

a quadratic in $\lambda$. Minimise it:

$$\lambda^\star = \frac{\sigma^2/n_j}{\tau^2 + \sigma^2/n_j}, \qquad R(\lambda^\star) = \frac{\tau^2\sigma^2/n_j}{\tau^2 + \sigma^2/n_j} \;<\; \min\!\left(\tau^2,\; \frac{\sigma^2}{n_j}\right) \qquad (5)$$

So the optimal weight is exactly the posterior weight (2), and the risk at the
optimum is *below both endpoints* — the harmonic-mean-looking expression is
smaller than either argument. That is Stein's phenomenon in its simplest
hierarchical form, and it is why the demo's MSE ordering is a theorem rather
than a coincidence.

With the demo's numbers, $\sigma^2/n = 0.25$ and $\tau^2 = 0.36$ give
$\lambda = 0.410$: the estimate is 59% the group's own mean, 41% the grand mean.

Three of the twelve groups had a sample variance below $\sigma^2$. For those
groups `no_pooling` is *more* confident than the hierarchical answer, because a
single $s_j^2$ is itself a noisy estimate of $\sigma^2$ and happened to come out
low. The pooled estimate wins on average precisely because it does not chase
that noise.

### 3. Marginal likelihood and empirical Bayes

To fit $\tau^2$ and $\sigma^2$, integrate the group effects out — they are
Gaussian, so this is exact. Two sufficient statistics survive, and they are
independent:

$$\bar y_j \sim N\!\left(\mu,\; \tau^2 + \frac{\sigma^2}{n_j}\right), \qquad \frac{ss_j}{\sigma^2} \sim \chi^2_{n_j-1}, \qquad ss_j = \sum_i (y_{ij} - \bar y_j)^2 \qquad (6)$$

so the marginal log-likelihood is a sum of two easy pieces:

$$\log p(y) = \sum_j\left[-\tfrac12\log\!\left(2\pi(\tau^2 + \sigma^2/n_j)\right) - \frac{(\bar y_j - \mu)^2}{2(\tau^2 + \sigma^2/n_j)}\right] + \sum_j\left[-\tfrac{n_j-1}{2}\log(2\pi\sigma^2) - \frac{ss_j}{2\sigma^2}\right] \qquad (7)$$

This is the objective `empirical_bayes` maximises, and it is also the quantity
reported as `loglik_`. The second bracket is the part that is easy to forget —
drop it and $\sigma^2 \to 0$ becomes a free lunch, as Section 4 of the failure
modes describes.

EM alternates the posterior of the missing $a_j$ with the M-step that maximises
the expected complete-data log-likelihood. With $\theta_j$ the posterior mean of
$a_j$ and $v_j$ its variance:

$$\mu = \frac{1}{N}\sum_j n_j(\bar y_j - \theta_j), \qquad \tau^2 = \frac{1}{J}\sum_j\left[(\theta_j - \mu)^2 + v_j\right] \qquad (8)$$

$$\sigma^2 = \frac{1}{N}\sum_j\left[ss_j + n_j\left((\bar y_j - \mu - \theta_j)^2 + v_j\right)\right] \qquad (9)$$

Every term has a job. The $v_j$ in (8) is what stops $\tau^2$ collapsing to zero:
the effects are not observed, and their posterior variance is part of their
second moment. The $(\bar y_j - \mu - \theta_j)^2$ in (9) is what stops $\sigma^2$
being biased low: the residual of the group mean around its fitted value is
still a residual. In the demo, EM raises the marginal likelihood from $-60.497$
to $-59.917$ monotonically, and lands at $\tau^2 = 0.259$, $\sigma^2 = 0.830$
against the truth $0.360$ and $1.000$ — both within sampling error at twelve
groups.

### 4. Centered and non-centered

Write the group effect two ways:

$$a_j \sim N(0, \tau^2) \qquad\text{or}\qquad a_j = \tau z_j, \quad z_j \sim N(0, 1) \qquad (10)$$

They describe the same distribution, and the joint density in
$(\mu, z, \log\tau, \log\sigma)$ is the same function of the data. But the
*geometry* is not. In the centered form the conditional precision of $a_j$ is

$$P_j = \frac{n_j}{\sigma^2} + \frac{1}{\tau^2} \qquad (11)$$

so as $\tau \to 0$ the coordinate $a_j$ becomes arbitrarily stiff, while its
typical value is of order $\tau$ — a neck of width $\tau$ inside a funnel of
height $1/\tau^2$. A leapfrog step small enough for the neck cannot cross the
mouth; a step large enough for the mouth overshoots the neck. In the non-centered
form $z_j$ is always $O(1)$ and $\tau$ enters only through the likelihood, so the
curvature stays bounded as $\tau \to 0$.

The demo samples both with the same kernel, the same step size (0.15, sixteen
leapfrog steps), the same 500 warm-up draws, and reports the ESS of $\log\tau$:
**2839 non-centered against 96 centered** out of 3000 draws. The centered
sampler is not wrong — it is exact, just catastrophically autocorrelated, which
is what an ESS of 3% means.

The target includes priors, since $\tau$ and $\sigma$ are now parameters:

$$\mu \sim N(0, 5^2), \qquad \log\tau \sim N(0, 2^2), \qquad \log\sigma \sim N(0, 2^2) \qquad (12)$$

Weak on purpose. A tight prior on $\log\tau$ would decide the variance
components, and then the comparison between parameterisations would be measuring
the prior rather than the sampler.

### 5. Checking the sampler against the exact answer

The group effects can be integrated out analytically, so the marginal posterior
over $(\mu, \log\tau, \log\sigma)$ is a three-dimensional integral of (7) times
the priors. The demo evaluates it on a $61 \times 111 \times 81$ grid and uses it
as an independent answer:

| Quantity | Sampler (3000 draws) | Exact grid |
|:--|--:|--:|
| $\mathbb{E}[\tau^2]$ | 0.097 | 0.108 |
| $\mathbb{E}[\sigma^2]$ | 1.710 | 1.726 |
| group means | — | max difference 0.015 |

The grid never touches the sampler, so this is evidence rather than a tautology.
It also surfaces something a single number would hide: the data put
$\tau^2 \approx 0.11$, not the true $0.0225$. With ten groups of two observations,
$\tau$ is barely identified — the spread of the group means is dominated by
$\sigma/\sqrt{n}$ — and that weak identification is exactly why the funnel is
severe and why the parameterisation matters.

## Code walkthrough

The public surface is one frozenset and a class whose `fit` dispatches on it.

```python
VARIANTS = frozenset(
    ["partial_pooling", "full_pooling", "no_pooling", "empirical_bayes", "noncentered"]
)
```

Three of the five branches are one-liners, and they are worth reading as the
definitions they are: no pooling reports the group mean and its own standard
error; full pooling reports the grand mean for everyone, with the variance of
*that* estimate, $\sigma^2/N$; partial pooling calls the shrinkage routine.

```python
        elif self.variant == "empirical_bayes":
            self._em(ybar, n, ss)
            self._shrink(ybar, n, ss, mu=self.mu_)
        elif self.variant == "noncentered":
            self._hmc(ys, centered=False)
        else:
            self.tau2_, self.sigma2_ = self.tau2, self.sigma2
            self._shrink(ybar, n, ss)
```

`_shrink` is (2) and its variance identity in five lines. The clamp is the
$\tau^2 \to 0$ limit: empirical Bayes can legitimately return zero between-group
variance, and the correct answer there is full pooling ($\lambda = 1$), not a
division by zero.

```python
        tau2 = max(self.tau2_, 1e-12)
        prec = n / self.sigma2_ + 1.0 / tau2
        self.group_mean_ = (n * ybar / self.sigma2_ + self.mu_ / tau2) / prec
        self.group_var_ = 1.0 / prec
        self.lambda_ = (self.sigma2_ / n) / (tau2 + self.sigma2_ / n)
```

The EM loop is (8)–(9). The E-step computes the posterior of $a_j$, not of the
group mean: `theta` is the deviation from $\mu$, which is why the M-step for
$\mu$ subtracts it rather than averaging it.

```python
            prec = n / sigma2 + 1.0 / tau2
            var = 1.0 / prec
            theta = (n * (ybar - mu) / sigma2) / prec
```

The M-step is where the two easily-dropped terms live. Both are in the code, and
both have a comment, because each one alone produces a plausible-looking fit that
is quietly wrong.

```python
            mu = float(np.sum(n * (ybar - theta)) / N)
            tau2 = float(np.mean((theta - mu) ** 2 + var))
            sigma2 = float(np.sum(ss + n * ((ybar - mu - theta) ** 2 + var)) / N)
```

The marginal likelihood is (7), and it is the same function the EM is
monotone in — which is what makes the demo's monotonicity check meaningful
rather than decorative.

```python
    var = tau2 + sigma2 / n
    out = float(np.sum(-0.5 * (LOG2PI + np.log(var)) - 0.5 * (ybar - mu) ** 2 / var))
    if ss is not None:
        out += float(np.sum(-0.5 * (n - 1.0) * (LOG2PI + math.log(sigma2))
                            - 0.5 * ss / sigma2))
```

`_model` is the sampler's target, and the one line that is not obvious is the
normalisation: `-N * ls`, not `-N/2 * ls`, because `ls` is $\log\sigma$ and
there is one factor of $\sigma$ per observation. The `ss_total` term is the
within-group sum of squares; without it the posterior is improper.

```python
        ll = (-0.5 * N * LOG2PI - N * ls
              - 0.5 * (np.sum(n * resid**2) + ss_total) / sigma2)
        ll += -0.5 * J * LOG2PI - 0.5 * np.sum(eff**2) / tau2
        if centered:
            ll -= J * lt
```

The gradient is written out by hand, in both parameterisations, because the whole
point is that they differ. In the centered branch the prior contributes
$-a_j/\tau^2$; in the non-centered branch the Jacobian of $a = \tau z$ cancels
the $-\log\tau$ of the prior, so $z_j$ is pulled to zero by $-z_j$ alone and
$\tau$ appears only through the likelihood.

```python
        if centered:
            # `a` is the effect itself; the prior pulls it to 0 with precision
            # 1/τ², so a small τ makes this coordinate stiff — the funnel.
            g[1:1 + J] = n * resid / sigma2 - a / tau2
            g[-2] = -0.25 * lt - J + np.sum(a**2) / tau2
        else:
            # z = a/τ is N(0, 1), so its prior term has no τ in it at all: the
            # log τ from the change of variables cancels the -log τ in N(0, τ²)
            g[1:1 + J] = tau * n * resid / sigma2 - a
            g[-2] = -0.25 * lt + tau * np.sum(n * resid * a) / sigma2
```

Both closures are handed to the same kernel, with the same step size. That is
what makes the ESS comparison a comparison of *geometry* and not of tuning.

```python
        sampler = MCMC(log_prob, variant="hmc", grad=grad, step_size=0.15,
                       n_leapfrog=16, n_warmup=self.n_warmup)
        chain = sampler.sample(self.n_samples, x0, rng=np.random.default_rng(self.seed))
```

The ESS call has one trap: `mcmc.ess` wants `(n_chains, n_samples)`, so a single
coordinate is passed as `chain[None, :, k]`. Passing `chain[:, k][:, None]`
instead makes it think there are 3000 chains of length 1 and it returns the
number of draws, which looks like perfect mixing.

```python
        self.ess_ = np.array([ess(chain[None, :, k]) for k in range(chain.shape[1])])
```

## Complexity

With $J$ groups, $N$ observations, and $N = \sum_j n_j$:

| Variant | Time | Memory |
|:--|:--|:--|
| `no_pooling` | $O(N)$ | $O(J)$ |
| `full_pooling` | $O(N)$ | $O(J)$ |
| `partial_pooling` | $O(N + J)$ | $O(J)$ |
| `empirical_bayes` | $O(IJ)$ for $I$ EM sweeps | $O(J)$ |
| `noncentered` | $O(T L (J+3))$ for $T$ draws, $L$ leapfrogs | $O(T(J+3))$ |

The closed forms are linear in the data and effectively free. EM is linear in the
number of groups per sweep and converged in 5 sweeps here, 60 in the
hand-rolled monotonicity check that starts from $\tau^2 = \sigma^2 = 1$ — the
number of sweeps is set by how far the start is from the answer, not by $J$.

The sampler is the expensive one: 3000 draws plus 500 warm-up, 16 leapfrog steps
each, on a parameter of dimension $J + 3 = 13$, so about 730,000 gradient
evaluations per chain and two chains in the demo. That is the price of
integrating over $(\tau^2, \sigma^2)$ instead of plugging in their MLEs, and it is
why `empirical_bayes` is the default and `noncentered` is the honest answer when
you can afford it.

The grid check is $O(A B C J)$: $61 \times 111 \times 81 \times 10 \approx 5.5$
million evaluations, vectorised in numpy, about 0.1 s. It is affordable only
because the model has three scalar parameters after marginalising the groups; the
same trick in a general hierarchical model is what INLA and Laplace
approximations are for.

## Failure modes

**Dropping the within-group sum of squares.** The first version of the sampler
targeted the group means only, as if the data were one observation per group.
That makes the posterior improper: send $\sigma \to 0$ and every group effect to
its group mean, and the log density diverges to $+\infty$ while the prior
penalty stays finite. HMC does exactly what the target says and runs down the
ridge, ending at $\sigma^2 \approx 0.01$ with a 3% acceptance rate. The fix is
one term, `ss_total`, and it is the difference between a proper posterior and a
divergence. *Diagnostic:* the sampler's $\log\sigma$ quantiles. If they sit far
below the plausible noise scale, the target is missing a term, not mis-tuned.

**Dropping the residual in the $\sigma^2$ M-step.** Use
$\sigma^2 = \sum_j(ss_j + n_j v_j)/N$ and the estimate is biased low — $0.741$
against the correct $0.830$ on this data — and the EM is no longer monotone in
the marginal likelihood it claims to maximise (the demo catches this as a
$-0.004$ step). The missing piece is
$n_j(\bar y_j - \mu - \theta_j)^2$: the group mean is not at its fitted value, and
that residual is data.

**Dropping the $v_j$ in the $\tau^2$ M-step.** Then $\tau^2$ is the variance of
the *estimated* effects, which already include shrinkage toward zero, so it
shrinks further every sweep and collapses. With $\tau^2 = 0$ the model becomes
full pooling and every group mean is the same number. This is the classic
empirical-Bayes failure, and the fix is to remember that the missing effects have
posterior variance.

**The centered funnel.** With a small true $\tau$, a centered sampler's ESS for
$\log\tau$ can fall to a few percent of the draws while its point estimates still
look fine, because the rare accepted moves are large. The demo's 96 out of 3000
is a 3% ESS — and the posterior means it produces are still close, which is
exactly why ESS and not the estimate is the diagnostic. *Fix:* non-centered
parameterisation, or a mass matrix adapted to the funnel.

**A single observation in a group.** `no_pooling` raises, because
$s_j^2/(n_j-1)$ is undefined at $n_j = 1$. Partial pooling is defined: the group
contributes its one observation with a wide posterior, and $\lambda_j$ is large
because $\sigma^2/n_j = \sigma^2$ is large compared with $\tau^2$. This is the
case hierarchical models exist for — small-area estimation is entirely about
groups too small to estimate alone.

**The exchangeability assumption.** The model treats the group labels as
interchangeable draws from one distribution. If the groups are actually two
clusters — a treatment and a control, say — the prior is wrong and the shrinkage
pulls both toward a mean that neither belongs to. The symptom is a residual
pattern by group that no $\tau^2$ can fix. *Diagnostic:* plot the group means
against the shrunken estimates; structure in the residuals means the prior is
missing a covariate, not that $\tau^2$ is wrong.

**An informative prior on the log-scales.** With ten groups and two observations
each, the data barely identify $\tau$. A tight prior on $\log\tau$ — the default
in some software is a half-Cauchy with a small scale — will dominate the
posterior, and the "estimate" is then the prior. The demo uses $N(0, 2^2)$
deliberately and still finds $\mathbb{E}[\tau^2] \approx 0.11$ against a true
$0.0225$: weak identification, not a bug.

## Applications

**Small-area estimation.** The original use: a survey with 4 respondents in a
county and 400 in a city, where the county estimate borrows from the state. The
US Census Bureau's SAIPE programme and every "model-based" official statistic is
this file with covariates added.

**Meta-analysis.** Each study estimates an effect with a standard error; the
between-study variance $\tau^2$ is the heterogeneity. Random-effects meta-analysis
is (7) with $n_j$ replaced by $1/\mathrm{se}_j^2$, and the shrinkage estimate is
the pooled effect for a study of that size. [`bayesian_inference.py`](bayesian_inference.py)
has the single-study conjugate version.

**A/B testing with many cells.** Ten variants, each with a few hundred users:
ranking the raw conversion rates is mostly noise, and ranking the shrunken ones
is not. This is the "empirical Bayes" that makes high-dimensional online
experiments work, and it is the same arithmetic as the demo.

**Education and health-care report cards.** Value-added models for teachers,
hospital mortality rates, and surgeon league tables are all hierarchical: the
whole point is that the noisy small-$n$ units are not ranked on their raw
averages. The policy debate about whether to publish shrunken or raw estimates
is a debate about (2).

**Genomics.** `limma`'s moderated $t$-statistic is empirical Bayes on gene-wise
variances: each gene's variance is shrunk toward a pooled estimate, exactly as
(2) shrinks a group mean, and the degrees of freedom are the $v_j$ correction in
a different guise. Millions of features, one variance component.

**Priors in machine learning.** Hierarchical priors are how multilevel models,
mixed-effects packages (`lme4`, `brms`), and Bayesian deep learning share
parameters across groups. The non-centered parameterisation is standard advice in
Stan and PyMC for exactly the funnel reason the demo measures.

Where it is a trap: when the groups are not exchangeable, when $J$ is very small
(fewer than about five groups, $\tau^2$ is not estimable and a fixed prior is
more honest), and when the group means are the quantity of interest and you
cannot afford to integrate over the variance components — the plug-in EB answer
is then overconfident by exactly the amount it ignores. [`mcmc.py`](mcmc.py) is
the sampler underneath the last variant, and [`variational_inference.py`](variational_inference.py)
is the alternative when $J$ is in the thousands and MCMC is too slow.

## References

- James & Stein (1961). *Estimation with quadratic loss.* Proceedings of the
  Fourth Berkeley Symposium. Shrinkage dominates the MLE; (4)–(5) is the
  one-dimensional case.
- Lindley & Smith (1972). *Bayes estimates for the linear model.* JRSS-B 34, 1.
  The hierarchical normal model and the shrinkage formula (2).
- Efron & Morris (1975). *Data analysis using Stein's estimator and its
  generalizations.* JASA 70, 311. The baseball example that made empirical Bayes
  mainstream.
- Dempster, Laird & Rubin (1977). *Maximum likelihood from incomplete data via
  the EM algorithm.* JRSS-B 39, 1. The algorithm behind `_em`.
- Laird & Ware (1982). *Random-effects models for longitudinal data.* Biometrics
  38, 963. The EM updates (8)–(9), including the two terms that are easy to drop.
- Casella (1985). *An introduction to empirical Bayes data analysis.* The
  American Statistician 39, 83. The best short account of the plug-in view.
- Robinson (1991). *That BLUP is a good thing: the estimation of random effects.*
  Statistical Science 6, 15. Why the shrinkage estimator is also the best linear
  unbiased predictor.
- Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013). *Bayesian Data
  Analysis*, 3rd ed. CRC. Chapters 5 and 11: hierarchical models, and the funnel
  in Figure 11.6.
- Papaspiliopoulos, Roberts & Sköld (2007). *A general framework for the
  parametrisation of hierarchical Bayesian models.* Bayesian Analysis 2, 59. The
  centered/non-centered dichotomy, and when each is better.
- Betancourt & Girolami (2015). *Hamiltonian Monte Carlo for hierarchical
  models.* In *Current Trends in Bayesian Methodology with Applications.* The
  funnel as a geometric pathology, with the diagnostics.
- Rue, Martino & Chopin (2009). *Approximate Bayesian inference for latent
  Gaussian models using integrated nested Laplace approximations.* JRSS-B 71,
  319. What to do when the grid check is not affordable.
