# Particle filter

`particle_filter.py` · Probabilistic models · Tier 2 · 6 variants

## Intuition

A Kalman filter represents the belief with a mean and a covariance, and that is
only correct when the model is linear and the noise is Gaussian. A particle
filter represents the same belief with samples — a few thousand state values,
each carrying a weight — and is correct when the samples are, which is to say
asymptotically, at a rate of $1/\sqrt{N}$ and with no assumptions about shape.

The mechanism is Bayes' rule applied to a list. Each particle is a hypothesis
about the state. Move every hypothesis through the dynamics; score it by how
well it predicted what you just observed; renormalise. The posterior is the
weighted cloud.

$$w_t^i \;\propto\; w_{t-1}^i\, g\!\left(z_t \mid x_t^i\right) \qquad (1)$$

The problem is that scoring is multiplicative, and so is the decay. After a few
observations a handful of particles hold essentially all the weight, and $N-10$
particles are computing nothing. The demo measures it exactly: without
resampling, the effective sample size of 4000 particles falls from 2772 to
**1.2** over 40 steps. Resampling is the fix — duplicate the particles that
survived, in proportion to their weight, and reset all weights to $1/N$ — and it
is not optional.

Everything in this file follows from that one tension. Resample too often and
you lose diversity (the same particle copied $N$ times); too rarely and the
weights degenerate. Resample well and you spend fewer particles; resample poorly
and you add noise for nothing.

## Variants

| Variant | Resamples | Ancestors drawn by | Use when |
|:--|:--|:--|:--|
| `sir` | when `ESS < 0.5 N` | multinomial | the default: bootstrap filter |
| `sis` | never | — | a short sequence, or as a baseline to show degeneracy |
| `auxiliary_pf` | every step, *before* propagating | first-stage predictive weights | the likelihood is informative and you can afford `p(z_t\mid x_{t-1})` |
| `systematic_resample` | when `ESS < 0.5 N` | one uniform, $N$ evenly spaced positions | the standard low-variance choice |
| `stratified_resample` | when `ESS < 0.5 N` | one uniform per stratum | same variance as systematic, slightly easier to parallelise |
| `regularized_pf` | when `ESS < 0.5 N` | multinomial + kernel jitter | the transition is nearly deterministic, or the state is continuous and the cloud collapses |

The split is not along "accuracy" — all six target the same posterior. It is
along *where the variance goes*: `sir` spends it on weight noise, `auxiliary_pf`
spends it on a better proposal, `systematic_resample` spends it on the resampling
step, and `regularized_pf` spends a little bias to buy back diversity.

## Derivation

### 1. Importance sampling, sequentially

We want $\pi_t(x_{1:t}) = p(x_{1:t} \mid z_{1:t})$, which is proportional to the
joint $p(x_{1:t}, z_{1:t})$ and normalised by the evidence. Draw from a proposal
$q(x_{1:t})$ you *can* sample — here the transition prior — and correct with

$$w_t = \frac{p(x_{1:t}, z_{1:t})}{q(x_{1:t})} = \frac{p(x_t \mid x_{t-1})\,g(z_t \mid x_t)}{q(x_t \mid x_{1:t-1})}\,\frac{p(x_{1:t-1}, z_{1:t-1})}{q(x_{1:t-1})} \qquad (2)$$

The second factor is yesterday's weight, so the weights update multiplicatively
by an *incremental* factor. With the bootstrap proposal
$q(x_t \mid x_{1:t-1}) = p(x_t \mid x_{t-1})$, that factor is exactly the
likelihood — which is (1), and the reason the file takes a `transition` that
samples rather than a density.

### 2. The evidence comes free

The normalising constant of the weighted cloud at step $t$ is an estimate of the
predictive likelihood:

$$\widehat{p}(z_t \mid z_{1:t-1}) = \sum_i w_{t-1}^i\, g\!\left(z_t \mid x_t^i\right), \qquad \widehat{p}(z_{1:T}) = \prod_{t=1}^{T}\widehat{p}(z_t \mid z_{1:t-1}) \qquad (3)$$

This estimator is unbiased for $p(z_{1:T})$, which is a strong statement with a
sharp practical edge: $\log\widehat{p}$ is *not* unbiased, and its bias is
downward by roughly $\mathrm{Var}(\widehat{p})/2p^2$. That is why the demo
compares a single run's number, $-57.06$ against the exact $-57.23$, rather than
averaging logs.

The product form is also the reason a particle filter can be used as a likelihood
inside a parameter sampler: fix $\theta$, filter, and (3) is a noisy but unbiased
$p(z \mid \theta)$.

### 3. Effective sample size

The variance of the weights is what destroys the estimate, and it is measurable
without knowing the truth:

$$\mathrm{ESS} = \frac{1}{\sum_i \left(w_t^i\right)^2}, \qquad 1 \le \mathrm{ESS} \le N \qquad (4)$$

Equal weights give $N$; a single particle holding all the weight gives 1. The
threshold in the code is a fraction of $N$ — resample when the cloud is carrying
fewer than half the particles' worth of information. Note what ESS does *not*
say: it is the size of an equally-weighted sample with the same weight variance,
not the number of particles that represent the posterior well. A cloud of 2000
distinct particles all in the wrong place has a healthy ESS.

### 4. Resampling, and why systematic is better

Resampling draws ancestors $a_i \sim \mathrm{Categorical}(w)$. Multinomial draws
$N$ independent uniforms and inverts the CDF. Systematic draws **one** uniform
$u \sim U[0, 1/N)$ and takes positions $u, u + 1/N, \dots$; stratified draws one
uniform inside each stratum $[i/N, (i+1)/N)$.

All three are unbiased. Their variances differ, and the difference has an exact
consequence: with uniform weights $w_i = 1/N$ the CDF is $i/N$, and the systematic
positions $u + i/N$ land one per interval — **every particle is copied exactly
once**. Multinomial scatters counts around 1 with variance $1 - 1/N$. The demo
asserts the exact property for both low-variance schemes (all counts equal 1) and
shows multinomial producing up to 4 copies of a particle, which is noise injected
into a cloud that was already correct.

This is the "variance of the resampling step", and it is why systematic
resampling is the default in every serious implementation.

### 5. The auxiliary filter

The bootstrap proposal ignores the next observation when it decides where to put
particles. If the likelihood is narrow, most particles land where $g$ is tiny and
the weights collapse. The auxiliary filter (Pitt & Shephard) moves the decision
one step earlier:

$$\lambda_i \;\propto\; w_{t-1}^i\, p\!\left(z_t \mid x_{t-1}^i\right) \qquad (5)$$

where $p(z_t \mid x_{t-1})$ is the *predictive* likelihood, with the transition
integrated out. Sample ancestors by $\lambda$, propagate them, then correct:

$$w_t^i \;\propto\; \frac{g\!\left(z_t \mid x_t^i\right)}{p\!\left(z_t \mid x_{t-1}^{a_i}\right)} \qquad (6)$$

and the evidence increment is the normaliser of (5). The demo needs an extra
argument — `log_predictive` — because that integrated likelihood is exactly what
the bootstrap filter never computes. With it, the spread of $\log\widehat{p}$
across 40 seeds falls from $0.277$ to $0.107$ nats at the same particle count: the
same estimate, a quarter of the variance.

### 6. Regularisation

Resampling copies particles, so a nearly deterministic transition produces
duplicates that stay duplicates. At the end of the demo run, the plain filter has **19 distinct values out of
500**; the rest are copies. The fix is to
jitter after resampling, with a kernel whose bandwidth is the standard
rule-of-thumb for kernel density estimation:

$$x^{i} \leftarrow x^{i} + h\,L\,\varepsilon_i, \qquad h = \left(\frac{4}{N(d+2)}\right)^{1/(d+4)} \qquad (7)$$

with $LL^{\top}$ the empirical covariance of the cloud and
$\varepsilon_i \sim N(0, I)$. This is Silverman's rule-of-thumb bandwidth. It introduces a small bias — you are smoothing the posterior — in
exchange for the diversity the next likelihood evaluation needs. The demo asserts
both halves: 500 distinct particles instead of 19, and a mean that agrees with
the unregularised filter to $0.005$.

## Code walkthrough

The public surface is a frozenset and one function; the model arrives as three
callables, which is what makes the same code work for any state space.

```python
VARIANTS = frozenset(
    ["sir", "sis", "auxiliary_pf", "systematic_resample", "stratified_resample",
     "regularized_pf"]
)
```

`init` draws from the prior, `transition` propagates the *whole cloud* in one
vectorised call (state noise included, which is why it takes an `rng` rather than
a covariance), and `log_likelihood` scores every particle against one
observation. That signature is the entire interface to the model.

The loop is the algorithm. Note where the evidence increment sits: it is the
normaliser of the weights *before* they are normalised, computed with
`logsumexp` so that a weight of $e^{-800}$ is a small number and not a zero.

```python
        else:
            X = np.asarray(transition(X, rng), dtype=float)
            lw = lw + log_likelihood(z[t], X)
            loglik += _logsumexp(lw)
            lw -= _logsumexp(lw)
```

The auxiliary branch is (5)–(6) and reads as an inversion of the bootstrap order.
It adds the predictive log-likelihood to the *ancestor* weights, resamples on
that, propagates, and then subtracts the same quantity from the descendants,
which is (6) in log space.

```python
            lpred = log_predictive(z[t], X)
            lw = lw + lpred
            loglik += _logsumexp(lw)
            lw -= _logsumexp(lw)
            idx = _resample(lw, "systematic", rng)
            X = np.asarray(transition(X[idx], rng), dtype=float)
            n_resample += 1
            lw = log_likelihood(z[t], X) - lpred[idx]
            lw -= _logsumexp(lw)
```

The three resampling schemes are four lines each because they differ only in
where the uniforms are placed. `np.searchsorted` on the CDF is the inverse-CDF
transform, and it returns `N` ancestors rather than `N` distinct particles —
duplicates are the point.

```python
    if scheme == "multinomial":
        return np.searchsorted(cdf, rng.random(N))
    if scheme == "systematic":
        u = (rng.random() + np.arange(N)) / N
        return np.searchsorted(cdf, u)
    if scheme == "stratified":
        u = (np.arange(N) + rng.random(N)) / N
        return np.searchsorted(cdf, u)
```

`cdf[-1] = 1.0` is the one defensive line: `cumsum` of $N$ floats can land a ulp
below 1, and a uniform draw in that last ulp would return index $N$ and crash the
next gather.

The weighted moments are a single expression, and the `w[:, None]` is what makes
it a covariance rather than a scalar product. This is the "estimate" the filter
returns — everything else in `PFResult` is diagnostics.

```python
def _weighted_cov(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`Σ_i w_i (x_i − m)(x_i − m)ᵀ` for a weighted cloud, shape `(d, d)`."""
    m = w @ X
    Z = X - m
    return (Z * w[:, None]).T @ Z
```

Regularisation is (7). The `1e-12 I` is not cosmetic: after resampling the
empirical covariance can be singular (all copies identical), and a Cholesky of a
singular matrix raises. Falling back to returning the cloud unchanged is the
honest behaviour — jitter with an undefined bandwidth is worse than none.

```python
    h = factor * (4.0 / (N * (d + 2.0))) ** (1.0 / (d + 4.0))
    try:
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(d))
    except np.linalg.LinAlgError:
        return X
    return X + h * rng.standard_normal((N, d)) @ L.T
```

## Complexity

With $N$ particles, $d$ states, $m$ observations and $T$ steps, letting $c_x$ be
the cost of one transition sample and $c_g$ one likelihood evaluation:

| Variant | Time | Memory | Resampling steps |
|:--|:--|:--|--:|
| `sir`, `systematic_resample`, `stratified_resample` | $O(T(N(c_x + c_g) + N\log N))$ | $O(Nd + Td^2)$ | $O(T\,\Pr[\mathrm{ESS}<\tau N])$ |
| `sis` | $O(TN(c_x + c_g))$ | $O(Nd + Td^2)$ | 0 |
| `auxiliary_pf` | $O(T(N(c_x + c_g + c_p) + N\log N))$ | $O(Nd + Td^2)$ | $T$ |
| `regularized_pf` | $O(T(N(c_x + c_g) + Nd^2 + d^3))$ | $O(Nd + Td^2)$ | as `sir` |

The $N\log N$ is the `searchsorted` (a binary search per ancestor); a linear-time
CDF walk is possible but only wins at large $N$. `regularized_pf` adds a $d\times
d$ Cholesky per resampling, which is negligible unless $d$ is large — and if $d$
is large, a particle filter is the wrong tool anyway.

The cost that matters is not in the table: the number of particles needed for a
given accuracy grows exponentially in $d$ (Section 5 of the failure modes). In
one dimension, 1000 particles are plenty; in twenty, $N$ must be in the millions
before the weights stop degenerating.

## Failure modes

**Weight degeneracy.** The default failure, and the one the demo makes visible:
without resampling ESS falls from 2772 to 1.2, so the posterior is being carried
by one particle. The estimates do not look broken — they look confident — which
is why the ESS trace must be plotted, not inferred from the output. *Diagnostic:*
`ess` in the result; anything below $0.1N$ for several consecutive steps means
the proposal is not putting particles where the data is.

**Sample impoverishment.** The mirror image: resample a nearly deterministic
model and the cloud becomes 19 copies of one number, so the filter can no longer
respond to new data and reports a variance far too small. *Diagnostic:* count
`np.unique` on returned particles; *fix:* `regularized_pf`, or a better proposal
(the auxiliary filter), or just more process noise — if the model allows it.

**The curse of dimensionality.** The variance of the incremental weights is
$O(1)$ *per dimension* for the bootstrap proposal, so the number of particles
needed for a fixed ESS grows exponentially in $d$ (Bengtsson, Bickel & Li). Ten
dimensions is comfortable; a hundred is hopeless without a proposal that exploits
structure — the auxiliary filter, a locally optimal proposal, or a
Rao-Blackwellised filter that marginalises the linear-Gaussian part.

**Resampling bias in the wrong place.** Resampling every step is legal but adds
variance for nothing when the weights are still healthy, and it is what drives
impoverishment. Resampling never is also legal and degenerates. The threshold is
the whole design decision, and there is no universal value: the file defaults to
$0.5N$ and the demo shows both extremes.

**A likelihood that is too narrow for the proposal.** If $R \ll Q$, the
observation is so precise that a propagated particle is never near it, every
weight is $e^{-100}$, and float32-style underflow — or, in log space, a single
particle holding all the weight — is guaranteed rather than occasional. The
auxiliary filter and locally optimal proposals exist for this case; so does the
much simpler fix of inflating the process noise and admitting the model is wrong.

## Applications

**Target tracking.** The original application (Gordon, Salmond & Smith, 1993) and
still the most common: bearings-only tracking, where the observation is an angle
and the posterior is a banana that no Gaussian can hold. Add multiple targets and
the state space becomes multimodal, which is exactly what particles are for.

**Robot localisation.** Monte Carlo localisation with a map: particles are poses,
the likelihood is a laser scan. Thousands of hypotheses, and the reason the
method is in every ROS stack. The failure mode is famous — a robot kidnapped and
moved, whose particle cloud confidently reports the wrong room — and it is why
"augmented MCL" injects random particles on purpose.

**Econometrics and finance.** Stochastic volatility models are nonlinear state
space models with a latent variance: $y_t = e^{h_t/2}\varepsilon_t$,
$h_t = \mu + \phi(h_{t-1} - \mu) + \sigma\eta_t$. The particle filter gives the
likelihood, and PMCMC puts it inside a parameter sampler.

**Data assimilation and ecology.** Weather and ocean models run ensemble filters
with $10^6$–$10^8$ states; animal movement models fit hidden behavioural states
to telemetry. Both are state space models where the transition is a simulator,
which is the case this file's `transition` callable is designed for.

**Nested inference.** Andrieu, Doucet & Holenstein's particle MCMC uses (3) as an
unbiased likelihood estimate inside Metropolis–Hastings, which makes the sampler
exact for the static parameters — the reason a particle filter is often a
component rather than a whole method.

Where it is a trap: low dimension with a linear-Gaussian model, where
[`kalman.py`](kalman.py) is exact and cheaper by orders of magnitude; and any
problem where the posterior is smooth and unimodal, where a Laplace
approximation or [`variational_inference.py`](variational_inference.py) will
answer in a fraction of the time.

## References

- Gordon, Salmond & Smith (1993). *Novel approach to nonlinear/non-Gaussian
  Bayesian state estimation.* IEE Proceedings F 140, 107. The bootstrap filter.
- Kitagawa (1996). *Monte Carlo smoother and filter for non-Gaussian nonlinear
  state space models.* Journal of Computational and Graphical Statistics 5, 1.
  Resampling and the smoother, independently.
- Liu & Chen (1998). *Sequential Monte Carlo methods for dynamic systems.* JASA
  93, 1032. The unifying importance-sampling view, and the ESS.
- Pitt & Shephard (1999). *Filtering via simulation: auxiliary particle
  filters.* JASA 94, 590. The auxiliary filter, (5)–(6).
- Carpenter, Clifford & Fearnhead (1999). *Improved particle filter for nonlinear
  problems.* IEE Proceedings Radar, Sonar and Navigation 146, 2. Systematic
  resampling, and why it wins.
- Doucet, de Freitas & Gordon (2001). *Sequential Monte Carlo Methods in
  Practice.* Springer. The reference collection; Chapter 1 is this file.
- Musso, Oudjane & Le Gland (2001). *Improving regularised particle filters.* In
  *Sequential Monte Carlo Methods in Practice.* The kernel jitter of (7).
- Bengtsson, Bickel & Li (2008). *Curse-of-dimensionality revisited: collapse of
  the particle filter in very large scale systems.* IMS Collections 2, 316. The
  exponential-in-$d$ statement, with constants.
- Andrieu, Doucet & Holenstein (2010). *Particle Markov chain Monte Carlo
  methods.* JRSS-B 72, 269. PMCMC, and the unbiased-likelihood argument.
