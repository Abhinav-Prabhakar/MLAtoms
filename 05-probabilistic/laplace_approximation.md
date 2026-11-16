# Laplace approximation

`laplace_approximation.py` · Probabilistic models · Tier 3 · 4 variants

## Intuition

The Laplace approximation is one sentence: **fit a parabola to the log-density at
its peak, and exponentiate it.** A parabola exponentiated is a Gaussian, so the
posterior becomes

$$p(\theta \mid y) \approx N\!\left(\hat\theta,\; P^{-1}\right), \qquad P = -\nabla^2\log p(\theta\mid y)\Big|_{\theta=\hat\theta} \qquad (1)$$

Two steps, and only the second is an approximation. Finding $\hat\theta$ is exact
— it is a root of the gradient, and Newton's method on a log-concave density
gets there in a handful of iterations (seven for the demo's logistic regression,
with a final gradient of $9\times10^{-16}$). Everything approximate is in the
curvature: the Gaussian matches the height and the width at the mode, and knows
nothing else.

That sounds like a crude approximation, and near the mode it is a very good one.
The reason is the Bernstein–von Mises theorem: as data accumulate, the posterior
becomes Gaussian, with width shrinking like $n^{-1/2}$. Laplace is not a hack
that happens to work — it is the leading term of the asymptotic expansion, and
it is the reason MLEs have standard errors.

```
   log p(θ|y)                        p(θ|y)
      │  ╭─╮                          │   ╭╮
      │ ╱   ╲        ← θ̂            │  ╱  ╲
      │╱     ╲___                    │ ╱    ╲___
      └─────────── θ                 └─────────── θ
        the parabola matches           its exponential is the
        the top and the width          Gaussian we report
```

The variants are four answers to "how much of the curvature can you afford":

- **`full_hessian`** — all $d^2$ entries of $P$. What the method means.
- **`diag_hessian`** — `1/diag(P)`, $O(d)$ storage. It treats the other
  parameters as known, so it is *overconfident* by exactly $1-\rho^2$; the demo
  measures $\rho = -0.469$ and variances 22% too small.
- **`kfac_laplace`** — a Kronecker product per layer, $O(k^3 + d^3)$ to invert.
  Exact when the output-side factor is shared across samples, otherwise not; the
  demo is 40% off in the precision at one output unit.
- **`predictive`** — full Hessian, and then *integrate* over it instead of
  plugging the mode into the link. The difference is the second-order term the
  delta method drops: 0.0086 in probability at the demo's test point, with the
  sign of $-\sigma''(\mu)$.

The demonstration model is Bayesian logistic regression, chosen because its exact
posterior is a 2-D integral that a grid can compute. Every claim below is checked
against that grid, not against another approximation: the covariance is within
13.9% entry by entry, the evidence within 0.0136 nats.

## Variants

| Variant | Covariance | Cost | When |
|:--|:--|:--|:--|
| `full_hessian` | $P^{-1}$ | $O(d^3)$ | $d$ up to a few thousand; the default and the reference |
| `diag_hessian` | $\mathrm{diag}(1/P_{ii})$ | $O(d)$ | very large $d$; overconfident, and useful as a cheap sanity check |
| `kfac_laplace` | $(\bar A \otimes X^\top X)^{-1}$ | $O(k^3 + d^3)$ | neural-network layers, where the block structure is real |
| `predictive` | $P^{-1}$, then integrated | $O(d^3 + m d^2)$ | you want probabilities, not logits |

`diag_hessian` is the only variant here that is *provably* wrong in a direction:
for positive-definite $P$, $(P^{-1})_{ii} \ge 1/P_{ii}$ always. The demo asserts
the inequality and the exact shrink factor.

`kfac_laplace` is the odd one out. It exists because in a network the precision
of a layer's weights is a sum of Kronecker products, and a Kronecker product can
be inverted factor by factor. With a single output unit there is no second
factor to exploit — the approximation degenerates to replacing each sample's
curvature $a_n$ with its mean — and the demo shows it is 40% off. With $k$
classes, the $k\times k$ factor carries the structure and the identity is exact
when $a_n$ does not vary with $n$.

`predictive` changes what is computed, not how: the covariance is the full one,
and the question is whether to report $\sigma(\hat\mu)$ or
$\mathbb{E}[\sigma(\mu)]$. The second is the posterior predictive, and it is the
one that is calibrated.

## Derivation

### 1. The expansion

Write $L(\theta) = \log p(\theta, y)$ for the unnormalised log-joint. Since
$\nabla L(\hat\theta) = 0$,

$$L(\theta) = L(\hat\theta) - \tfrac12(\theta-\hat\theta)^\top P(\theta-\hat\theta) + O\big(\lVert\theta-\hat\theta\rVert^3\big), \qquad P = -\nabla^2 L(\hat\theta) \qquad (2)$$

Exponentiating and normalising the Gaussian gives (1). For the *evidence*, keep
the constant too and integrate the quadratic:

$$\log p(y) = \log\int e^{L(\theta)}\,d\theta \approx L(\hat\theta) + \frac{d}{2}\log 2\pi - \frac12\log\lvert P\rvert \qquad (3)$$

The three terms are the height at the mode, the volume of a unit Gaussian in $d$
dimensions, and the volume of *this* Gaussian. The last is the Occam factor: a
model with a broad posterior pays for the parameter space it did not use. It is
why the evidence can compare models of different dimension, and why it can refuse
a model that fits the data well with a prior it had to travel far from.

Taylor-expanding $L$ one order further and integrating gives the leading error:

$$\mathbb{E}[\theta] \approx \hat\theta + \tfrac12 P^{-1}\,\nabla\!\left(\mathrm{tr}\big(P^{-1}\nabla^2 L(\theta)\big)\right)\Big|_{\hat\theta} \qquad (4)$$

The correction is driven by the third derivatives of $L$ — the skewness — and it
is zero when the posterior is symmetric. The demo's posterior is not: its mode
and mean differ by 0.057, which no Gaussian can represent.

The rate is the classical one. Under regularity, the posterior contracts at
$n^{-1/2}$ and the error of the Laplace approximation to $\log p(y)$ is
$O(n^{-1})$; on the demo's $n=60$ logistic problem the evidence is off by 0.0136
nats.

### 2. Finding the mode

Newton's method for $\nabla L(\theta) = 0$ is

$$\theta_{k+1} = \theta_k + P(\theta_k)^{-1}\nabla L(\theta_k) \qquad (5)$$

with $P(\theta) = -\nabla^2 L(\theta)$. This is the same matrix that becomes the
covariance at convergence, which is why the two steps share code: `_precision`
is called by both the optimiser and `covariance()`.

For a log-concave likelihood — logistic regression is — Newton is quadratically
convergent, so the step count barely depends on $d$. It is *not* monotone:
a full Newton step can jump across the mode into the flat tail where the
quadratic model is worthless, which is why the loop halves the step until the
objective stops decreasing. On a well-behaved problem the line search never
triggers; on a nearly separable one it is the difference between converging and
diverging.

For Bayesian logistic regression with a Gaussian prior $w\sim N(0,\lambda^{-1}I)$:

$$\nabla L(w) = X^\top(y - \sigma(Xw)) - \lambda w, \qquad P(w) = X^\top\mathrm{diag}\big(p(1-p)\big)X + \lambda I \qquad (6)$$

$P$ is positive definite because of the prior term and because $p(1-p) > 0$.
Without the prior, separable data drives $\lVert w\rVert\to\infty$, $p(1-p)\to0$,
and $P$ becomes singular — the posterior is improper, and no Gaussian exists.

### 3. Why the diagonal version is overconfident

Partition a positive-definite $P$ and use the Schur complement:

$$P = \begin{pmatrix} a & b^\top \\ b & C\end{pmatrix}, \qquad (P^{-1})_{11} = \frac{1}{a - b^\top C^{-1}b} \ge \frac{1}{a} \qquad (7)$$

with equality iff $b = 0$. The diagonal approximation reports $1/a$, the
variance you would have *if you knew the other parameters*. It is a conditional
variance passed off as a marginal one, so it is always too small — the estimate
is overconfident, not merely imprecise. For two parameters the inflation factor
is exact:

$$\frac{(P^{-1})_{11}}{1/P_{11}} = \frac{1}{1-\rho^2} \qquad (8)$$

The demo's $\rho = -0.469$ gives a factor of 1.28, and the measured variances are
22% low.

### 4. Kronecker structure

For a likelihood that factorises over samples, the precision of a linear layer's
weight matrix $W\in\mathbb{R}^{k\times d}$ is

$$P = \sum_{n=1}^{N} A_n \otimes x_n x_n^\top, \qquad A_n = \mathrm{diag}(p_n) - p_n p_n^\top \qquad (9)$$

— one Kronecker product per sample, and a sum of $N$ of them. KFAC replaces the
sum of products with the product of means:

$$P \approx \bar A \otimes \Big(\sum_n x_n x_n^\top\Big), \qquad \bar A = \frac1N\sum_n A_n \qquad (10)$$

The two agree exactly when $A_n$ does not depend on $n$; otherwise the
approximation discards the correlation between a sample's input and its output
curvature. The payoff is the inversion:

$$\big(A\otimes B\big)^{-1} = A^{-1}\otimes B^{-1} \qquad (11)$$

an identity the demo asserts, and the reason a $(kd)\times(kd)$ precision costs
$O(k^3 + d^3)$ instead of $O(k^3d^3)$. In a deep network this is applied per
layer, with $A$ from the layer's inputs and $B$ from the backpropagated
gradients; here, with one output unit, $A$ is a scalar and the approximation has
nothing left to exploit — which is exactly what the demo's 40% error measures.

### 5. The predictive, and the delta method

For a test point $x$ the logit is $\mu = x^\top\hat w$ with variance
$v = x^\top P^{-1}x$. The plug-in predictive is $\sigma(\mu)$; the integrated one
is

$$\mathbb{E}\big[\sigma(\mu+\varepsilon)\big] = \int \sigma\big(\mu+\sqrt{v}\,z\big)\varphi(z)\,dz \qquad (12)$$

which has no elementary closed form, so the code evaluates it by 20-point
Gauss–Hermite quadrature — exact to machine precision for a smooth integrand
with a Gaussian weight. The second-order expansion is the delta method's missing
term:

$$\mathbb{E}\big[\sigma(\mu+\varepsilon)\big] = \sigma(\mu) + \tfrac12\sigma''(\mu)\,v + O(v^2), \qquad \sigma''(\mu) = \sigma(\mu)\big(1-\sigma(\mu)\big)\big(1-2\sigma(\mu)\big) \qquad (13)$$

$\sigma''$ is negative above $\mu=0$ and positive below it, so integrating always
pulls the prediction toward $\tfrac12$: uncertainty in the weights makes the
model less confident, which is the correct behaviour and which the plug-in
cannot produce. The demo checks (13) to within 5% at $\mu = \pm2.02$, and finds
the gap is 0.0086 in probability — small here because the posterior is tight, and
growing like $v$ as the data thin out.

## Code walkthrough

The likelihood is evaluated through `logaddexp` rather than
$\log\sigma(a)$ written directly, because the direct form overflows for
$a \ll 0$ and underflows for $a \gg 0$:

```python
        a = self.X @ w
        # log σ(a) = −log(1+e^{−a}) and log(1−σ(a)) = −log(1+e^{a}); the
        # second form is what logaddexp computes without overflow
        return float(np.sum(self.y * a - np.logaddexp(0.0, a))
                     - 0.5 * self.prior_prec * float(w @ w))
```

The precision is (6) in one line, and the only thing to notice is that it is
built as `(X.T * weights) @ X` — column scaling, not a diagonal matrix:

```python
    def _precision(self, w: np.ndarray) -> np.ndarray:
        """`−∇² log p(w | y)`: positive definite by construction."""
        p = self._p(w)
        return (self.X.T * (p * (1.0 - p))) @ self.X + self.prior_prec * np.eye(w.size)
```

The optimiser is (5) with a backtracking line search. The loop condition matters:
without it, a Newton step on a nearly separated dataset leaves the region where
the quadratic model holds, and the objective can *decrease* — which is the one
thing a maximiser must not do:

```python
            step = np.linalg.solve(self._precision(w), self._grad(w))
            # Newton on a logistic posterior can overshoot into a flat region
            # where the quadratic model is worthless; halving until the
            # objective does not decrease costs one extra evaluation per step
            t = 1.0
            while t > 1e-12:
                cand = w + t * step
                new = self._log_post(cand)
                if new >= obj - 1e-12:
                    break
                t *= 0.5
```

The variant dispatch is where the approximation is chosen. The diagonal branch
carries the Schur-complement argument from (7) as a comment because the direction
of the error is the point: it is not "less accurate", it is systematically too
narrow.

```python
        P = self._precision(self.w_)
        if self.variant == "diag_hessian":
            # for a positive-definite P, (P⁻¹)_ii ≥ 1/P_ii with equality iff the
            # row is uncorrelated: dropping the off-diagonals treats the other
            # parameters as known, so it can only *narrow* a marginal
            return np.diag(1.0 / np.diag(P))
```

The evidence is (3). `slogdet` rather than `log(det(...))` because a determinant
of a $10^3\times10^3$ precision underflows to zero long before the log does:

```python
        P = self._precision(self.w_)
        _, logdet = np.linalg.slogdet(P)
        return self.objective_ + 0.5 * self.w_.size * LOG2PI - 0.5 * logdet
```

KFAC is (10), and the mean on the left factor is what makes it exact in the
shared case: with $A_n = a$ for all $n$, the exact sum is $a\otimes\sum x_nx_n^\top$
and the approximation is $\bar A\otimes\sum x_nx_n^\top$ with $\bar A = a$.

```python
    f = np.asarray(factors, dtype=float)
    if f.ndim not in (1, 3):
        raise ValueError("factors must be (n,) or (n, k, k)")
    A = np.atleast_2d(f.mean(axis=0) if f.ndim == 3 else f.mean())
    k = A.shape[0]
    return np.kron(A, X.T @ X) + prior_prec * np.eye(k * X.shape[1])
```

The exact side of the comparison in (9) is built sample by sample, which is only
affordable because the demo's model is small — and is the reason KFAC exists:

```python
    for x, p in zip(X, probs, strict=True):
        P += np.kron(np.diag(p) - np.outer(p, p), np.outer(x, x))
```

The integrated predictive is (12). Gauss–Hermite is the right tool: it is exact
for polynomials of degree $2n-1$ times the Gaussian weight, so 20 nodes resolve
the integrand to machine precision, and it needs no samples:

```python
        var = np.einsum("nd,de,ne->n", X_new, self.covariance(), X_new)
        x, w = np.polynomial.hermite.hermgauss(20)
        # ∫ σ(μ + √v z) φ(z) dz = π^{-1/2} Σ w_i σ(μ + √(2v) x_i)
        out = np.sqrt(2.0 * np.maximum(var, 0.0))[:, None] * x + mu[:, None]
        return (w * sigmoid(out)).sum(axis=1) / math.sqrt(math.pi)
```

## Complexity

With $n$ observations, $d$ parameters, $I$ Newton iterations, $m$ test points, and
KFAC factors of size $k\times k$:

| Operation | Time | Memory |
|:--|:--|:--|
| gradient and precision | $O(nd^2)$ | $O(d^2)$ |
| Newton step | $O(nd^2 + d^3)$ | $O(d^2)$ |
| `full_hessian` covariance | $O(d^3)$ | $O(d^2)$ |
| `diag_hessian` covariance | $O(d)$ | $O(d)$ |
| `kfac_laplace` covariance | $O(k^3 + d^3)$ | $O(k^2 + d^2)$ |
| `log_evidence` | $O(d^3)$ | $O(d^2)$ |
| `predict_proba` | $O(md^2 + 20m)$ | $O(m)$ |

The whole fit is $O\big(I(nd^2 + d^3)\big)$, and $I$ is small: seven for the
demo, and roughly independent of $n$ because Newton is quadratically convergent.
The cost is dominated by forming $P$, which is a $d\times d$ matrix built from
$n$ outer products — the same operation as one gradient evaluation of a linear
model, so Laplace doubles the cost of fitting rather than multiplying it.

The `predictive` overhead is $O(md^2)$: one quadratic form per test point, which
for a large $m$ is worth batching (it is `einsum` in the code).

The cubic terms are where this stops being free. At $d = 10^4$ the precision
alone is 800 MB, which is why no one runs `full_hessian` on a network, and why
`kfac_laplace` and `diag_hessian` exist at all.

## Failure modes

**A non-Gaussian posterior.** Laplace reports a Gaussian no matter what. When
the posterior is skewed or multimodal — hierarchical models with a funnel,
mixtures, anything with a boundary — the mode can sit far from the mean, and the
Gaussian's mass is in the wrong place. The demo's own posterior is mildly skewed:
mode and mean differ by 0.057, and no reparameterisation of the output fixes
that. *Diagnostic:* compare the mode with a short MCMC run's mean (see
[`mcmc.py`](mcmc.py)); a large gap or a heavy tail means the Gaussian is the
wrong shape, not just the wrong width.

**Separable data.** If a hyperplane separates the classes, the likelihood has no
maximum at any finite $w$ and the posterior is improper in the flat direction.
Newton walks off and the precision becomes numerically singular. The prior is
what prevents this; with `prior_prec` too small the code will produce a huge
covariance and an evidence that looks excellent. *Diagnostic:* the largest
eigenvalue of $P$ relative to the prior term, and the norm of $\hat w$.

**Trusting the diagonal.** It is overconfident by exactly $1-\rho^2$ (8), so it
under-covers. A 95% interval built from `diag_hessian` is not a 95% interval.
*Diagnostic:* compare with `full_hessian` on a subsample; if the ratio of
variances is far from one, the diagonal is not an approximation, it is a
different model.

**KFAC where there is no factor structure.** For a single output unit the
approximation collapses to a mean curvature, and the demo measures 40% error in
the precision and 48% in the marginal variances. It is not a milder version of
the full Hessian; it is a different matrix. *Diagnostic:* the ratio of the KFAC
and full precisions' traces, which should be near one if the factorisation is
holding.

**A plug-in predictive.** Reporting $\sigma(x^\top\hat w)$ instead of (12) omits
a term of order $v$, so it is overconfident by an amount that grows exactly when
you need it — at extrapolation points and with little data. *Diagnostic:* the
difference between `integrate=True` and `integrate=False`; if it is larger than
the precision you report, the plug-in number is not the one to report.

**The evidence as a model-comparison tool.** (3) is an integral over the whole
posterior, approximated by its behaviour at one point. It is excellent when the
posterior is nearly Gaussian and misleading when it is not — singular models
(mixtures with unused components, neural nets with permutation symmetry) have
evidence that Laplace cannot see. *Diagnostic:* compare with a nested or
bridge-sampling estimate before trusting a Bayes factor computed this way.

**Inverting instead of solving.** `np.linalg.inv` on an ill-conditioned $P$ loses
half the digits, and Laplace is precisely where ill-conditioning shows up. The
code uses `solve` for the Newton step and `slogdet` for the evidence; only the
covariance needs an explicit inverse, and there `cholesky`-based solves would be
more stable at large $d$.

## Applications

**Generalised linear models.** For a GLM the Laplace approximation *is* the
standard inference: $P$ is the IRLS weight matrix at convergence, so the
covariance is the one every GLM summary prints. A logistic regression fit with
`statsmodels` is this file with the prior removed.

**Gaussian process classification.** The posterior over the latent function is
not Gaussian, and Laplace's approximation of it is the classical method —
[`gp_classification.py`](gp_classification.py) implements it, and the
derivation there is (2)–(3) with $P = I + K^{1/2}W K^{1/2}$.

**Neural-network uncertainty.** Last-layer Laplace — fit the network, then apply
this file to the final layer — is one of the few cheap calibrated uncertainty
methods for deep nets; `kfac_laplace` is the version that scales to a full
network, applied layer by layer. [`variational_inference.py`](variational_inference.py)
is the alternative when you want a global fit rather than a local one.

**Model comparison and Bayes factors.** (3) is the marginal likelihood, so
Laplace gives a free Bayes factor between two models. This is where BIC comes
from: dropping the $O(1)$ terms of (3) under a unit prior gives
$-2\log p(y) \approx -2L(\hat\theta) + d\log n$.

**Sequential filtering.** The extended Kalman filter is a Laplace approximation
applied recursively: linearise at the current mean, propagate a Gaussian, repeat.
[`kalman.py`](kalman.py) is the exact linear-Gaussian case this degrades from.

**Where it is a trap.** Hierarchical models (the funnel in
[`hierarchical_bayes.py`](hierarchical_bayes.py) is exactly the geometry where a
Gaussian at the mode is worst), mixtures and anything with label switching,
models with a boundary-constrained parameter, and any high-dimensional posterior
where the mode is not where the mass is.

## References

- Laplace (1774). *Mémoire sur la probabilité des causes par les événements.* The
  original integral approximation, and the reason the method carries his name.
- Tierney & Kadane (1986). *Accurate approximations for posterior moments and
  marginal densities.* JASA 81, 82. The modern statement, with the error terms
  behind (4).
- Kass, Tierney & Kadane (1990). *The validity of posterior expansions based on
  Laplace's method.* In *Bayesian and Likelihood Methods in Statistics and
  Econometrics.* When the expansion is justified, and what regularity buys.
- Kass & Raftery (1995). *Bayes factors.* JASA 90, 773. (3) as a model-comparison
  tool, and the BIC approximation.
- Martens & Grosse (2015). *Optimizing neural networks with Kronecker-factored
  approximate curvature.* ICML. The KFAC factorisation (10)–(11), there for the
  Fisher rather than the Hessian.
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning.* MIT
  Press, Chapter 3. The Laplace approximation for GP classification, with the
  same structure and a different $P$.
- Bishop (2006). *Pattern Recognition and Machine Learning.* Springer, §4.4. The
  logistic-regression derivation of (6), and the evidence in §4.5.
- MacKay (2003). *Information Theory, Inference, and Learning Algorithms.*
  Cambridge University Press, Chapter 27. The Occam factor reading of (3), and
  the Laplace method as the cheap alternative to sampling.
- Ritter, Botev & Barber (2018). *A scalable Laplace approximation for neural
  networks.* ICLR workshop. KFAC-Laplace in practice, including the damping
  that makes the factors invertible.
- Daxberger, Kristiadi, Immer, Eschenhagen, Bauer & Hennig (2021). *Laplace
  redux — effortless Bayesian deep learning.* NeurIPS. The modern recipe:
  last-layer, subnetwork and KFAC variants, and when each is calibrated.
