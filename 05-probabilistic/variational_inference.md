# Variational inference

`variational_inference.py` · Probabilistic models · Tier 2 · 5 variants

## Intuition

Integration is the bottleneck. The posterior is a ratio of two quantities, and
the denominator — the evidence — is an integral over every parameter. MCMC
sidesteps it by sampling; variational inference avoids it by *bounding* it.

The bound comes from one line of algebra. For any density $q(z)$,

$$\log p(x) = \log \int q(z)\,\frac{p(z, x)}{q(z)}\,\mathrm{d}z \;\ge\; \int q(z) \log \frac{p(z, x)}{q(z)}\,\mathrm{d}z \;=\; \mathrm{ELBO}(q)$$

which is Jensen's inequality, and it is an equality exactly when $q$ is the
posterior. Turning the inequality around gives the reason the method works:

$$\log p(x) - \mathrm{ELBO}(q) \;=\; \mathrm{KL}\!\left(q(z)\,\|\,p(z \mid x)\right) \;\ge\; 0 \qquad (1)$$

The gap is not an abstraction. It is a distance between two distributions, it is
computable in the conjugate cases, and it has exactly two sources:

- **the family.** If the truth is not inside the set of $q$'s you allow, no
  amount of optimisation closes the gap. This is the *approximation* gap.
- **the optimiser.** Finite steps, finite samples, a step size that is too
  large. This is the *estimation* gap, and it is supposed to be small.

Everything in this file is an attempt to move one of those two numbers, and the
demo is built to separate them: a target with a known answer is fitted by a
family that contains it (diagonal, 0.511 away from the truth; full covariance,
0.002) and by a family that does not.

The punchline of the mean-field story is a theorem you can see in two
dimensions. A factorised $q$ is *always* too narrow: the marginal variance it
reports is smaller than the true one. In the demo the truth has variance
$2.778$ and the diagonal approximation reports $1.000$. The means, remarkably,
are exact — the correlation is thrown away only in the second moment.

## Variants

All five variants maximise the same objective. They differ in the family and in
how the expectation under $q$ is differentiated.

| Variant | Family $q$ | Target it accepts | Gradient needed | Use when |
|:--|:--|:--|:--|:--|
| `mean_field` | diagonal Gaussian | `GaussianTarget` | none (closed form) | the target is quadratic; you want the exact mean-field solution in one pass |
| `cavi` | $q(\mu)q(\tau)$, Gaussian × Gamma | array of observations | none (closed form) | the model is conjugate and the coordinate updates are derivable |
| `reparameterization` | diagonal Gaussian | `log_prob` + `grad` | yes | differentiable target, $d$ from 2 to 10⁶ — the default |
| `structured_vi` | full-covariance Gaussian via Cholesky | `log_prob` + `grad` | yes | the posterior is strongly correlated and $d$ is small |
| `black_box_vi` | diagonal Gaussian | `log_prob` only | no | the density is a simulator, a compiled model, or has no usable gradient |

The three gradient variants are the same loop with a different estimator for
$\nabla_\phi \mathbb{E}_q[\log p]$. `mean_field` and `cavi` are the closed forms
available when the expectation is analytic; they exist here so that the Monte
Carlo estimators can be checked against something exact.

## Derivation

### 1. The bound and what it costs

$$\mathrm{ELBO}(q) = \mathbb{E}_q[\log p(z, x)] + \mathbb{H}[q(z)] \qquad (2)$$

Two terms, and the tension between them is the whole method. The first wants $q$
concentrated where the joint density is high; the second wants $q$ spread out.
Their sum is maximised in between, at the posterior. Maximising (2) is therefore
mode-seeking but entropy-regularised — the reason VI is never as sharp as a
Laplace approximation at the mode, and never as confident as a point estimate.

Because $\log p(x)$ does not depend on $q$, any $q$ that raises (2) is closer to
the posterior in KL. That is the licence to use *any* optimiser: the bound is a
proper objective function.

### 2. Mean field: why the variance is pinned

Factorise $q(z) = \prod_j q_j(z_j)$ and take the target in information form,

$$\log p(z) = -\tfrac12 z^{\top} P z + b^{\top} z \qquad (3)$$

unormalised, since the ELBO only needs $p$ up to a constant. The optimal factor
for coordinate $j$ is $\log q_j^\star(z_j) = \mathbb{E}_{-j}[\log p(z)] + \text{const}$, and the
expectation of a quadratic in $z_j$ is another quadratic:

$$\log q_j^\star(z_j) = -\tfrac12 P_{jj} z_j^2 + z_j\Big(b_j - \sum_{k \ne j} P_{jk} m_k\Big) + \text{const} \qquad (4)$$

Reading off the Gaussian gives both updates at once:

$$v_j = \frac{1}{P_{jj}}, \qquad m_j = \frac{b_j - \sum_{k \ne j} P_{jk} m_k}{P_{jj}} \qquad (5)$$

The variance is *not* the exact posterior variance $[P^{-1}]_{jj}$; it is fixed
by the diagonal entry alone, and by Cauchy–Schwarz $[P^{-1}]_{jj} \ge 1/P_{jj}$.
The means, however, solve $(Pm)_j = b_j$ at the fixed point, i.e. $m = P^{-1}b$
— and $P^{-1}b$ is exactly the posterior mean. **Mean field gets the means right
and the variances wrong, whenever the target is Gaussian.** That is not a
coincidence: it is the reason VI is used for means, and the reason its
uncertainties are unusable without a correction.

In the demo, $P = \begin{pmatrix}1 & 0.8\\ 0.8 & 1\end{pmatrix}$ and $b = (0.5, -0.25)^{\top}$.
The means converge to $(1.944, -1.806)$ to machine precision in 53 sweeps, the
variances are reported as $(1.000, 1.000)$ against a truth of $(2.778, 2.778)$,
and the bound sits at $2.550$ against an evidence of $3.061$ — a KL of
$0.511$, all of it the discarded correlation.

### 3. CAVI on a conjugate model

The Normal–Gamma model is the smallest model where coordinate ascent is exact
and the answer is known:

$$x_i \mid \mu, \tau \sim N(\mu, \tau^{-1}), \qquad \mu \mid \tau \sim N\!\left(\mu_0, (\kappa_0\tau)^{-1}\right), \qquad \tau \sim \mathrm{Gamma}(a_0, b_0) \qquad (6)$$

with $q(\mu, \tau) = q(\mu)q(\tau)$. The two conditionals are Gaussian and Gamma,
so every expectation in (2) is a closed form in $\psi(a)$, $\log b$, $a/b$:

$$a = a_0 + \frac{n+1}{2}, \qquad \mu = \frac{\kappa_0\mu_0 + n\bar x}{\kappa_0 + n} \qquad (7)$$

$$v = \frac{b}{(\kappa_0+n)\,a}, \qquad b = b_0 + \tfrac12\kappa_0\left(v + (\mu - \mu_0)^2\right) + \tfrac12\left(\textstyle\sum_i (x_i-\mu)^2 + n v\right) \qquad (8)$$

The `n+1` in (7) is the tell. The exact posterior for $\tau$ has shape
$a_0 + n/2$: the likelihood contributes $n/2$, the prior on $\mu$ contributes
$\tfrac12$ *through the integral* when you marginalise $\mu$ out. Coordinate
ascent does not integrate $\mu$; it replaces $\mu$ by its mean, so the
$\tfrac12\log\tau$ from $p(\mu \mid \tau)$ survives as an extra half-degree. The
variational factor is therefore a Gamma with the wrong shape — barely wrong at
$n=40$ ($22.5$ against $22.0$), catastrophic at $n=0$.

Substituting $v$ into $b$ makes the fixed point explicit, because the system is
linear in $b$:

$$b = b_N + \frac{b}{2a} \quad\Longrightarrow\quad b = \frac{b_N}{1 - 1/(2a)}, \qquad b_N = b_0 + \tfrac12\left[\textstyle\sum_i (x_i - \bar x)^2 + \frac{\kappa_0 n}{\kappa_n}(\bar x - \mu_0)^2\right] \qquad (9)$$

So $q(\tau)$ keeps the exact posterior's *rate*, inflated by $1/(1 - 1/(2a))$ —
about 2% at $a=22.5$ — and the whole approximation is one number. The demo
checks (9) to $10^{-12}$, and checks the consequence: $q(\mu)$ reports a variance
of $0.1007$ against the exact $0.1055$, the same underestimate as Section 2.

The evidence, computed by integrating (6) analytically, is

$$\log p(x) = -\frac{n}{2}\log 2\pi + \tfrac12\log\frac{\kappa_0}{\kappa_n} + a_0\log b_0 - a_n\log b_n + \log\Gamma(a_n) - \log\Gamma(a_0) \qquad (10)$$

with $\kappa_n = \kappa_0 + n$, $a_n = a_0 + n/2$, $b_n = b_N$. Using
$a_0 + (n+1)/2$ here instead of $a_0 + n/2$ — the variational shape, which is the
tempting copy-paste — makes the "ELBO" exceed the evidence by $0.7$ nats on this
demo. The bound is only a bound if both sides describe the same prior.

### 4. Pathwise gradients

For a reparameterisable family, $z = m + L\varepsilon$ with
$\varepsilon \sim N(0, I)$ moves the randomness outside the parameters, so the
gradient goes through the sample:

$$\nabla_m \mathrm{ELBO} = \mathbb{E}\left[\nabla_z \log p(z)\right], \qquad \nabla_L \mathbb{E}_q[\log p] = \mathbb{E}\left[g(z)\,\varepsilon^{\top}\right], \quad g = \nabla_z\log p \qquad (11)$$

Since $L$ is Cholesky, only the lower triangle is a free parameter; the upper
triangle is noise and must be masked. The entropy of a Gaussian in $L$ is
$\tfrac12\log\det(2\pi e\,LL^{\top})$, whose derivative with respect to each
diagonal entry is exactly $1/L_{ii}$ — a term with no sampling error at all.

For the diagonal family, parameterise $\log\sigma$ (unconstrained, and the
$+1$ in the entropy derivative becomes a literal `+ 1.0`):

$$z = m + \sigma\odot\varepsilon, \qquad \nabla_m = \mathbb{E}[g], \qquad \nabla_{\log\sigma} = \mathbb{E}[g \odot \sigma\odot\varepsilon] + 1 \qquad (12)$$

The reason this is the default is variance, not elegance. At a fixed $q$, the
demo measures the gradient of the logistic posterior and finds the score-function
estimator 921× and 887× noisier in the two coordinates. The pathwise estimator
costs one gradient evaluation per sample and has the variance of the gradient;
the score-function estimator has the variance of the *density times* the score,
which grows as $1/\sigma^2$ precisely when the fit is getting sharp.

### 5. Score-function gradients

When there is no gradient, the identity $\nabla_\phi q = q\,\nabla_\phi\log q$
still holds, so the derivative moves onto the log-density:

$$\nabla_\phi \mathbb{E}_q[f(z)] = \mathbb{E}_q\left[f(z)\,\nabla_\phi \log q(z)\right] \qquad (13)$$

For a diagonal Gaussian this is $\nabla_m = f(z)\,(z-m)/\sigma^2$ and
$\nabla_{\log\sigma} = f(z)\,(\varepsilon^2 - 1)$, plus the same entropy term.
Subtracting a baseline $B$ is free — $\mathbb{E}[\nabla\log q] = 0$ — and it is
the difference between usable and not: the estimator's variance is
$\mathrm{Var}(f)$, so centring $f$ removes the part of the noise that has nothing
to do with the parameter.

## Code walkthrough

The public API is one frozenset, and `fit` dispatches on it.

```python
VARIANTS = frozenset(
    ["mean_field", "cavi", "reparameterization", "structured_vi", "black_box_vi"]
)
```

`fit` takes three different kinds of target because the variants genuinely
consume different information: an array (a dataset, for `cavi`), a
`GaussianTarget` (an information-form quadratic, for `mean_field`), or a callable
pair. The `ValueError` in the last branch is deliberate — a `structured_vi` call
with no gradient fails at the door with what to do instead.

```python
    def fit(self, target: object, grad: Grad | None = None, *,
            mean0: np.ndarray | None = None) -> VariationalInference:
        """Run the variant. See the class docstring for what `target` must be."""
        if self.variant == "cavi":
            return self._fit_cavi(np.asarray(target, dtype=float))
        if self.variant == "mean_field":
            return self._fit_mean_field(target)
        if self.variant == "black_box_vi":
            return self._fit_gradient(target, None, mean0)
        if grad is None:
            raise ValueError(
                f"variant {self.variant!r} needs grad=...; use 'black_box_vi' when only "
                "the log density is available"
            )
        return self._fit_gradient(target, grad, mean0)
```

The mean-field loop is (5) written out. `off` is the off-diagonal contribution
$P_{j,\cdot}\,m$ minus the term already owned by coordinate $j$; recomputing it
per coordinate is what makes a sweep $O(d^2)$ rather than a matrix inverse. The
variance never appears in the loop at all — it is `1 / P_jj`, known before the
first step.

```python
        P, b = target.precision, target.linear
        lam = np.diag(P).copy()
        m = np.zeros(b.size)
        self.elbo_history_ = []
        n_done = 0
        for _ in range(self.n_iter):
            n_done += 1
            previous = m.copy()
            for j in range(b.size):
                off = P[j] @ m - P[j, j] * m[j]
                m[j] = (b[j] - off) / lam[j]
            self.elbo_history_.append(self._elbo_exact(target, m, 1.0 / lam))
            if np.max(np.abs(m - previous)) < self.tol:
                break
```

All four gradient variants share one optimiser loop and differ only in the
estimator. `_pack` turns the free parameters into `(mean, covariance)` once per
recorded step, so the ELBO is always evaluated at a valid Gaussian — the
`log σ` and Cholesky parameterisations never produce a non-positive covariance.

```python
            if self.variant == "black_box_vi":
                grads, baseline = self._score_grads(log_prob, m, log_s, rng, baseline)
            elif structured:
                grads = self._pathwise_grads(grad, m, rng, L=L)
            else:
                grads = self._pathwise_grads(grad, m, rng, log_s=log_s)
            opt.step(grads)
```

The pathwise estimator is (11) and (12). The structured branch's one subtle line
is `np.tril`: the gradient of $LL^{\top}$ with respect to the upper triangle is
zero, so any entry above the diagonal must be dropped or the factor slowly
drifts out of the triangular family. The two branches then add the entropy
derivative — `1 / diag(L)` for the Cholesky parameterisation, `1.0` for
$\log\sigma$.

```python
        eps = rng.standard_normal((self.n_samples, m.size))
        if L is not None:
            z = m + eps @ L.T
            g = np.array([grad(row) for row in z])
            g_m = g.mean(axis=0)
            g_L = np.tril((g.T @ eps) / self.n_samples)
            g_L[np.diag_indices_from(g_L)] += 1.0 / np.diag(L)
            return [g_m, g_L]
        s = np.exp(log_s)
        z = m + s * eps
        g = np.array([grad(row) for row in z])
        g_m = g.mean(axis=0)
        g_ls = (g * (s * eps)).mean(axis=0) + 1.0
        return [g_m, g_ls]
```

The score-function estimator is (13). The baseline is an exponential moving
average of `log p` at the current $q$, which is the cheap choice: a single
sample per step still produces a centred advantage, and the return value feeds
the next step.

```python
        s = np.exp(log_s)
        eps = rng.standard_normal((self.n_samples, m.size))
        z = m + s * eps
        lp = np.array([log_prob(row) for row in z])
        level = float(lp.mean()) if baseline is None else baseline
        adv = lp - level
        g_m = (adv[:, None] * (z - m) / s**2).mean(axis=0)
        g_ls = (adv[:, None] * (eps**2 - 1.0)).mean(axis=0) + 1.0
        return [g_m, g_ls], 0.9 * level + 0.1 * float(lp.mean())
```

The ELBO itself is estimated with `log q` analytic, because a Gaussian's log
density is known in closed form even when $p$ is not. That halves the variance
of the diagnostic: only the $\log p$ half is sampled.

```python
        d = mean.size
        L = np.linalg.cholesky(cov)
        eps = rng.standard_normal((n_samples, d))
        z = mean + eps @ L.T
        lp = np.array([log_prob(row) for row in z])
        solve = np.linalg.solve(L, (z - mean).T).T
        log_q = -0.5 * (d * LOG2PI + 2.0 * np.sum(np.log(np.diag(L))) + np.sum(solve**2, 1))
        return float(np.mean(lp - log_q))
```

CAVI is (7)–(9) in six lines. `mu` is computed once before the loop and never
moves, because the update for $\mu$ contains $\mathbb{E}[\tau]$ in both numerator
and denominator and it cancels — a fact worth noticing, since it is why the
Normal–Gamma mean-field problem is one-dimensional even though it looks like two.

```python
        a = a0 + 0.5 * (n + 1.0)
        mu = (kappa0 * mu0 + n * xbar) / (kappa0 + n)
        var_mu = 1.0 / (kappa0 + n)
        b = b0 + 0.5 * float(np.sum((x - xbar) ** 2))
```

The convergence test is the one non-obvious line in the file. Near a fixed point
the ELBO is *flat* — its gap is quadratic in the parameter error — so a threshold
on $\Delta\mathrm{ELBO}$ retires the loop while $b$ is still wrong in the fifth
digit. The loop therefore watches the parameter; the contraction factor is
$1/(2a)$, so seven sweeps are enough for twelve digits.

```python
            self.elbo_history_.append(self._cavi_elbo(x, mu, var_mu, a, b))
            # Stop when the *parameter* stops moving, not when the ELBO does:
            # near a fixed point the bound is flat (its gap is quadratic in the
            # error), so an ELBO threshold retires the loop while b is still
            # wrong in the fifth digit. This fixed point contracts by 1/(2a).
            if abs(b - previous) < self.tol * max(1.0, abs(b)):
                break
```

Finally the evidence, (10), which the demo uses as the ceiling. The docstring
records the half-degree, because that is the mistake this function exists to
detect.

```python
def _normal_gamma_evidence(x: np.ndarray, mu0: float, kappa0: float, a0: float,
                           b0: float) -> float:
    """`log p(x)` under the Normal-Gamma prior, in closed form.

    The shape here is `a0 + n/2`, *not* the `a0 + (n+1)/2` that the variational
    factor picks up: marginalising `μ` out of the joint is not the same as
    averaging it out of `log p`, and the extra half-degree is exactly the
    mean-field distortion in `q(τ)`.
    """
```

## Complexity

With $d$ parameters, $n$ observations, $S$ draws per step and $T$ steps, and
letting $c_p$, $c_g$ be the cost of one log-density and one gradient evaluation:

| Variant | Per sweep/step | Memory | Sweeps or steps to converge |
|:--|:--|:--|:--|
| `mean_field` | $O(d^2)$ | $O(d^2)$ for $P$ | $O(d)$ at worst, correlation-limited |
| `cavi` | $O(n)$ | $O(n)$ | $\sim\log(1/\epsilon)/\log(2a)$ — seven here |
| `reparameterization` | $O(S\,c_g + d)$ | $O(d)$ | thousands; set by the noise floor |
| `structured_vi` | $O(S\,c_g + d^2)$ | $O(d^2)$ | thousands, and the $d^2$ parameterisation dominates |
| `black_box_vi` | $O(S\,c_p + d)$ | $O(d)$ | thousands more, for the same accuracy |

The asymmetry in the last column is the practical content of the file. Closed
forms converge geometrically and exactly; stochastic gradients converge at
$O(1/\sqrt{T})$ and stop moving at a noise floor set by
$\mathrm{lr} \times \sqrt{\mathrm{Var}(g) / S}$. That is why the demo's
`structured_vi` recovers the covariance to $0.097$ rather than to machine
precision: it is a Monte Carlo ELBO, and the last three percent are noise.

Storing `elbo_history_` is $O(T)$; the ELBO itself is evaluated every
`n_iter // 100` steps, which is why the curve has 100 points regardless of $T$.

## Failure modes

**The variance is too small and nobody notices.** This is the signature failure,
not an edge case: a factorised $q$ is under-dispersed by construction. The demo
measures it three ways in one run — `1.000` against `2.778` on the quadratic
target, a correlation of $-0.81$ forcing a diagonal $q$ to be 40% narrower than
the Laplace approximation, and $0.1007$ against $0.1055$ in the conjugate model.
Every credible interval built from `std_` is too short, and downstream decision
rules that compare an uncertain quantity to a threshold will be overconfident.
*Diagnostic:* fit a `structured_vi` to the same target; if the covariance changes
the uncertainty, the diagonal family was the problem, not the data.

**The bound exceeds the evidence.** Impossible, so it means the two sides
disagree about the model. In this file that happens exactly once, and it is
instructive: using the variational shape $a_0 + (n+1)/2$ in (10) instead of the
posterior's $a_0 + n/2$ reports an evidence of $-93.209$ against a bound of
$-92.512$ — the bound stands $0.70$ nats *above* the ceiling it is supposed to
respect. The same class of bug appears when the target is unnormalised and its
constant is added back by hand. *Diagnostic:* evaluate at $q = p$. The ELBO must
come out equal to the evidence to numerical precision; if it does not, the two
formulas are not describing the same model.

**The step size is the accuracy.** With `lr = 0.06` and 256 samples the
`structured_vi` covariance is off by $0.097$; quadrupling the samples and
halving the step does not beat that by much, because the gradient's variance
never goes away. Two failure shapes follow. Too large a step: the iterate
oscillates around the optimum and the covariance jitters — visible as a
non-monotone `elbo_history_` tail. Too small a step in `black_box_vi`: the fit
stops short, and the reported `std_` $(0.277, 0.392)$ sits below the pathwise
fit's $(0.303, 0.416)$ — under-dispersion that is the optimiser's fault, on top
of the family's. *Diagnostic:* rerun with two seeds and one different step size.
An answer that moves by more than the reported standard error is not converged.

**The score-function gradient dies as the fit sharpens.** The estimator's
variance carries $1/\sigma^2$; a tight posterior is exactly when
`black_box_vi` stops being able to move. The demo needs 6,000 steps and 24
samples to match what 2,000 steps and 8 samples achieve with gradients, ending
$0.03$ away in the mean and narrower in both coordinates. *Fix:* a baseline
(implemented), control variates, or a reparameterisable family.

## Applications

**Latent variable models at scale.** The reparameterisation trick is what made
the variational autoencoder possible: an amortised encoder $q_\phi(z \mid x)$
turns per-datapoint inference into a shared network, and (11) makes the gradient
of the ELBO flow through the sample. Every VAE, and every normalising flow
trained variationally, is this file's `reparameterization` variant plus a neural
network for $q$.

**Topic models and mixture models.** LDA and its descendants are fit by CAVI —
the same coordinate ascent as `_fit_cavi`, with Dirichlet factors. When the
corpus is too large to sweep, stochastic VI replaces the full-data expectation
with a minibatch estimate and keeps the same bound; [`gmm.py`](../07-clustering/gmm.py)
does the mixture-model version of this with a Dirichlet prior on the weights.

**PPLs and general-purpose inference.** Stan, PyMC and NumPyro all ship an ADVI
mode that is `mean_field` with automatic differentiation supplying `grad`, plus
`fullrank` (this file's `structured_vi` with a transformed Cholesky factor).
It is the first thing to run on a new model because it is fast and gives a
gradient of the bound — not because it gives a correct posterior.

**Sparse Gaussian processes.** Titsias' variational approximation to a GP
posterior is an ELBO over inducing points, and it is the reason sparse GPs are
cheap; [`gaussian_process.py`](gaussian_process.py)'s `sparse_gp` variant is the
SoR form of the same idea.

Where it is a trap: a low-dimensional, strongly correlated, or multimodal
posterior. A diagonal Gaussian cannot represent a banana, and the KL it
minimises is the *reverse* KL — it will happily put mass in a valley between two
modes rather than commit to one. [`mcmc.py`](mcmc.py) exists for those cases,
and a model small enough for MCMC should not be approximated.

## References

- Jordan, Ghahramani, Jaakkola & Saul (1999). *An introduction to variational
  methods for graphical models.* Machine Learning 37, 183. The mean-field
  coordinate-ascent derivation, and the origin of the name CAVI.
- Wainwright & Jordan (2008). *Graphical models, exponential families, and
  variational inference.* Foundations and Trends in ML 1, 1. The convex-analytic
  view: the ELBO as a dual function, the gap as a divergence.
- Blei, Kucukelbir & McAuliffe (2017). *Variational inference: a review for
  statisticians.* JASA 112, 859. The best single survey; Sections 2–3 are this
  file.
- Hoffman, Blei, Wang & Paisley (2013). *Stochastic variational inference.*
  JMLR 14, 1303. Minibatched coordinate ascent, and the natural-gradient step
  that replaces Adam.
- Kingma & Welling (2014). *Auto-encoding variational Bayes.* ICLR. The
  reparameterisation trick, and the reason VAEs train at all.
- Rezende, Mohamed & Wierstra (2014). *Stochastic backpropagation and
  approximate inference in deep latent Gaussian models.* ICML. The same
  estimator, derived independently.
- Ranganath, Gerrish & Blei (2014). *Black box variational inference.* AISTATS.
  Score-function gradients with control variates — the `black_box_vi` variant.
- Kucukelbir, Tran, Ranganath, Gelman & Blei (2017). *Automatic differentiation
  variational inference.* JMLR 18, 1. ADVI, the default first pass in every PPL.
- Titsias (2009). *Variational learning of inducing variables in sparse Gaussian
  processes.* AISTATS. The sparse-GP ELBO.
- Turner & Sahani (2011). *Two problems with variational expectation
  maximisation for time-series models.* In *Bayesian Time Series Models.* Why
  the mean-field variance is too small, with the counterexamples that matter.
