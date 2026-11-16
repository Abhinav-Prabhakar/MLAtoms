# EM and friends

`em.py` · Optimization · Tier 1 · 6 variants

## Intuition

If the latent variables were visible, the model would be a one-line MLE. They are
not, so EM alternates between guessing them (E) and re-fitting the model as if the
guesses were data (M). The surprising part is that this cannot make the likelihood
worse — each iteration increases a lower bound, and the bound touches the true
likelihood at the current parameters.

Everything else here is a variation on that loop: what you do when the M-step has
no closed form (**MM**), when you have priors (**MAP-EM**), when the data does not
fit in memory (**incremental**), or when you deliberately throw away the soft
assignment (**hard EM**, a.k.a. k-means).

## Variants

| Variant | E-step | M-step | Guarantee | Use when |
|:--|:--|:--|:--|:--|
| `em` | responsibilities | closed form | LL non-decreasing | latent variable, tractable M |
| `mm` | construct a tangent majoriser | minimise surrogate | objective non-increasing | no tractable M-step |
| `map_em` | responsibilities | penalised MLE | posterior non-decreasing | priors / regularisation |
| `gmm_em` | Gaussian responsibilities | weighted moments | LL non-decreasing | the canonical instance |
| `incremental_em` | per-observation | `1/t` averaged stats | approximate | streaming / large `n` |
| `hard_em` | `argmax` assignment | cluster moments | classification LL | speed, k-means equivalence |

## Derivation

**The ELBO.** For observed `x`, latent `z`, parameters `θ`, and any distribution
`q(z)`:

$$
\log p(x\mid\theta) = \underbrace{\mathbb{E}_q\!\left[\log\frac{p(x,z\mid\theta)}{q(z)}\right]}_{\mathcal{L}(q,\theta)}
+ \underbrace{\mathrm{KL}\!\left(q(z)\,\|\,p(z\mid x,\theta)\right)}_{\ge 0}
$$

EM maximises `L` by coordinate ascent. The E-step sets `q(z) = p(z|x,θ_old)`, which
makes the KL zero and the bound tight:

$$
\mathcal{L} = \mathbb{E}_{z\sim p(z|x,\theta_{\text{old}})}\left[\log p(x,z\mid\theta)\right] + H(q)
$$

The M-step maximises that expectation over `θ` — the entropy term does not depend
on `θ`. Since the bound is tight both before and after, the likelihood cannot
decrease:

$$
\log p(x\mid\theta_{t+1}) \ge \mathcal{L}(q_t,\theta_{t+1}) \ge \mathcal{L}(q_t,\theta_t) = \log p(x\mid\theta_t)
$$

That is the entire proof, and it is why EM is robust and slow: it is a bound
optimiser, so it inherits the bound's slack.

**GMM equations.** With `π_k, μ_k, Σ_k`:

$$
r_{ik} = \frac{\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)},
\qquad
N_k = \sum_i r_{ik}
$$

$$
\pi_k = \frac{N_k}{n},\qquad
\mu_k = \frac{1}{N_k}\sum_i r_{ik}x_i,\qquad
\Sigma_k = \frac{1}{N_k}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top
$$

Notice that the M-step is exactly the MLE you would compute if `r` were a
one-hot assignment — that is the sense in which EM "fills in" the labels.

**MAP-EM.** Add a prior and maximise `E[log p(x,z|θ)] + log p(θ)`. For a Beta(a,b)
prior on the mixing weights the M-step becomes

$$
\pi_k = \frac{N_k + a - 1}{n + a + b - 2}
$$

which is the usual pseudo-count shrinkage. The monotonicity argument survives
verbatim with the posterior in place of the likelihood.

**MM.** When the M-step is intractable, replace the objective with a surrogate that
(a) touches it at the current point and (b) lies below it everywhere. Maximising the
surrogate can only improve the true objective. The Student-t location is the
cleanest example: `log(1+r²/ν)` is concave in `r²`, so its tangent at `r₀` is a
global upper bound, and minimising the resulting quadratic gives a weighted mean
with weights `(ν+1)/(ν+r²)`. Heavy-tailed data automatically gets down-weighted —
this is robust statistics falling out of an optimisation argument.

**Hard EM.** Replace `r_{ik}` with `1[k = argmax_j r_{ij}]`. This maximises a
different objective (the classification likelihood), so it is *not* monotone in
`p(x|θ)` — but it converges much faster and, for a shared spherical covariance,
the assignment rule is exactly nearest-mean. That is why k-means is hard EM with a
Gaussian mixture: the "algorithm" and the "model" are the same thing viewed from
two sides.

**Incremental EM.** Update the sufficient statistics with one observation at a
time, weighting the new information by `1/t`, and recompute the parameters from the
running averages. The cost per observation is `O(k)` instead of `O(nk)` per
iteration, and it converges to the same fixed point under a step-size schedule that
satisfies the Robbins–Monro conditions (`Σγ_t = ∞`, `Σγ_t² < ∞`).

## Code walkthrough

`run_em` is the template in twelve lines; the models are callbacks:

```python
def run_em(estep, mstep, init, iters: int = 200, tol: float = 1e-10, loglik=None):
    params = init
    hist = []
    for _ in range(iters):
        resp = estep(params)
        params = mstep(resp, params)
        if loglik is not None:
            hist.append(loglik(params))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol:
            break
    return params, hist
```

The likelihood is recorded **after** the M-step, so `hist[0]` is the first updated
iterate, not the initialisation. That matters for the monotonicity assertions: they
compare successive `hist` entries, and a version that logged before the update would
show `hist[0]` as the (usually worse) starting point and appear to violate the
guarantee. The `len(hist) > 1` guard exists because `hist[-2]` is invalid on the
first pass, and the tolerance is on the *change*, not on the gradient.

`bernoulli_em` shows why the E-step is a matrix operation:

```python
        return (np.log(pi)
                + X @ np.log(theta).T
                + (1.0 - X) @ np.log1p(-theta).T)
```

For `X` of shape `(n,d)` and `theta` of shape `(k,d)`, this is `(n,k)` in two
matmuls — no `(n,k,d)` tensor. `log1p(-theta)` rather than `log(1-theta)` keeps
precision when `theta` is near zero; `theta` is clipped to `[1e-12, 1-1e-12]` first
because a component can drive a feature probability to exactly 0 or 1, and both logs
are then infinite. The E-step subtracts the row max before exponentiating:

```python
        lp -= lp.max(axis=1, keepdims=True)
        r = np.exp(lp)
        return r / r.sum(axis=1, keepdims=True)
```

This is `softmax` up to the normalisation constant, which cancels in the ratio. The
M-step is the complete-data MLE with `r` in place of one-hot labels:

```python
        nk = r.sum(axis=0)
        ...
            theta = (r.T @ X + 1e-12) / (nk[:, None] + 1e-12)
```

`r.T @ X` sums each feature weighted by responsibility; dividing by the component's
soft count `nk` gives the weighted frequency. The `1e-12` in both numerator and
denominator is the smallest possible smoothing: without it, an empty component gives
`0/0`. With a prior, the same lines become pseudo-counts:

```python
            pi = (nk + a - 1.0) / (n + a + b - 2.0)
            theta = (r.T @ X + c - 1.0) / (nk[:, None] + c + e - 2.0)
```

`a−1` and `c−1` are the Beta pseudo-counts; the denominator adds the total prior mass
minus 2 so that `a=b=1` (uniform prior) reduces exactly to the MLE. Using `a` instead
of `a−1` is a one-off that looks harmless and shifts every weight.

`student_t_mm` is the smallest complete MM example:

```python
        r2 = (x - mu) ** 2
        w = (nu + 1.0) / (nu + r2)
        mu_new = float((w * x).sum() / w.sum())
        hist.append(0.5 * (nu + 1.0) * float(np.log1p(r2 / nu).sum()))
        if abs(mu_new - mu) < 1e-12:
            break
        mu = mu_new
```

`w_i = (ν+1)/(ν+r_i²)` is the tangent majoriser's weight; a point at `r=0` gets
weight `(ν+1)/ν`, a point at `r=10` gets `(ν+1)/(ν+100)` — the down-weighting is
automatic. The convergence test uses `mu_new − mu` but does **not** assign on the
final iteration; that is harmless for the output (`mu` and `mu_new` differ by
`<1e-12`) and keeps `hist` aligned with the iterates actually visited. `log1p(r²/ν)`
is the Student-t NLL up to an additive constant — dropping the constant is fine for
monotonicity but means the printed value is not comparable across `ν`.

`_log_gaussian` is the numerically important helper:

```python
    L = np.linalg.cholesky(cov)
    diff = X - mu
    sol = np.linalg.solve(L, diff.T).T
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + logdet + (sol * sol).sum(axis=1))
```

The quadratic form is `‖L⁻¹(x−μ)‖²`, computed by a triangular solve, not
`diff @ inv(cov) @ diff.T`. The inverse is `O(d³)` *and* loses precision when the
covariance is ill-conditioned; the Cholesky factor is already needed for the
log-determinant, so the solve is free. `logdet = 2Σ log L_ii` avoids `slogdet`'s sign
bookkeeping and is exact for a PD matrix.

`gmm_em`'s E-step handles the shared-covariance case by shape:

```python
        if covs.ndim == 2:  # shared spherical covariance
            lp = np.column_stack([_log_gaussian(X, means[j], covs) for j in range(k)])
        else:
            lp = np.column_stack([_log_gaussian(X, means[j], covs[j]) for j in range(k)])
```

A shared covariance is a single `(d,d)` matrix; per-component covariances are
`(k,d,d)`. The `ndim` check is what lets `cov="spherical"` reuse the same code path
with one matrix, rather than carrying `k` copies of the same array. The hard
assignment is a one-hot built by indexing the identity:

```python
        if hard:
            r = np.eye(k)[r.argmax(axis=1)]
```

This is both the assignment and a valid responsibility matrix, so the M-step below is
unchanged — which is exactly the statement that hard EM is the same loop with a
different E-step.

The M-step is weighted moments, with `reg` added *after* the normalisation:

```python
        nk = r.sum(axis=0) + 1e-12
        pi = nk / n
        means = (r.T @ X) / nk[:, None]
        ...
                covs[j] = (r[:, j, None] * diff).T @ diff / nk[j] + reg * np.eye(d)
```

`(r[:,j,None]*diff).T @ diff` is `Σ_i r_ij (x_i−μ_j)(x_i−μ_j)ᵀ`, the weighted scatter.
Adding `reg·I` after dividing keeps the result PD even when a component collapses
onto one point — adding it before dividing would scale the regulariser by `nk` and
make it vanish for small components, which is the opposite of what is needed. The
spherical branch pools the variance over all coordinates:

```python
            var = float((r * ((X[:, None, :] - means[None]) ** 2).sum(-1)).sum()
                        / (n * d))
            covs = np.eye(d) * max(var, reg)
```

`max(var, reg)` is a floor on the *variance*, not on the matrix; with `var` exactly
zero, `_log_gaussian` would still fail on a singular Cholesky.

`incremental_em` is the only loop without a full E-step:

```python
            gamma = 1.0 / max(t / k, 1.0)
            nk = (1 - gamma) * nk + gamma * r
            sx = (1 - gamma) * sx + gamma * r * x[i]
            means = sx / np.maximum(nk, 1e-12)
            pi = nk / nk.sum()
```

`t` counts observations, not epochs, so `t/k` is "observations per component"; the
`max(·, 1)` makes the first `k` updates fully replace the statistics (`gamma=1`)
instead of averaging against zeros. After that, `gamma = k/t` decays like `1/t`,
satisfying Robbins–Monro. `sx` is the weighted sum, `nk` the soft count, and the mean
is their ratio — exactly the batch M-step, applied to running statistics. Dividing by
`max(nk, 1e-12)` again handles a component that no observation has touched.

## Complexity

For a GMM with `n` points, `k` components, `d` dimensions, one EM iteration is
`O(nkd²)` for full covariances (the Cholesky of each `d×d` covariance dominates
when `d` is large) and `O(nkd)` for spherical ones. Memory is `O(nk)` for the
responsibilities — the term that forces incremental or mini-batch EM at scale.

## Failure modes

1. **Singular covariances.** A component that collapses onto one point has
   `|Σ| → 0` and the likelihood diverges. Fix: `reg·I` on the covariance (as here),
   a Wishart prior, or a minimum eigenvalue floor.
2. **Local optima.** EM is a bound optimiser and the bound has many stationary
   points. Run several restarts, or initialise with k-means++ instead of random
   points.
3. **Label switching.** Component indices are not identified; comparing two runs
   requires matching (Hungarian algorithm) before computing parameter error.
4. **Hard EM is biased.** Assigning `argmax` makes the effective sample size per
   component smaller than `N_k`, so variances are underestimated. Never use hard
   EM to fit a covariance you care about.
5. **Incremental step sizes.** A constant step size makes the estimate track the
   last few points rather than converge; `1/t` is required, and `1/t` makes early
   updates huge. The `max(t/k, 1)` guard in this implementation is the usual
   compromise.
6. **Missing data treated as zero.** EM handles missingness by integrating over it
   in the E-step; substituting the mean silently biases every covariance downward.

## Applications

- **Mixtures:** GMM for density estimation, speaker identification, and as the
  emission model in HMMs.
- **Missing data:** the original EM application — fill in the missing entries in
  the E-step, re-estimate the covariance in the M-step.
- **Latent variable models:** factor analysis, probabilistic PCA, and the VAE's
  precursor (the ELBO is EM's bound with an amortised `q`).
- **Robust estimation:** MM for Student-t, Huber and other M-estimators; the
  weighted-mean update is the same one used in iteratively reweighted least
  squares.
- **Clustering at scale:** hard EM = k-means, and mini-batch/incremental EM for
  datasets that do not fit in memory.

## References

- Dempster, Laird & Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm*, 1977. JRSS-B.
- Neal & Hinton, *A View of the EM Algorithm that Justifies Incremental, Sparse, and other Variants*, 1998.
- Hunter & Lange, *A Tutorial on MM Algorithms*, 2004. The American Statistician.
- Bishop, *Pattern Recognition and Machine Learning*, ch. 9 (mixtures and EM), 2006.
- Celeux & Govaert, *A Classification EM Algorithm for Clustering and Two Stochastic Versions*, 1992.
- Cappé & Moulines, *On-line Expectation–Maximization Algorithm for Latent Data Models*, 2009. JRSS-B.
