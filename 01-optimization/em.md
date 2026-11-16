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
