# Gaussian process regression

`gaussian_process.py` · Probabilistic models · Tier 1 · 9 variants

## Intuition

A Gaussian process is a distribution over *functions*: pick any finite set of
inputs and the function's values there are jointly Gaussian. The covariance of
that Gaussian is given by a kernel, which is where all modelling choices live.

$$f \sim \mathcal{GP}(0, k), \qquad
y = f(x) + \varepsilon, \quad \varepsilon \sim N(0, \sigma_n^2) \qquad (1)$$

Because the prior is Gaussian and the likelihood is Gaussian, conditioning is
linear algebra. No sampling loop, no gradient descent, no evidence lower bound:
the posterior mean is `K* K⁻¹ y` and the posterior variance is
`K** − K* K⁻¹ K*ᵀ`. Everything the model knows is a matrix inverse.

The consequence that matters is that the model returns a **distribution**, not a
point. `predict` gives a mean *and* a standard deviation, the standard deviation
shrinks toward zero as you approach observed data, and it grows back to the
prior when you leave the data behind. That is what makes GPs the default
surrogate in Bayesian optimisation and active learning, and it is also why
nobody uses them on a million rows: the inverse costs `O(n³)`.

The kernel is the model. `rbf` says the function is infinitely differentiable;
`matern` says it is differentiable a specific number of times; `periodic` says
it repeats; `linear` says it is a line, and a GP with that kernel is exactly
Bayesian linear regression. Composition (`sum_kernel`, `product_kernel`) is how
a "trend plus seasonal plus noise" model gets built without a new algorithm.

## Variants

| Variant | Kernel | Say it when |
|:--|:--|:--|
| `rbf` | `σ²exp(−r²/2ℓ²)` | smooth, no reason to doubt infinite differentiability |
| `matern` | `(1+√3r)exp(−√3r)` per `ν` | physical data; `ν = 1/2` gives rough, OU-like paths |
| `rational_quadratic` | `σ²(1+r²/2αℓ²)^{−α}` | a mixture of length scales (heavy-tailed spectrum) |
| `periodic` | `σ²exp(−2sin²(πΔ/p)/ℓ²)` | a hard period `p`: orbits, seasons, duty cycles |
| `linear` | `σ²xᵀx'` | you want to *prove* it is ridge regression |
| `sum_kernel` | `k₁ + k₂` | trend plus local wiggle, additive structure |
| `product_kernel` | `k₁ · k₂` | periodicity whose amplitude decays away from the trend |
| `sparse_gp` | `K ≈ K_nm K_mm⁻¹ K_mn` | `n` too large for `O(n³)`; `m` inducing points |
| `warped` | `y = log(z − a)` | positive, skewed targets; a plain GP over-smooths the peaks |

The first seven change the *prior*; the last two change the *computation*.
`sparse_gp` trades a little accuracy for a much smaller matrix, and `warped`
changes the observation model rather than the function class.

## Derivation

### Conditioning a joint Gaussian

Stack the observed outputs and the values at test inputs:

$$\begin{bmatrix} y \\ f_* \end{bmatrix} \sim
N\!\left(0, \begin{bmatrix} K + \sigma_n^2 I & K_* \\ K_*^\top & K_{**}\end{bmatrix}\right),
\qquad (2)$$

with `K = k(X,X)`, `K_* = k(X,X_*)`, `K_{**} = k(X_*,X_*)`. The conditional of a
partitioned Gaussian is the Schur complement:

$$\mu_* = K_*^\top (K+\sigma_n^2I)^{-1} y, \qquad
\Sigma_* = K_{**} - K_*^\top (K+\sigma_n^2I)^{-1} K_*. \qquad (3)$$

That is the whole algorithm, and every implementation detail below exists to
compute it without ever forming an inverse.

### Cholesky, not inverse

Let `K + σ²I = LLᵀ`. Then `α = L⁻ᵀL⁻¹y` and `v = L⁻¹K_*`, so

$$\mu_* = K_*^\top\alpha, \qquad
\Sigma_* = K_{**} - v^\top v. \qquad (4)$$

Two triangular solves per prediction, no `np.linalg.inv` anywhere. The variance
is a difference of two positive numbers, which is why it is clipped at zero in
the code — with a badly conditioned `K` the subtraction can go slightly
negative, and a negative variance is a silent `nan` in the standard deviation.

### The evidence: choosing a kernel without a validation set

The marginal likelihood of the data under the prior is available in closed form
because everything is Gaussian:

$$\log p(y \mid X) = -\tfrac12 y^\top(K+\sigma_n^2I)^{-1}y
- \tfrac12\log\lvert K+\sigma_n^2I\rvert - \tfrac{n}{2}\log 2\pi. \qquad (5)$$

The first term rewards fitting the data; the second is an **Occam factor** — a
kernel that claims the data is surprising has a small `|K|`, and the log
determinant punishes it. This is the principled way to compare `rbf` against
`matern` on the same data, and it is why the demo reports both RMSE and
evidence: they do not have to agree, and the disagreement is informative.

### The kernels

*RBF.* Stationary, isotropic, infinitely mean-square differentiable:

$$k(x,x') = \sigma_f^2\exp\!\left(-\frac{\lVert x-x'\rVert^2}{2\ell^2}\right). \qquad (6)$$

*Matérn.* The spectral density is a Student-t, so the sample paths are only
`ν`-times differentiable in mean square. Only half-integer `ν` avoid a Bessel
function, and those are the three that matter:

$$k_{1/2} = e^{-r}, \qquad k_{3/2} = (1+\sqrt3 r)e^{-\sqrt3 r}, \qquad
k_{5/2} = (1+\sqrt5 r+\tfrac53 r^2)e^{-\sqrt5 r}, \qquad (7)$$

with `r = ‖x−x'‖/ℓ`. `ν = 1/2` is the Ornstein–Uhlenbeck process: continuous
everywhere, differentiable nowhere. `ν → ∞` recovers the RBF. If a fitted RBF
looks too smooth for the data, the fix is a Matérn, not a smaller length scale.

*Rational quadratic.* An infinite scale mixture of RBFs, `ℓ` distributed
inverse-gamma:

$$k = \sigma_f^2\left(1+\frac{r^2}{2\alpha\ell^2}\right)^{-\alpha},
\qquad \alpha \to \infty \text{ gives RBF}. \qquad (8)$$

*Periodic.* Correlation depends on `sin` of the distance, so it is exactly
periodic in `p` and smooth in between:

$$k = \sigma_f^2\exp\!\left(-\frac{2\sin^2(\pi\lvert x-x'\rvert/p)}{\ell^2}\right). \qquad (9)$$

*Linear.* `k = σ_f² xᵀx'`. Substituting into (4) and using Woodbury gives

$$\mu_* = x_*^\top\left(X^\top X + \frac{\sigma_n^2}{\sigma_f^2}I\right)^{-1}X^\top y,
\qquad (10)$$

which is ridge regression with `λ = σ_n²/\sigma_f²` and no intercept. A GP with
a linear kernel *is* a Bayesian linear model — the prior on `w` is
`N(0, I/σ_f²)`. The demo asserts this to `1e-8`, which is a much stronger test
of the GP solver than any regression score.

*Composition.* Covariances of independent functions add when the functions add
and multiply when the functions multiply, so

$$k_{1+2} = k_1 + k_2, \qquad k_{1\cdot2} = k_1\,k_2, \qquad (11)$$

and the result is a valid kernel because the pointwise operations preserve
positive semi-definiteness (Schur product theorem for the product). That is the
entire "automatic statistician" idea: a grammar of kernels, and one search over
it.

### Sparse: replace `K_nn` by a low-rank approximation

With `m` inducing points `Z`, the subset-of-regressors approximation is

$$K_{nn} \approx Q = K_{nm}K_{mm}^{-1}K_{mn}, \qquad (12)$$

which is the exact covariance if the function is determined by its values at
`Z`. Substituting into (3) would form an `n×n` matrix; the Woodbury identity
keeps everything at `m×m`. Writing `A = L_m^{-1}K_{mn}` (so `Q = A^\top A`) and
`B = \sigma_n^2 I_m + AA^\top`:

$$(Q+\sigma_n^2I)^{-1} = \sigma_n^{-2}\left(I - A^\top B^{-1}A\right), \qquad
\lvert Q+\sigma_n^2I\rvert = \sigma_n^{2(n-m)}\lvert B\rvert, \qquad (13)$$

so both the mean and the evidence cost `O(nm²)` and no `n×n` matrix is allocated.
The price is a variance that is too small between inducing points, because the
approximation is exact only *on* the manifold spanned by `k(·, Z)`.

### Warped: a Gaussian process for a non-Gaussian target

If `y` is positive and skewed, a GP on `y` wastes its prior on impossible values
and over-smooths the peaks. Warped GPs put the GP on a transformed scale,

$$g(y) = \log(y - a), \qquad p(y) = p(g)\,\lvert g'(y)\rvert,
\qquad \log p(y) = \log p(g) - \log(y-a), \qquad (14)$$

and choose `a` by maximising that Jacobian-corrected likelihood. Forgetting the
Jacobian term is the classic bug: it biases `a` toward the boundary and makes
the warp look better than it is.

Prediction has to come back through a non-linear map, and the mean of a
log-normal is not the exponential of the mean:

$$\mathbb{E}[y] = a + e^{\mu+\sigma^2/2}, \qquad
\mathrm{Var}[y] = e^{2\mu+\sigma^2}\left(e^{\sigma^2}-1\right). \qquad (15)$$

That `σ²/2` is not a detail — it is the difference between a calibrated
prediction and one that is biased low by exactly the uncertainty.

## Code walkthrough

`sqdist` is the only geometry primitive in the file, and it uses the expansion
`‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b` so nothing of size `n×m×d` is ever allocated. The
clip matters: floating point can return `-1e-17` for a point against itself, and
`sqrt` of that is `nan`:

```python
def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """`‖a − b‖²` for every pair, without forming the `n×m×d` tensor."""
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)
```

Every kernel is a `__call__(A, B)` plus a `diag`, and the diag override is the
reason prediction is `O(n)` per point rather than `O(n²)`: the diagonal of an RBF
Gram matrix is the constant `σ_f²`, and computing it as `np.diag(self(A, A))`
would build the whole matrix to read `n` entries:

```python
    def __call__(self, A, B):
        r2 = sqdist(A, B) / self.length_scale ** 2
        return self.sigma_f ** 2 * np.exp(-0.5 * r2)

    def diag(self, A):
        return np.full(A.shape[0], self.sigma_f ** 2)
```

The Matérn branch is a lookup, but the `ν` guard is load-bearing: `ν = 1` has no
closed form without a Bessel function and silently returns the wrong kernel if
it is allowed through:

```python
        r = np.sqrt(sqdist(A, B)) / self.length_scale
        if self.nu == 0.5:
            k = np.exp(-r)
        elif self.nu == 1.5:
            s = np.sqrt(3.0) * r
            k = (1.0 + s) * np.exp(-s)
        else:
            s = np.sqrt(5.0) * r
            k = (1.0 + s + s * s / 3.0) * np.exp(-s)
```

`_solve` is the numerical core: one Cholesky, two triangular solves, and the
evidence accumulated at the same time. `np.log(np.diag(L)).sum()` is the
log-determinant, and it is the only way to get it without underflow:

```python
    def _solve(self, X: np.ndarray, y: np.ndarray):
        """Cholesky of `K + σ²I`, the linear solve, and the log-likelihood."""
        s2 = self.sigma_n ** 2
        K = self.kernel_(X, X) + (s2 + self.jitter) * np.eye(X.shape[0])
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        ll = (-0.5 * float(y @ alpha) - np.log(np.diag(L)).sum()
              - 0.5 * y.size * np.log(2.0 * np.pi))
        return L, alpha, float(ll)
```

Prediction is (4) twice — once for the mean, once for the variance. Solving with
`L` rather than forming `K⁻¹` is `O(n²)` per test point instead of `O(n³)`:

```python
        Ks = self.kernel_(self.X_, X)
        mean = Ks.T @ self._alpha
        v = np.linalg.solve(self._L, Ks)
        var = self.kernel_.diag(X) - (v ** 2).sum(axis=0)
        return mean, np.maximum(var, 0.0)
```

The sparse fit is (13) in six lines. `A` is `L_m⁻¹K_mn`, `B` is the `m×m`
matrix the Woodbury identity asks for, and `w` is `(Q+σ²I)⁻¹y` computed without
ever knowing `Q`:

```python
        Kmm = self.kernel_(Z, Z) + self.jitter * np.eye(m)
        Lm = np.linalg.cholesky(Kmm)
        # A = Lm⁻¹ K_mn, so that Q = AᵀA = K_nm K_mm⁻¹ K_mn
        A = np.linalg.solve(Lm, self.kernel_(Z, X))              # (m, n)
        B = s2 * np.eye(m) + A @ A.T                             # (m, m)
        c = np.linalg.solve(B, A @ y)                            # (m,)
        w = (y - A.T @ c) / s2                                   # (Q+σ²I)⁻¹ y
        self.inducing_mean_ = np.linalg.solve(Lm.T, A @ w)       # K_mm⁻¹K_mn w
```

The warping objective is the marginal likelihood of the *transformed* data minus
`Σ log z`, the Jacobian of (14). One line of golden-section search over `θ`
recovers the offset, and there is no gradient because there is only one
parameter:

```python
        def objective(theta: float) -> float:
            a = ymin - np.exp(theta)
            z = y - a
            if (z <= 0.0).any():
                return -np.inf
            g = np.log(z)
            _, _, ll = self._solve(X, g)
            return ll - float(np.log(z).sum())      # the Jacobian of the warping
```

Back-transforming is where a warped GP quietly goes wrong. Both the mean and the
variance need the `σ²/2` term, and the observation noise enters scaled by
`(dg/dz)² = e^{2g}`:

```python
            scale = np.exp(2.0 * mean)
            ymean = self.warp_a_ + np.exp(mean + 0.5 * var)
            yvar = scale * (np.expm1(var) + self.sigma_n ** 2)
            return (ymean, np.sqrt(yvar)) if return_std else ymean
```

Sampling needs the full posterior covariance, not just its diagonal. Drawing
`N(mean_i, var_i)` independently at each point would match the marginals and
produce white noise instead of smooth functions, so `_posterior_cov` forms the
Schur complement and `sample` takes its square root:

```python
    def _posterior_cov(self, X: np.ndarray) -> np.ndarray:
        """`K** − K*ᵀ K⁻¹ K*`: the covariance `predict` only reports the diagonal of."""
        Kss = self.kernel_(X, X)
        if self.variant == "sparse_gp":
            Ksm = self.kernel_(X, self.inducing_)                  # (n*, m)
            C = Kss - Ksm @ np.linalg.solve(
                self._Lm.T, np.linalg.solve(self._Lm, Ksm.T))
        else:
            v = np.linalg.solve(self._L, self.kernel_(self.X_, X))
            C = Kss - v.T @ v
        return 0.5 * (C + C.T)
```

## Complexity

`n` training points, `n_*` test points, `d` dimensions, `m` inducing points.

| Stage | Exact GP | Sparse GP |
|:--|:--|:--|
| `fit` | `O(n³)` (Cholesky) | `O(nm² + m³)` |
| memory | `O(n²)` | `O(nm)` |
| `predict` mean | `O(nn_*)` | `O(mn_*)` |
| `predict` variance | `O(n²n_*)` | `O(m²n_*)` |
| `sample` | `O(n³ + n²n_*)` | `O(n_*³)` |

The `n³` is the wall. A laptop handles `n` in the low thousands; beyond that the
matrix alone does not fit and an approximation is not optional. Kernel
composition does not change the asymptotics — a sum of twenty kernels is still
one `n×n` Cholesky — it only changes the constant by the number of terms.

## Failure modes

**The `n³` wall.** At `n = 10⁴` the `K` matrix is 800 MB and a Cholesky is
`10¹²` flops; at `n = 10⁵` it is impossible. The diagnostic is simply the size of
the data. The fix is inducing points (`sparse_gp`), a variational bound, or a
different model — not a bigger machine.

**The length scale is the model, and it is not fitted here.** Hyperparameters
are inputs in this file. A length scale much smaller than the typical spacing
makes the GP interpolate noise and grow spurious wiggles between points; much
larger than the data range makes it a constant. The diagnostic is the evidence
from (5) compared across a grid of `ℓ`, and the fix is an optimiser over it
(which is `01-optimization`'s problem, not this atom's).

**Duplicate points plus a tiny noise floor.** Two identical rows make `K`
exactly singular, and `cholesky` raises `LinAlgError` rather than producing a
bad answer. `jitter` in the constructor is the fix; the diagnostic is the
exception itself, or `np.linalg.eigvalsh(K).min() ≤ 0`.

**Confident in the wrong direction.** Away from the data the variance returns to
`k(x,x)`, which looks reassuring, but "away" is measured in the kernel's metric.
In high dimensions every test point is far from every training point under
`‖·‖`, so the GP reports near-prior uncertainty everywhere and still returns a
mean that extrapolates linearly. The signal is a predictive band that never
shrinks regardless of how much data you add.

**A wrong warp is worse than no warp.** `warped` assumes a specific one-parameter
family. If the target is skewed in a way `log(y−a)` cannot represent, the
Jacobian term can make the likelihood prefer a `a` that flattens the data, and
the fit degrades below the plain GP. Compare the two on held-out data — the demo
does — rather than trusting the training evidence alone.

## Applications

Bayesian optimisation is the flagship: the GP is cheap to update and its
variance is exactly the exploration signal an acquisition function needs.
Surrogate modelling and computer-experiment emulation are the same use case with
a different name — fit a GP to 200 expensive simulations, optimise the cheap
surrogate. Kriging in geostatistics is a GP with a Matérn kernel under a
different vocabulary.

Time series with honest error bars, active learning (query where the variance
is largest), and calibration of simulation models all lean on the same two
formulas. Kernel composition is what makes it practical: `sum_kernel` of a
linear trend and a periodic component is a seasonal regression with a
principled uncertainty band.

Where it is a trap: high-dimensional inputs (`d > 20` needs a learned metric or
a deep kernel), more than a few thousand points, non-stationary processes
(a GP assumes the same smoothness everywhere), and any setting where only the
point prediction matters — a gradient-boosted tree will be faster and often more
accurate.

## References

- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
  Learning*, MIT Press 2006 — the reference; chapters 2 and 5 cover everything
  here, and chapter 8 the approximations.
- M. L. Stein, *Interpolation of Spatial Data*, Springer 1999 — the Matérn
  family and why the smoothness parameter is not cosmetic.
- J. Quiñonero-Candela and C. E. Rasmussen, *A unifying view of sparse
  approximate Gaussian process regression*, JMLR 6 (2005) — SoR, DTC, FITC and
  the rest, in one framework.
- E. Snelson and Z. Ghahramani, *Warped Gaussian processes*, NeurIPS 2004 — the
  warping, the Jacobian, and why the mean needs the variance term.
- D. Duvenaud, *Automatic model construction with Gaussian processes*, PhD
  thesis, Cambridge 2014 — the kernel grammar behind `sum_kernel` and
  `product_kernel`.
- C. K. I. Williams and D. Barber, *Bayesian classification with Gaussian
  processes*, IEEE TPAMI 20 (1998) — the Woodbury and determinant identities
  that make the sparse variant `O(nm²)`.
