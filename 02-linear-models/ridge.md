# Ridge

`ridge.py` · Linear models · Tier 1 · 6 variants

## Intuition

Ridge adds one sentence to least squares: *prefer smaller coefficients.* It
minimises `‖y − Xβ‖² + λ‖β‖²`, which replaces "find the closest point in the
column space" with "find the closest point, but pay for distance from the origin."

The consequence is the opposite of lasso's. Ridge never sets anything to zero —
it shrinks everything, in proportion to how *badly determined* each direction is.
A direction supported by a large singular value barely moves; a direction that is
almost in the null space (collinear columns, `p > n`) gets crushed. That is why
ridge is the fix for collinearity and lasso is the fix for "too many features."

## Variants

| Variant | What is penalised | Why it exists |
|:--|:--|:--|
| `closed_form` | `λ‖β‖²` | the default: one Cholesky solve |
| `svd` | same, in the singular basis | exposes the shrinkage factor; the numerically honest version |
| `generalized_ridge` | `Σ λ_j β_j²` | different penalty per coefficient (e.g. from a prior precision) |
| `loo_cv` | — | tunes `λ` with one factorisation, not `n` refits |
| `group_ridge` | one `λ_g` per block | multi-task / hierarchical structure |
| `nuclear_norm` | `λ‖X‖_*` on a matrix | the same idea for low-rank matrix completion |

## Derivation

**Closed form.** `∂/∂β [‖y − Xβ‖² + λ‖β‖²] = 0` gives

$$
(X^\top X + \lambda I)\,\hat\beta = X^\top y
\;\Longrightarrow\;
\hat\beta_\lambda = (X^\top X + \lambda I)^{-1}X^\top y
$$

The `+λI` also guarantees the matrix is invertible, which is the practical reason
ridge is often used even when nobody wants shrinkage.

**Shrinkage is explicit in the SVD.** Let `X = USVᵀ`. Then

$$
X^\top X + \lambda I = V(S^2 + \lambda I)V^\top,
\qquad
\hat\beta_\lambda = V\,\mathrm{diag}\!\left(\frac{s_i}{s_i^2+\lambda}\right)U^\top y
$$

so each singular direction is scaled by `s_i²/(s_i²+λ) ∈ (0,1)`. Two limits fall
out: `λ → 0` recovers OLS, and `λ → ∞` sends every coefficient to zero. The
shrinkage is *not* uniform — it is `1/(1 + λ/s_i²)`, so the least-supported
directions are shrunk hardest. That is exactly the behaviour you want on
collinear data.

**Degrees of freedom.** The hat matrix is `H = X(XᵀX + λI)⁻¹Xᵀ`, and

$$
\mathrm{df}(\lambda) = \mathrm{tr}(H) = \sum_i \frac{s_i^2}{s_i^2 + \lambda}
$$

which runs from `d` at `λ = 0` to `0` as `λ → ∞`. This is the right number to
compare against the residual sum of squares: `σ̂² = RSS/(n − df(λ))`.

**Bias and variance.** With `E[y] = Xβ`,

$$
\mathrm{bias}(\hat\beta_\lambda) = -\lambda(X^\top X + \lambda I)^{-1}\beta,
\qquad
\mathrm{Cov}(\hat\beta_\lambda) = \sigma^2 V\,\mathrm{diag}\!\left(\frac{s_i^2}{(s_i^2+\lambda)^2}\right)V^\top
$$

The variance always falls, the bias always grows, and the sum is minimised at some
`λ* > 0` unless `β` happens to lie exactly in the well-determined subspace. This is
the textbook bias–variance trade-off with a closed-form knob.

**Leave-one-out for free.** For any linear smoother with hat matrix `H`, the
leave-one-out residual is the ordinary residual divided by `1 − h_ii`:

$$
y_i - \hat y_{i}^{(-i)} = \frac{y_i - \hat y_i}{1 - h_{ii}}
$$

so `LOOCV(λ) = (1/n) Σ (r_i/(1−h_ii))²`, and every `λ` on a grid costs one
factorisation each — no refitting. This is what makes ridge tuning genuinely
cheap, and why `sklearn`'s `RidgeCV` is fast when it uses this form.

**Generalized and group penalties.** Replacing `λI` with `diag(λ_j)` is still a
one-line solve; it is the MAP estimate under a Gaussian prior with precisions
`λ_j`. Group ridge with a shared `λ_g` is the same thing with the entries of the
block set equal. The grouping only becomes interesting when the penalty is an L2
*norm* per group rather than a sum of squares — that is `group_lasso` in
`lasso.py`, which does produce exact zeros.

**Nuclear norm.** For a matrix `X`, the penalty `λ‖X‖_* = λΣσ_i` is the L1 norm
of the singular values, so its proximal operator is soft thresholding *of the
singular values*:

$$
\mathrm{prox}_{\lambda\|\cdot\|_*}(M) = U\,\mathrm{diag}\big((\sigma_i - \lambda)_+\big)V^\top
$$

Matrix completion alternates that step with re-imposing the observed entries
(`soft-impute`). Because the objective is convex, the iteration converges to the
global optimum — the low-rank analogue of the lasso's convexity argument.

## Code walkthrough

`_design` is the whole intercept policy in four lines:

```python
def _design(X: np.ndarray, fit_intercept: bool):
    if not fit_intercept:
        return X, np.zeros(X.shape[1])
    xm = X.mean(axis=0)
    return X - xm, xm
```

With an intercept, the fit is done on centred `X` and the offset is recovered at the
end (`ym − xm·β`). That is *not* the same as appending a ones column and penalising
it: the penalty is not translation-invariant, so a penalised intercept shrinks toward
zero and biases the slopes. Every variant here uses `Xc`, so the guarantee is uniform.

The three closed-form variants are one `solve` with different diagonals:

```python
    if variant == "closed_form":
        beta = np.linalg.solve(Xc.T @ Xc + np.diag(diag), Xc.T @ yc)
```

```python
    elif variant == "generalized_ridge":
        beta = np.linalg.solve(Xc.T @ Xc + np.diag(diag), Xc.T @ yc)
```

The second is identical apart from `diag` being supplied per coefficient. That is the
point: `λ_j β_j²` is still a quadratic, so the normal equations just get a different
diagonal. `group_ridge` fills the same diagonal blockwise:

```python
        pen = np.zeros(d)
        for j, g in enumerate(groups):
            pen[np.asarray(g)] = lam if penalty is None else np.asarray(penalty)[j]
```

`pen[g] = scalar` broadcasts the scalar across the group's indices — the same value
for every member, which is what "shared `λ_g`" means. Assigning `penalty[j]` per
index instead would give each member its own penalty and silently turn group ridge
into generalized ridge.

The SVD form is the one that explains the estimator:

```python
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        shrink = s / (s * s + lam)              # component-wise 1/(s + λ/s)
        beta = Vt.T @ (shrink * (U.T @ yc))
```

From `(S²+λ)⁻¹S`, not `S/(S²+λ)` applied to `Uᵀy` *after* multiplying by `V`: the
order in code is `s/(s²+λ)` scaled onto `Uᵀy`, then `Vᵀ`. For large `s`, `shrink ≈
1/s` and the direction is untouched (OLS); for small `s`, `shrink ≈ s/λ` and the
coefficient is crushed toward zero. The demo prints `shrink` and checks it is `< 1`
and decreasing, then verifies `λ=0` reproduces `lstsq` and `λ=1e8` gives `‖β‖ < 1e-4`.

`loo_cv` tunes with the hat matrix, and the grid is deliberately logarithmic:

```python
        grid = np.logspace(-2, 3, 40) if lams is None else np.asarray(lams)
        errors = [loo_error(X, y, ll, fit_intercept) for ll in grid]
        chosen = float(grid[int(np.argmin(errors))])
        beta = np.linalg.solve(Xc.T @ Xc + chosen * np.eye(d), Xc.T @ yc)
```

`loo_error` receives the **original** `X` and handles the intercept itself, because
the hat matrix needs the augmented design; the final refit uses the already-centred
`Xc`. Both are equivalent, but only if `fit_intercept` is passed consistently — a
`loo_error(Xc, ...)` call with `fit_intercept=True` would add a second intercept
column and double-count the offset.

The hat matrix is where the leave-one-out identity comes from:

```python
    Xa, D = _augmented(X, fit_intercept)
    return Xa @ np.linalg.solve(Xa.T @ Xa + lam * D, Xa.T)
```

`D` is `diag([0, 1, …, 1])`: the intercept column gets zero penalty. `H` is `n×n`,
which is the memory cost of the shortcut; only `diag(H)` is actually needed, so a
production version would compute `e_iᵀXa (XaᵀXa+λD)⁻¹ Xaᵀe_i` without forming `H`.

`loo_error` is then one line of algebra:

```python
    resid = y - Xa @ coef
    return float(np.mean((resid / (1.0 - np.diag(H))) ** 2))
```

The `(1 − h_ii)` denominator is the leverage correction: a point with `h_ii` near 1
is almost fitted by itself, so removing it changes the fit a lot and its LOO residual
is much larger than its training residual. As `λ → 0` and `n → d`, `h_ii → 1` and
the ratio blows up — which is the correct signal that LOO is unstable there, not a
numerical bug. The demo compares this to literally refitting `n` times for three
values of `λ` and matches to `1e-8`.

`nuclear_norm` is soft thresholding in the singular basis plus a projection:

```python
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_new = np.maximum(s - lam, 0.0)
        X_new = (U * s_new) @ Vt
        X_new[obs] = M[obs]
```

`s_new = max(s − λ, 0)` is `prox_{λ‖·‖_*}`; subtracting `λ` from every singular value
is exactly L1 shrinkage applied to the spectrum, and it is what produces a *low-rank*
answer rather than a small-norm one. The line `X_new[obs] = M[obs]` is the
data-consistency projection: the missing entries are free, the observed ones are
fixed. Without it, the iteration solves a different problem (denoising the full
matrix) and the held-out error is meaningless. The convergence test is relative
Frobenius, with a `1e-12` floor so an all-zero `X` does not divide by zero.

The demo's mask forces the first row and column to be observed:

```python
    mask[:, 0] = True                     # guarantee every row/column is seen
    mask[0, :] = True
```

A completely unobserved row is unconstrained by the objective, so its completion is
whatever the prior puts there; including it in the error would measure the prior, not
the algorithm. This is the operational version of the doc's failure mode.

## Complexity

| Variant | Time | Space |
|:--|:--|:--|
| `closed_form` | `O(nd² + d³)` | `O(d²)` |
| `svd` | `O(nd²)` | `O(nd)` |
| `generalized_ridge` / `group_ridge` | same as closed form | `O(d²)` |
| `loo_cv` over `G` values | `G·O(nd²)` (`XᵀX` reused) | `O(n²)` for `H`, or `O(n)` with the diagonal only |
| `nuclear_norm` per iteration | `O(min(m,n)²·max(m,n))` for the SVD | `O(mn)` |

The `loo_cv` row is the interesting one: `n` refits would be `n·O(nd²)`, while the
hat-matrix form is `O(nd² + Gn)` once `XᵀX` and its factorisation are reused.

## Failure modes

1. **Penalising the intercept.** Shrinking toward zero makes no sense unless the
   features are centred; the model then fits the mean badly. Always centre (or use
   an unpenalised intercept column, as `hat_matrix` does here).
2. **Forgetting to standardise.** `λ` penalises coefficients in their own units, so
   a feature in millimetres and one in kilometres get wildly different effective
   penalties. Standardise first, then unscale the coefficients.
3. **Reading `λ` across datasets.** `λ` has units of `XᵀX`, i.e. it scales with `n`
   and with the column norms. A `λ = 1` that works at `n = 1000` under-regularises
   at `n = 100`.
4. **Using ridge for sparsity.** Ridge coefficients are never exactly zero. If you
   need a support, you need lasso/elastic net (or thresholding afterwards, which is
   a different estimator with different properties).
5. **`loo_cv` on non-linear models.** The `r_i/(1−h_ii)` shortcut requires a linear
   smoother. Applying it to a model whose fit depends non-linearly on each point
   gives a wrong, too-optimistic CV curve.
6. **Nuclear norm with an unobserved row.** A matrix row with no entries is
   completely unconstrained; the completion is arbitrary there and the held-out
   error is meaningless. Guarantee at least one observation per row and column.

## Applications

- **Collinear or wide designs.** Genomics (`p ≫ n`), spectroscopy, and any dataset
  with duplicated or derived features.
- **Regularised deep learning.** Weight decay *is* ridge on the weights; the
  `l2` penalty in `SGD` is the same gradient term.
- **Kernel methods.** Kernel ridge regression is this file with `X → K`, which is
  how the closed form survives in infinite-dimensional feature spaces.
- **Recommender systems and matrix completion.** Nuclear-norm minimisation is the
  convex surrogate for low rank; Netflix-prize-style problems are its home.
- **Uncertainty quantification.** The Gaussian-prior reading gives a posterior
  covariance `σ²(XᵀX + λI)⁻¹`, so ridge is Bayesian linear regression with a
  zero-mean prior.

## References

- Hoerl & Kennard, *Ridge Regression: Biased Estimation for Nonorthogonal Problems*, 1970. Technometrics.
- Golub, Heath & Wahba, *Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter*, 1979. Technometrics.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, §3.4 (ridge, effective dof).
- Fazel, *Matrix Rank Minimization with Applications*, 2002 (nuclear norm as the convex envelope).
- Mazumder, Hastie & Tibshirani, *Spectral Regularization Algorithms for Learning Large Incomplete Matrices*, 2010 (soft-impute).
