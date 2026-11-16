# Elastic net

`elastic_net.py` · Linear models · Tier 1 · 4 variants

## Intuition

Lasso has two well-known failures. With `p > n` it can select at most `n`
variables — the support has to be linearly independent. And with a group of highly
correlated columns it picks one essentially at random and zeroes the rest, so the
selection is unstable under resampling.

Ridge has the opposite problem: it never zeroes anything, so it cannot select.

The elastic net is the interpolation. Add an `L2` term to the lasso objective and
the problem becomes *strictly convex*: the solution is unique, the correlated
columns share the weight instead of competing, and the `L1` term still produces
exact zeros. The `L2` term is doing exactly what `XᵀX` fails to do when columns
are collinear — making all directions equally well-conditioned.

## Variants

| Variant | Idea | Cost | Use |
|:--|:--|:--|:--|
| `naive` | the elastic net *is* a lasso on `[X; √(λ(1−α))I]` | `O((n+d)d)` memory | proof of equivalence, tiny `d` |
| `coordinate_descent` | closed-form 1-D update with the ridge term in the denominator | `O(nd)` per sweep | the workhorse |
| `path` | warm-started `λ` grid at fixed `α` | ≈ one fit | choosing `λ` |
| `group_elastic_net` | sparse-group prox: L2 shrink then L2 scale | `O(nd)` per iteration | grouped features |

## Derivation

**One coordinate.** Fix everything except `β_j`, let `a = ‖x_j‖²` and
`z = x_jᵀr_{−j}`. The subproblem gains a quadratic:

$$
\min_\beta\; \tfrac12 a\beta^2 - z\beta + \lambda\alpha|\beta| + \tfrac{\lambda(1-\alpha)}{2}\beta^2
$$

whose solution is the soft threshold with a rescaled denominator:

$$
\hat\beta_j = \frac{S_{\lambda\alpha}(z)}{a + \lambda(1-\alpha)}
$$

Compare with the lasso's `S_λ(z)/a`. The threshold moved from `λ` to `λα`, and the
denominator grew. Both changes point the same way: for `α < 1` the estimator
shrinks *more* than the lasso at the same `λ` — which is why `λ` grids are not
comparable across `α`, and why the `1/α` in `λ_max` matters.

**λ_max.** Setting `β = 0` makes the subgradient condition
`|x_jᵀy| ≤ λα` for every `j`, so

$$
\lambda_{\max}(\alpha) = \frac{\|X_c^\top y_c\|_\infty}{\alpha}
$$

At `α = 1` this is the lasso's `λ_max`; halving `α` doubles it. The demo checks
that one coefficient enters exactly at this value for three different `α`.

**The augmentation identity.** Let `X̃ = [X; √(λ(1−α))·I]` and `ỹ = [y; 0]`. Then

$$
\tfrac12\|\tilde y - \tilde X\beta\|^2
= \tfrac12\|y - X\beta\|^2 + \tfrac{\lambda(1-\alpha)}{2}\|\beta\|^2
$$

so the elastic net with parameters `(λ, α)` equals a **lasso with penalty `λα`** on
the augmented data. This is Zou & Hastie's observation, and it is the cleanest way
to see why `α < 1` cures both lasso pathologies: the augmented design has `n + d`
rows, so the "at most `n` selected" ceiling becomes `n + d`, and the appended
identity block makes the Gram matrix `XᵀX + λ(1−α)I` full rank by construction.
The demo verifies the identity to `4·10⁻¹⁶` over nine `(α, λ)` pairs.

**Why the columns share.** Take two identical columns `x₁ = x₂ = x`. Write a
solution as `(β₁ + β₂)` on `x` plus a difference component that costs nothing in
the loss. The `L1` term is minimised when the weight sits on one column
(`|β₁| + |β₂|` is constant only if the split is one-sided), so the pure lasso is
indifferent along that direction and any solver picks an arbitrary vertex. The
`L2` term is strictly convex, so the minimiser over the direction is the *even*
split `β₁ = β₂`. The demo shows the lasso returning `[2.32, 0.00]` and the elastic
net `[1.15, 1.15]`.

**Grouped extension.** For a block penalty `λα‖β_g‖₂ + λ(1−α)/2‖β_g‖₂²`, the
proximal operator of the sum is still closed-form. Completing the square in `β_g`:

$$
\arg\min_\beta \frac{1}{2t}\|\beta - z\|^2 + \lambda\alpha\|\beta\|_2 + \frac{\lambda(1-\alpha)}{2}\|\beta\|_2^2
= \frac{\max\!\big(1 - t\lambda\alpha/\|z\|_2,\,0\big)}{1 + t\lambda(1-\alpha)}\,z
$$

Two operations in sequence: shrink the whole block toward zero by an `L2` amount,
then scale it down by the ridge factor. `α = 1` gives the group lasso, `α = 0`
gives block ridge, and every intermediate value keeps the "all-in or all-out"
property of group selection.

**Strict convexity.** The Hessian is `XᵀX + λ(1−α)I`, positive definite for
`α < 1` even when `p ≫ n`. So the solution is unique — which is exactly what makes
the estimator stable under bootstrapping, and what makes the path continuous in
`λ` with a well-defined derivative almost everywhere.

## Code walkthrough

`lam_max` is where the `α` reparameterisation becomes visible:

```python
    Xc, yc, _, _ = _centre(X, y)
    return float(np.abs(Xc.T @ yc).max()) / max(alpha, 1e-12)
```

The subgradient at `β = 0` is `−Xᵀy + λα·sign(0)`, and the zero vector is optimal
when `‖Xᵀy‖∞ ≤ λα`. So the threshold is `‖Xᵀy‖∞/α`, not `‖Xᵀy‖∞`. Halving `α`
doubles the range of "λ that does nothing"; reusing a lasso grid at `α=0.5` therefore
starts *below* the true `λ_max` and the first few points are already non-trivial
fits. The `max(alpha, 1e-12)` avoids a division by zero at `α=0`, where the concept
of `λ_max` no longer exists (ridge never zeroes).

The coordinate update carries the ridge term in two places, and only one of them is
obvious:

```python
    sq = (Xc * Xc).sum(axis=0) + lam * (1.0 - alpha)
    ...
    thr = lam * alpha
    ...
            rho = Xc[:, j] @ r + (Xc[:, j] @ Xc[:, j]) * old
            # rho uses ‖x_j‖², not sq: sq includes the ridge term, which is not
            # part of the correlation.
            new = soft_threshold(rho, thr) / (sq[j])
```

The subproblem is `½(a + λ(1−α))β² − zβ + λα|β|`, so the denominator is `sq` and the
threshold is `λα`. But `z = x_jᵀr_{−j}` is a *correlation*, and the correction to the
full residual is `a·old = ‖x_j‖²·old` — not `sq·old`. Using `sq` there adds
`λ(1−α)·old` to the correlation, which is a spurious extra pull toward the current
value; the iteration still converges, to a different (over-shrunk) fixed point. This
is the single line in the file most likely to be copied wrong.

The zero guard is now on the combined denominator:

```python
    sq = np.where(sq <= 0, 1.0, sq)
```

For `α=1` and a constant column, `sq` is exactly zero; for `α<1`, the ridge term makes
it positive even then. The `<= 0` (rather than `== 0`) also catches a negative value
from floating-point subtraction, which would otherwise flip the sign of the update.

`naive` is the equivalence proof in code:

```python
    Xa = np.vstack([Xc, np.sqrt(lam * (1.0 - alpha)) * np.eye(d)])
    ya = np.concatenate([yc, np.zeros(d)])
    beta = coordinate_descent(Xa, ya, lam * alpha, alpha=1.0,
                              fit_intercept=False, iters=iters)["coef"]
```

The stacked rows contribute `½·λ(1−α)·β_j²` to the loss, exactly the ridge term, and
the lasso parameter passed is `λα` with `α=1` — i.e. a pure lasso. `fit_intercept=False`
is essential: the augmented matrix has no column of ones, and letting the inner call
centre it again would subtract a mean from the identity block and change the penalty.
The demo compares this to the closed-form update over nine `(α, λ)` pairs to `1e-8`,
which is a much stronger check than either solver alone.

`path` is the warm-start loop, unchanged from the lasso:

```python
    for i, lam in enumerate(lams):
        beta = coordinate_descent(X, y, lam, alpha, fit_intercept=fit_intercept,
                                  beta0=beta)["coef"]
        coefs[i] = beta
```

The monotonicity of the support in `λ` is what makes this safe at fixed `α`; it is
*not* monotone in `α`, which is the doc's last failure mode.

`group_elastic_net` is the proximal step for the block penalty:

```python
        for g in groups:
            g = np.asarray(g)
            nrm = np.linalg.norm(z[g])
            shrink = max(1.0 - step * lam * alpha / nrm, 0.0) if nrm > 0 else 0.0
            new[g] = shrink * z[g] / (1.0 + step * lam * (1.0 - alpha))
```

Two operations in order: the L2 block shrink `max(1 − tλα/‖z‖, 0)` and the ridge
scale `1/(1 + tλ(1−α))`. Swapping them is harmless (they are scalars), but folding
the `1/(1+…)` *inside* the `max` is not: it changes the group-zeroing threshold from
`tλα` to `tλα(1+tλ(1−α))`, so groups switch off at a different penalty. The `if nrm
> 0` branch and the `step = 1/L` choice are inherited from the group lasso; `L` is the
largest eigenvalue of `XᵀX`, so the gradient step never increases the smooth part.

## Complexity

| Variant | Time | Space |
|:--|:--|:--|
| `naive` | `O((n+d)d)` per sweep | `O((n+d)d)` — the augmented matrix |
| `coordinate_descent` | `O(nd)` per sweep, `O(nd·sweeps)` | `O(nd)` for `X`, `O(d)` state |
| `path` over `G` values | ≈ `O(nd)` × total sweeps (warm start) | `O(Gd)` for the curve |
| `group_elastic_net` | `O(nd)` per iteration, `O(1/L)` step | `O(d)` |

The augmentation column is the point of the table: forming `X̃` costs `d` extra rows
*and* destroys sparsity-friendliness, so the closed-form update is strictly better
in every practical setting. It survives as a variant because it is the proof.

## Failure modes

1. **Tuning `λ` without rescaling for `α`.** `λ_max` is `‖Xᵀy‖∞/α`. A grid tuned at
   `α = 0.5` puts *no* regularisation on the model at `α = 1` if reused. Recompute
   the grid per `α`, or tune `λ/λ_max`.
2. **Double shrinkage.** The elastic net shrinks the selected coefficients by an
   extra `1/(1 + λ(1−α))` relative to the lasso. Zou & Hastie recommend the
   rescaled estimate `(1 + λ(1−α))β̂` when the coefficients themselves are the
   deliverable; the demo's `naive` variant is the augmented (unrescaled) form, and
   the two differ by exactly that factor.
3. **Standardisation.** As with lasso and ridge, `λ` applies to coefficients in the
   features' own units. Always standardise, and unscale afterwards.
4. **Believing `α` fixed a correlated group completely.** The columns now *share*
   the weight, but the total mass on the group is still shrunk by the `L1` part.
   If you want a group treated as a unit, use the grouped penalty instead.
5. **Reading the support as truth.** Selection is still data-dependent; post-hoc
   `p`-values on the selected set are invalid. Sample-split or use de-biased
   estimates.
6. **Path over `α` with `λ` warm starts.** The support is not monotone in `α`, so
   warming from `α = 1` to `α = 0` can pass through solutions that are not on the
   `λ` path. Re-warm per `α`, from the previous `λ` grid point.

## Applications

- **Genomics.** Genes arrive in pathways and are highly co-expressed; the elastic
  net selects pathways and shares weight within them. This is its origin story.
- **Text and n-gram models.** Counts of overlapping n-grams are almost collinear.
- **Neuroimaging.** Voxels are spatially correlated by construction; lasso picks a
  single voxel per blob, the elastic net spreads the weight.
- **Sparse-group models.** When features have an obligatory group structure
  (categorical dummy blocks, Fourier coefficients, multi-task outputs), the sparse
  group lasso selects whole groups and sparsifies within them.
- **Portfolio construction.** Sparse weights (lasso) that are also stable under
  small changes to the return estimates (ridge) — the same objective, different
  names.

## References

- Zou & Hastie, *Regularization and Variable Selection via the Elastic Net*, 2005. JRSS-B (the objective and the augmentation).
- Friedman, Hastie & Tibshirani, *Regularization Paths for Generalized Linear Models via Coordinate Descent*, 2010. JSS.
- Simon, Friedman, Hastie & Tibshirani, *A Sparse-Group Lasso*, 2013. JCGS.
- Yuan & Lin, *Model Selection and Estimation in Regression with Grouped Variables*, 2006. JRSS-B.
- Tibshirani, *Regression Shrinkage and Selection via the Lasso*, 1996. JRSS-B (the baseline being improved on).
