# Lasso

`lasso.py` · Linear models · Tier 1 · 6 variants

## Intuition

Ridge charges for the *size* of a coefficient (`λβ²`), which never quite reaches
zero. The lasso charges for its *existence* (`λ|β|`): the same `λ` no matter how
small the coefficient is, so a coefficient that only earns a little correlation
with the residual is better off switching off entirely.

That single change of exponent turns a shrinkage estimator into a *selector*. The
price is that the problem is no longer differentiable at zero and has no closed
form — hence six algorithms for one objective:

$$
\min_\beta \; \tfrac12\|y - X\beta\|_2^2 + \lambda\|\beta\|_1
$$

## Variants

| Variant | Algorithm | Selects? | Cost |
|:--|:--|:--|:--|
| `coordinate_descent` | cyclic exact 1-D updates | yes | `O(nd)` per sweep, few sweeps |
| `lars` | exact piecewise-linear path | yes | `O(d³ + nd²)` for the whole path |
| `homotopy` | continuation on a `λ` grid | yes | ≈ one fit per grid point, warm-started |
| `path` | the full `β(λ)` curve | yes | ≈ one fit total |
| `positive_lasso` | CD with `β ≥ 0` | yes, sign-constrained | same as CD |
| `group_lasso` | proximal gradient, L2 block penalty | whole groups | `O(nd)` per iteration |

## Derivation

**One coordinate, exactly.** Hold every other coefficient fixed and let
`a = ‖x_j‖²`, `z = x_jᵀr_{−j}`. The subproblem is

$$
\min_\beta \; \tfrac12 a\beta^2 - z\beta + \lambda|\beta|
$$

whose subgradient is `aβ − z + λ·sign(β)` (with `[−λ, λ]` at zero). Hence

$$
\hat\beta_j =
\begin{cases}
(z - \lambda)/a & z > \lambda\\
0 & |z| \le \lambda\\
(z + \lambda)/a & z < -\lambda
\end{cases}
= \frac{1}{a}\,S_\lambda(z),
\qquad
S_\lambda(z) = \mathrm{sign}(z)\,(|z| - \lambda)_+
$$

This is the **soft threshold**: the ridge solution `z/a` shifted toward zero by a
constant `λ/a`, not scaled. The shift is what creates exact zeros, and it is the
proximal operator of `λ|·|`, so coordinate descent here *is* proximal coordinate
descent. Each update is exact, so the objective decreases monotonically.

**KKT.** At the optimum, with `r = y − Xβ̂`:

$$
x_j^\top r = \lambda\,\mathrm{sign}(\hat\beta_j) \;\;(\hat\beta_j \neq 0),
\qquad
|x_j^\top r| \le \lambda \;\;(\hat\beta_j = 0)
$$

So the residual has *equal* correlation `λ` with every selected column and at most
`λ` with every dropped one. Both halves are checked in the demo. Setting
`β = 0` gives `x_jᵀy`, so the whole path starts at

$$
\lambda_{\max} = \|X_c^\top y_c\|_\infty
$$

and any `λ ≥ λ_max` returns the empty model — a fact worth knowing before you
choose a `λ` grid, because a `λ` that is small relative to `λ_max` is not "mild
regularisation", it is "no regularisation".

**Why zero, geometrically.** The constraint form is
`min ‖y − Xβ‖² s.t. ‖β‖₁ ≤ t`. The level sets of the loss are ellipsoids centred at
`β̂_OLS`; the constraint set is a diamond (a cross-polytope in `d` dimensions) with
*corners on the coordinate axes*. The first contact between a growing diamond and
an ellipsoid lands on a corner or edge with positive probability, and a corner has
some coordinates exactly zero. For the `L2` ball (ridge) the boundary is smooth, so
contact almost never lands on an axis — the coefficients are small but nonzero.

**Why LARS is exact and piecewise linear.** Define the active set `A`, the
correlations `c = Xᵀr`, and `C = max|c_j|`. Move along
`w_A = (X_AᵀX_A)⁻¹ s_A` (with `s_A = sign(c_A)`). Then for `k ∈ A`

$$
\dot c_k = -x_k^\top X_A w_A = -(X_A^\top X_A w_A)_k = -s_k
$$

so every active correlation falls at **unit rate**, independently of the column
scaling and of `|A|`: `C(t) = C − t`. That identity is the whole trick, and it is
why the segment lengths are closed-form:

- a new column joins when `c_k(t) = ±C(t)`,
  `t = (C ∓ c_k)/(1 ± a_k)` with `a_k = x_kᵀX_A w_A`;
- an active coefficient hits zero when `t = −β_j / w_j`;
- you stop at the target when `t = C − λ`.

The path is therefore a polyline whose breakpoints are those events; the demo
checks that LARS and coordinate descent agree to `1e-12` at five values of `λ`.
There is a subtlety that costs everyone an afternoon: the entering variable must
be carried explicitly, not re-detected by `argmax` with a tolerance, or a
floating-point tie stalls the path (or re-admits a variable that just dropped).

**Warm starts and homotopy.** The support is monotone in `λ` and the path is
continuous, so solving `λ_{k+1}` starting from `β(λ_k)` takes a handful of
coordinate sweeps instead of hundreds. This is the reason `glmnet` computes a
40-point path in about the time of one fit, and it is what `path` and `homotopy`
demonstrate: identical solutions to cold solves, `O(1)` fits of work.

**Group lasso.** With `‖β_g‖₂` instead of `|β_j|`, the proximal operator becomes a
block shrink:

$$
\mathrm{prox}_{\lambda t\|\cdot\|_2}(z) =
\Big(1 - \frac{\lambda t}{\|z\|_2}\Big)_+ z
$$

A whole group is scaled by one factor, so it is either all-in or exactly zero —
never partly. That is what you want when features arrive in obligatory bundles
(dummy levels of a categorical, polynomial terms, a filter bank): lasso would keep
one level of a categorical and drop the others, which is rarely meaningful.

## Code walkthrough

The file starts with the one nonlinearity that makes the lasso work:

```python
def soft_threshold(z, t):
    """The lasso prox: sign(z)·max(|z| - t, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)
```

This is the proximal operator of `λ|·|`, and it is a *shift*, not a scale. `np.sign(0)`
is `0`, so the origin is handled without a branch. Every solver below either calls this
directly or reproduces it in a block form.

`coordinate_descent` is the workhorse, and its residual bookkeeping is the part to
read carefully:

```python
            old = beta[j]
            rho = Xc[:, j] @ r
            rho = Xc[:, j] @ r + sq[j] * old          # partial residual correlation
            new = soft_threshold(rho, lam) / sq[j]
            ...
            if new != old:
                beta[j] = new
                r -= Xc[:, j] * (new - old)
```

`r` is the residual of the *full* current model, so `x_jᵀr` already includes
`−β_j‖x_j‖²`; adding `sq[j]·old` back gives the correlation with the partial residual
`r_{−j}` — the quantity the one-coordinate subproblem needs. Dropping that `+sq[j]*old`
is the classic lasso bug: the update becomes `soft(x_jᵀr, λ)/a`, which shrinks the
current coefficient again every sweep and converges to a too-sparse solution. The
incremental residual update `r -= x_j(new−old)` is what makes a sweep `O(nd)` instead
of `O(nd²)`.

Two smaller guards matter:

```python
    sq = (Xc * Xc).sum(axis=0)
    sq = np.where(sq == 0, 1.0, sq)
```

A constant column has zero centred norm; dividing by it is `nan`. Setting `1.0` makes
its `rho` zero, so it gets coefficient zero — the right answer for an unidentifiable
column. And the convergence test is `max |Δβ_j| < tol`, not the objective change:
coordinate descent can stall with a tiny objective change while a coefficient is still
moving, and the KKT check in the demo would then fail intermittently.

`lars` follows the path exactly, and three lines encode the geometry:

```python
        s = np.where(np.sign(c[A]) == 0.0, 1.0, np.sign(c[A]))
        w = np.linalg.solve(Xc[:, A].T @ Xc[:, A] + 1e-12 * np.eye(A.size), s)
        a = Xc.T @ (Xc[:, A] @ w)          # dc_k/dt = -a_k for every column
```

`w_A = (X_AᵀX_A)⁻¹s_A` is chosen so that the active correlations fall at unit rate:
`dc_k/dt = −x_kᵀX_Aw = −s_k`. That is why the segment length to the next event is
`(C − c_k)/(1 − a_k)` and not some ratio of norms. **No column standardisation** is
applied: standardising changes the penalty (it reweights `λ|β_j|` by the column norm),
so the path would no longer be the lasso path. The `1e-12` ridge is only to keep the
solve defined when `A` is collinear; it is small enough not to move the answer at the
demo's scale.

The join and drop events are computed in the same loop:

```python
            g_plus = (cmax - c[k]) / (1.0 - a[k]) if 1.0 - a[k] > 1e-12 else np.inf
            g_minus = (cmax + c[k]) / (1.0 + a[k]) if 1.0 + a[k] > 1e-12 else np.inf
            g = min(g_plus, g_minus)
            if g < gamma - 1e-14:
                gamma, kind, arg = g, "join", k
```

`c_k(t) = c_k − t·a_k` and the corridor is `±(cmax − t)`. Solving `c_k − ta_k =
cmax − t` gives `t = (cmax − c_k)/(1 − a_k)`; the negative side gives `(cmax + c_k)/(1
+ a_k)`. The guards `1 ∓ a_k > 1e-12` avoid division by zero when a column is already
perfectly aligned with the active set; those events are at infinity, not at `0`.

The bookkeeping variables are what keep the path from stalling:

```python
        if kind == "drop":
            dropped, pending = {arg}, None
        elif kind == "join":
            dropped, pending = set(), arg
```

`pending` is the variable that just joined at this knot. On the next iteration it is
added to `candidates` even if its correlation is one ulp below the maximum, because a
floating-point tie would otherwise make `argmax` pick a different column and the
algorithm would never make progress. `dropped` excludes a variable that just hit zero
from being re-admitted in the *same* knot; without it, a drop can immediately re-join
and loop forever. This is exactly the "subtlety that costs everyone an afternoon"
named in the doc, and it is encoded rather than hidden in a tolerance.

`path` and `homotopy` are warm-start loops:

```python
    for i, lam in enumerate(lams):
        fit = coordinate_descent(X, y, lam, positive=positive,
                                 fit_intercept=fit_intercept, beta0=beta)
        beta = fit["coef"]
```

The only difference from a cold solve is `beta0=beta`. Since the solution is
continuous in `λ` and the support is monotone, a handful of sweeps per grid point is
enough; a cold solve at each point would cost the full `5000`-iteration budget. The
demo checks the warm-started path equals cold solves at every `λ` to `1e-8`, which is
the property that makes the speed-up safe rather than approximate.

`group_lasso` is proximal gradient, with the step from the Lipschitz constant:

```python
    L = float(np.linalg.eigvalsh(Xc.T @ Xc).max()) + 1e-12
    ...
        grad = Xc.T @ (Xc @ beta - yc)
        z = beta - step * grad
        ...
            nrm = np.linalg.norm(z[g])
            new[g] = z[g] * max(1.0 - step * lam / nrm, 0.0) if nrm > 0 else 0.0
```

`step = 1/L` is the largest step for which the gradient step is guaranteed not to
increase the smooth part, so the iteration is monotone without a line search.
`max(1 − step·λ/‖z_g‖, 0)` is the block soft threshold: one scalar multiplies the whole
group, so a group is either entirely zero or entirely nonzero. The `if nrm > 0` branch
avoids `0/0` for a group that is already at the origin; without it, NumPy emits a
warning and produces `nan` that then propagates through every later gradient.

## Complexity

| Variant | Time | Space |
|:--|:--|:--|
| `coordinate_descent` | `O(nd)` per sweep, `O(nd·sweeps)` | `O(d)` (`O(nd)` if `X` is not cached) |
| `lars` | `O(d³ + nd²)` for the full path | `O(d²)` |
| `path` (warm-started CD) | ≈ `O(nd)` × total sweeps across all `λ` | `O(n·n_lams)` for the curve |
| `group_lasso` | `O(nd)` per iteration, step `1/L` | `O(d)` |

For `d ≤ 10⁴` coordinate descent wins in practice; for `d` in the hundreds with a
sparse answer, the whole LARS path is often cheaper than one CD fit at a small `λ`.

## Failure modes

1. **`λ` without a scale.** `λ` is in units of `Xᵀy`. Always start the grid at
   `λ_max = ‖Xᵀy‖∞` and use fractions of it; a hard-coded `0.1` means something
   different for every dataset. This implementation's demo reports `λ_max ≈ 227`
   for data whose coefficients are `O(1)`.
2. **Unstandardised columns.** `λ` is applied to `β_j` per unit of `x_j`, so a
   feature measured in small units gets a free pass. Standardise (and remember to
   unscale `β` for interpretation).
3. **Correlated features.** Lasso picks one of a collinear pair essentially at
   random and zeroes the other; refitting on a bootstrap sample changes *which*.
   `elastic_net` fixes this by mixing in an `L2` term, which shares the weight.
4. **`p > n`.** At most `n` coefficients can be nonzero (the support must be
   linearly independent, and you run out of directions). If you expected 500
   active genes from 100 samples, you are reading noise.
5. **Inference after selection.** Post-lasso `p`-values from a naïve OLS refit on
   the selected support are invalid — the support was chosen using `y`. Use
   sample splitting, the de-biased lasso, or a selective-inference procedure.
6. **The intercept.** It must not be penalised (centre `X` and `y` first). This
   implementation always centres, which is why `λ_max` is computed on centred data.

## Applications

- **Genomics and compressed sensing.** The original motivation: `p` in the
  thousands, `n` in the hundreds, and a sparse truth. Lasso is the baseline every
  new method is compared to.
- **Sparse linear models in production.** Interpretable risk scores, credit
  models, and any place where "which features matter" is the deliverable.
- **Feature screening.** Run lasso, keep the support, fit a flexible model on it
  (screening + refit is the standard two-stage recipe).
- **Grouped structures.** `group_lasso` for categorical blocks, multi-task
  problems, and the `L2,1` norm in multi-output regression.
- **Compressed sensing theory.** `λ`-regularised `L1` recovers an `s`-sparse signal
  from `O(s log d)` random measurements — the result that made sparsity a
  mainstream tool rather than a trick.

## References

- Tibshirani, *Regression Shrinkage and Selection via the Lasso*, 1996. JRSS-B.
- Efron, Hastie, Johnstone & Tibshirani, *Least Angle Regression*, 2004. Ann. Stat. (LARS, and the lasso modification).
- Friedman, Hastie & Tibshirani, *Regularization Paths for Generalized Linear Models via Coordinate Descent*, 2010. JSS (glmnet).
- Zou & Hastie, *Regularization and Variable Selection via the Elastic Net*, 2005. JRSS-B (why lasso struggles with `p > n` and groups).
- Yuan & Lin, *Model Selection and Estimation in Regression with Grouped Variables*, 2006. JRSS-B.
- Candès & Tao, *Decoding by Linear Programming*, 2005 (the compressed-sensing guarantee).
