# OLS

`ols.py` · Linear models · Tier 1 · 7 variants

## Intuition

Least squares answers one question: which linear combination of the columns lands
closest to `y`? Geometrically, `Xβ̂` is the orthogonal projection of `y` onto the
column space of `X`, so the residual `r = y − Xβ̂` is perpendicular to every column.
That single picture generates the normal equations, the standard errors, and the
reason QR beats `(XᵀX)⁻¹` on a computer.

Everything else in the file is about the two ways this simple answer goes wrong:
the offset in your data (intercept) and the arithmetic (conditioning).

## Variants

| Variant | How β̂ is computed | Cost | When |
|:--|:--|:--|:--|
| `normal_equations` | `solve(XᵀX, Xᵀy)` | `O(nd² + d³)` | tiny `d`, well-scaled columns |
| `qr` | `Rβ = Qᵀy` | `O(nd²)` | the default; stable without extra work |
| `svd_pinv` | `V S⁺ Uᵀ y` | `O(nd²)` | rank-deficient `X`, or you want the pseudoinverse |
| `cholesky` | `XᵀX = LLᵀ`, two triangular solves | `O(nd²/2 + d³/3)` | repeated fits with the same `X` |
| `recursive` | rank-1 updates (RLS) | `O(nd²)` total, `O(d²)` per row | streaming, or `n` too large for one pass |
| `with_intercept` | centre `X, y`, fit, recover `β₀` | `+O(nd)` | any model with an offset (i.e. most) |
| `standard_errors` | QR fit + `σ̂²(XᵀX)⁻¹` | `+O(d³)` | inference: is this coefficient real? |

## Derivation

**Normal equations.** Minimise `½‖y − Xβ‖²`:

$$
\nabla_\beta = -X^\top(y - X\beta) = 0
\;\Longrightarrow\;
X^\top X\,\hat\beta = X^\top y
\;\Longrightarrow\;
\hat\beta = (X^\top X)^{-1}X^\top y
$$

The residual is orthogonal to the column space, `Xᵀr̂ = 0`, and the projection
matrix is `H = X(XᵀX)⁻¹Xᵀ` (idempotent, symmetric — a projection in the algebraic
sense too).

**Intercept by centring.** With an unpenalised intercept the first-order condition
for `β₀` is `Σrᵢ = 0`, so the residual has zero mean and the fitted plane passes
through the centroid `(x̄, ȳ)`. Substituting `β₀ = ȳ − x̄ᵀβ` leaves

$$
\hat\beta = (X_c^\top X_c)^{-1}X_c^\top y_c,\qquad X_c = X - \bar x,\; y_c = y - \bar y
$$

Centring is not cosmetic: it is the *same* fit as including an intercept column,
but it keeps the intercept out of the penalty in `ridge.py` and `lasso.py`, which
is where it matters.

**Conditioning.** `XᵀX` squares the condition number: `κ(XᵀX) = κ(X)²`. On the
degree-12 Vandermonde in the demo, `κ(X) ≈ 1.2·10⁸`, so the normal equations
square that to `10¹⁶` and lose every significant digit — coefficient error `0.29`
versus `8·10⁻⁹` for QR. QR factors `X = QR` and solves `Rβ = Qᵀy`; `R` has the
same condition number as `X`, so the error grows like `κ(X)·ε`, not `κ(X)²·ε`.
SVD is equally stable and additionally tells you *which* directions are
unreliable (`s_i` small), which is why `lstsq` uses it.

**Recursive least squares.** Suppose you have `P = (XᵀX)⁻¹` and a new row `x`.
Then `XᵀX → XᵀX + xxᵀ`, and Sherman–Morrison gives

$$
P \leftarrow P - \frac{Pxx^\top P}{1 + x^\top Px},
\qquad
\hat\beta \leftarrow \hat\beta + \frac{Px}{1 + x^\top Px}\Big(y - x^\top\hat\beta\Big)
$$

The gain is largest when `Px` is large, i.e. when the new row points in a
direction the data has not pinned down yet. That is the Kalman filter's update
with `P` as the posterior covariance, and the reason online least squares
converges to the batch answer (the demo checks this to `4·10⁻¹⁵`).

**Standard errors.** Under `y = Xβ + ε`, `ε ~ N(0, σ²I)`:

$$
\widehat{\mathrm{Cov}}(\hat\beta) = \sigma^2 (X^\top X)^{-1},
\qquad
\hat\sigma^2 = \frac{\|r\|^2}{n - d}
$$

`n − d` (not `n`) because the fit used `d` degrees of freedom. The standard error
`se_j = √(σ̂²[(XᵀX)⁻¹]_{jj})` is the sampling standard deviation of `β̂_j` across
datasets — the quantity a t-test needs. It scales as `1/√n`, which the demo shows
by comparing `n = 200` with `n = 50`.

## Code walkthrough

The file separates the numerical path (`_solve`) from the model-level concerns
(intercept, standard errors), so the four direct solvers are one `if`-chain:

```python
def _solve(X: np.ndarray, y: np.ndarray, variant: str) -> np.ndarray:
    if variant == "normal_equations":
        return np.linalg.solve(X.T @ X, X.T @ y)
```

`solve`, not `inv`: same asymptotic cost, backward-stable, and one fewer matrix to
materialise. The next representation error is already in `X.T @ X`, which is why the
demo's Vandermonde separates this branch from the others by eight orders of magnitude.

Cholesky reuses the Gram matrix but still solves twice:

```python
    if variant == "cholesky":
        G = X.T @ X
        L = np.linalg.cholesky(G)
        z = np.linalg.solve(L, X.T @ y)
        return np.linalg.solve(L.T, z)
```

`G = LLᵀ`, so `Gβ = b` becomes `L(Lᵀβ) = b`: forward solve for `z = Lᵀβ`, backward for
`β`. The cost is `d³/3` for the factorisation instead of `d³/3` for an LU — the saving
appears when the *same* `X` is fitted against many right-hand sides, where `L` is
computed once. Note that this branch inherits the normal equations' conditioning: it
is faster than `qr`, not more stable.

The pseudoinverse branch truncates small singular values:

```python
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        tol = max(X.shape) * np.finfo(float).eps * s[0]
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        return Vt.T @ (s_inv * (U.T @ y))
```

`tol` is NumPy's `rcond` convention: singular values below `max(n,d)·ε·s_max` are
treated as zero. Without the truncation, a rank-deficient `X` divides by `1e-17` and
the "solution" has coefficients of order `1e17` that cancel to a finite prediction.
With it, the answer is the minimum-norm least-squares solution — the one `lstsq`
returns. The order matters: `s_inv * (U.T @ y)` scales the *coefficients in the
singular basis*, then `Vt.T` maps back; computing `Vt.T @ diag(s_inv) @ U.T @ y`
is the same thing but allocates a `d×d` diagonal.

QR is the default and the two-line version of the derivation:

```python
    if variant == "qr":
        Q, R = np.linalg.qr(X)
        return np.linalg.solve(R, Q.T @ y)
```

`np.linalg.qr` returns the reduced `Q` (`n×d`) and `R` (`d×d`), so `Qᵀy` is `d`
numbers and the triangular solve is `O(d²)` on top of the `O(nd²)` factorisation.
There is no explicit `QᵀQ = I` check because LAPACK guarantees it; the demo checks it
indirectly by comparing to SVD at `1e-10`.

`with_intercept` is the one variant that changes the model, not the solver:

```python
def _fit_intercept(X: np.ndarray, y: np.ndarray):
    xm = X.mean(axis=0)
    ym = y.mean()
    beta = _solve(X - xm, y - ym, "qr")
    return beta, float(ym - xm @ beta), X - xm, y - ym
```

`β₀ = ȳ − x̄ᵀβ` follows from the residual having zero mean. Centring is algebraically
identical to appending a column of ones, but it keeps the intercept *out of the
design matrix*, which is exactly what ridge and lasso need: penalising a constant
column would shrink the intercept toward zero and bias every slope. The demo's
no-intercept slope of `2.49` against a truth of `2.0` is the same effect in its
crudest form — the missing offset is absorbed by the slope because `x̄ = 10`.

Standard errors reuse the QR fit and then form the covariance:

```python
            rss = float(resid @ resid)
            sigma2 = rss / max(n - d, 1)
            cov = sigma2 * np.linalg.pinv(X.T @ X)
```

`n − d`, not `n`: the fit consumed `d` degrees of freedom, and using `n` makes every
standard error too small by `√(n/(n−d))`. `pinv` rather than `inv` keeps the output
finite for a rank-deficient design, where the honest answer is that some directions
have no estimable variance. `np.sqrt(np.diag(cov))` is then the vector of `se_j`;
the demo checks it against the textbook formula to `1e-8` and checks the `1/√n`
scaling by refitting on the first 50 rows.

Recursive least squares is Sherman–Morrison applied to the information matrix:

```python
    warm = min(max(d, 1), n)
    beta = np.linalg.lstsq(X[:warm], y[:warm], rcond=None)[0]
    P = np.linalg.pinv(X[:warm].T @ X[:warm])
    for xi, yi in zip(X[warm:], y[warm:], strict=True):
        Px = P @ xi
        denom = 1.0 + xi @ Px
        gain = Px / denom
        beta = beta + gain * (yi - xi @ beta)
        P = P - np.outer(gain, Px)
```

The warm start uses `d` rows (or `n` if smaller) so that `P` is full rank; starting
from `P = 0` would make every gain zero forever. `gain = Px/(1+xᵀPx)` is the Kalman
gain, and `P ← P − gain·(Px)ᵀ` is `P − PxxᵀP/(1+xᵀPx)` expanded — `np.outer(gain, Px)`
is exactly that rank-1 update. The `1 +` is not a regulariser: it is the
Sherman–Morrison denominator, and dropping it gives a `P` that shrinks too fast and a
`β` that stops moving. The `strict=True` on `zip` catches a length mismatch between
rows and targets, which otherwise silently truncates the stream to the shorter one.

## Complexity

| Operation | Time | Space |
|:--|:--|:--|
| `XᵀX`, `Xᵀy` | `O(nd²)` | `O(d²)` |
| Cholesky of `XᵀX` | `O(d³/3)` | `O(d²)` |
| QR of `X` | `O(nd²)` | `O(nd)` for `Q`, less with Householder |
| SVD of `X` | `O(nd²)` (thin) | `O(nd)` |
| RLS, all `n` rows | `O(nd²)` | `O(d²)` |

Memory, not time, separates them: `QR`/`SVD` need the factorisation (`O(nd)`),
whereas the normal equations only ever hold the `d×d` Gram matrix — which is why
`d = 10⁴` problems are still solved from `XᵀX` when `n` is a million.

## Failure modes

1. **Forgetting the intercept.** If `x̄ ≠ 0` the slope absorbs `ȳ`, and the bias is
   `y`'s mean divided by `x`'s mean. The demo's no-intercept slope is 2.49 against
   a truth of 2.0.
2. **Using `inv(XᵀX)`.** It is slower and less accurate than `solve`, and
   catastrophically wrong when `X` is collinear. Never form the inverse to use it
   once.
3. **Rank deficiency.** `d > n` or duplicated columns makes `XᵀX` singular; `solve`
   raises or returns garbage. Use `svd_pinv` (minimum-norm solution) or a ridge
   penalty.
4. **Residuals that are not noise.** Standard errors assume i.i.d. Gaussian errors.
   Heteroscedasticity, autocorrelation or a missing nonlinearity makes them wrong
   in a way no diagnostic on `R²` will show. Plot `r` against the fitted values.
5. **Extrapolating.** `Xβ̂` is a projection; outside the convex hull of the data
   nothing constrains it. The model has no notion of "far away" unless you add one.
6. **`R²` as the only metric.** Adding any column increases `R²`; the unbiased
   version `1 − (1−R²)(n−1)/(n−d)` and cross-validation are the honest numbers.

## Applications

- **Baselines.** Any regression task starts here; a model that cannot beat OLS is
  not adding value.
- **Inference in the sciences.** Coefficient ± standard error, with `n − d` degrees
  of freedom, is the workhorse of econometrics and epidemiology.
- **Online calibration.** RLS keeps a model up to date one observation at a time
  with `O(d²)` work — sensor calibration, adaptive control, online advertising.
- **Residual diagnostics.** OLS residuals are the "signal" that tree models,
  splines and neural nets are later asked to explain.
- **Matrix completion / matrix sensing subroutines.** Least-squares solves appear
  inside alternating minimisation for factorisation models.

## References

- Gauss, *Theoria motus corporum coelestium*, 1809 (the method, and the first
  recorded priority dispute with Legendre).
- Golub & Van Loan, *Matrix Computations*, 4th ed., §5 (QR, SVD, conditioning).
- Plackett, *Some theorems in least squares*, 1950. Biometrika (why RLS works).
- Sherman & Morrison, *Adjustment of an inverse matrix…*, 1949. Ann. Math. Stat.
- Seber & Lee, *Linear Regression Analysis*, 2nd ed., 2003.
