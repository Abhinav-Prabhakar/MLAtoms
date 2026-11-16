# Linear algebra

`linalg.py` · Primitives · Tier 1 · 11 variants

## Intuition

Almost every algorithm in this repository reduces to one of four questions:

1. **Solve** `Ax = b` — regression, Kalman filters, Gaussian processes.
2. **Factorise** `A = LLᵀ` or `QR` — the stable way to answer (1).
3. **Compress** `A ≈ USVᵀ` — PCA, LSA, low-rank adapters.
4. **Update cheaply** — Woodbury, Sherman-Morrison, rank-1 Kalman updates.

The recurring lesson: *never form an inverse*, and *never form `XᵀX`*.

## Variants

| Variant | Cost | Gives you | Use for |
|:--|:--|:--|:--|
| `lu` | `2/3 n³` | `PA = LU` | general square solves |
| `qr` | `2mn²` | orthonormal basis | least squares, eigenvalue algorithms |
| `cholesky` | `1/3 n³` | `A = LLᵀ` | SPD systems (GP, ridge, Kalman) |
| `svd` | `O(mn²)` | optimal low rank | PCA, conditioning, pseudoinverse |
| `truncated_svd` | same | rank-k factors | compression, LSA |
| `randomized_svd` | `O(mnk)` | approximate rank-k | large matrices |
| `eigh` | `O(n³)` | symmetric spectrum | spectral clustering, PCA via covariance |
| `power_iteration` | `O(n² · iters)` | dominant eigenpair | PageRank, spectral methods |
| `woodbury` | `O(n²k + k³)` | inverse of `A + UCV` | LoRA, Kalman, low-rank updates |
| `solve_triangular` | `O(n²)` | `Lx = b` | after any factorisation |
| `condition_number` | `O(n³)` | `κ(A)` | deciding whether to trust a solve |

## Derivation

**Why not the normal equations.** `x = (XᵀX)⁻¹Xᵀy` squares the condition number:
`κ(XᵀX) = κ(X)²`. With `κ(X) = 10⁸` (common when features are collinear or
differ in scale), the solve loses all 16 digits. `QR` avoids it because
`κ(R) = κ(X)`:

$$
X = QR,\quad X\beta = y \;\Longrightarrow\; R\beta = Q^\top y
$$

which is a triangular solve — `O(nd²)` instead of `O(d³)`, and stable.

**Cholesky.** For SPD `A`, the factorisation `A = LLᵀ` exists and is unique with
positive diagonal. The recurrence is:

$$
L_{jj} = \sqrt{A_{jj} - \textstyle\sum_{k<j} L_{jk}^2},
\qquad
L_{ij} = \frac{A_{ij} - \sum_{k<j} L_{ik}L_{jk}}{L_{jj}}
$$

Half the work of LU, and it *fails* exactly when `A` is indefinite — a free
correctness check that Gaussian-process code relies on.

**Eckart-Young.** Among all rank-`k` matrices, the SVD truncation minimises both
the Frobenius and the spectral norm error:

$$
\|A - A_k\|_F^2 = \sum_{i>k}\sigma_i^2,
\qquad A_k = U_k S_k V_k^\top
$$

That optimality is why PCA is not a heuristic.

**Randomised range finder.** Compute `Y = AΩ` for a Gaussian `Ω ∈ ℝ^{n×(k+p)}`,
orthonormalise (`Q`), then solve the small problem `B = QᵀA`. Power iteration
`Y ← A(AᵀY)` sharpens the range when the spectrum decays slowly. The error is
`O(σ_{k+1})` with high probability.

**Woodbury.** For `A` invertible and `C` `k×k`:

$$
(A + UCV)^{-1} = A^{-1} - A^{-1}U\left(C^{-1} + VA^{-1}U\right)^{-1}VA^{-1}
$$

The inner inverse is `k×k`, so a rank-3 update to a `10⁵×10⁵` matrix costs
`O(n²k)`, not `O(n³)`. Sherman-Morrison is the `k=1` case; LoRA training is the
same identity applied to the Hessian/preconditioner.

## Code walkthrough

The module is deliberately split: `svd`, `eigh` and `condition_number` are thin
wrappers over LAPACK, while LU, QR, Cholesky, power iteration and Woodbury are
written out because their *structure* is the thing being taught. The docstring says
this so that a reader does not mistake `qr` for production code.

LU is Doolittle elimination with partial pivoting, and the pivot swap is the part
that is easy to get wrong:

```python
    for k in range(n - 1):
        p = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[p, k]) < 1e-14:
            raise ValueError("matrix is singular to working precision")
        if p != k:
            A[[k, p]] = A[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
```

`p = k + argmax(...)` because `argmax` is relative to the slice. Swapping rows of
`A` and `P` is expected; swapping the *first `k` columns* of `L` is the subtle one.
Those entries were computed in earlier steps and belong to the old row order, so if
they are not swapped the product `PᵀLU` no longer reconstructs `A`. A version that
forgets this line passes for matrices that need no pivoting (symmetric PD, diagonally
dominant) and fails on random ones — exactly the demo's matrix.

The elimination itself is a rank-1 update:

```python
        L[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k:] -= np.outer(L[k + 1 :, k], A[k, k:])
```

One `outer` replaces the double loop, and it is the same operation as a Schur
complement step. `np.triu(A)` at the end is not cosmetic: the lower triangle now
holds the multipliers, which have already been copied into `L`.

QR is Householder, and the sign convention is the numerical point:

```python
        x = A[k:, k]
        e = np.zeros_like(x)
        e[0] = np.sign(x[0]) if x[0] != 0 else 1.0
        v = x + e * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
```

The reflector is `I − 2vvᵀ` with `v = x + sign(x₀)‖x‖e₁`. Choosing `+sign(x₀)`
(rather than always `+`) makes `v` point *away* from `x`, so `‖v‖` is large and the
subtraction `x − 2v(vᵀx)` does not cancel. Always using `+` is unstable exactly when
`x₀ > 0`, which is half the time — and it still produces an orthogonal `Q`, so tests
that only check `QᵀQ = I` never catch it. The `else 1.0` handles `x[0] == 0`, where
`sign` would be zero and `v` would be zero.

The reflectors are applied on both sides:

```python
        A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
```

`A ← H A` on the trailing block, and `Q ← Q H` accumulated from the right, starting
from `I`. Applying the same reflector to the whole of `A` would be correct but
quadratic in the wrong dimension; the trailing-block slice is what makes it `2mn²`.
`Q[:, :n]` at the end truncates the thin `Q` for `m > n`.

Cholesky is the one factorisation that doubles as a definiteness test:

```python
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0:
            raise ValueError("matrix is not positive definite")
        L[j, j] = np.sqrt(s)
        L[j + 1 :, j] = (A[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
```

`s` is the Schur complement of the leading `j×j` block; if `A` is SPD it is strictly
positive, and if it is not, `sqrt` of a negative would be `nan` and every downstream
solve would silently return `nan`. Raising here is why GP code can trust it. The
`<=` (not `< 0`) also rejects a semidefinite matrix at the exact boundary, where the
factor exists but is singular.

`randomized_svd` is the range finder in four lines of linear algebra:

```python
    Q = rng.normal(size=(n, p))
    Y = A @ Q
    for _ in range(iters):
        Q, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(A.T @ Q)
        Y = A @ Z
    Q, _ = np.linalg.qr(Y)
```

`p = k + oversample` (not `k`) is what makes the range contain the top `k` singular
directions with high probability: the sketch has to be wider than the target rank,
or the estimate is biased low, not just noisy. Each loop iteration is one power step
`Y ← A Aᵀ Y` with reorthonormalisation, and reorthonormalising *inside* the loop
matters — without it the columns of `Y` collapse onto the dominant direction and the
extra iterations buy nothing. The final small SVD is of `B = QᵀA`, which is `p×n`;
`Q @ Ub[:, :k]` lifts the left singular vectors back.

`power_iteration` checks both signs when testing convergence:

```python
        v_new = w / nw
        if min(np.linalg.norm(v_new - v), np.linalg.norm(v_new + v)) < tol:
            v = v_new
            break
```

An eigenvector is defined up to sign, so `v` and `−v` are the same answer. Testing
only `v_new − v` makes the loop run the full `iters` whenever the iterates settle on
the negative sign — correct output, 10× the work. The returned eigenvalue is the
Rayleigh quotient `vᵀAv`, which is accurate to `O(ε)` even when `v` is only accurate
to `√ε`.

Woodbury is written with `inv` for readability, and the docstring is honest about
what it buys:

```python
    Ainv = np.linalg.inv(A)
    k = C.shape[0]
    inner = np.linalg.inv(np.linalg.inv(C) + V @ Ainv @ U) if k else np.zeros((0, 0))
    return Ainv - Ainv @ U @ inner @ V @ Ainv
```

The `k×k` inner inverse is the cheap part; the expensive part is `Ainv`, which is
`O(n³)` if computed here. The identity pays off when `A⁻¹` is *already known* — a
Kalman filter carrying `P`, an optimizer carrying a preconditioner, LoRA adapting a
frozen base — not when it has to be computed for the update. Calling this function in
a loop with a fresh `A` is slower than one `solve`. The `if k else` branch keeps the
zero-rank update exact instead of producing a `0×0` matmul error.

`solve_triangular` is the `O(n²)` kernel that every factorisation feeds:

```python
    idx = range(n) if lower else range(n - 1, -1, -1)
    for i in idx:
        s = b[i] - (L[i, :i] @ x[:i] if lower else L[i, i + 1 :] @ x[i + 1 :])
        x[i] = s / L[i, i]
```

Forward substitution reads `x[:i]` (already computed); backward substitution reads
`x[i+1:]`. The two slices are the entire difference, and swapping them produces a
result that is correct only for diagonal matrices — while still running to completion.
Note this indexes `L[i, i+1:]` but uses `x[i+1:]`; an off-by-one in either slice is
silent for the first few `i`.

## Complexity

| Operation | Time | Memory |
|:--|:--|:--|
| LU / Cholesky | `O(n³)` (`2/3` vs `1/3` constant) | `O(n²)` |
| QR (m ≥ n) | `O(mn²)` | `O(mn)` |
| SVD (full) | `O(mn²)` | `O(mn)` |
| Randomized SVD | `O(mnk)` | `O((m+n)k)` |
| Power iteration | `O(n²·iters)` | `O(n²)` for dense `A` |
| Woodbury | `O(n²k + k³)` | `O(nk)` |

## Failure modes

1. **`np.linalg.inv` in a loop.** It is `O(n³)` *and* less accurate than a solve.
   `solve(A, B)` is the same cost but backward-stable.
2. **Forming `XᵀX`.** Destroys precision on collinear features; use QR or SVD.
3. **Assuming `A` is SPD.** Cholesky on a covariance matrix with a duplicated
   column raises — good. Silently adding `1e-8·I` hides the real problem; check why
   it is singular first.
4. **Randomised SVD with too few oversamples.** With `k+p = k`, the range misses
   directions and the approximation is not just inaccurate, it is wrong. `p ≥ 10`
   and 1–2 power iterations is the usual default.
5. **Eigenvector sign/order.** `eigh` returns ascending eigenvalues; `svd` returns
   descending singular values. Mixing them silently reverses your components.

## Applications

- Ridge and GP: Cholesky of `XᵀX + λI` (or the kernel matrix), then triangular solves.
- PCA: SVD of the centred matrix (never `eigh` of `XᵀX` if you care about small
  singular values).
- Recommenders and LSA: truncated or randomised SVD.
- Fine-tuning: Woodbury for low-rank adapters and for K-FAC preconditioners.
- PageRank / spectral clustering: power iteration, which is just "multiply and
  renormalise until it stops moving".

## References

- Golub & Van Loan, *Matrix Computations*, 4th ed., 2013.
- Halko, Martinsson & Tropp, *Finding structure with randomness*, 2011. arXiv:0909.4061
- Trefethen & Bau, *Numerical Linear Algebra*, 1997.
