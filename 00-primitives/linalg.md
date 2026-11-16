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
