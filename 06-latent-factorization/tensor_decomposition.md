# Tensor decomposition

`tensor_decomposition.py` · Latent structure · Tier 3 · 5 variants

## Intuition

A matrix has rows and columns. A three-way array has rows, columns and *slices*,
and the slice direction is not like the other two: it can index trials, time,
subjects, conditions — anything that is a third thing rather than a fourth
coordinate. Flatten it and you assert that all three directions are
interchangeable, which is a claim about the data that is almost never true.

```
            slices (k)
        ┌───────────────┐
   i  ╱│  ░░░░░░░░░░░  │╲
     ╱ │  ░░░░░░░░░░░  │ ╲ j
    ┌──┴───────────────┴──┐
    │  ░░░░░░░░░░░░░░░░░  │      X ∈ R^{25×18×12}: 5400 numbers
    │  ░░░░░░░░░░░░░░░░░  │
    └─────────────────────┘
```

There are exactly two ways to say "this array is low rank", and they are not the
same statement.

**CP** says the array is a sum of $R$ rank-one pieces:

$$X = \sum_{r=1}^{R} a_r \circ b_r \circ c_r, \qquad X_{ijk} = \sum_r A_{ir}B_{jr}C_{kr}. \qquad (1)$$

One number, $R$, and no leftover. Each component is a triple of vectors, and the
component explains the same pattern along all three directions at once.

**Tucker** says the array lives in a small subspace along each direction:

$$X = G \times_1 U_1 \times_2 U_2 \times_3 U_3, \qquad X_{ijk} = \sum_{pqr} G_{pqr}U_{1,ip}U_{2,jq}U_{3,kr}. \qquad (2)$$

Three numbers — one rank per mode — and a core $G$ that is the entire model. A
Tucker decomposition with full ranks is not an approximation at all; it is a
change of basis.

The difference that matters is **identification**. A rank-$R$ CP decomposition of
a generic tensor is unique up to permuting the components and rescaling them
against each other (Kruskal's condition, §4), so the components are properties of
the data. Tucker is never unique: rotate inside any mode, compensate in the core,
and every number in the model changes while the fit changes by nothing (§7). CP
discovers; Tucker compresses. If you want to know what the components *are*, you
want CP, and if you want a compact basis you want Tucker.

The demo measures that distinction, and one more that is less comfortable. A
random $25\times18\times12$ rank-4 tensor is recovered from a random start in 32
sweeps at fit $1.0000000000$ with factors at cosine $1.000000$ in all three modes.
Then the demo moves one mode's second column to cosine $0.999$ of its first and
runs four starts: the fits read $0.9997021/0.9996775/0.9996774/0.9997918$ and the
factors disagree with the truth (cosines $0.897/0.982/0.970/0.940$) and with each
other. Every one of those runs reports a fit that looks like success. None of them
is the decomposition, which is representable at fit $1.0$ exactly.

## Variants

| Variant | Model | Update | Cost per sweep |
|:--|:--|:--|:--|
| `cp_als` | (1) | normal equations, $R\times R$ solve | $O(IJK\,R) + O(\sum_n I_n R^2)$ |
| `cp_als_ls` | (1) | least squares on the design matrix | $O(JK\,R^2)$ SVD per mode |
| `tucker_hooi` | (2) | alternate subspaces, then the core | $O(\sum_n I_n r_n \prod_{m\ne n}I_m r_m)$ |
| `hosvd` | (2) | truncate each mode once, then the core | $O(\sum_n I_n \prod_{m\ne n}I_m)$ |
| `parafac2` | ragged slices, shared `A` | Procrustes + three stacked solves | $O(\sum_k I J_k R^2)$ per sweep |

**`cp_als`** is alternating least squares on (1). Hold two factors and the third is
an ordinary linear least-squares problem, solved exactly — so the objective falls at
every half-step and the only thing that can go wrong is which stationary point you
land on. Each mode update solves an $R\times R$ system whose matrix is
$(B^{\mathsf T}B)*(C^{\mathsf T}C)$, the Khatri-Rao normal matrix; on the demo's
$25\times18\times12$ rank-4 tensor that is a $4\times4$ solve against a $216\times4$
design.

**`cp_als_ls`** is the same half-step without the normal equations: it forms the
$(JK)\times R$ Khatri-Rao design and calls `lstsq`, which uses an SVD and therefore
never squares the condition number. On the demo it reaches the same fit to
$10^{-16}$. Its reason to exist is the case where $(B^{\mathsf T}B)*(C^{\mathsf T}C)$
is nearly singular — near-collinear factor columns — where the normal equations
lose half the digits and the ridge in `cp_als` substitutes a bias for an answer.
The demo does not exhibit that failure at this size, and the doc says so rather than
manufacturing it: the two routes agree here, and the $216\times4$ SVD is the price
of the safer route.

**`tucker_hooi`** is HOOI, the higher-order orthogonal iteration: fit each mode's
subspace by the leading singular vectors of the array contracted with the other
modes' current subspaces, repeat. It is the workhorse because it is the only one of
the five with a monotone objective *and* an orthogonal basis.

**`hosvd`** is the one-pass version: truncate each mode independently, form the
core, stop. It is the tensor analogue of a truncated SVD and it is not optimal,
because the three truncations are chosen without reference to each other. The
demo's truncated problem shows the size of that: fitting rank $(2,3,2)$ to a noisy
rank-$(4,4,4)$ tensor, HOSVD stops at fit $0.275125$ and HOOI climbs to $0.340872$.

**`parafac2`** is the one that exists for a data-format reason. In chromatography
the second mode is a measured profile, and it drifts between runs, so the slices
have different widths *and* different loadings. PARAFAC2 keeps the loading
identified by allowing only a rotation between slices, so what is shared is the
inner-product structure rather than the loadings themselves. The demo's four
slices of widths $8/9/10/9$ come back with the shared factor and every slice's
loading at cosine $1.000000$, at fit $0.99999905$ — after 7658 sweeps, which is
the honest cost: ordinary CP on the zero-padded version of the same data, scored
on the true slices, stops at $0.5106$, so the model is what buys the difference and
the optimiser only decides how long it takes.

## Derivation

### 1. Unfolding, and the one convention that has to be right

Every algorithm here turns the array into matrices, and the index arithmetic of
that turn is where implementations quietly break. Unfolding mode $0$ of $X$ is

$$X_{(0)} \in \mathbb{R}^{I \times JK}, \qquad \big(X_{(0)}\big)_{i,\,jK+k} = X_{ijk}, \qquad (3)$$

so the columns are the other two modes in C order: the *first* remaining mode
varies slowest. The matching Khatri-Rao product of $B$ and $C$, both $R$-rowed, is

$$B \odot C \in \mathbb{R}^{JK\times R}, \qquad (B\odot C)_{jK+k,\,r} = B_{jr}C_{kr}. \qquad (4)$$

With (3) and (4) agreed, (1) becomes the matrix statement

$$X_{(0)} \approx A\,(C \odot B)^{\mathsf T}, \qquad (5)$$

where the factors appear in the *reverse* order of the remaining modes. Multiply
both sides by $(C\odot B)$ and the half-step is a least-squares problem:

$$A = X_{(0)}\,(C \odot B)\,\big[(C\odot B)^{\mathsf T}(C\odot B)\big]^{-1}. \qquad (6)$$

The inverse in (6) is not a $JK \times JK$ object:

$$(C\odot B)^{\mathsf T}(C\odot B) = (C^{\mathsf T}C) * (B^{\mathsf T}B), \qquad (7)$$

the elementwise (Hadamard) product of two $R\times R$ Gram matrices. That identity
is what makes CP-ALS cheap — $O(IJK\,R)$ to form the right-hand side, $O(R^3)$ to
solve — and it is also what makes it numerically fragile: (7) squares the condition
number of the factors, because each Gram already squares it once.

Get the pairing wrong — $X_{(0)}$ against $B \odot C$ instead of $C \odot B$ — and
the algorithm still converges, to a decomposition of the transposed-reshaped
tensor. The fit is plausible and the answer is wrong. That is the single most
common bug in from-scratch tensor code, and the reason `_unfold` and `_khatri_rao`
carry their index conventions in the docstring rather than a comment.

### 2. The design-matrix route

Equation (6) is the normal-equation solution. The same half-step solved directly is

$$A^{\mathsf T} = \arg\min_{Y}\; \big\lVert X_{(0)}^{\mathsf T} - (C\odot B)\,Y \big\rVert_F^2, \qquad (8)$$

a $(JK)\times R$ least-squares problem with $I$ right-hand sides. An SVD-based
solve of (8) has condition number $\kappa$ rather than the $\kappa^2$ of (7), and
costs $O(JK\,R^2 + I\,JK\,R)$ against $O(IJK\,R + R^3)$. At $R = 4$ and $JK = 216$
the difference is a $4\times4$ solve against a $216\times4$ SVD, and the demo
measures both reaching fit $1.0000000000$ — the same number to $10^{-16}$. The
route matters only when $\kappa$ is large enough that $\kappa^2$ exhausts double
precision, which in this file is where the ridge in `cp_als` fires.

### 3. Why the objective falls but the answer moves

Each half-step of (6) minimises the objective exactly over that block, so
$\lVert X - \hat X\rVert$ is non-increasing and bounded below: ALS converges. It
converges to a stationary point, and a stationary point of a non-convex problem is
not the answer. The demo's near-degenerate case is the measurement: the global
optimum has fit $1.0$ and is exactly attainable, and four starts sit at
$0.9997021$, $0.9996775$, $0.9996774$, $0.9997918$ respectively — four different
tensors-with-factors, none of them the truth, none of them moving under another
8000 sweeps.

Why near-collinearity does this is equation (7): the second mode's factor has
condition number $1679$ instead of $1.6$, so the normal matrix has condition
number $\sim 10^6$ before the second squaring, and the direction that
distinguishes component 1 from component 2 is the direction the solver resolves
worst. The fit is dominated by the well-conditioned directions and the components
are not.

### 4. Identifiability: Kruskal's condition

The reason CP is worth running at all is that its components are usually
determined by the data. The standard sufficient condition is in terms of the
*k-rank* of a factor — the largest $k$ such that every set of $k$ columns is
linearly independent — and it reads

$$k_A + k_B + k_C \;\ge\; 2R + 2 \quad\Longrightarrow\quad \text{the decomposition is unique up to permutation and scaling.} \qquad (9)$$

For generic factors (all $k$-ranks equal $R$) this holds for $R \ge 2$, so
"generically identifiable" is not an exaggeration. What (9) also says is how
identification fails: lower one mode's k-rank to 2 by making two columns nearly
dependent and the sum falls to $2 + R + R$, which for $R = 4$ is $10 < 10$ — on the
boundary, and the demo lands on the wrong side of it. The factors that come back
still satisfy the model to $10^{-4}$; there is simply a near-continuum of factor
triples that do.

### 5. Tucker: truncate, then project

For (2), the core is not a parameter to fit if the factors are orthonormal — it is
a projection:

$$G = X \times_1 U_1^{\mathsf T} \times_2 U_2^{\mathsf T} \times_3 U_3^{\mathsf T}, \qquad (10)$$

and substituting (10) into (2) shows $\hat X$ is the orthogonal projection of $X$
onto the tensor product of the three subspaces. So a Tucker fit is a choice of
three subspaces, and the fit is $\lVert P_{U_1\otimes U_2\otimes U_3}X\rVert^2$.

HOSVD chooses each $U_n$ as the leading $r_n$ left singular vectors of $X_{(n)}$,
independently, once. That is optimal for each mode in isolation and not optimal
jointly, because the best subspace for mode 1 depends on which subspaces modes 2
and 3 kept — the truncations interact through the core. HOOI fixes that by
iteration: with the other two subspaces fixed, the objective is quadratic in $U_n$
and its maximiser is the leading $r_n$ left singular vectors of

$$Y_n = X_{(n)}\,\big(U_3 \otimes U_2\big), \qquad (11)$$

with $\otimes$ the Kronecker product in the *same* order convention as (3). Note
(11) is not a Khatri-Rao product: the modes have different ranks, so the columns
cannot be paired. Using (4) here instead of the Kronecker product is a shape error
at best and, when the ranks happen to match, a silently wrong answer — which is
why the code says so in a comment.

HOOI starts from the HOSVD truncation and improves the projection norm at every
sweep, so it can never be worse. The demo's matched-rank case gets no benefit
(both at $0.929431$) because at the true rank the independent truncation is
already inside the noise; the truncated case gets $0.275125 \to 0.340872$.

### 6. What Tucker does not identify

Rotate inside a mode and the fit is unchanged:

$$U_n \mapsto U_n Q, \qquad G \mapsto G \times_n Q^{\mathsf T}, \qquad (12)$$

for any orthogonal $Q$ of the mode's rank. Every $Q$ gives a different set of
factors, a different core, and a bit-for-bit identical reconstruction. So a Tucker
factor has no interpretation and a Tucker core has no interpretation *except
relative to the factors that produced it*. The demo measures what is actually
identifiable — the subspace, via $\lVert U^{\mathsf T}\hat U\rVert_F/\sqrt{r}$ — and
gets $0.99990/0.99984/0.99996$ against the generating subspaces. That is the whole
content of a Tucker fit. Anyone who reports "the loadings" from a Tucker
decomposition is reporting one representative of a continuous family.

### 7. PARAFAC2: the model that fits ragged slices

The data are $K$ matrices $X_k \in \mathbb{R}^{I\times J_k}$ with different widths,
sharing the row mode. PARAFAC2 says

$$X_k \;\approx\; A\,\mathrm{diag}(d_k)\,(P_kB)^{\mathsf T},
\qquad P_k^{\mathsf T}P_k = I_J, \qquad (13)$$

with $A$ $(I\times R)$, $B$ $(J\times R)$, $J = \min_k J_k$, and $P_k$
$(J_k\times J)$ with orthonormal columns. Each slice has its own loading
$P_kB \in \mathbb{R}^{J_k\times R}$ — different shape, different values — and the
constraint is that all of them are rotations of one $B$. Two slices' loadings
therefore satisfy $(P_kB)^{\mathsf T}(P_hB) = B^{\mathsf T}P_k^{\mathsf T}P_hB$,
which is the inner-product structure the model claims is common.

The four blocks of the alternation:

**$P_k$** given $(A, B, d_k)$. Because $P_k^{\mathsf T}P_k = I$ makes the norm term
constant, minimising (13) over $P_k$ is maximising the cross term, and the maximiser
is an orthogonal Procrustes solution:

$$P_k = U V^{\mathsf T}, \qquad U\Sigma V^{\mathsf T} = \mathrm{svd}\big(X_k^{\mathsf T} A\,\mathrm{diag}(d_k)\,B^{\mathsf T}\big). \qquad (14)$$

**$A$** given the rest. Stacking the slices, $G_k = P_kB$ and

$$A = \Big[\sum_k X_kG_k\,\mathrm{diag}(d_k)\Big]\Big[\sum_k \mathrm{diag}(d_k)(B^{\mathsf T}B)\mathrm{diag}(d_k)\Big]^{-1}, \qquad (15)$$

where the second factor uses $G_k^{\mathsf T}G_k = B^{\mathsf T}B$ — the reason the
constraint (13) is on $P_k$ and not on the loadings.

**$B$** given the rest. Rotate each slice's transpose back through its own $P_k$ and
the same stacked problem appears:

$$B = \Big[\sum_k P_k^{\mathsf T}X_k^{\mathsf T}A\,\mathrm{diag}(d_k)\Big]\Big[\sum_k \mathrm{diag}(d_k)(A^{\mathsf T}A)\mathrm{diag}(d_k)\Big]^{-1}. \qquad (16)$$

**$d_k$** given the rest. This is where the obvious implementation is wrong.
Minimising (13) over $d_k$ looks like $R$ independent projections,

$$\tilde d_{k,r} = \frac{a_r^{\mathsf T}X_k g_{k,r}}{\lVert a_r\rVert^2\lVert g_{k,r}\rVert^2},
\qquad (17)$$

and that is only the minimiser when the $R$ components are orthogonal. They are
not: $a_r \circ g_{k,r}$ and $a_s \circ g_{k,s}$ overlap, so (17) is a fixed-point
step whose fixed point is not the least-squares solution. The correct update solves
the $R$-dimensional problem

$$d_k = \arg\min_{\delta}\; \big\lVert \mathrm{vec}(X_k) - W_k\delta\big\rVert^2,
\qquad (W_k)_{:,r} = \mathrm{vec}\big(a_r g_{k,r}^{\mathsf T}\big), \qquad (18)$$

which the code solves with `lstsq` on an $IJ_k\times R$ design $W_k$. Using (17)
instead gives a fixed-point iteration whose limit is not the least-squares
solution: the fit stalls at $0.9889$ with the shared factor at $0.997$, which is
close enough to look converged, and the residual is not large enough to notice.

## Code walkthrough

`TensorDecomposition(rank=3, *, variant="cp_als", n_iter=100, tol=1e-9, seed=0)`
with `fit(X)`, `reconstruct()`, `fit_score(X)`, the attributes `factors_`,
`core_`, `fits_` (fit after each sweep), and for `parafac2` also `weights_` and
`transforms_`. `rank` is an int or a per-mode tuple; `fit` takes a three-way array,
or a list of matrices for `parafac2`.

### 1. The unfolding

```python
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)
```

(3) in one line: bring the mode to the front and flatten the rest in C order.
`np.moveaxis` copies, so the reshape is contiguous and the column order is exactly
"first remaining mode slowest". The alternative — `X.reshape` on a transposed array
without `moveaxis` — is the bug in §1: it produces a valid matrix with the wrong
column order, and every subsequent step is consistent with itself.

### 2. The Khatri-Rao product

```python
    return (A[:, None, :] * B[None, :, :]).reshape(A.shape[0] * B.shape[0], -1)
```

(4). The broadcast makes an $(I,J,R)$ array whose entry $(i,j,r)$ is
$A_{ir}B_{jr}$, and the reshape folds $(i,j)$ into $i\cdot J + j$ — the same index
(3) gives the columns of the unfolding. The alternative `np.einsum("ir,jr->ijr")`
is clearer and roughly twice as slow for $R$ small.

### 3. The mode update, and the ridge that hides a singularity

```python
                if self.variant == "cp_als":
                    V = _kernels(factors, mode)
                    if np.linalg.cond(V) > 1e12:
                        # (BᵀB)*(CᵀC) factorises through two nearly dependent
                        # columns; the solve is then pure noise amplified
                        V = V + 1e-10 * np.trace(V) / R * np.eye(R)
                    factors[mode] = np.linalg.solve(V, _mttkrp(X, factors, mode).T).T
```

(6)–(7). `_mttkrp` is $X_{(n)}(C\odot B)$, the right-hand side, in one matmul. The
interesting line is the ridge. Without it, an exactly dependent pair of factor
columns makes $V$ singular and `np.linalg.solve` either raises or returns a vector
with components of size $10^{15}$ — and the next sweep then squares those, so the
run dies two iterations later with a `nan` far from the cause. With it, the solve
returns a *biased* answer in those directions, which is honest in the sense that it
is finite and the fit is still meaningful, and dishonest in the sense that the bias
is invisible. `np.linalg.cond` on every mode of every sweep is $O(R^3)$ against the
$O(IJK\,R)$ already spent, so the check is free.

### 4. The design-matrix route

```python
                else:
                    KR = _khatri_rao(*[f for m, f in enumerate(factors) if m != mode])
                    factors[mode] = np.linalg.lstsq(
                        KR, _unfold(X, mode).T, rcond=None)[0].T
```

(8). The transpose is where this is easy to get wrong: `lstsq(A, b)` solves
$A x = b$, so the design is `KR` $(JK\times R)$ and the right-hand side is
$X_{(n)}^{\mathsf T}$ $(JK \times I_n)$ — one solve with $I_n$ columns, not $I_n$
solves. `rcond=None` lets LAPACK pick its own cutoff, which for a well-scaled design
is the right choice; passing $10^{-12}$ here would discard genuine small singular
values on a nearly degenerate problem, which is precisely the case the variant
exists for.

### 5. Normalising the factors, and what the scale means

```python
            for mode in range(2):
                nrm = np.linalg.norm(factors[mode], axis=0)
                nrm[nrm < EPS] = 1.0
                factors[mode] /= nrm
                factors[mode + 1] *= nrm
```

(1) is invariant under $a_r \mapsto s_ra_r$, $b_r \mapsto t_rb_r$,
$c_r \mapsto a_rc_r/(s_rt_r)$, so the factors are only defined up to a
per-component scale, and leaving it free lets two components drift to
$(10^{6}, 10^{-6})$ and lose all precision. Pushing each mode's norm into the next
mode fixes one representative and makes the factors comparable across runs and
across seeds. The `nrm < EPS` guard is the all-zero component, which has no scale
to normalise and would otherwise divide by zero; setting it to 1 leaves it at zero
instead of turning it into `nan`.

### 6. Kronecker, not Khatri-Rao

```python
            for mode in range(3):
                others = [f for m, f in enumerate(factors) if m != mode]
                # Kronecker, not Khatri-Rao: the modes have different ranks
                Y = _unfold(X, mode) @ np.kron(others[0], others[1])
                factors[mode] = np.linalg.svd(Y, full_matrices=False)[0][:, :ranks[mode]]
```

(11). A Khatri-Rao product needs both operands to have the same number of columns,
which for equal Tucker ranks is exactly what happens — so the wrong call does not
raise at the demo's $(2,3,2)$ and $(3,4,4)$, and the run continues with a design
matrix that mixes two modes' coordinates. This is the shape-flexible version of the
§1 bug and it is only caught by checking a fit against a known core.

### 7. HOSVD as the zero-pass loop

```python
        fits: list[float] = []
        for _ in range(0 if self.variant == "hosvd" else self.n_iter):
```

HOSVD is the initialisation and the core projection, and nothing else: the
independent truncations happen before the loop, so `hosvd` runs it zero times and
falls through to the core computation below. Writing the variant as
`range(1 if hosvd else n_iter)` — one alternating pass — silently turns HOSVD into
one step of HOOI, which is a different (better) estimator with the same name. The
demo's HOSVD number would then come out at $0.316932$ instead of $0.275125$: a
$15\%$ difference in fit, invisible without a reference implementation.

### 8. Procrustes

```python
        U, _, Vt = np.linalg.svd(x.T @ (A * d) @ B.T, full_matrices=False)
        return U @ Vt
```

(14). `A * d` is $\mathrm{diag}(d)A$ with the scaling broadcast down the columns,
so the matrix being decomposed is $X_k^{\mathsf T}A\mathrm{diag}(d_k)B^{\mathsf T}$
of shape $(J_k, J)$; `full_matrices=False` gives $U$ $(J_k,\min)$ and
$V^{\mathsf T}$ $(\min,J)$, and the product has orthonormal *columns* — which is
exactly the constraint (13) needs. It does not have orthonormal rows unless
$J_k = J$, which is why the model takes $J = \min_kJ_k$: a slice narrower than $J$
cannot even hold a $J$-dimensional rotation.

### 9. The scale update, in the right coordinates

```python
            Bk = [p @ B for p in P]
            D = [np.linalg.lstsq(np.einsum("ir,jr->ijr", A, bk).reshape(-1, R),
                                 x.reshape(-1), rcond=None)[0]
                 for x, bk in zip(X, Bk, strict=True)]
```

(18). `np.einsum("ir,jr->ijr", A, bk)` builds $W_k$ $(I, J_k, R)$ whose column $r$
is $a_rg_{k,r}^{\mathsf T}$, and the two `reshape(-1)` calls vectorise both sides
in the same order. The version this replaced computed (17) — a dot product per
component divided by that component's squared norm — which is the coordinate-wise
minimiser only under orthogonality. It converges, and it converges to the wrong
tensor: the demo's fit sat at $0.9889$ with the shared factor at $0.997$ instead of
$1.000$.

### 10. Measuring a decomposition that has no canonical form

```python
    out = []
    for f, t in zip(factors, truth, strict=True):
        fn = f / np.maximum(np.linalg.norm(f, axis=0), EPS)
        tn = t / np.maximum(np.linalg.norm(t, axis=0), EPS)
        out.append(float(np.mean(np.max(np.abs(fn.T @ tn), axis=1))))
    return out
```

For CP this is the strongest available statement: normalise both sides, take the
absolute cosine against the best-matching true component, and average over the
fitted components. Absolute value because of the sign freedom, max because of the
permutation freedom, mean rather than max so that one good component out of four
cannot carry the score. `_subspace_score` is the same idea for Tucker, where
$\lVert U^{\mathsf T}\hat U\rVert_F/\sqrt{r} = 1$ means the same subspace and no
per-column comparison is meaningful.

## Complexity

$I, J, K$ the mode sizes, $R$ the CP rank, $r_n$ the Tucker ranks, $K$ the number
of PARAFAC2 slices, $T$ sweeps.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_unfold` | $O(\prod I_n)$ | same | once per mode per sweep |
| `_khatri_rao` | $O(JK\,R)$ | $O(JK\,R)$ | the design matrix |
| `_mttkrp` | $O(IJK\,R)$ | $O(IR)$ | the right-hand side |
| one `cp_als` mode | $O(IJK\,R + R^3)$ | $O(JKR)$ | the dominant term is the matmul |
| one `cp_als_ls` mode | $O(JK\,R^2 + I\,JK\,R)$ | $O(JKR)$ | an SVD, not a solve |
| one `tucker_hooi` mode | $O(I_n r_n \prod_{m\ne n} I_m r_m)$ | $O(\prod_n I_nr_n)$ | one SVD per mode |
| `hosvd` | $O(\sum_n I_n\prod_{m\ne n}I_m)$ | $O(\prod I_n)$ | three SVDs, once |
| one `parafac2` sweep | $O(\sum_k I J_k R^2)$ | $O(\sum_k I J_k)$ | the slices are the memory |
| `_factor_recovery` | $O(\sum_n I_nR\,R_{\rm true})$ | — | once, at the end |
| full demo | — | — | four fits, 1.0 s |

Two rows decide what you can run. The first is `_mttkrp`: CP-ALS is
$O(IJK\,R)$ per mode, so a $1000^3$ tensor at $R = 10$ is $10^{10}$ multiply-adds
per sweep — which is why nobody runs CP on a dense $1000^3$ array and why the
sparse and sampled variants exist. The second is memory: CP never materialises
anything bigger than an unfolding, but `_unfold` *does* copy, so a mode of size
$10^6$ with a $10^3 \times 10^3$ remainder allocates $10^{12}$ floats if you call it
on the wrong mode. The right move is `_mttkrp` without the copy, which is a
different function.

`parafac2`'s cost is not in the arithmetic but in the iteration count. The demo
takes 7658 sweeps to reach its $10^{-9}$ tolerance, and after 2000 sweeps it is
still at $0.999657$ — a fit that reads as finished at three decimals and is not.
The Procrustes step and the stacked solves converge linearly, and the rate shrinks
as the rotations grow. That is two orders of magnitude more sweeps than CP for the
same problem size, and it is the variant's real price.

## Failure modes

**1. The fit is not the answer.** Four starts on the near-degenerate tensor:
$0.9997021/0.9996775/0.9996774/0.9997918$, four different factor triples, none of
them the truth, and a global optimum at $1.0$ that ALS does not find. Any pipeline
that reports a CP fit as evidence of a successful decomposition is reporting the
model's flexibility. The diagnostic is a second start: run two seeds and compare
the factor cosines, not the fits.

**2. Non-convergence looks like convergence.** The same runs hit the sweep cap and
report a fit that is flat to $10^{-7}$ per sweep. A stopping rule based on the fit
change fires; a stopping rule based on the factor change fires; a stopping rule
based on the objective is the one everybody writes. Compare factors across starts
before believing either.

**3. Tucker factors are subspaces, so "the loading" is a fiction.** Rotating inside
a mode and compensating in the core leaves the reconstruction identical to the last
bit. Reporting loadings from a Tucker fit is reporting an arbitrary representative
of a continuous equivalence class. The interpretable objects are the subspaces and
the fit — and if the components themselves matter, the model you want is CP.

**4. CP rank is not a matrix rank.** A tensor's CP rank can exceed $\max(I,J,K)$:
the maximal rank of a $2\times2\times2$ tensor is 3, and the generic rank of a
$3\times3\times3$ is 5. Choosing $R$ by analogy with SVD — "at most
$\min(I,J,K)$" — under-fits, and choosing it by fitting until the residual is zero
over-fits, because the last components absorb noise at whatever cost in
conditioning they like.

**5. Padding ragged slices is a modelling error, not a convenience.** The demo's
zero-padded CP reaches $0.5106$ on the same data PARAFAC2 fits at $0.99999905$, and
the gap is not the optimiser: CP's model has one loading per column, and the four
slices have four different rotations of it. Padding is the default because it is
the only thing you can do with an array, and it destroys exactly the structure the
data had.

**6. PARAFAC2's Procrustes step needs the rotation to be the only difference.** If
slices differ in scale, baseline, or width *and* in loading shape — which real
chromatograms do — the model absorbs the misfit into the rotations and the shared
factor, and the fit stays high while the interpretation drifts. The constraint
$P_k^{\mathsf T}P_k = I$ is a claim about the measurement, not a numerical
convenience.

**7. The unfolding and the Khatri-Rao product must be agreed.** §1. Getting the
factor order or the column order wrong gives a decomposition of a reshuffled
tensor, with a plausible fit and components that look almost right. Check the
implementation on a synthetic rank-$R$ tensor with known factors before pointing it
at data: the demo's first section does exactly that, and it is the only cheap test
that catches the family of bugs.

**8. A rank larger than the data supports adds dead components, not capacity.** The
extra components must either cancel in pairs — which costs conditioning, as in §3 —
or collapse to near-zero norm, in which case the normalisation has nothing to
scale and the fit stops improving. Watch the per-component norms, not the total.

## Applications

- **Chemometrics.** PARAFAC is the standard tool for fluorescence excitation-emission
  matrices and for chromatographic runs with drift; PARAFAC2 exists because the
  second mode is a measurement that moves. This is the application the whole family
  was built for.
- **Neuroscience.** A trial × neuron × time tensor factors into cell assemblies,
  temporal patterns and trial weights; CP is used because the components are the
  claim, and Tucker when the question is which subspace of neurons is engaged.
- **Recommender systems and relational learning.** User × item × context, or
  subject × relation × object: CP on the latter is RESCAL-style latent feature
  learning, and the identifiability of (9) is what makes the recovered relations
  interpretable.
- **Signal processing.** Blind source separation with a third dimension (time,
  frequency, array element) becomes CP, and the identifiability condition is
  stronger than ICA's — which is the reason tensor methods are used where matrix
  factorisation would be ambiguous.
- **Compression of convolutional layers.** Tucker decomposition of a 4-D weight
  tensor is the classic network-compression baseline; the core is the compressed
  representation and the factors are the bases.
- **Topic modelling with a time axis.** Document × word × time, or author ×
  document × word: CP gives topics with a temporal signature, and the third mode is
  what separates it from `lda_topic` in this chapter.

Where it is a trap: reading CP components as "the" latent structure without a second
start; reporting Tucker loadings; choosing the rank by the fit; padding ragged data
to make it rectangular; and running dense ALS on a tensor whose mode products
exceed memory — which fails as an allocation, a long way from the line that chose
the algorithm.

## References

- **Hitchcock (1927),** *The expression of a tensor or a polyadic as a sum of
  products* — CP, before it had a name, and the first statement that the
  decomposition might be unique.
- **Carroll & Chang (1970),** *Analysis of individual differences in
  multidimensional scaling via an N-way generalization of "Eckart-Young"
  decomposition* — CANDECOMP, and the alternating least-squares algorithm that is
  still the default.
- **Harshman (1970),** *Foundations of the PARAFAC procedure* — PARAFAC, the same
  decomposition from psychometrics, and the source of the "components are the
  answer" reading.
- **Kruskal (1977),** *Three-way arrays: rank and uniqueness of trilinear
  decompositions* — the k-rank condition (9), which is why CP is used for discovery.
- **De Lathauwer, De Moor & Vandewalle (2000),** *On the best rank-1 and
  rank-$(R_1,R_2,\ldots,R_N)$ approximation of higher-order tensors* — HOSVD, and
  the proof that the truncated version is not optimal, which is what HOOI fixes.
- **De Lathauwer, De Moor & Vandewalle (2000),** *A multilinear singular value
  decomposition* — the Tucker core as a projection, equation (10), including the
  all-orthogonality that makes the core interpretable only relative to its factors.
- **Kiers, ten Berge & Bro (1999),** *PARAFAC2: a new, iterative algorithm for
  three-way analysis with one varying mode* — the model of (13) and the Procrustes
  step of (14); the reference the ragged-slice variant follows.
- **Bro, Andersson & Kiers (1999),** *PARAFAC2, part II: modeling chromatographic
  data with retention time shifts* — the application that made PARAFAC2 standard,
  and the honest account of what the rotation constraint assumes.
- **Tomasi & Bro (2006),** *A comparison of algorithms for fitting the PARAFAC
  model* — the survey of CP-ALS variants, including the line-search and
  compression tricks that make the slow convergence of §7 survivable.
- **Kolda & Bader (2009),** *Tensor decompositions and applications* — the review
  that every implementation is measured against, including the unfolding and
  Khatri-Rao conventions of §1 and the identifiability discussion of §4.
- **Comon, Golub, Lim & Mourrain (2008),** *Symmetric tensors and symmetric tensor
  rank* — the rank-exceeds-mode-size fact of failure mode 4, with the small
  counterexamples.
- **Anandkumar, Ge, Hsu, Kakade & Telgarsky (2014),** *Tensor decompositions for
  learning latent variable models* — the method-of-moments view, where CP's
  uniqueness is what turns a latent-variable model into an estimable one.
