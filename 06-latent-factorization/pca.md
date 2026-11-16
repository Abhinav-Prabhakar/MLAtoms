# PCA

`pca.py` · Latent structure · Tier 1 · 7 variants

## Intuition

PCA answers one question: which direction, if you had to throw everything else
away, loses the least? Centred data $X_c \in \mathbb{R}^{n \times d}$, a unit
vector $v$, and the variance of the projection $X_c v$ is $v^\top \Sigma v$
with $\Sigma = X_c^\top X_c/(n-1)$. Maximise it, then repeat in the orthogonal
complement:

$$\max_{V^\top V = I_k}\ \mathrm{tr}(V^\top \Sigma V), \qquad (1)$$

whose solution is the top $k$ eigenvectors of $\Sigma$ — the right singular
vectors of $X_c$. That is the whole algorithm. Everything else in this file is
a different way to compute the same $V$, or a deliberate change to the
objective.

The reason it is worth seven variants is that "the same $V$" can be very
different in cost and in what it can express:

```
  full_svd          X_c ──SVD──► U S Vᵀ        O(nd²), exact
  randomized        X_c ──sketch──► Q ──► SVD   O(ndk), exact whp
  kernel_pca        Φ(X) ──Gram──► K_c ──► eig  O(n²d), nonlinear
  incremental       [X; B] ──update──► (V, S)   O(bd²) per block
  robust_pca        X = L + S, minimise a mix   O(nd²) per iteration
  sparse_pca        add λ‖v‖₁, lose orthogonality
  whitening         V, then divide by √λ
```

The first four compute the same subspace and the demo asserts that they agree
to $10^{-10}$. The last three change the answer on purpose, and the demo
asserts what each change buys: robust PCA survives corruption that moves plain
PCA by 26%, sparse PCA puts 69% of its loadings at exactly zero while keeping
99% of the variance, and whitening makes the covariance exactly $I$.

**Eckart–Young** is what makes truncation meaningful. Among all rank-$k$
matrices, $X_c V_k V_k^\top$ is the closest to $X_c$ in Frobenius norm, and the
error is the sum of the eigenvalues you discarded:

$$\min_{\mathrm{rank}(M) \le k} \|X_c - M\|_F^2 = \sum_{j>k} \lambda_j, \qquad (2)$$

with $\lambda_j$ the eigenvalues of $\Sigma$. This is not an approximation
statement — it is an equality, and the demo checks it to $10^{-8}$. It is also
the only honest way to choose $k$ without a validation set: look at the tail of
the spectrum, not at a round number.

## Variants

| Variant | Computes | Cost | Exact? | When |
|:--|:--|:--|:--|:--|
| `full_svd` | all $V$, truncate | $O(nd\min(n,d))$ | yes | the default; $n, d \lesssim 5000$ |
| `randomized` | top $k$ from a sketch | $O(ndk)$ | whp, with power iterations | $d$ large, $k \ll d$ |
| `kernel_pca` | top $k$ of $\Phi(X)$ | $O(n^2d + n^3)$ | yes, in feature space | the manifold is nonlinear |
| `incremental` | $V$ updated per block | $O(bd^2)$ per block | yes, if the mean is fixed | data arrives in batches |
| `robust_pca` | $L$ in $X = L + S$ | $O(nd^2)$ per iteration | convex, not exact | corrupted cells, outliers |
| `sparse_pca` | $V$ with zeros | $O(Tnd)$ | no, non-convex | interpretable loadings |
| `whitening` | $V \Lambda^{-1/2}$ | same as `full_svd` | yes | downstream models want $I$ covariance |

**`full_svd`** is the definition: centre, call `np.linalg.svd`, keep the top
rows of $V^\top$. `np.linalg.svd` on an $n \times d$ matrix costs
$O(nd\min(n,d))$ and is the right choice until the matrix does not fit in
memory. The sign of each component is arbitrary; `_flip` fixes it so that the
largest-magnitude loading is positive, which is what makes two runs comparable.

**`randomized`** exploits the fact that you do not need the whole spectrum,
only the top $k$. Draw a Gaussian sketch $\Omega \in \mathbb{R}^{d \times (k+p)}$,
form $Y = (X_c X_c^\top)^q X_c \Omega$, orthonormalise $Y$ to get $Q$, and SVD
the small matrix $Q^\top X_c$. With $q = 0$ the sketch sees the spectrum
weighted by the Gaussian, and the error is governed by
$\sigma_{k+1}/\sigma_1$; each power iteration replaces that ratio by its
$(2q+1)$-th power. On a rank-3 signal buried in full-rank noise, with a sketch
of width 8, $q=0$ leaves 8% error in the singular values, $q=2$ leaves
$10^{-7}$, and $q=4$ leaves $4\times10^{-13}$. The cost is $O(ndk)$ plus $q$
passes over the data, which is the whole point when $d = 10^6$ and $k = 50$.

**`kernel_pca`** replaces $x$ with $\Phi(x)$ and never forms $\Phi$. The
covariance in feature space is an $n \times n$ Gram matrix, and its top
eigenvectors give the components through the kernel trick:

$$K_{ij} = k(x_i, x_j) = \exp(-\gamma\|x_i - x_j\|^2), \qquad
K_c = HKH, \qquad H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top. \qquad (3)$$

The centring is not optional: without $H$, the first eigenvector of $K$ is
approximately the all-ones vector, which measures nothing but the mean of the
feature map. The demo's two moons are the standard illustration — a linear
rotation cannot separate them, so linear PCA cannot either, but one threshold
on the first kernel component reaches a Fisher ratio of 9.9 against linear
PCA's 1.55.

**`incremental`** keeps $(V, S)$ and folds in a block $B$ of new rows without
touching the old ones. The identity is

$$[X; B] = \begin{bmatrix} U & 0 \\ 0 & I \end{bmatrix}
\begin{bmatrix} S & 0 \\ P & R_q^\top \end{bmatrix}
\begin{bmatrix} V & 0 \\ 0 & Q \end{bmatrix}^\top, \qquad (4)$$

with $P = BV$, $R = B - PV^\top$, and $R^\top = QR_q$. The middle matrix is
$(k+b) \times (k+l)$, so the SVD that matters is small; the update is exact up
to the truncation you choose. The demo recovers the full SVD of 400 rows from
eight 50-row blocks to $1.6\times10^{-15}$ — provided the mean is fixed. Left to
itself the variant freezes the first block's mean, and that alone moves the
subspace by $3\times10^{-4}$ rad, which the demo also asserts.

**`robust_pca`** changes the objective because squared error is not robust: one
cell of magnitude 10 will rotate a component to accommodate it. The
decomposition $X = L + S$ with $L$ low-rank and $S$ sparse is convex:

$$\min_{L, S}\ \|L\|_* + \lambda\|S\|_1 \quad \text{subject to}\quad X = L + S,
\qquad \lambda = \frac{1}{\sqrt{\max(n,d)}}. \qquad (5)$$

The nuclear norm is the convex envelope of rank, the L1 norm is the convex
envelope of sparsity, and the constraint forces the two to add up to the data.
The demo corrupts 2% of the cells with $\pm10$, and the decomposition recovers
$L$ to $1.1\times10^{-2}$ where rank-5 PCA misses by $0.26$ — an order of
magnitude. That is the variant to reach for when "outlier" means a bad sensor
reading rather than a heavy tail.

**`sparse_pca`** relaxes orthogonality to get zeros. The objective becomes

$$\max_{v}\ v^\top \Sigma v - \lambda\|v\|_1 \quad \text{subject to}\quad \|v\|_2 = 1,
\qquad (6)$$

which is non-convex but has a simple algorithm: power iteration where each step
soft-thresholds the gradient. Components are extracted one at a time with
deflation. The price is that they are no longer orthogonal and the explained
variance is no longer additive across components; the demo's "99% of dense
variance" is a statement about the sum of the deflated singular values, not
about a partition.

**`whitening`** is a post-processing, not a new estimator: after the projection,
divide each score by its standard deviation. The transformed covariance is the
identity, the transform is invertible on the $k$-subspace, and any downstream
model that assumes isotropic features (k-means, Gaussian mixtures, linear
regression with a spherical prior) now gets it. The cost is noise amplification:
the smallest eigenvalue's square root is in the denominator.

## Derivation

### The eigenproblem, and why SVD is the same thing

Lagrange multipliers on (1) with the constraint $V^\top V = I$:

$$\Sigma V = V\Lambda, \qquad V^\top \Sigma V = \Lambda. \qquad (7)$$

So the columns of $V$ are eigenvectors and the variances are the eigenvalues.
For centred data, $\Sigma = X_c^\top X_c/(n-1)$ and the SVD $X_c = USV^\top$
gives

$$X_c^\top X_c = VS^2V^\top, \qquad \lambda_j = \frac{s_j^2}{n-1}. \qquad (8)$$

The code computes $V$ from the SVD, never forms $\Sigma$, and never
eigendecomposes a $d \times d$ matrix. That matters: forming $\Sigma$ squares
the condition number, and for $d > n$ it is a $d \times d$ matrix of rank at
most $n$.

The scores are $Z = X_c V_k = U_k S_k$, and reconstruction is
$\hat X = ZV_k^\top + \mu$. The residual $X_c - ZV_k^\top$ has squared norm
$\sum_{j>k} s_j^2$, which divided by $n-1$ is exactly the discarded variance in
(2). The demo checks the two sides separately: the residual against the
discarded eigenvalues, and orthogonality of the residual to $V$, which is what
makes it a projection rather than merely a low-rank approximation.

### Randomized range finding

Write the SVD as $X_c = U\Sigma V^\top$ and sketch with a Gaussian
$\Omega \in \mathbb{R}^{d \times \ell}$, $\ell = k + p$. The claim is that
$\mathrm{range}(X_c\Omega)$ captures the top $\ell$ left singular vectors with
high probability, because a random vector has a nonzero component along every
direction. The quality is governed by how much of the spectrum is left outside:

$$\|X_c - QQ^\top X_c\|_2 \lesssim \sigma_{k+1} + \text{(a term } \propto \sigma_1 \sqrt{k+p}\,/\,\text{gap}\text{)}. \qquad (9)$$

Power iterations change the underlying matrix from $X_c$ to
$(X_cX_c^\top)^qX_c$, whose singular values are $s_j^{2q+1}$: the same ordering,
with the top amplified relative to the rest. Two iterations are usually enough
because they square the ratio three times. The demo's assertion is deliberately
sharp — $4\times10^{-16}$ in the singular values, zero angle in the subspace on
a rank-3 signal — and it fails at $q=0$ as soon as the data has a noise tail
(8% error on the same construction), which is the point of the variant.

### Kernel PCA, in four lines of algebra

In feature space the covariance is $\frac{1}{n}\sum_i \phi(x_i)\phi(x_i)^\top$
after centring, and its eigenvector $w$ lies in the span of the data:
$w = \sum_i \alpha_i \phi(x_i)$. Substituting into the eigenproblem and
multiplying by $\phi(x_j)^\top$ turns it into

$$K_c\alpha = n\lambda\alpha, \qquad (10)$$

an $n \times n$ eigenproblem whose matrix is the centred Gram matrix. The
projection of a new point is the kernel between it and the training set, times
$\alpha$:

$$z(x) = \sum_i \alpha_i k(x, x_i) = K_c(x, X)\,\alpha. \qquad (11)$$

The code stores $\alpha$ (scaled by $\sqrt{\lambda}$) as `components_`, which is
why the shape is $(k, n_{\text{fit}})$ rather than $(k, d)$: there is no vector
in input space to return. Centring a *test* block needs the training means, not
its own:

$$K_c^{\text{test}} = K^{\text{test}} - \mathbf{1}\bar K_{\text{row}}^\top
- \bar K_{\text{col}}\mathbf{1}^\top + \bar K_{\text{grand}}, \qquad (12)$$

with the three means computed on the training Gram matrix. Centring the test
block by its own mean is the classic kernel-PCA bug: the scores of training and
test points land in different coordinate systems and nothing downstream works.

### Incremental SVD

Equation (4) is a change of basis, not an approximation. Starting from
$X = USV^\top$, the new rows are split into the part the old basis already spans
($P = BV$) and the residual $R$. Factor the residual as $R = R_q^\top Q^\top$
and the augmented matrix is a product of three matrices, the middle one small.
Truncating the small SVD to $k$ is the only approximation, and it is the same
truncation a full refactorisation would make. The demo's exactness claim is
therefore conditional on keeping the full rank of the first block: throw away
directions on block one and no later block can bring them back.

### Robust PCA: the IALM updates

The augmented Lagrangian for (5) is

$$\mathcal{L} = \|L\|_* + \lambda\|S\|_1 + \langle Y, X - L - S\rangle
+ \frac{\mu}{2}\|X - L - S\|_F^2. \qquad (13)$$

Minimising over $L$ and $S$ in turn gives two proximal operators, each with a
closed form:

$$L \leftarrow \mathrm{SVT}_{1/\mu}\!\left(X - S + Y/\mu\right), \qquad
S \leftarrow \mathrm{soft}_{\lambda/\mu}\!\left(X - L + Y/\mu\right), \qquad (14)$$

and the multiplier update $Y \leftarrow Y + \mu(X - L - S)$ with $\mu$ increased
geometrically. $\mathrm{SVT}$ is singular-value soft thresholding: SVD, subtract
$\tau$ from each singular value, clip at zero, reassemble. The threshold is what
sets the rank, and the L1 threshold is what sets the sparsity, so the two
matrices negotiate over which part of each cell is signal.

The choice $\lambda = 1/\sqrt{\max(n,d)}$ comes from the recovery guarantee: it
balances the nuclear-norm penalty (which grows like $\sqrt{n}$) against the L1
penalty (which grows like $\sqrt{\text{number of corrupted cells}}$). It is not
a tuning knob, and changing it trades rank against sparsity in a way that is
usually wrong.

### Sparse PCA's iteration

For fixed $\|v\|=1$, the objective (6) is the Lagrangian of a lasso problem in
$v$, and the standard power-method-with-thresholding iterates

$$v \leftarrow \frac{\mathrm{soft}(X_c^\top X_c v,\ \lambda\,\|X_c^\top X_c v\|_\infty)}
{\|\cdot\|} \qquad (15)$$

until the direction stops moving. The threshold is relative to the largest
entry of the gradient, so `lam=0.1` means "zero anything below 10% of the
strongest loading", which is scale-free. After convergence the component is
deflated, $X_c \leftarrow X_c - (X_c v)v^\top$, so the next one explains what is
left. Deflation is what keeps the components from collapsing onto the same
direction once orthogonality is gone.

## Code walkthrough

`PCA(n_components, variant=..., seed=..., gamma=..., lam=..., ...)` is the
whole API: `fit`, `transform`, `inverse_transform`, and `partial_fit` for the
streaming variant. The variants share the bookkeeping — mean, explained
variance, ratios — and differ only in how `components_` is produced.

### 1. Centre, SVD, truncate

```python
def _reduce(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, SVD, truncate; return `(mean, components, singular_values)`."""
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, s.size)
    return mean, _flip(Vt[:k]), s[:k]
```

Three details are load-bearing. `full_matrices=False` is the economy SVD: with
`True`, $U$ is $n \times n$ and a $10^6$-row matrix allocates $10^{12}$ floats.
`k = min(k, s.size)` is what makes `n_components > min(n, d)` legal rather than
a broadcast error. And `_flip` is determinism: LAPACK is free to return either
sign for a singular vector, so without it two runs can disagree on every entry
of a component while representing the same subspace.

```python
def _flip(V: np.ndarray) -> np.ndarray:
    """Sign-flip rows so the largest-magnitude entry of each is positive."""
    idx = np.argmax(np.abs(V), axis=1)
    signs = np.sign(V[np.arange(V.shape[0]), idx])
    signs[signs == 0.0] = 1.0
    return V * signs[:, None]
```

The `signs == 0` guard is not paranoia: a component whose largest entry is
exactly zero gets sign 0 and the whole row is annihilated. That happens on
one-hot data, which is where PCA is used most.

### 2. The incremental update is a small SVD

```python
            V, S = self._V, self._S                 # d×r, r
            kc = V.shape[1]
            P = Xc @ V                              # b×r: what the old basis sees
            R = Xc - P @ V.T                        # b×d: what it does not
            Q, Rq = np.linalg.qr(R.T)               # Q: d×l, Rq: l×b
            lq = Q.shape[1]
            M = np.zeros((kc + Xc.shape[0], kc + lq))
            M[:kc, :kc] = np.diag(S)
            M[kc:, :kc] = P
            M[kc:, kc:] = Rq.T
            _, S2, V2t = np.linalg.svd(M, full_matrices=False)
```

This is (4) written out. The subtle line is `M[kc:, kc:] = Rq.T`: `Rq` comes
from `qr(R.T)`, so it is $l \times b$ and its transpose is the $b \times l$
block the identity needs. Writing `Rq` instead transposes the residual and the
singular values drift by percent-level amounts — exactly the bug this demo's
$1.6\times10^{-15}$ assertion exists to catch.

Keeping `self._V` at full rank rather than at $k$ is the other non-obvious
choice, and the comment says why: truncating on the first block discards
directions that later blocks may need, and no update can recover them. The
public `components_` is truncated to $k$ afterwards.

### 3. The sketch, and two passes that matter

```python
        width = min(k + self.oversample, d, n)
        Q, _ = np.linalg.qr(Xc @ rng.standard_normal((d, width)))
        for _ in range(self.power_iter):            # sharpen the spectrum
            Q, _ = np.linalg.qr(Xc.T @ Q)
            Q, _ = np.linalg.qr(Xc @ Q)
        B = Q.T @ Xc
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
```

The loop alternates $X_c^\top$ and $X_c$, which is $(X_cX_c^\top)^qX_c$ applied
to $\Omega$; each trip re-orthonormalises, which is what keeps the sketch from
overflowing on large data. `B = Q.T @ Xc` is $O(nd\ell)$, and the SVD of $B$ is
$O(n\ell^2)$ — with $\ell \approx k$, that is the $O(ndk)$ claim.

One line that is easy to get wrong: the sketch matrix is `(d, width)`, so
`Xc @ Omega` is $n \times \text{width}$. Transposing the sketch gives an
$n \times n$ matrix and a different algorithm.

### 4. Kernel PCA: centre the Gram matrix, not the data

```python
        K = self._rbf(X, X)
        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H @ K @ H
        w, A = np.linalg.eigh(Kc)
        order = np.argsort(w)[::-1][: self.n_components]
        w, A = np.maximum(w[order], 0.0), A[:, order]
```

`H @ K @ H` is $O(n^3)$ if written as three matrix products, and `eigh` is
$O(n^3)$ anyway, so for $n$ above a few thousand the whole variant is the
bottleneck. `np.linalg.eigh` rather than `eig` because $K_c$ is symmetric: it
returns real eigenvalues in ascending order, and the reversal is the sort. The
`np.maximum(w, 0.0)` clip is for the eigenvalues that come out as
$-10^{-15}$ from roundoff and would otherwise poison `sqrt`.

```python
    def _rbf(self, A, B) -> np.ndarray:
        d2 = (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
              - 2.0 * A @ B.T)
        return np.exp(-self.gamma * np.maximum(d2, 0.0))
```

The expansion $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a^\top b$ turns an
$n \times m \times d$ tensor into one matrix product. The `maximum(d2, 0.0)` is
not cosmetic: for nearby points the three terms cancel and `d2` can come out as
$-10^{-14}$, and `exp(+10^{-14})` is slightly greater than one, which breaks the
positive-semidefiniteness of $K$.

```python
        if self.variant == "kernel_pca":
            K = self._rbf(X, self.X_fit_)
            Kc = (K - K.mean(axis=1, keepdims=True) - self._col_mean_
                  + self._grand_mean_)
            return Kc @ self.components_.T
```

This is (12): `K.mean(axis=1)` centres each *test* row by its own kernel mean,
`self._col_mean_` subtracts the training mean of each column, and
`self._grand_mean_` adds it back once. Getting the signs right is the difference
between a transform and a projection; the demo checks the training set against
`eigh` because that is the only case where both sides are computable.

### 5. Robust PCA: two proximal operators and a multiplier

```python
            L = _svt(X - S + Y / mu, 1.0 / mu)
            S = _soft(X - L + Y / mu, lam / mu)
            residual = X - L - S
            Y += mu * residual
            mu = min(mu * 1.5, 1e12)
            if np.linalg.norm(residual) <= self.tol * np.linalg.norm(X):
                break
```

That is (14) in five lines. The `Y / mu` term is the dual variable, and the
`mu` growth is what makes the constraint $X = L + S$ bind: early iterations are
a smooth penalty, late ones are a hard constraint. The initial
`mu = 1.25 / max|X|` keeps the first soft threshold from swallowing the whole
matrix on data with a large offset.

```python
def _svt(A: np.ndarray, tau: float) -> np.ndarray:
    """Singular-value soft threshold: the proximal operator of the nuclear norm."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return (U * s) @ Vt
```

`(U * s)` scales the *columns* of $U$ by the singular values before the product,
which is the whole trick: the reassembled matrix is $\sum_i (s_i-\tau)_+
u_iv_i^\top$, the closest matrix with nuclear norm reduced by $\tau$. Writing
`U @ np.diag(s) @ Vt` is the same thing with an extra allocation, and it is the
version that appears in most write-ups.

### 6. Sparse PCA: threshold the gradient, then deflate

```python
            for _ in range(self.n_iter):
                u = Xc.T @ (Xc @ v)
                u = _soft(u, self.lam * np.abs(u).max())
                nrm = np.linalg.norm(u)
                if nrm < 1e-12:
                    break
                v_new = u / nrm
                if np.linalg.norm(v_new - v) < self.tol:
                    v = v_new
                    break
                v = v_new
            comps[c] = v
            Xc = Xc - np.outer(Xc @ v, v)           # deflate one rank-one term
```

The threshold is relative to `np.abs(u).max()` inside the loop, which makes
`lam` a fraction rather than an absolute value — otherwise the same `lam` would
mean different things on standardised and unstandardised data. The `nrm < 1e-12`
break is the degenerate case where the threshold has zeroed the entire gradient;
returning that as a component would divide by zero.

Deflation uses `np.outer(Xc @ v, v)` rather than `np.outer(u, v)`: after
thresholding, $u$ is no longer $X_c^\top X_c v$, and deflating by the thresholded
vector would subtract the wrong rank-one term and lose variance that the next
component should have seen.

### 7. Whitening, and the refusal that keeps kernel PCA honest

```python
        Z = (X - self.mean_) @ self.components_.T
        if self.variant == "whitening":
            Z = Z / np.sqrt(self.explained_variance_)
        return Z
```

One branch, no second class. `explained_variance_` is $s^2/(n-1)$ from the same
SVD, so the division makes the covariance exactly $I$ — the demo checks
$5\times10^{-16}$. Note that it divides *after* the projection, so the
components themselves are unchanged; whitening is a diagonal rescale in score
space, not a different basis.

```python
        if self.variant == "kernel_pca":
            raise NotImplementedError(
                "kernel PCA has no exact preimage in input space; a preimage can be "
                "approximated by gradient descent on ||Φ(x) − z V^T||², which is not "
                "the same object and not implemented here"
            )
```

Kernel PCA has no `inverse_transform`, and the message says precisely why: a
preimage is an optimisation, not an inverse, and its answer depends on the
initialisation. Returning the nearest training point would look like an inverse
and be wrong in a way that is hard to detect downstream.

## Complexity

$n$ rows, $d$ columns, $k$ components, $T$ iterations.

| Variant | Time | Memory | Dominant term |
|:--|:--|:--|:--|
| `full_svd` | $O(nd\min(n,d))$ | $O(nd)$ | the SVD |
| `randomized` | $O(q\,nd(k+p) + n(k+p)^2)$ | $O(nd)$ | two passes per power iteration |
| `kernel_pca` | $O(n^2d + n^3)$ | $O(n^2)$ | the Gram matrix, then `eigh` |
| `incremental` | $O(bd^2 + b^2d)$ per block | $O(dk + bd)$ | the QR of the residual |
| `robust_pca` | $O(Tnd^2)$ | $O(nd)$ | one SVD per iteration |
| `sparse_pca` | $O(T\,k\,nd)$ | $O(nd)$ | $T$ passes per component |
| `whitening` | same as `full_svd` | $O(nd)$ | the SVD |

Two entries deserve a note. `robust_pca` is not competitive for large $n$: each
iteration is a full SVD, and the demo's 200×40 matrix converges in ~100
iterations, but a 4K video frame (the classic background-subtraction use case)
needs a truncated SVT rather than a full one. `kernel_pca` is $O(n^2)$ in
memory, so it is a small-data method; the $n^3$ eigendecomposition is why the
practical limit is a few thousand points.

## Failure modes

**1. Not centring, or centring the wrong thing.** `np.linalg.svd(X)` without
subtracting the mean gives components that point at the mean rather than at the
variance. The symptom is that the first component is nearly constant across
rows and the reconstruction is dominated by the offset. The code centres inside
`_reduce` for this reason. The subtler version is scaling: PCA on features in
different units is dominated by whichever has the largest variance, and the
"principal component" is that feature. Standardise first unless the units are
comparable, and say which you did.

**2. Reading explained variance as model quality.** For $k$ equal to the rank
of the data, the ratio is exactly 1, no matter how much noise there is — the
ratio is a property of the spectrum, not of the fit. On the demo's rank-3 data,
three components explain 100% of the variance and the discarded variance is
still $0.0177$. If you need to know how much of the *signal* you kept, you need
a noise estimate (`noise_variance_` is the mean of the discarded ones) or a
held-out reconstruction error. A scree plot is a heuristic, not a test.

**3. Randomized PCA with no power iterations on a flat spectrum.** The sketch
sees the spectrum through the Gaussian, so when $\sigma_{k+1}/\sigma_k$ is close
to 1 the top-$k$ subspace is not separated from the rest and the sketch returns
a mixture. The diagnostic is the singular values: if the $k$-th and $(k+1)$-th
are within 10%, increase `power_iter` or use `full_svd`. Power iterations do not
help when the sketch width is too small — the fix there is `oversample`.

**4. Kernel PCA with a badly scaled bandwidth.** $\gamma$ sets the length scale
of the RBF. Too small and $K \approx \mathbf{1}\mathbf{1}^\top$, so the centred
Gram matrix is numerical noise; too large and $K \approx I$, so every point is
its own cluster and the components are arbitrary. The default $\gamma = 1/d$ is
a convention, not a calibration; the diagnostic is the effective rank of $K_c$
(count eigenvalues above $10^{-6}\lambda_{\max}$). It should be well above $k$
and well below $n$.

**5. Test-time centring in kernel PCA.** Covered above and repeated here because
it is the most common bug in from-scratch implementations: a test block centred
by its own mean has scores in a different coordinate system from the training
scores. Every downstream comparison (nearest neighbour, classifier, clustering)
is then invalid, and nothing crashes. The demo checks the training case
explicitly because that is where the two conventions coincide.

**6. Robust PCA on dense corruption.** The recovery guarantee needs $S$ to be
sparse — the constant in $\lambda = 1/\sqrt{\max(n,d)}$ comes from a
sparse-corruption model. At 5% corrupted cells with magnitude 12, the demo's
own decomposition misses $L$ by 30%; at 2% it is 1%. If a large fraction of the
data is corrupted, robust PCA is not the tool; a model of the corruption is.

**7. Sparse PCA's variance bookkeeping.** After deflation the components are not
orthogonal, so "component 2 explains 20% of the variance" does not mean what it
means in PCA: the percentages do not add to 100 across a partition. Report the
reconstruction error of the $k$-component fit instead, or orthogonalise the
components afterwards if downstream code needs it.

## Applications

- **Visualisation and exploratory structure.** Two components of standardised
  features are the default first plot of any tabular dataset. It is a rotation,
  so it preserves distances and shows the dominant directions; it does not show
  clusters that are not linearly separable, and reading clusters off a PCA plot
  is the most common over-interpretation in the literature.
- **Denoising and compression.** Keep $k$ components, reconstruct, and the
  discarded variance is gone — this is lossy compression with an $L_2$ optimal
  basis. Image patches, sensor arrays, and spectra are the usual cases.
- **Anomaly detection.** The reconstruction error of a new point against a
  fitted subspace is a score; large error means the point does not lie in the
  learned low-dimensional structure. This is the basis of most
  "unusual-behaviour" detectors, and it fails exactly when the anomaly is
  low-rank too.
- **Factor extraction in finance.** The first principal component of a returns
  matrix is the market, the second is a curve/slope factor, and the residual is
  idiosyncratic. This is a descriptive decomposition, not a causal model, and
  the 2008 crisis was partly a story about people treating it as one.
- **Background subtraction.** Frames of a video form a matrix whose columns are
  pixels; robust PCA puts the static background in $L$ and moving objects in
  $S$. This is the canonical application of the nuclear-norm + L1 decomposition.
- **Preprocessing.** Whitening before k-means, Gaussian mixtures, or any model
  with an isotropic prior removes the need to learn the covariance. Kernel PCA
  as a feature map before a linear classifier is the nonlinear version.
- **Sparse loadings for interpretation.** In genomics and chemometrics, a
  component with 400 non-zero loadings is unreadable; sparse PCA gives one with
  20, at the cost of a slightly worse fit. When the goal is a named factor
  rather than a variance-maximising direction, this is the right trade.

Where PCA is a trap: any claim about *which* features matter. The loadings of a
component are a rotation of the original basis, and if two features are
correlated, PCA is free to split their contribution arbitrarily. Use sparse PCA
or a supervised method if the question is "which feature drives the outcome",
and use PCA only if the question is "which direction has the most variance".

## References

- **Pearson (1901),** *On lines and planes of closest fit to systems of points
  in space* — the original geometric statement: the best-fitting subspace, not
  the eigenproblem.
- **Hotelling (1933),** *Analysis of a complex of statistical variables into
  principal components* — the variance-maximisation formulation and the name.
- **Eckart & Young (1936),** *The approximation of one matrix by another of
  lower rank* — the optimality of the truncated SVD, equation (2).
- **Jolliffe (2002),** *Principal Component Analysis*, 2nd ed. — the reference
  for the statistics, the scaling questions, and the many ways PCA is misused.
- **Halko, Martinsson & Tropp (2011),** *Finding structure with randomness* —
  the randomized range finder and the error bounds behind equation (9).
- **Schölkopf, Smola & Müller (1998),** *Nonlinear component analysis as a
  kernel eigenvalue problem* — kernel PCA, including the centring identity.
- **Brand (2006),** *Fast low-rank modifications of the thin singular value
  decomposition* — the incremental update in equation (4).
- **Candès, Li, Ma & Wright (2011),** *Robust principal component analysis?* —
  the exact-recovery result and the choice $\lambda = 1/\sqrt{\max(n,d)}$.
- **Zou, Hastie & Tibshirani (2006),** *Sparse principal component analysis* —
  the elastic-net formulation; **Witten, Tibshirani & Hastie (2009)** for the
  penalized matrix decomposition used here.
