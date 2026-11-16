# Independent component analysis

`ica.py` · Latent structure · Tier 2 · 6 variants

## Intuition

PCA and factor analysis find directions in which the data is **uncorrelated**.
ICA finds directions in which it is **independent**:

$$x = A s, \qquad s_1, \ldots, s_d \text{ independent}, \qquad (1)$$

with no assumption that $A$ is orthogonal, and no assumption about the shape of
the sources beyond "at most one of them is Gaussian". Independence is a
statement about the whole joint distribution, so it is far stronger than zero
covariance — and, remarkably, it is a statement strong enough to determine $A$.

The demo's first line is the argument for why second moments are not enough.
Whiten the mixture and every rotation is invisible to covariance:

$$\mathrm{cov}(Z) = I \quad\text{to } 1.3 \times 10^{-15}, \qquad (2)$$

and yet the same projections still have squared values correlated at $0.30$.
They are uncorrelated and they are obviously not independent — the correlation
of $y_1^2$ with $y_2^2$ is a fourth moment, and nothing in the covariance can see
it. Whitening is a *coordinate change* that removes all the second-order
information, which is exactly why the remaining problem is a search over
rotations.

```
      sources                     mixture x = A s                whitened Z
   s2 │  ·  ·  ·                 x2 │   ╱                        z2 │    ●
      │ ·  ·  ·                    │  ╱                            │  ●   ●
      │  ·  ·  ·                   │ ╱                             │ ●      ●
      └──────── s1                 └──────── x1                    └──────── z1
    independent                correlated, sheared              uncorrelated:
    (separable)                (a direction of maximum           every rotation
                                variance is not a source)        looks the same
```

The three panels are the whole story. In the left panel the density factorises:
knowing $s_1$ tells you nothing about $s_2$. In the middle, $A$ has sheared and
rotated the cloud, so the axes of the ellipse are *not* the sources. In the
right, whitening has undone the ellipse — and produced something that looks
identical from every direction. The information that identifies the sources is
not in the shape of the cloud but in the shapes of the **marginals**, and
whitening has deliberately thrown away the axes that would tell you where they
are.

| | PCA | Factor analysis | ICA |
|:--|:--|:--|:--|
| model | $x = W^\top f + \mu$ | $x = \Lambda f + \mu + \varepsilon$ | $x = As$ |
| fitted by | variance | likelihood, $\Phi = I$ | non-Gaussianity |
| moments used | second | second | third and higher |
| mixing identified | up to rotation | up to rotation | up to permutation and scale |
| what breaks it | nothing | nothing | two Gaussian sources |

The last two rows are the trade. PCA's answer is arbitrary inside the retained
subspace and that is fine, because it is not claiming to be the true generating
directions. ICA claims exactly that, and pays for the claim with an assumption
about the source distributions that can be checked but not assumed away.

## Variants

| Variant | Objective | Algorithm | What it is for |
|:--|:--|:--|:--|
| `whitening` | none | one eigendecomposition | the baseline, and the necessary first step |
| `fastica_deflation` | negentropy of one projection | fixed point, then Gram–Schmidt | $k < d$ components, in order |
| `fastica_symmetric` | negentropy of all projections | fixed point, then symmetric decorrelation | all components at once |
| `infomax` | logistic likelihood | natural gradient, switching score | a likelihood, and mutual information |
| `jade` | fourth cumulant | eigenmatrices, then Jacobi | exact, and cheap for small $d$ |
| `kernel_ica` | kernel alignment | coordinate descent over Givens angles | dependence that is not a moment |

**`whitening`** is the coordinate change $Z = (x - \mu) W_w^\top$ with
$\mathrm{cov}(Z) = I$. It is a variant for a reason: it is what the other five
all start with, it is a legitimate answer to "decorrelate my data", and it makes
the size of the remaining problem visible. In whitened coordinates the mixture
is $z = B s$ with $B = W_w A$ **orthogonal**, because

$$B B^\top = W_w A A^\top W_w^\top = W_w \Sigma_x W_w^\top = I. \qquad (3)$$

So after whitening there are only $d(d-1)/2$ numbers left to find, and the whole
of ICA is a search over the orthogonal group. The demo measures what skipping
that search costs: the whitened basis is $67°$ away from the true one.

**`fastica_deflation`** finds one source at a time. Each row is the fixed point
of a negentropy contrast, then projected off the rows already found, so the
components come out in order of non-Gaussianity. It is the right variant when
you want $k < d$ of them, or when the sources are known to be ordered, and it is
the wrong one when you want all of them: the deflation error accumulates,
because row $k$ is estimated in the complement of estimates that already have
error.

**`fastica_symmetric`** updates all rows at once and repairs orthogonality with
one matrix square root per step,

$$W \leftarrow \left(W W^\top\right)^{-1/2} W, \qquad (4)$$

which is the orthogonal polar factor of $W$ and hence the nearest orthogonal
matrix to it. No row is favoured, the errors do not compound, and the iteration
is the same fixed point as the deflation form. The demo measures them agreeing
to $1.0°$ after 6 and 5 iterations respectively.

**`infomax`** comes at the same problem from a likelihood. Model each source
with a density $p_i$, so that $y = Wz$ has

$$\log p(x) = \log \lvert \det W \rvert + \sum_i \log p_i(w_i^\top z) + \text{const}, \qquad (5)$$

and ascend it with the **natural gradient**, which is what makes this practical:
the ordinary gradient's scale depends on $W^\top W$ and the step size then has to
be retuned as the matrix grows or shrinks. The variant's cost is the density
choice. A logistic density assumes a super-Gaussian source; the *extended* form
reads the sign of each source's kurtosis from the data and uses a symmetric
sub-Gaussian density for the ones that need it. On the demo's three sources —
Laplace, uniform, and a heavy-tailed product — the difference is the difference
between recovering them to $1.3°$ and not recovering them at all: forced to be
super-Gaussian, the uniform source is fitted as a mixture and the answer is
$48°$ wrong.

**`jade`** uses the fourth-order cumulant instead of optimising a marginal
contrast. In whitened coordinates the cumulant tensor defines an operator on
symmetric matrices whose eigenvectors are exactly the outer products of the
source directions, with the kurtoses as eigenvalues. Diagonalising them jointly
gives the unmixing in closed form — no step size, no learning rate, four Jacobi
sweeps on the demo's data, and exact in the sense that no step size or stopping
rule is involved. It is also the most brittle: a source with a kurtosis near zero
is indistinguishable from the exact zeros that whitening puts in the other
directions.

**`kernel_ica`** does not use a moment at all. It embeds each component's samples
in a Gaussian kernel's reproducing kernel Hilbert space and measures dependence
by the **normalised Hilbert–Schmidt inner product** of the two embeddings — 0
exactly when the components are independent, for a characteristic kernel, and
positive otherwise. The contrast is minimised over rotations by coordinate
descent. It is the slowest variant per fit and the one that works when the
non-Gaussianity is too weak for a cumulant: on a pair whose second source has
excess kurtosis $-0.08$ it recovers the mixing to $3.8°$ where JADE is $58°$ off.

## Derivation

### 1. Whitening, and the rotation that is left over

Let $S = \frac{1}{n} X_c^\top X_c = V \Lambda V^\top$. Then

$$W_w = \Lambda^{-1/2} V^\top, \qquad W_w S W_w^\top = I, \qquad (6)$$

which is exactly what `_whiten` computes: eigenvectors descending, each divided
by the square root of its eigenvalue. Any other whitening matrix is
$Q W_w$ for orthogonal $Q$, so **whitening itself has a rotation ambiguity of
the same size as the problem it is supposed to simplify**. The pair of facts
together are the structure of ICA:

- whitening fixes everything second-order, and leaves an orthogonal $B$ (3);
- the remaining ambiguity is a rotation, and independence is what picks it.

Note what whitening is *not*: a projection. No dimension is dropped unless its
variance is numerically zero, because dropping one would make $B$ rectangular
and the "rotation" a projection, which is a different (and much harder) problem.
The component count is decided by the estimator, not by the whitening.

### 2. Identifiability, in one polynomial identity

Write $\kappa_x(u) = \mathrm{cum}(u^\top x, u^\top x, u^\top x, u^\top x)$, the
fourth cumulant of a projection. Cumulants add over independent variables and
are homogeneous of degree four, so under (1)

$$\kappa_x(u) = \sum_{i=1}^{d} \kappa_i \left(a_i^\top u\right)^4, \qquad (7)$$

where $a_i$ is the $i$-th column of $A$ and $\kappa_i$ the kurtosis of $s_i$.
The left side is a quartic form in $u$ that you can estimate from data; the
right side is a decomposition of it into fourth powers of linear forms.

The decomposition is unique up to permutation and scale **provided at most one
$\kappa_i$ is zero**. That is Comon's theorem, and (7) is the whole proof
strategy: a quartic form has a finite set of rank-one decompositions, and the
$\kappa_i$ are what label them. Two corollaries the demo exercises:

- the scale of each column of $A$ is not identified, and neither is the order —
  a permutation of the columns and a rescaling of each one give the same $x$;
- if two sources are Gaussian, $\kappa_i = 0$ for both, their two terms are
  missing from (7), and the corresponding $2\times 2$ block of the decomposition
  is not determined at all. That is not a limitation of any algorithm; it is the
  distribution being invariant along that block.

### 3. Negentropy, and a contrast that can be estimated

Entropy is maximised, for a fixed variance, by the Gaussian. So

$$J(y) = H\!\left(y_{\text{Gauss}}\right) - H(y) \;\ge\; 0, \qquad (8)$$

with equality exactly when $y$ is Gaussian, is a natural measure of
non-Gaussianity: it is non-negative, scale-invariant at fixed variance, and zero
for the one case where ICA cannot work. It is also not a moment, which makes it
hard to estimate directly. Truncating the Gram–Charlier expansion of the density
gives the standard approximation

$$J(y) \approx \frac{\kappa_3(y)^2}{12} + \frac{\kappa_4(y)^2}{48}, \qquad (9)$$

for a unit-variance $y$, and `_negentropy` uses the second term:

$$J(y) \approx \frac{1}{48}\left(\mathbb{E}[y^4] - 3\right)^2. \qquad (10)$$

Two things about (10) are worth stating because both were bugs in this file.
First, the reference value is exactly 3 — a standard normal has
$\mathbb{E}[\nu^4] = 3$ — so no constant has to be estimated or calibrated, and
the proxy is exactly zero for Gaussian data. An earlier version used a tanh-based
proxy with a nonzero reference constant, and it was minimised by a *non*-Gaussian;
carrying a wrong constant inside a contrast does not produce a slightly worse
estimate, it produces a different criterion. Second, (10) is a function of $y^2$,
so it is invariant under $y \mapsto -y$, as negentropy is. Any proxy has to be,
since the sign of a source is not identified by (1).

### 4. FastICA, as a Newton step

Maximise $J(w^\top z) \approx \left(\mathbb{E}[G(w^\top z)] - \mathbb{E}[G(\nu)]\right)^2$
over $\lVert w \rVert = 1$, with $G$ a smooth even function — here
$G = \log\cosh$, so $g = G' = \tanh$. At an optimum of the squared quantity,
either $\mathbb{E}[G(w^\top z)] - \mathbb{E}[G(\nu)]$ is zero (the Gaussian
solution, which is what we are trying to avoid) or $w$ is a stationary point of
$\mathbb{E}[G(w^\top z)]$ under the constraint:

$$\mathbb{E}\left[z\, g(w^\top z)\right] - \beta w = 0, \qquad \beta = \mathbb{E}\left[g'(w^\top z)\right]. \qquad (11)$$

Newton's method on (11) needs the Jacobian
$\mathbb{E}[zz^\top g'] - \beta I$, and the approximation that makes FastICA
fast is to replace it by $\mathbb{E}[g']I - \beta I \propto I$: the data is
whitened, so $zz^\top$ is the identity in expectation and the outer product
contributes only its average. One Newton step is then

$$w^+ = \mathbb{E}\left[z\, g(w^\top z)\right] - \mathbb{E}\left[g'(w^\top z)\right] w, \qquad (12)$$

which is `_fastica_step` and it is three lines, because both expectations are
just means over rows.

The symmetric form applies (12) to every row and then repairs orthogonality. If
$W$ has orthonormal rows the fixed point is unchanged by (4): a symmetric
decorrelation of an already orthogonal matrix is the identity. If it does not,
(4) is the projection onto the orthogonal group in the Frobenius norm, so it is
the smallest correction that restores the constraint. Deflation instead projects
each new row off the previous ones — and does it **twice**, because one pass
leaves a residue of the size of the projection error, which is enough to stop the
iteration converging to its own tolerance.

### 5. The natural gradient, and why the identity term is not a detail

Differentiate (5) with respect to $W$:

$$\frac{\partial \ell}{\partial W} = W^{-\top} + \mathbb{E}\left[\psi(y) z^\top\right], \qquad \psi(y) = -\frac{d}{dy}\log p(y). \qquad (13)$$

The natural gradient preconditions by the Fisher information, which for this
parameterisation is $W^\top W$, so

$$\Delta W = \left(W^{-\top} + \mathbb{E}[\psi z^\top]\right) W^\top W = \left(I + \mathbb{E}\left[\psi(y)y^\top\right]\right) W. \qquad (14)$$

The identity term in (14) is not a convention: $W^{-\top}W^\top W = W$, and it is
what turns a gradient whose scale depends on $W$ into a multiplicative update
that is equivariant under linear changes of coordinates.

Now choose the density. For a super-Gaussian source, $p \propto 1/\cosh$, so
$\psi = -\tanh(y) - y$; for a sub-Gaussian one the signs of the odd part flip and
$\psi = +\tanh(y) - y$. Writing $\psi = -K\tanh(y) - y$ with $K = \pm 1$ per
component and using $\mathbb{E}[yy^\top] = I$ in (14),

$$\Delta W = \left(I - K\,\mathbb{E}\left[\tanh(y)y^\top\right] - I\right)W = -K\,\mathbb{E}\left[\tanh(y)y^\top\right]W, \qquad (15)$$

which is the form in the code. A fixed point of (14) satisfies
$\mathbb{E}[\psi(y)y^\top] = -I$, and `score_residual_` is the max-abs of that
residual — the demo measures $8 \times 10^{-8}$ on a converged fit. It is a
sharper check than the step size, because row scaling changes both $y$ and
$\psi$: a stationary step and a stationary score are not the same statement.

$K$ is estimated once, from the kurtosis of the initial projections, and then
**frozen**. The online extended-infomax update re-estimates it every step, which
is fine when each sample updates the matrix once; in a batch iteration that
switches the score function back and forth it oscillates between two fixed
points. And a wrong $K$ is not a small error: forcing $K = +1$ on a uniform
source asks the model to fit a sub-Gaussian marginal with a super-Gaussian
density, and the demo's answer lands $48°$ away with the source's apparent
kurtosis pushed from $-1.2$ to $0.1$.

### 6. JADE: the cumulant is an operator, and its eigenvectors are the sources

For whitened $z$, define

$$Q(M) = \mathbb{E}\left[z z^\top \left(z^\top M z\right)\right] - \mathrm{tr}(M) I - 2M. \qquad (16)$$

The subtraction is forced by whiteness. Expanding the expectation in cumulants,

$$\mathbb{E}\left[z_i z_j (z^\top M z)\right] = \kappa_{ijkl} M_{kl} + \delta_{ij}\,\mathrm{tr}(M) + 2M_{ij}, \qquad (17)$$

so (16) is the fourth-cumulant part alone. It is a linear operator on symmetric
matrices, self-adjoint in the Frobenius inner product.

In the source basis $z = Bs$ with $B$ orthogonal, $\kappa_{ijkl} = \sum_p \kappa_p B_{ip}B_{jp}B_{kp}B_{lp}$, and therefore

$$Q\left(u_p u_p^\top\right) = \kappa_p\, u_p u_p^\top, \qquad Q\left(u_p u_q^\top + u_q u_p^\top\right) = 0 \ \ (p \ne q), \qquad (18)$$

where $u_p$ is the $p$-th column of $B$. So the operator's eigenvectors are the
rank-one matrices $u_p u_p^\top$, its nonzero eigenvalues are the kurtoses, and
everything else sits in an eigenvalue-zero subspace of dimension $d(d-1)/2$. The
code builds $Q$ in an orthonormal basis of symmetric matrices, takes the $d$
eigenvectors of largest $\lvert \lambda \rvert$, and joint-diagonalises them.

Joint diagonalisation means finding orthogonal $V$ with $V^\top M_l V$ diagonal
for every $l$. For a single $l$ that is the symmetric eigenvalue problem; for a
set, the Jacobi approach is to rotate one $(p, q)$ plane at a time to the angle
that minimises that plane's off-diagonal contribution over all $l$. Under a
rotation by $\theta$ of the plane, with $a = M_{pp}$, $d = M_{qq}$,
$b = M_{pq}$ and $u = (a - d)/2$,

$$M'_{pq} = b \cos 2\theta - u \sin 2\theta, \qquad (19)$$

and summing the square over the set,

$$\sum_l \left(M'_{pq}\right)^2 = \tfrac{1}{2}\left(\Sigma u^2 + \Sigma b^2\right) - \tfrac{1}{2}\left(\Sigma u^2 - \Sigma b^2\right)\cos 4\theta - \Sigma ub \sin 4\theta, \qquad (20)$$

which is minimised at

$$4\theta = \mathrm{atan2}\left(2\Sigma ub,\; \Sigma u^2 - \Sigma b^2\right). \qquad (21)$$

The factor of four is the whole subtlety, and it cost this file a day. Because
$4\theta$ appears twice in (20), the criterion has period $\pi/2$ in $\theta$; a
formula derived from the eigenvector of the quadratic form in
$(\cos 2\theta, \sin 2\theta)$ gives $\theta$ twice too large unless that is
noticed, and the resulting rotation still *decreases* the criterion at every
step. It converges monotonically, in a handful of sweeps, to a rotation that is
$10$–$20°$ short of the answer and looks entirely plausible. The symptom is not
divergence; it is an off-diagonal mass that stops at 70% instead of 8%.

### 7. Kernel ICA: dependence without moments

Let $K$ be a Gaussian kernel matrix on one component's samples and $L$ on
another's, both centred in feature space by $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$
so that $HKH$ has zero row and column means. The Hilbert–Schmidt inner product
$\langle HKH, HLH \rangle_F$ is the empirical HSIC of the pair, and HSIC is zero
exactly when the two variables are independent — not merely uncorrelated —
provided the kernel is characteristic, which the Gaussian one is. Normalising by
the two Frobenius norms gives a bounded correlation,

$$\rho = \frac{\langle HKH,\, HLH\rangle_F}{\lVert HKH \rVert_F \, \lVert HLH \rVert_F} \in [-1, 1], \qquad (22)$$

and the contrast is $\sum_{i<j} \rho_{ij}$, minimised over the rotation. For
$d = 2$ there is one pair and one angle, so a grid search is an exhaustive
minimisation up to its resolution: the demo's fitted contrast, $0.00655$, is the
minimum of an independent 721-point scan to five decimals.

Three implementation notes that are not incidental:

- the bandwidth is one scalar, from the **median off-diagonal** pairwise distance
  of the whitened sample. Whitening has already fixed the scale, so one knob is
  right, but the median must exclude the diagonal: including $n$ zeros in $n^2$
  entries halves the median and drives the kernel towards the identity;
- the contrast is evaluated on a subsample (300 rows here), because each
  evaluation costs $O(n^2 d)$ and a grid search needs dozens of them;
- the search is coordinate descent over Givens angles, coarse-to-fine, not a
  gradient method. With $d = 2$ it is exact by construction.

### 8. The boundary: why two Gaussians cannot be separated

If $s \sim \mathcal{N}(0, I)$ and $z = Bs$ with $B$ orthogonal, then
$z \sim \mathcal{N}(0, BB^\top) = \mathcal{N}(0, I)$ — the same distribution, for
**every** orthogonal $B$. The likelihood, the negentropy contrast, the cumulant
tensor and the kernel contrast are all constant along the rotation group, so no
method has a maximum to find. The demo measures the flatness directly: the
negentropy proxy varies by $10^{-4}$ over the whole circle, which is less than
its own sampling noise, and six FastICA seeds stop at $40.5°$ and $49.5°$ — both
"converged", both stationary, both meaningless.

This is Comon's condition from §2 seen from the other side. With $d = 2$ and both
kurtoses zero, (7) is the zero polynomial and carries no information; with one
Gaussian the polynomial still determines the other direction, which is why the
demo's JADE recovers a mixture to $2.8°$ even though it contains a nearly
Gaussian source, and why it fails at $58°$ when the *only* non-Gaussian source is
that weak and $d = 2$.

## Code walkthrough

`ICA(n_components=None, *, variant=..., seed=..., extend=True, max_iter=...,
tol=..., lr=...)` with `fit`, `transform`, `fit_transform`, and the attributes
`components_` $(k, d)$, `mixing_` $(d, k)$, `whitening_`, `rotation_`,
`contrast_curve_`, `n_iter_`, `converged_`, plus the variant-specific
`bandwidth_`, `kernel_sample_`, `kurtosis_signs_`, `negentropy_` and
`score_residual_`.

Two conventions are load-bearing. `components_` has the same orientation as
sklearn's — one row per component, so `transform(X) = (X - mean_) @ components_.T`
— and `components_ = rotation_ @ whitening_` holds for every variant, which is
what makes them comparable: they whiten identically and differ only in the
orthogonal map. `transform` returns the sources with whitening's scaling, not the
true sources' scaling, because the scaling is not identified in the first place.

### 1. Whitening in six lines

```python
    mean = X.mean(axis=0)
    Xc = X - mean
    S = Xc.T @ Xc / X.shape[0]
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    keep = w > EPS * max(float(w[0]), 1.0)
    W_w = (V[:, keep] / np.sqrt(w[keep])).T          # (r, d), rows are directions
    return mean, W_w, Xc @ W_w.T
```

(6), with the two decisions that matter made explicit. `keep` is a **rank test,
not a variance cut**: the threshold is `EPS` times the largest eigenvalue, so a
direction is discarded only when it is numerically indistinguishable from zero,
never for being small. That is what keeps whitening a change of coordinates
rather than a projection. Dividing eigenvectors by $\sqrt{w}$ and transposing
puts the directions in the **rows**, which is why `Z = Xc @ W_w.T` reads as a
projection onto them.

### 2. The fixed-point step

```python
        U = Z @ W.T
        G = np.tanh(U)                                # g = tanh, logcosh's derivative
        Gp = 1.0 - G * G                              # g'
        return (G.T @ Z) / Z.shape[0] - Gp.mean(axis=0)[:, None] * W
```

(12), for all rows at once. `G.T @ Z` is $\mathbb{E}[z g(w^\top z)]$ with the
expectation taken over rows, and `Gp.mean(axis=0)` is
$\mathbb{E}[g'(w^\top z)]$ per row — one scalar per component, which is exactly
the approximation that made the Newton step cheap. $\tanh$ and
$1 - \tanh^2$ are the same two numbers, so `Gp` costs no second transcendental.

### 3. Symmetric decorrelation

```python
                W_new = self._fastica_step(Z, W)
                # symmetric decorrelation: (W⁺W⁺ᵀ)^{-1/2}W⁺ makes the rows
                # orthonormal without singling any of them out
                W_new = _sym_inv_sqrt(W_new @ W_new.T) @ W_new
                change = float(np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0)))
```

(4) in one line. The convergence test is the row-wise dot product with the
previous iterate: at a fixed point the rows are unchanged up to sign, so
$\lvert \langle w_i^{\text{new}}, w_i \rangle \rvert \to 1$ and the max deviation
from 1 is the natural residual. Comparing the matrices entrywise instead would
fail, because a sign flip changes every entry of a row.

### 4. Deflation's double projection

```python
                w_new = Z.T @ g / n - gp.mean() * w
                # two Gram–Schmidt passes: one leaves a residue of the size of
                # the projection error, which is enough to spoil convergence
                for _ in range(2):
                    if i:
                        w_new -= (W[:i] @ w_new) @ W[:i]
                w_new /= max(np.linalg.norm(w_new), EPS)
```

The same (12) without the matrix form, then the constraint: $w$ must be
orthogonal to the $i$ rows already found. `W[:i] @ w_new` are the projections and
the subtraction removes them; doing it twice is classical Gram–Schmidt
reorthogonalisation, and one pass is not enough because the first row's error is
amplified by the second projection. The final normalisation is what makes
`done = abs(w_new @ w) > 1 - tol` a meaningful test.

### 5. Infomax: the update and the score

```python
            for it in range(1, self.max_iter + 1):
                U = Z @ W.T
                psi = -K * np.tanh(U) - U
                # ΔW = lr (I + E[ψ(u)uᵀ]) W: the natural gradient, whose identity
                # part cancels the −u in ψ and leaves −K E[tanh(u)uᵀ]
                W_new = W + self.lr * ((np.eye(d) + psi.T @ U / n) @ W)
```

(14) and (15) together. `psi.T @ U / n` is $\mathbb{E}[\psi(u)u^\top]$, a $d
\times d$ matrix whose diagonal is the natural step size for each component and
whose off-diagonals are the cross-talk that the natural gradient is removing.
`K` is a vector of $\pm 1$, one per component, frozen before the loop.

```python
            U = Z @ W.T
            psi = -K * np.tanh(U) - U
            resid = float(np.max(np.abs(np.eye(d) + psi.T @ U / n)))
```

The stationarity check, computed on the $W$ at which the loop stopped — before
the row normalisation, because normalising changes $u$ and therefore $\psi$. The
demo's converged run gives $8 \times 10^{-8}$; the stubborn one, which hits its
iteration cap, gives $2 \times 10^{-6}$ and reports `converged_ = False`.

### 6. The cumulant operator

```python
        q = np.einsum("ij,jk,ik->i", Z, M, Z)
        return (Z * q[:, None]).T @ Z / Z.shape[0] - np.trace(M) * np.eye(len(M)) - 2.0 * M
```

(16) literally. `einsum("ij,jk,ik->i", Z, M, Z)` is the row-wise quadratic form
$z_i^\top M z_i$ without materialising an $n \times d \times d$ tensor, and the
three terms are the expectation, the trace correction and the $2M$ correction
from (17). It is a static method precisely because it depends on nothing but the
data.

### 7. From the operator to the matrices to diagonalise

```python
        m = len(pairs)
        Op = np.zeros((m, m))
        for col, E in enumerate(basis):
            Q = self._cumulant_apply(Z, E)
            for row, F in enumerate(basis):
                Op[row, col] = float(np.sum(Q * F))
        lam, vec = np.linalg.eigh(Op)
        order = np.argsort(np.abs(lam))[::-1][:d]     # the d significant eigenmatrices
        Ms = [sum(vec[i, c] * basis[i] for i in range(m)) for c in order]
```

The operator applied to each basis element, then projected onto every basis
element: $O_{rc} = \langle F_r, Q(E_c) \rangle_F$, an $m \times m$ matrix with
$m = d(d+1)/2$. Because the basis is orthonormal, `eigh` on it gives both the
eigenvalues and the eigenmatrices, and the last line turns a coordinate vector
back into a matrix. Sorting by $\lvert \lambda \rvert$ rather than by $\lambda$
matters: kurtoses come with both signs, and a strongly sub-Gaussian source is as
informative as a super-Gaussian one. Taking $d$ of them is (18).

### 8. The Jacobi angle

```python
                    u = np.array([(M[p, p] - M[q, q]) / 2.0 for M in Ms])
                    b = np.array([M[p, q] for M in Ms])
```

then

```python
                    theta = 0.25 * math.atan2(2.0 * float(u @ b), float(u @ u - b @ b))
                    c, s = math.cos(theta), math.sin(theta)
                    R = np.eye(d)
                    R[p, p] = R[q, q] = c
                    R[p, q], R[q, p] = -s, s
                    Ms = [R.T @ M @ R for M in Ms]
                    V = V @ R
```

(19)–(21). `u` and `b` are the two vectors-of-scalars over the set of matrices:
the half-difference of diagonals and the off-diagonal, both for the pair $(p,q)$.
Everything after the angle is bookkeeping — build the Givens rotation, apply it
to every matrix by congruence, and accumulate it in `V` so the total map is
recoverable. `V` starts as the identity and ends as the diagonaliser; the returned
`V.T` is the *unmixing*, because `sources = Z @ V` and the class composes
`rotation_` on the left.

Note the difference between "apply" and "replace": `Ms = [R.T @ M @ R ...]` is
what makes the sweep monotonically increase the diagonal mass, and the stopping
test compares that mass before and after the whole sweep, not the angle.

### 9. Kernel alignment, and the bandwidth that must not include the diagonal

```python
        Y = Z @ U.T
        n = Y.shape[0]
        H = np.eye(n) - 1.0 / n
        Ks = []
        for i in range(Y.shape[1]):
            D = (Y[:, i][:, None] - Y[:, i][None, :]) ** 2
            K = H @ np.exp(-0.5 * D / self.bandwidth_) @ H
            Ks.append(K / max(float(np.linalg.norm(K)), EPS))
        return float(sum(np.sum(Ks[i] * Ks[j])
                         for i in range(len(Ks)) for j in range(i + 1, len(Ks))))
```

(22). `H @ K @ H` is the double centring, `D` is the squared-distance matrix, and
the normalisation to unit Frobenius norm is what makes each pair's score
comparable across bandwidths. `np.eye(n)` is an $n \times n$ matrix and this is
the reason the contrast is restricted to a subsample: at $n = 1500$ one
evaluation would be $2$ kernel matrices and four $1500^2$ multiplications, times
dozens of grid points.

```python
        D = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2)
        self.bandwidth_ = max(float(np.median(D[np.triu_indices(len(sub), 1)])), 1e-6)
```

One bandwidth, from the median pairwise distance — of the **full** vectors, so
each pair contributes one number, and restricted to the strict upper triangle, so
the $n$ zeros on the diagonal do not drag the median down. Getting this wrong is
not a tuning issue: a median of zero makes the kernel the identity and the
contrast a number with no dependence signal in it at all.

### 10. Composition, and the one refusal

```python
        self.unmixing_ = self.rotation_ @ self.whitening_
        return self
```

The whole model in one line: whiten, then rotate. Every variant is a different
`rotation_`, and every attribute downstream — `components_`, `mixing_`,
`transform` — goes through this product, so a variant with a transposed
convention cannot hide.

```python
        if k < d and self.variant != "fastica_deflation":
            raise ValueError(
```

Only deflation can extract fewer components than there are features, and the
reason is structural rather than a missing feature: the other variants optimise a
contrast over a full orthogonal matrix, and dropping to a $k \times d$
rectangular one changes the constraint set from $O(d)$ to a Stiefel manifold. The
error says which variant to use instead, because the alternative — silently
returning the first $k$ rows of a $d$-component fit — would look like it worked.

## Complexity

$n$ rows, $d$ features, $T$ iterations.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_whiten` | $O(nd^2 + d^3)$ | $O(d^2)$ | one covariance, one eigendecomposition |
| `fastica_symmetric` | $O(Tnd^2)$ | $O(d^2)$ | two products per iteration |
| `fastica_deflation` | $O(Tnd^2)$ | $O(d^2)$ | $d$ rows, each with its own iteration |
| `infomax` | $O(Tnd^2)$ | $O(d^2)$ | three starts by default |
| `jade` | $O(nd^4 + d^6)$ | $O(d^4)$ | the $m \times m$ operator, $m = d(d+1)/2$ |
| `kernel_ica` | $O(Gn_s^2d)$ | $O(n_s^2)$ | $G$ contrast evaluations, $n_s$ subsample |
| `transform` | $O(mdk)$ | $O(mk)$ | one product |

Three things follow. First, the gradient variants are linear in $n$ and
**quadratic in $d$**, and the cost per iteration is dominated by producing
$U = ZW^\top$; the constant is small enough that 4000 rows and three components
converge in a few milliseconds. Second, JADE is the outlier: the operator is
$d(d+1)/2$ squared, so it is the cheapest method at $d = 3$ and hopeless at
$d = 100$ — it is the variant for small, well-conditioned problems where
exactness is worth the price. Third, kernel ICA is quadratic in the *subsample*,
not in $n$, which is why it is affordable at all; the price is a contrast with
sampling noise, and $3.8°$ rather than $1°$ of accuracy.

The whole six-section demo runs in 2.0 s.

## Failure modes

**1. Two Gaussians are not separable, and more data does not help.** The demo
measures a negentropy range of $10^{-4}$ over the entire rotation circle, and six
FastICA seeds stopping at $40.5°$ and $49.5°$ — both `converged_ = True`. This is
the one failure mode that no amount of engineering fixes, because the objective
is flat (8). The practical test is to look at the recovered kurtoses: if one is
near zero, its component is not identified, and the reported direction is
whatever the seed produced. A corollary that catches people: whitening a
Gaussian mixture produces something *exactly* spherical, so a pipeline that
whitens, runs ICA and reports the components will report five orthogonal
directions for pure noise.

**2. A near-zero cumulant defeats a cumulant method.** On the demo's two-source
pair with kurtoses $2.09$ and $-0.08$, JADE lands $58°$ from the truth. The
operator's eigenvalues are $2.09$ and $-0.08$; the remaining $d(d-1)/2$ are
exactly zero by whiteness, so "the $d$ largest in absolute value" picks a noise
eigenvector and the joint diagonalisation does its best with it. The fix is not a
better Jacobi sweep — there is nothing there to diagonalise. Use a contrast that
is not a moment (`kernel_ica` gets $3.8°$ here), or get more data so that $-0.08$
separates from zero, or reduce $d$ so that the zero subspace is smaller.

**3. A wrong score function is a wrong model, not a slow fit.** Forced to assume
every source is super-Gaussian, infomax converges cleanly, satisfies its own
fixed-point condition, and lands $48°$ away with the uniform source's kurtosis
reported as $0.1$ instead of $-1.2$ — the model has fitted the source as a
mixture of two super-Gaussian directions. Nothing in the output flags this. The
tell is again the recovered kurtoses: a source that was strongly sub-Gaussian in
the data and comes back near Gaussian has been absorbed.

**4. Extended infomax oscillates if the switch is not frozen.** Re-estimating
$K$ every iteration gives a run whose relative step never falls below about
$10^{-2}$ — it cycles between two score functions instead of settling on one, so
it neither converges nor satisfies the stationarity condition, and 20 000
iterations do not help. Frozen after the first estimate, nine of ten seeds reach
the same stationary point and the tenth lands $48°$ away, which is what the
three-start selection in `_fit_infomax` is for. This is a batch-versus-online
difference: the online form updates the matrix once per sample, so the switch is
a slowly varying function of the data, while a full-batch iteration that
re-evaluates it sees a discontinuity every step.

**5. Whitening alone is not an unmixing.** It is $67°$ from the truth on the
demo's data, and it is the answer to a different question. It is also easy to
mistake for ICA because it produces decorrelated components with unit variance
and a plausible-looking mixing matrix. The diagnostic is the same one the demo
opens with: correlate the *squares* of the components. If those are not near
zero, the components are still dependent, whatever the covariance says.

**6. FastICA's answer is accurate to about a degree, not exactly.** The demo's
two implementations recover the true directions to $1.20°$ and $1.25°$ on 4000
rows. That is sampling error, not a convergence failure: the contrast is
estimated from data and its maximum moves with the sample. Reporting six decimal
places from a 4000-row fit is reporting the optimiser's tolerance, not the
statistical accuracy. It also means that "deflation and symmetric agree to
$1.0°$" is the strongest statement available — they agree to within the noise,
which is what the demo asserts.

**7. Deflation's error accumulates, and one Gram–Schmidt pass is not enough.**
Row $i$ is estimated in the orthogonal complement of rows $1 \ldots i-1$, so
their errors are inherited. One projection pass leaves a residual of the size of
the projection error, which is the same order as the tolerance and enough to stop
the iteration converging; two passes are standard and cost almost nothing. With
$k$ components the composite error grows roughly like $\sqrt{k}$ and the last
component is the worst — the reason to prefer the symmetric variant when all
$d$ are wanted.

**8. Kernel ICA is slow, and its contrast is noisy.** 70 contrast evaluations on
a 300-row subsample cost about 0.1 s here, and the search is over
$d(d-1)/2$ angles in sequence, so the cost grows like $d^2$ evaluations with
$O(n_s^2 d)$ each. The contrast itself is a $U$-statistic with $O(1/n_s)$ bias,
so the subsample sets a floor on accuracy: $3.8°$ on the demo's pair against
FastICA's $2°$. It is the variant to reach for when the moments are uninformative,
not the variant to reach for by default.

**9. The scale of the recovered components is arbitrary.** $s$ and $Ds$ for
diagonal $D$ give the same $x$, so no method can recover the true scale, and the
one this code returns is fixed by whitening: every component has unit variance on
the training data. A coefficient with a physical meaning, such as a source
amplitude in physical units, has to be calibrated against something external. The
same applies to sign: `components_` rows may come out flipped, which is why every
comparison in the demo uses $\lvert\cos\rvert$ rather than a dot product.

**10. It assumes as many sensors as sources, and no noise.** With $A$ square and
whitening exact, (3) makes the problem orthogonal. With more sources than sensors
the problem is underdetermined and the answer is a sparse decomposition rather
than a rotation; with noise the whitening is biased and the bias propagates into
every component. Neither is handled here, and both are common in the
applications below — which is why the classical ICA pipeline is nearly always
preceded by a dimension reduction (PCA to $k$ components) and a noise-whitening
step, at the cost of assuming the discarded directions are noise.

**11. It needs more rows than columns, and it says so.** The check refuses
$n < 2d$. Whitening needs $n \ge d$ to have a nonsingular covariance; the
higher-order statistics need considerably more, because a fourth moment's
standard error is $O(1/\sqrt{n})$ and the contrast differences between
candidate rotations are small. At $n = 2d$ the fit runs and the answer is noise.

## Applications

- **The cocktail party.** The original problem: $d$ microphones, $d$ speakers,
  and a room that convolves as well as mixes. Instantaneous ICA is what this file
  implements; the convolutive version moves to the frequency domain and runs one
  ICA per bin, then has to solve the permutation ambiguity across bins — the same
  non-identifiability, in a form that costs real engineering.
- **EEG and MEG.** Blind source separation of eye blinks, cardiac artefact and
  muscle from neural signal. The sources are super-Gaussian and few, the mixing
  is instantaneous because the skull conducts slowly, and ICA is the standard
  tool. Care is needed: the components are only as interpretable as the
  independence assumption, and a dipole fitted to an artefact component is a
  dipole fitted to a rotation.
- **fMRI.** Spatial ICA for resting-state networks, where the "sources" are
  spatial maps and the samples are time points. The independence assumption is
  weaker here — networks are not independent in any physical sense — and the
  method works anyway, which is worth knowing but does not make the components
  identifiable.
- **Telecommunications.** MIMO demultiplexing, where the sources are symbol
  streams and the mixing is the channel matrix. This is the cleanest application:
  the sources are genuinely independent by construction, at most one is Gaussian,
  and the sample size is large.
- **Audio and music.** Separating instruments and vocals; also the honest
  counterexample, since musical sources are *dependent* by design — they are
  played together, in time.
- **Financial returns.** Independent components of a return matrix are used as
  statistical risk factors, the same job as PCA factors but with a different
  criterion. The independence assumption is plainly false (factors comove in a
  crisis), and the honest reading is "a rotation chosen by a different rule",
  not "the true economic drivers".
- **Astronomy.** Separating the cosmic microwave background from foregrounds;
  the components have very different marginal shapes, which is the regime ICA is
  built for.
- **A testbed for identifiability.** Along with factor analysis's rotation
  problem, this is the cleanest example in statistics of a model whose likelihood
  is flat along a manifold of parameters — and here the manifold is continuous
  only in the Gaussian case. §8 and the demo's sixth section are the same
  phenomenon as factor analysis's rotation non-identifiability, with the
  difference that ICA's criterion resolves it and the Gaussian case does not.

Where it is a trap: whitening, running ICA, and reading the components as the
data's true structure when the data is nearly Gaussian; reporting component
order or sign; fitting a physical model to a component that is one of several
equivalent rotations; and treating a deflation fit's $k$-th component as being as
reliable as its first.

## References

- **Comon (1994),** *Independent component analysis, a new concept?* — the
  identifiability theorem and the quartic-form argument of §2, and the first
  statement of the problem as one of contrast maximisation.
- **Hyvärinen & Oja (2000),** *Independent component analysis: algorithms and
  applications* — the survey the FastICA derivation follows, including the
  symmetric decorrelation and the choice of contrast functions.
- **Hyvärinen (1999),** *Fast and robust fixed-point algorithms for independent
  component analysis* — the fixed-point iteration (12), the Newton approximation
  that makes it cheap, and the deflation form.
- **Bell & Sejnowski (1995),** *An information-maximization approach to blind
  separation and blind deconvolution* — infomax, the logistic score, and the
  natural gradient (14); the identity term in it is theirs.
- **Lee, Girolami & Sejnowski (1999),** *Independent component analysis using an
  extended infomax algorithm for mixed sub-Gaussian and super-Gaussian sources* —
  the switching score function of §5, and the reason the sign of the kurtosis
  decides which density is used.
- **Cardoso & Souloumiac (1993),** *Blind beamforming for non-Gaussian signals* —
  JADE: the cumulant operator (16), its eigenmatrices (18), and the Jacobi
  joint-diagonalisation this file implements.
- **Bach & Jordan (2002),** *Kernel independent component analysis* — the kernel
  contrast of §7, and the generalised-variance version this file replaces with a
  normalised HSIC because the determinant version goes singular on subsamples.
- **Gretton, Bousquet, Smola & Schölkopf (2005),** *Measuring statistical
  dependence with Hilbert–Schmidt norms* — HSIC, its bias, and the characteristic
  kernel condition that makes zero dependence mean independence.
- **Cardoso (1998),** *Blind signal separation: statistical principles* — the
  performance bounds, and why the Gaussian case is not a hard problem but an
  impossible one.
