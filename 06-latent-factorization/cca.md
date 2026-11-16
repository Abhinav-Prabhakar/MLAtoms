# Canonical correlation

`cca.py` · Latent structure · Tier 2 · 5 variants

## Intuition

Two data sets, the same $n$ objects. Maybe the same patients with gene expression
in one matrix and methylation in another; the same images with pixels and with
captions; the same time points with two different sensor arrays. You want the
one direction in each view along which the two views agree most.

```
        X (n × p)                  Y (n × q)
        ┌────────┐                 ┌────────┐
   rows │ · · ·  │                 │ · · ·  │     the same n objects
        │ · · ·  │                 │ · · ·  │     different measurements
        └───┬────┘                 └───┬────┘
            │ w_x                      │ w_y
            ▼                          ▼
        u = X w_x  ──── corr ────  v = Y w_y
                  ρ = max
```

That is the whole problem. It is a maximisation over $p + q$ numbers of a single
cosine, and unlike almost everything else in this chapter it has a closed form: a
generalised eigenvalue problem, solved by an SVD after whitening each view.

The reason it deserves an atom of its own is what the answer *means*. CCA does
not return coordinates; it returns **directions in the span of each view**. Apply
any invertible linear map to $X$ — rotate it, scale its columns, mix them — or
any shift, and the canonical correlations are unchanged, because the map does not
change the span and the answer was never about the coordinates. The demo asserts
this to $2.8 \times 10^{-14}$: the fitted correlation moves in the fourteenth
digit when each view is multiplied by a random invertible matrix and shifted.

That invariance is the whole difference between CCA and regression:

| | asks | invariant to |
|:--|:--|:--|
| regression $y \sim X$ | which of $X$'s columns predict $y$ | nothing — the columns are the model |
| PCA | which directions carry variance in one view | rotation, not scaling |
| CCA | which *directions* two views share | any invertible linear map |

The other side of the same coin: the weights are *only* defined up to that map.
A coefficient of $0.4$ in `weights_[0]` means nothing on its own, and a sparse
CCA's support is a statement about the coordinates you happened to supply. That
is fine when the coordinates are genes with names and wrong when they are the
first 30 principal components of something.

The four remaining variants are all responses to one failure, and the failure is
worth stating precisely because it is not a numerical accident. The objective
maximises over $p + q$ parameters subject to two constraints. Once $p + q \ge n$
there are enough degrees of freedom for the two centred views to intersect, and
then a correlation of exactly $1$ is attainable between views that are, in fact,
independent. The demo measures it: two $40 \times 20$ matrices of pure noise give
$\hat\rho = 1.000000$. Ridge it, sparsify it, kernelise it, or add views — those
are the four standard answers, and they fail in four different ways.

## Variants

| Variant | What it does | Cost | Where it wins |
|:--|:--|:--|:--|
| `cca` | whiten both views, SVD the cross-covariance | $O(n(p^2+q^2))$ | the answer, when $n$ is large |
| `regularized_cca` | add $\lambda I$ to each within-view covariance | same | $p$ comparable to $n$; held-out correlation |
| `kernel_cca` | CCA in a feature space given by a Gram matrix | $O(n^3)$ | relations that are not linear |
| `sparse_cca` | L1 penalty on the weights, alternating exact lasso steps | $O(p^2)$ per inner sweep | when a direction must be a short list |
| `multi_view` | one shared variate for $D \ge 2$ views (SUMCOR) | $O(D n p^2)$ | more than two views |

**`cca`** is Hotelling's 1936 problem, and "canonical" means the same thing it
means for a basis: the components come out ordered and mutually orthogonal under
each view's own metric. The demo's planted problem has 20 columns per view, a
5-column true direction in each, and a population correlation of $0.90$; the fit
returns $0.9122$ with cosine $0.964$ and $0.949$ to the two planted directions.
Getting the direction right to three digits out of five non-zero coefficients is
what 400 samples buy at that noise level, and it is the number to compare every
other variant against.

**`regularized_cca`** is the ridge, and it is the variant most often used without
being named. On 40 training rows of a planted $0.60$ signal it reads $1.00$
training and $0.02$ held-out; at $\lambda = 0.3$, $0.91$ and $0.37$; at
$\lambda = 3$, $0.80$ and $0.47$. Both curves are monotone, which is the entire
trade in one line: the ridge gives up the quantity it maximises in exchange for
one it does not. The held-out number still falls short of the true $0.60$ — the
ridge stops 40 rows from lying, it does not give them information.

**`kernel_cca`** replaces $X$ and $Y$ with Gram matrices. On the same planted
data with a linear kernel it reproduces `cca` to $2 \times 10^{-9}$ in correlation
and $1.00000$ cosine between the variates, which is the dual derivation checked
against the primal one. On a circle and its double angle — $x = (\cos t, \sin t)$,
$y = (\cos 2t, \sin 2t)$, a deterministic quadratic relation — linear CCA finds
$0.11$ on 100 training points and $0.00$ held out, while the RBF kernel finds
$0.90$ and $0.79$. The kernel adds no information; it changes the space in which
a linear direction is a nonlinear function.

**`sparse_cca`** puts an L1 penalty on the weights and solves each half-step
exactly. At $\lambda = 0$ the fit is bit-for-bit `cca` ($1.000000$ cosine, same
correlation to four decimals), which is the correct sanity check and one that
most implementations fail. At $\lambda = 0.02$ on 800 rows it keeps 10 of 30
coordinates per view and 4 of the 5 that are real, at $0.9069$ against the dense
$0.9124$; at $\lambda = 0.05$ it keeps 4 and hits 4, at $0.9035$; at
$\lambda = 0.2$ every coordinate is thresholded away and the solver raises rather
than returning a zero vector.

**`multi_view`** generalises to $D$ views by asking every view's variate to agree
with one shared $z$ rather than pairwise. With two views it reduces to `cca`
exactly ($1.3 \times 10^{-13}$ apart). With three — one shared latent, noise
$0.5$, $0.6$ and $2.5$ — the joint correlation is $0.40$ and the individual
correlations with the latent are $0.83$, $0.83$ and $0.21$. That $0.40$ is not an
average over the views, it is the mean of the three pairwise numbers ($0.71$,
$0.26$, $0.22$), which is what the shared-variate objective maximises: one bad
view costs a third of the score without destroying the good pair.

## Derivation

### 1. The objective is a correlation, and correlations are scale-free

Write $\Sigma_{xx}$, $\Sigma_{yy}$ for the within-view covariances and
$\Sigma_{xy}$ for the cross-covariance, all computed on centred data. For
directions $w_x \in \mathbb{R}^p$ and $w_y \in \mathbb{R}^q$ the correlation of
the variates is

$$\rho(w_x, w_y) = \frac{w_x^{\mathsf T}\Sigma_{xy} w_y}{\sqrt{w_x^{\mathsf T}\Sigma_{xx} w_x}\;\sqrt{w_y^{\mathsf T}\Sigma_{yy} w_y}}. \qquad (1)$$

It is invariant to rescaling either $w$ by a positive constant, so the maximum is
a direction and not a point. Fixing the scale of each:

$$\max_{w_x,\,w_y}\; w_x^{\mathsf T}\Sigma_{xy} w_y \quad \text{subject to} \quad w_x^{\mathsf T}\Sigma_{xx} w_x = 1,\;\; w_y^{\mathsf T}\Sigma_{yy} w_y = 1. \qquad (2)$$

The constraints are not cosmetic. They are $\Sigma$-orthonormality, and they are
what makes the successive solutions orthogonal in the right metric, exactly as
the eigenvectors of a symmetric matrix are orthogonal in the Euclidean one.

### 2. The generalised eigenproblem

The Lagrangian of (2) is
$w_x^{\mathsf T}\Sigma_{xy}w_y - \tfrac{\rho}{2}(w_x^{\mathsf T}\Sigma_{xx}w_x - 1) - \tfrac{\rho}{2}(w_y^{\mathsf T}\Sigma_{yy}w_y - 1)$.
Stationarity gives two equations that must hold simultaneously,

$$\Sigma_{xy} w_y = \rho\,\Sigma_{xx} w_x, \qquad \Sigma_{yx} w_x = \rho\,\Sigma_{yy} w_y. \qquad (3)$$

Eliminating $w_y$ from the second and substituting into the first,

$$\Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}\, w_x = \rho^2\,\Sigma_{xx}\, w_x, \qquad (4)$$

a generalised symmetric eigenproblem with $\min(p,q)$ solutions. The eigenvalues
are the squared canonical correlations, ordered; the eigenvectors are the
canonical directions. It is a *generalised* problem, not an ordinary one, because
the metric on each side is $\Sigma_{xx}$, and it is symmetric because
$\Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}$ is a product of a matrix with its own
transpose in that metric.

Solving (4) directly is a bad idea: $\Sigma_{yy}^{-1}$ destroys symmetry and
conditioning, and the standard dense eigensolver is $O(p^3)$ with no structure
used. There is a two-line route that does better and, more importantly, makes the
geometry obvious.

### 3. Whiten, then SVD

Let $\Sigma_{xx} = L_xL_x^{\mathsf T}$ be a Cholesky factor and define the
whitened views

$$X_w = X_c L_x^{-\mathsf T}, \qquad Y_w = Y_c L_y^{-\mathsf T}, \qquad (5)$$

so that $X_w^{\mathsf T}X_w/(n-1) = I$ and likewise for $Y_w$: every column now
has unit variance and they are uncorrelated. Substituting $w_x = L_x^{-\mathsf T}u$
and $w_y = L_y^{-\mathsf T}v$ into (1) turns the constraints into $\lVert u\rVert =
\lVert v\rVert = 1$ and the objective into

$$\rho = \frac{u^{\mathsf T} C\, v}{\lVert u\rVert\,\lVert v\rVert},
\qquad C = \frac{1}{n-1} X_w^{\mathsf T} Y_w = L_x^{-1}\Sigma_{xy}L_y^{-\mathsf T}. \qquad (6)$$

Maximising a bilinear form over the unit sphere is the definition of the leading
singular pair, so **the canonical correlations are the singular values of $C$**,
the canonical directions are $w_x = L_x^{-\mathsf T}u$, and one SVD returns all
$\min(p,q)$ components at once. No deflation, no iteration, no ordering logic.

This is the route the code takes, and it is the one Golub and Zha recommend: the
Cholesky is $O(p^3/3)$, the SVD of a $p\times q$ matrix is $O(pq\min(p,q))$, and
no covariance is ever inverted. The price is
that $\Sigma_{xx}$ must be positive definite, which it is not when $p \ge n$ —
that is §7, and it is why `_whiten` refuses rather than letting `cholesky` fail on
its own.

### 4. Why whitening makes the invariance visible

Two things fall out of (6) that are hard to see in (4).

**The correlations are properties of the whitened views.** $X_w$ and $Y_w$ are
canonical representatives of their spans: any two bases of the same subspace differ
by an invertible map, and whitening is the map that fixes the representative. So
the answer depends on the row space of $X$ and nothing else. Replacing
$X \mapsto XA + \mathbf{1}b^{\mathsf T}$ gives $X_w \mapsto X_w U$ for some
orthogonal $U$ — a rotation of the canonical coordinates — whose singular values
are unchanged. The demo's $2.8\times10^{-14}$ is that statement executed.

**A ridge is not invariant, and cannot be.** Adding $\lambda I$ to $\Sigma_{xx}$
picks a coordinate system: the ridge penalises directions according to their
variance *in the units you supplied*. It does not commute with a linear map, so
the invariance survives only to the size of the ridge. The demo's gap is
$3\times10^{-10}$ at $\lambda = 10^{-6}$ and $3\times10^{-6}$ at $\lambda =
10^{-4}$ — small, monotone in $\lambda$, and the honest way to report it. Any
claim that regularised CCA is "invariant" is a claim about a problem that has
stopped being CCA.

### 5. Ridge: where to put it and what it does

`_whiten` adds $\lambda$ to the covariance before the Cholesky, so (4) becomes

$$\Sigma_{xy}\left(\Sigma_{yy} + \lambda I\right)^{-1}\Sigma_{yx}\, w_x = \rho^2\left(\Sigma_{xx} + \lambda I\right)w_x. \qquad (7)$$

The important part is what this is *not*. It is not "compute CCA, then shrink the
correlations"; the directions move too. It is not a Bayesian prior with a
posterior; it is a numerical statement that directions of small within-view
variance are not trustworthy, which is the same trade as ridge regression and
has the same justification. For `cca` the ridge is a crutch — $10^{-6}$ is enough
to keep `cholesky` happy on nearly singular data, and small enough that the
invariance gap is $10^{-9}$. For `regularized_cca` it is the model, and the demo's
sweep is the evidence: the number the fit reports and the number it generalises to
are different numbers, and only the second one is worth optimising.

### 6. Kernel CCA: the same problem in a feature space

Map each row to $\varphi(x_i)$ in a feature space $\mathcal{H}$ and ask (2) there.
The solution is a linear combination of the mapped training points — the usual
representer argument — so $w_x = \Phi^{\mathsf T}\alpha$. Substituting, with
$K_x = \Phi\Phi^{\mathsf T}$ the $n\times n$ Gram matrix,

$$w_x^{\mathsf T}\Sigma_{xx}w_x \propto \alpha^{\mathsf T}K_x^2\alpha, \qquad
w_x^{\mathsf T}\Sigma_{xy}w_y \propto \alpha^{\mathsf T}K_xK_y\beta, \qquad (8)$$

and the regularised problem — regularised in the only place the kernel formulation
can be — is

$$\max_{\alpha,\beta}\; \alpha^{\mathsf T}K_xK_y\beta \quad \text{subject to}\quad
\alpha^{\mathsf T}(K_x^2 + \kappa K_x)\alpha = 1,\;\; \beta^{\mathsf T}(K_y^2 + \kappa K_y)\beta = 1. \qquad (9)$$

Now substitute $\alpha = S_x^{-1}u$ with $S_x = (K_x^2 + \kappa K_x)^{1/2}$. The
constraint becomes $\lVert u\rVert = 1$ and the objective becomes
$u^{\mathsf T}F_xF_yv$ with

$$F_x = S_x^{-1}K_x = K_x^{1/2}\left(K_x + \kappa I\right)^{-1/2}, \qquad (10)$$

so the canonical correlations are the singular values of $F_xF_y$, the training
variates are $F_xu$ and $F_yv$, and the dual coefficients are $S_x^{-1}u$. The
code computes exactly (10) and stores $\alpha$, which is why `transform` cannot
use $x\,w$ — see the walkthrough.

The direction of the shrink in (10) is the part that goes wrong. $F_x$ has
eigenvalues $\sqrt{w/(w+\kappa)}$: eigenvalues of $K$ that are large survive,
small ones are driven to zero. The naive reading of "ridge" — add $\kappa$ to the
eigenvalues of $K$ before taking a square root — leaves $n - \mathrm{rank}(K)$
directions of size $\sqrt{\kappa}$, each whitened to unit variance by the next
step, each carrying pure noise. That version trains to a correlation of $1$ and
generalises to nothing; the difference between it and (10) is a factor of
$w/(w+\kappa)$ versus $w + \kappa$ applied to the *same* eigendecomposition, and
it is invisible in the fit's own numbers.

### 7. Why $p + q \ge n$ is fatal, not merely bad

The centred $X$ defines a linear map $\mathbb{R}^p \to \mathbb{R}^n$ whose image
is a subspace of dimension $\min(p, n-1)$. Similarly for $Y$. If the two images
intersect in a non-zero vector, then some $w_x, w_y$ satisfy $X w_x = Y w_y$ and
$\rho = 1$ exactly. Two subspaces of dimensions $p'$ and $q'$ inside an
$(n-1)$-dimensional space must intersect once $p' + q' > n - 1$, i.e. once
$p + q \ge n$ (for $p, q \le n-1$). So the maximum is attained by an algebraic
identity, not by any shared structure.

This is the same phenomenon as $R^2 = 1$ from a regression with more features than
rows, and the demo makes it concrete: $40 \times 20$ of noise plus $40 \times 20$
of noise gives $1.000000$, and $40 \times 10$ gives $0.845$ — not a smaller
effect but the same effect partially diluted. The pattern is entirely the ratio:
at $n = 60$ with 30 columns each it is $1.000000$ again, and with 20 each $0.908$. With $\mathrm{reg} > 0$ the problem is no longer singular, and the
ridge at $10^{-8}$ is enough to reproduce the same $1.000000$, which is worth
knowing: *a tiny ridge does not protect you here*, because the ridge changes the
answer by less than the sampling error it is supposed to fix.

### 8. Sparse CCA: an exact lasso half-step

Write the penalised problem for one component as a two-block maximisation with
$\ell_1$ budgets, and hold $v$ fixed. The problem in $u$ is

$$\min_{u}\; \tfrac{1}{2}u^{\mathsf T}\Sigma_{xx}u - u^{\mathsf T}\Sigma_{xy}v + \lambda_u\lVert u\rVert_1 \qquad (11)$$

followed by a rescaling to $u^{\mathsf T}\Sigma_{xx}u = 1$. At $\lambda_u = 0$
the minimiser is $u = \Sigma_{xx}^{-1}\Sigma_{xy}v$, which is precisely the
stationarity equation (3) with $v$ fixed — that is why $\lambda = 0$ reproduces
`cca` and why the alternating scheme is CCA's power iteration rather than a
different algorithm.

(11) is a lasso with a $p \times p$ Gram matrix, so cyclic coordinate descent is
exact and needs no step size. Coordinate $j$, with $S = \Sigma_{xx}$:

$$u_j \leftarrow \frac{1}{S_{jj}}\,\mathcal{S}\!\left(c_j - \sum_{k\ne j} S_{jk}u_k,\;\lambda_u\right),
\qquad c = \Sigma_{xy}v, \qquad (12)$$

where $\mathcal{S}(z,\lambda) = \mathrm{sign}(z)\max(|z| - \lambda, 0)$. The
optimality conditions are worth stating because they are what "sparse" means
here:

$$u_j \ne 0 \;\Rightarrow\; (Su)_j - c_j = -\lambda_u\,\mathrm{sign}(u_j), \qquad
u_j = 0 \;\Rightarrow\; |(Su)_j - c_j| \le \lambda_u. \qquad (13)$$

A coordinate is zero exactly when its partial residual is smaller than the
penalty. Two consequences the demo measures: the support is a property of the
*standardised* coordinates (the code divides each column by its standard deviation
before the solve, so $\lambda$ means the same thing across features of different
units), and $\lambda$ has to be read against the scale of $c$, which grows with
$n$ — $0.02$ is an interesting penalty at $n = 800$ and $0.2$ annihilates the
solution, while at $n = 400$ the interesting range is $0.01$–$0.05$. There is no
scale-free $\lambda$.

The deflation step is what makes the second component meaningful: after
convergence the code subtracts $\hat\rho\,\hat u\hat v^{\mathsf T}$ from the
cross-covariance, which removes the found pair from the problem and leaves the
next one. This is greedy — the components are not jointly optimal, and the
support of component 2 depends on what component 1 found — which is true of every
sparse multi-component method in use.

### 9. More than two views: SUMCOR, and why it collapses to CCA at $D = 2$

For $D$ views there is no single "the" canonical correlation, because there is no
single pair. Kettenring's 1971 survey lists five criteria — SUMCOR, MAXVAR,
SSQCOR, MINVAR and GENVAR — which differ in what they do with the $D$ block
correlations; all five agree at $D = 2$ and disagree above it. This file
implements SUMCOR: one shared variate $z$, and the sum of the correlations
between $z$ and each view's variate maximised,

$$\max_{z,\,w_1,\dots,w_D}\; \sum_{i=1}^{D} \mathrm{corr}(z,\, X_iw_i)
\quad\text{subject to}\quad w_i^{\mathsf T}\Sigma_{ii}w_i = 1. \qquad (14)$$

Given $z$, each $w_i$ is a regression in the $\Sigma_{ii}$ metric,
$w_i \propto \Sigma_{ii}^{-1}X_i^{\mathsf T}z$ rescaled to the constraint; given
the $w_i$, the best $z$ is the mean of the unit-variance variates. Both steps are
exact, so the loop is coordinate ascent on (14), and at $D=2$ its fixed point is
(3) — hence the $10^{-13}$ agreement with `cca`.

The reported `correlations_` is the mean pairwise correlation of the fitted
variates, which is not (14)'s value but is a function of it. With $\lVert z\rVert
= 1$ the objective is $\sum_i z^{\mathsf T}s_i/\lVert s_i\rVert$, maximised by
$z \propto \sum_i s_i/\lVert s_i\rVert$, so

$$\sum_i \mathrm{corr}(z, s_i) = \Bigl\lVert \sum_i s_i/\lVert s_i\rVert \Bigr\rVert
= \sqrt{D + D(D-1)\,J},$$

where $J$ is the mean pairwise correlation the fit reports. The demo's three-view
fit satisfies the identity to six decimals, $2.318165$ on both sides: the number
being maximised and the number being reported determine each other, so nothing is
lost by reporting the mean.

The deflation removes $z$ from every view at once, $C_i \leftarrow C_i -
z(z^{\mathsf T}C_i)$, which is what keeps component $c+1$ orthogonal to component
$c$ in all views simultaneously.

(14) also has a one-step solution: concatenate the views, let $R$ be the
covariance and $D$ its block diagonal, and take the leading eigenvector of
$D^{-1}R$. The demo computes it and checks the loop against it, block by block:
cosine $0.994$, $0.995$ and $1.000$. The directions are what match — the
eigenvector normalises the blocks jointly while the loop gives each unit
variance — and the disagreement in the third digit is that difference in scaling,
not an optimisation gap.

SUMCOR's weakness is the one the demo measures: the objective is an average over
views, so a view that shares nothing drags the joint number down without being
ignored ($0.40$ against $0.71$ for the good pair). MAXVAR or SSQCOR would weight
it differently, and none is "correct" — the choice is a modelling statement about
whether a bad view is noise to be averaged out or a view to be discarded.

## Code walkthrough

`CCA(n_components=1, *, variant="cca", reg=1e-6, lam_u=0.0, lam_v=0.0,
kernel="linear", gamma=None, n_iter=300, tol=1e-9, seed=0)`, with `fit`,
`transform`, `fit_transform`, `score`, and the attributes `weights_` (a list of
`(p_i, k)` blocks — `(n, k)` for a kernel fit, where they are dual
coefficients), `correlations_` $(k,)$, `mean_`, `n_iter_`.

`fit(X, Y)` for two views, `fit([X1, X2, ...])` for `multi_view`. `weights_` is a
list even for two views because a kernel fit stores dual coefficients rather than
primal weights, and because `multi_view` has one block per view; making the shape
uniform removes a special case from every caller.

### 1. Whitening, and the one refusal that matters

```python
    n, p = Xc.shape
    cov = Xc.T @ Xc / max(n - 1, 1)
    if reg <= 0.0 and p > n - 1:
        raise ValueError(
            f"a view with {p} columns and {n} rows has a singular covariance; "
            f"unregularised CCA needs n > p — use regularized_cca with reg > 0"
        )
    L = np.linalg.cholesky(cov + reg * np.eye(p))
    Xw = np.linalg.solve(L, Xc.T).T                    # n × p, XwᵀXw ≈ I
    return L, Xw
```

(5). `np.linalg.solve(L, Xc.T).T` computes $X_cL^{-\mathsf T}$ without forming an
inverse — the solve is triangular and costs $O(np^2)$, the same as the covariance
it follows. The alternative, `Xc @ inv(L).T`, is the same arithmetic with an
explicit inverse and worse conditioning; it is the first thing to fix in a CCA
implementation that "works on small data".

The refusal is deliberate and is not a `try/except` around `cholesky`. `cholesky`
fails only when the matrix is numerically indefinite, which for `cov + 0·I` on
wide data is not guaranteed — a $40\times60$ covariance has rank 39 in exact
arithmetic but rounding can leave a tiny positive eigenvalue, and then the code
proceeds with a whitening that amplifies rounding into the answer. The explicit
test states the condition the mathematics requires, and the message names the
remedy, which is the only useful thing an exception can do.

### 2. The closed form, in six lines

```python
            Lx, Xw = _whiten(m, self.reg)
            Ly, Yw = _whiten(n, self.reg)
            P, s, Qt = np.linalg.svd(Xw.T @ Yw / max(X.shape[0] - 1, 1))
            k = self.n_components
            wx = np.linalg.solve(Lx.T, P[:, :k])
            wy = np.linalg.solve(Ly.T, Qt.T[:, :k])
            rho = s[:k]
```

(6), with every piece in the order the derivation gives it. `Xw.T @ Yw` is $C$ up
to the $n-1$; `svd` returns $P$, $s$, $Q^{\mathsf T}$ with $s$ sorted descending,
so `[:k]` is "the first $k$ canonical components" with no further sorting. The
weights are $w = L^{-\mathsf T}u$, i.e. `solve(Lx.T, u)`: `Lx.T` is upper
triangular, so this is a back-substitution and not a matrix inversion.

`Qt.T[:, :k]` deserves a second look: `np.linalg.svd` returns $Q^{\mathsf T}$, so
the right singular vectors are the *columns* of `Qt.T`, which is why the slice is
`Qt.T[:, :k]` and not `Qt[:k]`. Writing `Qt[:k]` transposes the answer and gives a
fit that is wrong in a way that still runs — one of the two classic CCA bugs, the
other being $L^{-1}$ instead of $L^{-\mathsf T}$.

### 3. Measuring the correlation rather than trusting the SVD

```python
        if U.shape[0] < 3:
            return np.asarray(fallback, dtype=float)
        return np.array([abs(float(np.corrcoef(U[:, j], V[:, j])[0, 1]))
                         for j in range(U.shape[1])])
```

The singular values of $C$ are the canonical correlations only when (5)'s
whitening is exact, and with a ridge it is not. The discrepancy is the size of the
ridge, so trusting `s` makes the primal path report a correlation that is
$8.7\times10^{-7}$ away from the variates' own at $\mathrm{reg}=10^{-6}$, and
$8.7\times10^{-5}$ at $10^{-4}$ — and it is enough to make two variants that
should agree to machine precision look like different estimators. Measuring the
correlation of the fitted variates costs two matrix products and makes every
variant comparable with every other by construction. This is also why
`correlations_` is always in-sample: it is the thing that was maximised, and
`score` is the only honest number in the file.

The `U.shape[0] < 3` branch exists because `transform` can be called on a single
row, where `corrcoef` is undefined; it falls back to the singular value, which is
the right limit.

### 4. The lasso half-step

```python
    u = np.zeros(S.shape[0])
    d = np.diag(S)
    for _ in range(500):
        u_old = u.copy()
        for j in range(S.shape[0]):
            partial = c[j] - (S[j] @ u - d[j] * u[j])
            u[j] = _soft(partial, lam) / d[j]
        if float(np.linalg.norm(u - u_old)) < 1e-12 * (1.0 + float(np.linalg.norm(u_old))):
            break
    if np.all(u == 0.0):
        raise ValueError("every coordinate was thresholded away — λ is too large")
    return u
```

(12), and the reason sparse CCA is implementable without a convex solver. `S[j] @
u` is the full matrix-vector product recomputed for every coordinate, which is
$O(p)$ per coordinate and $O(p^2)$ per sweep — the naive form, and the right one
at the sizes this file handles. The standard optimisation is to maintain $Su$
incrementally: subtract the old contribution and add the new, $O(1)$ per
coordinate. That is a factor of $p$ and it is what a production implementation
does; it is also six more lines and one more place to introduce a bug that
converges to the wrong answer, which is why it is not here.

`partial` is the partial residual $c_j - \sum_{k\ne j}S_{jk}u_k$, written as
`c[j] - (S[j] @ u - d[j] * u[j])` because the diagonal term is already included in
`S[j] @ u`. `d = np.diag(S)` is hoisted out of the loop and is safe: `S` has a
ridge on its diagonal, so $S_{jj} > 0$ always and the division cannot blow up.

The convergence test is relative to `1 + ‖u_old‖` rather than absolute, because
the scale of $u$ is problem-dependent and `S⁻¹c` can be large. And the all-zero
exit is a *feature*: at large $\lambda$ the true minimiser is exactly $0$, and
returning it would make `_unit_in` divide by zero and produce a NaN correlation
that propagates silently. Raising names the cause.

### 5. The alternating loop, and the constraint everyone forgets

```python
            for _ in range(self.n_iter):
                u = _unit_in(Sxx, _lasso(Sxx, Sxy @ v, self.lam_u))
                v_new = _unit_in(Syy, _lasso(Syy, Sxy.T @ u, self.lam_v))
                self.n_iter_ += 1
                if float(np.linalg.norm(v_new - v)) < self.tol:
                    v = v_new
                    break
                v = v_new
            u = _unit_in(Sxx, _lasso(Sxx, Sxy @ v, self.lam_u))
```

(11) twice per iteration. Each half-step solves its lasso and then rescales so
that $u^{\mathsf T}\Sigma_{xx}u = 1$:

```python
    return u / math.sqrt(max(float(u @ S @ u), EPS))
```

Without it both vectors shrink geometrically. On the demo's data the norms fall
from $0.21$ to $0.037$ in six steps at $\lambda = 0$, and at $\lambda = 0.02$ they
reach zero and the lasso raises. The correlation is scale-free, so the *direction*
survives for a while — what breaks first is the deflation, $\Sigma_{xy} - \rho\,
uv^{\mathsf T}$, which is scaled by the same disappearing factor: component 2
comes out as a copy of component 1, and the fit reports two components with the
same correlation and no warning.

The final `u` after the loop is not redundant. The loop breaks when `v` stops
moving, at which point `u` was computed from the *previous* `v`; recomputing it
from the final one makes the pair consistent — at `tol` when the loop converged,
and at the last iterate when it hit `n_iter`. It also means the returned pair
always satisfies both stationarity equations to the same tolerance, which is what
makes the reported support a property of a fixed point rather than of a
half-step.

### 6. Kernel CCA, and the two matrices the derivation produces

```python
    K = 0.5 * (K + K.T)
    w, U = np.linalg.eigh(K)
    w = np.maximum(w, 0.0)
    F = (U * np.sqrt(w / (w + reg))) @ U.T
    S = (U * np.sqrt(w * (w + reg))) @ U.T
    return F, S
```

(10), computed as a symmetric eigendecomposition rather than as a matrix square
root. `0.5 * (K + K.T)` symmetrises: a Gram matrix of an RBF kernel is positive
definite in exact arithmetic and routinely has small negative eigenvalues in
floating point, and `sqrt` of a negative eigenvalue is `nan` three functions
later. `np.maximum(w, 0.0)` clips them.

Both $F$ and $S$ are returned because the two are used in different places and
both are needed: $F$ for the correlation and the training variate, $S$ for the
dual coefficients. Computing $F$ as `(U * np.sqrt(w / (w + reg))) @ U.T` rather
than as `sqrtm` is exact and cheap — the eigendecomposition is already in hand —
and it is the only place the shrink direction is expressed. Reverse the ratio to
`(w + reg) / w` and the kernel becomes a noise amplifier; use `w + reg` under the
root and you get the `n − rank` noise dimensions of the derivation's §6.

### 7. The kernel path

```python
        Fx, Sx = _kcca_factors(_kernel(self._train_[0], self._train_[0],
                                       self.kernel, self._gammas[0]), self.reg)
        Fy, Sy = _kcca_factors(_kernel(self._train_[1], self._train_[1],
                                       self.kernel, self._gammas[1]), self.reg)
        P, s, Qt = np.linalg.svd(Fx @ Fy)
        k = self.n_components
        u, v = P[:, :k], Qt.T[:, :k]
        self.weights_ = [np.linalg.solve(Sx, u), np.linalg.solve(Sy, v)]
        self.correlations_ = self._corr_of(Fx @ u, Fy @ v, s[:k])
```

This is the primal §2 code with $X_w$ replaced by $F$ and $L$ replaced by $S$,
which is the point of (10): once the constraint is in $F$, no whitening is needed.
`np.linalg.solve(Sx, u)` is $S_x^{-1}u$, the dual coefficients — note `solve(Sx,
u)` and not `solve(Sx.T, u)`, because $S$ is symmetric and the transpose
distinction that matters in the primal path has collapsed.

The variates passed to `_corr_of` are $F_xu$ and $F_yv$, *not* $K_x\alpha$ and
$K_y\beta$. They are equal in exact arithmetic ($K\alpha = KS^{-1}u = Fu$) and
differ in floating point; using $F$ is cheaper and keeps the correlation a
property of the same objects the SVD saw.

### 8. Inference through the dual

```python
        Xc = X - self.mean_[view]
        if self._kernel:
            # in kernel space the weights are dual coefficients, so the variate is
            # `k(x, X_train) α` rather than `x w`
            K = _kernel(Xc, self._train_[view], self.kernel, self._gammas[view])
            return K @ self.weights_[view]
        return Xc @ self.weights_[view]
```

The one place a kernel CCA differs from a linear one at prediction time. `w` lives
in the $n$-dimensional feature space, so `X @ w` is a shape error; the projection
of a new point is $\varphi(x)^{\mathsf T}w = k(x, X_{\text{train}})\alpha$, an
$n$-vector times the stored coefficients. Getting this wrong — storing $F^{-1}u$
instead of $S^{-1}u$ — gives a *training* correlation that is correct and a
*held-out* correlation that is garbage, which is exactly the failure the demo's
harmonics section would show as "kernel CCA overfits" rather than "kernel CCA is
implemented wrong".

Centring is the other trap: `Xc` subtracts the training mean, and the Gram matrix
is built on centred rows, which is not the same as centring the Gram matrix. For a
linear kernel the two agree up to a rank-one correction; for an RBF kernel they do
not agree at all, and the code's choice (centre the features, then kernelise) is
the one consistent with the primal derivation.

### 9. Many views, one variate

```python
        for c in range(self.n_components):
            z = _unit(rng.standard_normal(n))
            for it in range(self.n_iter):
                scores = []
                for i, C in enumerate(centred):
                    w = _canonical_weight(C, covs[i], z)
                    scores.append(_unit(C @ w))
                z_new = _unit(np.mean(scores, axis=0))
                self.n_iter_ = it + 1
                if float(np.linalg.norm(z_new - z)) < self.tol:
                    z = z_new
                    break
                z = z_new
```

(14), alternating. Given $z$, each view's best direction is a $\Sigma$-metric
regression solved in closed form:

```python
    w = np.linalg.solve(cov, C.T @ z)
    return w / math.sqrt(max(float(w @ cov @ w), EPS))
```

Given the directions, the best shared variate is their mean after unit-scaling —
`_unit(C @ w)` per view and `np.mean` across views, then `_unit` again. The
per-view `_unit` is redundant at the fixed point, where the constraint already
gives every variate unit variance; it matters in the first few iterations, when
the weights are still far from satisfying it and one view would otherwise
dominate the average by scale alone.

The demo checks the loop against the one-step solution of (14), which it computes
in ten lines:

```python
    X = np.hstack([_centre(V) for V in views])
    R = X.T @ X / (X.shape[0] - 1)
    D = np.zeros_like(R)
    start = 0
    for V in views:
        k = V.shape[1]
        D[start:start + k, start:start + k] = R[start:start + k, start:start + k]
        start += k
    vals, vecs = np.linalg.eig(np.linalg.solve(D, R))
    lead = vecs[:, int(np.argmax(vals.real))].real
```

`np.linalg.eig` and not `eigh`: `solve(D, R)` is not symmetric, and `eigh` reads
one triangle of whatever it is given and returns a plausible answer — it gave
cosine $0.94$ against the loop where the correct call gives $0.994$. That is the
kind of bug that survives review, because the numbers it produces are close.

The initialisation is a random unit vector from `seed`, which is the only
stochastic ingredient in the whole file. It matters only for sign and for which
component comes out first when two are nearly tied; with the deflation in place
the components are determined up to sign, and `_cos` and `score` both take
absolute values for that reason.

### 10. What gets stored, and what "correlation" is measured on

```python
        U, V = self.transform(X, 0), self.transform(Y, 1)
        return np.array([abs(float(np.corrcoef(U[:, j], V[:, j])[0, 1]))
                         for j in range(self.correlations_.shape[0])])
```

`score` is the held-out twin of `correlations_`, and the reason every number in
this file's demo is reported as a pair. `transform` centres with the *training*
mean and applies the *training* weights, so `score(X, Y)` on new data measures the
variates the fit would actually produce — which is the only quantity a
generalisation claim can be about. The `abs` is because canonical directions are
defined up to sign; a caller who needs the sign should look at the weights, not at
the correlation.

### 11. The planted problem

```python
    sigma = 0.0 if rho <= 0.0 else math.sqrt((1.0 - rho) / rho)
    X = np.outer(z, wx) + sigma * rng.standard_normal((n, p))
    Y = np.outer(z, wy) + sigma * rng.standard_normal((n, q))
    return X, Y, wx, wy
```

The demo's generator, and worth reading as the model CCA assumes: one latent $z$,
a sparse loading in each view, independent noise. The noise scale solves
$\rho = 1/(1 + \sigma^2)$ for $\sigma$, so the *population* canonical
correlation is $\rho$ by construction and a fit can be scored against a number
that was not itself estimated. `nnz=5` of 20 columns is what makes sparse CCA's
support recovery a meaningful measurement rather than a tautology.

## Complexity

$n$ rows, $p$ and $q$ columns, $D$ views, $k$ components, $T$ outer iterations.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_centre` + covariances | $O(n(p^2+q^2))$ | $O(n(p+q))$ | dominates when $n \gg p$ |
| `_whiten` | $O(np^2 + p^3)$ | $O(np)$ | Cholesky plus a triangular solve |
| `svd(Xwᵀ Yw)` | $O(pq\min(p,q))$ | $O(pq)$ | all $k$ components at once |
| `_fit_pair` total | $O(n(p^2+q^2) + p^3 + q^3)$ | $O(n(p+q))$ | $1$ ms at $n=800$, $p=q=30$ |
| `_lasso` | $O(p^2)$ per sweep | $O(p^2)$ | naive; $O(p)$ per coordinate update |
| `_sparse_pair` | $O(T(p^2+q^2))$ | $O(p^2+q^2)$ | $T \approx 10$–$20$ here |
| `_kernel` (RBF) | $O(n^2(p+q))$ | $O(n^2)$ | the Gram matrices |
| `eigh` of the Grams | $O(n^3)$ | $O(n^2)$ | why kernel CCA is $0.30$ s at $n=800$ |
| `_fit_multi` | $O\big(T\sum_i (np_i + p_i^3)\big)$ | $O(nD)$ | 71 iterations on the demo's 3 views |
| `transform` (kernel) | $O(n_{\text{new}}\,n)$ | $O(n_{\text{new}}n)$ | one kernel row per point |
| `score` | $O(n(p+q))$ | $O(n)$ | two matvecs per component |

The row that matters is the first against the second: the covariance products are
$O(n(p^2+q^2))$ and the factorisation is $O(p^3)$, so CCA is *linear in the number
of rows* once the columns are fixed. That is why the closed form scales to
$n = 10^6$ with $p = q = 50$ and why the ridge sweep in the demo is cheap enough
to run three times: $n$ enters only through a matrix product.

Kernel CCA is the opposite: $O(n^3)$ time and $O(n^2)$ memory in the number of
*rows*, which is the entire capacity story. The demo's 100-training-point fit takes
milliseconds; the $n = 800$ timing above is $0.30$ s; at $n = 10^4$ the Gram matrix
alone is 800 MB and the eigendecomposition is hours. Every practical kernel CCA
uses a low-rank approximation of $K$ (Nyström, random features, or a truncated
eigendecomposition) and pays for it in exactly the accuracy the demo's harmonics
experiment measures.

`_sparse_pair` is the one place the naive implementation is asymptotically worse
than it needs to be: `S[j] @ u` inside the coordinate loop is $O(p)$ per
coordinate, so a sweep is $O(p^2)$, whereas maintaining $Su$ incrementally is
$O(p)$ per coordinate and $O(p)$ per sweep. The demo's $p = 30$ makes the
difference invisible — the whole sparse fit is $30$ ms — and the fix is a standard
exercise that is not here, for the reason §4 of the walkthrough gives.

## Failure modes

**1. $p + q \ge n$ and the correlation is $1$ by algebra.** §7, measured at
$1.000000$ on two independent $40\times20$ noise matrices. This is the failure that
matters because it is *silent*: the fit reports a perfect correlation, the weights
are large and dense, and there is no warning. The explicit refusal in `_whiten`
catches the $\mathrm{reg} = 0$ case, but with the default $10^{-6}$ it does not
fire and you get the same $1.000000$ — the check that matters is on $p + q$ against
$n$ in the pipeline, and the honest response to a genuinely wide problem is
`regularized_cca`.

**2. A small ridge does not fix the wide-data problem.** At $n=40$, $p=q=20$, a
ridge of $10^{-8}$ still returns $1.000000$, because $10^{-8}$ changes the answer by
less than the estimation error it is meant to control. The ridge has to be large
enough to be a model — the demo's sweep needs $\lambda \approx 1$, the same order
as the column variances themselves — and choosing it is a model-selection problem,
not a numerical-hygiene setting.

**3. The ridge breaks the invariance, in proportion to the ridge.** $2.8\times
10^{-14}$ at $\lambda = 10^{-8}$, $3\times10^{-6}$ at $10^{-4}$. If a pipeline
depends on "CCA doesn't care about the units", it stops being true the moment a
ridge is added, and the failure is a slow drift rather than an error.

**4. The weights are not identified, so a coefficient is not an effect.** Anything
in the span is an equally valid answer; the fitted $w_x$ is one representative, and
a linear map of $X$ produces a different representative with the same
correlation. Reporting "gene 412 has weight 0.4" is meaningful only because gene
412 is a coordinate someone chose, and averaging weights across fits is averaging
different bases.

**5. Sparse CCA's support is a coordinate statement, and its $\lambda$ is not
scale-free.** §8: the code standardises columns so $\lambda$ is comparable across
them, but the support is still "these coordinates, in this basis". The demo's
$\lambda = 0.02$ is interesting at $n=800$ and collapses the solution at
$\lambda = 0.2$; the same $\lambda$ on $n=400$ data would be a different model.
Read $\lambda$ as a fraction of the scale of $X^{\mathsf T}Y$, or tune it by
held-out correlation, and do not reuse a value across data sets.

**6. Sparse CCA is greedy and non-convex.** One component at a time, deflated from
the residual cross-covariance: component 2's support depends on component 1's, and
a different random start (`seed`) can land in a different local optimum. The demo
uses one component for exactly this reason, and any multi-component sparse fit
should report the objective across seeds rather than one number.

**7. Kernel CCA has $n$ parameters and $n$ samples.** The complexity table: the
model can fit any $n$ points, so the ridge and the bandwidth are the model, and the
median heuristic is a starting point that can be beaten badly. On the demo's
near-deterministic harmonics, the median-heuristic bandwidth gives $0.79$ held out
while $20\times$ that bandwidth gives $0.95$ — the kernel wants to interpolate
because the relation really is a function, and the heuristic is tuned for noisier
data. There is no default that is right for both.

**8. The kernel ridge must scale with $n$ and with the kernel.** $\kappa$ is added
to the eigenvalues of $K$, and those eigenvalues scale like $n$ times the variance
in that direction. So $\kappa = 10^{-3}$ is a strong regulariser at $n = 100$ and
a weak one at $n = 10^5$, and a pipeline that fixes $\kappa$ and changes the sample
size has silently changed its model.

**9. More than two views: "the" canonical correlation is an average, and the
choice of average is a model.** SUMCOR's joint number is the mean over pairwise
agreement, so a view sharing nothing costs a third of the score without being
discarded ($0.40$ against $0.71$). MAXVAR, SSQCOR or a view-selection step is a
different answer to a different question. With $D=2$ every criterion agrees, which is why
this file's two-view path is exact and its three-view path is one of five
defensible choices.

**10. High canonical correlation is not evidence of a shared cause.** Two views
can correlate at $0.9$ because of a confound measured in both — batch, age, site —
and CCA will report the direction that encodes it, with a sparse support and a
convincing plot. This is the same failure as a spurious regression, and it is worse
here because the direction is chosen to maximise the correlation: there is always
a direction, because the fit is searching over $p + q$ coefficients for a single
number.

**11. `correlations_` is in-sample and the variates are linear.** The reported
number is what the fit maximised, so it is upward-biased by construction; the demo
reports every result as train/held-out for that reason. And the whole file is
linear in the rows of each view after centring: a relation that is quadratic in
the features needs `kernel_cca`, and a relation that is linear only after a
transformation of the *response* is not this atom at all.

## Applications

- **Cross-modal retrieval and matching.** Image features and text features of the
  same items, projected to a shared space where the paired embeddings correlate.
  CCA is the linear ancestor of every contrastive multi-modal model; the difference
  is that a neural version learns the features and CCA takes them as given.
- **Multi-omics integration.** Gene expression, methylation, proteomics on the same
  samples: the first canonical variate is a candidate shared pathway, and the
  sparse variant is used precisely because a pathway is a short list of genes.
  This is where `sparse_cca`'s coordinate-dependence bites hardest and where the
  biology supplies the coordinates that make it meaningful.
- **Comparing neural representations.** SVCCA and PWCCA use CCA between the
  activations of two layers or two networks to measure how similar their
  representations are, with the invariance of §4 as the justification: the
  comparison must not depend on the basis the network happened to learn. This is
  the most active current use of the atom.
- **A test statistic, not just an estimator.** Wilks' $\Lambda$ and Bartlett's
  approximation turn the canonical correlations into a likelihood-ratio test of
  "are these two sets independent", which is the classical multivariate-statistics
  use and still the right one when the question is confirmatory rather than
  predictive.
- **EEG, audio and sensor fusion.** Two recordings of the same event, one per
  modality; the canonical variate is the shared component and the residual is the
  modality-specific one, which is a cheap alternative to ICA when the two views
  are known to be time-aligned.
- **As a lens on other atoms.** CCA with class-indicator columns on $Y$ is Fisher's
  linear discriminant; CCA between $X$ and $Y$ with the covariance replacing the
  correlation is PLS; CCA restricted to one view is PCA. Reading those three
  identities off (2) is the fastest way to see what each of them is actually
  optimising.
- **Seismic and remote sensing multiset analysis.** Nielsen's multiset CCA is the
  standard tool for fusing many co-registered image channels, and it is the
  practical reason the multi-view formulation exists rather than being a
  generalisation for its own sake.

Where it is a trap: fitting CCA with more columns than rows and reporting the
correlation; adding a ridge and still claiming invariance; reading a weight as an
effect; choosing $\lambda$ or $\kappa$ on the training correlation; and treating a
high canonical correlation as evidence that the two views measure the same thing
rather than that they share a confound.

## References

- **Hotelling (1936),** *Relations between two sets of variates* — the original
  paper, and still the clearest statement of (1)–(2). Everything else in this
  literature is a numerical method or a regulariser for this problem.
- **Golub & Zha (1995),** *The canonical correlations of matrix pairs and their
  numerical computation* — why the Cholesky-then-SVD route of §3 is the stable
  one, and the analysis of what the generalised eigenproblem does to conditioning.
- **Kettenring (1971),** *Canonical analysis of several sets of variables* — the
  five multi-set criteria of §9, and the proof that they coincide only at $D=2$.
- **Vinod (1976),** *Canonical ridge and econometrics of joint production* — the
  ridge of §5, including the observation that it is not invariant under rescaling
  of the views.
- **Hardoon, Szedmak & Shawe-Taylor (2004),** *Canonical correlation analysis: an
  overview with application to learning methods* — the standard reference for the
  kernel formulation (8)–(10), and the source of the regularisation used here.
- **Bach & Jordan (2002),** *Kernel independent component analysis* — the
  relationship between kernel CCA and dependence measures, and the reason
  kernel CCA with a small ridge can report correlation $1$ on dependent data.
- **Witten, Tibshirani & Hastie (2009),** *A penalized matrix decomposition, with
  applications to sparse principal components and canonical correlation analysis*
  — the PMD formulation of §8, including the alternating lasso structure and the
  deflation.
- **Witten & Tibshirani (2009),** *Extensions of sparse canonical correlation
  analysis with applications to genomic data* — the variant that penalises one
  side only, and the discussion of why the support is coordinate-dependent.
- **Parkhomenko, Tritchler & Beyene (2009),** *Sparse canonical correlation
  analysis with application to genomic data integration* — the alternating
  soft-thresholding form, and the empirical comparison that motivates exact lasso
  steps over threshold-then-rescale.
- **Nielsen (2002),** *Multiset canonical correlations analysis and multispectral,
  truly multitemporal remote sensing data* — the multiset criteria in practice,
  with the comparison of SUMCOR, SSQCOR and MAXVAR that makes clear they are
  different answers, and the deflation scheme of §9.
- **Tenenhaus & Tenenhaus (2011),** *Regularized generalized canonical
  correlation analysis* — RGCCA, the modern unification of the multi-set
  criteria with per-view penalties; the right place to look when SUMCOR is not
  the answer you want.
- **Parra (2018),** *Multiset canonical correlation analysis simply explained* —
  the one-step SUMCOR solution of §9, its derivation from $R\mathbf{v} =
  D\mathbf{v}\lambda$, and the two-step reading (whiten each block, then PCA the
  concatenation) that makes the criterion intuitive.
- **Raghu, Gilmer, Yosinski & Sohl-Dickstein (2017),** *SVCCA: singular vector
  canonical correlation analysis for deep learning dynamics and
  interpretability* — the invariance of §4 used as a measurement tool, with the
  PCA pre-processing that makes it practical at layer width.
- **Morcos, Raghu & Bengio (2018),** *Insights on representational similarity in
  neural networks with canonical correlation* — PWCCA, the weighted variant that
  fixes SVCCA's sensitivity to low-variance directions.
