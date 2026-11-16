# Classical MDS

`mds.py` · Manifolds & embeddings · Tier 2 · 5 variants

## Intuition

You are given numbers that say how far apart things are, and nothing else. Not
coordinates, not directions, not a metric space — a table of dissimilarities, one
per pair. Market research asks respondents how similar two brands feel; ecology
counts species and computes Bray–Curtis between sites; genetics has distances
between sequences; a graph has shortest-path lengths. In every case the question
is the same: *is there a configuration of points whose distances look like this
table?*

```
     δ (a table of numbers)            X (a plot)
      ┌───────────────┐              y│      ·   ·
   b  │ 0   .   .   . │               │   ·    ·
   r  │ .   0   .   . │     ──▶       │      ·    ·
   a  │ .   .   0   . │               │  ·      ·
   n  │ .   .   .   0 │               └────────────── x
      └───────────────┘
```

MDS is the answer, and the answer is not unique: any rotation, reflection or
translation of the configuration has the same distances. That is a feature, not a
defect — it means distances are the data and coordinates are a choice of picture.

What the picture should preserve is where the design space opens. The standard
objective is **stress**,

$$\sigma(X) \;=\; \sum_{i<j} w_{ij}\big(\lVert x_i - x_j\rVert - \delta_{ij}\big)^2, \qquad (1)$$

the summed squared disagreement between the embedding's distances and the
targets. It is not convex, it has many local minima, and every variant in this
file is a different answer to which $\delta$ should be matched and how.

There are five, and they differ more than the name suggests. **Classical** MDS
ignores (1) entirely: it minimises a linear surrogate with a closed form, no seed
and no local optima, and the demo recovers a 40-point planar configuration from
its distances to a Gram error of $1.1\times10^{-15}$. **SMACOF** minimises (1) by
majorisation and cannot go uphill. **Metric** minimises (1) by gradient descent
with a line search and cannot promise that. **Nonmetric** matches only the *order*
of the $\delta$, which on the demo's cubed distances recovers the geometry at
correlation $1.0000$ where metric MDS gets $0.9022$. **Weighted** matches a subset
of the pairs, and recovers a 40-point configuration from 230 of its 780
distances — where filling the missing entries with the mean gives a Gram error of
$0.8670$.

## Variants

| Variant | Objective | Algorithm | Cost |
|:--|:--|:--|:--|
| `classical` | strain, $\lVert B - XX^{\mathsf T}\rVert^2$ | double centre, eigendecompose | $O(n^3)$ once |
| `smacof` | (1) | majorisation, monotone | $O(n^3)$ per sweep |
| `metric` | (1) | gradient descent + line search | $O(n^2 k)$ per step |
| `nonmetric` | Kruskal's stress-1 | isotonic regression + SMACOF | $O(n^2\log n)$ per sweep |
| `weighted` | (1) with $w_{ij}=0$ on missing | SMACOF with a weight matrix | $O(n^3)$ per sweep |

**`classical`** is Torgerson scaling. Square the distances, double-centre them,
take the top eigenvectors, and the coordinates fall out (§2). It minimises
**strain**, a different objective from stress, and the two coincide exactly when
the dissimilarities are Euclidean distances of points in the requested dimension.
That is a strong assumption and it is why Isomap computes geodesic distances and
*then* calls this: on a manifold, the ambient Euclidean distances are not the
right $\delta$, the graph distances are.

**`smacof`** is the workhorse. Each sweep replaces the non-convex problem with a
quadratic that touches it at the current point and lies above it everywhere, then
minimises that exactly. The result is the Guttman transform, and the stress cannot
increase. On the demo's 50 noisy distances it takes 131 sweeps with **zero**
uphill steps to reach $0.004864$, which is $27.5\%$ below the closed form's
$0.006707$ — the closed form is not minimising the thing the iteration does.

**`metric`** is the same objective by a different route: the analytic gradient of
(1), a step, and a backtracking line search that halves until the stress falls.
It is here because SMACOF's monotonicity is bought with a specific update that
ignores the geometry of the problem, and descent is the alternative that uses it.
The demo's honest result is that descent reaches the same stress to $2\times
10^{-10}$ in **fewer** steps — 33 against 131 — because a line search is
aggressive when it works. The difference is the guarantee, not the speed:
majorisation cannot go uphill for any weights and any dissimilarity, while
descent is monotone only while its search succeeds.

**`nonmetric`** is Kruskal's ordinal MDS. It assumes only that the numbers are
*ordered*, not that they are on a ratio scale — which is what a Likert score or a
similarity judgement actually gives you. Each sweep regresses the current
distances onto the order of the $\delta$ by isotonic regression (pool adjacent
violators, §5), then does a SMACOF step towards those disparities. On the demo's
distances cubed — strictly increasing, nowhere near linear — metric MDS faithfully
reproduces the cube (correlation $0.9022$ with the truth, $0.8354$ of the local
neighbourhoods kept) while nonmetric recovers the geometry ($1.0000$ and
$1.0000$).

**`weighted`** is (1) with a weight matrix, where $w_{ij} = 0$ means "this pair
was never observed". It is the same solver as `smacof` — the weights are a
first-class part of the formulation, not a separate algorithm — and the code
refuses to run it without at least one zero so that it cannot silently duplicate
`smacof`. With $230$ of $780$ pairs observed the demo recovers the configuration
at Gram error $0.0005$ and a mean relative residual of $3.3\times10^{-12}$ on the
pairs it saw; imputing the mean and running the closed form gives $0.8670$.

## Derivation

### 1. Stress is a graph problem, not a matrix one

Write (1) out as three terms:

$$\sigma(X) \;=\; \sum_{i<j} w_{ij}\lVert x_i-x_j\rVert^2 \;-\; 2\sum_{i<j}w_{ij}\delta_{ij}\lVert x_i-x_j\rVert \;+\; \sum_{i<j}w_{ij}\delta_{ij}^2. \qquad (2)$$

The first term is a quadratic form in $X$ — it is $\mathrm{tr}(X^{\mathsf T} V X)$
with $V$ the weighted Laplacian of the graph whose edges are the observed pairs.
The second is the only hard part: $-\lVert x_i - x_j\rVert$ is concave, and
maximising a concave function is the non-convexity in the problem. The third term
does not involve $X$ at all.

That decomposition is the whole derivation. SMACOF linearises the concave term at
the current point, which turns (2) into a quadratic in $X$ with a closed-form
minimiser; gradient descent differentiates it; classical MDS throws it away.

### 2. Classical MDS: strain, and why it has a closed form

Suppose the $\delta$ *are* Euclidean distances, $\delta_{ij} = \lVert z_i -
z_j\rVert$ for some unknown $z$. Then

$$\delta_{ij}^2 \;=\; \lVert z_i\rVert^2 - 2\,z_i^{\mathsf T}z_j + \lVert z_j\rVert^2,
\qquad (3)$$

which says the matrix $D^{(2)}$ of squared distances is a rank-$\le k$ Gram matrix
plus two rank-one terms from the norms. Centring removes those terms. With
$J = I - \tfrac1n \mathbf{1}\mathbf{1}^{\mathsf T}$,

$$B \;=\; -\tfrac12\, J D^{(2)} J, \qquad B_{ij} = (z_i - \bar z)^{\mathsf T}(z_j - \bar z), \qquad (4)$$

so $B$ is the Gram matrix of the centred coordinates, and any factorisation
$B = XX^{\mathsf T}$ is an embedding with the right distances. Since $B$ is
symmetric and positive semidefinite, the factorisation is an eigendecomposition,

$$B = V\Lambda V^{\mathsf T} \;\Longrightarrow\; X = V_k \Lambda_k^{1/2}, \qquad (5)$$

keeping the $k$ largest eigenvalues. No iteration, no seed, and — when the data is
exactly Euclidean in $k$ dimensions — no error: the demo's Gram error is
$1.1\times10^{-15}$ and its stress $1.3\times10^{-31}$, floating-point zero.

When $B$ has negative eigenvalues, (5) truncates them to zero, which is a
*projection* onto the nearest positive semidefinite matrix. That is the classical
answer to non-Euclidean dissimilarities: not the minimiser of stress, the
minimiser of **strain**,

$$\lVert B - XX^{\mathsf T}\rVert_F^2, \qquad (6)$$

a least-squares problem in the *squared* distances. Strain and stress disagree
whenever the $\delta$ are not Euclidean, and the demo's 1-D fit shows the cost of
asking for too few dimensions: stress $0.0860$ where two dimensions give zero.

### 3. The majorisation, and the Guttman transform

The concave term in (2) is $-\lVert x_i-x_j\rVert$. The function $u \mapsto
-\sqrt{u}$ is convex, so its tangent at $u_0$ lies below it, which gives the
inequality

$$-\lVert x_i - x_j\rVert \;\le\; -\frac{(x_i-x_j)^{\mathsf T}(x_i^{t}-x_j^{t})}{\lVert x_i^{t}-x_j^{t}\rVert}
\;+\;\text{const}, \qquad (7)$$

and substituting into (2) gives a quadratic upper bound on the stress that touches
it at $X^t$. Minimising that bound is a linear system:

$$\frac{\partial}{\partial X}\Big[\mathrm{tr}(X^{\mathsf T}VX) - 2\,\mathrm{tr}\big(X^{\mathsf T}B(X^t)X^t\big)\Big] = 0
\;\Longrightarrow\; VX = B(X^t)X^t, \qquad (8)$$

with $V$ the weighted Laplacian and

$$B_{ij}(X) \;=\; \frac{w_{ij}\,\hat\delta_{ij}}{\lVert x_i - x_j\rVert}\;(i\ne j),
\qquad B_{ii} = -\sum_{j\ne i}B_{ij}. \qquad (9)$$

For unit weights $V = nI - \mathbf{1}\mathbf{1}^{\mathsf T}$, whose pseudo-inverse
is $I/n$ on the subspace orthogonal to $\mathbf{1}$, so (8) becomes the textbook
update $X \leftarrow \tfrac1n B(X)X$. The code solves the system instead, because
a general weight matrix is not the complete graph and $\tfrac1n$ is then wrong.
Because the bound touches the stress and lies above it, the updated point can only
lower the stress — the monotonicity is a theorem, not an observation, and the demo
counts zero uphill steps as a check on the implementation rather than on the
theory.

### 4. Descent

Differentiating (1) directly, with $d_{ij} = \lVert x_i-x_j\rVert$,

$$\frac{\partial\sigma}{\partial x_i} \;=\; 2\sum_{j\ne i} w_{ij}\,\frac{d_{ij}-\delta_{ij}}{d_{ij}}\,(x_i - x_j), \qquad (10)$$

which is a Laplacian-times-$X$ product in disguise: build the matrix of
coefficients $c_{ij} = w_{ij}(d_{ij}-\delta_{ij})/d_{ij}$, and the gradient is
$(\mathrm{diag}(c\mathbf{1}) - c)X$. A step along $-\nabla\sigma$ lowers the
stress for a small enough step, and the line search finds "small enough" by
halving until it does. Backtracking guarantees monotonicity per step, but only
because it checks; a fixed step can and does diverge, which is the sense in which
descent is the less safe of the two methods.

### 5. Ordinal constraints and isotonic regression

Nonmetric MDS replaces each $\delta_{ij}$ with a free value $\hat\delta_{ij}$
subject to preserving the order,

$$\hat\delta_{ij} \le \hat\delta_{hk} \quad\text{whenever}\quad \delta_{ij} \le \delta_{hk}, \qquad (11)$$

and chooses those values to sit as close as possible to the current distances:

$$\hat\delta \;=\; \arg\min_{z\ \text{monotone}}\ \sum_{i<j}\big(d_{ij}(X) - z_{ij}\big)^2, \qquad (12)$$

which is isotonic regression and is solved exactly by pool-adjacent-violators:
scan in the order of the $\delta$, and whenever the fitted value would decrease,
merge the offending block into its mean. The result is the closest monotone
sequence in squared error — no step size, no tolerance.

Then one SMACOF step towards $\hat\delta$ and the loop repeats. This is why any
strictly increasing transformation of the $\delta$ leaves the answer unchanged:
the algorithm only ever reads their **order**.

The objective is Kruskal's stress-1,

$$\mathrm{stress}_1 \;=\; \sqrt{\frac{\sum_{i<j}\big(d_{ij}-\hat\delta_{ij}\big)^2}{\sum_{i<j} d_{ij}^2}}, \qquad (13)$$

and the choice of that denominator matters. Raw stress against re-fitted
disparities can be driven to zero by shrinking the configuration towards a point,
because a collapsing configuration is perfectly monotone in almost anything.
Stress-1 divides by the scale of the current distances, so it measures only the
departure from monotonicity — which is the quantity the model claims to preserve.
The code records stress-1 for `nonmetric` and normalised raw stress for the other
four.

### 6. Missing pairs

A zero weight removes the pair from both sums in (1) and from the Laplacian in
(8). The objective is then defined on the observed graph only, and the embedding
is determined by the connectivity of that graph: a point with no observed pair
contributes nothing and is placed arbitrarily. The demo asserts every point keeps
at least one pair, and on a sparse graph the degrees of freedom are visibly
smaller — $230$ distances constrain $2n-3 = 77$ coordinates, which is why the fit
to those distances is essentially exact ($3.3\times10^{-12}$) and the recovery of
the unseen pairs is the interesting number.

This is a genuinely different operation from imputing the missing entries. An
imputed value enters the objective at full weight and pulls the configuration
towards a number nobody measured; a zero weight lets the geometry be determined by
its neighbours. The demo's $0.0005$ against $0.8670$ Gram error is that difference
at $70\%$ missingness.

### 7. What is identified

Distances determine coordinates only up to an isometry:

$$X \mapsto XQ + \mathbf{1}t^{\mathsf T}, \qquad Q^{\mathsf T}Q = I, \qquad (14)$$

so comparing embeddings coordinate by coordinate is meaningless, and comparing
them by Procrustes rotation only slightly less so. The honest comparison is the
Gram matrix $XX^{\mathsf T}$, which is invariant to (14), or the distances
themselves. Every claim in the demo is a Gram error, a distance correlation or a
neighbourhood overlap, for this reason.

## Code walkthrough

`MDS(n_components=2, *, variant="classical", n_iter=300, tol=1e-9, seed=0)` with
`fit(D, weights=None)` and `fit_transform(D, weights=None)`. `D` is `(n, n)`,
symmetric, non-negative, with a zero diagonal; `weights` is the same shape, and a
zero weight means a missing pair. Attributes: `embedding_` `(n, n_components)`,
`stress_`, `stresses_` (one value per sweep) and `disparities_` (the monotone
targets, `nonmetric` only).

### 1. Distances without a Python loop

```python
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
```

The broadcast builds the $(n,n,k)$ array of differences, and the `einsum` contracts
the last axis. Writing $\sqrt{\sum_k (x_{ik}-x_{jk})^2}$ this way costs $n^2k$
memory but removes the loop that every naive implementation has and every profile
then shows. `np.linalg.norm(diff, axis=2)` is the same thing and allocates the same
array; the `einsum` skips the intermediate square root of a sum along a strided
axis.

### 2. The closed form, and the truncation that hides in it

```python
        n = D.shape[0]
        J = np.eye(n) - np.full((n, n), 1.0 / n)
        B = -0.5 * J @ (D ** 2) @ J
        evals, evecs = np.linalg.eigh(B)
        order = np.argsort(evals)[::-1][: self.n_components]
        keep = np.maximum(evals[order], 0.0)
        return evecs[:, order] * np.sqrt(keep)
```

(4)–(5). Two details are the whole function. `eigh` returns ascending eigenvalues,
so the `[::-1]` is not decoration — remove it and the embedding is the *worst*
dimensions, which is a plausible-looking plot of noise. And `np.maximum(..., 0.0)`
is the projection of (6): a negative eigenvalue has no real square root, and
clipping it is the least-squares answer rather than an error. Both are silent when
wrong, which is why the demo's first assertion is a Gram error against a known
configuration and not a fit statistic.

### 3. Stress counts each pair once

```python
    d = _distances(X)
    iu = np.triu_indices_from(D, k=1)
    return float(np.sum(W[iu] * (d[iu] - D[iu]) ** 2))
```

(1). The upper triangle, because the full matrix counts every pair twice and
doubles the diagonal, which is zero anyway. Normalising by
$\sum_{i<j} w_{ij}\delta_{ij}^2$ — done by the caller — makes the number
comparable across datasets and puts it in a range where $0.004864$ and $0.006707$
are visibly different rather than both "small".

### 4. The Guttman transform with weights

```python
    d = _distances(X)
    ratio = W * target / np.maximum(d, EPS)
    np.fill_diagonal(ratio, 0.0)
    B = ratio.copy()
    np.fill_diagonal(B, -B.sum(axis=1))

    V = -W.copy()
    np.fill_diagonal(V, 0.0)
    np.fill_diagonal(V, -V.sum(axis=1))
    return np.linalg.lstsq(V + 1e-12 * np.eye(n), B @ X, rcond=None)[0]
```

(8)–(9). `B`'s diagonal is the negative row sum, which makes it the matrix whose
quadratic form is the majoriser; `V` is the Laplacian, built the same way from the
weights. `V` is singular — constants are in its null space, because shifting every
point changes nothing — so a plain solve fails and `lstsq` returns the minimum-norm
solution, which is the one with the points centred at the origin. The `1e-12`
ridge is there for the degenerate case of a point with no observed pair: without
it the system is inconsistent rather than merely singular, and `lstsq` returns a
least-squares compromise that moves every other point to accommodate the
unconstrained one.

### 5. Pool adjacent violators

```python
    means: list[float] = []
    sizes: list[int] = []
    for v in y:
        means.append(float(v))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            v2, s2 = means.pop(), sizes.pop()
            v1, s1 = means.pop(), sizes.pop()
            means.append((v1 * s1 + v2 * s2) / (s1 + s2))
            sizes.append(s1 + s2)
    return np.repeat(np.asarray(means), np.asarray(sizes))
```

(12). The blocks are kept as (mean, size) pairs, and a violation merges the last
two blocks into their weighted mean — one pass, $O(n)$ amortised, exact. Two
things are easy to get wrong: the merge has to be weighted by block size (an
unweighted mean of means is not the least-squares fit), and the `while` has to
keep checking after a merge, because merging can create a new violation with the
block before it. A single `if` passes the small examples and fails on real data.

### 6. Ordinal targets, and the reconstruction that must be symmetric

```python
        iu = np.triu_indices_from(D, k=1)
        order = np.argsort(D[iu], kind="stable")
        d = _distances(X)[iu]
        fitted = np.empty_like(d)
        fitted[order] = _isotonic(d[order])
```

The sort is on the *targets* and the regression is on the *distances*: the
isotonic fit is monotone in $\delta$, applied to $d$. `fitted[order] = ...` is the
inverse permutation that puts the fitted values back in pair order, so the
disparity matrix can be rebuilt and symmetrised. `kind="stable"` matters: ties in
the $\delta$ are common on ordinal data (a 1–7 scale has six possible ties per
pair), and an unstable sort makes the fitted values depend on memory layout. This
is Kruskal's *primary* approach — ties are allowed to be broken by the embedding —
as opposed to the secondary approach, which forces tied dissimilarities to tied
distances.

### 7. Descent with a line search

```python
            ratio[iu] = W[iu] * (d[iu] - D[iu]) / np.maximum(d[iu], EPS)
            ratio = ratio + ratio.T
            grad = 2.0 * (np.diag(ratio.sum(axis=1)) - ratio) @ X
            slope = float(np.sum(grad * grad))
            if slope < EPS:
                break
            trial, scale = X, step
            while scale > 1e-16:
                trial = X - scale * grad
                candidate = _stress(trial, D, W)
                if candidate <= sigma - 1e-4 * scale * slope:
                    break
                scale *= 0.5
```

(10). `ratio` holds $c_{ij}$ on the upper triangle and is symmetrised, which is
what makes the Laplacian product correct; forgetting the `+ ratio.T` halves the
gradient and doubles the iteration count without changing the fixed point. The
Armijo-style test `candidate ≤ σ − 10⁻⁴·scale·‖g‖²` is stronger than "the stress
went down": it requires a decrease proportional to the step, which rules out the
near-zero steps that a bare comparison accepts and then stalls on.

### 8. A zero weight is not a small number

```python
        if self.variant == "weighted" and np.all(W > 0.0):
            raise ValueError("weighted MDS needs a missing pair; pass weights with zeros")
```

The guard is about honesty of the API: `weighted` with a full matrix *is* `smacof`,
and silently returning the same numbers under a different name would make the
variant table a fiction. The same logic is why `_init_missing` fills the holes and
then throws the fill away — it is a starting configuration, not a model of the
missing data, and the numbers it invents never appear in the objective.

## Complexity

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_distances` | $O(n^2k)$ | $O(n^2k)$ | the broadcast is the cost |
| `classical` | $O(n^3)$ | $O(n^2)$ | one symmetric eigendecomposition |
| one `smacof` sweep | $O(n^3)$ | $O(n^2)$ | the Laplacian solve |
| one `metric` step | $O(n^2k)$ | $O(n^2)$ | gradient plus a line-search evaluation |
| one `nonmetric` sweep | $O(n^2\log n)$ | $O(n^2)$ | the sort dominates |
| `weighted` | $O(n^3)$ per sweep | $O(n^2)$ | sparse weights do not make `lstsq` sparse |
| full demo | — | — | 6 fits, 0.5 s |

The awkward row is `smacof`'s $O(n^3)$. For the complete graph the update is
$B(X)X/n$, which is $O(n^2k)$, and that is what every production implementation
uses. The code solves a general Laplacian system because the weighted variant
needs it, and pays $n^3$ for the generality. Above a few thousand points, the
right structure is to keep the unweighted update and handle missing pairs by
masking the *edges* rather than the matrix entries.

Classical MDS is $O(n^3)$ once, which is cheaper than a hundred SMACOF sweeps for
small $n$ and hopeless for large $n$: at $n = 10^5$ the eigendecomposition is
$10^{15}$ operations. That is the reason Isomap has a landmark variant and why
large-scale MDS is done by subsampling or by stochastic descent on a minibatch of
pairs.

## Failure modes

**1. The closed form is not minimising stress.** On non-Euclidean dissimilarities
classical MDS still returns coordinates, with negative eigenvalues quietly
truncated to zero, and they are the minimiser of strain — a different objective.
The visible symptom is a configuration that looks plausible and a stress that is
worse than SMACOF's: $0.006707$ against $0.004864$ on the demo's noisy Euclidean
distances, and a much larger gap on genuinely non-Euclidean data.

**2. Rotation, reflection and translation are free.** (14). Comparing embeddings
across runs by coordinates, or by which point ends up at the top of the plot, is
comparing an arbitrary basis. Use Gram errors, distance correlations or
neighbourhood overlap.

**3. Local minima are real and the start decides.** SMACOF is monotone but not
global, and the code starts from the classical embedding precisely because a
random start lands in worse basins. If two restarts give different stress, report
both; if they give the same stress with different configurations, the problem has
a symmetry you have not noticed.

**4. Nonmetric stress can be driven to zero without being right.** Raw stress
against re-fitted disparities goes to zero as the configuration collapses, since a
collapsing configuration is monotone in everything. That is why the variant
reports stress-1 and why the demo's nonmetric run shows stress-1 still falling
geometrically after 200 sweeps: the objective has no interior minimum, it has a
limit.

**5. Ordinal data does not give you metric answers.** Nonmetric MDS recovers the
configuration only up to a monotone transformation of the distances, so a distance
in the embedding cannot be read as "twice as far". The demo's correlation of
$1.0000$ is a statement about geometry, not about scale.

**6. Missing pairs must leave the graph connected enough.** A point with no
observed pair is unconstrained and wanders; a graph that splits into two
components leaves their relative placement arbitrary — each component can be
rotated and translated independently. The demo asserts every point keeps a pair,
which is the weakest useful condition; the real one is that the graph is connected.

**7. A zero dissimilarity is not a missing pair.** Two distinct objects can
legitimately be at distance zero (identical species counts, tied ranks), and a
weight of zero says something completely different. The code distinguishes them —
`D` may contain zeros, `W` may contain zeros — but a caller who confuses the two
gets a configuration that ignores real data or trusts an invented one.

**8. Truncating to too few dimensions is silent.** The 1-D fit of planar data
costs stress $0.0860$ and still produces a plot. There is no warning, no negative
eigenvalue to notice, and the picture looks like data. Always report the stress
and compare against one more dimension.

## Applications

- **Perceptual mapping.** Brands, products or political candidates placed by
  judged similarity; the axes are interpreted post hoc, which is legitimate as
  long as nobody claims they are unique.
- **Psychometrics and survey research.** Ordinal similarity judgements,
  preference rankings and Likert-scale profiles are exactly the data nonmetric
  MDS was built for.
- **Ecology and genomics.** Bray–Curtis, UniFrac and Jaccard dissimilarities
  between sites or samples; classical or weighted MDS is the standard ordination,
  and the missing-data variant handles incomplete sampling.
- **Isomap and manifold learning.** The second stage of `isomap` is classical MDS
  on graph geodesics, which is why this file is a prerequisite for that one.
- **Graph layout.** Fruchterman–Reingold and its descendants are stress
  majorisation on graph distances; Kamada–Kawai *is* SMACOF on shortest paths.
- **Molecular conformation.** Distances from NMR or cross-linking experiments,
  with many pairs unmeasured — the weighted variant's original application.

Where it is a trap: reading coordinates as features; choosing the number of
dimensions by looking at the plot; trusting classical MDS on non-Euclidean
dissimilarities; reporting nonmetric distances quantitatively; and running an
$O(n^3)$ solve on a dataset that has ten thousand points and could have been
subsampled.

## References

- **Torgerson (1952),** *Multidimensional scaling: I. Theory and method* — the
  double-centring construction of (4) and the closed form of (5), which is still
  the fastest thing to run and still the wrong objective half the time.
- **Kruskal (1964),** *Multidimensional scaling by optimizing goodness of fit to a
  nonmetric hypothesis* — the ordinal model, stress-1, and the insight that the
  data may only be ordered.
- **Kruskal (1964),** *Nonmetric multidimensional scaling: a numerical method* —
  the isotonic regression inside the loop, and the gradient method of (10).
- **Guttman (1968),** *A general nonmetric technique for finding the smallest
  coordinate space for a configuration of points* — the transform of (8), which
  is the reason SMACOF is monotone.
- **de Leeuw (1977),** *Applications of convex analysis to multidimensional
  scaling* — the majorisation view that turns Guttman's transform into a proof.
- **de Leeuw & Heiser (1980),** *Multidimensional scaling with restrictions on the
  configuration* — the weighted formulation and its use for missing data.
- **Borg & Groenen (2005),** *Modern Multidimensional Scaling* — the reference
  treatment of every variant here, including the tie-handling distinction between
  the primary and secondary approaches.
- **Cox & Cox (2001),** *Multidimensional Scaling* — the shorter, more applied
  account, with the perceptual-mapping case studies that made the method famous.
- **Buja, Swayne, Littman, Dean, Hofmann & Chen (2008),** *Data visualization with
  multidimensional scaling* — the modern review, including why stress and strain
  disagree and what to report instead.
- **Tenenbaum, de Silva & Langford (2000),** *A global geometric framework for
  nonlinear dimensionality reduction* — Isomap, which is classical MDS on
  geodesic distances and the clearest example of choosing $\delta$ well.
