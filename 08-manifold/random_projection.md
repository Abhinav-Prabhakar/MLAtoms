# Random projection

`random_projection.py` · Manifolds & embeddings · Tier 1 · 4 variants

## Intuition

Every other method in this chapter learns something from the data: neighbours,
eigenvectors, weights, an autoencoder. Random projection learns nothing. It
multiplies the data by a matrix of random numbers and keeps the result.

The reason that works is a theorem with no assumptions about the data at all.
Johnson and Lindenstrauss (1984) proved that for any $n$ points in any Euclidean
space, a projection onto

$$k \;=\; O\!\left(\frac{\log n}{\varepsilon^2}\right) \qquad (1)$$

random dimensions preserves all pairwise distances within a factor $1\pm
\varepsilon$ with high probability. The bound does not contain the input
dimension. A dataset in $10^6$ dimensions and a dataset in $10$ need the same
number of random coordinates to keep their geometry, because the guarantee is
about $n$ points, and $n$ points have at most $n(n-1)/2$ distances to preserve.

```
   200 points, 2000 dimensions          200 points, 64 dimensions
   ┌──────────────────────────┐         ┌────────────┐
   │  ·    ·        ·    ·    │         │  ·   ·     │
   │      ·    ·  ·           │   ──▶   │    ·   ·   │   distances within 1±ε
   │   ·       ·       ·   ·  │  R x    │ ·    ·     │   and nothing else
   └──────────────────────────┘         └────────────┘
```

It is worth being precise about what is *not* promised. Distances are preserved;
directions, clusters, densities, margins and nearest-neighbour identities are not.
The projection cannot exploit structure, so on data that is genuinely low-rank it
is beaten by the directions that structure defines: the demo's 300 points lie
exactly in a 10-dimensional subspace of $\mathbb{R}^{800}$, and ten principal
directions reconstruct them to $1.9\times10^{-14}$ while ten random directions
distort distances by $0.776$.

What the random projection buys is speed, and a property that no learned method
has: it is **data-oblivious**. The matrix can be drawn before the data exists, so
the projection can be applied at the point of collection, inside a database, or
before a privacy budget is spent — and the sketch of a dataset can be computed by
sketching its parts and adding.

## Variants

| Variant | Entries of $R$ | Non-zero fraction | Multiply cost |
|:--|:--|:--|:--|
| `gaussian` | $\mathcal{N}(0, 1/k)$ | 1 | $O(ndk)$ |
| `achlioptas` | $\pm 1/\sqrt{k}$ | 1 | $O(ndk)$, no RNG for values |
| `sparse` | $\pm 1/\sqrt{sk}$, prob. $s = 1/3$ | $1/3$ | $O(ndk/3)$ |
| `very_sparse` | $\pm 1/\sqrt{sk}$, prob. $s = 1/\sqrt{d}$ | $1/\sqrt{d}$ | $O(nk\sqrt{d})$ |

**`gaussian`** is the textbook construction and the one the proofs are usually
written for: i.i.d. normal entries scaled by $1/\sqrt{k}$. The demo projects 400
points from 1000 to 64 dimensions and the worst pairwise distortion is $0.396$.

**`achlioptas`** replaces the normal draws with Rademacher signs, $\pm 1/\sqrt{k}$.
The rows already have norm exactly 1, the analysis goes through the same
concentration inequalities, and generating $\pm1$ is cheaper than generating
normals. Its measured distortion is $0.375$ — indistinguishable from Gaussian at
this size, which is the point.

**`sparse`** sets each entry to zero with probability $2/3$ and to
$\pm 1/\sqrt{sk}$ otherwise, with $s = 1/3$ (Achlioptas 2003). The row norm is
still 1 in expectation, so the guarantee is unchanged, but two thirds of the
matrix is gone: the multiply becomes several scatter-adds instead of one dense
product, and on sparse input it touches only the non-zeros.

**`very_sparse`** pushes the density down to $1/\sqrt{d}$ (Li, Hastie & Church
2006). At $d = 20000$ that is $0.7\%$ non-zero and the demo's $128\times20000$
matrix holds 17,945 non-zeros against 2.56 million for the Gaussian one — a factor
143 fewer entries to touch. The price is a heavier-tailed distribution, so the
constant in the concentration bound is worse and the same $k$ gives slightly more
distortion ($0.438$ against $0.396$ in the demo). At large $d$ that is a trade
worth making; at small $d$ it is not.

All four are the same three lines of code with a different distribution. The one
thing they must share is the scaling: the entries' variance has to be $1/k$, not
"about right", and §2 shows exactly what goes wrong otherwise.

## Derivation

### 1. What has to be true for a random matrix to preserve norms

Let $R$ be $k\times d$ with i.i.d. entries of mean zero, and consider one vector
$x\in\mathbb{R}^d$. Then

$$\mathbb{E}\lVert Rx\rVert^2 \;=\; \mathbb{E}\big[x^{\mathsf T}R^{\mathsf T}Rx\big]
\;=\; x^{\mathsf T}\big(k\,\mathbb{E}[rr^{\mathsf T}]\big)x, \qquad (2)$$

where $r$ is a row of $R$. So the projection is unbiased — preserves squared norms
in expectation — exactly when

$$\mathbb{E}[rr^{\mathsf T}] \;=\; \frac{1}{k}I, \qquad (3)$$

that is, when each entry has variance $1/k$ and the entries are uncorrelated. Every
distribution in the table is a way of satisfying (3) with a different amount of
work.

The variance requirement is where sparse constructions get subtle, and where this
file had a bug for one run. If a fraction $s$ of the entries is non-zero and each
non-zero is $\pm a$, then the variance is $s a^2$, so (3) demands

$$a \;=\; \frac{1}{\sqrt{sk}}, \qquad (4)$$

not $\sqrt{s/k}$. The wrong scaling inflates every row's norm by $1/\sqrt{s}$,
which at $s = 1/3$ is a factor $1.7$: the demo's first attempt measured distortions
of $0.78$ and $0.98$ for the two sparse variants against $0.40$ and $0.38$ for the
dense ones, and it was the assertion — not the eye — that caught it.

### 2. Concentration

Unbiased is not enough; the actual norm has to be close to its expectation for
every point. For Gaussian entries, $k\lVert Rx\rVert^2/\lVert x\rVert^2$ is
$\chi^2_k$, and the standard tail bound gives, for $\varepsilon\in(0,1)$,

$$\mathbb{P}\Big(\big|\,\lVert Rx\rVert^2/\lVert x\rVert^2 - 1\,\big| > \varepsilon\Big)
\;\le\; 2\exp\!\left(-\frac{k(\varepsilon^2-\varepsilon^3)}{4}\right). \qquad (5)$$

Rademacher and sparse entries have the same shape of bound with worse constants,
which is why they behave identically in the demo.

For a *pair* of points, apply (5) to $x_i - x_j$: the difference of two projections
is the projection of the difference, so distance preservation is norm preservation
of the difference vector. Preserving all $\binom{n}{2}$ pairs at once is a union
bound:

$$\mathbb{P}\big(\text{some pair distorted}\big) \;\le\; n^2 \cdot
2\exp\!\left(-\frac{k(\varepsilon^2-\varepsilon^3)}{4}\right) \;\le\; \delta
\;\;\Longleftrightarrow\;\;
k \;\ge\; \frac{4\ln(2n^2/\delta)}{\varepsilon^2-\varepsilon^3}. \qquad (6)$$

Equation (6) is the lemma in the form you actually use: fix $n$, $\varepsilon$ and
a failure probability $\delta$, and read off $k$. It depends on $\ln n$, which is
why the number of dimensions needed is so small and why it grows so slowly.

### 3. The distortion the demo measures

Everything above is about squared norms; the demo reports the worst relative
change in a distance,

$$\Delta \;=\; \max_{i<j}\left|\frac{\lVert y_i-y_j\rVert}{\lVert x_i-x_j\rVert} - 1\right|,
\qquad (7)$$

which is the quantity a downstream algorithm actually sees. On 200 points in 2000
dimensions it reads $0.976$ at $k=8$, $0.473$ at $k=32$, $0.262$ at $k=128$ and
$0.112$ at $k=512$: each four-fold increase in $k$ halves the worst distortion,
which is the $\sqrt{1/k}$ rate that (5) implies. The typical distortion at $k$ is
about $\sqrt{2/k}$ — $0.18$ at $k = 64$ — and the maximum over 20,000 pairs lands
at roughly twice that. The union bound of (6) is much more pessimistic: with
$n = 200$, $\delta = 0.05$ and $\varepsilon = 0.5$ it asks for
$k \approx 460$. The lemma is a guarantee, not a prediction, and at these sizes the
measured distortion is far better than the guarantee requires.

### 4. The bound has no $d$ in it

The demo's sharpest check is to multiply the input dimension by ten and change
nothing else. At $k = 64$ the distortion goes from $0.346$ at $d = 2000$ to
$0.391$ at $d = 20000$ — the same number, twice measured with different noise. The
ambient dimension enters only through the arithmetic of the multiply; it does not
enter the guarantee.

That is counter-intuitive enough to be worth restating: a random projection from
$10^6$ dimensions to $10^3$ preserves the distances between $10^6$ points, because
there are only $10^6$ of them and each distance is a one-dimensional quantity.

### 5. Why it loses to PCA on low-rank data
Suppose the data lies in an $r$-dimensional subspace, $r \ll d$. PCA chooses that
subspace and needs exactly $k = r$ coordinates to represent it with no error at
all. A random $k$-dimensional projection cannot choose, and — this is the point
that is easy to get wrong — it does not systematically *lose* the subspace either.
For $x$ inside the subspace, the projected norm is still unbiased,

$$\mathbb{E}\left[\frac{\lVert Rx\rVert^2}{\lVert x\rVert^2}\right] = 1,
\qquad \mathrm{Var}\left[\frac{\lVert Rx\rVert^2}{\lVert x\rVert^2}\right] = \frac{2}{k},
\qquad (8)$$

whatever $r$ is, because $R$ mixes the $r$ coordinates rather than discarding
them. The difference is not bias, it is exactness: PCA's error is identically zero
once $k \ge r$, while a random projection's relative error stays of order
$\sqrt{2/k}$ no matter how simple the data is. The demo's numbers are the measured
version: at $k = r = 10$, random directions give $0.776$ and the principal
directions give $1.9\times10^{-14}$.

So the comparison is not "random projection is worse". It is: random projection is
$O(ndk)$ and data-oblivious; PCA is $O(nd\min(n,d))$, needs the data, and is exact
on low-rank data. Which is cheaper depends on whether the data has exploitable
structure, whether $k$ can be as small as $r$, and whether you are allowed to look
at the data before you project it.

### 6. What the sparse constructions cost

A dense $k\times d$ projection of an $n\times d$ matrix is $ndk$ multiply-adds. A
sparse one with density $s$ is $ndks$ multiply-adds but *the same number of memory
touches per non-zero*, so the win is proportional to $1/s$ only when the data is
dense; on sparse data the touches are proportional to the data's non-zeros.

The demo's $128\times20000$ example makes the scale concrete: 2.56 million
non-zeros for Gaussian, 853,215 for sparse, 17,945 for very sparse. The dense
multiply is $5.1$ billion multiply-adds; the very sparse one touches 36 million
entries — a factor 143 fewer. That is what "cheaper" means here, and it is why
feature hashing (which is a random projection with a fixed sparse structure) is
the default in large-scale text pipelines.

### 7. What is preserved, and what is not

Distances are preserved, and by the polarisation identity so are inner products:

$$\langle Rx, Ry\rangle \;=\; \tfrac14\big(\lVert R(x+y)\rVert^2 - \lVert R(x-y)\rVert^2\big),
\qquad (9)$$

so a bound on norms gives a bound on inner products with the same $\varepsilon$ up
to constants. Norms are preserved, hence so is the norm of the centroid, but the
*identity* of the nearest neighbour is not: two points at nearly equal distance
from a query can swap order under any perturbation of size $\varepsilon$, and a
random projection is a perturbation of exactly that size. Clustering structure,
margins, and density estimates all degrade faster than distances do.

## Code walkthrough

`RandomProjection(n_components=50, *, variant="gaussian", density=None, seed=0)`
with `fit(X)`, `transform(X)`, `fit_transform(X)` and the attribute `components_`
`(n_components, d)`. The only thing `fit` uses from `X` is its number of columns —
which is the point.

### 1. Two dense distributions, and the scaling that matters

```python
        if self.variant == "gaussian":
            return rng.standard_normal((k, d)) / np.sqrt(k)
        if self.variant == "achlioptas":
            return rng.choice([-1.0, 1.0], size=(k, d)) / np.sqrt(k)
```

(3). Both divide by $\sqrt{k}$, which sets the entry variance to $1/k$; the
Rademacher version's rows then have norm *exactly* 1, not just in expectation,
because every entry is $\pm 1/\sqrt{k}$. Dropping the division is the classic
mistake: the projection still preserves distances up to a global scale factor,
which looks like a successful embedding and is wrong by $\sqrt{k}$ — the demo's
first assertion would pass on ranks and fail on values.

### 2. The sparse construction, and the variance

```python
        s = self._density(d)
        # variance must be 1/k, so a non-zero of density s has magnitude
        # 1/√(sk); using √(s/k) inflates the row norm by 1/√s
        values = rng.choice([-1.0, 1.0], size=(k, d)) * np.sqrt(1.0 / (s * k))
        mask = rng.random((k, d)) < s
        return values * mask
```

(4). Generating the signs and the mask separately and multiplying is clearer than
a three-way `np.where`, and it makes the two independent random draws explicit.
The comment is there because this file shipped the wrong line first: with
$\sqrt{s/k}$ the very sparse variant distorted distances by $0.98$, which still
looks like a projection and is not one. `density` is exposed so a caller can tune
$s$; the defaults are the ones from the papers.

### 3. Fit uses only the shape

```python
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (samples, features), got shape {X.shape}")
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed the input dimension")
        self.components_ = self._draw(X.shape[1])
        self.n_features_in_ = X.shape[1]
        return self
```

This is the only `fit` in the chapter that does not look at the values, and the
guard on `n_components` is the one that matters: projecting up is a valid matrix
product and a meaningless operation, so it raises instead of returning a
higher-dimensional embedding that looks like it did something.

### 4. The projection, and the transpose

```python
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return X @ self.components_.T
```

`components_` is `(k, d)` so that the matrix reads as $k$ rows of $d$ coefficients —
the natural orientation for inspecting a projection direction — and the transform
therefore needs the transpose. Storing it as `(d, k)` and writing `X @ components_`
is the same computation and one fewer mental step, at the cost of a matrix whose
rows are the *features* of the projection, which is the wrong thing to look at.

### 5. Distances without an $(n,n,d)$ array

```python
    sq = np.einsum("ij,ij->i", X, X)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    return np.sqrt(d2)
```

(7) needs distances, and the obvious implementation —
`X[:, None, :] - X[None, :, :]` — allocates $n^2d$ floats. At the demo's
$400\times400\times1000$ that is 1.3 GB for a scalar. The Gram identity
$\lVert x_i-x_j\rVert^2 = \lVert x_i\rVert^2 + \lVert x_j\rVert^2 - 2x_i^{\mathsf
T}x_j$ is $O(n^2)$ memory and $O(n^2d)$ time, and the `maximum(..., 0.0)` is not
cosmetic: the identity is exact in reals and can go negative in floating point for
nearly identical points, and `sqrt` of a small negative is `nan`.

### 6. The comparison that keeps the atom honest

```python
    rp = RandomProjection(10, variant="gaussian", seed=7)
    rp_dist = _distortion(rp.fit_transform(X), X)

    centred = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pca = centred @ vt[:10].T
    pca_dist = _distortion(pca, centred)
```

The demo would be marketing without this section: on data with an exact
10-dimensional structure, the random projection is $0.776$ and PCA is
$1.9\times10^{-14}$. The SVD is used here as a *reference implementation*, not as
part of the atom — it is the cheapest honest way to show what an informed choice
of ten directions achieves, and it makes the case for random projection narrower
than "it works": it works when there is nothing to learn, or when learning is too
expensive.

## Complexity

$n$ samples, $d$ input dimensions, $k$ output dimensions, density $s$ of the
projection matrix.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| draw `components_` | $O(kd)$ | $O(kd)$ | once, independent of $n$ |
| `gaussian` / `achlioptas` | $O(ndk)$ | $O(nk)$ | one dense product |
| `sparse` | $O(ndks)$ | $O(nk)$ | $s = 1/3$ by default |
| `very_sparse` | $O(nk\sqrt{d})$ | $O(nk)$ | $s = 1/\sqrt d$ |
| `_distortion` (demo) | $O(n^2d)$ | $O(n^2)$ | the Gram identity |
| PCA, for comparison | $O(nd\min(n,d))$ | $O(nd)$ | and it needs the data |

Three numbers decide whether to use this. The first is $k$ from (6): for $n =
10^6$, $\varepsilon = 0.1$, $\delta = 0.05$, the bound gives
$k \approx 4\ln(2\times10^{12}/0.05)/0.009 \approx 14{,}000$ — large, and still an
order of magnitude below the vocabulary a bag-of-words model would carry. The second is the $O(ndk)$ product, which is
linear in everything and therefore the cheapest reduction in this chapter. The
third is that the projection matrix is $kd$ numbers: at $k = 10^3$ and $d = 10^6$
that is $10^9$ entries, which is where the very sparse construction stops being an
optimisation and becomes the only option that fits.

## Failure modes

**1. The scaling is wrong and everything still "works".** Using $\sqrt{s/k}$ for a
sparse projection inflates row norms by $1/\sqrt{s}$; the embedding is still a
linear map, still rank $k$, still looks like a cloud. The demo's first run
measured $0.98$ against $0.40$ and the assertion caught it. Check (3): entry
variance exactly $1/k$.

**2. $k$ chosen by the input dimension.** The guarantee depends on $n$, not $d$, so
"keep 10% of the features" is not a rule. At $n = 10^3$ and $k = 64$ the distortion
is already $0.35$; at $n = 10^6$ it needs thousands of dimensions for $\varepsilon
= 0.1$. Compute $k$ from (6) and the tolerance you need.

**3. Random projection is not a substitute for PCA on low-rank data.** $0.776$
against $1.9\times10^{-14}$ at the same $k$, measured, and the gap is structural:
a random projection is unbiased but it cannot be exact. If a covariance pass is
affordable, PCA (or a randomised SVD) is exact as soon as $k$ reaches the data's
rank, and it dominates the sketch once $k$ is large enough to matter.

**4. Distances preserved does not mean structure preserved.** Nearest-neighbour
identity, cluster membership, margins and density estimates all shift under an
$\varepsilon$ perturbation, and the pairs most likely to flip are exactly the ones
a classifier cares about — the close ones. Test the downstream task, not the
distance distortion.

**5. The projection is oblivious to the *future*.** A sketch of a dataset and a
sketch of its new points are addable only if they use the same $R$, which means
$k$ must be chosen for the eventual $n$, not today's. Re-drawing $R$ per batch
produces embeddings that cannot be concatenated, and the error looks like a
modelling failure rather than a bookkeeping one.

**6. Streaming order matters for nothing, and for everything.** Projection itself
is order-free — that is the appeal — but the *centring* is not: if you centre
before projecting you need the global mean, and if you centre after you must
remember that centring the projected data is equivalent to projecting the centred
data. Mixing the two gives a systematic offset in the sketch.

**7. Very sparse has heavier tails.** The same $k$ gives more distortion
($0.438$ against $0.396$ in the demo) because the entries' fourth moment is
larger. At large $d$ the cost win dominates; at small $d$ the very sparse
construction is strictly worse and the "sparse" default is the right one.

**8. Do not interpret the coordinates.** A random projection's axes are random
directions; a loading has no meaning. That is fine for distances and features fed
to a model, and fatal for a plot with labelled axes. If you need interpretable
dimensions, you wanted PCA, NMF or a topic model.

## Applications

- **Feature hashing.** A random projection with a fixed, sparse, hash-defined
  structure: text features are mapped to a fixed number of buckets before
  training. The demo's sparse variants are the same idea with a drawn matrix
  instead of a hash, and the reason it is standard at web scale is the
  $O(nk\sqrt d)$ cost.
- **Sketching for linear algebra.** Count-sketch and sparse Johnson-Lindenstrauss
  transforms are the sampling step inside randomised least-squares and
  low-rank-approximation algorithms; the same bound (6) is what makes their error
  guarantees unconditional.
- **Privacy and federated settings.** The projection is data-independent, so it
  can be applied before the data reaches a server and before a differential
  privacy budget is spent; the sketch is not invertible in any practical sense,
  and the geometry survives.
- **Approximate nearest neighbours.** Random projection trees, and the LSH family
  built from random hyperplanes, use the same distance-preservation argument to
  bucket points so that near neighbours collide with high probability.
- **Compression before clustering.** k-means on a sketch costs $O(ndk)$ to build
  and $O(nk_{\text{centres}}k)$ per iteration instead of $O(ndk_{\text{centres}})$;
  for large $d$ this is a large constant-factor win with a bounded distortion of
  the objective.
- **Sanity checks and baselines.** Because it learns nothing, a random projection
  is the right baseline for a learned embedding: if a trained autoencoder cannot
  beat it on the downstream task, the training is not buying structure.

Where it is a trap: choosing $k$ from $d$; comparing coordinates across runs;
using it on low-rank data where PCA is exact; assuming nearest neighbours are
preserved; and forgetting that the failure probability in (6) is a union bound
over $n^2$ pairs, so it degrades quadratically with dataset size unless $k$ grows
logarithmically.

## References

- **Johnson & Lindenstrauss (1984),** *Extensions of Lipschitz mappings into a
  Hilbert space* — the original lemma, in a paper about fixed points, with a
  constant far worse than anything used today.
- **Frankl & Maehara (1988),** *The Johnson-Lindenstrauss lemma and the sphericity
  of some convex bodies* — the geometric form and a much better constant, which is
  where the $4\ln(2n^2/\delta)/\varepsilon^2$ shape of (6) comes from.
- **Dasgupta & Gupta (2003),** *An elementary proof of a theorem of
  Johnson and Lindenstrauss* — the proof by random rotations, and the source of
  the $\chi^2$ tail bound in (5).
- **Achlioptas (2003),** *Database-friendly random projections* — the $\pm1$ and
  sparse constructions of the `achlioptas` and `sparse` variants, including the
  variance argument of (4).
- **Li, Hastie & Church (2006),** *Very sparse random projections* — the
  $1/\sqrt d$ density of `very_sparse`, with the cost analysis that makes it the
  default at scale.
- **Indyk & Motwani (1998),** *Approximate nearest neighbors: towards removing the
  curse of dimensionality* — the locality-sensitive-hashing use of the lemma, and
  the reason random projections are in every ANN library.
- **Bingham & Mannila (2001),** *Random projection in dimensionality reduction* —
  the empirical study that made the method standard in data mining, including the
  comparison against PCA that §5 formalises.
- **Vempala (2004),** *The Random Projection Method* — the book-length treatment,
  with the linear-algebra applications (sketching, regression) that the bound
  underpins.
- **Kane & Nelson (2014),** *Sparser Johnson-Lindenstrauss transforms* — the sharp
  sparse construction, with the $O(\log n)$ non-zeros per column that (4) only
  approximates.
- **Larsen & Nelson (2017),** *Optimality of the Johnson-Lindenstrauss lemma* —
  the matching lower bound: no distribution and no non-linear map does better than
  $k = \Theta(\log n/\varepsilon^2)$, which is what makes (1) tight rather than
  merely sufficient.
