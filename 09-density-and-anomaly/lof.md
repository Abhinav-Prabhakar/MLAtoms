# Local outlier factor

`lof.py` · Density & anomaly · Tier 1 · 4 variants

## Intuition

A point is anomalous if it is sparser than its neighbours. Not sparser than the
data set — sparser than the handful of points around it. That is the whole idea,
and it is what separates LOF from everything above it in this chapter:

```
   dense cluster              sparse region            a hole
   · · · · · ·                ·        ·              · · · ·
   · · x · · ·                ·   ·        ·          ·  x   ·
   · · · · · ·                ·        ·              · · · ·
   LOF(x) ≈ 1                 LOF ≈ 1                  LOF > 1
   (dense, like its           (sparse, but its         (sparse, and its
    neighbours)                neighbours agree)        neighbours do not)
```

The middle case is the one that costs a global detector its accuracy. A point in
a sparse region is far from everything, so a k-nearest-neighbour distance ranks
it as anomalous; but every point around it is equally far from everything, so
relative to its own neighbourhood it is perfectly ordinary. LOF divides, and the
division cancels the region's scale. The demo measures this: on a tight blob plus
a deliberately wide one, the eight genuinely odd points on the edge of the tight
blob are separated by LOF at AUC 0.971, while among the same 68 points the global
distance score gets AUC 0.000 — it ranks every wide-blob point above every odd
one, because wide-blob points are further from everything.

Formally, with $k$ a neighbourhood size,

$$
N_k(p) = \text{the } k \text{ nearest points to } p, \qquad
k\text{-dist}(p) = \max_{o \in N_k(p)} d(p, o)
\qquad (1)$$

the quantity that gets compared is a *local reachability density*,

$$
\mathrm{reach}_k(p, o) = \max\bigl(k\text{-dist}(o),\, d(p, o)\bigr)
\qquad (2)$$

$$
\mathrm{lrd}_k(p) = \left(\frac{1}{k}
\sum_{o \in N_k(p)} \mathrm{reach}_k(p, o)\right)^{-1}
\qquad (3)$$

and the factor is the ratio of the neighbours' density to the point's own:

$$
\mathrm{LOF}_k(p) = \frac{1}{k} \sum_{o \in N_k(p)}
\frac{\mathrm{lrd}_k(o)}{\mathrm{lrd}_k(p)}
\qquad (4)$$

One is the neutral value. Above one, the point is sparser than its
neighbourhood; below one, denser, which happens at the centre of a cluster and
is not usually interesting. The score is dimensionless and scale-free: the demo
translates, scales by three and rotates a sample, and every score is unchanged to
$8.9 \times 10^{-16}$.

Equation (2) is the part that is easy to get wrong, and the part that makes LOF
work. The reachability distance is not $d(p,o)$: it is $d(p,o)$ floored by the
*neighbour's* own k-distance. A point at the edge of a dense cluster may have one
neighbour that is very close and $k-1$ that are not; without the floor, the one
close neighbour would make its density look high. The floor says: a neighbour
cannot give you credit for being closer than its own scale. The demo's chain of
five points makes the effect visible in the other direction: the two end points of
a uniform chain score exactly 1, where `simplified_lof` — the same estimator with
the floor removed — gives them 1.5 and the middle points 0.83. The floor is what
stops an end from looking sparse merely for being an end.

`k` sets what "local" means and is the only real parameter. The published value
is 20; a `k` of 1 compares a point with a single neighbour and is mostly noise,
and a `k` close to $n$ turns LOF into a global density ratio. The variants are
four answers to "what is a local density":

- `lof` is (3), the paper's definition.
- `simplified_lof` drops the reachability floor and uses the plain mean distance.
- `lof_knn` drops the mean too, keeping only the k-th distance.
- `cluster_lof` computes the ratio between micro-clusters, weighting each by its
  point count, so the score is piecewise constant and a scattered cluster is
  judged as a unit.

## Variants

| variant | density of a point | scores constant within | measured |
|:--|:--|:--|:--|
| `lof` | $1/\text{mean reach}$ (3) | no | ring 6.78, sparse blob 1.21 |
| `simplified_lof` | $1/\text{mean } d$ | no | ring 8.39, sparse blob 1.33 |
| `lof_knn` | $1/k\text{-dist}$ | no | ring 6.61, noisier point to point |
| `cluster_lof` | mass / volume (5) | yes, per micro-cluster | ring 11.29, flags 7.6% |

The table is 300 tight points, 60 wide ones, and 10 on a ring of radius 4.5,
scored with $k = 20$. All four find the ring (AUC between 0.907 and 0.981) and
all four are near 1 through the interior; they disagree exactly where "density"
stops being obvious.

`simplified_lof` is the cleanest comparison, because it changes one thing: it
removes the $\max$ from (2). The wide blob's score rises from 1.207 to 1.331,
because a point there is allowed to be credited for a neighbour that is close by
its own standard but far by the region's. The reachability floor is a
regulariser, and removing it makes the score more sensitive and less stable.

`lof_knn` removes the averaging instead. Its density is $1/k\text{-dist}(p)$, so
one unusually placed neighbour moves the score; the ring's mean falls from 8.39
to 6.61 and the ranking within the ring gets noisier even though the AUC is the
highest of the four here. It is the cheapest to compute — one stored distance per
point — and the least smooth.

`cluster_lof` is a different estimator, not a tweak. It runs k-means, treats each
cluster as one weighted point, and applies (4) at that resolution:

$$
\rho_c = \frac{w_c}{\left(\frac{1}{k}\sum_{c' \in N_k(c)}
\mathrm{reach}(c, c')\right)^{d}}
\qquad (5)$$

with $w_c$ the number of points in cluster $c$. Mass over volume is what makes it
a density rather than a distance ratio: a cluster with twice the points at the
same radius is twice as dense, which the bare centroid would not know. The price
is resolution. Scores are constant within a cluster — the demo asserts the spread
is exactly zero — so it cannot rank points inside one, and with only ten distinct
scores it cannot hit a 10% threshold exactly: it flags 7.6% instead. In exchange
it is robust to a cluster of anomalies that is too small to fill a point-level
neighbourhood but large enough to be its own cluster, which is precisely the case
that defeats ordinary LOF below.

## Derivation

### 1. Why a ratio, and why it is near one

For a homogeneous sample, the $k$ nearest neighbours of every point are at
comparable distances, so every $\mathrm{lrd}$ in (4) is about the same number and
the ratio is about one. That is the calibration: LOF is not a probability and its
scale is not absolute, but 1 is the value a point gets when its neighbourhood is
typical *for its own region*. The demo's uniform chain gives exactly 1 at all four
non-outlier points, and on the variants demo's 300-point tight Gaussian blob the
mean score is 1.111, against 1.207 for the deliberately wide blob beside it.

The ratio is also what makes the estimator invariant to a similarity transform.
Scaling every distance by $c$ multiplies every $\mathrm{lrd}$ by $1/c$, so the
ratio in (4) is unchanged; rotating or translating the data does not change any
distance at all. The demo asserts this to $8.9 \times 10^{-16}$, and it is worth
stating because it is *not* true of the threshold: the threshold is a quantile of
the scores, so it moves with the shape of the distribution even though the scores
do not.

### 2. The reachability floor

Without the floor, $\mathrm{lrd}$ is $1/\text{mean } d$, which is a density
estimate with a bandwidth equal to the neighbourhood radius. That is fine inside
a homogeneous region and bad at a boundary, where the mean includes one short and
$k-1$ long distances. The floor replaces each short distance with the
neighbour's own $k$-distance, which is the radius at which that neighbour's
density is defined. The result is a density that is continuous across a density
boundary rather than jumping at it, which is why LOF gives the same answer on
both sides of a sharp edge.

The cost is that the floor discards information: two points at distances 0.1 and
1.0 from a neighbour whose k-distance is 2.0 are treated identically. That is the
intended behaviour — they are both inside the neighbour's own scale — and it is
also why `simplified_lof` can be sharper on data with genuine fine structure
inside a neighbourhood.

### 3. Why the mean is over the neighbours' densities

Equation (4) compares $p$'s density with the mean density *of the points $p$ is
compared with*, not with the density of the region. The two differ when the
neighbourhood is asymmetric: a point on the edge of a cluster has most neighbours
inside the cluster, so its reference is the cluster's density, and it is flagged
if it is sparser than the cluster even though the empty side is not represented.
This is the definition of "local": the reference set is the neighbourhood, not a
fixed radius. It also means the score depends on the density gradient — a point
on a slope is compared with points uphill of it.

### 4. Choosing k

$k$ controls both the variance and the scale of the comparison. Small $k$ makes
the score noisy, because the mean in (3) is over few distances and a single
neighbour can dominate; large $k$ smooths the score but compares a point with a
region so wide that "local" stops meaning anything. There is no closed-form
optimum, and the paper's 20 is a default rather than a result. The failure-mode
section shows the one case where the choice is structural rather than a
trade-off: if an anomaly group has at least $k$ members, the group supplies its
own reference and disappears.

### 5. The threshold

`fit` stores the $1 - \text{contamination}$ quantile of the training scores, and
`predict` returns $-1$ when a score exceeds it. The parameter is a budget, not a
detector, exactly as in isolation forest: 10% contamination flags 10% of any
sample, including pure noise. The difference is that LOF's scores are ratios near
1 with a right tail, so the quantile is usually close to 1 and the flag set is
"the points whose neighbourhood is sparsest relative to its own surroundings".
That is a more meaningful flag than a raw distance threshold, and it is still not
a statement about the true anomaly rate.

### 6. The cluster variant

`cluster_lof` applies (4) to micro-clusters. Each cluster is a point with a mass,
and (5) converts the mass and the mean reachability into a density. Two details
matter. The exponent $d$ is the dimension: volume grows like radius to the power
$d$, so without it a cluster of twice the mass at twice the radius would look
equally dense in any dimension, which is wrong above one dimension. And the
neighbour search is over clusters, so $k$ in the cluster variant is a number of
clusters, not points; the default `n_clusters = 8` makes `k = min(n_neighbors, 7)`
in practice, which is why the demo passes `n_clusters = 10` for a $k$ of 9.

## Code walkthrough

### 1. Neighbourhoods

```python
    distances = D.copy()
    if exclude_self:
        np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(distances, order, axis=1)
```

The `exclude_self` flag is the whole reason this is a function and not two lines.
For a square matrix of the training points against themselves, the diagonal is
zero and must be masked before the sort, or every point's nearest neighbour is
itself and the k-distance is zero. For a query against the training set, the
matrix is not square and row $i$'s zero entry is the distance to training point
$i$ — masking it would silently drop a genuine neighbour. Passing the flag
explicitly, rather than inferring it from the shape, is what keeps `score_samples`
correct on both paths. `stable` sorting makes ties deterministic, which the
demo's exact assertions depend on.

### 2. The three point-level densities

```python
        if self.variant == "lof":
            reach = np.maximum(dists, self.kdist_[self.neighbours_])
            spread = reach.mean(axis=1)
        elif self.variant == "simplified_lof":
            spread = dists.mean(axis=1)
        else:  # lof_knn keeps only the k-th distance
            spread = self.kdist_
        return 1.0 / np.maximum(spread, self._eps)
```

Equations (3) and its two variants, in the order of the table. The indexing is
the subtle part: `self.kdist_[self.neighbours_]` is an $(n, k)$ array whose entry
$i, j$ is the k-distance of neighbour $j$ of point $i$. That is (2)'s
asymmetry — the floor comes from the *neighbour* — and transposing it, or using
`self.kdist_[:, None]`, would floor each reach by the point's own k-distance,
which is a different estimator and is not LOF. The `np.maximum` against `_eps` is
the only numerical guard: duplicate points give zero distance, and a density of
infinity would make every ratio involving them `nan`.

### 3. The factor

```python
        neighbour_density = density[self.neighbours_]
        return neighbour_density.mean(axis=1) / density
```

Equation (4) in two lines, and the shape is the point: `density[self.neighbours_]`
is $(n, k)$, the mean over the $k$ neighbours is the numerator, and dividing by
the $(n,)$ vector broadcasts the denominator across it. The same expression
serves both levels of the estimator — points and clusters — because both are
"density array plus neighbour index array". That is the design decision that
keeps `cluster_lof` from being a separate class.

### 4. Fitting

```python
            D = _distances(X)
            scale = float(np.median(D[~np.eye(n, dtype=bool)]))
            self._eps = 1e-12 * max(scale, 1e-12)
            self.neighbours_, dists = _neighbourhoods(D, self.n_neighbors,
                                                      exclude_self=True)
            self.kdist_ = dists[:, -1]
            self.lrd_ = self._density(dists)
            self.scores_ = self._factor(self.lrd_)
```

The order matters: distances, then the numerical floor, then neighbourhoods. The
floor is $10^{-12}$ times the median off-diagonal distance, not a fixed constant,
so that the guard is scale-free — the same code works on data measured in
nanometres and in light years. `dists[:, -1]` is the k-distance because the
neighbours are sorted, and `self.kdist_` is stored rather than recomputed because
`score_samples` needs the *training* k-distances to floor a query's reachability
against.

### 5. k-means for the cluster variant

```python
    centres[0] = X[rng.integers(n)]
    closest = _distances(X, centres[:1])[:, 0] ** 2
    for j in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            centres[j] = X[rng.integers(n)]
        else:
            centres[j] = X[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, _distances(X, centres[j:j + 1])[:, 0] ** 2)
```

k-means++ seeding: the first centre is uniform, each next one is drawn with
probability proportional to its squared distance to the nearest centre chosen so
far. That is what makes the clustering reproducible enough to assert properties
about and good enough that ten clusters on the demo's data are stable across
runs. The `total <= 0` branch is the degenerate case where every point coincides
with a centre; without it the probabilities are `nan` and `rng.choice` raises a
confusing error much later. The Lloyd loop that follows is the ordinary one, with
an empty cluster re-seeded at the point furthest from its centre.

### 6. Cluster density

```python
        reach = np.maximum(dists, self.kdist_[self.neighbours_])
        # Mass per unit volume: a cluster holding twice the points at the same
        # radius is twice as dense, which a bare centroid would not know.
        density = weights / np.maximum(reach.mean(axis=1), self._eps) ** d
        self.lrd_ = density
        self.cluster_scores_ = self._factor(density)
        self.labels_ = labels
        self.scores_ = self.cluster_scores_[labels]
```

Equation (5). `weights` is `np.bincount(labels)`, so the mass is exact and not a
weighted average of points. The `** d` is the volume term, and it is the one line
that makes the variant dimension-aware. `self.scores_` broadcasts the cluster
factor back to points through the labels, so the training scores and the
per-cluster scores stay consistent; `score_samples` then assigns a query to its
nearest centre, which is why a query's score is exactly the cluster's and why the
spread inside a cluster is zero.

## Complexity

For $n$ points in $d$ dimensions, $k$ neighbours, $m$ queries, $c$ clusters and
$T$ Lloyd iterations:

| stage | time | memory |
|:--|:--|:--|
| pairwise distances | $O(n^2 d)$ | $O(n^2)$ |
| k nearest neighbours | $O(n^2 \log n)$ implemented, $O(n^2 \log k)$ with selection | $O(nk)$ |
| lrd and LOF | $O(nk)$ | $O(n)$ |
| query score | $O(mn(d + \log k))$ | $O(n)$ |
| k-means, `cluster_lof` | $O(T n c d)$ | $O(nc)$ |
| cluster distances | $O(c^2 d)$ | $O(c^2)$ |

The quadratic distance matrix is the honest cost of this implementation and the
reason production LOF uses a k-d tree or a ball tree; with an index, the
neighbour search is $O(n \log n)$ in low dimension and the matrix is never
materialised. The cluster variant is the exception: after k-means, everything
happens on $c$ centroids, so its cost is dominated by the assignment step and is
linear in $n$ for fixed $c$. That is also why it scales to data where point-level
LOF cannot be built — at the price of the resolution documented above.

## Failure modes

### Masking: a group that fills its own neighbourhood

This is the one to know, and it is structural rather than numerical. Take thirty
anomalies packed within 0.15 of one point, at radius 6 from a normal blob, and
score with $k = 15$: they average 1.08, AUC 0.439, and only 2 of 30 are flagged.
The same thirty points spread at radii 7 to 9 average 7.66, AUC 1.000, and all 30
are in the top 30. The mechanism is in (4): if the group has at least $k$ members,
every member's neighbour set is made of group members, so its density is compared
with the group's own density and the ratio is near one. The group is its own
reference and looks normal.

The demo also measures the boundary of the condition: eight points packed just as
tightly, scored with $k = 20$, average 8.78 — they cannot hide, because 13 of
their 20 neighbours are blob points. So masking is a relation between the group
size and $k$, not a property of tightness. Lowering $k$ makes it worse, and the
defence is a $k$ larger than the plausible group size, or a variant that does not
let a group be its own reference — `cluster_lof` at a finer resolution, or the isolation forest above,
which counts the group as a cluster of points but still has to separate it from
the blob.

### Duplicates are invisible

With $k = 1$, a duplicated row and its original are each other's only neighbour,
both k-distances are zero, the reachability floor in (2) is zero, and the ratio
is exactly 1.0. With $k = 5$ the pair scores 1.110 against a threshold of 1.205.
Either way the copy is not flagged: two coincident points make each other look
normal, which is masking at the smallest possible scale. Deduplicate first; a
near-duplicate burst in a log is exactly this case, and it is the reason a
production pipeline should not treat LOF's silence as evidence.

### The distance metric is a modelling decision

LOF uses Euclidean distance, so it mixes units. A column measured in thousands
dominates one measured in fractions, and two columns with different variances
are effectively weighted by their scales. Unlike a density estimator on a
standardised space, there is no bandwidth to absorb it. Standardise or whiten
before fitting, and remember that (4) is invariant to a global scale but not to
per-column scales — the invariance that makes the score dimensionless does not
make it unit-free.

### High dimensions

Above roughly ten dimensions the distances to every point concentrate, every
neighbourhood has the same radius, and (4) converges to 1 for every point. LOF
does not fail loudly: it returns a plausible set of numbers near one and a
threshold that flags an arbitrary 10%. This is the same curse that defeats the
KDE at the start of the chapter, and the local methods do not escape it; they
only postpone it. If a 20-dimensional LOF run separates nothing, that is usually
the reason, not a bug.

### The threshold is still a budget

`contamination` sets the flag rate, not the detection rate. On the masking data,
LOF's AUC is 0.439 — worse than chance, because the tight group's scores are
below the blob's — while `predict` still flags 10% of the points, chosen from the
least-dense blob points. A high flag rate and a bad ranking coexist happily, and
only the AUC (or a labelled sample) tells them apart.

## Applications

- **Intrusion detection.** The original application: network connections have
  wildly different baselines per host and service, and a login rate that is
  anomalous for one server is normal for another. LOF's local reference is the
  host's own history, not the fleet's.
- **Fraud and account takeover.** A transaction is judged against the account's
  recent behaviour rather than against all accounts; the ratio cancels the
  account's scale, which is what a global threshold cannot do.
- **Manufacturing and sensor faults.** A machine's vibration signature is dense
  in its healthy region; LOF flags excursions from its own operating point, and
  the wide-normal-region case is handled without flagging every point in a
  naturally noisy machine.
- **Feature construction.** The LOF score, the k-distance and the neighbour
  count are useful features for a supervised model even when LOF is not the
  detector, in the same way as the isolation-forest path lengths.
- **As a cross-check.** LOF and isolation forest disagree in a informative way:
  the forest is global and saturates, LOF is local and masks groups. Running both
  and inspecting the disagreements is a cheap way to find the cases where the
  answer depends on the definition of anomalous.
- **Where it is a trap.** Clustered anomalies, duplicates, and anything with more
  than about ten effective dimensions, per the failure modes above. For those,
  angle-based or isolation-based scores are the better default.

## References

- M. M. Breunig, H.-P. Kriegel, R. T. Ng and J. Sander, *LOF: identifying
  density-based local outliers*, ACM SIGMOD International Conference on
  Management of Data (2000). The algorithm, the reachability distance in (2) and
  the factor in (4).
- E. Schubert, A. Zimek and H.-P. Kriegel, *Local outlier detection reconsidered:
  a generalized view on locality with applications to spatial, video, and network
  outlier detection*, Data Mining and Knowledge Discovery 28(1):190–237 (2014).
  Why local density is a family of estimators and where LOF sits in it.
- H.-P. Kriegel, P. Kröger, E. Schubert and A. Zimek, *LoOP: local outlier
  probabilities*, ACM Conference on Information and Knowledge Management (2009).
  The probabilistic rescaling of a local score, and the instability of unbounded
  ratios like (4).
- L. Duan, L. Xu, Y. Liu and J. Lee, *Cluster-based outlier detection*, Annals of
  Operations Research 168(1):151–168 (2009). The cluster-level factor behind
  `cluster_lof`, and the mass-over-volume reading of density.
- F. Angiulli and C. Pizzuti, *Fast outlier detection in high dimensional
  spaces*, European Conference on Principles of Data Mining and Knowledge
  Discovery (2002). The k-nearest-neighbour distance score used as the global
  baseline in the demo.
- M. Goldstein and S. Uchida, *A comparative evaluation of unsupervised anomaly
  detection algorithms for multivariate data*, PLoS ONE 11(4):e0152173 (2016).
  Where LOF wins, where it loses to isolation forest, and how sensitive the
  ranking is to $k$.
- V. Chandola, A. Banerjee and V. Kumar, *Anomaly detection: a survey*, ACM
  Computing Surveys 41(3):1–58 (2009). The taxonomy that separates local from
  global scores, and the masking and swamping vocabulary.
