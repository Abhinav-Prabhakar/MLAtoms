# Isolation forest

`isolation_forest.py` · Density & anomaly · Tier 1 · 4 variants

## Intuition

Cut the data set in half at random, repeatedly. A point in the middle of a cloud
survives many cuts because there is always more cloud around it; a point on the
edge, or in a gap, is alone after three or four. Count the cuts, and you have an
anomaly score that never estimates a density, never computes a distance and
never looks at a neighbour list.

The algorithm, in four lines:

$$
\text{split a node by projecting on a random } n,\ \text{cutting at }
u \sim \mathrm{Uniform}[\min_i x_i \cdot n,\ \max_i x_i \cdot n]
\qquad (1)$$

$$
h(x) = \text{depth of the leaf } x \text{ lands in} + c(\text{size of that leaf})
\qquad (2)$$

$$
c(m) = 2H_{m-1} - \frac{2(m-1)}{m}, \qquad H_k = \sum_{j=1}^{k} \frac{1}{j}
\qquad (3)$$

$$
s(x) = 2^{-E[h(x)] / c(\psi)}, \qquad \psi = \text{points per tree}
\qquad (4)$$

Equation (2) is the only subtle one. A tree stops when its node has one point, or
when it hits the depth limit $\lceil \log_2 \psi \rceil$; if it stopped at every
single point the trees would be too deep to build. The leaf's remaining points
are all isolated at the same depth, so (2) adds the depth that *would* have been
needed to separate them on average, which is $c(m)$ — the expected isolation
depth of a random BST on $m$ points. A leaf of one point adds nothing, and a leaf
of a hundred adds about seven levels the tree did not build.

Equation (3) is the normaliser that turns a path length into a calibrated
number. For a typical
point, $E[h] \approx c(\psi)$, so the exponent is about $-1$ and $s \approx
\tfrac{1}{2}$; for an isolated point $E[h] \approx 0$ and $s \approx 1$. The score
is bounded, monotone and — this is the useful part — comparable across data sets
of different sizes, because both the numerator and the denominator grow like the
logarithm of the sample size. A threshold of 0.6 means something; a mean path
length of 9 does not.

```
   depth 0   ────────────────┬──────────────────────
                             │
   depth 1        ┌──────────┴─────────┐
                  │                    │
   depth 2    ────┴─────         ──────┴──────
                  │                    │
   depth 3        x               ┌────┴────┐
                                  │         │
                               · · · ·   · · · ·    x is alone after 3 cuts
```

Two structural choices follow, and they are the two variants worth knowing about.
The cuts are **axis-parallel** by default: every split is orthogonal to a
coordinate axis, so a point diagonal to the cloud needs more cuts than a point on
an axis at the same distance. That is a bias, and `extended` removes most of it
by cutting with random hyperplanes instead. The trees are also **subsampled**:
each one sees $\psi = 256$ points, not the data set, because a tree grown on a
million points spends its depth resolving density that no anomaly cares about.
`max_samples` exists so that the cost of that choice can be measured instead of
assumed.

## Variants

| variant | split direction | points per tree | output | measured |
|:--|:--|:--|:--|:--|
| `iforest` | one axis (1) | 256 | $s \in (0,1]$ | AUC 1.000, threshold 0.481 |
| `extended` | hyperplane, $\ell+1$ axes | 256 | $s \in (0,1]$ | AUC 1.000, angular bias −64% |
| `anomaly_score` | one axis | 256 | $s - \text{threshold}$ | same AUC, 10.0% flagged |
| `max_samples` | one axis | the full sample | $s \in (0,1]$ | AUC 1.000, planted score 0.738 |

The detection table is 300 standard normal points with 10 planted on a ring of
radius 5.5 to 6, contamination 0.1. All four variants find the ring — it is not
a hard problem — and the differences are in what the number means rather than in
whether it works.

`anomaly_score` is the same computation with the score shifted by the training
threshold: the furthest planted point scores 0.723 under `iforest` and 0.242
under `anomaly_score`, and 0.481 is the threshold, so zero is the decision
boundary and `predict` is the sign. Exactly 10.0% of the training set is flagged,
because the threshold *is* that quantile. That is worth stating plainly: the
contamination parameter does not detect anything, it sets the false-positive
budget, and a forest with `contamination=0.1` will flag ten percent of any data
set, including one with no anomalies in it.

`max_samples` changes the estimator, and the effect is measurable in two places.
With $\psi = 16$ the AUC falls to 0.958 and the seed-to-seed spread rises to
0.020: a tree of sixteen cannot separate a ring point from the edge of a
600-point cloud. At 256 the AUC is 1.000 and the spread is 0.000, and the full
sample — 612 points, so 2.4 times the work per tree — gives the same AUC without
moving the spread off zero. The second effect is on the scale of the score: the
planted points average 0.616 at $\psi = 16$, 0.680 at 64, 0.728 at 256 and
0.738 at the full sample, because $E[h]$ changes with the tree size while
$c(\psi)$ is the normaliser. A threshold fitted at one sample size is not a
threshold at another — which is why the variant normalises by `max_samples_`
and not by a constant.

## Derivation

### 1. The tree

A node holds a set of points. If it holds one point, or the depth limit is
reached, it becomes a leaf. Otherwise a direction $n$ and an offset $u$ are
drawn, and the points are split by the sign of $x \cdot n - u$. For
`iforest`, $n$ is a coordinate axis and $u$ is uniform between the smallest and
largest projection in the node — the standard rule, and the one that makes the
tree's structure independent of the data's density inside the node. For
`extended`, $n$ is a random hyperplane normal.

The recursion terminates with probability one because a uniform offset strictly
between the minimum and maximum projection separates at least one point on each
side, and the depth limit caps the work regardless. Two degenerate cases are
handled by returning a leaf: a node whose projections are all equal (duplicates,
or a direction along which the node has no extent) and a node of size one.

### 2. The normaliser

$c(m)$ is the expected depth at which a random cut sequence isolates one of $m$
points, equivalently the expected depth of an unsuccessful search in a random
binary search tree. It satisfies

$$
c(1) = 0, \qquad c(m) = c(m-1) + \frac{2}{m}
\qquad (5)$$

whose solution is (3): the harmonic sum, so $c(m) \approx 2\log m + 2\gamma - 2$
for large $m$. The demo checks the recurrence for every $m$ up to 256 and checks
$c(2) = 1$ exactly; at $m = 256$ the exact value is 10.2487 against 10.2448 for
the $\log$ approximation, a gap of $3.9 \times 10^{-3}$ that is the harmonic
sum's tail.

The choice to compute the harmonic sum exactly rather than use the log form is
not pedantry. Leaf sizes in a real tree are small — the depth limit guarantees
it — and at $m = 2$ the exact value is 1 while the approximation is 0.154. The leaf correction is added to every query, so an error there is a
systematic offset in every score.

### 3. The score, and what "calibrated" means

$s$ in (4) is a monotone decreasing map from mean path length to $(0, 1]$, with
$s = 0.5$ at $E[h] = c(\psi)$. It is not a probability, despite the range: the
demo's forest scores a standard normal cloud at 0.447 on average and 0.401 at the
centre, both under the 0.5 that the normaliser suggests, because a tree is capped
at $\lceil \log_2 \psi \rceil$ levels and the leaf correction only approximates
the depth an unlimited tree would have used. What the scale buys is
comparability: doubling the sample size changes $E[h]$ and $c(\psi)$ by the same
logarithm, so the score of a point stays put while a raw path length would grow.

The saturation in the demo's ray — 0.551 at radius 2, 0.660 at radius 5 and the
same 0.660 at radius 9 — is the same fact seen from the other side. Isolation
counts cuts, not distance, so once a point is outside the cloud its score stops
growing. An isolation forest ranks "unusual" but does not measure "how far".

### 4. Why axis-parallel cuts are biased

Every split is perpendicular to an axis, so a region that is diagonal to the
coordinate frame is harder to cut than one that is aligned with it. For a point
at fixed distance from a cloud, the number of cuts needed depends on the angle.
The demo measures it on 720 points of a circle of radius 4 around a normal
cloud: the score runs from 0.590 to 0.759 around the circle under `iforest`, a
spread of 0.169, and from 0.652 to 0.712 under `extended`, a spread of 0.060.

Extended isolation forest replaces the axis with a random normal vector:

$$
n = \frac{g}{\lVert g \rVert}, \qquad g_j \sim N(0,1) \text{ on }
\ell + 1 \text{ coordinates}, \quad g_j = 0 \text{ otherwise}
\qquad (6)$$

with the offset drawn uniformly in the projection range as before. The extension
level $\ell$ controls how many coordinates participate: $\ell = 0$ has one
non-zero coordinate and reproduces the axis-parallel forest exactly, and
$\ell \ge d - 1$ is a fully extended hyperplane. In two dimensions the demo's
default of 1 is already full. The remaining 0.060 of variation is six times the
seed-to-seed noise at a fixed point, so a little anisotropy survives: some of it
is the split geometry and some is the 400-point cloud's own asymmetry, which no
orientation of the cuts can average away.

### 5. Subsampling, swamping and masking

The sample size per tree is a variance–cost trade. With $\psi$ points the tree
has depth $O(\log \psi)$ and the score's variance over trees falls like
$1/\sqrt{t}$, where $t$ is the number of trees — so $\psi = 256$ and $t = 100$
is the published combination, and the demo's sweep shows why: at $\psi = 16$ a
tree cannot see enough of the cloud to separate its edge from a planted point,
and the AUC drops to 0.958.

The two failure modes the subsampling is meant to avoid are *swamping* and
*masking*. Swamping is a normal point being scored anomalous because the
anomalies changed the path lengths around it; masking is several anomalies
hiding one another because a region containing them looks, to a tree, like a
region with more points in it. Both are consequences of the score being a
property of the sample rather than of the point, and both are visible in the
demo's duplicate experiment: adding a copy of a point that is flagged at
contamination 0.1 makes both copies unflagged, because the pair now looks like a
small cluster. The density-based detectors in the rest of this chapter have the
same response, which is a reminder that "isolation" is not the same as
"density" but is still a property of the sample.

## Code walkthrough

### 1. The normaliser, exactly

```python
    if size <= 1:
        return 0.0
    harmonic = sum(1.0 / k for k in range(1, int(size)))
    return 2.0 * harmonic - 2.0 * (size - 1) / size
```

Equation (3), with the harmonic sum done rather than approximated, and the whole
function memoised with `functools.cache` because it is called once per leaf per
query. The `int(size)` is defensive: leaf sizes come from `len(indices)` and are
already Python ints, but a caller passing a numpy integer would otherwise make
`range` fine and the cache key a different object. Returning zero for
`size <= 1` is what makes a single-point leaf contribute exactly its depth.

### 2. Choosing the split direction

```python
        aim = 1 if self.variant != "extended" else min(
            self.extension_level + 1, d)
        normal = np.zeros(d)
        chosen = rng.choice(d, size=aim, replace=False)
        normal[chosen] = rng.normal(size=aim)
        length = float(np.linalg.norm(normal))
        if length == 0.0:  # measure-zero, but the caller must not divide by it
            normal[chosen[0]] = 1.0
            length = 1.0
        return normal / length
```

This is the entire difference between the first two variants, and it is four
lines: draw a Gaussian vector supported on `aim` coordinates, normalise it. For
`iforest`, `aim = 1` and the vector is a unit basis vector, so the projection is
one coordinate and the split is axis-parallel. Making the margin `aim = 1`
rather than a special case keeps the traversal — which always does a dot product
— free of a branch. The zero-norm guard is measure-zero and would be dead code
in exact arithmetic; it exists because `rng.normal(size=aim)` can return all
zeros in floating point only if every draw underflows, and a `nan` normal would
poison every path in the tree silently.

### 3. Growing a node

```python
        rows = X[indices]
        normal = self._draw_normal(rng)
        projection = rows @ normal
        low, high = float(projection.min()), float(projection.max())
        if high - low <= 0.0:
            return node  # every point projects to the same place: a leaf
        node.normal = normal
        node.offset = float(rng.uniform(low, high))
        goes_left = projection <= node.offset
```

The projection is computed once for the node and reused for both the split
decision and the child indices, which is why `indices` is kept as an array of
row numbers rather than a list of points. `<=` versus `<` puts points exactly on
the offset to the left; with a continuous uniform offset that happens only for
duplicated values on the split coordinate, where the choice is arbitrary but has
to be made consistently or the recursion would not shrink. The degenerate leaf
is the important branch: without it, a node of identical points would recurse
until the depth limit and return a tree whose paths are pure noise.

### 4. Walking a point down a tree

```python
        for tree in self.trees_:
            for i, row in enumerate(X):
                node = tree
                while node.normal is not None:
                    side = node.left if float(row @ node.normal) <= node.offset else node.right
                    node = side
                total[i] += node.depth + average_path(node.size)
```

Equation (2). The `while` is a loop rather than a recursion because the depth is
bounded by $\lceil \log_2 \psi \rceil$ and the function is called $t \times m$
times; a recursive Python call per level would dominate the runtime. `node.depth`
was stored at build time rather than recomputed, and `average_path(node.size)` is
the leaf correction — the only place where a leaf of many points differs from a
leaf of one. Note the loop is over points inside the loop over trees, not the
other way round: every point walks one tree before the next tree is touched, so
that tree's nodes stay hot instead of being reloaded $m$ times.

### 5. The score and the variant

```python
        depth = self.path_length(X)
        return 2.0 ** (-depth / average_path(self.max_samples_))
```

```python
        scores = self._calibrated(X)
        if self.variant == "anomaly_score":
            return scores - self.threshold_
        return scores
```

Equation (4), and then the one line that makes `anomaly_score` a variant rather
than a documentation note. The threshold is fitted in `fit` as the
`1 - contamination` quantile of the training scores, so `predict` returns `-1`
exactly when the calibrated score exceeds it. Keeping `_calibrated` separate from
`score_samples` is what lets the first three variants return a probability-like
number and the fourth return a decision function without duplicating the
arithmetic.

### 6. Fitting

```python
        if self.max_samples is None:
            size = n if self.variant == "max_samples" else 256
        else:
            size = self.max_samples
        self.max_samples_ = int(min(size, n))
        self.n_features_ = d
        self.depth_limit_ = int(math.ceil(math.log2(self.max_samples_)))
```

The variant's default sample size, the cap at $n$, and the depth limit that
makes the leaf correction necessary. `max_samples_` is stored rather than
recomputed because both `average_path(self.max_samples_)` and the depth limit
depend on the *effective* size after the cap — a forest fitted on 100 points with
`max_samples=256` must normalise by $c(100)$, not $c(256)$, or every score is
systematically too high.

## Complexity

For $n$ points, $d$ dimensions, $t$ trees and $\psi$ points per tree:

| stage | time | memory |
|:--|:--|:--|
| subsample each tree | $O(t\psi)$ | $O(\psi)$ |
| build one tree | $O(\psi \log \psi \cdot d)$ | $O(\psi)$ |
| fit the ensemble | $O(t\,\psi \log \psi \cdot d)$ | $O(t\psi)$ |
| threshold | $O(nt\log\psi \cdot d)$ | $O(n)$ |
| score | $O(mt\log\psi \cdot d)$ | $O(1)$ per point |

The $d$ in the build and score rows is the dot product: the code always projects
onto a full vector, so an axis-parallel forest pays $d$ multiplications per
split where one comparison would do. Production implementations store the
feature index for the axis-parallel case, which is why their constant is so much
better in high dimension. The dependence on $n$ is only through the subsampling
and the threshold; a forest's per-tree cost does not grow with the data set,
which is the point of $\psi$.

## Failure modes

### The score saturates, so it cannot rank the far field

On the demo's ray the score is 0.551 at radius 2, 0.660 at radius 5 and 0.660 at
radius 9. The method counts cuts, and once a point is outside the cloud the
number of cuts stops changing. Consequences: an isolation forest is a good
*detector* and a poor *ranking* of how extreme the extreme points are; two
outliers a hundred and a million units away get the same score, and the "most
anomalous" point in a report may be the one that happened to sit in the sparsest
region rather than the one that is furthest out. If an ordering in the tail
matters, use a distance-based score (LOF, Mahalanobis) on the points the forest
flags, or use the forest as a filter and rank inside its output.

### Axis-parallel cuts score by direction

The circle experiment: 0.590 to 0.759 around a fixed radius, a spread of 0.169,
while the seed-to-seed noise at a fixed point is 0.011. That is a real bias — a
point at 45° to the axes is scored as less anomalous than a point on an axis at
the same distance — and it is why `extended` exists. Fully extended hyperplanes
reduce the spread to 0.060 but do not eliminate it, because the offset is still
uniform over a projection range that depends on the cloud's shape. On data whose
anomalies are isolated along diagonals (correlated sensors, image patches) the
bias systematically understates them.

### Masking, and duplicates that hide

Adding a duplicate of a point that is flagged at contamination 0.1 makes both
copies unflagged: the score moves by −0.088, and the pair now looks like a
two-point cluster rather than an isolated point. This is masking, and it is the
reason the method is not a density estimator in the way KDE is — but it is not a
defence against duplication either. Near-duplicate rows from a deduplication
step, or a burst of identical events, will lower each other's scores.

### The sample size is part of the model

`max_samples` is not a speed knob. At $\psi = 16$ the AUC on the demo's data
falls to 0.958 with a seed spread of 0.020, and the planted points average 0.680
at $\psi = 64$ against 0.728 at 256. A threshold fitted with one setting is
wrong under another; retrain, do not reuse. The published 256 is a compromise
between variance (small $\psi$) and the swamping that a large $\psi$ lets in, and
it is the number the paper's experiments support, not a theorem.

### Contamination is a budget, not a detector

`contamination=0.1` flags 10.0% of the training set by construction, on any data,
including a sample of pure noise. If the true rate is 0.1% then 99% of the
flagged points are false positives caused by the threshold rather than by the
score. Fit with a contamination that reflects the operational budget, or drop
the parameter and use the score's quantiles downstream where the cost of a false
positive is known.

## Applications

- **Tabular anomaly detection.** The default first attempt on structured data,
  because its cost is bounded by the subsample rather than by the data set,
  it handles mixed scales without normalisation, and it needs no distance
  metric. Fraud, intrusion detection,
  sensor faults and quality control all start here.
- **Feature engineering for supervised models.** The score, the mean path
  length and the per-tree depth are useful features even when the forest is not
  the detector; a supervised model on top of them often beats either alone.
- **Cheap pre-filtering.** Use the forest to discard the obvious 90% and run an
  expensive detector (a density model, a reconstruction autoencoder) on the rest.
  The saturation failure mode is harmless in this role: the filter only has to
  rank, not to measure.
- **Streaming and distributed settings.** Each tree is independent and built on
  a bounded sample, so the ensemble shards trivially and can be refit on a
  sliding window without touching the history.
  That is the practical reason it beat density-based detectors in production
  even before it beat them on accuracy.
- **Where it is a trap.** Local anomalies — a point that is normal globally and
  strange relative to its own neighbourhood — are invisible, because isolation is
  a global property of the tree. That is exactly what LOF, the next atom, is for.
  And anything where the *magnitude* of the anomaly matters, per the saturation
  failure mode.

## References

- F. T. Liu, K. M. Ting and Z.-H. Zhou, *Isolation forest*, IEEE International
  Conference on Data Mining (2008). The algorithm, the score in (4) and the
  normaliser in (3).
- F. T. Liu, K. M. Ting and Z.-H. Zhou, *Isolation-based anomaly detection*, ACM
  Transactions on Knowledge Discovery from Data 6(1):1–39 (2012). The journal
  version: the subsampling argument, the swamping and masking analysis, and the
  experiments that fix $\psi = 256$.
- S. Hariri, M. C. Kind and R. J. Brunner, *Extended isolation forest*, IEEE
  Transactions on Knowledge and Data Engineering 33(4):1479–1489 (2021). The
  hyperplane split in (6) and the axis-parallel bias measured above.
- D. E. Knuth, *The Art of Computer Programming, Volume 3: Sorting and
  Searching*, 2nd edition, Addison-Wesley (1998), §6.2.2. The expected path
  length of a search in a random binary tree, i.e. the recurrence in (5).
- C. C. Aggarwal and S. Sathe, *Theoretical foundations and evaluations of
  anomaly detection algorithms*, ACM Transactions on Knowledge Discovery from
  Data 10(1):1–43 (2015). Why isolation-based scores and density-based scores
  disagree, and on which data.
- R. Domingues, M. Filippone, P. Michiardi and J. Zouaoui, *A comparative
  evaluation of anomaly detection methods for multivariate data*, ACM Computing
  Surveys 51(5):1–36 (2018). Where the forest wins and where the local methods
  win.
