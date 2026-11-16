# k-nearest neighbours

`knn.py` · Kernels & instance-based · Tier 1 · 9 variants

## Intuition

`fit` does nothing. Prediction scans the training set for the `k` closest
points and lets them vote (or averages them). There is no parameter vector, no
loss to minimise, no gradient — the training set *is* the model.

With `k = 1` the decision boundary is the Voronoi tessellation of the training
points: every cell is labelled by the point at its centre.

```
        class 0                 class 1
      ●     ●                ○       ○
        ╲   ╱                 ╲     ╱
   ●─────●─────●───────────────○───○─────○
        ╱   ╲                 ╱     ╲
      ●     ●                ○       ○
              ↑ boundary is an edge of the Voronoi diagram
```

That is also the whole problem. A single noisy point owns a whole cell, so
`k = 1` has zero training error and a jagged boundary. Raising `k` averages
away the noise and blurs the boundary — the entire bias-variance trade-off of
this atom is one integer.

The cost model is inverted relative to every other atom in the book: training
is free and prediction is `O(nd)`. That is why most of the interesting work
here is not in the vote but in the two things that make the scan survivable —
spatial trees and pruning the training set itself.

## Variants

| Variant | What changes | Why you would pick it |
|:--|:--|:--|
| `classification` | majority vote over `k` | labels, not numbers |
| `regression` | mean of the `k` targets | local averaging; no linearity assumption |
| `uniform` | every neighbour counts 1 | simplest, and robust to a single very close point |
| `distance_weighted` | weight `1/d` | smooths the discontinuity when a neighbour enters or leaves |
| `kd_tree` | axis-aligned median splits | low `d`, many queries, exact search |
| `ball_tree` | nested balls, radius pruning | clustered data; better than kd for `d ≳ 10` |
| `cosine` | `1 − cos(x, z)` | text, embeddings; direction matters, length does not |
| `edited_knn` | drop points its neighbours disagree with | label noise in the training set |
| `condensed_knn` | keep only decision-changing points | storage and query cost; smaller store, same boundary |

## Derivation

### Why it works

For `k = 1`, let `R*` be the Bayes error. Cover and Hart showed that the
`1`-NN error `R` satisfies

$$
R^* \le R \le 2R^*\left(1 - \frac{k}{2}R^*\right)^{-1} \xrightarrow{k=1} 2R^*,
$$

so half the information in an infinite sample is captured by the single nearest
neighbour. Consistency needs more: Stone's theorem requires `k → ∞` and
`k/n → 0`, which is the honest statement of the bias-variance trade — the cell
must shrink (so the label is constant inside it) and must contain many points
(so the vote is stable), and in a fixed dimension both can happen at once.

In dimension `d` the cell radius shrinks like `n^{-1/d}`, so the bias is
`O(n^{-2/d})` and the variance `O(1/k)`:

$$
\mathbb{E}[\text{error}] \approx R^* + c_1 n^{-2/d} + c_2/k ,
\qquad k^\star \sim n^{4/(d+4)} .
$$

The exponent is the curse in one symbol: the sample size needed for a fixed
cell radius grows exponentially in `d`. Concretely, to have `m` points inside a
ball of radius `r` in `[0,1]^d` you need `n ≈ m/r^d`.

### Distance concentration

The same fact destroys the ranking, not just the rate. For i.i.d. coordinates,
`(d_max − d_min)/d_min → 0` as `d → ∞`, so the nearest and the farthest point
become equally far and "nearest" loses its meaning.

The diagnostic that matters is the **hubness** ratio: for each query, compare
the `k`-th neighbour distance to the 1st. A ratio near 1 means every point is
equally close — the vote is a coin flip dressed up as a model.

### kd-tree: prune with the splitting plane

Build by splitting on the axis with the largest spread, at the median (so the
tree is balanced and depth is `≈ log₂ n`). A query descends to the leaf that
contains `x`, keeping a max-heap of the `k` best distances seen. Then it
backtracks, and here is the only idea in the structure: the whole far subtree
lies on the far side of one axis-aligned plane, so every point in it is at
least `|x_axis − value|` away. If that is larger than the current `k`-th best
distance, the subtree cannot contain a neighbour and is skipped unvisited.

```
   │            query ●
   │        ·   │
   │     ·      │      ← gap = |x[axis] − value|
   │  ·         │
   └────────────┼──────────────
        near    far: skip if gap² ≥ kth²
```

The expected query cost is `O(log n)` only in fixed dimension. In `d` dimensions
the ball of radius `k`-th distance overlaps `n^{1-1/d}` cells on average, so the
cost degrades to that — at `d = 20` a kd-tree is a linear scan with extra steps.
That is why the demo reports the *share* of brute force rather than a speedup.

### Ball tree: prune with a radius

Each node stores a centre `c` and a radius `r` that contains all of its points.
A query can then skip a node entirely when

$$
\|q - c\| - r \;\ge\; \text{(current k-th best distance)} .
$$

The bound is looser than an axis-aligned plane in low dimension but much
tighter when the data is clustered, and it does not degrade as fast with `d`
because it adapts to the local shape. Building uses the two-farthest-point
heuristic: pick the point farthest from the centroid, then the point farthest
from that one, and split by which is closer.

### Distance weighting

`uniform` is a step function of the geometry: a point crossing the `k`-th
distance boundary changes the vote discontinuously. Weighting by `1/d` removes
the jump and biases the vote towards the closest neighbour, at the cost of
letting one point with `d ≈ 0` decide alone (hence the `EPS` floor).

It also changes the effective `k`. With `1/d` weights on well-spread data the
vote behaves like a much smaller neighbourhood; the demo shows a case where
`1/0.1 = 10` outweighs five neighbours at `d ≈ 0.9`.

### Editing: attack the training set

**Wilson editing** removes every point whose `k` neighbours (excluding itself)
disagree with its label, iterating until stable. Noise points have label
inconsistent neighbours and go; the boundary sharpens because the noisy points
were voting. The failure is asymmetric and worth knowing: in an imbalanced
problem, *rare-class* points often look like noise, so editing can delete the
class you care about. Edit the majority class only, or edit per class.

**Hart's condensed nearest neighbour** keeps the smallest store that still
classifies the training set correctly with `1`-NN. Start with one stored point,
scan the data; every misclassified point is added. A correctly classified point
never needs to be added, because the store only grows — the decision boundary
can only move towards the point, never away. Termination is guaranteed
(each pass adds at least one point or stops), the store is consistent by
construction, and the price is that a single pass is `O(n²)` and the result
depends on the scan order.

## Code walkthrough

Three metrics, and the Euclidean one uses the expansion rather than a broadcast
subtraction — `(n, m, d)` temporaries are how a nearest-neighbour search runs
out of memory before it runs out of time:

```python
    if metric == "euclidean":
        d2 = (X * X).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2.0 * X @ Z.T
        return np.sqrt(np.maximum(d2, 0.0))
    if metric == "manhattan":
        return np.abs(X[:, None, :] - Z[None, :, :]).sum(axis=2)
    if metric == "cosine":
        xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
        zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), EPS)
        return 1.0 - xn @ zn.T
```

The cosine branch is worth pausing on: `‖x̂ − ẑ‖² = 2(1 − cos)`, so cosine k-NN
*is* Euclidean k-NN on `L2`-normalised rows. The variant exists as a metric
because normalising up front is cheaper than remembering to do it.

The kd-tree backtrack is the prune, in one line. `heap[0][0]` is `−dist`, so
squaring it recovers the `k`-th best *squared* distance:

```python
        self._query(near, x, k, heap)
        gap = x[axis] - node["value"]
        # the plane is the only thing the far subtree can be closer than
        if len(heap) < k or gap * gap < heap[0][0] ** 2:
            self._query(far, x, k, heap)
```

Getting that comparison wrong in either direction is silent: too aggressive and
the search returns wrong neighbours, too conservative and it is merely a slow
linear scan. The demo pins it down by requiring *identical* neighbour indices
against brute force, not just similar accuracy.

The ball-tree prune is the same shape with a different bound, and the early
return is what makes the structure worth its build cost:

```python
        if len(heap) >= k:
            bound = float(np.sqrt(((x - node["centre"]) ** 2).sum())) - node["radius"]
            if bound > -heap[0][0]:
                return                      # the whole ball is farther than k
```

The vote is a scatter over classes. `np.bincount`-style logic in matrix form
keeps the tie-break deterministic: `argmax` takes the lowest class index, which
matters when `k` is even and the vote is split.

```python
    def _vote(self, dist: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Weighted vote; ties go to the lowest class index, deterministically."""
        w = 1.0 / np.maximum(dist, EPS) if self.weights == "distance" else np.ones_like(dist)
        score = np.zeros((idx.shape[0], self.classes_.size))
        for c in range(self.classes_.size):
            score[:, c] = (w * (self.y_enc_[idx] == c)).sum(axis=1)
        return self.classes_[np.argmax(score, axis=1)]
```

Editing is a fixpoint loop, and the `pool` is recomputed every round because
removing points changes every subsequent vote:

```python
        keep = np.ones(self.X_.shape[0], dtype=bool)
        for _ in range(self.max_rounds):
            pool = np.flatnonzero(keep)
            drop = np.array([keep[i] and self._label_at(i, pool, self.k + 1) != self.y_enc_[i]
                             for i in range(self.X_.shape[0])])
            if not drop.any():
                break
            keep &= ~drop
```

Condensing is the mirror image — a store that only grows:

```python
                if label != self.y_enc_[i]:
                    store.append(i)
                    added += 1
```

## Complexity

| Path | Fit | Predict (per point) | Memory |
|:--|:--|:--|:--|
| brute force | `O(1)` | `O(nd)` | `O(nd)` |
| kd-tree build/query | `O(n log n)` | `O(log n)` in fixed `d`, `O(n^{1-1/d})` asymptotically | `O(nd)` |
| ball-tree build/query | `O(n log n)` | same order, better constants on clustered data | `O(nd)` |
| `edited_knn` | `O(r·n²)` for `r` rounds | unchanged | `O(nd)` |
| `condensed_knn` | `O(p·n²)` for `p` passes | `O(sd)` with `s` stored points | `O(sd)` |

Trees trade build time for query time; they pay off only when the number of
queries is much larger than the number of points, which is exactly the
inference-server regime.

## Failure modes

**Unscaled features.** Euclidean distance sums squared differences over
dimensions, so a feature measured in thousands dominates one measured in
fractions. Standardise first; this is not optional and there is no `k` that
fixes it.

**High dimension.** Past `d ≈ 20`, the neighbour ranking is noise: distances
concentrate and the vote approaches the class prior. Diagnostic: the ratio of
the `k`-th to the 1st neighbour distance, or simply the gap between train and
test accuracy (`k=1` giving 1.0 train and 0.6 test is a symptom, not a result).
Reduce dimension first — PCA, or an embedding.

**Imbalance.** The majority class occupies most of the space, so most cells
belong to it. `k=5` on a 95/5 split predicts the majority almost everywhere.
Fix by weighting votes by inverse class frequency, or by editing *within* each
class rather than globally.

**Even `k` in a two-class problem.** Ties are resolved by class index, which is
an arbitrary ordering. Use odd `k`, or distance weighting, or accept that
roughly `(k/2)`-sized neighbourhoods are decided by an accident of `np.unique`.

**Editing the wrong set.** `edited_knn` must see only training data. Editing the
combined train+test set to "clean" the evaluation is leakage and inflates every
number that follows.

**`k = n`.** The model predicts the global majority class everywhere. This is
the correct limit of the algorithm and a useful sanity check, not a bug.

## Applications

The default baseline for recommender systems ("users who bought this also
bought…") is item-item nearest neighbours with a cosine metric, and it is still
hard to beat at small scale.

Distance to the `k`-th neighbour is a density estimate, which makes k-NN the
backbone of anomaly detection and of the `k`-NN graph construction used by
spectral clustering, UMAP, and every manifold-learning method in chapter 08.

Imputation, record linkage, and duplicate detection are all nearest-neighbour
problems in disguise — usually with a learned metric (`metric_learning.py`)
because the default one is the whole difficulty.

It is a trap when `n` is large and queries are few (build a model instead), when
the data is high-dimensional without a meaningful metric, and whenever the
answer must be smooth in the inputs rather than piecewise constant.

## References

- Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Trans. Information Theory* 13(1).
- Stone, C. (1977). Consistent nonparametric regression. *Annals of Statistics* 5(4).
- Friedman, J., Bentley, J. & Finkel, R. (1977). An algorithm for finding best matches in logarithmic expected time. *ACM TOMS* 3(3).
- Omohundro, S. (1989). Five balltree construction algorithms. ICSI technical report.
- Wilson, D. (1972). Asymptotic properties of nearest neighbor rules using edited data. *IEEE Trans. Systems, Man and Cybernetics* 2(3).
- Hart, P. (1968). The condensed nearest neighbor rule. *IEEE Trans. Information Theory* 14(3).
- Devroye, L., Györfi, L. & Lugosi, G. (1996). *A Probabilistic Theory of Pattern Recognition*, ch. 5–11. Springer.
