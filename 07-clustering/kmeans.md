# k-means

`kmeans.py` · Clustering · Tier 1 · 7 variants

## Intuition

k-means partitions `n` points into `k` groups by minimising the total squared
distance from each point to its group's centre:

$$
J(C,z) = \sum_{i=1}^n \|x_i - c_{z_i}\|^2
$$

Two facts make this easy to optimise approximately. If the centres are fixed, the
best assignment sends each point to its nearest centre. If the assignments are
fixed, the best centre is the mean of its points. Alternating the two never
increases `J`, so the algorithm descends to a local minimum — always a local one,
never guaranteed global, because the problem is NP-hard in general.

Everything else in this file is a response to one of k-means' practical weaknesses:

- the result depends on initialisation → `kmeans_plusplus`
- every iteration touches every point → `mini_batch`, `elkan`
- the number of clusters must be chosen → `bisecting`
- squared Euclidean distance assumes isotropic blobs → `spherical`
- a centre can end up with no points → `empty_cluster_fix`

## Variants

| Variant | What changes | When it wins |
|:--|:--|:--|
| `lloyd` | alternate assign / mean | the reference; small `n`, small `k` |
| `kmeans_plusplus` | `D²` seeding, multiple restarts | unbalanced cluster sizes |
| `mini_batch` | centres updated from 64-row batches | large `n`, streaming data |
| `elkan` | triangle-inequality bounds skip distances | many clusters, low dimension |
| `bisecting` | recursive 2-means splits | a hierarchy is wanted, or `k` grows |
| `spherical` | cosine assignment, normalised centres | direction data (text, embeddings) |
| `empty_cluster_fix` | re-seed empty clusters at worst-fit points | degenerate `k`, duplicate points |

`lloyd` is the algorithm everyone means by "k-means". `kmeans_plusplus` is the same
loop with a better start: the demo's unbalanced dataset gives a mean inertia of
`326` against `341` for random seeding, and the worst random run is much worse than
the worst `++` run. `elkan` returns *identical* labels to Lloyd — it is an exact
accelerator, not an approximation — with `1380` distance evaluations instead of
`2700`. `mini_batch` reaches the same inertia after visiting `6840` rows instead of
`9000`, and the gap widens with `n` because each batch costs `O(batch·k)`
regardless of the dataset size.

## Derivation

**Assignment step.** For fixed centres, `J` separates over points:

$$
z_i = \arg\min_j \|x_i-c_j\|^2
$$

so the optimal assignment is independent per point. This is why the distance
computation can be blocked, cached, or bounded — the decision is local.

**Update step.** For a fixed partition, `J = Σ_j Σ_{i∈S_j} ||x_i−c_j||²`, and each
`c_j` appears in its own term only. Setting the gradient to zero gives

$$
\frac{\partial}{\partial c_j}\sum_{i\in S_j}\|x_i-c_j\|^2 = -2\sum_{i\in S_j}(x_i-c_j) = 0
\implies c_j = \frac{1}{|S_j|}\sum_{i\in S_j} x_i
$$

**Monotone descent.** The assignment step can only decrease `J` (each point moves to
a centre at least as close), and the update step can only decrease it (the mean is the
minimiser of the within-cluster sum). So `J` is non-increasing and bounded below by
zero; the algorithm converges in finitely many steps because there are finitely many
partitions. It converges to a *local* optimum, and the fixed point satisfies both
first-order conditions: every point is closer to its own centre than to any other, and
every centre is the mean of its cluster.

**k-means++.** Choose the first centre uniformly, then choose each subsequent centre
`x` with probability proportional to `D(x)²`, the squared distance to the nearest
chosen centre. The `D²` weighting makes a point that is far from everything already
chosen very likely to be picked, so the initial centres are spread out. The
approximation guarantee is

$$
\mathbb{E}[J_{\text{++}}] \le 8(\ln k + 2)\,J_{\text{opt}}
$$

which is why the seeding is worth its cost: random seeding has no such bound and can
place two centres inside the same cluster, from which no local move escapes.

**Elkan's bounds.** Let `a_i` be the current centre of `x_i`, `u_i` an upper bound on
`d(x_i, c_{a_i})`, and `l_i` a lower bound on `min_{j≠a_i} d(x_i, c_j)`. If
`u_i ≤ l_i`, no centre can beat the current one and the point can be skipped. After
the centres move by `δ_j = d(c_j, c_j')`, the triangle inequality gives

$$
u_i \leftarrow u_i + \delta_{a_i},
\qquad
l_i \leftarrow l_i - \max_j \delta_j
$$

and only points with `u_i > l_i` recompute distances. The bounds are loose compared
with the full per-centre Elkan bounds, but they preserve exactness and still skip most
points once the centres settle.

**Mini-batch update.** For a batch `B` and cluster `j` with `m_j` batch points, the
centre moves toward the batch mean with a decaying step:

$$
c_j \leftarrow c_j + \eta_j\,(\bar x_{j,B} - c_j),
\qquad
\eta_j = \frac{m_j}{N_j}
$$

where `N_j` is the total number of points assigned to `j` so far (including this
batch). This is Robbins–Monro stochastic approximation: `Ση = ∞` and `Ση² < ∞`, so the
centres converge to a fixed point of the full-batch iteration. The initial `N_j = 0`
makes the first update jump straight to the batch mean, which is what lets the method
converge in a few dozen batches.

**Bisecting k-means.** Start with one cluster. Repeatedly take the cluster with the
largest SSE, run 2-means on it, and replace it with the two children. This is a greedy
hierarchy: it produces a dendrogram as a by-product, and with `O(k log k)` 2-means runs
it is often faster than `k`-means on large `n` because each split works on a subset.
The result is not the same local optimum as Lloyd's, and the demo's inertia (`271.5`
vs `261.8`) is slightly worse for that reason.

**Spherical k-means.** Replace squared Euclidean distance with cosine:

$$
d_{\cos}(x,c) = 1 - \frac{x^\top c}{\|x\|\|c\|}
$$

Assign by maximum cosine, and after the mean update renormalise each centre to unit
length. This is k-means on the unit sphere: it clusters by direction and ignores
magnitude, which is exactly right for normalised embeddings and term-frequency
vectors. The demo's data has radii spanning `25×`; Euclidean k-means splits by radius
(`0.700` accuracy) while spherical recovers the three directions (`1.000`).

**Empty clusters.** A centre can lose all its points when `k` exceeds the number of
distinct points, or when two centres coincide. The mean of an empty set is undefined,
so the usual fix is to re-seed: pick the points with the largest distance to their
current centre and make them the new centres. The repair keeps the partition valid and
`J` finite; it does not guarantee `k` non-empty clusters when the data has fewer than
`k` distinct points, which is a pigeonhole limit, not a bug.

## Code walkthrough

The distance matrix is computed by expanding the square, never by broadcasting an
`n×k×d` tensor:

```python
    @staticmethod
    def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        d = (X * X).sum(axis=1)[:, None] - 2.0 * X @ C.T + (C * C).sum(axis=1)[None]
        return np.maximum(d, 0.0)
```

`||x−c||² = ||x||² − 2xᵀc + ||c||²` is one matrix multiply plus two rank-1 terms,
which is the standard `O(nkd)` implementation and the one BLAS accelerates. The
`maximum(..., 0)` guards the small negative values that cancellation produces when a
point is very close to a centre; a negative distance would win an `argmin` and corrupt
the assignment.

The `++` seeding is a loop of weighted draws:

```python
        C = [X[int(self.rng.integers(n))]]
        d = ((X - C[0]) ** 2).sum(axis=1)
        for _ in range(1, self.k):
            total = float(d.sum())
            if total <= EPS:
                C.append(X[int(self.rng.integers(n))])
            else:
                C.append(X[int(self.rng.choice(n, p=d / total))])
            d = np.minimum(d, ((X - C[-1]) ** 2).sum(axis=1))
```

`d` always holds the squared distance to the *nearest* centre so far, updated with
`np.minimum` after each pick — that is the `D²` in the name. The `total <= EPS` branch
handles the case where every point coincides with a chosen centre: the distribution is
all zeros and `choice(p=...)` would raise, so a uniform pick is the sensible fallback.
Drawing with `p=d/total` rather than sampling the largest `d` is what makes the seeding
randomised and gives the approximation guarantee; deterministic farthest-point seeding
is a different algorithm.

Lloyd's loop is five lines because both steps are functions:

```python
        for _ in range(self.max_iter):
            new_labels = self._assign(X, C)
            newC = self._means(X, new_labels, C)
            shift = float(np.sqrt(((newC - C) ** 2).sum()))
            C = newC
            self.n_updates_ += 1
            converged = np.array_equal(new_labels, labels)
            labels = new_labels
            if converged and shift <= self.tol:
                break
```

The stop test requires *both* stable labels and a small centre shift. Labels alone are
enough in exact arithmetic, but floating-point means can drift by `1e-16` forever,
and a shift-only test can stop while assignments still oscillate between two tied
centres. Checking both is the cheap, robust choice.

Elkan's version maintains the two bounds:

```python
            u = u + delta[labels]
            lo = lo - float(delta.max())
            changed = False
            for i in range(n):
                if u[i] <= lo[i]:
                    continue
                dist = np.sqrt(((X[i] - C) ** 2).sum(axis=1))
                self.n_evals_ += self.k
                j = int(np.argmin(dist))
                u[i] = dist[j]
                dist[j] = np.inf
                lo[i] = float(dist.min())
```

`u` grows by the movement of the point's own centre and `lo` shrinks by the largest
movement of *any* centre; that asymmetry is what keeps the test conservative. When a
point must be recomputed, `dist[j] = inf` before `min` extracts the second-smallest
distance, which becomes the new lower bound. Everything is in Euclidean distance, not
squared, because the triangle inequality does not hold for squares. The result is
bit-identical to Lloyd's (the demo asserts `array_equal` on labels) because the skipped
points provably cannot change their assignment.

Mini-batch replaces the full mean with a decaying step:

```python
            labels = self._assign(X[idx], C)
            for j in range(self.k):
                mask = labels == j
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                counts[j] += cnt
                eta = cnt / counts[j]
                C[j] += eta * (X[idx][mask].mean(axis=0) - C[j])
```

`counts[j]` is the cumulative number of rows ever assigned to `j`, so `eta` decays like
`1/N_j` — the Robbins–Monro schedule. Starting `counts` at zero makes `eta = 1` on the
first batch, so the centre jumps to the batch mean instead of creeping toward it.
`if cnt == 0: continue` leaves the centre where it is; the empty-cluster repair is a
separate variant because silently moving centres changes the algorithm.

Bisecting repeats 2-means on the worst cluster:

```python
        while len(clusters) < self.k:
            sizes = np.array([sse(c) for c in clusters])
            i = int(np.argmax(sizes))
            idx = clusters.pop(i)
```

```python
            for _ in range(3):
                sub = KMeans("lloyd", n_clusters=2, max_iter=self.max_iter,
                             tol=self.tol, init="kmeans++",
                             random_state=int(self.rng.integers(2**31 - 1)))
                C2, lab = sub._lloyd(X[idx], sub._init_centres(X[idx]))
```

Splitting the largest-SSE cluster is the greedy choice that most reduces the objective
per split, and three restarts guard against a bad 2-means split. The sub-problem is a
full `KMeans` instance so the same Lloyd code is reused; its counters are folded into
the parent's so the demo's evaluation count is honest.

The empty-cluster repair picks distinct worst-fit points:

```python
        if empty:
            # Re-seed each empty cluster at a distinct worst-fit point.
            d = ((X - C[labels]) ** 2).sum(axis=1)
            order = np.argsort(-d, kind="mergesort")
            taken: set[int] = set()
            for j in empty:
                for i in order:
                    if int(i) not in taken:
                        taken.add(int(i))
                        C[j] = X[int(i)]
                        break
```

`taken` is what makes the repair work when several clusters are empty at once: without
it, every empty centre would be seeded at the same farthest point, recreating the
duplicate-centre problem the repair is meant to fix. `mergesort` makes the tie order
deterministic.

Spherical assignment normalises both sides:

```python
        if self.variant == "spherical":
            Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
            Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), EPS)
            self.n_evals_ += X.shape[0] * C.shape[0]
            return np.argmax(Xn @ Cn.T, axis=1)
```

Cosine similarity is the dot product of unit vectors, so the whole assignment is one
matrix multiply and an `argmax`. Normalising `C` on every call (not just at update
time) keeps `predict` correct even if a caller overwrites `cluster_centers_`. The `EPS`
denominator guards the zero vector, which has no direction and would otherwise produce
`nan` that propagates into every comparison.

## Complexity

| Variant | Time per iteration | Extra memory |
|:--|:--|:--|
| `lloyd` | `O(n k d)` | `O(n k)` for the distance block |
| `kmeans_plusplus` | `O(n k d)` per restart, `n_init` restarts | `O(n)` |
| `elkan` | `O(n k d)` worst case, often `O(n d + k²d)` | `O(n k)` |
| `mini_batch` | `O(batch · k d)` | `O(batch · k)` |
| `bisecting` | `O(n k d log k)` total | `O(n + k)` |
| `spherical` | `O(n k d)` | `O(n k)` |

The number of Lloyd iterations is not bounded by a useful function of `n`: it is
finite but can be exponential in the worst case, and `O(log n)` in practice. In `d`
dimensions each iteration is a matrix multiply, which is why k-means is one of the few
classical algorithms where BLAS matters more than the asymptotics. `elkan`'s worst
case is no better than Lloyd's — it is a constant-factor accelerator — but its typical
case skips most points once the centres stabilise.

## Failure modes

1. **Local minima.** Every variant except `elkan` (which is exact but still local)
   can land in a bad partition. Restart (`n_init`) and keep the best inertia; the demo
   does this everywhere except the deliberate single-run comparisons.
2. **Choosing `k`.** Inertia decreases monotonically with `k`, so it cannot select
   `k`. Use an elbow plot, silhouette, or BIC on a Gaussian mixture — the `gmm.py`
   atom has the model-based version.
3. **Non-spherical clusters.** Squared Euclidean distance implies isotropic,
   equal-variance blobs. Elongated or nested clusters need GMM, DBSCAN, or a kernel.
   The spherical variant fixes the *magnitude* part of the problem, not the shape.
4. **Scale.** A feature in the thousands dominates one in the tenths. Standardise
   before fitting; k-means is not invariant to affine rescaling.
5. **Empty clusters.** With duplicate points or `k` larger than the number of distinct
   points, some centres get no rows. `empty_cluster_fix` repairs the numeric failure,
   but the pigeonhole case (`k >` distinct points) still cannot produce `k` non-empty
   clusters.
6. **Outliers.** A single far point drags its centre and can consume a whole cluster.
   Trim, use k-medoids, or a robust criterion.
7. **`mini_batch` learning rate.** The `1/N_j` schedule is right for stationary data.
   If the data distribution drifts, the decaying rate stops adapting; reset `counts`
   or use a floor on `eta`.
8. **`elkan` in high dimensions.** The bound test costs `O(k)` per point and the
   pruning rate falls as `d` grows; above roughly `d=20`, plain Lloyd with a blocked
   distance matrix is usually faster.
9. **Spherical with zero vectors.** A zero row has no direction; the `EPS` guard
   assigns it to an arbitrary centre. Filter or impute zero rows before fitting.

## Applications

- **Customer and user segmentation.** The canonical use: group users by spend,
  frequency and recency into `k` personas, then treat each segment differently.
- **Vector quantisation and compression.** Codebook learning for product quantisation
  in nearest-neighbour search; `mini_batch` is the standard variant at index-build
  scale.
- **Colour quantisation.** Pixels as 3-D points, `k=16` or `256` for a palette; the
  spherical variant is not used here, but `bisecting` gives a palette hierarchy.
- **Document and embedding clustering.** Spherical k-means on TF-IDF or normalised
  sentence embeddings is the standard topic-grouping baseline.
- **Anomaly pre-screening.** Distance to the nearest centre is a cheap outlier score;
  `local_outlier_factor` refines it with local density.
- **Initialisation for other models.** GMM, RBF networks, and vector-quantised VAEs
  all use k-means to place their components before a finer EM or gradient fit.

## References

- Lloyd, *Least Squares Quantization in PCM*, 1982, IEEE Transactions on Information
  Theory — the original alternation (written in 1957, published in 1982).
- MacQueen, *Some Methods for Classification and Analysis of Multivariate
  Observations*, 1967, Berkeley Symposium — the name "k-means".
- Arthur & Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, 2007, SODA —
  the `D²` seeding and its `O(log k)` guarantee.
- Elkan, *Using the Triangle Inequality to Accelerate k-Means*, 2003, ICML — the
  bound-based pruning implemented here.
- Sculley, *Web-Scale k-Means Clustering*, 2010, WWW — the mini-batch algorithm and
  its `1/N` schedule.
- Steinbach, Karypis & Kumar, *A Comparison of Document Clustering Techniques*, 2000,
  KDD Workshop — bisecting k-means and spherical k-means for text.
- Kanungo et al., *A Local Search Approximation Algorithm for k-Means Clustering*,
  2002, Computational Geometry — the swap-based local search and its guarantees.
