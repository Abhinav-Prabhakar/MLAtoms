# HDBSCAN

`hdbscan.py` · Clustering · Tier 1 · 5 variants

## Intuition

DBSCAN has one knob that decides everything: `eps`. It also has a structural weakness —
a single `eps` cannot fit a dense blob and a diffuse cloud at the same time. Set it for
the blob and the cloud is all noise; set it for the cloud and the blob merges with its
neighbours.

HDBSCAN dissolves the problem instead of tuning it. Build the clustering at *every*
density, and then decide which clusters are worth keeping:

1. **Core distance.** `d_core(x)` is the distance to the `min_samples`-th nearest
   neighbour — a local density estimate at `x`.
2. **Mutual reachability.** Replace `d(a,b)` by
   `max(d_core(a), d_core(b), d(a,b))`. Two points in a sparse region are pushed apart
   even if they are physically close, so a chain of sparse points cannot bridge two
   dense clusters.
3. **Minimum spanning tree** of that graph. Its edges, sorted by weight, are the entire
   single-linkage hierarchy: every merge is one edge.
4. **Stability.** A cluster earns credit for every point that stays inside it over its
   `λ = 1/distance` lifetime.
5. **Excess of mass.** Keep a cluster when it earns more credit than its children
   combined; otherwise keep the children.

Because the threshold `eps` is gone, only `min_cluster_size` (how small a group may be)
and `min_samples` (how smooth the density estimate is) remain, and both are far less
sensitive than `eps`. The demo's three blobs of different density plus a uniform
background come out as exactly three clusters, with `0.989` accuracy among labelled
points and `0.907` of the true inliers recovered.

## Variants

| Variant | Selection | What it shows |
|:--|:--|:--|
| `hdbscan` | excess of mass on the whole hierarchy | the reference algorithm |
| `hdbscan_star` | one flat DBSCAN* cut at the best level | the hierarchy read without stability |
| `stability` | rank all nodes by stability, keep the best non-overlapping ones | stability as a first-class quantity |
| `soft_clustering` | EOM clusters, then inverse-distance memberships | graded assignment |
| `approximate` | EOM on an MST built from a k-NN graph | the scaling trick |

`hdbscan` is Campello et al.'s excess-of-mass (EOM) rule and is what "HDBSCAN" means
without qualification. `hdbscan_star` is the other classical reading: pick the level
`λ` where the hierarchy has the most valid clusters and take DBSCAN* there. It is
cheaper (no stability accumulation) and it labels nearly everything — `35` noise points
against `127`, with `1.000` inlier coverage — but it also absorbs part of the uniform
background, which is why its accuracy is `0.927` rather than `0.989`.

`stability` ranks the nodes directly: sort every cluster-size-valid node by stability
and greedily keep the best ones that do not overlap. On well-separated data it agrees
with EOM, which is the point of the demo (`[115.1 40.3 118.1]` from both), but it
exposes the raw numbers and behaves differently when a parent node is genuinely more
stable than its children.

`soft_clustering` keeps the EOM clusters and replaces the hard label with a membership
vector. Each cluster is represented by its densest point, memberships are
`1/d²`-weighted, and small memberships are zeroed and renormalised. `79` of the 650
points end up split between two clusters — the boundary points whose assignment is a
coin flip — while the median point has a maximum membership of `1.00`.

`approximate` builds the MST from a `k`-nearest-neighbour graph instead of every pair.
On the demo it examines `6383` candidate pairs instead of `210925`, returns the same
three clusters, and loses nothing in accuracy (`0.989`).

## Derivation

**Core distance and mutual reachability.** Let `k = min_samples` and let `d_k(x)` be the
distance from `x` to its `k`-th nearest neighbour. Then

$$
d_{\text{core}}(x) = d_k(x),
\qquad
d_{\text{mreach}}(a,b) = \max\{d_{\text{core}}(a),\, d_{\text{core}}(b),\, d(a,b)\}
$$

The mutual reachability graph is the complete graph with these weights. It has two
interpretations that matter:

- It is the DBSCAN graph at level `λ` if you keep only edges with `d_mreach ≤ 1/λ`:
  two points are connected exactly when both are core at that radius and close enough.
- It inflates the distance between a core point and a sparse neighbour to the core
  radius, so the sparse point's attachment to the dense cluster becomes as long as the
  cluster's own density scale. That is what suppresses single-linkage chaining.

**Why the MST is the hierarchy.** In a graph with edge weights, single-linkage
clustering at threshold `d` is the connected components of the edges of weight `≤ d`.
Kruskal's algorithm processes edges in increasing order and unions components — so it
*is* single-linkage, and the sequence of unions it performs is the dendrogram, at a cost
of one MST instead of all `n(n−1)/2` edges. Prim's algorithm computes the same MST
without materialising the sort. That equivalence is the theoretical core of HDBSCAN:
**one MST contains every single-linkage clustering of the data**.

Formally, for any threshold `d`, the components of the mutual reachability graph at `d`
are the components of the subgraph of the MST restricted to edges `≤ d`. So the
hierarchy can be built from `n−1` numbers rather than `n²`.

**λ and the tree.** Write `λ = 1/d`. Small `λ` is coarse (large distance), large `λ` is
fine. Each internal node `c` of the dendrogram is created by one merge at level
`λ_c = 1/w_c` and splits at `λ = level(c)`. Its lifetime is the interval
`(birth(c), level(c)]`, where `birth(c)` is the `λ` of the merge that created its
parent. Along any root-to-leaf path these intervals partition the `λ` axis, so at any
`λ` exactly one node per path is "active" — that is why a flat cut is well defined.

The value `λ_p = 1/d_core(p)` is where point `p` stops being a core point: beyond it,
`p` is too sparse to anchor a cluster.

**Stability.** The stability of `c` is the total `λ`-lifetime its points spend inside
it:

$$
s(c) = \sum_{p \in c}\Big(\min\big(\lambda_p,\, \text{level}(c)\big) - \text{birth}(c)\Big)_+
$$

The `min` says a point contributes only until it stops being core, and only while `c`
exists; the `(·)_+` says points that are already gone when `c` appears contribute
nothing. A tight, long-lived cluster accumulates a large `s`; a transient merge of two
distant blobs accumulates almost nothing.

**Excess of mass.** Define the selection recursively:

$$
\text{keep}(c) =
\begin{cases}
\{c\} & \text{if } s(c) > \sum_{c' \in \text{children}(c)} \text{keep}(c'),\\[4pt]
\bigcup_{c'} \text{keep}(c') & \text{otherwise}
\end{cases}
$$

with leaves that are smaller than `min_cluster_size` contributing nothing. A parent is
kept only when merging its children *adds* mass — that is the "excess" in excess of
mass. It prefers fewer, larger clusters when the children are unstable fragments of one
dense region, and more, smaller clusters when the parent spans a sparse gap.

The root always wins this comparison, because `s(root) = Σ_p min(λ_p, level_root)`
grows with `n` while its children's stabilities are bounded by their own short
lifetimes. HDBSCAN therefore excludes the root unless `allow_single_cluster=True` —
this is a real branch in the library, and forgetting it silently returns one cluster on
every dataset. The demo relies on the excluded-root path.

**Flat cut.** At level `λ`, the active nodes are `{c : birth(c) < λ ≤ level(c)}`, and
they partition the leaves. A DBSCAN* reading of the hierarchy is one such cut. Choosing
`λ` to maximise the number of clusters of size `≥ min_cluster_size` (ties broken by
covered points) is the greedy optimum over the `n` candidate levels, which is what
`_best_flat_cut` enumerates. It is cheap and exact, and it needs no stability at all —
at the cost of being sensitive to which level wins.

**Soft membership.** For selected cluster `c` with exemplar `e_c` (its densest point),

$$
\mu_c(x) = \frac{1/\|x-e_c\|^2}{\sum_{c'} 1/\|x-e_{c'}\|^2}
$$

then entries below `min_probability` are zeroed and the row renormalised. Inverse-square
distance is the natural kernel here: it is the harmonic potential of a point mass, so a
point twice as far from one exemplar gets a quarter of the weight, and a point sitting
on an exemplar gets essentially all of it.

**Outlier score.** The GLOSH score of Campello et al. is

$$
\text{score}(p) = 1 - \frac{\lambda_p}{\lambda_{\max}}
$$

with `λ_max` the largest core-`λ` in the sample. It is scale-free (it depends only on
the density ordering), it needs no labelled anomalies, and it is exactly the quantity
the hierarchy already computed. The demo's uniform background scores `0.943` against
`0.718` for clustered points.

## Code walkthrough

Core distances come from one `partition`, which is `O(n)` per row instead of a sort:

```python
    def _core_distances(self, D: np.ndarray) -> np.ndarray:
        n = D.shape[0]
        k = min(self.min_samples, n - 1)
        return np.partition(D, k, axis=1)[:, k]
```

`D[i, i] = 0`, so index `k` in the sorted row is the `k`-th *other* point — the
neighbour count convention that makes `min_samples=1` the nearest neighbour rather than
the point itself. `np.partition` places the `k`-th smallest value in position `k` and
leaves the rest unordered, which is all this needs; a full `sort` would be `O(n log n)`
for the same answer. `min(self.min_samples, n − 1)` guards a dataset smaller than `k`.

Prim's algorithm runs over the dense matrix, keeping one best edge per unseen node:

```python
        for _ in range(n):
            masked = np.where(seen, INF, best)
            u = int(np.argmin(masked))
            seen[u] = True
            if parent[u] >= 0:
                edges.append((float(mr[u, parent[u]]), int(parent[u]), u))
            upd = (~seen) & (mr[u] < best)
            best[upd] = mr[u][upd]
            parent[upd] = u
```

`masked` is the standard "argmin over unseen" and is a full `O(n)` copy each step, so
the loop is `O(n²)` — the same order as the distance matrix it operates on, which is why
it is not worth a heap here. `best[u]` starts at infinity except for node 0, so the tree
grows from an arbitrary root and the first iteration adds nothing. The recorded edge
weight is the *parent's* side of the cut, which is the MST edge; taking `mr[u, v]` for
the current `u` and its old parent would be the same number.

The k-NN approximation replaces the `n²` scan with a sparse edge list:

```python
        k = min(self.approx_neighbors or 3 * self.min_samples, n - 1)
        idx = np.argpartition(mr, k, axis=1)[:, :k + 1]
        edges: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in idx[i]:
```

`k+1` columns because `argpartition` includes the point itself. The dict keyed on the
*sorted* pair deduplicates: `(i,j)` and `(j,i)` are the same edge and must not be
counted twice, or Kruskal would see two edges for one connection and the union-find
would silently ignore the duplicate. `3 * min_samples` is the usual rule of thumb;
raising it makes the graph denser and the result more exact at the cost of the speed-up.

The tree is built by replaying MST edges in increasing order:

```python
        for w, i, j in sorted(edges):
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            level = 1.0 / w if w > EPS else INF
            a, b = node_of[ri], node_of[rj]
            nodes.append({
                "level": float(level), "birth": 0.0, "children": [a, b],
                "points": nodes[a]["points"] + nodes[b]["points"],
                "size": nodes[a]["size"] + nodes[b]["size"],
            })
            uf[ri] = rj
            node_of[rj] = len(nodes) - 1
```

Each accepted edge creates exactly one internal node, so the tree has `2n−1` nodes and
the loop is Kruskal with union by "attach the second root under the first". `level` is
`1/w`, and `nodes[child]["birth"]` is filled in afterwards in one pass, because a node's
birth is its *parent's* level and the parent does not exist yet when the child is
created. `w = 0` (duplicate points) gives an infinite level, which is correct: a
duplicate is infinitely dense and never splits. The union-find is path-halving, which is
enough for `n` operations.

Stability is the only place the point lists are used as more than bookkeeping:

```python
        for i, node in enumerate(self.nodes):
            lo, hi = node["birth"], node["level"]
            total = 0.0
            for p in node["points"]:
                lp = lam[p]
                if not np.isfinite(lp):
                    lp = 1e300
                total += max(0.0, min(lp, hi) - lo)
            stab[i] = float(total)
```

`min(lp, hi)` is the point's fall-out `λ` (its own density or the cluster's split,
whichever comes first) and `lo` is when the cluster appeared; the difference is the
`λ`-lifetime the point spends inside. `max(0, ...)` handles the case `lo > λ_p` — a point
that was already non-core when the cluster was born. The `1e300` substitution for
duplicate points (infinite `λ`) keeps the arithmetic finite without changing any
comparison.

Selection is a five-line recursion:

```python
        picks: list[int] = []
        score = 0.0
        for j in node["children"]:
            sub, s = self._eom(j)
            picks.extend(sub)
            score += s
        if self._stab[i] > score:
            return [i], self._stab[i]
        return picks, score
```

The returned pair is `(clusters, total stability of those clusters)`, so the parent's
comparison uses the stability of what the children would *actually* contribute — not
their raw stability, which would double count when a child is itself replaced by
grandchildren. Sizes below `min_cluster_size` return `([], 0.0)` at the top of the
function, which prunes entire subtrees of fragments in one test.

The flat cut is a sweep over the distinct levels:

```python
        for lam in levels:
            cut = [i for i, n in enumerate(nodes)
                   if n["birth"] < lam <= n["level"]
                   and n["size"] >= self.min_cluster_size]
            if not cut:
                continue
            key = (len(cut), float(sum(nodes[i]["size"] for i in cut)))
            if key > best_key:
                best_key, best = key, cut
```

The strict `birth < λ ≤ level` matters at the boundaries: adjacent nodes share an
endpoint (`level(child) = birth(parent)`), and using `≤` on both sides would put two
nested nodes in the same cut. The tuple comparison `(count, coverage)` means "most
clusters first, then the most points covered" — a deliberate choice, since a single huge
cluster covers the most points but is useless as a clustering.

Labels are assigned per selected subtree:

```python
        for idx, c in enumerate(selected):
            node = self.nodes[c]
            level = node["level"]
            for p in node["points"]:
                labels[p] = idx
                proba[p] = 1.0 if level <= 0.0 else min(
                    1.0, float(self._lam[p]) / level)
```

`proba` is the fraction of the cluster's `λ`-range the point survives: 1 for a point
denser than the cluster's split level, less for a point that barely made it in. It is
the HDBSCAN membership strength, and it is what separates a core member from a border
one without a second membership model.

Soft memberships use the densest point of each cluster as its exemplar:

```python
        E = np.asarray(exemplars)
        d2 = ((X[:, None, :] - E[None, :, :]) ** 2).sum(axis=2)
        w = 1.0 / np.maximum(d2, EPS)
        proba = w / w.sum(axis=1, keepdims=True)
        proba[proba < self.min_probability] = 0.0
        proba /= np.maximum(proba.sum(axis=1, keepdims=True), EPS)
```

The broadcast `X[:, None, :] - E[None, :, :]` is an `n×k×d` temporary, which is fine for
a handful of clusters and would need blocking for many. Multiplying by `1/d²` rather
than taking a `softmax` of `−d` is deliberate: no temperature to tune, and the weights
decay at the same rate as a physical potential. Zeroing then renormalising keeps rows
summing to one after the threshold, which is what makes the result a probability vector
rather than a truncated one.

The density score needs only `λ`:

```python
        lam = np.where(np.isfinite(self._lam), self._lam, np.nan)
        top = float(np.nanmax(lam)) if np.isfinite(lam).any() else 1.0
        self.outlier_scores_ = np.clip(1.0 - np.nan_to_num(lam, nan=0.0) / top,
                                       0.0, 1.0)
```

`np.isfinite` masks the infinite `λ` of duplicate points before the max, because
`inf/inf` is `nan` and would poison every score. `np.nanmax` over the finite mask is the
densest point in the sample; the clip keeps the scores in `[0,1]` when a point is denser
than the reference (impossible in-sample, possible for out-of-sample queries).

## Complexity

| Step | Time | Memory |
|:--|:--|:--|
| Distance matrix | `O(n² d)` | `O(n²)` |
| Core distances | `O(n²)` | `O(n)` |
| MST (Prim) | `O(n²)` | `O(n)` |
| Tree build | `O(n log n)` sort + `O(n α(n))` unions | `O(n)` nodes |
| Stability | `O(Σ_c |c|)` = `O(n · depth)` | `O(n · depth)` |
| EOM selection | `O(n)` | `O(n)` |
| Flat cut | `O(n²)` over levels | `O(n)` |
| Soft memberships | `O(n K d)` | `O(n K)` |

The `O(n²)` distance matrix is the bottleneck, and it is the reason HDBSCAN is usually
paired with a k-d tree or an approximate nearest-neighbour index: with one, the MST is
built from `O(n k)` edges and the whole pipeline becomes `O(n k log n)`. This file's
`approximate` variant is the graph half of that trick — the demo drops from `210925`
candidate pairs to `6383` — but it still computes the full matrix, so the timings stay
quadratic. `Σ_c |c|` is `O(n log n)` for a balanced tree and `O(n²)` in the worst case
(a caterpillar); it is the price of storing point lists per node.

## Failure modes

1. **`min_cluster_size` too small.** The hierarchy has a node for every merge, so a tiny
   threshold turns the tree into noise fragments and EOM selects many small clusters.
   Set it to the smallest group you would act on (15–50 is usual), not to `1`.
2. **`min_samples` too small.** With `min_samples = 1` the mutual reachability reduces to
   plain distance and HDBSCAN becomes single linkage: the demo's uniform background
   would chain the three blobs into one cluster. `min_samples` is the knob that buys
   robustness to chaining.
3. **The root trap.** `s(root)` grows with `n`, so selecting on the whole tree returns
   one cluster for every dataset. The `allow_single_cluster=False` branch is not a
   nicety; the demo would fail without it.
4. **`min_samples` larger than the cluster.** If a cluster has fewer than `min_samples`
   points, none of its members is core there, so its core distances equal the
   `min_samples`-th neighbour distance *outside* the cluster and the whole group looks
   sparse. Keep `min_samples < min_cluster_size`.
5. **Duplicate points.** `d_core = 0` gives `λ = ∞` and infinite stability, which can
   dominate the selection. The code guards the arithmetic (`1e300`) but not the
   semantics: deduplicate, or accept that exact duplicates form an unbreakable core.
6. **Flat-cut instability.** `hdbscan_star` picks one level, and the level with the most
   clusters is not necessarily the level with the best clusters. On the demo it labels
   `35` noise points and covers everything, absorbing background that EOM rejects —
   higher recall, lower precision.
7. **Soft memberships ignore density.** Inverse-distance memberships are geometric: a
   sparse background point gets a confident membership if it happens to sit near an
   exemplar. Combine them with `outlier_scores_` or with `probabilities_` from the EOM
   fit when density matters.
8. **GLOSH with one component.** The score is relative to the densest point in the
   sample, so on a single uniform blob every point scores near zero even if the blob is
   the anomaly relative to incoming data. Fit on reference data and score new points
   against the fitted `λ_max`.
9. **Approximate MST disconnect.** A k-NN graph can be disconnected, which would leave
   a forest instead of a tree. The code adds a synthetic root at `λ = 0` to keep the
   selection well-defined, but the resulting hierarchy has a fake top-level merge; check
   `n_edges_ == n − 1` before trusting an approximate fit.
10. **Quadratic memory.** `n = 100k` needs 80 GB for the distance matrix. Use the k-NN
    path, a spatial index, or Borůvka's algorithm on a sparse graph — HDBSCAN's
    asymptotics are fine, this file's implementation is not.

## Applications

- **Anomaly detection without labels.** `outlier_scores_` ranks points by local density;
  the demo's background separates at `0.943` against `0.718`. Used for intrusion
  detection, sensor faults, and transaction monitoring.
- **Variable-density clustering.** Astronomy (star clusters against a background field),
  geo-spatial event data, and single-cell expression data all have clusters whose
  density varies by orders of magnitude, where DBSCAN's single `eps` fails.
- **Topic discovery in embeddings.** Cluster document embeddings without choosing `K`;
  the noise label isolates boilerplate and one-off documents.
- **Hierarchical taxonomies.** The tree is a dendrogram, so `cluster_stability_` and the
  node structure give a hierarchy rather than a flat partition — useful for browsing and
  for choosing a granularity later.
- **Pre-labelling for supervision.** Cluster the unlabelled pool, label the clusters,
  and send the noise to human review; the soft memberships say which points are worth a
  second look.
- **As a stronger baseline than k-means.** When someone reaches for k-means on data with
  unknown `K` and outliers, HDBSCAN is usually the better first attempt and needs less
  tuning.

## References

- Campello, Moulavi & Sander, *Density-Based Clustering Based on Hierarchical Density
  Estimates*, 2013, PAKDD — HDBSCAN, mutual reachability, stability and excess of mass.
- Campello, Moulavi, Zimek & Sander, *Hierarchical Density Estimates for Data
  Clustering, Visualization, and Outlier Detection*, 2015, ACM TKDD — HDBSCAN*, the
  GLOSH outlier score, and the flat-extraction variants.
- McInnes, Healy & Astels, *hdbscan: Hierarchical Density Based Clustering*, 2017, JOSS
  — the reference implementation and the `allow_single_cluster` behaviour.
- Chaudhuri & Dasgupta, *Rates of Convergence for the Cluster Tree*, 2010, NeurIPS — why
  the single-linkage tree estimated from a sample converges to the population tree.
- Ester, Kriegel, Sander & Xu, *A Density-Based Algorithm for Discovering Clusters in
  Large Spatial Databases with Noise*, 1996, KDD — the `eps`-based ancestor whose
  parameter HDBSCAN removes.
