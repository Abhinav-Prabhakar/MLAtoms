# Agglomerative clustering

`agglomerative.py` · Clustering · Tier 1 · 8 variants

## Intuition

Divisive methods start with everything and split. Agglomerative methods do the opposite:
every point is its own cluster, and the two closest clusters are merged, over and over,
until one remains. The sequence of merges *is* the answer — a tree (a dendrogram) whose
leaves are points and whose internal nodes are merges, each labelled with the distance at
which it happened. Cutting the tree at any height gives a flat clustering, so one run
answers every `K`.

Everything interesting is in the phrase "closest". Two clusters have many point pairs
between them, and the linkage rule decides which of those distances counts:

- **single** uses the closest pair, so a thin chain of points is enough to join two dense
  blobs. That is chaining: sometimes a feature (a filament, a coastline), sometimes a bug.
- **complete** uses the farthest pair, so an elongated cluster is expensive to join and
  the result is compact and roughly equal-sized.
- **average** uses the mean over all cross pairs, which is the compromise and the
  phylogenetic standard (UPGMA).
- **ward** merges the pair whose union increases the total within-cluster sum of squares
  the least — the "keep the variance low" rule.

On the demo's two blobs joined by twelve collinear bridge points, single linkage joins
them at height `0.44` and complete linkage only at `7.45`: same data, same greedy loop,
two completely different clusters. That gap is the whole reason the linkage rule exists.

The other two variants answer different questions. `connectivity_constrained` restricts
merges to edges of a k-NN graph, so clusters must be *connected regions* rather than
merely close ones. `cut_height` replaces "how many clusters?" with "how far apart?", and
lets small leftover groups become noise.

## Variants

| Variant | Merge distance | Monotone | Character |
|:--|:--|:--|:--|
| `single` | `min` over cross pairs | yes | chains, arbitrary shapes, `K` from gaps |
| `complete` | `max` over cross pairs | yes | compact, equal-sized, sensitive to outliers |
| `average` | mean over cross pairs | yes | size-weighted compromise, UPGMA |
| `ward` | least increase in within-cluster SSE | yes | balanced, variance-minimising |
| `centroid` | distance between centroids | **no** | size-weighted centre distance |
| `median` | distance between centroids, unweighted | **no** | WPGMC, no size bias |
| `connectivity_constrained` | ward, only along a mutual k-NN graph | yes | connected regions, manifold-shaped |
| `cut_height` | single, stopped at `height` | yes | discovers `K`, marks noise |

All six linkages are the same loop with different Lance–Williams coefficients, so the
file has one merge implementation and a six-branch distance update. The three
"geometric" ones (`ward`, `centroid`, `median`) run on squared distances and report a
square root as the merge height; the three "set" ones (`single`, `complete`, `average`)
run on plain distances.

The differences that matter in practice:

- **`single` and `complete` are the two extremes**, and both are exact: they are the
  minimum spanning tree (single) and the maximum spanning tree (complete) read as a
  hierarchy. Everything between them is a compromise, which is what `average` is.
- **`ward` is the one to reach for on compact blobs**, and the demo shows why: on unequal
  blobs it produces the lowest within-cluster spread (`1.36` against `1.96` for
  `centroid`). It is also the one that can fail badly on unequal cluster sizes, because
  its cost `n_A n_B/(n_A+n_B)` grows with balance: the same demo's `ward` accuracy is
  `0.722` while `centroid` gets `0.922`. Ward *prefers* balanced clusters, so it will
  split a large true cluster to join a small one.
- **`centroid` and `median` are the ones that can invert.** A merge that moves a centroid
  toward an unrelated cluster can make the next merge cheaper than the previous one, so
  merge heights are not monotone. On the chain data that happens `7` times for `centroid`
  and `3` times for `median`, and zero times for the other four. An inverted hierarchy
  cannot be drawn as a tree with increasing heights, and cutting it at a height is
  meaningless — the code still reports `n_inversions_` so you notice.
- **`connectivity_constrained`** keeps `ward`'s merge rule but only allows merges along a
  mutual k-NN graph. On two moons, plain `ward` cuts the crescents arbitrarily
  (`0.770`) because "close" is not the same as "connected"; the constrained version
  scores `1.000` and never has to merge across the gap between the manifolds.
- **`cut_height`** ignores `K` entirely: merges stop when the next one is above `height`.
  On three blobs with uniform background, a cut at `0.5` returns exactly the three blobs
  plus `8` noise points (`min_cluster_size=20` deletes the singletons); a cut at `5.0`
  returns one cluster. It is DBSCAN's idea built out of a dendrogram.

## Derivation

**The greedy loop.** Start with `n` singletons and a symmetric distance matrix. Merge the
pair with the smallest distance, replace the two rows by one, repeat `n−1` times. The
number of merges is fixed, so the only freedom is *which* two clusters are closest — which
is the linkage rule.

**Lance–Williams.** Recomputing `d(A∪B, C)` from scratch costs `O(|A||B|)` per pair. The
Lance–Williams recursion does it from the three existing entries in `O(1)`:

$$
d(A\cup B, C) = \alpha_A\, d(A,C) + \alpha_B\, d(B,C) + \beta\, d(A,B)
+ \gamma\,\bigl|d(A,C) - d(B,C)\bigr|
$$

The identities `max(x,y) = ½(x+y) + ½|x−y|` and `min(x,y) = ½(x+y) − ½|x−y|` give

| Linkage | `α_A` | `α_B` | `β` | `γ` |
|:--|:--|:--|:--|:--|
| single | `½` | `½` | `0` | `−½` |
| complete | `½` | `½` | `0` | `+½` |
| average | `n_A/(n_A+n_B)` | `n_B/(n_A+n_B)` | `0` | `0` |
| ward | `(n_A+n_C)/T` | `(n_B+n_C)/T` | `−n_C/T` | `0` |
| centroid | `n_A/(n_A+n_B)` | `n_B/(n_A+n_B)` | `−n_A n_B/(n_A+n_B)²` | `0` |
| median | `½` | `½` | `−¼` | `0` |

with `T = n_A + n_B + n_C`. Single and complete are the `min`/`max` identities applied
to `d(A,C)` and `d(B,C)` — the `β·d(A,B)` term is zero because the new cluster's own
diameter does not affect its distance to a third cluster.

**Average (UPGMA).** The mean over all cross pairs:

$$
d(A\cup B,C) = \frac{\sum_{a\in A\cup B}\sum_{c\in C} d(a,c)}{|A\cup B|\,|C|}
= \frac{n_A\,d(A,C) + n_B\,d(B,C)}{n_A+n_B}
$$

The coefficients are proportions, so a large cluster's average dominates the merged
distance. That is the UPGMA "size weighting": a big cluster is hard to pull apart because
its internal averaging is inertially stable.

**Centroid (UPGMC).** Work with squared Euclidean distances, so the singleton distances
are `‖x_i − x_j‖²` and a cluster is represented by its mean `μ`. Expanding the variance
decomposition around `μ_{A∪B} = (n_A μ_A + n_B μ_B)/(n_A+n_B)`:

$$
\|\mu_{A\cup B} - \mu_C\|^2 =
\frac{n_A\|\mu_A-\mu_C\|^2 + n_B\|\mu_B-\mu_C\|^2}{n_A+n_B}
- \frac{n_A n_B}{(n_A+n_B)^2}\|\mu_A-\mu_B\|^2
$$

which is the `centroid` row above. The negative `β` is the "centroid moved toward `C`"
correction, and it is exactly the term that allows inversions.

**Median (WPGMC).** Replace the weighted mean by the unweighted midpoint
`(μ_A + μ_B)/2`. The same expansion with `½` weights gives `α = ½`, `β = −¼`, and the
merged cluster's size no longer enters the update. That removes the size weighting but
not the non-monotonicity: the midpoint can still jump toward `C`. The difference on
unequal data is visible in the demo — `median` differs from `centroid` while both stay
non-monotone.

**Ward.** Ward's objective is the total within-cluster sum of squares,
`SSE = Σ_c Σ_{x∈c} ‖x − μ_c‖²`, and each greedy step merges the pair that increases it
the least. For a union,

$$
\Delta(A,B) = \mathrm{SSE}(A\cup B) - \mathrm{SSE}(A) - \mathrm{SSE}(B)
= \frac{n_A n_B}{n_A+n_B}\,\|\mu_A-\mu_B\|^2
$$

Because `Δ ≥ 0`, Ward is automatically monotone. The stored distance is
`2Δ = 2 n_A n_B/(n_A+n_B)·‖μ_A − μ_B‖²`, and it satisfies the Lance–Williams recursion
with the coefficients in the table (this is the standard derivation; substituting the
centroid identity above and multiplying by `2(n_A+n_B)n_C/(n_A+n_B+n_C)` produces it
term by term). The merge height reported is `√(2Δ)`, matching the convention of
`scipy.cluster.hierarchy`.

The cost `n_A n_B/(n_A+n_B)` is maximised at `n_A = n_B` for fixed total, which is the
formal statement of Ward's balance bias: for a fixed centroid separation, merging two
medium clusters costs more than absorbing a small cluster into a large one, so Ward
prefers to keep sizes similar — even when the truth is lopsided.

**Why `single` and `complete` are MSTs.** At threshold `d`, single-linkage clusters are
the connected components of the graph with edges of weight `≤ d`. Kruskal's algorithm
builds exactly those components while adding edges in increasing order, so single linkage
is Kruskal on the complete graph — equivalently, the minimum spanning tree read as a
dendrogram. Complete linkage is the same argument with the maximum spanning tree. That is
why `SLINK` and `CLINK` can do in `O(n²)` time what the naive loop does in `O(n³)`: they
never look at all pairs again.

**Connectivity constraint.** Let `G` be a graph on the points (here: the mutual k-NN
graph). A constrained agglomerative clustering only merges `A` and `B` when some edge of
`G` crosses between them, and after merging, the new cluster inherits the union of its
parents' edges. Clusters are then connected subgraphs of `G` by construction, which is
the discrete analogue of "a cluster is a connected region of the data manifold". The rule
is soft: when no allowed pair remains the code falls back to the unrestricted minimum, so
a disconnected `G` still produces a complete hierarchy instead of deadlocking.

**Cutting.** For a monotone hierarchy the merges happen at non-decreasing heights, so
"stop after `n − k` merges" and "cut at the `k`-th smallest height" are the same thing,
and every cut yields a well-defined partition. `cut_height` uses the height form: it
stops as soon as the next merge exceeds `height`, then drops clusters smaller than
`min_cluster_size` to noise. A dendrogram for the demo's chained data looks like

```text
height
 7.45 |                              ┌──────── both blobs ────────┐   complete
 0.44 |  ┌──── blob A ────┐── chain ──┌──── blob B ────┐              single
 0.00 |  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
```

The same picture under single linkage collapses to one long chain: the first merge above
the chain's spacing already contains every point.

## Code walkthrough

All six linkages are one function of `(name, d_i, d_j, d_ij, n_i, n_j, n_k)`, vectorised
over every live cluster `k`:

```python
        if name == "single":
            return 0.5 * di + 0.5 * dj - 0.5 * np.abs(di - dj)
        if name == "complete":
            return 0.5 * di + 0.5 * dj + 0.5 * np.abs(di - dj)
        if name == "average":
            w = ni + nj
            return (ni * di + nj * dj) / w
        if name == "ward":
            w = ni + nj + nk
            return ((ni + nk) * di + (nj + nk) * dj - nk * dij) / w
        if name == "centroid":
            w = ni + nj
            return (ni * di + nj * dj) / w - (ni * nj / (w * w)) * dij
        if name == "median":
            return 0.5 * di + 0.5 * dj - 0.25 * dij
        raise ValueError(name)
```

`di` and `dj` are entire rows (`d(A,C)` and `d(B,C)` for every live `C`), so one call
rewrites the whole merged row — no Python loop over clusters. `dij` is a scalar, the
distance being consumed by this merge. `nk` is an array, so `ni + nk` broadcasts; that is
what makes `ward` and `centroid` size-weighted while `median` is not. The two `abs` forms
are the `min`/`max` identities; writing `np.minimum(di, dj)` would be equivalent and
marginally slower.

Whether the working numbers are distances or squared distances is decided once, at the
top of `fit`:

```python
        working = d * d if self.linkage in SQUARED else d

        cap = 2 * n - 1
        dist = np.full((cap, cap), np.inf)
        dist[:n, :n] = working
        np.fill_diagonal(dist, np.inf)
```

`SQUARED` is `{ward, centroid, median}` — the three whose algebra is derived on squared
distances. Squaring single/complete would be harmless (min and max commute with `x²`) but
squaring **average** would not: the mean of squares is not the square of the mean, and
the linkage would silently become something else. `cap = 2n − 1` is the number of nodes
in the full dendrogram, so cluster `n + m` is the result of merge `m` and no index
arithmetic is needed. Pre-filling with `inf` and setting the diagonal to `inf` means an
`argmin` over the matrix can never pick a dead row or a self-distance.

The merge loop is the naive `O(n³)` version, which is the clearest form:

```python
        for _ in range(n - 1):
            live = alive[:, None] & alive[None, :]
            cand = np.where(live, dist, np.inf)
            if self.variant == "connectivity_constrained":
                cand = np.where(allow, cand, np.inf)
            if not np.isfinite(cand).any():
                cand = np.where(live, dist, np.inf)   # drop the constraint, keep going
            i, j = divmod(int(np.argmin(cand)), cap)
            if i > j:
                i, j = j, i
            value = float(cand[i, j])
            h = self._height(value)
```

`live` masks the matrix down to pairs of live clusters and `np.argmin` on the flattened
array gives the pair; `divmod` recovers `(i, j)` from the flat index because the matrix is
`cap × cap`. The `if i > j` swap makes the pair canonical, which keeps `children_` and
the label propagation deterministic. The `not np.isfinite(...).any()` check is the
constraint's escape hatch: a disconnected k-NN graph would otherwise leave the loop with
nothing to merge and no way to finish the hierarchy, so the restriction is dropped rather
than raising.

The update of the new row is one Lance–Williams call plus a symmetric write:

```python
            ks = np.nonzero(alive)[0]
            ks = ks[(ks != i) & (ks != j)]
            new = self._lance_williams(
                self.linkage, dist[i, ks], dist[j, ks], value,
                float(size[i]), float(size[j]), size[ks].astype(float))
            dist[nxt, ks] = new
            dist[ks, nxt] = new
```

`ks` is selected *before* `i` and `j` are marked dead, so the filter has to exclude them
by hand. Writing both `dist[nxt, ks]` and `dist[ks, nxt]` keeps the matrix symmetric, which
matters because the next iteration's `argmin` scans both triangles. `value` (not
`dist[i, j]`) is passed as `dij` because under `cut_height` the loop may break before the
merge, and the height conversion happened already; passing the raw entry would be the
same number here but would break if a threshold ever modified it.

The constraint graph is a mutual k-NN graph, and the constraint is propagated by union:

```python
        idx = np.argpartition(d, k, axis=1)[:, :k + 1]
        adj = np.zeros((n, n), dtype=bool)
        adj[np.arange(n)[:, None], idx] = True
        adj &= adj.T
        np.fill_diagonal(adj, False)
```

`argpartition` picks the `k` nearest without sorting, and `k+1` columns include the point
itself, which `fill_diagonal` then removes. The single character that matters is `&=`
rather than `|=`: a *mutual* edge needs both endpoints to name each other, so a point
sitting near a different manifold cannot drag its cluster across on its own. The union
graph is the looser alternative and gives visibly worse results on the moons.

After a merge, the new cluster's allowed set is the union of its parents' sets:

```python
            if self.variant == "connectivity_constrained":
                link = allow[i, ks] | allow[j, ks]
                allow[nxt, ks] = link
                allow[ks, nxt] = link
```

`|` here, not `&`: once `A` and `B` are one cluster, an edge from `A` to `C` is an edge
from `A∪B` to `C`. Using `&` would require *both* parents to touch `C`, which shrinks the
graph with every merge until nothing can merge at all.

The flat labels come from replaying merges into a union-find:

```python
        for idx, (a, b, _, _) in enumerate(merges):
            if idx >= keep:
                break
            ra, rb = find(a), find(b)
            parent[ra] = n + idx
            parent[rb] = n + idx

        roots = np.array([find(p) for p in range(n)])
```

Stopping at `keep` merges is the cut: `n − k` for a fixed `k`, or every performed merge
for `cut_height`. `parent[ra] = n + idx` records that merge `idx` created node `n + idx`,
so `find` on any leaf walks up to the last merge that contained it. Path halving in `find`
keeps the walk short — the tree can be a caterpillar with `n` levels, and without
compression the label step would be quadratic. After the roots are known, sizes come from
`np.bincount` and clusters smaller than `min_cluster_size` become `-1`.

The height reported for the squared linkages is undone at the last moment:

```python
    def _height(self, value: float) -> float:
        return float(np.sqrt(max(value, 0.0))) if self.linkage in SQUARED else float(value)
```

`max(value, 0.0)` is not paranoia: `centroid` and `median` have negative `β`, so a
merged distance can come out slightly below zero from rounding, and `np.sqrt` of a
negative number is `nan`. Returning the square root keeps `distances_` in the same units
as the input, which is what makes the demo's `0.44` versus `7.45` comparison meaningful.

## Complexity

| Step | Time | Memory |
|:--|:--|:--|
| Initial distance matrix | `O(n² d)` | `O(n²)` |
| One merge: `argmin` over the matrix | `O(n²)` | — |
| `n−1` merges | `O(n³)` | `O(n²)` |
| Lance–Williams update | `O(n)` per merge | — |
| Labels | `O(n α(n))` | `O(n)` |
| `n_clusters` variants, all linkages | same loop, different constants | same |
| Constraint graph | `O(n²)` + `O(n k)` | `O(n²)` |

The `O(n³)` is the cost of rescanning the whole matrix for each merge, and it is why real
libraries do not do this. Nearest-neighbour-chain (`nn_chain`) reduces it to `O(n²)` in
time and memory; a priority queue of candidate pairs gives `O(n² log n)`; `SLINK` and
`CLINK` achieve `O(n²)` time with `O(n)` memory for the two extreme linkages. All of them
change the bookkeeping, not the result. The memory is the other wall: the distance matrix
is `8n²` bytes, so `n = 50k` needs 20 GB. For larger data the usual answers are sparse
constraints (only merge along a k-NN graph, `O(nk)` edges) or a subsample.

## Failure modes

1. **Chaining under `single`.** One bridge of points merges two dense blobs at a tiny
   height — the demo's `0.44` versus `7.45`. Real causes: a stray line of points, a
   duplicated axis, or a genuinely filamentary cluster. Use `average`/`complete`, or a
   connectivity constraint, and inspect the merge heights rather than only the labels.
2. **Ward's balance bias.** Ward prefers equal-sized clusters, so on a 300/20 split it
   scores `0.722` while `centroid` scores `0.922`. If cluster sizes are very unequal and
   the small ones matter, Ward is the wrong default.
3. **Inversions in `centroid`/`median`.** Merge heights can decrease, so the "tree" has
   crossing branches and a height cut is ambiguous. The demo counts `7` and `3`
   inversions against `0` for the monotone rules. If you need a threshold, do not use
   these two linkages.
4. **Greedy irreversibility.** A merge is never undone, so an early mistake propagates
   upward. The classic case is `complete` linkage merging two outlier points into a
   spurious cluster that then attracts everything else. There is no restart and no
   objective that later repairs it.
5. **`K` is a choice, not a result.** `n_clusters` decides where to cut, and different
   cuts of the same tree are all "valid". Use the merge-height gap (largest jump in
   `distances_`), a silhouette scan, or domain knowledge — the tree only tells you the
   candidate cuts.
6. **No noise model.** Every point ends in a cluster, including outliers, which is why
   `complete` linkage's accuracy collapses on data with background points. Only
   `cut_height` with `min_cluster_size` can say "this is noise"; single/complete/average
   cannot.
7. **Feature scale.** All linkages use Euclidean distance, so a feature measured in
   thousands dominates one measured in fractions. Standardise, or use a domain metric —
   the algorithm itself has no notion of units.
8. **`average`'s size weighting.** The `n_A/(n_A+n_B)` coefficients mean a large cluster
   barely moves when it absorbs a point, so UPGMA tends to produce one big cluster plus
   satellites rather than balanced ones.
9. **Disconnected constraint graph.** The mutual k-NN graph is not always connected, and
   the fallback silently drops the constraint and merges across the gap. Check that the
   graph has one component (or accept that the hierarchy is partly unconstrained); the
   `n_neighbors` value controls how likely this is.
10. **Quadratic memory.** `O(n²)` floats is the practical ceiling, well before the
    `O(n³)` time becomes painful. Use k-NN constraints, sparse linkage algorithms, or
    cluster a subsample and assign the rest to the nearest centroid.

## Applications

- **Phylogenetics and taxonomy.** UPGMA (`average`) on genetic distances is the classical
  tree-building method, and `single`/`complete` correspond to the minimax and maximum
  distance criteria. The dendrogram *is* the deliverable.
- **Document and topic hierarchies.** Agglomerative clustering over TF-IDF or embedding
  vectors gives a browsable tree, which is more useful than a flat partition when the
  consumer is a human looking for the right level of granularity.
- **Genomics and single-cell data.** Hierarchical clustering of expression profiles with
  Ward linkage and a correlation-derived distance is standard; the dendrogram's ordering
  is used to permute the heatmap.
- **Image segmentation.** Region merging on a graph of superpixels, with a connectivity
  constraint so segments stay spatially contiguous.
- **Anomaly-adjacent workflows.** Cutting low and marking small clusters as noise
  (`cut_height`) gives a cheap alternative to DBSCAN when a hierarchy is wanted anyway.
- **Customer and product taxonomies.** Ward on standardised behavioural features, cut at
  several heights to get segments, sub-segments and micro-segments from one fit.
- **As a sanity check for `K`.** Running `single` and `complete` on the same data and
  comparing the top merge heights says a lot about whether the clusters are genuinely
  separated or just an artefact of one linkage's bias.

## References

- Ward, *Hierarchical Grouping to Optimize an Objective Function*, 1963, JASA — the
  variance-minimising criterion and its merge cost.
- Lance & Williams, *A General Theory of Classificatory Sorting Strategies*, 1967, The
  Computer Journal — the recursive update formula shared by every linkage.
- Sneath & Sokal, *Numerical Taxonomy*, 1973 — UPGMA, UPGMC and WPGMC, and the
  non-monotonicity of the centroid methods.
- Sibson, *SLINK: An Optimally Efficient Algorithm for the Single-Link Cluster Method*,
  1973, The Computer Journal — `O(n²)` single linkage via the MST.
- Defays, *An Efficient Algorithm for a Complete Link Method*, 1977, The Computer Journal
  — `CLINK`, the complete-linkage counterpart.
- Gower & Ross, *Minimum Spanning Trees and Single Linkage Cluster Analysis*, 1969, JRSS-C
  — the MST equivalence this file's derivation relies on.
- Müllner, *Modern Hierarchical, Agglomerative Clustering Algorithms*, 2011, arXiv —
  nearest-neighbour-chain, `SLINK`/`CLINK`, and why the naive loop is `O(n³)`.
