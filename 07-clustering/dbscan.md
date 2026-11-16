# DBSCAN

`dbscan.py` · Clustering · Tier 1 · 4 variants

## Intuition

DBSCAN answers a different question from k-means. Instead of asking "which centre is
this point closest to?", it asks "is this point sitting in a dense region, and what is
that region connected to?". A cluster is a maximal set of points connected through
dense neighbourhoods — a *connected component of the density landscape*. Points in
sparse regions are labelled noise and left alone.

That one change buys the properties k-means cannot offer:

- **`K` is discovered, not specified.** The number of clusters is however many
  components the data has.
- **Arbitrary shapes.** Two interleaved crescents, a ring around a hole, a spiral — all
  are single clusters if the density along them stays above the threshold. k-means would
  cut them into convex pieces.
- **Explicit noise.** Sparse points get label `-1` instead of being forced into the
  nearest blob, which is why DBSCAN is a standard anomaly detector.

The two parameters are a radius `eps` and a count `min_samples`. A point is *core* when
its `eps`-ball contains at least `min_samples` points. Core points within `eps` of each
other are in the same cluster; non-core points within `eps` of a core point are
*border*; everything else is noise. The demo's two blobs plus uniform background come
out as exactly two clusters and 48 noise points, with `0.992` accuracy on the inliers.

The four variants are about *how* that definition is evaluated. `with_index` replaces
the brute-force scan with a grid, `weighted` replaces the row count with a weight sum,
and `fuzzy_dbscan` replaces the hard label with a membership degree.

## Variants

| Variant | Neighbourhood | Density test | Output |
|:--|:--|:--|:--|
| `standard` | scan all rows | `count ≥ min_samples` | hard labels |
| `with_index` | grid of `eps` cells | same | identical hard labels |
| `weighted` | scan all rows | `Σ weights ≥ min_samples` | hard labels |
| `fuzzy_dbscan` | scan all rows | same, plus a second radius | membership matrix |

`with_index` is an *exact* accelerator: it must return bit-identical labels, and the
demo asserts `array_equal` on both `labels_` and `core_sample_indices_`. The only thing
that changes is the number of distance evaluations — `67818` instead of `435600` on
660 points, and the gap grows with `n` and shrinks with `d`.

`weighted` matters because `min_samples` is otherwise a count of *rows*, and rows are
not always equally informative. A dataset that oversampled a region, or joined a
denormalised table, can have sixty identical rows where the original data had one
distinct point. Under row counting those duplicates are a dense region and become a
spurious cluster; the weighted variant gives each duplicate `1/60` and sees total mass
`1 < min_samples`. The demo shows exactly this: three clusters by row count, two when
weighted, with every duplicated row labelled noise.

`fuzzy_dbscan` is the soft version. With an inner radius `eps` and an outer radius
`eps2`, a point within `eps` of a cluster's core is a full member (`μ = 1`), a point
between the radii gets a linear membership `μ = (eps2 − d)/(eps2 − eps)`, and a point
beyond `eps2` gets zero. Because a point can sit inside two outer balls, memberships are
normalised to sum to at most one and the remainder is a *noise degree*. On the demo data
the noise points average `0.59` noise degree against `0.01` for inliers, and 15 points
have partial membership — the ones near a cluster boundary whose assignment is genuinely
ambiguous.

## Derivation

**Core, border, noise.** Fix `eps` and `min_samples`. Define the neighbourhood
`N_eps(x) = {y : d(x,y) ≤ eps}`. Then

$$
x \text{ is core} \iff |N_\epsilon(x)| \ge \text{min\_samples}
$$

and the relation *directly density-reachable* is `y ∈ N_eps(x)` with `x` core.
Density-reachability is not symmetric (a border point reaches nothing), so clustering
uses its symmetric closure: `p` and `q` are *density-connected* if some core `o` has
both density-reachable. A cluster is a maximal density-connected set. This definition is
the original Ester et al. one, and it is why the algorithm needs no `K`: the clusters
are the connected components of the core graph under `eps` adjacency.

**Why border points attach to the first cluster.** A border point can be within `eps` of
core points from two different components — a thin bridge between two dense regions. The
definition says it joins *a* cluster that reaches it, and DBSCAN's scan order decides
which. This is a real non-determinism in the original algorithm (the demo's
`standard` and `with_index` runs are identical only because the candidate list is
sorted), and it is one of the reasons the fuzzy variant exists.

**Complexity.** Without an index, each point needs one pass over the data:
`O(n² d)` distance work and `O(n)` neighbourhood queries per expansion. In low
dimensions the grid gives `O(n · c · d)` where `c` is the number of candidates in the
27 neighbouring cells — for uniformly spread data `c ≈ n · (3·eps)^d / volume`, which is
constant in `n` when the density is fixed. The catch is the exponent: `3^d` cells must be
visited, so the index stops helping around `d = 8–10`.

**Why the two radii give a membership.** The fuzzy extension keeps the core/border
structure but replaces the binary decision at the boundary with a grade. For a point
`x` and cluster `c` with core set `C_c`:

$$
d_c(x) = \min_{a \in C_c} \|x - a\|,
\qquad
\mu_c(x) =
\begin{cases}
1 & x \in C_c \\[2pt]
\operatorname{clip}\!\left(\dfrac{\epsilon_2 - d_c(x)}{\epsilon_2 - \epsilon},\,0,\,1\right) & \text{otherwise}
\end{cases}
$$

`eps2` is the *outer* radius: at `d = eps` the formula gives 1, at `d = eps2` it gives 0,
and between them it is linear. This is a deliberate simplification of the Ienco–Bordogna
construction, which grades the density itself (how many neighbours fall between the two
radii) rather than the distance to the nearest core; both produce a soft boundary and
the same qualitative behaviour.

Memberships from different clusters can overlap, so the vector is rescaled when
`Σ_c μ_c > 1`:

$$
\mu_c \leftarrow \frac{\mu_c}{\sum_j \mu_j},
\qquad
\nu(x) = 1 - \sum_c \mu_c
$$

`ν` is the noise degree: 1 for a point no cluster claims, 0 for a core point. The hard
label is the argmax, kept only when it clears `0.5` — which is what makes
`fuzzy_dbscan` comparable to `standard` on the same data (`0.995` vs `0.992` inlier
accuracy) while exposing the boundary cases as numbers.

**Weighted density.** Replace the count with the total mass of the neighbourhood:

$$
x \text{ is core} \iff \sum_{y \in N_\epsilon(x)} w_y \ge \text{min\_samples}
$$

With `w ≡ 1` this is exactly DBSCAN, so the weighted variant is a strict generalisation.
The natural weight for deduplicated data is `w_i = 1/m` for each of `m` copies of a
point: the copies then contribute mass 1 in total, as one observation should.

## Code walkthrough

Every neighbourhood question goes through one function, which is also where the
distance counter lives:

```python
    def _region(self, X: np.ndarray, i: int, use_index: bool):
        self.n_queries_ += 1
        cand = self._candidates(X, i) if use_index else np.arange(X.shape[0])
        self.n_dist_ += int(cand.size)
        d = np.linalg.norm(X[cand] - X[i], axis=1)
        keep = d <= self.eps + EPS
        return cand[keep], d[keep]
```

`n_dist_` counts *candidates*, not survivors, because the filtering itself is the cost
being measured. Returning `cand[keep]` rather than just the mask matters: neighbourhoods
are used as index lists immediately, and recomputing the mask at the call site would
double the work. The `+ EPS` is a tolerance for points exactly `eps` apart, which
otherwise flip in and out depending on rounding and can make labels order-dependent.

The grid index is a dictionary of cells:

```python
    def _build_grid(self, X: np.ndarray) -> None:
        keys = np.floor(X / self.eps).astype(int)
        grid: dict[tuple[int, ...], list[int]] = {}
        for i, key in enumerate(map(tuple, keys)):
            grid.setdefault(key, []).append(i)
        self._grid = grid
```

Cells are `eps` wide, so a point's neighbours can only be in its own cell or the 26
adjacent ones — that is the whole correctness argument for the index. `np.floor` is the
right operation (not integer division), because it behaves correctly for negative
coordinates: floor division would put `-0.1/0.35` in cell `0` and merge it with `0.1`.
The cell size is exactly `eps` rather than `2·eps` for the same reason: at `2·eps` the
`3^d` neighbourhood would have to be `5^d`.

Candidate collection enumerates those adjacent cells:

```python
        for offset in itertools.product((-1, 0, 1), repeat=X.shape[1]):
            cell = tuple(k + o for k, o in zip(key, offset, strict=True))
            out.extend(self._grid.get(cell, ()))
        return np.sort(np.array(out, dtype=int))
```

`strict=True` is not decoration: if `X.shape[1]` ever stopped matching the length of
`key`, `zip` would silently truncate and the index would return the wrong cells.
`np.sort` is what makes `with_index` produce *identical* labels to `standard` — border
points attach to the first cluster that reaches them, so candidate order is
semantically relevant here even though the neighbourhood set is not.

The expansion is the original algorithm, written as a queue with an index:

```python
            seeds = list(idx)
            head = 0
            while head < len(seeds):
                j = int(seeds[head])
                head += 1
                if not visited[j]:
                    visited[j] = True
                    nbr, _ = self._region(X, j, use_index)
                    if self._is_core(nbr):
                        core[j] = True
                        seeds.extend(nbr)
                if labels[j] < 0:
                    labels[j] = cluster
```

`visited` is set when a point is *popped*, not when it is pushed into `seeds`; that is
what allows a border point to be reached by a later cluster before it is processed, and
it is the source of the first-come-first-served behaviour. `labels[j] < 0` re-labels a
point that is still noise, but never overwrites an assignment — that single condition is
the difference between "border points join the first cluster that reaches them" and
"the last cluster wins". The `head` index implements a queue over a growing list; using
`pop(0)` would be `O(len(seeds))` per step and quadratic overall.

The density test is the only line the weighted variant changes:

```python
    def _is_core(self, idx: np.ndarray) -> bool:
        if self.variant == "weighted":
            return float(self._w[idx].sum()) >= self.min_samples
        return idx.size >= self.min_samples
```

Note that `_w` is populated for every variant in `fit`, so the branch is a policy
switch, not a separate code path. Because the weighted test can be satisfied by a single
heavily weighted point, a core set is no longer "at least `min_samples` distinct rows" —
document that, or a caller will be surprised that `min_samples=5` accepted one row.

Fuzzy clustering first finds the core components by breadth-first search over core
points:

```python
        comp = np.full(n, -1, dtype=int)
        k = -1
        for i in np.nonzero(core)[0]:
            if comp[i] >= 0:
                continue
            k += 1
            comp[i] = k
            stack = [int(i)]
            while stack:
                a = stack.pop()
                for b in nbrs[a]:
                    b = int(b)
                    if core[b] and comp[b] < 0:
                        comp[b] = k
                        stack.append(b)
```

`nbrs` was computed once for all points, so this pass is pure graph traversal. Only core
points are enqueued: a border point must not propagate a component, which is exactly the
asymmetry of density-reachability. This also gives `n_clusters_` directly, before any
membership is computed — a useful check that the fuzzy and hard versions agree on the
number of components.

Membership is a clipped linear ramp, then a normalisation:

```python
        span = self.eps2 - self.eps
        mu = np.clip((self.eps2 - d) / span, 0.0, 1.0)
        mu[core] = 0.0
        mu[core, comp[core]] = 1.0
        total = mu.sum(axis=1)
        over = total > 1.0
        mu[over] /= total[over, None]
```

`mu[core] = 0.0` clears whatever the ramp gave a core point — including its membership in
*other* clusters — and the next line sets its own cluster to exactly 1. A core point is
unambiguously in its own cluster by definition, so the ramp should not be allowed to
dilute it. `mu[core, comp[core]] = 1.0` uses boolean-mask row indexing with an integer
column array, which is the vectorised form of "for each core point, set its own column".
The `over` guard is needed because overlapping outer balls can give a row sum above 1,
and a membership matrix whose rows exceed 1 is not a distribution.

The hard label is a thresholded argmax:

```python
        labels = np.where(mu.max(axis=1) >= 0.5, np.argmax(mu, axis=1), -1)
```

`0.5` is a choice, not a theorem: it makes a point belong to a cluster when it is at
least halfway inside the outer band, and points below it become noise. Lower it to grow
clusters at the cost of absorbing background; raise it for conservative assignments.
`np.where` evaluates both branches, so `argmax` runs even for noise rows — harmless
here, and cheaper than a masked write.

## Complexity

| Variant | Time | Extra memory |
|:--|:--|:--|
| `standard` | `O(n² d)` | `O(n)` |
| `with_index` | `O(n · c · d)` with `c` candidates per cell block | `O(n)` plus the grid |
| `weighted` | `O(n² d)` | `O(n)` |
| `fuzzy_dbscan` | `O(n² d + n · n_core)` for the distance-to-core matrix | `O(n K)` memberships |

`c` is constant in `n` for fixed density and dimension, so the index changes the
asymptotic class in low `d`. In `d` dimensions it visits `3^d` cells per query, so the
practical crossover is around `d = 8`; beyond that the brute-force scan is competitive
and a k-d tree (which prunes by coordinate) is better. The fuzzy variant's second pass is
`n` distances per core point — `O(n·n_core)` — which is why it is the slowest variant
here; a real implementation would query the index per point instead.

Memory is the reason DBSCAN scales further than it looks: it never materialises an
`n×n` distance matrix, and the neighbourhoods are recomputed rather than stored.

## Failure modes

1. **Varying density.** One global `eps` cannot serve a tight cluster and a diffuse one.
   Set `eps` for the sparsest cluster you care about and accept that the tight one may
   merge with its neighbours, or move to HDBSCAN, which varies the threshold per point.
2. **Parameter sensitivity.** `eps` and `min_samples` interact: raising `min_samples`
   shrinks the neighbourhoods that qualify, and the pair must be tuned together. The
   standard heuristic is `min_samples ≈ d + 1` and `eps` from the knee of the
   k-distance plot.
3. **Bridging.** A thin chain of points between two blobs — duplicated rows, a data
   pipeline artefact, a genuine continuum — makes them one cluster, because connectivity
   is transitive. The demo's duplicate experiment is the synthetic version; weighted
   density and higher `min_samples` are the fixes.
4. **Duplicates inflate density.** Sixty copies of one point are a core neighbourhood
   under row counting. Deduplicate, or weight by `1/m`, as in the weighted variant.
5. **High dimensions.** Distance concentration makes every point roughly equidistant, so
   `eps` stops separating dense from sparse. Reduce dimensionality first; DBSCAN's
   premise is that "near" is meaningful.
6. **Border non-determinism.** A border point reachable from two clusters joins whichever
   the scan reaches first, so the partition can change under row reordering. Sorting
   candidates (as `with_index` does) makes it reproducible, and `fuzzy_dbscan` makes the
   ambiguity visible instead of silently resolving it.
7. **`min_samples` includes the point itself.** The neighbourhood of `x` contains `x`
   when `eps > 0`, so `min_samples=1` makes every point core and the whole dataset one
   cluster. The usual convention is `min_samples ≥ 3` for 2-D data.
8. **No `predict`.** A new point cannot be assigned without rerunning the clustering,
   because "is this region dense?" is a property of the fitted data. Production systems
   approximate it by keeping the core points and assigning to the nearest core within
   `eps`, which is exactly the fuzzy membership rule with `eps2 = eps`.
9. **Fuzzy threshold drift.** The `0.5` cut and the `eps2` choice jointly determine how
   many points are called noise; changing `eps2` changes the hard labels too, not just
   the degrees. Report the degrees, not only `labels_`, when the boundary matters.

## Applications

- **Anomaly and fraud detection.** Noise label as the alert, no supervision needed. The
  demo's background points are a stand-in for transactions that fit no spending pattern.
- **Spatial clustering.** Earthquake epicentres, crime incidents, GPS traces: DBSCAN is
  the default because clusters are geographic blobs of arbitrary shape and noise is
  meaningful.
- **Image and point-cloud segmentation.** Density in colour or coordinate space
  separates objects from speckle; `with_index` is what makes it usable on LiDAR clouds.
- **Customer behaviour.** Sessions in feature space with `-1` as "atypical user", which
  is often the segment a fraud or support team wants.
- **Scientific event detection.** Particle tracks, gene-expression clusters, star
  catalogues — any setting where the number of groups is unknown and outliers are data,
  not errors.
- **As a labelling step before supervision.** Cluster the unlabelled pool, then label
  each cluster; the noise set is the part of the pool that needs human review.

## References

- Ester, Kriegel, Sander & Xu, *A Density-Based Algorithm for Discovering Clusters in
  Large Spatial Databases with Noise*, 1996, KDD — the original DBSCAN, including the
  core/border/noise definitions.
- Schubert, Sander, Ester, Kriegel & Xu, *DBSCAN Revisited, Revisited: Why and How You
  Should (Still) Use DBSCAN*, 2017, ACM TODS — parameter heuristics, the `min_samples`
  convention, and the border-point non-determinism.
- Campello, Moulavi & Sander, *Density-Based Clustering Based on Hierarchical Density
  Estimates*, 2013, PAKDD — HDBSCAN, the answer to failure mode 1.
- Ienco & Bordogna, *Fuzzy Extensions of the DBSCAN Clustering Algorithm*, 2018, Soft
  Computing — the two-radius fuzzy core and the membership construction this file
  simplifies.
- Gan & Tao, *DBSCAN Revisited: Mis-Claim, Un-Fixability, and Approximation*, 2015,
  SIGMOD — why exact DBSCAN is hard to scale and what the approximations cost.
