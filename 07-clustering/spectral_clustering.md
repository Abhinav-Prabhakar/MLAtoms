# Spectral clustering

`spectral_clustering.py` · Clustering · Tier 1 · 5 variants

## Intuition

k-means and GMMs both assume that a cluster is a blob around a centre. Spectral
clustering assumes something weaker: **points that belong together are connected by many
short paths, and points in different clusters are connected by few long ones**. All the
geometry lives in a graph, and the clustering is a graph cut.

The construction is:

1. Build an affinity `W`, large when two points are similar. Here it is a k-NN graph
   with RBF weights.
2. Form the graph Laplacian `L = D − W`, where `D` is the diagonal of weighted degrees.
3. Take the `k` eigenvectors of `L` belonging to its smallest eigenvalues. Each row of
   that `n×k` matrix is a new coordinate vector for one point.
4. Run k-means on those coordinates.

Step 3 is the trick. Minimising the cut between two groups is NP-hard, but the relaxed
version is an eigenproblem, and its solution embeds the graph so that a good cut becomes
a *small distance* in `R^k`. On the demo's two interleaving moons, k-means on the raw
coordinates scores `0.793` because the moons are not linearly separable, while spectral
clustering on the same coordinates scores `1.000` — the graph already knows which points
are neighbours along each moon.

The variants differ in which Laplacian is used, how the embedding is normalised, how
the affinity scale is chosen, and how `k` clusters are extracted.

## Variants

| Variant | Laplacian | Embedding | Why it exists |
|:--|:--|:--|:--|
| `normalized` | `L_rw = I − D⁻¹W` | rows of `V = D^{-1/2}U` | Shi–Malik; the standard choice |
| `unnormalized` | `L = D − W` | rows of `U` | the original; minimises `RatioCut` |
| `ng_jordan_weiss` | `L_sym = I − D^{-1/2}WD^{-1/2}` | `U` with rows scaled to unit norm | Ng–Jordan–Weiss; best-conditioned |
| `self_tuning` | `L_sym` on a locally scaled affinity | unit rows | no global `σ` to choose |
| `multiway` | `L_sym` on each subgraph | Fiedler vector, recursively | avoids k-means entirely |

The first three are the same algorithm with different normalisations, and on easy data
they agree; the demo's circles give `1.000` for both `unnormalized` and
`ng_jordan_weiss`. Their differences show up in the *quality of the relaxation*:

- `unnormalized` minimises the ratio cut, which divides by the **number of points** in
  each side. It therefore prefers splitting off small groups, which is a bias toward
  unbalanced clusters.
- `normalized` minimises the normalised cut, which divides by the **volume** of each
  side. It is scale-free and prefers balanced clusters.
- `ng_jordan_weiss` uses the symmetric Laplacian and then projects each row onto the unit
  sphere. The row normalisation removes the effect of the degree from the embedding, so
  k-means sees directions rather than magnitudes; it is the most numerically robust of
  the three and the one to reach for by default.

`self_tuning` replaces the single `σ` by a per-point scale `σ_i` (the distance to the
`i`-th neighbour) and uses `exp(−d²/(σ_iσ_j))`. On the moons, a full RBF graph with
`σ = 1` scores `0.777`, because one global width that large blurs the two moons into a
single dense graph; local scaling scores `1.000` with no width parameter at all.

`multiway` never runs k-means. It splits the graph in two using the Fiedler vector (the
eigenvector of the second-smallest eigenvalue), then recursively splits the largest
piece until there are `k` clusters. Three concentric rings come out as `1.000` after
exactly `2` Fiedler solves for `k = 3`.

## Derivation

**The cut objective.** For a partition of the vertices into `A` and `Ā`, the cut is

$$
\text{cut}(A,\bar A) = \sum_{i\in A,\,j\in\bar A} W_{ij}
$$

Minimising `cut` alone has a trivial optimum: isolate one vertex. The two standard
fixes are

$$
\text{RatioCut}(A,\bar A) = \frac{\text{cut}(A,\bar A)}{|A|} + \frac{\text{cut}(A,\bar A)}{|\bar A|},
\qquad
\text{Ncut}(A,\bar A) = \frac{\text{cut}(A,\bar A)}{\text{vol}(A)} + \frac{\text{cut}(A,\bar A)}{\text{vol}(\bar A)}
$$

with `vol(A) = Σ_{i∈A} d_i`.

**Relaxation.** Encode the partition by an indicator vector `f ∈ {−1,+1}ⁿ` (or `{0,1}`).
For the unnormalised case,

$$
f^\top L f = \tfrac12\sum_{i,j} W_{ij}(f_i-f_j)^2 = \text{cut}(A,\bar A)\cdot 4
$$

because `(f_i − f_j)² = 4` exactly when the pair straddles the cut. So minimising
`fᵀLf` subject to `fᵀ1 = 0` (balanced sizes) and `‖f‖ = √n` is RatioCut in relaxed form.
`L` is symmetric positive semi-definite with `1` as its eigenvector for eigenvalue 0, so
the minimum over the orthogonal complement is the *second* smallest eigenvalue, attained
at the Fiedler vector. Dropping the integrality of `f` is the relaxation; rounding the
Fiedler vector's sign is the rounding step.

For `Ncut` the substitution `g = D^{1/2}f` gives

$$
\min_f \frac{f^\top L f}{f^\top D f}
\quad\Longleftrightarrow\quad
\min_g \frac{g^\top D^{-1/2} L D^{-1/2} g}{g^\top g}
$$

so the relaxed Ncut solution is the second eigenvector of the normalised Laplacian
`L_sym = I − D^{-1/2}WD^{-1/2}`. The random-walk form is similar: `L_rw = D⁻¹L = I − D⁻¹W`
has the same eigenvalues as `L_sym`, and its eigenvectors are `v = D^{-1/2}u`. Working
with `L_rw` and then normalising rows by `D^{-1/2}` is the Shi–Malik recipe.

**Why k eigenvectors for k clusters.** The same argument applies to a `k`-way cut with an
`n×k` indicator matrix `F` whose columns are the cluster indicators. The relaxation is
`min tr(FᵀLF)` subject to `FᵀF = I`, whose solution is the matrix of the `k` smallest
eigenvectors. k-means on the rows is a discrete approximation of that solution: it finds
`k` centroids in `R^k`, i.e. an `n×k` indicator matrix of the required form. That is why
the algorithm is "spectral embedding, then k-means" rather than "spectral cut, then
rounding".

**Row normalisation (NJW).** The theorem in Ng–Jordan–Weiss is about the matrix `U` of
the first `k` eigenvectors of `L_sym`. Their algorithm normalises each row to unit norm
before k-means, which makes the embedding invariant to the degree and turns the problem
into clustering *directions*. The theoretical guarantee is stated for that normalised
matrix: under a model with `k` well-separated clusters, the rows of `T = U/‖U‖` are close
to `k` orthogonal vectors, so k-means recovers the partition.

**Local scaling.** Zelnik-Manor and Perona's affinity is

$$
W_{ij} = \exp\!\left(-\frac{d^2(x_i,x_j)}{\sigma_i\sigma_j}\right),
\qquad \sigma_i = d(x_i, x_{K_i})
$$

The product `σ_iσ_j` is the geometric mean of the two local scales, so a pair inside a
dense region is compared at a short scale and a pair inside a sparse region at a long
one. The affinity between a dense and a sparse point is the geometric mean of the two,
which is exactly the regime where a global `σ` fails: too small and the sparse region
disconnects, too large and the dense region merges with everything.

**Multiway / Fiedler splitting.** For a subgraph with Laplacian `L_sym`, the Fiedler
vector `v₂` minimises the relaxed Ncut of that subgraph, and its sign gives the split.
Recursion approximates the `k`-way optimum by `k−1` binary decisions, which is cheaper
than a full `k`-eigenvector decomposition on a large graph (`O(m log n)` per split with a
Lanczos solver) and produces a hierarchy as a by-product.

## Code walkthrough

The affinity is a k-NN mask over an RBF kernel, with the mask taking the union of both
directions:

```python
        idx = np.argpartition(d2, k, axis=1)[:, :k + 1]
        mask = np.zeros((n, n), dtype=bool)
        mask[np.arange(n)[:, None], idx] = True
        mask |= mask.T
        np.fill_diagonal(mask, False)
```

`argpartition` finds the `k` nearest without sorting, and `k+1` columns include the point
itself. `mask |= mask.T` symmetrises the *graph*, not just the weights: if `i` chose `j`
but `j` did not choose `i`, the edge still exists. That is the standard "union" k-NN
graph, and it matters for spectral clustering because a one-sided edge would make `W`
asymmetric and the Laplacian's eigenvectors complex. `fill_diagonal(False)` removes
self-loops, which would only shift every degree by one.

The self-tuning branch needs the distance to the *local-scale* neighbour, not the graph
neighbour:

```python
            order = np.argsort(d2, axis=1)
            scale = np.sqrt(np.maximum(
                d2[np.arange(n), order[:, self.local_scale_neighbors]], EPS))
            denom = scale[:, None] * scale[None, :]
            W = np.exp(-d2 / np.maximum(denom, EPS))
```

`order[:, j]` is the index of the `j`-th nearest point for each row, and taking `d2` at
that index gives the squared distance without a second sort. The outer product
`scale[:, None] * scale[None, :]` is the `σ_iσ_j` in one expression — no loop over pairs.
`full_graph` is forced on for this variant because the local scales are what make a dense
affinity well behaved; sparsifying it would throw away the long-range edges that survive
the scaling.

The Laplacians are three lines each because `eigh` does the work:

```python
        dinv = 1.0 / np.sqrt(np.maximum(d, EPS))
        L = np.eye(n) - dinv[:, None] * W * dinv[None, :]
        vals, vecs = np.linalg.eigh(L)
```

`dinv[:, None] * W * dinv[None, :]` is `D^{-1/2}WD^{-1/2}` as two broadcast multiplies,
never a matrix product with a diagonal matrix. `np.linalg.eigh` is the symmetric
eigensolver: it returns eigenvalues in ascending order, so `vecs[:, :k]` is already the
`k` smallest. `np.maximum(d, EPS)` protects isolated vertices — a point with no edges has
degree 0, and without the guard the embedding row would be `nan` and poison k-means.

The random-walk embedding is a rescale of the symmetric one:

```python
        V = dinv[:, None] * vecs[:, :k]
        return vals, V
```

Because `L_rw = D^{-1/2}L_symD^{1/2}`, its eigenvectors are `D^{-1/2}u`. This is exactly
why `normalized` and `ng_jordan_weiss` differ only in the row scaling: they use the same
eigendecomposition, and the variant decides whether rows are left as `V` or normalised
to unit length.

The Fiedler solve for a subgraph reuses the same construction on the principal submatrix:

```python
        sub = W[np.ix_(idx, idx)]
        d = sub.sum(axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(d, EPS))
        L = np.eye(idx.size) - dinv[:, None] * sub * dinv[None, :]
        _, vecs = np.linalg.eigh(L)
        self.n_eig_ += 1
        return dinv * vecs[:, 1] if idx.size > 1 else vecs[:, 0]
```

`np.ix_(idx, idx)` extracts the principal submatrix in one operation, which is the whole
"run the algorithm on a subset" step. `vecs[:, 1]` is the second eigenvector because the
first is constant. The `dinv *` converts it to the random-walk eigenvector, which is the
correct object for the normalised cut; using the symmetric eigenvector directly would
still split reasonably but would bias the split toward low-degree vertices.

The recursion always splits the biggest cluster:

```python
        while len(clusters) < self.k:
            j = max(range(len(clusters)), key=lambda i: clusters[i].size)
            idx = clusters.pop(j)
            v = self._fiedler(W, idx)
            left = v > 0.0
            if left.all() or not left.any():
                left = v > float(np.median(v))
```

Splitting the largest cluster balances the recursion tree and is what makes "k−1 splits"
meaningful — splitting the smallest would spend solves on already-small pieces. The sign
test `v > 0` fails when the vector is one-signed, which happens on a connected subgraph
whose Fiedler value is tiny; the median fallback guarantees a non-empty split and keeps
the loop from spinning. A truly unsplittable cluster (`idx.size < 2`) is put back and the
loop breaks, so `k` is an upper bound rather than a promise.

The main dispatch is three lines:

```python
        if self.variant == "multiway":
            self.labels_ = self._multiway(self.affinity_)
            self.embedding_ = None
            self.eigenvalues_ = None
        else:
            self.eigenvalues_, self.embedding_ = self._laplacian_eigen(
                self.affinity_, self.k)
```

Setting `embedding_` and `eigenvalues_` to `None` for `multiway` is deliberate: there is
no single embedding, and returning a stale one from a previous fit would be worse than
returning nothing.

k-means on the embedding is a sibling import, not a reimplementation:

```python
            km = KMeans("lloyd", n_clusters=self.k, n_init=10,
                        random_state=int(self.rng.integers(2**31 - 1)))
            self.labels_ = km.fit_predict(self.embedding_)
```

`n_init=10` because the embedding can have clusters of very different sizes, and a single
Lloyd run on a bad seed merges two of them. The RNG is seeded from this model's generator
so the whole pipeline is reproducible from `random_state` alone.

## Complexity

| Step | Time | Memory |
|:--|:--|:--|
| Affinity matrix | `O(n² d )` (dense) | `O(n²)` |
| k-NN selection | `O(n²)` with `argpartition` | `O(n²)` |
| Eigendecomposition | `O(n³)` dense `eigh` | `O(n²)` |
| k-means on `n×k` | `O(n k² i)` | `O(n k)` |
| Multiway, `k−1` splits | `O(k n³)` worst case | `O(n²)` |

The dense eigendecomposition is the wall: `O(n³)` with a small constant, fine to a few
thousand points and hopeless beyond that. Real implementations never build `W`: they use
a sparse k-NN graph and a Lanczos/LOBPCG solver for the few smallest eigenvectors, which
is `O(m k)` per iteration with `m ≈ n·n_neighbors` edges. The affinity matrix is also the
memory wall: `n = 100k` needs 80 GB. `multiway` looks worse in the worst case but each
sub-solve is on a smaller matrix, so in practice it is competitive and uses less peak
memory.

## Failure modes

1. **Choosing `k`.** The eigengap heuristic — pick `k` where `λ_k` jumps — is the
   standard tool, and it fails when the graph has no clear gap. The demo's moons have
   `λ₀, λ₁ ≈ 0` and `λ₂ = 0.0025`: a gap, but a small absolute one. Try a few `k` and
   validate on a downstream task rather than trusting the spectrum alone.
2. **The affinity width.** With a dense RBF graph, `σ` decides everything; on the moons,
   `σ = 1` scores `0.777` while `σ` from the median neighbour distance scores `1.000`.
   With a k-NN graph the width matters much less because the *edges* are fixed by the
   neighbour count and `σ` only reweights them — which is why the k-NN graph is the
   default here.
3. **Disconnected graphs.** If the k-NN graph has more than `k` components, the first
   several eigenvalues are all zero and k-means splits arbitrary components. That is
   usually a correct answer (the data really is disconnected), but it is reported
   without warning. Check `λ_{k-1} ≈ 0` and whether the clusters are meaningful.
4. **Very unbalanced cluster sizes.** `unnormalized` minimises RatioCut, which prefers
   isolating small groups; on data with one dominant cluster it will carve off slivers.
   Use the normalised variants, which divide by volume.
5. **Degree-0 vertices.** An isolated point has `d = 0` and an undefined normalised
   Laplacian. The `EPS` guard gives it a zero row, so it lands wherever k-means puts the
   origin — usually noise. Drop isolated vertices before fitting if they matter.
6. **Outliers.** A far outlier is its own connected component in the k-NN graph, so it
   becomes a cluster of one and consumes a slot of `k`. Trim first, or increase
   `n_neighbors` so the outlier attaches to something.
7. **Large `n`.** The dense path is `O(n²)` memory and `O(n³)` time. Use a sparse graph
   and a Lanczos solver, or Nyström approximation; the algorithm's structure is fine,
   this implementation is the limit.
8. **`multiway` split degeneracy.** A one-signed Fiedler vector on a well-connected
   subgraph forces the median fallback, which is an arbitrary split. The recursion then
   produces a cluster that does not correspond to a real cut. Prefer the `k`-eigenvector
   method when the subgraphs are not clearly separable.
9. **Sign ambiguity of eigenvectors.** `v` and `−v` are both valid, so labels can flip
   between runs. Cluster *labels* must be compared with a permutation-invariant metric;
   the demo's accuracy helper does exactly that.

## Applications

- **Manifold-shaped clusters.** Two moons, concentric circles, spirals: spectral
  clustering is the standard answer when k-means visibly fails, and the demo's
  `1.000` vs `0.793` is the canonical comparison.
- **Image segmentation.** Pixels as vertices, colour/position similarity as edges; the
  normalised cut is the original Shi–Malik application.
- **Speaker diarisation.** Segments as vertices, similarity of speaker embeddings as
  weights; the number of speakers is `k`, and the eigengap suggests it.
- **Community detection.** Graph clustering with the normalised Laplacian is the
  spectral counterpart of modularity methods, and it scales when the graph is sparse.
- **Constrained and semi-supervised clustering.** Setting `W_ij = 0` for must-not-link
  pairs and `W_ij = large` for must-link pairs turns constraints into graph edits, with
  no change to the algorithm.
- **As a feature map.** The `k` eigenvectors are a non-linear embedding usable by any
  downstream model, not only by k-means.

## References

- Fiedler, *Algebraic Connectivity of Graphs*, 1973, Czechoslovak Mathematical Journal —
  the second eigenvector and its role in graph partitioning.
- Shi & Malik, *Normalized Cuts and Image Segmentation*, 2000, IEEE TPAMI — the
  normalised cut, the random-walk Laplacian and the algorithm this file's `normalized`
  variant follows.
- Ng, Jordan & Weiss, *On Spectral Clustering: Analysis and an Algorithm*, 2002, NeurIPS
  — the symmetric Laplacian with row normalisation and its guarantee.
- Zelnik-Manor & Perona, *Self-Tuning Spectral Clustering*, 2004, NeurIPS — local scaling
  and the `self_tuning` variant.
- von Luxburg, *A Tutorial on Spectral Clustering*, 2007, Statistics and Computing — the
  clearest derivation of the relaxation, the three Laplacians and their equivalences.
- Meilă & Shi, *A Random Walks View of Spectral Segmentation*, 2001, AISTATS — the
  recursive multiway splitting used by `multiway`.
