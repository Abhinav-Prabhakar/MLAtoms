# Distances

`distances.py` · Primitives · Tier 1 · 13 variants

## Intuition

A distance is a definition of "similar", and every downstream algorithm inherits
it. kNN, k-means, DBSCAN, LOF, t-SNE, kernel methods and retrieval all fail in
the same way when the distance is wrong — and the failure looks like "the data has
no structure" rather than "you measured the wrong thing".

Three questions pick the metric: *Is the scale of each feature meaningful?* (if
not, standardise or use correlation), *Is the data a set?* (Jaccard/Dice), *Do the
sequences move at different speeds?* (DTW).

## Variants

| Variant | Family | Invariant to | Breaks when |
|:--|:--|:--|:--|
| `euclidean` | L2 | rotation | features have different units |
| `squared_euclidean` | L2² | rotation | same; no `sqrt` (cheaper, changes gradients) |
| `manhattan` | L1 | rotation | — (more robust to outliers) |
| `chebyshev` | L∞ | rotation | one noisy feature dominates |
| `cosine` | angular | magnitude | non-negative data (all vectors in one orthant) |
| `correlation` | angular on centred | magnitude **and** offset | constant vectors (0/0) |
| `mahalanobis` | whitened L2 | linear rescaling | `n < d` (covariance singular) |
| `jaccard` | set | counts, only presence | non-binary weighted sets |
| `dice` | set | counts | same |
| `hamming` | edit | magnitude | continuous data |
| `levenshtein` | edit | length differences | very long strings (O(nm)) |
| `dtw` | alignment | time warping | unnormalised series (length bias) |
| `haversine` | geodesic | — | distances > ~1000 km on an ellipsoid |

## Derivation

**Mahalanobis.** Euclidean assumes every direction is equally informative. With
covariance `Σ`, decorrelate and rescale first:

$$
d_M(x, y) = \sqrt{(x-y)^\top \Sigma^{-1} (x-y)}
= \| \Sigma^{-1/2}(x-y) \|_2
$$

After whitening, the distance is Euclidean — which is why Mahalanobis is the right
metric whenever features are correlated, and why it collapses to Euclidean when
`Σ = I`. Use the pseudo-inverse when `n < d`.

**Cosine vs correlation.** `cos` ignores magnitude; `corr` ignores magnitude *and*
the mean. For `x = [1,2,3]` and `y = [11,12,13]`, cosine distance is small but
non-zero (the vectors are not parallel), while correlation distance is exactly
zero (the centred vectors are identical). Choosing cosine when you mean "same
shape" is a common silent error.

**DTW.**

$$
D(i,j) = |a_i - b_j| + \min\{D(i-1,j),\; D(i,j-1),\; D(i-1,j-1)\}
$$

The three predecessors are the three ways to advance: stretch `a`, stretch `b`, or
match both. The Sakoe-Chiba band restricts `|i − j| ≤ w`, turning `O(nm)` into
`O(nw)` — and if the band excludes the endpoint the answer is `∞`, not a wrong
finite number. That is a feature.

**Haversine.** For latitude `φ` and longitude `λ`:

$$
d = 2R\arcsin\sqrt{\sin^2\!\frac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\!\frac{\Delta\lambda}{2}}
$$

The `arcsin` form is numerically stable for small distances where the spherical law
of cosines loses precision.

## Code walkthrough

`cdist` is one dispatch over a `VECTOR` whitelist, so a sequence metric such as
`"dtw"` fails loudly instead of falling through to a Euclidean branch. The
Euclidean case uses the expanded identity:

```python
        d2 = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
        d2 = np.maximum(d2, 0.0)
        if metric == "squared_euclidean":
            return d2
        np.fill_diagonal(d2, 0.0) if X is Y else None
        return np.sqrt(d2)
```

`‖x−y‖² = ‖x‖² + ‖y‖² − 2x·y` turns `n·m·d` work into one `n×d` by `d×m` matmul.
Floating point cancellation can make a true zero come out as `−1e-16`, and
`np.sqrt` of that is `nan`; the `maximum(..., 0)` is mandatory, not cosmetic. The
diagonal is then forced to exactly zero **only when `Y is X`** — an identity check,
not an equality check. Forcing it in the general case would silently zero legitimate
distances between duplicate rows in different arrays, which is a real bug in several
widely used implementations.

The non-Gram metrics materialise the full `n×m×d` difference tensor:

```python
    if metric == "manhattan":
        return np.abs(X[:, None, :] - Y[None, :, :]).sum(-1)
```

Correct and readable, but the memory is `n·m·d` floats — at `n=m=10⁴, d=768` that is
600 GB. The doc says "demo-scale only" for a reason; the production form is a blocked
loop over rows of `X`.

Cosine and correlation share a branch because they differ by one preprocessing step:

```python
        if metric == "correlation":
            X = X - X.mean(1, keepdims=True)
            Y = Y - Y.mean(1, keepdims=True)
        nx = np.linalg.norm(X, axis=1)[:, None]
        ny = np.linalg.norm(Y, axis=1)[None, :]
        sim = X @ Y.T / np.maximum(nx * ny, 1e-12)
        return np.clip(1.0 - sim, 0.0, 2.0)
```

Centring *is* the whole difference: `x = [1,2,3]` and `y = x + 10` have cosine
distance `> 0` but correlation distance `0`. The `np.maximum(..., 1e-12)` handles the
zero vector, and the clip handles the `1e-16` overshoot that makes a self-distance
`-2e-16`. Note that correlation still returns a distance, so a constant vector has
distance `1` to everything, including itself — mathematically the correlation is
undefined there, and the clip quietly picks a convention.

Mahalanobis is the only branch that needs a second argument:

```python
        cov = kwargs.get("cov")
        if cov is None:
            cov = np.cov(np.vstack([X, Y]).T)
        vi = np.linalg.pinv(cov)
        diff = X[:, None, :] - Y[None, :, :]
        return np.sqrt(np.maximum(np.einsum("ijk,kl,ijl->ij", diff, vi, diff), 0.0))
```

The default covariance pools `X` and `Y`, which is right for "distance within one
dataset" and wrong if the two arrays come from different populations — pass `cov`
explicitly then. `pinv` rather than `inv` is what makes `n < d` (singular covariance)
return a number instead of raising. The einsum is `diff @ vi @ diffᵀ` per pair, and
the `maximum` is again guarding the `sqrt` of a numerical zero.

Jaccard and Dice look like they share a numerator but their denominators differ:

```python
        num = np.minimum(X[:, None, :], Y[None, :, :]).sum(-1)
        den = np.maximum(X[:, None, :], Y[None, :, :]).sum(-1)
        if metric == "jaccard":
            return 1.0 - num / np.maximum(den, 1e-12)
        return 1.0 - 2.0 * num / np.maximum(X.sum(1)[:, None] + Y.sum(1)[None, :], 1e-12)
```

For binary rows, `sum(min)` is `|A∩B|` and `sum(max)` is `|A∪B|`; Dice instead divides
by `|A|+|B|`, which is why it does not reuse `den`. Both formulae extend to
non-negative weights but stop meaning "set overlap" there — with counts, `min/max` is
the *generalised* Jaccard and Dice is its fuzzy counterpart. The demo uses a `[1,1,0]`
vs `[1,0,0]` pair precisely because Jaccard `0.5` and Dice `1/3` differ and a shared
denominator would give one of them wrong.

`hamming` returns a *fraction*, not a count:

```python
        return (X[:, None, :] != Y[None, :, :]).mean(-1)
```

`mean` over the feature axis. Returning the count is the intuitive reading of "Hamming
distance" but makes the number incomparable across vectors of different length; the
demo asserts `1/3` on a length-3 pair, which pins the convention.

`levenshtein` keeps two rows instead of the full table:

```python
    prev = np.arange(len(b) + 1, dtype=float)
    for i, ca in enumerate(a, start=1):
        cur = np.empty_like(prev)
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1.0,                      # delete
                cur[j - 1] + 1.0,                   # insert
                prev[j - 1] + (0.0 if ca == cb else substitute),
            )
```

`prev[j]` is the row above (delete `a[i-1]`), `cur[j-1]` is the cell to the left
(insert `b[j-1]`), and `prev[j-1]` is the diagonal (match or substitute). The
initial `prev = arange` encodes "turn the empty prefix into `b[:j]`", and `cur[0] = i`
the reverse. Swapping `prev[j]` and `cur[j-1]` — an easy copy-paste — still produces a
symmetric-looking distance for many pairs but fails on `"ab"` vs `"ba"`, where the
correct answer is `2` and the transposed recurrence gives `1`.

`dtw` is the same DP with a different predecessor set, and the band is the subtle part:

```python
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0, j1 = 1, m + 1
        if band is not None:
            j0 = max(1, i - band)
            j1 = min(m, i + band) + 1
```

The borders are `inf`, not `0`: a cell outside the band must be unreachable, and
initialising the first row/column to zero (the usual DTW shortcut) lets every path
start anywhere and understates the cost. `D[0,0] = 0` is the only zero. `j0..j1` is
*inclusive* of `i ± band`, hence the `+1` on the right; getting that off by one
narrows the band by one cell per row and, for a series that only fits at exactly
`|i−j| = band`, returns `inf` — which the demo asserts rather than hides.

`haversine` is the only stateless scalar function:

```python
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
```

The `arcsin` form avoids the precision loss of `arccos` near zero, and
`np.clip(h, 0, 1)` is needed because at antipodal points the computed `h` can be
`1 + 2e-16`, where `arcsin(sqrt(·))` is `nan`. `r = 6371.0088` km is the mean Earth
radius; using 6378 (equatorial) is a 0.1% error that matters for city-scale distances
and not for country-scale ones.

## Complexity

`euclidean`/`squared` via the Gram matrix: `O(nmd)` time, `O(nm)` memory — the
memory, not the time, is the limit at `n = m = 10⁵`. `manhattan`/`chebyshev` as
implemented materialise `n×m×d`, which is why they are demo-scale only; use a
blocked loop in production. `levenshtein` and `dtw` are `O(nm)` with `O(min(n,m))`
and `O(m)` memory respectively. Haversine is `O(1)` per pair.

## Failure modes

1. **The curse of dimensionality.** In high dimensions, the ratio of nearest to
   farthest distance tends to 1, so "nearest neighbour" carries no signal. Fix:
   reduce dimensions first, or use a metric learned for the task
   (`03-kernels-and-instance/metric_learning.py`).
2. **Unscaled features.** One feature in the thousands dominates every Euclidean
   distance. Standardise, or use Mahalanobis.
3. **Cosine on non-negative data.** All points sit in one orthant, so cosine ranges
   over `[0,1]` — not `[−1,1]` — and the "angle" is compressed. Centre first
   (i.e. use correlation) for embeddings like TF-IDF.
4. **DTW length bias.** Longer series accumulate more cost. Normalise by path
   length for comparisons across series of different lengths.
5. **`pdist` diagonal drift.** Expanded Euclidean can produce `1e-16` on the
   diagonal and `sqrt` of a negative elsewhere. The implementation clips at zero;
   hand-rolled versions produce NaNs.

## Applications

- Embeddings/retrieval: cosine (normalised), with product quantisation for scale.
- Sensor and audio series: DTW, or a learned metric on top of it.
- Text sets (tokens, tags): Jaccard/Dice — the same functions used by segmentation
  overlap loss.
- Anomaly detection: Mahalanobis is the multivariate z-score, and the basis of
  `elliptic_envelope` and `mahalanobis` in chapter 09.
- Geolocation features: haversine, then a spatial index (KD-tree with a
  projection, or H3 cells).

## References

- Deza & Deza, *Encyclopedia of Distances*, 4th ed., 2016.
- Ratanamahatana & Keogh, *Three myths about dynamic time warping*, SDM 2005.
- Aggarwal, Hinneburg & Keim, *On the surprising behavior of distance metrics in high dimensional space*, ICDT 2001.
