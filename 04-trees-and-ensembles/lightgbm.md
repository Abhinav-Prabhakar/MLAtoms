# LightGBM

`lightgbm.py` · Trees & ensembles · Tier 1 · 7 variants

## Intuition

LightGBM keeps XGBoost's second-order objective and changes the *engineering*
around it. Three of those changes are why it trains on hundreds of millions of rows
where exact XGBoost cannot:

- **Histogram splits.** Bin each feature once into `max_bins` buckets. A node's
  split search then accumulates gradients into bins — `O(n)` instead of
  `O(n log n)` — and every candidate threshold is a bin boundary, so ties and
  duplicates cost nothing.
- **Leaf-wise growth.** Level-wise growth splits every node at a depth. Leaf-wise
  growth keeps a priority queue of candidate splits and always splits the leaf with
  the largest gain, so a fixed leaf budget is spent where it reduces the loss most.
- **GOSS.** The rows with small gradients are nearly fitted already; keep all the
  large-gradient rows and sample a fraction of the rest, reweighting the sample so
  the gradient sum stays unbiased.

Two more variants are representation tricks: **EFB** packs mutually exclusive sparse
features into a single column, and **categorical splits** group levels by their
gradient statistics instead of one-hot expanding them. **Monotone constraints** are a
correctness feature, not a speed one: they clip every leaf to bounds inherited from
its ancestors so the fitted function cannot decrease in a feature you declare
increasing.

## Variants

| Variant | What changes | Why it exists |
|:--|:--|:--|
| `histogram` | global quantile bins, level-wise growth | the baseline fast booster |
| `leaf_wise` | best-first growth from a heap, capped by `num_leaves` | lower loss per leaf |
| `goss` | keep top-gradient rows, sample the rest, reweight | fewer rows per stage, same estimate |
| `efb` | bundle exclusive sparse columns | fewer features to scan |
| `dart` | drop trees during fitting with `1/(1−p)` scaling | spreads contribution across the ensemble |
| `categorical` | split categories by sorted `G/(H+λ)` | no one-hot explosion |
| `monotone` | clip leaves to path bounds | enforce a known monotone effect |

`histogram` and `leaf_wise` share the same split machinery and differ only in the
order in which nodes are expanded — the demo compares them at an identical leaf
count (`8`) and the leaf-wise tree wins (`0.0182` vs `0.0208` training loss).
`goss` and `efb` are data-reduction strategies that leave the objective untouched, so
their trees are ordinary histogram trees fitted on less work. `dart`, `categorical`
and `monotone` change the model class itself.

## Derivation

**Objective.** Same as XGBoost: expand the loss to second order around the current
prediction,

$$
\mathcal{L} \approx \sum_i \left[g_i f(x_i) + \tfrac12 h_i f(x_i)^2\right]
+ \tfrac12\lambda\sum_j w_j^2
$$

so a leaf with gradient sum `G` and Hessian sum `H` has value

$$
w = -\frac{G}{H+\lambda},
\qquad
\text{gain} = \frac12\left[\frac{G_L^2}{H_L+\lambda}
+ \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]
$$

The difference from `xgboost.py` is only how `G_L, H_L` are obtained.

**Histogram split.** For feature `j` and bin edges `e_1 < … < e_{B−1}`, map each value
to a bin `b_i = #{e ≤ x_ij}` and accumulate

$$
G_b = \sum_{i:\,b_i=b} g_i, \qquad H_b = \sum_{i:\,b_i=b} h_i, \qquad N_b = |\{i: b_i=b\}|
$$

The prefix sums `cumsum(G_b)` then give every candidate threshold's left child in one
pass. Cost per feature per node is `O(n + B)` instead of `O(n log n + n)`: the sort is
paid once, globally, at fit time. `B = 255` is the LightGBM default because it fits a
byte; this implementation defaults to `32` to keep ties frequent enough to be visible.

**Leaf-wise growth.** Let `S` be the set of leaves currently splittable. Level-wise
expands all of `S` at the current depth. Leaf-wise expands

$$
\ell^\* = \arg\max_{\ell \in S} \text{gain}(\ell)
$$

then inserts the two children back into `S`. With a leaf budget `L`, this is a greedy
best-first search that maximises the total gain of `L−1` splits subject to the tree
being a prefix-closed set — the same greedy principle as CART, applied to the order of
expansion rather than the choice of split. The price is depth: a leaf-wise tree can
be very unbalanced, so `max_depth` is kept as a second guard, exactly as LightGBM
does with `num_leaves` + `max_depth`.

**GOSS.** Let `a` be the top-gradient fraction, `b` the sampled fraction of the rest.
The estimate is

$$
\tilde G = \sum_{i\in A} g_i + \frac{1-a}{b}\sum_{i\in B} g_i
$$

with `A` the top-`a·n` rows and `B` a uniform sample of `b·n` rows from the remainder.
Since `E[|B|] = b·n_(1−a)` and each sampled row stands for `1/b` of its stratum,

$$
\mathbb{E}\left[\frac{1-a}{b}\sum_{i\in B} g_i\right]
= \frac{1-a}{b}\cdot b \sum_{i\notin A} g_i
= \sum_{i\notin A} g_i
$$

so `G̃` is unbiased for the full gradient sum. The variance comes only from the
sampled stratum, and gradients shrink as boosting converges, which is why the
approximation improves over stages. The demo's `a=0.2, b=0.1` uses 105 of 350 rows per
stage and still reaches `R²=0.960`, essentially the full-data number.

**EFB.** Two features are *exclusive* if they are never nonzero in the same row. A set
of mutually exclusive features can share one column if each feature is shifted into a
disjoint numeric range (`x_j + offset_j`), because at most one of them is nonzero and
the others contribute their zero. Real data is not perfectly exclusive, so LightGBM
allows a conflict rate `γ` (1% here): a feature joins a bundle if the fraction of rows
where it conflicts is below `γ`. The packing must be invertible, which is why the map
stores `(feature, offset)` pairs.

**Categorical splits.** For a categorical feature with levels `c`, compute
`G_c, H_c` per level and sort the levels by `G_c/(H_c+λ)`. For the second-order
objective, an optimal partition of the levels into "left" and "right" is a *contiguous*
prefix of this ordering. The proof is the same exchange argument as in Fisher's
optimal binning: if `u` sorts before `v` but `u` is on the right and `v` on the left,
swapping them cannot decrease the gain, because the gain depends on the partition only
through the sums, and `G/(H+λ)` is monotone in the direction that makes the swap
favourable. That reduces `2^K` partitions to `K−1` prefix cuts.

**Monotone constraints.** If feature `j` must be nondecreasing, then for every split on
`j` the right child's values must be at least the left child's. Enforce it
recursively: a node holds bounds `[lo, hi]`, a split is rejected when `w_L > w_R`, and
the children inherit `[lo, w_R]` and `[w_L, hi]`. Clipping each leaf to its bounds
guarantees the invariant for every pair of leaves separated by a `j`-split, without
changing the tree's structure or the split search.

## Code walkthrough

EFB is the clearest place to start, because it rewrites the data before anything else
happens:

```python
        for j in sparse:
            for b in bundles:
                conflict = np.mean((X[:, j] != 0.0) & np.any(X[:, b] != 0.0, axis=1))
                if conflict <= 0.01:
                    b.append(j)
                    break
            else:
                bundles.append([j])
```

A feature is *sparse* if it is zero in more than half the rows and non-negative —
one-hot columns and count features qualify, a centred continuous column does not.
`np.any(X[:, b] != 0.0, axis=1)` is "this bundle is active in this row"; the mean of
its conjunction with the new feature's activity is the conflict rate. The `for … else`
runs only when no existing bundle accepted the feature, which is when a new bundle is
created — the same idiom CART uses for "no step accepted".

Packing is where the offsets come from:

```python
            for j in b:
                col = np.where(X[:, j] != 0.0, X[:, j] + offset, col)
                feats.append((j, offset))
                offset += float(X[:, j].max()) + 1.0
```

`offset` grows past each feature's maximum, so the ranges `[0, max_j]`, `[max_j+1, …]`
are disjoint and a nonzero value identifies its feature uniquely. `np.where` keeps the
existing bundle value for rows where `x_j` is zero, which is what makes the columns
*merge* rather than overwrite. The demo's one-hot block of 8 columns plus one dense
column becomes 2 columns, and the regression `R²` stays at `0.941`.

Bin edges are global and computed once, which is the whole speed argument:

```python
    def _edges(self, X):
        out = []
        for j in range(X.shape[1]):
            v = X[:, j]
            v = v[~np.isnan(v)]
            if v.size < 2:
                out.append(np.array([]))
            else:
                out.append(np.unique(np.quantile(
                    v, np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1])))
        return out
```

`[1:-1]` drops the 0% and 100% quantiles: a threshold at the minimum or maximum has an
empty child and is never useful. `np.unique` collapses duplicated quantiles, which is
common on discrete features and would otherwise create bins with no rows. Because the
edges are global, every node's `G_b` refers to the same partition, so the cumulative
sums are comparable across nodes and across trees.

A node's split search is then two `bincount`s and two `cumsum`s:

```python
        b = np.searchsorted(edges, v, side="right")
        nb = edges.size + 1
        Gb = np.bincount(b, weights=g[rows], minlength=nb)
        Hb = np.bincount(b, weights=h[rows], minlength=nb)
        Nb = np.bincount(b, minlength=nb)
        cG, cH, cN = np.cumsum(Gb), np.cumsum(Hb), np.cumsum(Nb)
```

`side="right"` makes a value exactly equal to an edge fall into the upper bin, which
matches the `x <= threshold` convention: the threshold *is* the edge, so values equal
to it go left in the split but are binned with the upper group. Since the split at bin
`k` means "left = bins `0..k`", the two conventions agree through `cG[:-1]`, which is
the left child for every candidate `k` at once. `Nb` counts rows, not weights, because
`min_samples_leaf` is a row constraint.

The gain is the XGBoost objective with `λ`, and it is shared by every variant:

```python
    def _split_gain(self, GL, HL, GR, HR, GP, HP):
        def score(G, H):
            return G * G / (H + self.reg_lambda)

        return 0.5 * (score(GL, HL) + score(GR, HR) - score(GP, HP))
```

No `gamma` here: LightGBM's `min_gain_to_split` is applied by the `gain <= 0.0` test
in `_grow`. The `½` is the Taylor factor and cancels in comparisons, but keeping it
makes the value directly comparable to XGBoost's.

Leaf-wise growth is a heap of pending splits:

```python
            heapq.heappush(heap, (-split["gain"], counter, node, idx, depth, lo, hi, split))
            counter += 1
```

```python
        while heap and leaves < self.num_leaves:
            _, _, node, idx, depth, lo, hi, split = heapq.heappop(heap)
```

`heapq` is a min-heap, so gains are negated. `counter` is a strictly increasing
tie-breaker: without it, Python would try to compare the `dict` payloads when two
gains are equal, which raises `TypeError`. The popped entry carries the *node dict
object*, so splitting it is `node.clear(); node.update(...)` — the leaf is turned into
an internal node in place, keeping the parent's reference valid.

GOSS is the whole sampling strategy in seven lines:

```python
    def _goss(self, g: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        top = max(1, int(self.top_rate * n))
        rest = max(1, int(self.other_rate * n))
        order = np.argsort(-np.abs(g), kind="mergesort")
        big, small = order[:top], order[top:]
        take = self.rng.choice(small, size=min(rest, small.size), replace=False)
        idx = np.concatenate([big, take])
        w = np.ones(idx.size)
        w[top:] = (1.0 - self.top_rate) / self.other_rate
        return idx, w
```

`argsort(-abs(g))` ranks by gradient magnitude, and `mergesort` keeps ties in index
order so the result is reproducible. The weights are attached to the *samples*, not to
the rows: `w[top:]` is the inverse sampling probability `(1−a)/b` for the small-gradient
stratum, and the top rows keep weight 1 because none of them were dropped. `min(rest,
small.size)` guards a small `n` where `b·n` exceeds the stratum.

Missing values are handled as a third child, the same way as XGBoost:

```python
        gains = [
            (self._split_gain(GL, HL, GR, HR,
                              float(g[rows].sum()), float(h[rows].sum())),
             (n_l >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf)
             & (n_m == 0), True),
            (self._split_gain(GL + Gm, HL + Hm, GR, HR, GP, HP),
             ((n_l + n_m) >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf),
             True),
```

The first entry is only valid when `n_m == 0`, because its parent sum is over observed
rows only. Scoring a missing-left or missing-right split against that parent would
understate the parent objective and inflate the gain.

Categorical splits sort levels by their gradient-to-Hessian ratio:

```python
        order = np.argsort(Gc / (Hc + self.reg_lambda), kind="mergesort")
        cats, Gc, Hc, Nc = cats[order], Gc[order], Hc[order], Nc[order]
        cG, cH, cN = np.cumsum(Gc), np.cumsum(Hc), np.cumsum(Nc)
```

`Hc + λ` is the same regularised curvature as in the leaf value; a level with zero
Hessian would otherwise divide by zero. The prefix sums then evaluate every
"left = first `k` levels" partition. `cats[: k + 1]` is stored on the node, so
prediction is a set-membership test rather than an ordering test — the levels have no
natural order, and the stored set *is* the learned grouping.

Monotone constraints are enforced by bounds passed down the recursion:

```python
        if sign > 0:
            return True, (lo, min(hi, split["wr"])), (max(lo, split["wl"]), hi)
        return True, (max(lo, split["wr"]), hi), (lo, min(hi, split["wl"]))
```

The left child of an increasing split may not exceed the right child's optimal value,
and the right child may not fall below the left's. `min(hi, …)`/`max(lo, …)` composes
with the bounds already inherited from ancestors, which is what makes the constraint
hold *across* different splits on the same feature, not just at one node. The values
`wl`, `wr` are the unconstrained optimal leaf values computed in `_hist_split`; they
are stored on the split for exactly this check. The demo's cubic fits a nondecreasing
step function from `−4.95` to `9.27` with no violations.

The leaf value clips to those bounds:

```python
    def _leaf(self, idx, g, h, lo=-np.inf, hi=np.inf):
        v = float(np.clip(self._leaf_value(float(g[idx].sum()),
                                           float(h[idx].sum())), lo, hi))
        return {"leaf": True, "n": idx.size, "value": v}
```

`np.clip` with infinite bounds is a no-op for unconstrained features, so the same code
path serves both. Clipping rather than rejecting is what keeps the fit close to the
unconstrained optimum: a leaf that overshoots the bound is moved to the bound, not
discarded.

## Complexity

| Operation | Time | Memory |
|:--|:--|:--|
| Binning once at fit | `O(n d log n)` (quantiles per feature) | `O(n B)` if materialised |
| One histogram split, one feature | `O(n + B)` | `O(B)` |
| One tree, level-wise | `O(d n · depth)` after binning | `O(nodes)` |
| One tree, leaf-wise | `O(d n · num_leaves)` worst case | `O(num_leaves)` heap |
| Prediction, one row | `O(depth)` | `O(1)` |
| GOSS stage | `O(n log n)` for the top-`a` selection | `O(n)` |

The `O(n log n)` in GOSS is the partial sort of gradients; LightGBM uses a
threshold-based selection to make it `O(n)`. The histogram is what makes the per-node
cost linear in `n` instead of `n log n`: after binning, the split search touches each
row once to accumulate into a bin. `num_leaves` bounds the leaf-wise work: each split
costs one histogram pass over its node's rows, and the heap holds at most one entry per
splittable leaf.

## Failure modes

1. **Too few bins.** `max_bins=16` merges distinct regimes; a sharp threshold can be
   missed entirely. The demo's 16-bin run is the visible cost, and production uses
   255. Raise the bins before adding trees when the data has fine structure.
2. **Leaf-wise overfitting.** Best-first growth keeps splitting the largest-gain leaf,
   which on noisy data is often the one that memorises. Cap `num_leaves` and
   `min_samples_leaf` together, and validate — the demo's 12-leaf comparison is only
   fair because both models have the same leaf budget.
3. **GOSS on the wrong loss.** The top-gradient selection assumes gradients are
   comparable across rows. With weights (sample weights, class weights) or a
   non-additive loss the ranking must include them, or the estimate is biased.
4. **GOSS and small `n`.** `top_rate=0.2` on 100 rows keeps 20 rows, and `other_rate=0.1`
   samples 10; the tree sees 30 rows and the variance is large. Switch to full data
   below a few thousand rows.
5. **EFB conflicts.** The 1% threshold is a heuristic. Highly overlapping sparse
   columns will not bundle, and the "sparse" test (`mean(x≠0) < 0.5`) rejects a
   one-hot block whose levels are too dense — check `n_features_` against the input
   width to see whether bundling fired.
6. **Categorical levels with tiny support.** A level with three rows has a wild
   `G/(H+λ)` and can be sorted into a group of its own, encoding noise. LightGBM
   merges rare categories first; this implementation does not, so pre-group levels or
   set `min_samples_leaf` high enough to reject the split.
7. **Monotone constraints and noisy data.** The constraint is hard: if the true effect
   is non-monotone, the fit is biased and the loss can be much worse. Use it when the
   monotonicity is a domain requirement, not as a regulariser.
8. **Missing values and monotone bounds.** Bounds are computed from `wl`/`wr` on
   observed rows; a missing-value branch can land on the "wrong" side of the order.
   The demo has no missing values in the monotone test for exactly this reason.
9. **DART cost.** Recomputing `F` from scratch each stage is `O(B² n)`; with 1,000
   trees that dominates. The `1/(1−p)` scale must appear in training only — applying it
   at prediction inflates every tree.

## Applications

- **Web-scale ranking and CTR.** Histogram splits and GOSS make gradient boosting
  feasible at billions of rows; categorical splits handle user and item IDs without a
  one-hot explosion.
- **Fraud and risk with monotone effects.** A monotone constraint on "debt-to-income"
  or "transaction velocity" is often a compliance requirement, and the bound
  propagation gives it without hand-written rules.
- **Sparse text and one-hot pipelines.** EFB is the reason a bag-of-words matrix can go
  straight into a booster: exclusive indicator columns collapse into a handful of
  packed features.
- **Tabular AutoML.** LightGBM is the default learner in most AutoML stacks because it
  is fast, handles missing values and categories natively, and its leaf-wise growth
  reaches a given loss with fewer leaves.
- **As the fast baseline in research.** When the point is the experiment, not the
  model, a 30-second LightGBM fit is a stronger baseline than a tuned neural network on
  most structured datasets.
- **Teaching the systems view.** The three levers here — binning, growth order and data
  sampling — are the same levers behind every modern fast GBDT (XGBoost's `hist`,
  CatBoost's quantization, cuML's GPU histograms).

## References

- Ke, Meng, Finley et al., *LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree*, 2017, NeurIPS — GOSS, EFB and leaf-wise growth, the paper this file follows.
- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, 2016, KDD — the
  second-order objective and the sparsity-aware split.
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, 2001 — the
  functional-gradient ancestor.
- Fisher, *On Grouping for Maximum Homogeneity*, 1958, JASA — the exchange argument
  behind sorting categorical levels by their statistic.
- Prokhorenkova, Gusev, Vorobev et al., *CatBoost: Unbiased Boosting with Categorical
  Features*, 2018, NeurIPS — the ordered-target-statistics alternative to plain
  categorical splits.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  2009, ch. 10 — the statistical view of additive trees and shrinkage.
