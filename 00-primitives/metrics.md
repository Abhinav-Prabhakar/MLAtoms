# Metrics

`metrics.py` · Primitives · Tier 1 · 18 variants

## Intuition

Every metric is a loss you did not train on, and choosing one is choosing what the
model will be good at. Accuracy on a 99:1 dataset rewards predicting the majority
class; AUC rewards ranking and ignores calibration; NDCG rewards putting the
right item near the top and is indifferent to items at position 400.

So the question is never "which metric is best" but "which mistake am I willing
to make". State that, then pick.

## Variants

| Metric | Scale | Answers | Blind to |
|:--|:--|:--|:--|
| `accuracy` | [0,1] | How often right? | class imbalance |
| `balanced_accuracy` | [0,1] | How often right, per class? | prevalence, confidence |
| `precision_recall_f1` | [0,1] | Trade-off at one threshold | ranking, calibration |
| `micro/macro/weighted` | [0,1] | How to pool classes | — |
| `confusion` | counts | *Which* mistakes | everything else |
| `roc_auc` | [0,1] | Ranking of positives vs negatives | calibration, prevalence |
| `pr_auc` | [0,1] | Ranking under imbalance | calibration |
| `average_precision` | [0,1] | Step-wise PR area | calibration |
| `log_loss` | [0,∞) | Probability quality | decisions |
| `brier` | [0,1] | Probability MSE | direction of error |
| `mcc` | [−1,1] | Correlation of predictions and truth | confidence |
| `cohen_kappa` | [−1,1] | Agreement beyond chance | confidence |
| `ndcg` | [0,1] | Graded ranking quality | items beyond k |
| `map` | [0,1] | Precision across the whole list | graded relevance |
| `mrr` | [0,1] | Position of the first hit | everything after it |
| `hit_rate` | [0,1] | Any hit in top k | position within k |
| `silhouette` | [−1,1] | Cluster separation | ground truth |
| `adjusted_rand` | ~[−0.5,1] | Agreement with true labels | cluster count |

## Derivation

**ROC-AUC is a rank statistic.** With `n₊` positives and `n₋` negatives,

$$
\text{AUC} = \frac{1}{n_+ n_-}\sum_{i \in +}\sum_{j \in -} \left[\mathbb{1}(s_i > s_j) + \tfrac12\mathbb{1}(s_i = s_j)\right]
= \frac{\sum_{i \in +} r_i - \tfrac12 n_+(n_+ + 1)}{n_+ n_-}
$$

where `r_i` is the average rank of positive `i`. The rank form is `O(n log n)`
instead of `O(n₊n₋)`, and it is exact with ties — which matters, because
quantised scores make ties common.

**AP vs trapezoidal PR-AUC.** Precision is undefined between two operating
points; the trapezoid interpolates through regions the classifier can never
occupy. AP sums precision at the ranks where a positive actually occurs, which is
the quantity you can estimate without inventing points. That is why AP ≤ PR-AUC
almost always, and why AP is preferred under imbalance.

**MCC is a phi coefficient.** With `TP, TN, FP, FN`:

$$
\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

The numerator is the determinant of the 2×2 confusion matrix: zero whenever the
prediction is independent of the truth, regardless of prevalence. Accuracy cannot
make that statement; MCC can.

**Silhouette.** For point `i`:

$$
a(i) = \text{mean distance to own cluster},\quad
b(i) = \min_{c \ne \text{own}} \text{mean distance to cluster } c,
\quad s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

`a` measures compactness, `b` separability. A singleton cluster has `a = 0` and is
defined as `s = 0` here, not `1` — otherwise every outlier would look perfect.

## Code walkthrough

The file has four blocks — classification, ranking, clustering, and a demo that
works small cases by hand. The classification metrics all start from one matrix:

```python
    c = np.zeros((k, k), dtype=int)
    np.add.at(c, (y_true, y_pred), 1)
    return c
```

`np.add.at` rather than `c[y_true, y_pred] += 1`. The fancy-indexed `+=` is not a
loop: for repeated `(i, j)` pairs NumPy buffers and writes only the last increment,
so a cell with 500 examples records 1. This is the single most common bug in a
hand-rolled confusion matrix, and the demo's 990/10 imbalance case would still
"pass" accuracy while producing a wrong MCC.

`balanced_accuracy` and the macro average both have to decide what to do with a
class that never appears:

```python
    recall = np.diag(c) / np.maximum(c.sum(axis=1), 1)
    present = c.sum(axis=1) > 0
    return float(recall[present].mean())
```

The `present` mask is the difference between a metric and a `nan`. A class with zero
support contributes `0/1 = 0` to the mean, which silently halves the score if you
compute `k` from a label set larger than the data. Filtering first is the
convention sklearn uses; keeping the `np.maximum(..., 1)` guards the division for
the classes that *are* present.

`_rankdata` is five lines and everything downstream depends on it:

```python
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    return avg[inv]
```

`np.unique` returns sorted unique values, so the tied block starting at cumulative
position `cum - counts` occupies ranks `cum - counts + 1 … cum`; its average rank is
the midpoint `cum − (counts−1)/2`. The returned `avg[inv]` scatters those averages
back to the original order. An `argsort`-based ranker without tie handling gives
each tied element a *different* rank, which biases AUC upward whenever scores are
quantised — exactly the regime where AUC is most often reported.

AUC then is Mann–Whitney in closed form:

```python
    r = _rankdata(scores)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
```

`Σr₊` counts every pair where a positive outranks something; subtracting
`n₊(n₊+1)/2` removes the positive-vs-positive pairs, leaving the positive-vs-negative
count. Dividing by `n₊n₋` turns the count into a probability. Averaged ranks mean
ties contribute exactly `1/2` each — the definition in the docstring — with no
special case.

`average_precision` and `pr_auc` are deliberately different functions, and the
difference is the sorting:

```python
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / max(y.sum(), 1))
```

AP only samples precision *at ranks where a positive occurs* (`(precision * y).sum()`),
and divides by the number of positives. `pr_auc` instead walks every unique threshold
and trapezoids the resulting curve, which interpolates through points the classifier
can never reach — hence the docstring's "the honest AP". `kind="mergesort"` makes the
tie order stable, so two calls on equal scores are reproducible; `quicksort`
(introsort) is not stable and AP can wobble by a few points on heavily tied data.

`ndcg` computes the ideal DCG from the *same* discount vector:

```python
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(((2.0**ideal - 1.0) * discounts).sum())
```

The discount starts at `log2(2) = 1`, so the first position is undiscounted. Reusing
`discounts` for the ideal ordering is what makes `ndcg(rel, rel) == 1.0` exactly; a
recomputed vector with a different length (`k` vs `rel.size`) is off by a scale
factor whenever `k` truncates.

`map_at_k` contains a subtlety worth flagging rather than hiding:

```python
        if k is not None:
            order = np.argsort(-s, kind="mergesort")[:k]
            ap = average_precision(r[order].astype(bool), s[order])
```

It truncates the list and *then* calls AP, so the normaliser is the number of
positives in the top `k`, not all positives. That is AP@k, not "AP computed on the
top-k prefix"; the two differ whenever there are relevant items below `k`. The
demo's single-query case is small enough that both agree, which is why the doc has
to state the convention explicitly.

`silhouette` skips singletons instead of scoring them:

```python
        if own.sum() == 0:  # singleton cluster: silhouette is 0 by convention
            continue
```

`s` was initialised to zeros, so the `continue` leaves the point at `0`. Setting it
to `1` — the tempting choice, since a point alone in its cluster has `a = 0` and
`(b−0)/max(0,b) = 1` — rewards every outlier and makes the score improve as
clustering degrades. The `own[i] = False` line is the second trap: it excludes the
point from its own `a`, otherwise `a` is `(n_c−1)/n_c` too small and every score is
inflated.

`adjusted_rand` is the combinatorial form of a contingency table:

```python
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_comb = comb(c).sum()
    a = comb(c.sum(axis=1)).sum()
    b = comb(c.sum(axis=0)).sum()
    expected = a * b / comb(len(y_true))
    max_index = 0.5 * (a + b)
```

`comb(n)` counts agreeing *pairs*, not points, which is why the correction is exact
under a hypergeometric null rather than approximate. `max_index = (a+b)/2` is the
largest achievable pair count given the two marginals, so the denominator is
`max − expected`, not `1 − expected`. Using `1` gives a score that is close to ARI on
balanced problems and arbitrarily wrong on unbalanced ones.

## Complexity

`roc_auc`, `average_precision`, `mcc`, `kappa`: `O(n log n)` (sort or rank).
`confusion_matrix`: `O(n)`. `silhouette`: `O(n²)` time and memory in this
implementation — for `n > 10⁴` subsample or use the `O(nd·k)` approximation.
`pr_auc`: `O(n·|thresholds|)` in the naive form; keep the unique thresholds.

## Failure modes

1. **Accuracy on imbalance.** 99% accuracy with 1% positives is the do-nothing
   baseline. Report balanced accuracy or MCC alongside it, always.
2. **AUC with no positives in a slice.** `roc_auc` raises here, which is the right
   behaviour; the common bug is silently returning 0.5 and hiding a broken slice.
3. **NDCG without specifying the gain.** `2^rel − 1` (used here) is steep and
   rewards the top heavily; linear gain is gentler. Two teams comparing NDCG with
   different gains are comparing different metrics.
4. **MAP with graded relevance.** AP is defined for binary relevance; binarising
   `rel ≥ 2` changes the number, so state the threshold.
5. **Silhouette for non-convex clusters.** It is a distance-based score: DBSCAN's
   crescents can be correct and score poorly. Use it to compare solutions of the
   *same* family, never across families.

## Applications

- Model selection: `log_loss` for probability models, `average_precision` for rare
  positives, `ndcg@10` for retrieval, `silhouette` only within one clustering family.
- Reporting: always a confusion matrix plus one scalar; the scalar alone hides
  which class is failing.
- Guardrails: MCC is the safest single number for a binary dashboard because it
  cannot be gamed by prevalence.

## References

- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, ch. 8 (evaluation). 2008.
- Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot*, 2015. PLoS ONE.
- Chicco & Jurman, *The advantages of the Matthews correlation coefficient (MCC)*, 2020. PMC.
- Rousseeuw, *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*, 1987.
