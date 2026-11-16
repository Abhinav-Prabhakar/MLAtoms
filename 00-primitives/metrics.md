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
