# AdaBoost

`adaboost.py` · Trees & ensembles · Tier 1 · 6 variants

## Intuition

Boosting answers a simple question: what should the next weak learner look at? The
answer is *whatever the current ensemble gets wrong*. AdaBoost turns that into a
reweighting scheme — after each learner, upweight the rows it misclassified, fit the
next learner on the new weights, and combine everything in a weighted vote. No
gradient is computed explicitly; the weights *are* the gradient.

The surprising part is that reweighting is equivalent to gradient descent on the
exponential loss `exp(−yF(x))`. That is why AdaBoost is the ancestor of every
gradient-boosted tree: `gradient_boosting.py` is the same loop with the loss written
down and the tree fitted to its negative gradient.

The six variants are the historical answers to "what does a weak learner return?".
`discrete` returns a hard `±1` label. `real` returns the log-odds of its leaf, which is
a confidence and converges faster. `gentle` returns the weighted regression fit
directly, with no `log` and no `α`, which is the most numerically stable. `samme` and
`samme_r` are the multiclass extensions, and `stagewise` makes the gradient-descent
view explicit by line-searching the step.

## Variants

| Variant | Task | Weak learner output | Step | Weight update |
|:--|:--|:--|:--|:--|
| `discrete` | binary | hard `±1` | `α = ½ln((1−err)/err)` | `∝ exp(−yαh)` |
| `real` | binary | `½ln(p/(1−p))` per leaf | none (`α=1`) | `∝ exp(−yh)` |
| `gentle` | binary | `2p−1` per leaf | none (`α=1`) | `∝ exp(−yh)` |
| `samme` | `K` classes | hard class vote | `ln((1−err)/err) + ln(K−1)` | `∝ exp(α·1[wrong])` |
| `samme_r` | `K` classes | `(K−1)(log p_k − mean log p)` | none | `∝ p_true^{−(K−1)/K}` |
| `stagewise` | binary | `2p−1`, line-searched `γ` | `γ` from Newton | `∝ exp(−yγh)` |

`discrete` is the original Freund–Schapire algorithm and the one in every textbook.
`real` and `gentle` are Friedman, Hastie and Tibshirani's confidence-rated
refinements: the leaf probability is more informative than the leaf label, and using
it removes the need for a separate `α`. `samme` generalises the `α` to `K` classes by
adding `ln(K−1)`; without that term a weak learner at chance level
(`err = 1−1/K`) would get a positive weight. `samme_r` replaces the hard vote with
per-class log-probabilities, which is the multiclass analogue of `real`.

`stagewise` is the odd one out: it is the same exponential loss, but the step is
chosen by a line search over `γ` instead of the closed-form `α`, and `learning_rate`
multiplies it. It is the bridge to `gradient_boosting.py` — replace the exponential
loss with any differentiable loss, fit the tree to the negative gradient, and the loop
is identical.

## Derivation

**Exponential loss and the additive model.** With `y ∈ {−1, +1}` and
`F(x) = Σ_t α_t h_t(x)`, AdaBoost minimises

$$
L(F) = \sum_i \exp\!\big(-y_i F(x_i)\big)
$$

One stage at a time, holding the previous `F` fixed:

$$
(\alpha, h) = \arg\min_{\alpha,h} \sum_i w_i \exp(-\alpha y_i h(x_i)),
\qquad w_i = \exp(-y_i F_{m-1}(x_i))
$$

**Discrete case.** For a hard `h ∈ {−1,+1}`, split the sum by whether `h` is right:

$$
\sum_i w_i e^{-\alpha y_i h_i}
= e^{-\alpha} W_{\text{correct}} + e^{\alpha} W_{\text{wrong}}
= e^{-\alpha}(W - W_{\text{err}}) + e^{\alpha} W_{\text{err}}
$$

with `W = Σw` and `W_err = Σ w·1[h wrong]`. Setting the derivative to zero gives

$$
\alpha^\* = \frac12 \ln\frac{W - W_{\text{err}}}{W_{\text{err}}}
= \frac12\ln\frac{1-\text{err}}{\text{err}}
$$

This is why `err ≥ 0.5` is a stop condition: `α ≤ 0` means the learner is worse than
chance, and flipping its sign would be a different learner. The best hard `h` at each
stage is the one minimising `W_err`, which is exactly what a weighted classification
tree finds — the weights enter the impurity, not the split search.

**The normaliser.** After the update, the total weight is

$$
Z_t = (1-\text{err}_t)e^{-\alpha_t} + \text{err}_t e^{\alpha_t}
= 2\sqrt{\text{err}_t(1-\text{err}_t)}
$$

so the normalised weights are `w_i ← w_i e^{−α y h}/Z_t`, and the exponential loss of
the final model is exactly

$$
\frac1n\sum_i e^{-y_i F(x_i)} = \prod_{t=1}^T Z_t
$$

The demo checks this identity to nine digits (`0.475614 == 0.475614`), which is a much
stronger test than checking that the loss decreased. Since `Z_t ≤ 1`, the training
error obeys

$$
\frac1n\#\{y_i \ne \operatorname{sign}F(x_i)\} \le \prod_t Z_t
= \exp\!\Big(-2\sum_t \gamma_t^2\Big),
\qquad \gamma_t = \tfrac12 - \text{err}_t
$$

so a weak learner with a constant edge `γ` drives the training error to zero
exponentially fast — the margin explanation of why AdaBoost does not overfit as
quickly as its training error suggests.

**Real AdaBoost.** Instead of a hard label, let the leaf return
`h_j = ½ln(W_+^j/W_-^j)` for leaf `j`, where `W_±^j` are the weighted class masses in
that leaf. This is the minimiser of the same objective when `h` is allowed to be any
real number *within a leaf*, and it is the log-odds of the leaf's class probability.
The `α` disappears because the log-odds already carries the confidence: a pure leaf
gets a large step, a 50/50 leaf gets none. Friedman's analysis shows the stagewise
fit converges faster than `discrete` and produces better-calibrated scores, which the
demo reflects in the lower exponential loss (`0.292` vs `0.476` at the same 80 stumps,
with test accuracies `0.927` vs `0.942` — the hard vote is not worse, just less
confident).

**Gentle AdaBoost.** Replace the log-odds with the weighted least-squares fit
`f_j = Σ_{i∈j} w_i y_i / Σ_{i∈j} w_i`, and add it directly. This is a Newton step for
the exponential loss: the second derivative of `e^{−yF}` is `e^{−yF}`, so the local
quadratic has the same weight in numerator and denominator, and the minimiser is the
weighted mean of `y`. No `log`, no division by `err` — hence "gentle": it cannot blow
up when a leaf is nearly pure, and it is the recommended default in Friedman's
comparison. The demo's depth-3 comparison (`0.372` vs `0.632` at stage 6) shows the
other axis: stronger weak learners converge in fewer stages, at the cost of variance.

**SAMME.** For `K` classes, the weak learner returns a hard class `h(x) ∈ {1,…,K}` and
the model is `F_k(x) = Σ α_t 1[h_t(x)=k]`, predicted by `argmax_k F_k`. The stage
objective becomes

$$
\sum_i w_i \exp\!\big(-\alpha\, \mathbb{1}[y_i = h(x_i)]\big)
$$

because the multiclass exponential loss is `exp(−α)` for a correct vote and `1` for
every wrong class. Minimising gives

$$
\alpha^\* = \ln\frac{1-\text{err}}{\text{err}} + \ln(K-1)
$$

The extra `ln(K−1)` is exactly the correction that makes random guessing
(`err = 1−1/K`) give `α = 0`. That is why SAMME's stop condition is
`err ≥ 1−1/K`, not `err ≥ 1/2`.

**SAMME.R.** Use the leaf's class probabilities `p_k(x)` and set

$$
h_k(x) = (K-1)\left(\log p_k(x) - \frac1K\sum_j \log p_j(x)\right)
$$

so the `K` outputs sum to zero and only their differences matter. The weight update is

$$
w_i \leftarrow w_i \, p_{y_i}(x_i)^{-(K-1)/K}
$$

which is the multiclass counterpart of `exp(−α y h)`: a row whose true class has low
probability gets a large weight, and the exponent `(K−1)/K` is the factor that makes
the update consistent with the `K`-class exponential loss. `samme_r` needs far fewer
stages than `samme` for the same accuracy, which the demo shows as `0.983` vs `0.975`
with 40 vs 60 trees.

**Stagewise.** Treat `F` as a vector and take a gradient step in function space:
`F ← F + γh` where `h` is fitted to `−∂L/∂F = y·exp(−yF)` under the current weights.
Choosing `γ` by minimising `Σ w_i exp(−γ y_i h_i)` is a one-dimensional Newton problem:

$$
\gamma \leftarrow \gamma + \frac{\sum_i w_i s_i e^{-\gamma s_i}}{\sum_i w_i s_i^2 e^{-\gamma s_i}},
\qquad s_i = y_i h_i
$$

which is what `_line_search` does. With `γ` fixed to 1 and `h` a hard label, this
reduces to `discrete`; with `h` a weighted mean, to `gentle`. `learning_rate < 1`
shrinks the step, which is the same shrinkage knob as in gradient boosting — the demo's
`lr=0.3` run takes smaller steps (`0.369` on average) and reaches `0.943` accuracy in
150 stages, while `lr=1.0` gets `0.932` in 25.

## Code walkthrough

The weak learner is a weighted tree, and one line in the split search is the whole
difference from an unweighted tree:

```python
    counts = np.zeros((n, k))
    counts[np.arange(n), y] = w
    parent = 1.0 - float(((counts.sum(axis=0) / wsum) ** 2).sum())
```

`counts` holds weighted class mass rather than row counts. The parent impurity is the
gini of the weighted distribution, so a row with weight 0.1 contributes a tenth as much
as a row with weight 1. Everything downstream — the cumulative sums, the child
impurities, the gain — is the same arithmetic as CART, which is why the tree code is
short.

`min_samples_leaf` is the exception, and it is the subtle one:

```python
        wl = lc.sum(axis=1)
        wr = rc.sum(axis=1)
        # min_samples_leaf counts rows, not weights.
        n_l = np.cumsum(np.ones(n))[cut]
        n_r = n - n_l
        ok = (n_l >= min_samples_leaf) & (n_r >= min_samples_leaf)
```

Weight sums shrink as boosting proceeds (they are normalised to 1), so a leaf limit
applied to `wl` would reject every split after a few stages and the ensemble would stop
learning. Row counts are tracked separately and never renormalised. Getting this wrong
is the classic way to write an AdaBoost that looks correct and trains one tree.

Leaves store weighted class frequencies, which is what every variant reads:

```python
def _leaf(y: np.ndarray, w: np.ndarray, k: int) -> dict:
    proba = np.bincount(y, weights=w, minlength=k).astype(float)
    total = proba.sum()
    return {
        "leaf": True,
        "n": y.size,
        "proba": proba / total if total > 0 else np.full(k, 1.0 / k),
    }
```

`np.bincount(..., weights=w)` is the weighted histogram; normalising by `total` makes
`proba` a distribution even when the incoming weights do not sum to 1. `discrete` uses
`proba[:, 1] > 0.5`, `real` uses its log-odds, `gentle` and `stagewise` use `2p−1`, and
both multiclass variants use the whole vector. One leaf representation serves six
algorithms.

The discrete branch is the textbook formula, and its two guards are not optional:

```python
            if self.variant == "discrete":
                pred = (p[:, 1] > 0.5).astype(float)
                err = float((w * (pred != yi)).sum())
                err = float(np.clip(err, 1e-10, 1.0 - 1e-10))
                if err >= 0.5:
                    break
                alpha = 0.5 * np.log((1.0 - err) / err)
                h = 2.0 * pred - 1.0
                F += alpha * h
                self.alphas_.append(float(alpha))
```

`err` is a weighted error, not a mean of booleans. The clip prevents `log(0)` on
perfectly separable data, and the `err >= 0.5` break stops the ensemble instead of
adding a negatively weighted learner. Break happens *before* the tree is appended, so
`estimators_`, `alphas_` and `errors_` stay aligned.

`real` replaces the hard label with the leaf log-odds:

```python
            elif self.variant == "real":
                p1 = np.clip(p[:, 1], 1e-10, 1.0 - 1e-10)
                h = 0.5 * np.log(p1 / (1.0 - p1))
                F += h
                self.alphas_.append(1.0)
```

`alpha` is still recorded as 1 so the prediction loop is uniform, but the step size is
inside `h`. The clip matters here too: a pure leaf would otherwise produce an infinite
confidence and a saturated `F` after one stage.

Weights are recomputed from `F` instead of being multiplied in place:

```python
            if self.variant not in ("samme", "samme_r"):
                # exp(-yF) underflows/overflows on separable data; the clip keeps
                # the weights finite without changing the argmin.
                w = np.exp(-y_signed * np.clip(F, -50.0, 50.0))
                w /= w.sum()
```

This is algebraically identical to `w *= exp(−α y h)` and numerically better: the
product of many small factors would underflow, and recomputing from the margin keeps
the relative weights accurate. The clip is a guard for separable data, where `F` grows
without bound; the exponential loss is flat enough there that the clipped weights pick
the same splits.

Multiclass hard votes add one class score at a time:

```python
            elif self.variant == "samme":
                pred = np.argmax(p, axis=1)
                err = float((w * (pred != yi)).sum())
                err = float(np.clip(err, 1e-10, 1.0 - 1e-10))
                alpha = np.log((1.0 - err) / err) + np.log(k - 1.0)
                if err >= 1.0 - 1.0 / k:
                    break
                F[np.arange(n), pred] += alpha
```

`F[np.arange(n), pred] += alpha` is a scatter-add: row `i`'s score for its predicted
class increases. `argmax` of those scores is the vote, so the update is the multiclass
analogue of `sign`. The `ln(K−1)` term is what makes `α` zero at chance level; omit it
and SAMME keeps boosting a learner that knows nothing.

`samme_r` uses the log-probabilities directly, centred across classes:

```python
            else:                                     # samme_r
                pk = np.clip(p, 1e-10, 1.0)
                logp = np.log(pk)
                h = (k - 1.0) * (logp - logp.mean(axis=1, keepdims=True))
                F += h
                self.alphas_.append(1.0)
                err = float((w * (np.argmax(p, axis=1) != yi)).sum())
                w = w * pk[np.arange(n), yi] ** (-(k - 1.0) / k)
                w /= w.sum()
```

`logp.mean(axis=1)` is the per-row mean over classes, so `h` sums to zero across
classes and only the *relative* log-probabilities enter. The weight update indexes the
true class's probability with `pk[np.arange(n), yi]` — a fixed `k=…` index would
upweight every row by the same class's confidence and destroy the reweighting.

`stagewise` is the only variant whose step is computed numerically:

```python
    @staticmethod
    def _line_search(y_signed: np.ndarray, h: np.ndarray, w: np.ndarray) -> float:
        """Minimise sum w exp(-gamma * y * h) over gamma >= 0 by Newton steps."""
        s = y_signed * h
        gamma = 1.0
        for _ in range(20):
            e = np.exp(-gamma * s)
            f1 = -float((w * s * e).sum())
            f2 = float((w * s * s * e).sum())
            if f2 <= EPS:
                return 0.0
            step = f1 / f2
            gamma = float(np.clip(gamma - step, 0.0, 5.0))
            if abs(step) < 1e-9:
                break
        return gamma
```

`s = y·h` is the *signed* agreement, so a leaf that agrees with the label contributes a
positive exponent. `f2` is the curvature and is always positive unless every `s` is
zero; the `EPS` guard returns a zero step rather than dividing by it. The clip to
`[0, 5]` keeps the line search in a sensible range — a negative `γ` would fit the
learner backwards, which is `discrete`'s failure mode generalised.

Prediction replays the stored `(tree, alpha)` pairs, and the discrete branch rebuilds
the hard label from the stored probability:

```python
            for tree, alpha in zip(self.estimators_, self.alphas_, strict=True):
                p = _tree_proba(tree, X, k)
                if self.variant == "discrete":
                    F += alpha * (2.0 * (p[:, 1] > 0.5) - 1.0)
```

Storing probabilities rather than hard labels in the tree is what lets one tree serve
all six variants; the *variant* decides how to read them at prediction time. `strict=True`
on the `zip` turns an accidental mismatch between trees and alphas into an error instead
of silently dropping stages.

## Complexity

| Quantity | Cost |
|:--|:--|
| One weighted tree (stump) | `O(d n log n)` for the sort + `O(d n)` prefix sums |
| One weighted tree, depth `h` | `O(d n h log n)` |
| `T` stages | `T ×` the tree cost |
| Prediction, one row | `O(T · depth)` |
| Memory | `O(T · nodes)` |

The weights add no asymptotic cost: they replace counts in the cumulative sums. The
sort dominates, and it is repeated at every node. AdaBoost is sequential like gradient
boosting — the next tree depends on the current weights — so the only parallelism is
inside a tree's split search. Training a few hundred stumps on 400 rows is milliseconds,
which is why AdaBoost is still a reasonable first baseline on small tabular problems.

## Failure modes

1. **Weak learners at chance.** `discrete` stops when `err ≥ 0.5`. A stump on XOR is
   exactly at chance, so boosting stumps cannot learn XOR — the demo uses a circle for
   this reason. The fix is a deeper weak learner or a feature that carries signal.
2. **Label noise.** AdaBoost upweights misclassified rows, and a mislabelled row is
   permanently misclassified, so its weight grows without bound. After enough stages
   the ensemble fits the noise. The demo's 5% noise is survivable; 20% is not. Regularise
   with a depth cap and early stopping.
3. **Unbounded margins on separable data.** `err → 0` makes `α → ∞` and `F` grows
   without limit. The clip in the weight update keeps the code finite, but the
   probabilities saturate. Use `gentle` or `real` with shrinkage, or stop on a
   validation curve.
4. **`α` sign after clipping.** Clipping `err` to `1e-10` on a perfect learner gives
   `α ≈ 11.5`, which is finite but large; the next stage sees a nearly uniform weight
   and stops. Do not read the number of stages as convergence — read `errors_`.
5. **`min_samples_leaf` on weights.** As noted above, a row-count limit must not be
   applied to normalised weights, or every deep split becomes invalid. This is the most
   common implementation bug in hand-written AdaBoost.
6. **Multiclass base rate.** `samme` with `err = 1−1/K` gives `α = 0`, but the
   implementation must also *stop*: continuing adds zero-weight trees and wastes
   stages. The break tests the same threshold the `α` formula uses.
7. **`samme_r` with zero probabilities.** `log(0)` is `−inf`; the clip at `1e-10`
   caps the confidence at about `23`, which is large enough to dominate the ensemble.
   Calibrate the weak learner (it is a weighted tree, so its leaves are smoothed by
   `min_samples_leaf`) or clip harder.
8. **Stagewise step size.** The line search assumes `h` is fixed while `γ` varies. With
   a deep tree, `h` is already a fitted function and a large `γ` moves far from the
   linearisation point; keep `learning_rate` small (0.1–0.3) and use more stages.
9. **Probabilities are not calibrated.** `sigmoid(2F)` is a convenient monotone map,
   not a fitted calibration. The demo checks that probabilities are ordered and sum to
   one, not that they match frequencies; use `isotonic` or Platt scaling if the
   probability itself is the product.

## Applications

- **Face detection.** Viola–Jones is AdaBoost over Haar-like features with a cascade;
  the reweighting is what selects the few hundred features that matter out of tens of
  thousands.
- **Small tabular classification.** A few hundred stumps give a strong, interpretable
  baseline (each stump is a threshold on one feature) with no hyperparameter tuning
  beyond the stage count.
- **Feature selection.** The stump chosen at each stage is the most useful single
  threshold given the current weights; the sequence of chosen features is a cheap
  importance ranking, and it is the mechanism behind Viola–Jones' cascade.
- **Class-imbalanced detection.** The weight update naturally focuses on hard
  positives, which is useful for rare-event detection — but see failure mode 2, since
  label noise and rare positives look the same to the algorithm.
- **Teaching boosting.** AdaBoost is the cleanest derivation of the exponential-loss
  view; `gradient_boosting.py` is the same loop with a pluggable loss, and reading the
  two side by side is the fastest way to understand modern GBDTs.
- **Multiclass baselines with weak learners.** SAMME works with any classifier that
  accepts sample weights, so it is a wrapper for turning a binary or multiclass stump
  into a competitive ensemble.

## References

- Freund & Schapire, *A Decision-Theoretic Generalization of On-Line Learning and an
  Application to Boosting*, 1997, JCSS — the original AdaBoost and the `α` formula.
- Schapire & Singer, *Improved Boosting Algorithms Using Confidence-rated Predictions*,
  1999, Machine Learning — Real AdaBoost and the leaf log-odds.
- Friedman, Hastie & Tibshirani, *Additive Logistic Regression: A Statistical View of
  Boosting*, 2000, Annals of Statistics — the exponential-loss derivation, Gentle
  AdaBoost and the `Z_t` analysis.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., 2009,
  ch. 10 — the margin bound and the comparison of discrete/real/gentle.
- Zhu, Zou, Rosset & Hastie, *Multi-class AdaBoost*, 2009, Statistics and Its Interface
  — SAMME and SAMME.R, including the `ln(K−1)` correction.
- Viola & Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*,
  2001, CVPR — the application that made AdaBoost famous.
