# Calibration

`calibration.py` · Evaluation · Tier 1 · 6 variants

## Intuition

A probability is a promise about frequency. If a model says `0.9` on a thousand
different cases, roughly nine hundred of them should be positive. Nothing in the
training objective enforces that: a network trained to convergence on separable data
drives its logits to infinity, and boosted trees average log-odds rather than
probabilities, so both end up confidently wrong in opposite directions.

Calibration is a second, tiny model fitted *after* the first one, whose only job is to
repair the mapping from score to probability:

```text
base model  ──►  score s  ──►  calibrator g  ──►  p = g(s)
                                    ▲
                          fitted on held-out (s, y)
```

Every variant here is a different `g`, and all of them are fitted by minimising log-loss
on predictions the base model has never seen. The demo starts from a model that is
exactly twice as sharp as the truth (`logit(p) = 2 × logit(p_true)`), which gives a clean
target: the correct calibrator has to learn a slope of `0.5`. Platt lands on `0.541`,
temperature on `2.014` when the truth is `2.0`, and both cut expected calibration error
from `0.090` to `0.031`.

Two things calibration does *not* do. It does not change accuracy — `Platt`, `isotonic`
and `temperature` are strictly increasing, so the ranking of scores is untouched, and
`dirichlet` moves it only slightly (`0.800 → 0.805` in the demo). And it does not
generalise across distributions: a calibrator fitted on last quarter's traffic encodes
last quarter's base rates.

## Variants

| Variant | Map | Parameters | Fixes |
|:--|:--|:--|:--|
| `platt` | `σ(a·logit(p) + b)` | 2 | a global over/under-confidence |
| `isotonic` | monotone step function | `O(n)` breakpoints | any monotone distortion |
| `temperature` | `softmax(z / T)` | 1 | one shared over/under-confidence |
| `beta` | `σ(a·log p + b·log(1−p) + c)` | 3 | asymmetric distortion |
| `dirichlet` | `softmax(W log p + b)` | `k² + k` | per-class distortion |
| `reliability_diagram` | none — diagnostic | 0 | measures, does not fix |

`platt` and `temperature` are almost the same model. On binary probabilities,
`σ(logit(p)/T)` is Platt with `b = 0`, so temperature is the one-parameter version and
Platt the two-parameter version; on `k` classes only temperature generalises
(`softmax(z/T)`), because a single scalar cannot rescale a matrix.

`isotonic` is the non-parametric extreme: it fits *any* non-decreasing function, so it
cannot be wrong about a monotone distortion — and it will happily fit noise. In the demo
it uses `1000` breakpoints on `1000` training rows, and it still wins on test ECE
(`0.014` against Platt's `0.031`) because the true distortion is monotone and the sample
is large. Halve the calibration set and that ranking flips; isotonic is the variant that
needs the most data per unit of flexibility.

`beta` generalises Platt by allowing the two tails to be corrected differently: the
features are `log p` and `log(1−p)`, so it can fix a model that is overconfident near 0
and underconfident near 1. Its fitted coefficients in the demo (`0.615`, `−0.479`,
intercept `0.200`) are not equal to the "true" `(0.5, −0.5, 0)`, because over the observed
range of `p` the three parameters are nearly collinear — but the *function* they define is
within `0.018` of the true probability. Fit quality is a property of the map, not of the
coefficients.

`dirichlet` is the multiclass version: a full `k×k` matrix applied to the log-probabilities.
It contains temperature as `W = I/T`, and it can express per-class temperature and
per-class bias. The demo's 3-class fit cuts ECE from `0.095` to `0.029` against
temperature's `0.036`, at the cost of nine parameters and a fixed-point optimiser instead
of a one-dimensional search.

`reliability_diagram` is the odd one out: it fits nothing and transforms nothing. It bins
the predictions by confidence and reports the gap between confidence and accuracy in each
bin — the diagnostic that tells you whether any of the others is needed.

## Derivation

**What calibration means.** A model is *perfectly calibrated* if

$$
P\bigl(Y = 1 \mid \hat p(X) = p\bigr) = p \quad \text{for all } p \in [0,1]
$$

This is a statement about conditional frequencies, so it can only be checked by grouping:
bin the predictions, compare the mean prediction in each bin with the observed positive
rate. That is the reliability diagram, and its weighted average gap is the expected
calibration error:

$$
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{n}\,\bigl|\mathrm{acc}(b) - \mathrm{conf}(b)\bigr|,
\qquad
\mathrm{MCE} = \max_b \bigl|\mathrm{acc}(b) - \mathrm{conf}(b)\bigr|
$$

`ECE` is a weighted mean of bin gaps and `MCE` the worst bin. Neither is a proper scoring
rule: they depend on the binning, and they can be reduced by a degenerate predictor that
puts everything in one bin. They are diagnostics, not objectives — which is why every
calibrator here is fitted by minimising log-loss instead.

**Why log-loss.** A scoring rule `S(p, y)` is *proper* if its expectation is minimised by
the true probability. Log-loss is:

$$
\mathbb{E}\bigl[-\log \hat p\bigr] = H(Y) + \mathrm{KL}\bigl(P(Y\mid X)\,\|\,\hat p\bigr)
$$

The entropy term does not depend on the model and the KL is minimised (at zero) exactly
when `p̂ = P(Y|X)`. So minimising log-loss is minimising the KL to the true conditional
distribution — the strongest possible statement that "the calibrated probabilities are
right", and it holds for every bin simultaneously rather than on average over bins.

**Platt scaling.** Fit `p = σ(A·f + B)` where `f` is the score (here `f = logit(p)`) by
minimising `−Σ log p_i`. The gradient conditions are the logistic regression normal
equations:

$$
\frac{\partial}{\partial A} = -\sum_i f_i\,(t_i - p_i) = 0,
\qquad
\frac{\partial}{\partial B} = -\sum_i (t_i - p_i) = 0
$$

with `t_i ∈ {0,1}`. Platt's contribution is the *target smoothing*: replacing `t` by

$$
t_+ = \frac{N_+ + 1}{N_+ + 2}, \qquad t_- = \frac{1}{N_- + 2}
$$

Without it, a separable problem drives `|A| → ∞` (the likelihood keeps increasing as the
sigmoid steepens) and the calibrator becomes a step function. The smoothed targets are the
posterior means of `t` under a `Beta(1,1)`-like prior on the class frequencies, which caps
the slope at a finite value. The demo's data is not separable, so the correction is small,
but it is the difference between a usable fit and a divergence on real SVM outputs.

**Isotonic regression.** Find the non-decreasing `g` minimising

$$
\sum_i \bigl(y_i - g(p_i)\bigr)^2
\quad \text{subject to} \quad g(p_{(1)}) \le g(p_{(2)}) \le \cdots
$$

after sorting by `p`. The solution is a step function whose blocks are the maximal
segments over which the cumulative mean is monotone. Pool Adjacent Violators computes it in
one pass: append each point as its own block, and whenever the previous block's value
exceeds the current one, merge them and use the weighted mean. Each merge reduces the
number of blocks by one, so the whole algorithm is `O(n)` after the sort.

The fit is the maximum-likelihood estimate under the monotonicity constraint, and it is
consistent (Grenander's theorem) but not smooth: with `n` distinct scores it has up to `n`
breakpoints, which is exactly what the demo sees. That is also why it overfits small
calibration sets — each block is fitted to the rows inside it, and a block with three rows
has a standard error of about `0.3`.

**Temperature scaling.** For logits `z` with `p = softmax(z)`, fit the single scalar `T`:

$$
p_T = \operatorname{softmax}\!\left(\frac{z}{T}\right),
\qquad
T^\star = \arg\min_T \; -\sum_i \log p_T(y_i \mid x_i)
$$

`T > 1` softens (fixes overconfidence), `T < 1` sharpens (fixes underconfidence), and
`T = 1` is the identity. Dividing logits by a constant is a monotone operation per row, so
the argmax — and therefore accuracy — is untouched by construction. Temperature is the
standard tool for neural networks because one parameter cannot overfit a validation set,
which is exactly the regime where the calibration set is small.

**Beta calibration.** With `s = log p` and `t = log(1−p)`, fit

$$
\operatorname{logit}(\hat p) = a\,s + b\,t + c
$$

Kull et al. motivate it from the Beta family: if the score density is `Beta(α, β)` and the
class-conditional densities are also Beta, the posterior log-odds is linear in `s, t, c`.
Two consequences are worth knowing: it contains Platt scaling (`a = −b`), and it maps the
boundary values correctly — `p → 0` gives `−∞` whatever the coefficients, whereas Platt on
`logit(p)` needs a clipped score to avoid infinities.

**Dirichlet calibration.** Let `z = log p` and fit

$$
\hat p = \operatorname{softmax}\bigl(W z + b\bigr), \qquad W \in \mathbb{R}^{k\times k}
$$

This is multinomial logistic regression with the log-probabilities as features. It contains
temperature (`W = I/T`, `b = 0`) and per-class Platt scaling (diagonal `W`). The softmax
parameterisation has a redundancy — adding a constant to every row of `W` and to `b`
leaves `softmax` unchanged — so the objective needs a tie-break; here an L2 penalty toward
`(I, 0)` supplies one and also keeps `W` from collapsing. Kull et al.'s "off-diagonal
regularisation" penalises only the off-diagonal entries, on the argument that a good
calibrator mostly rescales each class and only mildly mixes them; the demo uses the simpler
shrink-toward-identity version.

**Why these are all "post-hoc".** Each variant is fitted to the pair `(score, label)` and
never touches the base model. That is what makes them cheap and composable — and what makes
the held-out requirement absolute: the calibrator is fitting the base model's errors, and
in-sample errors are not the ones it will make in production.

## Code walkthrough

The diagnostic is a binned comparison, and its whole content is three `bincount` calls:

```python
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
```

`np.digitize` returns `0` for values below the first interior edge and `n_bins` above the
last, so the `clip` folds the overflow into the end bins. `bincount` with `weights` sums
the confidences and the correctness indicators per bin in one pass each; dividing gives the
bin's mean confidence and mean accuracy. `keep = counts > 0` then drops empty bins, which
matters because an empty bin has `0/0` and would poison the ECE with a `nan`.

The ECE is the weighted mean gap, and the weights are the reason it is computable from
bins:

```python
    gaps = np.abs(mean_acc - mean_conf)
    ece = float(np.sum(counts[keep] / n * gaps))
    mce = float(gaps.max()) if gaps.size else 0.0
```

`counts[keep] / n` is the fraction of the sample in each occupied bin, so `ECE` is the
sample-weighted average of `|acc − conf|` — the quantity a reliability plot shows as the
area between the diagonal and the curve.

Platt scaling is logistic regression on the logit, with the smoothed targets:

```python
        n_pos = float(y.sum())
        n_neg = float(y.size - n_pos)
        target = np.where(y > 0.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        F = np.column_stack([_logit(p), np.ones(p.size)])
        w = _fit_logistic(F, target, self.l2)
```

`np.where` builds the two smoothed targets in one expression — positives get
`(N₊+1)/(N₊+2)`, negatives `1/(N₋+2)`. The design matrix is `[logit(p), 1]`, so the fitted
`w[0]` is the slope `A` and `w[1]` the intercept `B`. Fitting on `target` rather than `y` is
the entire Platt correction; everything else is ordinary logistic regression.

That regression is Newton's method, five lines of it:

```python
        p = _sigmoid(F @ w)
        grad = F.T @ (p - y) + l2 * w
        weight = np.maximum(p * (1.0 - p), EPS)
        hess = (F * weight[:, None]).T @ F + l2 * np.eye(F.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
```

The Hessian of the logistic loss is `Fᵀ diag(p(1−p)) F`, which is exactly
`(F * weight[:, None]).T @ F` — the `weight` column broadcast into `F` before the product.
`np.maximum` guards the saturated case where `p` is exactly 0 or 1 and the Hessian becomes
singular; with `l2 > 0` the ridge term makes it invertible anyway. Convergence is quadratic
and takes 5–8 iterations on these problems.

Isotonic regression is the pooling loop:

```python
    blocks: list[list[float]] = []       # [fitted value, weight, first unique index]
    for j in range(uniq.size):
        blocks.append([float(means[j]), float(counts[j]), j])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, _ = blocks.pop()
            v1, w1, start = blocks.pop()
            total = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / total, total, start])
```

Each block carries its own start index so the final expansion knows which scores it covers.
The `while` is what makes it PAVA rather than a single merge: pooling two blocks can create
a new violation with the block *before* them, and the loop keeps merging leftward until
monotonicity is restored. Keeping the *earlier* start (`start` from the popped `v1` block)
is what extends the merged block over both ranges. `> ` rather than `>=` keeps equal
neighbours as separate blocks, which is harmless and keeps the output stable.

Fitted values are then scattered back to the unique scores, and the transform is a lookup:

```python
            idx = np.searchsorted(self.x_, p, side="right") - 1
            out = self.y_[np.clip(idx, 0, self.y_.size - 1)]
```

`side="right" - 1` gives the index of the largest fitted score `≤ p`, which is the
piecewise-constant reading of the isotonic fit: a value between two breakpoints takes the
lower block's probability. Clipping handles `p` below the smallest or above the largest
observed score. (scikit-learn interpolates linearly between block endpoints instead; the
step is what the isotonic solution actually is, and the difference is confined to the gaps
between blocks.)

Temperature scaling is a one-dimensional search, and the search is over `log T`:

```python
        grid = np.linspace(-3.0, 3.0, 601)
        vals = np.array([loss(np.array([g])) for g in grid])
        best = float(grid[int(np.argmin(vals))])
        self.temperature_ = float(np.exp(best))
```

`log T` rather than `T` makes the grid symmetric around the identity: `T = 0.05` and
`T = 20` are equally far from `1`, which they are not on the linear scale. A 601-point
grid over `[e⁻³, e³]` resolves `T` to about 1%, which is far finer than its standard error.
The loss is the mean negative log-probability of the true class, computed in log-softmax
form so no probability ever underflows.

Dirichlet calibration is the one variant that needs a real optimiser, so it uses gradient
descent with a backtracking line search:

```python
        def grad(theta: np.ndarray) -> np.ndarray:
            W, b = unpack(theta)
            s = _softmax(z @ W.T + b)
            g = (s - target) / y.size
            gW = g.T @ z + 2.0 * self.l2 * (W - np.eye(k))
            gb = g.sum(axis=0) + 2.0 * self.l2 * b
            return np.concatenate([gW.ravel(), gb])
```

`(s - target)` is the softmax regression gradient — the same `p − y` as the binary case,
one row per sample. `g.T @ z` contracts over samples to give a `k×k` matrix, and the
regulariser pulls `W` toward the identity rather than toward zero, which is the prior "the
calibrator is close to changing nothing". Flattening `W` and `b` into one vector lets a
single generic minimiser handle it.

The transform for the multiclass variants keeps probabilities normalised end to end:

```python
            q = np.clip(p, EPS, 1.0)
            q = q / q.sum(axis=1, keepdims=True)
            out = _softmax(np.log(q) @ self.matrix_.T + self.bias_)
```

Renormalising before the log is not cosmetic: if a caller passes probabilities that sum to
`0.98` (a common artefact of ensembling), the log-probabilities are shifted, `softmax`
absorbs the shift only if all classes share it — and they do not, because the shortfall is
distributed unevenly. Normalising first makes the result independent of the input's
rounding.

## Complexity

| Variant | Fit | Transform | Memory |
|:--|:--|:--|:--|
| `platt` | Newton, `O(n·d²)` for `d = 2`, ~8 iterations | `O(n)` | `O(1)` params |
| `beta` | same, `d = 3` | `O(n)` | `O(1)` |
| `isotonic` | `O(n log n)` sort + `O(n)` pooling | `O(n log n)` binary search, or `O(n)` on sorted input | `O(n)` breakpoints |
| `temperature` | 601 × `O(n k)` | `O(n k)` | `O(1)` |
| `dirichlet` | ~400 × `O(n k²)` | `O(n k²)` | `O(k²)` |
| `reliability_diagram` | `O(n + B)` | identity | `O(B)` |

All of these are negligible next to one forward pass of the base model. `isotonic` is the
memory outlier because it stores the entire step function — up to one breakpoint per row —
which is also its overfitting risk. `dirichlet` is the compute outlier at `O(k²)` per
sample per iteration, which matters only for many-class problems (language models with
`k = 50k` do not use it; they use temperature or top-k variants).

## Failure modes

1. **Fitting on the training set.** The calibrator then learns the base model's *training*
   errors — usually near zero — and concludes the model needs no correction. Fit on a
   held-out split or on out-of-fold predictions.
2. **Distribution shift.** Calibration is a property of a dataset, not of a model. A
   calibrator fitted on a balanced benchmark is wrong on imbalanced production traffic
   because the base rate changed. Re-fit whenever the deployment distribution moves.
3. **Isotonic overfitting.** The demo's fit has `1000` breakpoints on `1000` rows; with
   `200` rows the same procedure memorises the calibration set and can *increase* test
   ECE. Use isotonic only with thousands of points, or smooth/pre-bin first.
4. **Temperature cannot fix shape.** One scalar fixes a uniform over- or under-confidence.
   If the model is overconfident in the middle and underconfident in the tails, temperature
   makes one end worse; use `beta` or `dirichlet`.
5. **`dirichlet` changes predictions.** The full matrix is not monotone per class, so the
   argmax can move — the demo's accuracy shifts `0.800 → 0.805`. Penalise off-diagonals
   toward zero if you need the ranking preserved.
6. **ECE is not a loss.** It depends on the number and placement of bins, it ignores the
   ordering within a bin, and it can be gamed by a constant predictor. Report log-loss or
   Brier score alongside it; optimise those, not ECE.
7. **Prior shift from resampling.** Fitting a calibrator on a class-balanced subsample
   changes the base rate, so the calibrated probabilities are calibrated *for that
   subsample*. Correct for the known prior ratio afterwards, or fit on a representative
   sample.
8. **Softmax identifiability.** In `dirichlet`, adding a constant to every row of `W` and
   to `b` leaves the output unchanged, so without regularisation the parameters are
   arbitrary and the optimiser wanders. The L2 penalty breaks the tie; do not remove it.
9. **Clipping at the extremes.** Any variant that takes `log p` needs `p` clipped away
   from 0 and 1. With `p = 0` exactly, the logit is `−∞` and the fit silently drops the
   row (or produces `nan`); the `EPS` clip is load-bearing.
10. **Calibration is not fairness.** Global calibration says nothing about subgroups; a
    model can be perfectly calibrated overall and badly miscalibrated for a minority
    group. Check the reliability diagram per group, not just pooled.

## Applications

- **Clinical risk scores.** A predicted 20% five-year risk has to mean 20% for the score
  to inform a decision, and the calibration curve is a standard part of any risk-model
  report.
- **Credit and fraud.** Calibrated default probabilities are needed for expected-loss
  calculations; a ranking is not enough because the decision is an expectation.
- **Churn and propensity models.** Marketing budgets are allocated by expected value, which
  multiplies probability by margin — so the probability scale matters, not just the order.
- **Model ensembling.** Averaging probabilities from differently calibrated models is
  dominated by the most overconfident one; calibrate each before averaging.
- **Threshold selection.** Choosing a cut-off from a cost matrix needs probabilities; with
  uncalibrated scores the threshold has to be tuned empirically for every deployment.
- **Neural network confidence.** Temperature scaling is the standard post-hoc fix for
  overconfident classifiers, and the cheapest baseline any uncertainty method must beat.
- **Distillation and compression.** A distilled model is trained on the teacher's softened
  probabilities, and temperature is the knob that decides how much of the dark knowledge
  survives.

## References

- Platt, *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized
  Likelihood Methods*, 1999 — Platt scaling and the smoothed-target correction.
- Zadrozny & Elkan, *Transforming Classifier Scores into Accurate Multiclass Probability
  Estimates*, 2002, KDD — isotonic calibration and its multiclass extensions.
- Niculescu-Mizil & Caruana, *Predicting Good Probabilities with Supervised Learning*,
  2005, ICML — the empirical comparison across model families that made calibration
  standard practice.
- Kull, Silva Filho & Flach, *Beyond Sigmoids: How to Obtain Well-Calibrated Probabilities
  from Binary Classifiers with Beta Calibration*, 2017, NeurIPS — the `beta` variant and
  its Beta-family derivation.
- Kull et al., *Dirichlet Calibration: Beyond Temperature Scaling*, 2019, NeurIPS — matrix
  scaling, off-diagonal regularisation, and the multiclass case.
- Guo, Pleiss, Sun & Weinberger, *On Calibration of Modern Neural Networks*, 2017, ICML —
  temperature scaling and the ECE metric used throughout this file.
- Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, 2007,
  JASA — why log-loss is the right objective and ECE is not.
- Ayer, Brunk, Ewing, Reid & Silverman, *An Empirical Distribution Function for Sampling
  with Incomplete Information*, 1955, Annals of Mathematical Statistics — the PAVA
  algorithm behind isotonic regression.
