# XGBoost

`xgboost.py` · Trees & ensembles · Tier 1 · 9 variants

## Intuition

Gradient boosting fits each tree to the negative gradient. XGBoost adds the second
derivative: every split is scored by how much it reduces a *second-order Taylor
expansion* of the loss, and every leaf is set to the Newton step rather than the
mean residual. That one change makes the same algorithm work for squared error,
logistic loss and ranking without writing a new splitting rule for each.

The other half of XGBoost is that the objective is written down explicitly:

$$
\mathcal{L} = \sum_i l(y_i,\hat y_i) + \sum_k\left(\gamma T_k
+ \tfrac12\lambda\|w_k\|^2 + \alpha\|w_k\|_1\right)
$$

`T_k` is the number of leaves, `w_k` their values. `gamma` is a price per split,
`lambda` shrinks every leaf, `alpha` makes leaves exactly zero. The split search
optimises this regularised objective directly, which is why XGBoost is less prone
to the unchecked growth of a vanilla GBM.

The variants are the ways to enumerate candidate splits (exact, approximate,
histogram), the DART dropout, linear leaves, and the four regularisation knobs.

## Variants

| Variant | What changes | Why it exists |
|:--|:--|:--|
| `exact` | every distinct feature value is a candidate | the reference split search |
| `approx` | per-node quantile candidates | sublinear in `n` for large data |
| `hist` | global histogram bins | the production default; cache-friendly |
| `dart` | randomly drop previous trees while fitting | prevents later trees from over-relying on early ones |
| `linear_booster` | leaves hold ridge models, not constants | captures local linear trends |
| `reg_alpha` | `L1` leaf penalty (default `α=1`) | sparse leaves, automatic pruning |
| `reg_lambda` | stronger `L2` leaf penalty (default `λ=10`) | smoother, smaller leaves |
| `colsample` | each tree sees 70% of the columns | decorrelation, faster fitting |
| `sparsity_aware` | missing values routed by gain | no imputation, missingness is information |

`exact`, `approx` and `hist` are the same objective with different candidate sets.
`reg_alpha` and `reg_lambda` are the same objective with different penalties. The
demo shows each default doing what it says: `L1` produces 160 zero leaves where the
unpenalised fit has none, and `λ=100` cuts the mean absolute leaf value from
`0.254` to `0.098`.

## Derivation

**Second-order expansion.** Around the current prediction `F_{m-1}(x_i)`, with
`g_i = ∂l/∂F` and `h_i = ∂²l/∂F²`,

$$
\mathcal{L}^{(m)} \approx \sum_i
\left[g_i f_m(x_i) + \tfrac12 h_i f_m(x_i)^2\right]
+ \gamma T + \tfrac12\lambda\sum_j w_j^2 + \alpha\sum_j|w_j|
$$

For a leaf containing rows `S`, the optimal value is

$$
w_S = -\frac{\operatorname{soft}(G_S,\alpha)}{H_S+\lambda},
\qquad
G_S=\sum_{i\in S}g_i,\quad H_S=\sum_{i\in S}h_i
$$

where `soft(G,α) = sign(G)·max(|G|−α, 0)`. The `L1` case is the proximal operator:
`α` eats the gradient before the `L2` shrink. With `α=0` this is the familiar
`−G/(H+λ)`.

**Split gain.** Substituting `w_S` back and subtracting the parent's score gives

$$
\text{gain} = \frac12\left[
\frac{\operatorname{soft}(G_L)^2}{H_L+\lambda}
+ \frac{\operatorname{soft}(G_R)^2}{H_R+\lambda}
- \frac{\operatorname{soft}(G)^2}{H+\lambda}
\right] - \gamma
$$

The `½` is the Taylor factor, `γ` is charged once per split. A split is taken only
if `gain > 0`, so `gamma` is a hard minimum-improvement threshold.

**Gradients and Hessians.** For squared loss, `g = F−y`, `h = 1`. For logistic loss,
`p = σ(F)`, `g = p−y`, `h = p(1−p)`. The Hessian is what makes XGBoost's leaves
different from a GBM's: a leaf with confident, well-classified rows has small `h`,
so the same `G` produces a large step; a leaf with uncertain rows is damped.

**Candidate enumeration.** Exact search sorts each feature and evaluates every
distinct value. That is `O(n log n)` per feature per node — fine for `n` in the
millions, too slow for billions. Approximate search proposes `O(max_bins)` quantile
candidates per node; histogram search uses *global* bin edges computed once, so
each node only accumulates `G` and `H` into bins, which is `O(n)` per feature and
cache-friendly. The demo's 16-bin histogram reaches `R²=0.950` against exact
`0.963` — the usual small loss for a large speed-up.

**DART.** Dropout meets boosting: at stage `m`, each previous tree is temporarily
removed with probability `p`, the gradients are computed from the surviving
ensemble (scaled by `1/(1−p)` to preserve the expected sum), and the new tree is
fitted to those gradients. At prediction time all trees are used, exactly like
inverted dropout. The effect is to stop late trees from fitting only the residual
of the early ones and to spread contribution across the ensemble.

**Linear leaves.** Instead of a constant `w_S`, fit

$$
\min_{w,b}\sum_{i\in S} h_i\left(-g_i/h_i - (x_i^\top w+b)\right)^2
+ \lambda\|w\|^2
$$

which is a weighted ridge regression with target `−g/h`. For squared loss this is
ordinary ridge on the residual; for logistic it is the Newton step in function
space. The split search is unchanged — it still uses `g,h` — so the tree decides
*where* a linear model is worth fitting, and the leaf decides *what* it is.

**Sparsity-aware splits.** For each candidate, evaluate the gain three ways: missing
values left, missing values right, and (when there are none) the ordinary split.
Store the winning direction at the node. At prediction, a missing value follows the
stored direction. This is why XGBoost needs no imputation: the tree learns where
missingness belongs, and the demo still reaches `R²=0.841` with 20% of the
informative feature removed.

## Code walkthrough

The regularised leaf is three lines, and the `L1` case is the only non-obvious one:

```python
def _soft(g, alpha: float):
    """Proximal operator of alpha|w|: sign(g) max(|g|-alpha, 0)."""
    return np.sign(g) * np.maximum(np.abs(g) - alpha, 0.0)
```

```python
    def _leaf_value(self, G: float, H: float) -> float:
        return -_soft(G, self.reg_alpha) / (H + self.reg_lambda)
```

`soft` is vectorised so the same function scores thousands of candidate splits;
`_leaf_value` is called with scalars. With `alpha=0` this is `−G/(H+λ)`; with
`alpha>0` a leaf whose gradient is smaller than `alpha` gets exactly zero, which is
the sparsity the demo counts.

The split gain is the objective difference, not an impurity heuristic:

```python
    def _split_gain(self, GL, HL, GR, HR, GP, HP):
        return 0.5 * (self._score(GL, HL) + self._score(GR, HR)
                      - self._score(GP, HP)) - self.gamma
```

```python
    def _score(self, G: float, H: float) -> float:
        """Negative of the minimised leaf objective, up to the factor 1/2."""
        s = _soft(G, self.reg_alpha)
        return s * s / (H + self.reg_lambda)
```

`_score` is `soft(G)²/(H+λ)`, which is `−2 ×` the minimised leaf objective. The
factor cancels in the gain, so it is left out. `gamma` is subtracted *after* the
`½`, which means it is in the same units as the Taylor-expanded loss — a common
source of "why does `gamma=1` kill every split?" confusion when the loss is mean
squared error and the gradients are tiny.

`_split_feature` is where the three enumerations and the missing-value logic meet:

```python
        if self.variant == "exact":
            cand = 0.5 * (v[:-1] + v[1:])
            cand = cand[v[1:] > v[:-1]]
        else:
            cand = self._candidates(v, j)
```

The `v[1:] > v[:-1]` mask drops zero-width candidates between duplicate values.
Keeping them is harmless numerically (gain 0) but doubles the work on discrete
features and can produce a split that sends nothing to one side. Histogram and
approximate candidates come from `_candidates`, which returns global bin edges or
per-node quantiles respectively.

The cumulative sums then give every candidate's left child at once:

```python
        cumG, cumH = np.cumsum(G), np.cumsum(H)
        # left side contains all values <= cand (searchsorted, side="right")
        pos = np.searchsorted(v, cand, side="right") - 1
```

`searchsorted(..., side="right") - 1` is the index of the last value `<= cand`.
Using `side="left"` would put values exactly equal to the threshold on the right,
which contradicts the `x <= t` convention used at prediction. The subtraction of 1
is the off-by-one that makes the left size `pos+1`, not `pos`.

Missing values are evaluated as a third child:

```python
        gain0 = self._split_gain(GL, HL, GR, HR, GP_obs, HP_obs)
        gain1 = self._split_gain(GL + Gm, HL + Hm, GR, HR, GP, HP)
        gain2 = self._split_gain(GL, HL, GR + Gm, HR + Hm, GP, HP)
```

`gain0` uses the observed parent (`GP_obs`) and is valid only when there are no
missing rows. `gain1` adds the missing gradients to the left, `gain2` to the right,
and both compare against the *full* parent `GP`. That distinction matters: scoring
a missing-left split against the observed parent understates the parent objective
and can make a bad split look good. `ok0` is explicitly disabled when `n_m > 0` for
the same reason.

Growth is short because the split carries its own default direction:

```python
        col = X[idx, split["feature"]]
        left = (col <= split["threshold"]) | (np.isnan(col) & split["missing_left"])
        li, ri = idx[left], idx[~left]
```

`split["missing_left"]` was chosen by gain in `_split_feature`, so the routing is
already decided; `_grow` only applies it. `idx` rather than a boolean mask over all
rows keeps each recursive call proportional to the node, not the dataset.

Gradients are the only task-specific part of fitting:

```python
    def _gradients(self, y, F):
        if self.task == "binary":
            p = _sigmoid(F)
            return p - y, p * (1.0 - p)
        return F - y, np.ones_like(F)
```

The squared-loss Hessian is exactly 1, so `H` is just the leaf count and `λ` is a
per-sample ridge. The logistic Hessian is `p(1−p)`, which is why a confident leaf
takes a bigger Newton step than an uncertain one even when their gradients match.

DART is the only branch that rewrites the current prediction instead of updating
it in place:

```python
            if self.drop_rate > 0.0 and self.estimators_:
                keep = self.rng.random(len(self.estimators_)) >= self.drop_rate
                self.n_dropped_ += int((~keep).sum())
                scale = 1.0 / (1.0 - self.drop_rate)
                F = np.full(n, base)
                for tree, k in zip(self.estimators_, keep, strict=True):
                    if k:
                        F += scale * self.lr * self._tree_predict(tree, X)
```

`scale` is inverted dropout: the surviving trees are amplified so the expected sum
equals the full ensemble's. At prediction `_full_raw` adds every tree *without*
the scale, which is the same convention as a dropout layer at test time. Computing
`F` from scratch each stage is `O(B² n)`; a production DART implementation
subtracts the dropped trees' contributions instead.

Linear leaves are a weighted ridge solve:

```python
        Xl = np.column_stack([X[idx], np.ones(idx.size)])
        hh = h[idx]
        target = -g[idx] / np.maximum(hh, 1e-12)
        A = (Xl * hh[:, None]).T @ Xl
        pen = np.full(Xl.shape[1], self.reg_lambda)
        pen[-1] = 0.0
        A += np.diag(pen)
        b = Xl.T @ (hh * target)
```

`target = −g/h` is the Newton pseudo-response; weighting by `h` turns the Taylor
expansion into a weighted least-squares problem. `pen[-1] = 0` leaves the intercept
unpenalised — penalising it would shrink every leaf toward zero and bias the model,
exactly as in ridge. `np.maximum(hh, 1e-12)` guards a saturated logistic leaf.

## Complexity

| Quantity | Cost |
|:--|:--|
| Exact split, one feature | `O(n log n)` per node |
| Histogram split, one feature | `O(n)` per node after a one-time `O(nd)` binning |
| Approximate | `O(n log n)` for the quantiles, then `O(n)` per feature |
| One tree | `O(d n log n · depth)` exact, `O(d n · depth)` hist |
| `B` trees | `B ×` the above |
| Prediction, one row | `O(B · depth)` |

Memory is `O(B · nodes + n · bins)` for the histogram version. The exact search is
the reference; histogram is what makes `n` in the hundreds of millions feasible.
The demo's exact, hist and approx models all train 120 depth-3 trees on 350 rows in
well under a second, which is why the differences show up as accuracy, not runtime.

## Failure modes

1. **`gamma` on the wrong scale.** `gamma` is compared to a Taylor-expanded gain;
   with mean squared error and small residuals the gains are tiny, so `gamma=1`
   prunes everything. Tune `gamma` on a validation set, or normalise the loss.
2. **`L1` with unstandardised features.** `alpha` is in gradient units, so a feature
   with a large scale produces large `G` and is penalised less. Standardise, or
   tune `alpha` per feature block.
3. **DART inference mismatch.** Training uses `1/(1−p)` scaling; if prediction also
   scales, every tree is too large by that factor. Use the scaling in exactly one
   place.
4. **Histogram bins too coarse.** `max_bins=16` merges distinct regimes and can
   underfit a sharp threshold; 64–256 is the usual range. The demo's 16 bins lose
   about `0.013` in `R²` against exact, which is a fair price.
5. **Approximate quantiles are not weighted.** This implementation uses plain
   `np.quantile`; the paper's sketch weights by Hessian, so a candidate can be
   placed where few *important* rows live. On imbalanced Hessians the plain version
   is noticeably worse.
6. **Missing values and linear leaves.** A linear leaf predicts `x·w+b`; a missing
   `x_j` becomes `nan`. This implementation fills `nan` with zero at prediction,
   which is a convention, not a model. Use tree leaves or impute for the linear
   booster.
7. **No early stopping.** `n_estimators` is fixed; a large value overfits the
   validation set even with `lambda`. Watch the validation curve or add a holdout
   loop.
8. **Binary labels must be 0/1.** The logistic gradients assume it; labels `±1` or
   strings silently fit the wrong model. Encode explicitly.
9. **`reg_lambda` default.** XGBoost's real default is 1; the `reg_lambda` variant
   here bumps it to 10 to make the effect visible. Do not read the demo's shrinkage
   as "XGBoost always uses 10".

## Applications

- **Tabular competitions and industry baselines.** XGBoost remains the model to
  beat on medium-sized structured data, especially with missing values and mixed
  feature types.
- **Credit scoring and fraud.** The regularised objective plus monotone constraints
  (not implemented here) gives an auditable model; `gamma` and `lambda` control
  complexity directly.
- **Ranking and recommendation.** LambdaMART is this objective with a pairwise
  ranking loss; the second-order gain is what makes the pair weights tractable.
- **Insurance and demand.** Tweedie and quantile variants of the same objective
  model claims and intervals; the histogram splitter is the production path.
- **Feature-store pipelines.** Sparsity-aware splits mean no imputation step at
  serving time, which removes a train/serve skew that plagues linear pipelines.
- **As a teaching reference.** Every knob in the real library maps to one line
  here: `eta`, `max_depth`, `min_child_weight`, `lambda`, `alpha`, `gamma`,
  `subsample`, `colsample_bytree`.

## References

- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, 2016, KDD — the
  regularised objective, the gain formula and the sparsity-aware split.
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, 2001 —
  the first-order ancestor.
- Rashmi & Gilad-Bachrach, *DART: Dropouts meet Multiple Additive Regression
  Trees*, 2015, AISTATS — the dropout variant.
- Chen, He, Benesty et al., *XGBoost documentation*, 2016– — the exact/approx/hist
  split descriptions.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  2009, ch. 10 — the statistical view of boosted trees.
