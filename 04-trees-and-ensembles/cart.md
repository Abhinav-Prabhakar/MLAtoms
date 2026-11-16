# CART

`cart.py` · Trees & ensembles · Tier 1 · 7 variants

## Intuition

A decision tree asks one question at a time: *is feature `j` above or below `t`?*
Each answer sends the sample left or right, and the process repeats until a leaf.
The leaf's value is the mean (regression) or the class distribution
(classification) of the training points that landed there.

Growing the tree is greedy. At each node, every feature is scanned for the
threshold whose split most reduces impurity, the best split is taken, and the two
children recurse. There is no global optimisation and no backtracking, yet the
result is a piecewise-constant function that can approximate any smooth surface to
arbitrary accuracy given enough depth.

The seven variants are not seven models. They are four impurity choices and three
practical fixes: pruning (stop overfitting after the fact), surrogate splits
(behave sensibly when the split feature is missing at prediction time), and
missing-value routing (learn where missing values should go during training).

## Variants

| Variant | Task | Impurity | What it adds |
|:--|:--|:--|:--|
| `gini` | classification | `1 − Σp²` | the fast default; measures label disagreement |
| `entropy` | classification | `−Σp log₂p` | information gain; more sensitive to rare classes |
| `variance` | regression | `Var(y)` | the standard regression tree |
| `friedman_mse` | regression | `n_l n_r/n · (μ_l−μ_r)²` | Friedman's split score; same ranking, cheaper |
| `cost_complexity_pruning` | classification | gini | grow deep, then prune with `α` |
| `surrogate_splits` | classification | gini | backup splits for missing features |
| `missing_values` | classification | gini | learn which side missing values go to |

`gini` and `entropy` usually produce the same tree; they differ when a node has
several small classes, where entropy's `log` penalises rare-class impurity more
than gini's quadratic. `variance` and `friedman_mse` rank every *binary* split
identically — the demo asserts that their predictions are equal — because variance
reduction is a positive multiple of the Friedman score. The difference is
numerical convenience and tie-breaking, not statistics.

## Derivation

**Impurity and gain.** For a node with `n` samples and class proportions `p_k`,

$$
\text{Gini} = 1-\sum_k p_k^2,
\qquad
\text{Entropy} = -\sum_k p_k\log_2 p_k,
\qquad
\text{Var} = \frac1n\sum_i (y_i-\bar y)^2
$$

The gain of a split into children `L, R` is

$$
\Delta = I(\text{parent}) - \frac{n_L}{n}I(L) - \frac{n_R}{n}I(R)
$$

For regression, substituting `Var` and simplifying gives

$$
\Delta = \frac{n_L n_R}{n^2}(\mu_L-\mu_R)^2
$$

so the Friedman score `n_L n_R/n · (μ_L−μ_R)²` is `n²Δ` — the same ordering, and
only one mean per child instead of a full pass over the labels. That is why
`friedman_mse` exists in every serious implementation.

**Greedy optimality is local.** For a fixed set of candidate thresholds, scanning
all of them finds the best *single* split. It does not find the best *tree*: a
feature that looks useless now may enable a great split two levels down (XOR is
the canonical example, and the demo's 7-leaf tree is what greedy growth needs to
represent it). This is the bias that makes trees high-variance and motivates
ensembles.

**Cost-complexity pruning.** Define the risk of a subtree `T` as

$$
R_\alpha(T) = R(T) + \alpha|T|,
\qquad
R(T) = \sum_{\text{leaves}} \frac{n_\ell}{n}\,I(\ell)
$$

For each internal node `t`, the weakest link is

$$
g_t = \frac{R(t)-R(T_t)}{|T_t|-1}
$$

where `R(t)` is the risk if `t` were a leaf and `T_t` is its subtree. Repeatedly
collapse the node with the smallest `g_t` while `g_t ≤ α`. The sequence of
collapses is the cost-complexity path; every `α` selects a subtree on it. This is
the pruning algorithm of Breiman et al., and it is exact for the given tree — it
does not search all trees.

**Surrogate splits.** When the primary feature `x_j` is missing at prediction
time, a surrogate is another feature `x_k` whose split best reproduces the primary
direction on the training data. For each candidate split of `x_k`, agreement is

$$
a = \frac{1}{n}\sum_i \mathbb{1}\big[(x_{ik}\le t_k) = (x_{ij}\le t_j)\big]
$$

and the surrogate is used with a flip if `a < 1/2`. The best surrogate per feature
is kept, sorted by `max(a, 1−a)`, and the first surrogate whose feature is observed
decides the direction. If none is observed, the learned majority direction is
used. This is the CART answer to missing data and it is why the demo's accuracy
stays at `0.998` when the informative feature is entirely absent at test time.

## Code walkthrough

The tree is a nested dict: a leaf stores `value`, `proba` and its own impurity; an
internal node stores the split and two children. `_grow` is the recursion, and the
only decision it makes is whether to stop:

```python
        if (
            y.size < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or node["leaf_impurity"] <= 1e-12
        ):
            return node
        split = self._best_split(X, y)
        if split is None or split["gain"] <= 1e-12:
            return node
```

`leaf_impurity <= 1e-12` is the purity stop, and the gain test is the second one:
a split that does not improve impurity is not taken even if it is geometrically
valid. Without the gain test, a feature with thousands of unique values would keep
splitting noise forever.

Split search sorts the feature once and evaluates every cut point with prefix
sums:

```python
        order = np.argsort(v, kind="mergesort")
        v, y_obs = v[order], y_obs[order]
        cut = np.nonzero(v[1:] > v[:-1])[0]
        thr = 0.5 * (v[cut] + v[cut + 1])
```

`cut` holds the positions where the sorted value changes, so only those positions
are legal splits. Splitting between equal values is never useful, and trying every
index instead of every *distinct* value is both slower and a source of
zero-gain nodes. `mergesort` is stable, which makes tie-breaking reproducible.

Classification evaluates all cuts at once with a cumulative class count:

```python
            counts = np.zeros((n_obs, self.n_classes))
            counts[np.arange(n_obs), y_obs] = 1.0
            cum = np.cumsum(counts, axis=0)
            lc = cum[cut]
            rc = cum[-1] - lc
            mc = np.bincount(y_miss, minlength=self.n_classes).astype(float)
            imp_l0 = self._imp_from_counts(lc, n_l)
            imp_r0 = self._imp_from_counts(rc, n_r)
```

`cum[cut]` is the left class histogram for every candidate; `cum[-1] - lc` is the
right one. `np.bincount` only needs the class labels of the missing rows. The three
gains then differ only in *where the missing rows are counted*:

```python
            gain0 = parent - (n_l * imp_l0 + n_r * imp_r0) / (n_l + n_r)
            gain1 = parent - ((n_l + n_m) * imp_l1 + n_r * imp_r0) / n
            gain2 = parent - (n_l * imp_l0 + (n_r + n_m) * imp_r1) / n
```

`gain0` is the split on observed rows, `gain1` sends missing values left, `gain2`
sends them right. Evaluating both is the difference between "missing is an error"
and "missing is information": the learned direction is whichever reduces impurity
more, and it is stored as `missing_left`. The demo's 25%-missing training run
reaches `0.892` accuracy largely because of these two lines.

Regression reuses the same prefix-sum trick with `y` and `y²`:

```python
            sy = np.cumsum(y_obs)
            sy2 = np.cumsum(y_obs * y_obs)
            sl, sl2 = sy[cut], sy2[cut]
            sr, sr2 = sy[-1] - sl, sy2[-1] - sl2
```

Child variance is `sy2/n − (sy/n)²`; no second pass over the data is needed. The
`np.maximum(..., 0)` on the variance guards the `1e-17` negative that
`E[y²] − E[y]²` produces for a nearly constant child, and a negative variance
would make the gain *increase* as the split gets worse.

Surrogates are learned only when the variant asks for them:

```python
            side = np.arange(v.size)[None, :] <= cut[:, None]      # (cuts, n)
            agree = (side == pl[None, :]).mean(axis=1)
            eff = np.maximum(agree, 1.0 - agree)
            i = int(np.argmax(eff))
```

`side` is a `cuts × n` boolean matrix, so agreement for every threshold is one
vectorised comparison. `eff = max(a, 1−a)` is essential: a surrogate that is
perfectly *anti*-correlated is as useful as one perfectly correlated, provided the
stored `flip` inverts the direction. Without it, the best anti-correlated surrogate
is discarded and the tree falls back to the majority branch.

Pruning is the weakest-link loop:

```python
                if leaves > 1:
                    g = (n["n"] / self.n_total * n["leaf_impurity"]
                         - self._risk(n)) / (leaves - 1)
```

`n["n"]/n_total * leaf_impurity` is `R(t)`, the risk if this node became a leaf;
`self._risk(n)` is `R(T_t)`, the sum over its current leaves. The denominator
`leaves − 1` is the number of parameters removed by collapsing, which is exactly
`|T_t| − 1`. The loop only collapses nodes with `g ≤ α`, so smaller `α` keeps more
leaves; the demo's `50 → 6` leaves at `α=0.02` also moves test error from `0.053`
to `0.004` on noisy labels.

Prediction is a while-loop, and the missing-value branch is the whole reason the
node stores more than a threshold:

```python
            if np.isnan(x[j]):
                left = node["missing_left"]
                for s in node["surrogates"]:
                    if not np.isnan(x[s["feature"]]):
                        side_left = x[s["feature"]] <= s["threshold"]
                        left = bool(side_left) != s["flip"]
                        break
            else:
                left = x[j] <= node["threshold"]
```

The surrogate only runs when the primary feature is missing *and* the surrogate's
feature is observed. `!= flip` is XOR: if the surrogate is anti-correlated, a
left-side surrogate answer means the primary would have gone right. Falling
through to `missing_left` is the unconditional default, and it is also what the
`n_surrogates=0` ablation uses.

## Complexity

| Operation | Time | Memory |
|:--|:--|:--|
| One split search at a node | `O(d n log n)` (sort) + `O(d n)` (prefix sums) | `O(n)` per feature |
| Growing a balanced tree of depth `h` | `O(d n h log n)` | `O(nodes)` |
| Prediction, one sample | `O(h)` | `O(1)` |
| Cost-complexity pruning | `O(nodes²)` in this implementation | `O(nodes)` |

The `n log n` per node is the sort; with presorted features it drops to `O(n)`.
Pruning is quadratic because each collapse recomputes risks from the root; a
production implementation maintains the `g` values in a heap and is `O(nodes log
nodes)`. The tree itself is the memory: one dict per node, with the class
histogram at every node in this version.

## Failure modes

1. **High variance.** A small change in the data can change the root split and the
   entire tree below it. A single tree is rarely the final model; bagging
   (`random_forest.py`) and boosting (`gradient_boosting.py`) exist for this.
2. **Axis-aligned splits.** A diagonal boundary must be approximated by a staircase
   of axis-aligned cuts, costing depth. Rotating the features (PCA) or using an
   oblique tree is the fix.
3. **Greedy myopia.** XOR needs four quadrants but the first split looks useless;
   the tree still finds it here, but deeper interactions can be missed. Ensembles
   and deeper trees compensate at the cost of variance.
4. **Class imbalance.** Gini and entropy both prefer splits that separate the
   majority class, and a leaf can predict the majority everywhere. Pass class
   weights (not implemented here) or resample; accuracy is not the right metric.
5. **Unordered categorical features.** Treating an integer category as a number
   imposes a false order. Encode with one-hot or use a categorical split that
   groups levels (LightGBM's `categorical` variant).
6. **Missing values at prediction time with no surrogate.** The tree falls back to
   `missing_left`, which is a constant learned from training. If missingness is
   informative at test time in a different way, the prediction is biased. Fit with
   surrogates if the feature can be absent.
7. **Pruning with the wrong risk.** `ccp_alpha` is on the scale of node impurity
   times `n_node/n_total`. A value tuned for a 50/50 classification problem is not
   meaningful for a regression tree with `Var(y) ≈ 10⁴`; rescale or tune on a
   validation set.
8. **Deep trees on tiny leaves.** `min_samples_leaf=1` lets a leaf hold one point,
   so the tree memorises noise. The default here is 1 to keep the mechanics
   visible; production trees use 1–20 with a validation-tuned depth.
9. **No monotonicity.** Trees can produce a lower prediction for a larger feature
   value in a neighbouring leaf. If the domain requires monotone responses
   (pricing, risk), constrain the splits or use a monotone booster (LightGBM).

## Applications

- **Tabular baselines.** A depth-4 CART is still the first model to beat on most
  tabular datasets, and its splits are readable by domain experts.
- **Credit and insurance underwriting.** Axis-aligned rules on income, age and
  utilisation map directly to policy; `cost_complexity_pruning` keeps the rule
  list short enough to audit.
- **Clinical decision rules.** A shallow tree is a triage score: "if age > 65 and
  systolic > 140, admit". Interpretability is the product, not an afterthought.
- **Feature interactions.** The depth at which a feature first appears is a crude
  measure of interaction order; `random_forest`'s permutation importance builds on
  this.
- **Missing-data pipelines.** Surrogate splits are the classical answer for
  questionnaires and sensors where a field can be absent at scoring time, and they
  remain the reference against which imputation is compared.
- **As the base learner for ensembles.** Every atom in this chapter — random
  forest, boosting, XGBoost, LightGBM, AdaBoost — is this split search wrapped in
  a different outer loop.

## References

- Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees*, 1984 —
  the original CART, including surrogate splits and cost-complexity pruning.
- Friedman, *A Comparison of Alternative Tests of Significance for the Problem of
  m Rankings*, 1937 / *Multivariate Adaptive Regression Splines*, 1991 — the
  Friedman MSE score used for regression splits.
- Quinlan, *Induction of Decision Trees*, 1986 — ID3 and information gain, the
  entropy counterpart.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  2009, ch. 9 — the bias-variance and pruning analysis.
- Loh, *Classification and Regression Trees*, 2011, WIREs Data Mining — a survey of
  the split-selection literature.
