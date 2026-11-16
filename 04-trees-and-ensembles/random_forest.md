# Random forest

`random_forest.py` · Trees & ensembles · Tier 1 · 6 variants

## Intuition

One tree is unstable. Move a few training points and the best root split can
change, which changes every split below it, which changes every prediction. The
tree has low bias but enormous variance.

Averaging many trees fixes variance, but only if the trees disagree. Bootstrap
resampling alone is not enough: on a problem with one dominant feature, nearly
every bootstrap tree splits on that feature first, so the trees are almost
identical and their average has the same variance as one tree. The forest's second
ingredient is **random feature subsets at every split**. Each tree is forced to
look at a different part of the feature space, so their errors decorrelate and the
average cancels them.

Extra trees push the same idea further: instead of searching for the best
threshold, each candidate feature gets a *random* threshold. The trees are more
biased individually but even less correlated, and the ensemble is faster because
there is no sort.

## Variants

| Variant | Task | What changes | Why |
|:--|:--|:--|:--|
| `classification` | classification | majority vote over probability averages | the default forest |
| `regression` | regression | average of tree means | continuous targets |
| `extra_trees` | either | random thresholds, no bootstrap | more decorrelation, faster |
| `oob` | classification | reports the out-of-bag score | free validation, no holdout |
| `permutation_importance` | either | shuffles each column and measures the drop | model-agnostic importance |
| `balanced_subsample` | classification | equal sample count per class in each draw | rare-class recall |

`classification` and `regression` differ only in the leaf value and the averaging:
probabilities versus means. `extra_trees` is the same forest with `splitter=
"random"` and `bootstrap=False`. `oob`, `permutation_importance` and
`balanced_subsample` are not different estimators — they change how the forest is
sampled or how it is inspected, and the demo measures each one.

## Derivation

**Variance of an average.** For `B` trees with individual variance `σ²` and
pairwise correlation `ρ`,

$$
\mathrm{Var}\!\left(\frac1B\sum_{b=1}^B T_b\right)
= \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

The second term vanishes as `B` grows; the first does not. This is the entire
design argument for the forest: increasing `B` cannot help if `ρ≈1`, so the
algorithm must reduce `ρ`. Restricting each split to `m` of `d` features does
that, at the cost of slightly weaker individual trees.

**Why `sqrt(d)`.** For classification, Breiman's default `m = √d` keeps the
correlation low without starving the trees. For regression, `m = d/3` is the
usual choice; this implementation defaults to all features because the demo's
`d=2` is too small for a fraction to mean anything. The right `m` is a
hyperparameter: smaller `m` means less correlation and more bias.

**Out-of-bag.** A bootstrap sample draws `n` times with replacement, so

$$
P(\text{sample } i \text{ is out-of-bag}) = \left(1-\frac1n\right)^n \to e^{-1}
\approx 0.368
$$

Each tree therefore has a validation set of about 37% of the data that it never
saw, and every training point is OOB for about `B/e` trees. Averaging only those
trees gives a validation prediction at no extra fitting cost. The estimate is
slightly pessimistic because each OOB prediction uses fewer trees than the final
forest, but it is unbiased for the sampling scheme.

**Impurity importance.** Mean decrease in impurity (MDI) credits a split with

$$
\text{MDI}_j = \sum_{\text{nodes splitting on }j}
\frac{n_{\text{node}}}{n_{\text{total}}}\,\Delta I_{\text{node}}
$$

This is what `feature_importances_` computes. It is cheap and it is biased: a
continuous feature with many distinct values offers more candidate thresholds, so
it is more likely to be chosen by chance, and correlated features split the credit
between them.

**Permutation importance.** Shuffle column `j`, predict again, and report

$$
\text{PI}_j = \text{score}(X,y) - \text{score}(X^{(j\text{ shuffled})},y)
$$

This measures how much the fitted model *uses* the feature, not how much the
feature could help in a different model. It is unbiased under the null (a noise
column scores zero in expectation) but it is fooled by correlated features: if
`x₀` and `x₁` are duplicates, shuffling either one alone leaves the other intact
and both importances are near zero.

**Balanced subsampling.** The variant draws the same number of rows from every
class, `n // K` per class, so each tree sees a balanced problem regardless of the
original prevalence. This changes the *sampling distribution*, not the impurity:
the tree still maximises gini on the sample it receives. The cost is that the
forest no longer estimates the population prior, so predicted probabilities are
biased toward the uniform; the variant is for recall, not calibration.

## Code walkthrough

The forest is a loop around `CART`, and the loop body is where all the randomness
enters:

```python
            tree = CART(
                "gini" if self.task == "classification" else "variance",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                splitter="random" if self.variant == "extra_trees" else "best",
                random_state=int(self.rng.integers(0, 2**31 - 1)),
            ).fit(X[idx], y[idx])
```

`max_features` is passed straight to the tree, so the feature subset is drawn
*per node*, not per tree — that is the correct granularity, and it is why the trees
differ even when they see the same bootstrap sample. The per-tree `random_state`
comes from the forest's generator, so the whole ensemble is reproducible from one
seed.

Sampling is the only place where the variants diverge:

```python
        if self.variant == "balanced_subsample" and self.task == "classification":
            classes = np.unique(y)
            per = max(1, n // classes.size)
            parts = []
            for c in classes:
                pool = np.nonzero(y == c)[0]
                parts.append(self.rng.choice(pool, size=per, replace=True))
            return np.concatenate(parts)
        if not self.bootstrap:
            return np.arange(n)
        return self.rng.integers(0, n, size=n)
```

The balanced branch returns `K·(n//K)` rows, not exactly `n`, so a tree may see a
few fewer points. That is deliberate: forcing an exact total would require
rounding one class differently and reintroducing a small imbalance. `replace=True`
is what makes the minority class drawable `per` times even when it has fewer than
`per` distinct rows; without it, `np.random.choice` raises.

Out-of-bag predictions are accumulated per tree and divided at the end:

```python
        for tree, inbag in zip(self.trees_, self.inbags_, strict=True):
            mask = ~inbag
            if not mask.any():
                continue
            counts[mask] += 1.0
            if self.task == "classification":
                sums[mask] += tree.predict_proba(X[mask])
            else:
                sums[mask] += tree.predict(X[mask])
        seen = counts > 0
```

`counts` is the number of trees for which each row was OOB. A row can be in-bag in
every tree (especially with small `n`), in which case `seen` is false and it is
excluded from `oob_score_` rather than scored against a zero prediction. The
`strict=True` on `zip` catches a tree/bag length mismatch, which would otherwise
silently misalign predictions.

Prediction is one line per task:

```python
        if self.task == "regression":
            return np.mean([t.predict(X) for t in self.trees_], axis=0)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
```

Averaging probabilities before `argmax` is not the same as majority-voting labels:
a class with 40% probability in every tree wins over one with 30% in a majority of
trees but 0% elsewhere. Probability averaging is the lower-variance rule and is
what `predict_proba` exposes. For regression, the mean of tree predictions is the
bagged estimator; the mean of leaf values is the same thing because each leaf is a
mean.

`_mdi` walks every tree and credits each split by its sample share:

```python
        def walk(node, total):
            if node["leaf"]:
                return
            imp[node["feature"]] += node["n"] / total * node["gain"]
            walk(node["left"], total)
            walk(node["right"], total)
```

`node["gain"]` is already the impurity decrease *on that node's sample*, so
multiplying by `n_node/n_total` converts it to a contribution to the whole
dataset. Normalising the sum to 1 at the end makes the vector comparable across
forests but removes the absolute scale — do not read `0.3` as "30% of the
variance", only as a ranking.

Permutation importance is deliberately the slow, honest version:

```python
        for j in range(self.n_features_):
            drops = np.empty(n_repeats)
            for r in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = self.rng.permutation(Xp[:, j])
                drops[r] = base - self.score(Xp, y)
```

`X.copy()` per repeat keeps every other column fixed; shuffling in place across
features would compound the permutations. The score is recomputed on the *same*
data, so the drop measures reliance on the column, not generalisation. The
demo's signal features score `0.195–0.303` and the noise features `0.001–0.019`,
which is the separation the method is supposed to provide.

`score` uses the right denominator for each task:

```python
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-12)
```

`R²` is `1 − SS_res/SS_tot`; the `max(..., 1e-12)` only guards a constant target,
where `R²` is undefined and `0` is a defensible convention. Classification uses
plain accuracy, so the permutation importance of a classifier is a drop in
accuracy — a coarse metric that can understate the importance of a feature that
changes probabilities without changing the argmax. Switch to log loss if that
matters.

## Complexity

| Quantity | Cost |
|:--|:--|
| Fit | `B · O(d n log n · depth)` for the forest; each split is `O(m n log n)` |
| Predict one sample | `O(B · depth)` |
| Predict `n` samples | `O(B · n · depth)` |
| Memory | `O(B · nodes)` |
| Permutation importance | `O(d · repeats · B · n · depth)` |

The forest is embarrassingly parallel across trees; this implementation is
sequential so the code stays readable. `extra_trees` is faster per tree because it
skips the sort and the full scan, at the price of needing more trees. Prediction is
the bottleneck in serving: `B=500` trees of depth 15 means 7,500 comparisons per
row, which is why production forests are often distilled or pruned.

## Failure modes

1. **Correlated trees.** If one feature dominates, `ρ` stays near 1 and adding
   trees stops helping. The fix is a smaller `max_features` or a different feature
   representation, not more trees.
2. **OOB is not a test set.** It estimates performance on the training
   distribution, so it cannot detect distribution shift or leakage across time.
   With `n` small, some rows are never OOB and the estimate quietly ignores them.
3. **MDI favours high-cardinality features.** Continuous and ID-like columns get
   more candidate thresholds and accumulate more credit. Use permutation
   importance, or group correlated features before interpreting.
4. **Permutation importance with correlated features.** Shuffling one of two
   duplicates leaves the model able to use the other, so both importances collapse
   to zero. Permute groups of correlated columns together, or use conditional
   permutation.
5. **Balanced sampling and probability calibration.** Equalising classes changes
   the prior, so `predict_proba` no longer matches the population frequency.
   Recalibrate on a held-out set if probabilities are the product.
6. **Extrapolation.** Trees predict a constant outside the training range; a
   regression forest cannot extrapolate a trend. Do not use it for time series
   with a trend unless the target is detrended.
7. **Many trees, no gain.** `n_estimators` has diminishing returns once the OOB
   curve flattens; 100–500 is usually enough. More trees cost latency forever,
   while the variance reduction saturates.
8. **Class imbalance without the variant.** Plain bootstrapping preserves the
   prevalence, so a 5% class can be absent from many bootstrap samples. The demo's
   balanced accuracy rises from `0.781` to `0.864` with `balanced_subsample`; if
   that is not enough, combine with threshold tuning.

## Applications

- **Tabular default.** The strongest off-the-shelf baseline for mixed-type,
  medium-sized data; it needs no scaling, handles missing values (through CART),
  and gives importances for free.
- **Remote sensing and land cover.** Thousands of spectral features, correlated
  bands, and a need for per-pixel uncertainty; the forest's OOB estimate and
  permutation importance are standard in the literature.
- **Bioinformatics.** Gene-expression classification with `p ≫ n`; random feature
  subsets make the forest robust to the correlated gene blocks that break linear
  models.
- **Fraud and anomaly triage.** Rare-event recall is the priority, so
  `balanced_subsample` plus a low decision threshold; the OOB score gives a quick
  read on whether the model learned anything.
- **Feature screening.** Permutation importance ranks inputs before a costly
  model; the demo's clean separation between signal and noise is the intended use.
- **As a component.** The forest is the base learner in stacking, the fallback in
  AutoML, and the reference model against which boosting is judged.

## References

- Breiman, *Random Forests*, 2001, Machine Learning — the original paper, including
  the `√d` default and the OOB estimate.
- Breiman, *Bagging Predictors*, 1996, Machine Learning — the variance argument.
- Geurts, Ernst & Wehenkel, *Extremely Randomized Trees*, 2006, Machine Learning —
  random thresholds and the bias/variance trade-off.
- Louppe, Wehenkel, Sutera & Geurts, *Understanding Variable Importances in Forests
  of Randomized Trees*, 2013, NeurIPS — the MDI bias analysis.
- Strobl, Boulesteix, Zeileis & Hothorn, *Bias in Random Forest Variable Importance
  Measures*, 2007, BMC Bioinformatics — why permutation importance is preferred
  under correlation.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  2009, ch. 15 — the correlation decomposition and forest asymptotics.
