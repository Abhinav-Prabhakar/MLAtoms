# Gradient boosting

`gradient_boosting.py` · Trees & ensembles · Tier 1 · 7 variants

## Intuition

Boosting is a sequence of weak models, each one correcting the mistakes of the
ensemble so far. A random forest averages independent trees and reduces variance;
boosting fits trees *sequentially* and reduces bias. The first tree is barely
better than a stump, the second fits the residual of the first, and after a few
hundred stages the sum is a flexible function.

The modern view is functional gradient descent. The model is a function `F(x)`,
the loss is `L(y, F)`, and the negative gradient

$$
r_i = -\left.\frac{\partial L(y_i, F)}{\partial F}\right|_{F=F_{m-1}(x_i)}
$$

is a *pseudo-residual*. A regression tree `h_m` is fitted to `r`, and the update is
`F_m = F_{m−1} + η·h_m`. With squared loss the pseudo-residual is `y−F` and the
tree's own mean is already the right leaf value. For every other loss, the leaf
value must be recomputed with a one-step Newton update — that is the part most
implementations get subtly wrong.

## Variants

| Variant | Loss | Pseudo-residual / leaf | Use |
|:--|:--|:--|:--|
| `regression` | squared | `y−F`, leaf mean | continuous targets |
| `binary` | logistic | `y−p`, Newton leaf `Σ(y−p)/Σp(1−p)` | two classes |
| `multiclass` | softmax cross-entropy | `Y_k−p_k`, `(K−1)/K · Σr/Σp(1−p)` | `K` classes, `K` trees per stage |
| `shrinkage` | squared | same fit, smaller `η` | the regulariser that makes boosting work |
| `subsample` | squared | same fit on a random row subset | stochastic boosting, faster and less correlated |
| `quantile` | pinball `ρ_α` | `α−1[y<F]`, leaf `α`-quantile of residual | prediction intervals, any quantile |
| `deviance` | logistic | identical to `binary` | the same model viewed as a likelihood ratio |

`shrinkage` and `subsample` are not different losses — they are the two knobs that
turn a high-variance fit into a generalising one. `deviance` is the logistic model
under its classical name; the demo asserts that it predicts identically to
`binary`, because sharing a likelihood is the point.

## Derivation

**Functional gradient descent.** Treat `F` as a vector `(F(x₁),…,F(xₙ))`. Gradient
descent in that space is `F ← F − η∇L(F)`, but `∇L` is only defined at the training
points, so the step is restricted to a tree-shaped function `h`. Fitting `h` to the
negative gradient by least squares and then choosing a per-leaf scale `γ` gives

$$
F_m(x) = F_{m-1}(x) + \eta\sum_{\ell} \gamma_{\ell}\,\mathbb{1}[x\in\ell]
$$

**Squared loss.** `L = ½Σ(y−F)²`, so `−∂L/∂F = y−F`. For a leaf with rows `S`,

$$
\gamma = \arg\min_\gamma \sum_{i\in S}(r_i-\gamma)^2 = \bar r
$$

which is exactly what a regression tree predicts. No Newton step is needed, which
is why the `regression` branch has no leaf post-processing.

**Logistic loss.** `L = Σ[log(1+e^F) − yF]`, `p = σ(F)`, and

$$
-\frac{\partial L}{\partial F} = y-p,
\qquad
\frac{\partial^2 L}{\partial F^2} = p(1-p)
$$

A one-step Newton update in leaf `S` minimises the second-order Taylor expansion:

$$
\gamma_S = \frac{\sum_{i\in S}(y_i-p_i)}{\sum_{i\in S}p_i(1-p_i)}
$$

The denominator is the curvature: leaves where the model is already confident have
small `p(1−p)` and take a correspondingly large step. Using the tree's mean
residual instead would give `γ = mean(y−p)`, which is the wrong scale whenever the
leaf is heterogeneous, and the model still trains — just more slowly and with worse
calibration.

**Multiclass.** With `K` scores and `p_k = softmax(F)_k`, the negative gradient is
`y_k−p_k` and the Hessian is `p_k(1−p_k)`. Friedman's factor

$$
\gamma_{k,S} = \frac{K-1}{K}\,
\frac{\sum_{i\in S}(y_{ik}-p_{ik})}{\sum_{i\in S}p_{ik}(1-p_{ik})}
$$

compensates for the fact that the `K` trees are fitted simultaneously and the
softmax couples their updates. Omitting `(K−1)/K` still fits, but the effective
learning rate is `K/(K−1)` times too large, which shows up as instability at
`K=10` and is invisible at `K=2`.

**Quantile loss.** For `ρ_α(r) = max(αr, (α−1)r)`,

$$
-\frac{\partial \rho_\alpha(y-F)}{\partial F}
= \begin{cases}\alpha & y>F\\ \alpha-1 & y<F\end{cases}
$$

The tree is fitted to this two-valued pseudo-residual (it needs only the *sign* of
the residual), and the leaf value is the line-search optimum

$$
\gamma_S = \text{quantile}_\alpha\{y_i-F_i : i\in S\}
$$

so a leaf's shift is the residual quantile, not its mean. With `α=0.5` this is the
median and the fit is robust; with `α=0.9` it estimates the conditional 90th
percentile.

**Shrinkage.** A stage of size `η` changes the function by `η·h` instead of `h`.
Small `η` means more stages but a smoother path: each tree sees a slightly
different residual and the ensemble averages their errors. The demo's `η=1.0` and
`η=0.05` use the same 300 trees, and the small step wins on test MSE (`0.070` vs
`0.032`).

**Subsampling.** Stochastic gradient boosting fits each tree on a random fraction
`s` of the rows. The gradient estimate is noisier but the trees decorrelate, which
is the same bias-variance argument as the forest's feature subsets. The demo uses
`s=0.5` and a test MSE of `0.028`, comparable to the full-batch fit at a fraction
of the cost.

## Code walkthrough

The initial prediction is loss-specific, and it matters:

```python
        elif self.task == "binary":
            self.classes_ = np.unique(y)
            yb = (y == self.classes_[1]).astype(float)
            p0 = float(np.clip(yb.mean(), 1e-6, 1.0 - 1e-6))
            F = np.full(n, np.log(p0 / (1.0 - p0)))
```

Starting the logistic model at zero log-odds means the first tree has to learn the
base rate; starting at `log(p/(1−p))` means it only learns the signal. For
multiclass the analogue is the class log-prior, centred so the scores sum to zero;
for quantiles it is `quantile(y, α)`, because a constant model at the mean is the
wrong starting point for an extreme quantile.

Each stage fits a shallow regression tree to the pseudo-residual. Binary is the
simplest non-trivial case:

```python
            elif self.task == "binary":
                p = _sigmoid(F)
                yb = (y == self.classes_[1]).astype(float)
                tree = self._tree().fit(X[idx], yb[idx] - p[idx])
                self._newton_leaves(tree, X[idx], yb[idx], p[idx], 1.0)
                F += self.lr * tree.predict(X)
                self.estimators_.append(tree)
```

`yb − p` is the negative gradient, and `_newton_leaves` then rewrites every leaf
value. The tree's structure is still chosen by variance reduction on the
pseudo-residual, which is a second-order approximation to the logistic loss; the
leaf rewrite makes the *scale* exact for the current leaf contents.

The Newton rewrite is a single pass over the rows:

```python
        for xi, yi, pi in zip(X, y, p, strict=True):
            leaf = tree._walk(xi)
            key = id(leaf)
            nodes[key] = leaf
            num[key] = num.get(key, 0.0) + (yi - pi)
            den[key] = den.get(key, 0.0) + pi * (1.0 - pi)
        for key, leaf in nodes.items():
            leaf["value"] = factor * num[key] / (den[key] + 1e-12)
```

`id(leaf)` is safe because the leaf dicts live as long as the tree and are not
copied; it is a compact way to group rows by leaf without storing leaf indices.
`den + 1e-12` is not cosmetic: a leaf whose rows all have `p` exactly 0 or 1 has
zero curvature, and the ratio would be `0/0`. The `factor` is `1` for binary and
`(K−1)/K` for multiclass.

Multiclass fits `K` trees per stage, all against the *same* `P`:

```python
                P = _softmax(F)
                trees = []
                for k in range(self.classes_.size):
                    resid = Y[idx, k] - P[idx, k]
                    tree = self._tree().fit(X[idx], resid)
                    self._newton_leaves(tree, X[idx], Y[idx, k], P[idx, k],
                                        (self.classes_.size - 1.0) / self.classes_.size)
                    F[:, k] += self.lr * tree.predict(X)
                    trees.append(tree)
```

`P` is computed once before the loop; recomputing it after updating `F[:, 0]` would
make the second class's residual depend on the first class's tree, turning a
simultaneous update into a sequential one. The `K` trees are stored as a list per
stage, which is the shape `_raw` expects.

Quantile boosting changes the pseudo-residual and the leaf value:

```python
            elif self.task == "quantile":
                resid = y - F
                pseudo = np.where(resid > 0.0, self.alpha, self.alpha - 1.0)
                tree = self._tree().fit(X[idx], pseudo[idx])
                self._quantile_leaves(tree, X[idx], resid[idx])
                F += self.lr * tree.predict(X)
```

`pseudo` takes only two values, so the tree is really learning where the residual
is positive and where it is negative. The magnitude of the step comes from the
residual quantile in each leaf. Passing `pseudo` to `_quantile_leaves` instead of
`resid` is the natural mistake: it would set every leaf to `α` or `α−1`, a constant
step that never converges to the target quantile.

Subsampling is one branch inside `_sample`:

```python
    def _sample(self, n: int) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n)
        k = max(1, int(self.subsample * n))
        return self.rng.choice(n, size=k, replace=False)
```

Without replacement is the usual choice for boosting; the forest samples *with*
replacement because it is bagging. `max(1, ...)` keeps a tiny fraction from
producing an empty sample and a crashed tree. The `n_rows_used_` list records the
draw so the demo can assert that the subsample actually happened.

Prediction replays the stages:

```python
        base = self._base_binary if self.task == "binary" else self._base_value
        F = np.full(X.shape[0], base)
        for tree in self.estimators_:
            F += self.lr * tree.predict(X)
        return F
```

The base is stored at fit time rather than recomputed from `y`, because at
prediction time there is no `y`. For binary, `_raw` returns log-odds and
`predict_proba` applies the stable sigmoid; for multiclass it returns the score
matrix and `predict_proba` applies softmax. `predict` thresholds at `0.5` for
binary and takes the argmax for multiclass, which is a decision rule separate from
the probabilities.

## Complexity

| Quantity | Cost |
|:--|:--|
| One stage, regression/binary | `O(n log n · d · depth)` with this split search |
| One stage, multiclass | `K ×` the above |
| `B` stages | `O(B n log n · d · depth)` (or `K×`) |
| Prediction, one row | `O(B · depth)` (or `B·K·depth`) |
| Memory | `O(B · nodes)` trees |

Boosting is sequential, so the trees cannot be fitted in parallel; the parallelism
is inside a tree's split search. `subsample` reduces the per-stage `n` and is the
main lever for large datasets. The demo's 300 stages of depth-3 trees on 350 rows
run in well under a second, which is the right order: boosting is cheap per stage
and expensive only in the number of stages.

## Failure modes

1. **Learning rate and stage count are one hyperparameter.** A small `η` without
   enough stages underfits; a large `η` overfits in a few dozen trees. Tune them
   together on a validation curve, and use early stopping (not implemented here)
   rather than a fixed `n_estimators`.
2. **Newton leaves with saturated probabilities.** When `p` is exactly 0 or 1, the
   denominator `p(1−p)` vanishes and the step is huge. The `1e-12` floor keeps it
   finite, but clipping logits or adding an L2 term on the leaf value is the real
   fix.
3. **Class imbalance.** The logistic loss is dominated by the majority class, and
   the base rate makes the first stages predict the prior. Reweight the loss or
   subsample; `balanced_subsample` in the forest has no direct analogue here.
4. **Quantile crossing.** Fitting `α=0.1, 0.5, 0.9` independently can produce
   `q̂₀.₁ > q̂₀.₅` at some `x`. Sort the predictions or fit a joint model if the
   ordering is a hard requirement.
5. **Extrapolation.** Trees are constant outside the training range, so boosting
   cannot continue a trend. Difference the series or add a linear base learner.
6. **Leakage through the pseudo-residual.** Every stage uses the *current* model's
   predictions on the same rows; fitting and evaluating on the same data makes the
   training loss look excellent while the model memorises. Use a validation set for
   the stage count.
7. **Multiclass `K` trees per stage.** Memory and time scale with `K`; for 1,000
   classes this is `1,000×` the binary cost. Use a softmax with a shared tree or a
   different decomposition.
8. **Feature scale.** Trees are invariant to monotone transforms, so scaling is
   unnecessary — but the *sparsity pattern* matters: a feature with thousands of
   distinct values gets more candidate thresholds and can dominate the split
   search. Bin the feature or limit `max_features`.

## Applications

- **Search ranking and recommendation.** LambdaMART is gradient boosting with a
  ranking loss; the same pseudo-residual/Newton-leaf template applies.
- **Credit risk and fraud.** Binary log loss with monotone constraints and
  calibrated probabilities; boosted trees are the standard tabular model in both
  domains.
- **Insurance and demand forecasting.** Quantile boosting gives prediction
  intervals and handles heteroscedasticity without assuming a distribution;
  Tweedie boosting (not in this file) is the claims counterpart.
- **Click-through rate.** Logistic boosting on sparse categorical features, with
  shrinkage and subsampling to keep the model stable at billions of rows.
- **Kaggle and AutoML.** XGBoost and LightGBM are this atom with better split
  search, regularisation and systems work; understanding the base loop is what
  makes their knobs interpretable.
- **As a bias-reduction layer.** Boosting a shallow tree on the residuals of a
  linear model is a simple way to add nonlinearity without abandoning
  interpretability.

## References

- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, 2001,
  Annals of Statistics — the canonical paper, including the Newton leaf values.
- Friedman, *Stochastic Gradient Boosting*, 2002, Computational Statistics & Data
  Analysis — row subsampling.
- Freund & Schapire, *A Decision-Theoretic Generalization of On-Line Learning and
  an Application to Boosting*, 1997 — the AdaBoost ancestor.
- Bühlmann & Yu, *Boosting with the L2 Loss: Regression and Classification*, 2003,
  JASA — the statistical view of shrinkage.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  2009, ch. 10 — the functional-gradient derivation and bias-variance analysis.
- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, 2016, KDD — the
  regularised objective that the next atom implements.
