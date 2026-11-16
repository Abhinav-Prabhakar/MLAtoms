# Resampling

`resampling.py` · Evaluation · Tier 1 · 8 variants

## Intuition

A model's score on the data it was trained on is a lie. It has already seen those rows,
so the number measures memory, not generalisation. Resampling is the machinery that
manufactures the missing evidence: hold part of the data back, fit on the rest, score on
the held-out part, and repeat with different parts.

The variants are not different algorithms so much as different answers to *what is
allowed to leak between the fit and the score*:

- **`holdout`** — one cut. Cheapest, noisier than it looks: with `n = 200` and a 30% test
  set, the standard error of an MSE estimate is large enough that two models differing by
  10% can look identical.
- **`kfold`** — cut into `k` pieces and score each piece once with the other `k−1` used to
  fit. Every row is a test row exactly once, so no data is wasted and the estimate uses
  the whole sample.
- **`stratified`** — keep the class ratio inside every fold. An unstratified 5-fold split
  of 190 negatives and 10 positives can put all ten positives in one fold (the demo does
  exactly this by keeping the rows in class order: `[0, 0, 0, 0, 10]`), which makes four folds
  unable to measure the minority class at all.
- **`group`** — no patient, user, device or document may appear on both sides of the cut.
  Otherwise the model memorises the group and the fold score measures recall of an
  identity, not skill.
- **`time_series_split`** — the training set is always in the past. A random split of a
  time series trains on the future and tests on the past, which is not a question anyone
  can ask in production.
- **`nested_cv`** — the inner folds choose hyperparameters, the outer folds score the
  whole procedure. Without the nesting, the selection step sees the scoring data and the
  reported number is optimistic.
- **`repeated_cv`** — repeat the whole `k`-fold partition with different shuffles and
  average. The point estimate barely moves; the *variance* of the estimate drops like
  `1/R`, which is what lets you tell a real 1% difference from a lucky split.
- **`leave_one_out`** — `k = n`. Nearly unbiased for the training-set size, but each fit
  differs from the final model by one point, so for unstable learners the estimate has
  high variance. For linear least squares it is free: one closed-form formula replaces
  `n` fits.

The two that stop silent errors are `group` and `time_series_split`. The rest reduce
variance; those two decide whether the number is measuring anything at all.

## Variants

| Variant | Assumption it protects | Cost | Estimate variance |
|:--|:--|:--|:--|
| `holdout` | nothing | 1 fit | high |
| `kfold` | rows are i.i.d. | `k` fits | moderate |
| `stratified` | rows are i.i.d., classes imbalanced | `k` fits | moderate, lower on rare classes |
| `group` | rows are i.i.d. *within* group | `k` fits | moderate, correct under dependence |
| `time_series_split` | train precedes test | `n_splits` fits | moderate, pessimistic |
| `nested_cv` | hyperparameters are chosen, not given | `k·(1 + c·k_inner)` fits | low, unbiased for the procedure |
| `repeated_cv` | the partition is random | `k·R` fits | `≈ σ²/(kR)` |
| `leave_one_out` | rows are i.i.d. | `n` fits (`O(1)` for OLS) | low bias, high variance |

`holdout` and `kfold` are the same idea at two granularities: holdout is `k = 2` with an
unequal split. `kfold` with `k = n` is `leave_one_out`. The interesting cases are the
conditioning ones — `stratified`, `group`, `time_series_split` — and `nested_cv`, which
fixes a bias rather than a variance.

One detail that catches people: `nested_cv` and `repeated_cv` reuse the machinery of
`kfold`. In this file `repeated_cv` is literally `n_repeats` k-fold partitions with
different seeds, and `nested_cv` is a k-fold outer loop plus an `inner` k-fold partitioner
exposed as an attribute. That is not a shortcut; it is the definition.

## Derivation

**The estimand.** Let `R(θ) = E_{(x,y)~P}[ℓ(y, f_θ(x))]` be the risk under the data
distribution, and let `R̂_S(θ)` be the average loss on a finite sample `S`. Every
resampling scheme is an attempt to estimate `R(θ)` using only one sample `D` of size `n`.

**Holdout.** Split `D = T ∪ V` with `|V| = m`. Then `R̂_V(θ_T)` is unbiased for the risk of
a model trained on `n − m` points, and its variance is `Var(ℓ)/m`, estimated by the
empirical variance of the test losses. Two problems are visible in that sentence: the
training set is smaller than the final one (bias), and with `m = 0.3n` the variance is
large. The bias is `O(1/n)` for a stable learner but can be substantial when the learning
curve is steep.

**k-fold.** Partition into `K` folds `F_1…F_K` of size `≈ n/K`; for each `j`, fit on
`D \ F_j` and score on `F_j`. Every row is scored exactly once, so the estimate uses all
`n` scores:

$$
\hat R_{\text{cv}} = \frac{1}{n}\sum_{i=1}^{n} \ell\bigl(y_i, f_{\theta_{D\setminus F(i)}}(x_i)\bigr)
$$

with `F(i)` the fold containing `i`. The bias is that of a model trained on `n(1−1/K)`
points, which shrinks as `K` grows; the variance comes from two sources — the test losses
themselves and the correlation between the `K` fitted models, which share `n(1−2/K)`
points. That correlation is why `k = 10` usually beats both `k = 5` and `k = n`: it trades
bias down and correlation up, and the optimum is around 5–10 for most learners.

**Stratification.** Suppose the folds are formed by assigning each class's rows to folds
independently. Then the number of minority rows in fold `j` is hypergeometric with mean
`n_rare/K` and variance `n_rare (n−n_rare)/(K−1) · …`, so a fold can receive zero minority
rows whenever `n_rare < K`. Stratifying makes that count constant (`n_rare // K`, plus one
for the first `n_rare % K` folds), which removes the between-fold variance of the class
ratio entirely. The demo's 190/10 split gives `[2,2,2,2,2]` stratified and `[0,0,0,0,10]`
unstratified — the first four folds of the unstratified split cannot compute a
minority-class recall at all.

**Groups.** If rows are exchangeable only within a group `g` — repeated measurements of
the same patient, multiple sessions of the same user — then a row from `g` in the test
set is predictable from its siblings in the training set regardless of the target
relationship. The fold score then estimates "can the model identify the group", which is
usually near-perfect and always useless. Grouped splitting makes the folds a partition of
*groups*, so the test rows come from groups the model has never seen. The demo measures
the leak directly: k-fold shares `133` groups between the training and test sides, grouped
splitting shares `0`.

**Time series.** For a series `y_1…y_n`, the random-fold assumption fails twice: the rows
are dependent (so test losses are correlated and the variance formula is wrong), and a
random split lets the model fit on `y_{t+1}` to predict `y_t`. Forward chaining fixes
both: window `i` trains on `[0, w(i+1))` and tests on the following block, so the training
set only ever grows into the past. The `gap` parameter drops rows immediately before the
test window — the forecast horizon, an embargo, or the label-delay window. Without it, the
last training label may be computed from data that overlaps the test period, which is a
leak that survives even forward chaining.

**Nested CV.** Let `c` index a candidate set and `S_j` be the outer fold. The flat
procedure is

$$
\hat R_{\text{flat}} = \min_c \frac{1}{K}\sum_j \ell_{j}(c)
$$

and `min` is a nonlinear function, so it does not commute with expectation:

$$
\mathbb{E}\Bigl[\min_c \hat R(c)\Bigr] \;\le\; \min_c \mathbb{E}\bigl[\hat R(c)\bigr]
$$

The left side is what flat CV reports; the right side is the risk of the best candidate.
The gap is the selection bias, and it grows with the number of candidates and with the
noise in each estimate. Nested CV reports instead

$$
\hat R_{\text{nested}} = \frac{1}{K}\sum_j \ell_{j}\bigl(\hat c_{-j}\bigr),
\qquad
\hat c_{-j} = \arg\min_c \frac{1}{K_{\text{in}}}\sum_{i \in \text{inner}} \ell_i(c)
$$

where `ĉ_{−j}` is chosen without touching `S_j`. Each term is an honest score for the
*procedure* that picks `c` from data, which is the thing that will run in production.

The demo makes the bias visible with ten candidates that are all equally useless: ten
disjoint pairs of pure-noise features, `y` independent of `X`. Every candidate has true
risk exactly `1.0`. Flat CV's best of ten reports `0.984`; the nested estimate reports
`1.089`, and it selects `3` different subsets across the five outer folds — the selection
is unstable, which is precisely when the flat minimum is most optimistic.

**Repeated CV.** A single k-fold estimate is a random variable because the partition is
random. If a repeat has variance `σ²`, then averaging `R` independent repeats has variance
`σ²/R`, so the standard error falls by `√R`:

$$
\operatorname{se}\bigl(\hat R^{(R)}\bigr) = \frac{\sigma}{\sqrt{kR}}
$$

Two distinct 10-repeat estimates should therefore agree about three times more closely
than two distinct 1-repeat estimates. The demo measures that directly: over 20 seeds the
spread is `0.0039` for one repeat and `0.0011` for ten, a ratio of `0.29 ≈ 1/√10`.

**Leave-one-out and the hat matrix.** LOO is `K = n`: fit on all but row `i`, score row
`i`. Naively that is `n` fits, but for linear least squares there is an identity. Let
`β̂` be the full fit, `e_i = y_i − x_iᵀβ̂` the full residual, and
`h_i = x_iᵀ(XᵀX)⁻¹x_i` the leverage (the `i`-th diagonal of the hat matrix `H = X(XᵀX)⁻¹Xᵀ`).
Then

$$
\hat\beta_{-i} = \hat\beta - \frac{(X^\top X)^{-1}x_i\, e_i}{1-h_i},
\qquad
\hat y_{-i}(x_i) = x_i^\top\hat\beta_{-i}
\;\Longrightarrow\;
e_i^{(-i)} = \frac{e_i}{1-h_i}
$$

The second identity is the useful one: the LOO residual is the full residual inflated by
`1/(1−h_i)`. So

$$
\text{LOO-MSE} = \frac{1}{n}\sum_i \Bigl(\frac{e_i}{1-h_i}\Bigr)^2
$$

with no refitting at all. The demo checks the implementation against it and gets
`0.198063` from both routes to six decimals. The `h_i → 1` case is a high-leverage point:
the inflation factor blows up, and the LOO estimate becomes unstable, which is LOO's
known weakness rather than a bug.

## Code walkthrough

One `split` generator dispatches on the variant, so every consumer — the demo, the two
scoring harnesses, any caller — sees the same `(train_idx, test_idx)` contract:

```python
        if v == "holdout":
            yield from [self._holdout(n, y, rng)]
        elif v == "kfold":
            yield from self._kfold(n, self.n_splits, rng)
```

`yield from [... ]` with a one-element list is deliberate: `split` must always be a
generator, otherwise callers that call `next(iter(...))` and callers that loop would need
different code paths. `n` comes from the first axis of `X`, or from `X` itself when it is
an integer, so index-only callers (the time-series demo) do not have to fabricate an
array.

The k-fold split is a permutation plus a size vector:

```python
        perm = rng.permutation(n) if self.shuffle else np.arange(n)
        sizes = np.full(k, n // k)
        sizes[: n % k] += 1
        start = 0
        for size in sizes:
            test = np.sort(perm[start:start + size])
            mask = np.ones(n, dtype=bool)
            mask[test] = False
            yield np.flatnonzero(mask), test
            start += size
```

`sizes[: n % k] += 1` is what makes folds differ by at most one row: with `n = 203`,
`k = 5` the sizes are `[41, 41, 41, 40, 40]`. The mask-and-`flatnonzero` construction
gives sorted train indices without `np.setdiff1d`, which would sort on every fold and is
much slower. `np.sort(perm[...])` sorts the test indices too; callers index with them, so
order does not matter, but reproducible output does.

Stratification loops over classes and slices each one into `k` chunks:

```python
        folds: list[list[int]] = [[] for _ in range(k)]
        for cls in np.unique(y):
            ids = np.flatnonzero(y == cls)
            ids = ids[rng.permutation(ids.size)]
            for j, chunk in enumerate(np.array_split(ids, k)):
                folds[j].extend(chunk.tolist())
```

`np.array_split` splits into `k` nearly equal pieces, so a class with `m` rows puts
`⌈m/k⌉` or `⌊m/k⌋` in each fold — the constant-proportion guarantee. Shuffling per class
before the split is what makes it random rather than dependent on row order. Note the
loop cannot produce an empty fold for a class with fewer than `k` rows; it produces
folds with zero of that class, which is the unavoidable floor and exactly the case where
stratification has nothing left to give.

Grouped splitting is a greedy load-balance over whole groups:

```python
        load = np.zeros(k, dtype=int)
        assign: dict[Any, int] = {}
        for g in uniq:                       # longest-processing-time: biggest first
            j = int(np.argmin(load))
            assign[g] = j
            load[j] += int((groups == g).sum())
        fold_of = np.array([assign[g] for g in groups])
```

This is longest-processing-time scheduling: each group goes to the currently lightest
fold, which keeps fold sizes within one group's size of each other. Since the folds are a
partition of *groups*, no group can appear on both sides of a cut — that is the whole
guarantee, and it holds regardless of how balanced the folds are. `fold_of` is a
per-row array, so the yielded train/test masks are computed by equality rather than by
set operations.

Time-series windows are an expanding prefix and a following block:

```python
        width = max(1, n // (self.n_splits + 1))
        for i in range(self.n_splits):
            train_end = width * (i + 1)
            test_start = train_end + self.gap
            test_end = min(test_start + width, n)
            if test_start >= n:
                break
            yield np.arange(train_end), np.arange(test_start, test_end)
```

`n // (n_splits + 1)` reserves one window for the first training block plus one test
window per split, so the last test block ends at the data. `gap` is added between
`train_end` and `test_start` and is *not* part of either side — those rows are dropped,
which is the embargo. `np.arange(train_end)` is a prefix, so the training set grows by
`width` each iteration and the test block slides forward: exactly forward chaining.

The stratified holdout is the same idea at one cut, and its `max(1, ...)` guard is
subtle:

```python
        for cls in np.unique(y):
            ids = np.flatnonzero(y == cls)
            take = max(1, int(round(self.test_size * ids.size)))
            test.extend(rng.permutation(ids)[:take].tolist())
```

Without `max(1, ...)`, a class with two rows and `test_size = 0.25` contributes
`round(0.5) = 0` rows and disappears from the test set entirely. The guarantee that every
class is represented costs one row of imbalance — a class with one row ends up in the
test set regardless of `test_size`, which is why the demo's split has 60 test rows rather
than 59.

The scoring harness rebuilds the model for every fold:

```python
    scores = []
    for train, test in resampling.split(X, y):
        model = make_model()
        model.fit(X[train], y[train])
        scores.append(scoring(y[test], model.predict(X[test])))
    return np.asarray(scores, dtype=float)
```

`make_model()` must be a factory, not an instance: reusing a fitted estimator across
folds keeps the state from the previous fit, so fold 2 would start from fold 1's
coefficients. This is the most common bug in hand-rolled CV, and it is silent — the scores
still look plausible.

Nested CV is the outer loop plus an inner selection, and the inner partition is taken
from the *outer training rows*:

```python
            for itr, ite in resampling.inner.split(Xtr, ytr):
                model = make_model(**params)
                model.fit(Xtr[itr], ytr[itr])
                inner_scores.append(scoring(ytr[ite], model.predict(Xtr[ite])))
            mean = float(np.mean(inner_scores))
            if best is None or mean < best[0]:
                best = (mean, dict(params))
```

`Xtr[itr]` — not `X[itr]` — is the load-bearing line. Splitting the inner fold over the
full dataset would put outer-test rows into the inner training set, and the selection
would see the data it is later scored on. That single indexing mistake reintroduces the
exact bias nesting exists to remove. `dict(params)` copies the candidate so a later
mutation cannot change the record of what was chosen.

The LOO branch is two lines, and its cost profile is the reason the closed form matters:

```python
        else:                                   # leave_one_out
            for i in range(n):
                yield np.delete(np.arange(n), i), np.array([i])
```

`np.delete` allocates a new array of length `n−1` per iteration, so the generator alone is
`O(n²)` memory traffic before any fitting. For `n = 10⁴` that is 800 MB of copying; for
linear models use the hat-matrix formula instead, and for anything else use `k = 10` and
accept the bias.

## Complexity

| Variant | Fits | Total cost | Memory |
|:--|:--|:--|:--|
| `holdout` | 1 | `T(n(1−p))` | `O(n)` |
| `kfold` | `k` | `k·T(n(1−1/k))` | `O(n)` |
| `stratified` | `k` | `k·T(n(1−1/k))` + `O(n log n)` sort | `O(n)` |
| `group` | `k` | `k·T(·)` + `O(G log G)` for the group sort | `O(n + G)` |
| `time_series_split` | `s` | `Σ_i T(w(i+1))` — cheapest early, dearest late | `O(n)` |
| `nested_cv` | `k(1 + c·k_in)` | `k·T(·) + k·c·k_in·T(·)` | `O(n)` |
| `repeated_cv` | `kR` | `R`× k-fold | `O(n)` |
| `leave_one_out` | `n` | `n·T(n−1)`; `O(nd²)` for OLS via `H` | `O(n²)` naive, `O(n)` closed form |

`T(m)` is the cost of one fit on `m` rows. Nested CV is the expensive one: with `k = 5`,
`k_in = 4` and 10 candidates it is `5 + 200 = 205` fits, which is why people nest only the
few hyperparameters that matter. Repeated CV is embarrassingly parallel across repeats.
The LOO row is the interesting one: for OLS the closed form makes `n` apparent fits cost
one `d×d` inverse, so the "expensive" variant becomes the cheapest per unit of
information.

## Failure modes

1. **Leakage through preprocessing.** Scaling, imputing, feature selection or target
   encoding fitted on the full dataset before splitting leaks test information into
   training. Put every transformation inside the fold: fit on train, apply to test. This
   is the single most common way a CV score is spectacularly wrong.
2. **Grouped data split randomly.** The demo's k-fold shares `133` of `40` groups across
   cuts — every fold has seen siblings of its test rows. Scores look excellent and
   deployment collapses. Any column that identifies a repeated unit (patient, session,
   device) belongs in `groups`.
3. **Time leakage.** Random splits, `shuffle=True` on ordered data, or features computed
   over the full series (a rolling mean that includes future rows) all train on the
   future. Use `time_series_split` with a `gap` at least as large as the label horizon.
4. **Tuning and scoring on the same folds.** Flat CV's `min` over candidates is biased
   low by construction; the demo's best-of-ten reports `0.984` when the truth is `1.000`.
   The bias is largest exactly when the candidates are near-tied and the estimates are
   noisy — the situation where choosing matters most.
5. **`k` too small or too large.** `k = 2` wastes half the data per fit and has high
   variance; `k = n` makes the fits almost identical to each other (high correlation) and
   for unstable learners has *higher* variance than `k = 10`. Five or ten is the default
   for a reason.
6. **A single split reported as a result.** `holdout` with `test_size = 0.2` has a standard
   error several times larger than 10-fold on the same data. Report the spread across
   folds, not just the mean; a bare mean invites over-reading.
7. **Class imbalance without stratification.** As above: folds with zero minority rows
   make per-class metrics undefined or `nan`, and the mean over folds silently ignores
   them.
8. **LOO with high leverage.** `1/(1−h_i)` explodes for a point that nearly determines its
   own fit, so the LOO estimate is dominated by a few rows. If `max(h_i)` is close to 1,
   prefer `k`-fold.
9. **Reusing a fitted model across folds.** Passing an instance instead of a factory keeps
   warm-started state and quietly reports the score of a model trained on the union of
   folds. `make_model()` every fold.
10. **`n_splits` larger than the data.** `k > n` produces empty test folds and division by
    zero in the mean. The constructor rejects `k < 2`; it cannot know `n`, so guard at the
    call site.

## Applications

- **Model selection.** Compare candidate models on the same folds and report mean ± standard
  error across folds; with `repeated_cv` the error bar is small enough to resolve 1%
  differences.
- **Hyperparameter search.** `nested_cv` is the only honest way to report the score of a
  tuned model; flat CV is for ranking candidates, not for publishing a number.
- **Clinical and user-level prediction.** Grouped splitting is mandatory whenever a patient
  or user contributes multiple rows, and it is the difference between a paper that
  replicates and one that does not.
- **Forecasting.** Forward chaining is the standard backtest for time-series models, and
  the `gap` encodes the forecast horizon.
- **Rare-event detection.** Stratified folds keep fraud, failure and disease rates
  measurable in every fold, which is what per-class metrics need.
- **Cheap model diagnostics.** LOO via the hat matrix gives a per-row error estimate in
  one fit, useful for spotting influential points and for comparing linear models on small
  data.
- **Baseline establishment.** A plain 5-fold CV with a fixed seed is the minimum evidence
  any new model should beat before it is worth deploying.

## References

- Kohavi, *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model
  Selection*, 1995, IJCAI — the `k = 10` recommendation and the bias-variance comparison
  with holdout and bootstrap.
- Stone, *Cross-Validatory Choice and Assessment of Statistical Predictions*, 1974,
  JRSS-B — the original cross-validation formulation and LOO.
- Geisser, *The Predictive Sample Reuse Method with Applications*, 1975, JASA — LOO as an
  estimate of predictive risk.
- Varma & Simon, *Bias in Error Estimation When Using Cross-Validation for Model
  Selection*, 2006, BMC Bioinformatics — the nested-CV selection bias quantified.
- Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in
  Performance Evaluation*, 2010, JMLR — why the flat minimum is optimistic and how to nest.
- Bergmeir & Benítez, *On the Use of Cross-Validation for Time Series Predictor
  Evaluation*, 2012, Information Sciences — when blocked CV is valid and why random folds
  are not.
- Roberts et al., *Cross-Validation Strategies for Data with Temporal, Spatial,
  Hierarchical, or Phylogenetic Structure*, 2017, Ecography — grouped and blocked splitting
  under dependence.
- Cook, *Detection of Influential Observation in Linear Regression*, 1977,
  Technometrics — leverage and the `e_i/(1−h_i)` identity this file checks against.
