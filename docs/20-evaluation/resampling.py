"""Resampling — how to split data so the score means something.

Chapter:   20-evaluation
Companion: resampling.md
Tier 1 · 8 variants: holdout, kfold, stratified, group, time_series_split,
nested_cv, repeated_cv, leave_one_out

A held-out score is only as honest as the split that produced it. Every variant
here answers the same question — which rows may be used to fit, and which may be
used to score — under a different assumption about the data:

    holdout            one train/test cut; cheapest, highest variance
    kfold              every row is scored exactly once
    stratified         the class ratio is preserved in every fold
    group              no group (patient, user, batch) spans the cut
    time_series_split  train is always in the past, test in the future
    nested_cv          the inner folds choose hyperparameters, the outer ones score
    repeated_cv        repeat the k-fold partition to average away the split noise
    leave_one_out      k = n; unbiased-ish, high variance, closed form for OLS

The important ones are `group` and `time_series_split`: those are not variance
reductions, they are the difference between a valid estimate and a leaked one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

VARIANTS = frozenset(
    ["holdout", "kfold", "stratified", "group", "time_series_split",
     "nested_cv", "repeated_cv", "leave_one_out"]
)


class Resampling:
    """Split a dataset into fit/score index pairs.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    n_splits : int
        Number of folds (kfold family) or windows (time_series_split).
    test_size : float
        Fraction of rows in the test half of `holdout`.
    shuffle : bool
        Permute before folding; `time_series_split` ignores it.
    random_state : int
        Seed for every permutation.
    n_repeats : int
        Number of partitions produced by `repeated_cv`.
    gap : int
        Rows dropped between the train and test window of `time_series_split`.
    inner_splits : int
        Number of folds used inside each outer fold of `nested_cv`.
    stratify : bool
        Preserve the class ratio in `holdout`.

    Notes
    -----
    `split` yields `(train_idx, test_idx)`; concatenating every test index
    recovers `range(n)` exactly once for the k-fold family.
    """

    def __init__(
        self,
        variant: str = "kfold",
        n_splits: int = 5,
        *,
        test_size: float = 0.25,
        shuffle: bool = True,
        random_state: int = 0,
        n_repeats: int = 3,
        gap: int = 0,
        inner_splits: int = 4,
        stratify: bool = False,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if variant in ("kfold", "stratified", "group", "nested_cv") and n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        self.variant = variant
        self.n_splits = int(n_splits)
        self.test_size = float(test_size)
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)
        self.n_repeats = int(n_repeats)
        self.gap = int(gap)
        self.inner_splits = int(inner_splits)
        self.stratify = bool(stratify)
        self.inner: Resampling | None = (
            Resampling("kfold", inner_splits, random_state=random_state + 1)
            if variant == "nested_cv" else None
        )

    # ------------------------------------------------------------------ #
    # Splitters
    # ------------------------------------------------------------------ #
    def split(self, X: Any, y: np.ndarray | None = None,
              groups: np.ndarray | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(train_idx, test_idx)` integer arrays."""
        n = int(X.shape[0]) if hasattr(X, "shape") else int(X)
        rng = np.random.default_rng(self.random_state)
        v = self.variant

        if v == "holdout":
            yield from [self._holdout(n, y, rng)]
        elif v == "kfold":
            yield from self._kfold(n, self.n_splits, rng)
        elif v == "stratified":
            yield from self._stratified(n, self.n_splits, y, rng)
        elif v == "group":
            yield from self._group(n, self.n_splits, groups, rng)
        elif v == "time_series_split":
            yield from self._time_series(n)
        elif v == "nested_cv":
            yield from self._kfold(n, self.n_splits, rng)
        elif v == "repeated_cv":
            for r in range(self.n_repeats):
                sub = np.random.default_rng(self.random_state + r)
                yield from self._kfold(n, self.n_splits, sub)
        else:                                   # leave_one_out
            for i in range(n):
                yield np.delete(np.arange(n), i), np.array([i])

    def _holdout(self, n: int, y: np.ndarray | None,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n_test = max(1, int(round(self.test_size * n)))
        if not (self.stratify and y is not None):
            perm = rng.permutation(n) if self.shuffle else np.arange(n)
            return np.sort(perm[n_test:]), np.sort(perm[:n_test])
        test: list[int] = []
        for cls in np.unique(y):
            ids = np.flatnonzero(y == cls)
            take = max(1, int(round(self.test_size * ids.size)))
            test.extend(rng.permutation(ids)[:take].tolist())
        test_arr = np.sort(np.array(test, dtype=int))
        mask = np.ones(n, dtype=bool)
        mask[test_arr] = False
        return np.flatnonzero(mask), test_arr

    def _kfold(self, n: int, k: int,
               rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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

    def _stratified(self, n: int, k: int, y: np.ndarray | None,
                    rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("stratified splitting needs y")
        folds: list[list[int]] = [[] for _ in range(k)]
        for cls in np.unique(y):
            ids = np.flatnonzero(y == cls)
            ids = ids[rng.permutation(ids.size)]
            for j, chunk in enumerate(np.array_split(ids, k)):
                folds[j].extend(chunk.tolist())
        for fold in folds:
            test = np.sort(np.array(fold, dtype=int))
            mask = np.ones(n, dtype=bool)
            mask[test] = False
            yield np.flatnonzero(mask), test

    def _group(self, n: int, k: int, groups: np.ndarray | None,
               rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("group splitting needs groups")
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        uniq = uniq[rng.permutation(uniq.size)]
        load = np.zeros(k, dtype=int)
        assign: dict[Any, int] = {}
        for g in uniq:                       # longest-processing-time: biggest first
            j = int(np.argmin(load))
            assign[g] = j
            load[j] += int((groups == g).sum())
        fold_of = np.array([assign[g] for g in groups])
        for j in range(k):
            test = np.flatnonzero(fold_of == j)
            yield np.flatnonzero(fold_of != j), test

    def _time_series(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        width = max(1, n // (self.n_splits + 1))
        for i in range(self.n_splits):
            train_end = width * (i + 1)
            test_start = train_end + self.gap
            test_end = min(test_start + width, n)
            if test_start >= n:
                break
            yield np.arange(train_end), np.arange(test_start, test_end)


# --------------------------------------------------------------------------- #
# Scoring harness
# --------------------------------------------------------------------------- #
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error — a loss, so lower is better."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - accuracy — same orientation as `mse`."""
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def cross_val_score(
    make_model: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    resampling: Resampling,
    scoring: Callable[[np.ndarray, np.ndarray], float] = mse,
) -> np.ndarray:
    """One score per test fold, lower is better."""
    scores = []
    for train, test in resampling.split(X, y):
        model = make_model()
        model.fit(X[train], y[train])
        scores.append(scoring(y[test], model.predict(X[test])))
    return np.asarray(scores, dtype=float)


def nested_cross_val_score(
    make_model: Callable[..., Any],
    X: np.ndarray,
    y: np.ndarray,
    resampling: Resampling,
    candidates: Sequence[dict[str, Any]],
    scoring: Callable[[np.ndarray, np.ndarray], float] = mse,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select on the inner folds, score on the outer ones.

    Returns the outer scores and the selected hyperparameters per outer fold.
    """
    if resampling.variant != "nested_cv" or resampling.inner is None:
        raise ValueError("pass a Resampling('nested_cv')")
    outer: list[float] = []
    chosen: list[dict[str, Any]] = []
    for train, test in resampling.split(X, y):
        Xtr, ytr = X[train], y[train]
        best: tuple[float, dict[str, Any]] | None = None
        for params in candidates:
            inner_scores = []
            for itr, ite in resampling.inner.split(Xtr, ytr):
                model = make_model(**params)
                model.fit(Xtr[itr], ytr[itr])
                inner_scores.append(scoring(ytr[ite], model.predict(Xtr[ite])))
            mean = float(np.mean(inner_scores))
            if best is None or mean < best[0]:
                best = (mean, dict(params))
        assert best is not None
        model = make_model(**best[1])
        model.fit(Xtr, ytr)
        outer.append(scoring(y[test], model.predict(X[test])))
        chosen.append(best[1])
    return np.asarray(outer, dtype=float), chosen


# --------------------------------------------------------------------------- #
# Two tiny estimators, so the demo scores something real
# --------------------------------------------------------------------------- #
class Ridge:
    """Closed-form ridge regression with an intercept.

    `alpha = 0` is OLS; the intercept is never penalised because the columns
    are centred first.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ridge:
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = float(np.mean(y))
        Xc = X - self.x_mean_
        yc = y - self.y_mean_
        d = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(d)
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean_) @ self.coef_ + self.y_mean_


class NearestCentroid:
    """Assign each row to the closest class mean — enough to score a split."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> NearestCentroid:
        self.classes_ = np.unique(y)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.means_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(d2, axis=1)]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
class _SubOLS:
    """OLS on a chosen pair of columns — the demo's interchangeable candidates."""

    def __init__(self, cols: tuple[int, int] = (0, 1)):
        self.cols = tuple(cols)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SubOLS:
        self.model_ = Ridge(alpha=0.0).fit(X[:, self.cols], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X[:, self.cols])


def main() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, -0.5, 2.0, 0.0]) + rng.normal(scale=0.5, size=n)

    # --- holdout: disjoint halves, reproducible, stratifiable ---
    r = Resampling("holdout", test_size=0.3, random_state=0)
    tr, te = next(iter(r.split(X)))
    assert tr.size + te.size == n
    assert np.intersect1d(tr, te).size == 0
    tr2, te2 = next(iter(Resampling("holdout", test_size=0.3, random_state=0).split(X)))
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
    y_cls = (y > np.median(y)).astype(int)
    s = Resampling("holdout", test_size=0.3, random_state=0, stratify=True)
    _, te_s = next(iter(s.split(X, y_cls)))
    gap = abs(y_cls[te_s].mean() - y_cls.mean())
    assert gap < 0.05, gap
    print(f"resampling: holdout {tr.size}/{te.size} rows, stratified test "
          f"balance {y_cls[te_s].mean():.3f} vs {y_cls.mean():.3f} overall")

    # --- kfold: every row is tested exactly once, folds differ by at most one ---
    folds = list(Resampling("kfold", 5, random_state=0).split(X))
    stacks = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(stacks), np.arange(n))
    sizes = np.array([te.size for _, te in folds])
    assert sizes.max() - sizes.min() <= 1, sizes
    assert all(np.intersect1d(tr, te).size == 0 for tr, te in folds)
    print(f"resampling: kfold 5 folds of {sizes.tolist()}, every row scored once")

    # --- stratified vs kfold on 190/10 class split stored in class order ---
    y_rare = np.r_[np.zeros(190, dtype=int), np.ones(10, dtype=int)]
    strat = list(Resampling("stratified", 5, random_state=0).split(X, y_rare))
    plain = list(Resampling("kfold", 5, shuffle=False).split(X, y_rare))
    rare_strat = np.array([y_rare[te].sum() for _, te in strat])
    rare_plain = np.array([y_rare[te].sum() for _, te in plain])
    assert rare_strat.tolist() == [2, 2, 2, 2, 2], rare_strat
    assert rare_plain.tolist() == [0, 0, 0, 0, 10], rare_plain
    print(f"resampling: stratified rare-class per fold {rare_strat.tolist()} vs "
          f"kfold {rare_plain.tolist()} on class-ordered rows")

    # --- group: patients never split across folds, and kfold leaks them ---
    groups = np.repeat(np.arange(40), 5)
    g_folds = list(Resampling("group", 5, random_state=0).split(X, groups=groups))
    leaks = 0
    for tr, te in g_folds:
        leaks += int(np.intersect1d(groups[tr], groups[te]).size)
    assert leaks == 0, leaks
    k_folds = list(Resampling("kfold", 5, random_state=0).split(X))
    k_leaks = sum(int(np.intersect1d(groups[tr], groups[te]).size)
                  for tr, te in k_folds)
    assert k_leaks > 0
    print(f"resampling: group 5 folds, {leaks} shared groups; kfold shares "
          f"{k_leaks} groups across cuts")

    # --- time_series_split: train strictly before test, expanding window ---
    ts = list(Resampling("time_series_split", 4, random_state=0).split(n))
    widths = []
    for tr, te in ts:
        assert tr.max() < te.min(), (tr.max(), te.min())
        widths.append((tr.size, te.size))
    assert all(widths[i][0] < widths[i + 1][0] for i in range(len(widths) - 1))
    gap_folds = list(Resampling("time_series_split", 3, gap=5).split(n))
    assert all(te.min() - tr.max() - 1 == 5 for tr, te in gap_folds)
    print(f"resampling: time series windows {widths}, gap 5 respected")

    # --- nested CV: choosing on the same folds you score on is optimistic ---
    noise = np.random.default_rng(0)
    Xn = noise.normal(size=(100, 20))
    yn = noise.normal(size=100)        # independent of X: every candidate risks 1.0
    subsets = [{"cols": (2 * j, 2 * j + 1)} for j in range(10)]
    flat = np.array([cross_val_score(lambda p=p: _SubOLS(**p), Xn, yn,
                                     Resampling("kfold", 5, random_state=0)).mean()
                     for p in subsets])
    nested, picked = nested_cross_val_score(
        _SubOLS, Xn, yn, Resampling("nested_cv", 5, random_state=0), subsets)
    assert flat.min() < nested.mean() - 0.05, (flat.min(), nested.mean())
    distinct = len({p["cols"] for p in picked})
    assert distinct > 1, distinct
    print(f"resampling: flat CV best of 10 subsets {flat.min():.3f} vs nested "
          f"{nested.mean():.3f} (true risk 1.0), {distinct} distinct subsets chosen")

    # --- repeated CV: averaging partitions shrinks the seed-to-seed spread ---
    def once(seed: int, repeats: int) -> float:
        scores = []
        rr = Resampling("repeated_cv", 5, random_state=seed, n_repeats=repeats)
        for train, test in rr.split(X):
            model = Ridge(alpha=1.0).fit(X[train], y[train])
            scores.append(mse(y[test], model.predict(X[test])))
        return float(np.mean(scores))

    single_runs = np.array([once(s, 1) for s in range(20)])
    repeated_runs = np.array([once(5000 + 10 * s, 10) for s in range(20)])
    assert np.std(repeated_runs) < 0.5 * np.std(single_runs), (
        np.std(repeated_runs), np.std(single_runs))
    print(f"resampling: seed-to-seed spread {np.std(single_runs):.4f} (1 repeat) "
          f"vs {np.std(repeated_runs):.4f} (10 repeats)")

    # --- leave-one-out matches the hat-matrix shortcut for OLS ---
    Xa = np.column_stack([np.ones(n), X])
    H = Xa @ np.linalg.pinv(Xa.T @ Xa) @ Xa.T
    resid = y - H @ y
    loo_closed = float(np.mean((resid / (1.0 - np.diag(H))) ** 2))
    loo = cross_val_score(lambda: Ridge(alpha=0.0), X, y,
                          Resampling("leave_one_out"))
    assert loo.size == n
    assert abs(loo.mean() - loo_closed) < 1e-8, (loo.mean(), loo_closed)
    print(f"resampling: leave_one_out {loo.mean():.6f} == hat-matrix "
          f"{loo_closed:.6f}")

    print("resampling demo passed")


if __name__ == "__main__":
    main()
