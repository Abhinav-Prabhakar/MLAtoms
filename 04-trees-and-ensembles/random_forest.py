"""Random forest — many decorrelated trees, averaged.

Chapter:   04-trees-and-ensembles
Companion: random_forest.md
Tier 1 · 6 variants: classification, regression, extra_trees, oob,
permutation_importance, balanced_subsample

A single CART tree has low bias and high variance: perturbing the data can change
the root split and therefore the whole tree. Bagging fixes this by averaging many
trees fitted to bootstrap resamples. The forest adds one more ingredient — each
split considers only a random subset of features — which decorrelates the trees so
that averaging actually reduces variance instead of just reproducing it.

    variance of the average = rho * sigma^2 + (1 - rho) * sigma^2 / B

With `rho = 1` (identical trees) more trees buy nothing; random feature subsets push
`rho` down, which is why the forest beats bagged CART.

The variants here are the two tasks, one more-random split rule (extra trees), the
out-of-bag estimate, permutation importance, and class-balanced subsampling.
"""

from __future__ import annotations

import numpy as np
from cart import CART

VARIANTS = frozenset(
    ["classification", "regression", "extra_trees", "oob",
     "permutation_importance", "balanced_subsample"]
)


class RandomForest:
    """Bagged CART with random feature subsets (the forest).

    n_estimators trees are fitted; predictions average probabilities (classification)
    or values (regression). max_features defaults to "sqrt" for classification and
    1.0 (all features) for regression, as in Breiman's paper.
    """

    def __init__(
        self,
        variant: str = "classification",
        *,
        task: str | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: int | str | float | None = None,
        bootstrap: bool | None = None,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.task = task or ("regression" if variant == "regression" else "classification")
        if self.task not in ("classification", "regression"):
            raise ValueError(f"unknown task {self.task!r}")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = (
            ("sqrt" if self.task == "classification" else 1.0)
            if max_features is None else max_features
        )
        self.bootstrap = (
            variant != "extra_trees" if bootstrap is None else bootstrap
        )
        if variant == "oob":
            self.bootstrap = True
        self.rng = np.random.default_rng(random_state)
        self.trees_: list[CART] = []
        self.inbags_: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _indices(self, y: np.ndarray) -> np.ndarray:
        n = y.size
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

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = y.size
        self.classes_ = np.unique(y) if self.task == "classification" else None
        self.n_features_ = X.shape[1]
        self.trees_, self.inbags_ = [], []
        for _ in range(self.n_estimators):
            idx = self._indices(y)
            inbag = np.zeros(n, dtype=bool)
            inbag[idx] = True
            tree = CART(
                "gini" if self.task == "classification" else "variance",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                splitter="random" if self.variant == "extra_trees" else "best",
                random_state=int(self.rng.integers(0, 2**31 - 1)),
            ).fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.inbags_.append(inbag)
        self._oob(X, y)
        self.feature_importances_ = self._mdi()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is classification only")
        return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.task == "regression":
            return np.mean([t.predict(X) for t in self.trees_], axis=0)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        if self.task == "classification":
            return float(np.mean(self.predict(X) == y))
        pred = self.predict(X)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    # ------------------------------------------------------------------ #
    # Out-of-bag estimate
    # ------------------------------------------------------------------ #
    def _oob(self, X: np.ndarray, y: np.ndarray) -> None:
        n = y.size
        counts = np.zeros(n)
        sums = (np.zeros((n, self.classes_.size))
                if self.task == "classification" else np.zeros(n))
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
        if not seen.any():
            self.oob_score_ = float("nan")
            self.oob_prediction_ = None
            return
        if self.task == "classification":
            proba = sums[seen] / counts[seen, None]
            self.oob_decision_function_ = proba
            self.oob_prediction_ = self.classes_[np.argmax(proba, axis=1)]
            self.oob_score_ = float(np.mean(self.oob_prediction_ == y[seen]))
        else:
            self.oob_prediction_ = sums[seen] / counts[seen]
            ss_res = float(((y[seen] - self.oob_prediction_) ** 2).sum())
            ss_tot = float(((y[seen] - y[seen].mean()) ** 2).sum())
            self.oob_score_ = 1.0 - ss_res / max(ss_tot, 1e-12)

    # ------------------------------------------------------------------ #
    # Importances
    # ------------------------------------------------------------------ #
    def _mdi(self) -> np.ndarray:
        imp = np.zeros(self.n_features_)

        def walk(node, total):
            if node["leaf"]:
                return
            imp[node["feature"]] += node["n"] / total * node["gain"]
            walk(node["left"], total)
            walk(node["right"], total)

        for tree in self.trees_:
            walk(tree.root, tree.n_total)
        s = imp.sum()
        return imp / s if s > 0 else imp

    def permutation_importance(
        self, X: np.ndarray, y: np.ndarray, *, n_repeats: int = 5
    ) -> dict:
        """Drop in score when one column is shuffled, averaged over repeats."""
        X = np.asarray(X, dtype=float)
        base = self.score(X, y)
        mean = np.zeros(self.n_features_)
        std = np.zeros(self.n_features_)
        for j in range(self.n_features_):
            drops = np.empty(n_repeats)
            for r in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = self.rng.permutation(Xp[:, j])
                drops[r] = base - self.score(Xp, y)
            mean[j] = drops.mean()
            std[j] = drops.std()
        return {"mean": mean, "std": std, "baseline": base}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)
    n = 300

    # --- classification: a noisy circle, not linearly separable ---
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1.0).astype(int)
    y[rng.random(n) < 0.08] ^= 1
    Xt = rng.uniform(-1.5, 1.5, size=(500, 2))
    yt = ((Xt[:, 0] ** 2 + Xt[:, 1] ** 2) > 1.0).astype(int)
    rf = RandomForest("classification", n_estimators=60, min_samples_leaf=2,
                      random_state=1).fit(X, y)
    single = CART("gini", min_samples_leaf=2).fit(X, y)
    acc = np.mean(rf.predict(Xt) == yt)
    acc1 = np.mean(single.predict(Xt) == yt)
    assert acc > 0.9 and acc > acc1, (acc, acc1)
    print(f"forest: test acc {acc:.3f} vs one tree {acc1:.3f}")

    # --- regression: sin + quadratic, forest beats a single tree ---
    def f(x):
        return np.sin(1.5 * x[:, 0]) + 0.5 * x[:, 1] ** 2

    Xr = rng.uniform(-2.0, 2.0, size=(n, 2))
    yr = f(Xr) + rng.normal(0.0, 0.15, size=n)
    Xrt = rng.uniform(-2.0, 2.0, size=(500, 2))
    yrt = f(Xrt) + rng.normal(0.0, 0.15, size=500)
    rfr = RandomForest("regression", n_estimators=80, min_samples_leaf=3,
                       random_state=2).fit(Xr, yr)
    r2 = rfr.score(Xrt, yrt)
    assert r2 > 0.8, r2
    print(f"forest: regression R^2 {r2:.3f}, OOB R^2 {rfr.oob_score_:.3f}")

    # --- extra trees: random thresholds decorrelate further ---
    et = RandomForest("extra_trees", n_estimators=60, min_samples_leaf=2,
                      random_state=3).fit(X, y)
    acc_et = np.mean(et.predict(Xt) == yt)
    assert acc_et > 0.88, acc_et
    print(f"forest: extra_trees test acc {acc_et:.3f}")

    # --- OOB is a free validation estimate ---
    assert np.isfinite(rf.oob_score_)
    gap = abs(rf.oob_score_ - acc)
    assert gap < 0.12, gap
    print(f"forest: OOB {rf.oob_score_:.3f} vs test {acc:.3f} (gap {gap:.3f})")

    # --- permutation importance separates signal from noise ---
    Xi = rng.normal(size=(n, 6))
    yi = (Xi[:, 0] + Xi[:, 1] > 0).astype(int) ^ (Xi[:, 2] > 0).astype(int)
    rf_i = RandomForest("classification", n_estimators=60, random_state=4).fit(Xi, yi)
    pi = rf_i.permutation_importance(Xi, yi, n_repeats=5)["mean"]
    signal, noise = pi[:3], pi[3:]
    assert signal.min() > noise.max(), (signal, noise)
    assert np.isclose(rf_i.feature_importances_.sum(), 1.0)
    print(f"forest: permutation importance signal {np.round(signal, 3)}, "
          f"noise {np.round(noise, 3)}")

    # --- balanced subsampling helps a rare class ---
    nb = 300
    Xb = np.column_stack([
        rng.normal(0.0, 1.0, size=nb),
        rng.normal(0.0, 1.0, size=nb),
    ])
    yb = np.zeros(nb, dtype=int)
    minority = rng.choice(nb, size=15, replace=False)
    yb[minority] = 1
    Xb[minority, 0] += 2.5
    Xbt = np.column_stack([
        rng.normal(0.0, 1.0, size=600),
        rng.normal(0.0, 1.0, size=600),
    ])
    ybt = np.zeros(600, dtype=int)
    minority_t = rng.choice(600, size=30, replace=False)
    ybt[minority_t] = 1
    Xbt[minority_t, 0] += 2.5

    def balanced_acc(pred, truth):
        recalls = [np.mean(pred[truth == c] == c) for c in (0, 1)]
        return float(np.mean(recalls))

    plain = RandomForest("classification", n_estimators=80, random_state=5).fit(Xb, yb)
    bal = RandomForest("balanced_subsample", n_estimators=80, random_state=5).fit(Xb, yb)
    ba_plain = balanced_acc(plain.predict(Xbt), ybt)
    ba_bal = balanced_acc(bal.predict(Xbt), ybt)
    assert ba_bal > ba_plain, (ba_plain, ba_bal)
    print(f"forest: balanced acc plain {ba_plain:.3f} -> balanced_subsample {ba_bal:.3f}")

    print("random_forest demo passed")


if __name__ == "__main__":
    main()
