"""Gradient boosting — stagewise functional gradient descent.

Chapter:   04-trees-and-ensembles
Companion: gradient_boosting.md
Tier 1 · 7 variants: regression, binary, multiclass, shrinkage, subsample,
quantile, deviance

Boosting fits shallow trees one at a time, each to the negative gradient of the
loss with respect to the current prediction. The update is

    F_m(x) = F_{m-1}(x) + lr * h_m(x)

where h_m is a regression tree fitted to the pseudo-residuals. The learning rate
(`shrinkage`) and the row subsample are the two regularisers that make this work
in practice. Unlike a forest, boosting reduces *bias*; the trees are weak and
sequential on purpose.

The exact leaf values come from a one-step Newton update on the loss, not from the
tree's own mean. For squared loss the two coincide; for logistic and softmax they
do not, and using the mean is the classic mistake that makes boosting converge
slowly and miscalibrate.
"""

from __future__ import annotations

import numpy as np
from cart import CART

VARIANTS = frozenset(
    ["regression", "binary", "multiclass", "shrinkage", "subsample",
     "quantile", "deviance"]
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class GradientBoosting:
    """Gradient boosted trees for regression, binary, multiclass and quantiles."""

    def __init__(
        self,
        variant: str = "regression",
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        alpha: float = 0.5,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.alpha = alpha
        self.rng = np.random.default_rng(random_state)
        self.task = (
            "binary" if variant in ("binary", "deviance")
            else "multiclass" if variant == "multiclass"
            else "quantile" if variant == "quantile"
            else "regression"
        )

    # ------------------------------------------------------------------ #
    # Leaf values
    # ------------------------------------------------------------------ #
    def _newton_leaves(self, tree: CART, X: np.ndarray, y: np.ndarray,
                       p: np.ndarray, factor: float) -> None:
        """Replace each leaf mean with sum(y-p) / sum(p(1-p)), times factor."""
        num: dict[int, float] = {}
        den: dict[int, float] = {}
        nodes: dict[int, dict] = {}
        for xi, yi, pi in zip(X, y, p, strict=True):
            leaf = tree._walk(xi)
            key = id(leaf)
            nodes[key] = leaf
            num[key] = num.get(key, 0.0) + (yi - pi)
            den[key] = den.get(key, 0.0) + pi * (1.0 - pi)
        for key, leaf in nodes.items():
            leaf["value"] = factor * num[key] / (den[key] + 1e-12)

    def _quantile_leaves(self, tree: CART, X: np.ndarray, resid: np.ndarray) -> None:
        buckets: dict[int, list[float]] = {}
        nodes: dict[int, dict] = {}
        for xi, ri in zip(X, resid, strict=True):
            leaf = tree._walk(xi)
            key = id(leaf)
            nodes[key] = leaf
            buckets.setdefault(key, []).append(ri)
        for key, leaf in nodes.items():
            leaf["value"] = float(np.quantile(buckets[key], self.alpha))

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoosting:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = y.size
        self.n_features_ = X.shape[1]
        self.n_rows_used_: list[int] = []
        self.train_loss_: list[float] = []
        self.estimators_: list = []

        if self.task == "multiclass":
            self.classes_, yi = np.unique(y, return_inverse=True)
            K = self.classes_.size
            Y = np.eye(K)[yi]
            prior = np.clip(Y.mean(axis=0), 1e-6, 1.0)
            F = np.tile(np.log(prior), (n, 1))
            F -= F.mean(axis=1, keepdims=True)
        elif self.task == "binary":
            self.classes_ = np.unique(y)
            yb = (y == self.classes_[1]).astype(float)
            p0 = float(np.clip(yb.mean(), 1e-6, 1.0 - 1e-6))
            F = np.full(n, np.log(p0 / (1.0 - p0)))
        elif self.task == "quantile":
            self.classes_ = None
            F = np.full(n, float(np.quantile(y, self.alpha)))
        else:
            self.classes_ = None
            base = (float(np.quantile(y, self.alpha))
                    if self.task == "quantile" else float(np.mean(y)))
            F = np.full(n, base)

        if self.task == "multiclass":
            self._base_multi = F[0].copy()
        elif self.task == "binary":
            self._base_binary = float(F[0])
        else:
            self._base_value = float(F[0])

        for _ in range(self.n_estimators):
            idx = self._sample(n)
            self.n_rows_used_.append(idx.size)
            if self.task == "multiclass":
                P = _softmax(F)
                trees = []
                for k in range(self.classes_.size):
                    resid = Y[idx, k] - P[idx, k]
                    tree = self._tree().fit(X[idx], resid)
                    self._newton_leaves(tree, X[idx], Y[idx, k], P[idx, k],
                                        (self.classes_.size - 1.0) / self.classes_.size)
                    F[:, k] += self.lr * tree.predict(X)
                    trees.append(tree)
                self.estimators_.append(trees)
            elif self.task == "binary":
                p = _sigmoid(F)
                yb = (y == self.classes_[1]).astype(float)
                tree = self._tree().fit(X[idx], yb[idx] - p[idx])
                self._newton_leaves(tree, X[idx], yb[idx], p[idx], 1.0)
                F += self.lr * tree.predict(X)
                self.estimators_.append(tree)
            elif self.task == "quantile":
                resid = y - F
                pseudo = np.where(resid > 0.0, self.alpha, self.alpha - 1.0)
                tree = self._tree().fit(X[idx], pseudo[idx])
                self._quantile_leaves(tree, X[idx], resid[idx])
                F += self.lr * tree.predict(X)
                self.estimators_.append(tree)
            else:
                resid = y - F
                tree = self._tree().fit(X[idx], resid[idx])
                F += self.lr * tree.predict(X)
                self.estimators_.append(tree)
            self.train_loss_.append(self._loss(y, F))
        return self

    def _sample(self, n: int) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n)
        k = max(1, int(self.subsample * n))
        return self.rng.choice(n, size=k, replace=False)

    def _tree(self) -> CART:
        return CART("variance", max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=int(self.rng.integers(0, 2**31 - 1)))

    def _loss(self, y: np.ndarray, F: np.ndarray) -> float:
        if self.task == "binary":
            yb = (y == self.classes_[1]).astype(float)
            return float(np.mean(np.logaddexp(0.0, F) - yb * F))
        if self.task == "multiclass":
            yi = np.searchsorted(self.classes_, y)
            P = _softmax(F)
            return float(-np.mean(np.log(np.maximum(P[np.arange(y.size), yi], 1e-15))))
        if self.task == "quantile":
            r = y - F
            return float(np.mean(np.maximum(self.alpha * r, (self.alpha - 1.0) * r)))
        return float(np.mean((y - F) ** 2))

    def _raw(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.task == "multiclass":
            F = np.tile(self._base_multi, (X.shape[0], 1))
            for trees in self.estimators_:
                for k, tree in enumerate(trees):
                    F[:, k] += self.lr * tree.predict(X)
            return F
        base = self._base_binary if self.task == "binary" else self._base_value
        F = np.full(X.shape[0], base)
        for tree in self.estimators_:
            F += self.lr * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "binary":
            p = _sigmoid(self._raw(X))
            return np.column_stack([1.0 - p, p])
        if self.task == "multiclass":
            return _softmax(self._raw(X))
        raise ValueError("predict_proba is for classification variants")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "binary":
            return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]
        if self.task == "multiclass":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return self._raw(X)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)
    n = 350

    def f(x):
        return np.sin(1.5 * x[:, 0]) + 0.5 * x[:, 1] ** 2

    Xr = rng.uniform(-2.0, 2.0, size=(n, 2))
    yr = f(Xr) + rng.normal(0.0, 0.15, size=n)
    Xrt = rng.uniform(-2.0, 2.0, size=(500, 2))
    yrt = f(Xrt) + rng.normal(0.0, 0.15, size=500)

    # --- squared loss ---
    gb = GradientBoosting("regression", n_estimators=150, learning_rate=0.1,
                          max_depth=3, min_samples_leaf=5, random_state=1).fit(Xr, yr)
    r2 = 1.0 - np.mean((gb.predict(Xrt) - yrt) ** 2) / np.var(yrt)
    assert r2 > 0.9, r2
    assert gb.train_loss_[0] > gb.train_loss_[-1]
    print(f"boosting: regression R^2 {r2:.3f}, loss {gb.train_loss_[0]:.3f} "
          f"-> {gb.train_loss_[-1]:.3f}")

    # --- binary log loss ---
    Xb = rng.uniform(-1.5, 1.5, size=(n, 2))
    yb = ((Xb[:, 0] ** 2 + Xb[:, 1] ** 2) > 1.0).astype(int)
    yb[rng.random(n) < 0.08] ^= 1
    Xbt = rng.uniform(-1.5, 1.5, size=(500, 2))
    ybt = ((Xbt[:, 0] ** 2 + Xbt[:, 1] ** 2) > 1.0).astype(int)
    gbc = GradientBoosting("binary", n_estimators=100, learning_rate=0.1,
                           max_depth=3, min_samples_leaf=3, random_state=2).fit(Xb, yb)
    acc = np.mean(gbc.predict(Xbt) == ybt)
    assert acc > 0.9, acc
    print(f"boosting: binary test acc {acc:.3f}, log loss "
          f"{gbc.train_loss_[0]:.3f} -> {gbc.train_loss_[-1]:.3f}")

    # --- multiclass softmax ---
    centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
    yc = rng.integers(0, 3, size=n)
    Xc = centers[yc] + rng.normal(0.0, 0.6, size=(n, 2))
    Xct = centers[rng.integers(0, 3, size=600)] + rng.normal(0.0, 0.6, size=(600, 2))
    yct = np.argmin(((Xct[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    gbm = GradientBoosting("multiclass", n_estimators=60, learning_rate=0.2,
                           max_depth=3, random_state=3).fit(Xc, yc)
    acc_m = np.mean(gbm.predict(Xct) == yct)
    assert acc_m > 0.9, acc_m
    print(f"boosting: multiclass test acc {acc_m:.3f}")

    # --- shrinkage: same number of trees, smaller steps ---
    fast = GradientBoosting("shrinkage", n_estimators=300, learning_rate=1.0,
                            max_depth=3, min_samples_leaf=5,
                            random_state=4).fit(Xr, yr)
    slow = GradientBoosting("shrinkage", n_estimators=300, learning_rate=0.05,
                            max_depth=3, min_samples_leaf=5,
                            random_state=4).fit(Xr, yr)
    err_fast = np.mean((fast.predict(Xrt) - yrt) ** 2)
    err_slow = np.mean((slow.predict(Xrt) - yrt) ** 2)
    assert err_slow < err_fast, (err_slow, err_fast)
    print(f"boosting: shrinkage lr=1.0 MSE {err_fast:.3f} vs lr=0.05 {err_slow:.3f}")

    # --- stochastic subsampling ---
    sub = GradientBoosting("subsample", n_estimators=200, learning_rate=0.1,
                           max_depth=3, min_samples_leaf=5, subsample=0.5,
                           random_state=5).fit(Xr, yr)
    assert max(sub.n_rows_used_) < n
    err_sub = np.mean((sub.predict(Xrt) - yrt) ** 2)
    assert err_sub < 0.1, err_sub
    print(f"boosting: subsample 0.5 uses {sub.n_rows_used_[0]} rows, "
          f"test MSE {err_sub:.3f}")

    # --- quantile loss: coverage tracks alpha ---
    nq = 800
    Xq = rng.uniform(-2.0, 2.0, size=(nq, 2))
    yq = f(Xq) + rng.normal(0.0, 0.5, size=nq)
    Xqt = rng.uniform(-2.0, 2.0, size=(2000, 2))
    yqt = f(Xqt) + rng.normal(0.0, 0.5, size=2000)
    q50 = GradientBoosting("quantile", n_estimators=200, learning_rate=0.1,
                           max_depth=2, min_samples_leaf=30, alpha=0.5,
                           random_state=6).fit(Xq, yq)
    q90 = GradientBoosting("quantile", n_estimators=200, learning_rate=0.1,
                           max_depth=2, min_samples_leaf=30, alpha=0.9,
                           random_state=6).fit(Xq, yq)
    cov50 = float(np.mean(yqt <= q50.predict(Xqt)))
    cov90 = float(np.mean(yqt <= q90.predict(Xqt)))
    assert 0.45 < cov50 < 0.55, cov50
    assert 0.82 < cov90 < 0.93, cov90
    assert np.mean(q90.predict(Xqt)) > np.mean(q50.predict(Xqt))
    print(f"boosting: quantile coverage alpha=0.5 {cov50:.3f}, "
          f"alpha=0.9 {cov90:.3f}")

    # --- deviance is the same fit, viewed as a loss ---
    dev = GradientBoosting("deviance", n_estimators=100, learning_rate=0.1,
                           max_depth=3, min_samples_leaf=3, random_state=2).fit(Xb, yb)
    assert np.allclose(dev.predict_proba(Xbt), gbc.predict_proba(Xbt))
    assert dev.train_loss_[-1] < dev.train_loss_[0]
    print(f"boosting: deviance matches binary, final {dev.train_loss_[-1]:.3f}")

    print("gradient_boosting demo passed")


if __name__ == "__main__":
    main()
