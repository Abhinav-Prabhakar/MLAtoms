"""AdaBoost — boosting as reweighting, in six forms.

Chapter:   04-trees-and-ensembles
Companion: adaboost.md
Tier 1 · 6 variants: discrete, real, gentle, samme, samme_r, stagewise

AdaBoost fits weak learners one at a time. After each one, the rows it got wrong
have their weights increased, so the next learner is forced to focus on them, and
the final prediction is a weighted vote:

    F(x) = sum_t alpha_t h_t(x),        predict = sign(F(x))

The classic `discrete` form uses hard +/-1 predictions and a closed-form
`alpha = 0.5 log((1-err)/err)`. The other five variants differ in what the weak
learner outputs and how the step is chosen:

    real       confidence 0.5 log(p/(1-p)) per leaf
    gentle     weighted regression on y in {-1, +1}, no alpha
    samme      multiclass hard votes, alpha = log((1-err)/err) + log(K-1)
    samme_r    multiclass per-class log-probabilities (real SAMME)
    stagewise  gradient descent in function space with a line-searched step

The weak learner is a depth-limited weighted tree whose split criterion is weighted
gini. For binary labels weighted gini is exactly half the weighted variance, so the
same tree serves the classification and the regression variants.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["discrete", "real", "gentle", "samme", "samme_r", "stagewise"]
)

BINARY = frozenset(["discrete", "real", "gentle", "stagewise"])
MULTICLASS = frozenset(["samme", "samme_r"])

EPS = 1e-12


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# --------------------------------------------------------------------------- #
# Weak learner: a weighted tree
# --------------------------------------------------------------------------- #
def _leaf(y: np.ndarray, w: np.ndarray, k: int) -> dict:
    proba = np.bincount(y, weights=w, minlength=k).astype(float)
    total = proba.sum()
    return {
        "leaf": True,
        "n": y.size,
        "proba": proba / total if total > 0 else np.full(k, 1.0 / k),
    }


def _best_split(X, y, w, k, min_samples_leaf):
    n = X.shape[0]
    wsum = w.sum()
    counts = np.zeros((n, k))
    counts[np.arange(n), y] = w
    parent = 1.0 - float(((counts.sum(axis=0) / wsum) ** 2).sum())
    best = None
    for j in range(X.shape[1]):
        v = X[:, j]
        order = np.argsort(v, kind="mergesort")
        vs = v[order]
        cut = np.nonzero(vs[1:] > vs[:-1])[0]
        if cut.size == 0:
            continue
        cum = np.cumsum(counts[order], axis=0)
        lc = cum[cut]
        rc = cum[-1] - lc
        wl = lc.sum(axis=1)
        wr = rc.sum(axis=1)
        # min_samples_leaf counts rows, not weights.
        n_l = np.cumsum(np.ones(n))[cut]
        n_r = n - n_l
        ok = (n_l >= min_samples_leaf) & (n_r >= min_samples_leaf)
        if not ok.any():
            continue
        pl = lc / np.maximum(wl, EPS)[:, None]
        pr = rc / np.maximum(wr, EPS)[:, None]
        gl = 1.0 - (pl * pl).sum(axis=1)
        gr = 1.0 - (pr * pr).sum(axis=1)
        gain = parent - (wl * gl + wr * gr) / np.maximum(wl + wr, EPS)
        gain = np.where(ok, gain, -np.inf)
        i = int(np.argmax(gain))
        if not np.isfinite(gain[i]):
            continue
        if best is None or gain[i] > best["gain"]:
            best = {
                "feature": j,
                "threshold": float(0.5 * (vs[cut[i]] + vs[cut[i] + 1])),
                "gain": float(gain[i]),
            }
    return best


def _weighted_tree(X, y, w, k, depth, max_depth, min_samples_leaf) -> dict:
    node = _leaf(y, w, k)
    if depth >= max_depth or y.size < 2 * min_samples_leaf:
        return node
    split = _best_split(X, y, w, k, min_samples_leaf)
    if split is None or split["gain"] <= 1e-12:
        return node
    left = X[:, split["feature"]] <= split["threshold"]
    if not left.any() or left.all():
        return node
    node = {
        "leaf": False,
        "n": y.size,
        "feature": split["feature"],
        "threshold": split["threshold"],
        "left": _weighted_tree(X[left], y[left], w[left], k,
                               depth + 1, max_depth, min_samples_leaf),
        "right": _weighted_tree(X[~left], y[~left], w[~left], k,
                                depth + 1, max_depth, min_samples_leaf),
    }
    return node


def _tree_proba(tree: dict, X: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((X.shape[0], k))
    for i, x in enumerate(X):
        node = tree
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        out[i] = node["proba"]
    return out


class AdaBoost:
    """Boosted weighted trees.

    `variant` fixes the weak learner's output and the step rule. `max_depth` is the
    depth of the weak tree (1 = stump, the classical choice). `learning_rate`
    shrinks the step of the `stagewise` variant only; the other variants derive
    their step from the data.
    """

    def __init__(
        self,
        variant: str = "discrete",
        *,
        n_estimators: int = 50,
        max_depth: int = 1,
        min_samples_leaf: int = 1,
        learning_rate: float = 1.0,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lr = learning_rate
        self.rng = np.random.default_rng(random_state)
        self.task = "binary" if variant in BINARY else "multiclass"

    # ------------------------------------------------------------------ #
    # Fit helpers
    # ------------------------------------------------------------------ #
    def _fit_tree(self, X: np.ndarray, yi: np.ndarray, w: np.ndarray, k: int) -> dict:
        return _weighted_tree(X, yi, w, k, 0, self.max_depth, self.min_samples_leaf)

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

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> AdaBoost:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = y.size
        self.classes_, yi = np.unique(y, return_inverse=True)
        k = self.classes_.size
        if self.task == "binary" and k != 2:
            raise ValueError(f"{self.variant} needs exactly two classes, got {k}")
        self.estimators_: list[dict] = []
        self.alphas_: list[float] = []
        self.errors_: list[float] = []
        self.train_loss_: list[float] = []
        self.train_error_: list[float] = []
        w = np.full(n, 1.0 / n)
        F = np.zeros(n) if self.task == "binary" else np.zeros((n, k))
        y_signed = 2.0 * yi - 1.0 if self.task == "binary" else None

        for _ in range(self.n_estimators):
            tree = self._fit_tree(X, yi, w, k)
            p = _tree_proba(tree, X, k)

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
            elif self.variant == "real":
                p1 = np.clip(p[:, 1], 1e-10, 1.0 - 1e-10)
                h = 0.5 * np.log(p1 / (1.0 - p1))
                F += h
                self.alphas_.append(1.0)
                err = float((w * ((h > 0.0).astype(float) != yi)).sum())
            elif self.variant == "gentle":
                h = 2.0 * p[:, 1] - 1.0
                F += h
                self.alphas_.append(1.0)
                err = float((w * ((h > 0.0).astype(float) != yi)).sum())
            elif self.variant == "stagewise":
                h = 2.0 * p[:, 1] - 1.0
                gamma = self.lr * self._line_search(y_signed, h, w)
                F += gamma * h
                self.alphas_.append(float(gamma))
                err = float((w * ((h > 0.0).astype(float) != yi)).sum())
            elif self.variant == "samme":
                pred = np.argmax(p, axis=1)
                err = float((w * (pred != yi)).sum())
                err = float(np.clip(err, 1e-10, 1.0 - 1e-10))
                alpha = np.log((1.0 - err) / err) + np.log(k - 1.0)
                if err >= 1.0 - 1.0 / k:
                    break
                F[np.arange(n), pred] += alpha
                self.alphas_.append(float(alpha))
                w = w * np.exp(alpha * (pred != yi))
                w /= w.sum()
            else:                                     # samme_r
                pk = np.clip(p, 1e-10, 1.0)
                logp = np.log(pk)
                h = (k - 1.0) * (logp - logp.mean(axis=1, keepdims=True))
                F += h
                self.alphas_.append(1.0)
                err = float((w * (np.argmax(p, axis=1) != yi)).sum())
                w = w * pk[np.arange(n), yi] ** (-(k - 1.0) / k)
                w /= w.sum()

            if self.variant not in ("samme", "samme_r"):
                # exp(-yF) underflows/overflows on separable data; the clip keeps
                # the weights finite without changing the argmin.
                w = np.exp(-y_signed * np.clip(F, -50.0, 50.0))
                w /= w.sum()

            self.estimators_.append(tree)
            self.errors_.append(err)
            if self.task == "binary":
                self.train_loss_.append(float(np.mean(np.exp(-y_signed * F))))
                self.train_error_.append(float(np.mean((F > 0.0) != yi)))
            else:
                P = _softmax(F)
                self.train_loss_.append(float(-np.mean(
                    np.log(np.maximum(P[np.arange(n), yi], EPS)))))
                self.train_error_.append(float(np.mean(np.argmax(F, axis=1) != yi)))
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("model is not fitted")
        X = np.asarray(X, dtype=float)
        k = self.classes_.size
        if self.task == "binary":
            F = np.zeros(X.shape[0])
            for tree, alpha in zip(self.estimators_, self.alphas_, strict=True):
                p = _tree_proba(tree, X, k)
                if self.variant == "discrete":
                    F += alpha * (2.0 * (p[:, 1] > 0.5) - 1.0)
                elif self.variant == "real":
                    p1 = np.clip(p[:, 1], 1e-10, 1.0 - 1e-10)
                    F += alpha * 0.5 * np.log(p1 / (1.0 - p1))
                else:
                    F += alpha * (2.0 * p[:, 1] - 1.0)
            return F
        F = np.zeros((X.shape[0], k))
        for tree, alpha in zip(self.estimators_, self.alphas_, strict=True):
            p = _tree_proba(tree, X, k)
            if self.variant == "samme":
                F[np.arange(X.shape[0]), np.argmax(p, axis=1)] += alpha
            else:
                logp = np.log(np.clip(p, 1e-10, 1.0))
                F += alpha * (k - 1.0) * (logp - logp.mean(axis=1, keepdims=True))
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = self.decision_function(X)
        if self.task == "binary":
            p = _sigmoid(2.0 * F)
            return np.column_stack([1.0 - p, p])
        return _softmax(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self.decision_function(X)
        if self.task == "binary":
            return self.classes_[(F > 0.0).astype(int)]
        return self.classes_[np.argmax(F, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)
    n = 400

    # --- binary: a noisy circle, stumps only ---
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1.0).astype(int)
    flip = rng.random(n) < 0.05
    y[flip] ^= 1
    Xt = rng.uniform(-1.5, 1.5, size=(600, 2))
    yt = ((Xt[:, 0] ** 2 + Xt[:, 1] ** 2) > 1.0).astype(int)

    disc = AdaBoost("discrete", n_estimators=80, max_depth=1).fit(X, y)
    acc_d = disc.score(Xt, yt)
    assert acc_d > 0.9, acc_d
    assert len(disc.estimators_) >= 4, len(disc.estimators_)
    print(f"adaboost: discrete test acc {acc_d:.3f} "
          f"({len(disc.estimators_)} stumps, loss {disc.train_loss_[-1]:.4f})")

    # The exponential-loss bound: L <= prod_t 2 sqrt(err_t (1 - err_t)).
    bound = 1.0
    for e in disc.errors_:
        bound *= 2.0 * np.sqrt(e * (1.0 - e))
    assert np.isclose(disc.train_loss_[-1], bound, rtol=1e-9), (
        disc.train_loss_[-1], bound)
    print(f"adaboost: exp loss {disc.train_loss_[-1]:.6f} == prod Z_t {bound:.6f}")

    # --- real and gentle: confidence-rated updates ---
    real = AdaBoost("real", n_estimators=80).fit(X, y)
    gentle = AdaBoost("gentle", n_estimators=80).fit(X, y)
    acc_r, acc_g = real.score(Xt, yt), gentle.score(Xt, yt)
    assert acc_r > 0.9 and acc_g > 0.9, (acc_r, acc_g)
    for m in (real, gentle):
        assert m.train_loss_[-1] < m.train_loss_[0]
    print(f"adaboost: real acc {acc_r:.3f} (loss {real.train_loss_[-1]:.4f}), "
          f"gentle acc {acc_g:.3f} (loss {gentle.train_loss_[-1]:.4f})")

    # --- stagewise: line-searched step, shrinkage needs more stages ---
    slow = AdaBoost("stagewise", n_estimators=150, learning_rate=0.3).fit(X, y)
    fast = AdaBoost("stagewise", n_estimators=25, learning_rate=1.0).fit(X, y)
    acc_s = slow.score(Xt, yt)
    assert acc_s > 0.9, acc_s
    assert slow.train_loss_[-1] < slow.train_loss_[0]
    assert all(0.0 <= g <= 5.0 for g in slow.alphas_)
    print(f"adaboost: stagewise lr=0.3 acc {acc_s:.3f}, "
          f"mean step {np.mean(slow.alphas_):.3f}; lr=1.0 acc {fast.score(Xt, yt):.3f}")

    # --- multiclass: three blobs ---
    centers = np.array([[-2.2, 0.0], [2.2, 0.0], [0.0, 2.4]])
    nc = 450
    labels = rng.integers(0, 3, size=nc)
    Xc = centers[labels] + rng.normal(0.0, 0.7, size=(nc, 2))
    lc = rng.integers(0, 3, size=800)
    Xct = centers[lc] + rng.normal(0.0, 0.7, size=(800, 2))

    sam = AdaBoost("samme", n_estimators=60, max_depth=2).fit(Xc, labels)
    acc_sam = sam.score(Xct, lc)
    assert acc_sam > 0.85, acc_sam
    assert all(a > 0.0 for a in sam.alphas_)
    print(f"adaboost: samme acc {acc_sam:.3f} "
          f"({len(sam.estimators_)} trees, err {sam.train_error_[-1]:.4f})")

    samr = AdaBoost("samme_r", n_estimators=40, max_depth=2).fit(Xc, labels)
    acc_samr = samr.score(Xct, lc)
    assert acc_samr > 0.85, acc_samr
    print(f"adaboost: samme_r acc {acc_samr:.3f} "
          f"(loss {samr.train_loss_[-1]:.4f})")

    # --- probabilities are ordered, and a deeper weak learner fits faster ---
    proba = real.predict_proba(Xt)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba > 0.0) & (proba < 1.0))
    deep = AdaBoost("gentle", n_estimators=15, max_depth=3).fit(X, y)
    assert deep.train_loss_[5] < gentle.train_loss_[5]
    print(f"adaboost: gentle depth-3 loss@6 {deep.train_loss_[5]:.4f} "
          f"vs stumps {gentle.train_loss_[5]:.4f}")

    print("adaboost demo passed")


if __name__ == "__main__":
    main()
