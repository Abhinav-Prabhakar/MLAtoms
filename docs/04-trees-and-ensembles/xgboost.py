"""XGBoost — second-order boosting with a regularised objective.

Chapter:   04-trees-and-ensembles
Companion: xgboost.md
Tier 1 · 9 variants: exact, approx, hist, dart, linear_booster, reg_alpha,
reg_lambda, colsample, sparsity_aware

XGBoost is gradient boosting with three changes that matter:

1. **Second-order objective.** Every split and leaf uses the gradient `g` and the
   Hessian `h`, not just the gradient:

       gain = 1/2 [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                    - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

   and the leaf value is `-G/(H+lambda)`.

2. **Explicit regularisation.** `lambda` shrinks leaves, `alpha` makes them sparse,
   `gamma` charges for each split, and column subsampling decorrelates trees.

3. **A sparsity-aware split search.** Missing values are sent to whichever side
   improves the objective, and the direction is stored with the node.

The variants here are the split enumerations (exact/approx/hist), the DART
dropout, linear leaves, and the four regularisation knobs. This is a compact
reimplementation of the 2016 paper, not a production system.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["exact", "approx", "hist", "dart", "linear_booster", "reg_alpha",
     "reg_lambda", "colsample", "sparsity_aware"]
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _soft(g, alpha: float):
    """Proximal operator of alpha|w|: sign(g) max(|g|-alpha, 0)."""
    return np.sign(g) * np.maximum(np.abs(g) - alpha, 0.0)


class XGBoost:
    """A compact second-order booster.

    task is "regression" (squared loss) or "binary" (logistic).
    `variant` selects the split enumeration and the regularisation defaults.
    """

    def __init__(
        self,
        variant: str = "exact",
        *,
        task: str = "regression",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample: float = 1.0,
        max_bins: int = 32,
        drop_rate: float = 0.1,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if task not in ("regression", "binary"):
            raise ValueError(f"unknown task {task!r}")
        self.variant = variant
        self.task = task
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = 10.0 if variant == "reg_lambda" and reg_lambda == 1.0 else reg_lambda
        self.reg_alpha = 1.0 if variant == "reg_alpha" and reg_alpha == 0.0 else reg_alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = 0.7 if variant == "colsample" and colsample == 1.0 else colsample
        self.max_bins = max_bins
        self.drop_rate = drop_rate if variant == "dart" else 0.0
        self.linear = variant == "linear_booster"
        self.rng = np.random.default_rng(random_state)
        self.estimators_: list[dict] = []
        self.n_features_used_: list[int] = []
        self.n_dropped_ = 0

    # ------------------------------------------------------------------ #
    # Objective pieces
    # ------------------------------------------------------------------ #
    def _leaf_value(self, G: float, H: float) -> float:
        return -_soft(G, self.reg_alpha) / (H + self.reg_lambda)

    def _score(self, G: float, H: float) -> float:
        """Negative of the minimised leaf objective, up to the factor 1/2."""
        s = _soft(G, self.reg_alpha)
        return s * s / (H + self.reg_lambda)

    def _split_gain(self, GL, HL, GR, HR, GP, HP):
        return 0.5 * (self._score(GL, HL) + self._score(GR, HR)
                      - self._score(GP, HP)) - self.gamma

    # ------------------------------------------------------------------ #
    # Leaf fitting
    # ------------------------------------------------------------------ #
    def _fit_linear(self, X: np.ndarray, idx: np.ndarray,
                    g: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
        Xl = np.column_stack([X[idx], np.ones(idx.size)])
        hh = h[idx]
        target = -g[idx] / np.maximum(hh, 1e-12)
        A = (Xl * hh[:, None]).T @ Xl
        pen = np.full(Xl.shape[1], self.reg_lambda)
        pen[-1] = 0.0
        A += np.diag(pen)
        b = Xl.T @ (hh * target)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b
        return w[:-1], float(w[-1])

    def _make_leaf(self, X, idx, g, h) -> dict:
        G, H = float(g[idx].sum()), float(h[idx].sum())
        node = {"leaf": True, "n": idx.size,
                "value": float(self._leaf_value(G, H))}
        if self.linear:
            if idx.size >= 2:
                node["linear"] = self._fit_linear(X, idx, g, h)
            else:
                node["linear"] = (np.zeros(X.shape[1]), float(node["value"]))
        return node

    # ------------------------------------------------------------------ #
    # Split search
    # ------------------------------------------------------------------ #
    def _bin_edges(self, X: np.ndarray) -> list[np.ndarray]:
        edges = []
        for j in range(X.shape[1]):
            v = X[:, j]
            v = v[~np.isnan(v)]
            if v.size < 2:
                edges.append(np.array([]))
            else:
                q = np.quantile(v, np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1])
                edges.append(np.unique(q))
        return edges

    def _candidates(self, v: np.ndarray, j: int) -> np.ndarray:
        if self.variant in ("hist", "colsample", "sparsity_aware", "dart",
                            "reg_alpha", "reg_lambda"):
            edges = self.bin_edges_[j]
            return edges
        if self.variant == "approx":
            if v.size < 2:
                return np.array([])
            q = np.quantile(v, np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1])
            return np.unique(q)
        return 0.5 * (v[:-1] + v[1:])

    def _best_split(self, X, g, h, idx, features):
        best = None
        for j in features:
            cand = self._split_feature(X, g, h, idx, j)
            if cand is not None and (best is None or cand["gain"] > best["gain"]):
                best = cand
        return best

    def _split_feature(self, X, g, h, idx, j):
        col = X[idx, j]
        obs = ~np.isnan(col)
        n_obs = int(obs.sum())
        if n_obs < 2:
            return None
        rows = idx[obs]
        v = col[obs]
        order = np.argsort(v, kind="mergesort")
        v = v[order]
        G = g[rows][order]
        H = h[rows][order]
        if self.variant == "exact":
            cand = 0.5 * (v[:-1] + v[1:])
            cand = cand[v[1:] > v[:-1]]
        else:
            cand = self._candidates(v, j)
        if cand.size == 0:
            return None
        cumG, cumH = np.cumsum(G), np.cumsum(H)
        # left side contains all values <= cand (searchsorted, side="right")
        pos = np.searchsorted(v, cand, side="right") - 1
        ok = (pos >= 0) & (pos < n_obs - 1)
        if not ok.any():
            return None
        pos = pos[ok]
        cand = cand[ok]
        GL, HL = cumG[pos], cumH[pos]
        GR, HR = cumG[-1] - GL, cumH[-1] - HL
        GP_obs, HP_obs = float(cumG[-1]), float(cumH[-1])
        GP, HP = float(g[idx].sum()), float(h[idx].sum())
        n_l = (pos + 1).astype(float)
        n_r = (n_obs - pos - 1).astype(float)
        n_m = float(idx.size - n_obs)
        Gm = float(g[idx[~obs]].sum()) if n_m else 0.0
        Hm = float(h[idx[~obs]].sum()) if n_m else 0.0
        gain0 = self._split_gain(GL, HL, GR, HR, GP_obs, HP_obs)
        gain1 = self._split_gain(GL + Gm, HL + Hm, GR, HR, GP, HP)
        gain2 = self._split_gain(GL, HL, GR + Gm, HR + Hm, GP, HP)
        ok0 = ((n_l >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf)
               & (n_m == 0))
        ok1 = ((n_l + n_m) >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf)
        ok2 = (n_l >= self.min_samples_leaf) & ((n_r + n_m) >= self.min_samples_leaf)
        best = None
        for gain, valid_mask, missing_left in (
            (gain0, ok0, True), (gain1, ok1, True), (gain2, ok2, False)
        ):
            if not valid_mask.any():
                continue
            k = int(np.argmax(np.where(valid_mask, gain, -np.inf)))
            if best is None or gain[k] > best["gain"]:
                best = {"feature": j, "threshold": float(cand[k]),
                        "gain": float(gain[k]), "missing_left": bool(missing_left)}
        return best

    # ------------------------------------------------------------------ #
    # Growth
    # ------------------------------------------------------------------ #
    def _grow(self, X, g, h, idx, depth, features) -> dict:
        if (depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf
                or idx.size < 2):
            return self._make_leaf(X, idx, g, h)
        split = self._best_split(X, g, h, idx, features)
        if split is None or split["gain"] <= 0.0:
            return self._make_leaf(X, idx, g, h)
        col = X[idx, split["feature"]]
        left = (col <= split["threshold"]) | (np.isnan(col) & split["missing_left"])
        li, ri = idx[left], idx[~left]
        if li.size == 0 or ri.size == 0:
            return self._make_leaf(X, idx, g, h)
        return {
            "leaf": False,
            "n": idx.size,
            "feature": split["feature"],
            "threshold": split["threshold"],
            "missing_left": split["missing_left"],
            "gain": split["gain"],
            "left": self._grow(X, g, h, li, depth + 1, features),
            "right": self._grow(X, g, h, ri, depth + 1, features),
        }

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def _gradients(self, y, F):
        if self.task == "binary":
            p = _sigmoid(F)
            return p - y, p * (1.0 - p)
        return F - y, np.ones_like(F)

    def _tree_predict(self, tree: dict, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape[0])
        for i, x in enumerate(X):
            node = tree
            while not node["leaf"]:
                j = node["feature"]
                left = (node["missing_left"] if np.isnan(x[j])
                        else x[j] <= node["threshold"])
                node = node["left"] if left else node["right"]
            if self.linear:
                w, b = node["linear"]
                out[i] = np.nan_to_num(x) @ w + b
            else:
                out[i] = node["value"]
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoost:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_ = d
        self.bin_edges_ = self._bin_edges(X)
        self.estimators_ = []
        self.train_loss_: list[float] = []
        if self.task == "binary":
            p0 = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
            base = float(np.log(p0 / (1.0 - p0)))
            self.classes_ = np.unique(y)
        else:
            base = float(y.mean())
            self.classes_ = None
        self._base = base
        F = np.full(n, base)

        for _ in range(self.n_estimators):
            if self.drop_rate > 0.0 and self.estimators_:
                keep = self.rng.random(len(self.estimators_)) >= self.drop_rate
                self.n_dropped_ += int((~keep).sum())
                scale = 1.0 / (1.0 - self.drop_rate)
                F = np.full(n, base)
                for tree, k in zip(self.estimators_, keep, strict=True):
                    if k:
                        F += scale * self.lr * self._tree_predict(tree, X)
            g, h = self._gradients(y, F)
            if self.subsample < 1.0:
                k = max(2, int(self.subsample * n))
                idx = self.rng.choice(n, size=k, replace=False)
            else:
                idx = np.arange(n)
            m = max(1, int(self.colsample * d))
            features = (np.arange(d) if m >= d
                        else self.rng.choice(d, size=m, replace=False))
            self.n_features_used_.append(features.size)
            tree = self._grow(X, g, h, idx, 0, features)
            self.estimators_.append(tree)
            if self.drop_rate == 0.0:
                F += self.lr * self._tree_predict(tree, X)
            self.train_loss_.append(self._loss(y, self._full_raw(X)))
        return self

    def _full_raw(self, X: np.ndarray) -> np.ndarray:
        F = np.full(X.shape[0], self._base)
        for tree in self.estimators_:
            F += self.lr * self._tree_predict(tree, X)
        return F

    def _loss(self, y, F):
        if self.task == "binary":
            return float(np.mean(np.logaddexp(0.0, F) - y * F))
        return float(np.mean((y - F) ** 2))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "binary":
            raise ValueError("predict_proba is binary only")
        p = _sigmoid(self._full_raw(np.asarray(X, dtype=float)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.task == "binary":
            return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]
        return self._full_raw(X)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)
    n = 350

    def f(x):
        return np.sin(1.5 * x[:, 0]) + 0.5 * x[:, 1] ** 2

    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = f(X) + rng.normal(0.0, 0.15, size=n)
    Xt = rng.uniform(-2.0, 2.0, size=(500, 2))
    yt = f(Xt) + rng.normal(0.0, 0.15, size=500)

    def r2(pred):
        return 1.0 - np.mean((pred - yt) ** 2) / np.var(yt)

    # --- exact split enumeration ---
    exact = XGBoost("exact", n_estimators=120, learning_rate=0.1, max_depth=3,
                    min_samples_leaf=5, random_state=1).fit(X, y)
    r2_exact = r2(exact.predict(Xt))
    assert r2_exact > 0.92, r2_exact
    print(f"xgboost: exact R^2 {r2_exact:.3f}, loss {exact.train_loss_[0]:.3f} "
          f"-> {exact.train_loss_[-1]:.3f}")

    # --- histogram and approximate splits ---
    hist = XGBoost("hist", n_estimators=120, learning_rate=0.1, max_depth=3,
                   min_samples_leaf=5, max_bins=16, random_state=1).fit(X, y)
    approx = XGBoost("approx", n_estimators=120, learning_rate=0.1, max_depth=3,
                     min_samples_leaf=5, max_bins=16, random_state=1).fit(X, y)
    r2_hist, r2_approx = r2(hist.predict(Xt)), r2(approx.predict(Xt))
    assert min(r2_hist, r2_approx) > 0.88, (r2_hist, r2_approx)
    print(f"xgboost: hist R^2 {r2_hist:.3f} (16 bins), approx R^2 {r2_approx:.3f}")

    # --- binary logistic objective ---
    Xb = rng.uniform(-1.5, 1.5, size=(n, 2))
    yb = ((Xb[:, 0] ** 2 + Xb[:, 1] ** 2) > 1.0).astype(float)
    flip = rng.random(n) < 0.08
    yb[flip] = 1.0 - yb[flip]
    Xbt = rng.uniform(-1.5, 1.5, size=(500, 2))
    ybt = ((Xbt[:, 0] ** 2 + Xbt[:, 1] ** 2) > 1.0).astype(float)
    xbc = XGBoost("exact", task="binary", n_estimators=100, learning_rate=0.1,
                  max_depth=3, min_samples_leaf=3, random_state=2).fit(Xb, yb)
    acc = np.mean(xbc.predict(Xbt) == ybt)
    assert acc > 0.9, acc
    print(f"xgboost: binary acc {acc:.3f}, log loss "
          f"{xbc.train_loss_[0]:.3f} -> {xbc.train_loss_[-1]:.3f}")

    # --- DART dropout ---
    dart = XGBoost("dart", n_estimators=120, learning_rate=0.1, max_depth=3,
                   min_samples_leaf=5, drop_rate=0.15,
                   random_state=3).fit(X, y)
    r2_dart = r2(dart.predict(Xt))
    assert r2_dart > 0.88, r2_dart
    assert dart.n_dropped_ > 0
    print(f"xgboost: DART R^2 {r2_dart:.3f}, {dart.n_dropped_} tree-drops")

    # --- linear leaves ---
    lin = XGBoost("linear_booster", n_estimators=40, learning_rate=0.2,
                  max_depth=2, min_samples_leaf=10,
                  random_state=4).fit(X, y)
    r2_lin = r2(lin.predict(Xt))
    assert r2_lin > 0.95, r2_lin
    print(f"xgboost: linear_booster R^2 {r2_lin:.3f}")

    # --- L1 makes leaves sparse ---
    def count_zero_leaves(model):
        zeros = 0

        def walk(node):
            nonlocal zeros
            if node["leaf"]:
                zeros += int(abs(node["value"]) < 1e-12)
                return
            walk(node["left"])
            walk(node["right"])

        for tree in model.estimators_:
            walk(tree)
        return zeros

    l1 = XGBoost("reg_alpha", n_estimators=60, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=5, reg_alpha=20.0, random_state=5).fit(X, y)
    no_l1 = XGBoost("exact", n_estimators=60, learning_rate=0.1, max_depth=3,
                    min_samples_leaf=5, random_state=5).fit(X, y)
    assert count_zero_leaves(l1) > count_zero_leaves(no_l1)
    print(f"xgboost: L1 zero leaves {count_zero_leaves(l1)} "
          f"vs {count_zero_leaves(no_l1)} without")

    # --- L2 shrinks leaves ---
    def mean_leaf(model):
        vals = []

        def walk(node):
            if node["leaf"]:
                vals.append(abs(node["value"]))
                return
            walk(node["left"])
            walk(node["right"])

        for tree in model.estimators_:
            walk(tree)
        return float(np.mean(vals))

    weak = XGBoost("reg_lambda", n_estimators=40, learning_rate=0.1, max_depth=3,
                   min_samples_leaf=5, reg_lambda=0.1, random_state=6).fit(X, y)
    strong = XGBoost("reg_lambda", n_estimators=40, learning_rate=0.1, max_depth=3,
                     min_samples_leaf=5, reg_lambda=100.0, random_state=6).fit(X, y)
    assert mean_leaf(strong) < mean_leaf(weak)
    print(f"xgboost: mean |leaf| lambda=100 {mean_leaf(strong):.4f} "
          f"vs 0.1 {mean_leaf(weak):.4f}")

    # --- column subsampling ---
    col = XGBoost("colsample", n_estimators=100, learning_rate=0.1, max_depth=3,
                  min_samples_leaf=5, colsample=0.5,
                  random_state=7).fit(X, y)
    assert max(col.n_features_used_) < X.shape[1]
    assert r2(col.predict(Xt)) > 0.88
    print(f"xgboost: colsample uses {col.n_features_used_[0]}/{X.shape[1]} features, "
          f"R^2 {r2(col.predict(Xt)):.3f}")

    # --- sparsity-aware splits: train and test with missing values ---
    Xm = X.copy()
    Xm[rng.random(n) < 0.2, 0] = np.nan
    Xmt = Xt.copy()
    Xmt[rng.random(Xt.shape[0]) < 0.2, 0] = np.nan
    sparse = XGBoost("sparsity_aware", n_estimators=120, learning_rate=0.1,
                     max_depth=3, min_samples_leaf=5,
                     random_state=8).fit(Xm, y)
    r2_sparse = r2(sparse.predict(Xmt))
    assert r2_sparse > 0.8, r2_sparse
    print(f"xgboost: sparsity_aware R^2 {r2_sparse:.3f} with 20% missing x0")

    print("xgboost demo passed")


if __name__ == "__main__":
    main()
