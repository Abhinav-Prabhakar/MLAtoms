"""CART — one tree, seven ways to decide what a split is worth.

Chapter:   04-trees-and-ensembles
Companion: cart.md
Tier 1 · 7 variants: gini, entropy, variance, friedman_mse,
cost_complexity_pruning, surrogate_splits, missing_values

The tree is grown greedily: at every node, scan every feature for the threshold
with the largest impurity decrease, split, recurse. The variants change only two
things — the impurity (gini/entropy for classification, variance/Friedman MSE for
regression) and what happens around the edges (pruning, surrogate splits, missing
values).

    risk(node) = n_node / n_total * impurity(node)
    gain       = impurity(parent) - (n_l*impurity(l) + n_r*impurity(r)) / n

Splits are binary and axis-aligned, so every prediction is a walk down a chain of
`x_j <= t` tests. That is the whole model.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["gini", "entropy", "variance", "friedman_mse", "cost_complexity_pruning",
     "surrogate_splits", "missing_values"]
)

CLASSIFICATION = frozenset(
    ["gini", "entropy", "cost_complexity_pruning", "surrogate_splits", "missing_values"]
)
REGRESSION = frozenset(["variance", "friedman_mse"])


class CART:
    """A single CART tree with optional pruning, surrogates and missing values.

    max_features may be an int, "sqrt", "log2" or None (all features).
    n_surrogates is only used by the surrogate_splits variant.
    """

    def __init__(
        self,
        variant: str = "gini",
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = None,
        ccp_alpha: float = 0.0,
        n_surrogates: int = 3,
        splitter: str = "best",
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.n_surrogates = n_surrogates
        self.splitter = splitter
        self.rng = np.random.default_rng(random_state)
        self.task = "classification" if variant in CLASSIFICATION else "regression"
        self.criterion = variant
        self.use_surrogates = variant == "surrogate_splits"
        self.root: dict | None = None

    # ------------------------------------------------------------------ #
    # Impurity
    # ------------------------------------------------------------------ #
    def _impurity(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        if self.task == "classification":
            p = np.bincount(y, minlength=self.n_classes) / y.size
            if self.criterion == "entropy":
                p = p[p > 0]
                return float(-(p * np.log2(p)).sum())
            return float(1.0 - (p * p).sum())
        return float(np.var(y))

    def _imp_from_counts(self, c: np.ndarray, n: np.ndarray) -> np.ndarray:
        p = c / np.maximum(n, 1)[:, None]
        if self.criterion == "entropy":
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(p > 0.0, p * np.log2(np.maximum(p, 1e-300)), 0.0)
            return -t.sum(axis=1)
        return 1.0 - (p * p).sum(axis=1)

    # ------------------------------------------------------------------ #
    # Split search
    # ------------------------------------------------------------------ #
    def _feature_subset(self, d: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(d)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(d)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(d)))
        elif isinstance(self.max_features, float) and self.max_features <= 1.0:
            k = max(1, int(self.max_features * d))
        else:
            k = int(self.max_features)
        if k >= d:
            return np.arange(d)
        return self.rng.choice(d, size=k, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> dict | None:
        best = None
        for j in self._feature_subset(X.shape[1]):
            cand = self._best_split_feature(X, y, j)
            if cand is not None and (best is None or cand["gain"] > best["gain"]):
                best = cand
        return best

    def _best_split_feature(self, X: np.ndarray, y: np.ndarray, j: int) -> dict | None:
        col = X[:, j]
        obs = ~np.isnan(col)
        if int(obs.sum()) < 2:
            return None
        v = col[obs]
        y_obs = y[obs]
        y_miss = y[~obs]
        order = np.argsort(v, kind="mergesort")
        v, y_obs = v[order], y_obs[order]
        cut = np.nonzero(v[1:] > v[:-1])[0]
        if cut.size == 0:
            return None
        thr = 0.5 * (v[cut] + v[cut + 1])
        if self.splitter == "random":
            k = int(self.rng.integers(0, cut.size))
            cut, thr = cut[[k]], thr[[k]]
        n_obs = v.size
        n_l = (cut + 1).astype(float)
        n_r = (n_obs - cut - 1).astype(float)
        n_m = float(y_miss.size)
        parent = self._impurity(y)

        if self.task == "classification":
            counts = np.zeros((n_obs, self.n_classes))
            counts[np.arange(n_obs), y_obs] = 1.0
            cum = np.cumsum(counts, axis=0)
            lc = cum[cut]
            rc = cum[-1] - lc
            mc = np.bincount(y_miss, minlength=self.n_classes).astype(float)
            imp_l0 = self._imp_from_counts(lc, n_l)
            imp_r0 = self._imp_from_counts(rc, n_r)
            imp_l1 = self._imp_from_counts(lc + mc, n_l + n_m) if n_m else imp_l0
            imp_r1 = self._imp_from_counts(rc + mc, n_r + n_m) if n_m else imp_r0
            n = n_l + n_r + n_m
            gain0 = parent - (n_l * imp_l0 + n_r * imp_r0) / (n_l + n_r)
            gain1 = parent - ((n_l + n_m) * imp_l1 + n_r * imp_r0) / n
            gain2 = parent - (n_l * imp_l0 + (n_r + n_m) * imp_r1) / n
        else:
            sy = np.cumsum(y_obs)
            sy2 = np.cumsum(y_obs * y_obs)
            sl, sl2 = sy[cut], sy2[cut]
            sr, sr2 = sy[-1] - sl, sy2[-1] - sl2
            sm = float(y_miss.sum())
            sm2 = float((y_miss ** 2).sum())
            mu_l = sl / n_l
            mu_r = sr / n_r
            n = n_l + n_r + n_m
            if self.criterion == "friedman_mse":
                gain0 = (n_l * n_r / (n_l + n_r)) * (mu_l - mu_r) ** 2
                gain1 = ((n_l + n_m) * n_r / (n_l + n_m + n_r)) * (
                    (sl + sm) / (n_l + n_m) - mu_r) ** 2
                gain2 = (n_l * (n_r + n_m) / (n_l + n_r + n_m)) * (
                    mu_l - (sr + sm) / (n_r + n_m)) ** 2
            else:
                vl = np.maximum(sl2 / n_l - mu_l ** 2, 0.0)
                vr = np.maximum(sr2 / n_r - mu_r ** 2, 0.0)
                vl1 = np.maximum((sl2 + sm2) / (n_l + n_m)
                                 - ((sl + sm) / (n_l + n_m)) ** 2, 0.0)
                vr1 = np.maximum((sr2 + sm2) / (n_r + n_m)
                                 - ((sr + sm) / (n_r + n_m)) ** 2, 0.0)
                gain0 = parent - (n_l * vl + n_r * vr) / (n_l + n_r)
                gain1 = parent - ((n_l + n_m) * vl1 + n_r * vr) / n
                gain2 = parent - (n_l * vl + (n_r + n_m) * vr1) / n

        ok = (n_l >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf)
        ok1 = ((n_l + n_m) >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf)
        ok2 = (n_l >= self.min_samples_leaf) & ((n_r + n_m) >= self.min_samples_leaf)

        best = None
        for gain, valid, missing_left in (
            (gain0, ok, True), (gain1, ok1, True), (gain2, ok2, False)
        ):
            if not valid.any():
                continue
            k = int(np.argmax(np.where(valid, gain, -np.inf)))
            if best is None or gain[k] > best["gain"]:
                best = {
                    "feature": j,
                    "threshold": float(thr[k]),
                    "gain": float(gain[k]),
                    "missing_left": bool(missing_left),
                }
        return best

    # ------------------------------------------------------------------ #
    # Surrogates
    # ------------------------------------------------------------------ #
    def _surrogates(self, X: np.ndarray, y: np.ndarray, split: dict) -> list[dict]:
        j, t = split["feature"], split["threshold"]
        col = X[:, j]
        known = ~np.isnan(col)
        primary_left = np.zeros(X.shape[0], dtype=bool)
        primary_left[known] = col[known] <= t
        out = []
        for k in range(X.shape[1]):
            if k == j:
                continue
            ck = X[:, k]
            both = known & ~np.isnan(ck)
            if int(both.sum()) < 4:
                continue
            v = ck[both]
            pl = primary_left[both]
            order = np.argsort(v, kind="mergesort")
            v, pl = v[order], pl[order]
            cut = np.nonzero(v[1:] > v[:-1])[0]
            if cut.size == 0:
                continue
            side = np.arange(v.size)[None, :] <= cut[:, None]      # (cuts, n)
            agree = (side == pl[None, :]).mean(axis=1)
            eff = np.maximum(agree, 1.0 - agree)
            i = int(np.argmax(eff))
            if eff[i] <= 0.5:
                continue
            out.append({
                "feature": k,
                "threshold": float(0.5 * (v[cut[i]] + v[cut[i] + 1])),
                "flip": bool(agree[i] < 0.5),
                "agreement": float(eff[i]),
            })
        out.sort(key=lambda s: -s["agreement"])
        return out[: self.n_surrogates]

    # ------------------------------------------------------------------ #
    # Growth
    # ------------------------------------------------------------------ #
    def _leaf(self, y: np.ndarray) -> dict:
        node = {
            "leaf": True,
            "n": y.size,
            "leaf_impurity": self._impurity(y),
        }
        if self.task == "classification":
            proba = np.bincount(y, minlength=self.n_classes).astype(float)
            node["proba"] = proba / proba.sum()
            node["value"] = int(np.argmax(proba))
        else:
            node["value"] = float(np.mean(y))
        return node

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        node = self._leaf(y)
        if (
            y.size < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or node["leaf_impurity"] <= 1e-12
        ):
            return node
        split = self._best_split(X, y)
        if split is None or split["gain"] <= 1e-12:
            return node

        j, t, ml = split["feature"], split["threshold"], split["missing_left"]
        col = X[:, j]
        left = (col <= t) | (np.isnan(col) & ml)
        right = ~left
        node = {
            "leaf": False,
            "n": y.size,
            "leaf_impurity": node["leaf_impurity"],
            "value": node["value"],
            "proba": node.get("proba"),
            "feature": j,
            "threshold": t,
            "missing_left": ml,
            "gain": split["gain"],
            "surrogates": self._surrogates(X, y, split) if self.use_surrogates else [],
            "left": self._grow(X[left], y[left], depth + 1),
            "right": self._grow(X[right], y[right], depth + 1),
        }
        return node

    # ------------------------------------------------------------------ #
    # Cost-complexity pruning
    # ------------------------------------------------------------------ #
    def _risk(self, node: dict) -> float:
        if node["leaf"]:
            return node["n"] / self.n_total * node["leaf_impurity"]
        return self._risk(node["left"]) + self._risk(node["right"])

    def _leaves(self, node: dict) -> int:
        if node["leaf"]:
            return 1
        return self._leaves(node["left"]) + self._leaves(node["right"])

    @staticmethod
    def _as_leaf(node: dict) -> dict:
        return {
            "leaf": True,
            "n": node["n"],
            "leaf_impurity": node["leaf_impurity"],
            "value": node["value"],
            "proba": node.get("proba"),
        }

    def _prune(self, node: dict, alpha: float) -> dict:
        while True:
            best = None

            def walk(n, parent, side):
                nonlocal best
                if n["leaf"]:
                    return
                leaves = self._leaves(n)
                if leaves > 1:
                    g = (n["n"] / self.n_total * n["leaf_impurity"]
                         - self._risk(n)) / (leaves - 1)
                    if best is None or g < best[0]:
                        best = (g, parent, side, n)
                walk(n["left"], n, "left")
                walk(n["right"], n, "right")

            walk(node, None, None)
            if best is None or best[0] > alpha:
                break
            _, parent, side, target = best
            leaf = self._as_leaf(target)
            if parent is None:
                node = leaf
            else:
                parent[side] = leaf
        return node

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> CART:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_total = y.size
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes = self.classes_.size
        self.root = self._grow(X, y, 0)
        if self.ccp_alpha > 0.0 or self.variant == "cost_complexity_pruning":
            self.root = self._prune(self.root, self.ccp_alpha)
        return self

    def _walk(self, x: np.ndarray) -> dict:
        node = self.root
        while not node["leaf"]:
            j = node["feature"]
            if np.isnan(x[j]):
                left = node["missing_left"]
                for s in node["surrogates"]:
                    if not np.isnan(x[s["feature"]]):
                        side_left = x[s["feature"]] <= s["threshold"]
                        left = bool(side_left) != s["flip"]
                        break
            else:
                left = x[j] <= node["threshold"]
            node = node["left"] if left else node["right"]
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.array([self._walk(x)["value"] for x in X])
        if self.task == "classification":
            return self.classes_[out.astype(int)]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is classification only")
        X = np.asarray(X, dtype=float)
        return np.array([self._walk(x)["proba"] for x in X])

    @property
    def n_leaves(self) -> int:
        return self._leaves(self.root)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- classification: XOR is not linearly separable ---
    n = 400
    X = rng.normal(size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    tree_g = CART("gini", max_depth=None).fit(X, y)
    tree_e = CART("entropy", max_depth=None).fit(X, y)
    acc_g = float(np.mean(tree_g.predict(X) == y))
    acc_e = float(np.mean(tree_e.predict(X) == y))
    assert acc_g > 0.97 and acc_e > 0.97, (acc_g, acc_e)
    assert tree_g.n_leaves <= 8, tree_g.n_leaves
    print(f"cart: XOR train acc gini {acc_g:.3f} ({tree_g.n_leaves} leaves), "
          f"entropy {acc_e:.3f} ({tree_e.n_leaves} leaves)")

    # --- regression: variance and Friedman MSE rank splits identically ---
    def f(x):
        return np.sin(1.5 * x[:, 0]) + 0.5 * x[:, 1] ** 2

    Xr = rng.uniform(-2.0, 2.0, size=(n, 2))
    yr = f(Xr) + rng.normal(0.0, 0.1, size=n)
    t_var = CART("variance", max_depth=6, min_samples_leaf=5).fit(Xr, yr)
    t_fri = CART("friedman_mse", max_depth=6, min_samples_leaf=5).fit(Xr, yr)
    pred_var = t_var.predict(Xr)
    assert np.allclose(pred_var, t_fri.predict(Xr)), "criteria must rank splits the same"
    rmse = float(np.sqrt(np.mean((pred_var - yr) ** 2)))
    assert rmse < 0.25, rmse
    print(f"cart: regression RMSE {rmse:.3f}, variance == friedman_mse predictions")

    # --- cost-complexity pruning: fewer leaves, better test error on noisy labels ---
    Xn = rng.normal(size=(n, 2))
    yn = ((Xn[:, 0] > 0) ^ (Xn[:, 1] > 0)).astype(int)
    yn[rng.random(n) < 0.15] ^= 1                     # 15% label noise
    Xt = rng.normal(size=(1000, 2))
    yt = ((Xt[:, 0] > 0) ^ (Xt[:, 1] > 0)).astype(int)
    full = CART("cost_complexity_pruning", max_depth=8, ccp_alpha=0.0).fit(Xn, yn)
    pruned = CART("cost_complexity_pruning", max_depth=8, ccp_alpha=0.02).fit(Xn, yn)
    assert pruned.n_leaves < full.n_leaves, (pruned.n_leaves, full.n_leaves)
    err_full = 1.0 - np.mean(full.predict(Xn) == yn)
    err_pruned = 1.0 - np.mean(pruned.predict(Xn) == yn)
    test_full = 1.0 - np.mean(full.predict(Xt) == yt)
    test_pruned = 1.0 - np.mean(pruned.predict(Xt) == yt)
    assert err_pruned >= err_full
    assert test_pruned < test_full, (test_full, test_pruned)
    print(f"cart: ccp leaves {full.n_leaves} -> {pruned.n_leaves}, "
          f"train error {err_full:.3f} -> {err_pruned:.3f}, "
          f"test error {test_full:.3f} -> {test_pruned:.3f}")

    # --- surrogate splits: recover the split when x0 is missing ---
    Xs = np.column_stack([X[:, 0], X[:, 0] + 0.02 * rng.normal(size=n)])
    ys = (Xs[:, 0] > 0).astype(int)
    with_sur = CART("surrogate_splits", max_depth=2).fit(Xs, ys)
    without = CART("surrogate_splits", max_depth=2, n_surrogates=0).fit(Xs, ys)
    Xmiss = Xs.copy()
    Xmiss[:, 0] = np.nan
    acc_sur = np.mean(with_sur.predict(Xmiss) == ys)
    acc_def = np.mean(without.predict(Xmiss) == ys)
    assert acc_sur > 0.9, acc_sur
    assert acc_def < 0.6, acc_def
    print(f"cart: surrogate acc {acc_sur:.3f} vs default {acc_def:.3f} when x0 missing")

    # --- missing values during training ---
    Xm = X.copy()
    mask = rng.random(n) < 0.25
    Xm[mask, 0] = np.nan
    tree_m = CART("missing_values", max_depth=4).fit(Xm, y)
    acc_m = np.mean(tree_m.predict(Xm) == y)
    assert acc_m > 0.85, acc_m
    print(f"cart: trained with 25% missing x0, train acc {acc_m:.3f}")

    # --- depth controls the bias/variance trade-off ---
    shallow = CART("gini", max_depth=1).fit(X, y)
    deep = CART("gini", max_depth=None).fit(X, y)
    assert np.mean(shallow.predict(X) == y) < np.mean(deep.predict(X) == y)
    print("cart demo passed")


if __name__ == "__main__":
    main()
