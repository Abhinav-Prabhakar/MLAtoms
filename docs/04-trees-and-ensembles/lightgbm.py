"""LightGBM — histogram boosting with the systems tricks that make it fast.

Chapter:   04-trees-and-ensembles
Companion: lightgbm.md
Tier 1 · 7 variants: goss, efb, leaf_wise, histogram, dart, categorical, monotone

LightGBM keeps XGBoost's second-order objective but changes how the tree is grown
and how the data is represented:

- **Histogram splits.** Bin every feature once; a node's split search is a pass over
  bins, not over rows.
- **Leaf-wise growth.** Split the leaf with the largest gain, not every leaf at the
  current depth. Fewer leaves reach a lower loss.
- **GOSS.** Keep the large-gradient rows, sample the rest, and reweight so the
  gradient estimate stays unbiased.
- **EFB.** Bundle mutually exclusive sparse features into one column.
- **Categorical splits.** Group categories by their gradient statistics instead of
  one-hot encoding.
- **Monotone constraints.** Clip leaf values to path bounds so the fitted function
  cannot decrease in a feature you declare increasing.

The implementation is compact but keeps the defining mechanism of each variant.
"""

from __future__ import annotations

import heapq

import numpy as np

VARIANTS = frozenset(
    ["goss", "efb", "leaf_wise", "histogram", "dart", "categorical", "monotone"]
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LightGBM:
    """Histogram gradient boosting with LightGBM's main variants."""

    def __init__(
        self,
        variant: str = "histogram",
        *,
        task: str = "regression",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = 6,
        min_samples_leaf: int = 1,
        reg_lambda: float = 1.0,
        max_bins: int = 32,
        subsample: float = 1.0,
        colsample: float = 1.0,
        top_rate: float = 0.2,
        other_rate: float = 0.1,
        drop_rate: float = 0.1,
        categorical_features: tuple[int, ...] = (),
        monotone: dict[int, int] | None = None,
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
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.drop_rate = drop_rate if variant == "dart" else 0.0
        self.categorical = set(categorical_features)
        self.monotone = monotone or {}
        self.rng = np.random.default_rng(random_state)
        self.estimators_: list[dict] = []
        self.train_loss_: list[float] = []
        self.n_rows_used_: list[int] = []

    # ------------------------------------------------------------------ #
    # EFB
    # ------------------------------------------------------------------ #
    def _efb(self, X: np.ndarray):
        d = X.shape[1]
        dense = [j for j in range(d) if not (
            np.mean(X[:, j] != 0.0) < 0.5 and np.all(X[:, j] >= 0.0))]
        sparse = [j for j in range(d) if j not in dense]
        bundles: list[list[int]] = []
        for j in sparse:
            for b in bundles:
                conflict = np.mean((X[:, j] != 0.0) & np.any(X[:, b] != 0.0, axis=1))
                if conflict <= 0.01:
                    b.append(j)
                    break
            else:
                bundles.append([j])
        cols, mapping = [], []
        for j in dense:
            cols.append(X[:, j])
            mapping.append(("single", j))
        for b in bundles:
            col = np.zeros(X.shape[0])
            feats = []
            offset = 0.0
            for j in b:
                col = np.where(X[:, j] != 0.0, X[:, j] + offset, col)
                feats.append((j, offset))
                offset += float(X[:, j].max()) + 1.0
            cols.append(col)
            mapping.append(("bundle", feats))
        return np.column_stack(cols), mapping

    def _apply_efb(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for kind, payload in self.efb_map_:
            if kind == "single":
                cols.append(X[:, payload])
            else:
                col = np.zeros(X.shape[0])
                for j, off in payload:
                    col = np.where(X[:, j] != 0.0, X[:, j] + off, col)
                cols.append(col)
        return np.column_stack(cols)

    # ------------------------------------------------------------------ #
    # Histograms
    # ------------------------------------------------------------------ #
    def _edges(self, X):
        out = []
        for j in range(X.shape[1]):
            v = X[:, j]
            v = v[~np.isnan(v)]
            if v.size < 2:
                out.append(np.array([]))
            else:
                out.append(np.unique(np.quantile(
                    v, np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1])))
        return out

    def _leaf_value(self, G, H) -> float:
        return -G / (H + self.reg_lambda)

    def _split_gain(self, GL, HL, GR, HR, GP, HP):
        def score(G, H):
            return G * G / (H + self.reg_lambda)

        return 0.5 * (score(GL, HL) + score(GR, HR) - score(GP, HP))

    def _hist_split(self, X, g, h, idx, j):
        col = X[idx, j]
        obs = ~np.isnan(col)
        n_obs = int(obs.sum())
        if n_obs < 2:
            return None
        rows = idx[obs]
        v = col[obs]
        edges = self.edges_[j]
        b = np.searchsorted(edges, v, side="right")
        nb = edges.size + 1
        Gb = np.bincount(b, weights=g[rows], minlength=nb)
        Hb = np.bincount(b, weights=h[rows], minlength=nb)
        Nb = np.bincount(b, minlength=nb)
        cG, cH, cN = np.cumsum(Gb), np.cumsum(Hb), np.cumsum(Nb)
        if edges.size == 0:
            return None
        GL, HL, n_l = cG[:-1], cH[:-1], cN[:-1].astype(float)
        GR, HR, n_r = cG[-1] - GL, cH[-1] - HL, n_obs - n_l
        GP, HP = float(g[idx].sum()), float(h[idx].sum())
        Gm = float(g[idx[~obs]].sum()) if (~obs).any() else 0.0
        Hm = float(h[idx[~obs]].sum()) if (~obs).any() else 0.0
        n_m = float(idx.size - n_obs)
        gains = [
            (self._split_gain(GL, HL, GR, HR,
                              float(g[rows].sum()), float(h[rows].sum())),
             (n_l >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf)
             & (n_m == 0), True),
            (self._split_gain(GL + Gm, HL + Hm, GR, HR, GP, HP),
             ((n_l + n_m) >= self.min_samples_leaf) & (n_r >= self.min_samples_leaf),
             True),
            (self._split_gain(GL, HL, GR + Gm, HR + Hm, GP, HP),
             (n_l >= self.min_samples_leaf) & ((n_r + n_m) >= self.min_samples_leaf),
             False),
        ]
        best = None
        for gain, valid, missing_left in gains:
            if not valid.any():
                continue
            k = int(np.argmax(np.where(valid, gain, -np.inf)))
            if best is None or gain[k] > best["gain"]:
                best = {"feature": j, "threshold": float(edges[k]),
                        "gain": float(gain[k]), "missing_left": bool(missing_left)}
        if best is not None:
            # optimal child values, used by monotone checks
            k = int(np.searchsorted(edges, best["threshold"], side="left"))
            best["wl"] = float(self._leaf_value(GL[k], HL[k]))
            best["wr"] = float(self._leaf_value(GR[k], HR[k]))
        return best

    def _cat_split(self, X, g, h, idx, j):
        col = X[idx, j]
        obs = ~np.isnan(col)
        if int(obs.sum()) < 2:
            return None
        rows = idx[obs]
        cats = np.unique(col[obs]).astype(int)
        if cats.size < 2:
            return None
        Gc = np.array([g[rows][col[obs].astype(int) == c].sum() for c in cats])
        Hc = np.array([h[rows][col[obs].astype(int) == c].sum() for c in cats])
        Nc = np.array([(col[obs].astype(int) == c).sum() for c in cats], dtype=float)
        order = np.argsort(Gc / (Hc + self.reg_lambda), kind="mergesort")
        cats, Gc, Hc, Nc = cats[order], Gc[order], Hc[order], Nc[order]
        cG, cH, cN = np.cumsum(Gc), np.cumsum(Hc), np.cumsum(Nc)
        GP, HP = float(g[idx].sum()), float(h[idx].sum())
        n_m = float(idx.size - rows.size)
        Gm = float(g[idx[~obs]].sum()) if n_m else 0.0
        Hm = float(h[idx[~obs]].sum()) if n_m else 0.0
        best = None
        for k in range(cats.size - 1):
            nl = cN[k]
            nr = cN[-1] - nl
            if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                continue
            gl = self._split_gain(cG[k] + (Gm if n_m else 0), cH[k] + (Hm if n_m else 0),
                                  cG[-1] - cG[k], cH[-1] - cH[k], GP, HP)
            gr = self._split_gain(cG[k], cH[k],
                                  cG[-1] - cG[k] + Gm, cH[-1] - cH[k] + Hm, GP, HP)
            for gain, missing_left in ((gl, True), (gr, False)):
                if best is None or gain > best["gain"]:
                    best = {"feature": j, "categories": cats[: k + 1].copy(),
                            "gain": float(gain), "missing_left": missing_left,
                            "wl": float(self._leaf_value(cG[k], cH[k])),
                            "wr": float(self._leaf_value(cG[-1] - cG[k],
                                                         cH[-1] - cH[k]))}
        return best

    def _best_split(self, X, g, h, idx, features):
        best = None
        for j in features:
            cand = (self._cat_split(X, g, h, idx, j) if j in self.categorical
                    else self._hist_split(X, g, h, idx, j))
            if cand is not None and (best is None or cand["gain"] > best["gain"]):
                best = cand
        return best

    # ------------------------------------------------------------------ #
    # Growth
    # ------------------------------------------------------------------ #
    def _leaf(self, idx, g, h, lo=-np.inf, hi=np.inf):
        v = float(np.clip(self._leaf_value(float(g[idx].sum()),
                                           float(h[idx].sum())), lo, hi))
        return {"leaf": True, "n": idx.size, "value": v}

    def _split_ok(self, split, lo, hi):
        """Monotone constraint and the child bounds it implies."""
        j = split["feature"]
        sign = self.monotone.get(j, 0)
        if sign == 0:
            return True, (lo, hi), (lo, hi)
        if sign > 0 and split["wl"] > split["wr"]:
            return False, None, None
        if sign < 0 and split["wl"] < split["wr"]:
            return False, None, None
        if sign > 0:
            return True, (lo, min(hi, split["wr"])), (max(lo, split["wl"]), hi)
        return True, (max(lo, split["wr"]), hi), (lo, min(hi, split["wl"]))

    def _grow(self, X, g, h, idx, depth, features, lo=-np.inf, hi=np.inf):
        if (depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf
                or idx.size < 2):
            return self._leaf(idx, g, h, lo, hi)
        split = self._best_split(X, g, h, idx, features)
        if split is None or split["gain"] <= 0.0:
            return self._leaf(idx, g, h, lo, hi)
        ok, lb, rb = self._split_ok(split, lo, hi)
        if not ok:
            return self._leaf(idx, g, h, lo, hi)
        col = X[idx, split["feature"]]
        if "categories" in split:
            left = np.isin(col, split["categories"])
            if np.isnan(col).any():
                left |= np.isnan(col) & split["missing_left"]
        else:
            left = (col <= split["threshold"]) | (np.isnan(col) & split["missing_left"])
        li, ri = idx[left], idx[~left]
        if li.size == 0 or ri.size == 0:
            return self._leaf(idx, g, h, lo, hi)
        node = {"leaf": False, "n": idx.size, "feature": split["feature"],
                "threshold": split.get("threshold"),
                "categories": split.get("categories"),
                "missing_left": split["missing_left"], "gain": split["gain"]}
        node["left"] = self._grow(X, g, h, li, depth + 1, features, *lb)
        node["right"] = self._grow(X, g, h, ri, depth + 1, features, *rb)
        return node

    def _grow_leafwise(self, X, g, h, features):
        root = self._leaf(np.arange(X.shape[0]), g, h)
        heap: list[tuple[float, int, dict, np.ndarray, int, float, float, dict]] = []
        counter = 0

        def push(node, idx, depth, lo, hi):
            nonlocal counter
            if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf:
                return
            split = self._best_split(X, g, h, idx, features)
            if split is None or split["gain"] <= 0.0:
                return
            ok, lb, rb = self._split_ok(split, lo, hi)
            if not ok:
                return
            heapq.heappush(heap, (-split["gain"], counter, node, idx, depth, lo, hi, split))
            counter += 1

        push(root, np.arange(X.shape[0]), 0, -np.inf, np.inf)
        leaves = 1
        while heap and leaves < self.num_leaves:
            _, _, node, idx, depth, lo, hi, split = heapq.heappop(heap)
            ok, lb, rb = self._split_ok(split, lo, hi)
            if not ok:
                continue
            col = X[idx, split["feature"]]
            if "categories" in split:
                left = np.isin(col, split["categories"])
                if np.isnan(col).any():
                    left |= np.isnan(col) & split["missing_left"]
            else:
                left = ((col <= split["threshold"])
                        | (np.isnan(col) & split["missing_left"]))
            li, ri = idx[left], idx[~left]
            if li.size == 0 or ri.size == 0:
                continue
            node.clear()
            node.update({"leaf": False, "n": idx.size, "feature": split["feature"],
                         "threshold": split.get("threshold"),
                         "categories": split.get("categories"),
                         "missing_left": split["missing_left"], "gain": split["gain"]})
            node["left"] = self._leaf(li, g, h, *lb)
            node["right"] = self._leaf(ri, g, h, *rb)
            leaves += 1
            push(node["left"], li, depth + 1, *lb)
            push(node["right"], ri, depth + 1, *rb)
        return root

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def _goss(self, g: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        top = max(1, int(self.top_rate * n))
        rest = max(1, int(self.other_rate * n))
        order = np.argsort(-np.abs(g), kind="mergesort")
        big, small = order[:top], order[top:]
        take = self.rng.choice(small, size=min(rest, small.size), replace=False)
        idx = np.concatenate([big, take])
        w = np.ones(idx.size)
        w[top:] = (1.0 - self.top_rate) / self.other_rate
        return idx, w

    def _tree_predict(self, tree, X):
        out = np.empty(X.shape[0])
        for i, x in enumerate(X):
            node = tree
            while not node["leaf"]:
                j = node["feature"]
                if np.isnan(x[j]):
                    go_left = node["missing_left"]
                elif node["categories"] is not None:
                    go_left = int(x[j]) in set(node["categories"].tolist())
                else:
                    go_left = x[j] <= node["threshold"]
                node = node["left"] if go_left else node["right"]
            out[i] = node["value"]
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBM:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.variant == "efb":
            X, self.efb_map_ = self._efb(X)
        else:
            self.efb_map_ = None
        n, d = X.shape
        self.n_features_ = d
        self.edges_ = self._edges(X)
        self.estimators_ = []
        self.train_loss_ = []
        self.n_rows_used_ = []
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
                scale = 1.0 / (1.0 - self.drop_rate)
                F = np.full(n, base)
                for tree, k in zip(self.estimators_, keep, strict=True):
                    if k:
                        F += scale * self.lr * self._tree_predict(tree, X)
            if self.task == "binary":
                p = _sigmoid(F)
                g, h = p - y, p * (1.0 - p)
            else:
                g, h = F - y, np.ones_like(F)

            if self.variant == "goss":
                idx, w = self._goss(g, n)
                g, h = g.copy(), h.copy()
                g[idx] *= w
                h[idx] *= w
            elif self.subsample < 1.0:
                k = max(2, int(self.subsample * n))
                idx = self.rng.choice(n, size=k, replace=False)
            else:
                idx = np.arange(n)
            self.n_rows_used_.append(idx.size)
            m = max(1, int(self.colsample * d))
            features = (np.arange(d) if m >= d
                        else self.rng.choice(d, size=m, replace=False))
            if self.variant == "leaf_wise":
                tree = self._grow_leafwise(X, g, h, features)
            else:
                tree = self._grow(X, g, h, idx, 0, features)
            self.estimators_.append(tree)
            if self.drop_rate == 0.0:
                F += self.lr * self._tree_predict(tree, X)
            self.train_loss_.append(self._loss(y, self._full_raw(X)))
        return self

    def _full_raw(self, X):
        F = np.full(X.shape[0], self._base)
        for tree in self.estimators_:
            F += self.lr * self._tree_predict(tree, X)
        return F

    def _loss(self, y, F):
        if self.task == "binary":
            return float(np.mean(np.logaddexp(0.0, F) - y * F))
        return float(np.mean((y - F) ** 2))

    def predict_proba(self, X):
        if self.task != "binary":
            raise ValueError("predict_proba is binary only")
        X = np.asarray(X, dtype=float)
        if self.efb_map_ is not None:
            X = self._apply_efb(X)
        p = _sigmoid(self._full_raw(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.efb_map_ is not None:
            X = self._apply_efb(X)
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

    # --- histogram: the baseline ---
    hist = LightGBM("histogram", n_estimators=120, learning_rate=0.1,
                    max_depth=4, min_samples_leaf=5,
                    random_state=1).fit(X, y)
    r2_hist = r2(hist.predict(Xt))
    assert r2_hist > 0.9, r2_hist
    print(f"lightgbm: histogram R^2 {r2_hist:.3f}, loss "
          f"{hist.train_loss_[0]:.3f} -> {hist.train_loss_[-1]:.3f}")

    # --- leaf-wise reaches a lower loss with the same leaf budget ---
    def n_leaves(tree):
        if tree["leaf"]:
            return 1
        return n_leaves(tree["left"]) + n_leaves(tree["right"])

    level = LightGBM("histogram", n_estimators=80, learning_rate=0.1,
                     max_depth=3, min_samples_leaf=5, random_state=2).fit(X, y)
    leaf = LightGBM("leaf_wise", n_estimators=80, learning_rate=0.1,
                    num_leaves=8, max_depth=6, min_samples_leaf=5,
                    random_state=2).fit(X, y)
    assert n_leaves(leaf.estimators_[0]) == n_leaves(level.estimators_[0])
    assert leaf.train_loss_[-1] < level.train_loss_[-1], (
        leaf.train_loss_[-1], level.train_loss_[-1])
    print(f"lightgbm: leaf_wise loss {leaf.train_loss_[-1]:.4f} "
          f"vs level-wise {level.train_loss_[-1]:.4f} "
          f"at {n_leaves(leaf.estimators_[0])} leaves")

    # --- GOSS uses a fraction of the rows ---
    goss = LightGBM("goss", n_estimators=120, learning_rate=0.1, max_depth=4,
                    min_samples_leaf=5, top_rate=0.2, other_rate=0.1,
                    random_state=3).fit(X, y)
    assert max(goss.n_rows_used_) < n
    assert r2(goss.predict(Xt)) > 0.88
    print(f"lightgbm: GOSS uses {goss.n_rows_used_[0]}/{n} rows, "
          f"R^2 {r2(goss.predict(Xt)):.3f}")

    # --- DART ---
    dart = LightGBM("dart", n_estimators=100, learning_rate=0.1, max_depth=4,
                    min_samples_leaf=5, drop_rate=0.15,
                    random_state=4).fit(X, y)
    assert r2(dart.predict(Xt)) > 0.85
    print(f"lightgbm: DART R^2 {r2(dart.predict(Xt)):.3f}")

    # --- categorical split: one column with 6 levels ---
    nc = 400
    cat = rng.integers(0, 6, size=nc)
    Xc = np.column_stack([rng.normal(size=nc), cat.astype(float)])
    yc = np.array([0.0, 1.5, -1.0, 0.3, 2.0, -0.5])[cat] + 0.2 * Xc[:, 0]
    yc += rng.normal(0.0, 0.2, size=nc)
    lc = LightGBM("categorical", task="regression", n_estimators=80,
                  learning_rate=0.15, max_depth=4, min_samples_leaf=5,
                  categorical_features=(1,), random_state=5).fit(Xc, yc)
    pred_c = lc.predict(Xc)
    r2_c = 1.0 - np.mean((pred_c - yc) ** 2) / np.var(yc)
    assert r2_c > 0.85, r2_c
    print(f"lightgbm: categorical R^2 {r2_c:.3f}")

    # --- EFB bundles sparse one-hot columns ---
    ne = 500
    base = rng.normal(size=(ne, 1))
    groups = rng.integers(0, 8, size=ne)
    Xe = np.column_stack([base, np.eye(8)[groups]])
    ye = groups.astype(float) + 0.1 * base[:, 0] + rng.normal(0, 0.2, size=ne)
    efb = LightGBM("efb", n_estimators=80, learning_rate=0.15, max_depth=4,
                   min_samples_leaf=5, random_state=6).fit(Xe, ye)
    assert efb.n_features_ < Xe.shape[1]
    r2_e = 1.0 - np.mean((efb.predict(Xe) - ye) ** 2) / np.var(ye)
    assert r2_e > 0.85, r2_e
    print(f"lightgbm: EFB {Xe.shape[1]} -> {efb.n_features_} features, R^2 {r2_e:.3f}")

    # --- monotone constraint ---
    nm = 400
    Xm = rng.uniform(-2.0, 2.0, size=(nm, 1))
    ym = 2.0 + Xm[:, 0] ** 3 + rng.normal(0.0, 0.4, size=nm)
    mono = LightGBM("monotone", n_estimators=100, learning_rate=0.1, max_depth=4,
                    min_samples_leaf=5, monotone={0: 1},
                    random_state=7).fit(Xm, ym)
    grid = np.linspace(-2.0, 2.0, 400).reshape(-1, 1)
    pred = mono.predict(grid)
    assert np.all(np.diff(pred) >= -1e-12), "monotone constraint violated"
    print(f"lightgbm: monotone predictions nondecreasing "
          f"({pred[0]:.2f} -> {pred[-1]:.2f})")

    print("lightgbm demo passed")


if __name__ == "__main__":
    main()
