"""Interpretability — attributing a prediction to its inputs.

Chapter:   20-evaluation
Companion: interpretability.md
Tier 1 · 9 variants: permutation, shap, kernel_shap, tree_shap, pdp, ice, ale,
integrated_gradients, lime

There are two different questions hiding behind "why did the model say that":

    how much does feature j matter to the model overall?   -> permutation, pdp/ice/ale
    how much did feature j matter to *this* prediction?    -> shap family, lime, ig

The first is a property of the fitted function; the second is a property of one
row. Feature importance answers questions about the model; attribution answers
questions about a decision, and attribution is what a user asks for after a
surprising prediction.

    permutation          drop-column, model-agnostic, measures the model
    shap                 exact Shapley values over a background distribution
    kernel_shap          the same values by weighted least squares, O(2^d) -> O(d^2)
    tree_shap            the same idea for a tree, but exponentially cheaper
    pdp / ice            average effect of one feature over the population
    ale                  local effects, immune to correlated-feature bias
    integrated_gradients path integral of the gradient, for differentiable models
    lime                 local linear surrogate fitted around one row

Every variant here is checked against something exact in the demo: Shapley
efficiency (the attributions sum to the prediction minus the baseline), a
brute-force coalition enumeration, an analytic gradient, or a known linear
coefficient vector. Interpretability is the one area where a plausible-looking
plot is the normal failure mode, so the assertions are the point.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(
    ["permutation", "shap", "kernel_shap", "tree_shap", "pdp", "ice", "ale",
     "integrated_gradients", "lime"]
)

EPS = 1e-12


# --------------------------------------------------------------------------- #
# A minimal CART regressor: small enough to read, complete enough for TreeSHAP
# --------------------------------------------------------------------------- #
class _Tree:
    """Depth-limited regression tree stored in flat arrays.

    `cover_` is the number of training rows in each node — TreeSHAP needs it,
    because the conditional expectation at a split that is *not* in the
    coalition is the cover-weighted average of the two children.
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 5,
                 max_features: int | None = None, random_state: int = 0):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _Tree:
        self.feature_: list[int] = []
        self.threshold_: list[float] = []
        self.left_: list[int] = []
        self.right_: list[int] = []
        self.value_: list[float] = []
        self.cover_: list[float] = []
        self._rng = np.random.default_rng(self.random_state)
        self._build(np.asarray(X, dtype=float), np.asarray(y, dtype=float), 0)
        return self

    # -- construction ---------------------------------------------------- #
    def _add(self, y: np.ndarray) -> int:
        node = len(self.feature_)
        self.feature_.append(-1)
        self.threshold_.append(0.0)
        self.left_.append(-1)
        self.right_.append(-1)
        self.value_.append(float(y.mean()) if y.size else 0.0)
        self.cover_.append(float(y.size))
        return node

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> int:
        node = self._add(y)
        n = y.size
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return node
        sse, feature, threshold = self._best_split(X, y)
        if feature < 0:
            return node
        mask = X[:, feature] <= threshold
        self.feature_[node] = feature
        self.threshold_[node] = threshold
        self.left_[node] = self._build(X[mask], y[mask], depth + 1)
        self.right_[node] = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
        """Exhaustive or subsampled split search, vectorised over sorted rows."""
        n, d = X.shape
        best = (float(np.sum((y - y.mean()) ** 2)), -1, 0.0)
        m = self.max_features if self.max_features is not None else d
        feats = (np.arange(d) if m >= d
                 else self._rng.choice(d, size=m, replace=False))
        for j in feats:
            order = np.argsort(X[:, j], kind="stable")
            xs, ys = X[order, j], y[order]
            c1 = np.cumsum(ys)
            c2 = np.cumsum(ys * ys)
            total1, total2 = c1[-1], c2[-1]
            left_n = np.arange(1, n, dtype=float)
            right_n = n - left_n
            left_sse = c2[:-1] - c1[:-1] ** 2 / left_n
            right_sse = (total2 - c2[:-1]) - (total1 - c1[:-1]) ** 2 / right_n
            valid = ((xs[1:] > xs[:-1])
                     & (left_n >= self.min_samples_leaf)
                     & (right_n >= self.min_samples_leaf))
            if not valid.any():
                continue
            total = np.where(valid, left_sse + right_sse, np.inf)
            i = int(np.argmin(total))
            if total[i] < best[0]:
                best = (float(total[i]), int(j), float(0.5 * (xs[i] + xs[i + 1])))
        return best

    # -- use -------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0])
        for r in range(X.shape[0]):
            node = 0
            while self.feature_[node] >= 0:
                j = self.feature_[node]
                node = (self.left_[node] if X[r, j] <= self.threshold_[node]
                        else self.right_[node])
            out[r] = self.value_[node]
        return out


class _Forest:
    """Bagged regression trees — a stand-in for any black-box model."""

    def __init__(self, n_trees: int = 20, max_depth: int = 4,
                 min_samples_leaf: int = 5, random_state: int = 0):
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _Forest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for t in range(self.n_trees):
            rows = rng.integers(0, n, size=n)
            tree = _Tree(self.max_depth, self.min_samples_leaf, random_state=t)
            self.trees_.append(tree.fit(X[rows], y[rows]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([t.predict(X) for t in self.trees_], axis=0)


# --------------------------------------------------------------------------- #
# Shapley plumbing shared by three variants
# --------------------------------------------------------------------------- #
def shapley_coefficients(d: int) -> np.ndarray:
    """`c[k] = k!(d-k-1)!/d!` — the weight of a coalition of size `k`."""
    return np.array([math.factorial(k) * math.factorial(d - k - 1)
                     / math.factorial(d) for k in range(d)])


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Accept anything with `predict`; also accept a bare callable."""
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=float).reshape(-1)
    return np.asarray(model(X), dtype=float).reshape(-1)


def _mask_bits(mask: int, d: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(d)], dtype=bool)


def tree_shap(tree: _Tree, x: np.ndarray, n_features: int) -> np.ndarray:
    """Exact TreeSHAP for one tree: path-dependent Shapley values.

    Walks every root-to-leaf path once. A leaf only survives a coalition `S`
    if `S` avoids the features where that leaf diverges from `x`, so each leaf
    contributes a single polynomial in the number of *agreeing* features that
    are inside `S`.
    """
    x = np.asarray(x, dtype=float)
    c = shapley_coefficients(n_features)
    phi = np.zeros(n_features)

    def poly(features: list[int], q: dict[int, float]) -> np.ndarray:
        """Coefficients of prod_{i in features} (Q_i + t), degree len(features)."""
        out = np.ones(1)
        for i in features:
            step = np.zeros(out.size + 1)
            step[:out.size] = out * q[i]        # feature i is not in S
            step[1:] += out                     # feature i is in S
            out = step
        return out

    def visit(node: int, feats: list[int], qs: list[float], diverge: set) -> None:
        j = tree.feature_[node]
        if j < 0:
            q: dict[int, float] = {}
            for f, qv in zip(feats, qs, strict=True):
                q[f] = q.get(f, 1.0) * qv
            distinct = list(q)
            n_path = len(distinct)
            n_off = n_features - n_path
            inside = [f for f in distinct if f not in diverge]
            # weight of a coalition that puts `k` of the agreeing features in S,
            # with the other d - n_path features free
            size = np.array([
                sum(math.comb(n_off, m) * c[k + m]
                    for m in range(n_off + 1) if k + m <= n_features - 1)
                for k in range(len(inside) + 1)
            ])
            value = tree.value_[node]
            # coalitions must avoid every divergence feature, whose factors stay in
            # the product no matter what -- collect them once
            forced = 1.0
            for f in diverge:
                forced *= q[f]
            for i in distinct:
                if i in diverge:
                    # forcing i into S kills this leaf entirely
                    total = float(np.dot(poly(inside, q), size))
                    phi[i] -= value * forced * total
                else:
                    others = [f for f in inside if f != i]
                    total = float(np.dot(poly(others, q), size[:len(others) + 1]))
                    phi[i] += value * forced * (1.0 - q[i]) * total
            return
        go_left = bool(x[j] <= tree.threshold_[node])
        cover = tree.cover_[node]
        for child, taken in ((tree.left_[node], go_left),
                             (tree.right_[node], not go_left)):
            if cover <= 0 or tree.cover_[child] <= 0:
                continue
            visit(child, feats + [j], qs + [tree.cover_[child] / cover],
                  diverge if taken else diverge | {j})

    visit(0, [], [], set())
    return phi


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
class Explainer:
    """Attribute model behaviour to features.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    n_repeats : int
        Column shuffles per feature in `permutation`.
    background : int
        Rows of the fitted data used as the background distribution for
        `shap`, `kernel_shap` and `lime`.
    n_samples : int
        Perturbations drawn by `lime`.
    steps : int
        Quadrature points along the `integrated_gradients` path.
    n_bins : int
        Bin count for `ale`.
    l2 : float
        Ridge strength of the `lime` surrogate.
    random_state : int
        Seed for every stochastic step.

    Notes
    -----
    `fit(X, y)` stores the data; which of the three entry points applies
    depends on the variant:

        importance(model)                  permutation
        contributions(model, x, ...)       shap, kernel_shap, tree_shap,
                                           integrated_gradients, lime
        effect(model, feature)             pdp, ice, ale
    """

    def __init__(self, variant: str = "permutation", *, n_repeats: int = 10,
                 background: int = 200, n_samples: int = 500, steps: int = 64,
                 n_bins: int = 20, l2: float = 1e-6, random_state: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.n_repeats = int(n_repeats)
        self.background = int(background)
        self.n_samples = int(n_samples)
        self.steps = int(steps)
        self.n_bins = int(n_bins)
        self.l2 = float(l2)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Explainer:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-d array")
        self.X_ = X
        self.y_ = None if y is None else np.asarray(y, dtype=float)
        if self.y_ is not None and self.y_.shape[0] != X.shape[0]:
            raise ValueError("X and y disagree on row count")
        self.n_features_ = X.shape[1]
        return self

    def _background(self) -> np.ndarray:
        if self.X_.shape[0] <= self.background:
            return self.X_
        rng = np.random.default_rng(self.random_state)
        idx = np.sort(rng.choice(self.X_.shape[0], self.background, replace=False))
        return self.X_[idx]

    def _coalition(self, model, x: np.ndarray, mask: np.ndarray) -> float:
        """`E[f(x_S, X_bg_{\\bar S})]` — the Shapley value function."""
        Z = self._background().copy()
        Z[:, mask] = x[mask]
        return float(np.mean(_predict(model, Z)))

    # ------------------------------------------------------------------ #
    # Global importance
    # ------------------------------------------------------------------ #
    def importance(self, model) -> np.ndarray:
        """Loss increase when a column is shuffled: positive means it matters."""
        if self.variant != "permutation":
            raise ValueError("importance() is the permutation variant")
        if self.y_ is None:
            raise ValueError("permutation importance needs y at fit time")
        X, y = self.X_, self.y_
        base = float(np.mean((y - _predict(model, X)) ** 2))
        rng = np.random.default_rng(self.random_state)
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            deltas = np.empty(self.n_repeats)
            for r in range(self.n_repeats):
                Z = X.copy()
                Z[:, j] = X[rng.permutation(X.shape[0]), j]
                deltas[r] = float(np.mean((y - _predict(model, Z)) ** 2)) - base
            out[j] = float(np.mean(deltas))
        return out

    # ------------------------------------------------------------------ #
    # Local attribution
    # ------------------------------------------------------------------ #
    def contributions(self, model, x: np.ndarray, *,
                      grad_fn=None, baseline=None) -> np.ndarray:
        """Per-feature attribution for one row, in prediction units."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_features_,):
            raise ValueError(f"x must have shape ({self.n_features_},)")
        v = self.variant
        if v == "shap":
            return self._exact_shapley(model, x)
        if v == "kernel_shap":
            return self._kernel_shap(model, x)
        if v == "tree_shap":
            return self._tree_shap(model, x)
        if v == "integrated_gradients":
            if grad_fn is None:
                raise ValueError("integrated_gradients needs grad_fn")
            return self._integrated_gradients(grad_fn, x, baseline)
        if v == "lime":
            return self._lime(model, x)
        raise ValueError(f"{v} has no local attribution; use effect() or importance()")

    def _exact_shapley(self, model, x: np.ndarray) -> np.ndarray:
        d = self.n_features_
        c = shapley_coefficients(d)
        value = np.array([self._coalition(model, x, _mask_bits(m, d))
                          for m in range(1 << d)])
        phi = np.zeros(d)
        for i in range(d):
            bit = 1 << i
            for mask in range(1 << d):
                if mask & bit:
                    continue
                k = bin(mask).count("1")
                phi[i] += c[k] * (value[mask | bit] - value[mask])
        return phi

    def _kernel_shap(self, model, x: np.ndarray) -> np.ndarray:
        d = self.n_features_
        if d < 2:
            return np.array([self._coalition(model, x, np.ones(d, bool))
                             - self._coalition(model, x, np.zeros(d, bool))])
        value = np.array([self._coalition(model, x, _mask_bits(m, d))
                          for m in range(1 << d)])
        total = value[-1] - value[0]
        rows = [m for m in range(1, (1 << d) - 1)]
        A = np.array([_mask_bits(m, d) for m in rows], dtype=float)
        target = value[rows] - value[0]
        k = A.sum(axis=1)
        weight = (d - 1) / (np.array([math.comb(d, int(ki)) for ki in k])
                            * k * (d - k))
        # impose efficiency exactly by eliminating the last coordinate
        reduced = A[:, :d - 1] - A[:, d - 1:d]
        rhs = target - A[:, d - 1] * total
        root = np.sqrt(weight)
        coef, *_ = np.linalg.lstsq(reduced * root[:, None], rhs * root, rcond=None)
        phi = np.empty(d)
        phi[:d - 1] = coef
        phi[d - 1] = total - float(coef.sum())
        return phi

    def _tree_shap(self, model, x: np.ndarray) -> np.ndarray:
        trees = getattr(model, "trees_", [model])
        phi = np.zeros(self.n_features_)
        for tree in trees:
            phi += tree_shap(tree, x, self.n_features_)
        return phi / len(trees)

    def _integrated_gradients(self, grad_fn, x: np.ndarray,
                              baseline: np.ndarray | None = None) -> np.ndarray:
        if baseline is None:
            baseline = self._background().mean(axis=0)
        base = np.asarray(baseline, dtype=float)
        delta = x - base
        alphas = (np.arange(self.steps) + 0.5) / self.steps   # midpoint rule
        grads = np.array([grad_fn(base + a * delta) for a in alphas])
        return delta * grads.mean(axis=0)

    def _lime(self, model, x: np.ndarray) -> np.ndarray:
        background = self._background()
        scale = background.std(axis=0)
        scale[scale <= 0] = 1.0
        rng = np.random.default_rng(self.random_state)
        Z = x + rng.normal(size=(self.n_samples, x.size)) * 0.5 * scale
        offset = Z - x
        target = _predict(model, Z) - float(_predict(model, x[None, :])[0])
        u = offset / scale
        weight = np.exp(-0.5 * np.sum(u * u, axis=1) / (0.75 ** 2 * x.size))
        A = offset * weight[:, None]
        gram = A.T @ offset
        ridge = self.l2 * float(np.trace(gram)) / x.size + EPS
        coef = np.linalg.solve(gram + ridge * np.eye(x.size), A.T @ target)
        return coef

    # ------------------------------------------------------------------ #
    # Feature effects
    # ------------------------------------------------------------------ #
    def effect(self, model, feature: int, grid=None) -> dict[str, np.ndarray]:
        """Population effect of one feature: `pdp`, `ice` or `ale`."""
        v = self.variant
        if v not in ("pdp", "ice", "ale"):
            raise ValueError(f"{v} has no feature effect")
        if not 0 <= feature < self.n_features_:
            raise ValueError("feature index out of range")
        if v == "ale":
            return self._ale(model, feature, grid)
        X = self.X_
        if grid is None:
            grid = np.quantile(X[:, feature], np.linspace(0.05, 0.95, 20))
        grid = np.unique(np.asarray(grid, dtype=float))
        curves = np.empty((X.shape[0], grid.size))
        for k, g in enumerate(grid):
            Z = X.copy()
            Z[:, feature] = g
            curves[:, k] = _predict(model, Z)
        out: dict[str, np.ndarray] = {"grid": grid, "mean": curves.mean(axis=0)}
        if v == "ice":
            out["curves"] = curves
        else:
            out["values"] = out["mean"]
        return out

    def _ale(self, model, feature: int, edges=None) -> dict[str, np.ndarray]:
        X = self.X_
        if edges is None:
            edges = np.unique(np.quantile(X[:, feature],
                                          np.linspace(0.0, 1.0, self.n_bins + 1)))
        edges = np.asarray(edges, dtype=float)
        if edges.size < 2:
            raise ValueError("ale needs at least two distinct edges")
        xj = X[:, feature]
        idx = np.clip(np.digitize(xj, edges[1:-1]), 0, edges.size - 2)
        delta = np.zeros(edges.size - 1)
        for k in range(edges.size - 1):
            rows = idx == k
            if not rows.any():
                continue
            Z = X[rows]
            lo = Z.copy()
            hi = Z.copy()
            lo[:, feature] = edges[k]
            hi[:, feature] = edges[k + 1]
            delta[k] = float(np.mean(_predict(model, hi) - _predict(model, lo)))
        curve = np.concatenate([[0.0], np.cumsum(delta)])
        counts = np.bincount(idx, minlength=edges.size - 1).astype(float)
        centre = float(np.sum(counts / counts.sum()
                              * (curve[:-1] + curve[1:]) / 2.0))
        return {"edges": edges, "values": curve - centre, "delta": delta,
                "counts": counts}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _tree_value(tree: _Tree, x: np.ndarray, mask: int, n_features: int) -> float:
    """Coalition value under the tree's path-dependent distribution."""

    def rec(node: int) -> float:
        j = tree.feature_[node]
        if j < 0:
            return tree.value_[node]
        if (mask >> j) & 1:
            left = x[j] <= tree.threshold_[node]
            return rec(tree.left_[node] if left else tree.right_[node])
        nl, nr = tree.cover_[tree.left_[node]], tree.cover_[tree.right_[node]]
        return (nl * rec(tree.left_[node]) + nr * rec(tree.right_[node])) / (nl + nr)

    return rec(0)


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 400, 4
    X = rng.normal(size=(n, d))
    y = (2.0 * X[:, 0] + X[:, 1] ** 2 - X[:, 2]
         + 0.5 * X[:, 0] * X[:, 2] + rng.normal(scale=0.1, size=n))
    model = _Forest(n_trees=25, max_depth=4, random_state=0).fit(X, y)
    x0 = np.array([1.0, 0.5, -0.5, 0.0])

    # --- permutation: a noise column is worth nothing ---
    perm = Explainer("permutation", n_repeats=10, random_state=0).fit(X, y)
    imp = perm.importance(model)
    noise = int(np.argmin(np.abs(imp)))
    assert noise == 3 and abs(imp[3]) < 0.25 * np.max(imp), imp
    assert int(np.argmax(imp)) == 0, imp
    print(f"permutation: importance {np.round(imp, 3).tolist()}, weakest "
          f"feature {noise} at {imp[noise]:+.3f}")

    # --- exact Shapley over 2^4 coalitions, with efficiency ---
    shap = Explainer("shap", background=200, random_state=0).fit(X)
    phi = shap.contributions(model, x0)
    base = float(np.mean(_predict(model, shap._background())))
    gap = abs(phi.sum() - (float(_predict(model, x0[None, :])[0]) - base))
    assert gap < 1e-8, gap
    print(f"shap: phi {np.round(phi, 3).tolist()} sums to "
          f"{phi.sum():+.4f} = prediction {_predict(model, x0[None, :])[0]:.4f}"
          f" - baseline {base:.4f}")

    # --- kernel_shap: same numbers by weighted least squares ---
    kernel = Explainer("kernel_shap", background=200, random_state=0).fit(X)
    phi_k = kernel.contributions(model, x0)
    delta = float(np.max(np.abs(phi_k - phi)))
    assert delta < 1e-8, delta
    print(f"kernel_shap: max deviation from exact Shapley {delta:.1e} "
          f"over {2 ** d - 2} coalitions")

    # --- tree_shap: exact against enumeration of all coalitions, per tree ---
    explainer_ts = Explainer("tree_shap").fit(X)
    tree = model.trees_[0]
    fast = tree_shap(tree, x0, d)
    values = np.array([_tree_value(tree, x0, m, d) for m in range(1 << d)])
    c = shapley_coefficients(d)
    brute = np.zeros(d)
    for i in range(d):
        for mask in range(1 << d):
            if mask >> i & 1:
                continue
            brute[i] += c[bin(mask).count("1")] * (values[mask | (1 << i)]
                                                   - values[mask])
    err = float(np.max(np.abs(fast - brute)))
    assert err < 1e-10, (fast, brute)
    assert abs(fast.sum() - (values[-1] - values[0])) < 1e-10
    phi_t = explainer_ts.contributions(model, x0)
    tree_base = np.mean([_tree_value(t, x0, 0, d) for t in model.trees_])
    total = float(_predict(model, x0[None, :])[0]) - tree_base
    assert abs(phi_t.sum() - total) < 1e-10, (phi_t.sum(), total)
    print(f"tree_shap: max deviation from brute force {err:.1e}; forest phi "
          f"{np.round(phi_t, 3).tolist()} sums to {phi_t.sum():+.4f}")

    # --- pdp: recovers a convex squared effect, flat on a noise column ---
    pdp = Explainer("pdp").fit(X)
    eff1 = pdp.effect(model, 1)
    second = np.diff(eff1["values"], 2)
    assert np.mean(second) > 0.0, second.mean()
    eff3 = pdp.effect(model, 3)
    flat = np.ptp(eff3["values"])
    assert flat < 0.05 * np.ptp(eff1["values"]), (flat, np.ptp(eff1["values"]))
    print(f"pdp: x1 spans {np.ptp(eff1['values']):.3f} and is convex "
          f"({np.mean(second):+.4f} per unit); noise column x3 spans {flat:.3f}")

    # --- ice: additive features give parallel curves, interactions do not ---
    def additive(Z: np.ndarray) -> np.ndarray:
        return 2.0 * Z[:, 0] + Z[:, 1] ** 2 - Z[:, 2]

    def interaction(Z: np.ndarray) -> np.ndarray:
        return 5.0 * Z[:, 0] * Z[:, 2]

    ice = Explainer("ice").fit(X)
    add = ice.effect(additive, 1)["curves"]
    inter = ice.effect(interaction, 2)["curves"]
    spread_add = float(np.ptp(add[:, -1] - add[:, 0]))
    spread_int = float(np.ptp(inter[:, -1] - inter[:, 0]))
    assert spread_add < 1e-10, spread_add
    assert spread_int > 5.0, spread_int
    print(f"ice: x1 slope spread {spread_add:.1e} (additive, parallel curves) vs "
          f"x2 {spread_int:.2f} (slope is 5*x0)")

    # --- ale: with a correlated feature, pdp averages away the interaction ---
    Xc = X.copy()
    Xc[:, 2] = 0.9 * Xc[:, 0] + 0.1 * rng.normal(size=n)

    def product(Z: np.ndarray) -> np.ndarray:
        return Z[:, 0] * Z[:, 2]

    ale = Explainer("ale", n_bins=10).fit(Xc)
    ale_x0 = ale.effect(product, 0)
    pdp_x0 = Explainer("pdp").fit(Xc).effect(product, 0)
    a_span = float(np.ptp(ale_x0["values"]))
    p_span = float(np.ptp(pdp_x0["values"]))
    assert a_span > 5.0 * p_span, (a_span, p_span)
    assert np.mean(np.diff(ale_x0["values"], 2)) > 0.0
    counts = ale_x0["counts"] / ale_x0["counts"].sum()
    middle = (ale_x0["values"][:-1] + ale_x0["values"][1:]) / 2.0
    assert abs(float(np.sum(counts * middle))) < 1e-10
    print(f"ale: x0*x2 with x2 = 0.9*x0 -> ale spans {a_span:.2f} (convex) while "
          f"pdp spans only {p_span:.2f}; centred mean "
          f"{float(np.sum(counts * middle)):+.1e}")

    # --- integrated gradients: exact for a linear model ---
    w = np.array([1.5, -2.0, 0.5, 3.0])

    def lin(Z: np.ndarray) -> np.ndarray:
        return Z @ w

    def grad(z: np.ndarray) -> np.ndarray:
        return w

    ig = Explainer("integrated_gradients", steps=32).fit(X)
    base = np.array([0.3, -0.2, 0.1, 0.0])
    phi_ig = ig.contributions(lin, x0, grad_fn=grad, baseline=base)
    assert np.max(np.abs(phi_ig - w * (x0 - base))) < 1e-10
    def quad(z: np.ndarray) -> float:
        return float(z[0] * z[1] + z[2] ** 2)

    def quad_grad(z: np.ndarray) -> np.ndarray:
        return np.array([z[1], z[0], 2.0 * z[2], 0.0])

    phi_q = ig.contributions(quad, x0, grad_fn=quad_grad, baseline=np.zeros(d))
    assert abs(phi_q.sum() - (quad(x0) - quad(np.zeros(d)))) < 1e-6
    print(f"integrated_gradients: linear exact to {np.max(np.abs(phi_ig - w * (x0 - base))):.1e},"
          f" quadratic completeness residual "
          f"{abs(phi_q.sum() - (quad(x0) - quad(np.zeros(d)))):.1e}")

    # --- lime: recovers the linear coefficients exactly ---
    lime = Explainer("lime", n_samples=2000, random_state=0).fit(X)
    coef = lime.contributions(lin, x0)
    assert np.max(np.abs(coef - w)) < 1e-3, coef
    local = lime.contributions(model, x0)
    agree = float(np.corrcoef(local, phi_t)[0, 1])
    assert agree > 0.8, agree
    print(f"lime: linear recovery error {np.max(np.abs(coef - w)):.1e}; "
          f"surrogate on the forest correlates {agree:.3f} with tree_shap")

    print("interpretability demo passed")


if __name__ == "__main__":
    main()
