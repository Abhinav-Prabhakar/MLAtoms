"""k-nearest neighbours — the training set *is* the model.

Chapter:   03-kernels-and-instance
Companion: knn.md
Tier 1 · 9 variants: classification, regression, uniform, distance_weighted,
kd_tree, ball_tree, cosine, edited_knn, condensed_knn

    ŷ(x) = vote of the k training points closest to x

No parameters are learned and nothing is fitted — `fit` stores the data. That
makes k-NN the cleanest example of an instance-based method: all the work moves
to prediction time, and every failure mode is a statement about the distance.

    classification / regression      vote or average
    uniform / distance_weighted      treat the k neighbours equally or by 1/d
    kd_tree / ball_tree              prune the scan with geometry
    cosine                           angle instead of length
    edited_knn                       delete points their neighbours disagree with
    condensed_knn                    keep only the points that change a decision

The two editing variants are the interesting ones: they attack the two real
costs of k-NN — storage and noise — instead of the search.
"""

from __future__ import annotations

import heapq

import numpy as np

VARIANTS = frozenset(
    ["classification", "regression", "uniform", "distance_weighted", "kd_tree",
     "ball_tree", "cosine", "edited_knn", "condensed_knn"]
)
METRICS = frozenset(["euclidean", "manhattan", "cosine"])
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #
def pairwise(X: np.ndarray, Z: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distance matrix `(n, m)` for the three metrics k-NN actually uses."""
    if metric == "euclidean":
        d2 = (X * X).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2.0 * X @ Z.T
        return np.sqrt(np.maximum(d2, 0.0))
    if metric == "manhattan":
        return np.abs(X[:, None, :] - Z[None, :, :]).sum(axis=2)
    if metric == "cosine":
        xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
        zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), EPS)
        return 1.0 - xn @ zn.T
    raise ValueError(f"unknown metric {metric!r}")


# --------------------------------------------------------------------------- #
# Trees
# --------------------------------------------------------------------------- #
class _KDTree:
    """Axis-aligned median splits, with the classic ball-within-bounds prune.

    Nodes split on the axis with the largest spread. A query descends to the
    leaf containing `x`, then backtracks only while the splitting plane is
    closer than the current k-th best distance — every other subtree is
    guaranteed worse, which is what turns the scan into a search.
    """

    def __init__(self, X: np.ndarray, leaf_size: int = 16):
        self.X = X
        self.leaf_size = int(leaf_size)
        self.evals = 0
        self.root = self._build(np.arange(X.shape[0]))

    def _build(self, idx: np.ndarray) -> dict:
        pts = self.X[idx]
        if idx.size <= self.leaf_size:
            return {"idx": idx}
        axis = int(np.argmax(pts.max(0) - pts.min(0)))
        order = idx[np.argsort(pts[:, axis], kind="stable")]
        mid = order.size // 2
        return {"axis": axis, "value": float(self.X[order[mid], axis]),
                "lo": self._build(order[:mid]), "hi": self._build(order[mid:])}

    def _push(self, heap: list, dist: float, i: int, k: int) -> None:
        if len(heap) < k:
            heapq.heappush(heap, (-dist, i))
        elif dist < -heap[0][0]:
            heapq.heapreplace(heap, (-dist, i))

    def _query(self, node: dict, x: np.ndarray, k: int, heap: list) -> None:
        if "idx" in node:
            d = np.sqrt(((self.X[node["idx"]] - x) ** 2).sum(1))
            self.evals += node["idx"].size
            for dist, i in zip(d, node["idx"], strict=True):
                self._push(heap, float(dist), int(i), k)
            return
        axis = node["axis"]
        near, far = (node["lo"], node["hi"]) if x[axis] <= node["value"] else (
            node["hi"], node["lo"])
        self._query(near, x, k, heap)
        gap = x[axis] - node["value"]
        # the plane is the only thing the far subtree can be closer than
        if len(heap) < k or gap * gap < heap[0][0] ** 2:
            self._query(far, x, k, heap)

    def query(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        out_d = np.empty((X.shape[0], k))
        out_i = np.empty((X.shape[0], k), dtype=int)
        for r, x in enumerate(X):
            heap: list = []
            self._query(self.root, x, k, heap)
            best = sorted((-d, i) for d, i in heap)
            out_d[r] = [d for d, _ in best]
            out_i[r] = [i for _, i in best]
        return out_d, out_i


class _BallTree:
    """Nested balls: split by the two farthest points, prune by `d(q,c) − r`.

    A ball node bounds all of its points by one centre and one radius, so the
    distance from a query to anything inside is at least `d(q, c) − r`. That
    bound is much tighter than a single splitting plane when the data is
    clustered, which is why ball trees win on high-dimensional or clustered
    inputs where kd-trees degenerate to a linear scan.
    """

    def __init__(self, X: np.ndarray, leaf_size: int = 16):
        self.X = X
        self.leaf_size = int(leaf_size)
        self.evals = 0
        self.root = self._build(np.arange(X.shape[0]))

    def _build(self, idx: np.ndarray) -> dict:
        pts = self.X[idx]
        centre = pts.mean(axis=0)
        radius = float(np.sqrt(((pts - centre) ** 2).sum(1)).max())
        node = {"idx": idx, "centre": centre, "radius": radius}
        if idx.size <= self.leaf_size:
            return node
        # two-farthest-points split: cheap, balanced, and axis-free
        a = int(np.argmax(((pts - centre) ** 2).sum(1)))
        far = ((pts - pts[a]) ** 2).sum(1)
        b = int(np.argmax(far))
        left = ((pts - pts[a]) ** 2).sum(1) <= ((pts - pts[b]) ** 2).sum(1)
        node["lo"] = self._build(idx[left])
        node["hi"] = self._build(idx[~left])
        return node

    def _push(self, heap: list, dist: float, i: int, k: int) -> None:
        if len(heap) < k:
            heapq.heappush(heap, (-dist, i))
        elif dist < -heap[0][0]:
            heapq.heapreplace(heap, (-dist, i))

    def _query(self, node: dict, x: np.ndarray, k: int, heap: list) -> None:
        if len(heap) >= k:
            bound = float(np.sqrt(((x - node["centre"]) ** 2).sum())) - node["radius"]
            if bound > -heap[0][0]:
                return                      # the whole ball is farther than k
        if "lo" not in node:
            d = np.sqrt(((self.X[node["idx"]] - x) ** 2).sum(1))
            self.evals += node["idx"].size
            for dist, i in zip(d, node["idx"], strict=True):
                self._push(heap, float(dist), int(i), k)
            return
        near, far = ((node["lo"], node["hi"])
                     if (((x - node["lo"]["centre"]) ** 2).sum()
                         <= ((x - node["hi"]["centre"]) ** 2).sum())
                     else (node["hi"], node["lo"]))
        self._query(near, x, k, heap)
        self._query(far, x, k, heap)

    def query(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        out_d = np.empty((X.shape[0], k))
        out_i = np.empty((X.shape[0], k), dtype=int)
        for r, x in enumerate(X):
            heap: list = []
            self._query(self.root, x, k, heap)
            best = sorted((-d, i) for d, i in heap)
            out_d[r] = [d for d, _ in best]
            out_i[r] = [i for _, i in best]
        return out_d, out_i


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class KNN:
    """Nearest-neighbour classification and regression.

    Parameters
    ----------
    variant : str
        One of VARIANTS. `uniform` and `distance_weighted` choose the vote;
        `classification`/`regression` choose the output; a tree name chooses the
        search; `edited_knn`/`condensed_knn` choose the training subset.
    k : int
        Neighbour count. `k=1` reproduces the training set exactly.
    metric : str
        One of METRICS. Trees require `euclidean`.
    leaf_size : int
        Points per tree leaf; larger leaves mean fewer nodes and more brute
        force inside each one.
    """

    def __init__(self, variant: str = "classification", k: int = 5, *,
                 metric: str = "euclidean", leaf_size: int = 16,
                 max_rounds: int = 64):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        if variant in ("kd_tree", "ball_tree") and metric != "euclidean":
            raise ValueError("tree variants need the euclidean metric")
        self.variant = variant
        self.k = int(k)
        self.metric = "cosine" if variant == "cosine" else metric
        self.leaf_size = int(leaf_size)
        self.max_rounds = int(max_rounds)
        self.task = "regression" if variant == "regression" else "classification"
        self.weights = "distance" if variant == "distance_weighted" else "uniform"

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> KNN:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_, yi = np.unique(y, return_inverse=True)
            self.y_enc_ = yi
        else:
            self.y_ = y.astype(float)
        self.X_ = X
        if self.variant == "kd_tree":
            self.tree_ = _KDTree(X, self.leaf_size)
        elif self.variant == "ball_tree":
            self.tree_ = _BallTree(X, self.leaf_size)
        elif self.variant == "edited_knn":
            self._edit()
        elif self.variant == "condensed_knn":
            self._condense()
        return self

    # ------------------------------------------------------------------ #
    def _search(self, Xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k nearest training points, as `(distance, index)` sorted ascending."""
        k = min(k, self.X_.shape[0])
        if self.variant in ("kd_tree", "ball_tree"):
            return self.tree_.query(Xq, k)
        D = pairwise(Xq, self.X_, self.metric)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(D, idx, axis=1)
        order = np.argsort(d, axis=1)
        return np.take_along_axis(d, order, 1), np.take_along_axis(idx, order, 1)

    def kneighbors(self, X: np.ndarray, k: int | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
        return self._search(np.asarray(X, dtype=float),
                            self.k if k is None else k)

    # ------------------------------------------------------------------ #
    def _vote(self, dist: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Weighted vote; ties go to the lowest class index, deterministically."""
        w = 1.0 / np.maximum(dist, EPS) if self.weights == "distance" else np.ones_like(dist)
        score = np.zeros((idx.shape[0], self.classes_.size))
        for c in range(self.classes_.size):
            score[:, c] = (w * (self.y_enc_[idx] == c)).sum(axis=1)
        return self.classes_[np.argmax(score, axis=1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dist, idx = self._search(X, self.k)
        if self.task == "regression":
            w = 1.0 / np.maximum(dist, EPS) if self.weights == "distance" else np.ones_like(dist)
            return (w * self.y_[idx]).sum(axis=1) / w.sum(axis=1)
        return self._vote(dist, idx)

    # ------------------------------------------------------------------ #
    def _label_at(self, i: int, pool: np.ndarray, k: int) -> int:
        """Majority label of `i`'s `k` nearest neighbours inside `pool`.

        `i` itself is excluded: leaving it in would make every point its own
        neighbour at distance zero, and the vote would always be unanimous.
        """
        others = pool[pool != i]
        if others.size == 0:
            return int(self.y_enc_[i])
        d = np.sqrt(((self.X_[others] - self.X_[i]) ** 2).sum(1))
        near = np.argsort(d)[:k]
        labels = self.y_enc_[others[near]]
        return int(np.bincount(labels, minlength=self.classes_.size).argmax())

    def _edit(self) -> None:
        """Wilson editing: drop every point its own neighbours disagree with.

        Noise removal, not compression — an edited set usually loses only the
        mislabelled points, but a single pass already changes the decision
        boundary because those points were voting.
        """
        keep = np.ones(self.X_.shape[0], dtype=bool)
        for _ in range(self.max_rounds):
            pool = np.flatnonzero(keep)
            drop = np.array([keep[i] and self._label_at(i, pool, self.k + 1) != self.y_enc_[i]
                             for i in range(self.X_.shape[0])])
            if not drop.any():
                break
            keep &= ~drop
        self.removed_ = ~keep
        self.X_, self.y_enc_ = self.X_[keep], self.y_enc_[keep]

    def _condense(self) -> None:
        """Hart's condensed nearest neighbour: keep only decision-changing points.

        Scan the training set once per pass; if the current store misclassifies
        a point, add it. A point that is already classified correctly can never
        be needed, because the store only grows — this is the classic greedy
        set-cover argument.
        """
        n = self.X_.shape[0]
        store = [0]
        for _ in range(self.max_rounds):
            added = 0
            for i in range(n):
                if i in store:
                    continue
                d = np.sqrt(((self.X_[store] - self.X_[i]) ** 2).sum(1))
                label = int(self.y_enc_[store[int(np.argmin(d))]])
                if label != self.y_enc_[i]:
                    store.append(i)
                    added += 1
            if added == 0:
                break
        self.keep_ = np.asarray(store)
        self.X_, self.y_enc_ = self.X_[self.keep_], self.y_enc_[self.keep_]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- classification: 1-NN memorises, k-NN smooths ---
    # deliberately overlapping blobs: k-NN cannot beat the class overlap, and
    # the interesting comparison is k=1 (memorise) against k=5 (smooth)
    centres = [(-1.5, 0.0), (1.5, 0.0), (0.0, 2.0)]
    X = np.vstack([np.array(c) + rng.normal(0, 1.0, size=(50, 2)) for c in centres])
    y = np.repeat(np.arange(3), 50)
    te = rng.permutation(X.shape[0])[:60]
    tr = np.setdiff1d(np.arange(X.shape[0]), te)
    one = KNN("classification", k=1).fit(X[tr], y[tr])
    five = KNN("classification", k=5).fit(X[tr], y[tr])
    assert np.mean(one.predict(X[tr]) == y[tr]) == 1.0      # 1-NN has zero train error
    acc1 = float(np.mean(one.predict(X[te]) == y[te]))
    acc5 = float(np.mean(five.predict(X[te]) == y[te]))
    assert acc1 < 0.85 and acc5 > acc1, (acc1, acc5)
    print(f"knn: overlapping blobs — 1-NN train accuracy 1.000 (memorised), "
          f"test {acc1:.3f}; k=5 test {acc5:.3f} (the ceiling is the overlap)")

    # --- regression: k-NN is a piecewise-constant mean of the k targets ---
    xr = np.sort(rng.uniform(-3, 3, 200))
    yr = np.sin(xr) + 0.1 * rng.normal(size=200)
    grid = np.linspace(-3, 3, 61)
    truth = np.sin(grid)
    rmse = {}
    for k in (1, 5, 25):
        reg = KNN("regression", k=k).fit(xr[:, None], yr)
        rmse[k] = float(np.sqrt(np.mean((reg.predict(grid[:, None]) - truth) ** 2)))
    assert rmse[1] > rmse[5] and rmse[5] < rmse[25], rmse
    print(f"knn: 1-D regression — rmse {rmse[1]:.3f} (k=1), {rmse[5]:.3f} "
          f"(k=5), {rmse[25]:.3f} (k=25); bias-variance in one line")

    # --- uniform vs distance_weighted, on a fixture where they must differ ---
    Xw = np.array([[0.0], [1.0], [1.1], [1.2], [1.3], [1.4]])
    yw = np.array([0, 1, 1, 1, 1, 1])
    q = np.array([[0.1]])
    uni = KNN("classification", k=5, metric="euclidean").fit(Xw, yw)
    dis = KNN("distance_weighted", k=5, metric="euclidean").fit(Xw, yw)
    assert uni.predict(q)[0] == 1 and dis.predict(q)[0] == 0
    print("knn: one near neighbour vs five far ones — uniform votes 1, "
          "distance-weighted votes 0 (1/0.1 beats 5×(1/0.9))")

    # --- trees: identical neighbours, fewer distance evaluations ---
    Xt = rng.normal(size=(2000, 2))
    probe = rng.normal(size=(200, 2))
    labels = np.arange(2000) % 3
    brute = KNN("classification", k=5).fit(Xt, labels)
    kdt = KNN("kd_tree", k=5).fit(Xt, labels)
    ball = KNN("ball_tree", k=5).fit(Xt, labels)
    db, ib = brute.kneighbors(probe)
    dk, ik = kdt.kneighbors(probe)
    dl, il = ball.kneighbors(probe)
    assert np.array_equal(ib, ik) and np.array_equal(ib, il)
    assert np.allclose(db, dk) and np.allclose(db, dl, atol=1e-12)
    cost_brute = probe.shape[0] * Xt.shape[0]
    assert kdt.tree_.evals < cost_brute // 10
    assert ball.tree_.evals < cost_brute // 10
    assert np.mean(brute.predict(probe) == kdt.predict(probe)) == 1.0
    print(f"knn: 2000×2, k=5 — kd_tree {kdt.tree_.evals} and ball_tree "
          f"{ball.tree_.evals} distance evaluations vs {cost_brute} brute "
          f"force ({100 * kdt.tree_.evals / cost_brute:.1f}% and "
          f"{100 * ball.tree_.evals / cost_brute:.1f}%); neighbours identical")

    # --- cosine: an angle metric disagrees with a length metric ---
    Xc = np.array([[0.1, 0.1], [2.0, 0.5]])
    qc = np.array([[1.0, 1.0]])
    eu = KNN("classification", k=1, metric="euclidean").fit(Xc, [0, 1])
    co = KNN("cosine", k=1).fit(Xc, [0, 1])
    d_eu = pairwise(qc, Xc, "euclidean")[0]
    d_co = pairwise(qc, Xc, "cosine")[0]
    assert eu.predict(qc)[0] == 1 and co.predict(qc)[0] == 0, (d_eu, d_co)
    print(f"knn: euclidean picks the longer vector (d={d_eu[1]:.2f} < "
          f"{d_eu[0]:.2f}), cosine picks the aligned one "
          f"(1−cos={d_co[0]:.3f} < {d_co[1]:.3f})")

    # --- edited_knn: delete the points their neighbours disagree with ---
    Xe = np.vstack([rng.normal(-1.0, 0.6, size=(120, 2)),
                    rng.normal(1.0, 0.6, size=(120, 2))])
    ye = np.r_[np.zeros(120, int), np.ones(120, int)]
    flip = rng.choice(240, size=24, replace=False)      # 10% label noise
    ye_noisy = ye.copy()
    ye_noisy[flip] = 1 - ye_noisy[flip]
    plain = KNN("classification", k=5).fit(Xe, ye_noisy)
    edit = KNN("edited_knn", k=5).fit(Xe, ye_noisy)
    kept_noisy = float(np.mean(~edit.removed_[flip]))
    acc_plain = float(np.mean(plain.predict(Xe) == ye))
    acc_edit = float(np.mean(edit.predict(Xe) == ye))
    assert edit.removed_.sum() > 0 and acc_edit > acc_plain, (acc_plain, acc_edit)
    print(f"knn: edited_knn — removed {int(edit.removed_.sum())} of 240 points "
          f"({kept_noisy:.0%} of the flips survived), accuracy on the clean "
          f"labels {acc_plain:.3f} → {acc_edit:.3f}")

    # --- condensed_knn: keep only the points that change a decision ---
    cond = KNN("condensed_knn", k=1).fit(Xe, ye)
    consistency = float(np.mean(cond.predict(Xe) == ye))
    assert cond.X_.shape[0] < 240
    assert consistency > 0.99, consistency
    assert cond.X_.shape[0] > 2
    print(f"knn: condensed_knn — {cond.X_.shape[0]} of 240 points reproduce the "
          f"training labels at {consistency:.3f} with 1-NN")

    print("knn demo passed")


if __name__ == "__main__":
    main()
