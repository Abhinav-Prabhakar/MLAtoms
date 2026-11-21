"""Agglomerative clustering — merge the closest pair until one cluster remains.

Chapter:   07-clustering
Companion: agglomerative.md
Tier 1 · 8 variants: single, complete, average, ward, centroid, median,
connectivity_constrained, cut_height

Bottom-up clustering starts with `n` singleton clusters and repeatedly merges the
two closest. Everything else is a definition of "closest":

    single       d(A,B) = min  d(a,b)      nearest pair; chaining, arbitrary shapes
    complete     d(A,B) = max  d(a,b)      diameter; compact, equal-sized clusters
    average      d(A,B) = mean d(a,b)      UPGMA; compromise, size-weighted
    ward         Δ = n_A n_B/(n_A+n_B) |mu_A - mu_B|^2   variance-minimising
    centroid     |mu_A - mu_B|^2           distance between centroids
    median       |mu_A - mu_B|^2, unweighted   WPGMC, no size bias

All six satisfy the Lance-Williams recursion

    d(A u B, C) = a_A d(A,C) + a_B d(B,C) + b d(A,B) + g |d(A,C) - d(B,C)|

with different (a_A, a_B, b, g), so one merge loop covers every linkage. Ward,
centroid and median are defined on *squared* Euclidean distances; their merge
heights are square roots of the working values.

The remaining two variants are about the merge rule rather than the metric:
`connectivity_constrained` only allows merges along a k-NN graph, and
`cut_height` stops at a distance threshold instead of a cluster count.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["single", "complete", "average", "ward", "centroid", "median",
     "connectivity_constrained", "cut_height"]
)

# Linkage used underneath each variant.
BASE = {
    "single": "single",
    "complete": "complete",
    "average": "average",
    "ward": "ward",
    "centroid": "centroid",
    "median": "median",
    "connectivity_constrained": "ward",
    "cut_height": "single",
}

# Linkages whose Lance-Williams recursion runs on squared distances.
SQUARED = frozenset(["ward", "centroid", "median"])

# Linkages that are guaranteed monotone: merge heights never decrease.
MONOTONE = frozenset(["single", "complete", "average", "ward"])

EPS = 1e-12


class Agglomerative:
    """Bottom-up hierarchical clustering.

    `n_clusters` is used by every variant except `cut_height`, which uses
    `height` instead; `min_cluster_size` turns smaller groups into noise after
    the cut.
    """

    def __init__(
        self,
        variant: str = "ward",
        n_clusters: int = 2,
        *,
        height: float | None = None,
        n_neighbors: int = 10,
        min_cluster_size: int = 1,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.linkage = BASE[variant]
        self.n_clusters = int(n_clusters)
        self.height = None if height is None else float(height)
        self.n_neighbors = int(n_neighbors)
        self.min_cluster_size = int(min_cluster_size)
        if variant == "cut_height" and self.height is None:
            raise ValueError("cut_height needs height=")
        self.n_inversions_ = 0
        self.n_dist_ = 0

    # ------------------------------------------------------------------ #
    # Distances
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pairwise(X: np.ndarray) -> np.ndarray:
        d2 = (X * X).sum(axis=1)[:, None] - 2.0 * X @ X.T + (X * X).sum(axis=1)[None]
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    @staticmethod
    def _lance_williams(
        name: str,
        di: np.ndarray,
        dj: np.ndarray,
        dij: float,
        ni: float,
        nj: float,
        nk: np.ndarray,
    ) -> np.ndarray:
        """Lance-Williams update for cluster `i u j` against every live `k`."""
        if name == "single":
            return 0.5 * di + 0.5 * dj - 0.5 * np.abs(di - dj)
        if name == "complete":
            return 0.5 * di + 0.5 * dj + 0.5 * np.abs(di - dj)
        if name == "average":
            w = ni + nj
            return (ni * di + nj * dj) / w
        if name == "ward":
            w = ni + nj + nk
            return ((ni + nk) * di + (nj + nk) * dj - nk * dij) / w
        if name == "centroid":
            w = ni + nj
            return (ni * di + nj * dj) / w - (ni * nj / (w * w)) * dij
        if name == "median":
            return 0.5 * di + 0.5 * dj - 0.25 * dij
        raise ValueError(name)

    def _height(self, value: float) -> float:
        return float(np.sqrt(max(value, 0.0))) if self.linkage in SQUARED else float(value)

    def _constraints(self, d: np.ndarray) -> np.ndarray:
        """Mutual k-NN graph: which singleton pairs may merge.

        An edge needs both endpoints to name each other, so a point near a
        different manifold cannot drag a cluster across on its own.
        """
        n = d.shape[0]
        k = min(self.n_neighbors, n - 1)
        idx = np.argpartition(d, k, axis=1)[:, :k + 1]
        adj = np.zeros((n, n), dtype=bool)
        adj[np.arange(n)[:, None], idx] = True
        adj &= adj.T
        np.fill_diagonal(adj, False)
        return adj

    # ------------------------------------------------------------------ #
    # Merge loop
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> Agglomerative:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if n < 2:
            raise ValueError("need at least two points")
        d = self._pairwise(X)
        self.n_dist_ = n * n
        working = d * d if self.linkage in SQUARED else d

        cap = 2 * n - 1
        dist = np.full((cap, cap), np.inf)
        dist[:n, :n] = working
        np.fill_diagonal(dist, np.inf)
        size = np.ones(cap, dtype=int)
        alive = np.zeros(cap, dtype=bool)
        alive[:n] = True

        allow = np.ones((cap, cap), dtype=bool)
        if self.variant == "connectivity_constrained":
            allow = np.zeros((cap, cap), dtype=bool)
            allow[:n, :n] = self._constraints(d)
            np.fill_diagonal(allow, False)

        merges: list[tuple[int, int, float, int]] = []
        nxt = n
        top = -np.inf
        for _ in range(n - 1):
            live = alive[:, None] & alive[None, :]
            cand = np.where(live, dist, np.inf)
            if self.variant == "connectivity_constrained":
                cand = np.where(allow, cand, np.inf)
            if not np.isfinite(cand).any():
                cand = np.where(live, dist, np.inf)   # drop the constraint, keep going
            i, j = divmod(int(np.argmin(cand)), cap)
            if i > j:
                i, j = j, i
            value = float(cand[i, j])
            h = self._height(value)
            if self.variant == "cut_height" and h > self.height:
                break
            if h < top - 1e-12:
                self.n_inversions_ += 1
            top = max(top, h)

            ks = np.nonzero(alive)[0]
            ks = ks[(ks != i) & (ks != j)]
            new = self._lance_williams(
                self.linkage, dist[i, ks], dist[j, ks], value,
                float(size[i]), float(size[j]), size[ks].astype(float))
            dist[nxt, ks] = new
            dist[ks, nxt] = new
            if self.variant == "connectivity_constrained":
                link = allow[i, ks] | allow[j, ks]
                allow[nxt, ks] = link
                allow[ks, nxt] = link
            size[nxt] = size[i] + size[j]
            alive[i] = False
            alive[j] = False
            alive[nxt] = True
            merges.append((int(i), int(j), h, int(size[nxt])))
            nxt += 1

        self.children_ = np.array([(a, b) for a, b, _, _ in merges], dtype=int)
        self.distances_ = np.array([h for _, _, h, _ in merges], dtype=float)
        self.sizes_ = np.array([s for _, _, _, s in merges], dtype=int)
        self.n_inversions_ = int(self.n_inversions_)

        keep = len(merges) if self.variant == "cut_height" else max(
            n - self.n_clusters, 0)
        labels = self._labels(merges, keep, n)
        if self.min_cluster_size > 1:
            counts = np.bincount(labels[labels >= 0])
            small = np.zeros(counts.size, dtype=bool)
            small[counts < self.min_cluster_size] = True
            labels = np.where(small[labels], -1, labels)
        self.labels_ = labels
        self.n_clusters_ = int(np.unique(labels[labels >= 0]).size)
        self.n_noise_ = int((labels < 0).sum())
        return self

    def _labels(self, merges: list[tuple[int, int, float, int]], keep: int,
                n: int) -> np.ndarray:
        parent = list(range(2 * n - 1))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for idx, (a, b, _, _) in enumerate(merges):
            if idx >= keep:
                break
            ra, rb = find(a), find(b)
            parent[ra] = n + idx
            parent[rb] = n + idx

        roots = np.array([find(p) for p in range(n)])
        uniq, ids = np.unique(roots, return_inverse=True)
        order = np.argsort(-np.bincount(ids), kind="stable")
        rank = np.empty(order.size, dtype=int)
        rank[order] = np.arange(order.size)
        return rank[ids]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _match_accuracy(labels: np.ndarray, truth: np.ndarray) -> float:
    import itertools
    mask = labels >= 0
    ids = np.unique(labels[mask])
    if ids.size == 0:
        return 0.0
    best = 0.0
    for perm in itertools.permutations(range(int(ids.max()) + 1)):
        best = max(best, float(np.mean(np.asarray(perm)[labels][mask] == truth[mask])))
    return best


def _spread(X: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total / X.shape[0]


def main() -> None:
    rng = np.random.default_rng(0)

    # --- two blobs joined by a thin chain: single linkage chains, complete resists ---
    a = rng.normal([0.0, 0.0], 0.35, size=(60, 2))
    b = rng.normal([6.0, 0.0], 0.35, size=(60, 2))
    chain = np.column_stack([np.linspace(0.9, 5.1, 12), np.zeros(12)])
    Xc = np.vstack([a, b, chain])

    sl = Agglomerative("single", n_clusters=2).fit(Xc)
    cl = Agglomerative("complete", n_clusters=2).fit(Xc)
    h_single = float(sl.distances_[-1])
    h_complete = float(cl.distances_[-1])
    assert h_complete > 5.0 * h_single, (h_complete, h_single)
    assert sl.n_clusters_ == cl.n_clusters_ == 2
    print(f"agglomerative: chain merge height single {h_single:.2f} vs "
          f"complete {h_complete:.2f}")

    # --- monotonicity: single/complete/average/ward never invert ---
    for name in ("single", "complete", "average", "ward"):
        fit = Agglomerative(name, n_clusters=2).fit(Xc)
        assert fit.n_inversions_ == 0, (name, fit.n_inversions_)
    inv = {name: Agglomerative(name, n_clusters=2).fit(Xc).n_inversions_
           for name in ("centroid", "median")}
    assert inv["centroid"] > 0 and inv["median"] > 0, inv
    print(f"agglomerative: inversions centroid {inv['centroid']}, "
          f"median {inv['median']}, zero for the monotone linkages")

    # --- unequal blobs: ward minimises spread, centroid/median ignore size ---
    big = rng.normal([0.0, 0.0], 1.00, size=(300, 2))
    small = rng.normal([2.2, 0.0], 0.30, size=(20, 2))
    Xu = np.vstack([big, small])
    yu = np.r_[np.zeros(300, int), np.ones(20, int)]

    fits = {name: Agglomerative(name, n_clusters=2).fit(Xu)
            for name in ("ward", "centroid", "median", "average")}
    spreads = {name: _spread(Xu, f.labels_) for name, f in fits.items()}
    accs = {name: _match_accuracy(f.labels_, yu) for name, f in fits.items()}
    assert all(f.n_clusters_ == 2 for f in fits.values())
    assert min(spreads, key=spreads.get) == "ward", spreads
    assert accs["centroid"] > accs["ward"], accs
    print(f"agglomerative: unequal blobs spread ward {spreads['ward']:.2f} "
          f"< centroid {spreads['centroid']:.2f}; accuracy ward "
          f"{accs['ward']:.3f} vs centroid {accs['centroid']:.3f}")

    # --- connectivity constraint: follow the manifold instead of cutting it ---
    t = rng.uniform(0.0, np.pi, 150)
    ma = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0.0, 0.05, (150, 2))
    mb = np.column_stack([1.0 - np.cos(t), 0.5 - np.sin(t)])
    mb = mb + rng.normal(0.0, 0.05, (150, 2))
    Xm = np.vstack([ma, mb])
    ym = np.r_[np.zeros(150, int), np.ones(150, int)]

    plain = Agglomerative("ward", n_clusters=2).fit(Xm)
    cons = Agglomerative("connectivity_constrained", n_clusters=2,
                         n_neighbors=10).fit(Xm)
    acc_p = _match_accuracy(plain.labels_, ym)
    acc_k = _match_accuracy(cons.labels_, ym)
    assert acc_k > 0.95, acc_k
    assert acc_k > acc_p + 0.1, (acc_k, acc_p)
    print(f"agglomerative: constrained ward moons {acc_k:.3f} vs ward "
          f"{acc_p:.3f}")

    # --- three blobs, cut by height: K and noise fall out of the threshold ---
    p1 = rng.normal([0.0, 0.0], 0.35, size=(80, 2))
    p2 = rng.normal([6.0, 0.0], 0.35, size=(80, 2))
    p3 = rng.normal([3.0, 5.0], 0.35, size=(80, 2))
    tk = rng.uniform([-1.0, -1.0], [7.0, 6.0], size=(9, 2))
    Xt = np.vstack([p1, p2, p3, tk])
    yt = np.r_[np.zeros(80, int), np.ones(80, int),
               np.full(80, 2), -np.ones(9, int)]

    cut = Agglomerative("cut_height", height=0.5, min_cluster_size=20).fit(Xt)
    coarse = Agglomerative("cut_height", height=5.0, min_cluster_size=20).fit(Xt)
    acc_t = _match_accuracy(cut.labels_, yt)
    assert cut.n_clusters_ == 3, cut.n_clusters_
    assert cut.n_noise_ >= 6, cut.n_noise_
    assert coarse.n_clusters_ == 1, coarse.n_clusters_
    print(f"agglomerative: cut_height 0.5 -> {cut.n_clusters_} clusters, "
          f"{cut.n_noise_} noise, acc {acc_t:.3f}; 5.0 -> "
          f"{coarse.n_clusters_} cluster")

    print("agglomerative demo passed")


if __name__ == "__main__":
    main()
