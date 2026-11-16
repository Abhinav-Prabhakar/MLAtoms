"""HDBSCAN — DBSCAN with a hierarchy, so one threshold fits every density.

Chapter:   07-clustering
Companion: hdbscan.md
Tier 1 · 5 variants: hdbscan, hdbscan_star, soft_clustering, approximate, stability

DBSCAN needs one global `eps`, and a dataset with a tight cluster and a diffuse one
cannot satisfy it. HDBSCAN removes the parameter by building the clustering at *every*
density and choosing which clusters persist:

    1. core distance   d_core(x) = distance to the min_samples-th neighbour
    2. mutual reachability
                       d_mreach(a,b) = max(d_core(a), d_core(b), d(a,b))
    3. minimum spanning tree of the mutual-reachability graph
    4. single-linkage hierarchy: sort the MST edges from long to short and merge;
       each merge happens at lambda = 1 / distance
    5. stability       how much lambda-lifetime a cluster's points spend inside it
    6. selection       keep a cluster when its own stability exceeds the sum of its
                       children's, otherwise keep the children (excess of mass)

The variants are three ways of reading that hierarchy (excess of mass, HDBSCAN*'s
strict core membership, a persistence threshold), one soft assignment, and one
approximation of the MST.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["hdbscan", "hdbscan_star", "soft_clustering", "approximate", "stability"]
)

EPS = 1e-12
INF = np.inf


class HDBSCAN:
    """Hierarchical density-based clustering.

    `min_cluster_size` is the smallest group the selection may return and
    `min_samples` the density order (the k in core distance). `min_persistence`
    is the relative lambda-lifetime required by the `stability` variant;
    `min_probability` is the membership floor of `soft_clustering`.
    """

    def __init__(
        self,
        variant: str = "hdbscan",
        *,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        min_persistence: float = 0.3,
        min_probability: float = 0.05,
        allow_single_cluster: bool = False,
        approx_neighbors: int | None = None,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples or min_cluster_size)
        self.min_persistence = float(min_persistence)
        self.min_probability = float(min_probability)
        self.allow_single_cluster = bool(allow_single_cluster)
        self.approx_neighbors = approx_neighbors
        self.rng = np.random.default_rng(random_state)
        self.n_dist_ = 0
        self.n_edges_ = 0
        self.n_pairs_ = 0
        self.nodes: list[dict] = []

    # ------------------------------------------------------------------ #
    # Distances and graph
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pairwise(X: np.ndarray) -> np.ndarray:
        d2 = (X * X).sum(axis=1)[:, None] - 2.0 * X @ X.T + (X * X).sum(axis=1)[None]
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    def _core_distances(self, D: np.ndarray) -> np.ndarray:
        n = D.shape[0]
        k = min(self.min_samples, n - 1)
        return np.partition(D, k, axis=1)[:, k]

    def _prim(self, mr: np.ndarray) -> list[tuple[float, int, int]]:
        n = mr.shape[0]
        best = np.full(n, INF)
        parent = np.full(n, -1, dtype=int)
        seen = np.zeros(n, dtype=bool)
        best[0] = 0.0
        edges: list[tuple[float, int, int]] = []
        for _ in range(n):
            masked = np.where(seen, INF, best)
            u = int(np.argmin(masked))
            seen[u] = True
            if parent[u] >= 0:
                edges.append((float(mr[u, parent[u]]), int(parent[u]), u))
            upd = (~seen) & (mr[u] < best)
            best[upd] = mr[u][upd]
            parent[upd] = u
        return edges

    def _knn_edges(self, mr: np.ndarray) -> list[tuple[float, int, int]]:
        n = mr.shape[0]
        k = min(self.approx_neighbors or 3 * self.min_samples, n - 1)
        idx = np.argpartition(mr, k, axis=1)[:, :k + 1]
        edges: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in idx[i]:
                j = int(j)
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                edges[key] = float(mr[i, j])
        return [(w, i, j) for (i, j), w in edges.items()]

    # ------------------------------------------------------------------ #
    # Hierarchy
    # ------------------------------------------------------------------ #
    def _build_tree(self, n: int, edges: list[tuple[float, int, int]]) -> int:
        uf = list(range(n))
        node_of = list(range(n))
        lam = self._lam

        def find(a: int) -> int:
            while uf[a] != a:
                uf[a] = uf[uf[a]]
                a = uf[a]
            return a

        nodes: list[dict] = [
            {"level": float(lam[p]), "birth": 0.0, "children": [], "points": [p],
             "size": 1}
            for p in range(n)
        ]
        for w, i, j in sorted(edges):
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            level = 1.0 / w if w > EPS else INF
            a, b = node_of[ri], node_of[rj]
            nodes.append({
                "level": float(level), "birth": 0.0, "children": [a, b],
                "points": nodes[a]["points"] + nodes[b]["points"],
                "size": nodes[a]["size"] + nodes[b]["size"],
            })
            uf[ri] = rj
            node_of[rj] = len(nodes) - 1

        roots = sorted({find(i) for i in range(n)})
        if len(roots) > 1:                    # only for a disconnected k-NN graph
            nodes.append({
                "level": INF, "birth": 0.0,
                "children": [node_of[r] for r in roots],
                "points": list(range(n)), "size": n,
            })
        root = node_of[roots[0]] if len(roots) == 1 else len(nodes) - 1
        for node in nodes:
            for child in node["children"]:
                nodes[child]["birth"] = node["level"]
        nodes[root]["birth"] = 0.0
        self.nodes = nodes
        self._parent = {}
        for i, node in enumerate(nodes):
            for child in node["children"]:
                self._parent[child] = i
        return root

    def _compute_stability(self) -> dict[int, float]:
        """Lambda-lifetime mass: how long each point stays inside the cluster.

        A node exists for lambda in [birth, level]: it appears at the level its
        parent split and disappears when it splits itself. A point stops
        contributing at its own core-density lambda, whichever comes first.
        """
        stab: dict[int, float] = {}
        lam = self._lam
        for i, node in enumerate(self.nodes):
            lo, hi = node["birth"], node["level"]
            total = 0.0
            for p in node["points"]:
                lp = lam[p]
                if not np.isfinite(lp):
                    lp = 1e300
                total += max(0.0, min(lp, hi) - lo)
            stab[i] = float(total)
        return stab

    # ------------------------------------------------------------------ #
    # Cluster selection
    # ------------------------------------------------------------------ #
    def _eom(self, i: int) -> tuple[list[int], float]:
        node = self.nodes[i]
        if node["size"] < self.min_cluster_size:
            return [], 0.0
        if not node["children"]:
            return [], 0.0
        picks: list[int] = []
        score = 0.0
        for j in node["children"]:
            sub, s = self._eom(j)
            picks.extend(sub)
            score += s
        if self._stab[i] > score:
            return [i], self._stab[i]
        return picks, score

    def _best_flat_cut(self) -> list[int]:
        """DBSCAN* at the level that yields the most clusters.

        Every node is active for lambda in (birth, level] and the active nodes
        partition the leaves, so one sweep over the distinct levels finds the
        best flat reading of the hierarchy.
        """
        nodes = self.nodes
        levels = np.unique([n["level"] for n in nodes if np.isfinite(n["level"])])
        best_key = (0, 0.0)
        best: list[int] = []
        for lam in levels:
            cut = [i for i, n in enumerate(nodes)
                   if n["birth"] < lam <= n["level"]
                   and n["size"] >= self.min_cluster_size]
            if not cut:
                continue
            key = (len(cut), float(sum(nodes[i]["size"] for i in cut)))
            if key > best_key:
                best_key, best = key, cut
        return best

    def _ranked(self, root: int) -> list[int]:
        """The most stable clusters, skipping any that overlaps a better one."""
        cands = [(self._stab[i], i) for i, n in enumerate(self.nodes)
                 if i != root and n["size"] >= self.min_cluster_size
                 and self._stab[i] > 0.0]
        cands.sort(reverse=True)
        chosen: list[int] = []
        for _, i in cands:
            if any(self._related(i, c) for c in chosen):
                continue
            chosen.append(i)
        return chosen

    def _related(self, a: int, b: int) -> bool:
        """True when one node is an ancestor of the other."""
        x = a
        while x != 0 and self._parent.get(x) is not None:
            x = self._parent[x]
            if x == b:
                return True
        x = b
        while x != 0 and self._parent.get(x) is not None:
            x = self._parent[x]
            if x == a:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Membership
    # ------------------------------------------------------------------ #
    def _hard_labels(self, selected: list[int]):
        n = self._lam.size
        labels = np.full(n, -1, dtype=int)
        proba = np.zeros(n)
        for idx, c in enumerate(selected):
            node = self.nodes[c]
            level = node["level"]
            for p in node["points"]:
                labels[p] = idx
                proba[p] = 1.0 if level <= 0.0 else min(
                    1.0, float(self._lam[p]) / level)
        return labels, proba

    def _soft(self, X: np.ndarray, selected: list[int]) -> np.ndarray:
        exemplars = []
        for c in selected:
            pts = self.nodes[c]["points"]
            best = max(pts, key=lambda p: float(np.nan_to_num(self._lam[p],
                                                              posinf=1e300)))
            exemplars.append(X[best])
        E = np.asarray(exemplars)
        d2 = ((X[:, None, :] - E[None, :, :]) ** 2).sum(axis=2)
        w = 1.0 / np.maximum(d2, EPS)
        proba = w / w.sum(axis=1, keepdims=True)
        proba[proba < self.min_probability] = 0.0
        proba /= np.maximum(proba.sum(axis=1, keepdims=True), EPS)
        return proba

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> HDBSCAN:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        D = self._pairwise(X)
        self.n_dist_ = n * n
        self.core_distances_ = self._core_distances(D)
        core = self.core_distances_
        self._lam = np.where(core > EPS, 1.0 / np.maximum(core, EPS), INF)
        mr = np.maximum(D, core[:, None])
        mr = np.maximum(mr, core[None, :])

        edges = (self._knn_edges(mr) if self.variant == "approximate"
                 else self._prim(mr))
        self.n_edges_ = len(edges)
        self.n_pairs_ = (len(edges) if self.variant == "approximate"
                         else n * (n - 1) // 2)

        root = self._build_tree(n, edges)
        self._stab = self._compute_stability()

        if self.variant == "stability":
            selected = self._ranked(root)
        elif self.variant == "hdbscan_star":
            selected = self._best_flat_cut()
        elif self.allow_single_cluster:
            selected, _ = self._eom(root)
        else:
            # The root's stability is n * its lambda level, so it beats every
            # child sum; HDBSCAN excludes it unless one cluster is wanted.
            selected = []
            for j in self.nodes[root]["children"]:
                sub, _ = self._eom(j)
                selected.extend(sub)
        selected = sorted(selected, key=lambda c: -self.nodes[c]["size"])
        self.n_clusters_ = len(selected)

        if self.variant == "soft_clustering":
            self.membership_ = self._soft(X, selected)
            labels = np.where(self.membership_.max(axis=1) >= 0.5,
                              np.argmax(self.membership_, axis=1), -1)
            self.probabilities_ = self.membership_.max(axis=1)
        else:
            labels, proba = self._hard_labels(selected)
            self.probabilities_ = proba

        self.labels_ = labels
        self.cluster_stability_ = np.array([self._stab[c] for c in selected])
        self.n_noise_ = int((labels < 0).sum())
        lam = np.where(np.isfinite(self._lam), self._lam, np.nan)
        top = float(np.nanmax(lam)) if np.isfinite(lam).any() else 1.0
        self.outlier_scores_ = np.clip(1.0 - np.nan_to_num(lam, nan=0.0) / top,
                                       0.0, 1.0)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _match_accuracy(labels, truth, mask):
    import itertools
    ids = np.unique(labels[mask])
    ids = ids[ids >= 0]
    if ids.size == 0:
        return 0.0
    best = 0.0
    for perm in itertools.permutations(range(int(ids.max()) + 1)):
        best = max(best, float(np.mean(np.asarray(perm)[labels][mask] == truth[mask])))
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    Xa = rng.normal([0.0, 0.0], 0.30, size=(250, 2))
    Xb = rng.normal([5.0, 0.0], 0.55, size=(200, 2))
    Xc = rng.normal([0.0, 5.0], 0.22, size=(120, 2))
    Xn = rng.uniform([-2.0, -2.0], [7.5, 7.5], size=(80, 2))
    X = np.vstack([Xa, Xb, Xc, Xn])
    truth = np.r_[np.zeros(250, int), np.ones(200, int),
                  np.full(120, 2), -np.ones(80, int)]
    inlier = truth >= 0

    hdb = HDBSCAN("hdbscan", min_cluster_size=20, min_samples=5).fit(X)
    got = hdb.labels_ >= 0
    acc = _match_accuracy(hdb.labels_, truth, got)
    assert hdb.n_clusters_ == 3, hdb.n_clusters_
    assert acc > 0.98, acc
    assert got[inlier].mean() > 0.85, got[inlier].mean()
    print(f"hdbscan: {hdb.n_clusters_} clusters, {hdb.n_noise_} noise, "
          f"coverage {got[inlier].mean():.3f} of inliers, acc {acc:.3f}, "
          f"stability {np.round(hdb.cluster_stability_, 1)}")

    # --- HDBSCAN*: one DBSCAN* at the level with the most clusters ---
    star = HDBSCAN("hdbscan_star", min_cluster_size=20, min_samples=5).fit(X)
    star_got = star.labels_ >= 0
    acc_s = _match_accuracy(star.labels_, truth, star_got)
    assert star.n_clusters_ == 3, star.n_clusters_
    assert star_got[inlier].mean() > got[inlier].mean()
    print(f"hdbscan: star flat cut {star.n_clusters_} clusters, "
          f"{star.n_noise_} noise, coverage {star_got[inlier].mean():.3f}, "
          f"acc {acc_s:.3f}")

    # --- stability ranking: the most persistent nodes, no EOM comparison ---
    per = HDBSCAN("stability", min_cluster_size=20, min_samples=5).fit(X)
    assert per.n_clusters_ >= 2, per.n_clusters_
    assert per.cluster_stability_.min() > 0.0
    print(f"hdbscan: stability ranking {per.n_clusters_} clusters, "
          f"stabilities {np.round(per.cluster_stability_, 1)}")

    # --- soft membership ---
    soft = HDBSCAN("soft_clustering", min_cluster_size=20, min_samples=5).fit(X)
    mu = soft.membership_
    assert mu.shape == (X.shape[0], soft.n_clusters_)
    assert np.allclose(mu.sum(axis=1), 1.0)
    ambiguous = ((mu > 0.2) & (mu < 0.8)).sum(axis=1)
    assert ambiguous.sum() > 0, ambiguous.sum()
    print(f"hdbscan: soft {soft.n_clusters_} clusters, {int(ambiguous.sum())} "
          f"points split between clusters, median max membership "
          f"{np.median(mu.max(axis=1)):.2f}")

    # --- approximate MST: a k-NN graph instead of every pair ---
    app = HDBSCAN("approximate", min_cluster_size=20, min_samples=5).fit(X)
    assert app.n_clusters_ == hdb.n_clusters_, app.n_clusters_
    assert app.n_pairs_ < hdb.n_pairs_ / 10, (app.n_pairs_, hdb.n_pairs_)
    acc_a = _match_accuracy(app.labels_, truth, app.labels_ >= 0)
    assert abs(acc_a - acc) < 0.02, (acc_a, acc)
    print(f"hdbscan: approximate {app.n_pairs_} candidate pairs vs "
          f"{hdb.n_pairs_}, same {app.n_clusters_} clusters, acc {acc_a:.3f}")

    # --- the density score separates background from clusters ---
    score = hdb.outlier_scores_
    inside = hdb.labels_ >= 0
    assert score[~inside].mean() > score[inside].mean(), (
        score[~inside].mean(), score[inside].mean())
    assert np.all((score >= 0.0) & (score <= 1.0))
    print(f"hdbscan: outlier score {score[~inside].mean():.3f} for noise vs "
          f"{score[inside].mean():.3f} for clustered points")

    print("hdbscan demo passed")


if __name__ == "__main__":
    main()
