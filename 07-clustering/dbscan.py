"""DBSCAN — clusters as connected regions of high density.

Chapter:   07-clustering
Companion: dbscan.md
Tier 1 · 4 variants: standard, with_index, weighted, fuzzy_dbscan

DBSCAN defines a cluster by density, not by a centroid. With radius `eps` and a
count `min_samples`:

    core point     has at least min_samples neighbours within eps
    border point   is within eps of a core point but is not core itself
    noise          neither

Clusters are the connected components of the core points (connected when two core
points are within eps of each other), and border points join the component that
reaches them first. K is never specified: it is however many components the
density landscape has.

    standard       the algorithm above, brute-force neighbourhoods
    with_index     the same answers through a grid index over eps-sized cells
    weighted       density is a sum of sample weights, not a row count
    fuzzy_dbscan   membership degrees instead of a hard label, via a second radius
"""

from __future__ import annotations

import itertools

import numpy as np

VARIANTS = frozenset(["standard", "with_index", "weighted", "fuzzy_dbscan"])

EPS = 1e-12


class DBSCAN:
    """Density-based spatial clustering.

    `eps` is the neighbourhood radius and `min_samples` the density threshold.
    `eps2` is the outer radius of `fuzzy_dbscan` (default `1.5 * eps`); points
    between the two radii get a partial membership.
    """

    def __init__(
        self,
        variant: str = "standard",
        *,
        eps: float = 0.5,
        min_samples: int = 5,
        eps2: float | None = None,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.eps2 = float(eps2) if eps2 is not None else 1.5 * self.eps
        if not self.eps2 > self.eps:
            raise ValueError("eps2 must exceed eps")
        self.rng = np.random.default_rng(random_state)
        self.n_dist_ = 0      # pairwise distances computed
        self.n_queries_ = 0   # neighbourhood queries issued

    # ------------------------------------------------------------------ #
    # Neighbourhoods
    # ------------------------------------------------------------------ #
    def _build_grid(self, X: np.ndarray) -> None:
        keys = np.floor(X / self.eps).astype(int)
        grid: dict[tuple[int, ...], list[int]] = {}
        for i, key in enumerate(map(tuple, keys)):
            grid.setdefault(key, []).append(i)
        self._grid = grid

    def _candidates(self, X: np.ndarray, i: int) -> np.ndarray:
        key = tuple(np.floor(X[i] / self.eps).astype(int))
        out: list[int] = []
        for offset in itertools.product((-1, 0, 1), repeat=X.shape[1]):
            cell = tuple(k + o for k, o in zip(key, offset, strict=True))
            out.extend(self._grid.get(cell, ()))
        return np.sort(np.array(out, dtype=int))

    def _region(self, X: np.ndarray, i: int, use_index: bool):
        self.n_queries_ += 1
        cand = self._candidates(X, i) if use_index else np.arange(X.shape[0])
        self.n_dist_ += int(cand.size)
        d = np.linalg.norm(X[cand] - X[i], axis=1)
        keep = d <= self.eps + EPS
        return cand[keep], d[keep]

    def _is_core(self, idx: np.ndarray) -> bool:
        if self.variant == "weighted":
            return float(self._w[idx].sum()) >= self.min_samples
        return idx.size >= self.min_samples

    # ------------------------------------------------------------------ #
    # Algorithms
    # ------------------------------------------------------------------ #
    def _dbscan(self, X: np.ndarray, use_index: bool) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        core = np.zeros(n, dtype=bool)
        cluster = -1
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            idx, _ = self._region(X, i, use_index)
            if not self._is_core(idx):
                continue
            core[i] = True
            cluster += 1
            labels[i] = cluster
            seeds = list(idx)
            head = 0
            while head < len(seeds):
                j = int(seeds[head])
                head += 1
                if not visited[j]:
                    visited[j] = True
                    nbr, _ = self._region(X, j, use_index)
                    if self._is_core(nbr):
                        core[j] = True
                        seeds.extend(nbr)
                if labels[j] < 0:
                    labels[j] = cluster
        return labels, core

    def _fuzzy(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        nbrs = [self._region(X, i, False)[0] for i in range(n)]
        core = np.array([self._is_core(nb) for nb in nbrs], dtype=bool)

        # Connected components of the core graph.
        comp = np.full(n, -1, dtype=int)
        k = -1
        for i in np.nonzero(core)[0]:
            if comp[i] >= 0:
                continue
            k += 1
            comp[i] = k
            stack = [int(i)]
            while stack:
                a = stack.pop()
                for b in nbrs[a]:
                    b = int(b)
                    if core[b] and comp[b] < 0:
                        comp[b] = k
                        stack.append(b)
        k += 1
        self.n_clusters_ = k
        if k == 0:
            self.membership_ = np.zeros((n, 0))
            self.noise_degree_ = np.ones(n)
            return np.full(n, -1, dtype=int), core

        # Distance from every point to the nearest core point of each cluster.
        d = np.full((n, k), np.inf)
        for a in np.nonzero(core)[0]:
            dist = np.linalg.norm(X - X[a], axis=1)
            self.n_dist_ += n
            c = comp[a]
            d[:, c] = np.minimum(d[:, c], dist)

        span = self.eps2 - self.eps
        mu = np.clip((self.eps2 - d) / span, 0.0, 1.0)
        mu[core] = 0.0
        mu[core, comp[core]] = 1.0
        total = mu.sum(axis=1)
        over = total > 1.0
        mu[over] /= total[over, None]
        self.membership_ = mu
        self.noise_degree_ = 1.0 - mu.sum(axis=1)
        labels = np.where(mu.max(axis=1) >= 0.5, np.argmax(mu, axis=1), -1)
        return labels, core

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, sample_weight: np.ndarray | None = None) -> DBSCAN:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if sample_weight is None:
            self._w = np.ones(n)
        else:
            self._w = np.asarray(sample_weight, dtype=float)
            if self._w.shape != (n,):
                raise ValueError("sample_weight must have one entry per row")
        self.n_dist_ = 0
        self.n_queries_ = 0

        if self.variant == "fuzzy_dbscan":
            labels, core = self._fuzzy(X)
        elif self.variant == "with_index":
            self._build_grid(X)
            labels, core = self._dbscan(X, use_index=True)
        else:
            labels, core = self._dbscan(X, use_index=False)

        self.labels_ = labels
        self.core_sample_indices_ = np.nonzero(core)[0]
        keep = labels[labels >= 0]
        self.n_clusters_ = int(keep.max()) + 1 if keep.size else 0
        self.n_noise_ = int((labels < 0).sum())
        return self

    def fit_predict(self, X: np.ndarray,
                    sample_weight: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, sample_weight).labels_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _match_accuracy(labels: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Best accuracy over cluster permutations, restricted to `mask` rows."""
    ids = np.unique(labels[mask])
    ids = ids[ids >= 0]
    if ids.size == 0:
        return 0.0
    best = 0.0
    for perm in itertools.permutations(range(ids.max() + 1)):
        mapped = np.asarray(perm)[labels]
        best = max(best, float(np.mean(mapped[mask] == truth[mask])))
    return best


def main() -> None:
    rng = np.random.default_rng(0)

    # --- two dense blobs inside a sparse background ---
    n_a, n_b, n_n = 300, 300, 60
    Xa = rng.normal([0.0, 0.0], 0.45, size=(n_a, 2))
    Xb = rng.normal([4.5, 0.0], 0.45, size=(n_b, 2))
    Xn = rng.uniform([-2.0, -2.0], [6.5, 2.0], size=(n_n, 2))
    X = np.vstack([Xa, Xb, Xn])
    truth = np.r_[np.zeros(n_a, int), np.ones(n_b, int), -np.ones(n_n, int)]
    inlier = truth >= 0

    db = DBSCAN("standard", eps=0.35, min_samples=8).fit(X)
    acc = _match_accuracy(db.labels_, truth, inlier)
    assert db.n_clusters_ == 2, db.n_clusters_
    assert acc > 0.95, acc
    print(f"dbscan: standard {db.n_clusters_} clusters, {db.n_noise_} noise, "
          f"inlier acc {acc:.3f}, dist evals {db.n_dist_}")

    # --- the grid index returns the same partition with fewer distance evals ---
    ix = DBSCAN("with_index", eps=0.35, min_samples=8).fit(X)
    assert np.array_equal(ix.labels_, db.labels_), "index must be exact"
    assert np.array_equal(ix.core_sample_indices_, db.core_sample_indices_)
    assert ix.n_dist_ < db.n_dist_ / 4, (ix.n_dist_, db.n_dist_)
    print(f"dbscan: with_index same labels, dist evals {ix.n_dist_} "
          f"vs {db.n_dist_}")

    # --- weights stop duplicated rows from manufacturing a cluster ---
    Xa_w = rng.normal([0.0, 0.0], 0.25, size=(40, 2))
    Xb_w = rng.normal([4.5, 0.0], 0.25, size=(40, 2))
    dups = np.repeat(np.array([[2.25, 0.0]]), 60, axis=0)
    Xd = np.vstack([Xa_w, Xb_w, dups])
    w = np.r_[np.ones(80), np.full(60, 1.0 / 60.0)]
    row_db = DBSCAN("standard", eps=0.35, min_samples=5).fit(Xd)
    w_db = DBSCAN("weighted", eps=0.35, min_samples=5).fit(Xd, w)
    assert row_db.n_clusters_ == 3, row_db.n_clusters_
    assert w_db.n_clusters_ == 2, w_db.n_clusters_
    assert np.all(w_db.labels_[80:] == -1), w_db.labels_[80:]
    print(f"dbscan: weighted {w_db.n_clusters_} clusters vs {row_db.n_clusters_} "
          f"by row count; duplicated rows are "
          f"{'noise' if np.all(w_db.labels_[80:] < 0) else 'clustered'}")

    # --- fuzzy: degrees instead of labels ---
    fz = DBSCAN("fuzzy_dbscan", eps=0.35, min_samples=8, eps2=0.7).fit(X)
    mu, noise = fz.membership_, fz.noise_degree_
    assert mu.shape == (X.shape[0], fz.n_clusters_)
    assert np.all(mu >= -EPS) and np.all(mu <= 1.0 + EPS)
    assert np.all(noise >= -EPS) and np.all(noise <= 1.0 + EPS)
    assert np.all(mu.sum(axis=1) + noise <= 1.0 + 1e-9)
    assert np.allclose(mu[fz.core_sample_indices_].max(axis=1), 1.0)
    partial = ((mu > 0.05) & (mu < 0.95)).any(axis=1)
    assert partial.sum() > 0, partial.sum()
    acc_f = _match_accuracy(fz.labels_, truth, inlier)
    assert acc_f > 0.95, acc_f
    print(f"dbscan: fuzzy {fz.n_clusters_} clusters, {partial.sum()} partial "
          f"memberships, noise degree mean {noise[~inlier].mean():.2f} vs "
          f"{noise[inlier].mean():.2f}, acc {acc_f:.3f}")

    print("dbscan demo passed")


if __name__ == "__main__":
    main()
