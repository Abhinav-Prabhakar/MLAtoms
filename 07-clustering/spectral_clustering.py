"""Spectral clustering — cut the graph, not the space.

Chapter:   07-clustering
Companion: spectral_clustering.md
Tier 1 · 5 variants: normalized, unnormalized, ng_jordan_weiss, self_tuning, multiway

Spectral clustering turns a similarity graph into a linear problem. Build an
affinity matrix W (a k-NN graph with RBF weights), form the graph Laplacian, and
read the cluster structure from its smallest eigenvectors:

    L_sym = I - D^{-1/2} W D^{-1/2}

The first k eigenvectors of L_sym embed the points in R^k, where clusters that
were tangled in the original space become separated blobs, and k-means finishes
the job. Nothing about the geometry is assumed — two interleaving moons are two
clusters because the graph between them is weak, not because they are far apart.

    normalized       random-walk Laplacian (Shi-Malik), k-means on V
    unnormalized     L = D - W, k-means on the raw eigenvectors
    ng_jordan_weiss   symmetric Laplacian, rows of U normalised to unit length
    self_tuning      local scaling: sigma_i sigma_j instead of one global sigma
    multiway         recursive bipartition by the Fiedler vector
"""

from __future__ import annotations

import numpy as np
from kmeans import KMeans

VARIANTS = frozenset(
    ["normalized", "unnormalized", "ng_jordan_weiss", "self_tuning", "multiway"]
)

EPS = 1e-12


class SpectralClustering:
    """Graph-spectral clustering with an RBF k-NN affinity.

    `n_neighbors` sets the graph degree, `sigma` the RBF width (the median
    neighbour distance by default). `local_scale_neighbors` is the neighbour
    order used by `self_tuning`.
    """

    def __init__(
        self,
        variant: str = "normalized",
        *,
        n_clusters: int = 2,
        n_neighbors: int = 10,
        sigma: float | None = None,
        full_graph: bool = False,
        local_scale_neighbors: int = 7,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.k = int(n_clusters)
        self.n_neighbors = int(n_neighbors)
        self.sigma = sigma
        self.full_graph = bool(full_graph) or variant == "self_tuning"
        self.local_scale_neighbors = int(local_scale_neighbors)
        self.rng = np.random.default_rng(random_state)
        self.n_eig_ = 0
        self.n_dist_ = 0

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sq_dists(X: np.ndarray) -> np.ndarray:
        d2 = (X * X).sum(axis=1)[:, None] - 2.0 * X @ X.T + (X * X).sum(axis=1)[None]
        return np.maximum(d2, 0.0)

    def _affinity(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        d2 = self._sq_dists(X)
        self.n_dist_ = n * n
        k = min(self.n_neighbors, n - 1)

        # Union of the k-NN graphs: an edge exists if either endpoint chose it.
        idx = np.argpartition(d2, k, axis=1)[:, :k + 1]
        mask = np.zeros((n, n), dtype=bool)
        mask[np.arange(n)[:, None], idx] = True
        mask |= mask.T
        np.fill_diagonal(mask, False)

        if self.variant == "self_tuning":
            order = np.argsort(d2, axis=1)
            scale = np.sqrt(np.maximum(
                d2[np.arange(n), order[:, self.local_scale_neighbors]], EPS))
            denom = scale[:, None] * scale[None, :]
            W = np.exp(-d2 / np.maximum(denom, EPS))
        else:
            sigma = self.sigma
            if sigma is None:
                knn = np.sqrt(np.maximum(np.sort(d2, axis=1)[:, k], EPS))
                sigma = float(np.median(knn))
            W = np.exp(-d2 / (2.0 * max(sigma, EPS) ** 2))
        if not self.full_graph:
            W = np.where(mask, W, 0.0)
        return 0.5 * (W + W.T)

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def _laplacian_eigen(self, W: np.ndarray, k: int):
        n = W.shape[0]
        d = W.sum(axis=1)
        if self.variant == "unnormalized":
            L = np.diag(d) - W
            vals, vecs = np.linalg.eigh(L)
            self.n_eig_ += 1
            return vals, vecs[:, :k]
        dinv = 1.0 / np.sqrt(np.maximum(d, EPS))
        L = np.eye(n) - dinv[:, None] * W * dinv[None, :]
        vals, vecs = np.linalg.eigh(L)
        self.n_eig_ += 1
        if self.variant in ("ng_jordan_weiss", "self_tuning"):
            U = vecs[:, :k]
            norm = np.linalg.norm(U, axis=1, keepdims=True)
            return vals, U / np.maximum(norm, EPS)
        # random-walk eigenvectors: v = D^{-1/2} u
        V = dinv[:, None] * vecs[:, :k]
        return vals, V

    def _fiedler(self, W: np.ndarray, idx: np.ndarray) -> np.ndarray:
        sub = W[np.ix_(idx, idx)]
        d = sub.sum(axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(d, EPS))
        L = np.eye(idx.size) - dinv[:, None] * sub * dinv[None, :]
        _, vecs = np.linalg.eigh(L)
        self.n_eig_ += 1
        return dinv * vecs[:, 1] if idx.size > 1 else vecs[:, 0]

    def _multiway(self, W: np.ndarray) -> np.ndarray:
        n = W.shape[0]
        clusters = [np.arange(n)]
        while len(clusters) < self.k:
            j = max(range(len(clusters)), key=lambda i: clusters[i].size)
            idx = clusters.pop(j)
            if idx.size < 2:
                clusters.append(idx)
                break
            v = self._fiedler(W, idx)
            left = v > 0.0
            if left.all() or not left.any():
                left = v > float(np.median(v))
            if left.all() or not left.any():
                clusters.append(idx)
                break
            clusters.extend([idx[left], idx[~left]])
        labels = np.empty(n, dtype=int)
        for c, idx in enumerate(clusters):
            labels[idx] = c
        return labels

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> SpectralClustering:
        X = np.asarray(X, dtype=float)
        self.affinity_ = self._affinity(X)
        self.n_eig_ = 0
        if self.variant == "multiway":
            self.labels_ = self._multiway(self.affinity_)
            self.embedding_ = None
            self.eigenvalues_ = None
        else:
            self.eigenvalues_, self.embedding_ = self._laplacian_eigen(
                self.affinity_, self.k)
            km = KMeans("lloyd", n_clusters=self.k, n_init=10,
                        random_state=int(self.rng.integers(2**31 - 1)))
            self.labels_ = km.fit_predict(self.embedding_)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _match_accuracy(labels: np.ndarray, truth: np.ndarray) -> float:
    import itertools
    k = int(max(labels.max(), truth.max())) + 1
    best = 0.0
    for perm in itertools.permutations(range(k)):
        best = max(best, float(np.mean(np.asarray(perm)[labels] == truth)))
    return best


def main() -> None:
    rng = np.random.default_rng(0)

    # --- two moons: k-means cuts the wrong way ---
    t = rng.uniform(0.0, np.pi, 150)
    moon_a = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0.0, 0.07, (150, 2))
    moon_b = np.column_stack([1.0 - np.cos(t), 0.5 - np.sin(t)])
    moon_b = moon_b + rng.normal(0.0, 0.07, (150, 2))
    Xm = np.vstack([moon_a, moon_b])
    ym = np.r_[np.zeros(150, int), np.ones(150, int)]

    km_acc = _match_accuracy(
        KMeans("lloyd", n_clusters=2, n_init=10, random_state=0).fit_predict(Xm), ym)
    spec = SpectralClustering("normalized", n_clusters=2, n_neighbors=10).fit(Xm)
    acc = _match_accuracy(spec.labels_, ym)
    assert acc > 0.95, acc
    assert acc > km_acc + 0.2, (acc, km_acc)
    print(f"spectral: normalized moons acc {acc:.3f} vs kmeans {km_acc:.3f}, "
          f"smallest eigenvalues {np.round(spec.eigenvalues_[:3], 4)}")

    # --- circles: the graph, not the radius, separates the rings ---
    ang = rng.uniform(0.0, 2.0 * np.pi, 150)
    inner = 0.7 * np.column_stack([np.cos(ang), np.sin(ang)])
    outer = 2.0 * np.column_stack([np.cos(ang), np.sin(ang)])
    Xc = np.vstack([inner, outer]) + rng.normal(0.0, 0.05, (300, 2))
    yc = np.r_[np.zeros(150, int), np.ones(150, int)]

    unm = SpectralClustering("unnormalized", n_clusters=2, n_neighbors=8).fit(Xc)
    acc_u = _match_accuracy(unm.labels_, yc)
    assert acc_u > 0.95, acc_u
    print(f"spectral: unnormalized circles acc {acc_u:.3f}, "
          f"lambda1 {unm.eigenvalues_[1]:.4f}")

    # --- Ng-Jordan-Weiss: unit rows in the embedding ---
    njw = SpectralClustering("ng_jordan_weiss", n_clusters=2, n_neighbors=8).fit(Xc)
    norms = np.linalg.norm(njw.embedding_, axis=1)
    assert np.allclose(norms, 1.0), norms.min()
    acc_n = _match_accuracy(njw.labels_, yc)
    assert acc_n > 0.95, acc_n
    print(f"spectral: ng_jordan_weiss circles acc {acc_n:.3f}, "
          f"row norms {norms.min():.3f}..{norms.max():.3f}")

    # --- self-tuning: local scaling needs no global sigma ---
    fixed = SpectralClustering("normalized", n_clusters=2, n_neighbors=10,
                               full_graph=True, sigma=1.0).fit(Xm)
    local = SpectralClustering("self_tuning", n_clusters=2, n_neighbors=10).fit(Xm)
    acc_f = _match_accuracy(fixed.labels_, ym)
    acc_l = _match_accuracy(local.labels_, ym)
    assert acc_l > 0.95, acc_l
    assert acc_l > acc_f + 0.15, (acc_l, acc_f)
    print(f"spectral: self_tuning moons acc {acc_l:.3f} vs global sigma "
          f"1.0 {acc_f:.3f}")

    # --- multiway: recursive Fiedler splits, three rings ---
    angles = rng.uniform(0.0, 2.0 * np.pi, 360)
    radii = np.repeat([0.6, 1.4, 2.4], 120)
    Xr = radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    Xr = Xr + rng.normal(0.0, 0.04, Xr.shape)
    yr = np.repeat(np.arange(3), 120)
    mw = SpectralClustering("multiway", n_clusters=3, n_neighbors=8).fit(Xr)
    acc_m = _match_accuracy(mw.labels_, yr)
    assert acc_m > 0.95, acc_m
    assert mw.n_eig_ == 2, mw.n_eig_
    print(f"spectral: multiway 3 rings acc {acc_m:.3f} after {mw.n_eig_} "
          f"Fiedler solves")

    # --- the eigengap is the evidence that k=2 is the right number ---
    assert spec.embedding_.shape == (300, 2)
    assert spec.eigenvalues_[1] < 1e-6, spec.eigenvalues_[1]
    assert spec.eigenvalues_[2] > 1e-3, spec.eigenvalues_[2]
    print(f"spectral: moons lambda0 {spec.eigenvalues_[0]:.1e}, lambda1 "
          f"{spec.eigenvalues_[1]:.1e}, lambda2 {spec.eigenvalues_[2]:.4f}")

    print("spectral_clustering demo passed")


if __name__ == "__main__":
    main()
