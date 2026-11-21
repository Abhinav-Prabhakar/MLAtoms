"""Isomap — geodesic distances from a neighbour graph, then classical MDS.

Chapter:   08-manifold
Companion: isomap.md
Tier 1 · 4 variants: isomap, landmark, conformal, sparse

    G = kNN graph of X,   w_ij = ‖x_i − x_j‖                                (1)
    d_ij = length of the shortest path i → j in G                           (2)
    Y = classical_mds(d)                                                    (3)

Isomap is one idea with a large consequence: replace the Euclidean distance
between two points with the length of the shortest path through the data. On a
swiss roll the ambient distance between two points on adjacent turns is small and
their distance along the surface is large, so ambient distances lie about the
geometry; graph distances do not, because the path has to walk along the sheet.

The price is that (2) is only a lower bound on the true geodesic when the graph
has an edge across the fold — a *short-circuit* — and that computing it is
all-pairs shortest paths. The four variants trade those two costs differently:
`isomap` is the reference, `landmark` computes distances from a subset of nodes
and triangulates the rest, `conformal` rescales the input metric by a local scale
estimated from the sampling density, and `sparse` computes exact distances from a
subset of sources and completes the matrix by a min-plus closure.

The demo measures the claim and the failure: on a swiss roll the geodesic
embedding tracks the exact isometric parameterisation where ambient MDS does not,
and the same code collapses when `n_neighbors` grows past the point where edges
jump between turns.
"""

from __future__ import annotations

import heapq

import numpy as np

EPS = 1e-12
VARIANTS = ("isomap", "landmark", "conformal", "sparse")


# --------------------------------------------------------------------------- #
# Graph, distances and the MDS core
# --------------------------------------------------------------------------- #
def _pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix via the Gram identity, `(n, n)`, `O(n²)` memory.

    The `(n, n, d)` broadcast of differences is `O(n²d)` memory — 1.3 GB at
    n = 400, d = 1000 — for a number that fits in `n²` doubles. The clamp matters
    because the identity is exact in reals and can go negative in floating point,
    and `sqrt` of a small negative is `nan`.
    """
    sq = np.einsum("ij,ij->i", X, X)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))


def _knn(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of each point's `k` nearest neighbours, excluding itself, `(n, k)`."""
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def _weight_matrix(X: np.ndarray, k: int) -> np.ndarray:
    """`(n, n)` edge weights: `‖x_i − x_j‖` for kNN edges (symmetrised), `inf` else.

    Symmetrising by union rather than intersection is the usual choice: it keeps
    the graph connected where a one-sided kNN rule would not, at the cost of a few
    nodes with degree above `k`.
    """
    n = X.shape[0]
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    W = np.full((n, n), np.inf)
    np.fill_diagonal(W, 0.0)
    np.minimum.at(W, (rows, cols), D[rows, cols])
    np.minimum.at(W, (cols, rows), D[rows, cols])
    return W


def _adjacency(W: np.ndarray) -> list[list[tuple[int, float]]]:
    """Sparse adjacency lists from the dense weight matrix, for Dijkstra."""
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    keep = np.isfinite(W[iu])
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, w in zip(iu[0][keep], iu[1][keep], W[iu][keep], strict=True):
        adj[i].append((int(j), float(w)))
        adj[j].append((int(i), float(w)))
    return adj


def _n_components(W: np.ndarray) -> int:
    """Number of connected components of the graph of finite edges."""
    n = W.shape[0]
    seen = np.zeros(n, dtype=bool)
    count = 0
    for s in range(n):
        if seen[s]:
            continue
        count += 1
        seen[s] = True
        stack = [s]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(np.isfinite(W[i])):
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
    return count


def _floyd(W: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths by vectorised Floyd–Warshall, `O(n³)` time."""
    D = W.copy()
    for k in range(D.shape[0]):
        np.minimum(D, D[:, k][:, None] + D[k][None, :], out=D)
    return D


def _dijkstra(adj: list[list[tuple[int, float]]], source: int, n: int) -> np.ndarray:
    """Single-source shortest paths on the sparse graph; `inf` where unreachable."""
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue                       # stale heap entry, already improved
        for j, w in adj[i]:
            candidate = d + w
            if candidate < dist[j]:
                dist[j] = candidate
                heapq.heappush(heap, (candidate, j))
    return dist


def _classical(D: np.ndarray, n_components: int) -> np.ndarray:
    """Torgerson scaling: eigenvectors of the double-centred squared distances."""
    n = D.shape[0]
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ (D ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1][:n_components]
    return evecs[:, order] * np.sqrt(np.maximum(evals[order], 0.0))


def _stress(Y: np.ndarray, D: np.ndarray) -> float:
    """Normalised raw stress, `Σ(d − δ)² / Σδ²`, so it is comparable across data."""
    d = _pairwise(Y)
    iu = np.triu_indices_from(D, k=1)
    return float(np.sum((d[iu] - D[iu]) ** 2) / max(float(np.sum(D[iu] ** 2)), EPS))


def _triangulate(D: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Place points from their distances to landmarks, by linear least squares.

    Expand `‖y − y_l‖² = D_li²`, subtract the equation for landmark 0 and the
    quadratic term cancels:

        2 (y_l − y_0)·y = D_0i² − D_li² + ‖y_l‖² − ‖y_0‖²,

    which is `k` unknowns and `m − 1` equations per point: over-determined as soon
    as `m ≥ k + 2`, and then the least-squares solution is the triangulation.
    """
    base = np.sum(Y[1:] ** 2, axis=1) - float(np.sum(Y[0] ** 2))
    rows = 2.0 * (Y[1:] - Y[0])                              # (m − 1, k)
    rhs = D[0][None, :] ** 2 - D[1:] ** 2 + base[:, None]    # (m − 1, points)
    coef, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    return coef.T


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class Isomap:
    """Geodesic embedding of a point cloud.

    `Isomap(n_components=2, *, variant="isomap", n_neighbors=10,
    n_landmarks=None, seed=0)` with `fit(X)`, `fit_transform(X)`, and the
    attributes `embedding_` `(n, n_components)`, `distances_` (the geodesic
    matrix the embedding fits), `stress_`, and `landmarks_` / `scales_` for the
    variants that use them.

    `n_neighbors` sets the kNN graph and doubles as the neighbourhood whose mean
    radius estimates the conformal factor. `n_landmarks` defaults to
    `max(n_components + 2, n // 10)` for `landmark` and `sparse`; `seed` only
    chooses the sparse variant's source nodes.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "isomap",
                 n_neighbors: int = 10, n_landmarks: int | None = None,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if n_landmarks is not None and n_landmarks <= n_components + 1:
            raise ValueError("n_landmarks must exceed n_components + 1")
        self.n_components = int(n_components)
        self.variant = variant
        self.n_neighbors = int(n_neighbors)
        self.n_landmarks = n_landmarks
        self.seed = int(seed)

    # ------------------------------------------------------------- helpers ---- #
    def _landmark_count(self, n: int) -> int:
        """`n_landmarks`, or a tenth of the data with a floor for triangulation."""
        if self.n_landmarks is not None:
            return int(min(self.n_landmarks, n))
        return int(min(n, max(self.n_components + 2, n // 10)))

    def _farthest_point(self, D: np.ndarray, m: int) -> np.ndarray:
        """Deterministic farthest-point sampling: start at the point nearest the
        centroid, then repeatedly take the point farthest from everything chosen.
        """
        first = int(np.argmin(np.sum(D ** 2, axis=1)))
        chosen = [first]
        dist = D[first].copy()
        dist[first] = -1.0
        for _ in range(m - 1):
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, D[nxt])
            dist[nxt] = -1.0
        return np.asarray(chosen)

    # ----------------------------------------------------------------- fit ---- #
    def fit(self, X: np.ndarray) -> Isomap:
        """Build the graph, compute geodesics and embed; returns `self`."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (samples, features), got shape {X.shape}")
        n = X.shape[0]
        if self.n_neighbors >= n:
            raise ValueError("n_neighbors must be smaller than the number of samples")
        if n <= self.n_components:
            raise ValueError("need more samples than components")

        W = _weight_matrix(X, self.n_neighbors)
        if _n_components(W) > 1:
            raise ValueError(
                f"the {self.n_neighbors}-neighbour graph is disconnected; "
                "increase n_neighbors or sample more densely"
            )

        self.landmarks_ = None
        self.scales_ = None
        if self.variant == "landmark":
            self._fit_landmark(X, W)
        elif self.variant == "sparse":
            self._fit_sparse(W)
        elif self.variant == "conformal":
            self._fit_conformal(X, W)
        else:
            D = _floyd(W)
            self.embedding_ = _classical(D, self.n_components)
            self.distances_ = D
            self.stress_ = _stress(self.embedding_, D)
        return self

    def _fit_conformal(self, X: np.ndarray, W: np.ndarray) -> None:
        """Divide the geodesic metric by a local scale estimated from the sampling.

        If the data are uniform in the metric we want and the measured space is
        stretched by `λ`, the mean radius of a k-neighbourhood is `λ` times what
        it would be, so it estimates `λ` up to a constant. A conformal distortion
        scales distances near `i` by `λ_i`, hence `d_ij / √(λ_i λ_j)`; the
        constant is irrelevant because classical MDS is scale-equivariant.

        One neighbourhood is noisy — a Poisson process gives its radius a
        standard error of order `1/√k` — so the value used is the mean of the raw
        radii over one hop of the same graph, which averages `k²`-ish points
        without blurring `λ` on the scale of the point spacing.
        """
        D = _floyd(W)
        n = X.shape[0]
        idx = _knn(X, min(self.n_neighbors, n - 1))
        radius = _pairwise(X)[np.arange(n)[:, None], idx].mean(axis=1)
        smoothed = radius[idx].mean(axis=1)
        self.scales_ = smoothed / max(float(np.exp(np.mean(np.log(smoothed)))), EPS)
        scaled = D / np.sqrt(self.scales_[:, None] * self.scales_[None, :])
        self.embedding_ = _classical(scaled, self.n_components)
        self.distances_ = scaled
        self.stress_ = _stress(self.embedding_, scaled)

    def _fit_landmark(self, X: np.ndarray, W: np.ndarray) -> None:
        """Distance from a subset of landmarks, then triangulate everything else."""
        n = X.shape[0]
        m = self._landmark_count(n)
        if m < self.n_components + 2:
            raise ValueError("not enough landmarks to triangulate")
        land = self._farthest_point(_pairwise(X), m)
        adj = _adjacency(W)
        D_land = np.stack([_dijkstra(adj, int(s), n) for s in land])   # (m, n)
        Y = np.empty((n, self.n_components))
        Y[land] = _classical(D_land[:, land], self.n_components)
        rest = np.setdiff1d(np.arange(n), land, assume_unique=False)
        Y[rest] = _triangulate(D_land[:, rest], Y[land])

        self.landmarks_ = land
        self.embedding_ = Y
        # the distances this variant actually fits are those to the landmarks
        d = _pairwise(Y)[np.arange(n)[:, None], land[None, :]]
        self.stress_ = float(np.sum((d - D_land.T) ** 2)
                             / max(float(np.sum(D_land ** 2)), EPS))
        self.distances_ = D_land

    def _fit_sparse(self, W: np.ndarray) -> None:
        """Exact rows from a subset of sources, completed by a min-plus closure.

        `D̂_ij = min_l (D_il + D_lj)` over the source set is an upper bound on the
        true geodesic (each term is a walk through `l`), and it is exact when one
        of `i`, `j` is a source. The cost is `m` single-source runs plus `m n²`
        for the closure, against `n` runs for the full matrix.
        """
        n = W.shape[0]
        m = self._landmark_count(n)
        rng = np.random.default_rng(self.seed)
        sources = rng.permutation(n)[:m]
        adj = _adjacency(W)
        rows = np.stack([_dijkstra(adj, int(s), n) for s in sources])  # (m, n)

        D = np.full((n, n), np.inf)
        for row in rows:
            np.minimum(D, row[:, None] + row[None, :], out=D)
        np.fill_diagonal(D, 0.0)

        self.landmarks_ = np.sort(sources)
        self.embedding_ = _classical(D, self.n_components)
        self.distances_ = D
        self.stress_ = _stress(self.embedding_, D)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """`fit(X).embedding_`."""
        return self.fit(X).embedding_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _swiss_roll(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on the roll, and their exact isometric coordinates `(s, h)`.

    The surface `p(t, h) = (t cos t, h, t sin t)` has induced metric
    `ds² = (1 + t²) dt² + dh²`, so `s(t) = ½(t√(1+t²) + asinh t)` and `h` are
    arc-length coordinates: the roll is *exactly* isometric to a flat rectangle.
    That gives the demo a ground truth, not an approximation of one.
    """
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    h = 12.0 * rng.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    s = 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))
    return X, np.column_stack([s, h])


def _distance_correlation(Y: np.ndarray, truth: np.ndarray) -> float:
    """Correlation between embedded and true distances — rotation-free."""
    iu = np.triu_indices(truth.shape[0], k=1)
    return float(np.corrcoef(_pairwise(Y)[iu], _pairwise(truth)[iu])[0, 1])


def _neighbour_overlap(Y: np.ndarray, truth: np.ndarray, k: int = 8) -> float:
    """Mean fraction of each point's true `k` nearest neighbours kept by `Y`."""
    a, b = _pairwise(truth), _pairwise(Y)
    kept = []
    for i in range(truth.shape[0]):
        left = set(np.argsort(a[i])[1:k + 1])
        right = set(np.argsort(b[i])[1:k + 1])
        kept.append(len(left & right) / k)
    return float(np.mean(kept))


def _demo_isomap() -> None:
    """The geodesic is the right distance, and too many neighbours destroys it."""
    X, truth = _swiss_roll(600, seed=1)
    iso = Isomap(2, variant="isomap", n_neighbors=10).fit(X)
    r_iso = _distance_correlation(iso.embedding_, truth)
    knn_iso = _neighbour_overlap(iso.embedding_, truth)

    # ambient distances lie about the geometry: points on neighbouring turns are
    # close in R³ and far along the sheet
    ambient = _classical(_pairwise(X), 2)
    r_amb = _distance_correlation(ambient, truth)

    assert r_iso > 0.99 and knn_iso > 0.8, (r_iso, knn_iso)
    assert r_amb < 0.8 * r_iso, (r_amb, r_iso)

    sweep = []
    for k in (7, 10, 14, 16, 32):
        fit = Isomap(2, variant="isomap", n_neighbors=k).fit(X)
        sweep.append((k, _distance_correlation(fit.embedding_, truth),
                      float(np.max(fit.distances_))))

    print(
        f"isomap: {X.shape[0]} points on a swiss roll, which is exactly isometric "
        f"to a flat rectangle with coordinates (s, h). Graph geodesics over a "
        f"10-neighbour graph, then classical MDS, recover that rectangle at "
        f"distance correlation {r_iso:.4f} with {knn_iso:.3f} of each point's 8 "
        f"local neighbours intact. Classical MDS on the ambient Euclidean "
        f"distances gets {r_amb:.4f}: points on neighbouring turns are close in "
        f"R³ and far along the sheet, so the ambient metric is not the geometry. "
        f"Growing the neighbourhood past the gap between turns makes it worse, "
        f"not better — correlation "
        + ", ".join(f"{k}→{r:.4f}" for k, r, _ in sweep)
        + f" — because an edge across the fold short-circuits every geodesic that "
        f"crosses it, and max graph distance falls to "
        f"{sweep[-1][2]:.1f} from {sweep[1][2]:.1f} as the shortcuts open"
    )


def _demo_landmark() -> None:
    """A tenth of the shortest-path runs, and the same picture."""
    X, truth = _swiss_roll(700, seed=2)
    full = Isomap(2, variant="isomap", n_neighbors=10).fit(X)
    land = Isomap(2, variant="landmark", n_neighbors=10,
                  n_landmarks=70).fit(X)
    r_land = _distance_correlation(land.embedding_, truth)
    r_full = _distance_correlation(full.embedding_, truth)
    iu = np.triu_indices(X.shape[0], k=1)
    agree = float(np.corrcoef(_pairwise(land.embedding_)[iu],
                              _pairwise(full.embedding_)[iu])[0, 1])

    assert r_land > 0.95 and agree > 0.98, (r_land, agree)

    print(
        f"landmark: {X.shape[0]} points, {land.landmarks_.size} landmarks chosen by "
        f"farthest-point sampling. Seventy single-source shortest-path runs "
        f"instead of {X.shape[0]}, then the landmarks are embedded by classical "
        f"MDS and every other point is placed by least-squares triangulation from "
        f"its distances to them. Correlation with the true rectangle "
        f"{r_land:.4f} against {r_full:.4f} for the full computation, and the two "
        f"embeddings agree at distance correlation {agree:.4f} — the residual "
        f"landmark stress is {land.stress_:.1e}, which is the triangulation "
        f"fitting distances it never measured directly"
    )


def _demo_conformal() -> None:
    """A conformal distortion, and the local scale that undoes it."""
    rng = np.random.default_rng(3)
    n = 500
    r = 2.0 * np.sqrt(rng.random(n))                  # uniform in the disk's area
    theta = 2.0 * np.pi * rng.random(n)
    plane = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    # stereographic projection: the sphere's metric is λ² × the plane's, with
    # λ = 2 / (1 + |z|²), and it is exactly conformal
    denom = 1.0 + np.sum(plane ** 2, axis=1)
    sphere = np.column_stack([2.0 * plane / denom[:, None],
                              (np.sum(plane ** 2, axis=1) - 1.0) / denom])

    raw = Isomap(2, variant="isomap", n_neighbors=10).fit(sphere)
    conf = Isomap(2, variant="conformal", n_neighbors=10).fit(sphere)
    r_raw = _distance_correlation(raw.embedding_, plane)
    r_conf = _distance_correlation(conf.embedding_, plane)
    corr_scale = float(np.corrcoef(conf.scales_, 1.0 / denom)[0, 1])
    assert r_conf > r_raw + 0.05, (r_conf, r_raw)

    print(
        f"conformal: {n} points uniform in a disk of radius 2, pushed onto the "
        f"sphere by stereographic projection, which is exactly conformal with "
        f"λ = 2/(1+|z|²). The sphere's geodesics are great circles and no 2-D "
        f"Euclidean embedding reproduces them: raw Isomap reaches correlation "
        f"{r_raw:.4f} with the plane it came from. Dividing each geodesic by "
        f"√(λ_i λ_j), where λ is estimated from the mean k-neighbour radius — "
        f"which tracks the true factor at correlation {corr_scale:.4f} — recovers "
        f"the plane at {r_conf:.4f}. The correction uses no information the graph "
        f"does not have; it uses the sampling density, which is the assumption the "
        f"method rests on and the one that fails silently when it is wrong"
    )

    # ... and here is what "wrong" looks like: flat data, Euclidean metric,
    # sampled with a density gradient and nothing to correct
    rng = np.random.default_rng(5)
    rad = 2.0 * rng.random(n) ** 1.5              # density ∝ 1/r
    ang = 2.0 * np.pi * rng.random(n)
    flat = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    flat_raw = Isomap(2, variant="isomap", n_neighbors=10).fit(flat)
    flat_conf = Isomap(2, variant="conformal", n_neighbors=10).fit(flat)
    f_raw = _distance_correlation(flat_raw.embedding_, flat)
    f_conf = _distance_correlation(flat_conf.embedding_, flat)
    assert f_raw > 0.99 and f_conf < 0.5, (f_raw, f_conf)

    print(
        f"conformal, unchecked: take the flat plane — Euclidean metric, nothing "
        f"to correct — and sample it with density ∝ 1/r. The same estimator reads "
        f"the density gradient as a conformal factor and divides it out: "
        f"correlation with the truth falls from {f_raw:.4f} for raw Isomap to "
        f"{f_conf:.4f}. Nothing in the data distinguishes 'the metric is stretched "
        f"here' from 'there are more points here', which is why C-Isomap is a "
        f"method for data whose sampling you control or believe in"
    )


def _demo_sparse() -> None:
    """Fewer sources, a min-plus closure, and the error that buys."""
    X, truth = _swiss_roll(600, seed=4)
    full = Isomap(2, variant="isomap", n_neighbors=10).fit(X)
    rows = []
    for m in (30, 60, 120, 240):
        fit = Isomap(2, variant="sparse", n_neighbors=10, n_landmarks=m,
                     seed=0).fit(X)
        rows.append((m, _distance_correlation(fit.embedding_, truth),
                     float(np.mean(fit.distances_ - full.distances_))))

    assert rows[0][1] < rows[-1][1] and rows[-1][1] > 0.95, rows
    assert all(gap >= -1e-9 for _, _, gap in rows), rows

    print(
        f"sparse: exact geodesics from 30, 60, 120 and 240 source nodes instead of "
        f"{X.shape[0]}, with the rest of the matrix completed by "
        f"D̂_ij = min_l (D_il + D_lj). Correlation with the true rectangle "
        f"{'/'.join(f'{r:.4f}' for _, r, _ in rows)} against "
        f"{_distance_correlation(full.embedding_, truth):.4f} for all "
        f"{X.shape[0]} sources. Every completion is an upper bound on the true "
        f"geodesic, so the mean gap to the exact matrix is non-negative and falls "
        f"from {rows[0][2]:.4f} to {rows[-1][2]:.4f}: fewer sources means a longer "
        f"walk through a source, never a shorter one"
    )


def main() -> None:
    _demo_isomap()
    _demo_landmark()
    _demo_conformal()
    _demo_sparse()
    print("isomap demo passed")


if __name__ == "__main__":
    main()
