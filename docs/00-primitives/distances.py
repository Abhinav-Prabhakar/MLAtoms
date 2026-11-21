"""Distances — 13 dissimilarities, vector, set, string and time-series.

Chapter:   00-primitives
Companion: distances.md
Tier 1 · 13 variants: euclidean, squared_euclidean, manhattan, chebyshev, cosine,
correlation, mahalanobis, jaccard, dice, hamming, levenshtein, dtw, haversine

`cdist(X, Y, metric)` returns the (n, m) pairwise matrix for vector metrics.
Levenshtein and DTW take sequences directly; haversine takes coordinates in degrees.
"""

from __future__ import annotations

import numpy as np

VECTOR = frozenset(
    ["euclidean", "squared_euclidean", "manhattan", "chebyshev", "cosine",
     "correlation", "mahalanobis", "jaccard", "dice", "hamming"]
)


def cdist(
    X: np.ndarray, Y: np.ndarray | None = None, metric: str = "euclidean", **kwargs
) -> np.ndarray:
    """Pairwise distances between rows of X and rows of Y (default Y = X).

    Returns (len(X), len(Y)). Diagonal of cdist(X, X) is exactly zero for the
    metrics where that is mathematically true; `euclidean` uses the expanded form
    and clips at zero to avoid `sqrt(-1e-16)`.
    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    if metric not in VECTOR:
        raise ValueError(f"unknown metric {metric!r}; use cdist_seq for sequences")

    if metric in ("euclidean", "squared_euclidean"):
        d2 = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
        d2 = np.maximum(d2, 0.0)
        if metric == "squared_euclidean":
            return d2
        np.fill_diagonal(d2, 0.0) if X is Y else None
        return np.sqrt(d2)
    if metric == "manhattan":
        return np.abs(X[:, None, :] - Y[None, :, :]).sum(-1)
    if metric == "chebyshev":
        return np.abs(X[:, None, :] - Y[None, :, :]).max(-1)
    if metric in ("cosine", "correlation"):
        if metric == "correlation":
            X = X - X.mean(1, keepdims=True)
            Y = Y - Y.mean(1, keepdims=True)
        nx = np.linalg.norm(X, axis=1)[:, None]
        ny = np.linalg.norm(Y, axis=1)[None, :]
        sim = X @ Y.T / np.maximum(nx * ny, 1e-12)
        return np.clip(1.0 - sim, 0.0, 2.0)
    if metric == "mahalanobis":
        cov = kwargs.get("cov")
        if cov is None:
            cov = np.cov(np.vstack([X, Y]).T)
        vi = np.linalg.pinv(cov)
        diff = X[:, None, :] - Y[None, :, :]
        return np.sqrt(np.maximum(np.einsum("ijk,kl,ijl->ij", diff, vi, diff), 0.0))
    if metric in ("jaccard", "dice"):
        num = np.minimum(X[:, None, :], Y[None, :, :]).sum(-1)
        den = np.maximum(X[:, None, :], Y[None, :, :]).sum(-1)
        if metric == "jaccard":
            return 1.0 - num / np.maximum(den, 1e-12)
        return 1.0 - 2.0 * num / np.maximum(X.sum(1)[:, None] + Y.sum(1)[None, :], 1e-12)
    if metric == "hamming":
        return (X[:, None, :] != Y[None, :, :]).mean(-1)
    raise ValueError(metric)


def levenshtein(a, b, substitute: float = 1.0) -> float:
    """Edit distance with unit insert/delete and `substitute` for replace."""
    a, b = list(a), list(b)
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))
    prev = np.arange(len(b) + 1, dtype=float)
    for i, ca in enumerate(a, start=1):
        cur = np.empty_like(prev)
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1.0,                      # delete
                cur[j - 1] + 1.0,                   # insert
                prev[j - 1] + (0.0 if ca == cb else substitute),
            )
        prev = cur
    return float(prev[-1])


def dtw(a: np.ndarray, b: np.ndarray, band: int | None = None) -> float:
    """Dynamic time warping distance: allows stretching one series onto the other.

    band limits |i - j| (Sakoe-Chiba band); None is the full O(nm) table.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n, m = a.size, b.size
    inf = np.inf
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0, j1 = 1, m + 1
        if band is not None:
            j0 = max(1, i - band)
            j1 = min(m, i + band) + 1
        for j in range(j0, j1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres. Inputs in degrees, any broadcast shape."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def main() -> None:
    # 1. Hand-checked Euclidean and its variants.
    P = np.array([[0.0, 0.0], [3.0, 4.0]])
    D = cdist(P, metric="euclidean")
    assert D[0, 1] == 5.0
    assert cdist(P, metric="squared_euclidean")[0, 1] == 25.0
    assert cdist(P, metric="manhattan")[0, 1] == 7.0
    assert cdist(P, metric="chebyshev")[0, 1] == 4.0
    print("metric identities ok (3-4-5 triangle)")

    # 2. Cosine vs correlation: a constant offset leaves cosine unchanged but
    #    does not affect correlation distance.
    a = np.array([[1.0, 2.0, 3.0]])
    b = a + 10.0
    assert cdist(a, b, "cosine")[0, 0] > 0.01
    assert abs(cdist(a, b, "correlation")[0, 0]) < 1e-12
    assert abs(cdist(a, a, "cosine")[0, 0]) < 1e-12
    print("cosine != correlation under shift")

    # 3. Mahalanobis with identity covariance is Euclidean.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    d_m = cdist(X, X, "mahalanobis", cov=np.eye(4))
    assert np.allclose(d_m, cdist(X, X, "euclidean"), atol=1e-8)
    print("mahalanobis(I) == euclidean")

    # 4. Set metrics.
    assert cdist(np.array([[1.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), "jaccard")[0, 0] == 0.5
    assert abs(cdist(np.array([[1.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), "dice")[0, 0] - 1 / 3) < 1e-12
    assert abs(cdist(np.array([[1.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), "hamming")[0, 0] - 1 / 3) < 1e-12
    print("jaccard/dice/hamming ok")

    # 5. Edit distance.
    assert levenshtein("kitten", "sitting") == 3.0
    assert levenshtein("", "abc") == 3.0
    assert levenshtein("abc", "abc") == 0.0
    print("levenshtein(kitten, sitting) = 3")

    # 6. DTW is zero for a warped copy; a narrow band can make the endpoint
    #    unreachable, which returns inf rather than a silently wrong number.
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    assert dtw(a, b) == 0.0
    assert np.linalg.norm(a - b[:4]) > 0
    wide = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    assert np.isinf(dtw(a, wide, band=1))
    print("dtw: warped copy has zero distance; band too narrow -> inf")

    # 7. Haversine against a known city pair.
    d = haversine(51.5074, -0.1278, 48.8566, 2.3522)
    assert 340.0 < d < 347.0, d
    assert haversine(0.0, 0.0, 0.0, 0.0) == 0.0
    print(f"london-paris = {float(d):.1f} km")
    print("distances demo passed")


if __name__ == "__main__":
    main()
