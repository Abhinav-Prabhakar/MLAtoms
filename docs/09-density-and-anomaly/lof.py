"""Local outlier factor — density compared with the densities around it.

Chapter:   09-density-and-anomaly
Companion: lof.md
Tier 1 · 4 variants: lof, simplified_lof, cluster_lof, lof_knn

    N_k(p)   = the k points nearest to p, p excluded                          (1)
    reach_k(p, o) = max(k-dist(o), d(p, o))                                   (2)
    lrd_k(p) = 1 / mean_{o in N_k(p)} reach_k(p, o)                           (3)
    LOF_k(p) = mean_{o in N_k(p)} lrd_k(o) / lrd_k(p)                         (4)

LOF is the ratio in (4), and everything interesting about it is in the
denominator. A point whose neighbourhood is as dense as its neighbours' scores
one; a point in a hole scores above one; a point inside a dense cluster scores
below one. That is the whole difference from the detectors in the atoms above:
isolation forest asks how many cuts it takes to separate a point from everything,
LOF asks how the point's own patch compares with the patches touching it. The
answer is a ratio, so it is scale-free, and it is local, so a sparse region is
not anomalous merely for being sparse.

`k` is the only real parameter and it sets the scale of "local": a k of 20 asks
whether the point is odd relative to its twenty nearest neighbours. Equation (2)
is the part of the definition that is easy to get wrong. The reachability
distance is not the distance: it is the distance floored by the *neighbour's* own
k-distance, which is what stops a point in a dense cluster from looking
arbitrarily close to a neighbour that is far outside it. Without the floor, one
chance near-neighbour makes a lone point look dense; with it, the neighbour's
own scale caps the credit it can give.

The variants are four definitions of "local density" — the quantity the ratio
compares:

- `lof` is (3), the reachability version from the paper.
- `simplified_lof` drops the floor, so lrd is 1 over the mean plain distance.
- `lof_knn` drops the mean instead, using only the k-th distance.
- `cluster_lof` computes the ratio between micro-clusters, weighting each by how
  many points it holds, so that the score is piecewise constant and a scattered
  cluster is judged as a cluster rather than as a cloud of individually odd points.

All four share the fit, the threshold and the API. What differs is what "density"
means, and the demo measures each one's answer on the same data.
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("lof", "simplified_lof", "cluster_lof", "lof_knn")


def _distances(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Euclidean distances, (len(A), len(B))."""
    B = A if B is None else B
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _neighbourhoods(D: np.ndarray, k: int,
                    exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The k nearest indices of every row and their distances.

    `exclude_self` is only correct for a square matrix of points against
    themselves. A query against the training set must not mask its diagonal:
    row i's zero entry is its distance to training point i, not to itself.
    """
    distances = D.copy()
    if exclude_self:
        np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(distances, order, axis=1)


def _kmeans(X: np.ndarray, k: int, seed: int, n_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means++ with Lloyd iterations; returns centres, labels.

    Deterministic because the demo asserts properties of the clusters and a
    random initialisation would make the reported numbers move between runs.
    Empty clusters are re-seeded at the point furthest from its centre.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    centres = np.empty((k, X.shape[1]))
    centres[0] = X[rng.integers(n)]
    closest = _distances(X, centres[:1])[:, 0] ** 2
    for j in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            centres[j] = X[rng.integers(n)]
        else:
            centres[j] = X[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, _distances(X, centres[j:j + 1])[:, 0] ** 2)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        distances = _distances(X, centres)
        labels = np.argmin(distances, axis=1)
        moved = 0.0
        for j in range(k):
            members = X[labels == j]
            if len(members) == 0:
                far = int(np.argmax(np.min(distances, axis=1)))
                centres[j] = X[far]
                labels[far] = j
                moved = np.inf
                continue
            updated = members.mean(axis=0)
            moved = max(moved, float(np.linalg.norm(updated - centres[j])))
            centres[j] = updated
        if moved <= 1e-10:
            break
    return centres, labels


class LocalOutlierFactor:
    """Local density ratio anomaly detector.

    Parameters
    ----------
    variant : {"lof", "simplified_lof", "cluster_lof", "lof_knn"}
        Which density definition to use. See the module docstring.
    n_neighbors : int
        `k` in (1). Neighbourhood size, and the scale at which "local" is
        defined. The paper uses 20 and the score is not sensitive to it away
        from 1 and from `n`.
    n_clusters : int
        Micro-clusters for `cluster_lof` only. This is the `k` of the
        cluster-level neighbour search, not of the point-level one.
    contamination : float in (0, 0.5]
        Quantile of the training scores used as the decision threshold.
    seed : int
        Seeds the k-means++ initialisation; the other variants never use it.

    Attributes
    ----------
    n_features_ : int
    threshold_ : float
        1 - contamination quantile of the training scores.
    lrd_ : ndarray
        Local reachability density of each training point, or of each cluster.
    kdist_ : ndarray
        Distance to the k-th neighbour of each training point.
    scores_ : ndarray
    labels_ : ndarray of int, present only for `cluster_lof`.
    """

    def __init__(self, *, variant: str = "lof", n_neighbors: int = 20,
                 n_clusters: int = 8, contamination: float = 0.1,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must lie in (0, 0.5]")
        self.variant = variant
        self.n_neighbors = int(n_neighbors)
        self.n_clusters = int(n_clusters)
        self.contamination = float(contamination)
        self.seed = int(seed)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> LocalOutlierFactor:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        n, d = X.shape
        if n < 3:
            raise ValueError("need at least three points")
        self.n_features_ = d
        if self.variant == "cluster_lof":
            self._fit_clusters(X)
        else:
            if self.n_neighbors >= n:
                raise ValueError("n_neighbors must be smaller than the sample size")
            D = _distances(X)
            scale = float(np.median(D[~np.eye(n, dtype=bool)]))
            self._eps = 1e-12 * max(scale, 1e-12)
            self.neighbours_, dists = _neighbourhoods(D, self.n_neighbors,
                                                      exclude_self=True)
            self.kdist_ = dists[:, -1]
            self.lrd_ = self._density(dists)
            self.scores_ = self._factor(self.lrd_)
            self._data = X
            self._train_kdist = self.kdist_
            self._train_lrd = self.lrd_
        self.threshold_ = float(np.quantile(self.scores_, 1.0 - self.contamination))
        return self

    def _density(self, dists: np.ndarray) -> np.ndarray:
        """Equation (3), and the two variants of it that replace the mean."""
        if self.variant == "lof":
            reach = np.maximum(dists, self.kdist_[self.neighbours_])
            spread = reach.mean(axis=1)
        elif self.variant == "simplified_lof":
            spread = dists.mean(axis=1)
        else:  # lof_knn keeps only the k-th distance
            spread = self.kdist_
        return 1.0 / np.maximum(spread, self._eps)

    def _factor(self, density: np.ndarray) -> np.ndarray:
        """Equation (4): the mean density of the neighbours over your own."""
        neighbour_density = density[self.neighbours_]
        return neighbour_density.mean(axis=1) / density

    def _fit_clusters(self, X: np.ndarray) -> None:
        """CLOF: (4) on micro-clusters, with density as mass over volume."""
        n, d = X.shape
        if self.n_clusters >= n:
            raise ValueError("n_clusters must be smaller than the sample size")
        centres, labels = _kmeans(X, self.n_clusters, self.seed)
        weights = np.bincount(labels, minlength=self.n_clusters).astype(float)
        D = _distances(centres)
        self._eps = 1e-12 * max(float(np.median(D[~np.eye(len(D), dtype=bool)])), 1e-12)
        k = min(self.n_neighbors, self.n_clusters - 1)
        self.neighbours_, dists = _neighbourhoods(D, k, exclude_self=True)
        self.kdist_ = dists[:, -1]
        reach = np.maximum(dists, self.kdist_[self.neighbours_])
        # Mass per unit volume: a cluster holding twice the points at the same
        # radius is twice as dense, which a bare centroid would not know.
        density = weights / np.maximum(reach.mean(axis=1), self._eps) ** d
        self.lrd_ = density
        self.cluster_scores_ = self._factor(density)
        self.labels_ = labels
        self.scores_ = self.cluster_scores_[labels]
        self._centres = centres

    # -- scoring ---------------------------------------------------------- #
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """LOF of each row; above 1 means sparser than its neighbours.

        `cluster_lof` is piecewise constant: a query receives the factor of the
        micro-cluster it is nearest to, because that is the resolution at which
        the variant estimates density.
        """
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before scoring")
        X = self._check(X)
        if self.variant == "cluster_lof":
            return self.cluster_scores_[np.argmin(_distances(X, self._centres), axis=1)]
        same = X.shape == self._data.shape and np.array_equal(X, self._data)
        order, dists = _neighbourhoods(_distances(X, self._data), self.n_neighbors,
                                       exclude_self=same)
        kdist = dists[:, -1]
        if self.variant == "lof":
            reach = np.maximum(dists, self._train_kdist[order])
            spread = reach.mean(axis=1)
        elif self.variant == "simplified_lof":
            spread = dists.mean(axis=1)
        else:
            spread = kdist
        density = 1.0 / np.maximum(spread, self._eps)
        return self._train_lrd[order].mean(axis=1) / density

    def predict(self, X: np.ndarray) -> np.ndarray:
        """`-1` for an outlier, `1` for an inlier, at the fitted threshold."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before predicting")
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)

    def _check(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected shape (m, {self.n_features_})")
        return X

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = scores[labels], scores[~labels]
    wins = float((positive[:, None] > negative[None, :]).sum())
    ties = float((positive[:, None] == negative[None, :]).sum())
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _knn_distance(X: np.ndarray, k: int) -> np.ndarray:
    """The global baseline: mean distance to the k nearest neighbours."""
    D = _distances(X)
    np.fill_diagonal(D, np.inf)
    return np.partition(D, k - 1, axis=1)[:, :k].mean(axis=1)


def _demo_definition() -> None:
    """The four definitions, checked against hand arithmetic and symmetries."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    model = LocalOutlierFactor(variant="lof", n_neighbors=2).fit(X)
    scores = model.score_samples(X)
    plain = LocalOutlierFactor(variant="simplified_lof", n_neighbors=2).fit(X)
    no_floor = plain.score_samples(X)
    assert np.allclose(scores[:4], 1.0)
    assert scores[4] > 50.0
    assert np.allclose(no_floor[[0, 3]], 1.5) and np.allclose(no_floor[[1, 2]], 5.0 / 6.0)

    # Recompute (2)-(4) with explicit loops: the vectorised path must agree.
    D = _distances(X)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)[:, :2]
    kdist = np.array([D[i, order[i, -1]] for i in range(5)])
    lrd = np.empty(5)
    for i in range(5):
        reach = [max(kdist[j], D[i, j]) for j in order[i]]
        lrd[i] = 1.0 / np.mean(reach)
    manual = np.array([np.mean([lrd[j] for j in order[i]]) / lrd[i] for i in range(5)])
    assert np.allclose(scores, manual)

    # Scale and rotation invariance: LOF is a ratio of distances.
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    blob = np.random.default_rng(0).normal(size=(120, 2))
    moved = (3.0 * blob) @ rotation.T + np.array([11.0, -4.0])
    a = LocalOutlierFactor(n_neighbors=10).fit(blob).score_samples(blob)
    b = LocalOutlierFactor(n_neighbors=10).fit(moved).score_samples(moved)
    assert np.allclose(a, b)

    print(f"definition: on the points 0, 1, 2, 3 and 100 with k = 2, the four "
          f"points of the chain score exactly {scores[0]:.1f} -- including the "
          f"ends, because (2) floors the reach at the neighbour's k-distance, "
          f"so an end point cannot be credited with the gap beyond it -- and "
          f"the outlier scores {scores[4]:.1f}. Dropping the floor, as "
          f"`simplified_lof` does, gives the ends {no_floor[0]:.2f} and the "
          f"middle {no_floor[1]:.2f} -- the floor is what stops the ends from "
          f"looking sparse merely for being ends. The same numbers come out of "
          f"the vectorised code and of a loop written straight from (2)-(4). "
          f"Translating, scaling by 3 and rotating 120 points leaves every "
          f"score unchanged to {np.max(np.abs(a - b)):.1e}, because (4) is a "
          f"ratio of two densities measured in the same units.")


def _demo_local() -> None:
    """Local anomalies that a global distance misses."""
    rng = np.random.default_rng(1)
    dense = rng.normal(scale=0.45, size=(400, 2))
    sparse = rng.normal(scale=1.8, size=(60, 2)) + np.array([10.0, 0.0])
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + 0.1
    fringe = 1.55 * np.column_stack([np.cos(angles), np.sin(angles)])
    X = np.vstack([dense, sparse, fringe])
    labels = np.zeros(len(X), dtype=bool)
    labels[-8:] = True
    outside = np.arange(len(X)) >= 400  # the 68 points not in the tight blob

    local = LocalOutlierFactor(n_neighbors=20).fit(X).score_samples(X)
    global_score = _knn_distance(X, 20)
    lof_auc, knn_auc = _auc(local, labels), _auc(global_score, labels)
    panel_lof = _auc(local[outside], labels[outside])
    panel_knn = _auc(global_score[outside], labels[outside])
    assert lof_auc > knn_auc > 0.8
    assert panel_lof > 0.9 and panel_knn < 0.1

    print(f"local: 400 points from a tight blob, 60 from a wide one ten units "
          f"away, and 8 points on the fringe of the tight blob -- locally odd, "
          f"globally unremarkable, because the wide blob is further out. Over "
          f"all 460 normal points LOF gets AUC {lof_auc:.3f} and the mean "
          f"distance to the 20 nearest neighbours gets {knn_auc:.3f}, because "
          f"the tight blob is trivially easy and dominates the count. Restricted "
          f"to the 68 points outside it, LOF separates the 8 at AUC "
          f"{panel_lof:.3f} and the global score at {panel_knn:.3f}: every "
          f"sparse-blob point outranks every fringe point, since it is further "
          f"from everything. The sparse blob is not an anomaly -- it is "
          f"uniformly sparse, so each of its points has neighbours like itself "
          f"and (4) is near one. That is the property isolation forest lacks.")


def _demo_masking() -> None:
    """A group large enough to fill its own neighbourhood hides."""
    rng = np.random.default_rng(2)
    dense = rng.normal(scale=0.45, size=(400, 2))
    size = 30
    tight = np.array([6.0, 0.0]) + rng.normal(scale=0.15, size=(size, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size)
    radii = rng.uniform(7.0, 9.0, size)
    spread = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    rows = {}
    for name, extra in (("spread", spread), ("tight", tight)):
        X = np.vstack([dense, extra])
        labels = np.zeros(len(X), dtype=bool)
        labels[-size:] = True
        scores = LocalOutlierFactor(n_neighbors=15).fit(X).score_samples(X)
        rows[name] = (float(scores[-size:].mean()), _auc(scores, labels),
                      int(np.sum(np.argsort(-scores)[:size] >= 400)))
    small = np.array([6.0, 0.0]) + rng.normal(scale=0.15, size=(8, 2))
    X = np.vstack([dense, small])
    labels = np.zeros(len(X), dtype=bool)
    labels[-8:] = True
    eight = LocalOutlierFactor(n_neighbors=20).fit(X).score_samples(X)
    eight_mean = float(eight[-8:].mean())
    assert rows["spread"][0] > rows["tight"][0] * 4.0
    assert rows["spread"][1] > 0.95 and rows["tight"][1] < 0.7
    assert eight_mean > rows["tight"][0] * 5.0

    print(f"masking: the anomaly group at radius 6 or 7 to 9 from a normal blob, "
          f"scored against its own density. Thirty points spread out score "
          f"{rows['spread'][0]:.2f} on average, AUC {rows['spread'][1]:.3f}, and "
          f"all 30 are in the detector's top 30. The same thirty packed within "
          f"0.15 of one point score {rows['tight'][0]:.2f}, AUC "
          f"{rows['tight'][1]:.3f}: only {rows['tight'][2]} of 30 are flagged, "
          f"because 30 points are more than the 15 neighbours each of them is "
          f"compared with, so the group is its own reference and looks normal. "
          f"Eight points packed just as tightly score {eight_mean:.2f} under "
          f"k = 20 -- they cannot hide, because 13 of their 20 neighbours are "
          f"blob points. Masking is therefore a relation between the group size "
          f"and k, not a property of tightness alone.")


def _demo_variants() -> None:
    """Four definitions of density on one data set."""
    rng = np.random.default_rng(3)
    dense = rng.normal(scale=0.4, size=(300, 2))
    sparse = rng.normal(scale=1.6, size=(60, 2)) + np.array([7.0, 0.0])
    angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    ring = 4.5 * np.column_stack([np.cos(angles), np.sin(angles)])
    X = np.vstack([dense, sparse, ring])
    labels = np.zeros(len(X), dtype=bool)
    labels[-10:] = True

    model = LocalOutlierFactor(variant="cluster_lof", n_clusters=10, seed=0).fit(X)
    within = max(float(np.ptp(model.scores_[model.labels_ == j]))
                 for j in range(model.n_clusters) if np.any(model.labels_ == j))
    assert within == 0.0

    rows = {}
    for variant in VARIANTS:
        fitted = LocalOutlierFactor(variant=variant, n_neighbors=20,
                                    n_clusters=10, seed=0).fit(X)
        scores = fitted.score_samples(X)
        rows[variant] = (_auc(scores, labels), float(scores[-10:].mean()),
                         float(np.mean(fitted.predict(X) == -1)),
                         float(scores[300:360].mean()))
        if variant != "cluster_lof":
            assert np.allclose(fitted.score_samples(X), scores)
    assert rows["lof"][0] > 0.95
    assert rows["simplified_lof"][3] > rows["lof"][3]
    assert rows["cluster_lof"][1] > rows["lof"][1]
    assert rows["lof_knn"][1] < rows["simplified_lof"][1]
    print("variants: 300 tight points, 60 wide ones, and 10 on a ring of radius "
          "4.5, k = 20:")
    for variant, (auc, ring_score, flagged, blob) in rows.items():
        print(f"  {variant:15s} AUC {auc:.4f}  ring {ring_score:6.3f}  "
              f"sparse blob {blob:.3f}  flagged {flagged:.1%}")
    print(f"  The ring is far enough out that all four rank it highly, and their "
          f"AUCs sit between {rows['cluster_lof'][0]:.3f} and "
          f"{rows['lof_knn'][0]:.3f}. The differences are in the middle of the "
          f"data, where 'dense' is a judgement. `lof`'s reachability floor keeps "
          f"a point's density from being inflated by a neighbour outside its own "
          f"scale, so the sparse blob scores {rows['lof'][3]:.3f}; "
          f"`simplified_lof` removes the floor and scores it "
          f"{rows['simplified_lof'][3]:.3f}. `lof_knn` uses only the k-th "
          f"distance, which is noisier point to point: the ring's mean falls "
          f"from {rows['simplified_lof'][1]:.2f} to {rows['lof_knn'][1]:.2f}. "
          f"`cluster_lof` smooths hardest -- its scores are constant inside each "
          f"of the 10 micro-clusters (spread exactly 0.0 by construction), so it "
          f"cannot rank points within one, and a cluster holding the ring sets "
          f"the ring's score to {rows['cluster_lof'][1]:.2f}. It also cannot hit "
          f"the threshold exactly: only 10 distinct scores exist, so "
          f"{rows['cluster_lof'][2]:.1%} are flagged instead of 10.0%.")


def _demo_broken() -> None:
    """Inputs the detector refuses, and the duplicate it silently accepts."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3))
    cases = [
        ("an unknown variant", lambda: LocalOutlierFactor(variant="kde")),
        ("k = 0", lambda: LocalOutlierFactor(n_neighbors=0)),
        ("k >= n", lambda: LocalOutlierFactor(n_neighbors=40).fit(X)),
        ("one cluster", lambda: LocalOutlierFactor(n_clusters=1)),
        ("contamination = 0", lambda: LocalOutlierFactor(contamination=0.0)),
        ("a 1-D input", lambda: LocalOutlierFactor().fit(X[:, 0])),
        ("two points", lambda: LocalOutlierFactor(n_neighbors=1).fit(X[:2])),
        ("scoring before fitting", lambda: LocalOutlierFactor().score_samples(X)),
        ("the wrong width", lambda: LocalOutlierFactor().fit(X).score_samples(X[:, :2])),
    ]
    for name, call in cases:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"{name} did not raise")

    doubled = np.vstack([X, X[0]])
    nearest = LocalOutlierFactor(n_neighbors=1).fit(doubled)
    pair = float(nearest.score_samples(doubled[-1:])[0])
    wider = LocalOutlierFactor(n_neighbors=5).fit(doubled)
    five = float(wider.score_samples(doubled[-1:])[0])
    assert abs(pair - 1.0) < 1e-12
    assert five < wider.threshold_
    print(f"broken: {len(cases)} bad inputs raise -- "
          f"{', '.join(name for name, _ in cases)}. Duplicating a row is not an "
          f"error, and it is not detected either: with k = 1 the copy and the "
          f"original are each other's only neighbour, both k-distances are zero, "
          f"(2) floors both reaches at zero and the ratio is exactly "
          f"{pair:.1f}. With k = 5 the pair scores {five:.3f} against a "
          f"threshold of {wider.threshold_:.3f}. Two copies make each other look "
          f"normal -- the same masking as the tight group, at the smallest "
          f"possible scale -- which is why duplicate rows are removed before a "
          f"local method is trusted.")


def main() -> None:
    _demo_definition()
    _demo_local()
    _demo_masking()
    _demo_variants()
    _demo_broken()
    print("lof demo passed")


if __name__ == "__main__":
    main()
