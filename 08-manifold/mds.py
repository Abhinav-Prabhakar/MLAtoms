"""Classical MDS — distances in, coordinates out, five ways to get there.

Chapter:   08-manifold
Companion: mds.md
Tier 2 · 5 variants: classical, metric, nonmetric, weighted, smacof

    σ(X) = Σ_{i<j} w_ij (‖x_i − x_j‖ − δ_ij)²                                (1)

MDS takes dissimilarities and returns coordinates whose distances reproduce
them. What "reproduce" means is the whole design space: minimise (1) directly
(metric MDS), minimise it only up to a monotone transform of the δ (nonmetric
MDS), ignore some of the δ (weighted MDS), or replace the problem with a linear
one that has a closed form (classical MDS).

The closed form is the reason MDS is everywhere: double-centre the squared
distance matrix, take the top eigenvectors, and you have the answer — no
iteration, no seed, no local optima. It minimises a *different* objective
(strain, not stress), and the two objectives agree exactly when the distances
are Euclidean, which is the assumption behind Isomap, kernel PCA and every
"distances between points in a metric space" pipeline.

The demo measures the places that distinction matters: exact recovery in closed
form, the stress that SMACOF removes and classical leaves behind on
non-Euclidean distances, and a monotone distortion of the distances on which
metric MDS reports a good fit and nonmetric MDS finds the true geometry.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
VARIANTS = ("classical", "metric", "nonmetric", "smacof", "weighted")


def _distances(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix, `(n, n)`, symmetric with a zero diagonal."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _check_dissimilarities(D: np.ndarray) -> np.ndarray:
    """Validate a dissimilarity matrix: square, symmetric, zero diagonal, `≥ 0`."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape {D.shape}")
    if not np.allclose(D, D.T, atol=1e-10):
        raise ValueError("D must be symmetric")
    if np.any(np.diag(D) > 1e-10):
        raise ValueError("D must have a zero diagonal")
    if float(D.min()) < 0.0:
        raise ValueError("dissimilarities cannot be negative")
    return D


def _stress(X: np.ndarray, D: np.ndarray, W: np.ndarray) -> float:
    """(1) over the upper triangle, so every pair is counted once."""
    d = _distances(X)
    iu = np.triu_indices_from(D, k=1)
    return float(np.sum(W[iu] * (d[iu] - D[iu]) ** 2))


def _stress1(d: np.ndarray, disp: np.ndarray) -> float:
    """Kruskal's stress-1, `√(Σ(d − d̂)²/Σd²)`: scale-free, unlike the raw stress,
    which a shrinking configuration can drive to zero without being right.
    """
    return float(np.sqrt(np.sum((d - disp) ** 2) / max(float(np.sum(d ** 2)), EPS)))


def _isotonic(y: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit to `y`, by pool-adjacent-violators.

    A violating pair is merged into its mean, which is the exact minimiser of
    `Σ (ŷ − y)²` under monotonicity — no step size, no tolerance.
    """
    means: list[float] = []
    sizes: list[int] = []
    for v in y:
        means.append(float(v))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            v2, s2 = means.pop(), sizes.pop()
            v1, s1 = means.pop(), sizes.pop()
            means.append((v1 * s1 + v2 * s2) / (s1 + s2))
            sizes.append(s1 + s2)
    return np.repeat(np.asarray(means), np.asarray(sizes))


def _guttman_step(X: np.ndarray, target: np.ndarray, W: np.ndarray,
                  n: int) -> np.ndarray:
    """One SMACOF iteration: solve `V X⁺ = B(X) X`, the majorised problem.

    `V` is the weights' Laplacian, `B` the ratios `w_ij δ̂_ij/d_ij`, and the
    minimum-norm solution pins the translation at the origin. With unit weights
    this is the textbook `X⁺ = B X/n`, written as a solve so that zero weights
    can express missing pairs.
    """
    d = _distances(X)
    ratio = W * target / np.maximum(d, EPS)
    np.fill_diagonal(ratio, 0.0)
    B = ratio.copy()
    np.fill_diagonal(B, -B.sum(axis=1))

    V = -W.copy()
    np.fill_diagonal(V, 0.0)
    np.fill_diagonal(V, -V.sum(axis=1))
    return np.linalg.lstsq(V + 1e-12 * np.eye(n), B @ X, rcond=None)[0]


class MDS:
    """Distance-preserving embedding of a dissimilarity matrix.

    `MDS(n_components=2, *, variant="classical", n_iter=300, tol=1e-9, seed=0)`
    with `fit(D, weights=None)`, `fit_transform(...)`, and the attributes
    `embedding_` `(n, n_components)`, `stress_` (normalised by `Σ w δ²`),
    `stresses_` (one per iteration) and `disparities_` (the monotone targets of
    the `nonmetric` variant).

    `weights` is an `(n, n)` non-negative matrix with a zero diagonal; `None`
    means every pair counts once. A zero weight is a missing dissimilarity, and
    `weighted` requires at least one, so that it cannot silently reproduce
    `smacof`.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "classical",
                 n_iter: int = 300, tol: float = 1e-9, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = int(n_components)
        self.variant = variant
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.seed = int(seed)

    # ---------------------------------------------------------- classical ---- #
    def _fit_classical(self, D: np.ndarray) -> np.ndarray:
        """Torgerson scaling: eigenvectors of the double-centred `D²`."""
        n = D.shape[0]
        J = np.eye(n) - np.full((n, n), 1.0 / n)
        B = -0.5 * J @ (D ** 2) @ J
        evals, evecs = np.linalg.eigh(B)
        order = np.argsort(evals)[::-1][: self.n_components]
        keep = np.maximum(evals[order], 0.0)
        return evecs[:, order] * np.sqrt(keep)

    # ------------------------------------------------------------- stress ---- #
    def fit(self, D: np.ndarray, weights: np.ndarray | None = None) -> MDS:
        """Embed `D` `(n, n)` into `n_components` dimensions; returns `self`."""
        D = _check_dissimilarities(D)
        n = D.shape[0]
        if self.n_components > n - 1:
            raise ValueError("n_components must be smaller than n - 1")
        W = np.ones_like(D) if weights is None else np.asarray(weights, dtype=float)
        if W.shape != D.shape:
            raise ValueError(f"weights must have shape {D.shape}, got {W.shape}")
        if np.any(W < 0.0) or not np.allclose(W, W.T):
            raise ValueError("weights must be symmetric and non-negative")
        if self.variant == "weighted" and np.all(W > 0.0):
            raise ValueError("weighted MDS needs a missing pair; pass weights with zeros")
        np.fill_diagonal(W, 0.0)
        denom = float(np.sum(W * D ** 2))
        self.weights_ = W
        self.stresses_: list[float] = []
        self.disparities_ = None

        if self.variant == "classical":
            X = self._fit_classical(D)
            self.embedding_, self.stress_ = X, _stress(X, D, W) / max(denom, EPS)
            self.stresses_ = [self.stress_]
            return self

        X = self._fit_classical(D) if self.variant != "weighted" else self._init_missing(D, W)
        if not np.all(np.isfinite(X)):
            X = np.random.default_rng(self.seed).standard_normal((n, self.n_components))
        X -= X.mean(axis=0, keepdims=True)

        if self.variant == "metric":
            self._fit_metric(X, D, W, denom)
        else:
            self._fit_smacof(X, D, W, denom)
        return self

    def _init_missing(self, D: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Start a partial MDS from the observed entries, with the rest filled once.

        Zero-weighted pairs are given the mean observed dissimilarity before the
        classical solve. The point is not accuracy — the fill is then discarded —
        it is a deterministic, translation-free start in the right region.
        """
        D_filled = D.copy()
        missing = W <= 0.0
        D_filled[missing] = float(np.mean(D[W > 0.0]))
        return self._fit_classical(D_filled)

    def _fit_smacof(self, X: np.ndarray, D: np.ndarray, W: np.ndarray,
                    denom: float) -> None:
        """Majorisation, optionally with the ordinal step that defines nonmetric."""
        target = D.copy()
        n = D.shape[0]
        for _ in range(self.n_iter):
            disp = None
            if self.variant == "nonmetric":
                target, disp = self._ordinal_targets(X, D)
                self.disparities_ = disp
            X = _guttman_step(X, target, W, n)
            if disp is None:
                value = _stress(X, target, W) / max(denom, EPS)
            else:
                iu = np.triu_indices_from(D, k=1)
                value = _stress1(_distances(X)[iu], disp[iu])
            self.stresses_.append(value)
            if (len(self.stresses_) > 1
                    and abs(self.stresses_[-2] - self.stresses_[-1])
                    < self.tol * max(self.stresses_[-2], EPS)):
                break
        self.embedding_ = X
        self.stress_ = self.stresses_[-1] if self.stresses_ else 0.0

    def _ordinal_targets(self, X: np.ndarray,
                         D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Monotone regression of the current distances onto the order of `D`.

        Only the *order* of the dissimilarities is kept, so any strictly
        increasing distortion of `δ` leaves the targets unchanged — which is
        exactly the invariance the nonmetric variant claims.
        """
        iu = np.triu_indices_from(D, k=1)
        order = np.argsort(D[iu], kind="stable")
        d = _distances(X)[iu]
        fitted = np.empty_like(d)
        fitted[order] = _isotonic(d[order])
        disp = np.zeros_like(D)
        disp[iu] = fitted
        disp = disp + disp.T
        target = np.zeros_like(D)
        target[iu] = fitted
        return target + target.T, disp

    def _fit_metric(self, X: np.ndarray, D: np.ndarray, W: np.ndarray,
                    denom: float) -> None:
        """Steepest descent on (1) — Kruskal's original method, with backtracking."""
        step = 1.0 / max(float(np.max(np.sum(W, axis=1))), 1.0)
        sigma = _stress(X, D, W)
        for _ in range(self.n_iter):
            iu = np.triu_indices_from(D, k=1)
            ratio = np.zeros_like(D)
            d = _distances(X)
            ratio[iu] = W[iu] * (d[iu] - D[iu]) / np.maximum(d[iu], EPS)
            ratio = ratio + ratio.T
            grad = 2.0 * (np.diag(ratio.sum(axis=1)) - ratio) @ X
            slope = float(np.sum(grad * grad))
            if slope < EPS:
                break
            trial, scale = X, step
            while scale > 1e-16:
                trial = X - scale * grad
                candidate = _stress(trial, D, W)
                if candidate <= sigma - 1e-4 * scale * slope:
                    break
                scale *= 0.5
            X, sigma, step = trial, candidate, min(scale * 2.0, 1.0)
            self.stresses_.append(sigma / max(denom, EPS))
            if len(self.stresses_) > 1 and 1e-4 * abs(
                    self.stresses_[-2] - self.stresses_[-1]) < self.tol * max(sigma / denom, EPS):
                break
        self.embedding_, self.stress_ = X, sigma / max(denom, EPS)

    def fit_transform(self, D: np.ndarray,
                      weights: np.ndarray | None = None) -> np.ndarray:
        """`fit(D).embedding_`, the `sklearn`-style convenience."""
        return self.fit(D, weights).embedding_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _cloud(n: int, dim: int, seed: int) -> np.ndarray:
    """A random point cloud with a non-degenerate covariance."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    return X - X.mean(axis=0) + 0.5 * rng.standard_normal(dim)


def _gram_error(X: np.ndarray, truth: np.ndarray) -> float:
    """Relative error of the Gram matrix — the rotation- and sign-free comparison."""
    centred = truth - truth.mean(axis=0, keepdims=True)
    P, Q = X @ X.T, centred @ centred.T
    return float(np.linalg.norm(P - Q) / np.linalg.norm(Q))


def _distance_correlation(X: np.ndarray, D: np.ndarray) -> float:
    """Pearson correlation between embedded distances and the targets."""
    iu = np.triu_indices_from(D, k=1)
    d = _distances(X)[iu]
    return float(np.corrcoef(d, D[iu])[0, 1])


def _demo_classical() -> None:
    """Exact recovery from Euclidean distances, in closed form."""
    truth = _cloud(40, 2, seed=1)
    D = _distances(truth)
    fit = MDS(2, variant="classical").fit(D)
    assert _gram_error(fit.embedding_, truth) < 1e-10
    assert fit.stress_ < 1e-20

    # two dimensions in, one out: the embedding is the best rank-1 strain
    one = MDS(1, variant="classical").fit(D)
    assert one.stress_ > 1e-3

    print(
        f"classical: 40 points in the plane, distances in, coordinates out. The "
        f"embedding reproduces the true Gram matrix to {_gram_error(fit.embedding_, truth):.1e} "
        f"with normalised stress {fit.stress_:.1e}, in one eigendecomposition and "
        f"no seed — rotation and reflection are free, so the Gram matrix is the "
        f"strongest statement available, and it is exact. Ask for one dimension "
        f"instead and the same data costs stress {one.stress_:.4f}: the second "
        f"dimension is real, and the closed form reports exactly what dropping it "
        f"costs"
    )


def _demo_smacof() -> None:
    """The majorisation is monotone, and it lands on the same answer as descent."""
    rng = np.random.default_rng(5)
    truth = _cloud(50, 3, seed=2)
    D = _distances(truth) * (1.0 + 0.15 * rng.standard_normal((50, 50)))
    D = 0.5 * (np.abs(D) + np.abs(D).T)
    np.fill_diagonal(D, 0.0)

    sma = MDS(3, variant="smacof", n_iter=400, tol=1e-12).fit(D)
    met = MDS(3, variant="metric", n_iter=2000, tol=1e-12).fit(D)
    cls = MDS(3, variant="classical").fit(D)
    rises = int(np.sum(np.diff(sma.stresses_) > 1e-12))
    assert rises == 0, rises
    assert abs(sma.stress_ - met.stress_) < 1e-6, (sma.stress_, met.stress_)
    assert sma.stress_ < 0.75 * cls.stress_, (sma.stress_, cls.stress_)

    print(
        f"smacof: 50 points in three dimensions with 15% multiplicative noise on "
        f"every distance. The majorisation never increases the stress — {rises} "
        f"rises across {len(sma.stresses_)} iterations — and reaches "
        f"{sma.stress_:.6f}, against {cls.stress_:.6f} for the closed form, so the "
        f"iteration is worth "
        f"{100 * (1 - sma.stress_ / cls.stress_):.1f}% of the objective. Kruskal's "
        f"steepest descent reaches the same number to "
        f"{abs(sma.stress_ - met.stress_):.0e} in {len(met.stresses_)} steps, which "
        f"is *fewer*: a backtracking line search is aggressive when it works. That "
        f"is the honest trade — the Guttman transform is monotone for any weights "
        f"and any dissimilarities, so it has no step size to tune and no search to "
        f"fail, while descent is monotone only as long as its search succeeds"
    )


def _neighbour_overlap(X: np.ndarray, truth: np.ndarray, k: int = 8) -> float:
    """Mean fraction of each point's true `k` nearest neighbours kept by `X`."""
    d_true, d_emb = _distances(truth), _distances(X)
    kept = []
    for i in range(truth.shape[0]):
        a = set(np.argsort(d_true[i])[1:k + 1])
        b = set(np.argsort(d_emb[i])[1:k + 1])
        kept.append(len(a & b) / k)
    return float(np.mean(kept))


def _demo_nonmetric() -> None:
    """A monotone distortion: metric fits the distortion, nonmetric fits the order."""
    truth = _cloud(60, 2, seed=4)
    D = _distances(truth)
    scale = float(D.max())
    distorted = scale * (D / scale) ** 3      # strictly increasing, far from linear

    met = MDS(2, variant="metric", n_iter=2000, tol=1e-13).fit(distorted)
    non = MDS(2, variant="nonmetric", n_iter=200, tol=1e-9).fit(distorted)
    r_metric, r_non = (_distance_correlation(f.embedding_, D) for f in (met, non))
    knn_metric = _neighbour_overlap(met.embedding_, truth)
    knn_non = _neighbour_overlap(non.embedding_, truth)
    assert r_non > 0.999 and r_metric < 0.95, (r_non, r_metric)
    assert knn_non > 0.99 and knn_metric < 0.9, (knn_non, knn_metric)

    iu = np.triu_indices_from(D, k=1)
    order = np.argsort(distorted[iu], kind="stable")
    disp = non.disparities_[iu][order]
    assert np.all(np.diff(disp) >= -1e-9), "disparities must be non-decreasing"

    print(
        f"nonmetric: the same 60 points with every distance cubed, which is "
        f"strictly increasing and nowhere near linear. Metric MDS does exactly what "
        f"it is asked and reproduces the cube: correlation with the true distances "
        f"{r_metric:.4f}, {knn_metric:.4f} of each point's 8 nearest neighbours kept, "
        f"and a stress of {met.stress_:.4f} that looks respectable because it is "
        f"stress against the wrong targets. Nonmetric MDS keeps only the ordering, "
        f"fits monotone disparities by pool-adjacent-violators, and recovers the "
        f"geometry at correlation {r_non:.4f} with {knn_non:.4f} of the neighbourhoods "
        f"intact; after {len(non.stresses_)} sweeps its stress-1 is "
        f"{non.stress_:.1e} and still falling geometrically, because the true "
        f"configuration fits the ordering exactly. Same data, same optimiser, "
        f"different question — which is why 'MDS' without a variant name is "
        f"under-specified"
    )


def _demo_weighted() -> None:
    """Seventy percent of the pairs missing, and the configuration still comes back."""
    truth = _cloud(40, 2, seed=6)
    D = _distances(truth)
    n = D.shape[0]
    rng = np.random.default_rng(7)
    keep = rng.random((n, n)) < 0.3
    W = np.triu(keep, k=1).astype(float)
    W = W + W.T
    # every point needs at least one observed pair, or it is unconstrained
    assert np.all(W.sum(axis=1) > 0)

    fit = MDS(2, variant="weighted", n_iter=600, tol=1e-12).fit(D, weights=W)
    err = _gram_error(fit.embedding_, truth)
    iu = np.triu_indices_from(W, k=1)
    observed = W[iu] > 0
    residual = float(np.mean(np.abs(_distances(fit.embedding_)[iu][observed]
                                    - D[iu][observed]) / D[iu][observed]))
    assert err < 0.01 and residual < 1e-6, (err, residual)

    # the obvious alternative: put a number in the hole and run the closed form
    filled = D.copy()
    filled[W <= 0.0] = float(np.mean(D[W > 0.0]))
    np.fill_diagonal(filled, 0.0)
    fill = MDS(2, variant="classical").fit(filled)
    fill_err = _gram_error(fill.embedding_, truth)
    assert fill_err > 20.0 * err, (fill_err, err)

    print(
        f"weighted: {int(observed.sum())} of the {iu[0].size} pairs observed, the "
        f"rest unknown, every point keeping at least one. Weighted SMACOF, with a "
        f"zero weight meaning 'no opinion', recovers the configuration at Gram "
        f"error {err:.4f} and a mean relative residual of {residual:.1e} on the "
        f"pairs it did see — it fits the observed distances almost exactly, because "
        f"{int(observed.sum())} constraints against {2 * n - 3} coordinates is "
        f"generous. The obvious alternative, filling the holes with the mean "
        f"observed dissimilarity and running the closed form, gives Gram error "
        f"{fill_err:.4f} and stress {fill.stress_:.4f}. The zero weights are not "
        f"small numbers: the missing pairs never enter the objective at all — the "
        f"mean fill only chooses a starting configuration, which the first sweep "
        f"then discards — and that is why partial comparisons can still be embedded"
    )


def main() -> None:
    _demo_classical()
    _demo_smacof()
    _demo_nonmetric()
    _demo_weighted()
    print("mds demo passed")


if __name__ == "__main__":
    main()
