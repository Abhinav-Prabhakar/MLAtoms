"""Locally linear embedding — a point is a mixture of its neighbours.

Chapter:   08-manifold
Companion: lle.md
Tier 2 · 4 variants: standard, modified, hessian, supervised

    W = argmin_{Σ_j w_ij = 1} Σ_i ‖x_i − Σ_j w_ij x_j‖²,  j ∈ kNN(i)      (1)
    Y = argmin_{YᵀY = I} Σ_i ‖y_i − Σ_j w_ij y_j‖² = bottom eigenvectors   (2)
        of M = (I − W)ᵀ(I − W)

The method is an inversion. Given a graph, Isomap estimates the distances
between all pairs on the manifold and then embeds those distances. LLE never
estimates a distance at all: it writes each point as a weighted average of its
neighbours, and then asks for the coordinates in which those same averages hold.

That is why it needs no metric. The weights are the coefficients of a local
affine fit, so they are unchanged by rotating, translating or rescaling a
neighbourhood — they describe how the point sits *inside* its neighbourhood, not
how big the neighbourhood is. The embedding inherits the property: it recovers
the manifold up to a rotation, a translation and a scale, and it cannot tell you
how large the manifold is.

The four variants differ in what they do with the second eigenvector problem,
which is where all the difficulty lives. With `k` neighbours and `d` input
dimensions the local Gram matrix is `d × d` and has rank `d`; the weight vector
lives in a `k`-dimensional space, so as soon as `k > d + 1` the weights are not
unique, and *which* solution the solver returns is an arbitrary choice that the
embedding then amplifies. `modified` averages over that whole family, `hessian`
throws the weights away and fits a quadratic instead, and `supervised` moves the
graph rather than the objective.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
VARIANTS = ("standard", "modified", "hessian", "supervised")


# --------------------------------------------------------------------------- #
# Neighbour graph and weights
# --------------------------------------------------------------------------- #
def _pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distances via the Gram identity, `(n, n)`."""
    sq = np.einsum("ij,ij->i", X, X)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))


def _knn(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The `k` nearest neighbours of every row, ascending; `D` needs an inf diagonal."""
    n = D.shape[0]
    k = int(min(k, n - 1))
    idx = np.argpartition(D, k, axis=1)[:, :k]
    dk = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dk, axis=1)
    return (np.take_along_axis(dk, order, axis=1),
            np.take_along_axis(idx, order, axis=1))


def _barycentric(X: np.ndarray, idx: np.ndarray,
                 reg: float) -> np.ndarray:
    """The reconstruction weights of (1), `(n, k)`, rows summing to one.

    Each row solves `G w = 1` for the local Gram matrix `G` of the centred
    neighbourhood, then rescales so that the weights sum to one — which is what
    the constraint in (1) turns the least-squares problem into. The ridge
    `reg · trace(G)` is not cosmetic: `G` is `k × k` with rank at most `d`, so
    for `k > d` it is singular and the solution set is an affine subspace.
    """
    n, k = idx.shape
    W = np.empty((n, k))
    ones = np.ones(k)
    for i in range(n):
        Z = X[idx[i]] - X[i]
        G = Z @ Z.T
        trace = float(np.trace(G))
        G.flat[::k + 1] += reg * (trace if trace > EPS else 1.0)
        w = np.linalg.solve(G, ones)
        W[i] = w / w.sum()
    return W


def _alignment_standard(W: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`M = (I − W)ᵀ(I − W)` with `W` scattered into an `(n, n)` matrix."""
    n = idx.shape[0]
    full = np.zeros((n, n))
    np.put_along_axis(full, idx, W, axis=1)
    A = np.eye(n) - full
    return A.T @ A


# --------------------------------------------------------------------------- #
# Modified LLE — average over the whole family of optimal weights
# --------------------------------------------------------------------------- #
def _alignment_modified(X: np.ndarray, idx: np.ndarray, d: int, reg: float,
                        tol: float) -> np.ndarray:
    """The MLLE alignment matrix, one `k × s_i` block per point.

    With `k > d` the optimal weights are a `(k − d)`-dimensional family. MLLE
    keeps the family instead of a member: it takes the `s_i` eigenvectors of the
    local Gram whose eigenvalues are small relative to the rest, rotates them by
    a Householder reflection so that every one of the resulting weight vectors
    sums to one, and adds `(1 − α_i)` of the regularised single weight vector to
    guarantee that the span contains a genuine barycentre. The embedding then
    penalises every member of the family at once, so no arbitrary choice is
    amplified.

    `s_i` is chosen per point from the spectrum: `η` is the median across points
    of `Σ_{r>d} λ_r / Σ_{r≤d} λ_r`, and `s_i` counts how many of the smallest
    eigenvalues fall below that ratio.
    """
    n, k = idx.shape
    nev = min(X.shape[1], k)
    U = np.empty((n, k, k))
    evals = np.zeros((n, nev))
    for i in range(n):
        left, sv, _ = np.linalg.svd(X[idx[i]] - X[i], full_matrices=True)
        U[i] = left
        evals[i] = sv * sv

    scale = np.maximum(evals.sum(axis=1), EPS) * reg
    tmp = np.einsum("nji,nj->ni", U, np.ones((n, k)))
    tmp[:, :nev] /= evals + scale[:, None]
    tmp[:, nev:] /= scale[:, None]
    w_reg = np.einsum("nij,nj->ni", U, tmp)
    w_reg /= w_reg.sum(axis=1, keepdims=True)

    lead = np.maximum(evals[:, :d].sum(axis=1), EPS)
    rho = evals[:, d:].sum(axis=1) / lead
    eta = float(np.median(rho))
    cum = np.cumsum(evals, axis=1)
    ratio = cum[:, -1:] / np.maximum(cum[:, :-1], EPS) - 1.0
    s = np.array([np.searchsorted(ratio[i, ::-1], eta) for i in range(n)])
    s = np.maximum(s + k - nev, 1)

    M = np.zeros((n, n))
    for i in range(n):
        si = int(s[i])
        Vi = U[i][:, k - si:]
        alpha = float(np.linalg.norm(Vi.sum(axis=0)) / np.sqrt(si))
        h = alpha - Vi.T @ np.ones(k)
        norm = float(np.linalg.norm(h))
        h = h / norm if norm > tol else np.zeros(si)
        Wi = Vi - 2.0 * np.outer(Vi @ h, h) + (1.0 - alpha) * w_reg[i][:, None]
        nb = idx[i]
        M[np.ix_(nb, nb)] += Wi @ Wi.T
        rows = Wi.sum(axis=1)
        M[i, nb] -= rows
        M[nb, i] -= rows
        M[i, i] += si
    return M


# --------------------------------------------------------------------------- #
# Hessian LLE — fit a quadratic, penalise its curvature
# --------------------------------------------------------------------------- #
def _alignment_hessian(X: np.ndarray, idx: np.ndarray, d: int,
                       tol: float) -> np.ndarray:
    """The Hessian alignment matrix, `Σ_i w_i w_iᵀ` over the null directions.

    Per neighbourhood: project onto the tangent space from the SVD, build the
    design matrix of `1`, the `d` tangent coordinates and their `d(d+1)/2`
    products, and orthonormalise it. The last `d(d+1)/2` columns of `Q` span
    the directions orthogonal to that basis, and a function whose local
    quadratic fit is zero lies in their span. So `w_i` — the residual basis
    rescaled to sum to one — measures the curvature of the coordinate at `i`,
    and the embedding minimises the total curvature rather than the
    reconstruction error.

    This is what makes it able to see a manifold that LLE confuses with its own
    tangent space: a linear coordinate has zero Hessian, so the linear functions
    are in the null space and are what the eigen-solver returns.
    """
    n, k = idx.shape
    p = 1 + d + d * (d + 1) // 2
    if k < p:
        raise ValueError(
            f"hessian LLE needs n_neighbors >= {p} for {d} components, got {k}")
    M = np.zeros((n, n))
    for i in range(n):
        G = X[idx[i]] - X[idx[i]].mean(axis=0)
        U = np.linalg.svd(G, full_matrices=False)[0]
        Y = np.empty((k, p))
        Y[:, 0] = 1.0
        Y[:, 1:1 + d] = U[:, :d]
        col = 1 + d
        for a in range(d):
            Y[:, col:col + d - a] = U[:, a:a + 1] * U[:, a:d]
            col += d - a
        Q, _ = np.linalg.qr(Y)
        w = Q[:, d + 1:]
        s = w.sum(axis=0)
        s[np.abs(s) < tol] = 1.0
        w = w / s
        nb = idx[i]
        M[np.ix_(nb, nb)] += w @ w.T
    return M


def _null_space(M: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """The `d` eigenvectors after the constant one, and the full spectrum."""
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)
    return vecs[:, order][:, 1:1 + d], vals[order]


class LLE:
    """Locally linear embedding.

    `LLE(n_components=2, *, variant="standard", n_neighbors=10, reg=1e-3,
    class_weight=1.0, hessian_tol=1e-4, modified_tol=1e-12)` with `fit(X,
    y=None)` and `fit_transform(X, y=None)`.

    Attributes after `fit`: `embedding_` `(n, n_components)`, the orthonormal
    eigenvectors themselves — the method fixes the scale, so the output is a
    direction and not a size; `weights_` `(n, k)`, `neighbours_`, `distances_`,
    `eigenvalues_` (the full spectrum of the alignment matrix) and `objective_`
    `= Σ` of the `d` chosen eigenvalues, which is `tr(YᵀMY)` at the solution.

    `class_weight` is used by `supervised` only: cross-class distances are
    multiplied by `1 + class_weight` before the neighbours are chosen, which
    moves the graph without touching the objective.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "standard",
                 n_neighbors: int = 10, reg: float = 1e-3,
                 class_weight: float = 1.0, hessian_tol: float = 1e-4,
                 modified_tol: float = 1e-12) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if reg < 0.0:
            raise ValueError("reg must be non-negative")
        self.n_components = int(n_components)
        self.variant = variant
        self.n_neighbors = int(n_neighbors)
        self.reg = float(reg)
        self.class_weight = float(class_weight)
        self.hessian_tol = float(hessian_tol)
        self.modified_tol = float(modified_tol)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> LLE:
        """Fit the embedding; returns `self`."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (samples, features), got shape {X.shape}")
        n = X.shape[0]
        if n <= self.n_neighbors:
            raise ValueError("n_neighbors must be smaller than the sample size")
        d = self.n_components

        D = _pairwise(X)
        if self.variant == "supervised":
            if y is None:
                raise ValueError("variant='supervised' needs labels: fit(X, y)")
            labels = np.asarray(y)
            same = labels[:, None] == labels[None, :]
            D = D * np.where(same, 1.0, 1.0 + self.class_weight)
        np.fill_diagonal(D, np.inf)
        dk, idx = _knn(D, self.n_neighbors)

        if self.variant == "hessian":
            M = _alignment_hessian(X, idx, d, self.hessian_tol)
        elif self.variant == "modified":
            M = _alignment_modified(X, idx, d, self.reg, self.modified_tol)
        else:
            M = _alignment_standard(_barycentric(X, idx, self.reg), idx)

        Y, vals = _null_space(M, d)

        self.embedding_ = Y
        self.weights_ = _barycentric(X, idx, self.reg)
        self.neighbours_ = idx
        self.distances_ = dk
        self.eigenvalues_ = vals
        self.objective_ = float(np.sum(vals[1:1 + d]))
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the embedding and return it."""
        return self.fit(X, y).embedding_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _plane(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on a random plane through the origin in 3-D, and their coordinates."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    coords = rng.random((n, 2))
    return coords @ basis[:2, :], coords


def _scurve(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on an S-shaped sheet in 3-D, and their intrinsic coordinates."""
    rng = np.random.default_rng(seed)
    t = 3.0 * np.pi * (rng.random(n) - 0.5)
    h = 2.0 * rng.random(n)
    X = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])
    return X, np.column_stack([t, h])


def _spiral(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved spiral arms in 2-D, and the arm label."""
    rng = np.random.default_rng(seed)
    half = n // 2
    theta = 4.0 * np.pi * rng.random(n)
    arm = np.repeat(np.arange(2), half)
    radius = 0.3 + 0.15 * theta / np.pi
    angle = theta + np.pi * arm
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    return X + 0.02 * rng.standard_normal(X.shape), arm


def _overlap(A: np.ndarray, B: np.ndarray, k: int = 8) -> float:
    """Mean fraction of each point's `B`-nearest neighbours kept by `A`."""
    da, db = _pairwise(A), _pairwise(B)
    kept = []
    for i in range(A.shape[0]):
        left = set(np.argsort(da[i])[1:k + 1])
        right = set(np.argsort(db[i])[1:k + 1])
        kept.append(len(left & right) / k)
    return float(np.mean(kept))


def _per_point_overlap(A: np.ndarray, B: np.ndarray, k: int = 8) -> np.ndarray:
    """The same fraction, one number per point."""
    da, db = _pairwise(A), _pairwise(B)
    return np.array([len(set(np.argsort(da[i])[1:k + 1])
                         & set(np.argsort(db[i])[1:k + 1])) / k
                     for i in range(A.shape[0])])


def _distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Correlation of the two distance matrices — invariant to rigid motions."""
    iu = np.triu_indices(A.shape[0], k=1)
    return float(np.corrcoef(_pairwise(A)[iu], _pairwise(B)[iu])[0, 1])


def _purity(embedding: np.ndarray, labels: np.ndarray, k: int = 8) -> float:
    """Fraction of each point's `k` nearest layout neighbours sharing its label."""
    D = _pairwise(embedding)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    return float(np.mean(labels[idx] == labels[:, None]))


def _best_affine(embedding: np.ndarray,
                 target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The least-squares affine map onto `target`, and its linear part."""
    design = np.column_stack([np.ones(embedding.shape[0]), embedding])
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    return design @ coef, coef[1:]


def _demo_exact() -> None:
    """On an affine patch the algebra is exact — and the weights are untouched."""
    X, coords = _plane(200, seed=0)
    fit = LLE(2, n_neighbors=10).fit(X)

    rowsum = float(np.abs(fit.weights_.sum(axis=1) - 1.0).max())
    assert rowsum < 1e-12, rowsum
    assert abs(fit.objective_ - fit.eigenvalues_[1:3].sum()) < 1e-12
    assert _distance_correlation(fit.embedding_, coords) > 0.99
    assert _overlap(fit.embedding_, coords) > 0.95

    rng = np.random.default_rng(4)
    rotation = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    rotated = LLE(2, n_neighbors=10).fit(X @ rotation)
    moved = LLE(2, n_neighbors=10).fit(7.0 * X + 3.0)
    drift = max(float(np.abs(rotated.weights_ - fit.weights_).max()),
                float(np.abs(moved.weights_ - fit.weights_).max()))
    assert drift < 1e-10, drift
    assert np.array_equal(rotated.neighbours_, fit.neighbours_)

    print(
        f"plane: {X.shape[0]} points on a random plane in three dimensions. The "
        f"200 rows of weights are barycentres to {rowsum:.0e}, the objective is "
        f"the eigenvalue sum to twelve digits, and the embedding correlates "
        f"{_distance_correlation(fit.embedding_, coords):.5f} with the plane's "
        f"own coordinates while keeping {_overlap(fit.embedding_, coords):.3f} "
        f"of each point's eight nearest neighbours. Rotating the input and "
        f"mapping it through 7x + 3 leave every weight unchanged to "
        f"{drift:.0e} and every neighbourhood identical, which is the property "
        f"the method is built on"
    )


def _demo_sheet() -> None:
    """A curved sheet: what survives, and what does not."""
    X, coords = _scurve(300, seed=5)
    fit = LLE(2, n_neighbors=10).fit(X)
    per = _per_point_overlap(fit.embedding_, coords)
    aligned, _ = _best_affine(fit.embedding_, coords)
    centred = X - X.mean(axis=0)
    left, spectrum, _ = np.linalg.svd(centred, full_matrices=False)
    pca = left[:, :2] * spectrum[:2]

    assert _distance_correlation(fit.embedding_, coords) > 0.8
    assert _overlap(fit.embedding_, coords, 40) > _overlap(fit.embedding_, coords)
    assert float(np.median(per)) < 0.7

    print(
        f"S-curve: {X.shape[0]} points on a sheet folded twice in three "
        f"dimensions. LLE keeps {_overlap(fit.embedding_, coords):.3f} of each "
        f"point's eight nearest neighbours — the median point keeps "
        f"{float(np.median(per)):.2f} of eight, and only "
        f"{float((per > 0.8).mean()):.0%} of points keep seven or more — while it "
        f"correlates {_distance_correlation(fit.embedding_, coords):.3f} with "
        f"the true sheet and {_overlap(fit.embedding_, coords, 40):.3f} at forty "
        f"neighbours. The leading two principal components of the same data keep "
        f"{_overlap(pca, coords):.3f} and {_overlap(pca, coords, 40):.3f} and "
        f"correlate {_distance_correlation(pca, coords):.3f} — LLE comes within "
        f"0.04 of a linear projection on all three and beats it on none, which "
        f"is the honest summary of the method on a smooth sheet. One affine map "
        f"onto the truth lifts its own correlation to "
        f"{_distance_correlation(aligned, coords):.3f} and drops its "
        f"eight-neighbour overlap to {_overlap(aligned, coords):.3f}"
    )


def _demo_variants() -> None:
    """What the other three alignment matrices buy, at the same settings."""
    X, coords = _scurve(300, seed=5)
    plain = LLE(2, n_neighbors=10).fit(X)
    modified = LLE(2, variant="modified", n_neighbors=10).fit(X)
    hessian = LLE(2, variant="hessian", n_neighbors=10).fit(X)

    local = _overlap(plain.embedding_, coords)
    assert _overlap(modified.embedding_, coords) > local
    assert _overlap(hessian.embedding_, coords) > local
    assert _distance_correlation(modified.embedding_, coords) < \
        _distance_correlation(plain.embedding_, coords)
    assert _distance_correlation(hessian.embedding_, coords) < \
        _distance_correlation(plain.embedding_, coords)
    try:
        LLE(2, variant="hessian", n_neighbors=5).fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError("hessian needs n_neighbors >= 6 for 2 components")

    print(
        f"modified and hessian: the same sheet with the other two alignment "
        f"matrices. modified keeps {_overlap(modified.embedding_, coords):.3f} "
        f"at eight neighbours against {local:.3f}, and hessian "
        f"{_overlap(hessian.embedding_, coords):.3f}; both pay in the middle and "
        f"the distances, where they score "
        f"{_overlap(modified.embedding_, coords, 40):.3f} and "
        f"{_overlap(hessian.embedding_, coords, 40):.3f} at forty against "
        f"{_overlap(plain.embedding_, coords, 40):.3f}, and correlate "
        f"{_distance_correlation(modified.embedding_, coords):.3f} and "
        f"{_distance_correlation(hessian.embedding_, coords):.3f} against "
        f"{_distance_correlation(plain.embedding_, coords):.3f}. Both are doing "
        f"what they were designed to do — modified stops the solver from "
        f"committing to one arbitrary member of the weight family, hessian stops "
        f"it from caring about weights at all — and both trade the global map "
        f"for local fidelity. hessian refuses k = 5 outright: with two "
        f"components its design matrix already has six columns"
    )


def _demo_supervised() -> None:
    """Inflating cross-class distances moves the graph and nothing else."""
    X, labels = _spiral(300, seed=7)
    plain = LLE(2, n_neighbors=10).fit(X)
    guided = LLE(2, variant="supervised", n_neighbors=10, class_weight=2.0).fit(X, labels)
    pushed = LLE(2, variant="supervised", n_neighbors=10, class_weight=5.0).fit(X, labels)
    stretched = _purity(pushed.embedding_, labels)

    before, after = _purity(plain.embedding_, labels), _purity(guided.embedding_, labels)
    cross_before = float(np.mean(labels[plain.neighbours_] != labels[:, None]))
    cross_after = float(np.mean(labels[guided.neighbours_] != labels[:, None]))
    assert after > before + 0.15, (before, after)
    assert cross_after < 0.1 * cross_before, (cross_before, cross_after)
    try:
        LLE(2, variant="supervised").fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError("supervised without labels should raise")

    print(
        f"supervised: two interleaved spiral arms, {X.shape[0]} points, "
        f"half in each arm. The arms are close in space, so the unweighted graph "
        f"spends {cross_before:.3f} of its edges across them and each point's "
        f"eight layout neighbours match its arm {before:.3f} of the time. "
        f"Multiplying cross-arm distances by 1 + 2 before the neighbour search "
        f"— the objective is untouched — takes the cross-arm edge rate to "
        f"{cross_after:.3f} and purity to {after:.3f}. Pushed to 1 + 5 the purity "
        f"is {stretched:.3f}, which is also what a method that had stopped "
        f"looking at the geometry would score"
    )

    sheet, coords = _scurve(300, seed=5)
    bands = np.digitize(coords[:, 0], (-2.5, 0.0, 2.5))
    flat = LLE(2, n_neighbors=10).fit(sheet)
    forced = LLE(2, variant="supervised", n_neighbors=10, class_weight=3.0).fit(
        sheet, bands)
    plain_purity = _purity(flat.embedding_, bands)
    forced_purity = _purity(forced.embedding_, bands)
    assert forced_purity < plain_purity
    assert _overlap(forced.embedding_, coords) < _overlap(flat.embedding_, coords)

    print(
        f"supervised, wrong prior: the same sheet, labelled by four bands of "
        f"the parameter t, so the labels already run along one axis of the "
        f"answer. At 1 + 3 the purity falls from {plain_purity:.3f} to "
        f"{forced_purity:.3f} and eight-neighbour overlap from "
        f"{_overlap(flat.embedding_, coords):.3f} to "
        f"{_overlap(forced.embedding_, coords):.3f} — a prior that disagrees "
        f"with the manifold is damage, not guidance"
    )


def _demo_ridge() -> None:
    """The ridge is load-bearing: exact weights are the wrong answer."""
    X, coords = _scurve(300, seed=5)
    rows = {}
    for reg in (1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        fit = LLE(2, n_neighbors=10, reg=reg).fit(X)
        rows[reg] = (fit.objective_, _distance_correlation(fit.embedding_, coords),
                     _overlap(fit.embedding_, coords))

    assert rows[1e-9][0] < 1e-9 * rows[1e-3][0]
    assert rows[1e-9][1] < 0.5 < rows[1e-3][1]
    assert abs(rows[1e-1][1] - rows[1e-3][1]) < 0.05
    try:
        LLE(2, n_neighbors=10, reg=0.0).fit(X)
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("an unregularised Gram matrix should be singular")

    print(
        "ridge: the same fit at six ridges. Objective "
        + ", ".join(f"{row[0]:.0e}" for row in rows.values())
        + "; distance correlation "
        + ", ".join(f"{row[1]:.3f}" for row in rows.values())
        + "; eight-neighbour overlap "
        + ", ".join(f"{row[2]:.3f}" for row in rows.values())
        + ", at reg 1e-9 rising to 1e-1. At the smallest ridge the weights "
        "reconstruct every point to machine precision — the objective the method "
        "is written to minimise is essentially zero — and the correlation with "
        f"the truth is {rows[1e-9][1]:.3f}, worse than a linear projection. "
        f"At reg = 1e-3 the objective is {rows[1e-3][0]:.1e} and the correlation "
        f"{rows[1e-3][1]:.3f}. Exactness is not the goal: an exactly reconstructed "
        "neighbourhood can only be reconstructed by weights that cancel, and "
        "cancellation is what the eigen-solver cannot resolve. reg = 0 does not "
        "even get an answer — the Gram matrix is singular and numpy says so"
    )


def _demo_gauge() -> None:
    """What the embedding fixes, and what it leaves to an arbitrary 2x2 matrix."""
    rows = []
    for aspect in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        rng = np.random.default_rng(1)
        rectangle = np.column_stack([aspect * rng.random(300), rng.random(300)])
        fit = LLE(2, n_neighbors=10).fit(rectangle)
        aligned, linear = _best_affine(fit.embedding_, rectangle)
        singular = np.linalg.svd(linear, compute_uv=False)
        rows.append((aspect, _distance_correlation(fit.embedding_, rectangle),
                     _overlap(fit.embedding_, rectangle),
                     _distance_correlation(aligned, rectangle),
                     float(singular[0] / singular[1])))

    assert all(row[3] > 0.99 for row in rows), rows
    assert rows[0][4] < 2.0 < rows[1][4]
    assert rows[-1][4] > 20.0 * rows[3][4]

    print(
        "gauge: uniform rectangles at six aspect ratios, and the same fits "
        "after one least-squares affine map onto the truth. Raw distance "
        "correlation " + ", ".join(f"{row[1]:.4f}" for row in rows) + "; aligned "
        + ", ".join(f"{row[3]:.4f}" for row in rows) + ". Raw eight-neighbour "
        "overlap " + ", ".join(f"{row[2]:.3f}" for row in rows) + ". The linear "
        "part of that map has singular-value ratio "
        + ", ".join(f"{row[4]:.2f}" for row in rows)
        + " — it is the gauge the eigen-solver happened to leave the layout in, "
        "and it is the whole of the raw loss"
    )


def _demo_broken() -> None:
    """Two clouds the graph cannot join: the objective is satisfied exactly."""
    rng = np.random.default_rng(0)
    first = 0.3 * rng.standard_normal((100, 2))
    second = 0.3 * rng.standard_normal((100, 2)) + np.array([12.0, 0.0])
    X = np.vstack([first, second])

    wide = LLE(2, n_neighbors=8).fit(X)
    narrow = LLE(2, n_neighbors=3).fit(X)
    nullity = int(np.sum(np.abs(wide.eigenvalues_) < 1e-10))
    indicator = float(max(wide.embedding_[:100, 0].std(), wide.embedding_[100:, 0].std()))
    collapsed = float(wide.embedding_[:100, 1].std())
    patches = int(np.sum(np.abs(narrow.eigenvalues_) < 1e-10))

    assert nullity == 2 and indicator < 1e-9, (nullity, indicator)
    assert collapsed < 1e-9, collapsed
    assert patches > 2, patches

    print(
        f"disconnected: two clouds of 100 points, twelve units apart, k = 8. "
        f"Every neighbourhood lies inside one cloud, so the alignment matrix has "
        f"{nullity} eigenvalues at zero and the two eigenvectors it returns span "
        f"the two indicator vectors: the first coordinate takes one value per "
        f"cloud and varies within neither to {indicator:.0e}, and the second "
        f"coordinate is constant across the first cloud to {collapsed:.0e}, so "
        f"that cloud collapses to a single point. At k = 3 the same data splits "
        f"into {patches} independently solvable patches and the first coordinate "
        f"is exactly zero for half the points"
    )


def _demo_identity() -> None:
    """Hand the method the answer: it returns the answer up to a linear map."""
    _, coords = _scurve(300, seed=5)
    rows = []
    for k in (10, 15):
        fit = LLE(2, n_neighbors=k).fit(coords)
        aligned, linear = _best_affine(fit.embedding_, coords)
        singular = np.linalg.svd(linear, compute_uv=False)
        rows.append((k, _distance_correlation(fit.embedding_, coords),
                     _distance_correlation(aligned, coords),
                     float(singular[0] / singular[1])))

    assert rows[-1][1] < 0.8 < rows[-1][2], rows
    assert rows[-1][2] > 0.999, rows

    print(
        f"identity: the sheet's own two coordinates as input, so the problem is "
        f"to reproduce a rectangle. At k = 10 LLE's output correlates "
        f"{rows[0][1]:.4f} with the rectangle it was given; after one affine map "
        f"the same output correlates {rows[0][2]:.4f}, and at k = 15, "
        f"{rows[1][2]:.4f}. The map has a singular-value ratio of "
        f"{rows[1][3]:.2f}, so the layout is the rectangle squashed on one axis — "
        f"and the five per cent of distance correlation it lost on the way is "
        f"entirely that squash"
    )


def _roll(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on a swiss roll, and their coordinates in the flat sheet."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    h = 12.0 * rng.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    s = 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))
    return X, np.column_stack([s, h])


def _demo_roll() -> None:
    """A second surface, where the graph has a harder job and k is decisive."""
    X, coords = _roll(250, seed=1)
    sweep = {}
    for k in (6, 8, 10, 12, 15):
        fit = LLE(2, n_neighbors=k).fit(X)
        sweep[k] = (_overlap(fit.embedding_, coords),
                    _distance_correlation(fit.embedding_, coords))
    modified = LLE(2, variant="modified", n_neighbors=10).fit(X)
    hessian = LLE(2, variant="hessian", n_neighbors=10).fit(X)
    bigger = {}
    for n in (500, 1000):
        Xn, cn = _roll(n, seed=1)
        fit = LLE(2, n_neighbors=10).fit(Xn)
        bigger[n] = (_overlap(fit.embedding_, cn),
                     _distance_correlation(fit.embedding_, cn))

    assert sweep[6][0] > 0.6 > sweep[10][0]
    assert _overlap(modified.embedding_, coords) < sweep[10][0]
    assert _overlap(hessian.embedding_, coords) > sweep[10][0]
    assert _distance_correlation(hessian.embedding_, coords) < sweep[10][1]
    assert all(0.4 < row[0] < 0.65 for row in bigger.values())

    print(
        "swiss roll: 250 points, k from 6 to 15. Eight-neighbour overlap "
        + ", ".join(f"{sweep[k][0]:.3f}" for k in sweep)
        + " and distance correlation "
        + ", ".join(f"{sweep[k][1]:.3f}" for k in sweep)
        + " — a sequence of graphs rather than a curve, and the best of them "
        f"({sweep[6][0]:.3f}) is measured at k = 6. At k = 10 the variants "
        f"diverge: modified keeps {_overlap(modified.embedding_, coords):.3f} and "
        f"correlates {_distance_correlation(modified.embedding_, coords):.3f}, "
        f"hessian {_overlap(hessian.embedding_, coords):.3f} and "
        f"{_distance_correlation(hessian.embedding_, coords):.3f}, against "
        f"{sweep[10][0]:.3f} and {sweep[10][1]:.3f} for standard"
    )

    print(
        "swiss roll, more points: at k = 10 the eight-neighbour overlap is "
        + ", ".join(f"{bigger[n][0]:.3f}" for n in bigger)
        + " at 500 and 1000 points against "
        f"{sweep[10][0]:.3f} at 250, while distance correlation goes to "
        + ", ".join(f"{bigger[n][1]:.3f}" for n in bigger)
        + " from "
        f"{sweep[10][1]:.3f} — the global map improves and the local one does not"
    )


def main() -> None:
    _demo_exact()
    _demo_sheet()
    _demo_variants()
    _demo_supervised()
    _demo_ridge()
    _demo_gauge()
    _demo_identity()
    _demo_roll()
    _demo_broken()
    print("lle demo passed")


if __name__ == "__main__":
    main()
