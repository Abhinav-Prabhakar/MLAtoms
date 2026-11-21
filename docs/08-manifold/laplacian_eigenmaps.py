"""Laplacian eigenmaps — the low spectrum of a proximity graph is the map.

Chapter:   08-manifold
Companion: laplacian_eigenmaps.md
Tier 2 · 3 variants: standard, normalized, diffusion_link

    W_ij = exp(−‖x_i − x_j‖² / t)   on the symmetric kNN graph                  (1)
    L = D − W,   D = diag(Σ_j W_ij)                                             (2)
    L f = λ D f,   embedding = (f_2, …, f_{d+1}),  λ_1 = 0 with f_1 = 1         (3)
    L_sym = I − D^{−1/2} W D^{−1/2},   Y = D^{−1/2} v                          (4)
    P = D^{−1} W,   Y_k = ψ_k μ_k^s,   ψ_k = D^{−1/2} v_k,  μ_k = 1 − λ_k       (5)

There are no weights to fit and no global objective to minimise: the graph is
given, the Laplacian is a fixed matrix, and the embedding is a slice of its
spectrum. What makes it an embedding is the Dirichlet form — for any vector f on
the vertices,

    fᵀLf = ½ Σ_ij W_ij (f_i − f_j)²                                            (6)

so a small eigenvalue means the coordinate changes little across every edge, a
smooth function of the graph. Because (3) is a *generalized* problem the ordering
is by λ and the normalisation is fᵀDf = 1, weighting each vertex by its degree:
a vibrating membrane on the graph, whose low modes are the shapes it takes without
stretching and whose first mode is flat.

`standard` uses the combinatorial Laplacian and the generalized problem (3),
`normalized` the heat kernel and the symmetric form (4) — the same subspace, better
conditioned, one scalar scale to choose — and `diffusion_link` the Markov operator
(5), the same eigenvectors with a per-axis decay μ_k^s. Wandering s steps around
the graph multiplies mode k by μ_k^s, which is the seed of diffusion maps.

Unlike LLE there is no gauge freedom in the linear part: the generalized problem
fixes f up to sign and D fixes the weighting, so the layout is comparable across
runs. What it does not fix is *which* manifold: the spectrum is a property of the
graph, and a graph that cuts the sheet is a different graph.
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("standard", "normalized", "diffusion_link")
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def _pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of `X`."""
    sq = np.einsum("ij,ij->i", X, X)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0))


def _knn(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The `k` nearest neighbours of each row, by distance, diagonal excluded."""
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1)
    return (np.take_along_axis(dist, order, axis=1),
            np.take_along_axis(idx, order, axis=1))


def _graph(dist: np.ndarray, idx: np.ndarray, heat: float | None) -> np.ndarray:
    """Symmetric adjacency: ones (`heat is None`) or a heat kernel, unioned."""
    n, k = idx.shape
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    vals = (np.ones(rows.size) if heat is None
            else np.exp(-dist.ravel() ** 2 / heat))
    W[rows, idx.ravel()] = vals
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


# --------------------------------------------------------------------------- #
# The embedding
# --------------------------------------------------------------------------- #
class LaplacianEigenmaps:
    """Spectral embedding of a proximity graph.

    `LaplacianEigenmaps(n_components=2, *, variant="standard", n_neighbors=10,
    t=None, n_steps=1)` with `fit(X)` and `fit_transform(X)`.

    Attributes after `fit`: `embedding_` `(n, n_components)`; `eigenvalues_`,
    the Laplacian eigenvalues sorted ascending, so `eigenvalues_[0]` is the
    trivial zero; `decay_`, the matching `μ = 1 − λ` sorted descending (the
    Markov eigenvalues that `diffusion_link` raises to `n_steps`); `weights_`,
    the symmetric adjacency; `neighbours_`, `distances_` and `degrees_`.

    `t` is the heat-kernel bandwidth in the squared distance; `None` sets it to
    the mean squared distance to the k-th neighbour, which is the usual scale of
    the graph. It is unused by `standard`, whose graph is unweighted.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "standard",
                 n_neighbors: int = 10, t: float | None = None,
                 n_steps: int = 1) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        self.n_components = n_components
        self.variant = variant
        self.n_neighbors = n_neighbors
        self.t = t
        self.n_steps = n_steps

    def fit(self, X: np.ndarray) -> LaplacianEigenmaps:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        n = X.shape[0]
        if self.n_neighbors >= n:
            raise ValueError("n_neighbors must be smaller than the sample size")

        D = _pairwise(X)
        dist, idx = _knn(D, self.n_neighbors)
        heat = None if self.variant == "standard" else self.t
        if heat is None and self.variant != "standard":
            heat = float(np.mean(dist[:, -1] ** 2))
        W = _graph(dist, idx, heat)

        deg = W.sum(axis=1)
        if np.any(deg <= 0.0):
            raise ValueError("every vertex needs at least one neighbour")
        inv = 1.0 / np.sqrt(deg)

        # The symmetric conjugate of the operator. L_sym = I − S and P has the
        # same eigenvectors as S acting on D^{1/2}, so one eigh serves all three
        # variants; only the weights and the post-processing differ.
        S = inv[:, None] * W * inv[None, :]
        mu, vecs = np.linalg.eigh(S)
        order = np.argsort(mu)[::-1]
        mu = mu[order]
        vecs = vecs[:, order]

        keep = slice(1, 1 + self.n_components)
        coords = vecs[:, keep] * inv[:, None]
        if self.variant == "diffusion_link":
            coords = coords * (mu[keep] ** self.n_steps)[None, :]

        self.embedding_ = coords
        self.decay_ = mu
        self.eigenvalues_ = 1.0 - mu
        self.weights_ = W
        self.neighbours_ = idx
        self.distances_ = dist
        self.degrees_ = deg
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).embedding_


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _plane(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A uniform square on a random two-plane through the origin in R³."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((3, 2)))[0]
    coords = rng.random((n, 2)) - 0.5
    return coords @ basis.T, coords


def _scurve(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on a sheet folded twice, and the parameters that generated them."""
    rng = np.random.default_rng(seed)
    t = 3.0 * np.pi * (rng.random(n) - 0.5)
    h = 2.0 * rng.random(n)
    X = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])
    return X, np.column_stack([t, h])


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def _overlap(A: np.ndarray, B: np.ndarray, k: int = 8) -> float:
    """Mean fraction of each row's `k` nearest neighbours in `A` found in `B`."""
    da, db = _pairwise(A), _pairwise(B)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    ia = np.argsort(da, axis=1)[:, :k]
    ib = np.argsort(db, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ia, ib, strict=True)]))


def _distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Correlation of the two pairwise distance matrices."""
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(_pairwise(A)[iu], _pairwise(B)[iu])[0, 1])


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_energy() -> None:
    """The eigenproblem, the Dirichlet form and the trivial mode, exactly."""
    X, coords = _plane(200, seed=0)
    fit = LaplacianEigenmaps(2, n_neighbors=10).fit(X)
    W, deg = fit.weights_, fit.degrees_
    L = np.diag(deg) - W
    Y = fit.embedding_

    lam = fit.eigenvalues_[1:1 + 2]
    R = L @ Y - (deg[:, None] * Y) * lam
    residual = float(np.abs(R).max())
    weight = float(np.abs(Y.T @ (deg[:, None] * Y) - np.eye(2)).max())
    energy = np.array([float(y @ L @ y / (y @ (deg * y))) for y in Y.T])
    constant = float(np.abs(L @ np.ones(X.shape[0])).max())

    assert abs(fit.eigenvalues_[0]) < 1e-10 and constant < 1e-10
    assert residual < 1e-9 and weight < 1e-9
    assert np.allclose(energy, lam, atol=1e-10)
    assert np.all(np.diff(fit.eigenvalues_) >= -1e-12)

    print(
        f"energy: {X.shape[0]} points on a random plane in three dimensions, "
        f"k = 10, the unweighted Laplacian. The constant vector is in the kernel "
        f"to {constant:.0e}, its eigenvalue is {abs(fit.eigenvalues_[0]):.0e} in "
        f"magnitude, so "
        f"the first coordinate is discarded rather than used. The two returned "
        f"coordinates satisfy the generalized problem to {residual:.0e}, are "
        f"D-orthonormal to {weight:.0e}, and their Dirichlet quotients "
        f"fᵀLf / fᵀDf equal their eigenvalues {energy[0]:.6f} and "
        f"{energy[1]:.6f} — the energy identity is not a claim about the "
        f"embedding, it is what the embedding was chosen to minimise"
    )


def _demo_variants() -> None:
    """Three variants on the same sheet: the subspace is shared, the gauge is not."""
    X, coords = _scurve(300, seed=5)
    rows = {}
    for variant in VARIANTS:
        fit = LaplacianEigenmaps(2, variant=variant, n_neighbors=10).fit(X)
        rows[variant] = (fit, _overlap(fit.embedding_, coords),
                         _overlap(fit.embedding_, coords, 40),
                         _distance_correlation(fit.embedding_, coords))

    centred = X - X.mean(axis=0)
    left, spectrum, _ = np.linalg.svd(centred, full_matrices=False)
    pca = left[:, :2] * spectrum[:2]

    base = rows["normalized"][0]
    diffusion = rows["diffusion_link"][0]
    decay = base.decay_[1:1 + 2]
    assert np.allclose(diffusion.embedding_, base.embedding_ * decay, atol=1e-12)
    gain = np.sqrt((diffusion.embedding_ ** 2).mean(0)
                   / (base.embedding_ ** 2).mean(0))

    print(
        "variants: the same 300-point S-curve at k = 10. "
        + "; ".join(
            f"{name} keeps {row[1]:.3f} at eight neighbours, {row[2]:.3f} at "
            f"forty and correlates {row[3]:.3f}"
            for name, row in rows.items())
        + f". standard and normalized differ only through the graph — unweighted "
        f"against a heat kernel — and their second eigenvalues move from "
        f"{rows['standard'][0].eigenvalues_[1]:.4f} to {base.eigenvalues_[1]:.4f}. "
        f"normalized and diffusion_link agree to machine precision up to a "
        f"per-axis factor: the decays are {decay[0]:.4f} and {decay[1]:.4f}, and "
        f"the amplitude ratio is {gain[0]:.4f} and {gain[1]:.4f}. Shrinking that "
        f"factor with n_steps is the move from a Laplacian eigenmap to a diffusion "
        f"map. The leading two principal components keep {_overlap(pca, coords):.3f} "
        f"at eight neighbours and correlate {_distance_correlation(pca, coords):.3f}, "
        f"so on a smooth sheet the graph buys nothing over a rotation"
    )


def _demo_heat() -> None:
    """The bandwidth is the only free parameter: too small kills the graph."""
    X, coords = _scurve(300, seed=5)
    base = LaplacianEigenmaps(2, variant="normalized", n_neighbors=10).fit(X)
    auto = float(np.mean(base.distances_[:, -1] ** 2))
    rows = {}
    for factor in (0.02, 0.2, 1.0, 25.0):
        fit = LaplacianEigenmaps(2, variant="normalized", n_neighbors=10,
                                 t=auto * factor).fit(X)
        rows[factor] = (fit, _distance_correlation(fit.embedding_, coords),
                        _overlap(fit.embedding_, coords), fit.eigenvalues_[1])

    plain = LaplacianEigenmaps(2, n_neighbors=10).fit(X)
    tiny = rows[0.02][0]
    weights = tiny.weights_[tiny.weights_ > 0]
    near_zero = int(np.sum(tiny.eigenvalues_ < 1e-10))
    assert rows[0.02][3] < 1e-6 * rows[1.0][3], rows
    assert rows[1.0][1] > 0.8 and rows[1.0][1] > rows[0.02][1]
    assert abs(rows[25.0][3] - plain.eigenvalues_[1]) < 1e-2
    assert near_zero == 7 and weights.min() < 1e-70

    print(
        "bandwidth: the same sheet, the heat scale swept over four decades from "
        "0.02x to 25x the mean squared distance to the tenth neighbour. Distance "
        "correlation "
        + ", ".join(f"{row[1]:.3f}" for row in rows.values())
        + "; eight-neighbour overlap "
        + ", ".join(f"{row[2]:.3f}" for row in rows.values())
        + "; second eigenvalue "
        + ", ".join(f"{row[3]:.1e}" for row in rows.values())
        + f". At 0.02x the kernel underflows: the weights span {weights.min():.0e} "
        f"to {weights.max():.2f}, {near_zero} eigenvalues sit at 1e-10 or below "
        f"where the graph wants one, and the layout is the indicator of "
        f"{near_zero} near-isolated clumps — which is why overlap still reads "
        f"{rows[0.02][2]:.3f} while distance correlation collapses to "
        f"{rows[0.02][1]:.3f}. Local neighbours survive a clumped layout; global "
        f"structure does not. The other direction is benign: exp(−d²/t) → 1 makes "
        f"the graph unweighted and λ₂ settles at {rows[25.0][3]:.4f}, matching "
        f"standard's {plain.eigenvalues_[1]:.4f}. That limit is the standard "
        f"variant, which is why the two differ so little here"
    )


def _demo_circle() -> None:
    """Closed form: a sampled circle is a cycle graph, and its spectrum is known."""
    n = 200
    theta = 2.0 * np.pi * np.arange(n) / n
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    fit = LaplacianEigenmaps(2, n_neighbors=2).fit(X)

    closed = 1.0 - np.cos(2.0 * np.pi * np.repeat([1, 2, 3], 2) / n)
    error = float(np.abs(fit.eigenvalues_[1:7] - closed).max())
    pairs = float(np.abs(np.diff(fit.eigenvalues_[1:7].reshape(3, 2), axis=1)).max())
    truth = np.column_stack([np.cos(theta), np.sin(theta)])

    assert np.allclose(fit.degrees_, 2.0)
    assert error < 1e-12 and pairs < 1e-12
    assert _distance_correlation(fit.embedding_, truth) > 0.9999

    print(
        f"circle: {n} points uniformly spaced on a circle, k = 2, so the graph is "
        f"a cycle and every degree is exactly 2. Its Laplacian eigenvalues are "
        f"1 − cos(2πj/n), each twice, and the six returned ones match to "
        f"{error:.0e}: {fit.eigenvalues_[1]:.6f}, {fit.eigenvalues_[2]:.6f}, "
        f"{fit.eigenvalues_[3]:.6f}. The spectrum arrives in exactly degenerate "
        f"pairs — the gap inside each pair is {pairs:.0e} — because cos jθ and "
        f"sin jθ have the same energy. The embedding correlates "
        f"{_distance_correlation(fit.embedding_, truth):.4f} with (cos θ, sin θ) "
        f"and keeps all eight neighbours: here the graph, the shape and the answer "
        f"are the same object, which is what 'the spectrum of the manifold' means "
        f"when nothing is approximate"
    )


def _demo_broken() -> None:
    """A graph below the connectivity threshold: the objective is satisfied."""
    rng = np.random.default_rng(0)
    first = 0.3 * rng.standard_normal((100, 2))
    second = 0.3 * rng.standard_normal((100, 2)) + np.array([12.0, 0.0])
    X = np.vstack([first, second])

    wide = LaplacianEigenmaps(2, n_neighbors=8).fit(X)
    narrow = LaplacianEigenmaps(2, n_neighbors=3).fit(X)
    nullity = int(np.sum(np.abs(wide.eigenvalues_) < 1e-10))
    indicator = float(max(wide.embedding_[:100, 0].std(), wide.embedding_[100:, 0].std()))
    patches = int(np.sum(np.abs(narrow.eigenvalues_) < 1e-10))

    assert nullity == 2 and indicator < 1e-9, (nullity, indicator)
    assert patches > 2, patches

    print(
        f"disconnected: two clouds of 100 points twelve units apart, k = 8. Every "
        f"neighbourhood lies inside one cloud, so the Laplacian is block diagonal "
        f"and has {nullity} eigenvalues at zero — one per component. The first "
        f"coordinate is the indicator: one value per cloud, constant within each "
        f"to {indicator:.0e}. At k = 3 the same data splits into {patches} "
        f"pieces, every eigenvalue below the first three is zero to machine "
        f"precision, and the method reports a perfect fit to a graph it cannot "
        f"see across"
    )

    sheet, coords = _scurve(300, seed=5)
    thin = LaplacianEigenmaps(2, variant="normalized", n_neighbors=4).fit(sheet)
    loose = LaplacianEigenmaps(2, variant="normalized", n_neighbors=10).fit(sheet)
    thin_ov = _overlap(thin.embedding_, coords)
    thin_dc = _distance_correlation(thin.embedding_, coords)

    assert thin.eigenvalues_[1] < 1e-12 < loose.eigenvalues_[1]
    assert thin_ov > _overlap(loose.embedding_, coords)
    assert thin_dc > _distance_correlation(loose.embedding_, coords)

    print(
        f"disconnected, silently: the same sheet at k = 4. λ₂ is exactly "
        f"{thin.eigenvalues_[1]:.0e} and both scores improve — {thin_ov:.3f} at "
        f"eight neighbours and {thin_dc:.3f} correlation against "
        f"{_overlap(loose.embedding_, coords):.3f} and "
        f"{_distance_correlation(loose.embedding_, coords):.3f} at k = 10. The "
        f"graph tore along the fold, and the indicator of the two halves is close "
        f"to the sign of the sheet's own first parameter, so the collapse is "
        f"rewarded by the metric meant to catch it. Only the spectrum reports it: "
        f"one zero per component"
    )


def main() -> None:
    _demo_energy()
    _demo_variants()
    _demo_heat()
    _demo_circle()
    _demo_broken()
    print("laplacian_eigenmaps demo passed")


if __name__ == "__main__":
    main()
