"""Diffusion maps — the geometry of a random walk on the data.

Chapter:   08-manifold
Companion: diffusion_maps.md
Tier 3 · 3 variants: diffusion_map, diffusion_potential, multiscale

    K_ij = exp(−‖x_i − x_j‖² / ε)                                        (1)
    K̃_ij = K_ij / (q_i^α q_j^α),   q_i = Σ_j K_ij                        (2)
    P = D⁻¹ K̃,   D_i = Σ_j K̃_ij                                          (3)
    P ψ_l = λ_l ψ_l,   1 = λ_0 > λ_1 ≥ λ_2 ≥ …                           (4)
    Ψ_t = ( λ_1^t ψ_1, …, λ_d^t ψ_d )                                    (5)
    D_t(i,j)² = Σ_k (p_t(i,k) − p_t(j,k))² / π_k                         (6)

Every nonlinear method in this chapter has to decide what "close" means. This
one refuses to decide: it lets closeness be the probability that a random walk
gets from one point to another, which is a property of the whole data set rather
than of a pair. Two points are close at scale `t` if a walk of `t` steps starting
at either one is likely to be found at the other.

That is a strange definition until you compute it, and then the miracle is that
it is a Euclidean distance. Equation (6) is the squared distance between two
*rows* of the `t`-step transition matrix, weighted by the stationary
distribution, and because `P` is similar to a symmetric matrix it diagonalises
with an orthonormal basis — so the same distance can be written as a sum over
eigenvectors:

    D_t(i,j)² = Σ_{l ≥ 1} λ_l^{2t} (ψ_l(i) − ψ_l(j))²                   (7)

The embedding in (5) is exactly what you get by truncating that sum at `d` terms:
the coordinate vector Ψ_t(i) has squared Euclidean distance between `i` and `j`
equal to the truncated diffusion distance. Nothing was fitted, no stress was
minimised, and the coordinates are eigenfunctions of a Markov operator — which
is why they are smooth, why they compose across scales, and why the same
decomposition appears in spectral clustering and in the graph Laplacian atom
next door.

Two approximations stand outside that algebra, and both are measured below. The
graph decides what the walk can see: `ε`, `k` and `α` fix which edges exist and
how much weight each carries. The truncation decides how much of the distance
survives: (7) needs `n − 1` terms to be exact, and `d` of them are not the same
distance — on 60 random points at `t = 1`, two components leave 63% relative
error and five leave 43%, because at short times the distance is carried by fast
modes that are exactly the ones thrown away. The walk has to run before the slow
modes dominate.

The variants are three answers to one question: which function of the spectrum
do you keep? `diffusion_map` keeps λ^t — the walk at one scale. `multiscale`
keeps a band of scales, Σ_{s=t}^{t+L−1} λ^s, which is the same walk observed
through a window instead of an instant. `diffusion_potential` keeps the whole
future, Σ_{s≥t} λ^s = λ^t/(1−λ), which is the potential whose gradient is the
diffusion distance and which is finite because the walk mixes. Each is a filter
g(λ) applied to the same eigenvectors, so the three embeddings differ by
coordinate scaling and nothing else — a fact the demo checks to machine
precision on a graph whose spectrum is known in closed form.
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("diffusion_map", "diffusion_potential", "multiscale")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _pairwise(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    B = A if B is None else B
    sa = np.einsum("ij,ij->i", A, A)
    sb = np.einsum("ij,ij->i", B, B)
    return np.sqrt(np.maximum(sa[:, None] + sb[None, :] - 2.0 * (A @ B.T), 0.0))


def _knn_graph(distances: np.ndarray, k: int) -> np.ndarray:
    """Union of the `k`-nearest-neighbour relation, as a boolean matrix."""
    n = distances.shape[0]
    D = distances.copy()
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    mask[np.repeat(np.arange(n), k), idx.ravel()] = True
    return mask | mask.T


def diffusion_map_coordinates(eigenvalues: np.ndarray, vectors: np.ndarray,
                             variant: str, t: int, n_scales: int) -> np.ndarray:
    """Apply the variant's filter g(λ) to the eigenvectors.

    `eigenvalues` and `vectors` are the non-trivial spectrum (`λ_1 … λ_d` and
    the matching right eigenvectors). Returns `(n, d)`.
    """
    lam = np.clip(eigenvalues, 0.0, 1.0 - 1e-12)
    if variant == "diffusion_map":
        gain = lam ** t
    elif variant == "diffusion_potential":
        gain = lam ** t / (1.0 - lam)
    elif variant == "multiscale":
        gain = lam ** t * (1.0 - lam ** n_scales) / (1.0 - lam)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return vectors * gain[None, :]


# --------------------------------------------------------------------------- #
# The embedding
# --------------------------------------------------------------------------- #
class DiffusionMaps:
    """Diffusion coordinates from a Markov chain on a kNN graph.

    `DiffusionMaps(n_components=2, *, variant="diffusion_map", n_neighbors=10,
    alpha=1.0, t=1, n_scales=8, epsilon=None)` with `fit(X)` and
    `fit_transform(X)`; `X` is `(n, n_features)`.

    Attributes after `fit`: `embedding_` `(n, n_components)`; `eigenvalues_`,
    all `n` eigenvalues of the transition matrix, descending; `spectrum_`, the
    first `n_components` after the unit eigenvalue; `vectors_`, the matching
    right eigenvectors; `right_vectors_`, all of them; `stationary_`, the
    stationary distribution `π`; `diffusion_distances_`, the full `(n, n)`
    matrix at the fitted `t`; `kernel_`, `transition_` and `epsilon_`, the
    kernel width actually used.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "diffusion_map",
                 n_neighbors: int = 10, alpha: float = 1.0, t: int = 1,
                 n_scales: int = 8, epsilon: float | None = None) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1 or n_neighbors < 1 or t < 1 or n_scales < 1:
            raise ValueError("n_components, n_neighbors, t and n_scales "
                             "must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.n_components = n_components
        self.variant = variant
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.t = t
        self.n_scales = n_scales
        self.epsilon = epsilon

    def fit(self, X: np.ndarray) -> DiffusionMaps:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        n = X.shape[0]
        if self.n_neighbors >= n:
            raise ValueError("n_neighbors must be smaller than n")
        if self.n_components >= n:
            raise ValueError("n_components must be smaller than n")

        distances = _pairwise(X)
        graph = _knn_graph(distances, self.n_neighbors)
        scale = float(np.mean(np.sort(np.where(graph, distances, np.inf),
                                      axis=1)[:, self.n_neighbors - 1] ** 2))
        epsilon = scale if self.epsilon is None else float(self.epsilon)
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon_ = epsilon

        K = np.where(graph, np.exp(-distances ** 2 / epsilon), 0.0)
        q = K.sum(axis=1)
        if np.any(q <= 0.0):
            raise ValueError("some row of the kernel is zero — epsilon too small "
                             "for the distances it has to cover, or a vertex with "
                             "no neighbours")
        self.kernel_ = K / (q[:, None] ** self.alpha * q[None, :] ** self.alpha)

        degree = self.kernel_.sum(axis=1)
        self.transition_ = self.kernel_ / degree[:, None]
        self.stationary_ = degree / degree.sum()

        symmetric = (self.kernel_ / np.sqrt(degree)[:, None]
                     / np.sqrt(degree)[None, :])
        mu, vecs = np.linalg.eigh(symmetric)
        order = np.argsort(mu)[::-1]
        mu = mu[order]
        vecs = vecs[:, order]
        # Deterministic sign: the largest-magnitude entry of each vector is positive.
        signs = np.sign(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(n)])
        vecs = vecs * np.where(signs == 0.0, 1.0, signs)[None, :]

        self.eigenvalues_ = mu
        # ψ = √(Z/D) v is orthonormal in the stationary inner product
        # ⟨ψ_i, ψ_j⟩_π = δ_ij, which is the normalisation that makes (7) exact.
        normaliser = np.sqrt(degree / degree.sum())
        self.right_vectors_ = vecs / normaliser[:, None]
        # The unit eigenvalue has one eigenvector per connected component — the
        # functions constant on each. A walk never crosses, so they carry no
        # geometry; the embedding starts after them. For a connected graph this
        # drops exactly the constant vector.
        self.multiplicity_ = int(np.sum(mu > 1.0 - 1e-10))
        first = self.multiplicity_
        if first + self.n_components > n:
            raise ValueError("n_components plus the number of connected "
                             "components must not exceed n")
        last = first + self.n_components
        self.spectrum_ = mu[first:last]
        self.vectors_ = self.right_vectors_[:, first:last]
        self.embedding_ = diffusion_map_coordinates(
            self.spectrum_, self.vectors_, self.variant, self.t, self.n_scales)

        self.diffusion_distances_ = self.distance_matrix(self.t)
        self.n_features_ = X.shape[1]
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).embedding_

    def distance_matrix(self, t: int) -> np.ndarray:
        """Equation (6) at time `t`, computed from the spectral expansion (7)."""
        coords = self.right_vectors_ * (self.eigenvalues_ ** t)[None, :]
        sq = np.einsum("ij,ij->i", coords, coords)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T), 0.0)
        return np.sqrt(d2)

    def transition_power(self, t: int) -> np.ndarray:
        """`P^t` by matrix power, for checking the spectral identity."""
        return np.linalg.matrix_power(self.transition_, t)


# --------------------------------------------------------------------------- #
# Data and measurements
# --------------------------------------------------------------------------- #
def _circle(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`n` evenly spaced points on the unit circle, and their angles."""
    theta = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(theta), np.sin(theta)]), theta


def _density_circle(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A circle whose points crowd toward angle zero."""
    rng = np.random.default_rng(seed)
    theta = 2.0 * np.pi * np.sort(rng.beta(1.3, 1.0, n))
    return np.column_stack([np.cos(theta), np.sin(theta)]), theta


def _scurve(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A sheet folded twice, plus the parameters behind it."""
    rng = np.random.default_rng(seed)
    t = 3.0 * np.pi * (rng.random(n) - 0.5)
    h = 2.0 * rng.random(n)
    X = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])
    return X, np.column_stack([t, h])


def _spiral(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two and a half turns of a cone spiral, and the arc parameter."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n)) * 2.5 * 2.0 * np.pi
    return np.column_stack([t * np.cos(t), t * np.sin(t), 0.6 * t]), t


def _overlap(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    da, db = _pairwise(A), _pairwise(B)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    ia = np.argsort(da, axis=1)[:, :k]
    ib = np.argsort(db, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(ia, ib, strict=True)]))


def _distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(_pairwise(A)[iu], _pairwise(B)[iu])[0, 1])


def _pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    C = X - X.mean(axis=0)
    left, spectrum, _ = np.linalg.svd(C, full_matrices=False)
    return left[:, :n_components] * spectrum[:n_components]


def _truncation_error(fitted: DiffusionMaps, t: int, n_components: int) -> float:
    """Relative Frobenius error of the distance matrix cut at `n_components`."""
    index = fitted.multiplicity_ + np.arange(n_components)
    coords = fitted.right_vectors_[:, index] * fitted.eigenvalues_[index] ** t
    sq = np.einsum("ij,ij->i", coords, coords)
    squared = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T), 0.0)
    full = fitted.distance_matrix(t)
    return float(np.linalg.norm(np.sqrt(squared) - full) / np.linalg.norm(full))


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_circle() -> None:
    n = 200
    X, theta = _circle(n)
    target = np.column_stack([np.cos(theta), np.sin(theta)])
    fitted = DiffusionMaps(n_neighbors=2).fit(X)

    first = np.cos(2.0 * np.pi / n)
    second = np.cos(4.0 * np.pi / n)
    lam = fitted.eigenvalues_
    spectral_error = max(abs(lam[1] - first), abs(lam[2] - first),
                         abs(lam[3] - second), abs(lam[4] - second))
    correlation = _distance_correlation(fitted.embedding_, target)
    assert spectral_error < 1e-12, spectral_error
    assert correlation > 1.0 - 1e-9, correlation

    gains, ratios = {}, {}
    for variant in VARIANTS:
        other = DiffusionMaps(n_neighbors=2, variant=variant).fit(X)
        if variant == "diffusion_map":
            gains[variant] = float(lam[1])
        elif variant == "diffusion_potential":
            gains[variant] = float(lam[1] / (1.0 - lam[1]))
        else:
            gains[variant] = float(lam[1] * (1.0 - lam[1] ** 8) / (1.0 - lam[1]))
        column = np.linalg.norm(other.embedding_, axis=0) / np.linalg.norm(
            other.vectors_, axis=0)
        ratios[variant] = float(column[0])
        assert np.allclose(column, gains[variant], rtol=1e-12, atol=0.0)
        assert _distance_correlation(other.embedding_, target) > 1.0 - 1e-9

    worst = max(abs(ratios[v] / gains[v] - 1.0) for v in VARIANTS)
    print(
        f"circle: {n} points on a ring, each joined to its two neighbours. The "
        f"transition matrix is circulant, so its spectrum is known in closed form "
        f"— the eigenvalues are cos(2πm/{n}) with multiplicity two. The fit "
        f"returns {lam[1]:.6f} twice against cos(2π/{n}) = {first:.6f} and "
        f"{lam[3]:.6f} twice against cos(4π/{n}) = {second:.6f}, to "
        f"{spectral_error:.0e}. Keeping the first pair of eigenvectors and "
        f"walking once gives coordinates whose distances correlate "
        f"{correlation:.6f} with the circle's own: not an approximation, because "
        f"on a ring the two slowest modes *are* the circle. Those two eigenvalues "
        f"are equal, so the three filters can only rescale the picture — the gain "
        f"λ₁, λ₁/(1−λ₁) and λ₁(1−λ₁⁸)/(1−λ₁) is {gains['diffusion_map']:.6f}, "
        f"{gains['diffusion_potential']:.2f} and {gains['multiscale']:.4f} for "
        f"the map, potential and multiscale variants, and the column gains read "
        f"back off each fit agree to {worst:.0e}."
    )


def _demo_identity() -> None:
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((60, 4))
    fitted = DiffusionMaps(n_components=3, n_neighbors=6, t=2).fit(Z)

    row_error = float(np.abs(fitted.transition_.sum(axis=1) - 1.0).max())
    invariance = float(np.abs(fitted.stationary_ @ fitted.transition_
                              - fitted.stationary_).max())
    sqrt_pi = np.sqrt(fitted.stationary_)
    similar = fitted.transition_ * sqrt_pi[:, None] / sqrt_pi[None, :]
    symmetry = float(np.abs(similar - similar.T).max())
    gram = fitted.right_vectors_.T @ (fitted.stationary_[:, None]
                                      * fitted.right_vectors_)
    orthonormal = float(np.abs(gram - np.eye(len(Z))).max())
    assert max(row_error, invariance, symmetry, orthonormal) < 1e-12

    worst = 0.0
    for t in (1, 2, 3, 7):
        left = fitted.transition_power(t) / sqrt_pi[None, :]
        brute = _pairwise(left)
        spectral = fitted.distance_matrix(t)
        worst = max(worst, float(np.linalg.norm(brute - spectral)
                                 / np.linalg.norm(brute)))
    assert worst < 1e-6, worst

    closed = [_truncation_error(fitted, 1, 1), _truncation_error(fitted, 1, 2),
              _truncation_error(fitted, 1, 5), _truncation_error(fitted, 2, 5)]
    assert closed[2] < closed[1] < closed[0]
    assert closed[3] < closed[2]

    print(
        f"identity: {len(Z)} random points in four dimensions, k=6, α=1. The "
        f"bookkeeping first — P has row sums one to {row_error:.1e}, the "
        f"stationary distribution is invariant to {invariance:.1e}, "
        f"D⁻¹ᐟ²KD⁻¹ᐟ² is symmetric to {symmetry:.1e}, and the right eigenvectors "
        f"are orthonormal in the π inner product to {orthonormal:.1e}, all four "
        f"asserted. Then the identity itself: the distance between rows i and j "
        f"of Pᵗ/√π, computed by matrix power and a dense distance matrix, against "
        f"the spectral sum (7) over all {len(Z) - 1} modes, for t = 1, 2, 3, 7 — "
        f"worst relative error {worst:.1e}. So (7) is exact. The method is the "
        f"truncation, and truncation is expensive at small t: at t = 1, one mode "
        f"leaves {closed[0]:.0%} relative error in the distance matrix, two leave "
        f"{closed[1]:.0%} and five leave {closed[2]:.0%}; at t = 2 five leave "
        f"{closed[3]:.0%}. A diffusion distance needs time or components before a "
        f"handful of coordinates can carry it."
    )


def _demo_density() -> None:
    X, theta = _density_circle(600, seed=0)
    separation = 0.4
    left, right = [], []
    for i in range(len(theta) - 1):
        j = int(np.searchsorted(theta, theta[i] + separation))
        if j < len(theta):
            left.append(i)
            right.append(j)
    left = np.array(left)
    right = np.array(right)
    dense = 0.5 * (theta[left] + theta[right]) < 0.6 * np.pi
    assert 0.2 < dense.mean() < 0.8

    ratios = {}
    for alpha in (0.0, 0.5, 1.0):
        fitted = DiffusionMaps(n_neighbors=10, alpha=alpha, t=5).fit(X)
        distances = _pairwise(fitted.embedding_)[left, right]
        ratios[alpha] = (float(np.median(distances[dense]))
                         / float(np.median(distances[~dense])))
    assert ratios[1.0] < ratios[0.5] < ratios[0.0]
    assert abs(ratios[1.0] - 1.0) < 0.1

    print(
        f"density: {len(theta)} points on the same circle, sampled with a density "
        f"that rises with the angle. Take {len(left)} pairs exactly "
        f"{separation} radians apart and ask how far apart the embedding puts "
        f"them, on the dense side of the circle against the sparse side. With no "
        f"density correction (α=0) the dense pairs are {ratios[0.0]:.2f}× further "
        f"apart than the sparse pairs at the same arc length; with α=½, "
        f"{ratios[0.5]:.2f}×; with α=1, {ratios[1.0]:.2f}×. At α=1 the diffusion "
        f"distance has stopped caring where the points happen to be: it recovers "
        f"arc length. The kernel normalisation (2) is what buys this — dividing "
        f"K_ij by q_i^α q_j^α makes the stationary distribution uniform at α=1 "
        f"and proportional to the sampling density at α=0."
    )


def _demo_trajectory() -> None:
    X, arc = _spiral(600, seed=1)
    fitted = DiffusionMaps(n_components=1, n_neighbors=10).fit(X)
    psi = fitted.embedding_[:, 0]
    if np.corrcoef(psi, arc)[0, 1] < 0.0:
        psi = -psi
    recovered = abs(_distance_correlation(psi[:, None], arc[:, None]))
    pca = _distance_correlation(_pca(X), arc[:, None])
    assert recovered > 0.95
    assert pca < 0.5

    # How much of each of the first three modes is a function of the arc alone.
    basis = np.column_stack([np.ones_like(arc), arc, arc ** 2])
    explained = []
    for column in (1, 2, 3):
        vector = fitted.right_vectors_[:, column]
        residual = vector - basis @ np.linalg.lstsq(basis, vector, rcond=None)[0]
        explained.append(1.0 - residual.var() / vector.var())
    assert explained[0] > 0.9 and explained[1] > 0.8 and explained[2] < 0.2

    print(
        f"trajectory: {len(arc)} points along two and a half turns of a cone "
        f"spiral in three dimensions — a one-dimensional curve, so one "
        f"coordinate should be enough, and nobody told the method that. One "
        f"diffusion coordinate reproduces the arc parameter with a distance "
        f"correlation of {recovered:.3f}; PCA's two components reach {pca:.3f}, "
        f"because a spiral projects onto a disc and points half a turn apart land "
        f"on top of each other. The spectrum says the same thing: λ₁, λ₂ and λ₃ "
        f"are {fitted.eigenvalues_[1]:.5f}, {fitted.eigenvalues_[2]:.5f} and "
        f"{fitted.eigenvalues_[3]:.5f}, and a quadratic in the arc explains "
        f"{explained[0]:.0%} of the first eigenvector, {explained[1]:.0%} of the "
        f"second and {explained[2]:.0%} of the third. The first two are the first "
        f"two harmonics *along* the curve and decay slowly; the third leaves the "
        f"curve, varies across the tube the graph sees, and decays fastest."
    )


def _demo_filters() -> None:
    X, coords = _scurve(400, seed=3)
    pca = _pca(X)
    pca_overlap = _overlap(pca, coords)
    pca_correlation = _distance_correlation(pca, coords)

    rows = {}
    for variant in VARIANTS:
        fitted = DiffusionMaps(n_neighbors=10, variant=variant, t=1).fit(X)
        column_gain = (np.linalg.norm(fitted.embedding_, axis=0)
                       / np.linalg.norm(fitted.vectors_, axis=0))
        rows[variant] = (column_gain, _overlap(fitted.embedding_, coords),
                         _distance_correlation(fitted.embedding_, coords))
    assert all(row[1] > pca_overlap for row in rows.values())
    assert rows["diffusion_potential"][2] > pca_correlation
    assert rows["diffusion_map"][2] < pca_correlation
    tilt = rows["diffusion_potential"][0][0] / rows["diffusion_potential"][0][1]

    sweep = {}
    for t in (1, 2, 5, 10, 25):
        fitted = DiffusionMaps(n_neighbors=10, t=t).fit(X)
        sweep[t] = _distance_correlation(fitted.embedding_, coords)
    order = [sweep[t] for t in (1, 2, 5, 10, 25)]
    assert all(later > earlier for earlier, later in zip(order, order[1:],
                                                         strict=False))

    print(
        f"filters: {len(X)} points of an S-curve — a folded *sheet*, not a curve — "
        f"and the three variants with two components. Overlap with the sheet's own "
        f"parameters is {rows['diffusion_map'][1]:.3f} for the map, "
        f"{rows['multiscale'][1]:.3f} for the multiscale window and "
        f"{rows['diffusion_potential'][1]:.3f} for the potential, against PCA's "
        f"{pca_overlap:.3f}; distance correlation is "
        f"{rows['diffusion_map'][2]:.3f}, {rows['multiscale'][2]:.3f} and "
        f"{rows['diffusion_potential'][2]:.3f} against PCA's "
        f"{pca_correlation:.3f}. So every filter keeps more neighbours than PCA, "
        f"the potential also beats it on distances, and the plain map is slightly "
        f"behind there — on a folded sheet diffusion geometry does not unfold, and "
        f"the honest reading is that the potential is the variant to use. It wins "
        f"because (1−λ)⁻¹ tilts the two kept modes by {tilt:.1f}×, which is the "
        f"same thing as running the walk longer: the map's distance correlation "
        f"climbs {sweep[1]:.3f} → {sweep[2]:.3f} → {sweep[5]:.3f} → "
        f"{sweep[10]:.3f} → {sweep[25]:.3f} as t goes 1 → 2 → 5 → 10 → 25, with "
        f"the overlap unchanged at {rows['diffusion_map'][1]:.3f} throughout."
    )


def _demo_broken() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cases = [
        ({"variant": "laplacian"}, "variant must be one of"),
        ({"alpha": 1.5}, "alpha must lie in [0, 1]"),
        ({"t": 0}, "must be positive"),
        ({"n_scales": 0}, "must be positive"),
    ]
    for kwargs, message in cases:
        try:
            DiffusionMaps(**kwargs)
        except ValueError as error:
            assert message in str(error), (kwargs, error)
        else:
            raise AssertionError(f"{kwargs} was accepted")

    for kwargs, message in [({"n_neighbors": 4}, "n_neighbors must be smaller"),
                            ({"n_components": 4, "n_neighbors": 1},
                             "n_components must be smaller"),
                            ({"epsilon": -1.0, "n_neighbors": 1},
                             "epsilon must be positive")]:
        try:
            DiffusionMaps(**kwargs).fit(X)
        except ValueError as error:
            assert message in str(error), (kwargs, error)
        else:
            raise AssertionError(f"{kwargs} was accepted")

    # A kernel that underflows to zero on every row: the walk has no edges.
    tight = np.random.default_rng(0).normal(0.0, 1e-3, (40, 3))
    tight = np.vstack([tight, [5.0, 0.0, 0.0]])
    try:
        DiffusionMaps(n_neighbors=10, epsilon=1e-12).fit(tight)
    except ValueError as error:
        assert "epsilon too small" in str(error), error
    else:
        raise AssertionError("an underflowing kernel was accepted")

    # Two clumps joined by nothing: eigenvalue 1 has multiplicity two.
    rng = np.random.default_rng(0)
    near = rng.normal(0.0, 0.01, (5, 2))
    apart = np.vstack([near, near + 100.0])
    fitted = DiffusionMaps(n_neighbors=1).fit(apart)
    unit = int(np.sum(fitted.eigenvalues_ > 1.0 - 1e-10))
    assert unit == 2
    assert fitted.spectrum_[0] < 0.7

    print(
        f"failures: a bad variant, α outside [0, 1], t = 0, n_neighbors = n, "
        f"n_components = n and a negative epsilon all raise instead of returning "
        f"nonsense. A kernel that underflows — 40 points in a ball of radius "
        f"0.001 plus one 5 units away, with ε = 1e-12 — raises too, rather than "
        f"embedding a graph with no edges. The quiet failure is disconnection: "
        f"five points here and five points a hundred units away, with one "
        f"neighbour each. Nothing raises, because every vertex has an edge, but "
        f"the walk has two components and eigenvalue one has multiplicity {unit}. "
        f"Those two eigenvectors are the indicators of the components, and the fit "
        f"skips both — otherwise one coordinate would be no more than \"am I in "
        f"the first clump\" — so the spectrum starts at "
        f"{fitted.spectrum_[0]:.3f}. What is left is still wrong in a deeper way: "
        f"no walk ever crosses, so the distance between the clumps is the distance "
        f"between two distributions that each stay on their own side, and where "
        f"the clumps sit relative to each other is arbitrary. Check λ₂ < 1 before "
        f"believing any picture."
    )


def main() -> None:
    _demo_circle()
    _demo_identity()
    _demo_density()
    _demo_trajectory()
    _demo_filters()
    _demo_broken()
    print("diffusion_maps demo passed")


if __name__ == "__main__":
    main()
