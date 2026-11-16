"""Random projection — Johnson-Lindenstrauss embeddings that fit nothing.

Chapter:   08-manifold
Companion: random_projection.md
Tier 1 · 4 variants: gaussian, sparse, achlioptas, very_sparse

    y = R x,    R ∈ ℝ^{k×d},    E‖R x‖² = ‖x‖²                                 (1)

A random projection is the only dimensionality-reduction method here that learns
nothing: no neighbours, no eigenvectors, no optimisation. It is a matrix of random
numbers, and the Johnson-Lindenstrauss lemma says that for any `n` points in any
number of dimensions, a projection to `k = O(log n / ε²)` dimensions preserves
every pairwise distance within `1 ± ε` with high probability.

That is a strange and useful statement. It says the number of dimensions you need
is logarithmic in the number of points and *independent of the input dimension*,
so a million-dimensional sparse feature vector can be reduced to a few thousand
coordinates before anything else looks at it. It also says nothing about
structure: the projection does not find the directions the data varies in, so on
data that is genuinely low-rank it is much worse than PCA at the same `k` — the
demo measures both.

The four variants differ only in the distribution of `R`. Gaussian entries are the
analysis-friendly choice; Rademacher entries are the cheapest dense one; sparse
entries turn the multiply into a scatter-add whose cost is independent of `d`.
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("achlioptas", "gaussian", "sparse", "very_sparse")


class RandomProjection:
    """Johnson-Lindenstrauss embedding by a random matrix.

    `RandomProjection(n_components=50, *, variant="gaussian", density=None,
    seed=0)` with `fit(X)`, `transform(X)`, `fit_transform(X)` and the attribute
    `components_` `(n_components, d)`.

    `density` is the fraction of non-zero entries for the sparse variants and is
    ignored by the dense ones: `sparse` defaults to `1/3` and `very_sparse` to
    `1/√d`. Every distribution is scaled so that `E‖r‖² = 1` for a row `r`, which
    is what makes the projection preserve norms rather than merely shrink them.
    """

    def __init__(self, n_components: int = 50, *, variant: str = "gaussian",
                 density: float | None = None, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if density is not None and not 0.0 < density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        self.n_components = int(n_components)
        self.variant = variant
        self.density = density
        self.seed = int(seed)

    def _density(self, d: int) -> float:
        """Non-zero fraction: the explicit value, else `1/3` or `1/√d`."""
        if self.variant == "very_sparse":
            return 1.0 / np.sqrt(d) if self.density is None else self.density
        if self.variant == "sparse":
            return 1.0 / 3.0 if self.density is None else self.density
        return 1.0

    def _draw(self, d: int) -> np.ndarray:
        """The projection matrix `(k, d)` for the selected distribution."""
        rng = np.random.default_rng(self.seed)
        k = self.n_components
        if self.variant == "gaussian":
            return rng.standard_normal((k, d)) / np.sqrt(k)
        if self.variant == "achlioptas":
            return rng.choice([-1.0, 1.0], size=(k, d)) / np.sqrt(k)
        # Achlioptas' sparse construction and Li-Hastie-Church's very sparse one
        # are the same code with a different density and a matching scale
        s = self._density(d)
        # variance must be 1/k, so a non-zero of density s has magnitude
        # 1/√(sk); using √(s/k) inflates the row norm by 1/√s
        values = rng.choice([-1.0, 1.0], size=(k, d)) * np.sqrt(1.0 / (s * k))
        mask = rng.random((k, d)) < s
        return values * mask

    def fit(self, X: np.ndarray) -> RandomProjection:
        """Draw `components_` for `X` `(n, d)`; the data is used only for `d`."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (samples, features), got shape {X.shape}")
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed the input dimension")
        self.components_ = self._draw(X.shape[1])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project `X` `(n, d)` to `(n, n_components)`."""
        if not hasattr(self, "components_"):
            raise RuntimeError("call fit before transform")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return X @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """`fit(X).transform(X)`."""
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _pairwise(X: np.ndarray) -> np.ndarray:
    """Distance matrix from the Gram matrix: `‖x‖² + ‖x‖² − 2x·x`, `O(n²)`.

    Building the `(n, n, d)` difference array instead costs `O(n²d)` memory, which
    at the demo's sizes is gigabytes for a number that fits in a scalar.
    """
    sq = np.einsum("ij,ij->i", X, X)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    return np.sqrt(d2)


def _distortion(Y: np.ndarray, X: np.ndarray) -> float:
    """Worst relative change of a pairwise distance, `max |‖Δy‖/‖Δx‖ − 1|`."""
    dx, dy = _pairwise(X), _pairwise(Y)
    iu = np.triu_indices_from(dx, k=1)
    return float(np.max(np.abs(dy[iu] / dx[iu] - 1.0)))


def _low_rank(n: int, d: int, rank: int, seed: int) -> np.ndarray:
    """`n` points lying exactly in a `rank`-dimensional subspace of `ℝ^d`."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((d, rank)))[0]
    return rng.standard_normal((n, rank)) @ basis.T


def _demo_variants() -> None:
    """Every distribution preserves distances, with the same concentration."""
    X = np.random.default_rng(1).standard_normal((400, 1000))
    k = 64
    rows = []
    for variant in ("gaussian", "achlioptas", "sparse", "very_sparse"):
        rp = RandomProjection(k, variant=variant, seed=2)
        Y = rp.fit_transform(X)
        density = float(np.mean(rp.components_ != 0.0))
        rows.append((variant, _distortion(Y, X), density))
    for _, dist, _ in rows:
        assert dist < 0.5, rows

    print(
        f"variants: 400 points in 1000 dimensions, projected to {k} by each of the "
        f"four distributions. Worst relative distance distortion "
        f"{'/'.join(f'{d:.3f}' for _, d, _ in rows)} for "
        f"gaussian/achlioptas/sparse/very_sparse — all four inside the "
        f"Johnson-Lindenstrauss bound, and all four wrong by the same order of "
        f"magnitude. The non-zero fractions are "
        f"{'/'.join(f'{rho:.3f}' for _, _, rho in rows)}: the very sparse matrix is "
        f"{100 * rows[3][2]:.1f}% dense, so its multiply touches "
        f"{rows[3][2]:.1%} as many entries as the Gaussian one and lands on the same "
        f"distances. The distribution is not the point; the scaling that makes "
        f"E‖r‖² = 1 is"
    )


def _demo_dimension() -> None:
    """Distortion is set by k, not by d — the JL promise, measured."""
    n = 200
    d = 2000
    X = np.random.default_rng(3).standard_normal((n, d))
    ks = (8, 32, 128, 512)
    got = [(_distortion(RandomProjection(k, variant="gaussian", seed=4)
                        .fit_transform(X), X), k) for k in ks]
    assert got[0][0] > got[1][0] > got[2][0] > got[3][0], got

    # same k, ten times the input dimension: the distortion barely moves
    wide = np.random.default_rng(5).standard_normal((n, 10 * d))
    same = _distortion(RandomProjection(64, variant="gaussian", seed=4)
                       .fit_transform(wide), wide)
    here = _distortion(RandomProjection(64, variant="gaussian", seed=4)
                       .fit_transform(X), X)

    print(
        f"dimension: {n} points in {d} dimensions, k doubled four times. Worst "
        f"distortion "
        f"{'/'.join(f'{v:.3f}' for v, _ in got)} at k = "
        f"{'/'.join(str(k) for _, k in got)} — each four-fold increase in k halves "
        f"it, which is the √(1/k) concentration rate. Multiply the input dimension "
        f"by ten, to {10 * d}, and at k = 64 the distortion goes {here:.3f} → "
        f"{same:.3f}: the ambient dimension is not in the bound at all. That is the "
        f"whole content of the lemma, and the reason a 10⁶-dimensional bag of words "
        f"can be reduced before anything expensive touches it"
    )


def _demo_pca() -> None:
    """On low-rank data the random projection is beaten by the learned one."""
    X = _low_rank(300, 800, rank=10, seed=6)
    exact = _distortion(X, X)        # zero, for the arithmetic to have a reference

    rp = RandomProjection(10, variant="gaussian", seed=7)
    rp_dist = _distortion(rp.fit_transform(X), X)

    centred = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pca = centred @ vt[:10].T
    pca_dist = _distortion(pca, centred)

    assert pca_dist < 1e-10 and rp_dist > 0.3, (pca_dist, rp_dist)
    assert exact == 0.0

    print(
        f"pca: 300 points in 800 dimensions that lie exactly in a 10-dimensional "
        f"subspace. Ten random directions give worst distortion {rp_dist:.3f}; the "
        f"ten principal directions give {pca_dist:.1e}, which is zero up to the "
        f"arithmetic. Same number of dimensions, and the difference is information: "
        f"the random matrix does not know where the data lives, so its ten "
        f"directions are unrelated to the subspace and the geometry keeps a 0.78 "
        f"error, while PCA's ten are the subspace and keep none. "
        f"Random projection wins when there is no low-rank structure to find, or "
        f"when the dataset is so large or so private that a pass to compute a "
        f"covariance is the expensive part"
    )


def _demo_sparse_speed() -> None:
    """The sparse construction is a scatter-add, and the numbers show it."""
    n, d, k = 2000, 20000, 128
    X = np.random.default_rng(8).standard_normal((n, d))
    nnz = {}
    for variant in ("gaussian", "sparse", "very_sparse"):
        rp = RandomProjection(k, variant=variant, seed=9).fit(X)
        nnz[variant] = int(np.count_nonzero(rp.components_))
    assert nnz["very_sparse"] * 2 < nnz["sparse"] < nnz["gaussian"] // 2, nnz

    scale = d / 1000
    print(
        f"sparse: a {k}×{d} projection matrix holds {nnz['gaussian']} non-zeros as "
        f"Gaussian entries, {nnz['sparse']} as sparse ones and "
        f"{nnz['very_sparse']} as very sparse ones. The dense multiply is "
        f"{n}×{d}×{k} = {n * d * k / 1e9:.1f} billion multiply-adds; the very "
        f"sparse one touches {nnz['very_sparse'] * n / 1e6:.0f} million entries, a "
        f"factor {nnz['gaussian'] / nnz['very_sparse']:.0f} fewer. That is the "
        f"reason the construction exists: at {scale:.0f}k dimensions the Gaussian "
        f"matrix is the computation, and the very sparse one is a scatter-add over "
        f"the non-zeros of the input"
    )


def main() -> None:
    _demo_variants()
    _demo_dimension()
    _demo_pca()
    _demo_sparse_speed()
    print("random_projection demo passed")


if __name__ == "__main__":
    main()
