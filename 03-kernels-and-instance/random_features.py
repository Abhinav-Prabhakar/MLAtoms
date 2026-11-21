"""Random features — a kernel replaced by a finite inner product.

Chapter:   03-kernels-and-instance
Companion: random_features.md
Tier 2 · 6 variants: rff, orf, laplacian, polynomial, sparse_jlt, nystrom

    K(x, z) = <phi(x), phi(z)>                      an infinite-dimensional map
    K(x, z) ~= z(x)^T z(z)                          D of them, drawn at random

A kernel method needs one number per pair of points, so it needs the whole
n x n Gram matrix before it can do anything. Random features break that
dependence: build a map z: R^d -> R^D whose inner products approximate the
kernel, then run any linear model on z(x). Training becomes O(nD^2) instead of
O(n^2), the model is a weight vector instead of a stored dataset, and the price
is a variance of order 1/D in every kernel entry.

Bochner's theorem is what makes this possible for a shift-invariant kernel. If
K(x, z) = k(x - z) is positive definite, then k is the Fourier transform of a
probability measure p, so

    k(x - z) = E_{w ~ p}[cos(w^T (x - z))]

and drawing w from p turns the integral into an average. One random frequency
per feature, one random phase, a cosine: that is `rff`. The rest of the file is
the same idea applied honestly. `orf` couples the frequencies so they do not
clump. `laplacian` changes p, because p *is* the kernel and the Gaussian
frequency on a Laplacian kernel estimates a different function. `polynomial`
cannot use Bochner at all, so it sketches the tensor product instead.
`sparse_jlt` is the linear kernel's version, a sparse random projection.
`nystrom` gives up randomness in the features and draws the features from the
data itself.

The demo measures what the theory promises: the average over seeds converges to
the true kernel, the error falls like 1/sqrt(D), orthogonal frequencies cut the
variance at the same D, a mismatched spectral density does not converge at all,
and a kernel ridge regression on 400 random features matches the exact one to a
few thousandths of held-out RMSE.

Numbers to carry away (measured, see the demo):

  - D = 50 -> 800 takes the kernel error from 0.127 to 0.029, halving per
    quadrupling of D, and averaging 400 seeds at D = 200 leaves 0.0063
  - orthogonal frequencies cut the spread from 0.0911 to 0.0849 at D = 100,
    and by 6 to 15 percent across D
  - a Gaussian spectral density on a Laplacian kernel is off by 0.34 no matter
    how many features are drawn; the Cauchy one is off by 0.0023
  - a sparse projection with 200 nonzeros instead of 12800 is unbiased
  - Nystrom with 60 of 80 landmarks leaves a worst entry error of 0.0058
  - kernel ridge: exact 0.0408, random features at D = 512 0.0364, Nystrom
    with every point a landmark 0.0408

"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("rff", "orf", "laplacian", "polynomial", "sparse_jlt", "nystrom")


def _sq_dists(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape ``(len(X), len(Z))``."""
    squared = (np.sum(X ** 2, axis=1)[:, None]
               + np.sum(Z ** 2, axis=1)[None, :]
               - 2.0 * X @ Z.T)
    return np.maximum(squared, 0.0)


def gaussian_kernel(X: np.ndarray, Z: np.ndarray,
                    gamma: float) -> np.ndarray:
    """``exp(-gamma * ||x - z||^2)`` between two point sets."""
    return np.exp(-gamma * _sq_dists(X, Z))


def laplacian_kernel(X: np.ndarray, Z: np.ndarray,
                     bandwidth: float) -> np.ndarray:
    """``exp(-||x - z||_1 / bandwidth)`` between two point sets."""
    gaps = np.abs(X[:, None, :] - Z[None, :, :])
    return np.exp(-gaps.sum(axis=2) / bandwidth)


class RandomFeatures:
    """A finite feature map whose inner products approximate a kernel.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    n_features : int
        Output dimension ``D``. For `nystrom` this is the number of landmarks.
    gamma : float or None
        Gaussian bandwidth; `None` uses ``1 / n_features_of_X``.
    bandwidth : float
        Laplacian scale, used by the `laplacian` variant only.
    degree : int
        Polynomial degree, used by the `polynomial` variant only.
    coef0 : float
        Constant inside the polynomial, added as one extra coordinate.
    sparsity : int
        Non-zeros per column of the sparse projection.
    seed : int
        Seed for every random draw.

    Attributes
    ----------
    omega_ : ndarray
        Frequencies for the Fourier variants, shape ``(D, d)``.
    phase_ : ndarray
        Phases for the Fourier variants, shape ``(D,)``.
    hashes_, signs_ : ndarray
        Tensor-sketch hashes and signs, shape ``(degree, d_aug)``.
    rows_, values_ : ndarray
        Sparse projection indices and entries, shape ``(sparsity, D)``.
    whitener_ : ndarray
        ``K_mm^{-1/2}`` for the Nystrom variant, shape ``(m, m)``.
    """

    def __init__(self, *, variant: str = "rff", n_features: int = 200,
                 gamma: float | None = None, bandwidth: float = 1.0,
                 degree: int = 2, coef0: float = 1.0, sparsity: int = 3,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not isinstance(n_features, (int, np.integer)) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if gamma is not None and (not np.isfinite(gamma) or gamma <= 0.0):
            raise ValueError("gamma must be finite and positive")
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("bandwidth must be finite and positive")
        if not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError("degree must be a positive integer")
        if not np.isfinite(coef0) or coef0 < 0.0:
            raise ValueError("coef0 must be finite and non-negative")
        if not isinstance(sparsity, (int, np.integer)) or sparsity < 1:
            raise ValueError("sparsity must be a positive integer")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.variant = variant
        self.n_features = int(n_features)
        self.gamma = gamma
        self.bandwidth = float(bandwidth)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.sparsity = int(sparsity)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_X(X) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or min(array.shape) < 1:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError("X must be finite")
        return array

    # ------------------------------------------------------------------ #
    # fitting the map
    # ------------------------------------------------------------------ #
    def _draw_fourier(self, width: int, rng: np.random.Generator) -> None:
        """Frequencies from the spectral density of the chosen kernel."""
        if self.variant == "laplacian":
            self.omega_ = rng.standard_cauchy((self.n_features, width))
            self.omega_ /= self.bandwidth
        elif self.variant == "orf":
            self.omega_ = self._orthogonal_frequencies(width, rng)
        else:
            self.omega_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma_),
                                     (self.n_features, width))
        self.phase_ = rng.uniform(0.0, 2.0 * np.pi, self.n_features)

    def _orthogonal_frequencies(self, width: int,
                                rng: np.random.Generator) -> np.ndarray:
        """Rows of a Gaussian matrix made orthogonal, then rescaled."""
        blocks = -(-self.n_features // width)
        rows = []
        for _ in range(blocks):
            draw = rng.normal(size=(width, width))
            orthogonal, _ = np.linalg.qr(draw)
            norms = np.linalg.norm(rng.normal(size=(width, width)), axis=1)
            rows.append(orthogonal.T * norms[:, None])
        stacked = np.concatenate(rows, axis=0)[: self.n_features]
        stacked = stacked[rng.permutation(self.n_features)]
        return np.sqrt(2.0 * self.gamma_) * stacked

    def _draw_polynomial(self, width: int,
                         rng: np.random.Generator) -> None:
        """Two independent count sketches per tensor copy."""
        augmented = width + (1 if self.coef0 > 0.0 else 0)
        self.hashes_ = rng.integers(0, self.n_features,
                                    size=(self.degree, augmented))
        self.signs_ = rng.choice([-1.0, 1.0],
                                 size=(self.degree, augmented))
        self.augment_ = float(np.sqrt(self.coef0))

    def _draw_sparse(self, width: int, rng: np.random.Generator) -> None:
        """``sparsity`` signed entries in every column, scaled to unit norm."""
        keep = min(self.sparsity, width)
        self.rows_ = np.stack([
            rng.permutation(width)[:keep] for _ in range(self.n_features)
        ]).T
        signs = rng.choice([-1.0, 1.0], size=(keep, self.n_features))
        self.values_ = signs * np.sqrt(width / (keep * self.n_features))

    def _fit_nystrom(self, points: np.ndarray,
                     rng: np.random.Generator) -> None:
        """Landmarks, then the inverse square root of their Gram matrix."""
        count = min(self.n_features, points.shape[0])
        chosen = rng.choice(points.shape[0], count, replace=False)
        self.landmarks_ = points[np.sort(chosen)]
        gram = gaussian_kernel(self.landmarks_, self.landmarks_, self.gamma_)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        floor = max(float(eigenvalues.max()), 1e-12) * 1e-10
        eigenvalues = np.maximum(eigenvalues, floor)
        self.whitener_ = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T

    def fit(self, X) -> RandomFeatures:
        """Draw the map. Only `nystrom` reads the data itself."""
        points = self._check_X(X)
        width = points.shape[1]
        self.gamma_ = 1.0 / width if self.gamma is None else float(self.gamma)
        rng = np.random.default_rng(self.seed)
        if self.variant in ("rff", "orf", "laplacian"):
            self._draw_fourier(width, rng)
        elif self.variant == "polynomial":
            self._draw_polynomial(width, rng)
        elif self.variant == "sparse_jlt":
            self._draw_sparse(width, rng)
        else:
            self._fit_nystrom(points, rng)
        self.width_ = width
        return self

    # ------------------------------------------------------------------ #
    # the map
    # ------------------------------------------------------------------ #
    def _fourier(self, points: np.ndarray) -> np.ndarray:
        angles = points @ self.omega_.T
        angles += self.phase_
        return np.sqrt(2.0 / self.n_features) * np.cos(angles)

    def _tensor_sketch(self, points: np.ndarray) -> np.ndarray:
        rows = points.shape[0]
        if self.augment_ > 0.0:
            augmented = np.concatenate(
                [points, np.full((rows, 1), self.augment_)], axis=1)
        else:
            augmented = points
        copies = np.zeros((self.degree, rows, self.n_features))
        slot = np.arange(rows)[:, None]
        for copy in range(self.degree):
            index = np.broadcast_to(self.hashes_[copy],
                                    (rows, augmented.shape[1]))
            np.add.at(copies[copy], (slot, index),
                      augmented * self.signs_[copy])
        spectrum = np.fft.rfft(copies, axis=2)
        product = np.prod(spectrum, axis=0)
        return np.fft.irfft(product, n=self.n_features, axis=1)

    def _sparse(self, points: np.ndarray) -> np.ndarray:
        result = np.zeros((points.shape[0], self.n_features))
        for entry in range(self.values_.shape[0]):
            result += points[:, self.rows_[entry]] * self.values_[entry]
        return result

    def transform(self, X) -> np.ndarray:
        """The feature map, shape ``(len(X), D)``."""
        if not hasattr(self, "width_"):
            raise RuntimeError("call fit before transforming")
        points = self._check_X(X)
        if points.shape[1] != self.width_:
            raise ValueError("X has the wrong number of features")
        if self.variant == "polynomial":
            return self._tensor_sketch(points)
        if self.variant == "sparse_jlt":
            return self._sparse(points)
        if self.variant == "nystrom":
            return gaussian_kernel(points, self.landmarks_, self.gamma_) \
                @ self.whitener_
        return self._fourier(points)

    def fit_transform(self, X) -> np.ndarray:
        """Fit the map and apply it to the same points."""
        return self.fit(X).transform(X)

    def approximately(self, A, B=None) -> np.ndarray:
        """The kernel matrix rebuilt from features."""
        left = self.transform(A)
        right = left if B is None else self.transform(B)
        return left @ right.T

    def n_nonzeros(self) -> int:
        """Non-zeros in the projection, or the feature count for the rest."""
        if not hasattr(self, "width_"):
            raise RuntimeError("call fit before asking")
        if self.variant == "sparse_jlt":
            return int(self.values_.size)
        if self.variant == "polynomial":
            return int(self.degree * self.hashes_.size)
        if self.variant == "nystrom":
            return int(self.whitener_.size + self.landmarks_.size)
        return int(self.omega_.size + self.phase_.size)


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _target(X: np.ndarray) -> np.ndarray:
    """The 2-D function the regression demo fits."""
    return np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.2 * X[:, 0]


def _ridge(features: np.ndarray, targets: np.ndarray,
           penalty: float) -> np.ndarray:
    """Ridge readout on a feature matrix, no intercept."""
    system = features.T @ features + penalty * np.eye(features.shape[1])
    return np.linalg.solve(system, features.T @ targets)


def _demo_convergence() -> None:
    """The average over seeds is the kernel, and the spread is 1/sqrt(D)."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(60, 3))
    probe = rng.normal(size=(5, 3))
    gamma = 0.5
    truth = gaussian_kernel(probe, probe, gamma)
    print("five points in three dimensions, kernel exp(-0.5 ||x-z||^2):")
    print("  D     rmse over the 25 pairs   ratio to the next row")
    errors = []
    for size in (50, 200, 800):
        draws = []
        for seed in range(40):
            model = RandomFeatures(variant="rff", n_features=size,
                                   gamma=gamma, seed=seed).fit(data)
            estimate = model.approximately(probe)
            draws.append(float(np.sqrt(np.mean((estimate - truth) ** 2))))
        errors.append(float(np.mean(draws)))
        ratio = "" if len(errors) == 1 else f"{errors[-2] / errors[-1]:.2f}"
        print(f"  {size:>4}   {errors[-1]:>22.5f}   {ratio:>18}")
        assert np.all(np.isfinite(estimate))
    print("  quadrupling D halves the error: the variance of one kernel entry")
    print("  is O(1/D), so the standard deviation is O(1/sqrt(D))")
    assert errors[0] > errors[1] > errors[2]
    assert 3.0 < errors[0] / errors[2] < 5.0

    average = np.mean([
        RandomFeatures(variant="rff", n_features=200, gamma=gamma,
                       seed=seed).fit(data).approximately(probe)
        for seed in range(400)
    ], axis=0)
    worst = float(np.abs(average - truth).max())
    print(f"  averaged over 400 seeds at D = 200 the worst entry is off by "
          f"{worst:.4f}")
    print(f"  the largest true entry is {truth.max():.4f} (the diagonal), and")
    print("  the estimate's diagonal centres on one with it")
    assert worst < 0.01
    assert np.allclose(np.diag(average), 1.0, atol=0.01)


def _demo_orthogonal() -> None:
    """Coupling the frequencies cuts the variance at the same D."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(40, 10))
    target = gaussian_kernel(data[:8], data[:8], 0.1)
    print("eight points in ten dimensions, D = 100, 120 seeds:")
    print("  frequencies   bias     standard deviation")
    spreads = {}
    for variant in ("rff", "orf"):
        draws = np.stack([
            RandomFeatures(variant=variant, n_features=100, gamma=0.1,
                           seed=seed).fit(data).approximately(data[:8])
            for seed in range(120)
        ])
        bias = float(np.abs(draws.mean(axis=0) - target).mean())
        spreads[variant] = float(draws.std(axis=0).mean())
        print(f"  {variant:<12}  {bias:.4f}   {spreads[variant]:.4f}")
    print("  both are unbiased; the orthogonal draw has the smaller spread")
    print("  because its frequencies cannot pile up in one direction, which")
    print("  is the only failure mode the 1/D variance bound does not see")
    assert spreads["orf"] < 0.95 * spreads["rff"]

    print("how much of the gain survives a different D (40 seeds each):")
    print("  D     rff std   orf std   ratio")
    ratios = []
    for size in (40, 100, 200):
        measured = {}
        for variant in ("rff", "orf"):
            draws = np.stack([
                RandomFeatures(variant=variant, n_features=size, gamma=0.1,
                               seed=seed).fit(data).approximately(data[:8])
                for seed in range(40)
            ])
            measured[variant] = float(draws.std(axis=0).mean())
        ratios.append(measured["rff"] / measured["orf"])
        print(f"  {size:>4}   {measured['rff']:.4f}   {measured['orf']:.4f}   "
              f"{ratios[-1]:.3f}")
        assert measured["orf"] < measured["rff"]


def _demo_spectrum() -> None:
    """The spectral density is the kernel: the wrong one is biased."""
    rng = np.random.default_rng(2)
    data = rng.uniform(-2.0, 2.0, size=(80, 2))
    probe = rng.uniform(-2.0, 2.0, size=(4, 2))
    print("a Laplacian kernel exp(-||x-z||_1), bandwidth 1, D = 800:")
    truth = laplacian_kernel(probe, probe, 1.0)
    print("  frequencies        worst bias over 16 pairs")
    measured = {}
    for variant in ("laplacian", "rff"):
        average = np.mean([
            RandomFeatures(variant=variant, n_features=800, bandwidth=1.0,
                           gamma=0.5, seed=seed).fit(data).approximately(probe)
            for seed in range(160)
        ], axis=0)
        measured[variant] = float(np.abs(average - truth).max())
        print(f"  {variant:<16}   {measured[variant]:>24.4f}")
    print("  the Cauchy density is the Laplacian kernel's Fourier transform,")
    print("  so its cosines average to it; the Gaussian density is a")
    print("  different measure and no number of draws will fix the bias")
    assert measured["laplacian"] < 0.05
    assert measured["rff"] > 4.0 * measured["laplacian"]

    print("the same two densities against a Gaussian kernel, bandwidth 0.5:")
    gauss = gaussian_kernel(probe, probe, 0.5)
    for variant in ("rff", "laplacian"):
        average = np.mean([
            RandomFeatures(variant=variant, n_features=800, bandwidth=1.0,
                           gamma=0.5, seed=seed).fit(data).approximately(probe)
            for seed in range(160)
        ], axis=0)
        worst = float(np.abs(average - gauss).max())
        print(f"  {variant:<16}   {worst:>24.4f}")
        if variant == "rff":
            assert worst < 0.05
        else:
            assert worst > 0.2


def _demo_polynomial() -> None:
    """A kernel with no spectral density is sketched as a tensor."""
    rng = np.random.default_rng(3)
    data = rng.normal(size=(50, 3))
    probe = rng.normal(size=(6, 3))
    print("a polynomial kernel (x.z + 1)^2 in three dimensions:")
    truth = (probe @ probe.T + 1.0) ** 2
    print("  D     bias over 36 pairs   spread over seeds")
    for size in (64, 256):
        draws = np.stack([
            RandomFeatures(variant="polynomial", n_features=size, degree=2,
                           coef0=1.0, seed=seed).fit(data).approximately(probe)
            for seed in range(300)
        ])
        bias = float(np.abs(draws.mean(axis=0) - truth).mean())
        spread = float(draws.std(axis=0).mean())
        print(f"  {size:>4}   {bias:>18.4f}   {spread:>16.4f}")
        assert bias < 0.1
    print("  the estimate is unbiased and very noisy: a tensor sketch is the")
    print("  worst-behaved map in the file, and D has to be large.")

    print("degree 3, D = 256, the same data:")
    truth = (probe @ probe.T + 1.0) ** 3
    draws = np.stack([
        RandomFeatures(variant="polynomial", n_features=256, degree=3,
                       coef0=1.0, seed=seed).fit(data).approximately(probe)
        for seed in range(300)
    ])
    bias = float(np.abs(draws.mean(axis=0) - truth).mean())
    print(f"  bias {bias:.4f}, spread {draws.std(axis=0).mean():.4f}, "
          f"largest true entry {truth.max():.1f}")
    assert bias < 0.3
    print("  the sketch hashes an explicit tensor product, so nothing about")
    print("  it is specific to two dimensions or to this degree")


def _demo_sparse() -> None:
    """A sparse projection for the linear kernel keeps the distances."""
    rng = np.random.default_rng(4)
    data = rng.normal(size=(40, 64))
    probe = rng.normal(size=(6, 64)) / 8.0
    truth = probe @ probe.T
    print("a linear kernel in 64 dimensions, D = 200, 40 seeds:")
    print("  sparsity   nonzeros   bias     spread")
    for sparsity in (1, 3, 8):
        model = RandomFeatures(variant="sparse_jlt", n_features=200,
                               sparsity=sparsity, seed=0).fit(data)
        draws = np.stack([
            RandomFeatures(variant="sparse_jlt", n_features=200,
                           sparsity=sparsity, seed=seed).fit(data)
            .approximately(probe)
            for seed in range(40)
        ])
        bias = float(np.abs(draws.mean(axis=0) - truth).mean())
        spread = float(draws.std(axis=0).mean())
        print(f"  {sparsity:>8}   {model.n_nonzeros():>8}   {bias:.4f}   "
              f"{spread:.4f}")
        assert model.n_nonzeros() == sparsity * 200
        assert bias < 0.2
    print("  the dense projection would need 64 * 200 = 12800 nonzeros; at")
    print("  sparsity 1 this one needs 200 and is still unbiased. The price")
    print("  is variance: a coordinate that is dropped is not estimated.")
    dense = RandomFeatures(variant="sparse_jlt", n_features=200,
                           sparsity=64, seed=0).fit(data)
    print(f"  at sparsity 64 the map is dense and uses {dense.n_nonzeros()}")
    print("  nonzeros, the same as an ordinary projection")
    assert dense.n_nonzeros() == 64 * 200


def _demo_nystrom() -> None:
    """Data-dependent features reproduce the kernel exactly on the landmarks."""
    rng = np.random.default_rng(5)
    data = rng.normal(size=(80, 2))
    gamma = 0.7
    truth = gaussian_kernel(data, data, gamma)
    model = RandomFeatures(variant="nystrom", n_features=80, gamma=gamma,
                           seed=0).fit(data)
    inside = model.approximately(model.landmarks_)
    exact = gaussian_kernel(model.landmarks_, model.landmarks_, gamma)
    print("a Nystrom map with every point a landmark reproduces the kernel:")
    print(f"  worst entry error   {np.abs(inside - exact).max():.1e}")
    print(f"  feature dimension   {model.transform(data).shape[1]}")
    assert np.abs(inside - exact).max() < 1e-8
    print("  with a subset it is a rank-m approximation, so the error is the")
    print("  part of the kernel the landmarks cannot see:")
    print("  landmarks   worst entry error   smallest diagonal")
    errors = []
    for count in (5, 20, 60, 80):
        model = RandomFeatures(variant="nystrom", n_features=count,
                               gamma=gamma, seed=0).fit(data)
        estimate = model.approximately(data)
        errors.append(float(np.abs(estimate - truth).max()))
        diagonal = np.diag(estimate)
        at_landmarks = np.diag(model.approximately(model.landmarks_))
        print(f"  {count:>9}   {errors[-1]:>17.4f}   {diagonal.min():>17.4f}")
        assert np.allclose(at_landmarks, 1.0, atol=1e-9)
        assert diagonal.min() <= 1.0 + 1e-9
    print("  the diagonal is the one entry a projection can never overstate:")
    print("  away from the landmarks the self-similarity is underestimated,")
    print("  so the approximation is a shrinker, not an interpolant")
    assert errors == sorted(errors, reverse=True)


def _demo_regression() -> None:
    """Random features are a kernel method with a different cost."""
    rng = np.random.default_rng(6)
    train = rng.uniform(-2.5, 2.5, size=(120, 2))
    test = rng.uniform(-2.5, 2.5, size=(400, 2))
    targets = _target(train) + rng.normal(0.0, 0.05, 120)
    held = _target(test)
    gamma, penalty = 0.4, 1e-3
    gram = gaussian_kernel(train, train, gamma)
    dual = np.linalg.solve(gram + penalty * np.eye(120), targets)
    exact = gaussian_kernel(test, train, gamma) @ dual
    exact_rmse = float(np.sqrt(np.mean((exact - held) ** 2)))
    print("kernel ridge on 120 points of sin(x1) cos(x2), 400 held out:")
    print(f"  exact dual                                {exact_rmse:.4f}")
    print("  random features, D growing:")
    scores = []
    for size in (64, 128, 256, 512):
        model = RandomFeatures(variant="rff", n_features=size, gamma=gamma,
                               seed=0).fit(train)
        weights = _ridge(model.transform(train), targets, penalty)
        score = float(np.sqrt(np.mean(
            (model.transform(test) @ weights - held) ** 2)))
        scores.append(score)
        print(f"    D = {size:>4}                                  {score:.4f}")
    print("  orthogonal frequencies at the same D:")
    for size in (64, 256):
        model = RandomFeatures(variant="orf", n_features=size, gamma=gamma,
                               seed=0).fit(train)
        weights = _ridge(model.transform(train), targets, penalty)
        score = float(np.sqrt(np.mean(
            (model.transform(test) @ weights - held) ** 2)))
        print(f"    D = {size:>4}                                  {score:.4f}")
        assert score < 1.5 * exact_rmse
    print("  and Nystrom landmarks:")
    for count in (120, 50):
        model = RandomFeatures(variant="nystrom", n_features=count,
                               gamma=gamma, seed=0).fit(train)
        weights = _ridge(model.transform(train), targets, penalty)
        score = float(np.sqrt(np.mean(
            (model.transform(test) @ weights - held) ** 2)))
        print(f"    m = {count:>4}                                  {score:.4f}")
        if count == 120:
            assert abs(score - exact_rmse) < 1e-6
    print("  every column is a linear model on a feature map, so the whole")
    print("  kernel toolbox reduces to ridge regression on 64 to 512 columns.")
    print("  the error climbs towards the exact one as D grows: more features")
    print("  buys fidelity to the kernel, and the low-rank fit at D = 64 is a")
    print("  stronger regulariser that happens to win on this sample")
    assert scores == sorted(scores)
    assert abs(scores[-1] - exact_rmse) < 0.02


def _demo_guards() -> None:
    """Malformed settings and inputs, all refused."""
    X = np.linspace(-1.0, 1.0, 12).reshape(6, 2)
    model = RandomFeatures().fit(X)
    attempts = [
        lambda: RandomFeatures(variant="nope"),
        lambda: RandomFeatures(n_features=0),
        lambda: RandomFeatures(n_features=2.5),
        lambda: RandomFeatures(gamma=0.0),
        lambda: RandomFeatures(gamma=np.nan),
        lambda: RandomFeatures(bandwidth=-1.0),
        lambda: RandomFeatures(degree=0),
        lambda: RandomFeatures(coef0=-1.0),
        lambda: RandomFeatures(sparsity=0),
        lambda: RandomFeatures(seed=-1),
        lambda: RandomFeatures().transform(X),
        lambda: RandomFeatures().n_nonzeros(),
        lambda: model.transform([[1.0, 2.0, 3.0]]),
        lambda: RandomFeatures().fit([[[1.0]]]),
        lambda: RandomFeatures().fit([[np.nan, 1.0]]),
        lambda: RandomFeatures().fit([[]]),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant, a zero or fractional feature count, a zero or non-finite")
    print("  gamma, a negative bandwidth, a zero degree, a negative constant, a")
    print("  zero sparsity, a negative seed, transform or the nonzero count")
    print("  before fit, the wrong feature count, a three-dimensional design, a")
    print("  non-finite input, and an empty input.")
    assert refused == len(attempts)


def main() -> None:
    _demo_convergence()
    _demo_orthogonal()
    _demo_spectrum()
    _demo_polynomial()
    _demo_sparse()
    _demo_nystrom()
    _demo_regression()
    _demo_guards()
    print("random_features demo passed")


if __name__ == "__main__":
    main()
