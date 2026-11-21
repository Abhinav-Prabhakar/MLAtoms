"""Nyström approximation — read a kernel from a handful of its rows.

Chapter:   03-kernels-and-instance
Companion: nystrom.md
Tier 3 · 4 variants: standard, ensemble, streaming, with_kernel_ridge

    K  ≈  K_nm K_mm^{-1} K_mn  =  Φ Φᵀ,     Φ = K_nm K_mm^{-1/2}      (1)

A kernel method needs the n × n Gram matrix, and that is the wall: memory
O(n²), time O(n²d), and for kernel ridge a solve that is O(n³). The Nyström
method keeps m rows of the matrix — the columns of m landmarks — and
reconstructs the rest from them. It is the numerical analyst's quadrature rule
applied to a kernel: sample the function, weight the samples by the inverse of
their own Gram matrix, and integrate.

Three things make it more than an approximation trick. First, Φ is a genuine
feature map: any linear model on Φ is a kernel model with the kernel
Φ Φᵀ, so the same m-dimensional matrix serves regression, classification and
clustering. Second, the approximation is exact on the landmarks and never
overstates a self-similarity, so it behaves like a shrinkage estimator rather
than a noisy interpolant. Third, the landmarks are rows of the data, so the
features are interpretable and the cost is one kernel evaluation per pair.

The variants are four ways to spend the m landmarks. `standard` samples them
uniformly. `ensemble` spends E times the memory on independent samples and
averages, which leaves each map's bias alone and divides its seed-to-seed
spread. `streaming` keeps a reservoir sample as the data arrives, so the Gram
matrix is never formed and memory is O(md) rather than O(nd).
`with_kernel_ridge` puts the features to work: a ridge readout on Φ is exactly
kernel ridge with the approximated kernel, by the Woodbury identity, and the
demo checks that equality rather than assuming it.

The demo measures the wall and the escape. On 400 points from a two-dimensional
cloud, the relative Frobenius error falls from 0.31 with 5 landmarks to 0.002
with 160, and kernel ridge test error follows it from 0.086 to 0.0004. At the
same total dimension, one 60-landmark map reconstructs at 0.0071 while four
15-landmark maps averaged reach only 0.0596: the error is set by the landmarks
per map, not by the total. A streaming pass over 600 points in chunks of 37
holds 60 rows and matches the batch error distribution, 0.0059 against 0.0064.
And the primal readout on Φ matches the dual readout on Φ Φᵀ to 7e-14, which
is the identity that makes the variant worth having.

Numbers to carry away (measured, see the demo):

  - 400 points, RBF kernel: Frobenius error 0.3123 at m = 5, 0.0557 at m = 20,
    0.0023 at m = 160, and 5.5e-13 at m = 400
  - kernel ridge test error 0.08633 at m = 5, 0.00399 at m = 20, 0.00044 at
    m = 160, against 0.00047 for the full 400-point solve
  - ensemble at 15 landmarks per map: spread 0.0165 for one map and 0.0031 for
    twelve, at twelve times the feature dimension
  - equal total dimension: one 60-landmark map 0.0071, four 15-landmark maps
    0.0596
  - streaming 60 rows over 600 points: 0.0059 +/- 0.0014 against the batch
    0.0064 +/- 0.0013
  - feature readout against dual readout: 6.7e-14; all landmarks against the
    exact 300 x 300 solve: 3.9e-11

"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("standard", "ensemble", "streaming", "with_kernel_ridge")
KERNELS = ("linear", "rbf", "laplacian", "polynomial")


def _sq_dists(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape ``(len(X), len(Z))``."""
    squared = (np.sum(X ** 2, axis=1)[:, None]
               + np.sum(Z ** 2, axis=1)[None, :]
               - 2.0 * X @ Z.T)
    return np.maximum(squared, 0.0)


def _kernel_matrix(X: np.ndarray, Z: np.ndarray, kind: str, gamma: float,
                   degree: int, coef0: float) -> np.ndarray:
    """A kernel between two point sets, with the bandwidth already resolved."""
    if kind == "linear":
        return X @ Z.T
    if kind == "polynomial":
        return (gamma * (X @ Z.T) + coef0) ** degree
    if kind == "laplacian":
        gaps = np.abs(X[:, None, :] - Z[None, :, :])
        return np.exp(-gamma * gaps.sum(axis=2))
    return np.exp(-gamma * _sq_dists(X, Z))


class Nystrom:
    """A rank-m feature map read off m rows of the kernel matrix.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    n_components : int
        Total feature dimension, equal to the number of landmarks for
        `standard` and `with_kernel_ridge`. For `ensemble` it is rounded
        down to a multiple of `n_ensemble`.
    kernel : str
        One of `KERNELS`.
    gamma : float or None
        Bandwidth; `None` means ``1 / n_features``.
    degree, coef0 : float
        Polynomial parameters, used by the `polynomial` kernel only.
    n_ensemble : int
        Independent landmark samples for `ensemble`.
    lam : float
        Ridge penalty for `with_kernel_ridge`.
    chunk_size : int
        Rows per call in the internal streaming loop of `fit`.
    seed : int
        Seed for landmark selection and for the reservoir.

    Attributes
    ----------
    landmarks_ : ndarray
        The selected rows, shape ``(m, d)``.
    map_ : ndarray
        The whitening matrix ``K_mm^{-1/2}`` in the eigenbasis, shape
        ``(m, m)``; features are ``kernel(X, landmarks_) @ map_``.
    coef_ : ndarray
        Ridge readout in feature space, for `with_kernel_ridge` only.
    parts_ : list of (ndarray, ndarray)
        Per-model landmarks and maps, for `ensemble` only.
    n_seen_ : int
        Rows consumed by the reservoir, for `streaming` only.
    """

    def __init__(self, *, variant: str = "standard", n_components: int = 50,
                 kernel: str = "rbf", gamma: float | None = None,
                 degree: int = 3, coef0: float = 1.0, n_ensemble: int = 8,
                 lam: float = 1e-2, chunk_size: int = 64,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if (not isinstance(n_ensemble, (int, np.integer))
                or n_ensemble < 1):
            raise ValueError("n_ensemble must be a positive integer")
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lam must be finite and positive")
        if not np.isfinite(coef0):
            raise ValueError("coef0 must be finite")
        if gamma is not None and (not np.isfinite(gamma) or gamma <= 0.0):
            raise ValueError("gamma must be finite and positive")
        if not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError("degree must be a positive integer")
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.variant = variant
        self.n_components = int(n_components)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.n_ensemble = int(n_ensemble)
        self.lam = float(lam)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # inputs and kernel
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

    def _kernel(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return _kernel_matrix(X, Z, self.kernel, self.gamma_, self.degree,
                              self.coef0)

    @staticmethod
    def _whitener(gram: np.ndarray) -> np.ndarray:
        """``K^{-1/2}`` through an eigendecomposition with a floor."""
        values, vectors = np.linalg.eigh(gram)
        scale = float(np.max(np.abs(values)))
        floor = max(scale, 1.0) * 1e-12
        values = np.maximum(values, floor)
        return vectors / np.sqrt(values)

    def _set_landmarks(self, points: np.ndarray,
                       rng: np.random.Generator) -> None:
        """Choose ``n_components`` rows and factor their Gram matrix."""
        count = min(self.n_components, points.shape[0])
        chosen = rng.choice(points.shape[0], size=count, replace=False)
        chosen.sort()
        self.landmarks_ = points[chosen].copy()
        self.map_ = self._whitener(self._kernel(self.landmarks_,
                                                self.landmarks_))
        self.n_components_ = count

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def fit(self, X, y=None) -> Nystrom:
        """Select landmarks, factor their Gram matrix, and fit the readout."""
        points = self._check_X(X)
        width = points.shape[1]
        self.n_features_in_ = width
        self.gamma_ = (1.0 / width if self.gamma is None
                       else float(self.gamma))
        if y is not None:
            targets = np.asarray(y, dtype=float)
            if targets.ndim != 1 or targets.size != points.shape[0]:
                raise ValueError("y must have one target per row of X")
            if not np.all(np.isfinite(targets)):
                raise ValueError("y must be finite")
        else:
            targets = None
        if self.variant == "ensemble":
            self._fit_ensemble(points)
            return self
        if self.variant == "streaming":
            if targets is not None:
                raise ValueError("streaming approximates the kernel; "
                                 "use with_kernel_ridge for a readout")
            self._start_stream(width)
            for start in range(0, points.shape[0], self.chunk_size):
                self.partial_fit(points[start:start + self.chunk_size])
            return self
        rng = np.random.default_rng(self.seed)
        self._set_landmarks(points, rng)
        if self.variant == "with_kernel_ridge":
            if targets is None:
                raise ValueError("with_kernel_ridge needs targets y")
            features = self.transform(points)
            system = (features.T @ features
                      + self.lam * np.eye(self.n_components_))
            self.coef_ = np.linalg.solve(system, features.T @ targets)
        elif targets is not None:
            raise ValueError("this variant approximates the kernel only; "
                             "use with_kernel_ridge for a readout")
        return self

    def _fit_ensemble(self, points: np.ndarray) -> None:
        """Spend the budget on several independent landmark samples."""
        per = max(1, self.n_components // self.n_ensemble)
        rng = np.random.default_rng(self.seed)
        self.parts_ = []
        for _ in range(self.n_ensemble):
            count = min(per, points.shape[0])
            chosen = rng.choice(points.shape[0], size=count, replace=False)
            chosen.sort()
            landmarks = points[chosen].copy()
            self.parts_.append((landmarks,
                                self._whitener(self._kernel(landmarks,
                                                            landmarks))))
        self.n_components_ = per * self.n_ensemble

    def _start_stream(self, width: int) -> None:
        """Initialise the reservoir sampler without touching the data."""
        self._buffer = np.zeros((self.n_components, width))
        self._rng = np.random.default_rng(self.seed)
        self.landmarks_ = self._buffer[:0]
        self.map_ = None
        self.n_components_ = self.n_components
        self.n_seen_ = 0
        self.n_chunks_ = 0

    # ------------------------------------------------------------------ #
    # streaming
    # ------------------------------------------------------------------ #
    def partial_fit(self, X) -> Nystrom:
        """Add a chunk to the reservoir, keeping at most m rows."""
        if self.variant != "streaming":
            raise RuntimeError("partial_fit is only for the streaming variant")
        if not hasattr(self, "_buffer"):
            raise RuntimeError("call fit or _start_stream before partial_fit")
        points = self._check_X(X)
        if points.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        for row in points:
            self.n_seen_ += 1
            held = self.landmarks_.shape[0]
            if held < self.n_components:
                self._buffer[held] = row
                self.landmarks_ = self._buffer[:held + 1]
                self.map_ = None
            else:
                slot = int(self._rng.integers(self.n_seen_))
                if slot < self.n_components:
                    self._buffer[slot] = row
                    self.map_ = None
        self.n_chunks_ += 1
        return self

    # ------------------------------------------------------------------ #
    # features and kernel
    # ------------------------------------------------------------------ #
    def transform(self, X) -> np.ndarray:
        """The m-dimensional feature map, shape ``(len(X), n_components_)``."""
        points = self._check_X(X)
        if points.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        if self.variant == "ensemble":
            width = np.sqrt(len(self.parts_))
            return np.concatenate(
                [self._kernel(points, landmarks) @ mapping
                 for landmarks, mapping in self.parts_], axis=1) / width
        if self.landmarks_.shape[0] == 0:
            return np.zeros((points.shape[0], 0))
        if self.map_ is None:
            self.map_ = self._whitener(self._kernel(self.landmarks_,
                                                    self.landmarks_))
        return self._kernel(points, self.landmarks_) @ self.map_

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and return the features of the training data."""
        return self.fit(X, y).transform(X)

    def approximate_kernel(self, A, B=None) -> np.ndarray:
        """``Φ(A) Φ(B)ᵀ``, the reconstructed kernel block."""
        left = self.transform(A)
        if B is None:
            return left @ left.T
        return left @ self.transform(B).T

    def reconstruction_error(self, X) -> float:
        """Relative Frobenius error of the reconstruction on ``X``."""
        points = self._check_X(X)
        exact = self._kernel(points, points)
        estimate = self.approximate_kernel(points)
        return float(np.linalg.norm(exact - estimate) / np.linalg.norm(exact))

    # ------------------------------------------------------------------ #
    # readout
    # ------------------------------------------------------------------ #
    def predict(self, X) -> np.ndarray:
        """Ridge prediction in feature space, for `with_kernel_ridge`."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("call fit with targets before predicting")
        return self.transform(X) @ self.coef_

    def n_stored(self) -> int:
        """Rows kept in memory: landmarks, not training points."""
        if self.variant == "ensemble":
            if not hasattr(self, "parts_"):
                raise RuntimeError("call fit before asking")
            return int(sum(part.shape[0] for part, mapping in self.parts_))
        if not hasattr(self, "landmarks_"):
            raise RuntimeError("call fit before asking")
        return int(self.landmarks_.shape[0])


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _target(X: np.ndarray) -> np.ndarray:
    """The smooth 2-D function every regression demo fits."""
    return np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.2 * X[:, 0]


def _exact_ridge(train: np.ndarray, targets: np.ndarray, test: np.ndarray,
                 gamma: float, penalty: float) -> np.ndarray:
    """Kernel ridge with the full Gram matrix, no intercept."""
    gram = _kernel_matrix(train, train, "rbf", gamma, 3, 1.0)
    weights = np.linalg.solve(gram + penalty * np.eye(train.shape[0]), targets)
    return _kernel_matrix(test, train, "rbf", gamma, 3, 1.0) @ weights


def _demo_standard() -> None:
    """The rank of the approximation is the number of rows you keep."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 2))
    test = rng.normal(size=(200, 2))
    gamma, penalty = 0.5, 1e-2
    targets, truth = _target(train), _target(test)
    exact = _exact_ridge(train, targets, test, gamma, penalty)
    print("400 points, RBF kernel with gamma 0.5, ridge penalty 0.01:")
    print(f"  exact kernel ridge, all 400 points   test MSE "
          f"{np.mean((exact - truth) ** 2):.5f}")
    print("  landmarks   Frobenius error   worst entry   test MSE")
    errors = []
    for count in (5, 10, 20, 40, 80, 160, 400):
        model = Nystrom(n_components=count, gamma=gamma, seed=0).fit(train)
        error = model.reconstruction_error(train)
        worst = float(np.abs(model.approximate_kernel(train)
                             - _kernel_matrix(train, train, "rbf", gamma, 3,
                                              1.0)).max())
        ridge = Nystrom(variant="with_kernel_ridge", n_components=count,
                        gamma=gamma, lam=penalty, seed=0).fit(train, targets)
        mse = float(np.mean((ridge.predict(test) - truth) ** 2))
        errors.append(error)
        print(f"  {count:>9}   {error:>16.4f}   {worst:>11.4f}   {mse:.5f}")
        assert model.n_stored() == count
    print("  the approximation is built from m rows, so its rank is at most m")
    print("  and the error is the part of the kernel those rows cannot see")
    assert errors == sorted(errors, reverse=True)
    full = Nystrom(n_components=400, gamma=gamma, seed=0).fit(train)
    assert full.reconstruction_error(train) < 1e-12
    assert np.abs(full.approximate_kernel(train)
                  - _kernel_matrix(train, train, "rbf", gamma, 3, 1.0)).max() < 1e-10
    print("  with every point a landmark the reconstruction is exact to "
          f"{full.reconstruction_error(train):.1e},")
    print("  which is the check that the whitener and the kernel agree")


def _demo_ensemble() -> None:
    """Independent samples average out, and dimension is not free."""
    rng = np.random.default_rng(0)
    points = rng.normal(size=(400, 2))
    print("400 points, 30 seeds, each map has 15 landmarks:")
    print("  maps   dimension   mean error   spread")
    spreads = []
    for count in (1, 2, 3, 4, 6, 12):
        errors = []
        for seed in range(30):
            model = Nystrom(variant="ensemble", n_components=15 * count,
                            n_ensemble=count, gamma=0.5, seed=seed)
            model.fit(points)
            errors.append(model.reconstruction_error(points))
        spreads.append(float(np.std(errors)))
        print(f"  {count:>4}   {model.n_components_:>9}   "
              f"{np.mean(errors):>10.4f}   {spreads[-1]:.5f}")
        assert model.n_stored() == 15 * count
    print("  each map is an independent draw, so averaging E of them leaves")
    print("  each map's bias unchanged and divides its variance: the spread")
    print(f"  falls from {spreads[0]:.4f} to {spreads[-1]:.4f} at twelve times the")
    print("  dimension, and the expected error falls with it")
    assert spreads == sorted(spreads, reverse=True)
    assert spreads[-1] < 0.5 * spreads[0]

    print("the same total dimension spent on one map is a different trade:")
    single = Nystrom(n_components=60, gamma=0.5, seed=0).fit(points)
    split = Nystrom(variant="ensemble", n_components=60, n_ensemble=4,
                    gamma=0.5, seed=0).fit(points)
    print(f"  one map of 60 landmarks     {single.reconstruction_error(points):.4f}")
    print(f"  four maps of 15 landmarks   {split.reconstruction_error(points):.4f}")
    print("  the error is driven by the landmarks per map, not by the total,")
    print("  so an ensemble costs E times the memory and buys variance only")
    assert single.reconstruction_error(points) < split.reconstruction_error(points)


def _demo_streaming() -> None:
    """A reservoir keeps m rows, whatever the stream length."""
    rng = np.random.default_rng(1)
    points = rng.normal(size=(600, 2))
    count = 60
    print("600 points in chunks of 37, a reservoir of 60 landmarks:")
    print("  landmarks   rows seen   chunks   rows held   error")
    models = []
    for size in (20, 60, 120):
        model = Nystrom(variant="streaming", n_components=size, gamma=0.5,
                        chunk_size=37, seed=0).fit(points)
        models.append(model)
        print(f"  {size:>9}   {model.n_seen_:>9}   {model.n_chunks_:>6}   "
              f"{model.n_stored():>9}   {model.reconstruction_error(points):.4f}")
        assert model.n_stored() == size
        assert model.n_seen_ == 600
        assert model.n_chunks_ == 17
    print("  the last chunk has eight rows, which is why 600 / 37 is 17")
    print("  chunks and not 16: the reservoir does not need equal chunks")

    batch = []
    for seed in range(30):
        model = Nystrom(n_components=count, gamma=0.5, seed=seed).fit(points)
        batch.append(model.reconstruction_error(points))
    stream = []
    for seed in range(30):
        model = Nystrom(variant="streaming", n_components=count, gamma=0.5,
                        chunk_size=37, seed=seed).fit(points)
        stream.append(model.reconstruction_error(points))
    print("  the reservoir is a uniform sample, so the error distribution is")
    print(f"  the batch one: streaming {np.mean(stream):.4f} +/- "
          f"{np.std(stream):.4f}, batch {np.mean(batch):.4f} +/- "
          f"{np.std(batch):.4f}")
    assert abs(np.mean(stream) - np.mean(batch)) < 0.002
    print("  and memory is 60 rows rather than 600, which is the point")

    replay = Nystrom(variant="streaming", n_components=count, gamma=0.5,
                     chunk_size=37, seed=0).fit(points)
    assert np.array_equal(np.sort(replay.landmarks_, axis=0),
                          np.sort(models[1].landmarks_, axis=0))
    print("  the same seed replays the same reservoir")


def _demo_ridge() -> None:
    """A ridge readout on the features is kernel ridge on the kernel."""
    rng = np.random.default_rng(2)
    train = rng.normal(size=(300, 2))
    test = rng.normal(size=(200, 2))
    targets, truth = _target(train), _target(test)
    gamma, penalty = 0.5, 1e-2
    exact = _exact_ridge(train, targets, test, gamma, penalty)
    print("300 training points, the readout in the m-dimensional feature space:")
    print("  landmarks   test MSE   against exact")
    for count in (10, 20, 40, 80, 160):
        model = Nystrom(variant="with_kernel_ridge", n_components=count,
                        gamma=gamma, lam=penalty, seed=0).fit(train, targets)
        mse = float(np.mean((model.predict(test) - truth) ** 2))
        print(f"  {count:>9}   {mse:.5f}   "
              f"{mse / np.mean((exact - truth) ** 2):>13.2f}x")
    print("  no kernel matrix of the training set is ever formed: the solve")
    print("  is m x m and the features are m columns")

    model = Nystrom(variant="with_kernel_ridge", n_components=40, gamma=gamma,
                    lam=penalty, seed=0).fit(train, targets)
    features = model.transform(train)
    mapping = model.transform(test)
    dual = np.linalg.solve(features @ features.T + penalty * np.eye(300),
                           targets)
    primal = mapping @ model.coef_
    through_dual = (mapping @ features.T) @ dual
    print("the feature readout and the dual readout are the same estimator:")
    print(f"  primal  Phi_t coef_             {primal[0]:.10f}")
    print(f"  dual    K_tn (K_nn+lam)^-1 y  {through_dual[0]:.10f}")
    print(f"  worst difference over 200 rows   "
          f"{np.abs(primal - through_dual).max():.1e}")
    print("  that is the Woodbury identity, and it is why a kernel method can")
    print("  be served as a linear model once the landmarks are chosen")
    assert np.abs(primal - through_dual).max() < 1e-10

    full = Nystrom(variant="with_kernel_ridge", n_components=300,
                   gamma=gamma, lam=penalty, seed=0).fit(train, targets)
    print("with every point a landmark the readout is exact kernel ridge:")
    print(f"  worst difference from the 300 x 300 solve   "
          f"{np.abs(full.predict(test) - exact).max():.1e}")
    assert np.abs(full.predict(test) - exact).max() < 1e-8


def _demo_guards() -> None:
    """Malformed settings and inputs, all refused."""
    X = np.linspace(-1.0, 1.0, 40).reshape(20, 2)
    y = np.sin(X[:, 0])
    model = Nystrom(n_components=5).fit(X)
    ridge = Nystrom(variant="with_kernel_ridge", n_components=5).fit(X, y)
    stream = Nystrom(variant="streaming", n_components=5).fit(X)
    attempts = [
        lambda: Nystrom(variant="nope"),
        lambda: Nystrom(kernel="cosine"),
        lambda: Nystrom(n_components=0),
        lambda: Nystrom(n_ensemble=0),
        lambda: Nystrom(lam=0.0),
        lambda: Nystrom(lam=np.nan),
        lambda: Nystrom(coef0=np.nan),
        lambda: Nystrom(gamma=0.0),
        lambda: Nystrom(gamma=-1.0),
        lambda: Nystrom(degree=0),
        lambda: Nystrom(chunk_size=0),
        lambda: Nystrom(seed=-1),
        lambda: Nystrom().fit(X, y),
        lambda: Nystrom(variant="with_kernel_ridge").fit(X),
        lambda: ridge.fit(X, y[:3]),
        lambda: ridge.fit(X, np.full(20, np.nan)),
        lambda: model.fit([[np.nan, 1.0]]),
        lambda: model.fit([[[1.0]]]),
        lambda: Nystrom().predict(X),
        lambda: Nystrom().n_stored(),
        lambda: model.partial_fit(X),
        lambda: Nystrom(variant="streaming").partial_fit(X),
        lambda: stream.partial_fit(X[:, :1]),
        lambda: model.transform(X[:, :1]),
        lambda: Nystrom(variant="streaming").fit(X, y),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant or kernel, a zero dimension, zero ensemble members, a")
    print("  zero or non-finite penalty, a non-finite offset, a non-positive")
    print("  gamma, zero degree or chunk, a negative seed, targets given to a")
    print("  kernel-only variant, no targets for the readout, a short or")
    print("  non-finite target vector, a non-finite or three-dimensional")
    print("  design, predict or the stored count before fit, a chunk given to")
    print("  a batch variant, a chunk before the reservoir exists, a chunk of")
    print("  the wrong width, a transform of the wrong width, and targets to")
    print("  the streaming variant.")
    assert refused == len(attempts)


def main() -> None:
    _demo_standard()
    _demo_ensemble()
    _demo_streaming()
    _demo_ridge()
    _demo_guards()
    print("nystrom demo passed")


if __name__ == "__main__":
    main()
