"""Radial basis function networks — bumps, centres and a linear readout.

Chapter:   03-kernels-and-instance
Companion: rbf_network.md
Tier 2 · 5 variants: exact_interpolation, kmeans_centers, ols_readout,
                     normalized_rbf, grnn

    phi_j(x) = exp(-||x - c_j||^2 / (2 sigma^2))                      (1)
    f(x)     = sum_j w_j phi_j(x) + b                                 (2)

A radial basis function network is two decisions wearing one name. The hidden
layer places a bump — a Gaussian centred at c_j with width sigma — and the
output layer is linear in the bumps. Fitting is therefore not backpropagation:
choose the centres, choose the width, then solve a linear least-squares problem
for the weights. The atom is the smallest member of the kernel family that is
not a kernel method in disguise, and the reference point for every "just add a
hidden layer" argument.

The variants are the two decisions. `exact_interpolation` puts a centre on every
training point and solves the square system, which interpolates the data exactly
because the Gaussian interpolation matrix is nonsingular. `kmeans_centers` picks
a handful of centres with Lloyd's algorithm, so the problem becomes
overdetermined and the readout becomes a least-squares fit. `ols_readout` adds
an intercept and a ridge term, which is what makes a wide model stable.
`normalized_rbf` divides by the sum of the activations, turning the basis into a
partition of unity. `grnn` goes further and skips the readout entirely: the
weights are the training targets, so the prediction is a distance-weighted
average of them, which is the Nadaraya-Watson estimator.

Everything is numpy. The demo checks that the interpolation matrix is positive
definite and solves to rounding, that the width trades between-point accuracy
against the conditioning of the system, that a handful of k-means centres beats
both two of them and all sixty, that the normalised basis reproduces a constant
exactly where the raw one cannot, and that the GRNN has the two limits it is
supposed to have: nearest neighbour as the width goes to zero, the target mean
as it goes to infinity.

Numbers to carry away (measured, see the demo):

  - exact interpolation reaches a residual of 3.7e-15 at the auto width 0.1085
  - widening sigma from 0.1 to 1.5 drops the between-point error from 0.386 to
    2e-05 while the condition number climbs from 7e+04 to 1e+18
  - eight k-means centres reach a held-out RMSE of 0.0771, sixty reach 1.8042
  - the normalised basis reproduces a constant to 2e-15, the raw basis to 3.8e-02
  - the GRNN matches one-nearest-neighbour exactly at width 1e-03 and the target
    mean to 8e-13 at width 1e+06

"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("exact_interpolation", "kmeans_centers", "ols_readout",
            "normalized_rbf", "grnn")

_NORMALIZED = ("normalized_rbf", "grnn")


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape ``(len(X), len(C))``."""
    squared = (np.sum(X ** 2, axis=1)[:, None]
               + np.sum(C ** 2, axis=1)[None, :]
               - 2.0 * X @ C.T)
    return np.maximum(squared, 0.0)


def _kmeans(X: np.ndarray, count: int, rng: np.random.Generator,
            iterations: int) -> np.ndarray:
    """Lloyd's algorithm with a k-means++ initialisation."""
    rows = X.shape[0]
    chosen = [int(rng.integers(rows))]
    for _ in range(1, count):
        nearest = _sq_dists(X, X[chosen]).min(axis=1)
        total = float(nearest.sum())
        weights = None if total <= 0.0 else nearest / total
        chosen.append(int(rng.choice(rows, p=weights)))
    centres = X[chosen].copy()
    for _ in range(iterations):
        labels = _sq_dists(X, centres).argmin(axis=1)
        updated = centres.copy()
        for index in range(count):
            members = labels == index
            if members.any():
                updated[index] = X[members].mean(axis=0)
        if np.allclose(updated, centres):
            break
        centres = updated
    return centres


class RBFNetwork:
    """Gaussian radial basis network with a closed-form readout.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `exact_interpolation` and `grnn` ignore
        `n_centers` and use every training point.
    n_centers : int or None
        Number of hidden units; `None` picks ``min(16, n)``.
    sigma : float or None
        Width of the bumps; `None` uses the mean nearest-centre distance,
        which is the scale at which the bumps just touch.
    ridge : float
        Penalty on the readout weights, used by `ols_readout`.
    iterations : int
        Lloyd iterations for the centre search.
    seed : int
        Seed for the k-means++ draw.

    Attributes
    ----------
    centers_ : ndarray
        Hidden-unit centres, shape ``(k, n_features)``.
    sigma_ : float
        The width actually used.
    weights_ : ndarray
        Readout weights, shape ``(k,)``.
    bias_ : float
        Intercept, zero when the variant has none.
    """

    def __init__(self, *, variant: str = "kmeans_centers",
                 n_centers: int | None = None, sigma: float | None = None,
                 ridge: float = 1e-6, iterations: int = 100,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if n_centers is not None and (not isinstance(n_centers, (int, np.integer))
                                      or n_centers < 1):
            raise ValueError("n_centers must be a positive integer or None")
        if sigma is not None and (not np.isfinite(sigma) or sigma <= 0.0):
            raise ValueError("sigma must be finite and positive")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        if not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.variant = variant
        self.n_centers = n_centers
        self.sigma = sigma
        self.ridge = float(ridge)
        self.iterations = int(iterations)
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

    @staticmethod
    def _check_y(y, rows: int) -> np.ndarray:
        array = np.asarray(y, dtype=float).ravel()
        if array.size != rows:
            raise ValueError("y must have one entry per row of X")
        if not np.all(np.isfinite(array)):
            raise ValueError("y must be finite")
        return array

    # ------------------------------------------------------------------ #
    # activations
    # ------------------------------------------------------------------ #
    def _design(self, X: np.ndarray) -> np.ndarray:
        """The hidden layer, shape ``(len(X), k)``."""
        squared = _sq_dists(X, self.centers_)
        if self.variant in _NORMALIZED:
            squared = squared - squared.min(axis=1, keepdims=True)
        phi = np.exp(-squared / (2.0 * self.sigma_ ** 2))
        if self.variant in _NORMALIZED:
            phi = phi / phi.sum(axis=1, keepdims=True)
        return phi

    def activations(self, X) -> np.ndarray:
        """Public access to the hidden layer, for inspection and tests."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit before asking for activations")
        return self._design(self._check_X(X))

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def _choose_centers(self, X: np.ndarray) -> np.ndarray:
        rows = X.shape[0]
        if self.variant in ("exact_interpolation", "grnn"):
            return X.copy()
        wanted = 16 if self.n_centers is None else int(self.n_centers)
        count = min(wanted, rows)
        if count == rows:
            return X.copy()
        rng = np.random.default_rng(self.seed)
        return _kmeans(X, count, rng, self.iterations)

    def _choose_sigma(self, X: np.ndarray) -> float:
        if self.sigma is not None:
            return float(self.sigma)
        if self.centers_.shape[0] == 1:
            return float(np.std(X) + 1e-12)
        nearest = _sq_dists(self.centers_, self.centers_)
        np.fill_diagonal(nearest, np.inf)
        distance = float(np.sqrt(nearest.min(axis=1)).mean())
        if not np.isfinite(distance) or distance <= 0.0:
            return 1.0
        return distance

    def _solve_readout(self, phi: np.ndarray, y: np.ndarray) -> None:
        """Least squares with an optional intercept and ridge penalty."""
        if self.variant == "grnn":
            self.weights_ = y.copy()
            self.bias_ = 0.0
            return
        if self.variant == "ols_readout":
            design = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
            penalty = self.ridge * np.eye(design.shape[1])
            penalty[-1, -1] = 0.0
            solution, *_ = np.linalg.lstsq(
                design.T @ design + penalty, design.T @ y, rcond=None)
            self.weights_ = solution[:-1]
            self.bias_ = float(solution[-1])
            return
        solution, *_ = np.linalg.lstsq(phi, y, rcond=None)
        self.weights_ = solution
        self.bias_ = 0.0

    def fit(self, X, y) -> RBFNetwork:
        """Place the bumps, then solve the readout in closed form."""
        points = self._check_X(X)
        targets = self._check_y(y, points.shape[0])
        if points.shape[0] < 2:
            raise ValueError("at least two training points are required")
        if self.n_centers is not None and self.n_centers > points.shape[0]:
            raise ValueError("n_centers cannot exceed the number of rows")
        self.centers_ = self._choose_centers(points)
        self.sigma_ = self._choose_sigma(points)
        self.y_ = targets
        self.design_ = self._design(points)
        self._solve_readout(self.design_, targets)
        return self

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(self, X) -> np.ndarray:
        """The weighted sum of bumps, plus the intercept."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit before predicting")
        points = self._check_X(X)
        if points.shape[1] != self.centers_.shape[1]:
            raise ValueError("X has the wrong number of features")
        return self._design(points) @ self.weights_ + self.bias_

    def residual(self, X, y) -> np.ndarray:
        """Signed training residual, one entry per row."""
        targets = self._check_y(y, self._check_X(X).shape[0])
        return targets - self.predict(X)

    def rmse(self, X, y) -> float:
        """Root mean squared error on a set."""
        return float(np.sqrt(np.mean(self.residual(X, y) ** 2)))

    def n_parameters(self) -> int:
        """Weights, intercept and the coordinates of every centre."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit before asking")
        width = self.centers_.size + self.weights_.size
        return int(width + (1 if self.bias_ != 0.0 else 0))


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _target(x: np.ndarray) -> np.ndarray:
    """The 1-D function every demo fits."""
    return np.sin(2.0 * x) + 0.3 * x


def _demo_interpolation() -> None:
    """One centre per point interpolates exactly, at a price."""
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(-3.0, 3.0, 25))[:, None]
    y = _target(X[:, 0])
    grid = np.linspace(-3.0, 3.0, 400)[:, None]
    model = RBFNetwork(variant="exact_interpolation").fit(X, y)
    eigenvalues = np.linalg.eigvalsh(model.design_)
    print("25 points, one centre each, width from the mean spacing:")
    print(f"  width            {model.sigma_:.4f}")
    print(f"  training residual {model.rmse(X, y):.1e}")
    print(f"  smallest eigenvalue {eigenvalues.min():.1e}, so the system "
          "is positive definite")
    print("  the interpolation matrix of a Gaussian is nonsingular for")
    print("  distinct centres, so the square system always has one answer")
    assert model.rmse(X, y) < 1e-10
    assert eigenvalues.min() > 0.0
    print("what the width costs, on the same noiseless 25 points:")
    print("  sigma   grid rmse   condition number")
    errors, conditions = [], []
    for sigma in (0.1, 0.3, 0.8, 1.5):
        wide = RBFNetwork(variant="exact_interpolation", sigma=sigma).fit(X, y)
        error = float(np.sqrt(np.mean((wide.predict(grid) - _target(grid[:, 0])) ** 2)))
        condition = float(np.linalg.cond(wide.design_))
        errors.append(error)
        conditions.append(condition)
        print(f"  {sigma:>5.1f}   {error:>9.5f}   {condition:.1e}")
        assert wide.rmse(X, y) < 1e-6
    print("  a wider bump fits the gap between points far better and makes")
    print("  the linear system far worse: accuracy and stability pull apart")
    assert errors == sorted(errors, reverse=True)
    assert conditions == sorted(conditions)


def _demo_centers() -> None:
    """Fewer centres than points, chosen by k-means, fitted by least squares."""
    rng = np.random.default_rng(1)
    f = _target
    train = np.sort(rng.uniform(-3.0, 3.0, 60))[:, None]
    targets = f(train[:, 0]) + rng.normal(0.0, 0.1, 60)
    test = np.linspace(-3.0, 3.0, 200)[:, None]
    held = f(test[:, 0])
    print("60 noisy points, k-means centres, least-squares readout:")
    print("  centres   train rmse   test rmse")
    scores = {}
    for count in (2, 4, 8, 16, 60):
        model = RBFNetwork(variant="kmeans_centers",
                           n_centers=count).fit(train, targets)
        scores[count] = model.rmse(test, held)
        print(f"  {count:>7}   {model.rmse(train, targets):>10.4f}   "
              f"{scores[count]:>9.4f}")
    print("  two bumps cannot bend twice; sixty bumps follow the noise.")
    print(f"  the best of the five is {min(scores.values()):.4f} at "
          f"{min(scores, key=scores.get)} centres")
    assert scores[8] < scores[2]
    assert scores[8] < scores[60]
    model = RBFNetwork(variant="kmeans_centers", n_centers=8, seed=0)
    first = model.fit(train, targets).centers_
    second = RBFNetwork(variant="kmeans_centers", n_centers=8,
                        seed=0).fit(train, targets).centers_
    distances = _sq_dists(model.centers_, model.centers_)
    np.fill_diagonal(distances, np.inf)
    print(f"  the same seed gives the same centres: "
          f"{np.array_equal(first, second)}")
    print(f"  the nearest pair of centres is {np.sqrt(distances.min()):.4f} "
          "apart, and every centre is the mean of its own cluster")
    assert np.array_equal(first, second)
    labels = _sq_dists(train, model.centers_).argmin(axis=1)
    for index, centre in enumerate(model.centers_):
        members = labels == index
        assert members.any()
        assert np.allclose(train[members].mean(axis=0), centre)


def _demo_readout() -> None:
    """An intercept plus a ridge turns the readout into a shrinkage path."""
    rng = np.random.default_rng(2)
    train = np.sort(rng.uniform(-3.0, 3.0, 60))[:, None]
    targets = _target(train[:, 0]) + rng.normal(0.0, 0.1, 60)
    test = np.linspace(-3.0, 3.0, 200)[:, None]
    held = _target(test[:, 0])
    print("the same eight centres, three readouts:")
    print("  readout          bias     test rmse")
    scores = {}
    for variant in ("kmeans_centers", "ols_readout", "normalized_rbf"):
        model = RBFNetwork(variant=variant, n_centers=8).fit(train, targets)
        scores[variant] = model.rmse(test, held)
        print(f"  {variant:<16} {model.bias_:>6.3f}   {scores[variant]:>9.4f}")
    print("  only the least-squares variant has an intercept, and on this")
    print("  sample it costs accuracy rather than buying it; normalising the")
    print("  basis is what helps here. Neither fact is a theorem.")
    assert scores["normalized_rbf"] < scores["ols_readout"]
    assert abs(RBFNetwork(variant="ols_readout",
                          n_centers=8).fit(train, targets).bias_) > 0.1
    assert RBFNetwork(variant="kmeans_centers",
                      n_centers=8).fit(train, targets).bias_ == 0.0
    print("the ridge path, same centres, penalty from zero to a million:")
    print("  ridge     weight norm   bias    test rmse")
    norms, biases, errors = [], [], []
    for ridge in (0.0, 1.0, 1e2, 1e4, 1e6):
        model = RBFNetwork(variant="ols_readout", n_centers=8,
                           ridge=ridge).fit(train, targets)
        norms.append(float(np.linalg.norm(model.weights_)))
        biases.append(model.bias_)
        errors.append(model.rmse(test, held))
        print(f"  {ridge:>7.0e}   {norms[-1]:>11.4f}   {model.bias_:>6.3f}   "
              f"{errors[-1]:>9.4f}")
    print("  as the penalty grows the weights vanish and the bias tends to the")
    print(f"  target mean {targets.mean():.4f}; the model then predicts that")
    print(f"  constant and its error is the spread of the signal, "
          f"{np.std(held):.4f}")
    assert norms == sorted(norms, reverse=True)
    assert abs(biases[-1] - float(targets.mean())) < 1e-3
    assert abs(errors[-1] - float(np.std(held))) < 0.05


def _demo_normalized() -> None:
    """Dividing by the sum makes the basis a partition of unity."""
    rng = np.random.default_rng(3)
    train = np.sort(rng.uniform(-3.0, 3.0, 40))[:, None]
    targets = _target(train[:, 0]) + rng.normal(0.0, 0.1, 40)
    flat = np.full(40, 1.7)
    print("a constant target is the sharpest test of a basis "
          "(40 points, 8 centres):")
    print("  variant           residual   bias")
    constant = {}
    for variant in ("normalized_rbf", "kmeans_centers", "ols_readout"):
        model = RBFNetwork(variant=variant, n_centers=8).fit(train, flat)
        constant[variant] = (model.rmse(train, flat), model.bias_)
        print(f"  {variant:<16}  {constant[variant][0]:>9.1e}   "
              f"{model.bias_:>6.3f}")
    print("  the normalised basis sums to one everywhere, so a constant target")
    print("  is reproduced with every weight equal to it. The raw basis sums to")
    print("  between 1.09 and 2.70, so it cannot. The intercept is the other")
    print("  way to represent a constant, and it spends its one parameter on it.")
    assert constant["normalized_rbf"][0] < 1e-10
    assert constant["kmeans_centers"][0] > 1e-3
    assert abs(constant["ols_readout"][1] - 1.7) < 1e-9

    normalized = RBFNetwork(variant="normalized_rbf",
                            n_centers=8).fit(train, targets)
    plain = RBFNetwork(variant="kmeans_centers",
                       n_centers=8).fit(train, targets)
    grid = np.linspace(-3.0, 3.0, 300)[:, None]
    sums = normalized.activations(grid).sum(axis=1)
    raw = plain.activations(grid).sum(axis=1)
    inside = normalized.predict(grid)
    print("the same two models on the noisy targets:")
    print(f"  normalised row sums   {sums.min():.12f} to {sums.max():.12f}")
    print(f"  raw row sums          {raw.min():.4f} to {raw.max():.4f}")
    print(f"  weight range          {normalized.weights_.min():.4f} to "
          f"{normalized.weights_.max():.4f}")
    print(f"  prediction range      {inside.min():.4f} to {inside.max():.4f}")
    print("  a probability vector times the weights is an average, so the")
    print("  output stays inside the range of the weights whatever the input")
    assert np.allclose(sums, 1.0)
    assert raw.max() > 1.0
    assert inside.min() >= normalized.weights_.min() - 1e-12
    assert inside.max() <= normalized.weights_.max() + 1e-12
    test = np.linspace(-3.0, 3.0, 200)[:, None]
    held = _target(test[:, 0])
    print(f"  and the normalised model generalises better here: "
          f"{normalized.rmse(test, held):.4f} against "
          f"{plain.rmse(test, held):.4f}")


def _demo_grnn() -> None:
    """The two limits of a distance-weighted average."""
    rng = np.random.default_rng(4)
    train = np.sort(rng.uniform(-3.0, 3.0, 50))[:, None]
    targets = _target(train[:, 0]) + rng.normal(0.0, 0.1, 50)
    test = np.linspace(-2.8, 2.8, 25)[:, None]
    narrow = RBFNetwork(variant="grnn", sigma=1e-3).fit(train, targets)
    flat = RBFNetwork(variant="grnn", sigma=1e6).fit(train, targets)
    nearest = _sq_dists(test, train).argmin(axis=1)
    print("a generalised regression network is an average of the targets:")
    print(f"  width 1e-3: matches one-nearest-neighbour to "
          f"{np.abs(narrow.predict(test) - targets[nearest]).max():.1e}")
    print(f"  width 1e+6: matches the target mean {targets.mean():.4f} to "
          f"{np.abs(flat.predict(test) - targets.mean()).max():.1e}")
    assert np.allclose(narrow.predict(test), targets[nearest])
    assert np.allclose(flat.predict(test), targets.mean())
    print("  between the limits the width is the only knob:")
    print("  sigma     test rmse")
    scores = []
    for sigma in (0.02, 0.1, 0.3, 1.0, 3.0):
        model = RBFNetwork(variant="grnn", sigma=sigma).fit(train, targets)
        score = model.rmse(test, _target(test[:, 0]))
        scores.append(score)
        print(f"  {sigma:>6.2f}   {score:>9.4f}")
    assert min(scores) < scores[0]
    assert min(scores) < scores[-1]


def _demo_guards() -> None:
    """Malformed settings and inputs, all refused."""
    X = np.linspace(-1.0, 1.0, 6)[:, None]
    y = X[:, 0] ** 2
    model = RBFNetwork().fit(X, y)
    attempts = [
        lambda: RBFNetwork(variant="nope"),
        lambda: RBFNetwork(n_centers=0),
        lambda: RBFNetwork(n_centers=2.5),
        lambda: RBFNetwork(sigma=0.0),
        lambda: RBFNetwork(sigma=np.inf),
        lambda: RBFNetwork(ridge=-1.0),
        lambda: RBFNetwork(iterations=0),
        lambda: RBFNetwork(seed=-1),
        lambda: RBFNetwork(n_centers=10).fit(X, y),
        lambda: RBFNetwork().fit(X[:1], y[:1]),
        lambda: RBFNetwork().fit(X, y[:-1]),
        lambda: RBFNetwork().fit(X, [np.nan] * 6),
        lambda: RBFNetwork().fit([[[1.0]]], [1.0]),
        lambda: model.predict([[1.0, 2.0]]),
        lambda: RBFNetwork().predict(X),
        lambda: RBFNetwork().activations(X),
        lambda: RBFNetwork().n_parameters(),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant, a zero or fractional centre count, a zero or infinite")
    print("  width, a negative ridge, zero iterations, a negative seed, more")
    print("  centres than rows, a single point, mismatched or non-finite")
    print("  targets, a three-dimensional design, the wrong feature count, and")
    print("  predict, activations or the parameter count before fit.")
    assert refused == len(attempts)


def main() -> None:
    _demo_interpolation()
    _demo_centers()
    _demo_readout()
    _demo_normalized()
    _demo_grnn()
    _demo_guards()
    print("rbf_network demo passed")


if __name__ == "__main__":
    main()
