"""Support vector regression — a tube, a hinge and a dual.

Chapter:   03-kernels-and-instance
Companion: svr.md
Tier 2 · 5 variants: epsilon_svr, nu_svr, linear, rbf, smo

    f(x) = sum_i delta_i K(x_i, x) + b                                  (1)
    loss = max(0, |y - f(x)| - epsilon)                                (2)

Support vector regression keeps the support-vector idea and replaces the
classification margin with a tube. A prediction inside a band of half-width
epsilon around the target costs nothing; outside it the cost is linear in the
distance. The result is a sparse kernel expansion: most coefficients are exactly
zero, and the points that remain are the ones the tube cannot contain.

The dual is a box-constrained quadratic program in the difference
delta = alpha - alpha*:

    min 1/2 delta' K delta - y' delta + epsilon |delta|_1
    subject to  sum delta = 0,  -C <= delta_i <= C                     (3)

which is small enough for sequential minimal optimisation: pick the pair of
coefficients that violates the KKT conditions most, move them against each other
so the sum stays zero, and repeat. The atom implements that solver, the nu
parameterisation that replaces epsilon with a target fraction of support
vectors, and the linear and RBF kernels.

Everything is numpy. The demo checks the KKT conditions coefficient by
coefficient, checks strong duality (primal equals dual), checks that a noiseless
linear problem is fitted exactly, and checks the nu property: at nu = 0.5 at
least half the points are support vectors and at most half fall outside the
tube.

Numbers to carry away (measured, see the demo):

  - a noiseless line is recovered to 1.1e-13 with two nonzero weights
  - on noisy data the support set falls from 40 of 40 at epsilon = 0 to 4
  - KKT residual 1e-14, duality gap 1e-14 on every fitted epsilon model
  - RBF test error 0.0865 against 0.4786 for a linear kernel
  - nu = 0.5 with C = 1: 20 of 40 support vectors, 20 margin errors
"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("epsilon_svr", "nu_svr", "linear", "rbf", "smo")

_KERNELS = ("linear", "rbf", "polynomial", "sigmoid")


def _kernel_matrix(X: np.ndarray, Z: np.ndarray, kind: str, gamma: float,
                   degree: int, coef0: float) -> np.ndarray:
    """The Gram matrix between two point sets."""
    if kind == "linear":
        return X @ Z.T
    if kind == "rbf":
        squared = (np.sum(X ** 2, axis=1)[:, None]
                   + np.sum(Z ** 2, axis=1)[None, :] - 2.0 * X @ Z.T)
        return np.exp(-gamma * np.maximum(squared, 0.0))
    if kind == "polynomial":
        return (gamma * (X @ Z.T) + coef0) ** degree
    if kind == "sigmoid":
        return np.tanh(gamma * (X @ Z.T) + coef0)
    raise ValueError(f"unknown kernel {kind!r}")


class SVR:
    """Epsilon-insensitive support vector regression.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `linear` and `rbf` set the kernel; `nu_svr` sets the
        parameterisation; `smo` is the explicit solver path.
    kernel : str or None
        One of `linear`, `rbf`, `polynomial`, `sigmoid`; `None` picks the
        variant's default.
    C : float
        Box bound on the dual coefficients.
    epsilon : float
        Half-width of the insensitive tube.
    nu : float
        Target upper bound on the fraction of margin errors, in ``(0, 1]``.
    gamma : float or None
        Kernel scale; `None` means ``1 / n_features``.
    degree, coef0 : int, float
        Polynomial kernel parameters.
    tol : float
        KKT tolerance that stops the solver.
    max_iter : int
        Maximum number of SMO pair updates.

    Attributes
    ----------
    delta_ : ndarray
        ``alpha - alpha*``, one entry per training point.
    b_ : float
        Intercept.
    support_ : ndarray
        Indices whose dual coefficient is nonzero.
    epsilon_ : float
        The tube actually used: `epsilon`, or the value found by `nu_svr`.
    n_iter_ : int
        Pair updates used by the final solve.
    """

    def __init__(self, *, variant: str = "epsilon_svr", kernel: str | None = None,
                 C: float = 1.0, epsilon: float = 0.1, nu: float = 0.5,
                 gamma: float | None = None, degree: int = 3,
                 coef0: float = 1.0, tol: float = 1e-9,
                 max_iter: int = 20000) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if kernel is None:
            kernel = "linear" if variant == "linear" else "rbf"
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}")
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError("C must be finite and positive")
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")
        if not np.isfinite(nu) or not 0.0 < nu <= 1.0:
            raise ValueError("nu must lie in (0, 1]")
        if gamma is not None and (not np.isfinite(gamma) or gamma <= 0.0):
            raise ValueError("gamma must be finite and positive")
        if not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError("degree must be a positive integer")
        if not np.isfinite(coef0):
            raise ValueError("coef0 must be finite")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and positive")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.variant = variant
        self.kernel = kernel
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.nu = float(nu)
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

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

    def _gram(self, X: np.ndarray) -> np.ndarray:
        gamma = 1.0 / X.shape[1] if self.gamma is None else self.gamma
        return _kernel_matrix(X, X, self.kernel, gamma, self.degree, self.coef0)

    # ------------------------------------------------------------------ #
    # the solver
    # ------------------------------------------------------------------ #
    def _solve(self, gram: np.ndarray, targets: np.ndarray, box: float,
               epsilon: float, *, second_order: bool = True) -> tuple:
        """SMO for (3), written in the stacked form.

        The variables are ``beta = (alpha, alpha*)`` with the constraint
        ``sign' beta = 0`` for ``sign = (+1, ..., +1, -1, ..., -1)``. A pair
        step moves ``alpha_i`` and ``alpha_j`` in opposite directions when
        their signs agree and in the same direction when they differ, so the
        sum constraint survives every update.
        """
        rows = targets.size
        beta = np.zeros(2 * rows)
        sign = np.concatenate([np.ones(rows), -np.ones(rows)])
        offset = np.concatenate([epsilon - targets, epsilon + targets])
        delta = np.zeros(rows)
        fitted = np.zeros(rows)
        gradient = np.concatenate([fitted, -fitted]) + offset
        edge = 1e-12 * max(box, 1.0)
        index = np.arange(2 * rows)
        diagonal = np.concatenate([np.diag(gram), np.diag(gram)])
        iterations = 0
        while iterations < self.max_iter:
            iterations += 1
            up = ((beta < box - edge) & (sign > 0)) | ((beta > edge) & (sign < 0))
            low = ((beta < box - edge) & (sign < 0)) | ((beta > edge) & (sign > 0))
            if not (up.any() and low.any()):
                break
            score = -sign * gradient
            i = int(np.argmax(np.where(up, score, -np.inf)))
            head = i % rows
            gap = np.maximum(diagonal[head] + diagonal
                             - 2.0 * gram[head, index % rows], 1e-12)
            violation = score[i] - score
            candidate = low & (index != i) & (violation > self.tol)
            if not candidate.any():
                break
            if second_order:
                j = int(np.argmax(np.where(candidate, violation ** 2 / gap,
                                           -np.inf)))
            else:
                j = int(np.argmax(np.where(candidate, violation, -np.inf)))
            tail = j % rows
            agree = sign[i] == sign[j]
            slope = gap[j]
            if agree:
                step = (gradient[i] - gradient[j]) / slope
                low_step = max(-beta[j], beta[i] - box)
                high_step = min(beta[i], box - beta[j])
            else:
                step = (-gradient[i] - gradient[j]) / slope
                low_step = max(-beta[i], -beta[j])
                high_step = min(box - beta[i], box - beta[j])
            step = float(np.clip(step, low_step, high_step))
            if abs(step) < 1e-15:
                break
            if agree:
                beta[i] -= step
                beta[j] += step
                left, right = -step * sign[i], step * sign[j]
            else:
                beta[i] += step
                beta[j] += step
                left, right = step * sign[i], step * sign[j]
            delta[head] += left
            delta[tail] += right
            change = left * gram[:, head] + right * gram[:, tail]
            fitted += change
            gradient[:rows] += change
            gradient[rows:] -= change
        fitted = gram @ delta
        alpha, alpha_star = beta[:rows], beta[rows:]
        intercept = self._intercept(fitted, targets, alpha, alpha_star, box,
                                    epsilon)
        return delta, intercept, iterations

    @staticmethod
    def _intercept(fitted: np.ndarray, targets: np.ndarray, alpha: np.ndarray,
                   alpha_star: np.ndarray, box: float, epsilon: float) -> float:
        """The b that makes the free support vectors sit exactly on the tube."""
        residual = targets - fitted
        free_up = (alpha > 1e-8) & (alpha < box - 1e-8) & (alpha_star <= 1e-8)
        free_down = ((alpha_star > 1e-8) & (alpha_star < box - 1e-8)
                     & (alpha <= 1e-8))
        values = []
        if free_up.any():
            values.append(float(np.mean(residual[free_up] - epsilon)))
        if free_down.any():
            values.append(float(np.mean(residual[free_down] + epsilon)))
        if values:
            return float(np.mean(values))
        delta = alpha - alpha_star
        top = delta >= box - 1e-8
        bottom = delta <= -box + 1e-8
        middle = ~top & ~bottom
        lower = np.concatenate([residual[bottom] + epsilon,
                                residual[middle] - epsilon])
        upper = np.concatenate([residual[top] - epsilon,
                                residual[middle] + epsilon])
        if lower.size and upper.size:
            return 0.5 * float(lower.max() + upper.min())
        if lower.size:
            return float(lower.max())
        if upper.size:
            return float(upper.min())
        return float(np.mean(targets))

    def _fit_nu(self, gram: np.ndarray, targets: np.ndarray) -> tuple:
        """Bisect epsilon until the support mass equals ``C nu``.

        `sum |delta|` is non-increasing in epsilon: a wider tube explains more
        points for free, so fewer coefficients leave zero. The box shrinks to
        ``C / n`` so that the mass, not any single coefficient, carries the
        budget.
        """
        rows = targets.size
        box = self.C / rows
        target = self.C * self.nu
        low, high = 0.0, float(np.ptp(targets)) + 1.0
        zero = self._solve(gram, targets, box, 0.0)
        if float(np.abs(zero[0]).sum()) <= target:
            self.epsilon_ = 0.0
            return zero
        state = zero
        for _ in range(60):
            middle = 0.5 * (low + high)
            state = self._solve(gram, targets, box, middle)
            mass = float(np.abs(state[0]).sum())
            if mass > target:
                low = middle
            else:
                high = middle
            if high - low <= 1e-12:
                break
        self.epsilon_ = high
        return self._solve(gram, targets, box, high)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def fit(self, X, y) -> SVR:
        """Solve the dual and store the kernel expansion."""
        points = self._check_X(X)
        targets = self._check_y(y, points.shape[0])
        if points.shape[0] < 2:
            raise ValueError("at least two training points are required")
        self.X_ = points
        self.gamma_ = 1.0 / points.shape[1] if self.gamma is None else self.gamma
        self.y_ = targets
        gram = self._gram(points)
        if self.variant == "nu_svr":
            delta, intercept, iterations = self._fit_nu(gram, targets)
        else:
            delta, intercept, iterations = self._solve(
                gram, targets, self.C, self.epsilon,
                second_order=self.variant != "smo")
            self.epsilon_ = self.epsilon
        self.gram_ = gram
        self.delta_ = delta
        self.b_ = intercept
        self.n_iter_ = iterations
        self.support_ = np.flatnonzero(np.abs(delta) > 1e-8)
        return self

    def decision_function(self, X) -> np.ndarray:
        """The kernel expansion, without the intercept."""
        if not hasattr(self, "delta_"):
            raise RuntimeError("call fit before predicting")
        points = self._check_X(X)
        if points.shape[1] != self.X_.shape[1]:
            raise ValueError("X has the wrong number of features")
        gram = _kernel_matrix(points, self.X_, self.kernel, self.gamma_,
                              self.degree, self.coef0)
        return gram @ self.delta_

    def predict(self, X) -> np.ndarray:
        """Predicted targets."""
        return self.decision_function(X) + self.b_

    def residuals(self, X, y) -> np.ndarray:
        """Signed distance from each target to the prediction."""
        targets = self._check_y(y, self._check_X(X).shape[0])
        return targets - self.predict(X)

    def n_support(self) -> int:
        """Number of points with a nonzero dual coefficient."""
        if not hasattr(self, "delta_"):
            raise RuntimeError("call fit before asking")
        return int(self.support_.size)

    def n_margin_errors(self, X, y) -> int:
        """Points strictly outside the tube."""
        return int(np.sum(np.abs(self.residuals(X, y)) > self.epsilon_ + 1e-8))

    def primal_objective(self, X, y) -> float:
        """``1/2 w'w + C sum(xi + xi*)`` at the stored solution."""
        targets = self._check_y(y, self._check_X(X).shape[0])
        slack = np.maximum(np.abs(targets - self.predict(X)) - self.epsilon_, 0.0)
        quadratic = 0.5 * float(self.delta_ @ (self.gram_ @ self.delta_))
        return quadratic + self.C * float(slack.sum())

    def dual_objective(self) -> float:
        """``-1/2 delta'K delta + y'delta - epsilon |delta|_1``."""
        if not hasattr(self, "delta_"):
            raise RuntimeError("call fit before asking")
        quadratic = -0.5 * float(self.delta_ @ (self.gram_ @ self.delta_))
        linear = float(self.y_ @ self.delta_)
        tube = self.epsilon_ * float(np.abs(self.delta_).sum())
        return quadratic + linear - tube

    def duality_gap(self, X, y) -> float:
        """Primal objective minus dual objective; zero at the optimum."""
        if self.variant == "nu_svr":
            raise ValueError("the gap here is the epsilon-SVR gap")
        return self.primal_objective(X, y) - self.dual_objective()

    def kkt_violation(self, X, y) -> float:
        """Largest violation of the epsilon-SVR KKT conditions."""
        points = self._check_X(X)
        targets = self._check_y(y, points.shape[0])
        residual = targets - self.decision_function(points)
        epsilon = self.epsilon_
        scale = self.C / points.shape[0] if self.variant == "nu_svr" else self.C
        worst = 0.0
        for value, error in zip(self.delta_, residual, strict=True):
            free = 1e-8 < abs(value) < scale - 1e-8
            if free:
                worst = max(worst, abs(error - (self.b_ + np.sign(value) * epsilon)))
            elif abs(value) <= 1e-8:
                worst = max(worst, max(0.0, abs(error - self.b_) - epsilon))
            elif value >= scale - 1e-8:
                worst = max(worst, max(0.0, (self.b_ + epsilon) - error))
            else:
                worst = max(worst, max(0.0, error - (self.b_ - epsilon)))
        return float(worst)


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _demo_tube() -> None:
    """A noiseless line, then noise and a tube that grows."""
    rng = np.random.default_rng(0)
    count = 40
    X = np.sort(rng.uniform(-2.0, 2.0, count))[:, None]
    line = 2.0 * X[:, 0] + 1.0
    noisy = line + rng.normal(0.0, 0.3, count)
    model = SVR(variant="linear", C=10.0, epsilon=0.0).fit(X, line)
    error = float(np.abs(model.predict(X) - line).max())
    print("a noiseless line through 40 points, epsilon = 0:")
    print(f"  largest error   {error:.1e}")
    print(f"  KKT residual    {model.kkt_violation(X, line):.1e}")
    print(f"  duality gap     {abs(model.duality_gap(X, line)):.1e}")
    print(f"  support vectors {model.n_support()} of {count}, "
          f"weights {np.round(model.delta_[model.support_], 4)}")
    print("  two points pin the line; the other 38 weights are exactly zero")
    assert error < 1e-9
    assert model.kkt_violation(X, line) < 1e-10
    assert abs(model.duality_gap(X, line)) < 1e-10
    assert model.n_support() == 2

    print("the same line with noise 0.3, and a tube of increasing half-width:")
    print("  epsilon   support   kkt        gap")
    counts = []
    for epsilon in (0.0, 0.1, 0.2, 0.3, 0.6):
        model = SVR(variant="epsilon_svr", kernel="linear", C=1.0,
                    epsilon=epsilon).fit(X, noisy)
        counts.append(model.n_support())
        print(f"  {epsilon:>7.1f}   {model.n_support():>7}   "
              f"{model.kkt_violation(X, noisy):.1e}   "
              f"{abs(model.duality_gap(X, noisy)):.1e}")
        assert model.kkt_violation(X, noisy) < 1e-9
        assert abs(model.duality_gap(X, noisy)) < 1e-9
    print(f"  inside a wide tube a point costs nothing, so {counts[0]} weights")
    print(f"  fall to {counts[-1]}: the tube is what buys the sparsity")
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def _demo_kernels() -> None:
    """A linear kernel cannot bend; the RBF one can."""
    rng = np.random.default_rng(1)
    count = 80
    X = np.sort(rng.uniform(-3.0, 3.0, count))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(0.0, 0.05, count)
    order = rng.permutation(count)
    train = np.sort(order[:50])
    test = np.sort(order[50:])
    models = {
        "linear": SVR(variant="linear", C=1.0, epsilon=0.1),
        "polynomial": SVR(variant="epsilon_svr", kernel="polynomial",
                          degree=3, gamma=0.05, coef0=1.0, C=10.0,
                          epsilon=0.1),
        "rbf": SVR(variant="rbf", C=10.0, epsilon=0.1, gamma=1.0),
    }
    print("a sine wave plus noise 0.05, 50 points fitted, 30 held out:")
    print("  kernel       train rmse   test rmse   support")
    scores = {}
    for name, model in models.items():
        model.fit(X[train], y[train])
        fit = model.predict(X[train])
        held = model.predict(X[test])
        scores[name] = (
            float(np.sqrt(np.mean((fit - y[train]) ** 2))),
            float(np.sqrt(np.mean((held - y[test]) ** 2))),
        )
        assert model.kkt_violation(X[train], y[train]) < 1e-8
        print(f"  {name:<10}   {scores[name][0]:>10.4f}   "
              f"{scores[name][1]:>9.4f}   {model.n_support():>7}")
    print("  the straight line is the best it can do and still misses the")
    print("  curve; the RBF kernel puts a bump on every support vector")
    assert scores["rbf"][1] < 0.1
    assert scores["linear"][1] > 2.0 * scores["rbf"][1]


def _demo_nu() -> None:
    """nu trades the tube width for a fraction of support vectors."""
    rng = np.random.default_rng(2)
    count = 40
    X = np.sort(rng.uniform(-2.0, 2.0, count))[:, None]
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0.0, 0.3, count)
    print("nu is a budget on the fraction of points outside the tube")
    print("(C = 1, so the box is C/n = 0.025):")
    print("  nu    epsilon   support   margin errors")
    widths = []
    for nu in (0.1, 0.2, 0.3, 0.5, 0.8):
        model = SVR(variant="nu_svr", kernel="linear", C=1.0, nu=nu).fit(X, y)
        errors = model.n_margin_errors(X, y)
        widths.append(model.epsilon_)
        assert model.n_support() >= nu * count - 1e-9
        assert errors <= nu * count + 1e-9
        print(f"  {nu:>3.1f}   {model.epsilon_:>8.4f}   "
              f"{model.n_support():>7}   {errors:>13}")
    print("  the tube narrows as nu grows and the support set grows with it.")
    print("  both counts stay inside the budget; at nu = 0.3 ten of the twelve")
    print("  support vectors are margin errors, so the bound has room in it.")
    assert widths == sorted(widths, reverse=True)


def _demo_solver() -> None:
    """Two working-set rules, one optimum."""
    rng = np.random.default_rng(3)
    count = 40
    X = np.sort(rng.uniform(-2.0, 2.0, count))[:, None]
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0.0, 0.3, count)
    print("the same dual, solved by two working-set rules:")
    print("  epsilon   second order   largest violation")
    for epsilon in (0.0, 0.2, 0.6):
        modern = SVR(variant="epsilon_svr", kernel="linear",
                     epsilon=epsilon).fit(X, y)
        classic = SVR(variant="smo", kernel="linear",
                      epsilon=epsilon).fit(X, y)
        print(f"  {epsilon:>7.1f}   {modern.n_iter_:>12}   "
              f"{classic.n_iter_:>17}")
        assert np.allclose(modern.delta_, classic.delta_, atol=1e-9)
        assert abs(modern.b_ - classic.b_) < 1e-9
    print("  the second-order rule divides each violation by the curvature of")
    print("  the pair, so it spends its steps where they buy the most. It is")
    print("  not uniformly faster: at epsilon = 0 the classic rule wins here.")
    print("  Both rules stop on the same KKT tolerance, so they agree on the")
    print("  answer and differ only in how many updates they needed to get it.")


def _demo_guards() -> None:
    """Malformed settings and inputs, all refused."""
    X = np.linspace(-1.0, 1.0, 6)[:, None]
    y = X[:, 0] ** 2
    model = SVR(variant="linear").fit(X, y)
    attempts = [
        lambda: SVR(variant="nope"),
        lambda: SVR(kernel="cosine"),
        lambda: SVR(C=0.0),
        lambda: SVR(C=np.inf),
        lambda: SVR(epsilon=-0.1),
        lambda: SVR(epsilon=np.nan),
        lambda: SVR(nu=0.0),
        lambda: SVR(nu=1.5),
        lambda: SVR(gamma=0.0),
        lambda: SVR(degree=0),
        lambda: SVR(tol=0.0),
        lambda: SVR(max_iter=0),
        lambda: model.fit(X[:1], y[:1]),
        lambda: model.fit(X, y[:-1]),
        lambda: model.fit(X, [np.nan] * 6),
        lambda: model.predict([[1.0, 2.0]]),
        lambda: SVR(variant="nu_svr").fit(X, y).duality_gap(X, y),
        lambda: SVR().kkt_violation(X, y),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant, an unknown kernel, a zero or infinite C, a negative or")
    print("  non-finite tube, a nu outside (0, 1], a zero gamma, degree, tol or")
    print("  max_iter, a single training point, mismatched or non-finite")
    print("  targets, the wrong number of features, the epsilon gap asked of a")
    print("  nu model, and a check asked of a model that was never fitted.")
    assert refused == len(attempts)


def main() -> None:
    _demo_tube()
    _demo_kernels()
    _demo_nu()
    _demo_solver()
    _demo_guards()
    print("svr demo passed")


if __name__ == "__main__":
    main()
