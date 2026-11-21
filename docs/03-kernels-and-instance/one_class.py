"""One-class models — a boundary around the normal, learned from the normal.

Chapter:   03-kernels-and-instance
Companion: one_class.md
Tier 2 · 4 variants: ocsvm, svdd, kernel_ocsvm, nu_one_class

    min ½||w||^2 + C sum(xi_i) - rho   s.t.  w^T phi(x_i) >= rho - xi_i,  xi_i >= 0

Novelty detection has no negative class, so there is nothing to separate the data
from except the origin. The one-class SVM does exactly that: it looks for the
hyperplane in feature space with the largest margin from the training points,
with the origin on the other side. Everything the margin cannot cover pays a
slack, and the dual turns the problem into weights on the training points:

    min ½ alpha^T K alpha    s.t.  sum(alpha_i) = 1,  0 <= alpha_i <= C

so the model is again a subset of the data — the support vectors — and the
decision function is sum(alpha_i K(x_i, x)) - rho.

    ocsvm          the linear kernel, C is the box
    kernel_ocsvm   the RBF kernel, C is the box
    svdd           the smallest enclosing ball instead of the widest margin
    nu_one_class   the box is 1/(nu n), so nu bounds the outlier fraction

The four variants are two ideas and two parameterisations. The two ideas are
separating the origin and enclosing the points in a sphere; with the RBF kernel
they produce the *same* alpha, because the kernel diagonal is constant and the
sphere's radius term becomes a constant in the dual. With a linear kernel they
differ, and the demo shows it. The two parameterisations are C and nu: C is a box
bound with no units, while nu in (0, 1] is the fraction of points the model
allows itself to reject, and the demo checks the two inequalities that make it
interpretable — support vectors at least nu n, outliers at most nu n.

Numbers to carry away (measured, see the demo):

  - a linear model on 60 points keeps 54 weights at zero, 2 free and 4 at the
    box, rejects 4 points, and satisfies the KKT conditions to 6.7e-16
  - an RBF model with 11 support vectors keeps 92.0 percent of held-out
    inliers and rejects 100 percent of a shell at radius 1.2
  - the same boundary with gamma = 8 keeps 72.0 percent: a narrow kernel
    follows the sample rather than the distribution
  - SVDD and the one-class SVM differ with a linear kernel and are the same
    problem with an RBF one, where the decision functions differ by a factor
    of exactly 2 (measured to 1.4e-09)
  - at nu = 0.5 on 120 points: 62 support vectors and 59 outliers, against
    bounds of 60 and 60

"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("ocsvm", "svdd", "kernel_ocsvm", "nu_one_class")


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
    raise ValueError(f"unknown kernel {kind!r}")


class OneClass:
    """A boundary learned from normal data alone.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    kernel : str or None
        `linear`, `rbf` or `polynomial`; `None` picks the variant's default,
        which is `linear` for `ocsvm` and `svdd` and `rbf` otherwise.
    C : float
        Box bound on the dual weights. The problem is infeasible when
        ``C * n < 1``, because the weights have to sum to one.
    nu : float
        Allowed outlier fraction for `nu_one_class`; the box is ``1/(nu n)``.
    gamma, degree, coef0 : float, int, float
        Kernel parameters. `gamma` defaults to ``1 / n_features``.
    tol : float
        KKT tolerance on the pair violation.
    max_iter : int
        Hard cap on pair updates.

    Attributes
    ----------
    alpha_ : ndarray
        Dual weights, shape ``(n,)``.
    rho_ : float
        Offset of the decision function.
    support_ : ndarray
        Indices with a non-zero weight.
    free_ : ndarray
        Indices strictly inside the box; they satisfy ``f(x_i) = 0`` exactly.
    n_iter_ : int
        Pair updates performed.
    w_ : ndarray or None
        The expansion vector ``X^T alpha`` for a linear kernel: the normal of
        the separating hyperplane, or the centre of the ball for `svdd`.
    """

    def __init__(self, *, variant: str = "ocsvm", kernel: str | None = None,
                 C: float = 1.0, nu: float = 0.5, gamma: float | None = None,
                 degree: int = 3, coef0: float = 1.0, tol: float = 1e-9,
                 max_iter: int = 20000) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if kernel is None:
            kernel = "linear" if variant in ("ocsvm", "svdd") else "rbf"
        if kernel not in ("linear", "rbf", "polynomial"):
            raise ValueError(f"unknown kernel {kernel!r}")
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError("C must be finite and positive")
        if not 0.0 < nu <= 1.0:
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

    def _gram(self, points: np.ndarray) -> np.ndarray:
        return _kernel_matrix(points, points, self.kernel, self.gamma_,
                              self.degree, self.coef0)

    # ------------------------------------------------------------------ #
    # the dual solver
    # ------------------------------------------------------------------ #
    def _solve(self, hessian: np.ndarray, linear: np.ndarray,
               box: float) -> tuple:
        """SMO on ``min ½ a'Ha + c'a`` with ``sum(a) = 1`` and ``0 ≤ a ≤ box``."""
        rows = hessian.shape[0]
        alpha = np.full(rows, 1.0 / rows)
        gradient = hessian @ alpha + linear
        iterations = 0
        while iterations < self.max_iter:
            up = alpha < box - 1e-12 * box
            down = alpha > 1e-12 * box
            if not up.any() or not down.any():
                break
            i = int(np.argmax(np.where(down, gradient, -np.inf)))
            j = int(np.argmin(np.where(up, gradient, np.inf)))
            violation = gradient[i] - gradient[j]
            if violation <= self.tol:
                break
            curvature = (hessian[i, i] + hessian[j, j]
                         - 2.0 * hessian[i, j])
            if curvature <= 0.0:
                curvature = 1e-12
            step = violation / curvature
            low = max(-alpha[j], alpha[i] - box)
            high = min(box - alpha[j], alpha[i])
            step = float(np.clip(step, low, high))
            if step == 0.0:
                break
            alpha[i] -= step
            alpha[j] += step
            gradient += step * (hessian[:, j] - hessian[:, i])
            iterations += 1
        return alpha, iterations

    def _offset(self, alpha: np.ndarray, gradient: np.ndarray,
                box: float) -> float:
        """rho from the free support vectors, or from the KKT interval."""
        scale = 1e-9 * max(box, 1.0)
        free = (alpha > scale) & (alpha < box - scale)
        if free.any():
            return float(gradient[free].mean())
        at_zero = alpha <= scale
        at_box = alpha >= box - scale
        lower = float(gradient[at_box].max()) if at_box.any() else -np.inf
        upper = float(gradient[at_zero].min()) if at_zero.any() else np.inf
        if not np.isfinite(lower):
            return upper
        if not np.isfinite(upper):
            return lower
        return 0.5 * (lower + upper)

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def fit(self, X) -> OneClass:
        """Solve the dual and store the kernel expansion."""
        points = self._check_X(X)
        rows = points.shape[0]
        if rows < 2:
            raise ValueError("at least two training points are required")
        self.gamma_ = 1.0 / points.shape[1] if self.gamma is None else self.gamma
        box = 1.0 / (self.nu * rows) if self.variant == "nu_one_class" else self.C
        if box * rows < 1.0:
            raise ValueError("C is too small: C * n must be at least 1")
        self.X_ = points
        self.box_ = box
        gram = self._gram(points)
        if self.variant == "svdd":
            hessian = 2.0 * gram
            linear = -np.diag(gram)
        else:
            hessian = gram
            linear = np.zeros(rows)
        alpha, iterations = self._solve(hessian, linear, box)
        gradient = hessian @ alpha + linear
        self.alpha_ = alpha
        self.n_iter_ = iterations
        self.rho_ = self._offset(alpha, gradient, box)
        scale = 1e-9 * max(box, 1.0)
        self.support_ = np.flatnonzero(alpha > scale)
        self.free_ = np.flatnonzero((alpha > scale) & (alpha < box - scale))
        self.w_ = points.T @ alpha if self.kernel == "linear" else None
        return self

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def decision_function(self, X) -> np.ndarray:
        """Positive inside the boundary, negative outside."""
        if not hasattr(self, "alpha_"):
            raise RuntimeError("call fit before deciding")
        points = self._check_X(X)
        if points.shape[1] != self.X_.shape[1]:
            raise ValueError("X has the wrong number of features")
        kernel = _kernel_matrix(points, self.X_, self.kernel, self.gamma_,
                                self.degree, self.coef0)
        if self.variant == "svdd":
            diagonal = np.diag(_kernel_matrix(points, points, self.kernel,
                                              self.gamma_, self.degree,
                                              self.coef0))
            return 2.0 * (kernel @ self.alpha_) - diagonal - self.rho_
        return kernel @ self.alpha_ - self.rho_

    def predict(self, X) -> np.ndarray:
        """``+1`` for normal, ``-1`` for novel."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def n_support(self) -> int:
        """Non-zero dual weights."""
        if not hasattr(self, "alpha_"):
            raise RuntimeError("call fit before counting")
        return int(self.support_.size)

    def n_outliers(self, X) -> int:
        """Points the boundary rejects, with the boundary treated as a slab."""
        decision = self.decision_function(X)
        tolerance = 1e-9 * max(1.0, float(np.abs(decision).max()))
        return int(np.count_nonzero(decision < -tolerance))

    def primal_norm(self) -> float:
        """``||w||``, equal to ``sqrt(alpha'K alpha)`` in every kernel."""
        if not hasattr(self, "alpha_"):
            raise RuntimeError("call fit before asking")
        return float(np.sqrt(self.alpha_ @ self._gram(self.X_) @ self.alpha_))

    def radius(self) -> float:
        """Radius of the enclosing ball implied by the weights."""
        if not hasattr(self, "alpha_"):
            raise RuntimeError("call fit before asking")
        gram = self._gram(self.X_)
        centres = gram @ self.alpha_
        squared = np.diag(gram) - 2.0 * centres + self.alpha_ @ centres
        chosen = self.free_ if self.free_.size else self.support_
        return float(np.sqrt(max(np.mean(squared[chosen]), 0.0)))

    def kkt_violation(self) -> float:
        """Largest violation of the pair conditions, on the training set."""
        if not hasattr(self, "alpha_"):
            raise RuntimeError("call fit before checking")
        gram = self._gram(self.X_)
        if self.variant == "svdd":
            gradient = 2.0 * (gram @ self.alpha_) - np.diag(gram)
        else:
            gradient = gram @ self.alpha_
        box = self.box_
        scale = 1e-9 * max(box, 1.0)
        up = self.alpha_ < box - scale
        down = self.alpha_ > scale
        if not up.any() or not down.any():
            return 0.0
        return float(gradient[down].max() - gradient[up].min())


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _shell(rng: np.random.Generator, count: int, radius: float,
           spread: float) -> np.ndarray:
    """Points on a noisy circle of the given radius."""
    direction = rng.normal(size=(count, 2))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * (radius + rng.normal(0.0, spread, count))[:, None]


def _demo_linear() -> None:
    """A blob away from the origin, and the widest margin from it."""
    rng = np.random.default_rng(0)
    points = rng.normal(0.0, 0.6, size=(60, 2)) + np.array([1.6, 1.2])
    model = OneClass(variant="ocsvm", C=0.2).fit(points)
    decision = model.decision_function(points)
    direction = model.w_ / np.linalg.norm(model.w_)
    print("60 points in a blob at (1.6, 1.2), linear kernel, C = 0.2:")
    print(f"  normal direction    ({direction[0]:.3f}, {direction[1]:.3f})")
    print(f"  ||w||               {model.primal_norm():.4f}")
    print(f"  rho                 {model.rho_:.4f}")
    print(f"  margin rho / ||w||   {model.rho_ / model.primal_norm():.4f}")
    print("  weights: "
          f"{int((model.alpha_ < 1e-12).sum())} at zero, "
          f"{model.free_.size} free, "
          f"{int((model.alpha_ > 0.2 - 1e-12).sum())} at the box")
    print(f"  points outside      {model.n_outliers(points)} of 60, "
          f"bounded by min(n, 1/C) = 5")
    print(f"  KKT residual        {model.kkt_violation():.1e}")
    print(f"  pair updates        {model.n_iter_}")
    print("  the boundary is rho - w.x = 0, and the origin satisfies it with")
    print("  room to spare, which is what 'one class' means")
    assert abs(model.alpha_.sum() - 1.0) < 1e-12
    assert model.alpha_.max() <= 0.2 + 1e-12
    assert np.abs(decision[model.free_]).max() < 1e-8
    assert model.kkt_violation() < 1e-8
    assert np.abs(decision - (points @ model.w_ - model.rho_)).max() < 1e-12
    assert float(model.decision_function([[0.0, 0.0]])[0]) < -0.5
    assert model.n_support() < points.shape[0]
    print("  the kernel expansion and the primal hyperplane agree to rounding:")
    print(f"    max |f(x) - (x.w - rho)| = "
          f"{np.abs(decision - (points @ model.w_ - model.rho_)).max():.1e}")


def _demo_kernel() -> None:
    """A ball of normal points, and everything outside it is novel."""
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 0.35, size=(120, 2))
    held = rng.normal(0.0, 0.35, size=(200, 2))
    outer = _shell(rng, 200, 1.2, 0.1)
    model = OneClass(variant="kernel_ocsvm", C=0.1, gamma=0.5).fit(train)
    print("120 points in a ball of radius 0.35, RBF kernel, C = 0.1:")
    print(f"  support vectors     {model.n_support()} of 120 "
          f"({model.free_.size} free)")
    print(f"  training outliers   {model.n_outliers(train)}")
    print(f"  held-out inliers    {np.mean(model.predict(held) > 0):.3f} kept")
    print(f"  a shell at radius 1.2  {np.mean(model.predict(outer) < 0):.3f} "
          "rejected")
    print("  no negative examples were used: the shell is novel because the")
    print("  training density does not reach it")
    assert model.n_outliers(train) <= 0.1 * 120 + 1
    assert np.mean(model.predict(held) > 0) > 0.85
    assert np.mean(model.predict(outer) < 0) > 0.95

    print("the width decides how tight the boundary is (held-out inliers kept):")
    print("  gamma   support   inliers kept   shell rejected")
    kept = []
    for gamma in (0.1, 0.5, 2.0, 8.0):
        model = OneClass(variant="kernel_ocsvm", C=0.1, gamma=gamma).fit(train)
        score = float(np.mean(model.predict(held) > 0))
        rejected = float(np.mean(model.predict(outer) < 0))
        kept.append(score)
        assert model.kkt_violation() < 1e-8
        print(f"  {gamma:>5.1f}   {model.n_support():>7}   {score:>12.3f}   "
              f"{rejected:>14.3f}")
    print("  a narrow kernel puts a bump on every training point, so the")
    print("  boundary follows the sample instead of the distribution")
    assert kept[0] > kept[-1]


def _demo_svdd() -> None:
    """A sphere and a hyperplane: different with linear, identical with RBF."""
    rng = np.random.default_rng(2)
    points = rng.normal(0.0, 0.6, size=(60, 2)) + np.array([1.6, 1.2])
    sphere = OneClass(variant="svdd", C=0.2).fit(points)
    plane = OneClass(variant="ocsvm", C=0.2).fit(points)
    print("the same 60 points, a sphere and a hyperplane:")
    print("              centre or normal          rho     radius")
    print(f"  ocsvm       ({plane.w_[0]:>6.3f}, {plane.w_[1]:>6.3f})   "
          f"{plane.rho_:>8.4f}   {plane.radius():>6.4f}")
    print(f"  svdd        ({sphere.w_[0]:>6.3f}, {sphere.w_[1]:>6.3f})   "
          f"{sphere.rho_:>8.4f}   {sphere.radius():>6.4f}")
    print(f"  their weights agree: {np.allclose(sphere.alpha_, plane.alpha_)}")
    print("  with the linear kernel the sphere's centre is pulled towards the")
    print("  blob and the plane's normal is sized by the margin, so they differ")
    assert not np.allclose(sphere.alpha_, plane.alpha_)
    assert np.abs(sphere.w_ - points.mean(axis=0)).max() < 0.5

    print("with the RBF kernel the two problems coincide:")
    ring = _shell(rng, 150, 1.2, 0.1)
    sphere = OneClass(variant="svdd", kernel="rbf", C=0.1, gamma=1.0)
    plane = OneClass(variant="kernel_ocsvm", C=0.1, gamma=1.0)
    sphere.fit(points)
    plane.fit(points)
    gap = float(np.abs(sphere.decision_function(ring)
                        - 2.0 * plane.decision_function(ring)).max())
    print(f"  same weights            {np.allclose(sphere.alpha_, plane.alpha_)}")
    print(f"  same boundary, factor 2 {gap:.1e}")
    print(f"  rho: {sphere.rho_:.4f} against 2 * {plane.rho_:.4f} - 1 = "
          f"{2.0 * plane.rho_ - 1.0:.4f}")
    print("  the kernel diagonal is one everywhere, so the radius term is a")
    print("  constant in the dual and only the residual problem is left")
    assert np.allclose(sphere.alpha_, plane.alpha_)
    assert abs(sphere.rho_ - (2.0 * plane.rho_ - 1.0)) < 1e-7
    assert gap < 1e-7


def _demo_nu() -> None:
    """nu is a fraction, and the two bounds are what make it one."""
    rng = np.random.default_rng(3)
    train = rng.normal(0.0, 0.35, size=(120, 2)) + np.array([0.1, -0.2])
    print("120 points, RBF kernel, the box is 1/(nu n):")
    print("  nu     box     support   outliers   nu*n")
    for nu in (0.05, 0.1, 0.2, 0.5, 1.0):
        model = OneClass(variant="nu_one_class", nu=nu, gamma=2.0).fit(train)
        outliers = model.n_outliers(train)
        print(f"  {nu:>4.2f}   {model.box_:.4f}   {model.n_support():>7}   "
              f"{outliers:>8}   {nu * 120:>4.1f}")
        assert model.n_support() >= nu * 120 - 1e-9
        assert outliers <= nu * 120 + 1e-9
        assert model.kkt_violation() < 1e-8
    print("  both bounds come from the same two facts: the weights sum to")
    print("  one and each is at most 1/(nu n). At least nu n points must")
    print("  carry weight, and only a point at the box can be rejected, so")
    print("  the outliers are at most nu n as well. The bound is a count,")
    print("  which is why nu suits a problem with a known false-alarm cost")


def _demo_guards() -> None:
    """Malformed settings and inputs, all refused."""
    X = np.linspace(-1.0, 1.0, 12).reshape(6, 2)
    model = OneClass().fit(X)
    attempts = [
        lambda: OneClass(variant="nope"),
        lambda: OneClass(kernel="cosine"),
        lambda: OneClass(C=0.0),
        lambda: OneClass(C=np.inf),
        lambda: OneClass(nu=0.0),
        lambda: OneClass(nu=1.5),
        lambda: OneClass(gamma=0.0),
        lambda: OneClass(degree=0),
        lambda: OneClass(coef0=np.nan),
        lambda: OneClass(tol=0.0),
        lambda: OneClass(max_iter=0),
        lambda: OneClass(C=0.01).fit(X),
        lambda: OneClass().fit(X[:1]),
        lambda: OneClass().fit([[np.nan, 1.0]]),
        lambda: OneClass().fit([[[1.0]]]),
        lambda: model.predict([[1.0, 2.0, 3.0]]),
        lambda: OneClass().predict(X),
        lambda: OneClass().n_support(),
        lambda: OneClass().radius(),
        lambda: OneClass().kkt_violation(),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant or kernel, a zero or infinite C, a nu outside (0, 1], a")
    print("  zero gamma, degree or tol, a non-finite coef0, zero iterations,")
    print("  a box too small to hold the unit weight, a single point, a")
    print("  non-finite input, a three-dimensional design, the wrong feature")
    print("  count, and predict, the support count, the radius or the KKT")
    print("  check before fit.")
    assert refused == len(attempts)


def main() -> None:
    _demo_linear()
    _demo_kernel()
    _demo_svdd()
    _demo_nu()
    _demo_guards()
    print("one_class demo passed")


if __name__ == "__main__":
    main()
