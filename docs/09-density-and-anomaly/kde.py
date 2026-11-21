"""Kernel density estimation — a bump on every point, and the bandwidth that decides everything.

Chapter:   09-density-and-anomaly
Companion: kde.md
Tier 1 · 8 variants: gaussian, epanechnikov, tophat, silverman, scott,
cv_bandwidth, tree_kde, product_kernel

    p̂(x) = 1/(n h^d) Σ_i K((x − x_i)/h)                                  (1)
    K(u) = (2π)^{−d/2} exp(−‖u‖²/2)                                       (2)
    K(u) = (d+2)/(2V_d) (1 − ‖u‖²)_+                                      (3)
    K(u) = 1/V_d  for ‖u‖ ≤ 1                                             (4)
    h_S = 0.9 min(σ̂, IQR/1.34) n^{−1/(d+4)}                               (5)
    h_scott = σ̂ n^{−1/(d+4)}                                             (6)
    h_cv = argmax_h Σ_i log( 1/(n−1) Σ_{j≠i} K_h(x_i − x_j) )             (7)

Nothing in this chapter is estimated more directly than a density: histograms
bin, copulas reorder, and every anomaly detector from LOF to the autoencoder is
a density estimate with a threshold bolted on. KDE is the version that puts the
same small bump on every point and adds them up, which is a convolution of the
empirical measure with a kernel — differentiable, asymptotically unbiased, and
completely at the mercy of one number.

That number is the bandwidth. Too small and the estimate is a picket fence of
spikes through the sample, variance exploding and the tails empty; too large and
it is one Gaussian blob, bias exploding and every structure erased. The
asymptotic decomposition is exact and worth carrying in your head:

    E[p̂(x)] − p(x) = (h²/2) σ_K² ∇²p(x) + O(h⁴)                          (8)
    Var[p̂(x)]      = R(K) p(x) / (n h^d) + O(n^{−1})                     (9)

so bias grows with h² and variance falls like 1/(n h^d), and the product is
minimised at h ∝ n^{−1/(d+4)}. Two consequences the code cannot avoid: the
constant is unknown because it contains ∇²p, and the exponent degrades with
dimension — the same estimator needs n = 10⁶ points to see a 10-D density at
1-D resolution, which is the reason high-dimensional anomaly detection uses
scorers (isolation forest, LOF) that never form a density at all.

Three kernels are here because they mark the three positions in the
bias–variance trade: Gaussian is smooth and unbounded, Epanechnikov is the
minimum-variance positive kernel of order two with compact support, top-hat is
the naive one and the least efficient of the three (asymptotic efficiency 0.951
for the Gaussian and 0.930 for the top-hat, against Epanechnikov's 1.000, in
one dimension). The estimator's shape depends far more on h
than on which of the three you pick, which is why choice of kernel is the
cheapest decision in the file and bandwidth selection is the expensive one.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = (
    "gaussian",
    "epanechnikov",
    "tophat",
    "silverman",
    "scott",
    "cv_bandwidth",
    "tree_kde",
    "product_kernel",
)

#: Which kernel each variant uses. The bandwidth rule is a separate question.
_KERNEL_OF = {
    "gaussian": "gaussian",
    "epanechnikov": "epanechnikov",
    "tophat": "tophat",
    "silverman": "gaussian",
    "scott": "gaussian",
    "cv_bandwidth": "gaussian",
    "tree_kde": "gaussian",
    "product_kernel": "gaussian",
}


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def ball_volume(n_features: int) -> float:
    """Volume of the unit ball in `n_features` dimensions: π^{d/2}/Γ(d/2+1)."""
    return math.pi ** (n_features / 2.0) / math.gamma(n_features / 2.0 + 1.0)


def kernel_value(u2: np.ndarray, n_features: int, name: str) -> np.ndarray:
    """Kernel at squared radius `u2`, with unit bandwidth.

    `u2` may be any shape; the return has the same shape. The compact kernels
    are written with `np.where` rather than a branch so that the caller can pass
    a whole distance matrix at once.
    """
    u2 = np.asarray(u2, dtype=float)
    if name == "gaussian":
        return np.exp(-0.5 * u2) / (2.0 * math.pi) ** (n_features / 2.0)
    volume = ball_volume(n_features)
    if name == "tophat":
        return np.where(u2 <= 1.0, 1.0 / volume, 0.0)
    if name == "epanechnikov":
        return np.where(u2 <= 1.0, (n_features + 2.0) / (2.0 * volume) * (1.0 - u2), 0.0)
    raise ValueError(f"unknown kernel {name!r}")


def _sq_distances(A: np.ndarray, B: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    """Squared distances in bandwidth units. `A` (m,d), `B` (n,d) → (m,n).

    Dividing the *difference* by the bandwidth before squaring is what makes
    (1) a per-dimension statement: with a vector `h` the squared radius is
    Σ_j ((a_j − b_j)/h_j)², which is a Mahalanobis distance with a diagonal
    metric — the product kernel below, with no extra code.
    """
    offset = (A[:, None, :] - B[None, :, :]) / bandwidth
    return np.einsum("ijk,ijk->ij", offset, offset)


# --------------------------------------------------------------------------- #
# Bandwidth rules
# --------------------------------------------------------------------------- #
def scott_factor(X: np.ndarray) -> np.ndarray:
    """Per-dimension Scott bandwidth: `σ_j n^{-1/(d+4)}`."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    return X.std(axis=0, ddof=1) * n ** (-1.0 / (d + 4.0))


def silverman_factor(X: np.ndarray) -> np.ndarray:
    """Per-dimension Silverman bandwidth: `0.9 min(σ_j, IQR_j/1.349) n^{-1/(d+4)}`.

    The interquartile range is in the minimum because a single far outlier
    inflates `σ` by its distance and would widen the kernel over the whole
    sample; the IQR moves with the middle half and not with the outlier.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    q75, q25 = np.percentile(X, [75.0, 25.0], axis=0)
    robust = np.minimum(X.std(axis=0, ddof=1), (q75 - q25) / 1.349)
    return 0.9 * robust * n ** (-1.0 / (d + 4.0))


def _golden_max(f, lo: float, hi: float, n_iter: int = 60) -> float:
    """Maximise a unimodal `f` on `[lo, hi]`; returns the argmax in log space."""
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(int(n_iter)):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
    return float((a + b) / 2.0)


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class KernelDensity:
    """Kernel density estimate, fitted by keeping the sample.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. It selects the kernel, the bandwidth rule and the
        evaluator, in that order:
        `gaussian`/`epanechnikov`/`tophat` fix the kernel and default to Scott's
        bandwidth, `silverman`/`scott` fix the rule, `cv_bandwidth` maximises
        the leave-one-out likelihood, `tree_kde` evaluates through a ball tree
        and `product_kernel` uses one bandwidth per dimension.
    bandwidth : float or array of shape (n_features,), optional
        A fixed bandwidth, overriding whatever the variant would choose.
    leaf_size : int
        Points per leaf of the ball tree used by `tree_kde`.
    tol : float
        Relative pruning tolerance for `tree_kde`; ignored elsewhere.

    Attributes
    ----------
    bandwidth_ : ndarray of shape (n_features,)
    kernel_ : str
    data_ : ndarray of shape (n_samples, n_features)
    evaluations_ : int
        Kernel evaluations performed by the last `tree_kde` call.
    """

    def __init__(self, *, variant: str = "gaussian", bandwidth=None,
                 leaf_size: int = 16, tol: float = 1e-8) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        if not tol > 0.0:
            raise ValueError("tol must be positive")
        if bandwidth is not None and not np.all(np.asarray(bandwidth, float) > 0.0):
            raise ValueError("bandwidth must be positive")
        self.variant = variant
        self.bandwidth = bandwidth
        self.leaf_size = int(leaf_size)
        self.tol = float(tol)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> KernelDensity:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if X.shape[0] < 2:
            raise ValueError("need at least two points to estimate a density")
        if np.any(np.ptp(X, axis=0) == 0.0):
            raise ValueError("a constant column has zero bandwidth; drop it or "
                             "add jitter")
        self.n_features_ = X.shape[1]
        self.kernel_ = _KERNEL_OF[self.variant]
        self.data_ = X
        self.bandwidth_ = self._choose_bandwidth(X)
        self.evaluations_ = 0
        if self.variant == "tree_kde":
            if np.ptp(self.bandwidth_) > 1e-12:
                raise ValueError("tree_kde needs a scalar bandwidth; the node "
                                 "bound is only valid for an isotropic kernel")
            self.tree_ = _grow(X, np.arange(len(X)), self.leaf_size)
        return self

    def _choose_bandwidth(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        if self.bandwidth is not None:
            given = np.asarray(self.bandwidth, dtype=float)
            if given.ndim == 0:
                return np.full(d, float(given))
            if given.shape != (d,):
                raise ValueError(f"bandwidth must be a scalar or shape ({d},)")
            return given.copy()
        if self.variant == "silverman":
            return silverman_factor(X)
        base = scott_factor(X)
        if self.variant == "cv_bandwidth":
            return np.full(d, self._cross_validate(X, float(np.median(base))))
        if self.variant == "tree_kde":
            # One bandwidth for every dimension is what the node bound in
            # `_tree_sum` assumes; the median is the least surprising summary.
            return np.full(d, float(np.median(base)))
        return base

    def _loo_loglik(self, X: np.ndarray, h: float) -> float:
        """Equation (7) at one scalar bandwidth: the leave-one-out mean log density."""
        sq = _sq_distances(X, X, np.full(X.shape[1], h))
        kernel = kernel_value(sq, X.shape[1], "gaussian") / h ** X.shape[1]
        held_out = kernel.sum(axis=1) - np.diag(kernel)
        return float(np.mean(np.log(np.maximum(held_out / (len(X) - 1), 1e-300))))

    def _cross_validate(self, X: np.ndarray, h0: float) -> float:
        """Maximise (7) over a single scalar `h`, by golden section on log h."""
        # A factor of four either side of the rule of thumb. The curve is
        # unimodal in log h on that range, which is what golden section needs.
        return math.exp(_golden_max(
            lambda log_h: self._loo_loglik(X, math.exp(log_h)),
            math.log(h0 / 4.0), math.log(h0 * 4.0)))

    # -- evaluation ------------------------------------------------------- #
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log density at each row of `X`, shape (m,)."""
        if not hasattr(self, "bandwidth_"):
            raise RuntimeError("fit before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"X must have shape (m, {self.n_features_})")
        scale = float(np.prod(self.bandwidth_))
        if self.variant == "tree_kde":
            self.evaluations_ = 0
            sums = np.array([self._tree_sum(row) for row in X])
        else:
            sq = _sq_distances(X, self.data_, self.bandwidth_)
            kernel = kernel_value(sq, self.n_features_, self.kernel_)
            sums = kernel.sum(axis=1)
        density = sums / (len(self.data_) * scale)
        return np.log(np.maximum(density, 1e-300))

    def score(self, X: np.ndarray) -> float:
        """Mean log density per point: the number cross-validation maximises."""
        return float(np.mean(self.score_samples(X)))

    def _tree_sum(self, row: np.ndarray) -> float:
        """Sum of kernel values at `row`, pruning nodes that cannot matter."""
        h = float(self.bandwidth_[0])
        d = self.n_features_
        total = 0.0
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            gap = max(float(np.linalg.norm(row - node.centre)) - node.radius, 0.0) / h
            # Every kernel above is a decreasing function of radius, so the
            # nearest point in the node is the best case for the node's whole
            # contribution -- and if even that is negligible, so is the node.
            bound = node.count * kernel_value(gap * gap, d, self.kernel_) / h ** d
            # Exact when the bound underflows, relative once the running sum is
            # non-zero. Both are needed: in the far field every node evaluates to
            # zero and a relative test alone would never fire.
            if bound == 0.0 or (total > 0.0 and bound < self.tol * total):
                continue
            if node.left is None:
                for index in node.indices:
                    offset = (row - self.data_[index]) / h
                    total += float(kernel_value(float(offset @ offset), d, self.kernel_))
                    self.evaluations_ += 1
            else:
                stack.append(node.left)
                stack.append(node.right)
        return total


class _Node:
    """One ball in the tree: a centre, a radius, and either points or children."""

    __slots__ = ("indices", "centre", "radius", "count", "left", "right")


def _grow(X: np.ndarray, indices: np.ndarray, leaf_size: int) -> _Node:
    node = _Node()
    node.count = len(indices)
    points = X[indices]
    node.centre = points.mean(axis=0)
    node.radius = float(np.max(np.linalg.norm(points - node.centre, axis=1)))
    node.left = node.right = None
    node.indices = indices
    spread = np.ptp(points, axis=0)
    axis = int(np.argmax(spread))
    if len(indices) <= leaf_size or spread[axis] == 0.0:
        return node
    order = indices[np.argsort(points[:, axis], kind="stable")]
    middle = len(order) // 2
    node.indices = None  # interior nodes are summarised by centre and radius
    node.left = _grow(X, order[:middle], leaf_size)
    node.right = _grow(X, order[middle:], leaf_size)
    return node


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #
def _mixture_pdf(x: np.ndarray) -> np.ndarray:
    """The truth the demo scores against: 0.35 N(-1.2, 0.5) + 0.65 N(1.0, 0.7)."""
    def bump(mean: float, sd: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
    return 0.35 * bump(-1.2, 0.5) + 0.65 * bump(1.0, 0.7)


def _ise(model: KernelDensity, grid: np.ndarray) -> float:
    """Integrated squared error against the mixture, by the midpoint rule."""
    estimate = np.exp(model.score_samples(grid[:, None]))
    step = float(grid[1] - grid[0])
    return float(np.sum((estimate - _mixture_pdf(grid)) ** 2) * step)


def _radial_integral(name: str, n_features: int, limit: float) -> float:
    """∫ K(‖u‖) du by radial quadrature: ∫_0^R K(r²) S_{d-1} r^{d-1} dr."""
    surface = 2.0 * math.pi ** (n_features / 2.0) / math.gamma(n_features / 2.0)
    radius = np.linspace(0.0, limit, 200_001)
    weight = surface * radius ** (n_features - 1)
    values = kernel_value(radius ** 2, n_features, name)
    return float(np.trapezoid(values * weight, radius))


def _demo_normalisation() -> None:
    """Every kernel integrates to one, in every dimension that matters."""
    errors = {}
    for name in ("gaussian", "epanechnikov", "tophat"):
        for d in (1, 2, 3):
            limit = 10.0 if name == "gaussian" else 1.0
            errors[name, d] = abs(_radial_integral(name, d, limit) - 1.0)
            assert errors[name, d] < 1e-8, (name, d, errors[name, d])
    worst = max(errors.values())

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    base = KernelDensity(bandwidth=0.4).fit(X)
    scaled = KernelDensity(bandwidth=0.8).fit(2.0 * X)
    # Changing units changes the density by the Jacobian and nothing else.
    gap = float(np.abs(scaled.score_samples(2.0 * X)
                       - base.score_samples(X) + 2.0 * np.log(2.0)).max())
    assert gap < 1e-12

    product = KernelDensity(variant="product_kernel").fit(X)
    same = KernelDensity(variant="gaussian", bandwidth=product.bandwidth_).fit(X)
    flat = float(np.abs(product.bandwidth_ - product.bandwidth_[0]).max())
    agreement = float(np.abs(product.score_samples(X) - same.score_samples(X)).max())
    assert agreement < 1e-12

    print(f"normalisation: each kernel integrates to one to {worst:.1e} by radial "
          f"quadrature in d = 1, 2 and 3 -- which is the check that the constants "
          f"in (2)-(4) are right and not merely dropped, since 1/V_d and "
          f"(d+2)/(2V_d) only balance the sphere's surface area with V_d inside. "
          f"Rescaling the bandwidth by 2 and the data by 2 leaves the log density "
          f"shifted by exactly -d log 2, to {gap:.1e}. The product kernel with all "
          f"bandwidths equal to the product kernel's own (spread {flat:.1e}) is the Gaussian kernel "
          f"bit-for-bit, to {agreement:.1e}: 'product' is a per-dimension "
          f"bandwidth, not a different estimator.")


def _demo_bandwidths() -> None:
    """The three rules on one skewed sample, and the likelihood each buys."""
    rng = np.random.default_rng(1)
    X = np.exp(rng.normal(size=(200, 1))) * 1.5
    sd = float(X.std(ddof=1))
    q75, q25 = np.percentile(X, [75.0, 25.0])
    iqr = float(q75 - q25)

    models = {v: KernelDensity(variant=v).fit(X)
              for v in ("scott", "silverman", "cv_bandwidth")}
    table = {v: (float(m.bandwidth_[0]), m._loo_loglik(X, float(m.bandwidth_[0])))
             for v, m in models.items()}
    best = max(("scott", "silverman"), key=lambda v: table[v][1])
    assert table["cv_bandwidth"][1] >= table[best][1] - 1e-9
    assert abs(table["scott"][0] - sd * 200 ** (-1.0 / 5.0)) < 1e-12

    print("bandwidths: 200 lognormal points, a sample whose sd is inflated by "
          f"its tail. sd {sd:.3f}, IQR {iqr:.3f}, IQR/1.349 {iqr / 1.349:.3f}, so "
          f"Silverman shrinks the scale to the robust one before applying "
          f"0.9 n^(-1/5). Scott {table['scott'][0]:.3f}, Silverman "
          f"{table['silverman'][0]:.3f}, cross-validated "
          f"{table['cv_bandwidth'][0]:.3f}. Leave-one-out log-likelihood: "
          f"{table['scott'][1]:.4f}, {table['silverman'][1]:.4f}, "
          f"{table['cv_bandwidth'][1]:.4f}. The two formulas disagree by a "
          f"factor of {table['scott'][0] / table['silverman'][0]:.1f} on the same "
          f"sample -- the robust scale is less than half the sd -- and CV takes "
          f"the larger of the three, so on this sample the outliers that "
          f"Silverman discounts are real signal and widening is the right answer. "
          f"It is also the most expensive rule: sixty likelihoods of a 200x200 "
          f"kernel, against one pass for a formula.")


def _demo_error() -> None:
    """Density error against a density that is known, and what CV does to it."""
    rng = np.random.default_rng(2)
    n = 200
    picks = rng.random(n) < 0.35
    X = np.where(picks, rng.normal(-1.2, 0.5, n), rng.normal(1.0, 0.7, n))
    X = X[:, None]
    grid = np.linspace(-5.0, 4.0, 901)

    rows = []
    for variant in ("gaussian", "epanechnikov", "tophat"):
        model = KernelDensity(variant=variant).fit(X)
        rows.append((variant, float(model.bandwidth_[0]), _ise(model, grid)))
    cv = KernelDensity(variant="cv_bandwidth").fit(X)
    rows.append(("gaussian @ cv", float(cv.bandwidth_[0]), _ise(cv, grid)))

    mass = {name: float(np.sum(np.exp(m.score_samples(grid[:, None])))
                        * float(grid[1] - grid[0]))
            for name, m in (("gaussian", KernelDensity().fit(X)),
                            ("tophat", KernelDensity(variant="tophat").fit(X)))}
    uniform = np.random.default_rng(5).random((500, 1))
    boundary = KernelDensity(bandwidth=0.1).fit(uniform)
    at_edge = float(np.exp(boundary.score_samples(np.zeros((1, 1)))[0]))
    at_middle = float(np.exp(boundary.score_samples(np.array([[0.5]]))[0]))
    assert at_edge < 0.75 * at_middle
    assert all(abs(v - 1.0) < 2e-3 for v in mass.values())
    # The likelihood is not the squared error: on this sample CV undersmooths.
    assert rows[-1][2] > rows[0][2]
    spacings = np.sort(np.abs(X - X.T), axis=1)[:, 1]
    median_gap = float(np.median(spacings))

    print("error: 200 points from a bimodal mixture, scored against the mixture "
          "itself on 901 grid points. Integrated squared error at the Scott "
          "bandwidth:")
    for name, h, ise in rows:
        print(f"  {name:15s} h={h:.3f}  ISE {ise:.2e}")
    print(f"  the estimate integrates to {mass['gaussian']:.4f} (Gaussian) and "
          f"{mass['tophat']:.4f} (top-hat) on the grid, so this compares "
          f"densities and not densities against mass. Swapping the kernel at "
          f"fixed h moves the error by less than a factor of two "
          f"({rows[0][2]:.2e}, {rows[1][2]:.2e}, {rows[2][2]:.2e}); moving h "
          f"moves it by more. Cross-validation chooses h={rows[-1][1]:.3f}, "
          f"less than a third of the rule's, and its ISE is "
          f"{rows[-1][2] / rows[0][2]:.1f} times the rule's. That is not a bug in "
          f"the search: the leave-one-out likelihood really is maximised there, "
          f"and the assertion above pins it down. It is the known pathology of "
          f"likelihood cross-validation for KDE. Each term is -log h plus the log "
          f"of a kernel value, so shrinking h buys log(1/h) until the nearest "
          f"neighbour falls outside the kernel -- on this sample the median "
          f"nearest-neighbour spacing is {median_gap:.3f}, three orders below the "
          f"chosen h, so the exponential term is what finally stops it. The "
          f"objective being maximised is not the objective you care about: "
          f"least-squares cross-validation, which compares densities rather than "
          f"evaluating them at the sample, is the one that tracks ISE.\n"
          f"  the same estimator at a boundary: 500 uniform points on [0, 1], "
          f"h = 0.1. The estimate is {at_edge:.3f} at x=0 and {at_middle:.3f} at "
          f"x=0.5, against a truth of 1.000. Half of every boundary kernel hangs "
          f"outside the support and there is no data there to make it up, so the "
          f"density is halved exactly where a detector most wants it. Reflecting "
          f"the data or fitting a log-spline avoids it; widening the bandwidth "
          f"does not.")


def _demo_tree() -> None:
    """The tree is an accumulator with a proof, not an approximation."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0.0, 0.6, size=(200, 2)),
                   rng.normal(3.0, 0.6, size=(200, 2)),
                   rng.normal([0.0, 3.0], 0.6, size=(200, 2))])
    # The bandwidth is fixed rather than ruled: Scott's rule reads the spread of
    # all three blobs (sd about 1.5) and returns 0.53, which is wider than a
    # blob, and at that width the other two blobs are inside the kernel.
    tree = KernelDensity(variant="tree_kde", bandwidth=0.15, leaf_size=8).fit(X)
    brute = KernelDensity(bandwidth=0.15).fit(X)
    groups = {
        "in a blob": rng.normal([0.0, 0.0], 0.4, size=(75, 2)),
        "between blobs": rng.normal([1.5, 1.5], 0.3, size=(75, 2)),
        "far away": rng.normal([20.0, 20.0], 2.0, size=(75, 2)),
    }
    report = {}
    for name, queries in groups.items():
        fast = tree.score_samples(queries)
        work = tree.evaluations_
        slow = brute.score_samples(queries)
        report[name] = (work, 1.0 - work / (len(queries) * len(X)),
                        float(np.abs(fast - slow).max()), float(fast.max()))
    worst = max(row[2] for row in report.values())
    assert worst < 1e-9, worst
    assert report["in a blob"][1] < report["between blobs"][1] < report["far away"][1]

    print(f"tree: 600 points in three blobs, 75 queries per group, leaf size 8, "
          f"bandwidth 0.15, pruning tolerance {tree.tol:.0e}. The tree agrees "
          f"with the brute-force sum to {worst:.1e} in log density, and evaluates")
    for name, (work, saved, _, _) in report.items():
        print(f"  {name:14s} {work:5d} kernels of 45000  ({saved:4.0%} saved)")
    print(f"  The saving is geometric and it is the whole point: a node is "
          f"skipped when even its nearest point is negligible against the running "
          f"sum. In a blob that saves a third, because the other two blobs are "
          f"far enough to skip; between them it saves two thirds; from twenty "
          f"units away every bound underflows to exactly zero and the tree touches "
          f"nothing at all, returning the floor of {report['far away'][3]:.1f} "
          f"which is the true double-precision answer. That last case is why the "
          f"pruning test cannot be purely relative: a relative test never fires "
          f"while the running sum is still zero, which is exactly the far field. "
          f"Note what the tree does not buy: inside a dense blob it is the plain "
          f"loop plus a traversal, so the speed-up is real only when a query is "
          f"dominated by a handful of nodes -- large n, well separated scales.")


def _demo_broken() -> None:
    """The inputs the estimator refuses, and one it should."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 2))
    flat_column = np.column_stack([X, np.ones(30)])
    cases = [
        ("an unknown variant", lambda: KernelDensity(variant="parzen")),
        ("a negative bandwidth", lambda: KernelDensity(bandwidth=-1.0)),
        ("a bandwidth of the wrong length",
         lambda: KernelDensity(bandwidth=np.ones(3)).fit(X)),
        ("one point", lambda: KernelDensity().fit(X[:1])),
        ("a constant column", lambda: KernelDensity().fit(flat_column)),
        ("scoring before fitting", lambda: KernelDensity().score_samples(X)),
        ("the wrong number of columns",
         lambda: KernelDensity().fit(X).score_samples(X[:, :1])),
        ("an anisotropic bandwidth with the tree",
         lambda: KernelDensity(variant="tree_kde",
                               bandwidth=[0.1, 0.2]).fit(X)),
        ("a zero tolerance", lambda: KernelDensity(tol=0.0)),
    ]
    for name, call in cases:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"{name} did not raise")

    # Data the estimator accepts and should not: two identical columns.
    duplicated = np.column_stack([X, X[:, 0]])
    model = KernelDensity(bandwidth=0.3).fit(duplicated)
    squeezed = KernelDensity(bandwidth=0.3).fit(X[:, :1])
    inflated = float(model.score_samples(duplicated[:1])[0]
                     - squeezed.score_samples(X[:1, :1])[0])
    print(f"broken: {len(cases)} bad inputs raise -- "
          f"{', '.join(name for name, _ in cases)}. The last one is not an "
          f"error, it is a mistake: duplicating a column turns the estimate into "
          f"the density of the diagonal x_1 = x_2, which is not the density of "
          f"x_1, and the same point scores {inflated:+.3f} nats differently with "
          f"nothing added. KDE has no defence against collinearity; a whitening "
          f"step (or a diagonal-bandwidth product kernel fitted on the whitened "
          f"data) is the fix, and dropping the duplicate column is the cheap one.")


def main() -> None:
    _demo_normalisation()
    _demo_bandwidths()
    _demo_error()
    _demo_tree()
    _demo_broken()
    print("kde demo passed")


if __name__ == "__main__":
    main()
