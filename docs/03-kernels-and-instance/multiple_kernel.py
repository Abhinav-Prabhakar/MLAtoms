"""Multiple kernel learning — combine kernels and learn the weights.

Chapter:   03-kernels-and-instance
Companion: multiple_kernel.md
Tier 3 · 4 variants: average, alignment, mkl_svm, easy_mkl
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("average", "alignment", "mkl_svm", "easy_mkl")
KINDS = ("linear", "polynomial", "rbf", "laplacian")
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Building the kernel blocks
# --------------------------------------------------------------------------- #
def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape `(len(A), len(B))`."""
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def _l1dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Manhattan distances, shape `(len(A), len(B))`."""
    return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)


def _pair(A: np.ndarray, B: np.ndarray, spec: dict) -> np.ndarray:
    """One kernel block from a spec dict, shape `(len(A), len(B))`.

    Recognised keys: `kind`, `gamma`, `degree`, `coef0`, `columns`. The
    `columns` key restricts the kernel to a feature subset, which is how you
    hand the learner a "this group of axes only" expert.
    """
    kind = spec.get("kind", "rbf")
    if kind not in KINDS:
        raise ValueError(f"unknown kernel kind {kind!r}; expected one of {KINDS}")
    columns = spec.get("columns")
    if columns is not None:
        columns = np.asarray(columns, dtype=int)
        A, B = A[:, columns], B[:, columns]
    gamma = float(spec.get("gamma", 1.0 / A.shape[1]))
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    if kind == "linear":
        return A @ B.T
    if kind == "polynomial":
        degree = int(spec.get("degree", 3))
        if degree < 1:
            raise ValueError("degree must be a positive integer")
        return (gamma * (A @ B.T) + float(spec.get("coef0", 1.0))) ** degree
    if kind == "rbf":
        return np.exp(-gamma * _sqdist(A, B))
    return np.exp(-gamma * _l1dist(A, B))


def kernel_matrices(X: np.ndarray, Z: np.ndarray | None = None,
                    specs: list[dict] | None = None) -> list[np.ndarray]:
    """Build one Gram block per spec.

    Returns `[K_1, ..., K_q]` with `K_k` of shape `(len(X), len(Z))`; `Z=None`
    means `Z = X`. Use the same `specs` for training and test blocks, and
    normalise only with training statistics — the caller owns the scales.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    Z = X if Z is None else np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != X.shape[1]:
        raise ValueError("Z must be two-dimensional with the same width as X")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z must be finite")
    if specs is None or len(specs) == 0:
        raise ValueError("specs must be a non-empty list of kernel specifications")
    return [_pair(X, Z, spec) for spec in specs]


# --------------------------------------------------------------------------- #
# The binary SVM that scores a combined kernel
# --------------------------------------------------------------------------- #
def _svm_fit(K: np.ndarray, y: np.ndarray, C: float, *,
             tol: float = 1e-6, max_iter: int = 50000
             ) -> tuple[np.ndarray, float, int]:
    """Maximise the C-SVM dual `1ᵀα − ½αᵀQα`, `Q = (y yᵀ) ⊙ K`.

    Pairwise coordinate ascent with a second-order working set. Returns
    `(alpha, b, iterations)`; `alpha` has shape `(n,)` and `b` is the offset.
    """
    n = y.size
    alpha = np.zeros(n)
    f = np.zeros(n)                       # f = K (α ⊙ y), without the offset
    grad = -np.ones(n)
    diag = np.diag(K)
    it = 0
    for _ in range(max_iter):
        up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
        low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
        if not up.any() or not low.any():
            break
        viol = -y * grad
        gi = np.where(up, viol, -np.inf)
        i = int(np.argmax(gi))
        if gi[i] - np.where(low, viol, np.inf).min() < tol:
            break
        ok = low & (viol < gi[i])
        if not ok.any():
            break
        curv = np.maximum(diag[i] + diag - 2.0 * K[i], EPS)
        crit = np.where(ok, -((viol - gi[i]) ** 2) / curv, np.inf)
        j = int(np.argmin(crit))
        a = max(diag[i] + diag[j] - 2.0 * K[i, j], EPS)
        step = ((y[i] - y[j]) - (f[i] - f[j])) / a
        lo, hi = -np.inf, np.inf
        for m, sign in ((i, y[i]), (j, -y[j])):
            room = C - alpha[m]
            if sign > 0:
                lo, hi = max(lo, -alpha[m]), min(hi, room)
            else:
                lo, hi = max(lo, -room), min(hi, alpha[m])
        step = min(max(step, lo), hi)
        if abs(step) < EPS:
            break
        alpha[i] += y[i] * step
        alpha[j] -= y[j] * step
        f += step * (K[i] - K[j])
        grad = y * f - 1.0
        it += 1
    free = (alpha > 1e-8) & (alpha < C - 1e-8)
    if free.any():
        b = float(np.mean(y[free] - f[free]))
    else:
        plus = (y > 0) & (alpha < C)
        b = float(np.mean(1.0 - f[plus])) if plus.any() else 0.0
    return alpha, b, it


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex, shape `(q,)`.

    Sorts once and finds the threshold `θ` with the largest active set; the
    result is non-negative and sums to one. `q` is small here, so the `O(q log q)`
    sort is cheaper than an active-set iteration.
    """
    u = np.sort(v)[::-1]
    total = np.cumsum(u) - 1.0
    index = np.arange(1, v.size + 1)
    active = u - total / index > 0.0
    rho = index[active][-1]
    theta = total[active][-1] / rho
    return np.maximum(v - theta, 0.0)


def _alignment(K: np.ndarray, target: np.ndarray) -> float:
    """Centred kernel-target alignment in `[-1, 1]`.

    `target` is `(n, n)`, normally `y yᵀ`. Both blocks are centred by
    `H = I − 11ᵀ/n` before the Frobenius inner product, because a kernel's
    constant part carries no label information.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Tc = H @ target @ H
    denominator = np.linalg.norm(Kc) * np.linalg.norm(Tc)
    if denominator < EPS:
        return 0.0
    return float(np.sum(Kc * Tc) / denominator)


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
class MultipleKernel:
    """Learn a non-negative combination of precomputed kernels, then classify.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    C : float
        SVM box bound applied to the combined kernel.
    reg : float
        Ridge term of the `easy_mkl` reduced problem; ignored elsewhere.
    max_iter : int
        Outer iterations of the `mkl_svm` line search.
    steps : int
        Projected-gradient steps of the `easy_mkl` pair-mass solver.
    tol : float
        KKT tolerance of the SVM and stationarity tolerance of the weight step.
    normalize : bool
        Divide each block by its training mean diagonal, so that weights
        compare kernels rather than kernel scales.

    Notes
    -----
    `fit` takes a *list* of Gram blocks, because multiple kernel learning is
    defined on kernels, not on features. Binary only: multiclass MKL needs a
    separate weight vector per machine, which is a different atom.
    """

    def __init__(self, variant: str = "average", *, C: float = 1.0,
                 reg: float = 0.1, max_iter: int = 25, steps: int = 600,
                 tol: float = 1e-6, normalize: bool = True):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError("C must be positive and finite")
        if not np.isfinite(reg) or reg < 0.0:
            raise ValueError("reg must be non-negative and finite")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be positive and finite")
        self.variant = variant
        self.C = float(C)
        self.reg = float(reg)
        self.max_iter = int(max_iter)
        self.steps = int(steps)
        self.tol = float(tol)
        self.normalize = bool(normalize)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, Ks: list[np.ndarray], y: np.ndarray) -> MultipleKernel:
        """Learn `weights_` from `Ks`, a list of `(n, n)` Gram blocks."""
        blocks, codes, classes = self._validate_train(Ks, y)
        n_blocks = len(blocks)
        scales = [float(np.mean(np.diag(K))) for K in blocks]
        if not all(np.isfinite(s) and s > 0.0 for s in scales):
            raise ValueError("every kernel must have a positive mean diagonal")
        self.scales_ = np.array(scales if self.normalize else np.ones(n_blocks))
        self.classes_ = classes
        self.codes_ = codes
        self.n_features_in_ = blocks[0].shape[0]
        blocks = [K / s for K, s in zip(blocks, self.scales_, strict=True)]
        target = np.outer(codes, codes)
        self.path_ = []

        if self.variant == "average":
            weights = np.full(n_blocks, 1.0 / n_blocks)
            self.n_iter_ = 0
        elif self.variant == "alignment":
            scores = np.array([_alignment(K, target) for K in blocks])
            positive = np.maximum(scores, 0.0)
            kept = positive.sum() > EPS
            weights = positive / positive.sum() if kept else np.full(
                n_blocks, 1.0 / n_blocks)
            self.alignment_ = scores
            self.n_iter_ = 0
        elif self.variant == "mkl_svm":
            weights = self._fit_mkl_svm(blocks, codes)
        else:
            weights = self._fit_easy_mkl(blocks, codes)

        self.weights_ = weights
        self.combined_ = sum(w * K for w, K
                             in zip(weights, blocks, strict=True))
        self.alpha_, self.intercept_, self.solve_iter_ = _svm_fit(
            self.combined_, codes, self.C, tol=self.tol)
        self.support_ = self.alpha_ > 1e-8
        return self

    def _validate_train(self, Ks, y):
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        if not isinstance(Ks, (list, tuple)) or len(Ks) == 0:
            raise ValueError("Ks must be a non-empty list of Gram blocks")
        blocks = [np.asarray(K, dtype=float) for K in Ks]
        n = blocks[0].shape[0] if blocks[0].ndim == 2 else -1
        for K in blocks:
            if K.ndim != 2 or K.shape != (n, n):
                raise ValueError("every block must be a square (n, n) matrix")
            if not np.all(np.isfinite(K)):
                raise ValueError("kernels must be finite")
            if not np.allclose(K, K.T, atol=1e-8):
                raise ValueError("kernels must be symmetric")
        y = np.asarray(y)
        if y.ndim != 1 or y.size != n:
            raise ValueError("y must have one label per row of the kernels")
        if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
            raise ValueError("y must be finite")
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("exactly two classes are required; "
                             "multiclass MKL needs one weight vector per machine")
        return blocks, np.where(y == classes[0], -1.0, 1.0), classes

    def _svm_solve(self, K):
        alpha, b, _ = _svm_fit(K, self.codes_, self.C, tol=self.tol)
        return alpha, b

    def _mixed(self, blocks, weights):
        return sum(w * K for w, K in zip(weights, blocks, strict=True))

    # -- mkl_svm: projected gradient on the dual value -------------------- #
    def _fit_mkl_svm(self, blocks, codes):
        """Descend `J(w) = max_α 1ᵀα − ½αᵀ(Y K(w) Y)α` with a line search."""
        n_blocks = len(blocks)
        weights = np.full(n_blocks, 1.0 / n_blocks)
        alpha, _ = self._svm_solve(self._mixed(blocks, weights))
        q = self._quadratic(blocks, alpha, codes)
        value = float(alpha.sum() - 0.5 * np.dot(weights, q))
        self.path_.append(value)
        step_size = 2.0 / max(float(q.max()), EPS)
        for _ in range(self.max_iter):
            direction = self._quadratic(blocks, alpha, codes)
            moved = False
            for _ in range(12):
                trial = _project_simplex(weights + 0.5 * step_size * direction)
                shift = float(np.max(np.abs(trial - weights)))
                if shift < 1e-10:
                    break
                done, _ = self._svm_solve(self._mixed(blocks, trial))
                tq = self._quadratic(blocks, done, codes)
                tvalue = float(done.sum() - 0.5 * np.dot(trial, tq))
                if tvalue <= value - 1e-12 * abs(value):
                    weights, value = trial, tvalue
                    alpha = done
                    self.path_.append(value)
                    step_size *= 1.5
                    moved = True
                    break
                step_size *= 0.5
            if not moved:
                break
        self.n_iter_ = len(self.path_) - 1
        return weights

    @staticmethod
    def _quadratic(blocks, alpha, codes):
        v = alpha * codes
        return np.array([float(v @ K @ v) for K in blocks])

    # -- easy_mkl: quadratic over a pair distribution --------------------- #
    def _fit_easy_mkl(self, blocks, codes):
        """Solve `min_γ γᵀ(YK̄Y + reg·I)γ` on per-class simplices.

        The result is the mass EasyMKL places on the hardest pairs; the kernel
        weights are the same quadratic form evaluated per block.
        """
        mean_block = np.mean(blocks, axis=0)
        positive = codes > 0
        g = np.zeros(codes.size)
        g[positive] = 1.0 / positive.sum()
        g[~positive] = 1.0 / (~positive).sum()

        def apply(vec):
            return codes * (mean_block @ (codes * vec)) + self.reg * vec

        probe = np.full(codes.size, 1.0 / codes.size)
        top = 1.0
        for _ in range(30):
            image = apply(probe)
            top = float(np.linalg.norm(image))
            if top < EPS:
                break
            probe = image / top
        rate = 1.0 / max(2.0 * top, EPS)
        value = float(g @ apply(g))
        self.path_.append(value)
        self.n_iter_ = 0
        for _ in range(self.steps):
            relaxed = g - rate * apply(g)
            new = np.empty_like(g)
            new[positive] = _project_simplex(relaxed[positive])
            new[~positive] = _project_simplex(relaxed[~positive])
            shift = float(np.max(np.abs(new - g)))
            g = new
            self.path_.append(float(g @ apply(g)))
            self.n_iter_ += 1
            if shift < self.tol:
                break
        self.pair_mass_ = g
        v = g * codes
        scores = np.array([float(v @ K @ v) for K in blocks])
        positive_scores = np.maximum(scores, 0.0)
        if positive_scores.sum() < EPS:
            return np.full(len(blocks), 1.0 / len(blocks))
        return positive_scores / positive_scores.sum()

    # -- prediction ------------------------------------------------------- #
    def decision_function(self, Ks: list[np.ndarray]) -> np.ndarray:
        """Signed scores for cross-kernel blocks, shape `(m,)`."""
        self._check_fitted()
        if not isinstance(Ks, (list, tuple)) or len(Ks) != self.weights_.size:
            raise ValueError("predict needs one block per fitted kernel")
        blocks = [np.asarray(K, dtype=float) for K in Ks]
        for K in blocks:
            if K.ndim != 2 or K.shape[1] != self.n_features_in_:
                raise ValueError("every test block must be (m, n_train)")
            if not np.all(np.isfinite(K)):
                raise ValueError("test kernels must be finite")
        v = self.alpha_ * self.codes_
        scores = np.full(blocks[0].shape[0], self.intercept_)
        for weight, scale, K in zip(self.weights_, self.scales_, blocks,
                                    strict=True):
            scores += (weight / scale) * (K @ v)
        return scores

    def predict(self, Ks: list[np.ndarray]) -> np.ndarray:
        """Class labels for cross-kernel blocks, shape `(m,)`."""
        scores = self.decision_function(Ks)
        return np.where(scores > 0.0, self.classes_[1], self.classes_[0])

    def _check_fitted(self):
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit before predict or decision_function")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _xor_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """XOR in two axes plus six noise axes with six times the spread."""
    signal = rng.uniform(-1.0, 1.0, (n, 2))
    labels = np.where(signal[:, 0] * signal[:, 1] > 0.0, 1.0, -1.0)
    noise = rng.normal(0.0, 2.0, (n, 6))
    return np.hstack([signal, noise]), labels


def demo() -> None:
    rng = np.random.default_rng(0)
    train, train_y = _xor_data(150, rng)
    test, test_y = _xor_data(2000, rng)
    specs = [
        {"kind": "rbf", "gamma": 2.0, "columns": [0, 1]},
        {"kind": "polynomial", "degree": 3, "gamma": 1.0, "columns": [0, 1]},
        {"kind": "rbf", "gamma": 0.01},
        {"kind": "rbf", "gamma": 100.0},
        {"kind": "linear", "columns": [2, 3, 4, 5, 6, 7]},
        {"kind": "rbf", "gamma": 1.0, "columns": [2, 3, 4, 5, 6, 7]},
        {"kind": "linear"},
        {"kind": "polynomial", "degree": 1, "gamma": 1.0},
    ]
    names = ["rbf2/sig", "poly3/sig", "rbf0.01/all", "rbf100/all",
             "lin/noise", "rbf1/noise", "lin/all", "poly1/all"]
    train_k = kernel_matrices(train, None, specs)
    test_k = kernel_matrices(test, train, specs)
    print("eight kernels, four useful: two on the two label-carrying axes,")
    print("two noise-only, four global kernels blinded by six loud axes")
    solo = []
    for name, K_tr, K_te in zip(names, train_k, test_k, strict=True):
        one = MultipleKernel("average").fit([K_tr], train_y)
        assert np.allclose(one.combined_, K_tr / one.scales_[0])
        solo.append(float(np.mean(one.predict([K_te]) == test_y)))
        print(f"  {name:12s} {solo[-1]:.4f}")

    scores = {}
    for variant in ("average", "alignment", "mkl_svm", "easy_mkl"):
        model = MultipleKernel(variant).fit(train_k, train_y)
        weights = model.weights_
        scores[variant] = float(np.mean(model.predict(test_k) == test_y))
        assert np.isclose(weights.sum(), 1.0) and np.all(weights >= 0.0)
        kept = int(np.sum(weights > 1e-3))
        print(f"  {variant:12s} {scores[variant]:.4f}   "
              f"{kept} of 8 kernels kept")
        print("    weights  " + "  ".join(f"{w:.3f}" for w in weights))

    svm_model = MultipleKernel("mkl_svm").fit(train_k, train_y)
    print("mkl_svm drives the dual from a uniform kernel's "
          f"{svm_model.path_[0]:.2f} to {svm_model.path_[-1]:.2f}")
    print(f"over {svm_model.n_iter_} line-search steps, and keeps two kernels")
    print("that are chance alone: the dual rewards margin volume, and a")
    print("100-bandwidth RBF on pure noise supplies plenty of it")
    assert svm_model.path_[-1] < 0.75 * svm_model.path_[0]

    assert np.allclose(MultipleKernel("average").fit(train_k, train_y).weights_,
                       np.full(8, 0.125))
    assert scores["mkl_svm"] >= scores["average"] - 1e-12
    assert scores["alignment"] >= scores["average"] - 1e-12
    assert scores["easy_mkl"] >= scores["average"] - 1e-12
    assert max(solo) < 0.97

    print("the combined kernel is the weighted sum, and every weight vector")
    print("is a point of the simplex:")
    manual = sum(w * K / s for w, s, K in zip(model.weights_, model.scales_,
                                              train_k, strict=True))
    print(f"  max |K(w) - sum w_k K_k|  {np.abs(model.combined_ - manual).max():.1e}")
    assert np.allclose(model.combined_, manual)
    assert np.allclose(_project_simplex(np.array([3.0, -1.0, 0.2, 5.0])).sum(), 1.0)
    assert np.all(_project_simplex(np.array([3.0, -1.0, 0.2, 5.0])) >= 0.0)
    print("  projection of (3, -1, 0.2, 5) sums to one and stays non-negative")

    target = np.outer(train_y, train_y)
    assert np.isclose(_alignment(target, target), 1.0)
    assert np.isclose(_alignment(3.5 * target, target), 1.0)
    assert np.isclose(_alignment(np.ones((150, 150)), target), 0.0)
    print("alignment is scale free: A(yyT, yyT) = 1, A(3.5 yyT, yyT) = 1,")
    print("and a constant kernel carries no centred mass, so A(11T, yyT) = 0")
    flat = [np.ones((150, 150)), np.ones((150, 150))]
    assert np.allclose(MultipleKernel("alignment").fit(flat, train_y).weights_, 0.5)
    print("two kernels with no alignment fall back to uniform weights")

    assert all(np.diff(model.path_) <= 1e-9)
    assert model.n_iter_ < model.steps
    mass = model.pair_mass_
    assert np.isclose(mass[train_y > 0].sum(), 1.0)
    assert np.isclose(mass[train_y < 0].sum(), 1.0)
    print(f"easy_mkl drove its quadratic from {model.path_[0]:.4f} to "
          f"{model.path_[-1]:.4f} in {model.n_iter_} steps, with pair mass")
    print(f"{mass[train_y > 0].sum():.3f} on the positive and "
          f"{mass[train_y < 0].sum():.3f} on the negative class")

    v = model.alpha_ * model.codes_
    margin = model.codes_ * (model.combined_ @ v + model.intercept_)
    free = (model.alpha_ > 1e-8) & (model.alpha_ < model.C - 1e-8)
    assert np.all(model.alpha_ <= model.C + 1e-10)
    assert np.allclose(margin[free], 1.0, atol=1e-4)
    assert np.all(margin[model.alpha_ <= 1e-8] >= 1.0 - 1e-4)
    print(f"KKT holds on {int(model.support_.sum())} support vectors, "
          f"{int(free.sum())} of them free at margin one")

    bad = [
        lambda: MultipleKernel("vote"),
        lambda: MultipleKernel(C=0.0),
        lambda: MultipleKernel(reg=-1.0),
        lambda: MultipleKernel(max_iter=0),
        lambda: MultipleKernel(steps=0),
        lambda: MultipleKernel(tol=0.0),
        lambda: MultipleKernel().fit([], train_y),
        lambda: MultipleKernel().fit([np.ones((4, 4))], np.array([0, 1])),
        lambda: MultipleKernel().fit([np.ones((4, 5))], np.ones(4)),
        lambda: MultipleKernel().fit([np.full((4, 4), np.nan)], np.array([0, 0, 1, 1])),
        lambda: MultipleKernel().fit([np.ones((4, 4))], np.array([0, 1, 2, 0])),
        lambda: MultipleKernel().fit(
            [np.array([[1.0, 2.0], [0.0, 1.0]])], np.array([0, 1])),
        lambda: MultipleKernel().fit([np.eye(4)], np.array([0, 1, 0])),
        lambda: MultipleKernel().fit([np.eye(4)], np.array([0.0, np.nan, 1.0, 1.0])),
        lambda: MultipleKernel().predict([np.ones((2, 2))]),
        lambda: MultipleKernel().fit([np.eye(4)], np.ones(4)),
        lambda: MultipleKernel().fit([np.zeros((4, 4))], np.array([0, 0, 1, 1])),
        lambda: MultipleKernel().fit([np.eye(4)], np.array([0, 0, 1, 1])).predict(
            [np.ones((3, 5))]),
        lambda: MultipleKernel().fit([np.eye(4)], np.array([0, 0, 1, 1])).predict(
            [np.ones((3, 4)), np.ones((3, 4))]),
        lambda: kernel_matrices(np.zeros((3, 2)), specs=[]),
        lambda: kernel_matrices(np.zeros((3, 2)), specs=[{"kind": "cosine"}]),
        lambda: kernel_matrices(np.zeros((3, 2)), specs=[{"kind": "rbf", "gamma": 0}]),
        lambda: kernel_matrices(np.zeros(3), specs=[{"kind": "linear"}]),
        lambda: kernel_matrices(np.ones((3, 2)), np.ones((3, 3)),
                                specs=[{"kind": "linear"}]),
        lambda: kernel_matrices(np.zeros((3, 2)),
                                specs=[{"kind": "polynomial", "degree": 0}]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown "
          "variant, a non-positive C,")
    print("  a negative ridge, zero iterations, steps or tolerance,")
    print("  an empty kernel list, a mismatched block size or label vector, a")
    print("  non-square, non-finite or asymmetric block, three classes, one")
    print("  class, a non-finite label, predict before fit, a wrong test width")
    print("  or block count, an unknown kernel kind, a zero bandwidth, a 1-d")
    print("  design, a cross-kernel width mismatch, degree zero and empty specs")
    print("multiple_kernel demo passed")


if __name__ == "__main__":
    demo()
