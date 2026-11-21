"""PHATE — potential distances and information-geometric embeddings.

Diffusion maps answers "how close are these two points" with a diffusion
distance: the L2 gap between two rows of P^t, weighted by the stationary
distribution. That distance is small whenever the two rows overlap and *large*
whenever they do not, and on a data set with far-apart regions it saturates:
two points in different branches are both "infinitely far" from each other, so
the geometry between branches is lost. PHATE's move is to measure distance in
the exponent instead of in the probability. Define the potential

    U = -log(P^t)                                                   (4)

and put a Euclidean metric on its rows. A probability of 1e-3 and one of 1e-9
are then 6 nats apart rather than 1e-3 apart, so no distance saturates, and
multiplying P by a constant — which is what a longer walk mostly does — becomes
an additive constant, which distances ignore. That is why the picture is stable
in t where a diffusion map is not.

The pipeline, in the order the code runs it:

    (1) kernel       K_ij = exp(-(d_ij / sigma_i) ** decay) on kNN edges
    (2) walk         P = D^-1 K,  pi = d / sum(d)
    (3) diffusion    P^t, by repeated squaring
    (4) potential    U = -log(P^t + 1e-12)
    (5) embedding    metric MDS on the Euclidean distances between rows of U

Step (1) is where this departs from diffusion maps in a way that is easy to
miss: the kernel width is *per point*, sigma_i = the distance to i's k-th
neighbour, so d_ij / sigma_i is a distance measured in units of local spacing.
A dense region gets a narrow kernel and a sparse one gets a wide kernel, and
the graph stops encoding sampling density — the job diffusion maps gives to its
alpha normalisation, done here without an exponent to tune.

The gamma family. Equation (4) is one end of a one-parameter family of
monotone transforms, the Box-Cox transforms of the diffusion operator:

    phi_gamma(x) = x^c / c   with c = (1 - gamma) / 2,   phi_1(x) = log x

gamma = 1 is the log potential of (4); gamma = 0 is 2*sqrt(x); gamma = -1 is
the identity, whose distances are exactly the diffusion distances of the atom
next door. Smaller gamma compresses less, so the far field behaves more like a
diffusion map and less like a potential. The three variants of this atom are
not three methods — they sample this family and the choice of embedder:

    phate          gamma = 1, metric MDS      (the published pipeline)
    potential      gamma = 1, classical MDS  (the embedder is optional)
    kernel_phate   gamma = -1, kernel PCA    (the potential is not)

Everything here is dense: P is an n x n matrix, P^t costs O(n^3 log t) by
repeated squaring, and the potential distance matrix is another O(n^3). The
published implementation uses landmarks and sparse operators to reach 10^6
cells; this file is the definition, and the demo checks it against closed forms.
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("phate", "potential", "kernel_phate")


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


def adaptive_kernel(distances: np.ndarray, graph: np.ndarray,
                    sigma: np.ndarray, decay: float) -> np.ndarray:
    """Equation (1): a Gaussian in units of each point's own neighbour distance.

    `sigma[i]` is the distance from `i` to its `k`-th neighbour, so the exponent
    is dimensionless. The kernel is symmetrised by taking the larger of the two
    directions, which keeps every entry in `(0, 1]` and adds no weight the
    graph did not already have.
    """
    scaled = distances / sigma[:, None]
    kernel = np.where(graph, np.exp(-(scaled ** decay)), 0.0)
    return np.maximum(kernel, kernel.T)


def potential_transform(power: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Equation (4) and its family: `phi_gamma` applied entrywise to `P^t`.

    `gamma = 1` is the log potential, `gamma = -1` the identity, and everything
    between is a power. The 1e-12 floor is not cosmetic: `P^t` has entries that
    underflow to zero on a long walk, and the log of zero is the one value this
    transform cannot return.
    """
    if not -1.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [-1, 1]")
    if gamma == 1.0:
        return -np.log(np.maximum(power, 1e-12))
    c = (1.0 - gamma) / 2.0
    return power ** c / c


# --------------------------------------------------------------------------- #
# Choosing the diffusion time from the spectrum
# --------------------------------------------------------------------------- #
def von_neumann_entropy(eigenvalues: np.ndarray, t_max: int = 100):
    """Entropy of the density matrix `P^t / tr(P^t)`, as a function of `t`.

    `rho_t` has eigenvalues `p_l = |lambda_l|^t / sum_m |lambda_m|^t`, so the
    entropy `H = -sum_l p_l log p_l` is a function of the spectrum alone — no
    matrix powers are needed to plot it. It starts at `log n` (every mode
    equally weighted, the identity) and falls to zero (one mode, a rank-one
    operator) as the walk forgets where it started.
    """
    t_max = int(t_max)
    if t_max < 3:
        raise ValueError("t_max must be at least 3")
    lam = np.maximum(np.abs(eigenvalues), 1e-300)
    t = np.arange(t_max)
    # Work in logs: at t = 99 every weight in the row can underflow at once and
    # a direct `lam**t / sum(lam**t)` is 0/0.
    log_weights = t[:, None] * np.log(lam)[None, :]
    log_weights = log_weights - log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    prob = np.maximum(weights / weights.sum(axis=1, keepdims=True), 1e-300)
    return t, -np.sum(prob * np.log(prob), axis=1)


def knee_point(x: np.ndarray, y: np.ndarray) -> float:
    """Chord-distance (Kneedle) knee of a monotone curve.

    Rescale both axes to the unit square, draw the chord from the first point to
    the last, and return the x whose point is farthest below it. The published
    implementation fits a pair of least-squares lines and minimises absolute
    error instead; on a monotone curve the two rules pick the same turn.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional and the same length")
    if len(x) < 3:
        raise ValueError("cannot find a knee on fewer than three points")
    span_x = x[-1] - x[0]
    span_y = y[-1] - y[0]
    if span_x == 0.0 or span_y == 0.0:
        return float(x[0])
    xn = (x - x[0]) / span_x
    yn = (y - y[0]) / span_y
    chord = yn[0] + (yn[-1] - yn[0]) * xn
    # A curve below the chord has a knee; a convex curve does not, and then the
    # largest distance is the answer all the same.
    return float(x[int(np.argmax(np.abs(yn - chord)))])


# --------------------------------------------------------------------------- #
# Embedding a distance matrix
# --------------------------------------------------------------------------- #
def classical_mds(distances: np.ndarray, n_components: int) -> np.ndarray:
    """Coordinates whose Euclidean distances best match `distances` in L2.

    Double-centre the squared distances, `B = -1/2 J D^2 J`, and keep the top
    eigenpairs: for a genuinely Euclidean `D` this is exact, and otherwise it is
    the closest Euclidean distance matrix in Frobenius norm.
    """
    n = distances.shape[0]
    centring = np.eye(n) - 1.0 / n
    gram = -0.5 * centring @ (distances ** 2) @ centring
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:n_components]
    kept = np.maximum(values[order], 0.0)
    return vectors[:, order] * np.sqrt(kept)[None, :]


def _stress(distances: np.ndarray, X: np.ndarray, scale: float) -> float:
    """Normalised stress: the residual distance error, relative to the target."""
    return float(np.sqrt(np.sum((distances - _pairwise(X)) ** 2) / scale))


def _smacof(distances: np.ndarray, init: np.ndarray, n_iter: int,
            tol: float = 1e-10):
    """Metric MDS by SMACOF: the Guttman transform, iterated.

    Each step solves the weighted least-squares problem in closed form, so there
    is no learning rate: the transform is a descent direction and the stress
    falls monotonically. The init decides which local minimum is reached;
    classical MDS is the standard choice because it is the global answer for the
    Euclidean part of the problem.

    The backtracking loop is the price of a dense, ill-conditioned target. When
    two embedded points land on top of each other, `D_ij / d_ij` is enormous and
    the undamped step throws the configuration across the plane; halving the
    step until the stress falls keeps the iteration honest instead of relying on
    a floor that only postpones the problem.
    """
    X = init.copy()
    n = distances.shape[0]
    scale = float(np.sum(distances ** 2))
    stress = _stress(distances, X, scale)
    for _ in range(int(n_iter)):
        ratio = distances / np.maximum(_pairwise(X), 1e-9)
        transform = -ratio
        np.fill_diagonal(transform, ratio.sum(axis=1))
        step = (transform @ X) / n - X
        alpha = 1.0
        for _ in range(20):
            trial = X + alpha * step
            new_stress = _stress(distances, trial, scale)
            if new_stress < stress:
                break
            alpha *= 0.5
        else:
            break
        change = stress - new_stress
        X, stress = trial, new_stress
        if change < tol:
            break
    return X, stress


# --------------------------------------------------------------------------- #
# The embedding
# --------------------------------------------------------------------------- #
class PHATE:
    """Potential distances and their MDS embedding.

    `PHATE(n_components=2, *, variant="phate", n_neighbors=5, decay=2.0,
    t="auto", t_max=100, n_iter=300)` with `fit(X)` and `fit_transform(X)`;
    `X` is `(n, n_features)`.

    Attributes after `fit`: `embedding_` `(n, n_components)`; `t_`, the
    diffusion time used (`t` or the knee of the entropy curve); `potential_`,
    the `(n, n)` matrix that was embedded; `potential_distances_`; `kernel_`,
    `transition_` and `stationary_`, the graph and its walk; `sigma_`, the
    per-point bandwidths; `eigenvalues_` and `spectrum_`, the spectrum of the
    walk; `entropy_`, the `(t, H)` curve used to pick `t_`; `stress_`, the
    normalised stress of the MDS (zero for the spectral variants);
    `n_features_`.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "phate",
                 n_neighbors: int = 5, decay: float = 2.0,
                 t: int | str = "auto", t_max: int = 100,
                 n_iter: int = 300):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if not decay > 0.0:
            raise ValueError("decay must be positive")
        if t != "auto" and (not isinstance(t, (int, np.integer)) or t < 1):
            raise ValueError("t must be a positive integer or 'auto'")
        if n_iter < 1:
            raise ValueError("n_iter must be positive")
        self.n_components = int(n_components)
        self.variant = variant
        self.n_neighbors = int(n_neighbors)
        self.decay = float(decay)
        self.t = t
        self.t_max = int(t_max)
        self.n_iter = int(n_iter)

    def fit(self, X: np.ndarray) -> PHATE:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        n = X.shape[0]
        if not 1 <= self.n_neighbors < n:
            raise ValueError("n_neighbors must lie in [1, n)")
        if self.n_components >= n:
            raise ValueError("n_components must be smaller than n")

        # (1) the graph, with a bandwidth measured in local spacings
        self.distances_ = _pairwise(X)
        graph = _knn_graph(self.distances_, self.n_neighbors)
        self.sigma_ = np.sort(np.where(graph, self.distances_, np.inf), axis=1)[
            :, self.n_neighbors - 1]
        if np.any(self.sigma_ <= 0.0):
            raise ValueError("duplicate points: some point has zero distance to "
                             "its k-th neighbour, so its kernel is a delta")
        self.kernel_ = adaptive_kernel(self.distances_, graph, self.sigma_, self.decay)

        # (2) the walk
        degree = self.kernel_.sum(axis=1)
        self.transition_ = self.kernel_ / degree[:, None]
        self.stationary_ = degree / degree.sum()

        # the spectrum, via the symmetric twin of the walk
        symmetric = self.kernel_ / np.sqrt(degree)[:, None] / np.sqrt(degree)[None, :]
        lam, vecs = np.linalg.eigh(symmetric)
        order = np.argsort(lam)[::-1]
        self.eigenvalues_ = lam[order]
        self.eigenvectors_ = vecs[:, order]
        self.spectrum_ = self.eigenvalues_[:self.n_components + 1]

        # (3) the diffusion time
        self.entropy_ = von_neumann_entropy(self.eigenvalues_, self.t_max)
        if self.t == "auto":
            self.t_ = int(round(knee_point(*self.entropy_)))
        else:
            self.t_ = int(self.t)
        power = np.linalg.matrix_power(self.transition_, self.t_)
        self.power_ = power

        # (4) the potential, and (5) the embedding
        if self.variant == "kernel_phate":
            # gamma = -1: the diffusion kernel itself, embedded by kernel PCA.
            # The right eigenvectors scaled by lambda^(t/2) are the features of
            # the kernel with respect to the stationary measure, so their
            # Euclidean distances are the diffusion distances at time t.
            # A kNN-masked kernel is not positive semidefinite, so its
            # spectrum has negative modes; kernel PCA is only defined for a PSD
            # kernel, so they are clipped rather than exponentiated to a
            # fractional power. They carry |lambda|^t, which the walk shrinks
            # faster than the leading mode, so the clip is cheap.
            gains = np.maximum(self.eigenvalues_, 0.0) ** (self.t_ / 2.0)
            coords = (self.eigenvectors_ / np.sqrt(degree)[:, None]) * gains[None, :]
            first = int(np.sum(self.eigenvalues_ > 1.0 - 1e-10))
            self.potential_ = power
            self.potential_distances_ = _pairwise(coords)
            self.embedding_ = coords[:, first:first + self.n_components]
            self.stress_ = 0.0
        else:
            self.potential_ = potential_transform(power, gamma=1.0)
            self.potential_distances_ = _pairwise(self.potential_)
            start = classical_mds(self.potential_distances_, self.n_components)
            if self.variant == "phate":
                self.embedding_, self.stress_ = _smacof(
                    self.potential_distances_, start, self.n_iter)
            else:
                self.embedding_, self.stress_ = start, 0.0

        self.n_features_ = X.shape[1]
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).embedding_


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #
def _scurve(n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The folded sheet the neighbouring atoms are measured on, plus its chart.

    `t` is the angle along the fold and `h` the height across it; together they
    are the sheet's own two coordinates, and the embedding is scored against
    them rather than against the three ambient ones.
    """
    rng = np.random.default_rng(seed)
    t = 3.0 * np.pi * (rng.random(n) - 0.5)
    h = 2.0 * rng.random(n)
    X = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])
    return X, np.column_stack([t, h])


def _pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    centred = X - X.mean(axis=0)
    left, spectrum, _ = np.linalg.svd(centred, full_matrices=False)
    return left[:, :n_components] * spectrum[:n_components]


def _overlap(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    """Mean fraction of `k` nearest neighbours shared by two point sets."""
    da, db = _pairwise(A), _pairwise(B)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    ia = np.argsort(da, axis=1)[:, :k]
    ib = np.argsort(db, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(ia, ib, strict=True)]))


def _corr(Da: np.ndarray, Db: np.ndarray) -> float:
    """Correlation of the two distance matrices, over unordered pairs."""
    iu = np.triu_indices(Da.shape[0], 1)
    return float(np.corrcoef(Da[iu], Db[iu])[0, 1])


def _rank_corr(Da: np.ndarray, Db: np.ndarray) -> float:
    """Spearman correlation: the same measure after replacing values by ranks."""
    iu = np.triu_indices(Da.shape[0], 1)
    a = np.argsort(np.argsort(Da[iu])).astype(float)
    b = np.argsort(np.argsort(Db[iu])).astype(float)
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_identity(n: int = 80, n_features: int = 5, seed: int = 0) -> None:
    """The pieces that must be exact, checked against their definitions."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    model = PHATE(n_neighbors=6, t=8).fit(X)
    P, pi = model.transition_, model.stationary_
    row = float(np.abs(P.sum(axis=1) - 1.0).max())
    invariant = float(np.abs(pi @ P - pi).max())
    symmetry = float(np.abs(model.kernel_ - model.kernel_.T).max())
    assert row < 1e-12 and invariant < 1e-15 and symmetry == 0.0

    power = model.power_
    identity = float(np.abs(potential_transform(power, -1.0) - power).max())
    root = float(np.abs(potential_transform(power, 0.0) - 2.0 * np.sqrt(power)).max())
    mask = power > 1e-8
    c = 1e-6
    boxcox = float(np.abs((power[mask] ** c - 1.0) / c - np.log(power[mask])).max())
    assert identity == 0.0 and root == 0.0 and boxcox < 1e-4

    D = model.potential_distances_
    violation = float(max(D[i, k] - D[i, j] - D[j, k]
                          for i in range(n) for j in range(n) for k in range(n)))
    triples = n ** 3

    points = rng.normal(size=(30, 3))
    exact = classical_mds(_pairwise(points), 3)
    recovery = float(np.abs(_pairwise(exact) - _pairwise(points)).max() / np.abs(
        _pairwise(points)).max())

    kernel_model = PHATE(variant="kernel_phate", n_neighbors=6, t=8).fit(X)
    half = np.linalg.matrix_power(kernel_model.transition_, 4)
    stationary = kernel_model.stationary_
    definition = np.sqrt(np.sum((half[:, None, :] - half[None, :, :]) ** 2
                                / stationary[None, None, :], axis=2))
    spectral = float(np.linalg.norm(kernel_model.potential_distances_
                                    - definition / np.sqrt(kernel_model.kernel_.sum()))
                     / np.linalg.norm(definition))
    lam = kernel_model.eigenvalues_
    negative = float(np.sum(np.abs(lam[lam < 0]) ** kernel_model.t_)
                     / np.sum(np.abs(lam) ** kernel_model.t_))

    t, entropy = model.entropy_
    assert abs(entropy[0] - np.log(n)) < 1e-12
    assert np.all(np.diff(entropy) <= 1e-12)
    direct = []
    for s in t[:20]:
        vals = np.abs(np.linalg.eigvals(np.linalg.matrix_power(P, int(s))))
        vals = np.maximum(vals / vals.sum(), 1e-300)
        direct.append(-float(np.sum(vals * np.log(vals))))
    entropy_gap = float(np.abs(np.array(direct) - entropy[:20]).max())

    print(f"identity: {n} random points in {n_features} dimensions, k=6. The "
          f"bookkeeping first: rows of P sum to one to {row:.1e}, the stationary "
          f"distribution is invariant to {invariant:.1e}, the kernel is symmetric "
          f"exactly, all three asserted. Then the potential family: gamma=-1 is the "
          f"operator itself and gamma=0 is exactly 2*sqrt(P^t) (both to 0.0), and "
          f"the Box-Cox form (x^c-1)/c with c=1e-6 differs from log x by at most "
          f"{boxcox:.1e} -- first order in c, as the expansion says -- so the log "
          f"potential is the limit of a family, not a special case bolted on. The "
          f"potential distance is a Euclidean distance, so {triples} triangle "
          f"inequalities hold with worst excess {violation:.1e}; classical MDS "
          f"inverts a Euclidean distance matrix to {recovery:.1e} relative error; "
          f"and the kernel variant at t=8 reproduces the definition of the "
          f"diffusion distance at t=4 -- two powers, not eight, because a kernel "
          f"PCA coordinate is a square root -- to {spectral:.1e}, the whole of that "
          f"gap being the {negative:.0e} of the kernel's spectral mass that this "
          f"data puts on negative eigenvalues, which kernel PCA clips and the "
          f"distance does not. Finally the entropy curve: H(0) = log n to 1e-12, "
          f"non-increasing, and computed from the spectrum it agrees with twenty "
          f"matrix powers of P to {entropy_gap:.1e}.")


def _demo_sheet() -> None:
    """The three variants on the chapter's standard folded sheet."""
    X, chart = _scurve()
    truth = _pairwise(chart)
    pca = _pca(X)
    print("variants: 400 points of an S-curve, k=5, auto-t, scored against the "
          "sheet's own (angle, height) coordinates:")
    for variant in VARIANTS:
        model = PHATE(variant=variant, n_neighbors=5).fit(X)
        print(f"  {variant:12s} t={model.t_:3d}  overlap@10 "
              f"{_overlap(model.embedding_, chart):.3f}  dcorr "
              f"{_corr(_pairwise(model.embedding_), truth):.3f}  stress "
              f"{model.stress_:.3f}")
    print(f"  {'PCA':12s}       overlap@10 {_overlap(pca, chart):.3f}"
          f"{'':17s}  dcorr {_corr(_pairwise(pca), truth):.3f}")
    decays = [(d, PHATE(n_neighbors=5, decay=d).fit(X)) for d in (2.0, 10.0, 40.0)]
    spread = max(abs(_corr(a.potential_distances_, truth)
                     - _corr(b.potential_distances_, truth))
                 for (_, a) in decays for (_, b) in decays)
    print(f"  the kernel exponent, decay=2 (this file), 10 and 40 (the reference "
          f"default): potential-distance correlation "
          f"{', '.join(f'{_corr(m.potential_distances_, truth):.3f}' for _, m in decays)}"
          f" -- a spread of {spread:.3f}, so on a uniformly sampled sheet the "
          f"exponent is inert and the per-point bandwidth is doing the work.")


def _demo_time() -> None:
    """What the log buys: the potential's geometry keeps sharpening with t."""
    X, chart = _scurve()
    truth = _pairwise(chart)
    picks = [1, 2, 5, 10, 25, 50]
    potential, embedded, kernel = [], [], []
    for s in picks:
        model = PHATE(n_neighbors=5, t=s).fit(X)
        other = PHATE(variant="kernel_phate", n_neighbors=5, t=s).fit(X)
        potential.append(_corr(model.potential_distances_, truth))
        embedded.append(_corr(_pairwise(model.embedding_), truth))
        kernel.append(_corr(_pairwise(other.embedding_), truth))
    auto = PHATE(n_neighbors=5).fit(X)
    early = PHATE(n_neighbors=5, t=10).fit(X)
    late = PHATE(n_neighbors=5, t=50).fit(X)
    rank = _rank_corr(early.potential_distances_, late.potential_distances_)
    window = [i for i, s in enumerate(picks) if 10 <= s <= 50]
    print(f"time: one embedding per diffusion time on the same sheet. Correlation "
          f"with the sheet, for t = {', '.join(str(s) for s in picks)}:")
    print(f"  potential distances  {', '.join(f'{v:.3f}' for v in potential)}")
    print(f"  PHATE embedding      {', '.join(f'{v:.3f}' for v in embedded)}")
    print(f"  kernel embedding     {', '.join(f'{v:.3f}' for v in kernel)}")
    print(f"  The potential climbs from {potential[0]:.3f} to {potential[-1]:.3f} "
          f"and the kernel stays inside {min(kernel):.3f}-{max(kernel):.3f}: a "
          f"diffusion distance is dominated by the leading modes, which saturate, "
          f"while the log keeps every entry of the row in play, so running the walk "
          f"longer keeps buying resolution. Auto-t picks t={auto.t_} from the knee "
          f"of the entropy curve. That is not a claim that t does not matter: at "
          f"t=1 the potential is still local and the picture is a smear. It is a "
          f"claim about the far field. The rank correlation between the potential "
          f"distances at t=10 and t=50 is {rank:.3f}, and the embedding's score "
          f"moves by only "
          f"{max(embedded[i] for i in window) - min(embedded[i] for i in window):.3f} "
          f"across that window, so past the knee more diffusion reorders very "
          f"little and sharpens what is already there.")


def _demo_broken() -> None:
    """The ways the constructor and the graph refuse to guess."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    cases = [
        ("a variant that does not exist", lambda: PHATE(variant="phate2")),
        ("a non-positive decay", lambda: PHATE(decay=0.0)),
        ("t=0", lambda: PHATE(t=0)),
        ("t_max below three", lambda: PHATE(t_max=2).fit(X)),
        ("a gamma outside [-1, 1]", lambda: potential_transform(np.ones((2, 2)), 1.5)),
        ("as many neighbours as points", lambda: PHATE(n_neighbors=40).fit(X)),
        ("more components than points", lambda: PHATE(n_components=40).fit(X)),
        ("duplicate points", lambda: PHATE(n_neighbors=3).fit(
            np.repeat(X[:4], 10, axis=0))),
    ]
    for _name, call in cases:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{_name} did not raise")

    # A disconnected graph is not an error -- it is worse, it is silent.
    left = rng.normal(size=(5, 2))
    clumps = np.vstack([left, left + 100.0])
    model = PHATE(n_neighbors=2, t=3).fit(clumps)
    repeated = int(np.sum(model.eigenvalues_ > 1.0 - 1e-10))
    inside = float(np.max(model.potential_distances_[:5, :5]))
    across = model.potential_distances_[:5, 5:]
    between = (float(across.min()), float(across.max()))
    print(f"broken: {len(cases)} bad calls all raise ValueError -- "
          f"{', '.join(name for name, _ in cases)}. A graph of two clumps a hundred "
          f"units apart with two neighbours each raises nothing, and should not be "
          f"trusted: the unit eigenvalue has multiplicity {repeated}, so the walk "
          f"never crosses between the clumps and P^t is block diagonal. Nothing inside "
          f"a clump is further than {inside:.1f} from anything else in it, and "
          f"every one of the 25 pairs across the clumps sits between "
          f"{between[0]:.1f} and {between[1]:.1f}: a gap set by the floor of the "
          f"logarithm, since every cross-clump entry of P^3 underflows to the "
          f"floor, and not by anything the data says. Nothing downstream "
          f"can tell: MDS will place the clumps a sensible distance apart and the "
          f"plot will look like a result.")


def main() -> None:
    _demo_identity()
    _demo_sheet()
    _demo_time()
    _demo_broken()
    print("phate demo passed")


if __name__ == "__main__":
    main()
