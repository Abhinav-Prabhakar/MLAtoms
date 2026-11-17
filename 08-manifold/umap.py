"""UMAP — a fuzzy graph in, a low-dimensional layout out.

Chapter:   08-manifold
Companion: umap.md
Tier 1 · 5 variants: umap, fuzzy_simplicial, supervised, densmap, parametric

    w_{i|j} = exp(−(d_ij − ρ_i)/σ_i),   Σ_j w_{i|j} = log₂ k                (1)
    w_ij    = w_{i|j} + w_{j|i} − w_{i|j} w_{j|i}                           (2)
    q_ij    = (1 + a ‖y_i − y_j‖^{2b})^{−1}                                 (3)
    min     Σ_{i<j} w_ij log(w_ij/q_ij) + (1 − w_ij) log((1 − w_ij)/(1 − q_ij))  (4)

The method is two independent ideas glued together. The first is a *fuzzy
graph*: a k-nearest-neighbour graph whose edges carry membership strengths in
[0, 1], built so that every point sees the same amount of fuzzy mass no matter
how dense its neighbourhood is. The second is *cross entropy*: a layout is
scored by how well its own membership strengths q_ij agree with the graph's, and
the disagreement is punished in both directions — a graph edge with no layout
edge costs a lot, and a layout edge with no graph edge costs a lot too.

t-SNE has the same shape and a different second half. Its KL is asymmetric, so
it pays for pulling neighbours together and shrugs at spurious gaps; UMAP's
cross entropy charges for both, which is why its layouts hold global arrangement
better and its cluster sizes still mean nothing. The demo measures the pair on
the same swiss roll: UMAP keeps 0.833 of each point's eight true neighbours
against t-SNE's 0.857, and 0.834 of its forty against t-SNE's 0.769 — the
cross entropy's two-sided charge is what buys the mid-range. Neither recovers
the geometry: UMAP's distance correlation with the flat rectangle the roll
unrolls to is 0.534 and t-SNE's is 0.335, against Isomap's 0.9996.

The expensive part is not the objective but its support: (4) sums over all
n² pairs, and the graph touches only nk of them. `umap` samples the complement
instead of enumerating it; `fuzzy_simplicial` enumerates it, which is exact and
quadratic; `supervised` replaces the graph's weights with labels; `densmap` adds
a term that makes local density survive; `parametric` replaces the n free points
with a function of the input.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
VARIANTS = ("umap", "fuzzy_simplicial", "supervised", "densmap", "parametric")


# --------------------------------------------------------------------------- #
# The fuzzy graph
# --------------------------------------------------------------------------- #
def _pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distances via the Gram identity, `(n, n)`."""
    sq = np.einsum("ij,ij->i", X, X)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))


def _sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared distances, `(n, n)`, diagonal zero."""
    sq = np.einsum("ij,ij->i", X, X)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def _knn(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The `k` nearest neighbours of every row, distances sorted ascending.

    `D` must have an infinite diagonal, so a point is never its own neighbour.
    Returns `(distances, indices)`, both `(n, k)`, columns in ascending distance.
    """
    n = D.shape[0]
    k = int(min(k, n - 1))
    idx = np.argpartition(D, k, axis=1)[:, :k]
    dk = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dk, axis=1)
    return np.take_along_axis(dk, order, axis=1), np.take_along_axis(idx, order, axis=1)


def _smooth_knn(dk: np.ndarray) -> np.ndarray:
    """Membership strengths `w_{i|j}`, `(n, k)`, each row summing to `log₂ k`.

    (1), with the two knobs the paper introduces. `ρ_i` is the distance to the
    nearest neighbour: subtracting it makes the nearest neighbour carry
    membership exactly 1, so every point is connected to the graph with full
    strength and the layout has no isolated points to invent a position for.
    `σ_i` is then bisected until the row sums to `log₂ k`, which is the fuzzy
    analogue of t-SNE's perplexity: a row in a dense region gets a narrow kernel,
    a row in a sparse one gets a wide kernel, and both carry the same mass.

    The exponent is non-positive because `d_ij ≥ ρ_i`, so no overflow is
    possible and no shift is needed. The bracket on `σ` starts at `[0, ∞)` and
    doubles until it is finite; the paper's schedule uses the same search.
    """
    n, k = dk.shape
    rho = dk[:, 0]
    diff = dk - rho[:, None]
    target = np.log2(k)
    sigma = np.ones(n)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    for _ in range(64):
        mass = np.exp(-diff / np.maximum(sigma[:, None], EPS)).sum(axis=1)
        too_wide = mass > target                    # needs a smaller sigma
        hi = np.where(too_wide, sigma, hi)
        lo = np.where(too_wide, lo, sigma)
        sigma = np.where(np.isfinite(hi), 0.5 * (lo + hi), 2.0 * sigma)
    return np.exp(-diff / np.maximum(sigma[:, None], EPS))


def _fuzzy_graph(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """The symmetric fuzzy graph of (2), `(n, n)`, dense but mostly zero.

    The union is the probabilistic t-conorm `a + b − ab`, not the average. It is
    the membership of "i or j is a neighbour": if either direction is certain the
    union is certain, which is what makes a one-sided edge — j in i's list but
    not the reverse — survive as a full-strength edge rather than a half one.
    """
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    dk, idx = _knn(D, n_neighbors)
    memberships = _smooth_knn(dk)
    A = np.zeros_like(D)
    np.put_along_axis(A, idx, memberships, axis=1)
    return A + A.T - A * A.T


# --------------------------------------------------------------------------- #
# The layout kernel
# --------------------------------------------------------------------------- #
def _ab_params(min_dist: float, spread: float) -> tuple[float, float]:
    """Least-squares fit of `(a, b)` in `q = (1 + a r^{2b})^{-1}`.

    The target is the piecewise curve the paper prescribes: membership 1 for
    `r < min_dist`, and `exp(−(r − min_dist)/spread)` beyond it. `min_dist` is
    the smallest distance the layout is willing to give two neighbours — the
    knob that decides whether clusters come out as dots or as blobs — and
    `spread` is the length scale over which the kernel decays.

    Gauss–Newton on `log a, log b`, whose positivity is then automatic. For the
    default `(0.1, 1.0)` the fit lands on `a = 1.5769`, `b = 0.8951`, which are
    the constants in `umap-learn`; the demo checks that.
    """
    x = np.linspace(0.05 * spread, 3.0 * spread, 256)
    target = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))

    def residual(p: np.ndarray) -> np.ndarray:
        a, b = np.exp(p)
        return 1.0 / (1.0 + a * x ** (2.0 * b)) - target

    p = np.log(np.array([1.5, 0.9]))
    for _ in range(64):
        r = residual(p)
        J = np.empty((x.size, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = 1e-6
            J[:, j] = (residual(p + step) - r) / 1e-6
        move = np.linalg.lstsq(J, r, rcond=None)[0]
        p = np.clip(p - np.clip(move, -0.5, 0.5), [-4.0, -3.0], [4.0, 3.0])
    a, b = np.exp(p)
    return float(a), float(b)


def _cross_entropy(W: np.ndarray, Y: np.ndarray, a: float, b: float) -> float:
    """(4), evaluated densely. The objective itself, not a surrogate."""
    d2 = _sq_dists(Y)
    q = 1.0 / (1.0 + a * np.maximum(d2, 0.0) ** b)
    q = np.clip(q, EPS, 1.0 - EPS)          # a coincident pair has q = 1
    iu = np.triu_indices(W.shape[0], k=1)
    w = W[iu]
    q = q[iu]
    pos = w > 0.0
    neg = w < 1.0
    total = 0.0
    if pos.any():
        total += float((w[pos] * np.log(w[pos] / q[pos])).sum())
    if neg.any():
        total += float(((1.0 - w[neg])
                        * np.log((1.0 - w[neg]) / (1.0 - q[neg]))).sum())
    return total


def _pair_gradients(rows: np.ndarray, cols: np.ndarray, w: np.ndarray,
                    Y: np.ndarray, a: float, b: float,
                    clip: float | None = 4.0) -> np.ndarray:
    """Gradient of (4) for a batch of pairs, `(m, 2)`, clipped coordinatewise.

    For a pair at squared distance `d2` the two derivatives are

        d/d(d2) of −w log q        =  w a b d2^{b−1} / (1 + a d2^b)
        d/d(d2) of −(1−w) log(1−q) = −(1−w) b / (d2 (1 + a d2^b))

    and the chain rule with `d(d2)/dy_i = 2 (y_i − y_j)` turns them into the
    two coefficients below. The sign convention is the *force*, so the
    attraction coefficient is negative: it points from i towards j.

    The `max(d2, 1e-12)` guards `d2^{b−1}` at coincident points, where `b < 1`
    makes the derivative infinite; the 0.001 in the repulsion is the
    implementation's standard floor, which keeps a pair that has already
    collapsed from being pushed with unbounded force. Each coordinate is clipped
    to ±4 — the only regularisation the optimiser has.
    """
    d = Y[rows] - Y[cols]
    d2 = np.einsum("ij,ij->i", d, d)
    denom = 1.0 + a * np.maximum(d2, 0.0) ** b
    safe = np.maximum(d2, 1e-12)
    pull = -2.0 * w * a * b * safe ** (b - 1.0) / denom
    push = 2.0 * (1.0 - w) * b / (safe * denom)
    g = (pull + push)[:, None] * d
    return g if clip is None else np.clip(g, -clip, clip)


def _apply(Y: np.ndarray, idx: np.ndarray, g: np.ndarray, scale: float) -> None:
    """`Y[idx] += scale * g`, accumulating when an index repeats.

    `bincount` rather than `np.add.at`: the batched update touches every point
    once per chunk, and fancy assignment with duplicate indices would silently
    drop all but the last contribution.
    """
    for j in range(Y.shape[1]):
        Y[:, j] += scale * np.bincount(idx, weights=g[:, j], minlength=Y.shape[0])


def _dense_force(W: np.ndarray, Y: np.ndarray, a: float, b: float) -> np.ndarray:
    """The same gradient over all `n²` pairs at once, as one matrix product.

    `force_i = Σ_j c_ij (y_i − y_j)` is `y_i Σ_j c_ij − (C @ Y)[i]`, so the
    whole batch costs one product with the coefficient matrix. This is the exact
    objective of (4) and the reference the sampled optimiser is measured
    against; it is what `fuzzy_simplicial` runs.
    """
    d2 = _sq_dists(Y)
    denom = 1.0 + a * np.maximum(d2, 0.0) ** b
    safe = np.maximum(d2, 1e-12)
    C = (W * (-2.0 * a * b * safe ** (b - 1.0) / denom)
         + (1.0 - W) * (2.0 * b / (safe * denom)))
    np.fill_diagonal(C, 0.0)
    return np.clip(Y * C.sum(axis=1)[:, None] - C @ Y, -4.0, 4.0)


# --------------------------------------------------------------------------- #
# Initialisation and the density term
# --------------------------------------------------------------------------- #
def _spectral_init(W: np.ndarray, n_components: int) -> np.ndarray:
    """Spectral embedding of the fuzzy graph, scaled to a spread of ±10.

    The eigenvectors of the normalised affinity `D^{-1/2} W D^{-1/2}` are the
    low-frequency modes of the graph — the same objects Laplacian eigenmaps
    computes — and starting there is cheap and deterministic, where a random
    start is only deterministic once you are careful about the generator.

    Each eigenvector is sign-fixed by making its largest absolute entry
    positive, since `v` and `−v` are equally valid and eigh may return either.
    Without this the layout differs between two runs of the same seed on
    machines with different LAPACK builds.
    """
    d = np.maximum(W.sum(axis=1), EPS)
    dinv = 1.0 / np.sqrt(d)
    M = dinv[:, None] * W * dinv[None, :]
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    take = order[1:n_components + 1]        # drop the constant eigenvector
    V = vecs[:, take]
    signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    V = V * np.where(signs == 0.0, 1.0, signs)
    scale = float(np.abs(V).max())
    return V / max(scale, EPS) * 10.0


def _density_force(Y: np.ndarray, ranks: np.ndarray, k: int,
                   weight: float) -> np.ndarray:
    """Gradient of the density term: match the local radii rank for rank.

    DensMAP's insight is that the layout's metric is already being stretched to
    fit the graph, and the stretch is what destroys density. Its correction is
    a second objective on the *radius* of each point's neighbourhood.

    The published form regresses the embedding's log radius on the input's and
    penalises the residual. Implemented that way the objective has a degenerate
    optimum — the regression's best slope is zero, so flattening every radius
    drives the penalty to zero, and the demo measured the density correlation
    going from −0.14 to −0.01 while the cross entropy rose: uniform density, not
    preserved density. What is used here is the 1-D optimal-transport form of
    the same idea: sort the embedding's log radii and give each point the radius
    that currently sits at its rank in the input. The multiset of radii is then
    driven to the input's multiset up to a monotone transform, which is what
    "the relative densities are preserved" means, and there is no flat solution
    that satisfies it.
    """
    D = _pairwise(Y)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    near = np.take_along_axis(D, idx, axis=1)
    radius = near.mean(axis=1)
    log_r = np.log(np.maximum(radius, EPS))
    target = np.sort(log_r)[ranks]
    c = -weight * (log_r - target) / (np.maximum(radius, EPS) * k)
    step = Y[:, None, :] - Y[idx]
    unit = step / np.maximum(near, EPS)[:, :, None]
    g = c[:, None, None] * unit
    force = np.zeros_like(Y)
    np.add.at(force, np.repeat(np.arange(Y.shape[0]), k), g.reshape(-1, 2))
    np.add.at(force, idx.ravel(), -g.reshape(-1, 2))
    return force


# --------------------------------------------------------------------------- #
# The parametric map
# --------------------------------------------------------------------------- #
class _MLP:
    """One tanh hidden layer and a linear readout, fitted by Adam on `‖f(x)−y‖²`.

    This is the `parametric` variant's function, and it is deliberately small:
    the thing being tested is that *some* fixed-size map can carry a UMAP layout
    to new points, not that a large network is needed to do it.
    """

    def __init__(self, n_in: int, hidden: int, n_out: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((hidden, n_in)) / np.sqrt(n_in)
        self.b1 = np.zeros(hidden)
        self.W2 = np.zeros((n_out, hidden))
        self.b2 = np.zeros(n_out)
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """The learned map, applied to any `(m, n_in)` array of points."""
        A = np.tanh((X - self.mu) / self.sd @ self.W1.T + self.b1)
        return A @ self.W2.T + self.b2

    def fit(self, X: np.ndarray, Y: np.ndarray, steps: int = 600,
            learning_rate: float = 1e-2) -> None:
        """Least-squares readout first, then Adam on the residuals.

        With `W1` random, `tanh(W1 x)` is already a rich feature map and the
        best linear function of it is one `lstsq`. Adam then moves the features
        themselves. Fine-tuning the net directly on the embedding objective —
        the obvious thing to try — is scale-blind and collapses the layout.
        """
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + EPS
        Xs = (X - self.mu) / self.sd
        n = X.shape[0]
        A = np.tanh(Xs @ self.W1.T + self.b1)
        self.b2 = Y.mean(axis=0)
        self.W2 = np.linalg.lstsq(A, Y - self.b2, rcond=None)[0].T
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for step in range(steps):
            A = np.tanh(Xs @ self.W1.T + self.b1)
            residual = 2.0 * (A @ self.W2.T + self.b2 - Y) / n
            dA = (residual @ self.W2) * (1.0 - A * A)
            grads = [dA.T @ Xs, dA.sum(axis=0), residual.T @ A,
                     residual.sum(axis=0)]
            for i, g in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                mhat = m[i] / (1.0 - 0.9 ** (step + 1))
                vhat = v[i] / (1.0 - 0.999 ** (step + 1))
                params[i] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class UMAP:
    """Uniform Manifold Approximation and Projection.

    `UMAP(n_components=2, *, variant="umap", n_neighbors=15, min_dist=0.1,
    spread=1.0, n_epochs=200, learning_rate=1.0, negative_rate=5,
    dens_lambda=0.1, hidden=64, seed=0)` with `fit(X, y=None)`,
    `fit_transform(X, y=None)` and, for `parametric` only, `transform(X_new)`.

    Attributes after `fit`: `embedding_` `(n, n_components)`, `graph_` (the
    fuzzy weights (2)), `a_`, `b_`, `ce_` (the cross entropy (4) of the final
    layout), `history_` (sampled through the run) and, for `parametric`,
    `teacher_` — the layout the network was fitted to.

    `n_neighbors` is the size of the fuzzy neighbourhood and the parameter that
    changes the answer. `min_dist` sets how tightly the layout may pack
    neighbours into a cluster, `negative_rate` is the number of non-edges
    sampled per edge per epoch, `dens_lambda` weights the density term, and
    `hidden` is the network width for `parametric`.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "umap",
                 n_neighbors: int = 15, min_dist: float = 0.1,
                 spread: float = 1.0, n_epochs: int = 200,
                 learning_rate: float = 1.0, negative_rate: int = 5,
                 dens_lambda: float = 10.0, hidden: int = 64,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if min_dist < 0.0 or spread <= 0.0:
            raise ValueError("min_dist must be non-negative and spread positive")
        if n_epochs < 1 or negative_rate < 1:
            raise ValueError("n_epochs and negative_rate must be positive")
        self.n_components = int(n_components)
        self.variant = variant
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.spread = float(spread)
        self.n_epochs = int(n_epochs)
        self.learning_rate = float(learning_rate)
        self.negative_rate = int(negative_rate)
        self.dens_lambda = float(dens_lambda)
        self.hidden = int(hidden)
        self.seed = int(seed)

    # -- graph ------------------------------------------------------------- #
    def _graph(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        """The fuzzy graph this variant optimises.

        `supervised` keeps only the support of the kNN graph but replaces each
        weight with the label agreement: 1 for the same class, 0 otherwise. The
        kNN support still matters — it is what stops the objective from being
        all pairs, which would be quadratic in a different way — but the
        geometry of `X` is used only to decide which pairs are eligible.
        """
        W = _fuzzy_graph(X, self.n_neighbors)
        if self.variant != "supervised":
            return W
        if y is None:
            raise ValueError("variant='supervised' needs labels: fit(X, y)")
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have one label per sample")
        same = y[:, None] == y[None, :]
        return np.where(same & (W > 0.0), 1.0, 0.0)

    # -- optimisation ------------------------------------------------------- #
    def _sampled_layout(self, W: np.ndarray, Y: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        """Descend on (4) by sampling the non-edges, one edge chunk at a time.

        The graph has `O(nk)` edges and the complement has `O(n²)`, so the
        complement is sampled rather than enumerated: `negative_rate` uniform
        draws per edge, rejected if they hit one of the edge's endpoints. The
        learning rate decays linearly to zero, which is what makes a fixed
        number of epochs converge instead of oscillate.

        The updates are applied in chunks of 64 edges rather than one edge at a
        time. That is not a cosmetic choice: summing a whole epoch's forces and
        applying them once makes a point with 17 incident edges move by 17
        times a single edge's step, and the first epoch then throws the layout
        to a standard deviation of 87 where the spectral initialisation had 10.
        At that scale the kernel is flat, the attraction is `1/r²`, and the run
        freezes: the demo originally measured an eight-neighbour overlap of 0.22
        this way. A chunk keeps the sequential coupling that a per-edge loop has
        while staying inside numpy.
        """
        n = W.shape[0]
        rows, cols = np.nonzero(np.triu(W, 1))
        w = W[rows, cols]
        self.history_ = [_cross_entropy(W, Y, self.a_, self.b_)]
        sample = max(1, self.n_epochs // 10)
        chunk = 64
        for epoch in range(self.n_epochs):
            lr = self.learning_rate * (1.0 - epoch / self.n_epochs)
            order = rng.permutation(rows.size)
            for start in range(0, rows.size, chunk):
                take = order[start:start + chunk]
                r, c, wt = rows[take], cols[take], w[take]
                neg = rng.integers(0, n, size=(r.size, self.negative_rate))
                neg = np.where(neg == r[:, None], (neg + 1) % n, neg)
                neg = np.where(neg == c[:, None], (neg + 1) % n, neg)
                heads = np.concatenate([r, np.repeat(r, self.negative_rate)])
                tails = np.concatenate([c, neg.ravel()])
                weights = np.concatenate(
                    [wt, np.zeros(r.size * self.negative_rate)])
                g = _pair_gradients(heads, tails, weights, Y, self.a_, self.b_)
                _apply(Y, heads, g, lr)
                _apply(Y, tails, -g, lr)
            if self.variant == "densmap":
                Y = Y + lr * _density_force(
                    Y, self._ranks, self.n_neighbors, self.dens_lambda)
            if (epoch + 1) % sample == 0:
                self.history_.append(_cross_entropy(W, Y, self.a_, self.b_))
        return Y

    def _dense_layout(self, W: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Full-batch descent on (4): the reference, at `n²` per iteration."""
        self.history_ = [_cross_entropy(W, Y, self.a_, self.b_)]
        sample = max(1, self.n_epochs // 10)
        lr = self.learning_rate * 0.05
        for epoch in range(self.n_epochs):
            Y = Y + lr * (1.0 - epoch / self.n_epochs) * _dense_force(
                W, Y, self.a_, self.b_)
            if (epoch + 1) % sample == 0:
                self.history_.append(_cross_entropy(W, Y, self.a_, self.b_))
        return Y

    # -- public API --------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> UMAP:
        """Fit the layout; returns `self`."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (samples, features), got shape {X.shape}")
        n = X.shape[0]
        if n <= self.n_neighbors:
            raise ValueError("n_neighbors must be smaller than the sample size")
        rng = np.random.default_rng(self.seed)
        W = self._graph(X, y)
        self.a_, self.b_ = _ab_params(self.min_dist, self.spread)
        D = _pairwise(X)
        np.fill_diagonal(D, np.inf)
        dk, _ = _knn(D, self.n_neighbors)
        log_radius = np.log(np.maximum(dk.mean(axis=1), EPS))
        self._ranks = np.argsort(np.argsort(log_radius))
        Y = _spectral_init(W, self.n_components)

        self.mlp_ = None
        self.teacher_ = None
        if self.variant == "fuzzy_simplicial":
            self.embedding_ = self._dense_layout(W, Y)
        elif self.variant == "parametric":
            self.teacher_ = self._sampled_layout(W, Y.copy(), rng)
            self.mlp_ = _MLP(X.shape[1], self.hidden, self.n_components,
                             seed=self.seed)
            self.mlp_.fit(X, self.teacher_)
            self.embedding_ = self.mlp_.forward(X)
        else:
            self.embedding_ = self._sampled_layout(W, Y, rng)

        self.graph_ = W
        self.ce_ = _cross_entropy(W, self.embedding_, self.a_, self.b_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed new points with the fitted network; `parametric` only.

        There is no such map for the other variants, and returning a nearest
        training point instead would be a different algorithm wearing this one's
        name.
        """
        if self.mlp_ is None:
            raise NotImplementedError(
                f"variant {self.variant!r} has no out-of-sample transform; "
                "refit, or use variant='parametric'"
            )
        return self.mlp_.forward(np.asarray(X, dtype=float))

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """`fit(X, y).embedding_`."""
        return self.fit(X, y).embedding_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
N_EPOCHS = 500


def _swiss_roll(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on the roll and their exact isometric coordinates `(s, h)`."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    h = 12.0 * rng.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    s = 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))
    return X, np.column_stack([s, h])


def _blobs(n_per: int, n_clusters: int, spread: float,
           seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters on a circle in 8 dimensions, with their labels."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
    centres = np.zeros((n_clusters, 8))
    centres[:, 0] = 1.5 * np.cos(angles)
    centres[:, 1] = 1.5 * np.sin(angles)
    X = np.vstack([centres[c] + spread * rng.standard_normal((n_per, 8))
                   for c in range(n_clusters)])
    return X, np.repeat(np.arange(n_clusters), n_per)


def _cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise distances between two sets of rows, `(m, k)`."""
    sq_a = np.einsum("ij,ij->i", A, A)
    sq_b = np.einsum("ij,ij->i", B, B)
    return np.sqrt(np.maximum(sq_a[:, None] + sq_b[None, :] - 2.0 * A @ B.T, 0.0))


def _distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Correlation of the two distance matrices — invariant to rigid motions."""
    iu = np.triu_indices(A.shape[0], k=1)
    return float(np.corrcoef(_pairwise(A)[iu], _pairwise(B)[iu])[0, 1])


def _overlap(A: np.ndarray, B: np.ndarray, k: int = 8) -> float:
    """Mean fraction of each point's `B`-nearest neighbours kept by `A`."""
    da, db = _pairwise(A), _pairwise(B)
    kept = []
    for i in range(A.shape[0]):
        left = set(np.argsort(da[i])[1:k + 1])
        right = set(np.argsort(db[i])[1:k + 1])
        kept.append(len(left & right) / k)
    return float(np.mean(kept))


def _purity(embedding: np.ndarray, labels: np.ndarray, k: int = 8) -> float:
    """Fraction of each point's `k` nearest layout neighbours sharing its label."""
    D = _pairwise(embedding)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    return float(np.mean(labels[idx] == labels[:, None]))


def _radius(X: np.ndarray, k: int) -> np.ndarray:
    """Mean distance to the `k` nearest neighbours, one per row."""
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return np.take_along_axis(D, idx, axis=1).mean(axis=1)


def _demo_umap(X: np.ndarray, truth: np.ndarray) -> tuple[UMAP, np.ndarray]:
    """Neighbourhoods are kept, mid-range structure too, distances are not."""
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pca = U[:, :2] * S[:2]

    rows = []
    for neighbours in (5, 15, 50):
        fit = UMAP(2, n_neighbors=neighbours, n_epochs=N_EPOCHS,
                   seed=1).fit(X)
        rows.append((neighbours, _overlap(fit.embedding_, truth),
                     _overlap(fit.embedding_, truth, 40),
                     _distance_correlation(fit.embedding_, truth),
                     fit.history_[0], fit.ce_))
    best = max(rows, key=lambda r: r[1])
    reference = UMAP(2, n_epochs=N_EPOCHS, seed=1).fit(X)

    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    memberships = _smooth_knn(_knn(D, 15)[0])
    assert np.abs(memberships.sum(axis=1) - np.log2(15)).max() < 1e-6
    assert abs(reference.a_ - 1.5769) < 1e-3, reference.a_
    assert abs(reference.b_ - 0.8951) < 1e-3, reference.b_
    assert _overlap(pca, truth) < 0.55
    assert best[1] > 0.8, rows
    assert all(r[4] > r[5] for r in rows), rows
    assert rows[2][3] < 0.4, rows
    assert rows[0][3] > rows[2][3], rows

    print(
        f"umap: {X.shape[0]} points on a swiss roll, which is exactly isometric "
        f"to a flat rectangle. The leading two principal components keep "
        f"{_overlap(pca, truth):.3f} of each point's eight true nearest "
        f"neighbours; the fuzzy graph's layout keeps {best[1]:.3f} at its best "
        f"k and {rows[1][1]:.3f} at the default k = 15. Neighbourhood size, "
        f"overlap at k = 8, overlap at k = 40 and distance correlation with the "
        f"rectangle are "
        + ", ".join(f"{k}→{a:.3f}/{b:.3f}/{c:.3f}" for k, a, b, c, _, _ in rows)
        + f". The mid-range survives better than it does under t-SNE: at the "
        f"default the forty-neighbour overlap is {rows[1][2]:.3f}, and unlike a "
        f"KL the cross entropy charges for a spurious edge as well as a missing "
        f"one. The kernel is the fitted (a, b) = ({reference.a_:.4f}, "
        f"{reference.b_:.4f}), the published pair for min_dist = 0.1. Distances "
        f"are another matter. Correlation with the rectangle is "
        f"{rows[0][3]:.3f} at k = 5 and falls to {rows[2][3]:.3f} at k = 50, so "
        f"larger neighbourhoods buy a more global picture at the cost of the "
        f"local one — but the best of them is nowhere near Isomap's 0.9996. The "
        f"cross entropy falls from {rows[1][4]:.0f} to {rows[1][5]:.0f} over "
        f"{N_EPOCHS} epochs"
    )
    return reference, 15


def _demo_sampling(X: np.ndarray, sampled: UMAP, neighbours: int) -> None:
    """The dense objective and the sampled one land in the same place."""
    dense = UMAP(2, variant="fuzzy_simplicial", n_neighbors=neighbours,
                 n_epochs=N_EPOCHS, seed=1).fit(X)
    W, Y = dense.graph_, dense.embedding_
    a, b = dense.a_, dense.b_
    exact = _dense_force(W, Y, a, b)
    rows, cols = np.triu_indices(Y.shape[0], k=1)
    g = _pair_gradients(rows, cols, W[rows, cols], Y, a, b, clip=None)
    summed = np.zeros_like(Y)
    np.add.at(summed, rows, g)
    np.add.at(summed, cols, -g)

    rng = np.random.default_rng(0)
    scale = max(1.0, float(np.abs(exact).max()))
    worst = 0.0
    for _ in range(6):
        i = int(rng.integers(Y.shape[0]))
        j = int(rng.integers(Y.shape[1]))
        up = Y.copy()
        up[i, j] += 1e-7
        down = Y.copy()
        down[i, j] -= 1e-7
        fd = -(_cross_entropy(W, up, a, b)
               - _cross_entropy(W, down, a, b)) / 2e-7
        worst = max(worst, abs(fd - exact[i, j]) / scale)

    pairs = X.shape[0] * (X.shape[0] - 1) // 2
    edges = int((np.triu(W, 1) > 0).sum())
    terms = edges * (1 + dense.negative_rate)
    assert np.abs(summed - exact).max() < 1e-3 * scale, "pair sum disagrees"
    assert worst < 1e-4, worst
    agree = _overlap(dense.embedding_, sampled.embedding_)
    assert agree > 0.8, agree
    assert abs(dense.ce_ - sampled.ce_) < 0.1 * sampled.ce_

    print(
        f"fuzzy_simplicial: the same objective without sampling, as one product "
        f"with the coefficient matrix. Two independent implementations of the "
        f"gradient agree to {np.abs(summed - exact).max():.1e} and six "
        f"finite-difference probes agree to {worst:.1e}, so the matrix form is "
        f"the derivative of {pairs} pair terms and not a look-alike. The dense "
        f"solver reaches cross entropy {dense.ce_:.0f} where the sampled one "
        f"reaches {sampled.ce_:.0f}, and the two layouts share {agree:.3f} of "
        f"their eight-neighbourhoods — the objective is the same, the "
        f"trajectory is not, and near-equal minima are a fact of this landscape "
        f"rather than a bug. The price is the point: {pairs} pairs per "
        f"iteration against {terms} sampled terms per epoch, which is "
        f"{pairs / terms:.1f} times fewer for the sampler, and the gap widens "
        f"with n because the graph stays at O(nk) while the complement grows"
    )


def _demo_supervised() -> None:
    """Labels turn the graph's weights into targets, and the classes separate."""
    X, labels = _blobs(50, 5, 1.1, seed=3)
    plain = UMAP(2, n_epochs=N_EPOCHS, seed=1).fit(X)
    guided = UMAP(2, variant="supervised", n_epochs=N_EPOCHS, seed=1).fit(X, labels)
    loose = _purity(plain.embedding_, labels)
    tight = _purity(guided.embedding_, labels)

    assert tight > loose + 0.3, (loose, tight)
    assert tight > 0.9, tight

    print(
        f"supervised: five 8-dimensional Gaussian clusters whose centres are "
        f"1.5 apart and whose standard deviation is 1.1, so the plain kNN graph "
        f"mixes them. The unsupervised layout keeps {loose:.3f} of each point's "
        f"eight layout neighbours inside its own class. Replace each edge's "
        f"fuzzy weight with the label agreement — 1 inside a class, 0 across — "
        f"and the same optimiser separates them at {tight:.3f}. Nothing about "
        f"the geometry changed; only the target the cross entropy is measured "
        f"against. The cost is the geometry itself: the supervised layout is a "
        f"picture of the labels, so its distances say nothing about the data"
    )


def _demo_densmap() -> None:
    """A density term makes local radii survive, at a cross-entropy price."""
    rng = np.random.default_rng(7)
    t = rng.beta(0.5, 2.0, 300)
    X = np.column_stack([t, t ** 2, np.sin(2.0 * np.pi * t),
                         np.cos(2.0 * np.pi * t), t ** 3])
    plain = UMAP(2, n_epochs=N_EPOCHS, seed=1).fit(X)
    dense = UMAP(2, variant="densmap", dens_lambda=10.0, n_epochs=N_EPOCHS,
                 seed=1).fit(X)
    before = float(np.corrcoef(np.log(_radius(X, 15)),
                               np.log(_radius(plain.embedding_, 15)))[0, 1])
    after = float(np.corrcoef(np.log(_radius(X, 15)),
                              np.log(_radius(dense.embedding_, 15)))[0, 1])

    assert after > before + 0.5, (before, after)
    assert dense.ce_ < 1.25 * plain.ce_, (dense.ce_, plain.ce_)

    print(
        f"densmap: 300 points on a curve, with the parameter drawn from a Beta "
        f"distribution so the sampling is dense at one end and sparse at the "
        f"other. Correlation between log input radius and log embedding radius "
        f"is {before:.3f} for plain UMAP — the layout spreads the points evenly "
        f"along the curve, which is exactly what it was asked to do and exactly "
        f"what destroys density — and {after:.3f} with the radius term. The "
        f"cross entropy moves from {plain.ce_:.0f} to {dense.ce_:.0f}, so the "
        f"density is not free but it is cheap here. The term is a 1-D optimal "
        f"transport of the radii rather than the published regression: the "
        f"regression's best slope on a uniformly dense layout is zero, so it is "
        f"satisfied by flattening every radius and the measured correlation "
        f"never rose above zero"
    )


def _demo_parametric() -> None:
    """A network reproduces the fit and places points it never saw."""
    X, labels = _blobs(40, 6, 0.3, seed=5)
    rng = np.random.default_rng(105)
    perm = rng.permutation(len(X))
    train = np.sort(perm[:180])
    held = np.sort(perm[180:])
    fit = UMAP(2, variant="parametric", n_epochs=N_EPOCHS, seed=1).fit(X[train])
    embedded = fit.embedding_
    projected = fit.transform(X[held])
    reproduce = _overlap(embedded, fit.teacher_)
    teacher = _purity(fit.teacher_, labels[train])
    nearest = np.argsort(_cross(X[held], X[train]), axis=1)[:, :10]
    baseline = embedded[nearest].mean(axis=1)

    def purity(points: np.ndarray) -> float:
        return float(np.mean(
            labels[train][np.argmin(_cross(points, embedded), axis=1)]
            == labels[held]))

    net = purity(projected)
    base = purity(baseline)
    assert teacher > 0.9, teacher
    assert reproduce > 0.9, reproduce
    assert net > 0.7, net
    assert base >= net, (base, net)

    print(
        f"parametric: {train.size} points in six 8-dimensional clusters, and a "
        f"two-layer tanh network fitted to the layout rather than to the data. "
        f"The teacher separates the clusters at {teacher:.3f} and the network "
        f"reproduces it on the training points at {reproduce:.3f} "
        f"eight-neighbour overlap. Transform the {held.size} held-out points and "
        f"{net:.3f} land nearest their own cluster. The baseline is a 10-neighbour "
        f"average of the fitted coordinates and it scores {base:.3f}. That is "
        f"the honest shape of this variant: the map can be almost exact on the "
        f"points it was fitted to and still generalise worse than averaging "
        f"neighbours, because UMAP's layout is not a smooth function of x — "
        f"adding capacity makes the reproduction better and the generalisation "
        f"worse"
    )


def main() -> None:
    X, truth = _swiss_roll(250, seed=1)
    sampled, neighbours = _demo_umap(X, truth)
    _demo_sampling(X, sampled, neighbours)
    _demo_supervised()
    _demo_densmap()
    _demo_parametric()
    print("umap demo passed")


if __name__ == "__main__":
    main()
