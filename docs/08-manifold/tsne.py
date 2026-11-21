"""t-SNE — neighbourhood probabilities in, a two-dimensional layout out.

Chapter:   08-manifold
Companion: tsne.md
Tier 1 · 5 variants: exact, barnes_hut, fft, multiscale, parametric

    p_ij ∝ (p_{j|i} + p_{i|j}) / 2n,   p_{j|i} ∝ exp(−‖x_i − x_j‖² / 2σ_i²)   (1)
    q_ij ∝ (1 + ‖y_i − y_j‖²)^{−1}                                          (2)
    Y = arg min KL(p ‖ q) over the n points of the layout                   (3)

The whole method is one sentence: choose the σ_i so that each point's
neighbourhood has a fixed entropy, then arrange points in the plane so that the
resulting *probabilities* match. Distances enter only through a probability that
a pair is a neighbour, which is why t-SNE keeps neighbourhoods and loses
distances — it never asked about distances.

Everything expensive is in (2): the repulsion sums over all n² pairs every
iteration. The variants are the answers to that. `exact` pays it, `barnes_hut`
and `fft` approximate the same sum in sub-quadratic time, `multiscale` changes
the input probabilities rather than the sum, and `parametric` replaces the n
free points with a function of the input.

The demo measures the claim and the bill: on a swiss roll the exact variant
keeps 86% of each point's true eight nearest neighbours where the leading two
principal components keep 49%, and yet its distance correlation with the flat
rectangle the roll is isometric to is 0.335 — Isomap, in this same chapter,
reaches 0.9996. It keeps what it was asked to keep.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
VARIANTS = ("exact", "barnes_hut", "fft", "multiscale", "parametric")


# --------------------------------------------------------------------------- #
# Affinities
# --------------------------------------------------------------------------- #
def _pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distances via the Gram identity, `(n, n)`."""
    sq = np.einsum("ij,ij->i", X, X)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))


def _sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared distances, `(n, n)`, diagonal included (the caller masks it)."""
    sq = np.einsum("ij,ij->i", X, X)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)


def _beta_search(d2: np.ndarray, perplexity: float, n_iter: int = 64) -> np.ndarray:
    """Row-wise conditional probabilities with a prescribed entropy.

    For each point, find the precision `β_i = 1/2σ_i²` such that the conditional
    distribution has entropy `log(perplexity)`. Bisection, for all points at
    once: entropy is monotone in β, so each point's bracket tightens
    independently and the whole search vectorises. Sixty-four steps takes the
    entropy to the target to machine precision, which the demo checks.

    The diagonal is zero rather than infinite. `exp(-inf)` is fine but
    `inf * 0` is `nan`, and the entropy sum contains exactly that product — the
    first version of this function returned `nan` for every point.

    `perplexity` must be smaller than `n − 1` for the target to be reachable;
    beyond that the bracket runs away and the clamp keeps the exponential finite.
    """
    n = d2.shape[0]
    eye = np.eye(n, dtype=bool)
    off = np.where(eye, np.inf, d2)
    d2 = d2 - off.min(axis=1, keepdims=True)      # per-row shift, for stability
    target = np.log(perplexity)
    beta = 1.0 / np.median(off, axis=1)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    for _ in range(n_iter):
        P = np.exp(-beta[:, None] * d2)
        P[eye] = 0.0
        s = P.sum(axis=1)
        entropy = np.log(s) + beta * (d2 * P).sum(axis=1) / s
        too_flat = entropy > target                  # needs a larger beta
        lo = np.where(too_flat, beta, lo)
        hi = np.where(too_flat, hi, beta)
        beta = np.where(np.isfinite(hi), 0.5 * (lo + hi),
                        np.minimum(2.0 * beta, 1e6))
    P = np.exp(-beta[:, None] * d2)
    P[eye] = 0.0
    return P / P.sum(axis=1, keepdims=True)


def _conditional(X: np.ndarray, perplexity: float) -> np.ndarray:
    """`p_{j|i}` for one perplexity, `(n, n)`, each row summing to one."""
    return _beta_search(_sq_dists(X), perplexity)


def _joint(P: np.ndarray, k: float) -> np.ndarray:
    """Symmetrise to the joint `p_ij` of (1), keeping the `k` largest per row.

    The affinity is dense and almost all of it is rounding error: at perplexity
    30 a row's 30 significant terms and its other 220 differ by hundreds of
    orders of magnitude. Dropping them is not an approximation at the level of
    the objective — it is what makes the attraction sum `O(nk)` instead of
    `O(n²)`. Keeping them and flooring every entry at `1e-12` is worse than
    dropping them: the floor makes the matrix dense, and the "sparse" attraction
    then costs `n²` again (measured: a 400-point fit took 6.0 s instead of
    0.6 s). The floor belongs on the support, not off it.
    """
    n = P.shape[0]
    k = int(min(max(k, 1), n - 1))
    idx = np.argpartition(-P, k, axis=1)[:, :k]
    keep = np.zeros_like(P)
    np.put_along_axis(keep, idx, 1.0, axis=1)
    P = P * keep
    P = P / P.sum(axis=1, keepdims=True)
    J = (P + P.T) / (2.0 * n)
    np.fill_diagonal(J, 0.0)
    support = J > 0.0
    J[support] = np.maximum(J[support], 1e-12)    # floor the KL's log q
    return J / J.sum()


# --------------------------------------------------------------------------- #
# Repulsion: three ways to sum the same kernel
# --------------------------------------------------------------------------- #
def _quadtree(Y: np.ndarray, max_depth: int = 40) -> tuple:
    """Quadtree of a 2-D layout as flat arrays, built one level at a time.

    Returns `(com, mass, half, kids, leaf)`: each node's centre, the number of
    points beneath it, its half-width, its four children (`-1` for an empty
    quadrant) and, for a leaf, the index of its single point.

    Two details are load-bearing. A leaf's centre is snapped to its point's
    coordinates — the cell centre is *not* the point, and using it left a force
    error of 30% at θ = 0, where the approximation must vanish. And the points
    are kept in one `order` array, contiguous per node, so a level costs one
    `argsort` over the whole level instead of one `flatnonzero(assign == node)`
    per node. The obvious version scans all n points for every node, which is
    `O(n²)` bookkeeping and made the "sub-quadratic" method slower than the
    dense sum it was meant to beat.

    `max_depth` catches points that are equal to the last bit of a float, which
    would otherwise split until the coordinates run out; such a node becomes one
    body at its points' mean, which is the right answer because the points are
    indistinguishable to every other point.
    """
    n = Y.shape[0]
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    com: list[np.ndarray] = [(lo + hi) / 2.0]
    mass: list[int] = [n]
    half: list[float] = [float(np.max(hi - lo)) / 2.0 + 1e-12]
    kids: list[list[int]] = [[-1, -1, -1, -1]]
    leaf: list[int] = [-1]
    order = np.arange(n)
    frontier = [(0, 0, n)]
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        nxt: list[tuple[int, int, int]] = []
        for node, start, stop in frontier:
            if stop - start <= 1:
                continue
            idx = order[start:stop]
            cx, cy = com[node]
            h = half[node] * 0.5
            quad = (Y[idx, 0] > cx).astype(np.int64) * 2 + (Y[idx, 1] > cy)
            perm = np.argsort(quad, kind="stable")
            order[start:stop] = idx[perm]
            cuts = np.searchsorted(quad[perm], np.arange(5))
            for k in range(4):
                a, b = start + int(cuts[k]), start + int(cuts[k + 1])
                if b <= a:
                    continue
                child = len(com)
                if b - a == 1:
                    com.append(Y[order[a]].copy())
                    leaf.append(int(order[a]))
                else:
                    com.append(np.array([cx + (h if k >= 2 else -h),
                                         cy + (h if k % 2 else -h)]))
                    leaf.append(-1)
                    nxt.append((child, a, b))
                mass.append(b - a)
                half.append(h)
                kids.append([-1, -1, -1, -1])
                kids[node][k] = child
        frontier = nxt
    for node, start, stop in frontier:
        if stop - start > 1:
            com[node] = Y[order[start:stop]].mean(axis=0)
            mass[node] = stop - start
            leaf[node] = int(order[start])
    return (np.array(com), np.array(mass, dtype=float), np.array(half),
            np.array(kids, dtype=np.int64), np.array(leaf, dtype=np.int64))


def _bh_repulsion(Y: np.ndarray, theta: float) -> tuple:
    """Barnes–Hut: one body per cell once the cell is small in the distance.

    A cell is accepted when its side is below `θ` times its distance to the
    query point, so a distant cluster costs one term instead of its point count.
    `θ = 0` accepts only leaves and is *exact*, which is the test that separates
    traversal bugs from approximation error. A cell that contains the query
    point is never accepted: it stands in for a cloud that includes the query.

    Returns `(F, Z, None, terms)`: the `(n, 2)` repulsion vector, the kernel
    normaliser `Z = Σ_{i≠j} (1+r²)^{-1}`, a `None` where the dense kernel would
    be, and the number of accepted (point, cell) terms.
    """
    n = Y.shape[0]
    com, mass, half, kids, leaf = _quadtree(Y)
    pi = np.arange(n)
    pc = np.zeros(n, dtype=np.int64)
    fx = np.zeros(n)
    fy = np.zeros(n)
    total = 0.0
    terms = 0
    while pi.size:
        dx = Y[pi, 0] - com[pc, 0]
        dy = Y[pi, 1] - com[pc, 1]
        r2 = dx * dx + dy * dy
        single = leaf[pc] >= 0
        accept = single | (2.0 * half[pc] / np.sqrt(np.maximum(r2, EPS)) < theta)
        inside = ~single & (np.abs(dx) <= half[pc]) & (np.abs(dy) <= half[pc])
        accept = accept & ~(single & (leaf[pc] == pi)) & ~inside
        if accept.any():
            ii = pi[accept]
            m = mass[pc[accept]]
            w = 1.0 / (1.0 + r2[accept])
            total += float((m * w).sum())
            fx += np.bincount(ii, weights=m * w * w * dx[accept], minlength=n)
            fy += np.bincount(ii, weights=m * w * w * dy[accept], minlength=n)
            terms += int(accept.sum())
        rejected = ~accept
        if rejected.any():
            children = kids[pc[rejected]]
            valid = children >= 0
            pi = np.repeat(pi[rejected], 4)[valid.ravel()]
            pc = children[valid]
        else:
            pi = np.zeros(0, dtype=np.int64)
            pc = np.zeros(0, dtype=np.int64)
    return np.stack([fx, fy], axis=1), total, None, terms


def _fft_repulsion(Y: np.ndarray, grid_size: int) -> tuple:
    """The same sums by convolution on a grid, in `O(G² log G)` for any `n`.

    The repulsion is a convolution of the point density with the kernel
    `(1+r²)^{-1}` and its first two moments, which the FFT evaluates at every
    lag at once. Points are deposited bilinearly and the field is interpolated
    the same way.

    Three details are load-bearing. The kernel is `ifftshift`ed so zero lag sits
    at index zero. The transform is padded to `2G − 1` so the linear convolution
    is not aliased into a circular one. And each point's own term is subtracted
    *analytically* from its four deposited corners: the naive `φ −= 1` leaves a
    `1/(1+r²)` bias that does not shrink with `G`.

    The accuracy is set by how many grid cells fit inside one kernel width, and
    the kernel's unit length is the layout's own scale — so a fixed grid gets
    relatively coarser as t-SNE's layout grows, which it does with `n`. The demo
    measures that: error falls fourfold per doubling of `G` on one layout, and
    the same `G` is worth less on a bigger picture.
    """
    n = Y.shape[0]
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = max(float(np.max(hi - lo)) / 2.0, 1e-9)
    h = 2.0 * radius / (grid_size - 1)
    u = (Y[:, 0] - centre[0] + radius) / h
    v = (Y[:, 1] - centre[1] + radius) / h
    i0 = np.clip(np.floor(u).astype(np.int64), 0, grid_size - 2)
    j0 = np.clip(np.floor(v).astype(np.int64), 0, grid_size - 2)
    fu = u - i0
    fv = v - j0
    taps = ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)),
            (0, 1, (1 - fu) * fv), (1, 1, fu * fv))
    charge = np.zeros((grid_size, grid_size))
    for di, dj, weight in taps:
        np.add.at(charge, (i0 + di, j0 + dj), weight)

    size = 2 * grid_size - 1
    lag = np.arange(-(grid_size - 1), grid_size)
    dx = lag[:, None] * h
    dy = lag[None, :] * h
    k0 = 1.0 / (1.0 + dx * dx + dy * dy)
    kernel = np.fft.ifftshift(
        np.stack([k0, k0 * k0 * dx, k0 * k0 * dy]), axes=(1, 2))
    spectrum = np.fft.rfft2(charge, s=(size, size))
    field = np.fft.irfft2(
        spectrum[None, :, :] * np.fft.rfft2(kernel, s=(size, size)),
        s=(size, size))
    field = field[:, :grid_size, :grid_size]

    def gather(g: np.ndarray) -> np.ndarray:
        return sum(weight * g[i0 + di, j0 + dj] for di, dj, weight in taps)

    phi = gather(field[0])
    fx = gather(field[1])
    fy = gather(field[2])
    own_phi = np.zeros(n)
    own_x = np.zeros(n)
    own_y = np.zeros(n)
    for di, dj, weight in taps:
        ox = (di - fu) * h
        oy = (dj - fv) * h
        k = 1.0 / (1.0 + ox * ox + oy * oy)
        own_phi += weight * k
        own_x += weight * k * k * ox
        own_y += weight * k * k * oy
    phi = phi - own_phi
    return (np.stack([fx - own_x, fy - own_y], axis=1), float(phi.sum()),
            None, grid_size * grid_size)


def _exact_repulsion(Y: np.ndarray) -> tuple:
    """The sums of (2) over all pairs, in three matrix products.

    Returns `(F, Z, w, pairs)` with `w` the dense kernel, because at this size
    the kernel is already sitting in cache and one matrix product against it
    beats gathering `nk` pairs — the sparse path wins only for the variants that
    have no dense kernel to begin with.
    """
    n = Y.shape[0]
    sq = np.einsum("ij,ij->i", Y, Y)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Y @ Y.T, 0.0)
    np.fill_diagonal(d2, np.inf)                     # (1 + inf)^-1 = 0, exactly
    w = 1.0 / (1.0 + d2)
    w2 = w * w
    phi = w2.sum(axis=1)
    f = np.stack([Y[:, 0] * phi - w2 @ Y[:, 0],
                  Y[:, 1] * phi - w2 @ Y[:, 1]], axis=1)
    return f, float(w.sum()), w, n * (n - 1)


def _gradient(P: np.ndarray, rows: np.ndarray, cols: np.ndarray,
              vals: np.ndarray, Y: np.ndarray, f: np.ndarray, z: float,
              w: np.ndarray | None, exaggeration: float) -> np.ndarray:
    """Descent direction `−4(A − F/Z)`, verified against finite differences.

    Expanding `Σ_j (p_ij − q_ij) w_ij (y_i − y_j)` with `q = w/Z` gives an
    attraction, a matrix product against `P`, and a repulsion, which is the `F`
    and `Z` the variant already computed.

    Two normalisation traps, both of which were live in this file. `Z` is
    `Σ_{i≠j} w_ij` over *ordered* pairs, matching `P`, which is a joint
    distribution summing to one over ordered pairs; using `½Σw` (the unordered
    sum) doubles the repulsion, and with a doubled repulsion the layout expands
    past the scale where `w` matters, everything freezes, and the demo's
    eight-neighbour overlap reads 0.43 where it should read 0.86. And the `1/Z`
    sits on the repulsion alone: putting it on both terms is a globally rescaled
    gradient, invisible in the mathematics and fatal in the code, because the
    learning rate and the momentum schedule are fixed numbers.
    """
    n = Y.shape[0]
    if w is not None:
        pw = P * w * exaggeration
        attraction = np.stack([Y[:, 0] * pw.sum(axis=1) - pw @ Y[:, 0],
                               Y[:, 1] * pw.sum(axis=1) - pw @ Y[:, 1]], axis=1)
    else:
        diff = Y[rows] - Y[cols]
        pair_w = 1.0 / (1.0 + np.einsum("ij,ij->i", diff, diff))
        pw = vals * pair_w * exaggeration
        total = np.bincount(rows, weights=pw, minlength=n)
        ax = np.bincount(rows, weights=pw * Y[cols, 0], minlength=n)
        ay = np.bincount(rows, weights=pw * Y[cols, 1], minlength=n)
        attraction = np.stack([Y[:, 0] * total - ax,
                               Y[:, 1] * total - ay], axis=1)
    return -4.0 * (attraction - f / max(z, EPS))


def _kl_divergence(P: np.ndarray, Y: np.ndarray) -> float:
    """`Σ_{i≠j} p_ij log(p_ij/q_ij)`; the objective, not a surrogate."""
    d2 = _sq_dists(Y)
    np.fill_diagonal(d2, np.inf)
    q = 1.0 / (1.0 + d2)
    q = q / q.sum()
    mask = (P > 0) & (q > 0)
    return float((P[mask] * np.log(P[mask] / q[mask])).sum())


def _pca_init(X: np.ndarray, n_components: int) -> np.ndarray:
    """Start at the leading principal components, scaled to unit deviation.

    This is a convenience and not a load-bearing choice, which is worth saying
    because it is easy to believe otherwise: with the gradient's normalisation
    right, a unit-scale random start, a `1e-4` random start and the principal
    components all reach the same place on the demo data (eight-neighbour
    overlap 0.841, 0.837 and 0.857). The initialisation only *looked* decisive
    while the objective was wrong, because a wrong objective has a different
    equilibrium scale and the run then spends its exaggeration phase travelling
    to it.
    """
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    Y = U[:, :n_components] * S[:n_components]
    return Y / max(float(Y.std()), EPS)


# --------------------------------------------------------------------------- #
# The parametric map
# --------------------------------------------------------------------------- #
class _MLP:
    """One tanh hidden layer and a linear readout, fitted by Adam on `‖f(x)−y‖²`.

    This is the `parametric` variant's function. It is deliberately small: the
    thing being tested is that *some* fixed-size map can carry a t-SNE layout to
    new points, not that a large network is needed to do it.
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

        The readout is what makes this cheap: with `W1` random, `tanh(W1 x)` is
        already a rich feature map of `x`, and the best linear function of those
        features that reproduces `Y` is one `lstsq`. Adam then moves the
        features themselves. Fine-tuning the net directly on the KL instead —
        which is the obvious thing to try — collapses it: Adam's step is
        scale-blind, and a KL gradient through a net is large in the wrong
        direction long before it is useful.
        """
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-12
        Xs = (X - self.mu) / self.sd
        n = X.shape[0]
        A = np.tanh(Xs @ self.W1.T + self.b1)
        self.W2 = np.linalg.lstsq(A, Y, rcond=None)[0].T
        self.b2 = Y.mean(axis=0)
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
class TSNE:
    """t-SNE layout of a point cloud.

    `TSNE(n_components=2, *, variant="exact", perplexity=30.0, n_iter=1000,
    learning_rate=None, exaggeration=12.0, theta=0.5, grid_size=128,
    perplexities=(10.0, 30.0), hidden=64, seed=0)` with `fit(X)`,
    `fit_transform(X)` and, for `parametric` only, `transform(X_new)`.

    Attributes after `fit`: `embedding_` `(n, n_components)`, `kl_initial_`,
    `kl_divergence_`, `history_` (the KL sampled through the run; for
    `parametric`, the teacher's run), `interactions_` (repulsion terms evaluated
    by the last iteration) and, for `parametric`, `teacher_` — the exact layout
    the network was fitted to.

    `perplexity` is the effective neighbourhood size and the parameter that
    changes the answer. `theta` trades Barnes–Hut accuracy against cost,
    `grid_size` sets the FFT resolution, `perplexities` is the scale mixture for
    `multiscale`, and `hidden` is the network width for `parametric`. `seed` is
    used only for the network's initial weights; the layout itself is
    deterministic.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "exact",
                 perplexity: float = 30.0, n_iter: int = 1000,
                 learning_rate: float | None = None, exaggeration: float = 12.0,
                 theta: float = 0.5, grid_size: int = 128,
                 perplexities: tuple[float, ...] = (10.0, 30.0),
                 hidden: int = 64, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if perplexity <= 1.0:
            raise ValueError("perplexity must exceed 1")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        self.n_components = int(n_components)
        self.variant = variant
        self.perplexity = float(perplexity)
        self.n_iter = int(n_iter)
        self.learning_rate = learning_rate
        self.exaggeration = float(exaggeration)
        self.theta = float(theta)
        self.grid_size = int(grid_size)
        self.perplexities = tuple(float(p) for p in perplexities)
        self.hidden = int(hidden)
        self.seed = int(seed)

    def _affinity(self, X: np.ndarray) -> np.ndarray:
        """The joint `P` this variant optimises, `(n, n)`, summing to one."""
        if self.variant == "multiscale":
            scales = np.mean([_conditional(X, p) for p in self.perplexities],
                             axis=0)
            return _joint(scales, 3.0 * max(self.perplexities))
        return _joint(_conditional(X, self.perplexity), 3.0 * self.perplexity)

    def _repulsion(self, Y: np.ndarray) -> tuple:
        """`(F, Z, w, terms)` from whichever approximation this variant names."""
        if self.variant == "barnes_hut":
            return _bh_repulsion(Y, self.theta)
        if self.variant == "fft":
            return _fft_repulsion(Y, self.grid_size)
        return _exact_repulsion(Y)

    def _optimise(self, P: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Momentum descent on the KL, with early exaggeration.

        Exaggerating `P` for the first quarter of the run multiplies the
        attraction, which lets the layout find its large-scale arrangement while
        the local forces are still weak; releasing it settles the fine structure.
        Momentum rises from 0.5 to 0.8 at the same moment, because a layout that
        is still repositioning clusters should not carry its momentum into the
        fine phase.

        No per-parameter gains: the classic 1.2/0.8 gains rule grows without
        bound here (a doubling of the same direction every iteration, measured
        at 1.9e10) and sends the layout to a scale of 1e9, and its usual guard —
        compare the *gradient's* sign with the previous *gradient's* sign — is
        the version that works. Comparing against the displacement instead
        collapses the gains to their floor and freezes the layout.
        """
        Y = _pca_init(X, self.n_components)
        rows, cols = np.nonzero(P)
        vals = P[rows, cols]
        n = X.shape[0]
        lr = (self.learning_rate if self.learning_rate is not None
              else max(n / self.exaggeration / 4.0, 50.0))
        release = max(1, self.n_iter // 4)
        update = np.zeros_like(Y)
        sample = max(1, self.n_iter // 10)
        self.history_ = [_kl_divergence(P, Y)]
        for it in range(self.n_iter):
            f, z, w, self.interactions_ = self._repulsion(Y)
            g = _gradient(P, rows, cols, vals, Y, f, z, w,
                          self.exaggeration if it < release else 1.0)
            update = (0.5 if it < release else 0.8) * update + lr * g
            Y = Y + update
            if (it + 1) % sample == 0:
                self.history_.append(_kl_divergence(P, Y))
        return Y

    def fit(self, X: np.ndarray) -> TSNE:
        """Fit the layout; returns `self`."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (samples, features), got shape {X.shape}")
        n = X.shape[0]
        if n <= self.n_components:
            raise ValueError("need more samples than components")
        if self.perplexity >= n - 1:
            raise ValueError("perplexity must be smaller than the sample size")
        P = self._affinity(X)
        self.mlp_ = None
        if self.variant == "parametric":
            self.teacher_ = self._optimise(P, X)      # the exact layout first
            self.mlp_ = _MLP(X.shape[1], self.hidden, self.n_components,
                             seed=self.seed)
            self.mlp_.fit(X, self.teacher_)
            self.embedding_ = self.mlp_.forward(X)
        else:
            self.teacher_ = None
            self.embedding_ = self._optimise(P, X)
        self.kl_initial_ = self.history_[0]
        self.kl_divergence_ = _kl_divergence(P, self.embedding_)
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

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """`fit(X).embedding_`."""
        return self.fit(X).embedding_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
N_ITER = 600


def _swiss_roll(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on the roll and their exact isometric coordinates `(s, h)`."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    h = 12.0 * rng.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    s = 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))
    return X, np.column_stack([s, h])


def _blobs(n_per: int, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters on a circle in 12 dimensions, with their labels."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
    centres = np.zeros((n_clusters, 12))
    centres[:, 0] = 3.0 * np.cos(angles)
    centres[:, 1] = 3.0 * np.sin(angles)
    X = np.vstack([centres[c] + 0.55 * rng.standard_normal((n_per, 12))
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


def _demo_exact(X: np.ndarray, truth: np.ndarray) -> TSNE:
    """Neighbourhoods are kept, distances are not, and perplexity picks which."""
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pca = U[:, :2] * S[:2]
    rows = []
    for perplexity in (5.0, 30.0, 80.0):
        fit = TSNE(2, variant="exact", perplexity=perplexity, n_iter=N_ITER).fit(X)
        rows.append((perplexity, _overlap(fit.embedding_, truth),
                     _overlap(fit.embedding_, truth, 40),
                     _distance_correlation(fit.embedding_, truth),
                     fit.kl_initial_, fit.kl_divergence_))
    reference = TSNE(2, variant="exact", n_iter=N_ITER).fit(X)
    low, mid, high = rows
    pca_keep = _overlap(pca, truth)

    blobs, labels = _blobs(40, 6, seed=5)
    cluster_fit = TSNE(2, variant="exact", n_iter=N_ITER).fit(blobs)
    ci = np.array([blobs[labels == c].mean(axis=0) for c in range(6)])
    co = np.array([cluster_fit.embedding_[labels == c].mean(axis=0)
                   for c in range(6)])
    iu = np.triu_indices(6, k=1)
    gap_in = np.linalg.norm(ci[:, None] - ci[None], axis=2)[iu]
    gap_out = np.linalg.norm(co[:, None] - co[None], axis=2)[iu]
    within_in = np.mean([np.linalg.norm(blobs[labels == c] - ci[c], axis=1).mean()
                         for c in range(6)])
    within_out = np.mean([np.linalg.norm(cluster_fit.embedding_[labels == c] - co[c],
                                         axis=1).mean() for c in range(6)])
    tight_in = within_in / gap_in.min()
    tight_out = within_out / gap_out.min()
    order = float(np.corrcoef(gap_in, gap_out)[0, 1])

    assert pca_keep < 0.55, pca_keep
    assert mid[1] > 0.8 and mid[1] > low[1] and mid[1] > high[1], rows
    assert all(r[3] < 0.5 for r in rows), rows
    assert mid[5] < 0.5 * mid[4], mid
    assert tight_out < 0.5 * tight_in, (tight_in, tight_out)
    assert order > 0.9, order

    print(
        f"exact: {X.shape[0]} points on a swiss roll, which is exactly isometric "
        f"to a flat rectangle with coordinates (s, h) — the ground truth Isomap "
        f"recovers at distance correlation 0.9996. t-SNE keeps the "
        f"neighbourhoods and not the rectangle. The leading two principal "
        f"components keep {pca_keep:.3f} of each point's eight true nearest "
        f"neighbours; t-SNE keeps {mid[1]:.3f} at perplexity 30, where its "
        f"distance correlation with the rectangle is {mid[3]:.3f}. Perplexity is "
        f"the one dial: eight-neighbour overlap at k = 8 and distance "
        f"correlation are "
        + ", ".join(f"{p:g}→{o:.3f}/{c:.3f}" for p, o, _, c, _, _ in rows)
        + f", so 5 is too local to hold a neighbourhood together and 80 is "
        f"nearly global, and the KL falls from {mid[4]:.3f} to {mid[5]:.3f} on "
        f"the way. What it does to clusters is separate from what it does to "
        f"neighbourhoods: on six Gaussian clusters the ratio of within-cluster "
        f"spread to the closest gap tightens from {tight_in:.3f} to "
        f"{tight_out:.3f}, a factor of {(tight_out / tight_in):.3f}, while the "
        f"order of the gaps survives (correlation {order:.3f})"
    )
    return reference


def _demo_barnes_hut(X: np.ndarray, truth: np.ndarray) -> None:
    """The tree is exact at θ = 0 and cheap at θ = 0.5, and the layout agrees."""
    reference = TSNE(2, variant="exact", n_iter=300).fit(X)
    Y = reference.embedding_
    f_exact, z_exact, _, pairs = _exact_repulsion(Y)
    rows = []
    for theta in (0.0, 0.25, 0.5, 1.0):
        f_h, z_h, _, terms = _bh_repulsion(Y, theta)
        rows.append((theta, float(np.linalg.norm(f_h - f_exact)
                                  / np.linalg.norm(f_exact)), terms,
                     abs(z_h - z_exact) / z_exact))
    rng = np.random.default_rng(0)
    scaling = []
    for size in (200, 800, 3200):
        cloud = rng.standard_normal((size, 2)) * 5.0
        _, _, _, terms = _bh_repulsion(cloud, 0.5)
        scaling.append((size, terms, size * (size - 1)))
    approximate = TSNE(2, variant="barnes_hut", n_iter=300).fit(X)
    keep_approx = _overlap(approximate.embedding_, truth)
    keep_exact = _overlap(Y, truth)
    agree = _overlap(approximate.embedding_, Y)

    assert rows[0][1] < 1e-12 and rows[0][3] < 1e-12, rows
    assert rows[2][1] < 0.08 and rows[2][2] < 0.3 * pairs, rows
    assert rows[3][1] > rows[2][1], rows
    share = [t / p for _, t, p in scaling]
    assert all(share[i + 1] < share[i] for i in range(len(share) - 1)), scaling
    assert abs(keep_approx - keep_exact) < 0.1, (keep_approx, keep_exact)
    assert approximate.kl_divergence_ < 1.5 * reference.kl_divergence_, (
        approximate.kl_divergence_, reference.kl_divergence_)

    print(
        f"barnes_hut: on a fixed 250-point layout, θ = 0 accepts every leaf and "
        f"reproduces the exact force to {rows[0][1]:.1e} and the normaliser to "
        f"{rows[0][3]:.1e} — the traversal has no bookkeeping error, only the "
        f"approximation does. θ, relative force error, accepted terms and "
        f"normaliser error are "
        + ", ".join(f"{t:g}→{e:.4f}/{n}/{z:.1e}" for t, e, n, z in rows)
        + f" against {pairs} pairs. The term count is the point: at θ = 0.5 it "
        f"is {rows[2][2] / pairs:.3f} of the pairs here and shrinks with n, "
        + ", ".join(f"{n}→{t / p:.3f}" for n, t, p in scaling)
        + f" at n = 200, 800, 3200, because a cell stands in for its whole "
        f"cloud. That is the asymptotic win and not a constant-factor win: this "
        f"numpy traversal is slower than one BLAS matrix product until a few "
        f"thousand points. At an equal 300-iteration budget the tree keeps "
        f"{keep_approx:.3f} of the true eight neighbours against the exact "
        f"solver's {keep_exact:.3f} ({agree:.3f} of the two layouts' "
        f"neighbourhoods in common) and reaches KL "
        f"{approximate.kl_divergence_:.3f} against {reference.kl_divergence_:.3f}"
    )


def _demo_fft(X: np.ndarray, truth: np.ndarray) -> None:
    """A grid makes the repulsion a convolution, at a resolution you choose."""
    reference = TSNE(2, variant="exact", n_iter=300).fit(X)
    Y = reference.embedding_
    f_exact, z_exact, _, pairs = _exact_repulsion(Y)
    rows = []
    for grid in (32, 64, 128, 256):
        f_g, z_g, _, cells = _fft_repulsion(Y, grid)
        rows.append((grid, float(np.linalg.norm(f_g - f_exact)
                                 / np.linalg.norm(f_exact)),
                     abs(z_g - z_exact) / z_exact, cells))
    approximate = TSNE(2, variant="fft", grid_size=128, n_iter=300).fit(X)
    keep_approx = _overlap(approximate.embedding_, truth)
    keep_exact = _overlap(Y, truth)
    agree = _overlap(approximate.embedding_, Y)

    errs = [r[1] for r in rows]
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), rows
    assert rows[-1][1] < 0.01, rows
    assert abs(keep_approx - keep_exact) < 0.1, (keep_approx, keep_exact)

    print(
        "fft: the same sums, computed by convolving the point density with the "
        "kernel on a grid. Grid, relative force error, normaliser error and "
        "cells are "
        + ", ".join(f"{g:g}→{e:.4f}/{z:.1e}/{c}" for g, e, z, c in rows)
        + f" against {pairs} pairs, and each doubling of the grid buys a factor "
        f"of four in error, which is the bilinear deposit's second order. The "
        f"cost is the grid and not the data — {rows[2][3]} cells serves 250 "
        f"points or 250000 — but the *accuracy* is not: the grid must resolve "
        f"one kernel width, the kernel's unit length is the layout's own scale, "
        f"and t-SNE's layout grows with n, so this same grid is worth less on a "
        f"bigger picture. At an equal 300-iteration budget it keeps "
        f"{keep_approx:.3f} of the true eight neighbours against the exact "
        f"solver's {keep_exact:.3f} ({agree:.3f} in common) and reaches KL "
        f"{approximate.kl_divergence_:.3f} against "
        f"{reference.kl_divergence_:.3f}"
    )


def _demo_multiscale(X: np.ndarray, truth: np.ndarray,
                     single_30: TSNE) -> None:
    """A mixture is a hedge: near the best single scale, without choosing it."""
    mixtures = ((10.0,), (50.0,), (10.0, 30.0, 50.0))
    rows = []
    for scales in mixtures:
        fit = TSNE(2, variant="multiscale", perplexities=scales,
                   n_iter=N_ITER).fit(X)
        rows.append((scales, _overlap(fit.embedding_, truth),
                     _overlap(fit.embedding_, truth, 40)))
    baseline = (_overlap(single_30.embedding_, truth),
                _overlap(single_30.embedding_, truth, 40))
    best = max(r[1] for r in rows)
    worst = min(r[1] for r in rows)
    order = (10.0, 30.0, 50.0)
    mixed_8 = next(r[1] for r in rows if r[0] == order)
    mixed_40 = next(r[2] for r in rows if r[0] == order)

    assert mixed_8 > worst, (mixed_8, worst)
    assert mixed_8 > 0.93 * best, (mixed_8, best)
    assert mixed_40 > 0.93 * baseline[1], (mixed_40, baseline[1])

    print(
        "multiscale: average the conditional affinities over several "
        "perplexities and one run carries more than one notion of "
        "'neighbour'. Eight-neighbour overlap at k = 8 and k = 40 against the "
        "true sheet is "
        + ", ".join(f"{'/'.join(f'{s:g}' for s in sc)}→{a:.3f}/{b:.3f}"
                    for sc, a, b in rows)
        + f", with the single-scale perplexity 30 reference at "
        f"{baseline[0]:.3f}/{baseline[1]:.3f}. The mixture lands at "
        f"{mixed_8:.3f}/{mixed_40:.3f}: {mixed_8 / best:.1%} of the best single "
        f"scale at k = 8 and {mixed_40 / baseline[1]:.1%} of it at k = 40, "
        f"without anyone having to know which scale that is. That is what the "
        f"mixture buys — a hedge, not an improvement — and it is worth being "
        f"precise about, because the averaging gives the flattest scale in the "
        f"mixture equal weight. A wide astronomical scale would flatten every "
        f"row toward uniform; keeping the range to comparable scales is what "
        f"keeps the hedge honest"
    )


def _demo_parametric() -> None:
    """A network reproduces the fit and places points it never saw."""
    X, labels = _blobs(40, 6, seed=5)
    rng = np.random.default_rng(105)
    perm = rng.permutation(len(X))
    train = np.sort(perm[:180])
    held = np.sort(perm[180:])
    fit = TSNE(2, variant="parametric", n_iter=N_ITER).fit(X[train])
    embedded = fit.embedding_
    projected = fit.transform(X[held])
    reproduce = _overlap(embedded, fit.teacher_)

    def purity(points: np.ndarray, truth: np.ndarray) -> float:
        return float(np.mean(labels[train][np.argmin(_cross(points, embedded),
                                                     axis=1)] == truth))

    neighbours = np.argsort(_cross(X[held], X[train]), axis=1)[:, :10]
    baseline = embedded[neighbours].mean(axis=1)
    net = purity(projected, labels[held])
    base = purity(baseline, labels[held])

    roll, _ = _swiss_roll(300, seed=7)
    curved = TSNE(2, variant="parametric", n_iter=N_ITER).fit(roll)
    near = np.argsort(_cross(roll, roll), axis=1)[:, :10]
    smooth = curved.embedding_[near].mean(axis=1)
    net_roll = _overlap(curved.transform(roll), curved.teacher_)
    base_roll = _overlap(smooth, curved.teacher_)

    assert reproduce > 0.9, reproduce
    assert net > 0.8, net
    assert base >= net, (base, net)
    assert base_roll > net_roll, (base_roll, net_roll)

    print(
        f"parametric: {train.size} points in six 12-dimensional clusters, and a "
        f"two-layer tanh network fitted to the layout rather than to the data. "
        f"On the training points it reproduces the exact embedding at "
        f"{reproduce:.3f} eight-neighbour overlap, and the KL it reports, "
        f"{fit.kl_divergence_:.3f}, is the teacher's — which is the honest "
        f"description of distilling rather than optimising. Transform the "
        f"{held.size} held-out points and {net:.3f} of them land nearest their "
        f"own cluster in the layout. The baseline any parametric map has to "
        f"beat is a 10-neighbour average of the fitted coordinates, and it wins "
        f"here: {base:.3f} against the network's {net:.3f}. The network is a "
        f"smooth function of x and t-SNE is not, so a curved sheet is where the "
        f"readout stops being enough: on a 300-point swiss roll the network "
        f"scores {net_roll:.3f} against the neighbour average's {base_roll:.3f}, "
        f"both far below the cluster numbers, because the teacher's layout is "
        f"rough at the fold and no smooth function of x follows it. That is the "
        f"trade: `transform` exists, and it costs local fidelity"
    )


def main() -> None:
    X, truth = _swiss_roll(250, seed=1)
    reference = _demo_exact(X, truth)
    _demo_barnes_hut(X, truth)
    _demo_fft(X, truth)
    _demo_multiscale(X, truth, reference)
    _demo_parametric()
    print("tsne demo passed")


if __name__ == "__main__":
    main()
