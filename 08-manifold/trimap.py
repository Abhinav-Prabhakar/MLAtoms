"""TriMap — preserve the *order* of distances with triplets, not the distances.

Chapter:   08-manifold
Companion: trimap.md
Tier 3 · 3 variants: trimap, supervised, knn

    s(y_i, y_j) = 1 / (1 + ‖y_i − y_j‖²)                              (1)
    ℓ_ijk = ω_ijk · s(y_i, y_k) / (s(y_i, y_j) + s(y_i, y_k))          (2)
    ω̃_ijk = d²_ik − d²_ij,   d²_ij = ‖x_i − x_j‖² / (σ_i σ_j)         (3)
    ω_ijk = log_t(1 + ω̃_ijk − ω_min)                                   (4)

A triplet (i, j, k) asserts one thing: i is closer to j than to k. It carries
no distance, no angle and no scale — only an ordering, and an ordering is what
survives when the data are on a curved surface and the embedding is flat. That
is the whole design. t-SNE and UMAP ask the embedding to reproduce pairwise
similarities, which forces a global scale and a particular deformation of the
neighbourhoods; TriMap asks only that the *ranking* around each point is kept,
and spends a fixed budget of triplets per point to say so.

The second idea is where the triplets come from. Most of them are local: for
each of the `m` nearest neighbours `j` of `i`, an `m′`-th point `k` is drawn
uniformly from everything farther from `i` than `j` is. A few are global: `r`
per point, with both `j` and `k` drawn uniformly from the whole data set and
then ordered by distance. The local triplets fix the shape of each
neighbourhood; the global ones fix where the neighbourhoods sit relative to
each other. The demo measures what happens without them, and the answer is not
"a slightly worse layout" — it is a crumpled one.

The weights in (3) say how much a triplet matters: the further `k` is beyond
`j`, the stronger the claim. `σ_i` is the mean distance to `i`'s 4th–6th
neighbours, so a dense region and a sparse region contribute on the same scale.
Equation (4) then compresses the range with a tempered logarithm — without it a
handful of far-away triplets dominate the gradient — and shifts by the minimum
so every weight is non-negative.

What makes the result different from t-SNE is what it is *not* asked to do. Two
points may be close in the embedding without being neighbours in the data, as
long as no sampled triplet forbids it. That has a cost and a payoff: a triplet
budget of `(m + r)·n` makes each gradient step linear in `n`, so the optimisation
never dominates; what dominates is building the triplets, which needs a
neighbour search and, in this implementation, the full distance matrix. The
paper's own implementation replaces the latter with an approximate index, and
that — not the optimiser — is what carries it to millions of points.
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("trimap", "supervised", "knn")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _pairwise(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    B = A if B is None else B
    sa = np.einsum("ij,ij->i", A, A)
    sb = np.einsum("ij,ij->i", B, B)
    return np.sqrt(np.maximum(sa[:, None] + sb[None, :] - 2.0 * (A @ B.T), 0.0))


def _tempered_log(x: np.ndarray, t: float) -> np.ndarray:
    """log_t(x) = (x^(1−t) − 1)/(1 − t), the plain log in the limit t → 1."""
    if abs(t - 1.0) < 1e-6:
        return np.log(x)
    return (np.power(x, 1.0 - t) - 1.0) / (1.0 - t)


def global_score(X: np.ndarray, Y: np.ndarray) -> float:
    """The paper's global score: how well a linear map rebuilds X from Y.

    `X` is `(n, n_features)`, `Y` is `(n, n_components)`. The minimum
    reconstruction error is `min_A ‖X − Y Aᵀ‖²_F`, which the least-squares
    solution attains; the score is `exp(−(MRE − MRE_PCA)/MRE_PCA)`, so 1.0 means
    "as globally faithful as PCA" and 0.0 means "much worse".
    """
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    A = Xc.T @ Yc @ np.linalg.inv(Yc.T @ Yc)
    mre = float(((Xc - Yc @ A.T) ** 2).sum())
    left, spectrum, _ = np.linalg.svd(Xc, full_matrices=False)
    pca = left[:, :Y.shape[1]] * spectrum[:Y.shape[1]]
    A_pca = Xc.T @ pca @ np.linalg.inv(pca.T @ pca)
    mre_pca = float(((Xc - pca @ A_pca.T) ** 2).sum())
    if mre_pca <= 0.0:
        raise ValueError("PCA leaves no residual; the global score is undefined")
    return float(np.exp(-(mre - mre_pca) / mre_pca))


# --------------------------------------------------------------------------- #
# Triplet sampling
# --------------------------------------------------------------------------- #
def _self_tuning_scale(D: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """σ_i = mean distance to i's 4th–6th neighbours (the paper's choice).

    Rows of `D` are distances with a zero diagonal; `neighbours` is `(n, m)`.
    """
    scale = D[np.arange(D.shape[0])[:, None], neighbours[:, 3:6]].mean(axis=1)
    return np.maximum(scale, 1e-10)


def _sample_triplets(D: np.ndarray, neighbours: np.ndarray, *, variant: str,
                     n_triplets: int, n_random: int, random_weight: float,
                     labels: np.ndarray | None, scale: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample `(i, j, k)` triplets and their weights ω.

    `D` is `(n, n)` with a zero diagonal, `neighbours` is `(n, m)` sorted by
    distance, and `scale` is `(n,)`. Each of the `m` neighbours of a point
    contributes `n_triplets` triplets — the paper's m′ — and each point adds
    `n_random` fully random ones, so `T ≈ (m·m′ + r)·n`. Returns integer arrays
    `(T, 3)` and `(T,)`.
    """
    n = D.shape[0]
    D2 = D**2 / (scale[:, None] * scale[None, :])

    i_parts, j_parts, k_parts, w_parts = [], [], [], []
    ring = [np.setdiff1d(np.unique(neighbours[neighbours[row]]),
                         np.append(neighbours[row], row))
            for row in range(n)]

    for i in range(n):
        if variant == "supervised":
            same = np.flatnonzero(labels == labels[i])
            same = same[same != i]
            other = np.flatnonzero(labels != labels[i])
            if same.size == 0 or other.size == 0:
                continue
            count = n_triplets * neighbours.shape[1]
            j = same[rng.integers(same.size, size=count)]
            k = other[rng.integers(other.size, size=count)]
            i_parts.append(np.full(count, i))
            j_parts.append(j)
            k_parts.append(k)
            w_parts.append(D2[i, k] - D2[i, j])
            continue

        for j in neighbours[i]:
            candidates = (ring[i] if variant == "knn"
                          else np.flatnonzero(D[i] > D[i, j]))
            if candidates.size == 0:
                continue
            k = candidates[rng.integers(candidates.size, size=n_triplets)]
            i_parts.append(np.full(n_triplets, i))
            j_parts.append(np.full(n_triplets, j))
            k_parts.append(k)
            w_parts.append(D2[i, k] - D2[i, j])

        if variant == "trimap" and n_random > 0:
            for _ in range(n_random):
                j, k = rng.choice(n, size=2, replace=False)
                if D[i, k] < D[i, j]:
                    j, k = k, j
                i_parts.append(np.array([i]))
                j_parts.append(np.array([j]))
                k_parts.append(np.array([k]))
                w_parts.append(np.array([random_weight * (D2[i, k] - D2[i, j])]))

    triplets = np.column_stack([np.concatenate(i_parts),
                                np.concatenate(j_parts),
                                np.concatenate(k_parts)]).astype(int)
    weights = np.concatenate(w_parts)
    weights = weights - weights.min()
    weights = _tempered_log(1.0 + weights, 0.5)
    return triplets, weights


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def _loss_and_grad(Y: np.ndarray, triplets: np.ndarray, weights: np.ndarray
                   ) -> tuple[float, np.ndarray]:
    """The triplet loss (2) and its gradient. `Y` is `(n, n_components)`."""
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    yi, yj, yk = Y[i], Y[j], Y[k]
    # s = 1/(1+d²) written as a = 1/d_ij, b = 1/d_ik with d = 1+‖·‖².
    a = 1.0 / (1.0 + np.einsum("ij,ij->i", yi - yj, yi - yj))
    b = 1.0 / (1.0 + np.einsum("ij,ij->i", yi - yk, yi - yk))
    loss = float(np.sum(weights * b / (a + b)))

    denom = (a + b) ** 2
    coef = 2.0 * weights * a * b / denom
    g_i = coef[:, None] * (a[:, None] * (yi - yj) - b[:, None] * (yi - yk))
    g_j = -2.0 * (weights * a * a * b / denom)[:, None] * (yi - yj)
    g_k = 2.0 * (weights * a * b * b / denom)[:, None] * (yi - yk)

    grad = np.zeros_like(Y)
    np.add.at(grad, i, g_i)
    np.add.at(grad, j, g_j)
    np.add.at(grad, k, g_k)
    return loss, grad / len(triplets)


def _violation_rate(Y: np.ndarray, triplets: np.ndarray) -> float:
    """Fraction of triplets whose order is still wrong in the embedding.

    A triplet is violated when `k` ends up closer to `i` than `j` is. The
    reference implementation tracks exactly this number; it is a sharper
    diagnostic than the loss, which mixes in the weights.
    """
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    near = np.einsum("ij,ij->i", Y[i] - Y[j], Y[i] - Y[j])
    far = np.einsum("ij,ij->i", Y[i] - Y[k], Y[i] - Y[k])
    return float(np.mean(far < near))


# --------------------------------------------------------------------------- #
# The embedding
# --------------------------------------------------------------------------- #
class TriMap:
    """Triplet-constrained embedding.

    `TriMap(n_components=2, *, variant="trimap", n_neighbors=12, n_triplets=4,
    n_random=3, n_iter=400, learning_rate=0.1, random_weight=0.1, seed=0)` with
    `fit(X, y=None)` and `fit_transform(X, y=None)`; `X` is `(n, n_features)`
    and `y` is `(n,)` labels, required by `supervised` and ignored otherwise.
    `n_neighbors` and `n_triplets` are the paper's m and m′.

    Attributes after `fit`: `embedding_` `(n, n_components)`; `losses_`, the
    objective at every iteration, and `violations_`, the fraction of triplets
    whose order is still wrong; `triplets_` `(T, 3)` and `weights_` `(T,)`,
    the sampled constraints, which are the real model; `scale_` `(n,)`, the
    self-tuning σ_i. The initialisation is the PCA embedding rescaled to unit
    root-mean-square, and the optimiser is full-batch Adam from `seed`.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "trimap",
                 n_neighbors: int = 12, n_triplets: int = 4, n_random: int = 3,
                 n_iter: int = 400, learning_rate: float = 0.1,
                 random_weight: float = 0.1, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1 or n_iter < 1:
            raise ValueError("n_components and n_iter must be positive")
        if n_neighbors < 6:
            raise ValueError("n_neighbors must be at least 6 for the σ_i scale")
        if n_triplets < 1:
            raise ValueError("n_triplets must be positive")
        if n_random < 0:
            raise ValueError("n_random cannot be negative")
        self.n_components = n_components
        self.variant = variant
        self.n_neighbors = n_neighbors
        self.n_triplets = n_triplets
        self.n_random = n_random
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_weight = random_weight
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> TriMap:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        n = X.shape[0]
        if self.n_neighbors >= n:
            raise ValueError("n_neighbors must be smaller than n")
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed n_features")
        labels = None
        if self.variant == "supervised":
            if y is None:
                raise ValueError("the supervised variant needs labels y")
            labels = np.asarray(y)
            if labels.shape != (n,):
                raise ValueError("y must have shape (n,)")

        D = _pairwise(X)
        np.fill_diagonal(D, np.inf)
        neighbours = np.argsort(D, axis=1)[:, :self.n_neighbors]
        self.scale_ = _self_tuning_scale(D, neighbours)
        np.fill_diagonal(D, 0.0)

        rng = np.random.default_rng(self.seed)
        self.triplets_, self.weights_ = _sample_triplets(
            D, neighbours, variant=self.variant, n_triplets=self.n_triplets,
            n_random=self.n_random, random_weight=self.random_weight,
            labels=labels, scale=self.scale_, rng=rng)

        centred = X - X.mean(axis=0)
        left, spectrum, _ = np.linalg.svd(centred, full_matrices=False)
        Y = left[:, :self.n_components] * spectrum[:self.n_components]
        Y = Y / np.sqrt(np.mean(np.einsum("ij,ij->i", Y, Y)))

        first = np.zeros_like(Y)
        second = np.zeros_like(Y)
        self.losses_ = []
        self.violations_ = []
        for step in range(1, self.n_iter + 1):
            loss, grad = _loss_and_grad(Y, self.triplets_, self.weights_)
            self.losses_.append(loss)
            self.violations_.append(_violation_rate(Y, self.triplets_))
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad * grad
            m_hat = first / (1.0 - 0.9**step)
            v_hat = second / (1.0 - 0.999**step)
            Y = Y - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            Y -= Y.mean(axis=0)

        self.embedding_ = Y
        self.n_features_ = X.shape[1]
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None
                      ) -> np.ndarray:
        return self.fit(X, y).embedding_


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _scurve(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A sheet folded twice in three dimensions, and the parameters behind it."""
    rng = np.random.default_rng(seed)
    t = 3.0 * np.pi * (rng.random(n) - 0.5)
    h = 2.0 * rng.random(n)
    X = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])
    return X, np.column_stack([t, h])


def _blobs(n: int, seed: int, k: int = 5, dims: int = 8, spread: float = 1.2,
           separation: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters with labels, far enough apart to be worth separating."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dims)) * separation
    labels = rng.integers(0, k, n)
    return centres[labels] + spread * rng.standard_normal((n, dims)), labels


def _overlap(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each row's `k` nearest neighbours in `A` found in `B`."""
    da, db = _pairwise(A), _pairwise(B)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    ia = np.argsort(da, axis=1)[:, :k]
    ib = np.argsort(db, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k
                          for a, b in zip(ia, ib, strict=True)]))


def _distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(_pairwise(A)[iu], _pairwise(B)[iu])[0, 1])


def _pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    C = X - X.mean(axis=0)
    left, spectrum, _ = np.linalg.svd(C, full_matrices=False)
    return left[:, :n_components] * spectrum[:n_components]


def _knn_accuracy(Y: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Fraction of rows whose `k` nearest neighbours vote for their label."""
    D = _pairwise(Y)
    np.fill_diagonal(D, np.inf)
    nearest = np.argsort(D, axis=1)[:, :k]
    votes = [np.bincount(labels[row]).argmax() for row in nearest]
    return float(np.mean(votes == labels))


def _loss_on(Y: np.ndarray, triplets: np.ndarray, weights: np.ndarray) -> float:
    return _loss_and_grad(Y, triplets, weights)[0]


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_gradient() -> None:
    """The triplet gradient against central differences, and its invariants."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((6, 2))
    triplets = np.array([[0, 1, 2]])
    weights = np.array([1.7])
    _, grad = _loss_and_grad(Y, triplets, weights)

    worst = 0.0
    for idx in np.ndindex(Y.shape):
        up, down = Y.copy(), Y.copy()
        up[idx] += 1e-6
        down[idx] -= 1e-6
        numeric = (_loss_on(up, triplets, weights)
                   - _loss_on(down, triplets, weights)) / 2e-6
        analytic = grad[idx]
        worst = max(worst, abs(numeric - analytic)
                    / max(abs(numeric), abs(analytic), 1e-12))

    line = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    swapped = line[[0, 2, 1]]
    good = _loss_on(line, triplets, weights)
    bad = _loss_on(swapped, triplets, weights)

    # The same test with the doc's round weight, and the scale sweep that shows
    # the loss has no preferred scale.
    unit = np.array([6.0])
    long_line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    ordered = _loss_on(long_line, triplets, unit)
    reversed_ = _loss_on(long_line[[0, 2, 1]], triplets, unit)
    sweep = [_loss_on(long_line * s, triplets, unit) for s in (0.25, 1.0, 4.0, 16.0)]
    shrinking = all(later < earlier
                    for earlier, later in zip(sweep, sweep[1:], strict=False))

    assert worst < 1e-6, worst
    assert float(np.abs(grad.sum(axis=0)).max()) < 1e-12
    assert good < bad
    assert ordered < reversed_
    assert shrinking

    print(
        f"gradients: one triplet on a six-point embedding, all twelve partial "
        f"derivatives checked against central differences. Worst relative error "
        f"{worst:.1e}; the three role-gradients sum to "
        f"{float(np.abs(grad.sum(axis=0)).max()):.0e}, which they must because "
        f"moving the whole embedding does not change any distance. With ω = 1.7, "
        f"placing the far point at distance 3 and the near one at 1 costs "
        f"{good:.4f}; swapping them costs {bad:.4f}, five times more. With ω = 6 "
        f"and points at 0, 1, 4 the same pair costs {ordered:.4f} against "
        f"{reversed_:.4f}, and blowing the embedding up by 4× takes the ordered "
        f"cost from {sweep[1]:.4f} to {sweep[2]:.4f} and by 16× to "
        f"{sweep[3]:.4f} — monotonically, with the limit "
        f"{6.0 * 1.0 / 17.0:.4f}. The loss wants the embedding to expand"
    )


def _demo_scurve() -> None:
    """The claim: triplet order keeps more than a projection, globally too."""
    X, coords = _scurve(300, seed=5)
    fit = TriMap().fit(X)
    pca = _pca(X)
    ov = _overlap(fit.embedding_, coords)
    ov40 = _overlap(fit.embedding_, coords, 40)
    pca_ov = _overlap(pca, coords)
    pca_40 = _overlap(pca, coords, 40)

    assert ov > pca_ov and ov40 > pca_40
    assert global_score(X, fit.embedding_) > 0.6

    print(
        f"S-curve: 300 points folded twice in three dimensions. "
        f"{(fit.n_neighbors * fit.n_triplets + fit.n_random)} triplets per point "
        f"— twelve neighbours times four draws each, plus three uniformly random "
        f"pairs — give {len(fit.weights_)} constraints in all. TriMap keeps "
        f"{ov:.3f} of each point's ten nearest neighbours and {ov40:.3f} at "
        f"forty, and its distances correlate "
        f"{_distance_correlation(fit.embedding_, coords):.3f} with the sheet's "
        f"parameters. The leading two principal components keep {pca_ov:.3f} and "
        f"{pca_40:.3f} and correlate "
        f"{_distance_correlation(pca, coords):.3f}. The global score — how much "
        f"of the input a *linear* map can rebuild from the embedding, relative to "
        f"PCA's own residual — is {global_score(X, fit.embedding_):.3f} against "
        f"PCA's 1.000 by definition, and {fit.violations_[-1]:.1%} of the sampled "
        f"triplets still have the wrong order at the end, down from "
        f"{fit.violations_[0]:.1%} at the PCA initialisation. So the trade is "
        f"+{ov - pca_ov:.3f} neighbourhood overlap at k = 10 for "
        f"{1.0 - global_score(X, fit.embedding_):.3f} of the global score, and "
        f"that is what separates TriMap from t-SNE, which keeps the "
        f"neighbourhoods and gives up the global picture almost entirely"
    )


def _demo_ablation() -> None:
    """The knn variant, and the proof that its failure is the objective."""
    X, coords = _scurve(300, seed=5)
    good = TriMap().fit(X)
    local = TriMap(variant="knn").fit(X)

    same_scale = good.embedding_ * (
        np.sqrt(np.mean(np.einsum("ij,ij->i", local.embedding_, local.embedding_)))
        / np.sqrt(np.mean(np.einsum("ij,ij->i", good.embedding_, good.embedding_))))
    loss_local = _loss_on(local.embedding_, local.triplets_, local.weights_)
    loss_good = _loss_on(same_scale, local.triplets_, local.weights_)
    sweep = [TriMap(n_random=r).fit(X) for r in (0, 3, 6)]
    scores = [global_score(X, f.embedding_) for f in sweep]

    assert loss_local < loss_good
    assert scores[0] < scores[2]

    print(
        f"ablation: the knn variant keeps both j and k inside the two-hop "
        f"neighbourhood, {len(local.weights_)} local triplets and no global ones. "
        f"It keeps {_overlap(local.embedding_, coords):.3f} of each point's ten "
        f"nearest neighbours — less than TriMap's 0.774 — and its distances "
        f"correlate {_distance_correlation(local.embedding_, coords):.3f} with "
        f"the sheet, with a global score of "
        f"{global_score(X, local.embedding_):.3f}. The optimiser did not fail: "
        f"scored on the knn variant's own triplets, its crumpled embedding gets "
        f"{loss_local:.0f} and the good S-curve embedding gets {loss_good:.0f}, so "
        f"the local-only objective prefers the crumple. What matters is letting k "
        f"be anywhere farther than j: with r = 0 the global score is already "
        f"{scores[0]:.3f}, with the default three uniform pairs per point it is "
        f"{scores[1]:.3f}, and six do not improve on that ({scores[2]:.3f}) — the "
        f"budget saturates, the paper's own Figure 3 makes the same point"
    )


def _demo_supervised() -> None:
    """Labels as triplets: perfect classes, worse geometry."""
    X, labels = _blobs(300, seed=2)
    plain = TriMap().fit(X)
    guided = TriMap(variant="supervised").fit(X, labels)
    local = TriMap(variant="knn").fit(X)

    plain_acc = _knn_accuracy(plain.embedding_, labels)
    guided_acc = _knn_accuracy(guided.embedding_, labels)
    assert guided_acc > plain_acc
    assert global_score(X, guided.embedding_) < global_score(X, plain.embedding_)

    print(
        f"supervised: five Gaussian clusters in eight dimensions, overlapping "
        f"enough that a projection cannot separate them. Unsupervised TriMap "
        f"classifies {plain_acc:.1%} of points correctly by one nearest "
        f"neighbour in the embedding and scores "
        f"{global_score(X, plain.embedding_):.3f} globally; label-driven "
        f"triplets — j from the same class, k from another, "
        f"{len(guided.weights_)} of them — classify "
        f"{guided_acc:.1%} and score "
        f"{global_score(X, guided.embedding_):.3f}. The supervised variant "
        f"answers the question it was asked and loses global geometry doing it, "
        f"because a triplet that says 'same class' is silent about where the "
        f"class sits. The local-only variant classifies "
        f"{_knn_accuracy(local.embedding_, labels):.1%}"
    )


def _demo_weights() -> None:
    """What the shift, the tempered log and σ_i each do to the weights."""
    X, _ = _scurve(300, seed=5)
    fit = TriMap().fit(X)
    w = fit.weights_
    raw = (1.0 + w / 2.0) ** 2 - 1.0          # invert log_0.5
    ratio_raw = float(np.max(raw) / max(np.median(raw[raw > 0]), 1e-12))
    ratio_w = float(np.max(w) / max(np.median(w[w > 0]), 1e-12))
    ess_raw = float(raw.sum() ** 2 / np.sum(raw * raw))
    ess = float(w.sum() ** 2 / np.sum(w * w))
    top_raw = float(np.sort(raw)[-len(raw) // 10:].sum() / raw.sum())
    top = float(np.sort(w)[-len(w) // 10:].sum() / w.sum())

    assert w.min() == 0.0
    assert ratio_w < ratio_raw / 2.0
    assert ess > ess_raw
    assert top < top_raw

    print(
        f"weights: {len(w)} triplets, raw ω̃ from {raw.min():.0f} to "
        f"{raw.max():.1f} with median {np.median(raw):.2f}. The shift makes the "
        f"smallest weight exactly zero; the tempered log then cuts the "
        f"top-to-median ratio from {ratio_raw:.1f} to {ratio_w:.1f}, lifts the "
        f"effective sample size from {ess_raw:.0f} to {ess:.0f} of {len(w)} "
        f"({ess_raw / len(w):.0%} → {ess / len(w):.0%}), and drops the share "
        f"carried by the heaviest tenth from {top_raw:.0%} to {top:.0%}. "
        f"Compression helps but does not flatten the weights: a fifth of the "
        f"gradient still comes from a tenth of the constraints, which is the "
        f"price of claiming that a far-away point is *very* far away"
    )

    # Ablation: replace σ_i by one global number.
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((250, 3)) * 0.05
    sparse = rng.standard_normal((50, 3)) * 3.0
    Z = np.vstack([dense, sparse])
    membership = np.r_[np.zeros(250, dtype=int), np.ones(50, dtype=int)]
    D = _pairwise(Z)
    np.fill_diagonal(D, np.inf)
    near = np.argsort(D, axis=1)[:, :10]
    sigma = _self_tuning_scale(D, near)
    np.fill_diagonal(D, 0.0)

    shares = []
    for scale in (sigma, np.full(Z.shape[0], sigma.mean())):
        trip, weight = _sample_triplets(
            D, near, variant="trimap", n_triplets=4, n_random=3,
            random_weight=0.1, labels=None, scale=scale,
            rng=np.random.default_rng(0))
        dense_share = float(weight[membership[trip[:, 0]] == 0].sum() / weight.sum())
        shares.append((dense_share, float(weight.sum() ** 2 / np.sum(weight * weight))))

    assert sigma.max() / sigma.min() > 100.0
    assert shares[0][1] > 3.0 * shares[1][1]

    print(
        f"self-tuning scale: the same sampler on 250 points packed in a ball of "
        f"radius 0.05 and 50 points spread over radius 3. σ_i ranges over a "
        f"factor of {sigma.max() / sigma.min():.0f}, so equal distances count "
        f"equally in both regions. Replace σ_i by its mean and the dense "
        f"cluster's {shares[0][0]:.0%} share of the weight falls to "
        f"{shares[1][0]:.0%} even though it holds 83% of the points, and the "
        f"effective sample size collapses from {shares[0][1]:.0f} to "
        f"{shares[1][1]:.0f}: a few triplets in the sparse tail carry the "
        f"gradient. On this pair of blobs the embeddings come out about equally "
        f"separable — the ablation hurts when the scale ratios are large, not "
        f"always — but the objective is no longer the one the paper defines"
    )


def _demo_broken() -> None:
    """Three loud failures and one quiet one."""
    X, _ = _scurve(100, seed=1)
    messages = []
    for kwargs, y in [({"variant": "supervised"}, None),
                      ({"n_neighbors": 120}, None),
                      ({"n_neighbors": 4}, None)]:
        try:
            TriMap(**kwargs).fit(X, y)
        except ValueError as err:
            messages.append(str(err))

    first = TriMap().fit(X)
    second = TriMap().fit(X)

    assert len(messages) == 3
    assert "labels" in messages[0]
    assert "smaller than n" in messages[1]
    assert "at least 6" in messages[2]
    assert np.array_equal(first.triplets_, second.triplets_)
    assert np.array_equal(first.embedding_, second.embedding_)

    print(
        f"failures: the supervised variant without labels raises "
        f"\"{messages[0]}\"; more neighbours than points raises "
        f"\"{messages[1]}\"; fewer than six raises \"{messages[2]}\". The quiet "
        f"failure is that the model is the sample: two fits with the same seed "
        f"produce identical triplets and a bit-identical embedding, and a "
        f"different seed produces a different triplet set, a different loss and "
        f"an embedding that is not comparable without alignment"
    )


def main() -> None:
    _demo_gradient()
    _demo_scurve()
    _demo_ablation()
    _demo_supervised()
    _demo_weights()
    _demo_broken()
    print("trimap demo passed")


if __name__ == "__main__":
    main()
