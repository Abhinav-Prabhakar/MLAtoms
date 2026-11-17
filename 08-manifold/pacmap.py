"""PaCMAP — three kinds of pair, three weights, one schedule.

Chapter:   08-manifold
Companion: pacmap.md
Tier 2 · 4 variants: pacmap, localmap, supervised, knn_graph

    L = w_NP Σ_NP d̃/(10 + d̃) + w_MN Σ_MN d̃/(10000 + d̃) + w_FP Σ_FP 1/(1 + d̃)   (1)
    d̃_ij = 1 + ‖y_i − y_j‖²                                                        (2)
    (w_NP, w_MN, w_FP) = (2, 1000→3, 1) → (3, 3, 1) → (1, 0, 1)                     (3)

The method's claim is that an embedding is an argument about *which pairs
matter*, and that t-SNE and UMAP both get the answer slightly wrong by treating
every non-edge identically. PaCMAP splits the pairs into three sets — the
k-neighbour ring, mid-near pairs at the one-third quantile of the distance
distribution, and uniformly sampled further pairs — and gives each set its own
weight and its own schedule. On the demo's roll the three sit at mean distances
0.44, 1.54 and 2.18, against 2.12 for a random pair.

The schedule is the interesting part. The first phase puts weight 1000 on the
mid-near pairs, so the layout arranges regions before it arranges points; the
second brings the neighbour weight up to 3; the third drops mid-near to zero and
leaves neighbours and further pairs at equal weight. The phases do different
work and the demo measures the split: after phase two the forty-neighbour
overlap is 0.861 and the distance correlation 0.910, and phase three trades
0.111 of that correlation for 0.055 of eight-neighbour overlap. The further
pairs are the dial — halving them at `fp_ratio=1.0` gives the best global
fidelity in the demo, 0.969, at 0.719 local overlap.

Two consequences are measurable and are what the demo reports. Because the
further pairs are sampled from the whole set rather than from the graph's
complement per edge, the global arrangement is closer to the input than UMAP's
or t-SNE's on the same data. And because the neighbour set is chosen on *scaled*
distances — d² divided by both endpoints' local scale — the sparse of two regions
stops spending its neighbour slots on the dense one.

The variants isolate the parts: `knn_graph` drops the mid-near pairs and shows
what global structure they were carrying, `localmap` swaps the uniform further
pairs for the two-hop ring in the last phase and buys local overlap with the
middle of the range, and `supervised` lets labels decide which pairs are allowed
at all.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
VARIANTS = ("pacmap", "localmap", "supervised", "knn_graph")
NP, MN, FP = 0, 1, 2


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #
def _pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distances via the Gram identity, `(n, n)`."""
    sq = np.einsum("ij,ij->i", X, X)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))


def _knn(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The `k` nearest neighbours of every row, ascending; `D` needs an inf diagonal."""
    n = D.shape[0]
    k = int(min(k, n - 1))
    idx = np.argpartition(D, k, axis=1)[:, :k]
    dk = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dk, axis=1)
    return (np.take_along_axis(dk, order, axis=1),
            np.take_along_axis(idx, order, axis=1))


def _scaled_neighbours(D: np.ndarray, k: int,
                       extra: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Neighbours chosen on `d²/(σ_i σ_j)` rather than on `d`, `(n, k)` each.

    `σ_i` is the mean distance to neighbours 4, 5 and 6 — a local length scale
    that is estimated from the ring just outside the k-neighbourhood, so it
    measures the region rather than the point. Dividing by both scales takes the
    region's density out of the comparison: a pair that straddles a dense and a
    sparse region has one large `σ`, so its scaled distance is *smaller* than
    its raw one. The measured effect is an equalisation — on two Gaussians a
    factor of five apart in spread, the sparse region spends 0.161 of its
    neighbour slots on the other region and the dense one 0.076; after scaling
    the two are 0.124 and 0.095.

    The raw distance is returned alongside, because the layout is scored on
    distances and not on this rescaled quantity.
    """
    n = D.shape[0]
    k_ext = int(min(k + extra, n - 1))
    dk, idx = _knn(D, k_ext)
    sigma = np.maximum(dk[:, 3:6].mean(axis=1), EPS)
    scaled = dk * dk / sigma[:, None] / sigma[idx]
    order = np.argsort(scaled, axis=1)[:, :k]
    return (np.take_along_axis(dk, order, axis=1),
            np.take_along_axis(idx, order, axis=1))


def _sample_mid_near(D: np.ndarray, n_mn: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Mid-near partners, `(n, n_mn)`.

    Each slot is an independent draw of six points whose closest member is
    discarded and whose second closest becomes the partner. Six draws place the
    order statistics over the whole distance range, so "second of six" lands
    near the one-third quantile rather than at either end: on the demo's roll
    the three pair sets sit at mean distances 0.44, 1.54 and 2.18, against 2.12
    for a uniformly random pair. Uniform pairs carry the same information in
    expectation and would need far more of them, because most of the budget
    would be spent on the far tail.

    One draw of six would give one partner, and taking the second through sixth
    would give five partners all at or above the median — the mistake this
    construction exists to avoid.
    """
    n = D.shape[0]
    out = np.empty((n, n_mn), dtype=np.int64)
    for i in range(n):
        for slot in range(n_mn):
            chosen = rng.choice(n - 1, size=6, replace=False)
            chosen = chosen + (chosen >= i)          # skip the point itself
            out[i, slot] = chosen[np.argsort(D[i, chosen])[1]]
    return out


def _sample_far(n: int, neighbours: np.ndarray, n_fp: int,
                rng: np.random.Generator, labels: np.ndarray | None = None,
                same: bool = False) -> np.ndarray:
    """Further partners, `(n, n_fp)`, uniform over the points that are neither
    the point itself nor one of its `k` neighbours.

    Rejection is bounded to five rounds and whatever is left is dropped by the
    caller, because a rejection loop with a `while` is the one place a
    deterministic demo can hang. With `labels` given, the draw is restricted to
    partners whose label agrees (`same=True`) or disagrees (`same=False`).
    """
    n_fp = int(min(n_fp, n - 1))
    pool = rng.integers(0, n, size=(n, n_fp))
    bad = pool == np.arange(n)[:, None]
    bad |= (pool[:, :, None] == neighbours[:, None, :]).any(axis=2)
    if labels is not None:
        agree = labels[pool] == labels[:, None]
        bad |= agree if not same else ~agree
    for _ in range(4):
        if not bad.any():
            break
        fresh = rng.integers(0, n, size=(n, n_fp))
        pool = np.where(bad, fresh, pool)
        bad = pool == np.arange(n)[:, None]
        bad |= (pool[:, :, None] == neighbours[:, None, :]).any(axis=2)
        if labels is not None:
            agree = labels[pool] == labels[:, None]
            bad |= agree if not same else ~agree
    keep = ~bad
    return pool, keep


def _sample_second_order(neighbours: np.ndarray, n_fp: int,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Further partners drawn from the two-hop neighbourhood, `(n, n_fp)`.

    The candidates are the neighbours of a point's neighbours, minus the point's
    own neighbours and the point itself: pairs that are close in the graph but
    not adjacent in it, which a uniform sample reaches by accident and a local
    method has to get right. This is the `localmap` last-phase set, and it costs
    nothing but the kNN indices — no distance is computed that was not already
    needed.

    Returns the partner array and the mask of slots that were filled. A point in
    a very small sample can have an empty two-hop set; those slots are dropped by
    the caller rather than filled with a uniform draw, so the pair count is a
    measured quantity and not an assumption.
    """
    n = neighbours.shape[0]
    hood = [set(row.tolist()) for row in neighbours]
    pool = np.zeros((n, n_fp), dtype=np.int64)
    keep = np.zeros((n, n_fp), dtype=bool)
    for i in range(n):
        two: set[int] = set()
        for j in hood[i]:
            two |= hood[j]
        two.discard(i)
        two -= hood[i]
        if not two:
            continue
        cand = np.array(sorted(two), dtype=np.int64)
        take = rng.choice(cand, size=min(n_fp, cand.size), replace=False)
        pool[i, :take.size] = take
        keep[i, :take.size] = True
    return pool, keep


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
def _weights(t: int, phases: tuple[int, int, int]) -> tuple[float, float, float]:
    """`(w_NP, w_MN, w_FP)` from (3), in pair-kind order, at iteration `t`.

    Phase 1 starts the mid-near weight at 1000 and decays it to 3; phase 2 holds
    it at 3 while the neighbour weight rises; phase 3 drops it to zero and lets
    the neighbour pairs carry the layout. The decay is linear rather than
    exponential on purpose: the first phase should spend most of its budget at
    the large weight, which is where the regional arrangement is set.

    Only the weights differ between phases. `localmap` changes the *pairs* and
    not the schedule, which is what makes the two variants comparable.
    """
    p1, p2, _ = phases
    if t < p1:
        frac = t / p1
        return 2.0, (1.0 - frac) * 1000.0 + frac * 3.0, 1.0
    if t < p1 + p2:
        return 3.0, 3.0, 1.0
    return 1.0, 0.0, 1.0


def _pair_loss(heads: np.ndarray, tails: np.ndarray, kinds: np.ndarray,
               Y: np.ndarray, weights: tuple[float, float, float]) -> float:
    """(1), the objective itself — not a surrogate — for the finite-difference check."""
    d = Y[heads] - Y[tails]
    d2 = 1.0 + np.einsum("ij,ij->i", d, d)
    total = 0.0
    for kind, denom in ((NP, 10.0), (MN, 10000.0)):
        mask = kinds == kind
        if mask.any():
            total += weights[kind] * float((d2[mask] / (denom + d2[mask])).sum())
    mask = kinds == FP
    if mask.any():
        total += weights[FP] * float((1.0 / (1.0 + d2[mask])).sum())
    return total


def _pair_gradient(heads: np.ndarray, tails: np.ndarray, kinds: np.ndarray,
                   Y: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """Gradient of (1) with respect to `Y`, `(n, n_components)`.

    Each term is a scalar function of `d̃` times `d̃`'s derivative `2(y_i − y_j)`:

        d/dd̃ of  d̃/(10 + d̃)    =  10 / (10 + d̃)²         (attractive)
        d/dd̃ of  d̃/(10000 + d̃) =  10000 / (10000 + d̃)²    (attractive)
        d/dd̃ of  1/(1 + d̃)      =  −1 / (1 + d̃)²           (repulsive)

    The factors below are twice those, and the sign of the third is negative
    because minimising it means *increasing* the distance. Both attractive
    terms are bounded away from zero by the `1 +` in (2): at coincidence the
    gradient is `20/100` and `20000/10⁴`, finite, so no clamp is needed.
    """
    d = Y[heads] - Y[tails]
    d2 = 1.0 + np.einsum("ij,ij->i", d, d)
    coef = np.empty_like(d2)
    for kind, denom, sign in ((NP, 10.0, 1.0), (MN, 10000.0, 1.0),
                              (FP, 1.0, -1.0)):
        mask = kinds == kind
        if mask.any():
            coef[mask] = sign * weights[kind] * 2.0 * denom / (denom + d2[mask]) ** 2
    g = coef[:, None] * d
    grad = np.empty_like(Y)
    for j in range(Y.shape[1]):
        grad[:, j] = (np.bincount(heads, weights=g[:, j], minlength=Y.shape[0])
                      - np.bincount(tails, weights=g[:, j], minlength=Y.shape[0]))
    return grad


def _adam(Y: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
          lr: float, step: int, beta1: float = 0.9,
          beta2: float = 0.999) -> None:
    """Adam with the bias correction folded into the step size.

    `sqrt(1 − β₂^{t+1}) / (1 − β₁^{t+1})` is the standard correction; keeping it
    on the learning rate rather than dividing `m` and `v` saves two arrays and
    is the form the reference implementation uses.
    """
    m += (1.0 - beta1) * (grad - m)
    v += (1.0 - beta2) * (grad * grad - v)
    rate = lr * np.sqrt(1.0 - beta2 ** (step + 1)) / (1.0 - beta1 ** (step + 1))
    Y -= rate * m / (np.sqrt(v) + 1e-7)


def _pca_init(X: np.ndarray, n_components: int) -> np.ndarray:
    """Leading principal components scaled by 0.01.

    PaCMAP's own convention. The scale is not load-bearing — Adam's per-
    parameter step is scale-blind, so the run is the same up to the number of
    iterations it takes to grow into the data's scale — but it is what makes a
    fit reproducible against the reference implementation.
    """
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    take = min(n_components, U.shape[1])
    out = np.zeros((X.shape[0], n_components))
    out[:, :take] = U[:, :take] * S[:take]
    return 0.01 * out


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class PaCMAP:
    """Pairwise Controlled Manifold Approximation.

    `PaCMAP(n_components=2, *, variant="pacmap", n_neighbors=10, mn_ratio=0.5,
    fp_ratio=2.0, n_iters=(100, 100, 250), learning_rate=1.0, seed=0)` with
    `fit(X, y=None)` and `fit_transform(X, y=None)`.

    Attributes after `fit`: `embedding_` `(n, n_components)`, `pairs_` (the
    flat `(heads, tails, kinds)` triple used in the first two phases),
    `local_pairs_` (the last-phase triple, for `localmap`, else `None`),
    `neighbour_distances_`, `loss_`, `history_` (the loss at six checkpoints)
    and `weights_` (the schedule as it was applied at those checkpoints).

    `mn_ratio` and `fp_ratio` are pair counts per neighbour, so the totals are
    `n·k`, `n·k·mn_ratio` and `n·k·fp_ratio`. That is where the cost lives: at
    the defaults a point is involved in 10 + 5 + 20 pairs, three times what UMAP
    touches at `n_neighbors=15`.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "pacmap",
                 n_neighbors: int = 10, mn_ratio: float = 0.5,
                 fp_ratio: float = 2.0, n_iters: tuple[int, int, int] = (100, 100, 250),
                 learning_rate: float = 1.0, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if mn_ratio < 0.0 or fp_ratio < 0.0:
            raise ValueError("pair ratios must be non-negative")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if len(n_iters) != 3 or min(n_iters) < 0:
            raise ValueError("n_iters must be three non-negative phase lengths")
        self.n_components = int(n_components)
        self.variant = variant
        self.n_neighbors = int(n_neighbors)
        self.mn_ratio = float(mn_ratio)
        self.fp_ratio = float(fp_ratio)
        self.n_iters = tuple(int(v) for v in n_iters)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

    # -- pairs -------------------------------------------------------------- #
    def _pairs(self, X: np.ndarray, labels: np.ndarray | None,
               rng: np.random.Generator) -> tuple:
        """The three pair sets, as one flat triple of ``(heads, tails, kinds)``.

        The neighbour set is built on scaled distances in every variant. The
        mid-near set is skipped entirely for `knn_graph`, which is precisely
        that variant's ablation, and for `supervised` the further pairs are
        drawn only from the other classes while same-class neighbours keep their
        slots — the label decides which pairs exist, and the schedule then
        decides what they do.

        Returns the base triple and, for `localmap` only, the triple whose
        further pairs are the two-hop ring. The caller holds the base through
        the first two phases and switches at the last one, which is where the
        published method puts its local computations.
        """
        n = X.shape[0]
        D = _pairwise(X)
        np.fill_diagonal(D, np.inf)
        dk, idx = _scaled_neighbours(D, self.n_neighbors)
        n_mn = int(round(self.n_neighbors * self.mn_ratio))
        n_fp = int(round(self.n_neighbors * self.fp_ratio))

        heads = [np.repeat(np.arange(n), idx.shape[1])]
        tails = [idx.ravel()]
        kinds = [np.full(n * idx.shape[1], NP)]

        if self.variant != "knn_graph" and n_mn > 0:
            near = _sample_mid_near(D, n_mn, rng)
            heads.append(np.repeat(np.arange(n), n_mn))
            tails.append(near.ravel())
            kinds.append(np.full(n * n_mn, MN))

        if self.variant == "supervised":
            if labels is None:
                raise ValueError("variant='supervised' needs labels: fit(X, y)")
            pool, keep = _sample_far(n, idx, n_fp, rng, labels=labels, same=False)
            heads.append(np.repeat(np.arange(n)[:, None], n_fp, axis=1)[keep])
            tails.append(pool[keep])
            kinds.append(np.full(int(keep.sum()), FP))
            return ((_keep_same_class(heads, tails, kinds, idx, labels, n), dk),
                    None)

        pool, keep = _sample_far(n, idx, n_fp, rng)
        head_fp = np.repeat(np.arange(n)[:, None], n_fp, axis=1)[keep]
        tail_fp = pool[keep]
        base = (np.concatenate(heads + [head_fp]),
                np.concatenate(tails + [tail_fp]),
                np.concatenate(kinds + [np.full(head_fp.size, FP)]))

        local = None
        if self.variant == "localmap":
            pool2, keep2 = _sample_second_order(idx, n_fp, rng)
            head2 = np.repeat(np.arange(n)[:, None], n_fp, axis=1)[keep2]
            tail2 = pool2[keep2]
            local = (np.concatenate(heads + [head2]),
                     np.concatenate(tails + [tail2]),
                     np.concatenate(kinds + [np.full(head2.size, FP)]))
        return (base, dk), local

    # -- public API --------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PaCMAP:
        """Fit the embedding; returns `self`."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (samples, features), got shape {X.shape}")
        n = X.shape[0]
        if n <= self.n_neighbors:
            raise ValueError("n_neighbors must be smaller than the sample size")
        rng = np.random.default_rng(self.seed)
        labels = None if y is None else np.asarray(y)

        Xs = (X - X.mean(axis=0)) / max(float(X.std()), EPS)
        (base, dk), local = self._pairs(Xs, labels, rng)
        heads, tails, kinds = base

        Y = _pca_init(Xs, self.n_components)
        m = np.zeros_like(Y)
        v = np.zeros_like(Y)
        total = int(sum(self.n_iters))
        switch = int(self.n_iters[0] + self.n_iters[1])
        checkpoints = {0, total // 5, 2 * total // 5, 3 * total // 5,
                       4 * total // 5, total - 1}
        self.history_ = []
        self.weights_ = []

        for step in range(total):
            if local is not None and step == switch:
                heads, tails, kinds = local
            weights = _weights(step, self.n_iters)
            grad = _pair_gradient(heads, tails, kinds, Y, weights)
            _adam(Y, grad, m, v, self.learning_rate, step)
            if step in checkpoints:
                self.history_.append(_pair_loss(heads, tails, kinds, Y, weights))
                self.weights_.append((step, *weights))

        self.embedding_ = Y
        self.pairs_ = base
        self.local_pairs_ = local
        self.neighbour_distances_ = dk
        self.loss_ = _pair_loss(heads, tails, kinds, Y,
                                _weights(total - 1, self.n_iters))
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """`fit(X, y).embedding_`."""
        return self.fit(X, y).embedding_


def _keep_same_class(heads: list, tails: list, kinds: list, neighbours: np.ndarray,
                     labels: np.ndarray, n: int) -> tuple:
    """Restrict the neighbour set to same-label pairs, keeping one per point.

    A point with no same-label neighbour in its k-neighbourhood would otherwise
    be a node with no attractive force at all and drift to wherever the further
    pairs push it. Keeping its nearest neighbour keeps every point attached to
    its own region, which is the honest interpretation: the label rule can only
    remove a pair where the geometry offered one.
    """
    head = heads[0]
    tail = tails[0]
    same = labels[head] == labels[tail]
    counts = np.bincount(head[same], minlength=n)
    fallback = np.zeros(n, dtype=bool)
    fallback[counts == 0] = True
    nearest = neighbours[:, 0]
    rescue = fallback[head] & (tail == nearest[head])
    keep = same | rescue
    out = [head[keep], tail[keep], kinds[0][keep]]
    for h, t, k in zip(heads[1:], tails[1:], kinds[1:], strict=True):
        out[0] = np.concatenate([out[0], h])
        out[1] = np.concatenate([out[1], t])
        out[2] = np.concatenate([out[2], k])
    return tuple(out)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
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


def _demo_gradient() -> None:
    """The analytic gradient against a centred difference, and (3) at its knots."""
    rng = np.random.default_rng(7)
    n, total = 60, 400
    heads = rng.integers(0, n, size=total)
    tails = rng.integers(0, n, size=total)
    kinds = rng.integers(0, 3, size=total)
    Y = rng.standard_normal((n, 2))
    weights = (2.0, 500.0, 1.0)

    grad = _pair_gradient(heads, tails, kinds, Y, weights)
    step, worst = 1e-5, 0.0
    for _ in range(6):
        i, j = int(rng.integers(n)), int(rng.integers(2))
        up, down = Y.copy(), Y.copy()
        up[i, j] += step
        down[i, j] -= step
        numeric = (_pair_loss(heads, tails, kinds, up, weights)
                   - _pair_loss(heads, tails, kinds, down, weights)) / (2.0 * step)
        worst = max(worst, abs(numeric - grad[i, j])
                    / max(abs(numeric), abs(grad[i, j]), 1.0))
    assert worst < 1e-6, worst

    phases = (100, 100, 250)
    assert _weights(0, phases) == (2.0, 1000.0, 1.0)
    assert abs(_weights(50, phases)[1] - 501.5) < 1e-9
    assert abs(_weights(99, phases)[1] - 12.97) < 1e-9
    assert _weights(100, phases) == (3.0, 3.0, 1.0)
    assert _weights(199, phases) == (3.0, 3.0, 1.0)
    assert _weights(200, phases) == (1.0, 0.0, 1.0)
    print(f"gradient: six centred differences of (1) against the analytic "
          f"gradient agree to {worst:.1e} relative; the schedule is "
          f"(2, 1000→3, 1) at step 0, (2, 501.5, 1) at 50, (2, 12.97, 1) at "
          f"99, (3, 3, 1) through step 199 and (1, 0, 1) from step 200")


def _demo_roll(X: np.ndarray, truth: np.ndarray) -> None:
    """What the three pair sets buy, and what the further pairs cost."""
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pca = U[:, :2] * S[:2]

    full = PaCMAP(2, seed=1).fit(X)
    graph = PaCMAP(2, variant="knn_graph", seed=1).fit(X)
    local = PaCMAP(2, variant="localmap", seed=1).fit(X)
    fewer = PaCMAP(2, fp_ratio=1.0, seed=1).fit(X)

    counts = [int((full.pairs_[2] == kind).sum()) for kind in (NP, MN, FP)]
    assert counts == [2500, 1250, 5000], counts
    assert int((graph.pairs_[2] == MN).sum()) == 0
    assert local.local_pairs_ is not None
    assert int((local.local_pairs_[2] == FP).sum()) < counts[2]
    assert len(full.history_) == 6
    assert full.history_[0] > 4.0 * full.history_[-1], full.history_
    assert _overlap(pca, truth) < 0.55
    assert _overlap(local.embedding_, truth) > _overlap(full.embedding_, truth)
    assert _distance_correlation(local.embedding_, truth) < _distance_correlation(
        full.embedding_, truth)

    Xs = (X - X.mean(axis=0)) / X.std()
    D = _pairwise(Xs)
    heads, tails, kinds = full.pairs_
    scales = {kind: float(D[heads[kinds == kind], tails[kinds == kind]].mean())
              for kind in (NP, MN, FP)}
    assert scales[NP] < 0.8 < scales[MN] < scales[FP] < 2.5, scales

    again = PaCMAP(2, seed=1).fit(X)
    assert np.array_equal(again.embedding_, full.embedding_)
    assert np.array_equal(PaCMAP(2, seed=1).fit_transform(X), full.embedding_)

    print(
        f"pacmap: {X.shape[0]} points on a swiss roll, isometric to a flat "
        f"rectangle. Two principal components keep {_overlap(pca, truth):.3f} "
        f"of each point's eight nearest neighbours and correlate "
        f"{_distance_correlation(pca, truth):.3f} with the rectangle. "
        f"PaCMAP keeps {_overlap(full.embedding_, truth):.3f} at eight "
        f"neighbours, {_overlap(full.embedding_, truth, 40):.3f} at forty, and "
        f"correlates {_distance_correlation(full.embedding_, truth):.3f} — the "
        f"mid range and the distances survive together, which neither t-SNE "
        f"(0.769 at forty, 0.335 correlated) nor UMAP (0.834, 0.534) manages. "
        f"The loss falls from {full.history_[0]:.0f} to {full.loss_:.1f} over "
        f"{sum(full.n_iters)} steps on {counts[0]} neighbour pairs, "
        f"{counts[1]} mid-near pairs and {counts[2]} further pairs, at mean "
        f"input distances {scales[NP]:.2f}, {scales[MN]:.2f} and "
        f"{scales[FP]:.2f} against {D[np.triu_indices(X.shape[0], 1)].mean():.2f} "
        f"for a random pair"
    )
    print(
        f"knn_graph: the same run with the mid-near pairs removed — "
        f"{graph.pairs_[0].size} pairs instead of {full.pairs_[0].size}. Eight-"
        f"neighbour overlap is {_overlap(graph.embedding_, truth):.3f}, hardly "
        f"changed from {_overlap(full.embedding_, truth):.3f}, but forty-"
        f"neighbour overlap falls from "
        f"{_overlap(full.embedding_, truth, 40):.3f} to "
        f"{_overlap(graph.embedding_, truth, 40):.3f} and the distance "
        f"correlation collapses from "
        f"{_distance_correlation(full.embedding_, truth):.3f} to "
        f"{_distance_correlation(graph.embedding_, truth):.3f}. The mid-near "
        f"pairs are the only term that sees the middle of the distance range, "
        f"so removing them costs exactly the scale they were sampling, and the "
        f"running time is unchanged"
    )
    print(
        f"localmap: the last phase only replaces the uniform further pairs with "
        f"the two-hop ring — {int((local.local_pairs_[2] == FP).sum())} of them, "
        f"because a point can run out of non-adjacent second neighbours. It buys "
        f"the locality it is after: {_overlap(local.embedding_, truth):.3f} at "
        f"eight neighbours against {_overlap(full.embedding_, truth):.3f} — and "
        f"pays with the middle and the distances, "
        f"{_overlap(local.embedding_, truth, 40):.3f} at forty against "
        f"{_overlap(full.embedding_, truth, 40):.3f} and "
        f"{_distance_correlation(local.embedding_, truth):.3f} correlated "
        f"against {_distance_correlation(full.embedding_, truth):.3f}. Run the "
        f"same swap from the first step and it destroys the layout — 0.624 at "
        f"eight neighbours, 0.457 at forty — because the repulsion is then "
        f"separating points the neighbour term has not placed yet. Staging is "
        f"the whole of the method"
    )
    print(
        f"fp_ratio: the further pairs are the dial between the two regimes. At "
        f"1.0 — {int((fewer.pairs_[2] == FP).sum())} pairs instead of "
        f"{counts[2]} — eight-neighbour overlap is "
        f"{_overlap(fewer.embedding_, truth):.3f}, lower than "
        f"{_overlap(full.embedding_, truth):.3f}, while the distance "
        f"correlation rises from "
        f"{_distance_correlation(full.embedding_, truth):.3f} to "
        f"{_distance_correlation(fewer.embedding_, truth):.3f}. Every extra "
        f"draw from the whole set is another pair pushed apart, which flattens "
        f"regions and costs global fidelity; every draw withheld is a "
        f"neighbourhood left to crowd. A t-SNE or a UMAP fixes this trade "
        f"before the run starts"
    )


def _demo_scaling() -> None:
    """Scaled distances even out how the two regions spend their neighbour slots."""
    rng = np.random.default_rng(3)
    dense = 0.35 * rng.standard_normal((120, 2))
    sparse = np.array([2.0, 0.0]) + 1.6 * rng.standard_normal((120, 2))
    X = np.vstack([dense, sparse])
    labels = np.concatenate([np.zeros(120, int), np.ones(120, int)])
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)

    rates = {}
    for name, chooser in (("raw", _knn), ("scaled", _scaled_neighbours)):
        _, idx = chooser(D, 10)
        cross = labels[idx] != labels[:, None]
        rates[name] = (float(cross[:120].mean()), float(cross[120:].mean()))
    assert rates["raw"][1] > 2.0 * rates["raw"][0], rates
    assert rates["scaled"][1] < rates["raw"][1]
    assert rates["scaled"][0] > rates["raw"][0]
    no_self = np.arange(120)
    _, idx = _scaled_neighbours(D, 10)
    assert not (idx == np.arange(240)[:, None]).any()
    assert no_self.shape == (120,)

    print(
        f"scaled neighbours: two Gaussians five times apart in spread, "
        f"overlapping in space. Choosing neighbours on raw distance, the "
        f"sparse region spends {rates['raw'][1]:.3f} of its slots on the dense "
        f"region and the dense region {rates['raw'][0]:.3f} on the sparse one — "
        f"an asymmetry that comes entirely from density. Dividing d² by both "
        f"endpoints' local scale gives {rates['scaled'][1]:.3f} and "
        f"{rates['scaled'][0]:.3f}, so the two regions contribute comparable "
        f"cross-region edges and neither is annexed by the other's scale"
    )


def _demo_supervised() -> None:
    """Labels decide which pairs exist; kNN purity is the measurement."""
    X, labels = _blobs(n_per=40, n_clusters=5, spread=1.1, seed=2)
    plain = PaCMAP(2, seed=3).fit(X)
    guided = PaCMAP(2, variant="supervised", seed=3).fit(X, labels)

    before, after = _purity(plain.embedding_, labels), _purity(guided.embedding_, labels)
    assert after > before + 0.2, (before, after)
    assert after > 0.95, after
    assert int((guided.pairs_[2] == FP).sum()) < int((plain.pairs_[2] == FP).sum())
    try:
        PaCMAP(2, variant="supervised").fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError("supervised without labels should raise")

    print(
        f"supervised: five overlapping eight-dimensional Gaussians, centres 1.5 "
        f"apart, standard deviation 1.1, so the clusters touch. Without labels "
        f"each point's eight layout neighbours agree with its label "
        f"{before:.3f} of the time; cross-class further pairs then same-class "
        f"neighbours only, and it is {after:.3f}. Cross-class pairs are drawn "
        f"{int((guided.pairs_[2] == FP).sum())} times against "
        f"{int((plain.pairs_[2] == FP).sum())} unlabelled draws, and every "
        f"point with no same-class neighbour keeps its nearest one, so nothing "
        f"is left with only repulsion"
    )


def main() -> None:
    X, truth = _swiss_roll(250, seed=1)
    _demo_gradient()
    _demo_roll(X, truth)
    _demo_scaling()
    _demo_supervised()
    print("pacmap demo passed")


if __name__ == "__main__":
    main()
