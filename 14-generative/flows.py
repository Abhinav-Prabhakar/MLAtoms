"""Normalising flows — exact likelihoods through invertible maps.

Chapter:   14-generative
Companion: flows.md
Tier 2 · 6 variants: realnvp, glow, coupling, autoregressive_flow,
                    spline_flow, continuous_flow
"""

from __future__ import annotations

import numpy as np

VARIANTS = (
    "coupling",
    "realnvp",
    "glow",
    "autoregressive_flow",
    "spline_flow",
    "continuous_flow",
)

DIM = 2
MODES = 8
RADIUS = 0.8
SPREAD = 0.07
PER_MODE = 256
HIDDEN = 32
BATCH = 256
TRAIN_STEPS = 3000
LR = 2e-3
ANGLE = np.pi / MODES
SPLINE_BINS = 8
SPLINE_RANGE = 4.0
RK4_STEPS = 20

# (kind, number of blocks). The block count is the headline architectural
# difference; the continuous flow has no blocks, only a velocity field.
ARCH = {
    "coupling": ("affine", 6),
    "realnvp": ("realnvp", 6),
    "glow": ("glow", 5),
    "autoregressive_flow": ("maf", 2),
    "spline_flow": ("spline", 4),
    "continuous_flow": ("cnf", 0),
}


# --------------------------------------------------------------------------- #
# Data and the exact answer
# --------------------------------------------------------------------------- #
def centres() -> np.ndarray:
    """The eight mode centres, shape (8, 2)."""
    angle = 2.0 * np.pi * np.arange(MODES) / MODES + ANGLE
    return RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def make_data(seed: int, per_mode: int = PER_MODE) -> np.ndarray:
    """Eight Gaussians, shuffled. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    flat = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, DIM))
                      for c in centres()])
    return flat[rng.permutation(len(flat))]


def mixture_logpdf(x: np.ndarray) -> np.ndarray:
    """Log density of the true mixture at each row of `x`."""
    x = np.asarray(x, dtype=float)
    var = SPREAD ** 2
    lp = np.stack([-0.5 * ((x - c) ** 2).sum(axis=1) / var
                   - np.log(2.0 * np.pi * var) for c in centres()], axis=1)
    top = lp.max(axis=1, keepdims=True)
    return top[:, 0] + np.log(np.exp(lp - top).mean(axis=1))


def exact_nll(x: np.ndarray) -> float:
    """Negative log-likelihood of the true mixture, in nats per sample."""
    return float(-mixture_logpdf(x).mean())


def mode_stats(samples: np.ndarray) -> tuple:
    """(modes covered, largest share, mean dist to centre, mean radius)."""
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    share = np.bincount(d.argmin(axis=1), minlength=MODES) / len(s)
    return (int(np.sum(share > 0.03)), float(share.max()),
            float(d.min(axis=1).mean()), float(np.linalg.norm(s, axis=1).mean()))


# --------------------------------------------------------------------------- #
# Small MLP and optimiser
# --------------------------------------------------------------------------- #
def _silu(x: np.ndarray) -> np.ndarray:
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))


def _dsilu(x: np.ndarray) -> np.ndarray:
    s = 0.5 * (1.0 + np.tanh(0.5 * x))
    return s + x * s * (1.0 - s)


def _mlp_init(rng: np.random.Generator, sizes: list) -> list:
    """Two hidden layers; a flat list of (weight, bias) pairs."""
    out = []
    for a, b in zip(sizes, sizes[1:], strict=False):
        out += [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)]
    return out


def _mlp_forward(P: list, h: np.ndarray) -> tuple:
    h1 = h @ P[0] + P[1]
    a1 = _silu(h1)
    h2 = a1 @ P[2] + P[3]
    a2 = _silu(h2)
    return a2 @ P[4] + P[5], (h, h1, a1, h2, a2)


def _mlp_backward(P: list, cache: tuple, d: np.ndarray) -> tuple:
    """Return (gradient wrt the input, per-parameter gradients)."""
    h, h1, a1, h2, a2 = cache
    gW3, gb3 = a2.T @ d, d.sum(axis=0)
    d = (d @ P[4].T) * _dsilu(h2)
    gW2, gb2 = a1.T @ d, d.sum(axis=0)
    d = (d @ P[2].T) * _dsilu(h1)
    gW1, gb1 = h.T @ d, d.sum(axis=0)
    return d @ P[0].T, [gW1, gb1, gW2, gb2, gW3, gb3]


class _Adam:
    """Adam driven by summed (not averaged) gradients."""

    def __init__(self, params: list, lr: float) -> None:
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list, grads: list, scale: float = 1.0) -> None:
        self.t += 1
        c1 = 1.0 - 0.9 ** self.t
        c2 = 1.0 - 0.999 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            if g is None:
                continue
            g = g * scale
            m[...] = 0.9 * m + 0.1 * g
            v[...] = 0.999 * v + 0.001 * g ** 2
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
def _cond(params: list, block: dict, x1: np.ndarray) -> tuple:
    """Conditioner: x1 -> (bounded log-scale, shift) plus its cache."""
    q = block["p"]
    outs = []
    for k in range(2 if block["split"] else 1):
        sub = [params[i] for i in q[6 * k:6 * k + 6]]
        outs.append(_mlp_forward(sub, x1[:, None]))
    if len(outs) == 1:
        raw_s, raw_t = outs[0][0][:, 0], outs[0][0][:, 1]
    else:
        raw_s, raw_t = outs[0][0][:, 0], outs[1][0][:, 0]
    s = block["scale_range"] * np.tanh(0.5 * raw_s)
    return s, raw_t, (outs, raw_s)


def _cond_backward(block: dict, cache: tuple, params: list,
                   ds: np.ndarray, dt: np.ndarray) -> tuple:
    """Return (gradient wrt x1, per-parameter gradients)."""
    outs, raw_s = cache
    q = block["p"]
    r = block["scale_range"]
    ds = ds * r * 0.5 * (1.0 - np.tanh(0.5 * raw_s) ** 2)
    if len(outs) == 1:
        din, grads = _mlp_backward([params[i] for i in q[:6]], outs[0][1],
                                   np.stack([ds, dt], axis=1))
        return din[:, 0], grads
    din = np.zeros_like(dt)
    grads: list = []
    for k, col in enumerate((ds, dt)):
        sub = [params[i] for i in q[6 * k:6 * k + 6]]
        d = np.zeros((len(col), 2))
        d[:, 0] = col
        dsub, gsub = _mlp_backward(sub, outs[k][1], d)
        din = din + dsub[:, 0]
        grads += gsub
    return din, grads


def _affine_forward(block: dict, x: np.ndarray, params: list) -> tuple:
    if block["flip"]:
        x = x[:, ::-1]
    x1, x2 = x[:, 0], x[:, 1]
    s, t, cache = _cond(params, block, x1)
    if block["scale_first"]:
        mu, log_s = params[block["p"][6]], params[block["p"][7]]
        y1 = (x1 - mu) * np.exp(-log_s)
    else:
        y1, log_s = x1, None
    y2 = (x2 - t) * np.exp(-s)
    y = np.stack([y1, y2], axis=1)
    ld = -s - (np.full(len(x), log_s[0]) if log_s is not None else 0.0)
    if block["flip"]:
        y = y[:, ::-1]
    return y, ld, (x1, x2, s, t, cache, y1)


def _affine_backward(block: dict, d: np.ndarray, cache: tuple,
                     params: list) -> tuple:
    x1, x2, s, t, cond_cache, y1 = cache
    if block["flip"]:
        d = d[:, ::-1]
    d1, d2 = d[:, 0], d[:, 1]
    e = np.exp(-s)
    dx2 = d2 * e
    ds = 1.0 - d2 * (x2 - t) * e
    dt = -d2 * e
    din, grads = _cond_backward(block, cond_cache, params, ds, dt)
    if block["scale_first"]:
        log_s = params[block["p"][7]]
        dx1 = d1 * np.exp(-log_s) + din
        grads = grads + [np.array([-d1.sum() * np.exp(-log_s[0])]),
                         np.array([len(d1) - (d1 * y1).sum()])]
    else:
        dx1 = d1 + din
    out = np.stack([dx1, dx2], axis=1)
    if block["flip"]:
        out = out[:, ::-1]
    return out, grads


def _actnorm_forward(block: dict, x: np.ndarray, params: list) -> tuple:
    shift, log_scale = params[block["p"][0]], params[block["p"][1]]
    y = (x - shift) * np.exp(-log_scale)
    return y, np.full(len(x), -log_scale.sum()), (y,)


def _actnorm_backward(block: dict, d: np.ndarray, cache: tuple,
                      params: list) -> tuple:
    (y,) = cache
    log_scale = params[block["p"][1]]
    dx = d * np.exp(-log_scale)
    return dx, [-dx.sum(axis=0), len(d) - (d * y).sum(axis=0)]


def _mix_matrix(params: list, block: dict) -> np.ndarray:
    """Lower-triangular 1x1 convolution: [[e^a, 0], [b, e^d]]."""
    a, b, d = (params[i][0] for i in block["p"])
    return np.array([[np.exp(a), 0.0], [b, np.exp(d)]])


def _mix_forward(block: dict, x: np.ndarray, params: list) -> tuple:
    W = _mix_matrix(params, block)
    y = x @ W.T
    return y, np.full(len(x), np.log(W[0, 0]) + np.log(W[1, 1])), (x, y)


def _mix_backward(block: dict, d: np.ndarray, cache: tuple,
                  params: list) -> tuple:
    x, y = cache
    W = _mix_matrix(params, block)
    dx = d @ W
    n = len(d)
    grads = [np.array([-n + (d[:, 0] * y[:, 0]).sum()]),
             np.array([(d[:, 1] * x[:, 0]).sum()]),
             np.array([-n + (d[:, 1] * (y[:, 1] - W[1, 0] * x[:, 0])).sum()])]
    return dx, grads


def _spline_weights(u: np.ndarray) -> np.ndarray:
    z = u - u.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _spline_knots(w: np.ndarray) -> np.ndarray:
    """Left edges of the K bins, plus the closing edge; shape (n, K+1)."""
    return np.concatenate([np.full((len(w), 1), -SPLINE_RANGE),
                           -SPLINE_RANGE + 2.0 * SPLINE_RANGE
                           * np.cumsum(w, axis=1)], axis=1)


def _spline_forward(block: dict, x: np.ndarray, params: list) -> tuple:
    if block["flip"]:
        x = x[:, ::-1]
    x1, x2 = x[:, 0], x[:, 1]
    u, mlp_cache = _mlp_forward([params[i] for i in block["p"]],
                                x1[:, None])
    w = _spline_weights(u)
    knots = _spline_knots(w)
    slope = SPLINE_BINS * w
    width = 2.0 * SPLINE_RANGE / SPLINE_BINS
    b = np.clip(np.floor((x2 + SPLINE_RANGE) / width).astype(int),
                0, SPLINE_BINS - 1)
    inside = (x2 > -SPLINE_RANGE) & (x2 < SPLINE_RANGE)
    rows = np.arange(len(x2))
    y2 = np.where(inside, knots[rows, b] + slope[rows, b]
                  * (x2 + SPLINE_RANGE - b * width), x2)
    ld = np.where(inside, np.log(slope[rows, b]), 0.0)
    y = np.stack([x1, y2], axis=1)
    if block["flip"]:
        y = y[:, ::-1]
    return y, ld, (x1, x2, w, u, knots, slope, b, inside, mlp_cache, block)


def _spline_backward(block: dict, d: np.ndarray, cache: tuple,
                     params: list) -> tuple:
    x1, x2, w, u, knots, slope, b, inside, mlp_cache, blk = cache
    if blk["flip"]:
        d = d[:, ::-1]
    d1, d2 = d[:, 0], d[:, 1]
    rows = np.arange(len(x2))
    width = 2.0 * SPLINE_RANGE / SPLINE_BINS
    dx2 = np.where(inside, d2 / slope[rows, b], d2)
    below = (np.arange(SPLINE_BINS)[None, :] < b[:, None]).astype(float)
    g = inside[:, None] * 2.0 * SPLINE_RANGE * (d2[:, None] * below)
    g[rows, b] += inside * (SPLINE_BINS * (x2 + SPLINE_RANGE - b * width) * d2
                            - 1.0 / np.maximum(w[rows, b], 1e-9))
    du = w * (g - (g * w).sum(axis=1, keepdims=True))
    din, grads = _mlp_backward([params[i] for i in block["p"]], mlp_cache, du)
    out = np.stack([d1 + din[:, 0], dx2], axis=1)
    if blk["flip"]:
        out = out[:, ::-1]
    return out, grads


# --------------------------------------------------------------------------- #
# The flow
# --------------------------------------------------------------------------- #
class Flow:
    """A stack of invertible blocks with an exact log-density.

    `coupling`, `realnvp`, `glow` and `autoregressive_flow` differ only in
    which blocks are stacked; `spline_flow` swaps the affine map for a
    monotone piecewise-linear one. `continuous_flow` replaces the stack with
    a velocity field trained by flow matching, integrated with RK4, whose
    density comes from a divergence integral instead of a log-determinant sum.
    """

    def __init__(self, variant: str = "coupling", *, hidden: int = HIDDEN,
                 blocks: int | None = None, train_steps: int = TRAIN_STEPS,
                 batch: int = BATCH, lr: float = LR, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not isinstance(hidden, (int, np.integer)) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not isinstance(train_steps, (int, np.integer)) or train_steps < 1:
            raise ValueError("train_steps must be a positive integer")
        if not isinstance(batch, (int, np.integer)) or batch < 1:
            raise ValueError("batch must be a positive integer")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.kind, default = ARCH[variant]
        self.n_blocks = default if blocks is None else blocks
        if self.kind != "cnf" and self.n_blocks < 1:
            raise ValueError("blocks must be at least one")
        self.hidden = int(hidden)
        self.train_steps = int(train_steps)
        self.batch = int(batch)
        self.lr = float(lr)
        self.seed = seed
        self.params: list = []
        self.blocks: list = []
        self.fitted = False

    # -- construction --------------------------------------------------------- #
    def _build(self, rng: np.random.Generator) -> None:
        kind, n = self.kind, self.n_blocks
        self.params, self.blocks = [], []
        if kind == "cnf":
            self.params = _mlp_init(
                rng, [DIM + 1, self.hidden, self.hidden, DIM])
            return
        params, blocks = self.params, self.blocks

        def affine(flip: bool, scale_first: bool, split: bool) -> None:
            start = len(params)
            for _ in range(2 if split else 1):
                params.extend(_mlp_init(rng, [1, self.hidden, self.hidden, 2]))
            idx = list(range(start, len(params)))
            if scale_first:
                params.extend([np.zeros(1), np.zeros(1)])
                idx = idx + [len(idx) + start, len(idx) + start + 1]
            blocks.append({"kind": "affine", "p": idx, "flip": flip,
                           "scale_first": scale_first, "split": split,
                           "scale_range": 2.0})

        def actnorm() -> None:
            start = len(params)
            params.extend([np.zeros(DIM), np.zeros(DIM)])
            blocks.append({"kind": "actnorm", "p": [start, start + 1]})

        def mix() -> None:
            start = len(params)
            params.extend([np.zeros(1), np.zeros(1), np.zeros(1)])
            blocks.append({"kind": "mix", "p": [start, start + 1, start + 2]})

        def spline(flip: bool) -> None:
            start = len(params)
            params.extend(_mlp_init(rng, [1, self.hidden, self.hidden,
                                          SPLINE_BINS]))
            blocks.append({"kind": "spline",
                           "p": list(range(start, start + 6)), "flip": flip})

        for i in range(n):
            flip = bool(i % 2)
            if kind == "affine":
                affine(flip, False, False)
            elif kind == "realnvp":
                actnorm()
                affine(flip, False, True)
            elif kind == "glow":
                actnorm()
                mix()
                affine(flip, False, False)
            elif kind == "maf":
                affine(flip, True, False)
            else:
                spline(flip)

    def _init_actnorm(self, x: np.ndarray) -> None:
        """Data-dependent init: the first batch maps to zero mean unit var."""
        h = x
        for block in self.blocks:
            if block["kind"] == "actnorm":
                shift = self.params[block["p"][0]]
                log_scale = self.params[block["p"][1]]
                shift[...] = h.mean(axis=0)
                log_scale[...] = np.log(h.std(axis=0) + 1e-6)
            h = self._block_forward(block, h)[0]

    # -- forward / inverse ---------------------------------------------------- #
    def _block_forward(self, block: dict, x: np.ndarray) -> tuple:
        kind = block["kind"]
        if kind == "affine":
            return _affine_forward(block, x, self.params)
        if kind == "actnorm":
            return _actnorm_forward(block, x, self.params)
        if kind == "mix":
            return _mix_forward(block, x, self.params)
        return _spline_forward(block, x, self.params)

    def _forward(self, x: np.ndarray, keep: bool = False) -> tuple:
        y, logdet, caches = x, np.zeros(len(x)), []
        for block in self.blocks:
            y, ld, cache = self._block_forward(block, y)
            logdet = logdet + ld
            if keep:
                caches.append(cache)
        return y, logdet, caches

    def _backward(self, d: np.ndarray, caches: list) -> list:
        grads: list = [None] * len(self.params)
        pairs = zip(reversed(self.blocks), reversed(caches), strict=True)
        for block, cache in pairs:
            if block["kind"] == "affine":
                d, g = _affine_backward(block, d, cache, self.params)
            elif block["kind"] == "actnorm":
                d, g = _actnorm_backward(block, d, cache, self.params)
            elif block["kind"] == "mix":
                d, g = _mix_backward(block, d, cache, self.params)
            else:
                d, g = _spline_backward(block, d, cache, self.params)
            for i, gi in zip(block["p"], g, strict=True):
                grads[i] = gi
        return grads

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        for block in reversed(self.blocks):
            kind = block["kind"]
            if kind == "actnorm":
                shift = self.params[block["p"][0]]
                y = y * np.exp(self.params[block["p"][1]]) + shift
            elif kind == "mix":
                y = y @ np.linalg.inv(_mix_matrix(self.params, block)).T
            elif kind == "affine":
                y = self._affine_inverse(block, y)
            else:
                y = self._spline_inverse(block, y)
        return y

    def _affine_inverse(self, block: dict, y: np.ndarray) -> np.ndarray:
        if block["flip"]:
            y = y[:, ::-1]
        y1, y2 = y[:, 0], y[:, 1]
        if block["scale_first"]:
            mu = self.params[block["p"][6]][0]
            log_s = self.params[block["p"][7]][0]
            x1 = y1 * np.exp(log_s) + mu
        else:
            x1 = y1
        s, t, _ = _cond(self.params, block, x1)
        x2 = y2 * np.exp(s) + t
        out = np.stack([x1, x2], axis=1)
        return out[:, ::-1] if block["flip"] else out

    def _spline_inverse(self, block: dict, y: np.ndarray) -> np.ndarray:
        if block["flip"]:
            y = y[:, ::-1]
        y1, y2 = y[:, 0], y[:, 1]
        u, _ = _mlp_forward([self.params[i] for i in block["p"]], y1[:, None])
        w = _spline_weights(u)
        knots = _spline_knots(w)
        slope = SPLINE_BINS * w
        width = 2.0 * SPLINE_RANGE / SPLINE_BINS
        inside = (y2 > -SPLINE_RANGE) & (y2 < SPLINE_RANGE)
        b = np.clip((y2[:, None] > knots).sum(axis=1) - 1, 0, SPLINE_BINS - 1)
        rows = np.arange(len(y2))
        x2 = np.where(inside, -SPLINE_RANGE + b * width
                      + (y2 - knots[rows, b]) / slope[rows, b], y2)
        out = np.stack([y1, x2], axis=1)
        return out[:, ::-1] if block["flip"] else out

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray) -> Flow:
        x = _check_data(x)
        rng = np.random.default_rng(self.seed)
        self._build(rng)
        if self.kind == "cnf":
            self._fit_cnf(x, rng)
        else:
            self._init_actnorm(x)
            opt = _Adam(self.params, self.lr)
            for _ in range(self.train_steps):
                idx = rng.integers(0, len(x), self.batch)
                y, _, caches = self._forward(x[idx], keep=True)
                grads = self._backward(y, caches)
                opt.step(self.params, grads, scale=1.0 / self.batch)
        self.fitted = True
        return self

    def _fit_cnf(self, x: np.ndarray, rng: np.random.Generator) -> None:
        opt = _Adam(self.params, self.lr)
        for _ in range(self.train_steps):
            x1 = x[rng.integers(0, len(x), self.batch)]
            x0 = rng.normal(0.0, 1.0, x1.shape)
            t = rng.random(self.batch)
            xt = (1.0 - t)[:, None] * x0 + t[:, None] * x1
            v, cache = _mlp_forward(
                self.params,
                np.concatenate([xt, t[:, None]], axis=1))
            d = 2.0 * (v - (x1 - x0)) / DIM
            _, grads = _mlp_backward(self.params, cache, d)
            opt.step(self.params, grads, scale=1.0 / self.batch)

    # -- outputs -------------------------------------------------------------- #
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit first")
        x = _check_data(x)
        if self.kind == "cnf":
            z, acc = self._cnf_solve(x, 1.0, 0.0, want_div=True)
            return -0.5 * (z ** 2).sum(axis=1) - np.log(2.0 * np.pi) + acc
        y, logdet, _ = self._forward(x)
        return -0.5 * (y ** 2).sum(axis=1) - np.log(2.0 * np.pi) + logdet

    def nll(self, x: np.ndarray) -> float:
        return float(-self.log_prob(x).mean())

    def sample(self, n: int = 1024, *, seed: int = 0) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit first")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        z = np.random.default_rng(seed).normal(0.0, 1.0, (n, DIM))
        if self.kind == "cnf":
            return self._cnf_solve(z, 0.0, 1.0)
        return self._inverse(z)

    def round_trip(self, x: np.ndarray) -> float:
        """Largest coordinate error after forward then inverse."""
        if not self.fitted:
            raise RuntimeError("call fit first")
        x = _check_data(x)
        if self.kind == "cnf":
            z = self._cnf_solve(x, 1.0, 0.0)
            return float(np.abs(self._cnf_solve(z, 0.0, 1.0) - x).max())
        return float(np.abs(self._inverse(self._forward(x)[0]) - x).max())

    def jacobian(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Finite-difference Jacobian of the forward map, for the demo."""
        x = _check_data(x)
        cols = []
        for k in range(DIM):
            e = np.zeros(DIM)
            e[k] = eps
            up = (self._forward(x + e)[0] - self._forward(x - e)[0]) / (2 * eps)
            cols.append(up)
        return np.stack(cols, axis=2)

    def n_params(self) -> int:
        if not self.params:
            raise RuntimeError("call fit first")
        return int(sum(p.size for p in self.params))

    def spline_widths(self) -> np.ndarray:
        """Median learned bin widths across positions, per block."""
        if self.kind != "spline":
            raise ValueError("only spline_flow has bin widths")
        grid = np.linspace(-RADIUS - 0.4, RADIUS + 0.4, 256)[:, None]
        out = []
        for block in self.blocks:
            u, _ = _mlp_forward([self.params[i] for i in block["p"]], grid)
            out.append(2.0 * SPLINE_RANGE * _spline_weights(u).mean(axis=0))
        return np.stack(out)

    # -- continuous flow ------------------------------------------------------ #
    def _cnf_solve(self, x: np.ndarray, t0: float, t1: float,
                   want_div: bool = False) -> tuple:
        h = (t1 - t0) / RK4_STEPS
        acc = np.zeros(len(x))
        for i in range(RK4_STEPS):
            t = np.full(len(x), t0 + i * h)
            if want_div:
                k1, d1 = self._velocity_div(x, t)
                k2, d2 = self._velocity_div(x + 0.5 * h * k1, t + 0.5 * h)
                k3, d3 = self._velocity_div(x + 0.5 * h * k2, t + 0.5 * h)
                k4, d4 = self._velocity_div(x + h * k3, t + h)
                acc = acc + h / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            else:
                k1 = self._velocity(x, t)
                k2 = self._velocity(x + 0.5 * h * k1, t + 0.5 * h)
                k3 = self._velocity(x + 0.5 * h * k2, t + 0.5 * h)
                k4 = self._velocity(x + h * k3, t + h)
            x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return (x, acc) if want_div else x

    def _velocity(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        inp = np.concatenate([x, np.asarray(t)[:, None]], axis=1)
        return _mlp_forward(self.params, inp)[0]

    def _velocity_div(self, x: np.ndarray, t: np.ndarray) -> tuple:
        """Velocity and its exact divergence, via the 2x2 Jacobian."""
        inp = np.concatenate([x, np.asarray(t)[:, None]], axis=1)
        P = self.params
        h1 = inp @ P[0] + P[1]
        a1 = _silu(h1)
        h2 = a1 @ P[2] + P[3]
        a2 = _silu(h2)
        v = a2 @ P[4] + P[5]
        A = P[0][None] * _dsilu(h1)[:, None, :]
        B = P[2][None] * _dsilu(h2)[:, None, :]
        J = np.einsum("nij,njk,kl->nil", A, B, P[4])
        return v, J[:, 0, 0] + J[:, 1, 1]


def _check_data(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(f"x must have shape (n, {DIM})")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    return x


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _refused(fn) -> bool:
    try:
        fn()
    except (ValueError, RuntimeError, TypeError):
        return True
    return False


def demo() -> None:
    train, test = make_data(0), make_data(1)
    exact = exact_nll(test)
    base = float(np.mean(0.5 * (test ** 2).sum(axis=1) + np.log(2.0 * np.pi)))
    print(f"data: {len(train)} train + {len(test)} test, {MODES} modes "
          f"at radius {RADIUS} with sd {SPREAD}")
    print(f"reference: exact mixture {exact:.3f} nats, "
          f"standard normal {base:.3f} nats")

    span = 1.6
    ax = np.linspace(-span, span, 401)
    gx, gy = np.meshgrid(ax, ax)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    area = float((ax[1] - ax[0]) ** 2)
    mass = float(np.exp(mixture_logpdf(grid)).sum() * area)
    print(f"the exact density integrates to {mass:.6f} on "
          f"[-{span}, {span}]^2")
    assert abs(mass - 1.0) < 1e-4

    print("\nvariant               nll     gap  params  modes   share    dist"
          "  radius")
    rows = {}
    for name in VARIANTS:
        flow = Flow(name).fit(train)
        cov, share, dist, radius = mode_stats(flow.sample(1024, seed=1))
        rows[name] = flow
        print(f"{name:20s} {flow.nll(test):6.3f} {flow.nll(test) - exact:7.3f}"
              f" {flow.n_params():7d}   {cov}/{MODES}  {share:5.3f}  {dist:6.3f}"
              f"   {radius:6.3f}")
        assert flow.nll(test) < base - 1.0
        assert flow.nll(test) > exact - 0.2
        assert cov == MODES

    print("\ndepth (coupling, same data and budget)")
    shallow = {}
    for n in (1, 2, 3):
        shallow[n] = Flow("coupling", blocks=n).fit(train)
        print(f"  {n} block{'s' if n > 1 else ' '}  nll "
              f"{shallow[n].nll(test):6.3f}   params "
              f"{shallow[n].n_params():5d}")
    print(f"  6 blocks  nll {rows['coupling'].nll(test):6.3f}   "
          f"params {rows['coupling'].n_params():5d}")
    assert shallow[1].nll(test) > shallow[2].nll(test) > shallow[3].nll(test)
    assert rows["coupling"].nll(test) < shallow[3].nll(test)
    hot = Flow("coupling", lr=2e-2).fit(train)
    print(f"  lr 2e-2   nll {hot.nll(test):6.3f}   (ten times the step)")
    assert hot.nll(test) > rows["coupling"].nll(test)
    cold = Flow("coupling", lr=2e-3 / BATCH).fit(train)
    print(f"  lr 7.8e-6 nll {cold.nll(test):6.3f}   (a 256th of the step)")
    assert cold.nll(test) > base

    print("\nintegrity")
    for name in ("coupling", "realnvp", "glow", "autoregressive_flow",
                 "spline_flow"):
        err = rows[name].round_trip(test)
        print(f"  {name:20s} forward-then-inverse error {err:.2e}")
        assert err < 1e-10
    err = rows["continuous_flow"].round_trip(test[:64])
    print(f"  {'continuous_flow':20s} forward-then-inverse error {err:.2e} "
          f"(RK4, {RK4_STEPS} steps)")
    assert err < 1e-4

    flow = rows["coupling"]
    pts = test[:6]
    _, logged, _ = flow._forward(pts)
    numeric = np.linalg.slogdet(flow.jacobian(pts))[1]
    print(f"  {'coupling':20s} log-determinant vs log|det J| "
          f"{np.abs(logged - numeric).max():.2e}")
    assert np.abs(logged - numeric).max() < 1e-4

    cnf = rows["continuous_flow"]
    probe = np.random.default_rng(5).normal(0.0, 1.0, (8, DIM))
    times = np.linspace(0.05, 0.95, 8)
    _, div = cnf._velocity_div(probe, times)
    num = np.zeros(len(probe))
    for k in range(DIM):
        e = np.zeros(DIM)
        e[k] = 1e-6
        num += (cnf._velocity(probe + e, times)[:, k]
                - cnf._velocity(probe - e, times)[:, k]) / 2e-6
    print(f"  {'continuous_flow':20s} divergence vs trace of J "
          f"{np.abs(num - div).max():.2e}")
    assert np.abs(num - div).max() < 1e-8

    mass = float(np.exp(flow.log_prob(grid)).sum() * area)
    print(f"  {'coupling':20s} learned density integrates to {mass:.4f}")
    assert abs(mass - 1.0) < 0.05

    coarse = np.linspace(-span, span, 61)
    cx, cy = np.meshgrid(coarse, coarse)
    cgrid = np.stack([cx.ravel(), cy.ravel()], axis=1)
    cmass = float(np.exp(cnf.log_prob(cgrid)).sum()
                  * (coarse[1] - coarse[0]) ** 2)
    print(f"  {'continuous_flow':20s} learned density integrates to "
          f"{cmass:.4f}")
    assert abs(cmass - 1.0) < 0.08

    latent, _, _ = flow._forward(test)
    print(f"\ncoupling latent: mean {latent.mean():+.3f}, "
          f"sd {latent.std():.3f}, correlation "
          f"{np.corrcoef(latent.T)[0, 1]:+.3f}")
    assert abs(latent.mean()) < 0.06
    assert abs(latent.std() - 1.0) < 0.08

    widths = rows["spline_flow"].spline_widths()
    print("spline bin widths (8 bins spanning 8.0 units, per block):")
    for i, row in enumerate(widths):
        cells = " ".join(f"{v:.2f}" for v in row)
        print(f"  block {i}  {cells}   ratio {row.max() / row.min():.2f}")
    assert widths.max() / widths.min() > 1.5

    a = flow.sample(256, seed=7)
    b = flow.sample(256, seed=7)
    assert np.array_equal(a, b)

    cases = [
        lambda: Flow("nope"),
        lambda: Flow("coupling", hidden=0),
        lambda: Flow("coupling", blocks=0),
        lambda: Flow("coupling", train_steps=0),
        lambda: Flow("coupling", batch=0),
        lambda: Flow("coupling", lr=0.0),
        lambda: Flow("coupling", lr=float("nan")),
        lambda: Flow("coupling").sample(4),
        lambda: Flow("coupling").log_prob(test),
        lambda: Flow("coupling").round_trip(test),
        lambda: Flow("coupling").n_params(),
        lambda: flow.sample(0),
        lambda: flow.sample(2.5),
        lambda: flow.log_prob(np.zeros((4, 3))),
        lambda: flow.log_prob(np.array([[0.0, np.nan]])),
        lambda: flow.spline_widths(),
        lambda: Flow("coupling").fit(np.zeros((4, 3))),
    ]
    refused = sum(_refused(case) for case in cases)
    print(f"\n{refused} of {len(cases)} malformed calls refused")
    assert refused == len(cases)

    print("\nflows demo passed")


if __name__ == "__main__":
    demo()
