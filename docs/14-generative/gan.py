"""Generative adversarial network — seven losses, one minimax game.

Chapter:   14-generative
Companion: gan.md
Tier 1 · 7 variants: vanilla, dcgan, lsgan, hinge, non_saturating, conditional,
infogan
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("vanilla", "dcgan", "lsgan", "hinge", "non_saturating",
            "conditional", "infogan")

MODES = 8
RADIUS = 0.8
SPREAD = 0.07
ZDIM = 2
CODE = 8
HID = 64
BATCH = 128
STEPS = 4000
LR = 2e-4
BETA1 = 0.5
LEAK = 0.2
MI_WEIGHT = 1.0


# --------------------------------------------------------------------------- #
# Data: eight Gaussians on a circle, scaled to fit inside the tanh output range
# --------------------------------------------------------------------------- #
def centres() -> np.ndarray:
    """The eight mode centres, (8, 2)."""
    angle = 2.0 * np.pi * np.arange(MODES) / MODES
    return RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def make_data(seed: int, per_mode: int = 256) -> tuple:
    """Return (x, y) with `per_mode` points around each of the eight centres."""
    rng = np.random.default_rng(seed)
    x = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, 2))
                   for c in centres()])
    y = np.repeat(np.arange(MODES), per_mode)
    order = rng.permutation(len(y))
    return x[order], y[order]


def mode_stats(samples: np.ndarray) -> tuple:
    """Return (modes covered, largest share, mean distance to nearest centre).

    A mode counts as covered when it wins more than 3% of the samples, so a
    collapsed generator scores one and a uniform one scores eight.
    """
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    share = np.bincount(d.argmin(axis=1), minlength=MODES) / len(s)
    return int(np.sum(share > 0.03)), float(share.max()), float(d.min(1).mean())


def code_report(samples: np.ndarray) -> tuple:
    """Return (fraction landing on their own centre, mean distance to it).

    `samples` must be stacked by code: 64 draws for code 0, then 64 for code 1,
    and so on, which is how the demo builds it.
    """
    s = np.asarray(samples, dtype=float)
    if len(s) % MODES:
        raise ValueError("sample count must be a multiple of the code count")
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    own = np.repeat(np.arange(MODES), len(s) // MODES)
    return (float(np.mean(d.argmin(axis=1) == own)),
            float(d[np.arange(len(s)), own].mean()))


# --------------------------------------------------------------------------- #
# Networks: a list of dense layers, leaky ReLU inside, tanh on the generator
# --------------------------------------------------------------------------- #
def _init_net(rng: np.random.Generator, sizes: list) -> list:
    return [{"W": rng.normal(0.0, np.sqrt(2.0 / a), (a, b)),
             "b": np.zeros(b)}
            for a, b in zip(sizes, sizes[1:], strict=False)]


def _forward(layers: list, x: np.ndarray, leak: float = 0.0,
             tanh_last: bool = False) -> tuple:
    """Run the stack; returns (output, cache) with one entry per layer."""
    cache = []
    h = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        pre = h @ layer["W"] + layer["b"]
        if i == last:
            out = np.tanh(pre) if tanh_last else pre
        else:
            out = np.where(pre > 0.0, pre, leak * pre)
        cache.append((h, pre, out))
        h = out
    return h, cache


def _backward(layers: list, cache: list, dout: np.ndarray, leak: float = 0.0,
              tanh_last: bool = False) -> tuple:
    """Return (per-layer grads, gradient wrt the input).

    `dout` must already carry the 1/n of a batch mean, so that the parameter
    gradients and the input gradient share one scale.
    """
    grads = [None] * len(layers)
    d = dout
    last = len(layers) - 1
    for i in range(last, -1, -1):
        h, pre, out = cache[i]
        if i == last:
            if tanh_last:
                d = d * (1.0 - out ** 2)
        else:
            d = d * np.where(pre > 0.0, 1.0, leak)
        grads[i] = {"W": h.T @ d, "b": d.sum(axis=0)}
        d = d @ layers[i]["W"].T
    return grads, d


def _col(x, n: int) -> np.ndarray:
    """A per-sample gradient as an (n, 1) column, whether scalar or array."""
    return np.broadcast_to(np.asarray(x, dtype=float), (n,))[:, None]


def _add(one: list, two: list) -> list:
    return [{k: one[i][k] + two[i][k] for k in one[i]}
            for i in range(len(one))]


def _sig(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Adam:
    """Adam with a per-network state, mirrored to the layer lists."""

    def __init__(self, params: list, lr: float = LR,
                 beta1: float = BETA1) -> None:
        self.lr = lr
        self.b1 = beta1
        self.t = 0
        self.m = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]
        self.v = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - 0.999 ** self.t
        for L, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            for key in L:
                m[key] = self.b1 * m[key] + (1.0 - self.b1) * g[key]
                v[key] = 0.999 * v[key] + 0.001 * g[key] ** 2
                L[key] -= self.lr * (m[key] / c1) / (np.sqrt(v[key] / c2) + 1e-8)


# --------------------------------------------------------------------------- #
# The losses, as per-sample gradients with respect to the discriminator logit
# --------------------------------------------------------------------------- #
def loss_grads(variant: str, real: np.ndarray, fake: np.ndarray) -> tuple:
    """Return (dD/d real, dD/d fake, dG/d fake), one entry per sample.

    The discriminator descends its two rows; the generator descends the third.
    Keeping all three here is what makes the variants comparable: they differ
    in this table, in the network shapes, and in what the networks are shown.
    """
    if variant == "vanilla":
        return _sig(real) - 1.0, _sig(fake), -_sig(fake)
    if variant in ("dcgan", "non_saturating", "conditional", "infogan"):
        return _sig(real) - 1.0, _sig(fake), _sig(fake) - 1.0
    if variant == "lsgan":
        return real - 1.0, fake, fake - 1.0
    if variant == "hinge":
        return (-(real < 1.0).astype(float),
                (fake > -1.0).astype(float), -1.0)
    raise ValueError(f"unknown variant {variant!r}")


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class GAN:
    """Train a generator against a discriminator.

    `params["D"]` and `params["G"]` are lists of dense layers; `params["Q"]` is
    the InfoGAN auxiliary head and exists only for that variant. `history`
    records the generator's mean loss-gradient scale per step, which is the
    quantity that vanishes under the saturating objective.
    """

    def __init__(self, variant: str = "non_saturating", *, z_dim: int = ZDIM,
                 hidden: int = HID, batch: int = BATCH, steps: int = STEPS,
                 lr: float = LR, mi_weight: float = MI_WEIGHT,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if z_dim < 1:
            raise ValueError("z_dim must be at least one")
        if hidden < 1:
            raise ValueError("hidden must be at least one")
        if steps < 1 or batch < 1:
            raise ValueError("steps and batch must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        if not np.isfinite(mi_weight) or mi_weight < 0.0:
            raise ValueError("mi_weight must be non-negative and finite")
        self.variant = variant
        self.z_dim = z_dim
        self.hidden = hidden
        self.batch = batch
        self.steps = steps
        self.lr = lr
        self.mi_weight = mi_weight
        self.seed = seed
        self.conditional = variant == "conditional"
        self.infogan = variant == "infogan"
        if variant == "dcgan":
            self.z_dim = max(z_dim, 16)
            self.hidden = max(hidden, 128)
        self.params: dict | None = None
        self.history: list = []

    # -- shapes --------------------------------------------------------------- #
    def _sizes(self) -> tuple:
        d_in = 2 + (CODE if self.conditional else 0)
        g_in = self.z_dim + (CODE if self.conditional or self.infogan else 0)
        depth = 3 if self.variant == "dcgan" else 2
        return ([d_in] + [self.hidden] * depth + [1],
                [g_in] + [self.hidden] * depth + [2])

    def _code(self, c: np.ndarray | None, n: int,
              rng: np.random.Generator | None) -> np.ndarray:
        if c is None:
            if rng is None:
                raise ValueError("this variant needs an explicit code")
            c = rng.integers(0, CODE, n)
        c = np.asarray(c, dtype=int)
        if c.ndim != 1 or len(c) != n:
            raise ValueError("c must be one-dimensional and match the batch")
        if np.any((c < 0) | (c >= CODE)):
            raise ValueError(f"c must lie in [0, {CODE})")
        return c

    def _onehot(self, c: np.ndarray | None) -> np.ndarray | None:
        return None if c is None else np.eye(CODE)[c]

    def _g_input(self, z: np.ndarray, c: np.ndarray | None) -> np.ndarray:
        onehot = self._onehot(c)
        return z if onehot is None else np.concatenate([z, onehot], axis=1)

    def _d_input(self, x: np.ndarray, c: np.ndarray | None) -> np.ndarray:
        # Only the conditional variant shows the code to the discriminator;
        # InfoGAN's whole point is that it does not.
        onehot = self._onehot(c) if self.conditional else None
        return x if onehot is None else np.concatenate([x, onehot], axis=1)

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> GAN:
        x = _check_data(x)
        n = len(x)
        if self.conditional:
            if y is None:
                raise ValueError("the conditional variant needs labels")
            y = _check_code(y, n)
        rng = np.random.default_rng(self.seed)
        d_sizes, g_sizes = self._sizes()
        params = {"D": _init_net(rng, d_sizes),
                  "G": _init_net(rng, g_sizes)}
        if self.infogan:
            params["Q"] = {"W": rng.normal(0.0, 0.1, (self.hidden, CODE)),
                           "b": np.zeros(CODE)}
        opt_d = _Adam(params["D"], self.lr)
        opt_g = _Adam(params["G"], self.lr)
        opt_q = _Adam([params["Q"]], self.lr) if self.infogan else None
        self.history = []
        for _ in range(self.steps):
            idx = rng.integers(0, n, self.batch)
            real = x[idx]
            c_real = y[idx] if self.conditional else None
            z = rng.normal(0.0, 1.0, (self.batch, self.z_dim))
            c_fake = (self._code(None, self.batch, rng)
                      if self.conditional or self.infogan else None)
            fake, g_cache = _forward(params["G"], self._g_input(z, c_fake),
                                     tanh_last=True)

            d_real, dr_cache = _forward(
                params["D"], self._d_input(real, c_real), leak=LEAK)
            d_fake, df_cache = _forward(
                params["D"], self._d_input(fake, c_fake), leak=LEAK)
            kdr, kdf, kgen = loss_grads(self.variant, d_real[:, 0],
                                        d_fake[:, 0])
            gr, _ = _backward(params["D"], dr_cache,
                              _col(kdr, self.batch) / self.batch, LEAK)
            gf, _ = _backward(params["D"], df_cache,
                              _col(kdf, self.batch) / self.batch, LEAK)
            grads_d = _add(gr, gf)
            self.history.append(float(np.mean(np.abs(kgen))))

            if self.infogan:
                # Q is fitted on the generated batch only: the mutual
                # information being estimated is I(c; G(z, c)).
                feat, feat_cache = _forward(params["D"][:-1], fake, leak=LEAK)
                logits = feat @ params["Q"]["W"] + params["Q"]["b"]
                d_logits = (_softmax(logits) - np.eye(CODE)[c_fake]) \
                    / self.batch
                opt_q.step([params["Q"]],
                           [{"W": feat.T @ d_logits,
                             "b": d_logits.sum(axis=0)}])
                pull, _ = _backward(params["D"][:-1], feat_cache,
                                    d_logits @ params["Q"]["W"].T, LEAK)
                grads_d = _add(grads_d[:-1], pull) + [grads_d[-1]]
            opt_d.step(params["D"], grads_d)

            # Generator step, against the discriminator it just updated.
            d_fake, df_cache = _forward(
                params["D"], self._d_input(fake, c_fake), leak=LEAK)
            kgen = loss_grads(self.variant, d_real[:, 0], d_fake[:, 0])[2]
            _, dx = _backward(params["D"], df_cache,
                              _col(kgen, self.batch) / self.batch, LEAK)
            if self.infogan:
                feat, feat_cache = _forward(params["D"][:-1], fake, leak=LEAK)
                logits = feat @ params["Q"]["W"] + params["Q"]["b"]
                d_logits = (_softmax(logits) - np.eye(CODE)[c_fake]) \
                    / self.batch
                _, dx_mi = _backward(params["D"][:-1], feat_cache,
                                     d_logits @ params["Q"]["W"].T, LEAK)
                # Maximise I(c; G(z, c)): descend the auxiliary cross-entropy,
                # so add its gradient instead of subtracting it.
                dx = dx + self.mi_weight * dx_mi
            if self.conditional or self.infogan:
                dx = dx[:, :self.z_dim]
            grads_g, _ = _backward(params["G"], g_cache, dx, tanh_last=True)
            opt_g.step(params["G"], grads_g)
        self.params = params
        return self

    # -- use ------------------------------------------------------------------ #
    def sample(self, n: int, c: np.ndarray | None = None,
               seed: int = 0) -> np.ndarray:
        """Draw n points; `c` is required by `conditional`, optional elsewhere."""
        if self.params is None:
            raise RuntimeError("call fit first")
        if n < 1:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, 1.0, (n, self.z_dim))
        if self.conditional:
            if c is None:
                raise ValueError("the conditional variant needs a code")
            c = self._code(c, n, None)
        elif self.infogan:
            c = self._code(c, n, rng)
        else:
            c = None
        return _forward(self.params["G"], self._g_input(z, c),
                        tanh_last=True)[0]

    def code_stats(self, n: int = 2048, seed: int = 0) -> tuple:
        """Return (per-sample consistency, codes resolved to a single mode).

        The code need not be the mode's number: InfoGAN has no reason to use the
        same labelling. What matters is that each code lands on one mode, so the
        test asks, per code, which mode its draws hit most and how often. The
        first number is that share averaged over the draws, the second counts
        codes whose majority mode holds at least 60% of them — 0 is a generator
        that ignores the code, `CODE` is one that obeys it.
        """
        if not self.infogan:
            raise RuntimeError("only infogan has a code to recover")
        rng = np.random.default_rng(seed)
        c = rng.integers(0, CODE, n)
        x = self.sample(n, c, seed=seed + 1)
        d = np.linalg.norm(x[:, None, :] - centres()[None], axis=2)
        near = d.argmin(axis=1)
        hits = 0
        resolved = 0
        for k in range(CODE):
            mine = near[c == k]
            if not len(mine):
                continue
            top = int(np.bincount(mine, minlength=MODES).max())
            hits += top
            resolved += int(top >= 0.6 * len(mine))
        return hits / n, resolved

    def accuracy(self, x: np.ndarray, y: np.ndarray | None = None,
                 seed: int = 0) -> float:
        """Discriminator accuracy on real points against generated ones."""
        if self.params is None:
            raise RuntimeError("call fit first")
        x = _check_data(x)
        if self.conditional:
            if y is None:
                raise ValueError("the conditional variant needs labels")
            y = _check_code(y, len(x))
            fake = self.sample(len(x), y, seed=seed)
        else:
            y = None
            fake = self.sample(len(x), None, seed=seed)
        d_real = _forward(self.params["D"], self._d_input(x, y),
                          leak=LEAK)[0][:, 0]
        d_fake = _forward(self.params["D"], self._d_input(fake, y),
                          leak=LEAK)[0][:, 0]
        return 0.5 * (float(np.mean(d_real > 0.0))
                      + float(np.mean(d_fake <= 0.0)))


def _check_data(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must be (n, 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    return x


def _check_code(c: np.ndarray, n: int) -> np.ndarray:
    c = np.asarray(c, dtype=int)
    if c.ndim != 1 or len(c) != n:
        raise ValueError("codes must be one-dimensional and match x")
    if np.any((c < 0) | (c >= CODE)):
        raise ValueError(f"codes must lie in [0, {CODE})")
    return c


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    x, y = make_data(0)
    print(f"generative adversarial network. {MODES} modes on a circle, "
          f"{len(x)} points:")
    print(f"  generator {ZDIM} -> {HID} -> {HID} -> 2 tanh, discriminator "
          f"2 -> {HID} -> {HID} -> 1 with leak {LEAK}")
    print(f"  {STEPS} Adam steps, batch {BATCH}, lr {LR}, beta1 {BETA1}")

    runs = {}
    print("generative adversarial network. what each loss finds:")
    print(f"  {'variant':<15s} {'modes':>7s} {'share':>6s} {'dist':>6s} "
          f"{'signal':>7s} {'d acc':>6s}")
    for name in VARIANTS:
        model = GAN(name, seed=0)
        labels = y if model.conditional else None
        model.fit(x, labels)
        runs[name] = model
        codes = np.arange(1024) % CODE if model.conditional else None
        many, share, dist = mode_stats(model.sample(1024, codes, seed=1))
        print(f"  {name:<15s} {many:>5d}/{MODES} {share:6.2f} {dist:6.3f} "
              f"{np.mean(model.history):7.3f} "
              f"{model.accuracy(x, labels, seed=2):6.2f}")

    print("generative adversarial network. the conditional codes, one "
          "per class:")
    cond = runs["conditional"]
    stacked = np.vstack([cond.sample(64, np.full(64, k), seed=k)
                         for k in range(CODE)])
    hit, gap = code_report(stacked)
    print(f"  own-centre match {hit:.3f} of {CODE * 64} points, "
          f"mean distance to that centre {gap:.3f}")

    print("generative adversarial network. the infogan code, and without it:")
    info = runs["infogan"]
    plain = GAN("infogan", mi_weight=0.0, seed=0).fit(x)
    for tag, model in (("mi 1.0", info), ("mi 0.0", plain)):
        many, share, _ = mode_stats(model.sample(1024, None, seed=1))
        consistent, resolved = model.code_stats()
        print(f"  {tag:<8s} consistent {consistent:.3f}  "
              f"resolved {resolved}/{CODE}  modes {many}/{MODES}  "
              f"largest share {share:.2f}")

    print("generative adversarial network. the two generator objectives:")
    grid = np.array([-6.0, -4.0, -2.0, 0.0, 2.0])
    saturating = np.abs(loss_grads("vanilla", np.zeros(5), grid)[2])
    nonsat = np.abs(loss_grads("non_saturating", np.zeros(5), grid)[2])
    print("  logit     " + " ".join(f"{v:6.1f}" for v in grid))
    print("  log(1-D)  " + " ".join(f"{v:6.3f}" for v in saturating))
    print("  -log D    " + " ".join(f"{v:6.3f}" for v in nonsat))
    print(f"  at logit -6 the saturating signal is "
          f"{saturating[0] / nonsat[0]:.3f} of the other")

    print("generative adversarial network. the optimal constant "
          "discriminator:")
    gap_to_entropy = 0.0
    for rate in (0.1, 0.5, 0.9):
        labels = np.zeros(100)
        labels[:int(rate * 100)] = 1.0
        grid_c = np.linspace(-8.0, 8.0, 8001)
        losses = np.array([np.mean(np.logaddexp(0.0, c) - labels * c)
                           for c in grid_c])
        best = grid_c[int(np.argmin(losses))]
        entropy = -(rate * np.log(rate) + (1.0 - rate) * np.log(1.0 - rate))
        gap_to_entropy = max(gap_to_entropy, abs(losses.min() - entropy))
        print(f"  base rate {rate:.1f}: argmin at logit {best: .3f}, "
              f"logit of the rate {np.log(rate / (1.0 - rate)): .3f}, "
              f"loss {losses.min():.3f} vs entropy {entropy:.3f}")
        assert abs(best - np.log(rate / (1.0 - rate))) < 0.01

    print("generative adversarial network. malformed calls are refused:")
    bad = [
        lambda: GAN("wgan_gp"),
        lambda: GAN("non_saturating", z_dim=0),
        lambda: GAN("non_saturating", steps=0),
        lambda: GAN("non_saturating", lr=0.0),
        lambda: GAN("infogan", mi_weight=-1.0),
        lambda: GAN("non_saturating").sample(8),
        lambda: GAN("conditional").fit(x),
        lambda: GAN("non_saturating").fit(np.zeros((8, 3))),
        lambda: cond.sample(4),
        lambda: runs["non_saturating"].code_stats(),
        lambda: cond.fit(x, y + CODE),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)} malformed calls refused")

    keeps = {name: mode_stats(runs[name].sample(1024, None, seed=3))[0]
             for name in ("vanilla", "lsgan", "non_saturating", "hinge")}
    print(f"  modes kept from a fresh draw: vanilla {keeps['vanilla']}, "
          f"least squares {keeps['lsgan']}, non-saturating "
          f"{keeps['non_saturating']}, hinge {keeps['hinge']}")
    assert keeps["vanilla"] >= 7 and keeps["lsgan"] >= 7
    assert keeps["non_saturating"] >= 7
    assert keeps["hinge"] < keeps["vanilla"]
    assert hit > 0.9
    assert info.code_stats()[0] > 0.95
    assert info.code_stats()[1] == CODE
    assert plain.code_stats()[0] < 0.8
    assert plain.code_stats()[1] <= 4
    assert saturating[0] < 0.05 * nonsat[0]
    assert np.allclose(np.abs(loss_grads("hinge", np.zeros(3),
                                         np.zeros(3))[2]), 1.0)
    assert gap_to_entropy < 2e-3
    batch = np.random.default_rng(5).normal(0.0, 0.02, (64, 2))
    for model in runs.values():
        codes = np.zeros(64, dtype=int)
        out = _forward(model.params["D"], model._d_input(batch, codes),
                       leak=LEAK)[0]
        assert out.shape == (64, 1) and np.all(np.isfinite(out))
    print("generative adversarial network demo passed")


if __name__ == "__main__":
    demo()
