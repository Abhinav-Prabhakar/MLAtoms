"""Guidance — classifier guidance, CFG, negative prompts and rescaling.

Chapter:   14-generative
Companion: guidance.md
Tier 2 · 4 variants: classifier, classifier_free, negative_prompt, rescale_cfg
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("classifier", "classifier_free", "negative_prompt", "rescale_cfg")

DIM = 2
MODES = 8
RADIUS = 0.8
SPREAD = 0.07
PER_MODE = 256
HIDDEN = 64
BATCH = 256
TRAIN_STEPS = 3000
CLF_STEPS = 3000
LR = 2e-3
CLF_LR = 1e-3
TIME_FEATURES = 4
DROPOUT = 0.1
STEPS = 200
COSINE_S = 0.008

# Modes start at pi / 8 so no centre sits on the x = 0 boundary. The binary
# attribute used by classifier guidance is then a property of the mode.
ANGLE_OFFSET = np.pi / MODES
RIGHT = (0, 1, 6, 7)

CONFIG = {
    "classifier": {"gamma": 0.3, "window": (0, 100)},
    "classifier_free": {"weight": 3.0, "window": (0, STEPS)},
    "negative_prompt": {"weight": 0.0, "negative_weight": 1.0,
                        "suppress": 7, "window": (100, STEPS)},
    "rescale_cfg": {"weight": 7.0, "rescale": True, "window": (0, STEPS)},
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def centres() -> np.ndarray:
    """The eight mode centres, (8, 2)."""
    angle = 2.0 * np.pi * np.arange(MODES) / MODES + ANGLE_OFFSET
    return RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def make_data(seed: int, per_mode: int = PER_MODE) -> tuple:
    """Return (points, labels) with labels drawn uniformly over the modes."""
    rng = np.random.default_rng(seed)
    flat = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, DIM))
                      for c in centres()])
    labels = np.repeat(np.arange(MODES), per_mode)
    order = rng.permutation(len(labels))
    return flat[order], labels[order]


def onehot(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((len(labels), MODES))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def attribute(points: np.ndarray) -> np.ndarray:
    """The binary attribute classifier guidance steers: x > 0."""
    return (np.asarray(points, dtype=float)[:, 0] > 0.0).astype(int)


def mode_stats(samples: np.ndarray) -> tuple:
    """Return (modes covered, largest share, mean distance, mean radius)."""
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    near = d.argmin(axis=1)
    share = np.bincount(near, minlength=MODES) / len(s)
    return (int(np.sum(share > 0.03)), float(share.max()),
            float(d.min(axis=1).mean()), float(np.linalg.norm(s, axis=1).mean()))


def share_of(samples: np.ndarray, mode: int) -> float:
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    return float(np.mean(d.argmin(axis=1) == mode))


# --------------------------------------------------------------------------- #
# Schedule and network
# --------------------------------------------------------------------------- #
def make_schedule(steps: int = STEPS) -> tuple:
    """Cosine schedule; `abar` has a leading one so abar[t] is the t-th rung."""
    t = np.arange(steps + 1) / steps
    f = np.cos((t + COSINE_S) / (1.0 + COSINE_S) * np.pi / 2.0) ** 2
    abar = f / f[0]
    beta = np.clip(1.0 - abar[1:] / abar[:-1], 1e-5, 0.999)
    alpha = 1.0 - beta
    return beta, alpha, np.concatenate([[1.0], np.cumprod(alpha)])


def _sig(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _time_features(t: np.ndarray, steps: int) -> np.ndarray:
    """Four features: sin and cos at one and two cycles over the schedule."""
    phase = (np.asarray(t, dtype=float) / steps) * np.pi
    return np.stack([np.sin(phase), np.cos(phase),
                     np.sin(2.0 * phase), np.cos(2.0 * phase)], axis=1)


def _init_net(rng: np.random.Generator, sizes: list) -> list:
    layers = []
    for a, b in zip(sizes, sizes[1:], strict=False):
        layers.append({"W": rng.normal(0.0, np.sqrt(2.0 / a), (a, b)),
                       "b": np.zeros(b)})
    return layers


def _forward(params: list, h: np.ndarray) -> tuple:
    cache, last = [], len(params) - 1
    for i, layer in enumerate(params):
        pre = h @ layer["W"] + layer["b"]
        out = pre if i == last else pre * _sig(pre)
        cache.append((h, pre, out))
        h = out
    return h, cache


def _backward(params: list, cache: list, dout: np.ndarray) -> tuple:
    grads = [None] * len(params)
    d, last = dout, len(params) - 1
    for i in range(last, -1, -1):
        h, pre, out = cache[i]
        if i != last:
            s = _sig(pre)
            d = d * (s + pre * s * (1.0 - s))
        grads[i] = {"W": h.T @ d, "b": d.sum(axis=0)}
        d = d @ params[i]["W"].T
    return grads, d


class _Adam:
    def __init__(self, params: list, lr: float) -> None:
        self.lr, self.t = lr, 0
        self.m = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]
        self.v = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        c1 = 1.0 - 0.9 ** self.t
        c2 = 1.0 - 0.999 ** self.t
        for L, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            for key in L:
                m[key] = 0.9 * m[key] + 0.1 * g[key]
                v[key] = 0.999 * v[key] + 0.001 * g[key] ** 2
                L[key] -= self.lr * (m[key] / c1) / (
                    np.sqrt(v[key] / c2) + 1e-8)


# --------------------------------------------------------------------------- #
# The guided sampler
# --------------------------------------------------------------------------- #
class GuidedDiffusion:
    """A class-conditional v-prediction denoiser, read through four guidances.

    `fit` also trains the noisy-data attribute classifier when the variant is
    `classifier`. `sample` runs the deterministic DDIM update, optionally
    mixing an unconditional and a conditional prediction (`weight`), pushing
    away from a second label (`negative_weight`), adding the classifier's
    gradient (`gamma`), and rescaling the clean prediction (`rescale`).
    `window` is the rung range in which guidance acts, inclusive, with
    rung 200 the noisiest.
    """

    def __init__(self, variant: str = "classifier_free", *,
                 hidden: int = HIDDEN, train_steps: int = TRAIN_STEPS,
                 clf_steps: int = CLF_STEPS, batch: int = BATCH,
                 lr: float = LR, clf_lr: float = CLF_LR, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if hidden < 1:
            raise ValueError("hidden must be at least one")
        if train_steps < 1 or clf_steps < 1 or batch < 1:
            raise ValueError("train_steps, clf_steps and batch must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        if not np.isfinite(clf_lr) or clf_lr <= 0.0:
            raise ValueError("clf_lr must be positive and finite")
        cfg = CONFIG[variant]
        self.variant = variant
        self.hidden = hidden
        self.train_steps = train_steps
        self.clf_steps = clf_steps
        self.batch = batch
        self.lr = lr
        self.clf_lr = clf_lr
        self.seed = seed
        self.beta, self.alpha, self.abar = make_schedule(STEPS)
        self.weight = cfg.get("weight", 0.0)
        self.gamma = cfg.get("gamma", 0.0)
        self.negative_weight = cfg.get("negative_weight", 0.0)
        self.suppress = cfg.get("suppress", 0)
        self.rescale = cfg.get("rescale", False)
        self.window = cfg["window"]
        self.params: list | None = None
        self.clf: list | None = None

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray, y: np.ndarray) -> GuidedDiffusion:
        x, y = _check_data(x, y)
        rng = np.random.default_rng(self.seed)
        params = _init_net(rng, [DIM + TIME_FEATURES + MODES, self.hidden,
                                 self.hidden, DIM])
        opt = _Adam(params, self.lr)
        labels = onehot(y)
        for _ in range(self.train_steps):
            idx = rng.integers(0, len(x), self.batch)
            x0, lab = x[idx], labels[idx].copy()
            lab[rng.random(len(idx)) < DROPOUT] = 0.0
            t = rng.integers(1, STEPS + 1, self.batch)
            eps = rng.normal(0.0, 1.0, x0.shape)
            abar = self.abar[t][:, None]
            xt = np.sqrt(abar) * x0 + np.sqrt(1.0 - abar) * eps
            target = np.sqrt(abar) * eps - np.sqrt(1.0 - abar) * x0
            inp = np.concatenate([xt, _time_features(t, STEPS), lab], axis=1)
            out, cache = _forward(params, inp)
            dout = 2.0 * (out - target) / (len(idx) * DIM)
            opt.step(params, _backward(params, cache, dout)[0])
        self.params = params
        if self.variant == "classifier":
            self.clf = self._fit_classifier(x)
        return self

    def _fit_classifier(self, x: np.ndarray) -> list:
        """Binary attribute classifier trained on noised inputs at every t."""
        rng = np.random.default_rng(self.seed + 1)
        clf = _init_net(rng, [DIM + TIME_FEATURES, self.hidden, self.hidden, 2])
        opt = _Adam(clf, self.clf_lr)
        target = attribute(x)
        for _ in range(self.clf_steps):
            idx = rng.integers(0, len(x), self.batch)
            x0 = x[idx]
            t = rng.integers(1, STEPS + 1, self.batch)
            eps = rng.normal(0.0, 1.0, x0.shape)
            abar = self.abar[t][:, None]
            xt = np.sqrt(abar) * x0 + np.sqrt(1.0 - abar) * eps
            logits, cache = _forward(
                clf, np.concatenate([xt, _time_features(t, STEPS)], axis=1))
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            p[np.arange(len(idx)), target[idx]] -= 1.0
            opt.step(clf, _backward(clf, cache, p / len(idx))[0])
        return clf

    def n_params(self) -> int:
        if self.params is None:
            raise RuntimeError("call fit first")
        total = sum(L["W"].size + L["b"].size for L in self.params)
        if self.clf is not None:
            total += sum(L["W"].size + L["b"].size for L in self.clf)
        return total

    # -- predictions ---------------------------------------------------------- #
    def _predict(self, x: np.ndarray, t: int, lab: np.ndarray) -> np.ndarray:
        inp = np.concatenate([x, _time_features(np.full(len(x), t), STEPS),
                              lab], axis=1)
        return _forward(self.params, inp)[0]

    def _eps(self, x: np.ndarray, t: int, lab: np.ndarray) -> np.ndarray:
        """The velocity prediction turned into a noise prediction."""
        v = self._predict(x, t, lab)
        abar = self.abar[t]
        return np.sqrt(abar) * v + np.sqrt(1.0 - abar) * x

    def _classifier_grad(self, x: np.ndarray, t: int) -> np.ndarray:
        """dCE/dx for the positive attribute, shape (n, 2)."""
        inp = np.concatenate([x, _time_features(np.full(len(x), t), STEPS)],
                             axis=1)
        logits, cache = _forward(self.clf, inp)
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        p[:, 1] -= 1.0
        return _backward(self.clf, cache, p)[1][:, :DIM]

    def classifier_accuracy(self, x: np.ndarray, t: int,
                            seed: int = 0) -> float:
        """Accuracy of the trained attribute classifier on inputs noised to t."""
        if self.clf is None:
            raise RuntimeError("this variant has no classifier")
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(seed)
        abar = self.abar[t]
        xt = (np.sqrt(abar) * x
              + np.sqrt(1.0 - abar) * rng.normal(0.0, 1.0, x.shape))
        logits = _forward(
            self.clf,
            np.concatenate([xt, _time_features(np.full(len(x), t), STEPS)],
                           axis=1))[0]
        return float(np.mean(logits.argmax(axis=1) == attribute(x)))

    def bayes_accuracy(self, x: np.ndarray, t: int, seed: int = 0) -> float:
        """The best any attribute classifier can do at rung t.

        Under the Gaussian mixture the class posterior at `x_t` is exact, so
        the ceiling is a property of the data and the schedule, not of a
        trained network. The noise draw matches `classifier_accuracy`.
        """
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(seed)
        abar = self.abar[t]
        var = abar * SPREAD ** 2 + (1.0 - abar)
        mu = np.sqrt(abar) * centres()
        xt = (np.sqrt(abar) * x
              + np.sqrt(1.0 - abar) * rng.normal(0.0, 1.0, x.shape))
        logp = -0.5 * ((xt[:, None, :] - mu[None]) ** 2).sum(axis=2) / var
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        right = p[:, RIGHT].sum(axis=1) / p.sum(axis=1)
        return float(np.mean((right > 0.5) == (attribute(x) == 1)))

    # -- the trajectory ------------------------------------------------------- #
    def _times(self, steps: int) -> np.ndarray:
        """The descending rungs visited, from STEPS to zero inclusive."""
        if steps < 1 or steps > STEPS:
            raise ValueError(f"steps must lie in [1, {STEPS}]")
        grid = np.rint(np.linspace(0.0, STEPS, steps + 1)).astype(int)
        return grid[::-1]

    def sample(self, n: int = 1024, *, labels: np.ndarray | None = None,
               steps: int | None = None, seed: int = 0,
               weight: float | None = None, gamma: float | None = None,
               negative: np.ndarray | None = None,
               negative_weight: float | None = None,
               window: tuple | None = None,
               rescale: bool | None = None) -> np.ndarray:
        """Run the guided reverse trajectory from noise."""
        if self.params is None:
            raise RuntimeError("call fit first")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        steps = steps if steps is not None else STEPS
        weight = self.weight if weight is None else weight
        gamma = self.gamma if gamma is None else gamma
        neg_w = self.negative_weight if negative_weight is None else negative_weight
        rescale = self.rescale if rescale is None else rescale
        lo, hi = self.window if window is None else window
        _check_window(lo, hi)
        _check_scale("weight", weight)
        _check_scale("gamma", gamma)
        _check_scale("negative_weight", neg_w)
        if gamma > 0.0 and self.clf is None:
            raise ValueError("gamma needs the classifier variant")
        if (negative is None and neg_w > 0.0
                and self.variant == "negative_prompt"):
            negative = np.full(n, self.suppress)
        if negative is None and neg_w > 0.0:
            raise ValueError("negative_weight needs a negative label")
        labels = _check_labels(labels, n, "labels")
        negative = _check_labels(negative, n, "negative")
        if self.variant == "classifier" and labels is not None:
            raise ValueError("the classifier variant steers an attribute")

        times = self._times(steps)
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, DIM))
        zero = np.zeros((n, MODES))
        lab = zero if labels is None else onehot(labels)
        neg = zero if negative is None else onehot(negative)

        for t, s in zip(times[:-1], times[1:], strict=True):
            abar = self.abar[t]
            base = self._eps(x, t, zero)
            e = base
            if lo <= t <= hi and labels is not None and weight > 0.0:
                cond = self._eps(x, t, lab)
                e = base + weight * (cond - base)
                if rescale:
                    x0c = (x - np.sqrt(1.0 - abar) * cond) / np.sqrt(abar)
                    x0g = (x - np.sqrt(1.0 - abar) * e) / np.sqrt(abar)
                    scale = (np.linalg.norm(x0c, axis=1, keepdims=True)
                             / np.maximum(np.linalg.norm(
                                 x0g, axis=1, keepdims=True), 1e-12))
                    x0g = x0g * scale
                    e = (x - np.sqrt(abar) * x0g) / np.sqrt(1.0 - abar)
            if lo <= t <= hi and negative is not None and neg_w > 0.0:
                away = self._eps(x, t, neg)
                e = e - neg_w * (away - base)
            if lo <= t <= hi and gamma > 0.0:
                e = e + np.sqrt(1.0 - abar) * gamma * self._classifier_grad(x, t)
            x0 = (x - np.sqrt(1.0 - abar) * e) / np.sqrt(abar)
            x = np.sqrt(self.abar[s]) * x0 + np.sqrt(1.0 - self.abar[s]) * e
        return x


def _check_data(x: np.ndarray, y: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(f"x must be (n, {DIM})")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    if y.shape != (len(x),):
        raise ValueError("y must have one label per point")
    if len(y) and (y.min() < 0 or y.max() >= MODES):
        raise ValueError(f"labels must lie in [0, {MODES})")
    return x, y.astype(int)


def _check_labels(labels: np.ndarray | None, n: int, name: str):
    if labels is None:
        return None
    out = np.asarray(labels)
    if out.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.issubdtype(out.dtype, np.integer):
        raise ValueError(f"{name} must be integers")
    if out.min() < 0 or out.max() >= MODES:
        raise ValueError(f"{name} must lie in [0, {MODES})")
    return out


def _check_scale(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _check_window(lo: int, hi: int) -> None:
    if not (0 <= lo <= hi <= STEPS):
        raise ValueError(f"window must satisfy 0 <= lo <= hi <= {STEPS}")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _stats(samples: np.ndarray) -> dict:
    cover, share, dist, radius = mode_stats(samples)
    return {"cover": cover, "maxshare": share, "dist": dist, "radius": radius}


def demo() -> None:
    x, y = make_data(0)
    test, _ = make_data(1)
    print(f"guidance. eight Gaussians of radius {RADIUS} at angles "
          f"{np.degrees(ANGLE_OFFSET):.1f} + k*45 degrees, {len(x)} points:")
    print(f"  v-prediction, cosine schedule, {STEPS} rungs, "
          f"{HIDDEN}-wide SiLU net, {TRAIN_STEPS} steps, "
          f"{DROPOUT:.0%} label dropout")

    runs = {name: GuidedDiffusion(name).fit(x, y) for name in VARIANTS}
    size = runs["classifier_free"].n_params()
    extra = runs["classifier"].n_params() - size
    print(f"  {size} parameters in a denoiser; the classifier variant adds "
          f"{extra} more")

    print("guidance. what the classifier can still see, against the exact "
          "ceiling:")
    print(f"  {'t':>4s} {'trained':>8s} {'Bayes':>6s}")
    clf = runs["classifier"]
    ceiling = {}
    for t in (5, 20, 50, 100, 150, 200):
        ceiling[t] = (clf.classifier_accuracy(test, t),
                      clf.bayes_accuracy(test, t))
        print(f"  {t:>4d} {ceiling[t][0]:>8.3f} {ceiling[t][1]:>6.3f}")

    print("guidance. the four variants, 1024 samples each:")
    print(f"  {'variant':<16s} {'knob':<12s} {'metric':<15s} "
          f"{'value':>6s} {'cover':>5s} {'dist':>6s} {'radius':>6s}")
    for name in VARIANTS:
        m, cfg = runs[name], CONFIG[name]
        if name == "classifier":
            s = m.sample(1024, seed=1)
            label, value = "right half", float(np.mean(attribute(s) == 1))
            knob = f"gamma {cfg['gamma']:.1f}"
        elif name == "negative_prompt":
            s = m.sample(1024, seed=1)
            label, value = "mode 7", share_of(s, 7)
            knob = f"nw {cfg['negative_weight']:.1f}"
        else:
            s = m.sample(1024, labels=np.full(1024, 3), seed=1)
            label, value = "mode 3", share_of(s, 3)
            knob = f"w {cfg['weight']:.1f}"
        st = _stats(s)
        print(f"  {name:<16s} {knob:<12s} {label:<15s} {value:6.2f} "
              f"{st['cover']:>3d}/8 {st['dist']:6.3f} {st['radius']:6.3f}")

    print("guidance. CFG mixes the two models; above w = 1 it extrapolates:")
    print(f"  {'w':>4s} {'mode 3':>6s} {'cover':>5s} {'dist':>6s} "
          f"{'radius':>6s}")
    ladder = {}
    cfg_model = runs["classifier_free"]
    for w in (0.0, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0, 7.0, 14.0):
        s = cfg_model.sample(1024, labels=np.full(1024, 3), weight=w, seed=1)
        ladder[w] = {"share": share_of(s, 3), **_stats(s)}
        print(f"  {w:>4.1f} {ladder[w]['share']:6.3f} "
              f"{ladder[w]['cover']:>3d}/8 {ladder[w]['dist']:6.3f} "
              f"{ladder[w]['radius']:6.3f}")

    print("guidance. classifier guidance, late window against none:")
    print(f"  {'gamma':>5s} {'window':>7s} {'right half':>10s} "
          f"{'dist':>6s} {'radius':>6s}")
    guided = {}
    for window in ((0, 100), (0, STEPS)):
        for gamma in (0.0, 0.1, 0.3, 1.0):
            s = clf.sample(1024, gamma=gamma, window=window, seed=1)
            guided[window, gamma] = {"right": float(np.mean(attribute(s) == 1)),
                                     **_stats(s)}
            end = "late" if window[1] == 100 else "all"
            print(f"  {gamma:>5.1f} {end:>7s} "
                  f"{guided[window, gamma]['right']:>10.3f} "
                  f"{guided[window, gamma]['dist']:6.3f} "
                  f"{guided[window, gamma]['radius']:6.3f}")

    print("guidance. the negative prompt needs the middle of the trajectory:")
    print(f"  {'from t':>6s} {'mode 7':>6s} {'cover':>5s} {'dist':>6s} "
          f"{'radius':>6s}")
    windows = {}
    for lo in (0, 50, 100, 150):
        s = runs["negative_prompt"].sample(
            1024, negative=np.full(1024, 7), window=(lo, STEPS), seed=1)
        windows[lo] = {"share": share_of(s, 7), **_stats(s)}
        print(f"  {lo:>6d} {windows[lo]['share']:6.3f} "
              f"{windows[lo]['cover']:>3d}/8 {windows[lo]['dist']:6.3f} "
              f"{windows[lo]['radius']:6.3f}")
    print("guidance. and its weight, inside that window:")
    print(f"  {'nw':>4s} {'mode 7':>6s} {'cover':>5s} {'dist':>6s} "
          f"{'radius':>6s}")
    strengths = {}
    for nw in (0.0, 0.5, 1.0, 1.5):
        s = runs["negative_prompt"].sample(
            1024, negative=np.full(1024, 7), negative_weight=nw, seed=1)
        strengths[nw] = {"share": share_of(s, 7), **_stats(s)}
        print(f"  {nw:>4.1f} {strengths[nw]['share']:6.3f} "
              f"{strengths[nw]['cover']:>3d}/8 {strengths[nw]['dist']:6.3f} "
              f"{strengths[nw]['radius']:6.3f}")

    print("guidance. rescaling the clean prediction at large w:")
    print(f"  {'w':>4s} {'plain':>6s} {'resc.':>6s} {'radius':>7s} "
          f"{'resc.':>6s}")
    rescaling = {}
    for w in (3.0, 7.0, 14.0):
        plain = cfg_model.sample(1024, labels=np.full(1024, 3), weight=w,
                                 seed=1)
        scaled = cfg_model.sample(1024, labels=np.full(1024, 3), weight=w,
                                  rescale=True, seed=1)
        rescaling[w] = (_stats(plain), _stats(scaled))
        print(f"  {w:>4.1f} {rescaling[w][0]['dist']:6.3f} "
              f"{rescaling[w][1]['dist']:6.3f} "
              f"{rescaling[w][0]['radius']:7.3f} "
              f"{rescaling[w][1]['radius']:6.3f}")

    print("guidance. malformed calls are refused:")
    bad = [
        lambda: GuidedDiffusion("cfg"),
        lambda: GuidedDiffusion("classifier", hidden=0),
        lambda: GuidedDiffusion("classifier", train_steps=0),
        lambda: GuidedDiffusion("classifier", lr=0.0),
        lambda: GuidedDiffusion("classifier_free").sample(8),
        lambda: cfg_model.sample(0),
        lambda: cfg_model.sample(8, steps=0),
        lambda: cfg_model.sample(8, steps=STEPS + 1),
        lambda: cfg_model.sample(8, labels=np.full(8, 3), weight=-1.0),
        lambda: cfg_model.sample(8, labels=np.full(8, 3), gamma=1.0),
        lambda: cfg_model.sample(8, gamma=1.0),
        lambda: cfg_model.sample(8, labels=np.full(8, 3), negative_weight=1.0),
        lambda: clf.sample(8, labels=np.zeros(8, dtype=int)),
        lambda: cfg_model.sample(8, labels=np.full(7, 3)),
        lambda: cfg_model.sample(8, labels=np.full(8, MODES)),
        lambda: cfg_model.sample(8, window=(50, 10)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)} malformed calls refused")

    print("guidance. what the tables say:")
    assert ceiling[200][1] < 0.52
    assert ceiling[5][1] > 0.97
    assert abs(ceiling[50][0] - ceiling[50][1]) < 0.04
    assert abs(ceiling[100][0] - ceiling[100][1]) < 0.04
    assert ladder[0.0]["share"] < 0.2
    assert 0.5 < ladder[0.5]["share"] < 0.95
    assert ladder[1.0]["share"] > 0.99
    assert ladder[2.0]["dist"] < ladder[1.0]["dist"]
    assert ladder[14.0]["radius"] > ladder[1.0]["radius"] + 0.03
    assert abs(guided[(0, 100), 0.0]["right"] - 0.48) < 0.04
    assert guided[(0, 100), 0.3]["right"] > guided[(0, 100), 0.0]["right"] + 0.1
    assert guided[(0, STEPS), 1.0]["radius"] > guided[(0, 100), 1.0]["radius"]
    assert guided[(0, STEPS), 1.0]["dist"] > guided[(0, 100), 1.0]["dist"]
    assert abs(strengths[0.0]["share"] - 0.125) < 0.02
    assert windows[0]["cover"] < MODES
    assert windows[0]["radius"] > 1.2
    assert windows[100]["share"] < strengths[0.0]["share"] * 0.5
    assert windows[100]["cover"] == MODES
    assert windows[100]["radius"] < 0.9
    assert windows[150]["share"] > windows[100]["share"]
    assert strengths[1.0]["share"] < strengths[0.5]["share"]
    assert strengths[1.0]["cover"] == MODES
    assert strengths[1.5]["cover"] < MODES
    assert rescaling[14.0][1]["radius"] < rescaling[14.0][0]["radius"] - 0.1
    assert rescaling[14.0][1]["dist"] < rescaling[14.0][0]["dist"]
    first = cfg_model.sample(128, labels=np.full(128, 3), seed=3)
    second = cfg_model.sample(128, labels=np.full(128, 3), seed=3)
    assert np.array_equal(first, second)
    other = cfg_model.sample(128, labels=np.full(128, 3), seed=4)
    assert not np.allclose(first, other)
    print("  the classifier tracks the exact ceiling and runs out of")
    print("  information as noise grows; CFG is exact at w = 1 and sharpens")
    print("  beyond it; a negative prompt works only mid-trajectory; and")
    print("  rescaling keeps large-w samples on the data's shell.")
    print("guidance demo passed")


if __name__ == "__main__":
    demo()
