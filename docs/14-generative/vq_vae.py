"""VQ-VAE — learning a discrete codebook instead of a continuous latent.

Chapter:   14-generative
Companion: vq_vae.md
Tier 2 · 6 variants: vq_vae, vq_vae2, ema_codebook, commitment_loss,
                    straight_through, residual_vq
"""

from __future__ import annotations

import numpy as np

VARIANTS = (
    "vq_vae",
    "vq_vae2",
    "ema_codebook",
    "commitment_loss",
    "straight_through",
    "residual_vq",
)

DIM = 2
MODES = 8
RADIUS = 0.8
SPREAD = 0.07
PER_MODE = 256
LATENT = 8
CODES = 16
HIDDEN = 64
BATCH = 256
STEPS = 3000
LR = 2e-3
BETA = 0.25
TAU = 1.0
DECAY = 0.99
IDLE = 25
ANGLE = np.pi / MODES

CONFIG = {
    "vq_vae": {"levels": 1},
    "vq_vae2": {"levels": 1, "restart": True},
    "ema_codebook": {"levels": 1, "ema": True},
    "commitment_loss": {"levels": 1, "beta": 4.0},
    "straight_through": {"levels": 1, "grad": "soft"},
    "residual_vq": {"levels": 2},
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def centres() -> np.ndarray:
    """The eight mode centres, shape (8, 2)."""
    angle = 2.0 * np.pi * np.arange(MODES) / MODES + ANGLE
    return RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def make_data(seed: int, per_mode: int = PER_MODE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, DIM))
                      for c in centres()])
    return flat[rng.permutation(len(flat))]


def mode_stats(samples: np.ndarray) -> tuple:
    """(modes covered, largest share, mean distance to a centre, mean radius)."""
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    share = np.bincount(d.argmin(axis=1), minlength=MODES) / len(s)
    return (int(np.sum(share > 0.03)), float(share.max()),
            float(d.min(axis=1).mean()), float(np.linalg.norm(s, axis=1).mean()))


def kmeans(x: np.ndarray, k: int, *, iters: int = 100) -> tuple:
    """Lloyd's algorithm; returns (distortion, centres)."""
    c = x[np.linspace(0, len(x) - 1, k).astype(int)].copy()
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(axis=2), axis=1)
        new = np.stack([x[lab == j].mean(axis=0) if np.any(lab == j) else c[j]
                        for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(axis=2), axis=1)
    return float(((x - c[lab]) ** 2).sum(axis=1).mean()), c


# --------------------------------------------------------------------------- #
# MLP and optimiser
# --------------------------------------------------------------------------- #
def _silu(x: np.ndarray) -> np.ndarray:
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))


def _dsilu(x: np.ndarray) -> np.ndarray:
    s = 0.5 * (1.0 + np.tanh(0.5 * x))
    return s + x * s * (1.0 - s)


def _mlp_init(rng: np.random.Generator, sizes: list) -> list:
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
    h, h1, a1, h2, a2 = cache
    gW3, gb3 = a2.T @ d, d.sum(axis=0)
    d = (d @ P[4].T) * _dsilu(h2)
    gW2, gb2 = a1.T @ d, d.sum(axis=0)
    d = (d @ P[2].T) * _dsilu(h1)
    gW1, gb1 = h.T @ d, d.sum(axis=0)
    return d @ P[0].T, [gW1, gb1, gW2, gb2, gW3, gb3]


class _Adam:
    def __init__(self, params: list, lr: float) -> None:
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        c1 = 1.0 - 0.9 ** self.t
        c2 = 1.0 - 0.999 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            m[...] = 0.9 * m + 0.1 * g
            v[...] = 0.999 * v + 0.001 * g ** 2
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)


def _check_data(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(f"x must have shape (n, {DIM})")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    return x


def _soft_weights(ze: np.ndarray, E: np.ndarray, tau: float) -> tuple:
    """Softmax over negative squared distances, plus those distances.

    `tau` is relative to the codebook's own spread, so the relaxation means the
    same thing wherever the encoder happens to place the latent.
    """
    scale = float(((E - E.mean(axis=0)) ** 2).sum(axis=1).mean()) + 1e-9
    d2 = ((ze[:, None, :] - E[None]) ** 2).sum(axis=2)
    z = d2 - d2.min(axis=1, keepdims=True)
    w = np.exp(-z / (tau * scale))
    return w / w.sum(axis=1, keepdims=True), d2


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class VQVAE:
    """An encoder, a codebook and a decoder.

    The encoder maps a point to a continuous vector, the nearest codebook entry
    replaces it, and the decoder maps the entry back. Nearest-neighbour lookup
    has zero derivative almost everywhere, so the encoder is trained through a
    surrogate: the straight-through estimator, or the exact gradient of a
    temperature-relaxed quantiser evaluated at the hard code.
    """

    def __init__(self, variant: str = "vq_vae", *, latent: int = LATENT,
                 codes: int = CODES, hidden: int = HIDDEN,
                 beta: float | None = None, tau: float = TAU,
                 decay: float = DECAY, levels: int | None = None,
                 train_steps: int = STEPS, batch: int = BATCH,
                 lr: float = LR, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("latent", latent), ("codes", codes),
                            ("hidden", hidden), ("batch", batch),
                            ("train_steps", train_steps)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be positive and finite")
        if beta is not None and (not np.isfinite(beta) or beta < 0.0):
            raise ValueError("beta must be finite and non-negative")
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        cfg = CONFIG[variant]
        self.variant = variant
        self.levels = cfg.get("levels", 1) if levels is None else int(levels)
        if self.levels < 1:
            raise ValueError("levels must be a positive integer")
        self.restart = cfg.get("restart", False)
        self.ema = cfg.get("ema", False)
        self.grad = cfg.get("grad", "st")
        if self.grad == "soft" and self.levels > 1:
            raise ValueError("the relaxed gradient supports one level only")
        self.beta = cfg.get("beta", BETA) if beta is None else float(beta)
        self.latent = int(latent)
        self.codes = int(codes)
        self.hidden = int(hidden)
        self.tau = float(tau)
        self.decay = float(decay)
        self.train_steps = int(train_steps)
        self.batch = int(batch)
        self.lr = float(lr)
        self.seed = seed
        self.enc: list = []
        self.dec: list = []
        self.E: list = []
        self.reseeds = 0
        self.hist: np.ndarray | None = None
        self.last_loss = float("nan")
        self.fitted = False

    # -- quantiser ------------------------------------------------------------ #
    def _quantise(self, ze: np.ndarray) -> tuple:
        """Replace each row by its nearest code; returns (quantised, picks)."""
        resid = ze
        zq = np.zeros_like(ze)
        picks = []
        for level in range(self.levels):
            d2 = ((resid[:, None, :] - self.E[level][None]) ** 2).sum(axis=2)
            k = np.argmin(d2, axis=1)
            zq = zq + self.E[level][k]
            resid = resid - self.E[level][k]
            picks.append(k)
        return zq, picks

    def soft_gradient(self, ze: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Exact gradient of the relaxed objective, at the hard assignment.

        The surrogate is L(sum_k w_k(z) e_k) with w a softmax over negative
        squared distances. Its gradient at z is (2/tau) times the weighted
        spread of the codes around their weighted mean.
        """
        if self.levels != 1:
            raise ValueError("the relaxed gradient supports one level only")
        w, _ = _soft_weights(ze, self.E[0], self.tau)
        scale = float(((self.E[0] - self.E[0].mean(axis=0)) ** 2)
                      .sum(axis=1).mean()) + 1e-9
        bar = w @ self.E[0]
        proj = w * (self.E[0][None] * dq[:, None, :]).sum(axis=2)
        return (2.0 / (self.tau * scale)) * (
            proj @ self.E[0] - proj.sum(axis=1, keepdims=True) * bar)

    # -- fit ------------------------------------------------------------------ #
    def fit(self, x: np.ndarray) -> VQVAE:
        x = _check_data(x)
        rng = np.random.default_rng(self.seed)
        self.enc = _mlp_init(rng, [DIM, self.hidden, self.hidden, self.latent])
        self.dec = _mlp_init(rng, [self.latent, self.hidden, self.hidden, DIM])
        self.E = [rng.normal(0.0, 0.3, (self.codes, self.latent))
                  for _ in range(self.levels)]
        opt_e = _Adam(self.enc, self.lr)
        opt_d = _Adam(self.dec, self.lr)
        opt_c = [_Adam([e], self.lr) for e in self.E]
        counts = [np.zeros(self.codes) for _ in range(self.levels)]
        idle = [np.zeros(self.codes) for _ in range(self.levels)]

        # data-dependent init: the codebook starts where the encoder puts data
        seed_rows = x[rng.integers(0, len(x), self.codes * 4)]
        z0, _ = _mlp_forward(self.enc, seed_rows)
        for level in range(self.levels):
            self.E[level][...] = z0[rng.integers(0, len(z0), self.codes)]

        for step in range(self.train_steps):
            xb = x[rng.integers(0, len(x), self.batch)]
            n = len(xb)
            ze, cache_e = _mlp_forward(self.enc, xb)
            zq, picks = self._quantise(ze)
            xh, cache_d = _mlp_forward(self.dec, zq)

            recon = ((xb - xh) ** 2).sum(axis=1).mean() / (2.0 * SPREAD ** 2)
            dq, gd = _mlp_backward(self.dec, cache_d,
                                   (xh - xb) / (n * SPREAD ** 2))
            g_ze = (self.soft_gradient(ze, dq) if self.grad == "soft"
                    else dq.copy())
            g_E = [np.zeros_like(e) for e in self.E]
            resid, gap = ze, 0.0
            for level in range(self.levels):
                k = picks[level]
                eq = self.E[level][k]
                gap = gap + float(((resid - eq) ** 2).sum(axis=1).mean())
                if not self.ema:
                    np.add.at(g_E[level], k, 2.0 * (eq - resid) / n)
                g_ze = g_ze + 2.0 * self.beta * (resid - eq) / n
                resid = resid - eq
            if step % 100 == 0:
                self.last_loss = float(recon) + (1.0 + self.beta) * gap
            _, ge = _mlp_backward(self.enc, cache_e, g_ze)
            opt_e.step(self.enc, ge)
            opt_d.step(self.dec, gd)
            if self.ema:
                self._ema_step(ze, picks, counts)
            else:
                for level in range(self.levels):
                    opt_c[level].step([self.E[level]], [g_E[level]])
            if self.restart:
                self._restart(ze, picks, idle, rng)
        self.hist = self._histogram(x)
        self.fitted = True
        return self

    def _ema_step(self, ze: np.ndarray, picks: list, counts: list) -> None:
        """Smoothed exponential moving average, as in VQ-VAE-2."""
        resid = ze
        for level in range(self.levels):
            k = picks[level]
            for j in range(self.codes):
                m = k == j
                nj = float(m.sum())
                if nj == 0.0:
                    continue
                nk = self.decay * counts[level][j] + (1.0 - self.decay) * nj
                tot = (self.decay * counts[level][j] * self.E[level][j]
                       + (1.0 - self.decay) * ze[m].sum(axis=0))
                self.E[level][j] = tot / nk
                counts[level][j] = nk
            resid = resid - self.E[level][k]

    def _restart(self, ze: np.ndarray, picks: list, idle: list,
                 rng: np.random.Generator) -> None:
        """Re-seed a code after it has gone a while without an assignment."""
        for level in range(self.levels):
            hit = np.zeros(self.codes, dtype=bool)
            hit[picks[level]] = True
            idle[level] = np.where(hit, 0.0, idle[level] + 1.0)
            for j in np.where(idle[level] > IDLE)[0]:
                self.E[level][j] = ze[rng.integers(0, len(ze))]
                idle[level][j] = 0.0
                self.reseeds += 1

    # -- outputs -------------------------------------------------------------- #
    def _joint(self, x: np.ndarray, ze: np.ndarray | None = None,
               picks: list | None = None) -> np.ndarray:
        if ze is None:
            ze, _ = _mlp_forward(self.enc, _check_data(x))
        if picks is None:
            _, picks = self._quantise(ze)
        joint = np.zeros(len(ze), dtype=int)
        for level in range(self.levels):
            joint = joint * self.codes + picks[level]
        return joint

    def _histogram(self, x: np.ndarray) -> np.ndarray:
        counts = np.bincount(self._joint(x),
                             minlength=self.codes ** self.levels)
        return counts / counts.sum()

    def reconstruct(self, x: np.ndarray, upto: int | None = None) -> np.ndarray:
        """Decode, optionally stopping after `upto` codebook levels."""
        if not self.fitted:
            raise RuntimeError("call fit first")
        x = _check_data(x)
        if upto is None:
            upto = self.levels
        if not isinstance(upto, (int, np.integer)) or not 1 <= upto <= self.levels:
            raise ValueError(f"upto must lie in 1..{self.levels}")
        ze, _ = _mlp_forward(self.enc, x)
        resid = ze
        zq = np.zeros_like(ze)
        for level in range(upto):
            d2 = ((resid[:, None, :] - self.E[level][None]) ** 2).sum(axis=2)
            k = np.argmin(d2, axis=1)
            zq = zq + self.E[level][k]
            resid = resid - self.E[level][k]
        return _mlp_forward(self.dec, zq)[0]

    def encode(self, x: np.ndarray) -> tuple:
        """Return (continuous latent, code indices per level, quantised latent)."""
        if not self.fitted:
            raise RuntimeError("call fit first")
        ze, _ = _mlp_forward(self.enc, _check_data(x))
        zq, picks = self._quantise(ze)
        return ze, picks, zq

    def usage(self, x: np.ndarray) -> tuple:
        """(codes used, perplexity of the joint assignment, distribution)."""
        if not self.fitted:
            raise RuntimeError("call fit first")
        p = np.bincount(self._joint(x), minlength=self.codes ** self.levels)
        p = p / p.sum()
        used = int(np.sum(p > 1e-3))
        nz = p[p > 0.0]
        return used, float(np.exp(-(nz * np.log(nz)).sum())), p

    def sample(self, n: int = 1024, *, seed: int = 0) -> np.ndarray:
        """Decode codes drawn from the training set's code distribution."""
        if not self.fitted:
            raise RuntimeError("call fit first")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        joint = rng.choice(len(self.hist), size=n, p=self.hist)
        zq = np.zeros((n, self.latent))
        for level in reversed(range(self.levels)):
            zq = zq + self.E[level][joint % self.codes]
            joint = joint // self.codes
        return _mlp_forward(self.dec, zq)[0]

    def n_params(self) -> int:
        if not self.fitted:
            raise RuntimeError("call fit first")
        return int(sum(p.size for p in self.enc + self.dec)
                   + sum(e.size for e in self.E))

    def distances(self, x: np.ndarray) -> np.ndarray:
        """Per-sample squared distance from the latent to its chosen code."""
        ze, picks, _ = self.encode(x)
        resid, total = ze, np.zeros(len(ze))
        for level in range(self.levels):
            eq = self.E[level][picks[level]]
            total = total + ((resid - eq) ** 2).sum(axis=1)
            resid = resid - eq
        return total


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _refused(fn) -> bool:
    try:
        fn()
    except (ValueError, RuntimeError, TypeError):
        return True
    return False


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).sum(axis=1).mean())


def demo() -> None:
    train, test = make_data(0), make_data(1)
    floor = 2.0 * SPREAD ** 2
    print(f"data: {len(train)} train + {len(test)} test, {MODES} modes at "
          f"radius {RADIUS} with sd {SPREAD}")
    print(f"floor: a decoder that emits the mode centre scores {floor:.4f}, "
          f"the data's own variance")

    print("\nvariant               mse /floor  used   ppl   dist  params")
    rows = {}
    for name in VARIANTS:
        model = VQVAE(name).fit(train)
        xh = model.reconstruct(test)
        mse = _mse(test, xh)
        used, ppl, _ = model.usage(test)
        rows[name] = (model, mse, used, ppl)
        print(f"{name:20s} {mse:.5f} {mse / floor:6.2f} "
              f"{used:3d}/{model.codes ** model.levels:<3d} {ppl:5.2f} "
              f"{mode_stats(xh)[2]:6.3f} {model.n_params():6d}")
        assert mode_stats(model.sample(512, seed=1))[0] == MODES

    print("\nwhere each code sits  (all distances divided by latent norm)")
    print("  variant               used   to its cell   quantisation error")
    for name in ("vq_vae", "vq_vae2", "ema_codebook", "commitment_loss",
                 "straight_through"):
        model, _, used, _ = rows[name]
        ze, picks, _ = model.encode(test)
        gaps = [np.abs(model.E[0][j] - ze[picks[0] == j].mean(axis=0)).max()
                for j in range(CODES) if (picks[0] == j).sum() > 8]
        norm = float(np.linalg.norm(ze, axis=1).mean())
        error = float(model.distances(test).mean()) / norm ** 2
        print(f"  {name:18s} {used:3d}/{CODES:<3d}   {max(gaps) / norm:9.3f}"
              f"   {error:16.3f}")
        if name == "vq_vae":
            loose = max(gaps) / norm
        if name == "ema_codebook":
            tight = max(gaps) / norm
    print("  a code is the mean of its cell only if the encoder stops moving")
    assert loose > 2.0 * tight
    assert tight < 0.15

    print("\nlearned codebook vs Lloyd's optimum on the same latents")
    print("  variant              learned     Lloyd   ratio")
    for name in ("vq_vae", "vq_vae2", "ema_codebook", "commitment_loss",
                 "straight_through"):
        model, _, _, _ = rows[name]
        ze = model.encode(test)[0]
        learned = float(model.distances(test).mean())
        best, _ = kmeans(ze, CODES)
        print(f"  {name:18s} {learned:8.4f} {best:8.4f} {learned / best:7.2f}")
        if name == "straight_through":
            settled = learned / best
    assert settled < 1.5

    print("\ncollapses to fewer codes")
    print(f"  vq_vae   {rows['vq_vae'][2]} of {CODES}")
    print(f"  vq_vae2  {rows['vq_vae2'][2]} of {CODES} after "
          f"{rows['vq_vae2'][0].reseeds} re-seeds")
    print(f"  ema      {rows['ema_codebook'][2]} of {CODES}")
    assert rows["vq_vae"][2] < CODES
    assert rows["vq_vae2"][2] == CODES

    print("\ncommitment sweep (beta, same architecture)")
    for beta in (0.0, 0.25, 4.0, 16.0, 64.0):
        model = VQVAE("vq_vae", beta=beta).fit(train)
        ze = model.encode(test)[0]
        print(f"  beta {beta:5.2f}  mse {_mse(test, model.reconstruct(test)):.5f}"
              f"   mean latent norm {np.linalg.norm(ze, axis=1).mean():10.2f}"
              f"   used {model.usage(test)[0]:2d}/{CODES}")
    wild = VQVAE("vq_vae", beta=0.0, seed=0).fit(train)
    assert float(np.linalg.norm(wild.encode(test)[0], axis=1).mean()) > 50.0

    print("\ngradient rules")
    model = rows["straight_through"][0]
    sample_x = test[:64]
    ze, _, _ = model.encode(sample_x)
    dq = np.random.default_rng(0).normal(0.0, 1.0, ze.shape)
    soft = model.soft_gradient(ze, dq)
    cos = float((soft * dq).sum() / (np.linalg.norm(soft) * np.linalg.norm(dq)))
    print(f"  cosine between the relaxed gradient and the straight-through one"
          f" {cos:+.3f}")
    probe = model.E[0].copy()

    def relaxed(z):
        w, _ = _soft_weights(z, probe, model.tau)
        return float((w @ probe * dq).sum())

    num = np.zeros_like(ze)
    for i in range(4):
        for k in range(LATENT):
            e = np.zeros_like(ze)
            e[i, k] = 1e-6
            num[i, k] = (relaxed(ze + e) - relaxed(ze - e)) / 2e-6
    err = float(np.abs(num - soft)[:4].max())
    print(f"  relaxed gradient vs finite differences {err:.2e}")

    probe, near = model.E[0], ze[:4]
    w, _ = _soft_weights(near, probe, model.tau)
    diff = probe[None] - (w @ probe)[:, None]
    cov = np.einsum("nk,nki,nkj->nij", w, diff, diff)
    scale = float(((probe - probe.mean(axis=0)) ** 2).sum(axis=1).mean())
    jac = (2.0 / (model.tau * scale)) * cov
    by_cov = np.einsum("nij,nj->ni", jac, dq[:4])

    def soft_map(z):
        ww, _ = _soft_weights(z, probe, model.tau)
        return ww @ probe

    fd = np.zeros_like(near)
    for k in range(LATENT):
        step = np.zeros_like(near)
        step[:, k] = 1e-6
        fd[:, k] = ((soft_map(near + step) - soft_map(near - step))
                    / 2e-6 * dq[:4]).sum(axis=1)
    cov_err = float(np.abs(by_cov - fd).max())
    print(f"  covariance identity (7) vs finite differences {cov_err:.2e}")
    assert err < 1e-5
    assert cov_err < 1e-5
    assert cos > 0.0

    print("\nresidual quantisation")
    rvq = rows["residual_vq"][0]
    one = _mse(test, rvq.reconstruct(test, upto=1))
    two = _mse(test, rvq.reconstruct(test, upto=2))
    print(f"  level 1 alone {one:.5f}, two levels {two:.5f}, "
          f"ratio {one / two:.2f}")
    print(f"  joint vocabulary {rvq.usage(test)[2][rvq.usage(test)[2] > 1e-3].size}"
          f" of {rvq.codes ** rvq.levels} reachable pairs")
    assert two < one / 2.0

    print("\nintegrity")
    base = rows["vq_vae"][0]
    ze, picks, zq = base.encode(test[:256])
    d2 = ((ze[:, None, :] - base.E[0][None]) ** 2).sum(axis=2)
    assert np.array_equal(picks[0], np.argmin(d2, axis=1))
    print(f"  {'nearest-neighbour':22s} chosen code is the argmin for all "
          f"{len(ze)} probes")
    gap = np.abs(((ze[:, None, :] - base.E[0][None]) ** 2).sum(axis=2)
                 [np.arange(len(ze)), picks[0]]
                 - ((ze - zq) ** 2).sum(axis=1)).max()
    print(f"  {'assignment distance':22s} gathered vs recomputed {gap:.2e}")
    assert gap < 1e-9

    model = rows["commitment_loss"][0]
    xb = test[:24]
    ze, picks, _ = model.encode(xb)
    k = picks[0]

    def codebook_loss():
        eq = model.E[0][k]
        return float(((ze - eq) ** 2).sum(axis=1).mean())

    grad = np.zeros_like(model.E[0])
    np.add.at(grad, k, 2.0 * (model.E[0][k] - ze) / len(xb))
    worst = 0.0
    for j in range(model.codes):
        for d in range(model.latent):
            old = model.E[0][j, d]
            model.E[0][j, d] = old + 1e-6
            up = codebook_loss()
            model.E[0][j, d] = old - 1e-6
            dn = codebook_loss()
            model.E[0][j, d] = old
            num = (up - dn) / 2e-6
            worst = max(worst, abs(num - grad[j, d]) / max(1.0, abs(num)))
    print(f"  {'codebook gradient':22s} finite differences {worst:.2e}")
    assert worst < 1e-4

    dec_worst = 0.0
    zq = model.encode(xb)[2]
    xh, cache = _mlp_forward(model.dec, zq)
    _, gd = _mlp_backward(model.dec, cache, (xh - xb) / (len(xb) * SPREAD ** 2))
    saved = [q.copy() for q in model.dec]
    for pi in range(len(model.dec)):
        for j in range(model.dec[pi].size):
            old = model.dec[pi].flat[j]
            model.dec[pi].flat[j] = old + 1e-6
            up = _mse(xb, _mlp_forward(model.dec, zq)[0]) / (2.0 * SPREAD ** 2)
            model.dec[pi].flat[j] = old - 1e-6
            dn = _mse(xb, _mlp_forward(model.dec, zq)[0]) / (2.0 * SPREAD ** 2)
            model.dec[pi].flat[j] = old
            num = (up - dn) / 2e-6
            dec_worst = max(dec_worst,
                            abs(num - gd[pi].flat[j]) / max(1.0, abs(num)))
    model.dec[:] = saved
    print(f"  {'decoder gradient':22s} finite differences {dec_worst:.2e}")
    assert dec_worst < 1e-4

    first = base.sample(256, seed=3)
    second = base.sample(256, seed=3)
    assert np.array_equal(first, second)
    rec = base.reconstruct(test)
    assert np.array_equal(rec, base.reconstruct(test))

    cases = [
        lambda: VQVAE("nope"),
        lambda: VQVAE("vq_vae", latent=0),
        lambda: VQVAE("vq_vae", codes=0),
        lambda: VQVAE("vq_vae", hidden=0),
        lambda: VQVAE("vq_vae", batch=0),
        lambda: VQVAE("vq_vae", train_steps=0),
        lambda: VQVAE("vq_vae", lr=0.0),
        lambda: VQVAE("vq_vae", lr=float("nan")),
        lambda: VQVAE("vq_vae", tau=0.0),
        lambda: VQVAE("vq_vae", beta=-1.0),
        lambda: VQVAE("vq_vae", decay=1.0),
        lambda: VQVAE("residual_vq", levels=0),
        lambda: VQVAE("straight_through", levels=2),
        lambda: VQVAE("vq_vae").sample(4),
        lambda: VQVAE("vq_vae").reconstruct(test),
        lambda: VQVAE("vq_vae").encode(test),
        lambda: VQVAE("vq_vae").usage(test),
        lambda: VQVAE("vq_vae").n_params(),
        lambda: base.sample(0),
        lambda: base.sample(2.5),
        lambda: base.reconstruct(np.zeros((4, 3))),
        lambda: base.reconstruct(test, upto=0),
        lambda: base.reconstruct(test, upto=3),
        lambda: base.encode(np.array([[0.0, np.nan]])),
        lambda: VQVAE("vq_vae").fit(np.zeros((4, 3))),
        lambda: base.reconstruct(test, upto=-1),
    ]
    refused = sum(_refused(case) for case in cases)
    print(f"\n{refused} of {len(cases)} malformed calls refused")
    assert refused == len(cases)

    print("\nvq_vae demo passed")


if __name__ == "__main__":
    demo()
