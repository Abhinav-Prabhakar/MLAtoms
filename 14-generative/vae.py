"""Variational autoencoder — the ELBO, and four ways to tighten or break it.

Chapter:   14-generative
Companion: vae.md
Tier 1 · 5 variants: standard, conditional, iwae, beta, collapse_fix
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("standard", "conditional", "iwae", "beta", "collapse_fix")

NMODES = 8
RADIUS = 1.2
SPREAD = 0.12
LATENT = 8
HIDDEN = 48
DEC_VAR = 0.02
STEPS = 2000
BATCH = 128
LR = 3e-3
BETA = 8.0
HARD_BETA = 16.0
FREE_BITS = 0.5
K_IWAE = 8
LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Data: eight Gaussians on a circle — two degrees of freedom, eight identities
# --------------------------------------------------------------------------- #
def make_data(seed: int, per_mode: int = 100) -> tuple:
    """Return (x, y) with `per_mode` points around each of eight centres."""
    rng = np.random.default_rng(seed)
    angle = 2.0 * np.pi * np.arange(NMODES) / NMODES
    centres = RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    x = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, 2))
                   for c in centres])
    y = np.repeat(np.arange(NMODES), per_mode)
    order = rng.permutation(len(y))
    return x[order], y[order]


def mode_centres() -> np.ndarray:
    """The eight centres, (8, 2), used by the demo to score samples."""
    angle = 2.0 * np.pi * np.arange(NMODES) / NMODES
    return RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)


# --------------------------------------------------------------------------- #
# Network: 2 -> 48 -> (mu, log var) -> 8 -> 48 -> 2
# --------------------------------------------------------------------------- #
def _init(in_dim: int, dec_dim: int, latent: int, hidden: int,
          seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def mat(rows: int, cols: int) -> np.ndarray:
        return rng.normal(0.0, np.sqrt(1.0 / rows), (rows, cols))

    return {
        "E1": mat(in_dim, hidden), "b1": np.zeros(hidden),
        "Em": mat(hidden, latent), "bm": np.zeros(latent),
        "El": mat(hidden, latent), "bl": np.zeros(latent),
        "D1": mat(dec_dim, hidden), "d1": np.zeros(hidden),
        "D2": mat(hidden, 2), "d2": np.zeros(2),
    }


def _encode(params: dict, xc: np.ndarray) -> tuple:
    h = np.tanh(xc @ params["E1"] + params["b1"])
    return h, h @ params["Em"] + params["bm"], h @ params["El"] + params["bl"]


def _decode(params: dict, zc: np.ndarray) -> tuple:
    """Decode any leading batch shape; returns (hidden, reconstruction)."""
    shape = zc.shape[:-1]
    flat = zc.reshape(-1, zc.shape[-1])
    g = np.tanh(flat @ params["D1"] + params["d1"])
    out = g @ params["D2"] + params["d2"]
    return g, out.reshape(*shape, -1)


def _terms(params: dict, x: np.ndarray, xc: np.ndarray, eps: np.ndarray,
           onehot: np.ndarray | None = None) -> tuple:
    """Return (elbo, rec, kl, cache) per data point for one latent sample."""
    h, mu, lv = _encode(params, xc)
    sd = np.exp(0.5 * lv)
    z = mu + sd * eps
    zc = z if onehot is None else np.concatenate([z, onehot], axis=1)
    g, xr = _decode(params, zc)
    rec = (-0.5 * np.sum((xr - x) ** 2, axis=1) / DEC_VAR
           - 0.5 * x.shape[1] * np.log(2.0 * np.pi * DEC_VAR))
    kl = 0.5 * np.sum(mu ** 2 + np.exp(lv) - 1.0 - lv, axis=1)
    return rec - kl, rec, kl, (h, mu, lv, sd, z, zc, g, xr)


def _grads(params: dict, x: np.ndarray, xc: np.ndarray, eps: np.ndarray,
           beta: float = 1.0, mask: np.ndarray | None = None,
           onehot: np.ndarray | None = None) -> dict:
    """Gradient of mean(-rec + beta * KL), the single-sample objective.

    `mask` is the free-bits switch: a zero in dimension d drops that
    dimension's KL term from the objective.
    """
    n = len(x)
    _, _, _, (h, mu, lv, sd, _, zc, g, xr) = _terms(params, x, xc, eps, onehot)
    if mask is None:
        mask = np.ones(lv.shape[1])
    dxr = (xr - x) / DEC_VAR / n
    grads = {"D2": g.T @ dxr, "d2": dxr.sum(0)}
    dg = (dxr @ params["D2"].T) * (1.0 - g ** 2)
    grads["D1"] = zc.T @ dg
    grads["d1"] = dg.sum(0)
    dz = (dg @ params["D1"].T)[:, :lv.shape[1]]
    dmu = dz + beta * (mu / n) * mask
    dlv = (dz * eps * 0.5 * sd
           + beta * 0.5 * ((np.exp(lv) - 1.0) / n) * mask)
    grads["Em"] = h.T @ dmu
    grads["bm"] = dmu.sum(0)
    grads["El"] = h.T @ dlv
    grads["bl"] = dlv.sum(0)
    dh = (dmu @ params["Em"].T + dlv @ params["El"].T) * (1.0 - h ** 2)
    grads["E1"] = xc.T @ dh
    grads["b1"] = dh.sum(0)
    return grads


def iwae_log_weights(params: dict, x: np.ndarray,
                     eps: np.ndarray) -> np.ndarray:
    """Log importance weights, (n, k), for k latent samples per point.

    `logmeanexp` along the second axis is the k-sample bound. This is the
    unconditional path only, which is why it takes `x` directly.
    """
    n, k, latent = eps.shape
    _, mu, lv = _encode(params, x)
    sd = np.exp(0.5 * lv)
    z = mu[:, None, :] + sd[:, None, :] * eps
    _, xr = _decode(params, z)
    ss = np.sum((xr - x[:, None, :]) ** 2, axis=2)
    log_px = -0.5 * ss / DEC_VAR - np.log(2.0 * np.pi * DEC_VAR)
    log_pz = -0.5 * np.sum(z ** 2, axis=2) - 0.5 * latent * LOG_2PI
    log_qz = -0.5 * np.sum(eps ** 2 + lv[:, None, :], axis=2)
    log_qz -= 0.5 * latent * LOG_2PI
    return log_px + log_pz - log_qz


def _softmax_rows(log_w: np.ndarray) -> np.ndarray:
    shifted = log_w - log_w.max(axis=1, keepdims=True)
    weight = np.exp(shifted)
    return weight / weight.sum(axis=1, keepdims=True)


def _grads_iwae(params: dict, x: np.ndarray, eps: np.ndarray) -> dict:
    """Pathwise gradient of the *negative* k-sample bound, unconditional.

    The sign is the awkward part: every other variant minimises a loss, so this
    one returns the descent direction for -bound and `_adam` subtracts it.
    """
    n, k, latent = eps.shape
    weight = _softmax_rows(iwae_log_weights(params, x, eps))
    h, mu, lv = _encode(params, x)
    sd = np.exp(0.5 * lv)
    z = mu[:, None, :] + sd[:, None, :] * eps
    g, xr = _decode(params, z)
    dxr = weight[:, :, None] * (xr - x[:, None, :]) / DEC_VAR / n
    dz = weight[:, :, None] * z / n
    flat_dxr = dxr.reshape(n * k, 2)
    flat_g = g.reshape(n * k, h.shape[1])
    grads = {"D2": flat_g.T @ flat_dxr, "d2": flat_dxr.sum(0)}
    dg = (flat_dxr @ params["D2"].T) * (1.0 - flat_g ** 2)
    grads["D1"] = z.reshape(n * k, latent).T @ dg
    grads["d1"] = dg.sum(0)
    dz = dz + (dg @ params["D1"].T).reshape(n, k, latent)
    dmu = dz.sum(axis=1)
    dlv = np.sum(dz * eps * 0.5 * sd[:, None, :], axis=1) - 0.5 / n
    grads["Em"] = h.T @ dmu
    grads["bm"] = dmu.sum(0)
    grads["El"] = h.T @ dlv
    grads["bl"] = dlv.sum(0)
    dh = (dmu @ params["Em"].T + dlv @ params["El"].T) * (1.0 - h ** 2)
    grads["E1"] = x.T @ dh
    grads["b1"] = dh.sum(0)
    return grads


def _adam(params: dict, grads: dict, state: tuple, step: int,
          lr: float) -> None:
    first, second = state
    for key, grad in grads.items():
        first[key] = 0.9 * first[key] + 0.1 * grad
        second[key] = 0.999 * second[key] + 0.001 * grad ** 2
        params[key] -= lr * (first[key] / (1.0 - 0.9 ** step)) / (
            np.sqrt(second[key] / (1.0 - 0.999 ** step)) + 1e-8)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class VAE:
    """Fit a Gaussian-latent autoencoder by maximising the ELBO.

    `params` holds the ten arrays, `beta` scales the KL term, and `free` is the
    per-dimension KL floor used by `collapse_fix` (ignored otherwise).
    """

    def __init__(self, variant: str = "standard", *, latent: int = LATENT,
                 hidden: int = HIDDEN, beta: float = BETA,
                 free: float = FREE_BITS, steps: int = STEPS,
                 batch: int = BATCH, lr: float = LR, seed: int = 1):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if latent < 1:
            raise ValueError("latent must be at least one")
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be positive and finite")
        if not np.isfinite(free) or free < 0.0:
            raise ValueError("free must be non-negative and finite")
        if steps < 1 or batch < 1:
            raise ValueError("steps and batch must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.latent = latent
        self.hidden = hidden
        self.beta = beta
        self.free = free
        self.steps = steps
        self.batch = batch
        self.lr = lr
        self.seed = seed
        self.conditional = variant == "conditional"
        self.in_dim = 2 + (NMODES if self.conditional else 0)
        self.dec_dim = latent + (NMODES if self.conditional else 0)
        self.params: dict | None = None

    # -- packing -------------------------------------------------------------- #
    def _pack(self, x: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        x = _check(x, 2)
        if not self.conditional:
            return x
        if y is None:
            raise ValueError("the conditional variant needs labels")
        y = _check_label(y, len(x))
        return np.concatenate([x, np.eye(NMODES)[y]], axis=1)

    def _pack_z(self, z: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        if not self.conditional:
            return z
        onehot = np.eye(NMODES)[_check_label(y, len(z))]
        return np.concatenate([z, onehot], axis=1)

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> VAE:
        xc = self._pack(x, y)
        onehot = np.eye(NMODES)[y] if self.conditional else None
        n = len(xc)
        rng = np.random.default_rng(self.seed)
        params = _init(self.in_dim, self.dec_dim, self.latent, self.hidden,
                       self.seed)
        state = ({k: np.zeros_like(v) for k, v in params.items()},
                 {k: np.zeros_like(v) for k, v in params.items()})
        for step in range(1, self.steps + 1):
            idx = rng.integers(0, n, self.batch)
            if self.variant == "iwae":
                eps = rng.normal(0.0, 1.0, (self.batch, K_IWAE, self.latent))
                grads = _grads_iwae(params, x[idx], eps)
            else:
                eps = rng.normal(0.0, 1.0, (self.batch, self.latent))
                grads = _grads(params, x[idx], xc[idx], eps, self._beta(),
                               self._mask(params, xc),
                               None if onehot is None else onehot[idx])
            _adam(params, grads, state, step, self.lr)
        self.params = params
        return self

    def _beta(self) -> float:
        if self.variant == "beta":
            return self.beta
        if self.variant == "collapse_fix":
            return HARD_BETA
        return 1.0

    def _mask(self, params: dict, xc: np.ndarray) -> np.ndarray | None:
        if self.variant != "collapse_fix":
            return None
        _, mu, lv = _encode(params, xc)
        kld = 0.5 * (np.mean(mu ** 2, axis=0) + np.mean(np.exp(lv), axis=0)
                     - 1.0 - np.mean(lv, axis=0))
        return (kld > self.free).astype(float)

    # -- evaluation ----------------------------------------------------------- #
    def elbo(self, x: np.ndarray, y: np.ndarray | None = None, *,
             k: int = 32, seed: int = 0, importance: bool = False) -> float:
        """Mean bound per point; `importance=True` returns the k-sample IWAE."""
        if self.params is None:
            raise RuntimeError("call fit first")
        xc = self._pack(x, y)
        onehot = np.eye(NMODES)[y] if self.conditional else None
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, 1.0, (len(x), k, self.latent))
        if importance and self.conditional:
            raise ValueError("the iwae bound is unconditional only")
        if importance:
            log_w = iwae_log_weights(self.params, x, eps)
            top = log_w.max(axis=1, keepdims=True)
            return float(np.mean(np.log(np.exp(log_w - top).mean(axis=1))
                                 + top[:, 0]))
        values = [_terms(self.params, x, xc, eps[:, j, :], onehot)[0]
                  for j in range(k)]
        return float(np.mean(values))

    def reconstruct(self, x: np.ndarray, y: np.ndarray | None = None) -> tuple:
        """Return (mean reconstruction, (rec, kl)) at the posterior mean."""
        if self.params is None:
            raise RuntimeError("call fit first")
        xc = self._pack(x, y)
        _, mu, lv = _encode(self.params, xc)
        _, xr = _decode(self.params, self._pack_z(mu, y))
        rec = (-0.5 * np.sum((xr - x) ** 2, axis=1) / DEC_VAR
               - 0.5 * x.shape[1] * np.log(2.0 * np.pi * DEC_VAR))
        kl = 0.5 * np.sum(mu ** 2 + np.exp(lv) - 1.0 - lv, axis=1)
        return xr, (float(rec.mean()), float(kl.mean()))

    def posterior_sd(self, x: np.ndarray,
                     y: np.ndarray | None = None) -> np.ndarray:
        """Posterior standard deviation, (n, latent), one row per point."""
        if self.params is None:
            raise RuntimeError("call fit first")
        _, _, lv = _encode(self.params, self._pack(x, y))
        return np.exp(0.5 * lv)

    def active(self, x: np.ndarray, y: np.ndarray | None = None,
               threshold: float = 0.01) -> int:
        """Latent dimensions whose posterior mean varies across the data."""
        if self.params is None:
            raise RuntimeError("call fit first")
        _, mu, _ = _encode(self.params, self._pack(x, y))
        return int(np.sum(mu.var(axis=0) > threshold))

    def sample(self, n: int, y: np.ndarray | None = None,
               seed: int = 0) -> np.ndarray:
        """Draw n points from the prior, (n, 2)."""
        if self.params is None:
            raise RuntimeError("call fit first")
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, 1.0, (n, self.latent))
        return _decode(self.params, self._pack_z(z, y))[1]


def _check(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"x must be (n, {dim})")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    return x


def _check_label(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must be (n,) and match x")
    if np.any((y < 0) | (y >= NMODES)):
        raise ValueError(f"y must lie in [0, {NMODES})")
    return y


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    xtr, ytr = make_data(0)
    xte, yte = make_data(1)
    print(f"variational autoencoder. {NMODES} modes on a circle, "
          f"{len(xtr)} points, latent {LATENT}:")
    print(f"  decoder variance {DEC_VAR}, hidden width {HIDDEN}, "
          f"{STEPS} Adam steps, batch {BATCH}")

    runs = {}
    for name in VARIANTS:
        model = VAE(name)
        runs[name] = model.fit(xtr, ytr if model.conditional else None)

    print("variational autoencoder. what each variant optimises:")
    print(f"  {'variant':<14s} {'elbo':>8s} {'iwae8':>8s} {'active':>7s} "
          f"{'mean sd':>8s}")
    for name, model in runs.items():
        label = yte if model.conditional else None
        bound = ("      --" if model.conditional
                 else f"{model.elbo(xte, k=8, importance=True):9.3f}")
        print(f"  {name:<14s} {model.elbo(xte, label):8.3f} {bound} "
              f"{model.active(xte, label):4d}/{LATENT} "
              f"{model.posterior_sd(xte, label).mean():8.3f}")

    print("variational autoencoder. the fit behind each column:")
    print(f"  {'variant':<14s} {'beta':>5s} {'free':>5s} {'rec':>8s} "
          f"{'kl':>7s} {'elbo':>8s} {'test mse':>9s}")
    for name, model in runs.items():
        label = yte if model.conditional else None
        xr, (rec, kl) = model.reconstruct(xte, label)
        mse = float(np.mean(np.sum((xr - xte) ** 2, axis=1)))
        free = model.free if name == "collapse_fix" else 0.0
        print(f"  {name:<14s} {model._beta():5.1f} {free:5.2f} {rec:8.3f} "
              f"{kl:7.3f} {model.elbo(xte, label):8.3f} {mse:9.4f}")

    print("variational autoencoder. what the kl weight buys:")
    sweep = [(1.0, runs["standard"])]
    sweep += [(b, VAE("beta", beta=b).fit(xtr)) for b in (2.0, 4.0, 8.0)]
    for beta, model in sweep:
        xr, (rec, kl) = model.reconstruct(xte)
        mse = float(np.mean(np.sum((xr - xte) ** 2, axis=1)))
        print(f"  beta {beta:4.1f}  kl {kl:6.3f}  rec {rec:6.3f}  "
              f"elbo {model.elbo(xte):8.3f}  mse {mse:7.4f}")

    print("variational autoencoder. collapse at a hard beta, and free bits:")
    collapsed = VAE("beta", beta=HARD_BETA).fit(xtr)
    for tag, model in (("beta 16", collapsed),
                       ("free bits .5", runs["collapse_fix"])):
        xr, (rec, kl) = model.reconstruct(xte)
        print(f"  {tag:<13s} rec {rec:7.3f}  kl {kl:6.3f}  "
              f"active {model.active(xte)}/{LATENT}  "
              f"elbo {model.elbo(xte):8.3f}")

    cond = runs["conditional"]
    centres = mode_centres()
    near = []
    print("variational autoencoder. conditional samples, one per class:")
    for j in range(NMODES):
        draw = cond.sample(200, y=np.full(200, j), seed=j)
        near.append(float(np.mean(np.linalg.norm(draw - centres[j], axis=1))))
    print("  mean distance to the right centre: "
          + " ".join(f"{v:.2f}" for v in near))

    print("variational autoencoder. both gradients match finite differences:")
    probe = VAE("standard", steps=1)
    probe.params = _init(2, LATENT, LATENT, HIDDEN, 3)
    xb, _ = make_data(7, per_mode=4)

    def elbo_value(eps: np.ndarray) -> float:
        return float(np.mean(_terms(probe.params, xb, xb, eps)[0]))

    def bound_value(eps: np.ndarray) -> float:
        log_w = iwae_log_weights(probe.params, xb, eps)
        top = log_w.max(axis=1, keepdims=True)
        shifted = np.exp(log_w - top).mean(axis=1)
        return float(np.mean(np.log(shifted) + top[:, 0]))

    def elbo_grads(eps: np.ndarray) -> dict:
        return _grads(probe.params, xb, xb, eps, 1.0)

    def bound_grads(eps: np.ndarray) -> dict:
        return _grads_iwae(probe.params, xb, eps)

    rng = np.random.default_rng(4)
    worst = {}
    for tag, grad_fn, value, shape in (
            ("elbo", elbo_grads, elbo_value, (len(xb), LATENT)),
            ("8-sample bound", bound_grads, bound_value,
             (len(xb), K_IWAE, LATENT))):
        eps = rng.normal(0.0, 1.0, shape)
        grads = grad_fn(eps)
        worst[tag] = 0.0
        for key in ("E1", "Em", "El", "D2", "bl"):
            index = np.unravel_index(np.argmax(np.abs(grads[key])),
                                     grads[key].shape)
            past, step = probe.params[key][index], 1e-6
            probe.params[key][index] = past + step
            up = value(eps)
            probe.params[key][index] = past - step
            down = value(eps)
            probe.params[key][index] = past
            worst[tag] = max(worst[tag], abs((up - down) / (2.0 * step)
                                             + grads[key][index]))
    print(f"  worst disagreement {worst['elbo']:.2e} for the elbo and "
          f"{worst['8-sample bound']:.2e} for the bound")

    print("variational autoencoder. malformed calls are refused:")
    bad = [
        lambda: VAE("standard_v2"),
        lambda: VAE("standard", latent=0),
        lambda: VAE("beta", beta=0.0),
        lambda: VAE("collapse_fix", free=-1.0),
        lambda: VAE("standard", steps=0),
        lambda: VAE("standard").elbo(xte),
        lambda: VAE("conditional").fit(xtr),
        lambda: VAE("standard").fit(np.zeros((4, 3))),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)} malformed calls refused")

    standard, iwae = runs["standard"], runs["iwae"]
    _, (rec_std, kl_std) = standard.reconstruct(xte)
    _, (rec_beta, kl_beta) = runs["beta"].reconstruct(xte)
    _, (rec_hard, kl_hard) = collapsed.reconstruct(xte)
    _, (kl_free) = runs["collapse_fix"].reconstruct(xte)[1]
    std_b8 = standard.elbo(xte, k=8, importance=True)
    iw_b8 = iwae.elbo(xte, k=8, importance=True)
    print(f"  beta {BETA:g}: kl {kl_std:.2f} -> {kl_beta:.2f} for "
          f"{rec_std - rec_beta:.2f} nats of reconstruction, and "
          f"{standard.elbo(xte) - runs['beta'].elbo(xte):.2f} nats of elbo")
    print(f"  the 8-sample bound prefers iwae ({iw_b8:.3f} vs {std_b8:.3f}) "
          f"while its 1-sample estimate is worse "
          f"({iwae.elbo(xte):.2f} vs {standard.elbo(xte):.2f})")
    print(f"  hard beta 16 drops the elbo {runs['beta'].elbo(xte):.2f} -> "
          f"{collapsed.elbo(xte):.2f}; free bits holds it at "
          f"{runs['collapse_fix'].elbo(xte):.2f}")

    assert worst["elbo"] < 1e-5 and worst["8-sample bound"] < 1e-5
    assert standard.active(xte) == 2
    assert kl_beta < 0.6 * kl_std and rec_beta < rec_std
    assert kl_hard < 0.5 * kl_std and rec_hard < rec_beta
    assert kl_free > 2.0 * kl_hard
    assert runs["collapse_fix"].elbo(xte) > collapsed.elbo(xte) + 4.0
    assert iw_b8 > std_b8 + 0.2
    assert iwae.elbo(xte) < standard.elbo(xte) - 1.0
    kls = [m.reconstruct(xte)[1][1] for _, m in sweep]
    assert all(a > b for a, b in zip(kls, kls[1:], strict=False))
    assert max(near) < 0.5
    print("variational autoencoder demo passed")


if __name__ == "__main__":
    demo()
